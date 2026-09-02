"""
src/harness/reconciler.py

Serving-harness reconciler (cambia-256, design overview 4.2 / 4.3 / 5.7).

Replays a runner's synced run_db.sqlite into the client's authoritative
cfr/runs/cambia_runs.db. The runner-local journal db IS the reconciliation wire
format: run_meta.json and eval_summary.jsonl are lossy display artifacts and are
not consumed here (design 4.2).

Trust boundary (design 5.7): everything pulled from a runner is untrusted input.
The LXC boundary contains processes, not data flows, so a compromised job can
author arbitrary sync content. The reconciler therefore:

  - opens the synced sqlite READ-ONLY (mode=ro URI);
  - whitelists fields per table, ignoring anything else the pulled db carries;
  - range-checks every numeric and enum, rejecting out-of-range values;
  - re-validates the run name with the Go validateName rules before any
    directory join;
  - NEVER trusts a pulled checkpoint file_path (a pulled absolute/traversal path
    would later feed a client-side torch.load, i.e. pickle) and instead re-derives
    it from the known local synced layout under the given run dir;
  - refuses to overwrite a local run (origin_host IS NULL) that shares a name
    with a pulled run (same-name cross-host runs are unsupported in v1).

Replay is idempotent: every write keys on a natural key (runs.name,
checkpoints(run_id, iteration), eval_results(run_id, iteration, baseline)), so
partial pulls converge on re-replay. Replay order run -> checkpoints -> evals
respects the foreign keys. Runner surrogate ids are never carried across stores:
the local checkpoint id for each eval row is re-resolved by (run_id, iteration)
against the destination db.
"""

import json
import math
import os
import re
import shutil
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.run_db import (
    ALGO_TO_AGENT_TYPE,
    STABILITY_METRIC_BASELINES,
    algo_to_checkpoint_prefix,
    get_db,
    insert_eval_result,
    register_checkpoint,
    upsert_run,
)

# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class ReconcilerError(Exception):
    """Base class for reconciler failures."""


class ReconcilerValidationError(ReconcilerError):
    """A pulled row failed a whitelist / range / name-validation check."""


class ReconcilerCollisionError(ReconcilerError):
    """A pulled run name collides with a local run (or a different host)."""


# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------

# Go validateName rules mirrored (design 5.7 / process.go validateName): reject
# empty, ".", "..", "/" or any absolute path, characters outside
# [A-Za-z0-9._-], and anything over the length cap. "/" and a leading "/" are
# both excluded by the character class, so the regex alone rejects absolute and
# multi-segment paths; "." / ".." are rejected explicitly.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BASENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_MAX_NAME_LEN = 128
_MAX_HOST_LEN = 128
_MAX_BASELINE_LEN = 128
_MAX_SHORT_STR_LEN = 128
_MAX_NOTES_LEN = 65536

# Run status enum. Superset of the run_db lifecycle values and the runnerd job
# state machine (design 2.3): anything outside this set is treated as an
# out-of-range enum and rejected.
_ALLOWED_STATUS: Set[str] = {
    "created",
    "queued",
    "preparing",
    # procmgr writes "starting" (runnerd/procmgr/state.go); a run synced mid
    # launch carries it, and rejecting it would fail an otherwise valid replay.
    "starting",
    "running",
    "stopping",
    "stopped",
    "crashed",
    "canceled",
    "cancelled",
    "failed",
    "completed",
    "finished",
    "done",
    "interrupted",
    # A gate-driven stop (design 7 D62), distinct from an operator cancel: with
    # no promoted checkpoint the job returns to ready with no attempt increment,
    # with a checkpoint it is terminal pending an explicit operator resume.
    "preempted",
}


def _validate_run_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ReconcilerValidationError("run name is empty or not text")
    if len(name) > _MAX_NAME_LEN:
        raise ReconcilerValidationError(
            f"run name exceeds {_MAX_NAME_LEN} chars: {name!r}"
        )
    if name in (".", ".."):
        raise ReconcilerValidationError(f"run name is a reserved path segment: {name!r}")
    if not _NAME_RE.match(name):
        raise ReconcilerValidationError(f"run name has illegal characters: {name!r}")
    return name


def _validate_origin_host(host: Any) -> str:
    if not isinstance(host, str) or not host:
        raise ReconcilerValidationError("origin_host must be a non-empty string")
    if len(host) > _MAX_HOST_LEN or not _HOST_RE.match(host):
        raise ReconcilerValidationError(f"invalid origin_host: {host!r}")
    return host


def _validate_executed_on(value: Any) -> Optional[str]:
    """Validate a pulled env.json `executed_on` field (design 4.2 D23): "which
    node produced these numbers", the same name shape as origin_host. Unlike
    origin_host (a required replay() argument the collision guard's ownership
    authority depends on), executed_on is informational provenance attached
    from an optional file, so an invalid or absent value is dropped to NULL
    rather than failing the replay."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    if len(value) > _MAX_HOST_LEN or not _HOST_RE.match(value):
        return None
    return value


def _validate_status(status: Any) -> str:
    if status is None:
        return "created"
    if not isinstance(status, str) or status not in _ALLOWED_STATUS:
        raise ReconcilerValidationError(f"invalid run status (enum): {status!r}")
    return status


def _validate_algorithm(algorithm: Any) -> Optional[str]:
    if algorithm is None:
        return None
    if not isinstance(algorithm, str):
        raise ReconcilerValidationError("algorithm must be text or NULL")
    if algorithm not in ALGO_TO_AGENT_TYPE:
        raise ReconcilerValidationError(f"unknown algorithm (enum): {algorithm!r}")
    return algorithm


def _num_or_none(
    value: Any,
    field: str,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    integral: bool = False,
) -> Optional[Union[int, float]]:
    """Range-check a numeric cell (sqlite is dynamically typed, so a pulled db
    may store text/blobs in a numeric column; reject anything non-numeric,
    non-finite, or out of [lo, hi])."""
    if value is None:
        return None
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        raise ReconcilerValidationError(f"{field}: non-numeric value {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ReconcilerValidationError(f"{field}: non-finite value {value!r}")
    if integral:
        if isinstance(value, float):
            if not value.is_integer():
                raise ReconcilerValidationError(f"{field}: expected integer {value!r}")
            value = int(value)
    if lo is not None and value < lo:
        raise ReconcilerValidationError(f"{field}: {value} below minimum {lo}")
    if hi is not None and value > hi:
        raise ReconcilerValidationError(f"{field}: {value} above maximum {hi}")
    return value


def _int_or_none(value: Any, field: str, lo: Optional[int] = 0) -> Optional[int]:
    out = _num_or_none(value, field, lo=lo, hi=None, integral=True)
    return None if out is None else int(out)


def _prob_or_none(value: Any, field: str) -> Optional[float]:
    out = _num_or_none(value, field, lo=0.0, hi=1.0)
    return None if out is None else float(out)


def _flag01(value: Any, field: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    raise ReconcilerValidationError(f"{field}: expected 0/1, got {value!r}")


def _str_or_none(value: Any, field: str, maxlen: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReconcilerValidationError(
            f"{field}: expected text, got {type(value).__name__}"
        )
    if len(value) > maxlen:
        raise ReconcilerValidationError(f"{field}: exceeds {maxlen} chars")
    if "\x00" in value:
        raise ReconcilerValidationError(f"{field}: contains NUL byte")
    return value


# ---------------------------------------------------------------------------
# Read-only source access
# ---------------------------------------------------------------------------


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open a sqlite file strictly read-only via the mode=ro URI so replay can
    never mutate the untrusted source."""
    uri = "file:" + urllib.request.pathname2url(str(path.resolve())) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _present_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {row[1] for row in rows}


def _select_whitelisted(
    conn: sqlite3.Connection,
    table: str,
    whitelist: List[str],
    where: str = "",
    params: tuple = (),
    order_by: str = "",
) -> List[Dict[str, Any]]:
    """Read only the intersection of `whitelist` and the columns actually present
    in the (untrusted, possibly schema-drifted) source. Missing whitelisted
    columns read back as None; extra source columns are ignored entirely."""
    present = _present_columns(conn, table)
    cols = [c for c in whitelist if c in present]
    if not cols:
        return []
    col_sql = ", ".join(cols)
    sql = f"SELECT {col_sql} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    out: List[Dict[str, Any]] = []
    for row in conn.execute(sql, params).fetchall():
        d: Dict[str, Any] = {c: None for c in whitelist}
        for c in cols:
            d[c] = row[c]
        out.append(d)
    return out


_RUNS_WHITELIST = [
    "id",
    "name",
    "algorithm",
    "status",
    "engine_commit_hash",
    "tags",
    "notes",
]

_CHECKPOINTS_WHITELIST = [
    "run_id",
    "iteration",
    "file_path",
    "file_size_bytes",
    "is_best",
    "is_retained",
    "compressed",
]

_EVALS_WHITELIST = [
    "run_id",
    "iteration",
    "baseline",
    "win_rate",
    "ci_low",
    "ci_high",
    "games_played",
    "p0_wins",
    "p1_wins",
    "ties",
    "avg_game_turns",
    "t1_cambia_rate",
    "avg_score_margin",
    "adv_loss",
    "strat_loss",
    "seat_balanced",
    "selection_mode",
    "crn_seed",
    "seat_scheme",
    # cambia-1479 eval-integrity columns; a source db predating them (or a
    # source row measured before they existed) carries NULL here, which must
    # replay as NULL rather than a default that would masquerade as a
    # measurement (cambia-1937).
    "policy_errors",
    "engine_errors",
    "belief_protocol",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Checkpoint path re-derivation (design 4.2 step 2 / 5.7)
# ---------------------------------------------------------------------------


def _rederive_checkpoint_path(
    run_dir: Path,
    iteration: int,
    algorithm: Optional[str],
    pulled_file_path: Any,
) -> str:
    """Re-derive a checkpoint's local path from the synced layout under run_dir.

    The pulled file_path's directory component is discarded (only its basename is
    considered, and only if it is a safe simple filename); a path that would
    escape run_dir is impossible by construction, and the result is
    containment-checked as defense in depth against a symlinked run dir. If the
    basename is unusable, the filename is synthesized from the algorithm prefix
    and iteration. The path is located within the known snapshot/checkpoint
    subdirs when present, else defaulted to a contained snapshots/ path so a
    not-yet-synced file still records a valid local target.
    """
    run_root = run_dir.resolve()
    base = os.path.basename(str(pulled_file_path or ""))
    if not base or base in (".", "..") or not _SAFE_BASENAME_RE.match(base):
        prefix = algo_to_checkpoint_prefix(algorithm or "")
        base = f"{prefix}_iter_{iteration}.pt"

    candidate: Optional[Path] = None
    for sub in ("snapshots", "checkpoints", ""):
        probe = (run_root / sub / base) if sub else (run_root / base)
        if probe.exists():
            candidate = probe
            break
    if candidate is None:
        candidate = run_root / "snapshots" / base

    resolved = candidate.resolve()
    if resolved != run_root and run_root not in resolved.parents:
        raise ReconcilerValidationError(
            f"re-derived checkpoint path escapes run dir: {resolved}"
        )
    return str(candidate)


# ---------------------------------------------------------------------------
# Row readers (whitelist + range checks)
# ---------------------------------------------------------------------------


def _read_runs(src: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = _select_whitelisted(src, "runs", _RUNS_WHITELIST, order_by="id")
    out: List[Dict[str, Any]] = []
    for r in rows:
        import json as _json

        tags_raw = r.get("tags")
        tags: list = []
        if tags_raw is not None:
            if not isinstance(tags_raw, str):
                raise ReconcilerValidationError("tags must be JSON text")
            try:
                parsed = _json.loads(tags_raw)
            except (ValueError, TypeError):
                raise ReconcilerValidationError(f"tags not valid JSON: {tags_raw!r}")
            if not isinstance(parsed, list):
                raise ReconcilerValidationError("tags JSON must be a list")
            tags = [_str_or_none(t, "tag", _MAX_SHORT_STR_LEN) for t in parsed]
        out.append(
            {
                "_src_id": r.get("id"),
                "name": _validate_run_name(r.get("name")),
                "algorithm": _validate_algorithm(r.get("algorithm")),
                "status": _validate_status(r.get("status")),
                "engine_commit_hash": _str_or_none(
                    r.get("engine_commit_hash"), "engine_commit_hash", _MAX_SHORT_STR_LEN
                ),
                "tags": tags,
                "notes": _str_or_none(r.get("notes"), "notes", _MAX_NOTES_LEN),
            }
        )
    return out


def _read_checkpoints(src: sqlite3.Connection, src_run_id: Any) -> List[Dict[str, Any]]:
    rows = _select_whitelisted(
        src,
        "checkpoints",
        _CHECKPOINTS_WHITELIST,
        where="run_id=?",
        params=(src_run_id,),
        order_by="iteration",
    )
    out: List[Dict[str, Any]] = []
    for c in rows:
        iteration = _int_or_none(c.get("iteration"), "checkpoint.iteration", lo=0)
        if iteration is None:
            raise ReconcilerValidationError("checkpoint.iteration is NULL")
        out.append(
            {
                "iteration": iteration,
                "file_path": c.get("file_path"),  # untrusted; re-derived later
                "is_best": _flag01(c.get("is_best"), "checkpoint.is_best", default=0),
                "is_retained": _flag01(
                    c.get("is_retained"), "checkpoint.is_retained", default=1
                ),
                "compressed": _flag01(
                    c.get("compressed"), "checkpoint.compressed", default=0
                ),
            }
        )
    return out


def _read_evals(src: sqlite3.Connection, src_run_id: Any) -> List[Dict[str, Any]]:
    rows = _select_whitelisted(
        src,
        "eval_results",
        _EVALS_WHITELIST,
        where="run_id=?",
        params=(src_run_id,),
        order_by="iteration, baseline",
    )
    out: List[Dict[str, Any]] = []
    for e in rows:
        iteration = _int_or_none(e.get("iteration"), "eval.iteration", lo=0)
        if iteration is None:
            raise ReconcilerValidationError("eval.iteration is NULL")
        baseline = _str_or_none(e.get("baseline"), "eval.baseline", _MAX_BASELINE_LEN)
        if not baseline:
            raise ReconcilerValidationError("eval.baseline is empty or NULL")
        # Stability-metric journal rows (e.g. "nashconv" from the tiny gate's
        # _record_nashconv_in_db, "lbr_tier_b"/"ismcts_br" from the mixture
        # exploitability script) ride the exact metric value in win_rate but are
        # not a win probability -- exact NashConv on the tiny tree routinely
        # exceeds 1.0, and a symmetric CI on a near-zero exploitability can dip
        # below 0. Bound win_rate as finite/non-negative and the CI columns as
        # finite (a bound may be slightly negative near zero) instead of the
        # genuine win-rate [0, 1] bound; every other baseline (real agent win
        # rates, incl. MEAN_IMP_BASELINES) keeps the strict bound on all three.
        if baseline in STABILITY_METRIC_BASELINES:
            win_rate = _num_or_none(e.get("win_rate"), "eval.win_rate", lo=0.0)
            ci_low = _num_or_none(e.get("ci_low"), "eval.ci_low")
            ci_high = _num_or_none(e.get("ci_high"), "eval.ci_high")
        else:
            win_rate = _prob_or_none(e.get("win_rate"), "eval.win_rate")
            ci_low = _prob_or_none(e.get("ci_low"), "eval.ci_low")
            ci_high = _prob_or_none(e.get("ci_high"), "eval.ci_high")
        row_dict = {
            "iteration": iteration,
            "baseline": baseline,
            "win_rate": win_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "games_played": _int_or_none(e.get("games_played"), "eval.games_played"),
            "p0_wins": _int_or_none(e.get("p0_wins"), "eval.p0_wins"),
            "p1_wins": _int_or_none(e.get("p1_wins"), "eval.p1_wins"),
            "ties": _int_or_none(e.get("ties"), "eval.ties"),
            "avg_game_turns": _num_or_none(
                e.get("avg_game_turns"), "eval.avg_game_turns", lo=0.0
            ),
            "t1_cambia_rate": _prob_or_none(
                e.get("t1_cambia_rate"), "eval.t1_cambia_rate"
            ),
            "avg_score_margin": _num_or_none(
                e.get("avg_score_margin"), "eval.avg_score_margin"
            ),
            "adv_loss": _num_or_none(e.get("adv_loss"), "eval.adv_loss"),
            "strat_loss": _num_or_none(e.get("strat_loss"), "eval.strat_loss"),
            "seat_balanced": _flag01(
                e.get("seat_balanced"), "eval.seat_balanced", default=0
            ),
            "selection_mode": _str_or_none(
                e.get("selection_mode"), "eval.selection_mode", _MAX_SHORT_STR_LEN
            ),
            "crn_seed": _str_or_none(
                e.get("crn_seed"), "eval.crn_seed", _MAX_SHORT_STR_LEN
            ),
            "seat_scheme": _str_or_none(
                e.get("seat_scheme"), "eval.seat_scheme", _MAX_SHORT_STR_LEN
            ),
            "policy_errors": _int_or_none(e.get("policy_errors"), "eval.policy_errors"),
            "engine_errors": _int_or_none(e.get("engine_errors"), "eval.engine_errors"),
            "belief_protocol": _str_or_none(
                e.get("belief_protocol"), "eval.belief_protocol", _MAX_SHORT_STR_LEN
            ),
            "timestamp": _str_or_none(
                e.get("timestamp"), "eval.timestamp", _MAX_SHORT_STR_LEN
            ),
        }
        out.append(row_dict)
    return out


# ---------------------------------------------------------------------------
# Coordinator-authored sidecar files (design 4.2 D23, design 8 D64)
# ---------------------------------------------------------------------------


def _read_run_dir_json(run_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
    """Best-effort read of a coordinator-authored JSON sidecar (jobspec.json,
    env.json, process.json) from a synced run dir. Absent or unparseable is not
    an error: a pre-pool (v1.0) synced dir carries none of these, and a partial
    sync may not have pulled one yet."""
    path = Path(run_dir) / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_jobspec(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the coordinator-authored jobspec.json (kind, name, target, ...;
    runnerd/harness/jobspec.go) from a synced run dir, or None if absent."""
    return _read_run_dir_json(run_dir, "jobspec.json")


def _read_env_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read the coordinator-authored env.json (origin_host, executed_on, ...;
    design 4.2 D23) from a synced run dir, or None if absent."""
    return _read_run_dir_json(run_dir, "env.json")


def _read_process_status(run_dir: Path) -> Optional[str]:
    """Read the coordinator-authored process.json `status` field (the
    runnerd ProcessState projection) from a synced run dir, or None if absent
    or malformed. Used for an evaluate dir's lifecycle status (design 8 D64):
    its run_db.sqlite carries the target's runs row, not the eval job's own,
    so the target's status is not a usable proxy for the eval job's status."""
    data = _read_run_dir_json(run_dir, "process.json")
    if data is None:
        return None
    status = data.get("status")
    return status if isinstance(status, str) else None


# ---------------------------------------------------------------------------
# Collision guard (design 4.3)
# ---------------------------------------------------------------------------


def _guard_collision(dest: sqlite3.Connection, name: str, origin_host: str) -> None:
    # origin_host is the single ownership authority (design 4.3). A non-NULL value
    # arrives from one of two paths that both mean "the runner owns this run": a
    # prior reconcile of a runner-authored run, or `cambia harness push-run` marking
    # a client-local run it pushed up (mark_run_pushed). The guard treats both the
    # same: a matching host is an accepted owner; only a NULL host (a never-pushed
    # local run) is a true collision.
    row = dest.execute("SELECT origin_host FROM runs WHERE name=?", (name,)).fetchone()
    if row is None:
        return  # new name, safe to insert
    existing = row["origin_host"]
    if existing is None:
        raise ReconcilerCollisionError(
            f"run '{name}' already exists as a local run (origin_host IS NULL); "
            "same-name cross-host runs are unsupported in v1 (push-run it first to "
            "transfer ownership before pulling it back)"
        )
    if existing != origin_host:
        raise ReconcilerCollisionError(
            f"run '{name}' already exists with origin_host {existing!r} != "
            f"{origin_host!r}; same-name cross-host runs are unsupported in v1"
        )
    # existing == origin_host -> idempotent re-replay of a run this host owns
    # (reconciled earlier, or pushed up from here by push-run then pulled back).


# ---------------------------------------------------------------------------
# Evaluate-journal merge (design 8 D64)
# ---------------------------------------------------------------------------


def _merge_evaluate_artifacts(job_dir: Path, target_name: str) -> None:
    """Merge an evaluate job's own metrics.jsonl lines and evaluations/iter_N
    contents onto the target run's local dir (design 8 D64).

    job_dir is the evaluate job's own synced run dir; the target's local dir is
    its sibling under the shared local runs dir (the pull.py
    `local_runs_dir/run_name` convention: replay() is handed the job's own
    synced dir, not local_runs_dir directly, so the target dir is derived as
    job_dir.parent / target_name).

    metrics.jsonl is append-only on the source and may be pulled and replayed
    on every tick before the job goes terminal, so the merge is idempotent by
    line content: only source lines not already present at the target are
    appended. evaluations/iter_N files are copied with overwrite, which is
    naturally idempotent (same source bytes every tick).
    """
    target_dir = job_dir.parent / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    job_metrics = job_dir / "metrics.jsonl"
    if job_metrics.exists():
        source_lines = job_metrics.read_text(encoding="utf-8").splitlines()
        if source_lines:
            target_metrics = target_dir / "metrics.jsonl"
            existing_lines: Set[str] = set()
            if target_metrics.exists():
                existing_lines = set(
                    target_metrics.read_text(encoding="utf-8").splitlines()
                )
            new_lines = [ln for ln in source_lines if ln not in existing_lines]
            if new_lines:
                with open(target_metrics, "a", encoding="utf-8") as fh:
                    for ln in new_lines:
                        fh.write(ln + "\n")

    job_evaluations = job_dir / "evaluations"
    if job_evaluations.is_dir():
        target_evaluations = target_dir / "evaluations"
        for iter_dir in sorted(job_evaluations.iterdir()):
            if not iter_dir.is_dir():
                continue
            dest_iter_dir = target_evaluations / iter_dir.name
            dest_iter_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(iter_dir.iterdir()):
                if f.is_file():
                    shutil.copy2(f, dest_iter_dir / f.name)


def _replay_evaluate_merge(
    job_dir: Path,
    dest: sqlite3.Connection,
    src: sqlite3.Connection,
    run: Dict[str, Any],
) -> Dict[str, int]:
    """Merge an evaluate journal onto the local run named by the job's target
    (design 8 D64). Never touches the target's `runs` columns and never
    registers checkpoints from the eval journal: checkpoint_id is re-resolved
    locally by (run_id, iteration), exactly as a normal replay does."""
    target_name = run["name"]
    dest_row = dest.execute("SELECT id FROM runs WHERE name=?", (target_name,)).fetchone()
    if dest_row is None:
        raise ReconcilerValidationError(
            f"evaluate journal targets run {target_name!r}, which has no local "
            "run row yet; replay the target run before its evaluate journal"
        )
    local_run_id = dest_row["id"]

    eval_rows = _read_evals(src, run["_src_id"])
    summary = {"runs": 0, "checkpoints": 0, "evals": 0}
    for row_dict in eval_rows:
        local_ckpt = dest.execute(
            "SELECT id FROM checkpoints WHERE run_id=? AND iteration=?",
            (local_run_id, row_dict["iteration"]),
        ).fetchone()
        local_ckpt_id = local_ckpt["id"] if local_ckpt else None
        insert_eval_result(dest, local_run_id, local_ckpt_id, row_dict)
        summary["evals"] += 1

    _merge_evaluate_artifacts(job_dir, target_name)
    return summary


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def replay(
    synced_run_dir: Union[str, Path],
    dest_db: Union[str, Path, sqlite3.Connection],
    origin_host: str,
) -> Dict[str, int]:
    """Replay a runner's synced run_db.sqlite into the client's authoritative db.

    Single-row replay (design 5.7 D61): a journal carries at most one `runs`
    row. For a normal (train/bench/head-to-head) dir that row must be named for
    the synced run dir itself; for an evaluate dir (jobspec.json kind
    "evaluate", design 8 D64) it must be named for the job's target instead,
    and replay runs in eval-merge mode: eval_results rows, metrics.jsonl lines,
    and evaluations/iter_N contents are merged onto the target's own local run,
    and the target's `runs` row and checkpoints are never touched. A journal
    carrying more than one `runs` row is rejected outright, closing on the
    client the same hole D55 closes on the coordinator.

    Args:
        synced_run_dir: local directory holding the pulled run (the synced run
            dir); must contain run_db.sqlite and is the root for checkpoint path
            re-derivation.
        dest_db: path to the client's cambia_runs.db, or an already-open
            connection (from get_db, so the origin_host/executed_on migration
            is applied).
        origin_host: the source host stamped onto the replayed run (e.g.
            "runner"); must be a non-empty safe identifier. Unused in
            eval-merge mode (the target's origin_host is never touched).

    Returns:
        {"runs": n, "checkpoints": n, "evals": n} counts of rows replayed. In
        eval-merge mode "runs" and "checkpoints" are always 0.

    Raises:
        ReconcilerError: run_db.sqlite missing.
        ReconcilerValidationError: any pulled row fails whitelist/range/name
            checks, the journal carries more than one `runs` row, a row's name
            does not match the synced dir (or the jobspec target for an
            evaluate dir), or an evaluate journal targets a run with no local
            row yet.
        ReconcilerCollisionError: a pulled name collides with a local (or other-host) run.
    """
    origin_host = _validate_origin_host(origin_host)
    run_dir = Path(synced_run_dir)
    src_path = run_dir / "run_db.sqlite"
    if not src_path.exists():
        raise ReconcilerError(f"no run_db.sqlite under {synced_run_dir}")

    owns_dest = not isinstance(dest_db, sqlite3.Connection)
    dest = get_db(str(dest_db)) if owns_dest else dest_db
    src = _open_readonly(src_path)

    try:
        rows = _read_runs(src)
        if len(rows) > 1:
            raise ReconcilerValidationError(
                f"journal under {synced_run_dir} carries {len(rows)} runs rows; "
                "the reconciler replays a single row per journal (design 5.7 D61)"
            )
        if not rows:
            return {"runs": 0, "checkpoints": 0, "evals": 0}
        run = rows[0]

        jobspec = _read_jobspec(run_dir)
        if isinstance(jobspec, dict) and jobspec.get("kind") == "evaluate":
            target = jobspec.get("target")
            if not isinstance(target, str) or not target:
                raise ReconcilerValidationError(
                    "evaluate journal's jobspec.json carries no target"
                )
            target = _validate_run_name(target)
            if run["name"] != target:
                raise ReconcilerValidationError(
                    f"evaluate journal runs row {run['name']!r} does not match "
                    f"jobspec target {target!r} (design 8 D64)"
                )
            return _replay_evaluate_merge(run_dir, dest, src, run)

        expected_name = run_dir.resolve().name
        if run["name"] != expected_name:
            raise ReconcilerValidationError(
                f"runs row name {run['name']!r} does not match the synced run "
                f"dir {expected_name!r} (design 5.7 D61)"
            )
        name = run["name"]

        # Collision guard BEFORE any write for this run: refuse to touch a
        # local run that shares the name (design 4.3).
        _guard_collision(dest, name, origin_host)

        # Read and validate all child rows before any write, so an
        # out-of-range checkpoint/eval rejects the run without a partial write.
        ckpt_rows = _read_checkpoints(src, run["_src_id"])
        eval_rows = _read_evals(src, run["_src_id"])

        env = _read_env_json(run_dir)
        executed_on = _validate_executed_on(
            env.get("executed_on") if isinstance(env, dict) else None
        )

        local_run_id = upsert_run(
            dest,
            name=name,
            algorithm=run["algorithm"],
            status=run["status"],
            tags=run["tags"],
            notes=run["notes"],
            engine_commit_hash=run["engine_commit_hash"],
            origin_host=origin_host,
            executed_on=executed_on,
        )
        summary = {"runs": 1, "checkpoints": 0, "evals": 0}

        for c in ckpt_rows:
            file_path = _rederive_checkpoint_path(
                run_dir, c["iteration"], run["algorithm"], c["file_path"]
            )
            # Force a local stat (file_size_bytes=None) so size reflects the
            # actually-synced file, never the untrusted pulled size.
            ckpt_id = register_checkpoint(
                dest,
                local_run_id,
                c["iteration"],
                file_path,
                file_size_bytes=None,
            )
            # register_checkpoint does not set these; preserve them per row.
            dest.execute(
                "UPDATE checkpoints SET is_best=?, is_retained=?, compressed=? "
                "WHERE id=?",
                (c["is_best"], c["is_retained"], c["compressed"], ckpt_id),
            )
            dest.commit()
            summary["checkpoints"] += 1

        for row_dict in eval_rows:
            # ckpt_id re-resolved LOCALLY; runner surrogate ids never cross stores.
            local_ckpt = dest.execute(
                "SELECT id FROM checkpoints WHERE run_id=? AND iteration=?",
                (local_run_id, row_dict["iteration"]),
            ).fetchone()
            local_ckpt_id = local_ckpt["id"] if local_ckpt else None
            insert_eval_result(dest, local_run_id, local_ckpt_id, row_dict)
            summary["evals"] += 1
        return summary
    finally:
        src.close()
        if owns_dest:
            dest.close()
