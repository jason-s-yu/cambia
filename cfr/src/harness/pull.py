"""
src/harness/pull.py

Pull loop and reconcile orchestration for the `cambia harness watch` command
(cambia-256, design 4.1 / 4.4). The client pulls; the runner never pushes.

Per run, per tick:
  1. rsync the runner's run_db.sqlite down (phase 1); it is the reconciliation
     wire format (design 4.2) and says which checkpoints are retained.
  2. derive the include set from its retained-checkpoint rows (design 4.4);
     `--all-checkpoints` widens to the whole snapshots/ tree.
  3. rsync the run dir with that include set (phase 2). The DiskReservoir
     directory is NEVER pulled (design 4.1): resume locality stays on the runner.
  4. reconciler.replay() the synced dir into the client's authoritative db.
  5. upsert harness_sync (the per-run current-state store the dashboard reads for
     staleness, design 4.5).

Terminal transitions (surfaced by the queue WS or a status poll) trigger an
immediate pull with retry+backoff; a terminal run stays in the active pull set
until its SYNCED status is itself terminal, so a single dropped final pull cannot
freeze a dead run at "running".
"""

import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.harness.reconciler import (
    _ALLOWED_STATUS,
    ReconcilerValidationError,
    _validate_run_name,
)
from src.harness.reconciler import replay as reconciler_replay
from src.run_db import get_db, upsert_harness_sync

RESERVOIR_DIRNAME = "reservoir"
SNAPSHOTS_DIRNAME = "snapshots"

# L10: the enum harness_sync.last_status is clamped to before an upsert. Reuses
# the reconciler's canonical status set (design 2.3: run_db lifecycle values +
# the runnerd job state machine) so the rule is not reforked, and adds the
# ProcessState "starting" value that set omits, covering the full runnerd process
# state machine.
_KNOWN_SYNC_STATUSES: Set[str] = _ALLOWED_STATUS | {"starting"}

# Terminal run states (design 2.3 terminal set + run_db lifecycle completions).
# A run in one of these has no further artifact changes coming.
TERMINAL_STATUSES: Set[str] = {
    "stopped",
    "crashed",
    "canceled",
    "cancelled",
    "failed",
    "completed",
    "finished",
    "done",
}


class PullError(Exception):
    """A pull (rsync/replay) step failed."""


def is_terminal(status: Optional[str]) -> bool:
    return bool(status) and status in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Untrusted-input guards (design 5.7)
# ---------------------------------------------------------------------------


def is_valid_run_name(name: Any) -> bool:
    """Whether `name` clears the canonical Go validateName rules. Reuses the
    reconciler validator so the rule has a single source (H1 / design 5.7)."""
    try:
        _validate_run_name(name)
        return True
    except ReconcilerValidationError:
        return False


def _require_valid_run_name(name: Any) -> str:
    """Raise PullError unless `name` clears the canonical validator. The last
    gate before a runner-supplied name is joined into a local path or an rsync
    argument (H1)."""
    if not is_valid_run_name(name):
        raise PullError(f"refusing unsafe run name {name!r}")
    return name


def sanitize_last_status(status: Optional[str]) -> Optional[str]:
    """Clamp a synced run status to the known enum before it lands in
    harness_sync (L10). A pulled run_db is untrusted (design 5.7), so its status
    column can carry arbitrary text; anything outside the canonical set becomes
    "unknown". None (no status observed yet) is preserved as absence."""
    if status is None:
        return None
    return status if status in _KNOWN_SYNC_STATUSES else "unknown"


# ---------------------------------------------------------------------------
# rsync filter construction (design 4.1 / 4.4)
# ---------------------------------------------------------------------------


def build_pull_filters(
    include_basenames: Optional[Set[str]], all_checkpoints: bool
) -> List[str]:
    """Build rsync filter args for a run-dir pull.

    The DiskReservoir directory is excluded in BOTH modes (design 4.1). Default
    mode keeps only the retained snapshot files named by `include_basenames`
    (plus the best.pt symlink) and drops the rest of snapshots/; `--all-checkpoints`
    widens to the whole snapshots/ tree (reservoir still excluded).

    rsync applies rules first-match-wins, so the retained-file includes precede
    the snapshots/ catch-all exclude.
    """
    filters = [
        f"--exclude=/{RESERVOIR_DIRNAME}/",
        f"--exclude=/{RESERVOIR_DIRNAME}/**",
    ]
    if not all_checkpoints:
        for base in sorted(include_basenames or set()):
            filters.append(f"--include=/{SNAPSHOTS_DIRNAME}/{base}")
        # keep the best-checkpoint symlink and let rsync descend into snapshots/,
        # then drop every non-retained snapshot file.
        filters.append(f"--include=/{SNAPSHOTS_DIRNAME}/best.pt")
        filters.append(f"--include=/{SNAPSHOTS_DIRNAME}/")
        filters.append(f"--exclude=/{SNAPSHOTS_DIRNAME}/**")
    return filters


# ---------------------------------------------------------------------------
# Reading the synced run_db (include-set derivation + status)
# ---------------------------------------------------------------------------


def _connect_synced(db_path: Path) -> sqlite3.Connection:
    """Open a synced (local) run_db for reading. A normal connection (not the
    reconciler's mode=ro) so any pulled -wal frames are folded in; the file is a
    local synced copy, not the in-place untrusted source the reconciler guards."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def derive_include_set(synced_db_path: Path, all_checkpoints: bool) -> Optional[Set[str]]:
    """Derive the retained-checkpoint file basenames from a synced run_db.

    Returns None when `all_checkpoints` is set (no snapshot filtering). Otherwise
    returns the set of basenames of checkpoints flagged is_retained=1 across every
    run in the db (the runner writes one run per run-dir db). An empty set means
    "no retained checkpoints yet"; only best.pt and metadata will be pulled.
    """
    if all_checkpoints:
        return None
    if not synced_db_path.exists():
        return set()
    conn = _connect_synced(synced_db_path)
    try:
        rows = conn.execute(
            "SELECT file_path FROM checkpoints WHERE is_retained=1"
        ).fetchall()
    except sqlite3.Error:
        return set()
    finally:
        conn.close()
    return {os.path.basename(r["file_path"]) for r in rows if r["file_path"]}


def read_run_status(synced_db_path: Path, run_name: str) -> Optional[str]:
    """Read a run's status from the synced run_db (the pulled source of truth for
    the run's own lifecycle; design 4.2)."""
    if not synced_db_path.exists():
        return None
    conn = _connect_synced(synced_db_path)
    try:
        row = conn.execute("SELECT status FROM runs WHERE name=?", (run_name,)).fetchone()
        if row is None:
            # fall back to the single run present in a per-run-dir db
            row = conn.execute(
                "SELECT status FROM runs ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row["status"] if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Runner: the rsync/ssh executor (injectable for tests)
# ---------------------------------------------------------------------------


class Runner:
    """Data-plane executor. Default impl shells rsync over an ssh alias; tests
    inject a fake that copies temp dirs and records calls."""

    def pull_db(self, run_name: str, local_run_dir: Path) -> None:
        raise NotImplementedError

    def pull_run(self, run_name: str, local_run_dir: Path, filters: List[str]) -> None:
        raise NotImplementedError

    def push_run(self, run_name: str, local_run_dir: Path) -> None:
        raise NotImplementedError


class RsyncRunner(Runner):
    """rsync-over-ssh data-plane executor (design 1: ssh only, no new listener)."""

    def __init__(
        self,
        ssh_alias: str,
        runner_runs_dir: str,
        rsync_bin: str = "rsync",
    ):
        # An empty ssh_alias selects local rsync (no "host:" prefix); used for
        # same-host tests. Production always sets the alias (data plane is ssh).
        self._alias = ssh_alias
        self._remote_base = runner_runs_dir.rstrip("/")
        self._rsync = rsync_bin

    def _prefix(self) -> str:
        return f"{self._alias}:" if self._alias else ""

    def _remote(self, run_name: str, trailing_slash: bool = True) -> str:
        suffix = "/" if trailing_slash else ""
        return f"{self._prefix()}{self._remote_base}/{run_name}{suffix}"

    def _run(self, args: List[str]) -> None:
        proc = subprocess.run(args, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PullError(
                f"rsync failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr.strip()}"
            )

    def pull_db(self, run_name: str, local_run_dir: Path) -> None:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        # Pull run_db.sqlite and any WAL siblings so the synced snapshot is
        # internally consistent before replay reads it.
        for suffix in ("", "-wal", "-shm"):
            args = [
                self._rsync,
                "-a",
                "--safe-links",
                "--ignore-missing-args",
                f"{self._prefix()}{self._remote_base}/{run_name}/run_db.sqlite{suffix}",
                str(local_run_dir / f"run_db.sqlite{suffix}"),
            ]
            self._run(args)

    def pull_run(self, run_name: str, local_run_dir: Path, filters: List[str]) -> None:
        local_run_dir.mkdir(parents=True, exist_ok=True)
        args = [
            self._rsync,
            "-a",
            "--safe-links",
            *filters,
            self._remote(run_name),
            str(local_run_dir) + "/",
        ]
        self._run(args)

    def push_run(self, run_name: str, local_run_dir: Path) -> None:
        # Explicit reverse transfer for remote resume (design 2.5): the full run
        # dir INCLUDING the reservoir goes up, since remote resume needs it.
        args = [
            self._rsync,
            "-a",
            "--safe-links",
            str(local_run_dir).rstrip("/") + "/",
            self._remote(run_name),
        ]
        self._run(args)


# ---------------------------------------------------------------------------
# Pull coordinator (design 4.1)
# ---------------------------------------------------------------------------


class PullCoordinator:
    """Orchestrates one run's two-phase pull + replay + harness_sync upsert, with
    retry+backoff for terminal pulls.

    The reconciler replay function and the runner are injected so the loop is
    unit-testable without real rsync or a real control plane.
    """

    def __init__(
        self,
        runner: Runner,
        local_runs_dir: Path,
        dest_conn: sqlite3.Connection,
        origin_host: str,
        replay_fn: Callable[
            [Path, sqlite3.Connection, str], Dict[str, int]
        ] = reconciler_replay,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.runner = runner
        self.local_runs_dir = Path(local_runs_dir)
        self.dest = dest_conn
        self.origin_host = origin_host
        self.replay_fn = replay_fn
        self.sleep_fn = sleep_fn

    def local_run_dir(self, run_name: str) -> Path:
        # H1: the chokepoint where a run name becomes a local path. Every pull /
        # push path routes through here, so validating once guards pull_once,
        # pull_with_retry, and push_run against a runner-supplied traversal name,
        # independent of which caller (watch, cli pull, cli push-run) supplied it.
        _require_valid_run_name(run_name)
        return self.local_runs_dir / run_name

    def pull_once(self, run_name: str, all_checkpoints: bool = False) -> Optional[str]:
        """One full pull->replay->bookkeep cycle. Returns the synced run status."""
        local_dir = self.local_run_dir(run_name)
        db_path = local_dir / "run_db.sqlite"

        # Phase 1: the run_db drives the include set (design 4.2/4.4).
        self.runner.pull_db(run_name, local_dir)
        includes = derive_include_set(db_path, all_checkpoints)
        filters = build_pull_filters(includes, all_checkpoints)

        # Phase 2: the artifact set (reservoir excluded by build_pull_filters).
        self.runner.pull_run(run_name, local_dir, filters)

        # Replay into the authoritative db, then record freshness.
        self.replay_fn(local_dir, self.dest, self.origin_host)
        status = read_run_status(db_path, run_name)
        upsert_harness_sync(
            self.dest, run_name, self.origin_host, sanitize_last_status(status)
        )
        return status

    def pull_with_retry(
        self,
        run_name: str,
        all_checkpoints: bool = False,
        attempts: int = 5,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> Optional[str]:
        """Pull with exponential backoff; used on terminal transitions (design
        4.1: the immediate terminal pull retries with backoff)."""
        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self.pull_once(run_name, all_checkpoints)
            except Exception as exc:  # rsync / replay / validation failure
                last_exc = exc
                if i < attempts - 1:
                    self.sleep_fn(min(base_backoff * (2**i), max_backoff))
        raise PullError(
            f"pull of {run_name!r} failed after {attempts} attempts: {last_exc}"
        )

    def push_run(self, run_name: str) -> None:
        """Push a client-local run dir up for remote resume (design 2.5)."""
        local_dir = self.local_run_dir(run_name)
        if not local_dir.is_dir():
            raise PullError(f"no local run dir to push: {local_dir}")
        self.runner.push_run(run_name, local_dir)


# ---------------------------------------------------------------------------
# The foreground watch loop (design 4.1)
# ---------------------------------------------------------------------------


def watch(
    coordinator: PullCoordinator,
    job_lister: Callable[[], List[Dict[str, Any]]],
    interval_seconds: float = 60.0,
    all_checkpoints: bool = False,
    stop: Optional[Callable[[], bool]] = None,
    on_event: Optional[Callable[[str], None]] = None,
    max_ticks: Optional[int] = None,
) -> None:
    """Foreground pull loop (design 4.1). Not daemonized (out of scope).

    Args:
        coordinator: the pull/replay orchestrator.
        job_lister: returns [{"name", "status"}, ...] from the control plane;
            the source of live runs and terminal transitions.
        interval_seconds: periodic-pull cadence (default 60s).
        stop: optional predicate to break the loop (e.g. a signal flag).
        max_ticks: optional cap on iterations (tests inject a small value).
    """
    stop = stop or (lambda: False)
    log = on_event or (lambda msg: None)

    # active pull set: run_name -> {"terminal_seen": bool}. A run stays until its
    # SYNCED status is terminal (design 4.1).
    active: Dict[str, Dict[str, Any]] = {}
    ticks = 0

    while not stop():
        try:
            jobs = job_lister()
        except Exception as exc:  # control plane unreachable: stay stale, retry
            log(f"job list failed: {exc}")
            jobs = []

        for job in jobs:
            name = job.get("name")
            status = job.get("status")
            if not name:
                continue
            if not is_valid_run_name(name):
                # H1: a hostile job name never reaches the pull set, a local
                # path, or an rsync argument. Skip loudly, keep watching.
                log(f"skipping job with unsafe name {name!r}")
                continue
            entry = active.setdefault(name, {"terminal_seen": False})
            if is_terminal(status) and not entry["terminal_seen"]:
                entry["terminal_seen"] = True
                log(f"terminal transition {name} -> {status}; immediate pull")
                try:
                    synced = coordinator.pull_with_retry(name, all_checkpoints)
                    if is_terminal(synced):
                        active.pop(name, None)
                        log(f"{name} synced terminal ({synced}); dropped")
                except Exception as exc:
                    log(f"terminal pull of {name} failed: {exc}")

        # periodic pull of everything still active
        for name in list(active):
            try:
                synced = coordinator.pull_once(name, all_checkpoints)
            except Exception as exc:
                log(f"periodic pull of {name} failed: {exc}")
                continue
            if active[name]["terminal_seen"] and is_terminal(synced):
                active.pop(name, None)
                log(f"{name} synced terminal ({synced}); dropped")

        ticks += 1
        if max_ticks is not None and ticks >= max_ticks:
            return
        if stop():
            return
        coordinator.sleep_fn(interval_seconds)
