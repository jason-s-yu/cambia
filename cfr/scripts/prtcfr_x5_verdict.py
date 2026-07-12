"""cfr/scripts/prtcfr_x5_verdict.py

X5 terminal gate verdict analysis (v0.4 Phase 3, P3W3): renders the Phase 3
contract acceptance criteria (G2 + G3-ordering + G3-trend + budget-reached) for
the full-game PRT-CFR run on the deployable SD-CFR mixture, plus the
budget-versus-exploitability curve.

X5 is X4 at scale, not a new estimator. The slope-CI math (OLS + Student's t
confidence interval, no scipy dependency) is reused verbatim from
``cfr/scripts/prtcfr_x4_verdict.py`` (imported, not re-derived), as is the
metrics.jsonl loader and the informational mean_imp(5) floor read path.

Data sources (all reconciled onto pangu before this runs):
  - ``<run_dir>/metrics.jsonl``: the in-loop battery series. The G3-trend leg
    reads ``tier_a_lbr`` per iteration (null off the battery cadence, skipped);
    cumulative traversals are ``iteration * K`` with K read from the run config
    (``<run_dir>/config.yaml`` prt_cfr.k_games_per_iter, or ``--k-games``).
  - the reconciled run_db (``src/run_db.py``, the sole eval source):
      * mean_imp(5) per-baseline rows -> the informational floor (gates nothing).
      * the G2 per-baseline tie-aware not-a-loss rows (p0_wins / p1_wins / ties /
        games_played, agent-relative, latest iteration per baseline).
      * the MIXTURE exploitability rows under the dedicated metric baseline keys
        ``lbr_tier_b`` (primary) and ``ismcts_br`` (confirming), the value riding
        in the ``win_rate`` column exactly like the existing ``nashconv``
        stability rows (P3W2 determination, @chief-ruled: no harness kind emits a
        pure exploitability eval, so a post-pull local runner persists these rows
        via insert_eval_result). Mixture exploitability is never read from a JSON
        side-file.
  - the P3W5 baseline exploitability reference table
    (``cfr/runs/reference/baseline_exploitability.json``, ``--baseline-ref``
    override): the frozen per-baseline reference exploitability the G3-ordering
    leg compares the mixture against. The file may not exist yet; tests fixture it.

Gates (all on the deployable SD-CFR mixture at cumulative budget >= 10x DESCA):
  G2      per-baseline tie-aware not-a-loss (win + 0.5*tie) >= 0.50 vs each
          mean_imp baseline. The strategic legs (imperfect_greedy,
          memory_heuristic, aggressive_snap) are the binding sub-gate; the two
          random legs are reported non-binding. The pass rule is selectable via
          ``--g2-rule`` because the user ruling is pending: default is the lower
          CI bound clearing 0.50 on the strategic legs and the point estimate on
          the random legs; ``point`` / ``lower-ci`` force one rule everywhere. The
          emitted JSON records which rule evaluated each leg.
  G3-ord  mixture exploitability strictly below each baseline's reference
          exploitability with non-overlapping CIs. Tier-B LBR (``lbr_tier_b``) is
          the primary binding estimator; ISMCTS-BR (``ismcts_br``) is confirming:
          its disagreement with Tier-B LBR is raised as a flag, not a veto,
          unless ``--ismcts-binding`` is set.
  G3-trnd in-loop Tier-A LBR OLS slope over the in-budget 1000-iter window with a
          95% CI upper bound < 0, and the observed magnitude clearing the window's
          minimum detectable slope (computed from the in-window residual variance
          and point count). The window is the one whose end iteration is the
          largest available and whose start is at or beyond the 10x anchor
          iteration (ceil(2.0e7 / K)); cumulative traversals at the window start
          must be >= 2.0e7.

Budget curve: Tier-A LBR versus cumulative traversals, emitted in the verdict
JSON regardless of PASS or FAIL, plus a PNG plot when matplotlib is importable.
A flat-curve kill still writes the curve and exits nonzero (the kill deliverable).

Emits one structured verdict JSON (``--out`` to persist) plus a human summary.
Exits nonzero if any binding leg FAILs. mean_imp(5) is informational only.

Usage:
  cd cfr && python scripts/prtcfr_x5_verdict.py runs/v0.4-x5-full
  cd cfr && python scripts/prtcfr_x5_verdict.py runs/v0.4-x5-full \
      --out runs/v0.4-x5-full/x5_verdict.json --g2-rule lower-ci
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CFR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CFR_ROOT)

# Reuse the X4 verdict-script slope-CI math and metrics/floor readers verbatim
# (Phase 3 contract Interfaces: "The x4 verdict-script slope-CI math ... reused
# for the G3-trend leg"). Do not re-derive.
from scripts import prtcfr_x4_verdict as x4  # noqa: E402
from src.evaluate_agents import MEAN_IMP_BASELINES  # noqa: E402
from src.run_db import get_db  # noqa: E402

# Re-export the reused helpers so tests and callers reference one surface.
ols_slope_ci = x4.ols_slope_ci
load_metrics_rows = x4.load_metrics_rows
read_mean_imp5_floor = x4.read_mean_imp5_floor
_t_ppf = x4._t_ppf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 10x DESCA (1000 iters x K=2000 = 2.0e6 traversals) = 2.0e7 traversals.
BUDGET_ANCHOR_TRAVERSALS: float = 2.0e7

# The G3-trend window width (iterations). The contract fixes it at 1000.
DEFAULT_WINDOW_WIDTH: int = 1000

# Default statistical power for the minimum-detectable-slope bar. 0.80 is the
# conventional MDE power; power=0.5 collapses the MDS to the CI half-width, i.e.
# "observed magnitude clears MDS" becomes exactly "CI excludes zero".
DEFAULT_TREND_POWER: float = 0.80

# The five mean_imp baselines split into binding (strategic) and non-binding
# (random) legs for G2. imperfect_greedy / memory_heuristic / aggressive_snap
# are the strategic legs the p2-redesign amendment makes the binding sub-gate.
STRATEGIC_BASELINES: Tuple[str, ...] = (
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
)
RANDOM_BASELINES: Tuple[str, ...] = ("random_no_cambia", "random_late_cambia")

# run_db metric baseline keys for the mixture exploitability rows (P3W2 pin).
EXPLOIT_PRIMARY_KEY: str = "lbr_tier_b"
EXPLOIT_CONFIRMING_KEY: str = "ismcts_br"

# The frozen G3-ordering reference table (P3W5), relative to the cfr/ root.
DEFAULT_BASELINE_REF: str = os.path.join(
    _CFR_ROOT, "runs", "reference", "baseline_exploitability.json"
)

# Recognized reference-table metadata keys (skipped when the baselines map is at
# the top level rather than nested under a "baselines" key).
_REF_META_KEYS = frozenset(
    {"schema_version", "rule_profile", "generated_at", "estimators", "notes", "games", "rows"}
)


# ---------------------------------------------------------------------------
# K (traversals per iteration): read from the run config, never hardcoded
# ---------------------------------------------------------------------------


def resolve_k_games(
    run_dir: Path,
    k_override: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Tuple[Optional[int], str]:
    """Resolve K = traversals per iteration for the run.

    Priority: explicit ``k_override`` (CLI/test) > ``<run_dir>/config.yaml``
    prt_cfr.k_games_per_iter (the materialized run config cli.py writes) > the
    run_db config_snapshots yaml. Returns ``(k, source)``; ``(None, reason)`` if
    K cannot be resolved (the budget leg then FAILs with a clear message rather
    than the anchor iteration being guessed).
    """
    if k_override is not None:
        return int(k_override), "cli --k-games"

    cfg_path = Path(run_dir) / "config.yaml"
    if cfg_path.exists():
        k = _k_from_yaml_text(cfg_path.read_text(encoding="utf-8"))
        if k is not None:
            return k, str(cfg_path)

    # Fallback: the run_db config snapshot (harness reconciler / trainer writes it).
    try:
        db = get_db(db_path)
        run_row = db.execute(
            "SELECT id FROM runs WHERE name=?", (Path(run_dir).name,)
        ).fetchone()
        if run_row is not None:
            snap = db.execute(
                "SELECT config_yaml FROM config_snapshots WHERE run_id=? "
                "ORDER BY id DESC LIMIT 1",
                (run_row["id"],),
            ).fetchone()
            if snap is not None and snap["config_yaml"]:
                k = _k_from_yaml_text(snap["config_yaml"])
                if k is not None:
                    return k, "run_db config_snapshots"
    except sqlite3.Error:
        pass

    return None, (
        "K (prt_cfr.k_games_per_iter) not found in %s/config.yaml or run_db; "
        "pass --k-games" % run_dir
    )


def _k_from_yaml_text(text: str) -> Optional[int]:
    """Extract prt_cfr.k_games_per_iter (nested or top-level) from a config YAML."""
    try:
        import yaml

        cfg = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    prt = cfg.get("prt_cfr")
    if isinstance(prt, dict) and prt.get("k_games_per_iter") is not None:
        return int(prt["k_games_per_iter"])
    if cfg.get("k_games_per_iter") is not None:
        return int(cfg["k_games_per_iter"])
    return None


# ---------------------------------------------------------------------------
# run_db eval-row reader (latest iteration per baseline key)
# ---------------------------------------------------------------------------


def _open_run(run_dir: Path, db_path: Optional[str]) -> Tuple[sqlite3.Connection, Optional[int]]:
    db = get_db(db_path)
    row = db.execute(
        "SELECT id FROM runs WHERE name=?", (Path(run_dir).name,)
    ).fetchone()
    return db, (row["id"] if row is not None else None)


def _latest_eval_rows(
    db: sqlite3.Connection,
    run_id: int,
    baselines: Sequence[str],
    iteration: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Latest eval_results row per baseline key (or the exact ``iteration``).

    Matches how prtcfr_x4_verdict.py aggregates the agent's rows: one row per
    baseline, the largest iteration winning (ties keep the last seen). When
    ``iteration`` is given, only rows at that iteration are considered.
    """
    if not baselines:
        return {}
    placeholders = ",".join("?" * len(baselines))
    sql = (
        "SELECT iteration, baseline, win_rate, ci_low, ci_high, games_played, "
        "p0_wins, p1_wins, ties FROM eval_results WHERE run_id=? "
        "AND baseline IN (%s)" % placeholders
    )
    params: List[Any] = [run_id, *baselines]
    if iteration is not None:
        sql += " AND iteration=?"
        params.append(iteration)
    rows = db.execute(sql, params).fetchall()
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bl = row["baseline"]
        if bl not in latest or row["iteration"] > latest[bl]["iteration"]:
            latest[bl] = dict(row)
    return latest


# ---------------------------------------------------------------------------
# G2: per-baseline tie-aware not-a-loss
# ---------------------------------------------------------------------------


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion ``p_hat`` over ``n`` trials.

    The tie-aware not-a-loss rate is a [0,1] proportion (win=1, tie=0.5, loss=0),
    so the Wilson interval on ``(p_hat, n)`` is the same construction the eval
    persistence uses for the raw win rate. Clamped to [0,1].
    """
    if n <= 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _g2_rule_for_leg(g2_rule: str, baseline: str) -> str:
    """Resolve the per-leg pass rule ("point" or "lower-ci")."""
    if g2_rule == "point":
        return "point"
    if g2_rule == "lower-ci":
        return "lower-ci"
    # default: lower CI bound on the strategic legs, point on the random legs.
    return "lower-ci" if baseline in STRATEGIC_BASELINES else "point"


def compute_g2(
    run_dir: Path,
    db_path: Optional[str] = None,
    g2_rule: str = "default",
    iteration: Optional[int] = None,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """G2: per-baseline tie-aware not-a-loss with CI and a strategic-binding verdict."""
    db, run_id = _open_run(run_dir, db_path)
    legs: Dict[str, Any] = {}
    if run_id is None:
        return {
            "available": False,
            "rule": g2_rule,
            "threshold": threshold,
            "legs": {},
            "strategic_pass": False,
            "all_five_pass": False,
            "verdict": "FAIL",
            "note": "no run_db row named '%s'" % Path(run_dir).name,
        }

    rows = _latest_eval_rows(db, run_id, MEAN_IMP_BASELINES, iteration=iteration)
    for baseline in MEAN_IMP_BASELINES:
        row = rows.get(baseline)
        rule = _g2_rule_for_leg(g2_rule, baseline)
        binding = baseline in STRATEGIC_BASELINES
        if row is None:
            legs[baseline] = {
                "present": False, "binding": binding, "rule": rule,
                "not_a_loss": None, "ci_low": None, "ci_high": None,
                "games_played": None, "tie_aware": False, "pass": False,
                "note": "no eval_results row for this baseline",
            }
            continue
        n = row.get("games_played") or 0
        p0 = row.get("p0_wins")
        ties = row.get("ties")
        if p0 is not None and ties is not None and n > 0:
            not_a_loss = (p0 + 0.5 * ties) / n
            tie_aware = True
        elif row.get("win_rate") is not None and n > 0:
            # Older/partial row without seat counts: fall back to the raw win rate
            # (not tie-aware; ties are not credited). Flagged so the reader knows.
            not_a_loss = float(row["win_rate"])
            tie_aware = False
        else:
            legs[baseline] = {
                "present": True, "binding": binding, "rule": rule,
                "not_a_loss": None, "ci_low": None, "ci_high": None,
                "games_played": n, "tie_aware": False, "pass": False,
                "note": "row present but games_played/seat counts missing",
            }
            continue
        ci_low, ci_high = wilson_ci(not_a_loss, n)
        leg_pass = (not_a_loss >= threshold) if rule == "point" else (ci_low >= threshold)
        legs[baseline] = {
            "present": True, "binding": binding, "rule": rule,
            "not_a_loss": not_a_loss, "ci_low": ci_low, "ci_high": ci_high,
            "games_played": n, "iteration": row.get("iteration"),
            "p0_wins": p0, "p1_wins": row.get("p1_wins"), "ties": ties,
            "tie_aware": tie_aware, "pass": bool(leg_pass),
        }

    strategic_pass = all(legs[b]["pass"] for b in STRATEGIC_BASELINES)
    all_five_pass = all(legs[b]["pass"] for b in MEAN_IMP_BASELINES)
    return {
        "available": True,
        "rule": g2_rule,
        "threshold": threshold,
        "legs": legs,
        "strategic_pass": strategic_pass,
        "all_five_pass": all_five_pass,
        # The strategic legs are the binding sub-gate: passing the random legs
        # while failing a strategic leg is a G2 FAIL.
        "verdict": "PASS" if strategic_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# G3-ordering: mixture exploitability vs the baseline reference table
# ---------------------------------------------------------------------------


def _baselines_from_rows(rows: Sequence[Any]) -> Dict[str, Any]:
    """Convert the canonical row-list reference table into the per-baseline map.

    Builds ``{baseline: {estimator: {"exploitability", "ci_low", "ci_high",
    "games"}}}`` from flat rows (``baseline``, ``estimator``, ``value``,
    ``ci_low``, ``ci_high``, ``sample_count``) -- the producer's
    (``scripts/prtcfr_baseline_exploitability.py``) row shape. When duplicate
    ``(baseline, estimator)`` rows exist, the row with the latest ``timestamp``
    wins (ISO-8601 UTC strings sort lexicographically; ties keep the later row
    in list order).
    """
    latest: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline")
        estimator = row.get("estimator")
        if baseline is None or estimator is None:
            continue
        key = (baseline, estimator)
        ts = row.get("timestamp") or ""
        existing = latest.get(key)
        if existing is None or ts >= existing[0]:
            latest[key] = (ts, row)

    baselines: Dict[str, Any] = {}
    for (baseline, estimator), (_, row) in latest.items():
        baselines.setdefault(baseline, {})[estimator] = {
            "exploitability": row.get("value"),
            "ci_low": row.get("ci_low"),
            "ci_high": row.get("ci_high"),
            "games": row.get("sample_count"),
        }
    return baselines


def read_baseline_reference(path: str) -> Dict[str, Any]:
    """Load the P3W5 baseline exploitability reference table.

    Accepts the canonical ``{"schema_version": 1, "rows": [...]}`` row-list
    shape (the producer's ``write_table`` format) as the primary format: each
    row is ``{baseline, estimator, value, ci_low, ci_high, sample_count, ...}``,
    converted into the per-baseline ``{estimator: {exploitability, ci_low,
    ci_high, games}}`` structure via ``_baselines_from_rows`` (latest row wins
    per (baseline, estimator) on duplicates). Falls back to a nested
    ``{"baselines": {name: {estimator: {...}}}}`` map or a top-level
    ``{name: {estimator: {...}}}`` map (metadata keys skipped) for callers not
    on the canonical schema. Each estimator entry carries ``exploitability``
    and, when available, ``ci_low``/``ci_high``.
    Returns ``{"available": bool, "baselines": {...}, "note"/"error"}``.
    """
    p = Path(path)
    if not p.exists():
        return {
            "available": False, "path": str(p), "baselines": {},
            "note": "baseline reference table not found at %s (P3W5 not landed?)" % p,
        }
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "available": False, "path": str(p), "baselines": {},
            "error": "failed to parse baseline reference table: %s" % exc,
        }
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        baselines = _baselines_from_rows(data["rows"])
    elif isinstance(data, dict) and isinstance(data.get("baselines"), dict):
        baselines = data["baselines"]
    elif isinstance(data, dict):
        baselines = {k: v for k, v in data.items() if k not in _REF_META_KEYS}
    else:
        return {
            "available": False, "path": str(p), "baselines": {},
            "error": "unexpected reference table shape (not a dict)",
        }
    return {
        "available": True,
        "path": str(p),
        "rule_profile": data.get("rule_profile") if isinstance(data, dict) else None,
        "baselines": baselines,
    }


def _estimator_entry(baseline_ref_row: Any, estimator_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(baseline_ref_row, dict):
        return None
    entry = baseline_ref_row.get(estimator_key)
    return entry if isinstance(entry, dict) else None


def _compare_ordering(
    mixture: Optional[Dict[str, Any]], ref: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compare one mixture exploitability against one baseline reference.

    Pass requires the mixture strictly below the baseline with non-overlapping
    CIs: ``mixture.ci_high < ref.ci_low``. Missing CIs downgrade to a point-only
    comparison that cannot confirm non-overlap (pass=False, overlap=None).
    """
    if mixture is None or ref is None:
        return {"pass": False, "overlap": None, "note": "missing mixture or reference value"}
    m_val = mixture.get("exploitability")
    r_val = ref.get("exploitability")
    m_hi = mixture.get("ci_high")
    r_lo = ref.get("ci_low")
    point_below = (m_val is not None and r_val is not None and m_val < r_val)
    if m_hi is None or r_lo is None:
        return {
            "pass": False, "overlap": None, "point_below": point_below,
            "mixture_exploitability": m_val, "reference_exploitability": r_val,
            "note": "CI bounds missing; cannot confirm non-overlap",
        }
    non_overlap_below = m_hi < r_lo
    return {
        "pass": bool(non_overlap_below),
        "overlap": (not non_overlap_below),
        "point_below": point_below,
        "mixture_exploitability": m_val, "mixture_ci_high": m_hi,
        "reference_exploitability": r_val, "reference_ci_low": r_lo,
    }


def compute_g3_ordering(
    run_dir: Path,
    db_path: Optional[str] = None,
    baseline_ref_path: str = DEFAULT_BASELINE_REF,
    ismcts_binding: bool = False,
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    """G3-ordering: mixture exploitability below each baseline's reference.

    Tier-B LBR (``lbr_tier_b``) is the primary binding estimator over all five
    baselines. ISMCTS-BR (``ismcts_br``) is confirming: per-leg disagreement with
    the primary verdict is flagged; it gates only under ``ismcts_binding``.
    """
    db, run_id = _open_run(run_dir, db_path)
    ref = read_baseline_reference(baseline_ref_path)

    mixture: Dict[str, Optional[Dict[str, Any]]] = {
        EXPLOIT_PRIMARY_KEY: None,
        EXPLOIT_CONFIRMING_KEY: None,
    }
    if run_id is not None:
        rows = _latest_eval_rows(
            db, run_id, [EXPLOIT_PRIMARY_KEY, EXPLOIT_CONFIRMING_KEY], iteration=iteration
        )
        for key in (EXPLOIT_PRIMARY_KEY, EXPLOIT_CONFIRMING_KEY):
            row = rows.get(key)
            if row is not None:
                mixture[key] = {
                    "exploitability": row.get("win_rate"),
                    "ci_low": row.get("ci_low"),
                    "ci_high": row.get("ci_high"),
                    "games_played": row.get("games_played"),
                    "iteration": row.get("iteration"),
                }

    legs: Dict[str, Any] = {}
    flags: List[str] = []
    for baseline in MEAN_IMP_BASELINES:
        ref_row = ref["baselines"].get(baseline) if ref.get("available") else None
        primary = _compare_ordering(
            mixture[EXPLOIT_PRIMARY_KEY], _estimator_entry(ref_row, EXPLOIT_PRIMARY_KEY)
        )
        confirming = _compare_ordering(
            mixture[EXPLOIT_CONFIRMING_KEY], _estimator_entry(ref_row, EXPLOIT_CONFIRMING_KEY)
        )
        confirming_available = (
            mixture[EXPLOIT_CONFIRMING_KEY] is not None
            and _estimator_entry(ref_row, EXPLOIT_CONFIRMING_KEY) is not None
        )
        disagreement = confirming_available and (confirming["pass"] != primary["pass"])
        if disagreement:
            flags.append(
                "ISMCTS-BR disagrees with Tier-B LBR on %s (primary=%s, confirming=%s)"
                % (baseline, primary["pass"], confirming["pass"])
            )
        if primary.get("overlap"):
            flags.append("Tier-B LBR CI overlap on binding comparison: %s" % baseline)
        legs[baseline] = {
            "primary": primary,
            "confirming": confirming,
            "confirming_available": confirming_available,
            "disagreement_flag": disagreement,
        }

    # Binding set: the primary (Tier-B LBR) comparisons for all five baselines,
    # plus the confirming comparisons only when the user rules ISMCTS-BR binding.
    primary_pass = all(legs[b]["primary"]["pass"] for b in MEAN_IMP_BASELINES)
    confirming_pass = all(
        legs[b]["confirming"]["pass"]
        for b in MEAN_IMP_BASELINES
        if legs[b]["confirming_available"]
    )
    if ismcts_binding:
        overall = primary_pass and confirming_pass
    else:
        overall = primary_pass

    available = (
        ref.get("available", False)
        and mixture[EXPLOIT_PRIMARY_KEY] is not None
    )
    return {
        "available": available,
        "ismcts_binding": ismcts_binding,
        "reference_path": ref.get("path"),
        "reference_available": ref.get("available", False),
        "mixture": mixture,
        "legs": legs,
        "flags": flags,
        "primary_pass": primary_pass,
        "confirming_pass": confirming_pass,
        "verdict": ("PASS" if (available and overall) else "FAIL"),
    }


# ---------------------------------------------------------------------------
# G3-trend: in-budget window selection, OLS slope CI, minimum detectable slope
# ---------------------------------------------------------------------------


def _lbr_points(rows: Sequence[Dict[str, Any]], field: str = "tier_a_lbr") -> List[Tuple[int, float]]:
    pts = [
        (int(row["iteration"]), float(row[field]))
        for row in rows
        if "iteration" in row and field in row and row[field] is not None
    ]
    pts.sort(key=lambda pair: pair[0])
    return pts


def select_trend_window(
    rows: Sequence[Dict[str, Any]],
    k_games: int,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    anchor_traversals: float = BUDGET_ANCHOR_TRAVERSALS,
    field: str = "tier_a_lbr",
) -> Dict[str, Any]:
    """Select the in-budget 1000-iter trend window.

    The window ends at the largest available Tier-A LBR iteration; its nominal
    start is ``end - window_width``. It is valid only if the start is at or
    beyond the 10x anchor iteration ``ceil(anchor_traversals / K)`` (so every
    in-window point sits in budget), and the cumulative traversals at the window
    start (``start * K``) must reach ``anchor_traversals``.
    """
    anchor_iter = math.ceil(anchor_traversals / k_games)
    pts = _lbr_points(rows, field=field)
    if not pts:
        return {
            "valid": False, "anchor_iter": anchor_iter, "k_games": k_games,
            "window_width": window_width, "end_iter": None, "start_iter": None,
            "n_in_window": 0, "points": [], "budget_start_traversals": None,
            "budget_ok": False, "note": "no Tier-A LBR points in metrics.jsonl",
        }
    end_iter = pts[-1][0]
    start_iter = end_iter - window_width
    in_window = [(it, v) for it, v in pts if start_iter <= it <= end_iter]
    budget_start_traversals = start_iter * k_games
    valid = start_iter >= anchor_iter
    budget_ok = budget_start_traversals >= anchor_traversals
    return {
        "valid": bool(valid),
        "anchor_iter": anchor_iter,
        "k_games": k_games,
        "window_width": window_width,
        "end_iter": end_iter,
        "start_iter": start_iter,
        "n_in_window": len(in_window),
        "points": in_window,
        "budget_start_traversals": budget_start_traversals,
        "budget_ok": bool(budget_ok),
        "note": (
            "window start %d < 10x anchor iter %d (budget not reached at window)"
            % (start_iter, anchor_iter)
            if not valid
            else None
        ),
    }


def minimum_detectable_slope(
    slope_info: Dict[str, Any],
    power: float = DEFAULT_TREND_POWER,
    confidence: float = 0.95,
) -> Optional[float]:
    """Minimum detectable slope magnitude for the window's design.

    MDS = (t_{1-alpha/2, df} + t_{power, df}) * SE(slope), the standard
    minimum-detectable-effect for a two-sided test at significance ``1-confidence``
    and the given ``power``. SE(slope) already folds in the in-window residual
    variance and the x-spread (point count and cadence). At ``power=0.5`` the
    second term is zero and MDS is the CI half-width (equivalent to "CI excludes
    zero"). Returns None if the slope fit is degenerate.
    """
    se = slope_info.get("se")
    df = slope_info.get("df")
    if se is None or df is None or df <= 0:
        return None
    alpha = 1.0 - confidence
    t_alpha = _t_ppf(1.0 - alpha / 2.0, df)
    t_beta = _t_ppf(power, df) if power > 0.5 else 0.0
    return (t_alpha + t_beta) * se


def compute_g3_trend(
    rows: Sequence[Dict[str, Any]],
    k_games: int,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    trend_power: float = DEFAULT_TREND_POWER,
    anchor_traversals: float = BUDGET_ANCHOR_TRAVERSALS,
) -> Dict[str, Any]:
    """G3-trend: negative in-budget LBR slope clearing the minimum detectable slope."""
    window = select_trend_window(
        rows, k_games, window_width=window_width, anchor_traversals=anchor_traversals
    )
    xs = [float(it) for it, _ in window["points"]]
    ys = [float(v) for _, v in window["points"]]
    slope_info = ols_slope_ci(xs, ys)
    mds = minimum_detectable_slope(slope_info, power=trend_power)

    slope = slope_info.get("slope")
    ci_high = slope_info.get("ci_high")
    ci_negative = ci_high is not None and ci_high < 0.0
    clears_mds = (
        slope is not None and mds is not None and abs(slope) >= mds
    )
    trend_pass = bool(
        window["valid"] and window["budget_ok"] and ci_negative and clears_mds
    )
    return {
        "window": window,
        "slope_ci": slope_info,
        "trend_power": trend_power,
        "minimum_detectable_slope": mds,
        "observed_slope_magnitude": (abs(slope) if slope is not None else None),
        "ci_upper_negative": bool(ci_negative),
        "observed_clears_mds": bool(clears_mds),
        "budget_reached": bool(window["budget_ok"]),
        "verdict": "PASS" if trend_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Budget curve: Tier-A LBR versus cumulative traversals
# ---------------------------------------------------------------------------


def budget_curve_table(
    rows: Sequence[Dict[str, Any]], k_games: int, field: str = "tier_a_lbr"
) -> List[Dict[str, Any]]:
    """Full exploitability-vs-budget curve across all in-loop battery points."""
    return [
        {
            "iteration": it,
            "cumulative_traversals": it * k_games,
            "tier_a_lbr": v,
        }
        for it, v in _lbr_points(rows, field=field)
    ]


def write_budget_curve_plot(
    curve: Sequence[Dict[str, Any]], out_path: Path
) -> Optional[str]:
    """Write a PNG plot of the budget curve if matplotlib is importable.

    Returns the written path, or None (with the plot silently skipped) when
    matplotlib is absent or there is nothing to plot. The table in the verdict
    JSON is the authoritative artifact; the plot is a convenience.
    """
    if not curve:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    xs = [pt["cumulative_traversals"] for pt in curve]
    ys = [pt["tier_a_lbr"] for pt in curve]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker=".", linewidth=1.0)
    ax.axvline(BUDGET_ANCHOR_TRAVERSALS, color="gray", linestyle="--", linewidth=0.8,
               label="10x DESCA anchor (2.0e7)")
    ax.set_xlabel("cumulative traversals (iteration * K)")
    ax.set_ylabel("Tier-A LBR exploitability")
    ax.set_title("X5 budget curve: exploitability vs traversals")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    finally:
        plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------


def compute_verdict(
    run_dir: str,
    db_path: Optional[str] = None,
    baseline_ref_path: str = DEFAULT_BASELINE_REF,
    k_games: Optional[int] = None,
    g2_rule: str = "default",
    ismcts_binding: bool = False,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    trend_power: float = DEFAULT_TREND_POWER,
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    run_dir_path = Path(run_dir)
    metrics_path = run_dir_path / "metrics.jsonl"
    rows = load_metrics_rows(metrics_path)

    k_resolved, k_source = resolve_k_games(run_dir_path, k_override=k_games, db_path=db_path)

    g2 = compute_g2(run_dir_path, db_path=db_path, g2_rule=g2_rule, iteration=iteration)
    g3_ord = compute_g3_ordering(
        run_dir_path, db_path=db_path, baseline_ref_path=baseline_ref_path,
        ismcts_binding=ismcts_binding, iteration=iteration,
    )

    if k_resolved is None:
        g3_trend: Dict[str, Any] = {
            "verdict": "FAIL",
            "budget_reached": False,
            "error": k_source,
            "window": None, "slope_ci": None,
            "minimum_detectable_slope": None,
        }
        budget_curve: List[Dict[str, Any]] = []
    else:
        g3_trend = compute_g3_trend(
            rows, k_resolved, window_width=window_width, trend_power=trend_power
        )
        budget_curve = budget_curve_table(rows, k_resolved)

    mean_imp5_floor = read_mean_imp5_floor(run_dir_path, db_path)

    # Binding legs: G2 strategic, G3-ordering (primary + optionally confirming),
    # G3-trend (incl. the in-budget-window / budget-reached assertion).
    overall_pass = (
        g2["verdict"] == "PASS"
        and g3_ord["verdict"] == "PASS"
        and g3_trend["verdict"] == "PASS"
    )

    return {
        "run_dir": str(run_dir_path),
        "metrics_path": str(metrics_path),
        "n_metrics_rows": len(rows),
        "k_games": k_resolved,
        "k_source": k_source,
        "budget_anchor_traversals": BUDGET_ANCHOR_TRAVERSALS,
        "g2_not_a_loss": g2,
        "g3_ordering": g3_ord,
        "g3_trend": g3_trend,
        "budget_curve": budget_curve,
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "mean_imp5_floor": mean_imp5_floor,
    }


# ---------------------------------------------------------------------------
# Human summary
# ---------------------------------------------------------------------------


def human_summary(verdict: Dict[str, Any]) -> str:
    lines = [
        "X5 terminal gate verdict (Phase 3 contract) -- run_dir=%s" % verdict["run_dir"],
        "metrics rows: %d (%s)" % (verdict["n_metrics_rows"], verdict["metrics_path"]),
        "K (traversals/iter): %s [source: %s]"
        % (verdict.get("k_games"), verdict.get("k_source")),
        "",
    ]

    # G2
    g2 = verdict["g2_not_a_loss"]
    lines.append(
        "G2 tie-aware not-a-loss >= 0.50 (rule=%s; strategic legs binding) [%s]"
        % (g2["rule"], g2["verdict"])
    )
    if not g2.get("available"):
        lines.append("   %s" % g2.get("note", "unavailable"))
    for baseline in MEAN_IMP_BASELINES:
        leg = g2["legs"].get(baseline)
        if leg is None:
            continue
        tag = "strategic" if leg.get("binding") else "random"
        lines.append(
            "   %-20s [%s] not_a_loss=%s ci=[%s, %s] rule=%s games=%s tie_aware=%s -> %s"
            % (
                baseline, tag, _fmt(leg.get("not_a_loss")),
                _fmt(leg.get("ci_low")), _fmt(leg.get("ci_high")),
                leg.get("rule"), leg.get("games_played"), leg.get("tie_aware"),
                "PASS" if leg.get("pass") else "FAIL",
            )
        )

    # G3-ordering
    lines.append("")
    g3o = verdict["g3_ordering"]
    lines.append(
        "G3-ordering mixture exploitability below baselines (Tier-B LBR primary, "
        "ISMCTS-BR %s) [%s]"
        % ("binding" if g3o.get("ismcts_binding") else "confirming", g3o["verdict"])
    )
    if not g3o.get("reference_available"):
        lines.append("   reference table unavailable: %s" % g3o.get("reference_path"))
    for baseline in MEAN_IMP_BASELINES:
        leg = g3o["legs"].get(baseline)
        if leg is None:
            continue
        prim = leg["primary"]
        conf = leg["confirming"]
        lines.append(
            "   %-20s tierB: mix=%s vs ref=%s -> %s%s | ismcts: mix=%s vs ref=%s -> %s"
            % (
                baseline,
                _fmt(prim.get("mixture_exploitability")),
                _fmt(prim.get("reference_exploitability")),
                "PASS" if prim.get("pass") else "FAIL",
                " (CI overlap)" if prim.get("overlap") else "",
                _fmt(conf.get("mixture_exploitability")),
                _fmt(conf.get("reference_exploitability")),
                ("PASS" if conf.get("pass") else "FAIL")
                if leg.get("confirming_available") else "n/a",
            )
        )
    for flag in g3o.get("flags", []):
        lines.append("   FLAG: %s" % flag)

    # G3-trend
    lines.append("")
    g3t = verdict["g3_trend"]
    if g3t.get("error"):
        lines.append("G3-trend [FAIL]: %s" % g3t["error"])
    else:
        win = g3t["window"]
        sc = g3t["slope_ci"]
        lines.append(
            "G3-trend in-budget LBR slope CI upper < 0, magnitude clears MDS [%s]"
            % g3t["verdict"]
        )
        lines.append(
            "   window: iters [%s, %s] (width %s), 10x anchor iter=%s, n=%s, "
            "budget@start=%s traversals (reached=%s)"
            % (
                win.get("start_iter"), win.get("end_iter"), win.get("window_width"),
                win.get("anchor_iter"), win.get("n_in_window"),
                win.get("budget_start_traversals"), win.get("budget_ok"),
            )
        )
        if win.get("note"):
            lines.append("   note: %s" % win["note"])
        lines.append(
            "   slope=%s ci=[%s, %s] se=%s df=%s | MDS(power=%s)=%s | |slope|=%s clears=%s"
            % (
                _fmt(sc.get("slope")), _fmt(sc.get("ci_low")), _fmt(sc.get("ci_high")),
                _fmt(sc.get("se")), sc.get("df"), g3t.get("trend_power"),
                _fmt(g3t.get("minimum_detectable_slope")),
                _fmt(g3t.get("observed_slope_magnitude")), g3t.get("observed_clears_mds"),
            )
        )
        if sc.get("error"):
            lines.append("   slope note: %s" % sc["error"])

    # Overall
    lines.append("")
    overall = verdict["overall_verdict"]
    lines.append("OVERALL: %s" % overall)
    if overall == "FAIL" and g3t.get("verdict") == "FAIL" and g3t.get("budget_reached"):
        lines.append(
            "  (flat-curve kill: budget reached, trend not significant -> RC-F promotes; "
            "budget curve is the deliverable)"
        )

    # Budget curve
    curve = verdict.get("budget_curve", [])
    lines.append("")
    lines.append("Budget curve (Tier-A LBR vs cumulative traversals), %d points:" % len(curve))
    for pt in _curve_preview(curve):
        lines.append(
            "   iter=%-6s traversals=%-12s tier_a_lbr=%s"
            % (pt["iteration"], pt["cumulative_traversals"], _fmt(pt["tier_a_lbr"]))
        )

    # mean_imp(5) floor (informational)
    floor = verdict["mean_imp5_floor"]
    lines.append("")
    lines.append("mean_imp(5) per-baseline floor (informational, gates nothing):")
    if not floor.get("baselines"):
        lines.append("   %s" % floor.get("note", "unavailable"))
    else:
        for name, row in floor["baselines"].items():
            lines.append(
                "   %-20s win_rate=%s games=%s iter=%s"
                % (name, _fmt(row.get("win_rate")), row.get("games_played"), row.get("iteration"))
            )
        if floor.get("mean_imp") is not None:
            lines.append("   mean_imp(5) = %.4f" % floor["mean_imp"])
        else:
            lines.append("   %s" % floor.get("note", ""))

    return "\n".join(lines)


def _curve_preview(curve: Sequence[Dict[str, Any]], head: int = 5, tail: int = 5) -> List[Dict[str, Any]]:
    """Head + tail of the curve for the human summary (full table lives in JSON)."""
    if len(curve) <= head + tail:
        return list(curve)
    return list(curve[:head]) + list(curve[-tail:])


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    # Slope/SE/MDS magnitudes are commonly ~1e-7; scientific notation keeps them
    # legible while %.6f stays readable for the [0,1] exploitability/rate values.
    if value != 0.0 and abs(value) < 1e-4:
        return "%.3e" % value
    return "%.6f" % value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "PRT-CFR X5 terminal gate verdict: renders the Phase 3 contract "
            "(G2 tie-aware not-a-loss, G3-ordering exploitability, G3-trend in-budget "
            "LBR slope CI + minimum detectable slope, budget-reached assertion) from "
            "a full run's metrics.jsonl + reconciled run_db + baseline reference "
            "table, and emits the budget curve. Nonzero exit on any binding-leg FAIL."
        )
    )
    ap.add_argument("run_dir", help="X5 run directory (e.g. runs/v0.4-x5-full).")
    ap.add_argument(
        "--db-path", default=None,
        help="run_db sqlite path override (default: CAMBIA_RUN_DB env or cfr/runs/cambia_runs.db).",
    )
    ap.add_argument(
        "--baseline-ref", default=DEFAULT_BASELINE_REF,
        help="P3W5 baseline exploitability reference table JSON (default: %(default)s).",
    )
    ap.add_argument(
        "--k-games", type=int, default=None,
        help="Traversals per iteration (K). Default: read prt_cfr.k_games_per_iter "
        "from <run_dir>/config.yaml or run_db.",
    )
    ap.add_argument(
        "--g2-rule", choices=("default", "point", "lower-ci"), default="default",
        help="G2 pass rule. default = lower-ci on strategic legs, point on random "
        "legs (user ruling pending); point / lower-ci force one rule everywhere.",
    )
    ap.add_argument(
        "--ismcts-binding", action="store_true",
        help="Make ISMCTS-BR a binding G3-ordering estimator (default: confirming only, "
        "disagreement is a flag not a veto).",
    )
    ap.add_argument(
        "--window-width", type=int, default=DEFAULT_WINDOW_WIDTH,
        help="G3-trend window width in iterations (default: %(default)s).",
    )
    ap.add_argument(
        "--trend-power", type=float, default=DEFAULT_TREND_POWER,
        help="Power for the minimum-detectable-slope bar (default: %(default)s; "
        "0.5 makes MDS the CI half-width).",
    )
    ap.add_argument(
        "--iter", dest="iteration", type=int, default=None,
        help="Pin G2 and mixture-exploitability reads to this eval iteration "
        "(default: latest per baseline key).",
    )
    ap.add_argument("--out", default=None, help="JSON verdict output path.")
    ap.add_argument(
        "--plot-out", default=None,
        help="Budget-curve PNG output path (default: <run_dir>/x5_budget_curve.png "
        "when matplotlib is importable).",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verdict = compute_verdict(
        args.run_dir,
        db_path=args.db_path,
        baseline_ref_path=args.baseline_ref,
        k_games=args.k_games,
        g2_rule=args.g2_rule,
        ismcts_binding=args.ismcts_binding,
        window_width=args.window_width,
        trend_power=args.trend_power,
        iteration=args.iteration,
    )
    print(human_summary(verdict))

    # The budget curve is written regardless of PASS or FAIL (the flat-curve kill
    # deliverable). PNG is best-effort; the JSON table is authoritative.
    plot_path = Path(args.plot_out) if args.plot_out else (Path(args.run_dir) / "x5_budget_curve.png")
    written = write_budget_curve_plot(verdict.get("budget_curve", []), plot_path)
    verdict["budget_curve_plot"] = written
    if written:
        print("\n[prtcfr-x5-verdict] wrote budget-curve plot %s" % written)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, indent=2, sort_keys=False)
        print("[prtcfr-x5-verdict] wrote %s" % args.out)

    return 0 if verdict["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
