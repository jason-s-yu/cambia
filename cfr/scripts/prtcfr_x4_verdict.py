"""cfr/scripts/prtcfr_x4_verdict.py

X4 gate verdict analysis (v0.4 Phase 2 Sprint 2, S2W3): renders the Phase 2
contract AC2 verdict for a full-game PRT-CFR production pilot run.

Verbatim AC2 (phase2-throughput-pilot/contract.md): T1-Cambia < 5% by iter 150
AND Tier-A LBR slope over iters 100-300 negative with 95% CI excluding 0 AND
zero grad-norm violations, over a 300-iteration full-game pilot with the
battery every 10 iters.

This script reads two independent sources, per the sprint-2 plan's pinned
verdict contract:
  - ``<run_dir>/metrics.jsonl``: the in-loop battery series (S2W1 additive
    fields on ``PRTCFRProductionTrainState``/its metrics row: ``t1_cambia_rate``,
    ``tier_a_lbr``, ``grad_norm_violations``). Rows that lack a given field
    (the battery cadence -- not every iteration carries an eval_fn score) are
    skipped for that field's computation rather than treated as a zero or a
    crossing.
  - the reconciled run_db (``src/run_db.py``, the sole eval-results source):
    the post-pilot mean_imp(5) per-baseline rows, rendered as an
    informational floor table. Per protocol.md and the sprint-2 plan, this
    table gates nothing -- only the three AC2 sub-conditions below do.

Computes:
  1. ``t1_cambia_first_sub5_iter``: first iteration (ascending) with
     ``t1_cambia_rate < 0.05``. Gate: <= 150.
  2. ``tier_a_lbr_slope_ci``: OLS slope of ``tier_a_lbr`` over iters
     [100, 300] with a 95% confidence interval (Student's t, no scipy
     dependency -- the regularized incomplete beta function is implemented
     locally via a continued fraction, the standard Numerical-Recipes
     ``betacf``/``betai`` construction). Gate: CI upper bound < 0 (the slope
     is significantly negative).
  3. ``grad_norm_violations_total``: sum of ``grad_norm_violations`` across
     all metrics rows that carry the field. Gate: == 0.

Emits one structured verdict JSON (machine-readable, ``--out`` to persist)
plus a human summary printed to stdout. Exits nonzero if any AC2
sub-condition fails.

Usage:
  cd cfr && python scripts/prtcfr_x4_verdict.py runs/v0.4-x4-pilot
  cd cfr && python scripts/prtcfr_x4_verdict.py runs/v0.4-x4-pilot --out runs/v0.4-x4-pilot/x4_verdict.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_agents import MEAN_IMP_BASELINES  # noqa: E402
from src.run_db import get_db  # noqa: E402

# ---------------------------------------------------------------------------
# metrics.jsonl parsing
# ---------------------------------------------------------------------------


def load_metrics_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    """Parse a metrics.jsonl file into a list of row dicts.

    Missing file -> empty list (callers report this as a FAIL condition, not
    a crash -- a verdict script must degrade to a clear FAIL, never a
    traceback, when pointed at an incomplete or wrong run_dir).
    """
    rows: List[Dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    with open(metrics_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# AC2 sub-condition 1: T1-Cambia rate crossing
# ---------------------------------------------------------------------------


def first_sub_threshold_iter(
    rows: Sequence[Dict[str, Any]],
    field: str = "t1_cambia_rate",
    threshold: float = 0.05,
) -> Optional[int]:
    """First iteration (ascending) at which ``field`` drops below ``threshold``.

    Rows lacking ``field`` (battery-cadence misses -- the pinned schema adds
    the field to the metrics row but the eval_fn that populates it only runs
    at the stability cadence) are skipped, not treated as an observation.
    """
    observed = [
        (row["iteration"], row[field])
        for row in rows
        if "iteration" in row and field in row and row[field] is not None
    ]
    observed.sort(key=lambda pair: pair[0])
    for iteration, value in observed:
        if value < threshold:
            return iteration
    return None


# ---------------------------------------------------------------------------
# AC2 sub-condition 2: Tier-A LBR OLS slope + 95% CI
#
# No scipy dependency (not in cfr/pyproject.toml). The regularized incomplete
# beta function is implemented via the standard continued-fraction
# construction (Numerical Recipes' betacf/betai) to invert the Student's t
# CDF for the two-sided critical value, rather than approximating with a
# normal quantile (which understates the CI width at the modest sample
# counts -- ~20 points at a 10-iter cadence over a 200-iter window -- this
# gate operates at).
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation of the incomplete beta function."""
    max_iter = 200
    eps = 3.0e-16
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    """Student's t CDF via the incomplete-beta relation."""
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0.0 else tail


def _t_ppf(p: float, df: float) -> float:
    """Inverse Student's t CDF (quantile function) via bisection.

    ``p`` in (0.5, 1); returns the positive t such that P(T <= t) = p.
    """
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ols_slope_ci(
    xs: Sequence[float], ys: Sequence[float], confidence: float = 0.95
) -> Dict[str, Any]:
    """OLS slope of ``ys`` on ``xs`` with a two-sided confidence interval.

    Returns a dict with ``slope``/``intercept``/``se``/``t_crit``/``ci_low``/
    ``ci_high``/``n_points``/``df``, or an ``error`` string (and all numeric
    fields None) if there are too few points or ``xs`` has zero variance.
    """
    n = len(xs)
    if n < 3:
        return {
            "n_points": n,
            "df": None,
            "slope": None,
            "intercept": None,
            "se": None,
            "t_crit": None,
            "ci_low": None,
            "ci_high": None,
            "error": "insufficient data points for OLS (need >= 3, got %d)" % n,
        }
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0.0:
        return {
            "n_points": n,
            "df": n - 2,
            "slope": None,
            "intercept": None,
            "se": None,
            "t_crit": None,
            "ci_low": None,
            "ci_high": None,
            "error": "degenerate x series (zero variance)",
        }
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    df = n - 2
    variance = sse / df if df > 0 else 0.0
    se_slope = math.sqrt(variance / sxx)
    alpha = 1.0 - confidence
    if se_slope == 0.0:
        t_crit = 0.0
    else:
        t_crit = _t_ppf(1.0 - alpha / 2.0, df)
    margin = t_crit * se_slope
    return {
        "n_points": n,
        "df": df,
        "slope": slope,
        "intercept": intercept,
        "se": se_slope,
        "t_crit": t_crit,
        "ci_low": slope - margin,
        "ci_high": slope + margin,
        "error": None,
    }


def tier_a_lbr_slope(
    rows: Sequence[Dict[str, Any]],
    iter_lo: int = 100,
    iter_hi: int = 300,
    field: str = "tier_a_lbr",
) -> Dict[str, Any]:
    points: List[Tuple[float, float]] = [
        (float(row["iteration"]), float(row[field]))
        for row in rows
        if "iteration" in row
        and field in row
        and row[field] is not None
        and iter_lo <= row["iteration"] <= iter_hi
    ]
    points.sort(key=lambda pair: pair[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ols_slope_ci(xs, ys)


# ---------------------------------------------------------------------------
# AC2 sub-condition 3: grad-norm violations
# ---------------------------------------------------------------------------


def grad_norm_violations_total(
    rows: Sequence[Dict[str, Any]], field: str = "grad_norm_violations"
) -> int:
    return sum(int(row[field]) for row in rows if field in row and row[field] is not None)


# ---------------------------------------------------------------------------
# mean_imp(5) informational floor (run_db read path -- the sole eval source)
# ---------------------------------------------------------------------------


def read_mean_imp5_floor(run_dir: Path, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Read the post-pilot mean_imp(5) per-baseline floor from run_db.

    Looked up by run name = ``run_dir``'s basename (the client-side run_dir
    convention: ``upsert_run(name=...)`` and ``Path("runs")/name`` are the
    same string throughout cli.py). Informational only -- absence of a
    run_db row or of eval_results rows is reported, not raised.
    """
    run_name = Path(run_dir).name
    db = get_db(db_path)
    run_row = db.execute("SELECT id FROM runs WHERE name=?", (run_name,)).fetchone()
    if run_row is None:
        return {
            "run_name": run_name,
            "available": False,
            "baselines": {},
            "mean_imp": None,
            "note": "no run_db row named '%s'; mean_imp(5) not yet reconciled" % run_name,
        }
    run_id = run_row["id"]
    placeholders = ",".join("?" * len(MEAN_IMP_BASELINES))
    result_rows = db.execute(
        "SELECT iteration, baseline, win_rate, ci_low, ci_high, games_played "
        "FROM eval_results WHERE run_id=? AND baseline IN (%s)" % placeholders,
        (run_id, *MEAN_IMP_BASELINES),
    ).fetchall()
    if not result_rows:
        return {
            "run_name": run_name,
            "available": False,
            "baselines": {},
            "mean_imp": None,
            "note": "run_db row found but no mean_imp(5) baseline eval_results rows present",
        }
    latest: Dict[str, Dict[str, Any]] = {}
    for row in result_rows:
        bl = row["baseline"]
        if bl not in latest or row["iteration"] > latest[bl]["iteration"]:
            latest[bl] = dict(row)
    imp_values = [
        latest[b]["win_rate"]
        for b in MEAN_IMP_BASELINES
        if b in latest and latest[b]["win_rate"] is not None
    ]
    complete = len(imp_values) == len(MEAN_IMP_BASELINES)
    mean_imp = sum(imp_values) / len(imp_values) if imp_values else None
    return {
        "run_name": run_name,
        "available": True,
        "baselines": {b: latest[b] for b in MEAN_IMP_BASELINES if b in latest},
        "mean_imp": mean_imp if complete else None,
        "note": (
            "informational per-baseline floor; gates nothing (protocol.md)"
            if complete
            else "partial baseline coverage (%d/%d); mean_imp(5) requires all 5"
            % (len(imp_values), len(MEAN_IMP_BASELINES))
        ),
    }


# ---------------------------------------------------------------------------
# Verdict assembly
# ---------------------------------------------------------------------------


def compute_verdict(
    run_dir: str,
    db_path: Optional[str] = None,
    iter_lo: int = 100,
    iter_hi: int = 300,
    sub5_threshold: float = 0.05,
    sub5_iter_gate: int = 150,
) -> Dict[str, Any]:
    run_dir_path = Path(run_dir)
    metrics_path = run_dir_path / "metrics.jsonl"
    rows = load_metrics_rows(metrics_path)

    crossing_iter = first_sub_threshold_iter(rows, threshold=sub5_threshold)
    cond1_pass = crossing_iter is not None and crossing_iter <= sub5_iter_gate

    slope_info = tier_a_lbr_slope(rows, iter_lo=iter_lo, iter_hi=iter_hi)
    cond2_pass = slope_info.get("ci_high") is not None and slope_info["ci_high"] < 0.0

    violations_total = grad_norm_violations_total(rows)
    cond3_pass = violations_total == 0

    mean_imp5_floor = read_mean_imp5_floor(run_dir_path, db_path)

    overall_pass = cond1_pass and cond2_pass and cond3_pass

    return {
        "run_dir": str(run_dir_path),
        "metrics_path": str(metrics_path),
        "n_metrics_rows": len(rows),
        "t1_cambia_first_sub5_iter": {
            "value": crossing_iter,
            "threshold": sub5_threshold,
            "iter_gate": sub5_iter_gate,
            "verdict": "PASS" if cond1_pass else "FAIL",
        },
        "tier_a_lbr_slope_ci": {
            **slope_info,
            "iter_lo": iter_lo,
            "iter_hi": iter_hi,
            "verdict": "PASS" if cond2_pass else "FAIL",
        },
        "grad_norm_violations_total": {
            "value": violations_total,
            "verdict": "PASS" if cond3_pass else "FAIL",
        },
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "mean_imp5_floor": mean_imp5_floor,
    }


def human_summary(verdict: Dict[str, Any]) -> str:
    c1 = verdict["t1_cambia_first_sub5_iter"]
    c2 = verdict["tier_a_lbr_slope_ci"]
    c3 = verdict["grad_norm_violations_total"]
    lines = [
        "X4 gate verdict (Phase 2 contract AC2) -- run_dir=%s" % verdict["run_dir"],
        "metrics rows: %d (%s)" % (verdict["n_metrics_rows"], verdict["metrics_path"]),
        "",
        "1. T1-Cambia < %.2f by iter %d: first sub-threshold iter=%s [%s]"
        % (c1["threshold"], c1["iter_gate"], c1["value"], c1["verdict"]),
        "2. Tier-A LBR slope over iters [%d, %d] negative, 95%% CI excludes 0: "
        "slope=%s ci=[%s, %s] n=%s df=%s [%s]"
        % (
            c2["iter_lo"],
            c2["iter_hi"],
            _fmt(c2.get("slope")),
            _fmt(c2.get("ci_low")),
            _fmt(c2.get("ci_high")),
            c2.get("n_points"),
            c2.get("df"),
            c2["verdict"],
        ),
    ]
    if c2.get("error"):
        lines.append("   note: %s" % c2["error"])
    lines.append(
        "3. grad_norm_violations_total == 0: total=%d [%s]" % (c3["value"], c3["verdict"])
    )
    lines.append("")
    lines.append("OVERALL: %s" % verdict["overall_verdict"])

    floor = verdict["mean_imp5_floor"]
    lines.append("")
    lines.append("mean_imp(5) per-baseline floor (informational, gates nothing):")
    if not floor["baselines"]:
        lines.append("  %s" % floor["note"])
    else:
        for name, row in floor["baselines"].items():
            lines.append(
                "  %-22s win_rate=%s games=%s iter=%s"
                % (
                    name,
                    _fmt(row.get("win_rate")),
                    row.get("games_played"),
                    row.get("iteration"),
                )
            )
        if floor.get("mean_imp") is not None:
            lines.append("  mean_imp(5) = %.4f" % floor["mean_imp"])
        else:
            lines.append("  %s" % floor["note"])
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.6f" % value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "PRT-CFR X4 gate verdict: renders Phase 2 contract AC2 (T1-Cambia "
            "crossing, Tier-A LBR slope CI, grad-norm violations) from a pilot "
            "run's metrics.jsonl, plus the informational mean_imp(5) floor "
            "from run_db."
        )
    )
    ap.add_argument("run_dir", help="Pilot run directory (e.g. runs/v0.4-x4-pilot).")
    ap.add_argument(
        "--db-path",
        default=None,
        help="run_db sqlite path override (default: CAMBIA_RUN_DB env or cfr/runs/cambia_runs.db).",
    )
    ap.add_argument(
        "--iter-lo",
        type=int,
        default=100,
        help="Tier-A LBR slope window start (inclusive).",
    )
    ap.add_argument(
        "--iter-hi",
        type=int,
        default=300,
        help="Tier-A LBR slope window end (inclusive).",
    )
    ap.add_argument(
        "--sub5-threshold", type=float, default=0.05, help="T1-Cambia crossing threshold."
    )
    ap.add_argument(
        "--sub5-iter-gate",
        type=int,
        default=150,
        help="Gate: crossing iter must be <= this.",
    )
    ap.add_argument("--out", default=None, help="JSON verdict output path.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verdict = compute_verdict(
        args.run_dir,
        db_path=args.db_path,
        iter_lo=args.iter_lo,
        iter_hi=args.iter_hi,
        sub5_threshold=args.sub5_threshold,
        sub5_iter_gate=args.sub5_iter_gate,
    )
    print(human_summary(verdict))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(verdict, fh, indent=2, sort_keys=False)
        print("\n[prtcfr-x4-verdict] wrote %s" % args.out)
    return 0 if verdict["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
