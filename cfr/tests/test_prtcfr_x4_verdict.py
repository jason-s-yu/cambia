"""tests/test_prtcfr_x4_verdict.py

Scoped tests for scripts/prtcfr_x4_verdict.py (v0.4 Phase 2 Sprint 2, S2W3):
the X4 gate verdict analysis script. Synthetic metrics.jsonl fixtures drive
the three AC2 sub-conditions (T1-Cambia crossing, Tier-A LBR slope CI,
grad-norm violations) independently -- each FAIL fixture changes exactly one
axis off the PASS baseline so a failure pinpoints its condition. The
OLS-slope-with-CI math is unit-tested against numpy's polyfit (independent
implementation, cross-checks the point estimate) and a known Student's-t
critical-value table (cross-checks the confidence interval width).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import prtcfr_x4_verdict as verdict_mod  # noqa: E402
from src import run_db  # noqa: E402
from src.evaluate_agents import MEAN_IMP_BASELINES  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic metrics.jsonl fixture construction
#
# CADENCE_ITERS mirrors the plan's "battery every 10 iters" (plus iter 1, the
# stability controller's first check per _stability_check_due). t1_cambia_rate
# is tracked from iter 1; tier_a_lbr is populated from iter 100 on (nothing
# requires it start earlier -- the gate window is [100, 300] regardless).
# ---------------------------------------------------------------------------

CADENCE_ITERS = [1] + list(range(10, 301, 10))
LBR_ITERS = list(range(100, 301, 10))

# Deterministic (non-random) noise patterns so the fixtures are reproducible.
_NOISE_TIGHT = [
    0.01, -0.01, 0.005, -0.008, 0.012, -0.006, 0.009, -0.011, 0.004, -0.007,
    0.01, -0.009, 0.006, -0.005, 0.011, -0.01, 0.007, -0.008, 0.005, -0.006, 0.009,
]
_NOISE_WIDE = [
    0.05, -0.04, 0.03, -0.06, 0.04, -0.03, 0.05, -0.05, 0.02, -0.04,
    0.06, -0.03, 0.04, -0.05, 0.03, -0.04, 0.05, -0.03, 0.04, -0.05, 0.03,
]
assert len(_NOISE_TIGHT) == len(LBR_ITERS)
assert len(_NOISE_WIDE) == len(LBR_ITERS)


def _t1_crossing_at_120(it: int) -> float:
    """T1-Cambia rate: decreasing, first drops below 0.05 at iter 120."""
    if it <= 110:
        return round(0.35 - 0.0025 * it, 6)
    return round(max(0.001, 0.05 - 0.001 * (it - 110)), 6)


def _t1_crossing_at_170(it: int) -> float:
    """T1-Cambia rate: decreasing, first drops below 0.05 at iter 170."""
    if it <= 160:
        return round(0.35 - 0.00175 * it, 6)
    return round(max(0.001, 0.05 - 0.001 * (it - 160)), 6)


def _lbr_series(noise, slope: float, base: float = 1.0) -> dict:
    return {it: round(base + slope * (it - 100) + n, 6) for it, n in zip(LBR_ITERS, noise)}


def _build_rows(t1_fn, lbr_values: dict, violations_at: dict | None = None) -> list:
    violations_at = violations_at or {}
    rows = []
    for it in CADENCE_ITERS:
        row = {
            "iteration": it,
            "t1_cambia_rate": t1_fn(it),
            "grad_norm_violations": violations_at.get(it, 0),
        }
        if it in lbr_values:
            row["tier_a_lbr"] = lbr_values[it]
        rows.append(row)
    return rows


def _write_metrics_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


_STRONG_NEGATIVE_LBR = _lbr_series(_NOISE_TIGHT, slope=-0.002)
_FLAT_LBR = _lbr_series(_NOISE_WIDE, slope=0.0002)


def _pass_rows(**kwargs):
    return _build_rows(_t1_crossing_at_120, _STRONG_NEGATIVE_LBR, **kwargs)


# ---------------------------------------------------------------------------
# Student's t critical value (known table cross-check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "df,expected",
    [(1, 12.706), (5, 2.571), (10, 2.228), (19, 2.093), (30, 2.042), (100, 1.984)],
)
def test_t_ppf_matches_known_table(df, expected):
    assert verdict_mod._t_ppf(0.975, df) == pytest.approx(expected, abs=1e-3)


def test_t_ppf_approaches_normal_z_for_large_df():
    assert verdict_mod._t_ppf(0.975, 1_000_000) == pytest.approx(1.95996, abs=1e-3)


# ---------------------------------------------------------------------------
# OLS slope + CI (cross-checked against numpy polyfit, a known series)
# ---------------------------------------------------------------------------


def test_ols_slope_ci_matches_numpy_polyfit_on_known_series():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [8.9, 7.2, 6.8, 5.1, 4.9, 3.3, 2.7, 1.5, 0.6, -0.4]
    np_slope, np_intercept = np.polyfit(xs, ys, 1)

    result = verdict_mod.ols_slope_ci(xs, ys)

    assert result["error"] is None
    assert result["slope"] == pytest.approx(np_slope, abs=1e-9)
    assert result["intercept"] == pytest.approx(np_intercept, abs=1e-9)
    assert result["n_points"] == 10
    assert result["df"] == 8
    # Hand-verified reference values (Student's t, df=8, 95% CI).
    assert result["se"] == pytest.approx(0.033117760223792155, abs=1e-9)
    assert result["t_crit"] == pytest.approx(2.306004135204165, abs=1e-9)
    assert result["ci_low"] == pytest.approx(-1.077581813236886, abs=1e-9)
    assert result["ci_high"] == pytest.approx(-0.9248424291873565, abs=1e-9)


def test_ols_slope_ci_insufficient_points_reports_error():
    result = verdict_mod.ols_slope_ci([1, 2], [1.0, 2.0])
    assert result["error"] is not None
    assert result["slope"] is None
    assert result["ci_low"] is None
    assert result["ci_high"] is None


def test_ols_slope_ci_degenerate_x_reports_error():
    result = verdict_mod.ols_slope_ci([5, 5, 5], [1.0, 2.0, 3.0])
    assert result["error"] is not None
    assert result["slope"] is None


# ---------------------------------------------------------------------------
# AC2 sub-condition helpers
# ---------------------------------------------------------------------------


def test_first_sub_threshold_iter_finds_first_crossing():
    rows = _build_rows(_t1_crossing_at_120, {})
    assert verdict_mod.first_sub_threshold_iter(rows) == 120


def test_first_sub_threshold_iter_skips_rows_missing_field():
    rows = [
        {"iteration": 1, "t1_cambia_rate": 0.3},
        {"iteration": 10},  # missing field: must not be treated as a crossing
        {"iteration": 20, "t1_cambia_rate": 0.01},
    ]
    assert verdict_mod.first_sub_threshold_iter(rows) == 20


def test_first_sub_threshold_iter_returns_none_if_never_crosses():
    rows = [{"iteration": it, "t1_cambia_rate": 0.5} for it in [1, 10, 20]]
    assert verdict_mod.first_sub_threshold_iter(rows) is None


def test_tier_a_lbr_slope_filters_to_window():
    rows = _build_rows(_t1_crossing_at_120, _STRONG_NEGATIVE_LBR)
    result = verdict_mod.tier_a_lbr_slope(rows, iter_lo=100, iter_hi=300)
    assert result["n_points"] == len(LBR_ITERS)
    assert result["slope"] < 0
    assert result["ci_high"] < 0


def test_grad_norm_violations_total_sums_present_skips_missing():
    rows = [
        {"grad_norm_violations": 0},
        {},  # missing field: treated as no contribution, not an error
        {"grad_norm_violations": 2},
    ]
    assert verdict_mod.grad_norm_violations_total(rows) == 2


# ---------------------------------------------------------------------------
# mean_imp(5) informational floor (run_db read path)
# ---------------------------------------------------------------------------


def _seed_mean_imp5(db_path, run_name, iteration=300, win_rates=None):
    win_rates = win_rates or {
        "random_no_cambia": 0.70,
        "random_late_cambia": 0.62,
        "imperfect_greedy": 0.55,
        "memory_heuristic": 0.48,
        "aggressive_snap": 0.51,
    }
    db = run_db.get_db(str(db_path))
    run_id = run_db.upsert_run(db, name=run_name, algorithm="prt-cfr")
    for baseline, wr in win_rates.items():
        run_db.insert_eval_result(
            db, run_id, None,
            {"iteration": iteration, "baseline": baseline, "win_rate": wr, "games_played": 5000},
        )
    return run_id


def test_mean_imp5_floor_run_not_found(tmp_path):
    run_dir = tmp_path / "runs" / "no-such-run"
    run_dir.mkdir(parents=True)
    floor = verdict_mod.read_mean_imp5_floor(run_dir, db_path=str(tmp_path / "db.sqlite"))
    assert floor["available"] is False
    assert floor["mean_imp"] is None
    assert floor["baselines"] == {}


def test_mean_imp5_floor_reads_latest_iteration_per_baseline(tmp_path):
    run_dir = tmp_path / "runs" / "x4-pilot"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_mean_imp5(db_path, "x4-pilot", iteration=200, win_rates={"random_no_cambia": 0.40})
    _seed_mean_imp5(db_path, "x4-pilot", iteration=300)  # newer, full 5-baseline row set

    floor = verdict_mod.read_mean_imp5_floor(run_dir, db_path=str(db_path))

    assert floor["available"] is True
    assert set(floor["baselines"]) == set(MEAN_IMP_BASELINES)
    # The stale iteration=200 random_no_cambia=0.40 row must not win over 300's 0.70.
    assert floor["baselines"]["random_no_cambia"]["win_rate"] == pytest.approx(0.70)
    assert floor["mean_imp"] == pytest.approx((0.70 + 0.62 + 0.55 + 0.48 + 0.51) / 5.0)


def test_mean_imp5_floor_partial_baselines_no_mean_imp(tmp_path):
    run_dir = tmp_path / "runs" / "x4-partial"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    db = run_db.get_db(str(db_path))
    run_id = run_db.upsert_run(db, name="x4-partial", algorithm="prt-cfr")
    run_db.insert_eval_result(
        db, run_id, None,
        {"iteration": 300, "baseline": "random_no_cambia", "win_rate": 0.6, "games_played": 5000},
    )

    floor = verdict_mod.read_mean_imp5_floor(run_dir, db_path=str(db_path))

    assert floor["available"] is True
    assert floor["mean_imp"] is None
    assert "partial" in floor["note"]


# ---------------------------------------------------------------------------
# Full verdict: PASS + each FAIL mode (one axis changed per case)
# ---------------------------------------------------------------------------


def test_verdict_pass(tmp_path):
    run_dir = tmp_path / "runs" / "x4-pass"
    run_dir.mkdir(parents=True)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", _pass_rows())
    db_path = tmp_path / "db.sqlite"
    _seed_mean_imp5(db_path, "x4-pass")

    verdict = verdict_mod.compute_verdict(str(run_dir), db_path=str(db_path))

    assert verdict["t1_cambia_first_sub5_iter"]["value"] == 120
    assert verdict["t1_cambia_first_sub5_iter"]["verdict"] == "PASS"
    assert verdict["tier_a_lbr_slope_ci"]["verdict"] == "PASS"
    assert verdict["grad_norm_violations_total"]["value"] == 0
    assert verdict["grad_norm_violations_total"]["verdict"] == "PASS"
    assert verdict["overall_verdict"] == "PASS"
    assert verdict["mean_imp5_floor"]["mean_imp"] is not None

    summary = verdict_mod.human_summary(verdict)
    assert "OVERALL: PASS" in summary


def test_verdict_fail_t1_cambia_crosses_too_late(tmp_path):
    run_dir = tmp_path / "runs" / "x4-fail-t1"
    run_dir.mkdir(parents=True)
    rows = _build_rows(_t1_crossing_at_170, _STRONG_NEGATIVE_LBR)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", rows)

    verdict = verdict_mod.compute_verdict(str(run_dir))

    assert verdict["t1_cambia_first_sub5_iter"]["value"] == 170
    assert verdict["t1_cambia_first_sub5_iter"]["verdict"] == "FAIL"
    assert verdict["tier_a_lbr_slope_ci"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["grad_norm_violations_total"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["overall_verdict"] == "FAIL"


def test_verdict_fail_flat_lbr_slope_ci_spans_zero(tmp_path):
    run_dir = tmp_path / "runs" / "x4-fail-lbr"
    run_dir.mkdir(parents=True)
    rows = _build_rows(_t1_crossing_at_120, _FLAT_LBR)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", rows)

    verdict = verdict_mod.compute_verdict(str(run_dir))

    assert verdict["t1_cambia_first_sub5_iter"]["verdict"] == "PASS"  # unaffected axis
    lbr = verdict["tier_a_lbr_slope_ci"]
    assert lbr["ci_low"] < 0 < lbr["ci_high"]  # CI spans zero: not significant
    assert lbr["verdict"] == "FAIL"
    assert verdict["grad_norm_violations_total"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["overall_verdict"] == "FAIL"


def test_verdict_fail_nonzero_grad_norm_violations(tmp_path):
    run_dir = tmp_path / "runs" / "x4-fail-grad"
    run_dir.mkdir(parents=True)
    rows = _pass_rows(violations_at={150: 1})
    _write_metrics_jsonl(run_dir / "metrics.jsonl", rows)

    verdict = verdict_mod.compute_verdict(str(run_dir))

    assert verdict["t1_cambia_first_sub5_iter"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["tier_a_lbr_slope_ci"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["grad_norm_violations_total"]["value"] == 1
    assert verdict["grad_norm_violations_total"]["verdict"] == "FAIL"
    assert verdict["overall_verdict"] == "FAIL"


def test_verdict_missing_metrics_file_fails_clean(tmp_path):
    run_dir = tmp_path / "runs" / "x4-empty"
    run_dir.mkdir(parents=True)
    verdict = verdict_mod.compute_verdict(str(run_dir))
    assert verdict["n_metrics_rows"] == 0
    assert verdict["overall_verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_pass(tmp_path):
    run_dir = tmp_path / "runs" / "x4-pass"
    run_dir.mkdir(parents=True)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", _pass_rows())
    db_path = tmp_path / "db.sqlite"

    exit_code = verdict_mod.main([str(run_dir), "--db-path", str(db_path)])

    assert exit_code == 0


def test_main_returns_nonzero_on_fail(tmp_path):
    run_dir = tmp_path / "runs" / "x4-fail"
    run_dir.mkdir(parents=True)
    rows = _build_rows(_t1_crossing_at_170, _STRONG_NEGATIVE_LBR)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", rows)
    db_path = tmp_path / "db.sqlite"

    exit_code = verdict_mod.main([str(run_dir), "--db-path", str(db_path)])

    assert exit_code == 1


def test_main_writes_json_output(tmp_path):
    run_dir = tmp_path / "runs" / "x4-pass"
    run_dir.mkdir(parents=True)
    _write_metrics_jsonl(run_dir / "metrics.jsonl", _pass_rows())
    db_path = tmp_path / "db.sqlite"
    out_path = tmp_path / "verdict.json"

    verdict_mod.main([str(run_dir), "--db-path", str(db_path), "--out", str(out_path)])

    with open(out_path, "r", encoding="utf-8") as fh:
        persisted = json.load(fh)
    assert persisted["overall_verdict"] == "PASS"


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        verdict_mod.main(["--help"])
    assert exc_info.value.code == 0
