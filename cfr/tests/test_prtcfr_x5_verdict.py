"""tests/test_prtcfr_x5_verdict.py

Scoped tests for scripts/prtcfr_x5_verdict.py (v0.4 Phase 3, P3W3): the X5
terminal gate verdict. Synthetic metrics.jsonl + run_db rows + a baseline
exploitability reference-table JSON drive:

  - one full PASS (all G2 legs >= 0.50, mixture exploitability strictly below
    every baseline with non-overlapping CIs, strictly-negative in-budget LBR
    slope CI clearing the minimum detectable slope, budget reached);
  - each FAIL mode, one axis off the PASS baseline: a strategic G2 leg at 0.48;
    a baseline whose exploitability CI overlaps the mixture's; a flat LBR slope
    with CI spanning 0; a window that does not reach 2.0e7 traversals;
  - unit tests for the window-selection and minimum-detectable-slope math on
    known series;
  - the --g2-rule flag flipping a borderline strategic leg (point passes, lower
    CI bound does not);
  - the canonical row-list reference schema (cambia-415): read_baseline_reference
    accepting {"schema_version", "rows": [...]}, latest-row-wins on duplicate
    (baseline, estimator) rows, and row-list vs. nested-map agreement on
    G3-ordering results for identical underlying data.

The slope-CI math itself is imported from prtcfr_x4_verdict (already unit-tested
against numpy polyfit and a Student's-t table in test_prtcfr_x4_verdict.py), so
these tests exercise the X5-specific composition, not the OLS internals.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import prtcfr_x5_verdict as x5  # noqa: E402
from src import run_db  # noqa: E402
from src.evaluate_agents import MEAN_IMP_BASELINES  # noqa: E402


K_GAMES = 8192
# ceil(2.0e7 / 8192) = 2442 (the 10x anchor iteration).
ANCHOR_ITER = math.ceil(x5.BUDGET_ANCHOR_TRAVERSALS / K_GAMES)

# Battery cadence: every 10 iters. A PASS run reaches 4000 (window [3000, 4000],
# start 3000 >= anchor 2442, budget 3000*8192 = 24,576,000 >= 2.0e7).
PASS_ITERS = list(range(10, 4001, 10))
# A budget-not-reached run ends at 3000 (window start 2000 < anchor 2442).
SHORT_ITERS = list(range(10, 3001, 10))


# ---------------------------------------------------------------------------
# metrics.jsonl fixtures (deterministic; no randomness)
# ---------------------------------------------------------------------------


def _decreasing_lbr(iters, base=1.0, slope=-2.0e-4, noise_amp=5.0e-4):
    """Strongly, significantly decreasing Tier-A LBR with tiny alternating noise."""
    out = {}
    for i, it in enumerate(iters):
        noise = noise_amp * (1.0 if i % 2 == 0 else -1.0)
        out[it] = round(base + slope * it + noise, 8)
    return out


def _flat_lbr(iters, base=0.30, slope=1.0e-5, noise_amp=2.0e-2):
    """Flat (not significantly negative) Tier-A LBR: CI spans zero over the window."""
    out = {}
    for i, it in enumerate(iters):
        noise = noise_amp * (1.0 if i % 2 == 0 else -1.0)
        out[it] = round(base + slope * it + noise, 8)
    return out


def _write_metrics(path, lbr_values):
    with open(path, "w", encoding="utf-8") as fh:
        for it in sorted(lbr_values):
            row = {"iteration": it, "tier_a_lbr": lbr_values[it], "grad_norm_violations": 0}
            fh.write(json.dumps(row) + "\n")


def _write_config_yaml(run_dir, k_games=K_GAMES):
    (run_dir / "config.yaml").write_text(
        "prt_cfr:\n  k_games_per_iter: %d\n  iterations: 5000\n" % k_games,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# run_db fixtures: G2 not-a-loss rows + mixture exploitability rows
# ---------------------------------------------------------------------------

# games=5000 per baseline. (p0_wins, ties) chosen so not_a_loss = (p0 + 0.5*ties)/5000.
_G2_PASS = {
    "random_no_cambia": (3500, 100),   # na = 0.710
    "random_late_cambia": (3000, 100),  # na = 0.610
    "imperfect_greedy": (2950, 200),    # na = 0.610
    "memory_heuristic": (2900, 150),    # na = 0.595
    "aggressive_snap": (2850, 100),     # na = 0.580
}


def _seed_g2(db_path, run_name, seat_counts, iteration=5000, games=5000):
    db = run_db.get_db(str(db_path))
    run_id = run_db.upsert_run(db, name=run_name, algorithm="prt-cfr")
    for baseline, (p0, ties) in seat_counts.items():
        p1 = games - p0 - ties
        run_db.insert_eval_result(
            db, run_id, None,
            {
                "iteration": iteration, "baseline": baseline,
                "win_rate": p0 / games, "games_played": games,
                "p0_wins": p0, "p1_wins": p1, "ties": ties,
            },
        )
    return db, run_id


# Mixture exploitability (rides in win_rate): well below every baseline reference.
_MIX_PASS = {
    "lbr_tier_b": {"exploitability": 0.10, "ci_low": 0.08, "ci_high": 0.12},
    "ismcts_br": {"exploitability": 0.12, "ci_low": 0.10, "ci_high": 0.14},
}


def _seed_mixture_exploit(db_path, run_name, mixture, iteration=5000, games=2000):
    db = run_db.get_db(str(db_path))
    run_id = run_db.upsert_run(db, name=run_name, algorithm="prt-cfr")
    for key, vals in mixture.items():
        run_db.insert_eval_result(
            db, run_id, None,
            {
                "iteration": iteration, "baseline": key,
                "win_rate": vals["exploitability"],
                "ci_low": vals["ci_low"], "ci_high": vals["ci_high"],
                "games_played": games,
            },
        )
    return run_id


# Baseline reference table (P3W5): every baseline's exploitability strictly above
# the mixture, with non-overlapping CIs, under both estimators.
_REF_PASS = {
    "random_no_cambia": {
        "lbr_tier_b": {"exploitability": 0.55, "ci_low": 0.52, "ci_high": 0.58},
        "ismcts_br": {"exploitability": 0.60, "ci_low": 0.56, "ci_high": 0.64},
    },
    "random_late_cambia": {
        "lbr_tier_b": {"exploitability": 0.50, "ci_low": 0.47, "ci_high": 0.53},
        "ismcts_br": {"exploitability": 0.55, "ci_low": 0.51, "ci_high": 0.59},
    },
    "imperfect_greedy": {
        "lbr_tier_b": {"exploitability": 0.30, "ci_low": 0.27, "ci_high": 0.33},
        "ismcts_br": {"exploitability": 0.34, "ci_low": 0.30, "ci_high": 0.38},
    },
    "memory_heuristic": {
        "lbr_tier_b": {"exploitability": 0.28, "ci_low": 0.25, "ci_high": 0.31},
        "ismcts_br": {"exploitability": 0.32, "ci_low": 0.28, "ci_high": 0.36},
    },
    "aggressive_snap": {
        "lbr_tier_b": {"exploitability": 0.35, "ci_low": 0.32, "ci_high": 0.38},
        "ismcts_br": {"exploitability": 0.40, "ci_low": 0.36, "ci_high": 0.44},
    },
}


def _write_reference(path, baselines):
    payload = {"schema_version": 1, "rule_profile": "production", "baselines": baselines}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows_from_baselines(baselines, timestamp="2026-07-12T00:00:00Z", seed=42,
                          rule_profile="prtcfr_production", sample_count=2000):
    """Flatten a nested {baseline: {estimator: {exploitability, ci_low, ci_high}}}
    map into the canonical producer row-list shape (P3W5's write_table format),
    for asserting the row-list and nested-map reference loaders agree."""
    rows = []
    for baseline, estimators in baselines.items():
        for estimator, vals in estimators.items():
            rows.append({
                "baseline": baseline,
                "estimator": estimator,
                "value": vals["exploitability"],
                "ci_low": vals["ci_low"],
                "ci_high": vals["ci_high"],
                "sample_count": sample_count,
                "rule_profile": rule_profile,
                "seed": seed,
                "timestamp": timestamp,
            })
    return rows


def _write_row_list_reference(path, baselines, **kwargs):
    payload = {"schema_version": 1, "rows": _rows_from_baselines(baselines, **kwargs)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(tmp_path, name, lbr_values, g2_counts, mixture, reference, k_games=K_GAMES):
    """Assemble a full run fixture: run_dir + metrics + config + run_db + ref table."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    _write_metrics(run_dir / "metrics.jsonl", lbr_values)
    _write_config_yaml(run_dir, k_games=k_games)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, name, g2_counts)
    _seed_mixture_exploit(db_path, name, mixture)
    ref_path = tmp_path / "baseline_exploitability.json"
    _write_reference(ref_path, reference)
    return run_dir, str(db_path), str(ref_path)


# ---------------------------------------------------------------------------
# Unit: Wilson CI
# ---------------------------------------------------------------------------


def test_wilson_ci_known_value():
    lo, hi = x5.wilson_ci(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)


def test_wilson_ci_zero_games():
    assert x5.wilson_ci(0.5, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Unit: window selection
# ---------------------------------------------------------------------------


def test_select_trend_window_picks_trailing_in_budget_window():
    rows = [{"iteration": it, "tier_a_lbr": 1.0} for it in PASS_ITERS]
    w = x5.select_trend_window(rows, K_GAMES, window_width=1000)
    assert w["anchor_iter"] == ANCHOR_ITER == 2442
    assert w["end_iter"] == 4000
    assert w["start_iter"] == 3000
    assert w["valid"] is True
    assert w["budget_ok"] is True
    assert w["budget_start_traversals"] == 3000 * K_GAMES
    assert w["n_in_window"] == 101  # iters 3000..4000 inclusive at cadence 10


def test_select_trend_window_budget_not_reached():
    rows = [{"iteration": it, "tier_a_lbr": 1.0} for it in SHORT_ITERS]
    w = x5.select_trend_window(rows, K_GAMES, window_width=1000)
    assert w["end_iter"] == 3000
    assert w["start_iter"] == 2000
    assert w["valid"] is False  # start 2000 < anchor 2442
    assert w["budget_ok"] is False
    assert "budget not reached" in w["note"]


def test_select_trend_window_no_points():
    w = x5.select_trend_window([{"iteration": 5}], K_GAMES)
    assert w["valid"] is False
    assert w["n_in_window"] == 0


# ---------------------------------------------------------------------------
# Unit: minimum detectable slope
# ---------------------------------------------------------------------------


def test_minimum_detectable_slope_formula_and_power_half_collapses_to_margin():
    xs = list(range(0, 40))
    # A known noisy series so se/df are well defined.
    ys = [10.0 - 0.1 * x + (0.2 if x % 2 == 0 else -0.2) for x in xs]
    info = x5.ols_slope_ci(xs, ys)
    df = info["df"]
    se = info["se"]

    t_alpha = x5._t_ppf(0.975, df)
    t_beta = x5._t_ppf(0.80, df)
    mds = x5.minimum_detectable_slope(info, power=0.80)
    assert mds == pytest.approx((t_alpha + t_beta) * se, rel=1e-9)

    # power=0.5 -> t_beta term vanishes -> MDS is the CI half-width (== margin).
    mds_half = x5.minimum_detectable_slope(info, power=0.5)
    margin = info["ci_high"] - info["slope"]
    assert mds_half == pytest.approx(t_alpha * se, rel=1e-9)
    assert mds_half == pytest.approx(margin, rel=1e-9)


def test_minimum_detectable_slope_degenerate_returns_none():
    info = x5.ols_slope_ci([1, 2], [1.0, 2.0])  # insufficient points
    assert x5.minimum_detectable_slope(info) is None


# ---------------------------------------------------------------------------
# Unit: K resolution
# ---------------------------------------------------------------------------


def test_resolve_k_games_from_config_yaml(tmp_path):
    run_dir = tmp_path / "runs" / "k-run"
    run_dir.mkdir(parents=True)
    _write_config_yaml(run_dir, k_games=4096)
    k, source = x5.resolve_k_games(run_dir, db_path=str(tmp_path / "db.sqlite"))
    assert k == 4096
    assert "config.yaml" in source


def test_resolve_k_games_cli_override_wins(tmp_path):
    run_dir = tmp_path / "runs" / "k-run2"
    run_dir.mkdir(parents=True)
    _write_config_yaml(run_dir, k_games=8192)
    k, source = x5.resolve_k_games(run_dir, k_override=1234, db_path=str(tmp_path / "db.sqlite"))
    assert k == 1234
    assert "cli" in source


def test_resolve_k_games_missing_returns_none(tmp_path):
    run_dir = tmp_path / "runs" / "k-none"
    run_dir.mkdir(parents=True)
    k, reason = x5.resolve_k_games(run_dir, db_path=str(tmp_path / "db.sqlite"))
    assert k is None
    assert "k_games_per_iter" in reason


# ---------------------------------------------------------------------------
# G2 leg unit tests
# ---------------------------------------------------------------------------


def test_g2_tie_aware_not_a_loss_and_strategic_binding(tmp_path):
    run_dir = tmp_path / "runs" / "g2"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g2", _G2_PASS)
    g2 = x5.compute_g2(run_dir, db_path=str(db_path))
    assert g2["verdict"] == "PASS"
    assert g2["strategic_pass"] is True
    ig = g2["legs"]["imperfect_greedy"]
    assert ig["binding"] is True
    assert ig["tie_aware"] is True
    assert ig["not_a_loss"] == pytest.approx((2950 + 0.5 * 200) / 5000)
    # random leg is reported non-binding
    assert g2["legs"]["random_no_cambia"]["binding"] is False


def test_g2_strategic_leg_048_fails(tmp_path):
    run_dir = tmp_path / "runs" / "g2fail"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    counts = dict(_G2_PASS)
    counts["memory_heuristic"] = (2400, 0)  # na = 0.480
    _seed_g2(db_path, "g2fail", counts)
    g2 = x5.compute_g2(run_dir, db_path=str(db_path))
    assert g2["legs"]["memory_heuristic"]["not_a_loss"] == pytest.approx(0.48)
    assert g2["legs"]["memory_heuristic"]["pass"] is False
    assert g2["strategic_pass"] is False
    assert g2["verdict"] == "FAIL"


def test_g2_rule_flip_on_borderline_strategic_leg(tmp_path):
    """A strategic leg at 0.505 point passes but its lower CI bound does not.

    Under the default rule (lower-ci on strategic legs) G2 FAILs; forcing --g2-rule
    point flips it to PASS. This is the user-ruling-pending knob.
    """
    run_dir = tmp_path / "runs" / "g2borderline"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    counts = dict(_G2_PASS)
    counts["imperfect_greedy"] = (2525, 0)  # na = 0.505; wilson lower ~0.491 < 0.50
    _seed_g2(db_path, "g2borderline", counts)

    default_g2 = x5.compute_g2(run_dir, db_path=str(db_path), g2_rule="default")
    ig = default_g2["legs"]["imperfect_greedy"]
    assert ig["rule"] == "lower-ci"
    assert ig["not_a_loss"] == pytest.approx(0.505)
    assert ig["ci_low"] < 0.50
    assert ig["pass"] is False
    assert default_g2["verdict"] == "FAIL"

    point_g2 = x5.compute_g2(run_dir, db_path=str(db_path), g2_rule="point")
    ig_p = point_g2["legs"]["imperfect_greedy"]
    assert ig_p["rule"] == "point"
    assert ig_p["pass"] is True
    assert point_g2["verdict"] == "PASS"


def test_g2_run_not_found(tmp_path):
    run_dir = tmp_path / "runs" / "missing"
    run_dir.mkdir(parents=True)
    g2 = x5.compute_g2(run_dir, db_path=str(tmp_path / "db.sqlite"))
    assert g2["available"] is False
    assert g2["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# G3-ordering leg unit tests
# ---------------------------------------------------------------------------


def test_g3_ordering_pass_non_overlapping(tmp_path):
    run_dir = tmp_path / "runs" / "g3o"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3o", _G2_PASS)  # run must exist
    _seed_mixture_exploit(db_path, "g3o", _MIX_PASS)
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, _REF_PASS)

    g3 = x5.compute_g3_ordering(run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path))
    assert g3["verdict"] == "PASS"
    assert g3["primary_pass"] is True
    for baseline in MEAN_IMP_BASELINES:
        assert g3["legs"][baseline]["primary"]["pass"] is True
        assert g3["legs"][baseline]["primary"]["overlap"] is False


def test_g3_ordering_ci_overlap_fails_and_flags(tmp_path):
    run_dir = tmp_path / "runs" / "g3overlap"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3overlap", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3overlap", _MIX_PASS)
    ref = json.loads(json.dumps(_REF_PASS))  # deep copy
    # Drag memory_heuristic's Tier-B CI down so it overlaps the mixture (ci_high 0.12).
    ref["memory_heuristic"]["lbr_tier_b"] = {
        "exploitability": 0.13, "ci_low": 0.11, "ci_high": 0.15
    }
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, ref)

    g3 = x5.compute_g3_ordering(run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path))
    leg = g3["legs"]["memory_heuristic"]["primary"]
    assert leg["overlap"] is True
    assert leg["pass"] is False
    assert g3["primary_pass"] is False
    assert g3["verdict"] == "FAIL"
    assert any("memory_heuristic" in f for f in g3["flags"])


def test_g3_ordering_ismcts_disagreement_flag_not_veto(tmp_path):
    """ISMCTS-BR disagreement is a flag by default, not a veto."""
    run_dir = tmp_path / "runs" / "g3disagree"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3disagree", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3disagree", _MIX_PASS)
    ref = json.loads(json.dumps(_REF_PASS))
    # ISMCTS reference for aggressive_snap overlaps the mixture (0.14) -> confirming FAIL,
    # while Tier-B still passes. Default: flag, gate still PASS.
    ref["aggressive_snap"]["ismcts_br"] = {
        "exploitability": 0.15, "ci_low": 0.13, "ci_high": 0.17
    }
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, ref)

    default_g3 = x5.compute_g3_ordering(run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path))
    assert default_g3["legs"]["aggressive_snap"]["disagreement_flag"] is True
    assert default_g3["verdict"] == "PASS"  # confirming-only: not a veto
    assert any("aggressive_snap" in f for f in default_g3["flags"])

    binding_g3 = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path), ismcts_binding=True
    )
    assert binding_g3["verdict"] == "FAIL"  # now ISMCTS gates


def test_g3_ordering_reference_missing(tmp_path):
    run_dir = tmp_path / "runs" / "g3noref"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3noref", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3noref", _MIX_PASS)
    g3 = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(tmp_path / "nope.json")
    )
    assert g3["reference_available"] is False
    assert g3["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Canonical row-list reference schema (cambia-415): {"schema_version", "rows"}
# is the producer's (prtcfr_baseline_exploitability.py) write_table shape and
# must be accepted as the primary reference-loader format.
# ---------------------------------------------------------------------------


def test_read_baseline_reference_accepts_row_list_shape(tmp_path):
    path = tmp_path / "ref_rows.json"
    _write_row_list_reference(path, _REF_PASS)

    ref = x5.read_baseline_reference(str(path))
    assert ref["available"] is True
    entry = ref["baselines"]["imperfect_greedy"]["lbr_tier_b"]
    assert entry["exploitability"] == pytest.approx(0.30)
    assert entry["ci_low"] == pytest.approx(0.27)
    assert entry["ci_high"] == pytest.approx(0.33)
    assert entry["games"] == 2000
    assert ref["baselines"]["aggressive_snap"]["ismcts_br"]["exploitability"] == pytest.approx(0.40)


def test_baselines_from_rows_latest_timestamp_wins_on_duplicates():
    rows = [
        {
            "baseline": "aggressive_snap", "estimator": "lbr_tier_b", "value": 0.20,
            "ci_low": 0.18, "ci_high": 0.22, "sample_count": 1000,
            "timestamp": "2026-07-01T00:00:00Z",
        },
        {
            "baseline": "aggressive_snap", "estimator": "lbr_tier_b", "value": 0.35,
            "ci_low": 0.32, "ci_high": 0.38, "sample_count": 2000,
            "timestamp": "2026-07-12T00:00:00Z",
        },
    ]
    baselines = x5._baselines_from_rows(rows)
    entry = baselines["aggressive_snap"]["lbr_tier_b"]
    assert entry["exploitability"] == pytest.approx(0.35)
    assert entry["games"] == 2000


def test_g3_ordering_row_list_and_nested_map_reference_agree(tmp_path):
    """The canonical row-list reference and the nested-map reference for the
    same underlying data must produce identical G3-ordering results."""
    run_dir = tmp_path / "runs" / "g3rows"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3rows", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3rows", _MIX_PASS)

    nested_path = tmp_path / "ref_nested.json"
    _write_reference(nested_path, _REF_PASS)
    rows_path = tmp_path / "ref_rows.json"
    _write_row_list_reference(rows_path, _REF_PASS)

    g3_nested = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(nested_path)
    )
    g3_rows = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(rows_path)
    )

    assert g3_nested["verdict"] == g3_rows["verdict"] == "PASS"
    assert g3_nested["primary_pass"] == g3_rows["primary_pass"]
    assert g3_nested["legs"] == g3_rows["legs"]


# ---------------------------------------------------------------------------
# G3-trend leg unit tests
# ---------------------------------------------------------------------------


def test_g3_trend_pass_negative_in_budget(tmp_path):
    rows = [
        {"iteration": it, "tier_a_lbr": v}
        for it, v in _decreasing_lbr(PASS_ITERS).items()
    ]
    t = x5.compute_g3_trend(rows, K_GAMES)
    assert t["window"]["valid"] is True
    assert t["budget_reached"] is True
    assert t["slope_ci"]["slope"] < 0
    assert t["ci_upper_negative"] is True
    assert t["observed_clears_mds"] is True
    assert t["verdict"] == "PASS"


def test_g3_trend_flat_slope_ci_spans_zero(tmp_path):
    rows = [
        {"iteration": it, "tier_a_lbr": v}
        for it, v in _flat_lbr(PASS_ITERS).items()
    ]
    t = x5.compute_g3_trend(rows, K_GAMES)
    assert t["window"]["valid"] is True
    assert t["budget_reached"] is True  # budget reached, but slope not significant
    sc = t["slope_ci"]
    assert sc["ci_low"] < 0 < sc["ci_high"]
    assert t["verdict"] == "FAIL"


def test_g3_trend_budget_not_reached(tmp_path):
    rows = [
        {"iteration": it, "tier_a_lbr": v}
        for it, v in _decreasing_lbr(SHORT_ITERS).items()
    ]
    t = x5.compute_g3_trend(rows, K_GAMES)
    assert t["window"]["valid"] is False
    assert t["budget_reached"] is False
    assert t["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# Budget curve
# ---------------------------------------------------------------------------


def test_budget_curve_table_cumulative_traversals():
    rows = [{"iteration": it, "tier_a_lbr": 0.5} for it in [10, 20, 30]]
    curve = x5.budget_curve_table(rows, K_GAMES)
    assert curve[0] == {"iteration": 10, "cumulative_traversals": 10 * K_GAMES, "tier_a_lbr": 0.5}
    assert [pt["cumulative_traversals"] for pt in curve] == [10 * K_GAMES, 20 * K_GAMES, 30 * K_GAMES]


# ---------------------------------------------------------------------------
# Full verdict: PASS + each FAIL mode (one axis changed per case)
# ---------------------------------------------------------------------------


def test_verdict_full_pass(tmp_path):
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-pass", _decreasing_lbr(PASS_ITERS), _G2_PASS, _MIX_PASS, _REF_PASS
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)

    assert verdict["k_games"] == K_GAMES
    assert verdict["g2_not_a_loss"]["verdict"] == "PASS"
    assert verdict["g3_ordering"]["verdict"] == "PASS"
    assert verdict["g3_trend"]["verdict"] == "PASS"
    assert verdict["overall_verdict"] == "PASS"
    assert verdict["mean_imp5_floor"]["mean_imp"] is not None
    assert len(verdict["budget_curve"]) == len(PASS_ITERS)

    summary = x5.human_summary(verdict)
    assert "OVERALL: PASS" in summary
    # exit code path
    assert x5.main([str(run_dir), "--db-path", db_path, "--baseline-ref", ref_path]) == 0


def test_verdict_fail_strategic_g2_leg(tmp_path):
    counts = dict(_G2_PASS)
    counts["aggressive_snap"] = (2400, 0)  # na = 0.48
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-g2fail", _decreasing_lbr(PASS_ITERS), counts, _MIX_PASS, _REF_PASS
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    assert verdict["g2_not_a_loss"]["verdict"] == "FAIL"
    assert verdict["g3_ordering"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["g3_trend"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["overall_verdict"] == "FAIL"
    assert x5.main([str(run_dir), "--db-path", db_path, "--baseline-ref", ref_path]) == 1


def test_verdict_fail_exploitability_ci_overlap(tmp_path):
    ref = json.loads(json.dumps(_REF_PASS))
    ref["imperfect_greedy"]["lbr_tier_b"] = {
        "exploitability": 0.13, "ci_low": 0.11, "ci_high": 0.15  # overlaps mixture 0.12
    }
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-ordfail", _decreasing_lbr(PASS_ITERS), _G2_PASS, _MIX_PASS, ref
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    assert verdict["g2_not_a_loss"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["g3_ordering"]["verdict"] == "FAIL"
    assert verdict["g3_trend"]["verdict"] == "PASS"  # unaffected axis
    assert verdict["overall_verdict"] == "FAIL"


def test_verdict_fail_flat_curve_kill(tmp_path):
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-flat", _flat_lbr(PASS_ITERS), _G2_PASS, _MIX_PASS, _REF_PASS
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    assert verdict["g3_trend"]["verdict"] == "FAIL"
    assert verdict["g3_trend"]["budget_reached"] is True  # flat-curve kill: budget reached
    assert verdict["overall_verdict"] == "FAIL"
    # The budget curve is still written on the kill.
    assert len(verdict["budget_curve"]) == len(PASS_ITERS)
    summary = x5.human_summary(verdict)
    assert "flat-curve kill" in summary


def test_verdict_fail_budget_not_reached(tmp_path):
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-short", _decreasing_lbr(SHORT_ITERS), _G2_PASS, _MIX_PASS, _REF_PASS
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    assert verdict["g3_trend"]["verdict"] == "FAIL"
    assert verdict["g3_trend"]["budget_reached"] is False
    assert verdict["overall_verdict"] == "FAIL"


def test_verdict_missing_k_fails_trend_cleanly(tmp_path):
    """No config.yaml and no --k-games: budget/trend FAILs with a message, no crash."""
    run_dir = tmp_path / "runs" / "x5-nok"
    run_dir.mkdir(parents=True)
    _write_metrics(run_dir / "metrics.jsonl", _decreasing_lbr(PASS_ITERS))
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "x5-nok", _G2_PASS)
    _seed_mixture_exploit(db_path, "x5-nok", _MIX_PASS)
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, _REF_PASS)
    verdict = x5.compute_verdict(str(run_dir), db_path=str(db_path), baseline_ref_path=str(ref_path))
    assert verdict["k_games"] is None
    assert verdict["g3_trend"]["verdict"] == "FAIL"
    assert "k_games_per_iter" in verdict["g3_trend"]["error"]
    assert verdict["overall_verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# G3-binding mode (review finding, ruled): strategic-only binding is the
# default; the random legs' Tier-B reference exploitability is a known
# estimator artifact and must never veto the verdict on its own. --g3-binding
# all restores the pre-fix all-five-binding behavior.
# ---------------------------------------------------------------------------


def test_g3_ordering_strategic_default_passes_when_only_random_leg_fails(tmp_path):
    run_dir = tmp_path / "runs" / "g3randomfail"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3randomfail", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3randomfail", _MIX_PASS)
    ref = json.loads(json.dumps(_REF_PASS))
    # Drag random_no_cambia's Tier-B CI down so it overlaps the mixture (an
    # artifact-driven tight bar, per the module docstring rationale).
    ref["random_no_cambia"]["lbr_tier_b"] = {
        "exploitability": 0.11, "ci_low": 0.09, "ci_high": 0.13
    }
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, ref)

    g3 = x5.compute_g3_ordering(run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path))
    assert g3["g3_binding"] == "strategic"
    assert g3["legs"]["random_no_cambia"]["binding"] is False
    assert g3["legs"]["random_no_cambia"]["primary"]["pass"] is False  # the leg itself still fails
    assert g3["verdict"] == "PASS"  # non-binding leg does not veto
    assert any(
        "random_no_cambia" in f and "non-binding" in f for f in g3["flags"]
    )


def test_g3_ordering_all_binding_restores_old_behavior(tmp_path):
    run_dir = tmp_path / "runs" / "g3randomfail2"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3randomfail2", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3randomfail2", _MIX_PASS)
    ref = json.loads(json.dumps(_REF_PASS))
    ref["random_no_cambia"]["lbr_tier_b"] = {
        "exploitability": 0.11, "ci_low": 0.09, "ci_high": 0.13
    }
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, ref)

    g3 = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path), g3_binding="all"
    )
    assert g3["g3_binding"] == "all"
    assert g3["legs"]["random_no_cambia"]["binding"] is True
    assert g3["verdict"] == "FAIL"


def test_verdict_records_g3_binding_mode_and_defaults_to_strategic(tmp_path):
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-g3mode", _decreasing_lbr(PASS_ITERS), _G2_PASS, _MIX_PASS, _REF_PASS
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    assert verdict["g3_ordering"]["g3_binding"] == "strategic"
    summary = x5.human_summary(verdict)
    assert "g3-binding=strategic" in summary

    verdict_all = x5.compute_verdict(
        str(run_dir), db_path=db_path, baseline_ref_path=ref_path, g3_binding="all"
    )
    assert verdict_all["g3_ordering"]["g3_binding"] == "all"
    assert "g3-binding=all" in x5.human_summary(verdict_all)


def test_g3_ordering_invalid_binding_raises(tmp_path):
    run_dir = tmp_path / "runs" / "g3bogus"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        x5.compute_g3_ordering(run_dir, db_path=str(tmp_path / "db.sqlite"), g3_binding="bogus")


# ---------------------------------------------------------------------------
# ISMCTS-binding fail-closed (review finding): with --ismcts-binding set but
# zero confirming comparisons available, the leg must FAIL CLOSED with an
# explicit reason rather than passing vacuously by degrading to primary-only.
# Default (confirming-only) mode keeps the graceful degrade.
# ---------------------------------------------------------------------------


def test_ismcts_binding_fails_closed_when_mixture_has_no_ismcts_row(tmp_path):
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-noismcts", _decreasing_lbr(PASS_ITERS), _G2_PASS,
        {"lbr_tier_b": _MIX_PASS["lbr_tier_b"]}, _REF_PASS,
    )
    verdict = x5.compute_verdict(
        str(run_dir), db_path=db_path, baseline_ref_path=ref_path, ismcts_binding=True
    )
    g3 = verdict["g3_ordering"]
    assert g3["primary_pass"] is True  # primary alone would have passed
    assert g3["ismcts_binding_fail_reason"] == "ismcts binding requested but no ismcts rows"
    assert g3["verdict"] == "FAIL"
    assert verdict["overall_verdict"] == "FAIL"
    summary = x5.human_summary(verdict)
    assert "ismcts binding requested but no ismcts rows" in summary


def test_ismcts_binding_fails_closed_when_reference_has_no_ismcts_rows(tmp_path):
    run_dir = tmp_path / "runs" / "g3norefismcts"
    run_dir.mkdir(parents=True)
    db_path = tmp_path / "db.sqlite"
    _seed_g2(db_path, "g3norefismcts", _G2_PASS)
    _seed_mixture_exploit(db_path, "g3norefismcts", _MIX_PASS)  # has both keys
    ref = {b: {"lbr_tier_b": v["lbr_tier_b"]} for b, v in _REF_PASS.items()}  # no ismcts_br entries
    ref_path = tmp_path / "ref.json"
    _write_reference(ref_path, ref)

    g3 = x5.compute_g3_ordering(
        run_dir, db_path=str(db_path), baseline_ref_path=str(ref_path), ismcts_binding=True
    )
    assert g3["ismcts_binding_fail_reason"] == "ismcts binding requested but no ismcts rows"
    assert g3["verdict"] == "FAIL"


def test_ismcts_binding_default_confirming_only_still_degrades_gracefully(tmp_path):
    """Default (ismcts_binding=False) keeps passing on primary alone when
    there are no confirming rows -- only --ismcts-binding fails closed."""
    run_dir, db_path, ref_path = _make_run(
        tmp_path, "x5-noismcts2", _decreasing_lbr(PASS_ITERS), _G2_PASS,
        {"lbr_tier_b": _MIX_PASS["lbr_tier_b"]}, _REF_PASS,
    )
    verdict = x5.compute_verdict(str(run_dir), db_path=db_path, baseline_ref_path=ref_path)
    g3 = verdict["g3_ordering"]
    assert g3["ismcts_binding_fail_reason"] is None
    assert g3["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Key-constant provenance (review finding, FIX 2): the mixture-exploitability
# metric baseline keys must be sourced from src.run_db, not duplicated as
# string literals. Patch run_db's constants and reload the script module to
# confirm it follows rather than pinning its own frozen copy.
# ---------------------------------------------------------------------------


def test_exploit_keys_are_sourced_from_run_db_and_follow_patches():
    import importlib

    orig_primary = run_db.EXPLOIT_TIER_B_LBR_BASELINE
    orig_confirming = run_db.EXPLOIT_ISMCTS_BR_BASELINE
    assert x5.EXPLOIT_PRIMARY_KEY == orig_primary
    assert x5.EXPLOIT_CONFIRMING_KEY == orig_confirming
    try:
        run_db.EXPLOIT_TIER_B_LBR_BASELINE = "patched_primary_key"
        run_db.EXPLOIT_ISMCTS_BR_BASELINE = "patched_confirming_key"
        importlib.reload(x5)
        assert x5.EXPLOIT_PRIMARY_KEY == "patched_primary_key"
        assert x5.EXPLOIT_CONFIRMING_KEY == "patched_confirming_key"
    finally:
        run_db.EXPLOIT_TIER_B_LBR_BASELINE = orig_primary
        run_db.EXPLOIT_ISMCTS_BR_BASELINE = orig_confirming
        importlib.reload(x5)
        assert x5.EXPLOIT_PRIMARY_KEY == orig_primary
        assert x5.EXPLOIT_CONFIRMING_KEY == orig_confirming
