"""tests/test_prtcfr_x2_verdict.py

Scoped tests for scripts/prtcfr_x2_verdict.py: the X2 re-spec gate verdict tool.

The tool encodes a FROZEN pre-registration (x2-respec-preregistration.md) plus
amendment A3; a deviation poisons a gate verdict, so the tests pin the ground
truths from the closed record and the exact branch/boundary arithmetic:

  (i)   replaying the frozen plateau stop rule over the fresh 1000-horizon series
        fires at iteration 330 with NashConv 0.08894883814554927.
  (ii)  replaying over the original 530-iteration series fires at iteration 350
        with 0.08804.
  (iii) variance band B and the rule-2 floor-move threshold max(B, 0.10).
  (iv)  A3-3/A3-4 boundary: a crossing at exactly iteration 700 counts, a first
        crossing after 700 does not, checkpoints only land every 10 iterations.
  (v)   rules 3/4/5 branch logic, including C2 auto-queue only when BOTH C0 and
        C1 move the floor.
  (vi)  stage gating: insufficient data never yields a PASS/FAIL/INDETERMINATE.

Series (i) and (ii) are embedded verbatim from the closed record
(cfr/runs/v0.4-x2-fresh-1000-xpu/resume_state.json and the convergence-audit
fixture x2_trajectories.csv). Synthetic series drive the branches the record has
no fixture for; the frozen-stop generator is sanity-checked so its plateau value
is exactly recoverable.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import prtcfr_x2_verdict as x2  # noqa: E402

# ---------------------------------------------------------------------------
# Ground-truth series from the closed record
# ---------------------------------------------------------------------------

# v0.4-x2-fresh-1000-xpu controller.history (exact float64 NashConv, iters 1..330).
FRESH_SERIES = [
    (1, 1.6387356755350615),
    (10, 0.49660096169629664),
    (20, 0.3425478860832225),
    (30, 0.29296220236628584),
    (40, 0.2538829336714554),
    (50, 0.21694719785768263),
    (60, 0.18544832321619514),
    (70, 0.15327140914589008),
    (80, 0.13854530817640584),
    (90, 0.12889786639205114),
    (100, 0.1300570614935329),
    (110, 0.12677315894013902),
    (120, 0.1207570243113737),
    (130, 0.11858794859140351),
    (140, 0.11432192546263314),
    (150, 0.11065064587712842),
    (160, 0.11127347460482273),
    (170, 0.11019966054502178),
    (180, 0.10888378401281551),
    (190, 0.10906799182646404),
    (200, 0.10533939916907581),
    (210, 0.10338298055213552),
    (220, 0.09979755881531682),
    (230, 0.09705366473273916),
    (240, 0.09716159419836262),
    (250, 0.09633778455760966),
    (260, 0.0963053055769778),
    (270, 0.09347567540680327),
    (280, 0.09116417039102659),
    (290, 0.09082096350417737),
    (300, 0.09075384444848889),
    (310, 0.08959472522162801),
    (320, 0.08916034771122605),
    (330, 0.08894883814554927),
]

# x2-530-original neural trace (x2_trajectories.csv), iters 1..530.
ORIGINAL_530_SERIES = [
    (1, 1.65338),
    (10, 0.47586),
    (20, 0.35809),
    (30, 0.31938),
    (40, 0.2588),
    (50, 0.22483),
    (60, 0.20798),
    (70, 0.18761),
    (80, 0.17428),
    (90, 0.15906),
    (100, 0.14281),
    (110, 0.1313),
    (120, 0.1225),
    (130, 0.11613),
    (140, 0.11969),
    (150, 0.11426),
    (160, 0.11327),
    (170, 0.11114),
    (180, 0.1088),
    (190, 0.10458),
    (200, 0.10043),
    (210, 0.10073),
    (220, 0.10066),
    (230, 0.09859),
    (240, 0.09611),
    (250, 0.09653),
    (260, 0.09675),
    (270, 0.09533),
    (280, 0.09378),
    (290, 0.09091),
    (300, 0.08949),
    (310, 0.08786),
    (320, 0.08713),
    (330, 0.08742),
    (340, 0.08795),
    (350, 0.08804),
    (360, 0.09103),
    (370, 0.08989),
    (380, 0.08851),
    (390, 0.08741),
    (400, 0.08715),
    (410, 0.08582),
    (420, 0.0845),
    (430, 0.0832),
    (440, 0.08314),
    (450, 0.08122),
    (460, 0.07868),
    (470, 0.0766),
    (480, 0.07445),
    (490, 0.07285),
    (500, 0.07153),
    (510, 0.07043),
    (520, 0.06911),
    (530, 0.06834),
]


# ---------------------------------------------------------------------------
# Synthetic series builders
# ---------------------------------------------------------------------------


def stopping_series(plateau_value, continuation=None):
    """A series whose frozen plateau stop fires at exactly ``plateau_value``.

    Steep descent (iters 10..120) into a flat tail (iters 130..200) at
    ``plateau_value``; the plateau rule fires inside the flat, so the recovered
    plateau NashConv is ``plateau_value`` regardless of the exact stop iteration.
    ``continuation`` is an optional list of (iter, nc) pairs at iters > 200 that
    feed the A3 read-out (they never change the stop, which the replay reaches
    first).
    """
    pts = [(1, 1.0)]
    desc_iters = list(range(10, 121, 10))
    hi, lo = 0.9, plateau_value + 0.05
    for k, it in enumerate(desc_iters):
        frac = k / (len(desc_iters) - 1)
        pts.append((it, round(hi + (lo - hi) * frac, 6)))
    for it in range(130, 201, 10):
        pts.append((it, plateau_value))
    if continuation:
        pts += list(continuation)
    return pts


def flat_continuation(value, lo=210, hi=700):
    """Continuation checkpoints at a constant ``value`` over [lo, hi] step 10."""
    return [(it, value) for it in range(lo, hi + 1, 10)]


def never_stopping_series(n=20):
    """A steep, always-positive exponential descent that never plateau-stops.

    Constant relative improvement per step (~13% / 10 iters) keeps the plateau
    rate far above the 0.5% threshold, so the frozen stop rule never fires.
    """
    return [(1, 1.0)] + [(10 * k, round(0.82**k, 6)) for k in range(1, n + 1)]


# ---------------------------------------------------------------------------
# (i)/(ii) Frozen-stop replay ground truths
# ---------------------------------------------------------------------------


def test_replay_fresh_series_fires_at_330():
    result = x2.replay_frozen_stop(FRESH_SERIES, x2.FROZEN_STOP_PARAMS)
    assert result["stopped"] is True
    assert result["stop_iter"] == 330
    assert result["stop_nashconv"] == pytest.approx(0.08894883814554927, abs=1e-15)


def test_replay_original_530_series_fires_at_350():
    result = x2.replay_frozen_stop(ORIGINAL_530_SERIES, x2.FROZEN_STOP_PARAMS)
    assert result["stopped"] is True
    assert result["stop_iter"] == 350
    assert result["stop_nashconv"] == pytest.approx(0.08804, abs=1e-9)


def test_build_cell_readout_recovers_fresh_stop():
    cell = x2.build_cell_readout("C-rep", FRESH_SERIES)
    assert cell.replay_stopped is True
    assert cell.replay_stop_iter == 330
    assert cell.replay_stop_nashconv == pytest.approx(0.08894883814554927, abs=1e-15)


def test_synthetic_stopping_series_recovers_plateau_value():
    for value in (0.088, 0.079, 0.070, 0.056):
        cell = x2.build_cell_readout("cell", stopping_series(value))
        assert cell.replay_stopped is True
        assert cell.replay_stop_nashconv == pytest.approx(value, abs=1e-12)


# ---------------------------------------------------------------------------
# (iii) Variance band B and rule-2 threshold arithmetic
# ---------------------------------------------------------------------------


def test_variance_band_arithmetic():
    crep = x2.build_cell_readout("C-rep", stopping_series(0.088))
    band = x2.variance_band(crep)
    assert band == pytest.approx(abs(0.088 - 0.08895) / 0.08895, abs=1e-15)


def test_floor_move_threshold_floors_at_0p10():
    # Small band -> the 0.10 floor dominates.
    assert x2.floor_move_threshold(0.0107) == pytest.approx(0.10)
    # Large band -> B dominates.
    assert x2.floor_move_threshold(0.22) == pytest.approx(0.22)


def test_moved_the_floor_uses_baseline_relative_improvement():
    threshold = 0.10
    target = 0.08895 * (1.0 - threshold)  # 0.080055
    moved = x2.build_cell_readout("C0", stopping_series(0.070))  # 0.070 < target
    not_moved = x2.build_cell_readout("C1", stopping_series(0.085))  # 0.085 > target
    assert x2.moved_the_floor(moved, threshold) is True
    assert x2.moved_the_floor(not_moved, threshold) is False
    # A cell that has not stopped has no plateau value -> None (undefined).
    pre = x2.build_cell_readout("C2", never_stopping_series())
    assert pre.replay_stopped is False
    assert x2.moved_the_floor(pre, threshold) is None
    assert target == pytest.approx(0.080055, abs=1e-9)


# ---------------------------------------------------------------------------
# (iv) A3-3 / A3-4 read-out boundary
# ---------------------------------------------------------------------------


def test_a3_bar_cross_at_exactly_700_counts():
    cont = [(it, 0.06) for it in range(210, 700, 10)] + [(700, x2.BAR_RESPEC)]
    cell = x2.build_cell_readout("C-rep", stopping_series(0.088, cont))
    assert cell.bar_cross_iter == 700
    assert cell.crossed_bar() is True
    assert cell.floored() is False


def test_a3_first_cross_after_700_does_not_count_floored():
    # Above the bar through 700, first sub-bar checkpoint at 710: A3-4 floored,
    # the 700 window is the verdict input even though the run later crosses.
    cont = [(it, 0.06) for it in range(210, 701, 10)] + [(710, 0.056)]
    cell = x2.build_cell_readout("C-rep", stopping_series(0.088, cont))
    assert cell.bar_cross_iter is None
    assert cell.crossed_bar() is False
    assert cell.reaches_readout_end is True
    assert cell.floored() is True


def test_a3_stopped_but_not_reached_700_is_neither():
    # Stopped at 180, continuation ends at 300 above the bar: neither crossed nor
    # floored -> the read-out is still undecided (pending), not a verdict.
    cont = flat_continuation(0.08, lo=210, hi=300)
    cell = x2.build_cell_readout("C-rep", stopping_series(0.088, cont))
    assert cell.crossed_bar() is False
    assert cell.reaches_readout_end is False
    assert cell.floored() is False


def test_a3_readout_only_sees_grid_checkpoints():
    # Between-grid dips are invisible; only the 10-iter checkpoints are read.
    cont = [(it, 0.06) for it in range(210, 701, 10)]  # never <= bar on the grid
    cell = x2.build_cell_readout("C-rep", stopping_series(0.088, cont))
    assert cell.bar_cross_iter is None
    assert cell.floored() is True


# ---------------------------------------------------------------------------
# Descent-monotone-under-the-stop-rule (rule 3 conjunct)
# ---------------------------------------------------------------------------


def test_monotone_true_on_clean_descent():
    ev = x2._descent_monotone_under_stop_rule(FRESH_SERIES, 330, x2.FROZEN_STOP_PARAMS)
    assert ev["monotone"] is True
    assert ev["divergence_fired_at"] is None


def test_monotone_false_on_sustained_rise_before_crossing():
    # After the stop the metric rises for 5 consecutive checks (a divergence
    # excursion) then crashes below the bar: the bar-cross is real but the descent
    # is not monotone under the stop rule.
    rise = [(210, 0.089), (220, 0.090), (230, 0.091), (240, 0.092), (250, 0.093)]
    fall = [
        (it, round(0.093 - (0.093 - 0.056) * ((it - 250) / (400 - 250)), 6))
        for it in range(260, 401, 10)
    ]
    fall[-1] = (400, 0.056)
    cell = x2.build_cell_readout("C-rep", stopping_series(0.088, rise + fall))
    assert cell.bar_cross_iter == 400
    assert cell.descent_monotone is False
    assert cell.monotone_evidence["divergence_fired_at"] is not None


# ---------------------------------------------------------------------------
# (v) Rules 3/4/5 branch logic (via compute_verdict on pre-built cells)
# ---------------------------------------------------------------------------


def _floored_crep():
    """C-rep floored at or above the bar (A3-4): drives the C0/C1 fork."""
    return x2.build_cell_readout("C-rep", stopping_series(0.088, flat_continuation(0.06)))


def test_rule5_fail_when_neither_c0_nor_c1_moves():
    c0 = x2.build_cell_readout("C0", stopping_series(0.088, flat_continuation(0.088)))
    c1 = x2.build_cell_readout("C1", stopping_series(0.086, flat_continuation(0.086)))
    verdict = x2.compute_verdict(crep=_floored_crep(), c0=c0, c1=c1)
    assert verdict["stage"] == "d"
    assert verdict["overall_verdict"] == "FAIL"
    assert verdict["rule_4_5"]["rule"] == 5
    assert verdict["rule_4_5"]["c2_auto_queued"] is False


def test_rule4_indeterminate_one_moved_no_c2_autoqueue():
    c0 = x2.build_cell_readout("C0", stopping_series(0.070, flat_continuation(0.070)))
    c1 = x2.build_cell_readout("C1", stopping_series(0.088, flat_continuation(0.088)))
    verdict = x2.compute_verdict(crep=_floored_crep(), c0=c0, c1=c1)
    assert verdict["overall_verdict"] == "INDETERMINATE"
    assert verdict["rule_4_5"]["rule"] == 4
    assert verdict["rule_4_5"]["c0_moved_floor"] is True
    assert verdict["rule_4_5"]["c1_moved_floor"] is False
    assert verdict["rule_4_5"]["c2_auto_queued"] is False


def test_rule4_indeterminate_both_moved_autoqueues_c2():
    c0 = x2.build_cell_readout("C0", stopping_series(0.070, flat_continuation(0.070)))
    c1 = x2.build_cell_readout("C1", stopping_series(0.069, flat_continuation(0.069)))
    verdict = x2.compute_verdict(crep=_floored_crep(), c0=c0, c1=c1)
    assert verdict["overall_verdict"] == "INDETERMINATE"
    assert verdict["rule_4_5"]["rule"] == 4
    assert verdict["rule_4_5"]["c0_moved_floor"] is True
    assert verdict["rule_4_5"]["c1_moved_floor"] is True
    assert verdict["rule_4_5"]["c2_auto_queued"] is True


def _a3_continued_cell(name, plateau=0.088, floor=0.065):
    """A cell that plateau-stops at the shared ~330 flat (``plateau``) but in the
    A3 continuation descends to ``floor`` by iteration 700 (above bar_respec)."""
    cont = [
        (it, round(plateau - (plateau - floor) * ((it - 210) / (700 - 210)), 6))
        for it in range(210, 701, 10)
    ]
    cont[-1] = (700, floor)
    return x2.build_cell_readout(name, stopping_series(plateau, cont))


def test_rule2_reads_a3_continuation_floor_not_the_330_flat():
    # C0/C1 share the ~330 flat at 0.088 but descend in the A3 continuation to
    # 0.065 by iter 700 (still above bar_respec 0.057). Rule 2 must read the A3
    # continuation floor (0.065 -> moved), NOT the frozen ~330 plateau stop
    # (0.088 -> not moved). Reading the flat drives rule 4 out of reach and
    # misfires rule 5 FAIL; the continuation floor gives rule 4 INDETERMINATE
    # with C2 auto-queued.
    c0 = _a3_continued_cell("C0")
    c1 = _a3_continued_cell("C1")
    assert c0.replay_stop_nashconv == pytest.approx(0.088, abs=1e-9)
    assert c0.readout_min_nashconv == pytest.approx(0.065, abs=1e-9)
    # scored on the flat this would be False; on the continuation floor it is True
    assert x2.moved_the_floor(c0, 0.10) is True
    verdict = x2.compute_verdict(crep=_floored_crep(), c0=c0, c1=c1)
    assert verdict["overall_verdict"] == "INDETERMINATE"
    assert verdict["rule_4_5"]["rule"] == 4
    assert verdict["rule_4_5"]["c0_moved_floor"] is True
    assert verdict["rule_4_5"]["c1_moved_floor"] is True
    assert verdict["rule_4_5"]["c2_auto_queued"] is True
    c0_cell = verdict["rule_2_floor_move"]["cells"]["C0"]
    assert c0_cell["readout_floor_nashconv"] == pytest.approx(0.065, abs=1e-9)
    assert c0_cell["plateau_stop_nashconv"] == pytest.approx(0.088, abs=1e-9)


def _bar_crossing_cell(name, cross_iter=560, cross_value=0.056):
    ramp = {
        it: round(0.088 - (0.088 - cross_value) * ((it - 200) / (cross_iter - 200)), 6)
        for it in range(210, cross_iter + 1, 10)
    }
    ramp[cross_iter] = cross_value
    return x2.build_cell_readout(name, stopping_series(0.088, sorted(ramp.items())))


def _confirm_seed(name, plateau_iter=500, plateau_value=0.069):
    ramp = {
        it: round(
            0.088 - (0.088 - plateau_value) * ((it - 200) / (plateau_iter - 200)), 6
        )
        for it in range(210, plateau_iter + 1, 10)
    }
    ramp[plateau_iter] = plateau_value
    return x2.build_cell_readout(name, stopping_series(0.088, sorted(ramp.items())))


def test_rule3_pass_requires_bar_cross_plus_two_confirm_seeds():
    verdict = x2.compute_verdict(
        crep=_bar_crossing_cell("C-rep"),
        confirm_seeds=[_confirm_seed("seed-2"), _confirm_seed("seed-3")],
    )
    assert verdict["stage"] == "e"
    assert verdict["overall_verdict"] == "PASS"
    assert verdict["rule_3_pass"]["passing_cell"] == "C-rep"
    assert verdict["rule_3_pass"]["descent_monotone"] is True
    assert len(verdict["rule_3_pass"]["confirm_seeds_passed"]) == 2


def test_rule3_bar_cross_without_enough_confirm_seeds_is_pending():
    verdict = x2.compute_verdict(
        crep=_bar_crossing_cell("C-rep"),
        confirm_seeds=[_confirm_seed("seed-2")],  # only one
    )
    assert verdict["overall_verdict"] == "PENDING"
    assert any("confirm seed" in p for p in verdict["pending"])


def test_rule3_confirm_seed_must_be_within_1p25_bar():
    # A seed plateauing above 1.25 * bar_respec (0.07125) does not confirm.
    weak_seed = _confirm_seed("seed-2", plateau_iter=500, plateau_value=0.072)
    assert weak_seed.confirm_passes() is False
    good_seed = _confirm_seed("seed-3", plateau_iter=500, plateau_value=0.071)
    assert good_seed.confirm_passes() is True


# ---------------------------------------------------------------------------
# (vi) Stage gating: insufficient data never yields a verdict
# ---------------------------------------------------------------------------


def test_stage_a_crep_pre_stop_is_pending_only():
    verdict = x2.compute_verdict(
        crep=x2.build_cell_readout("C-rep", never_stopping_series())
    )
    assert verdict["stage"] == "a"
    assert verdict["overall_verdict"] == "PENDING"
    assert verdict["rule_1_variance_band"]["available"] is False


def test_stage_b_stopped_but_no_readout_reports_b_but_pending():
    crep = x2.build_cell_readout("C-rep", stopping_series(0.088))  # stop only, no cont
    verdict = x2.compute_verdict(crep=crep)
    assert verdict["overall_verdict"] == "PENDING"
    assert verdict["rule_1_variance_band"]["available"] is True
    assert verdict["rule_1_variance_band"]["B"] == pytest.approx(
        abs(0.088 - 0.08895) / 0.08895, abs=1e-12
    )


def test_c0c1_fork_pending_when_cells_missing():
    # C-rep floored but no C0/C1 provided: never a premature FAIL.
    verdict = x2.compute_verdict(crep=_floored_crep())
    assert verdict["overall_verdict"] == "PENDING"
    assert verdict["stage"] == "d"
    assert any("C0" in p or "C1" in p for p in verdict["pending"])


def test_c0c1_fork_pending_when_c1_not_read_out_to_700():
    c0 = x2.build_cell_readout("C0", stopping_series(0.070, flat_continuation(0.070)))
    c1 = x2.build_cell_readout(
        "C1", stopping_series(0.088, flat_continuation(0.088, hi=400))
    )  # only to 400
    verdict = x2.compute_verdict(crep=_floored_crep(), c0=c0, c1=c1)
    assert verdict["overall_verdict"] == "PENDING"
    assert any("700" in p for p in verdict["pending"])


def test_no_crep_is_pending():
    verdict = x2.compute_verdict()
    assert verdict["overall_verdict"] == "PENDING"
    assert verdict["stage"] == "a"


# ---------------------------------------------------------------------------
# Run-directory loader + discrepancy detection
# ---------------------------------------------------------------------------


def _write_run_dir(
    tmp_path,
    name,
    series,
    stopped=None,
    best_iteration=None,
    log_stop_iter=None,
    params=None,
):
    run_dir = tmp_path / name
    (run_dir / "logs").mkdir(parents=True)
    controller = {
        "history": [{"iteration": float(it), "metric": float(nc)} for it, nc in series],
        "stopped": stopped,
        "best_iteration": best_iteration,
    }
    controller.update(params or x2.FROZEN_STOP_PARAMS)
    with open(run_dir / "resume_state.json", "w", encoding="utf-8") as fh:
        json.dump({"controller": controller}, fh)
    if log_stop_iter is not None:
        with open(run_dir / "logs" / "training.log", "w", encoding="utf-8") as fh:
            fh.write(
                "2026-07-14 06:39:50 - INFO - [prtcfr] stability iter=%d "
                "nashconv=0.08895 best_iter=%d best=0.08895 worse_streak=0 stop=True\n"
                % (log_stop_iter, log_stop_iter)
            )
            fh.write(
                "2026-07-14 06:39:50 - INFO - [prtcfr] early-stop at iter=%d; "
                "deployable window pinned to [1..%d]\n" % (log_stop_iter, log_stop_iter)
            )
    return run_dir


def test_load_cell_reads_resume_state_and_crosschecks_log(tmp_path):
    run_dir = _write_run_dir(
        tmp_path,
        "v0.4-x2-fresh",
        FRESH_SERIES,
        stopped=True,
        best_iteration=330,
        log_stop_iter=330,
    )
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.load_error is None
    assert cell.replay_stop_iter == 330
    assert cell.replay_stop_nashconv == pytest.approx(0.08894883814554927, abs=1e-15)
    assert cell.recorded_stopped is True
    assert cell.recorded_stop_iter == 330
    assert cell.discrepancy is None


def test_load_cell_missing_resume_state_is_load_error(tmp_path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.load_error is not None
    assert not cell.series


def test_discrepancy_recorded_stopped_but_replay_does_not_stop(tmp_path):
    # A steep, never-flat descent that never triggers the plateau rule, but the
    # record claims stopped=True: the tool must refuse to render a verdict.
    run_dir = _write_run_dir(
        tmp_path,
        "bad-run",
        never_stopping_series(),
        stopped=True,
        best_iteration=200,
        log_stop_iter=200,
    )
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.discrepancy is not None
    verdict = x2.compute_verdict(crep=cell)
    assert verdict["overall_verdict"] == "DISCREPANCY"
    assert verdict["discrepancies"]


def test_discrepancy_stop_iter_mismatch(tmp_path):
    # Replay of FRESH_SERIES fires at 330, but the log records 320: mismatch.
    run_dir = _write_run_dir(
        tmp_path,
        "fresh",
        FRESH_SERIES,
        stopped=True,
        best_iteration=330,
        log_stop_iter=320,
    )
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.discrepancy is not None
    assert "320" in cell.discrepancy and "330" in cell.discrepancy


def test_divergence_stop_mode_is_not_a_false_discrepancy(tmp_path):
    # A3-3 permits a divergence stop mode. On a clean monotone descent the
    # divergence rule never fires, so the run records stopped=False while the
    # frozen plateau replay (always plateau) fires at the ~330 flat. That is not
    # a discrepancy: the run was governed by a different, authorized rule.
    series = _bar_crossing_cell("C-rep").series
    params = dict(x2.FROZEN_STOP_PARAMS)
    params["stop_mode"] = "divergence"
    run_dir = _write_run_dir(
        tmp_path,
        "crep-div",
        series,
        stopped=False,
        best_iteration=560,
        params=params,
    )
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.recorded_stop_mode == "divergence"
    assert cell.recorded_stopped is False
    assert cell.replay_stopped is True  # frozen plateau still fires at the flat
    assert cell.discrepancy is None  # no false discrepancy


def test_non_frozen_plateau_params_flagged_as_drift(tmp_path):
    # A run whose controller recorded a loose plateau_rel_improvement (0.05 vs
    # frozen 0.005) is scored on the FROZEN params (not the run's), and the
    # deviation is flagged loudly instead of silently adopted.
    params = dict(x2.FROZEN_STOP_PARAMS)
    params["plateau_rel_improvement"] = 0.05
    run_dir = _write_run_dir(
        tmp_path,
        "drift",
        FRESH_SERIES,
        stopped=True,
        best_iteration=330,
        log_stop_iter=330,
        params=params,
    )
    cell = x2.load_cell(str(run_dir), "C-rep")
    # replay used the frozen 0.005, so the stop is still the real 330 flat
    assert cell.replay_stop_iter == 330
    assert cell.replay_stop_nashconv == pytest.approx(0.08894883814554927, abs=1e-15)
    assert cell.param_drift is not None
    assert "plateau_rel_improvement" in cell.param_drift
    assert cell.discrepancy is not None
    verdict = x2.compute_verdict(crep=cell)
    assert verdict["overall_verdict"] == "DISCREPANCY"


def test_truncated_warmstart_history_is_load_error(tmp_path):
    # controller.history spanning only iters 540..600 (a net-only warm-start) is
    # not a valid X2 series source; the loader refuses it rather than replaying
    # the plateau stop over the tail and emitting a wrong verdict.
    tail = [(it, 0.20) for it in range(540, 601, 10)]
    run_dir = _write_run_dir(tmp_path, "warm", tail, stopped=True, best_iteration=580)
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.load_error is not None
    assert "540" in cell.load_error
    assert not cell.replay_stopped
    verdict = x2.compute_verdict(crep=cell)
    assert verdict["overall_verdict"] == "PENDING"


def test_absent_training_log_records_crosscheck_note(tmp_path):
    # No logs/training.log: the stop-iteration cross-check cannot run. The tool
    # records a note so the verdict is not presented as fully cross-checked,
    # instead of silently dropping the check.
    run_dir = _write_run_dir(
        tmp_path, "nolog", FRESH_SERIES, stopped=True, best_iteration=330
    )  # no log_stop_iter -> no training.log
    assert not (run_dir / "logs" / "training.log").is_file()
    cell = x2.load_cell(str(run_dir), "C-rep")
    assert cell.crosscheck_note is not None
    assert "training.log" in cell.crosscheck_note
    verdict = x2.compute_verdict(crep=cell)
    assert any("training.log" in n for n in verdict["crosscheck_notes"])


# ---------------------------------------------------------------------------
# Constants / bar consistency
# ---------------------------------------------------------------------------


def test_bar_matches_prtcfr_eval_constant():
    assert x2._eval_bar_from_source() == pytest.approx(x2.BAR_RESPEC, abs=1e-12)
    assert x2._EVAL_BAR_CONSISTENT is True


def test_frozen_constants_are_exact():
    assert x2.BASELINE_NASHCONV == 0.08895
    assert x2.BAR_RESPEC == 0.057
    assert x2.CONFIRM_SEED_BAR == pytest.approx(0.07125, abs=1e-12)
    assert x2.A3_READOUT_MAX_ITER == 700
    assert x2.FLOOR_MOVE_MIN_REL == 0.10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_pending_exit_code_and_results_block(tmp_path, capsys):
    run_dir = _write_run_dir(
        tmp_path, "crep", stopping_series(0.088), stopped=True, best_iteration=180
    )
    exit_code = x2.main([str(run_dir)])
    assert exit_code == 3  # PENDING
    out = capsys.readouterr().out
    assert "OVERALL: PENDING" in out
    assert "Results append block" in out


def test_cli_writes_json_and_pass_exit_code(tmp_path):
    crep = _write_run_dir(
        tmp_path,
        "crep",
        _bar_crossing_cell("C-rep").series,
        stopped=True,
        best_iteration=180,
    )
    seed2 = _write_run_dir(
        tmp_path, "seed2", _confirm_seed("s").series, stopped=True, best_iteration=180
    )
    seed3 = _write_run_dir(
        tmp_path, "seed3", _confirm_seed("s").series, stopped=True, best_iteration=180
    )
    out_path = tmp_path / "verdict.json"
    exit_code = x2.main(
        [
            str(crep),
            "--confirm-seed-dir",
            str(seed2),
            "--confirm-seed-dir",
            str(seed3),
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0  # PASS
    with open(out_path, "r", encoding="utf-8") as fh:
        persisted = json.load(fh)
    assert persisted["overall_verdict"] == "PASS"
    assert persisted["rule_3_pass"]["passing_cell"] == "C-rep"


def test_cli_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        x2.main(["--help"])
    assert exc_info.value.code == 0
