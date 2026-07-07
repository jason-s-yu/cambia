"""Scoped tests for PRT-CFR late-training stability.

Covers the best-snapshot / early-stop controller, the deployable manifest
round-trip, the per-iteration peak-LR schedule, and the trainer wiring
(config-gated knobs plumb through; defaults reproduce the original path;
stability_enabled + eval_fn drives the controller, writes the manifest, and
early-stops on the exploitability trend).
"""

import json
import os
import types

import pytest
import torch

from src.config import load_config, PRTCFRConfig
from src.cfr.prtcfr_net import PRTCFRNet
from src.cfr.prtcfr_stability import (
    DEPLOYABLE_MANIFEST,
    BestSnapshotController,
    read_deployable_iters,
    read_deployable_manifest,
    write_deployable_manifest,
)
from src.cfr.prtcfr_trainer import (
    PRTCFRTinyTrainer,
    _peak_lr_for_iter,
)
from tools.tiny_solver import build_tree


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_2CARD = "config/tiny_2card_plateau.yaml"


# ---------------------------------------------------------------------------
# BestSnapshotController
# ---------------------------------------------------------------------------


def test_controller_tracks_best_min_mode():
    c = BestSnapshotController(rel_tolerance=0.15, patience=3, min_iters=1, mode="min")
    for it, m in [(1, 0.5), (2, 0.3), (3, 0.4), (4, 0.2), (5, 0.25)]:
        c.update(it, m)
    assert c.best_iteration == 4
    assert c.best_metric == pytest.approx(0.2)


def test_controller_tracks_best_max_mode():
    c = BestSnapshotController(rel_tolerance=0.15, patience=3, min_iters=1, mode="max")
    for it, m in [(1, 0.4), (2, 0.6), (3, 0.55), (4, 0.7), (5, 0.65)]:
        c.update(it, m)
    assert c.best_iteration == 4
    assert c.best_metric == pytest.approx(0.7)


def test_controller_early_stop_after_patience_worse_checks():
    # best=0.20 at iter 4; tolerance band 0.20*1.15=0.23. Checks above it for
    # `patience` consecutive evals trip the stop.
    c = BestSnapshotController(rel_tolerance=0.15, patience=2, min_iters=1, mode="min")
    stops = []
    for it, m in [(1, 0.5), (2, 0.2), (3, 0.30), (4, 0.35)]:
        d = c.update(it, m)
        stops.append(d.should_stop)
    assert c.best_iteration == 2
    # iter 3 first-worse (streak 1, no stop), iter 4 second-worse (streak 2 -> stop)
    assert stops == [False, False, False, True]


def test_controller_tolerance_band_resets_streak():
    # a check within the tolerance band of best must NOT count toward the stop.
    c = BestSnapshotController(rel_tolerance=0.20, patience=2, min_iters=1, mode="min")
    c.update(1, 0.20)          # best
    c.update(2, 0.30)          # worse (> 0.24) streak 1
    d3 = c.update(3, 0.22)     # within band (<= 0.24) -> reset
    assert d3.num_worse_since_best == 0
    d4 = c.update(4, 0.30)     # worse streak 1 again (not 2)
    assert d4.should_stop is False


def test_controller_respects_min_iters():
    c = BestSnapshotController(rel_tolerance=0.1, patience=1, min_iters=100, mode="min")
    c.update(1, 0.2)
    d = c.update(2, 0.9)       # worse, but below min_iters -> no stop
    assert d.should_stop is False


def test_controller_deployable_window():
    c = BestSnapshotController(mode="min")
    for it, m in [(1, 0.5), (2, 0.3), (3, 0.4)]:
        c.update(it, m)
    assert c.deployable_iters([1, 2, 3]) == [1, 2]
    # no checks yet -> whole set deployable
    fresh = BestSnapshotController(mode="min")
    assert fresh.deployable_iters([1, 2, 3]) == [1, 2, 3]


def test_controller_pins_original_x2_trajectory():
    """The audit's warm-start NashConv trajectory: best in the plateau, diverge
    after iter 300. The controller must pin the deployable window BEFORE 300."""
    traj = [
        (1, 1.134), (25, 0.182), (50, 0.102), (75, 0.082), (100, 0.075),
        (125, 0.064), (150, 0.054), (175, 0.047), (200, 0.042), (225, 0.039),
        (250, 0.041), (275, 0.040), (300, 0.041), (325, 0.077), (350, 0.117),
        (375, 0.136), (400, 0.158),
    ]
    c = BestSnapshotController(rel_tolerance=0.15, patience=2, min_iters=50, mode="min")
    stop_at = None
    for it, m in traj:
        d = c.update(it, m)
        if d.should_stop and stop_at is None:
            stop_at = it
    assert c.best_iteration <= 300
    assert c.best_metric < 0.05
    assert stop_at is not None and stop_at <= 400
    deployable = c.deployable_iters([it for it, _ in traj])
    assert max(deployable) <= 300


# ---------------------------------------------------------------------------
# Deployable manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    c = BestSnapshotController(mode="min")
    for it, m in [(1, 0.5), (2, 0.2), (3, 0.4)]:
        c.update(it, m)
    d = str(tmp_path)
    path = write_deployable_manifest(d, c, [1, 2, 3], metric_name="nashconv", stopped_early=True)
    assert os.path.basename(path) == DEPLOYABLE_MANIFEST
    man = read_deployable_manifest(d)
    assert man["best_iteration"] == 2
    assert man["deployable_iters"] == [1, 2]
    assert man["stopped_early"] is True
    assert man["metric_name"] == "nashconv"
    assert read_deployable_iters(d) == [1, 2]


def test_manifest_absent_returns_none(tmp_path):
    assert read_deployable_manifest(str(tmp_path)) is None
    assert read_deployable_iters(str(tmp_path)) is None


def test_eval_path_consumption_drops_diverged_tail(tmp_path):
    """The eval path (discover_snapshots) filtered by the manifest serves only the
    pre-divergence window; the diverged tail snapshots are dropped from the served
    SD-CFR average. This is the deployable-set realization without touching the
    scorer: read_deployable_iters is the seam."""
    from src.cfr.prtcfr_eval import discover_snapshots

    d = str(tmp_path)
    for it in range(1, 13):
        torch.save(
            {"encoder_state_dict": {}, "head_state_dict": {}, "iteration": it},
            os.path.join(d, f"prtcfr_snapshot_iter_{it}.pt"),
        )
    c = BestSnapshotController(rel_tolerance=0.15, patience=2, min_iters=1, mode="min")
    for it, m in [(2, 0.3), (4, 0.2), (6, 0.15), (8, 0.12), (10, 0.30), (12, 0.45)]:
        c.update(it, m)
    write_deployable_manifest(d, c, list(range(1, 13)), stopped_early=True)

    all_iters = [it for it, _ in discover_snapshots(d)]
    dep = set(read_deployable_iters(d))
    served = [it for it in all_iters if it in dep]
    assert max(served) == 8
    assert not (dep & {9, 10, 11, 12})


# ---------------------------------------------------------------------------
# Peak-LR schedule
# ---------------------------------------------------------------------------


def test_peak_lr_restart_is_constant():
    for t in (1, 5, 50, 400):
        assert _peak_lr_for_iter(1e-3, 0.0, t, 400, "restart") == 1e-3


def test_peak_lr_global_cosine_decays_to_floor():
    lr, lr_min, n = 1e-3, 1e-5, 400
    assert _peak_lr_for_iter(lr, lr_min, 1, n, "global_cosine") == pytest.approx(lr)
    assert _peak_lr_for_iter(lr, lr_min, n, n, "global_cosine") == pytest.approx(lr_min)
    mid = _peak_lr_for_iter(lr, lr_min, (n + 1) // 2, n, "global_cosine")
    assert lr_min < mid < lr
    # monotone non-increasing across the run
    prev = None
    for t in range(1, n + 1, 20):
        v = _peak_lr_for_iter(lr, lr_min, t, n, "global_cosine")
        if prev is not None:
            assert v <= prev + 1e-12
        prev = v


def test_peak_lr_unknown_schedule_raises():
    with pytest.raises(ValueError):
        _peak_lr_for_iter(1e-3, 0.0, 1, 10, "bogus")


# ---------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tree():
    cfg = load_config(CONFIG_2CARD)
    root, _isets, _n, aborted = build_tree(
        cfg, n_deals=5, seed0=0, max_nodes_per_deal=2_000_000,
        enumerate_draws=True, perfect_recall=True, tokenize=True, seq_cap=256,
    )
    assert aborted == 0
    return root


def _fast_ns(**extra):
    """Duck-typed config carrying the fast PRTCFRConfig defaults plus extras.

    PRTCFRConfig ignores unknown keys (extra='ignore'), so stability/lr_schedule
    knobs are attached to a SimpleNamespace the trainer reads via getattr -- the
    same pattern the production config plumbing (S1W5) will land on PRTCFRConfig.
    """
    base = PRTCFRConfig(
        m_rollouts=2, k_games_per_iter=15, iterations=6, train_steps_per_iter=20,
        batch_size=256, warm_start=True, device=_DEVICE,
    ).model_dump()
    base.update(extra)
    return types.SimpleNamespace(**base)


def test_default_writes_no_manifest(tiny_tree, tmp_path):
    cfg = _fast_ns()  # stability disabled by default
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, str(tmp_path / "snaps"))
    trainer.train(iterations=3)
    assert read_deployable_manifest(str(tmp_path / "snaps")) is None


def test_reanchor_reinits_net_on_schedule(tiny_tree, tmp_path):
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return PRTCFRNet(device=_DEVICE)

    cfg = _fast_ns(warm_start=True, reanchor_every=2)
    trainer = PRTCFRTinyTrainer(
        tiny_tree, cfg, str(tmp_path / "snaps"), net_factory=factory
    )
    trainer.train(iterations=4)
    # sigma^1 init (1) + re-anchor at t=2 and t=4 (2) = 3 factory calls.
    assert calls["n"] == 3


def test_global_cosine_lowers_late_peak_lr(tiny_tree, tmp_path):
    cfg = _fast_ns(lr=1e-3, lr_min=1e-5, lr_schedule="global_cosine", iterations=6)
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, str(tmp_path / "snaps"))
    assert trainer.lr_schedule == "global_cosine"
    early = _peak_lr_for_iter(trainer.lr, trainer.lr_min, 1, 6, "global_cosine")
    late = _peak_lr_for_iter(trainer.lr, trainer.lr_min, 6, 6, "global_cosine")
    assert late < early


def test_stability_controller_drives_and_writes_manifest(tiny_tree, tmp_path):
    """A synthetic eval_fn that reports a diverging metric must early-stop and
    pin the deployable window to the pre-divergence best."""
    # metric by iteration: improves to iter 2 then diverges.
    metric_by_iter = {1: 0.5, 2: 0.2, 3: 0.6, 4: 0.7, 5: 0.8, 6: 0.9}

    def eval_fn(_trainer, t):
        return metric_by_iter[t]

    cfg = _fast_ns(
        stability_enabled=True, stability_eval_every=1, stability_patience=2,
        stability_rel_tolerance=0.15, stability_min_iters=1,
    )
    snap_dir = str(tmp_path / "snaps")
    trainer = PRTCFRTinyTrainer(tiny_tree, cfg, snap_dir, eval_fn=eval_fn)
    hist = trainer.train(iterations=6)
    # best at iter 2; iters 3,4 worse -> stop at iter 4 (patience 2). Training
    # stops early, so fewer than 6 iterations run.
    assert len(hist) < 6
    man = read_deployable_manifest(snap_dir)
    assert man is not None
    assert man["best_iteration"] == 2
    assert man["stopped_early"] is True
    assert max(man["deployable_iters"]) == 2
