"""tests/test_prtcfr_production_battery.py

Scoped tests for the PRT-CFR production in-loop X4 battery (Phase 2 S2W1):

  - src/cfr/prtcfr_battery.py::build_production_battery_eval_fn -- the Tier-A LBR
    fast-lane eval_fn (SD-CFR mixture over snapshots [1..t]).
  - src/cfr/prtcfr_trainer.py -- the three additive AC2 metrics-row fields
    (t1_cambia_rate / tier_a_lbr / grad_norm_violations), the generation-loop
    turn-1 Cambia tap, and the per-fit grad-norm violation counter.

All tests run on CPU with tiny K / train_steps. The trainer's generation uses a
scripted driver; the real Tier-A LBR lane (last test) plays real Cambia games
via the loaded snapshot mixture, sized tiny (2 infosets, 1 rollout, 8-turn cap).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import PRTCFRConfig
from src.encoding import MAX_HAND
from src.cfr.prtcfr_battery import build_production_battery_eval_fn
from src.cfr.prtcfr_trainer import (
    PRTCFRProductionTrainer,
    PRTCFRProductionTrainState,
    _CALL_CAMBIA_INDEX,
    _GenTurn1Observer,
)

_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Scripted drivers (deterministic, satisfy the GameDriver protocol).
# ---------------------------------------------------------------------------


class ScriptedDriver:
    """Tiny deterministic 2-player game (mirrors the production-trainer test)."""

    def __init__(self, seed=0, depth=4, branch=3, history=None, first_legal=None):
        self.seed = seed
        self.depth = depth
        self.branch = branch
        self.first_legal = first_legal  # override legal set on the opening move
        self.history = list(history) if history else []
        self._closed = False

    def current_player(self):
        return len(self.history) % 2

    def is_terminal(self):
        return len(self.history) >= self.depth

    def utility(self, player):
        even = (sum(self.history) % 2) == 0
        if even:
            return 1.0 if player == 0 else -1.0
        return -1.0 if player == 0 else 1.0

    def legal_actions(self):
        if not self.history and self.first_legal is not None:
            return list(self.first_legal)
        return list(range(self.branch))

    def apply(self, action):
        self.history.append(int(action))
        return True

    def tokens(self, player):
        toks = [1]  # BOS
        for a in self.history:
            toks.append(4 + (a % 4))
        toks.append(6 + player)
        return toks

    def clone(self):
        return ScriptedDriver(
            self.seed, self.depth, self.branch, self.history, self.first_legal
        )

    def close(self):
        self._closed = True

    def _get_all_cards_unsafe(self):
        return np.full(2 * MAX_HAND, 0xFF, dtype=np.uint8)


def _scripted_factory(seed, num_players=2, backend="go", **_kw):
    return ScriptedDriver(seed=seed, depth=4, branch=3)


def _first_legal_factory(first_legal):
    def factory(seed, num_players=2, backend="go", **_kw):
        # depth 6 so a first forced move still leaves a few normal decisions.
        return ScriptedDriver(seed=seed, depth=6, branch=2, first_legal=first_legal)

    return factory


def _prod_config(**overrides):
    base = dict(
        seq_cap=64,
        m_rollouts=1,
        k_games_per_iter=6,
        iterations=3,
        lr=1.0e-2,
        lr_min=1.0e-4,
        lr_schedule="global_cosine",
        batch_size=32,
        train_steps=5,
        warm_start=True,
        stability_enabled=True,
        stability_eval_every=1,
        stability_min_iters=1,
        stability_patience=5,
        reservoir_capacity=1000,
        backend="python",
        critic_enabled=True,
        critic_capacity=1000,
        critic_steps_per_iter=5,
        critic_batch_size=16,
        num_players=2,
        max_trajectory_steps=50,
        gru_embed_dim=8,
        gru_hidden_dim=16,
        gru_num_layers=1,
        gru_dropout=0.0,
        head_hidden_dim=16,
        device=_DEVICE,
        seed=0,
    )
    base.update(overrides)
    return PRTCFRConfig(**base)


def _read_rows(run_dir):
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    return [json.loads(x) for x in lines]


# ---------------------------------------------------------------------------
# Config surface + import smoke
# ---------------------------------------------------------------------------


def _real_config_module():
    """The real src.config (worktree file), bypassing conftest's stub module.

    Mirrors conftest's own _load_config pop/import pattern so the assertions
    exercise the actual PRTCFRConfig fields, not the hand-maintained stub."""
    import importlib
    import sys

    saved = sys.modules.pop("src.config", None)
    try:
        return importlib.import_module("src.config")
    finally:
        if saved is not None:
            sys.modules["src.config"] = saved


def test_config_has_small_battery_defaults():
    real = _real_config_module()
    fields = real.PRTCFRConfig.model_fields
    assert "battery_lbr_games" in fields
    assert "battery_lbr_depth" in fields
    c = real.PRTCFRConfig()
    assert isinstance(c.battery_lbr_games, int) and c.battery_lbr_games > 0
    assert isinstance(c.battery_lbr_depth, int) and c.battery_lbr_depth > 0
    # Small fast-lane sizes (a fraction of a K=8192 iteration).
    assert c.battery_lbr_games <= 512
    assert c.battery_lbr_depth <= 64
    # Default stability metric name is unchanged (production run sets tier_a_lbr).
    assert c.stability_metric_name == "nashconv"


def test_call_cambia_index_is_two():
    assert _CALL_CAMBIA_INDEX == 2


# ---------------------------------------------------------------------------
# Metrics-row schema: all three additive fields at the stability cadence.
# ---------------------------------------------------------------------------


def test_battery_rows_carry_all_three_fields(tmp_path):
    cfg = _prod_config()
    run_dir = tmp_path / "run"
    metrics_seq = {1: 0.50, 2: 0.30, 3: 0.20}

    def stub_eval_fn(trainer, t):
        # Mimic the battery's stash side-effect.
        trainer.tier_a_lbr = metrics_seq[t]
        return metrics_seq[t]

    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=stub_eval_fn, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-schema",
    )
    history = trainer.train(iterations=3)
    trainer.close()

    assert len(history) == 3
    rows = _read_rows(run_dir)
    assert len(rows) == 3
    for i, row in enumerate(rows, start=1):
        # Additive fields present in EVERY row.
        assert "t1_cambia_rate" in row
        assert "tier_a_lbr" in row
        assert "grad_norm_violations" in row
        # Pre-existing fields untouched (spot-check a few).
        for key in ("fit_loss", "peak_lr", "critic_held_out_mse", "gen_seconds"):
            assert key in row
        # At the (every-iter) cadence the LBR trend equals the eval_fn's return.
        assert row["tier_a_lbr"] == pytest.approx(metrics_seq[i])
        assert isinstance(row["grad_norm_violations"], int)

    # The dataclass carries the fields too (read by name, additive).
    st = history[-1]
    assert isinstance(st, PRTCFRProductionTrainState)
    assert st.tier_a_lbr == pytest.approx(metrics_seq[3])
    assert st.grad_norm_violations >= 0


def test_tier_a_lbr_null_off_cadence(tmp_path):
    # stability_eval_every=3 over 3 iters -> cadence fires at t=1 and t=3 (t==n),
    # NOT t=2, so t=2's row carries a null tier_a_lbr (present-but-unscored).
    cfg = _prod_config(iterations=3, stability_eval_every=3, stability_min_iters=1)
    run_dir = tmp_path / "run"

    def stub_eval_fn(trainer, t):
        trainer.tier_a_lbr = 0.1 * t
        return 0.1 * t

    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=stub_eval_fn, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-cadence",
    )
    trainer.train(iterations=3)
    trainer.close()

    rows = _read_rows(run_dir)
    assert rows[0]["tier_a_lbr"] is not None  # t=1 (t==1 always due)
    assert rows[1]["tier_a_lbr"] is None      # t=2 off cadence
    assert rows[2]["tier_a_lbr"] is not None  # t=3 (t==n due)
    # t1_cambia_rate / grad_norm_violations are per-iteration (every row).
    for row in rows:
        assert "t1_cambia_rate" in row
        assert isinstance(row["grad_norm_violations"], int)


# ---------------------------------------------------------------------------
# Grad-norm violation counter (pre-clip norm > grad_clip).
# ---------------------------------------------------------------------------


def test_grad_clip_below_norm_increments_violations(tmp_path):
    # grad_clip far below any real grad norm -> clipping fires every fit step.
    cfg = _prod_config(grad_clip=1.0e-9, iterations=2, train_steps=4)
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=lambda tr, t: 0.0, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-gradviol",
    )
    trainer.train(iterations=2)
    trainer.close()

    rows = _read_rows(run_dir)
    # Every fit step violated -> a positive count each iteration.
    assert all(row["grad_norm_violations"] > 0 for row in rows)


def test_grad_clip_above_norm_zero_violations(tmp_path):
    # grad_clip far above any real grad norm -> clipping never fires.
    cfg = _prod_config(grad_clip=1.0e9, iterations=2, train_steps=4)
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=lambda tr, t: 0.0, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-noviol",
    )
    trainer.train(iterations=2)
    trainer.close()

    rows = _read_rows(run_dir)
    assert all(row["grad_norm_violations"] == 0 for row in rows)


# ---------------------------------------------------------------------------
# Turn-1 Cambia tap (counted from the iteration's own K generation games).
# ---------------------------------------------------------------------------


def test_observer_clone_is_unwrapped():
    # Rollout / child clones must NOT be observed: clone() returns the inner
    # driver, so its applies never touch the shared counter.
    stats = {"games": 0, "t1_cambia": 0}
    inner = ScriptedDriver(depth=4, branch=2, first_legal=[2])
    obs = _GenTurn1Observer(inner, stats)
    assert stats["games"] == 1
    child = obs.clone()
    assert isinstance(child, ScriptedDriver)   # unwrapped
    child.apply(2)                             # a rollout apply
    assert stats["t1_cambia"] == 0             # not counted
    obs.apply(2)                               # the real turn-1 move
    assert stats["t1_cambia"] == 1
    # Delegation: non-overridden attributes reach the inner driver.
    assert obs.current_player() == inner.current_player()


@pytest.mark.parametrize("gen_batched", [False, True])
def test_t1_cambia_rate_all_games_open_cambia(tmp_path, gen_batched):
    # Opening legal set == [Cambia] -> every game's first move is a Cambia call.
    cfg = _prod_config(
        iterations=2, k_games_per_iter=4, critic_enabled=False,
        gen_batched=gen_batched,
    )
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir),
        driver_factory=_first_legal_factory([_CALL_CAMBIA_INDEX]),
        eval_fn=lambda tr, t: 0.0, db_path=str(tmp_path / "db.sqlite"),
        run_name=f"v0.4-prtcfr-battery-t1hi-{int(gen_batched)}",
    )
    trainer.train(iterations=2)
    trainer.close()

    rows = _read_rows(run_dir)
    for row in rows:
        assert row["t1_cambia_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize("gen_batched", [False, True])
def test_t1_cambia_rate_no_game_opens_cambia(tmp_path, gen_batched):
    # Opening legal set excludes Cambia -> rate 0.0.
    cfg = _prod_config(
        iterations=2, k_games_per_iter=4, critic_enabled=False,
        gen_batched=gen_batched,
    )
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir),
        driver_factory=_first_legal_factory([0]),
        eval_fn=lambda tr, t: 0.0, db_path=str(tmp_path / "db.sqlite"),
        run_name=f"v0.4-prtcfr-battery-t1lo-{int(gen_batched)}",
    )
    trainer.train(iterations=2)
    trainer.close()

    rows = _read_rows(run_dir)
    for row in rows:
        assert row["t1_cambia_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Real Tier-A LBR fast lane wired in-loop (build_production_battery_eval_fn).
# ---------------------------------------------------------------------------


def _fast_eval_config(tmp_path, max_turns=8):
    from src.config import load_config

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(project_root, "config", "prtcfr_production.yaml")
    cfgpath = tmp_path / "eval_config.yaml"
    cfgpath.write_text(
        f"_base: {base}\ncambia_rules:\n  max_game_turns: {max_turns}\n"
    )
    return load_config(str(cfgpath))


def test_build_production_battery_eval_fn_runs_real_lbr(tmp_path):
    # Train a tiny production run (scripted driver -> real net snapshots), wiring
    # the REAL Tier-A LBR battery as the trainer's stability eval_fn. The battery
    # loads the [1..t] snapshot mixture and plays real Cambia LBR games.
    eval_config = _fast_eval_config(tmp_path, max_turns=8)
    battery = build_production_battery_eval_fn(
        eval_config, device=_DEVICE, lbr_games=2, lbr_depth=1,
    )

    cfg = _prod_config(
        iterations=2, k_games_per_iter=4, stability_eval_every=2,
        stability_metric_name="tier_a_lbr", critic_enabled=False,
    )
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=battery, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-reallbr",
    )
    history = trainer.train(iterations=2)
    trainer.close()

    assert len(history) == 2
    # The battery stashed a finite Tier-A LBR on the trainer (last cadence).
    assert np.isfinite(trainer.tier_a_lbr)

    rows = _read_rows(run_dir)
    # Cadence fires at t=1 (t==1) and t=2 (t==n): both rows carry a real
    # (non-null, finite, >= 0) tier_a_lbr from the [1..t] mixture LBR.
    for row in rows:
        val = row["tier_a_lbr"]
        assert val is not None
        assert np.isfinite(val)
        assert val >= 0.0
    # Snapshots [1, 2] were written; the t=2 mixture spanned both.
    snap_dir = run_dir / "snapshots"
    assert (snap_dir / "prtcfr_snapshot_iter_1.pt").exists()
    assert (snap_dir / "prtcfr_snapshot_iter_2.pt").exists()


def test_upto_mixture_spans_one_to_t(tmp_path):
    # The [1..t] window is built from the snapshots on disk, NOT the deployable
    # manifest (which would pin [1..best] after the first cadence).
    from src.cfr.prtcfr_battery import _build_upto_mixture
    from src.cfr.prtcfr_worker import PRODUCTION_SEQ_CAP

    eval_config = _fast_eval_config(tmp_path, max_turns=8)
    cfg = _prod_config(iterations=3, k_games_per_iter=3, critic_enabled=False)
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=_scripted_factory,
        eval_fn=lambda tr, t: 0.0, db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-battery-window",
    )
    trainer.train(iterations=3)
    trainer.close()

    for t in (1, 2, 3):
        mix = _build_upto_mixture(trainer, t, _DEVICE, "linear", PRODUCTION_SEQ_CAP)
        assert sorted(mix.iters) == list(range(1, t + 1))
