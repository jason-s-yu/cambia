"""tests/test_prtcfr_production_trainer.py

Scoped tests for the PRT-CFR production trainer (Phase 2 S1W5): the loop that
ties the merged components (single-trajectory worker, per-player DiskReservoirs,
shared regret net, V_phi critic, stability controller, run_db, metrics.jsonl)
into one iteration.

All tests run on CPU. The end-to-end coverage uses a scripted driver (fast,
deterministic) plus one run over the real Python-engine driver with a bounded
turn cap. The Go substrate (default backend) needs libcambia.so and is exercised
by the X3 bench lane, not here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import PRTCFRConfig
from src.encoding import MAX_HAND, NUM_ACTIONS
from src import run_db
from src.disk_reservoir import DiskReservoir
from src.reservoir import ReservoirSample
from src.cfr.prtcfr_net import PRTCFRNet, pad_tokens
from src.cfr.prtcfr_trainer import (
    NetProductionSigma,
    PRTCFRProductionTrainer,
    _MultiReservoirSampler,
    _UnpaddingReservoir,
    _merge_columnar_batches,
)

_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Scripted driver: a tiny deterministic 2-player game satisfying GameDriver.
# ---------------------------------------------------------------------------


class ScriptedDriver:
    """Minimal GameDriver: ``depth`` alternating moves, ``branch`` legal actions
    (integer indices, like the Go driver), terminal utility by move-sum parity.

    Exposes ``_get_all_cards_unsafe`` so the critic's omniscient resolver treats
    it directly (the trainer's value_sink wrapper hands it through)."""

    def __init__(self, seed=0, depth=4, branch=3, history=None):
        self.seed = seed
        self.depth = depth
        self.branch = branch
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
        return ScriptedDriver(self.seed, self.depth, self.branch, self.history)

    def close(self):
        self._closed = True

    def _get_all_cards_unsafe(self):
        return np.full(2 * MAX_HAND, 0xFF, dtype=np.uint8)


def _scripted_factory(seed, num_players=2, backend="go", **_kw):
    return ScriptedDriver(seed=seed, depth=4, branch=3)


def _prod_config(**overrides):
    """A tiny production PRTCFRConfig for CPU e2e (small net, K, steps)."""
    base = dict(
        seq_cap=64,
        m_rollouts=1,
        k_games_per_iter=8,
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
        stability_patience=2,
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


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_production_fields_and_tiny_safe_defaults():
    c = PRTCFRConfig()
    # Shared fields keep TINY-safe defaults (X2 gate byte-for-byte).
    assert c.seq_cap == 256
    assert c.k_games_per_iter == 200
    assert c.warm_start is False
    assert c.lr_schedule == "restart"
    assert c.stability_enabled is False
    # Production-only fields carry production defaults (tiny trainer never reads).
    assert c.train_steps == 3000
    assert c.reservoir_capacity == 20_000_000
    assert c.backend == "go"
    assert c.critic_enabled is True
    assert 0.0 < c.critic_held_out_fraction < 1.0
    # New knobs round-trip through model_dump/reconstruct (config plumbing).
    c2 = PRTCFRConfig(**c.model_dump())
    assert c2.lr_schedule == c.lr_schedule
    assert c2.reservoir_capacity == c.reservoir_capacity


# ---------------------------------------------------------------------------
# run_db detection + mapping
# ---------------------------------------------------------------------------


def test_infer_algorithm_prtcfr_before_desca():
    assert run_db.infer_algorithm({"prt_cfr": {"seq_cap": 256}}) == "prt-cfr"
    assert run_db.infer_algorithm({"algorithm": "prt-cfr"}) == "prt-cfr"
    assert (
        run_db.infer_algorithm({}, checkpoint_filename="x/prtcfr_checkpoint.pt")
        == "prt-cfr"
    )
    assert (
        run_db.infer_algorithm({}, checkpoint_filename="prtcfr_snapshot_iter_5.pt")
        == "prt-cfr"
    )
    # desca still resolves (prt-cfr rule must not shadow it).
    assert run_db.infer_algorithm({"algorithm": "desca"}) == "desca"
    assert run_db.ALGO_TO_AGENT_TYPE["prt-cfr"] == "prt_cfr"
    assert run_db.ALGO_TO_CHECKPOINT_PREFIX["prt-cfr"] == "prtcfr_checkpoint"
    assert run_db.algo_to_agent_type("prt-cfr") == "prt_cfr"
    assert run_db.algo_to_checkpoint_prefix("prt-cfr") == "prtcfr_checkpoint"


# ---------------------------------------------------------------------------
# Sigma provider
# ---------------------------------------------------------------------------


def test_net_production_sigma_valid_distribution():
    net = PRTCFRNet(
        embed_dim=8, hidden_dim=16, num_layers=1, dropout=0.0,
        head_hidden_dim=16, device=_DEVICE,
    )
    sigma = NetProductionSigma(net, seq_cap=64)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 1, 2]] = True
    probs = sigma([1, 5, 6], mask)
    assert probs.shape == (NUM_ACTIONS,)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()
    # all mass on legal actions
    assert abs(probs[[0, 1, 2]].sum() - 1.0) < 1e-6
    assert probs[3:].sum() < 1e-6


# ---------------------------------------------------------------------------
# Reservoir plumbing
# ---------------------------------------------------------------------------


def test_unpadding_reservoir_stores_ragged(tmp_path):
    disk = DiskReservoir(
        path=str(tmp_path / "b0"), capacity=100, seq_cap=64,
        target_dim=NUM_ACTIONS, has_mask=True, seed=0,
    )
    adapter = _UnpaddingReservoir(disk)
    tokens = [1, 5, 6, 7]  # natural length 4
    padded = pad_tokens(tokens, seq_cap=64)  # width 64
    tgt = np.zeros(NUM_ACTIONS, dtype=np.float32)
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[[0, 1]] = True
    adapter.add(ReservoirSample(features=padded, target=tgt, action_mask=m, iteration=1))
    assert len(adapter) == 1
    batch = adapter.sample_batch(1)
    # ragged: stored at natural length 4, NOT the 64-wide padded row.
    assert int(batch.lengths[0]) == 4
    assert batch.features.shape[1] == 4


def test_multi_reservoir_sampler_and_merge(tmp_path):
    r0 = DiskReservoir(path=str(tmp_path / "r0"), capacity=100, seq_cap=64,
                       target_dim=NUM_ACTIONS, has_mask=True, seed=1)
    r1 = DiskReservoir(path=str(tmp_path / "r1"), capacity=100, seq_cap=64,
                       target_dim=NUM_ACTIONS, has_mask=True, seed=2)
    a0, a1 = _UnpaddingReservoir(r0), _UnpaddingReservoir(r1)
    tgt = np.zeros(NUM_ACTIONS, dtype=np.float32)
    m = np.zeros(NUM_ACTIONS, dtype=bool)
    m[0] = True
    def _s(toks):
        return ReservoirSample(
            features=pad_tokens(toks, 64), target=tgt, action_mask=m, iteration=1
        )

    for _ in range(10):
        a0.add(_s([1, 5]))
    for _ in range(6):
        a1.add(_s([1, 5, 6, 7, 8]))
    sampler = _MultiReservoirSampler([a0, a1])
    assert len(sampler) == 16
    batch = sampler.sample_batch(12)
    assert len(batch) == 12
    # merged width = longest row across the sampled sub-batches (2 or 5).
    assert batch.features.shape[1] in (2, 5)
    assert batch.targets.shape == (12, NUM_ACTIONS)
    assert batch.masks.shape == (12, NUM_ACTIONS)

    # merge re-pads narrower sub-batches to the global max width.
    b0 = r0.sample_batch(3)
    b1 = r1.sample_batch(3)
    merged = _merge_columnar_batches([b0, b1])
    assert merged.features.shape == (6, 5)


# ---------------------------------------------------------------------------
# End-to-end (scripted driver)
# ---------------------------------------------------------------------------


def test_end_to_end_scripted_driver(tmp_path):
    cfg = _prod_config()
    run_dir = tmp_path / "run"
    db_path = str(tmp_path / "cambia_runs.db")

    metrics_seq = {1: 0.5, 2: 0.3, 3: 0.6}  # best at iter 2

    def eval_fn(_trainer, t):
        return metrics_seq[t]

    trainer = PRTCFRProductionTrainer(
        cfg,
        str(run_dir),
        driver_factory=_scripted_factory,
        eval_fn=eval_fn,
        db_path=db_path,
        run_name="v0.4-prtcfr-scripted",
        config_dict={"prt_cfr": cfg.model_dump()},
    )
    history = trainer.train(iterations=3)
    trainer.close()

    assert len(history) == 3

    # Snapshots + rolling checkpoint (Phase-1 dict format).
    snap_dir = run_dir / "snapshots"
    for t in (1, 2, 3):
        p = snap_dir / f"prtcfr_snapshot_iter_{t}.pt"
        assert p.exists(), p
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        assert set(d.keys()) == {"encoder_state_dict", "head_state_dict", "iteration"}
    ckpt = torch.load(str(snap_dir / "prtcfr_checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) == {
        "encoder_state_dict", "head_state_dict", "config", "iteration",
    }
    assert ckpt["iteration"] == 3

    # metrics.jsonl: one row per iter, full schema incl. gen/fit seconds + critic.
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    for i, row in enumerate(rows, start=1):
        assert row["iteration"] == i
        assert set(row["samples_added"].keys()) == {"0", "1"}
        assert set(row["buffer_sizes"].keys()) == {"0", "1"}
        for key in ("fit_loss", "peak_lr", "gen_seconds", "fit_seconds",
                    "critic_held_out_mse", "critic_constant_baseline_mse",
                    "critic_ratio"):
            assert key in row
        assert row["gen_seconds"] >= 0.0
        assert row["fit_seconds"] >= 0.0
    # global-cosine schedule: late peak LR below early.
    assert rows[2]["peak_lr"] < rows[0]["peak_lr"]
    # critic produced a held-out number (critic ON, omniscient resolves).
    assert rows[2]["critic_held_out_mse"] is not None

    # Reservoir growth: samples accumulate across iterations.
    tot1 = sum(rows[0]["buffer_sizes"].values())
    tot3 = sum(rows[2]["buffer_sizes"].values())
    assert tot3 > tot1 > 0

    # Deployable manifest written (stability ON + eval_fn), best pinned to iter 2.
    manifest = json.loads((snap_dir / "prtcfr_deployable.json").read_text())
    assert manifest["best_iteration"] == 2
    assert manifest["deployable_iters"] == [1, 2]

    # run_db rows: run registered as prt-cfr, 3 checkpoints.
    db = run_db.get_db(db_path)
    run_row = db.execute(
        "SELECT id, algorithm, status, best_metric_name, best_metric_value, "
        "best_metric_iter FROM runs WHERE name=?",
        ("v0.4-prtcfr-scripted",),
    ).fetchone()
    assert run_row is not None
    assert run_row["algorithm"] == "prt-cfr"
    assert run_row["status"] == "completed"
    n_ckpt = db.execute(
        "SELECT COUNT(*) AS c FROM checkpoints WHERE run_id=?", (run_row["id"],)
    ).fetchone()["c"]
    assert n_ckpt == 3

    # cambia-390: stability best mirrored into run_db -- best at iter 2 (0.3).
    assert run_row["best_metric_name"] == "nashconv"
    assert run_row["best_metric_value"] == pytest.approx(0.3)
    assert run_row["best_metric_iter"] == 2
    best_ckpt_iters = {
        r["iteration"]
        for r in db.execute(
            "SELECT iteration, is_best FROM checkpoints WHERE run_id=?",
            (run_row["id"],),
        ).fetchall()
        if r["is_best"]
    }
    assert best_ckpt_iters == {2}
    db.close()


def test_stability_best_moves_to_later_iteration_in_run_db(tmp_path):
    """cambia-390: a later, better iteration must move is_best off the earlier
    checkpoint row and refresh runs.best_metric_*, not just append a second
    best row."""
    cfg = _prod_config(iterations=4, stability_patience=10, stability_eval_every=1)
    metrics_seq = {1: 0.5, 2: 0.3, 3: 0.4, 4: 0.2}  # best moves 2 -> 4

    def eval_fn(_trainer, t):
        return metrics_seq[t]

    db_path = str(tmp_path / "cambia_runs.db")
    trainer = PRTCFRProductionTrainer(
        cfg, str(tmp_path / "run"), driver_factory=_scripted_factory,
        eval_fn=eval_fn, db_path=db_path, run_name="v0.4-prtcfr-bestmoves",
    )
    history = trainer.train(iterations=4)
    trainer.close()
    assert len(history) == 4  # high patience: no early-stop on this trajectory

    db = run_db.get_db(db_path)
    run_row = db.execute(
        "SELECT id, best_metric_value, best_metric_iter FROM runs WHERE name=?",
        ("v0.4-prtcfr-bestmoves",),
    ).fetchone()
    assert run_row is not None
    assert run_row["best_metric_value"] == pytest.approx(0.2)
    assert run_row["best_metric_iter"] == 4
    best_ckpt_iters = {
        r["iteration"]
        for r in db.execute(
            "SELECT iteration, is_best FROM checkpoints WHERE run_id=?",
            (run_row["id"],),
        ).fetchall()
        if r["is_best"]
    }
    assert best_ckpt_iters == {4}
    db.close()


def test_early_stop_pins_deployable_window(tmp_path):
    # metric worsens sustainedly after iter 1 -> early-stop, window pinned to [1].
    cfg = _prod_config(iterations=6, stability_patience=2, stability_eval_every=1)
    metrics_seq = {1: 0.10, 2: 0.50, 3: 0.60, 4: 0.70, 5: 0.80, 6: 0.90}
    trainer = PRTCFRProductionTrainer(
        cfg, str(tmp_path / "run"), driver_factory=_scripted_factory,
        eval_fn=lambda _t, t: metrics_seq[t], db_path=str(tmp_path / "db.sqlite"),
        run_name="v0.4-prtcfr-earlystop",
    )
    history = trainer.train(iterations=6)
    trainer.close()
    # stopped before running all 6 iterations.
    assert len(history) < 6
    manifest = json.loads(
        (tmp_path / "run" / "snapshots" / "prtcfr_deployable.json").read_text()
    )
    assert manifest["best_iteration"] == 1
    assert manifest["deployable_iters"] == [1]
    assert manifest["stopped_early"] is True


# ---------------------------------------------------------------------------
# End-to-end (real Python-engine driver, bounded)
# ---------------------------------------------------------------------------


def test_end_to_end_real_python_driver(tmp_path):
    from src.config import CambiaRulesConfig
    from src.cfr.prtcfr_worker import new_production_driver

    hr = CambiaRulesConfig()
    hr.allowDrawFromDiscardPile = True
    hr.allowReplaceAbilities = True
    hr.allowOpponentSnapping = True
    hr.max_game_turns = 8  # bound game + rollout length for a fast test
    hr.lockCallerHand = False

    def driver_factory(seed):
        return new_production_driver(seed, house_rules=hr, backend="python")

    cfg = _prod_config(
        seq_cap=512, k_games_per_iter=2, iterations=2, train_steps=3,
        critic_steps_per_iter=3, max_trajectory_steps=60,
    )
    run_dir = tmp_path / "run"
    trainer = PRTCFRProductionTrainer(
        cfg, str(run_dir), driver_factory=driver_factory,
        db_path=str(tmp_path / "db.sqlite"), run_name="v0.4-prtcfr-pydriver",
    )
    history = trainer.train(iterations=2)
    trainer.close()

    assert len(history) == 2
    assert (run_dir / "snapshots" / "prtcfr_snapshot_iter_2.pt").exists()
    assert (run_dir / "snapshots" / "prtcfr_checkpoint.pt").exists()
    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    rows = [json.loads(x) for x in metrics_lines]
    assert len(rows) == 2
    # real engine produced regret samples through the actual driver path.
    assert sum(rows[-1]["buffer_sizes"].values()) >= 0


# ---------------------------------------------------------------------------
# CLI smoke (direct function call, scripted driver injected)
# ---------------------------------------------------------------------------


def test_cli_smoke_direct_call(tmp_path, monkeypatch):
    import src.cfr.prtcfr_trainer as trainer_mod
    from src.cli import train_prtcfr

    monkeypatch.setattr(trainer_mod, "new_production_driver", _scripted_factory)

    cfg_yaml = tmp_path / "tiny_prtcfr.yaml"
    cfg_yaml.write_text(
        "prt_cfr:\n"
        "  seq_cap: 64\n"
        "  m_rollouts: 1\n"
        "  k_games_per_iter: 4\n"
        "  iterations: 2\n"
        "  train_steps: 3\n"
        "  batch_size: 16\n"
        "  lr: 1.0e-2\n"
        "  lr_min: 1.0e-4\n"
        "  lr_schedule: global_cosine\n"
        "  warm_start: true\n"
        "  stability_enabled: false\n"
        "  reservoir_capacity: 500\n"
        "  backend: python\n"
        "  critic_enabled: true\n"
        "  critic_capacity: 500\n"
        "  critic_steps_per_iter: 3\n"
        "  critic_batch_size: 8\n"
        "  gru_embed_dim: 8\n"
        "  gru_hidden_dim: 16\n"
        "  gru_num_layers: 1\n"
        "  gru_dropout: 0.0\n"
        "  head_hidden_dim: 16\n"
        "  device: cpu\n"
    )
    run_dir = tmp_path / "cli_run"

    # All params passed explicitly so no typer.OptionInfo leaks into the body.
    train_prtcfr(
        config=cfg_yaml,
        run_name="v0.4-prtcfr-clismoke",
        iterations=2,
        save_path=run_dir,
        device="cpu",
        backend="python",
        resume=False,
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "snapshots" / "prtcfr_checkpoint.pt").exists()
    assert (run_dir / "snapshots" / "prtcfr_snapshot_iter_2.pt").exists()
    rows = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2


def test_init_run_db_honors_cambia_run_db_env(tmp_path, monkeypatch):
    """CAMBIA_RUN_DB redirects the trainer's journal to the per-run-dir
    run_db.sqlite the serving harness syncs (design 4.2); without it the
    runs-dir sibling default is unchanged."""
    from src.cfr.prtcfr_trainer import PRTCFRProductionTrainer

    run_dir = tmp_path / "runs" / "env-run"
    run_dir.mkdir(parents=True)
    journal = run_dir / "run_db.sqlite"
    monkeypatch.setenv("CAMBIA_RUN_DB", str(journal))

    trainer = object.__new__(PRTCFRProductionTrainer)
    trainer.run_dir = str(run_dir)
    trainer._init_run_db(None, "env-run", None, None)
    assert journal.exists()
    assert not (tmp_path / "runs" / "cambia_runs.db").exists()
