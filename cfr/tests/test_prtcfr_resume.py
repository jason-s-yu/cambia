"""tests/test_prtcfr_resume.py

Scoped tests for PRT-CFR production-trainer resume-from-disk (Phase 2 S1T3).

Covers: resume_state.json is written each iteration with the pinned schema; the
BestSnapshotController (de)serialization helpers round-trip (incl. the inf
sentinel); ``train(resume=True)`` reloads the rolling checkpoint + per-player
reservoirs + RNG + controller and continues at t+1; a resume with no checkpoint
raises cleanly and leaves the run dir intact; the reservoir/fit/torch RNG are
restored so a resumed step draws exactly what a non-interrupted run would; a
full uninterrupted run and an interrupt-then-resume run produce bit-identical
fits on the resumed iterations; and resume reuses the run_db row by name (no
duplicate).

CPU only, tiny configs, a deterministic scripted driver. No GPU, no full suite.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.config import PRTCFRConfig
from src.encoding import MAX_HAND
from src import run_db
from src.cfr.prtcfr_stability import BestSnapshotController
from src.cfr import prtcfr_trainer as trainer_mod
from src.cfr.prtcfr_trainer import (
    RESUME_SCHEMA_VERSION,
    PRTCFRProductionTrainer,
    PRTCFRResumeError,
    _controller_from_dict,
    _controller_to_dict,
)

# Measurement-trap guard: confirm the trainer under test is the worktree copy
# (same repo root as this test file), not the main-repo editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert os.path.abspath(trainer_mod.__file__).startswith(_REPO_ROOT), (
    f"prtcfr_trainer imported from {trainer_mod.__file__}, expected under "
    f"{_REPO_ROOT}"
)

_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Scripted driver (mirrors the production-trainer test): a tiny deterministic
# 2-player game satisfying GameDriver.
# ---------------------------------------------------------------------------


class ScriptedDriver:
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
    """A tiny CPU production config; resume-test-friendly defaults.

    critic OFF and the sequential (fp32) gen path by default so the strict
    reproducibility test isolates the regret loop; individual tests re-enable
    the production paths (batched gen, critic) where that is what they cover.
    """
    base = dict(
        seq_cap=64,
        m_rollouts=1,
        k_games_per_iter=8,
        iterations=4,
        lr=1.0e-2,
        lr_min=1.0e-4,
        lr_schedule="global_cosine",
        batch_size=32,
        train_steps=5,
        warm_start=True,
        stability_enabled=False,
        reservoir_capacity=1000,
        backend="python",
        critic_enabled=False,
        num_players=2,
        max_trajectory_steps=50,
        gen_batched=False,
        infer_dtype="fp32",
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


def _make_trainer(cfg, run_dir, db_path, run_name, **kw):
    return PRTCFRProductionTrainer(
        cfg,
        str(run_dir),
        driver_factory=_scripted_factory,
        db_path=str(db_path),
        run_name=run_name,
        **kw,
    )


# ---------------------------------------------------------------------------
# Controller (de)serialization helpers (unit)
# ---------------------------------------------------------------------------


def test_controller_dict_roundtrip_fresh_and_updated():
    # Fresh controller: best_metric is +inf (mode="min") -> serialized as None.
    c = BestSnapshotController(rel_tolerance=0.2, patience=3, min_iters=2, mode="min")
    d = _controller_to_dict(c)
    assert d["best_metric"] is None
    back = _controller_from_dict(d)
    assert back.best_metric == math.inf
    assert back.mode == "min" and back.patience == 3 and back.min_iters == 2

    # After some updates the state round-trips exactly.
    c.update(1, 0.5)
    c.update(2, 0.4)  # new best
    c.update(3, 0.9)  # worse
    d2 = _controller_to_dict(c)
    back2 = _controller_from_dict(d2)
    assert back2.best_iteration == c.best_iteration == 2
    assert back2.best_metric == c.best_metric == pytest.approx(0.4)
    assert back2.num_worse_since_best == c.num_worse_since_best
    assert back2.stopped == c.stopped
    assert back2.history == c.history

    # A subsequent update on the restored controller matches the original's.
    assert (
        back2.update(4, 0.95).num_worse_since_best
        == c.update(4, 0.95).num_worse_since_best
    )

    # mode="max" fresh sentinel is -inf.
    cmax = BestSnapshotController(mode="max")
    assert _controller_from_dict(_controller_to_dict(cmax)).best_metric == -math.inf


# ---------------------------------------------------------------------------
# resume_state.json is written each iteration with the pinned schema
# ---------------------------------------------------------------------------


def test_resume_state_written_each_iteration(tmp_path):
    cfg = _prod_config(iterations=2, stability_enabled=True, stability_eval_every=1,
                       stability_min_iters=1)
    run_dir = tmp_path / "run"
    tr = _make_trainer(cfg, run_dir, tmp_path / "db.sqlite", "v0.4-resume-schema")
    tr.train(iterations=2)
    tr.close()

    rs_path = run_dir / "resume_state.json"
    assert rs_path.exists()
    state = json.loads(rs_path.read_text())
    assert state["schema"] == RESUME_SCHEMA_VERSION
    assert state["iteration"] == 2
    assert state["snapshots"] == [1, 2]
    # numpy_rng is the _fit_rng Generator bit-generator state (a JSON dict).
    assert isinstance(state["numpy_rng"], dict)
    assert "bit_generator" in state["numpy_rng"]
    # torch_rng is a hex string that decodes to a non-empty byte buffer.
    assert isinstance(state["torch_rng"], str)
    assert len(bytes.fromhex(state["torch_rng"])) > 0
    # controller present (stability ON) with public fields.
    assert state["controller"] is not None
    assert "best_iteration" in state["controller"]


# ---------------------------------------------------------------------------
# (a) resume continues at t+1 and the reservoir length is preserved
#     -- exercised over the PRODUCTION paths (batched gen + critic ON).
# ---------------------------------------------------------------------------


def test_resume_two_iter_then_resume(tmp_path):
    """Acceptance (a): a 2-iteration run, then resume, starts at iter 3 and the
    reservoir length carried across the interruption (batched gen + critic ON,
    i.e. the production paths)."""
    cfg = _prod_config(
        iterations=4, gen_batched=True, infer_dtype="bf16", critic_enabled=True,
        critic_capacity=500, critic_steps_per_iter=3, critic_batch_size=8,
        stability_enabled=True, stability_eval_every=1, stability_min_iters=1,
        stability_patience=10,
    )
    run_dir = tmp_path / "run"
    db_path = tmp_path / "db.sqlite"

    # First segment: run exactly 2 iterations (self.iterations still reflects the
    # full target so the LR schedule matches the eventual 4-iter run).
    tr1 = _make_trainer(cfg, run_dir, db_path, "v0.4-resume-2then",
                        eval_fn=lambda _t, t: 1.0 / t)
    hist1 = tr1.train(iterations=2)
    base_total = sum(hist1[-1].buffer_sizes.values())
    assert base_total > 0
    tr1.close()

    # Resume toward the full 4 iterations.
    tr2 = _make_trainer(cfg, run_dir, db_path, "v0.4-resume-2then",
                        eval_fn=lambda _t, t: 1.0 / t)
    hist2 = tr2.train(iterations=4, resume=True)
    tr2.close()

    # The loop resumed at iteration 3 and returned only iters 3, 4.
    assert [s.iteration for s in hist2] == [3, 4]
    # Reservoir length preserved: iter-3 buffers == carried base + iter-3 adds.
    first_resumed = hist2[0]
    assert (
        sum(first_resumed.buffer_sizes.values())
        == base_total + sum(first_resumed.samples_added.values())
    )
    # metrics.jsonl is a continuous 1..4 log (resumed rows appended).
    rows = [json.loads(x) for x in
            (run_dir / "metrics.jsonl").read_text().strip().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4]
    # Final snapshot + resume_state reflect the completed run.
    assert (run_dir / "snapshots" / "prtcfr_snapshot_iter_4.pt").exists()
    assert json.loads((run_dir / "resume_state.json").read_text())["iteration"] == 4


# ---------------------------------------------------------------------------
# (b) resume with no checkpoint raises cleanly and leaves the dir intact
# ---------------------------------------------------------------------------


def test_resume_no_checkpoint_raises(tmp_path):
    cfg = _prod_config(iterations=2)
    run_dir = tmp_path / "run"
    tr = _make_trainer(cfg, run_dir, tmp_path / "db.sqlite", "v0.4-resume-empty")
    with pytest.raises(PRTCFRResumeError) as ei:
        tr.train(iterations=2, resume=True)
    msg = str(ei.value).lower()
    assert "resume" in msg and "checkpoint" in msg
    # Dir intact: no snapshot / rolling checkpoint / metrics were produced.
    assert run_dir.exists()
    assert not (run_dir / "snapshots" / "prtcfr_checkpoint.pt").exists()
    assert not (run_dir / "metrics.jsonl").exists()
    assert not (run_dir / "resume_state.json").exists()
    tr.close()


def test_resume_missing_resume_state_raises(tmp_path):
    """A checkpoint present but resume_state.json absent also raises cleanly."""
    cfg = _prod_config(iterations=2)
    run_dir = tmp_path / "run"
    tr = _make_trainer(cfg, run_dir, tmp_path / "db.sqlite", "v0.4-resume-nostate")
    tr.train(iterations=2)
    tr.close()
    # Remove the resume marker, keep the checkpoint.
    (run_dir / "resume_state.json").unlink()
    tr2 = _make_trainer(cfg, run_dir, tmp_path / "db.sqlite", "v0.4-resume-nostate")
    with pytest.raises(PRTCFRResumeError) as ei:
        tr2.train(iterations=4, resume=True)
    assert "resume_state" in str(ei.value)
    tr2.close()


# ---------------------------------------------------------------------------
# (c) RNG restore: a resumed step draws exactly what a non-interrupted run would
# ---------------------------------------------------------------------------


def test_resume_rng_restore_draws_identically(tmp_path):
    cfg = _prod_config(iterations=2)
    run_dir = tmp_path / "run"
    db_path = tmp_path / "db.sqlite"

    tr = _make_trainer(cfg, run_dir, db_path, "v0.4-resume-rng")
    tr.train(iterations=2)
    # Snapshot the live RNG state a non-interrupted run would continue from.
    fit_state_ref = tr._fit_rng.bit_generator.state
    torch_state_ref = torch.random.get_rng_state().clone()
    # The reference reservoir draw is taken from a COPY of the state so tr's own
    # (and therefore its persisted) reservoir RNG is not advanced -- otherwise
    # close()'s reservoir re-save would clobber the on-disk state with the
    # post-draw one and the resumed draw would legitimately differ.
    ref_gen = np.random.default_rng()
    ref_gen.bit_generator.state = tr.reservoirs[0].raw._rng.bit_generator.state
    reservoir_draw_ref = float(ref_gen.random())
    tr.close()

    # A fresh instance restores that exact state from disk.
    tr2 = _make_trainer(cfg, run_dir, db_path, "v0.4-resume-rng")
    last_t = tr2._load_resume_state()
    assert last_t == 2

    # fit-side numpy Generator: identical serialized state.
    assert tr2._fit_rng.bit_generator.state == fit_state_ref
    # torch global RNG: restored to the saved end-of-iter-2 state.
    assert torch.equal(torch.random.get_rng_state(), torch_state_ref)
    # reservoir sampling RNG: the next draw matches the non-interrupted draw
    # exactly (tolerance-free) -- the load-bearing property for the fit batches.
    reservoir_draw_got = float(tr2.reservoirs[0].raw._rng.random())
    assert reservoir_draw_got == reservoir_draw_ref
    tr2.close()


# ---------------------------------------------------------------------------
# End-to-end reproducibility: interrupt-then-resume == uninterrupted on the
# resumed iterations (bit-identical fit loss). Sequential fp32 gen, critic OFF.
# ---------------------------------------------------------------------------


def _fit_losses(history):
    return [float(s.fit_loss) for s in history]


def test_resume_reproduces_uninterrupted_run(tmp_path):
    cfg = _prod_config(iterations=4)  # sequential fp32 gen, critic OFF, no stability

    # Reference: a single uninterrupted 4-iteration run.
    torch.manual_seed(12345)
    tr_u = _make_trainer(cfg, tmp_path / "U", tmp_path / "dbU.sqlite", "v0.4-repro-U")
    hist_u = tr_u.train(iterations=4)
    tr_u.close()
    u_losses = _fit_losses(hist_u)  # [l1, l2, l3, l4]
    assert len(u_losses) == 4

    # Split: identical net-init seed, interrupt at the start of iteration 3.
    torch.manual_seed(12345)
    run_dir = tmp_path / "S"
    db_path = tmp_path / "dbS.sqlite"
    tr_s = _make_trainer(cfg, run_dir, db_path, "v0.4-repro-S")
    orig_run = tr_s.run_iteration

    def interrupting_run(t):
        if t == 3:
            raise KeyboardInterrupt("simulated SIGINT at iter 3 start")
        return orig_run(t)

    tr_s.run_iteration = interrupting_run
    with pytest.raises(KeyboardInterrupt):
        tr_s.train(iterations=4)  # completes 1, 2; interrupted before 3

    # Resume the split run toward the full 4 iterations.
    tr_r = _make_trainer(cfg, run_dir, db_path, "v0.4-repro-S")
    hist_r = tr_r.train(iterations=4, resume=True)
    tr_r.close()

    assert [s.iteration for s in hist_r] == [3, 4]
    # The resumed fits are bit-identical to the uninterrupted run's iters 3, 4.
    r_losses = _fit_losses(hist_r)
    assert r_losses[0] == u_losses[2]
    assert r_losses[1] == u_losses[3]


# ---------------------------------------------------------------------------
# run_db: resume reuses the row by name (no duplicate, id + created_at stable)
# ---------------------------------------------------------------------------


def test_resume_reuses_run_db_row(tmp_path):
    cfg = _prod_config(iterations=4)
    run_dir = tmp_path / "run"
    db_path = tmp_path / "db.sqlite"
    name = "v0.4-resume-db"

    tr1 = _make_trainer(cfg, run_dir, db_path, name)
    tr1.train(iterations=2)
    tr1.close()

    db = run_db.get_db(str(db_path))
    row1 = db.execute(
        "SELECT id, created_at FROM runs WHERE name=?", (name,)
    ).fetchone()
    assert row1 is not None
    id1, created1 = row1["id"], row1["created_at"]
    db.close()

    tr2 = _make_trainer(cfg, run_dir, db_path, name)
    tr2.train(iterations=4, resume=True)
    tr2.close()

    db = run_db.get_db(str(db_path))
    rows = db.execute("SELECT id, created_at FROM runs WHERE name=?", (name,)).fetchall()
    # Exactly one row for the name; resume did not mint a fresh run.
    assert len(rows) == 1
    assert rows[0]["id"] == id1
    assert rows[0]["created_at"] == created1
    status = db.execute(
        "SELECT status FROM runs WHERE name=?", (name,)
    ).fetchone()["status"]
    assert status == "completed"
    db.close()
