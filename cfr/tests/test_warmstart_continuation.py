"""Validation gate for the cambia-374 warm-start continuation fix.

Root cause: net-only warm start (warm_start_path -> a bare snapshot .pt)
restarts the reservoir EMPTY, so the trainer fine-tunes the imported net on
regret targets from a tiny immature buffer -- a fresh run with a net prior,
never a state-faithful continuation. Full-mode warm start (warm_start_path ->
a run dir / resume_state.json) already restores net + reservoir via
_load_full_state and is the correct continuation path.

This module proves both halves on a reduced, CPU-fast copy of the X2 tiny
gate config (config/x2_tiny_gate.yaml):

  - full-mode warm start from a source run's DIRECTORY carries the snapshot
    ledger, the reservoir sample count, and NashConv continuity forward (no
    blow-up jump);
  - net-only warm start raises by default (the cambia-374 guard) and, only
    with warm_start_net_only_ok=True, proceeds -- with an EMPTY reservoir,
    the exact root-cause behavior the guard exists to gate.
"""

import os
import random

import numpy as np
import pytest
import torch

from src.cfr.prtcfr_eval import materialize_policy_incremental
from src.cfr.prtcfr_trainer import PRTCFRResumeError, PRTCFRTinyTrainer
from src.config import PRTCFRConfig, load_config
from tools.tiny_solver import build_tree, exploitability

_SEQ_CAP = 32
_N_A = 12  # run A: fresh, completes to iteration 12
_N_B = 20  # run B: full-mode warm start from A, +8 new iterations (13..20)


@pytest.fixture(scope="module")
def tiny_tree():
    cfg = load_config("config/tiny_2card_plateau.yaml")
    root, _isets, _n, aborted = build_tree(
        cfg,
        1,
        0,
        2_000_000,
        enumerate_draws=True,
        perfect_recall=True,
        tokenize=True,
        seq_cap=_SEQ_CAP,
    )
    assert aborted == 0
    return root


def _seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _reduced_cfg(**over):
    """A reduced-scale copy of config/x2_tiny_gate.yaml: same net shape,
    LR schedule, and stability cadence as the real X2 gate config, shrunk
    (net dims, seq_cap, iteration/game/step counts) so the whole module runs
    in CPU seconds instead of the real gate's hours."""
    cfg_top = load_config("config/x2_tiny_gate.yaml")
    base = cfg_top.prt_cfr.model_dump()
    base.update(
        iterations=_N_A,
        k_games_per_iter=16,
        train_steps_per_iter=96,
        stability_eval_every=4,
        device="cpu",
        seed=0,
        seq_cap=_SEQ_CAP,
        gru_embed_dim=16,
        gru_hidden_dim=32,
        gru_num_layers=2,
        head_hidden_dim=32,
        batch_size=64,
    )
    base.update(over)
    return PRTCFRConfig(**base)


def _make_eval(root, out):
    """Current-net NashConv eval_fn; appends (t, metric); no global-RNG use."""

    def eval_fn(trainer, t):
        trainer.net.eval()
        policy = materialize_policy_incremental(
            root, [(1, trainer.net)], weighting="linear", seq_cap=_SEQ_CAP
        )
        nashconv, _c = exploitability(root, policy)
        out.append((t, float(nashconv)))
        return float(nashconv)

    return eval_fn


@pytest.fixture(scope="module")
def warm_run_a(tiny_tree, tmp_path_factory):
    """Run A to completion once; shared read-only by both tests below."""
    run_dir = str(tmp_path_factory.mktemp("run_a"))
    _seed_all(12345)
    evals = []
    tr = PRTCFRTinyTrainer(
        tiny_tree, _reduced_cfg(), run_dir=run_dir, eval_fn=_make_eval(tiny_tree, evals)
    )
    history = tr.train(iterations=_N_A)
    return {"trainer": tr, "run_dir": run_dir, "evals": evals, "history": history}


def test_full_mode_warm_start_is_continuous(tiny_tree, warm_run_a, tmp_path):
    """Full-mode warm start (a source run's DIRECTORY) is a state-faithful
    continuation: ledger, buffer, and NashConv trend all carry forward."""
    tr_a = warm_run_a["trainer"]
    run_a = warm_run_a["run_dir"]
    evals_a = warm_run_a["evals"]

    # A persisted exactly what a continuation needs.
    assert os.path.exists(os.path.join(run_a, "resume_state.json"))
    assert os.path.exists(os.path.join(run_a, "reservoir.npz"))
    for t in range(1, _N_A + 1):
        assert os.path.exists(
            os.path.join(run_a, "snapshots", f"prtcfr_snapshot_iter_{t}.pt")
        )

    _seed_all(999)  # ambient stream pre-construction; full mode overrides RNG on load
    evals_b = []
    run_b = str(tmp_path / "run_b_full")
    cfg_b = _reduced_cfg(warm_start_path=run_a, iterations=_N_B)
    tr_b = PRTCFRTinyTrainer(
        tiny_tree, cfg_b, run_dir=run_b, eval_fn=_make_eval(tiny_tree, evals_b)
    )

    # Capture the buffer length AFTER warm-start-load but BEFORE the first
    # new traversal adds anything, to prove it is seeded from A's state, not
    # restarted at zero.
    buf_at_start = {}
    orig_run_iteration = tr_b.run_iteration

    def _capture_then_run(t):
        buf_at_start.setdefault("n", len(tr_b.buffer))
        return orig_run_iteration(t)

    tr_b.run_iteration = _capture_then_run
    tr_b.train(iterations=_N_B)

    # Ledger: B's written-iteration set is ALL of A's snapshot iterations
    # plus its own new ones, not just the new tail.
    assert tr_b._written_iters == list(range(1, _N_B + 1))

    # Prior snapshots were imported into B's own snapshot dir too, so B's
    # SD-CFR average spans [1..N_B], not just the continued iterations.
    for t in range(1, _N_A + 1):
        assert os.path.exists(
            os.path.join(run_b, "snapshots", f"prtcfr_snapshot_iter_{t}.pt")
        )

    # Buffer starts at A's persisted sample count, not zero (the cambia-374
    # bug: net-only warm start restarts this at 0 -- see the guard test below).
    assert buf_at_start["n"] > 0
    assert buf_at_start["n"] == len(tr_a.buffer)

    # NashConv continuity: B's first stability eval stays close to A's last,
    # not the >2x-style degradation jump the net-only path produced (measured
    # on the reduced config: ratio ~0.69, i.e. the continuation keeps
    # improving). Bound pinned with generous margin against run-to-run float
    # noise while still catching an empty-reservoir-style regression.
    assert evals_a and evals_b
    nashconv_a_last = evals_a[-1][1]
    nashconv_b_first = evals_b[0][1]
    assert nashconv_b_first <= nashconv_a_last * 2.0 + 0.05, (
        f"B's first NashConv ({nashconv_b_first}) degraded >2x vs A's last "
        f"({nashconv_a_last}) -- looks like the cambia-374 empty-reservoir "
        f"regression, not a state-faithful continuation"
    )


def test_full_mode_warm_start_registers_imported_snapshots_in_run_db(
    tiny_tree, warm_run_a, tmp_path
):
    """Every prtcfr_snapshot_iter_N.pt imported by a warm start must get a
    checkpoints row in run_db, exactly like a natively-written checkpoint
    (cambia-389). The harness pull include-set is derived from checkpoint
    rows, so an unregistered import is invisible to artifact sync even
    though resume_state.json's snapshots ledger lists it."""
    run_a = warm_run_a["run_dir"]

    _seed_all(2026)
    run_b = str(tmp_path / "run_b_dbcheck")
    db_path = str(tmp_path / "cambia_runs.db")
    cfg_b = _reduced_cfg(warm_start_path=run_a, iterations=_N_A)
    tr_b = PRTCFRTinyTrainer(
        tiny_tree,
        cfg_b,
        run_dir=run_b,
        run_name="cambia-389-warmstart-dbcheck",
        db_path=db_path,
    )
    # iterations == _N_A == A's last iteration: the training loop runs zero
    # NEW iterations, isolating import-time registration from the
    # natively-produced-checkpoint registration run_iteration already does.
    history = tr_b.train(iterations=_N_A)
    assert history == []

    assert tr_b._db_conn is not None and tr_b._db_run_id is not None
    rows = tr_b._db_conn.execute(
        "SELECT iteration, file_path FROM checkpoints WHERE run_id=? ORDER BY iteration",
        (tr_b._db_run_id,),
    ).fetchall()
    got_iters = [r["iteration"] for r in rows]
    assert got_iters == list(range(1, _N_A + 1)), (
        f"expected a checkpoint row per imported snapshot iter 1..{_N_A}, got "
        f"{got_iters} -- the harness pull include-set derives from these rows, "
        f"so a gap here means iter_N.pt is skipped by artifact sync"
    )
    for r in rows:
        assert os.path.exists(r["file_path"])
        assert r["file_path"] == os.path.join(
            run_b, "snapshots", f"prtcfr_snapshot_iter_{r['iteration']}.pt"
        )


def test_net_only_warm_start_guard(tiny_tree, warm_run_a, tmp_path):
    """Net-only warm start (a bare snapshot .pt) raises by default
    (cambia-374) and is permitted -- with an EMPTY reservoir, the root-cause
    behavior -- only via the explicit warm_start_net_only_ok opt-in."""
    run_a = warm_run_a["run_dir"]
    snap_path = os.path.join(run_a, "snapshots", f"prtcfr_snapshot_iter_{_N_A}.pt")
    n_continue = _N_A + 2

    _seed_all(555)
    cfg_default = _reduced_cfg(warm_start_path=snap_path, iterations=n_continue)
    tr_default = PRTCFRTinyTrainer(
        tiny_tree, cfg_default, run_dir=str(tmp_path / "net_default")
    )
    with pytest.raises(PRTCFRResumeError, match="cambia-374"):
        tr_default.train(iterations=n_continue)

    _seed_all(555)
    cfg_ok = _reduced_cfg(
        warm_start_path=snap_path,
        warm_start_net_only_ok=True,
        iterations=n_continue,
    )
    tr_ok = PRTCFRTinyTrainer(tiny_tree, cfg_ok, run_dir=str(tmp_path / "net_ok"))

    buf_at_start = {}
    orig_run_iteration = tr_ok.run_iteration

    def _capture_then_run(t):
        buf_at_start.setdefault("n", len(tr_ok.buffer))
        return orig_run_iteration(t)

    tr_ok.run_iteration = _capture_then_run
    tr_ok.train(iterations=n_continue)

    # The opt-in proceeds, but the root cause is right there: empty reservoir.
    assert buf_at_start["n"] == 0
