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
    ledger, the reservoir sample count, and the net forward, and continues on
    the same trajectory an in-place resume of the source run would take;
  - net-only warm start raises by default (the cambia-374 guard) and, only
    with warm_start_net_only_ok=True, proceeds -- with an EMPTY reservoir,
    the exact root-cause behavior the guard exists to gate.
"""

import os
import random
import shutil

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


def _nashconv(root, net):
    """NashConv of ONE net's regret-matched policy on the tiny tree.

    A single snapshot is the degenerate case where the cambia-708 SERVED and
    PER_DECISION objects coincide: the served weighting divides each infoset
    by its own reach under that one snapshot, which cancels wherever the
    infoset is reachable at all. Both were measured at every eval point of
    runs A and B below and agreed to the bit."""
    net.eval()
    policy = materialize_policy_incremental(
        root, [(1, net)], weighting="linear", seq_cap=_SEQ_CAP
    )
    nashconv, _c = exploitability(root, policy)
    return float(nashconv)


def _make_eval(root, out):
    """Current-net NashConv eval_fn; appends (t, metric); no global-RNG use."""

    def eval_fn(trainer, t):
        metric = _nashconv(root, trainer.net)
        out.append((t, metric))
        return metric

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
    continuation: ledger, buffer, net, and the continued trajectory all carry
    forward.

    Continuity is asserted at the RUN BOUNDARY, not across a training gap
    (restated for cambia-2016). The original form compared A's last stability
    eval (t=12) with B's first (t=16) and allowed at most a 2x rise. Those two
    points are four fresh iterations apart, and NashConv is not monotone over
    four iterations on this reduced config: an uninterrupted 20-iteration run
    of the same seed reads 1.018, 1.235, 0.540, 0.861, 1.234, 0.734 at
    t=1,4,8,12,16,20. The bound also scaled with A's last value, so whenever A
    ended in a trough (it ends at 1/6 here) any ordinary wander tripped it.
    That form measured training luck rather than the warm start, and it failed
    identically on the master tip before cambia-708 landed, so the served
    objective is not what moved it.

    What replaces it is exact and deterministic:
      - at the boundary the net B loads scores A's last metric, because it IS
        A's net;
      - over the continued iterations, warm-starting A's state into a NEW run
        dir yields the same metrics as resuming A's own dir in place, which is
        the state-faithfulness claim itself (net, reservoir, RNG and LR
        schedule span all restored);
      - the reservoir starts at A's sample count, the direct cambia-374 guard.
    """
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

    # Capture the buffer length and the loaded net's score AFTER warm-start
    # load but BEFORE the first new traversal changes either, to prove both
    # are seeded from A's state rather than restarted.
    at_load = {}
    orig_run_iteration = tr_b.run_iteration

    def _capture_then_run(t):
        if not at_load:
            at_load["buffer"] = len(tr_b.buffer)
            at_load["nashconv"] = _nashconv(tiny_tree, tr_b.net)
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
    assert at_load["buffer"] > 0
    assert at_load["buffer"] == len(tr_a.buffer)

    # Boundary continuity: the net B carries into iteration _N_A + 1 is A's
    # net, so it scores A's last stability eval. Exact, and independent of how
    # the metric moves once new iterations start.
    assert evals_a and evals_b
    assert at_load["nashconv"] == pytest.approx(evals_a[-1][1], rel=1e-9, abs=1e-12)

    # Trajectory continuity: continuing A's state in a NEW dir via warm start
    # must match continuing A's OWN dir via the in-place resume path, which
    # restores the same net, reservoir and RNG under the same LR schedule span.
    # An unfaithful warm start (a dropped reservoir, an unrestored RNG, a
    # schedule spanning the wrong horizon) diverges here; a metric that merely
    # wanders does not, because both sides wander together.
    run_a_resumed = str(tmp_path / "run_a_resumed")
    shutil.copytree(run_a, run_a_resumed)
    _seed_all(999)  # same ambient stream B got; both paths override it on load
    evals_resumed = []
    tr_resumed = PRTCFRTinyTrainer(
        tiny_tree,
        _reduced_cfg(iterations=_N_B),
        run_dir=run_a_resumed,
        eval_fn=_make_eval(tiny_tree, evals_resumed),
    )
    tr_resumed.train(iterations=_N_B, resume=True)
    assert [t for t, _ in evals_b] == [t for t, _ in evals_resumed]
    assert [v for _, v in evals_b] == pytest.approx(
        [v for _, v in evals_resumed], rel=1e-9, abs=1e-12
    ), (
        f"warm-started continuation {evals_b} diverged from the in-place "
        f"resume of the same state {evals_resumed}: the warm start is not "
        f"carrying everything the continuation needs"
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
