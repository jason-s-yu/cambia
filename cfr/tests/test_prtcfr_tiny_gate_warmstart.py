"""Equivalence gate for the tiny-gate trainer's full-state warm-start (cambia-334).

The arbiter for W1: a run split at iteration j and continued from a saved state
must reproduce, bit-for-bit, the uninterrupted run's tail. On CPU with fixed
seeds and a tiny budget, this asserts:

  - per-iteration snapshot net weights identical (torch.equal per tensor) for
    every continued iteration j+1 .. N;
  - the final net state identical;
  - the NashConv eval trajectory identical over the overlapping evals;
  - the iteration counters and the stability controller's best-so-far identical.

A negative control perturbs only the saved RNG and asserts the tail DIVERGES, so
the positive test cannot pass vacuously (nothing stochastic happening).

The eval_fn scores the CURRENT net's exact NashConv (net.eval(), so no dropout
noise and no global-RNG consumption), which depends only on restored trainer
state -- the clean probe for state fidelity. The CLI wires the SD-CFR-average
scorer (build_tiny_nashconv_eval_fn) instead.
"""

import json
import os
import random
import shutil

import numpy as np
import pytest
import torch

from src.cfr.prtcfr_eval import materialize_policy_incremental
from src.cfr.prtcfr_trainer import PRTCFRTinyTrainer, _global_rng_save
from src.config import PRTCFRConfig, load_config
from tools.tiny_solver import build_tree, exploitability

_SEQ_CAP = 32
_SEED = 12345
_N = 6
_J = 4  # EVAL-DUE (eval_every=2 -> due at 1,2,4,6): resume_state(j) must carry
# iteration j's OWN controller update (cambia-341 regression coverage), not
# just the update from the last-due iteration before it.


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


def _cfg(**over):
    base = dict(
        tiny_gate=True,
        # Shrunk net keeps fit + eval fast; num_layers=2 keeps GRU dropout live
        # so a perturbed torch RNG actually changes the fit (negative control).
        gru_embed_dim=16,
        gru_hidden_dim=32,
        gru_num_layers=2,
        head_hidden_dim=32,
        gru_dropout=0.1,
        seq_cap=_SEQ_CAP,
        m_rollouts=1,
        k_games_per_iter=6,
        train_steps_per_iter=8,
        batch_size=8,
        lr=1.0e-3,
        lr_min=1.0e-4,
        lr_schedule="global_cosine",
        warm_start=True,
        device="cpu",
        seed=0,
        stability_enabled=True,
        stability_eval_every=2,
        stability_patience=100,  # never early-stop within N
        stability_min_iters=10_000,  # never early-stop within N
        stability_metric_mode="min",
        stability_metric_name="nashconv",
    )
    base.update(over)
    return PRTCFRConfig(**base)


def _seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _make_eval(root, out):
    """current-net NashConv eval_fn; appends (t, metric); no global-RNG use."""

    def eval_fn(trainer, t):
        trainer.net.eval()
        policy = materialize_policy_incremental(
            root, [(1, trainer.net)], weighting="linear", seq_cap=_SEQ_CAP
        )
        nashconv, _c = exploitability(root, policy)
        out.append((t, float(nashconv)))
        return float(nashconv)

    return eval_fn


def _state_equal(sd1, sd2):
    if set(sd1.keys()) != set(sd2.keys()):
        return False
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def _nets_equal(a, b):
    return _state_equal(a.encoder_state_dict(), b.encoder_state_dict()) and _state_equal(
        a.head_state_dict(), b.head_state_dict()
    )


def _snap_equal(p1, p2):
    d1 = torch.load(p1, map_location="cpu", weights_only=False)
    d2 = torch.load(p2, map_location="cpu", weights_only=False)
    return _state_equal(
        d1["encoder_state_dict"], d2["encoder_state_dict"]
    ) and _state_equal(d1["head_state_dict"], d2["head_state_dict"])


def _run_cold(root, run_dir, evals):
    _seed_all(_SEED)
    tr = PRTCFRTinyTrainer(root, _cfg(), run_dir=run_dir, eval_fn=_make_eval(root, evals))
    hist = tr.train(iterations=_N)
    return tr, hist


def _run_split_part1(root, run_dir, evals):
    """Run 1..J with the schedule spanning N, then simulate an interrupt at
    the J/(J+1) boundary so resume_state(J) is the on-disk resume point."""
    _seed_all(_SEED)
    tr = PRTCFRTinyTrainer(root, _cfg(), run_dir=run_dir, eval_fn=_make_eval(root, evals))
    orig = tr.run_iteration

    def stop_after_j(t):
        if t > _J:
            raise KeyboardInterrupt()
        return orig(t)

    tr.run_iteration = stop_after_j
    with pytest.raises(KeyboardInterrupt):
        tr.train(iterations=_N)  # self.iterations == N -> schedule spans N
    tr.close()
    return tr


def test_warm_start_full_state_is_bit_exact(tiny_tree, tmp_path):
    root = tiny_tree
    cold_evals, split_evals = [], []

    cold_dir = str(tmp_path / "cold")
    cold_tr, cold_hist = _run_cold(root, cold_dir, cold_evals)

    split_a = str(tmp_path / "split_a")
    _run_split_part1(root, split_a, [])

    # Fresh trainer, warm-start (full) from split_a's resume_state.json into a
    # new run dir; the schedule again spans N so the tail matches cold.
    split_b = str(tmp_path / "split_b")
    cfg_b = _cfg(warm_start_path=os.path.join(split_a, "resume_state.json"))
    split_tr = PRTCFRTinyTrainer(
        root, cfg_b, run_dir=split_b, eval_fn=_make_eval(root, split_evals)
    )
    split_hist = split_tr.train(iterations=_N)

    # Iteration counters: cold ran 1..N; the resumed part ran J+1..N.
    assert [h.iteration for h in cold_hist] == list(range(1, _N + 1))
    assert [h.iteration for h in split_hist] == list(range(_J + 1, _N + 1))

    # Per-iteration snapshot weights identical for every continued iteration.
    for t in range(_J + 1, _N + 1):
        assert _snap_equal(
            os.path.join(cold_dir, "snapshots", f"prtcfr_snapshot_iter_{t}.pt"),
            os.path.join(split_b, "snapshots", f"prtcfr_snapshot_iter_{t}.pt"),
        ), f"snapshot mismatch at iter {t}"

    # Final net identical.
    assert _nets_equal(cold_tr.net, split_tr.net)

    # NashConv trajectory identical over the overlapping evals (t >= J+1).
    cold_tail = {t: v for t, v in cold_evals if t > _J}
    split_tail = {t: v for t, v in split_evals if t > _J}
    assert cold_tail and split_tail
    assert set(cold_tail) == set(split_tail)
    for t in cold_tail:
        assert cold_tail[t] == split_tail[t], f"nashconv differs at iter {t}"

    # Controller best-so-far identical.
    assert cold_tr.controller.best_iteration == split_tr.controller.best_iteration
    assert cold_tr.controller.best_metric == split_tr.controller.best_metric


def test_warm_start_perturbed_rng_diverges(tiny_tree, tmp_path):
    """Negative control: perturb ONLY the saved RNG and the tail must differ."""
    root = tiny_tree
    cold_evals = []
    cold_dir = str(tmp_path / "cold")
    cold_tr, _ = _run_cold(root, cold_dir, cold_evals)

    split_a = str(tmp_path / "split_a")
    _run_split_part1(root, split_a, [])

    # Copy split_a, then overwrite ONLY the RNG fields of its resume_state with
    # a different seed's streams (net / buffer / controller / counter unchanged).
    perturbed = str(tmp_path / "perturbed")
    shutil.copytree(split_a, perturbed)
    rs_path = os.path.join(perturbed, "resume_state.json")
    with open(rs_path) as fh:
        state = json.load(fh)
    _seed_all(999)
    state.update(_global_rng_save())  # numpy_rng + python_rng + torch_rng
    with open(rs_path, "w") as fh:
        json.dump(state, fh)

    split_b = str(tmp_path / "split_b")
    cfg_b = _cfg(warm_start_path=os.path.join(perturbed, "resume_state.json"))
    split_tr = PRTCFRTinyTrainer(
        root, cfg_b, run_dir=split_b, eval_fn=_make_eval(root, [])
    )
    split_tr.train(iterations=_N)

    # A different RNG stream (GRU dropout masks + buffer sampling) must yield a
    # different fit, so the continued tail is NOT bit-identical to cold's.
    assert not _nets_equal(cold_tr.net, split_tr.net)
