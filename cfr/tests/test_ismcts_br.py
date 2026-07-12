"""Tests for ISMCTS-BR (cfr/src/cfr/ismcts_br.py): the information-set Monte-Carlo
tree-search best-response exploitability estimator.

Three properties, per the P3W4 spec:

  1. Calibration. On the tiny {A,6} game, coupled to the SAME K-deal Monte-Carlo
     chance root that tools/tiny_solver.py solves exactly, the ISMCTS-BR estimate
     converges to the solver's exact perfect-recall best-response gap
     (br0 - onp0 vs the uniform policy) within a measured tolerance as the search
     budget grows. Coupling to the identical deal seeds removes deal-sampling noise
     between estimator and reference, so the only residual is the estimator's
     search/greedy-extraction error, which shrinks with budget.

  2. Tighter bound than LBR. On a known-exploitable stub policy the ISMCTS-BR
     estimate is >= the one-ply sampled-LBR estimate for the same policy: the
     multi-ply search recovers exploitation a one-ply lookahead misses.

  3. Determinism. Identical output under a fixed seed.

Budgets are kept small so this file runs in well under a few minutes on CPU (the
tiny_solver reference build is the largest single cost).
"""

import random

import pytest

from src.config import load_config
from src.cfr.ismcts_br import ismcts_br
from src.cfr.sampled_lbr import sampled_lbr
from src.constants import ActionDrawStockpile, ActionDiscard
from tools.tiny_solver import build_tree, _br_value, _policy_value

TINY_CONFIG = "config/tiny_2card_plateau.yaml"

# --- Calibration budget + tolerance (measured; do not tighten without re-measuring). ---
# Reference: tools/tiny_solver exact perfect-recall BR gap (br0 - onp0) vs the
# uniform policy on a CALIB_K-deal subgame. The estimator is coupled to the same
# CALIB_K deal seeds (deal_seeds=range(K)), so it integrates over the identical
# chance root -- no deal-sampling mismatch, only estimator error.
#
# Measured across 10 seeds at (CALIB_HI_ITERS search sims, CALIB_HI_GAMES eval
# games): the coupled estimate lands 0.00-0.05 BELOW the exact gap (a systematic
# downward bias from greedy extraction + uniform fallback on unseen infosets);
# worst |error| = 0.050. At CALIB_LOW_ITERS the |error| is ~0.4. CALIB_TOL = 0.08
# clears the worst observed case with margin while staying tight relative to the
# ~0.76 exact gap (~10%).
CALIB_K = 8
CALIB_LOW_ITERS = 300
CALIB_LOW_GAMES = 1000
CALIB_HI_ITERS = 6000
CALIB_HI_GAMES = 4000
CALIB_TOL = 0.08
CALIB_SEED = 7


class _UniformWrapper:
    """Uniform-random target over the repr-sorted legal set (its own RNG so the
    game-value baseline is seed-deterministic). Matches the solver's uniform
    (empty) policy, the calibration reference.
    """

    def __init__(self, seed):
        self._rng = random.Random(seed)

    def choose_action(self, state, legal):
        actions = sorted(legal, key=repr)
        return actions[self._rng.randrange(len(actions))]


class _PassiveStub:
    """Deterministic, strongly-suboptimal target: draw, then always discard the
    drawn card; never replace, snap, or call Cambia. Highly exploitable, and the
    exploitation needs a multi-turn line, so a multi-ply BR far exceeds one-ply
    LBR.
    """

    def choose_action(self, state, legal):
        actions = sorted(legal, key=repr)
        draws = [a for a in actions if isinstance(a, ActionDrawStockpile)]
        if draws:
            return draws[0]
        discards = [a for a in actions if isinstance(a, ActionDiscard)]
        if discards:
            return discards[0]
        return actions[0]


@pytest.fixture(scope="module")
def cfg():
    return load_config(TINY_CONFIG)


@pytest.fixture(scope="module")
def exact_br_gap(cfg):
    """tools/tiny_solver exact perfect-recall BR gap (br0 - onp0) vs the uniform
    policy on the CALIB_K-deal {A,6} subgame -- the calibration reference. The
    empty policy ({}) resolves to uniform in the solver's _lookup, so br0 is the
    exact best-response value vs a uniform opponent and onp0 the uniform self-play
    value.
    """
    root, _isets, _n, aborted = build_tree(
        cfg,
        CALIB_K,
        0,
        2_000_000,
        enumerate_draws=False,
        perfect_recall=True,
        tokenize=False,
        quiet=True,
    )
    assert aborted == 0, "tiny reference tree truncated (raise max nodes)"
    br0 = _br_value(root, 0, {})
    onp0 = _policy_value(root, {0: {}, 1: {}}, 0)
    return br0 - onp0


def test_calibration_converges_to_exact_br(cfg, exact_br_gap):
    """Coupled to the solver's CALIB_K deals, ISMCTS-BR converges to the exact BR
    gap within CALIB_TOL as the budget grows, and the error strictly shrinks from
    the low to the high budget.
    """
    deal_seeds = list(range(CALIB_K))
    low = ismcts_br(
        _UniformWrapper(1234),
        cfg,
        ismcts_iterations=CALIB_LOW_ITERS,
        eval_games=CALIB_LOW_GAMES,
        seed=CALIB_SEED,
        deal_seeds=deal_seeds,
    )
    high = ismcts_br(
        _UniformWrapper(1234),
        cfg,
        ismcts_iterations=CALIB_HI_ITERS,
        eval_games=CALIB_HI_GAMES,
        seed=CALIB_SEED,
        deal_seeds=deal_seeds,
    )
    low_err = abs(low["exploitability"] - exact_br_gap)
    high_err = abs(high["exploitability"] - exact_br_gap)

    # The exact gap is large and positive: uniform play is highly exploitable here.
    assert exact_br_gap > 0.5, f"unexpected reference gap {exact_br_gap:.4f}"

    # (a) the high-budget estimate lands within the stated tolerance of exact BR.
    assert high_err <= CALIB_TOL, (
        f"ISMCTS-BR={high['exploitability']:.4f} vs exact BR gap={exact_br_gap:.4f} "
        f"(|error|={high_err:.4f} > tol={CALIB_TOL})"
    )
    # (b) it converges: growing the budget strictly reduces the error.
    assert high_err < low_err, (
        f"no convergence: |error| at {CALIB_HI_ITERS} iters ({high_err:.4f}) is not "
        f"below |error| at {CALIB_LOW_ITERS} iters ({low_err:.4f})"
    )


def test_tighter_bound_than_lbr(cfg):
    """On a known-exploitable stub, ISMCTS-BR (multi-ply) >= sampled LBR (one-ply)
    for the same policy: a tighter lower bound on true exploitability.
    """
    lbr = sampled_lbr(
        _PassiveStub(), cfg, num_infosets=800, br_rollouts_per_infoset=20, seed=13
    )
    ism = ismcts_br(
        _PassiveStub(), cfg, ismcts_iterations=4000, eval_games=3000, seed=13
    )
    assert ism["exploitability"] >= lbr["exploitability"], (
        f"ISMCTS-BR ({ism['exploitability']:.4f}) must be a tighter (>=) bound than "
        f"one-ply LBR ({lbr['exploitability']:.4f}) on the exploitable stub"
    )
    # Both must agree the stub is meaningfully exploitable (not a degenerate 0>=0).
    assert lbr["exploitability"] > 0.1


def test_deterministic_under_seed(cfg):
    """Identical output for identical (seed, arguments)."""
    a = ismcts_br(
        _UniformWrapper(0), cfg, ismcts_iterations=1500, eval_games=1500, seed=55
    )
    b = ismcts_br(
        _UniformWrapper(0), cfg, ismcts_iterations=1500, eval_games=1500, seed=55
    )
    assert a["exploitability"] == b["exploitability"]
    assert a["br_value"] == b["br_value"]
    assert a["game_value"] == b["game_value"]
    assert a["num_infosets_sampled"] == b["num_infosets_sampled"]


def test_result_shape_and_nonnegative(cfg):
    """Return-dict shape (mirrors src.cfr.lbr.tier_b_lbr) and basic invariants."""
    r = ismcts_br(
        _UniformWrapper(0), cfg, ismcts_iterations=400, eval_games=400, seed=1
    )
    for key in (
        "exploitability",
        "br_value",
        "game_value",
        "num_infosets_sampled",
        "std_err",
        "estimator",
        "ismcts_iterations",
        "eval_games",
        "ucb_c",
    ):
        assert key in r, f"missing key {key}"
    assert r["estimator"] == "ismcts_br"
    assert r["exploitability"] >= 0.0
    assert r["std_err"] >= 0.0
    assert r["num_infosets_sampled"] > 0
    assert isinstance(r["exploitability"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
