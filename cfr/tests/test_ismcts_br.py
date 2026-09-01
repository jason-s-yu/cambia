"""Tests for ISMCTS-BR (cfr/src/cfr/ismcts_br.py): the information-set Monte-Carlo
tree-search best-response exploitability estimator.

Four properties, per the P3W4 spec plus the cambia-1427 port:

  1. Calibration. On the tiny {A,6} game, coupled to the SAME K deals that
     tools/tiny_solver.py solves exactly, the ISMCTS-BR estimate converges to the
     solver's exact perfect-recall best-response gap (br0 - onp0 vs the uniform
     policy) within a measured tolerance as the search budget grows. Coupling to
     the identical deals removes deal-sampling noise between estimator and
     reference, so the only residual is the estimator's search/greedy-extraction
     error, which shrinks with budget.

  2. Tighter bound than LBR. On a known-exploitable stub policy the ISMCTS-BR
     estimate is >= the one-ply sampled-LBR estimate for the same policy: the
     multi-ply search recovers exploitation a one-ply lookahead misses.

  3. Determinism. Identical output under a fixed seed.

  4. Info-key exactness. The O(1)-extend incremental key stays content-identical
     to the O(L) rebuild and induces the same tree-node sharing.

Deal coupling after the Go port (cambia-1427 D3)
------------------------------------------------
The estimator now runs on the Go engine while tiny_solver still builds its exact
tree on the Python reference engine, and the two engines do NOT share a PRNG: a
seed means different cards to each. Seed coupling would therefore silently
reintroduce the deal-sampling mismatch the calibration exists to remove. The
fixture instead transplants the solver's actual deals -- it extracts each Python
deal's deck order and hands the pool to the estimator via ``deal_decks``, which
is engine-independent. This test imports the Python engine to build the
reference; the estimator modules themselves no longer do. When cambia-1429 lands
a Go-native tiny_solver, both sides can couple on Go seeds and the transplant
goes away.

Budgets are kept small so this file runs in well under a few minutes on CPU (the
tiny_solver reference build is the largest single cost).
"""

import random

import pytest

from src.agents.action_codec import actions_from_indices
from src.cfr.ismcts_br import ismcts_br
from src.cfr.sampled_lbr import sampled_lbr
from src.config import load_config
from src.constants import ActionDiscard, ActionDrawStockpile
from tools.tiny_solver import build_tree, _br_value, _policy_value

TINY_CONFIG = "config/tiny_2card_plateau.yaml"

# --- Calibration budget + tolerance (measured; do not tighten without re-measuring). ---
# Reference: tools/tiny_solver exact perfect-recall BR gap (br0 - onp0) vs the
# uniform policy on a CALIB_K-deal subgame. The estimator is coupled to the same
# CALIB_K deals (deal_decks transplanted from the solver's own deals), so it
# integrates over the identical chance root -- no deal-sampling mismatch, only
# estimator error.
CALIB_K = 8
CALIB_LOW_ITERS = 300
CALIB_LOW_GAMES = 1000
CALIB_HI_ITERS = 6000
CALIB_HI_GAMES = 4000
CALIB_TOL = 0.08
CALIB_SEED = 7


class _UniformWrapper:
    """Uniform-random target with its own RNG, so the game-value baseline is
    seed-deterministic. Matches the solver's uniform (empty) policy, the
    calibration reference.

    Written against the cambia-1427 policy boundary: a GameView plus a list of
    GameAction NamedTuples already in the engine's ascending index order (the old
    repr-sort existed only because the Python engine handed over an unordered
    set).
    """

    accepts_game_view = True

    def __init__(self, seed):
        self._rng = random.Random(seed)

    def choose_action(self, view, legal):
        actions = list(legal)
        return actions[self._rng.randrange(len(actions))]


class _PassiveStub:
    """Deterministic, strongly-suboptimal target: draw, then always discard the
    drawn card; never replace, snap, or call Cambia. Highly exploitable, and the
    exploitation needs a multi-turn line, so a multi-ply BR far exceeds one-ply
    LBR.
    """

    accepts_game_view = True

    def choose_action(self, view, legal):
        actions = list(legal)
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


# Deals for the uncoupled tests. The tiny config restricts the deck via
# deck_ranks; the Go rules struct carries that as a rank mask since cambia-1478,
# but these cases still deal from a transplanted deck pool so both engines walk
# the identical deal, which is what makes a Go/Python comparison meaningful.
# POOL_N is wide enough to stand in for "the deal distribution" without being
# the 8-deal calibration root.
POOL_N = 64


@pytest.fixture(scope="module")
def deal_pool(cfg):
    """A wider transplanted deck pool, standing in for the deal distribution."""
    return _transplant_decks(cfg, range(1000, 1000 + POOL_N))


def _transplant_decks(cfg, seeds):
    from src.ffi.bridge import extract_deck_from_python_game
    from src.game.engine import CambiaGameState

    decks = []
    for s in seeds:
        game = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(s))
        deck, starting_player = extract_deck_from_python_game(game)
        decks.append((deck, starting_player))
    return decks


@pytest.fixture(scope="module")
def calib_decks(cfg):
    """The solver's CALIB_K deals, as (deck order, starting player) pairs.

    build_tree deals with ``CambiaGameState(house_rules=cfg.cambia_rules,
    _rng=random.Random(seed0 + d))`` at seed0=0, so this reproduces exactly those
    K games and extracts each one's deck. Handing the decks to the estimator
    couples it to the reference's chance root without either side having to share
    a PRNG.
    """
    return _transplant_decks(cfg, range(CALIB_K))


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


def test_deck_transplant_reproduces_the_reference_deal(cfg, calib_decks):
    """The coupling itself: a transplanted deck must deal the same hands on the Go
    engine as the Python deal it came from. If this drifts, the calibration below
    is comparing two different chance roots and its tolerance is meaningless.
    """
    from src.game.engine import CambiaGameState
    from src.cfr.lbr import GoSearchState

    for d, (deck, starting_player) in enumerate(calib_decks):
        py = CambiaGameState(house_rules=cfg.cambia_rules, _rng=random.Random(d))
        go = GoSearchState.from_deck(cfg.cambia_rules, deck, starting_player)
        try:
            for seat in (0, 1):
                py_ranks = [c.rank for c in py.get_player_hand(seat)]
                go_ranks = [c.rank for c in go.view().get_player_hand(seat)]
                assert py_ranks == go_ranks, (
                    f"deal {d} seat {seat}: Go hand {go_ranks} != Python hand "
                    f"{py_ranks}; the deck transplant has drifted"
                )
            assert go.acting_player() == py.get_acting_player()
        finally:
            go.close()


def test_calibration_converges_to_exact_br(cfg, exact_br_gap, calib_decks):
    """Coupled to the solver's CALIB_K deals, ISMCTS-BR converges to the exact BR
    gap within CALIB_TOL as the budget grows, and the error strictly shrinks from
    the low to the high budget.
    """
    low = ismcts_br(
        _UniformWrapper(1234),
        cfg,
        ismcts_iterations=CALIB_LOW_ITERS,
        eval_games=CALIB_LOW_GAMES,
        seed=CALIB_SEED,
        deal_decks=calib_decks,
    )
    high = ismcts_br(
        _UniformWrapper(1234),
        cfg,
        ismcts_iterations=CALIB_HI_ITERS,
        eval_games=CALIB_HI_GAMES,
        seed=CALIB_SEED,
        deal_decks=calib_decks,
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


def test_tighter_bound_than_lbr(cfg, deal_pool):
    """On a known-exploitable stub, ISMCTS-BR (multi-ply) >= sampled LBR (one-ply)
    for the same policy: a tighter lower bound on true exploitability.
    """
    lbr = sampled_lbr(
        _PassiveStub(),
        cfg,
        num_infosets=800,
        br_rollouts_per_infoset=20,
        seed=13,
        deal_decks=deal_pool,
    )
    ism = ismcts_br(
        _PassiveStub(),
        cfg,
        ismcts_iterations=4000,
        eval_games=3000,
        seed=13,
        deal_decks=deal_pool,
    )
    assert ism["exploitability"] >= lbr["exploitability"], (
        f"ISMCTS-BR ({ism['exploitability']:.4f}) must be a tighter (>=) bound than "
        f"one-ply LBR ({lbr['exploitability']:.4f}) on the exploitable stub"
    )
    # Both must agree the stub is meaningfully exploitable (not a degenerate 0>=0).
    assert lbr["exploitability"] > 0.1


def test_deterministic_under_seed(cfg, deal_pool):
    """Identical output for identical (seed, arguments)."""
    a = ismcts_br(
        _UniformWrapper(0),
        cfg,
        ismcts_iterations=1500,
        eval_games=1500,
        seed=55,
        deal_decks=deal_pool,
    )
    b = ismcts_br(
        _UniformWrapper(0),
        cfg,
        ismcts_iterations=1500,
        eval_games=1500,
        seed=55,
        deal_decks=deal_pool,
    )
    assert a["exploitability"] == b["exploitability"]
    assert a["br_value"] == b["br_value"]
    assert a["game_value"] == b["game_value"]
    assert a["num_infosets_sampled"] == b["num_infosets_sampled"]


def test_result_shape_and_nonnegative(cfg, deal_pool):
    """Return-dict shape (mirrors src.cfr.lbr.tier_b_lbr) and basic invariants."""
    r = ismcts_br(
        _UniformWrapper(0),
        cfg,
        ismcts_iterations=400,
        eval_games=400,
        seed=1,
        deal_decks=deal_pool,
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
        "seed",
    ):
        assert key in r, f"missing key {key}"
    assert r["estimator"] == "ismcts_br"
    assert r["seed"] == 1
    assert r["exploitability"] >= 0.0
    assert r["std_err"] >= 0.0
    assert r["num_infosets_sampled"] > 0
    assert isinstance(r["exploitability"], float)


def test_no_handle_leak_across_a_search(cfg, deal_pool):
    """Every determinization holds a game handle plus two agent handles out of a
    finite pool; a full search-and-eval run must return all of them."""
    from src.ffi.bridge import get_handle_pool_stats

    before = get_handle_pool_stats()
    assert set(before) >= {"games", "agents", "snapshots"}, before
    ismcts_br(
        _UniformWrapper(0),
        cfg,
        ismcts_iterations=300,
        eval_games=200,
        seed=2,
        deal_decks=deal_pool,
    )
    after = get_handle_pool_stats()
    for key in ("games", "agents", "snapshots"):
        assert (
            after[key] == before[key]
        ), f"handle leak in {key}: {before[key]} -> {after[key]}"


def test_incremental_key_matches_rebuilt(cfg, deal_pool):
    """The info key carried incrementally down a playout equals the key rebuilt
    from scratch from the full streams, and induces exactly the same equivalence
    classes (tree-node sharing) as the pre-refactor
    ("PR", priv_init, tuple(priv_draw), tuple(pub_path)) tuple. This is the guard
    that the O(1)-extend incremental key stays content-identical to the O(L) rebuild.
    """
    from src.cfr.ismcts_br import (
        _InfoKey,
        _new_deal,
        _responder_priv_init,
        _step_tokens,
    )

    responder = 0
    house_rules = cfg.cambia_rules
    deal_rng = random.Random(2024)
    move_rng = random.Random(99)
    opp = _UniformWrapper(7)

    # (old_tuple, incremental_key) at every responder decision across many random
    # playouts; the two keying schemes must agree class-for-class.
    pairs = []
    for _ in range(80):
        state = _new_deal(house_rules, deal_rng, None, deal_pool)
        try:
            priv_init = _responder_priv_init(state.view(), responder)
            key = _InfoKey.root(priv_init)
            priv_draw = []
            pub_path = []
            turns = 0
            while not state.is_terminal() and turns < 200:
                turns += 1
                acting = state.acting_player()
                if acting == -1:
                    break
                legal_indices = state.legal_indices()
                if not legal_indices:
                    break
                legal = actions_from_indices(legal_indices)
                if acting == responder:
                    # At each responder decision: incremental == rebuilt.
                    rebuilt = _InfoKey.from_streams(priv_init, priv_draw, pub_path)
                    assert key == rebuilt, "incremental key != rebuilt-from-scratch key"
                    assert hash(key) == hash(rebuilt), "equal keys must hash equal"
                    old_tuple = ("PR", priv_init, tuple(priv_draw), tuple(pub_path))
                    pairs.append((old_tuple, key))
                    pos = move_rng.randrange(len(legal))
                else:
                    action = opp.choose_action(state.view(), legal)
                    pos = legal.index(action)
                if not state.apply_index(legal_indices[pos]):
                    break
                pub_entry, draw_token = _step_tokens(
                    state.view(), legal[pos], legal_indices[pos], acting, responder
                )
                key = key.extend_pub(pub_entry)
                pub_path.append(pub_entry)
                if draw_token is not None:
                    key = key.extend_draw(draw_token)
                    priv_draw.append(draw_token)
        finally:
            state.close()

    assert len(pairs) > 20, f"too few responder decisions sampled ({len(pairs)})"
    # Node sharing is exact: two decisions share a tree node under the incremental
    # key iff they shared one under the rebuilt tuple, and equal keys hash equal.
    saw_equal = saw_distinct = False
    for i in range(len(pairs)):
        old_i, key_i = pairs[i]
        for j in range(i + 1, len(pairs)):
            old_j, key_j = pairs[j]
            same_old = old_i == old_j
            same_new = key_i == key_j
            assert (
                same_old == same_new
            ), "info-key equivalence classes diverge from the rebuilt-tuple key"
            if same_new:
                assert hash(key_i) == hash(key_j)
                saw_equal = True
            else:
                saw_distinct = True
    # The sample must exercise both a shared node and a distinct one.
    assert saw_equal and saw_distinct, "sample did not cover both equal and distinct keys"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
