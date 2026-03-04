"""
tests/test_gtcfr_search.py

Unit tests for the GT-CFR growing-tree search engine.

Uses a small randomly-initialized CVPN for speed.
All tests use real GoEngine via FFI (requires libcambia.so).
"""

import warnings

import numpy as np
import pytest
import torch

from src.cfr.gtcfr_search import (
    GTCFRNode,
    GTCFRSearch,
    SearchResult,
    NUM_HAND_TYPES,
    VALUE_DIM,
)
from src.encoding import NUM_ACTIONS
from src.networks import CVPN
from src.pbs import uniform_range


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cvpn() -> CVPN:
    """Randomly-initialized CVPN with small hidden_dim for fast tests."""
    cvpn = CVPN(
        input_dim=956,
        hidden_dim=64,
        num_blocks=1,
        value_dim=936,
        policy_dim=146,
        validate_inputs=False,
    )
    cvpn.eval()
    return cvpn


@pytest.fixture
def searcher(small_cvpn: CVPN) -> GTCFRSearch:
    return GTCFRSearch(
        cvpn=small_cvpn,
        expansion_budget=5,
        c_puct=2.0,
        cfr_iters_per_expansion=3,
        device="cpu",
    )


def _make_game():
    """Create a GoEngine with Python-compatible default rules."""
    from src.ffi.bridge import GoEngine
    from src.config import CambiaRulesConfig

    rules = CambiaRulesConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GoEngine(seed=42, house_rules=rules)


def _uniform_ranges():
    return uniform_range(), uniform_range()


# ---------------------------------------------------------------------------
# Helper: pool stats
# ---------------------------------------------------------------------------


def _game_pool_count() -> int:
    """Return number of currently allocated game handles."""
    from src.ffi.bridge import _ffi, _get_lib

    lib = _get_lib()
    games_buf = _ffi.new("int32_t[1]")
    agents_buf = _ffi.new("int32_t[1]")
    snaps_buf = _ffi.new("int32_t[1]")
    lib.cambia_handle_pool_stats(games_buf, agents_buf, snaps_buf)
    return int(games_buf[0])


# ---------------------------------------------------------------------------
# Test: search() returns valid policy
# ---------------------------------------------------------------------------


def test_search_returns_valid_policy(searcher: GTCFRSearch):
    """search() returns (146,) policy summing to ~1 over legal actions."""
    r0, r1 = _uniform_ranges()
    with _make_game() as game:
        result = searcher.search(game, r0, r1)

    assert isinstance(result, SearchResult)
    assert result.policy.shape == (NUM_ACTIONS,)
    assert result.policy.dtype == np.float32

    # Policy sums to ~1
    assert abs(result.policy.sum() - 1.0) < 1e-4, f"Policy sum = {result.policy.sum()}"

    # No negative probabilities
    assert (result.policy >= -1e-6).all()


# ---------------------------------------------------------------------------
# Test: search() returns valid CFVs
# ---------------------------------------------------------------------------


def test_search_returns_valid_cfvs(searcher: GTCFRSearch):
    """search() returns (936,) CFVs with finite values."""
    r0, r1 = _uniform_ranges()
    with _make_game() as game:
        result = searcher.search(game, r0, r1)

    assert result.root_values.shape == (VALUE_DIM,), f"Got shape {result.root_values.shape}"
    assert result.root_values.dtype == np.float32
    assert np.isfinite(result.root_values).all(), "CFVs contain non-finite values"


# ---------------------------------------------------------------------------
# Test: tree grows with budget
# ---------------------------------------------------------------------------


def test_tree_grows_with_budget(small_cvpn: CVPN):
    """tree_size increases with expansion_budget."""
    r0, r1 = _uniform_ranges()

    with _make_game() as game:
        searcher_small = GTCFRSearch(small_cvpn, expansion_budget=1, cfr_iters_per_expansion=1)
        result_small = searcher_small.search(game, r0, r1)

    with _make_game() as game:
        searcher_large = GTCFRSearch(small_cvpn, expansion_budget=10, cfr_iters_per_expansion=1)
        result_large = searcher_large.search(game, r0, r1)

    assert result_large.tree_size >= result_small.tree_size, (
        f"Expected larger tree with larger budget: {result_large.tree_size} >= {result_small.tree_size}"
    )


# ---------------------------------------------------------------------------
# Test: PUCT scores prefer unvisited high-prior actions
# ---------------------------------------------------------------------------


def test_puct_scores_prefer_unvisited(small_cvpn: CVPN):
    """Unvisited actions with high prior get high PUCT scores."""
    searcher = GTCFRSearch(small_cvpn, expansion_budget=5, c_puct=2.0)

    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[0] = True
    legal_mask[1] = True
    legal_mask[2] = True

    prior = np.zeros(NUM_ACTIONS, dtype=np.float32)
    prior[0] = 0.9   # high prior, unvisited
    prior[1] = 0.05  # low prior, unvisited
    prior[2] = 0.05  # low prior, visited many times

    node = GTCFRNode(
        depth=0,
        acting_player=0,
        is_terminal=False,
        terminal_values=None,
        legal_mask=legal_mask,
        n_legal=3,
        children={},
        is_expanded=False,
        cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cfr_visits=0,
        visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
        total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
        policy_prior=prior,
        leaf_values=None,
        engine_handle=None,
    )
    # Action 2 has been visited many times with low value
    node.visit_counts[2] = 100
    node.total_action_value[2] = -50.0

    scores = searcher._puct_scores(node)

    # Action 0 (high prior, unvisited) should score highest
    assert scores[0] > scores[2], (
        f"Expected scores[0]={scores[0]:.3f} > scores[2]={scores[2]:.3f}"
    )
    # Illegal actions should have very low scores
    assert scores[3] < -100, f"Illegal action score should be -1e9, got {scores[3]}"


# ---------------------------------------------------------------------------
# Test: CFR traverse at terminal
# ---------------------------------------------------------------------------


def test_cfr_traverse_terminal(small_cvpn: CVPN):
    """_cfr_traverse at a terminal node returns terminal values broadcast across hand types."""
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1)
    util = np.array([1.0, -1.0], dtype=np.float32)

    terminal_node = GTCFRNode(
        depth=0,
        acting_player=-1,
        is_terminal=True,
        terminal_values=util,
        legal_mask=np.zeros(NUM_ACTIONS, dtype=bool),
        n_legal=0,
        children={},
        is_expanded=True,
        cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cfr_visits=0,
        visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
        total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
        policy_prior=np.zeros(NUM_ACTIONS, dtype=np.float32),
        leaf_values=None,
        engine_handle=None,
    )

    r0, r1 = _uniform_ranges()
    reach = np.ones(2, dtype=np.float32)
    cfvs = searcher._cfr_traverse(terminal_node, reach, r0, r1)

    assert cfvs.shape == (2, NUM_HAND_TYPES)
    # All hand types should have the same value (broadcast)
    assert np.allclose(cfvs[0], 1.0), f"Player 0 CFVs should all be 1.0, got {cfvs[0][:3]}"
    assert np.allclose(cfvs[1], -1.0), f"Player 1 CFVs should all be -1.0, got {cfvs[1][:3]}"


# ---------------------------------------------------------------------------
# Test: CFR traverse at unexpanded leaf
# ---------------------------------------------------------------------------


def test_cfr_traverse_leaf(small_cvpn: CVPN):
    """_cfr_traverse at an unexpanded leaf returns stored CVPN leaf values."""
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1)

    # Synthetic CVPN leaf values
    leaf_values = np.random.randn(2, NUM_HAND_TYPES).astype(np.float32)

    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[0] = True

    leaf_node = GTCFRNode(
        depth=1,
        acting_player=0,
        is_terminal=False,
        terminal_values=None,
        legal_mask=legal_mask,
        n_legal=1,
        children={},
        is_expanded=False,  # not expanded → leaf
        cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cfr_visits=0,
        visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
        total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
        policy_prior=np.zeros(NUM_ACTIONS, dtype=np.float32),
        leaf_values=leaf_values,
        engine_handle=None,
    )

    r0, r1 = _uniform_ranges()
    reach = np.ones(2, dtype=np.float32)
    cfvs = searcher._cfr_traverse(leaf_node, reach, r0, r1)

    assert cfvs.shape == (2, NUM_HAND_TYPES)
    assert np.allclose(cfvs, leaf_values), "Leaf node should return stored CVPN values"


# ---------------------------------------------------------------------------
# Test: depth_stats fields
# ---------------------------------------------------------------------------


def test_search_result_depth_stats(searcher: GTCFRSearch):
    """depth_stats contains min, max, and mean fields with correct types."""
    r0, r1 = _uniform_ranges()
    with _make_game() as game:
        result = searcher.search(game, r0, r1)

    stats = result.depth_stats
    assert "min" in stats, "depth_stats missing 'min'"
    assert "max" in stats, "depth_stats missing 'max'"
    assert "mean" in stats, "depth_stats missing 'mean'"

    assert isinstance(stats["min"], int), f"'min' should be int, got {type(stats['min'])}"
    assert isinstance(stats["max"], int), f"'max' should be int, got {type(stats['max'])}"
    assert isinstance(stats["mean"], float), f"'mean' should be float, got {type(stats['mean'])}"

    assert stats["min"] >= 0
    assert stats["max"] >= stats["min"]
    assert stats["mean"] >= stats["min"]


# ---------------------------------------------------------------------------
# Test: engine handles freed after search
# ---------------------------------------------------------------------------


def test_engine_handles_freed(small_cvpn: CVPN):
    """After search(), all GoEngine handles created internally are freed."""
    searcher = GTCFRSearch(small_cvpn, expansion_budget=5, cfr_iters_per_expansion=2)
    r0, r1 = _uniform_ranges()

    before = _game_pool_count()

    with _make_game() as game:
        # The test game is "inside" the with block, so not freed yet
        before_with_test_game = _game_pool_count()
        result = searcher.search(game, r0, r1)
        after_search = _game_pool_count()

    after_all = _game_pool_count()

    # After search() returns, all internally-created handles must be freed.
    # Pool count should return to the level before search was called
    # (accounting for the test game itself, which is freed by the with-block).
    assert after_search == before_with_test_game, (
        f"Handle leak detected: {after_search - before_with_test_game} handles "
        f"leaked during search (before_with_test_game={before_with_test_game}, "
        f"after_search={after_search})"
    )

    assert after_all == before, (
        f"Handle leak after context exit: before={before}, after={after_all}"
    )
