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
    _children_mask,
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

    assert result.root_values.shape == (
        VALUE_DIM,
    ), f"Got shape {result.root_values.shape}"
    assert result.root_values.dtype == np.float32
    assert np.isfinite(result.root_values).all(), "CFVs contain non-finite values"


# ---------------------------------------------------------------------------
# Test: tree grows with budget
# ---------------------------------------------------------------------------


def test_tree_grows_with_budget(small_cvpn: CVPN):
    """tree_size increases with expansion_budget."""
    r0, r1 = _uniform_ranges()

    with _make_game() as game:
        searcher_small = GTCFRSearch(
            small_cvpn, expansion_budget=1, cfr_iters_per_expansion=1
        )
        result_small = searcher_small.search(game, r0, r1)

    with _make_game() as game:
        searcher_large = GTCFRSearch(
            small_cvpn, expansion_budget=10, cfr_iters_per_expansion=1
        )
        result_large = searcher_large.search(game, r0, r1)

    assert (
        result_large.tree_size >= result_small.tree_size
    ), f"Expected larger tree with larger budget: {result_large.tree_size} >= {result_small.tree_size}"


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
    prior[0] = 0.9  # high prior, unvisited
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
    assert (
        scores[0] > scores[2]
    ), f"Expected scores[0]={scores[0]:.3f} > scores[2]={scores[2]:.3f}"
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
    assert np.allclose(
        cfvs[0], 1.0
    ), f"Player 0 CFVs should all be 1.0, got {cfvs[0][:3]}"
    assert np.allclose(
        cfvs[1], -1.0
    ), f"Player 1 CFVs should all be -1.0, got {cfvs[1][:3]}"


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
    assert isinstance(
        stats["mean"], float
    ), f"'mean' should be float, got {type(stats['mean'])}"

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

    assert (
        after_all == before
    ), f"Handle leak after context exit: before={before}, after={after_all}"


# ---------------------------------------------------------------------------
# Helper: synthetic nodes for traversal / backprop tests
# ---------------------------------------------------------------------------


def _make_node(
    acting_player: int,
    legal_mask: np.ndarray,
    depth: int = 0,
    is_expanded: bool = False,
    leaf_values=None,
    engine_handle=None,
) -> GTCFRNode:
    """Build a non-terminal GTCFRNode with zeroed CFR and PUCT state."""
    return GTCFRNode(
        depth=depth,
        acting_player=acting_player,
        is_terminal=False,
        terminal_values=None,
        legal_mask=legal_mask,
        n_legal=int(legal_mask.sum()),
        children={},
        is_expanded=is_expanded,
        cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cfr_visits=0,
        visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
        total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
        policy_prior=np.zeros(NUM_ACTIONS, dtype=np.float32),
        leaf_values=leaf_values,
        engine_handle=engine_handle,
    )


def _fallback_node(searcher: GTCFRSearch, n_legal: int, child_utils: dict) -> GTCFRNode:
    """Node in the uniform fallback: n_legal actions, len(child_utils) expanded.

    Every cumulative regret is negative, so regret matching falls back to the
    uniform strategy. Children are terminal, so their CFVs are the given
    utilities broadcast across hand types.
    """
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[:n_legal] = True

    node = _make_node(acting_player=0, legal_mask=legal_mask, is_expanded=True)
    node.cumulative_regret[legal_mask] = -1.0

    for a, util in child_utils.items():
        node.children[a] = searcher._make_terminal_node(
            1, None, np.array(util, dtype=np.float32)
        )
    return node


# ---------------------------------------------------------------------------
# Test: uniform fallback does not drop unexpanded-action mass (cambia-716)
# ---------------------------------------------------------------------------


def test_uniform_fallback_mixes_over_expanded_children(small_cvpn: CVPN):
    """The fallback node value is the expanded-children mean, not k/n_legal of it.

    With 10 legal actions and 3 expanded children, a 1/n_legal strategy summed
    over the children alone yields 3/10 of their mean and drives every regret
    delta the same direction.
    """
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, expansion_k=3)
    child_utils = {0: (-1.0, 1.0), 1: (-0.5, 0.5), 2: (-0.2, 0.2)}
    node = _fallback_node(searcher, n_legal=10, child_utils=child_utils)

    r0, r1 = _uniform_ranges()
    before = node.cumulative_regret.copy()
    cfvs = searcher._cfr_traverse(node, np.ones(2, dtype=np.float32), r0, r1)

    expected_p0 = np.mean([u[0] for u in child_utils.values()])
    deflated_p0 = expected_p0 * len(child_utils) / 10.0

    assert np.allclose(
        cfvs[0], expected_p0
    ), f"Expected expanded-children mean {expected_p0}, got {cfvs[0][0]}"
    assert not np.isclose(
        cfvs[0][0], deflated_p0
    ), f"Node value still deflated to k/n_legal ({deflated_p0})"

    # Regret deltas are centered on the node value, so they cannot all be negative.
    deltas = [float(node.cumulative_regret[a] - before[a]) for a in child_utils]
    assert max(deltas) > 0.0, f"All regret deltas non-positive: {deltas}"
    assert min(deltas) < 0.0, f"All regret deltas non-negative: {deltas}"
    assert sum(deltas) == pytest.approx(0.0, abs=1e-5)


def test_uniform_fallback_node_escapes_fallback(small_cvpn: CVPN):
    """Repeated iterations lift the best action's regret above zero.

    Under a k/n_legal node value every delta stays negative and the node is
    pinned to uniform-over-all-legal forever.
    """
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, expansion_k=3)
    child_utils = {0: (-1.0, 1.0), 1: (-0.5, 0.5), 2: (-0.2, 0.2)}
    node = _fallback_node(searcher, n_legal=10, child_utils=child_utils)

    r0, r1 = _uniform_ranges()
    for _ in range(5):
        searcher._cfr_traverse(node, np.ones(2, dtype=np.float32), r0, r1)

    assert (
        node.cumulative_regret > 0.0
    ).any(), f"Node still frozen in fallback: {node.cumulative_regret[:3]}"
    # Action 2 has the best value for the acting player and should hold the mass.
    strategy = node.current_strategy()
    assert strategy[2] > strategy[0]


# ---------------------------------------------------------------------------
# Test: PUCT backprop uses the ancestor's perspective (cambia-717)
# ---------------------------------------------------------------------------


class _FakeTerminalEngine:
    """Stand-in child engine reporting a terminal state with fixed utilities."""

    def __init__(self, util):
        self._util = np.asarray(util, dtype=np.float32)

    def is_terminal(self) -> bool:
        return True

    def get_utility(self) -> np.ndarray:
        return self._util.copy()

    def close(self) -> None:
        pass


def _constant_leaf_values(v0: float, v1: float) -> np.ndarray:
    """(2, NUM_HAND_TYPES) leaf values constant across hand types."""
    lv = np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)
    lv[0, :] = v0
    lv[1, :] = v1
    return lv


def test_backprop_projects_value_into_ancestor_perspective(small_cvpn: CVPN, monkeypatch):
    """A parent books the expanded node's value for the parent's own actor.

    The root acts as player 0 and both children act as player 1. Leaf values are
    deliberately not zero-sum, so indexing by the ancestor's actor is
    distinguishable from negating the expanded node's own value.
    """
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, c_puct=2.0, expansion_k=2)
    r0, r1 = _uniform_ranges()

    action_a, action_b = 0, 1
    root_legal = np.zeros(NUM_ACTIONS, dtype=bool)
    root_legal[[action_a, action_b]] = True
    root = _make_node(acting_player=0, legal_mask=root_legal, is_expanded=True)
    root.policy_prior[[action_a, action_b]] = 0.5  # equal exploration terms

    child_legal = np.zeros(NUM_ACTIONS, dtype=bool)
    child_legal[2] = True
    child_a = _make_node(
        acting_player=1,
        legal_mask=child_legal,
        depth=1,
        leaf_values=_constant_leaf_values(0.8, -0.5),
        engine_handle=_FakeTerminalEngine([0.0, 0.0]),
    )
    child_b = _make_node(
        acting_player=1,
        legal_mask=child_legal,
        depth=1,
        leaf_values=_constant_leaf_values(-0.6, 0.7),
        engine_handle=_FakeTerminalEngine([0.0, 0.0]),
    )
    root.children = {action_a: child_a, action_b: child_b}

    monkeypatch.setattr(
        searcher,
        "_make_child_engine",
        lambda parent, action: _FakeTerminalEngine([0.0, 0.0]),
    )
    selections = iter([action_a, action_b])
    monkeypatch.setattr(
        searcher, "_select_action", lambda node, support_mask=None: next(selections)
    )

    searcher._expand_once(root, None, r0, r1)
    searcher._expand_once(root, None, r0, r1)

    assert root.visit_counts[action_a] == 1
    assert root.visit_counts[action_b] == 1

    # Player 0's values (0.8 / -0.6), not player 1's (-0.5 / 0.7) and not their
    # negations (0.5 / -0.7).
    assert root.total_action_value[action_a] == pytest.approx(0.8, abs=1e-5)
    assert root.total_action_value[action_b] == pytest.approx(-0.6, abs=1e-5)

    # Equal priors and equal visit counts, so Q decides the ranking.
    scores = searcher._puct_scores(root)
    assert scores[action_a] > scores[action_b], (
        f"PUCT ranks against the root actor's preference: "
        f"{scores[action_a]:.3f} vs {scores[action_b]:.3f}"
    )


# ---------------------------------------------------------------------------
# Test: selection is masked to the expanded children (cambia-1869)
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Depth-limited stand-in GoEngine exposing only what the search reads.

    Engines past max_depth report terminal; the rest report a fixed legal set
    and enough public features for _build_pbs.
    """

    def __init__(self, n_legal: int, acting_player: int, depth: int, max_depth: int):
        self.n_legal = n_legal
        self._acting = acting_player
        self.depth = depth
        self.max_depth = max_depth

    def is_terminal(self) -> bool:
        return self.depth >= self.max_depth

    def get_utility(self) -> np.ndarray:
        return np.array([1.0, -1.0], dtype=np.float32)

    def acting_player(self) -> int:
        return self._acting

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[: self.n_legal] = True
        return mask

    def decision_ctx(self) -> int:
        return 0

    def turn_number(self) -> int:
        return self.depth

    def discard_top(self) -> int:
        return 0

    def stock_len(self) -> int:
        return max(1, 40 - self.depth)

    def close(self) -> None:
        pass

    def child(self, action: int) -> "_FakeEngine":
        return _FakeEngine(
            n_legal=self.n_legal,
            acting_player=1 - self._acting,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )


def test_select_action_restricted_to_expanded_children(small_cvpn: CVPN):
    """At an expanded node, selection only draws actions that hold a child.

    Drawing any other action ends the walk-down at an already-expanded node and
    the whole expansion step grows nothing.
    """
    np.random.seed(0)
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, expansion_k=3)

    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[:10] = True
    node = _make_node(acting_player=0, legal_mask=legal_mask, is_expanded=True)
    node.policy_prior[:10] = 0.1

    for a in (0, 1, 2):
        node.children[a] = searcher._make_terminal_node(
            1, None, np.array([0.0, 0.0], dtype=np.float32)
        )

    draws = {searcher._select_action(node, _children_mask(node)) for _ in range(200)}
    assert draws <= set(node.children), f"Selected childless actions: {draws}"
    assert len(draws) > 1, "Selection collapsed onto a single child"


def test_expansion_steps_never_waste_the_budget(small_cvpn: CVPN, monkeypatch):
    """Every expansion step grows the tree when unexpanded nodes remain.

    n_legal = 10 against expansion_k = 3 leaves 7 childless actions at every
    expanded node; before the mask those carried selection mass and the step
    returned without adding a node.
    """
    np.random.seed(0)
    steps = 12
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, expansion_k=3)
    r0, r1 = _uniform_ranges()

    # max_depth well past the reachable depth, so no step ends on a terminal.
    root_engine = _FakeEngine(n_legal=10, acting_player=0, depth=0, max_depth=20)
    root = _make_node(
        acting_player=0,
        legal_mask=root_engine.legal_actions_mask(),
        leaf_values=_constant_leaf_values(0.0, 0.0),
        engine_handle=root_engine,
    )
    root.policy_prior[:10] = 0.1

    monkeypatch.setattr(
        searcher, "_make_child_engine", lambda parent, action: parent.child(action)
    )

    added = [searcher._expand_once(root, root_engine, r0, r1) for _ in range(steps)]

    assert min(added) > 0, f"Expansion steps grew nothing: {added}"
    # Each step expands one unexpanded node into expansion_k children.
    assert added == [3] * steps, f"Unexpected growth per step: {added}"
    assert searcher._count_nodes(root) == 1 + sum(added)


# ---------------------------------------------------------------------------
# Test: progressive widening (cambia-1870)
# ---------------------------------------------------------------------------


def _widening_fixture(searcher: GTCFRSearch):
    """Root over 10 legal actions with a strictly decreasing PUCT prior.

    The prior order fixes both the initial top-k expansion and the order in
    which widening opens the rest, so child insertion order is deterministic.
    """
    engine = _FakeEngine(n_legal=10, acting_player=0, depth=0, max_depth=20)
    root = _make_node(
        acting_player=0,
        legal_mask=engine.legal_actions_mask(),
        leaf_values=_constant_leaf_values(0.0, 0.0),
        engine_handle=engine,
    )
    root.policy_prior[:10] = np.linspace(0.19, 0.01, 10).astype(np.float32)
    return root, engine


def test_widening_off_keeps_the_fixed_expansion_k(small_cvpn: CVPN, monkeypatch):
    """With widening off a node never grows past expansion_k children."""
    np.random.seed(0)
    searcher = GTCFRSearch(small_cvpn, expansion_budget=1, expansion_k=3)
    r0, r1 = _uniform_ranges()
    root, engine = _widening_fixture(searcher)

    monkeypatch.setattr(
        searcher, "_make_child_engine", lambda parent, action: parent.child(action)
    )
    for _ in range(20):
        searcher._expand_once(root, engine, r0, r1)

    assert len(root.children) == 3, f"Child count drifted: {sorted(root.children)}"
    assert sorted(root.children) == [0, 1, 2]


def test_widening_opens_every_action_in_prior_order(small_cvpn: CVPN, monkeypatch):
    """With widening on, all 10 actions open, in descending PUCT prior order.

    An unopened action has no visits, so its PUCT score is its prior term and
    the widening order matches the ranking the initial expansion uses.
    """
    np.random.seed(0)
    searcher = GTCFRSearch(
        small_cvpn,
        expansion_budget=1,
        expansion_k=3,
        widening_enabled=True,
        widening_c=3.0,
        widening_alpha=0.5,
    )
    r0, r1 = _uniform_ranges()
    root, engine = _widening_fixture(searcher)

    monkeypatch.setattr(
        searcher, "_make_child_engine", lambda parent, action: parent.child(action)
    )
    for _ in range(20):
        searcher._expand_once(root, engine, r0, r1)

    assert list(root.children.keys()) == list(
        range(10)
    ), f"Unexpected open order: {list(root.children.keys())}"


def test_widening_schedule_bounds(small_cvpn: CVPN):
    """_allowed_width is floored at expansion_k and capped at n_legal."""
    searcher = GTCFRSearch(
        small_cvpn,
        expansion_budget=1,
        expansion_k=3,
        widening_enabled=True,
        widening_c=1.0,
        widening_alpha=0.5,
    )
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[:10] = True
    node = _make_node(acting_player=0, legal_mask=legal_mask, is_expanded=True)

    assert searcher._allowed_width(node) == 3  # ceil(sqrt(1)) floored at k
    node.visit_counts[0] = 16
    assert searcher._allowed_width(node) == 4  # ceil(sqrt(16))
    node.visit_counts[0] = 10_000
    assert searcher._allowed_width(node) == 10  # capped at n_legal


def test_widening_on_search_returns_valid_policy(small_cvpn: CVPN):
    """search() with widening on keeps the policy contract its consumers read."""
    searcher = GTCFRSearch(
        small_cvpn,
        expansion_budget=5,
        cfr_iters_per_expansion=3,
        widening_enabled=True,
        widening_c=2.0,
        widening_alpha=0.5,
    )
    r0, r1 = _uniform_ranges()
    with _make_game() as game:
        result = searcher.search(game, r0, r1)

    assert result.policy.shape == (NUM_ACTIONS,)
    assert abs(result.policy.sum() - 1.0) < 1e-4
    assert result.root_values.shape == (VALUE_DIM,)
    assert np.isfinite(result.root_values).all()


# ---------------------------------------------------------------------------
# Test: widening config keys reach the searcher (cambia-1870)
# ---------------------------------------------------------------------------


def test_config_widening_keys_reach_the_searcher(small_cvpn: CVPN, monkeypatch):
    """A DeepCfrConfig with widening on builds a GTCFRSearch with widening on.

    gtcfr_self_play_episode builds the searcher before touching the engine, so
    a recording stand-in that stops there captures the settings it passes.
    """
    from src.cfr import gtcfr_worker
    from src.config import DeepCfrConfig

    config = DeepCfrConfig(
        gtcfr_widening_enabled=True,
        gtcfr_widening_c=2.5,
        gtcfr_widening_alpha=0.75,
    )

    class _StopAfterConstruction(Exception):
        pass

    captured: dict = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _StopAfterConstruction

    monkeypatch.setattr(gtcfr_worker, "GTCFRSearch", _recorder)
    with pytest.raises(_StopAfterConstruction):
        gtcfr_worker.gtcfr_self_play_episode(None, small_cvpn, config)

    assert captured["widening_enabled"] is True
    assert captured["widening_c"] == 2.5
    assert captured["widening_alpha"] == 0.75

    # The captured settings build a searcher that actually widens.
    searcher = GTCFRSearch(**captured)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[:10] = True
    node = _make_node(acting_player=0, legal_mask=legal_mask, is_expanded=True)
    node.visit_counts[0] = 16
    assert searcher._allowed_width(node) > config.gtcfr_expansion_k
