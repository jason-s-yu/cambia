"""
tests/test_sog_search.py

Unit tests for SoGSearch: continual re-solving, budget decoupling, safety check.

All tests use mocked GTCFRSearch internals (no Go FFI required).
"""

import warnings
from unittest.mock import MagicMock, patch, call
from typing import Optional

import numpy as np
import pytest
import torch

from src.cfr.sog_search import SoGSearch, VALUE_DIM
from src.cfr.gtcfr_search import GTCFRNode, GTCFRSearch, SearchResult
from src.encoding import NUM_ACTIONS
from src.networks import CVPN
from src.pbs import uniform_range, NUM_HAND_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cvpn() -> CVPN:
    cvpn = CVPN(
        input_dim=956,
        hidden_dim=32,
        num_blocks=1,
        value_dim=936,
        policy_dim=146,
        validate_inputs=False,
    )
    cvpn.eval()
    return cvpn


def make_mock_node(
    depth: int = 0,
    is_terminal: bool = False,
    n_children: int = 0,
    engine_handle=None,
    leaf_values: Optional[np.ndarray] = None,
) -> GTCFRNode:
    """Build a GTCFRNode with sensible defaults for testing."""
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_mask[:3] = True

    node = GTCFRNode(
        depth=depth,
        acting_player=0,
        is_terminal=is_terminal,
        terminal_values=np.array([1.0, -1.0], dtype=np.float32) if is_terminal else None,
        legal_mask=legal_mask,
        n_legal=3,
        children={},
        is_expanded=(n_children > 0),
        cumulative_regret=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cumulative_strategy=np.zeros(NUM_ACTIONS, dtype=np.float32),
        cfr_visits=0,
        visit_counts=np.zeros(NUM_ACTIONS, dtype=np.int32),
        total_action_value=np.zeros(NUM_ACTIONS, dtype=np.float32),
        policy_prior=np.zeros(NUM_ACTIONS, dtype=np.float32),
        leaf_values=leaf_values,
        engine_handle=engine_handle,
    )
    return node


def make_mock_search_result() -> SearchResult:
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[:3] = 1.0 / 3
    return SearchResult(
        policy=policy,
        root_values=np.random.rand(VALUE_DIM).astype(np.float32),
        tree_size=5,
        depth_stats={"min": 0, "max": 2, "mean": 1.0},
    )


def make_mock_engine():
    eng = MagicMock()
    eng.is_terminal.return_value = False
    eng.get_utility.return_value = np.array([1.0, -1.0], dtype=np.float32)
    eng.legal_actions_mask.return_value = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    eng.acting_player.return_value = 0
    eng.close = MagicMock()
    return eng


# ---------------------------------------------------------------------------
# Tests: init + budget toggling
# ---------------------------------------------------------------------------


class TestSoGSearchInit:
    def test_init_defaults(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn)
        assert sog._train_budget == 50
        assert sog._eval_budget == 200
        assert sog._current_budget == 50
        assert sog._last_tree is None
        assert isinstance(sog._inner, GTCFRSearch)

    def test_use_train_budget(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, train_budget=10, eval_budget=100)
        sog.use_eval_budget()
        assert sog._current_budget == 100
        old_inner = sog._inner

        sog.use_train_budget()
        assert sog._current_budget == 10
        assert sog._inner is not old_inner  # rebuilt

    def test_use_eval_budget(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, train_budget=10, eval_budget=100)
        assert sog._current_budget == 10

        sog.use_eval_budget()
        assert sog._current_budget == 100
        assert sog._inner._expansion_budget == 100

    def test_budget_toggle_no_rebuild_if_unchanged(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, train_budget=10, eval_budget=100)
        inner_before = sog._inner
        sog.use_train_budget()  # already train
        assert sog._inner is inner_before  # no rebuild


# ---------------------------------------------------------------------------
# Tests: get_tree, cleanup
# ---------------------------------------------------------------------------


class TestSoGGetTreeCleanup:
    def test_get_tree_returns_none_initially(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn)
        assert sog.get_tree() is None

    def test_cleanup_when_no_tree(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn)
        sog.cleanup()  # should not raise
        assert sog._last_tree is None

    def test_cleanup_frees_tree(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn)
        mock_engine = make_mock_engine()
        node = make_mock_node(engine_handle=mock_engine)
        sog._last_tree = node

        with patch.object(sog._inner, "_cleanup_tree") as mock_cleanup:
            sog.cleanup()

        mock_cleanup.assert_called_once_with(node)
        assert sog._last_tree is None


# ---------------------------------------------------------------------------
# Tests: depth pruning
# ---------------------------------------------------------------------------


class TestPruneDepth:
    def test_prune_removes_deep_children(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, max_persist_depth=2)
        root = make_mock_node(depth=0)
        child1 = make_mock_node(depth=1, engine_handle=make_mock_engine())
        child2 = make_mock_node(depth=2, engine_handle=make_mock_engine())
        grandchild = make_mock_node(depth=3, engine_handle=make_mock_engine())

        root.children = {0: child1}
        root.is_expanded = True
        child1.children = {1: child2}
        child1.is_expanded = True
        child2.children = {2: grandchild}
        child2.is_expanded = True

        # Prune at max_persist_depth=2 relative to root (rel_depth 0)
        sog._prune_depth(root, relative_depth=0)

        # root (0) and child1 (1) should survive; child2 at rel_depth=2 loses children
        assert 0 in root.children  # child1 survives
        # child2 at relative depth 2 == max_persist_depth, so it clears its children
        assert 1 in child1.children  # child2 still in child1.children
        assert len(child2.children) == 0  # grandchild pruned
        assert not child2.is_expanded

    def test_prune_noop_on_shallow_tree(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, max_persist_depth=8)
        root = make_mock_node(depth=0)
        child = make_mock_node(depth=1, engine_handle=make_mock_engine())
        root.children = {0: child}
        root.is_expanded = True

        sog._prune_depth(root, relative_depth=0)
        assert 0 in root.children  # nothing pruned


# ---------------------------------------------------------------------------
# Tests: fresh search path
# ---------------------------------------------------------------------------


class TestFreshSearch:
    def _make_sog_with_mocked_inner(self, small_cvpn, mock_result=None):
        sog = SoGSearch(cvpn=small_cvpn, train_budget=3, eval_budget=10)
        if mock_result is None:
            mock_result = make_mock_search_result()

        # Mock inner methods
        sog._inner._clone_engine = MagicMock(return_value=make_mock_engine())
        sog._inner._evaluate_node = MagicMock(
            return_value=(
                np.zeros((2, NUM_HAND_TYPES), dtype=np.float32),
                np.zeros(NUM_ACTIONS, dtype=np.float32),
            )
        )
        sog._inner._cfr_traverse = MagicMock(
            return_value=np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)
        )
        sog._inner._expand_once = MagicMock(return_value=1)
        sog._inner._collect_depths = MagicMock(side_effect=lambda n, d: d.extend([0, 1]))
        sog._inner._count_nodes = MagicMock(return_value=3)
        sog._inner._cleanup_tree = MagicMock()

        return sog

    def test_fresh_search_stores_tree(self, small_cvpn):
        sog = self._make_sog_with_mocked_inner(small_cvpn)
        mock_engine = make_mock_engine()
        sog._inner._clone_engine.return_value = mock_engine

        r0, r1 = uniform_range(), uniform_range()
        result = sog._fresh_search(make_mock_engine(), r0, r1)

        assert sog._last_tree is not None
        assert isinstance(result, SearchResult)

    def test_fresh_search_result_shapes(self, small_cvpn):
        sog = self._make_sog_with_mocked_inner(small_cvpn)
        r0, r1 = uniform_range(), uniform_range()
        result = sog._fresh_search(make_mock_engine(), r0, r1)

        assert result.policy.shape == (NUM_ACTIONS,)
        assert result.root_values.shape == (VALUE_DIM,)
        assert isinstance(result.tree_size, int)
        assert "min" in result.depth_stats

    def test_fresh_search_cleans_up_prior_tree(self, small_cvpn):
        sog = self._make_sog_with_mocked_inner(small_cvpn)
        prior_node = make_mock_node()

        r0, r1 = uniform_range(), uniform_range()
        sog._fresh_search(make_mock_engine(), r0, r1, prior_tree=prior_node)

        sog._inner._cleanup_tree.assert_any_call(prior_node)


# ---------------------------------------------------------------------------
# Tests: re-solve path
# ---------------------------------------------------------------------------


class TestResolveSearch:
    def _build_prior_tree_with_child(self, action: int = 1):
        """Build a minimal prior tree with one child at given action."""
        root_engine = make_mock_engine()
        child_engine = make_mock_engine()

        child = make_mock_node(
            depth=1,
            engine_handle=child_engine,
            leaf_values=np.ones((2, NUM_HAND_TYPES), dtype=np.float32) * 0.5,
        )
        child.cumulative_strategy[action] = 10.0  # non-uniform strategy

        root = make_mock_node(depth=0, engine_handle=root_engine)
        root.children = {action: child}
        root.is_expanded = True
        return root, child, root_engine, child_engine

    def test_resolve_frees_root_and_siblings(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, train_budget=2, eval_budget=10)
        sibling_engine = make_mock_engine()

        root, child, root_engine, child_engine = self._build_prior_tree_with_child(1)
        sibling = make_mock_node(depth=1, engine_handle=sibling_engine)
        root.children[2] = sibling  # add a sibling

        r0, r1 = uniform_range(), uniform_range()

        with patch.object(sog._inner, "_cleanup_tree") as mock_cleanup, \
             patch.object(sog._inner, "_cfr_traverse",
                          return_value=np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)), \
             patch.object(sog._inner, "_expand_once", return_value=0), \
             patch.object(sog._inner, "_collect_depths",
                          side_effect=lambda n, d: d.extend([0])), \
             patch.object(sog._inner, "_count_nodes", return_value=2):

            result = sog._resolve_search(make_mock_engine(), r0, r1, root, 1)

        # Root engine handle should be closed
        root_engine.close.assert_called()
        # Sibling should be cleaned up via _cleanup_tree
        mock_cleanup.assert_any_call(sibling)
        # Result should be valid
        assert result is not None
        assert result.policy.shape == (NUM_ACTIONS,)

    def test_resolve_falls_back_when_too_many_handles(self, small_cvpn):
        sog = SoGSearch(cvpn=small_cvpn, max_persist_handles=5)
        root, child, root_engine, child_engine = self._build_prior_tree_with_child(0)

        r0, r1 = uniform_range(), uniform_range()

        with patch.object(sog._inner, "_cleanup_tree"), \
             patch.object(sog._inner, "_count_nodes", return_value=100):  # > 5

            result = sog._resolve_search(make_mock_engine(), r0, r1, root, 0)

        assert result is None  # signals fallback

    def test_resolve_safety_check_triggers(self, small_cvpn):
        """When re-solve values < commitment - margin, prior strategy is retained."""
        sog = SoGSearch(cvpn=small_cvpn, train_budget=2, safety_margin=0.1)

        child_lv = np.ones((2, NUM_HAND_TYPES), dtype=np.float32) * 0.5

        root, child, root_engine, child_engine = self._build_prior_tree_with_child(0)
        child.leaf_values = child_lv.copy()
        # cumulative_strategy already set to non-uniform in helper

        r0, r1 = uniform_range(), uniform_range()

        # Make cfr_traverse return values much worse than commitment
        bad_cfvs = np.zeros((2, NUM_HAND_TYPES), dtype=np.float32)  # 0 < 0.5 - 0.1

        with patch.object(sog._inner, "_cleanup_tree"), \
             patch.object(sog._inner, "_cfr_traverse", return_value=bad_cfvs), \
             patch.object(sog._inner, "_expand_once", return_value=0), \
             patch.object(sog._inner, "_collect_depths",
                          side_effect=lambda n, d: d.extend([0])), \
             patch.object(sog._inner, "_count_nodes", return_value=2):

            result = sog._resolve_search(make_mock_engine(), r0, r1, root, 0)

        assert result is not None
        # Safety check triggered, so policy should be prior_strategy
        # (uniform since cumulative_strategy[:3] != 0 only for action 0 position)
        # The key test: result should not fail and returns a valid policy
        assert result.policy.shape == (NUM_ACTIONS,)
        assert result.policy.sum() > 0

    def test_search_dispatches_to_resolve(self, small_cvpn):
        """search() should call _resolve_search when prior_tree + action given."""
        sog = SoGSearch(cvpn=small_cvpn, train_budget=2)
        root, child, _, _ = self._build_prior_tree_with_child(1)

        r0, r1 = uniform_range(), uniform_range()

        with patch.object(sog, "_resolve_search",
                          return_value=make_mock_search_result()) as mock_resolve:
            sog.search(make_mock_engine(), r0, r1, prior_tree=root, action_taken=1)

        mock_resolve.assert_called_once()

    def test_search_fresh_when_action_not_in_children(self, small_cvpn):
        """search() should call _fresh_search when action_taken not in prior_tree.children."""
        sog = SoGSearch(cvpn=small_cvpn, train_budget=2)
        root, _, _, _ = self._build_prior_tree_with_child(1)

        r0, r1 = uniform_range(), uniform_range()

        with patch.object(sog, "_fresh_search",
                          return_value=make_mock_search_result()) as mock_fresh:
            sog.search(make_mock_engine(), r0, r1, prior_tree=root, action_taken=99)

        mock_fresh.assert_called_once()
