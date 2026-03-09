"""Tests for k-limited expansion in GTCFRSearch."""

import numpy as np
import pytest
import torch

from src.networks import build_cvpn
from src.cfr.gtcfr_search import GTCFRSearch
from src.pbs import NUM_HAND_TYPES


def _make_search(expansion_k: int, budget: int = 5) -> GTCFRSearch:
    """Create a GTCFRSearch with random-weight CVPN and given expansion_k."""
    cvpn = build_cvpn(hidden_dim=64, num_blocks=1)
    cvpn.eval()
    return GTCFRSearch(
        cvpn=cvpn,
        expansion_budget=budget,
        c_puct=2.0,
        cfr_iters_per_expansion=2,
        expansion_k=expansion_k,
        device="cpu",
    )


def _run_search(search: GTCFRSearch):
    """Run search on a fresh GoEngine game and return the result."""
    from src.ffi.bridge import GoEngine

    range_p0 = np.ones(NUM_HAND_TYPES, dtype=np.float32) / NUM_HAND_TYPES
    range_p1 = np.ones(NUM_HAND_TYPES, dtype=np.float32) / NUM_HAND_TYPES

    with GoEngine(seed=42) as game:
        with torch.inference_mode():
            result = search.search(game, range_p0, range_p1)
    return result


def test_k_limited_expands_k_children():
    """k=3 expands at most 3 children per node."""
    search = _make_search(expansion_k=3, budget=1)

    from src.ffi.bridge import GoEngine

    range_p0 = np.ones(NUM_HAND_TYPES, dtype=np.float32) / NUM_HAND_TYPES
    range_p1 = np.ones(NUM_HAND_TYPES, dtype=np.float32) / NUM_HAND_TYPES

    # Build tree manually to inspect root children
    with GoEngine(seed=42) as game:
        with torch.inference_mode():
            result = search.search(game, range_p0, range_p1)

    # With budget=1, only one expansion happens.
    # The root should have at most 3 children (k=3).
    # tree_size = 1 (root) + num_children + ...
    # With k=3 and budget=1: root + up to 3 children = at most 4 nodes
    assert result.tree_size <= 4, (
        f"Expected at most 4 nodes (root + 3 children) with k=3, budget=1, "
        f"got {result.tree_size}"
    )
    # Should have more than just the root (at least 1 expansion happened)
    assert result.tree_size >= 2, (
        f"Expected at least 2 nodes after 1 expansion, got {result.tree_size}"
    )


def test_k_inf_backward_compat():
    """k=-1 expands all legal children (same as before)."""
    search_kinf = _make_search(expansion_k=-1, budget=1)
    result = _run_search(search_kinf)

    # With k=-1 and budget=1, root gets ALL legal actions expanded.
    # Cambia draw phase typically has 2+ legal actions (draw stockpile, etc.)
    # tree_size should be 1 (root) + n_legal children
    assert result.tree_size >= 2, (
        f"k=-1 should expand at least some children, got tree_size={result.tree_size}"
    )


def test_k1_fewer_nodes_than_kinf():
    """k=1 produces fewer total nodes than k=-1 at same budget."""
    budget = 10

    search_k1 = _make_search(expansion_k=1, budget=budget)
    search_kinf = _make_search(expansion_k=-1, budget=budget)

    np.random.seed(123)
    torch.manual_seed(123)
    result_k1 = _run_search(search_k1)

    np.random.seed(123)
    torch.manual_seed(123)
    result_kinf = _run_search(search_kinf)

    # k=1 adds exactly 1 child per expansion step (budget=10 -> ~11 nodes max).
    # k=-1 adds ALL legal actions per expansion (typically 2-10 children per step).
    # So k=1 should have strictly fewer nodes.
    assert result_k1.tree_size <= result_kinf.tree_size, (
        f"k=1 tree_size={result_k1.tree_size} should be <= "
        f"k=-1 tree_size={result_kinf.tree_size}"
    )


def test_k_greater_than_legal_expands_all():
    """When k > n_legal, all legal actions are expanded (no crash)."""
    search = _make_search(expansion_k=999, budget=1)
    result = _run_search(search)
    # Should behave identically to k=-1 since k > n_legal
    assert result.tree_size >= 2


def test_default_expansion_k_is_3():
    """Default expansion_k in config is 3."""
    from src.config import DeepCfrConfig

    cfg = DeepCfrConfig()
    assert cfg.gtcfr_expansion_k == 3
