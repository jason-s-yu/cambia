"""
tests/test_subgame_bridge.py

Integration tests for SubgameSolver — verifies the Python cffi bridge wrapping
the Go subgame solver (cambia_subgame_* exports in libcambia.so).
"""

import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.ffi.bridge import GoEngine, SubgameSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game(seed: int = 42) -> GoEngine:
    """Return a fresh GoEngine using default (Go) house rules."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return GoEngine(seed=seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubgameSolverBuild:
    """Build a solver from an initial game state."""

    def test_leaf_count_positive(self):
        """After building from initial state, leaf_count must be > 0."""
        with _make_game() as g:
            with SubgameSolver(g, max_depth=2) as solver:
                assert solver.leaf_count > 0

    def test_leaf_count_increases_with_depth(self):
        """Deeper search should yield at least as many leaves."""
        with _make_game(seed=7) as g:
            with SubgameSolver(g, max_depth=1) as s1:
                count1 = s1.leaf_count
            with SubgameSolver(g, max_depth=2) as s2:
                count2 = s2.leaf_count
        assert count2 >= count1

    def test_handle_exhaustion_guard(self):
        """solver_h must be >= 0 after successful build."""
        with _make_game() as g:
            solver = SubgameSolver(g, max_depth=1)
            assert solver._solver_h >= 0
            solver.free()


class TestExportLeaves:
    """Export leaf states and verify they are usable GoEngine instances."""

    def test_returns_list_of_go_engines(self):
        with _make_game() as g:
            with SubgameSolver(g, max_depth=2) as solver:
                leaves = solver.export_leaves()
                assert len(leaves) == solver.leaf_count
                for leaf in leaves:
                    assert isinstance(leaf, GoEngine)

    def test_leaves_have_legal_actions(self):
        """Each exported leaf must return a valid action mask."""
        with _make_game() as g:
            with SubgameSolver(g, max_depth=2) as solver:
                leaves = solver.export_leaves()
                for leaf in leaves:
                    mask = leaf.legal_actions_mask()
                    assert mask.shape == (146,)
                    assert mask.dtype == np.uint8
                    # Non-terminal leaves should have at least one legal action
                    if not leaf.is_terminal():
                        assert mask.sum() > 0

    def test_non_owning_views_do_not_double_free(self):
        """GoEngine views from export_leaves must not free handles on GC."""
        with _make_game() as g:
            solver = SubgameSolver(g, max_depth=1)
            leaves = solver.export_leaves()
            # Close the view objects — they are non-owning, no double-free
            for leaf in leaves:
                leaf.close()
            # Solver should still be able to free cleanly
            solver.free()


class TestSolve:
    """Run CFR iterations and check output shapes/semantics."""

    def _run_solve(self, seed: int = 42, depth: int = 2, iters: int = 10):
        with _make_game(seed=seed) as g:
            with SubgameSolver(g, max_depth=depth) as solver:
                count = solver.leaf_count
                leaf_values = np.random.default_rng(0).random(
                    count * 2, dtype=np.float32
                )
                strategy, root_values = solver.solve(leaf_values, num_iterations=iters)
        return strategy, root_values

    def test_strategy_shape(self):
        strategy, _ = self._run_solve()
        assert strategy.shape == (146,)
        assert strategy.dtype == np.float32

    def test_strategy_sums_to_one(self):
        """Strategy over legal actions should sum to ~1.0."""
        strategy, _ = self._run_solve()
        total = float(strategy.sum())
        assert abs(total - 1.0) < 1e-4, f"Strategy sum = {total}"

    def test_root_values_shape(self):
        _, root_values = self._run_solve()
        assert root_values.shape == (2,)
        assert root_values.dtype == np.float32

    def test_2d_leaf_values_accepted(self):
        """solve() must accept (leaf_count, 2) shaped input."""
        with _make_game() as g:
            with SubgameSolver(g, max_depth=2) as solver:
                count = solver.leaf_count
                leaf_values = np.zeros((count, 2), dtype=np.float32)
                strategy, root_values = solver.solve(leaf_values)
        assert strategy.shape == (146,)
        assert root_values.shape == (2,)


class TestContextManager:
    """Verify context-manager lifecycle: build, solve, free, handle released."""

    def test_context_manager_frees_on_exit(self):
        with _make_game() as g:
            solver = SubgameSolver(g, max_depth=1)
            assert not solver._closed
            with solver:
                count = solver.leaf_count
                leaf_values = np.zeros(count * 2, dtype=np.float32)
                solver.solve(leaf_values)
            assert solver._closed
            assert solver._solver_h == -1

    def test_double_free_is_safe(self):
        """Calling free() twice must not raise."""
        with _make_game() as g:
            solver = SubgameSolver(g, max_depth=1)
            solver.free()
            solver.free()  # must not raise


class TestBufferReuse:
    """Multiple sequential solve() calls reuse buffers and produce consistent sums."""

    def test_multiple_solves_consistent(self):
        with _make_game(seed=99) as g:
            with SubgameSolver(g, max_depth=2) as solver:
                count = solver.leaf_count
                rng = np.random.default_rng(1)

                for _ in range(5):
                    leaf_values = rng.random(count * 2).astype(np.float32)
                    strategy, root_values = solver.solve(leaf_values, num_iterations=5)
                    total = float(strategy.sum())
                    assert abs(total - 1.0) < 1e-4, f"Strategy sum = {total}"
                    assert root_values.shape == (2,)
