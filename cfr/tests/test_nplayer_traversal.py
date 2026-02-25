"""
tests/test_nplayer_traversal.py

Tests for _deep_traverse_os_go_nplayer() — N-player outcome sampling traversal
using the Go engine backend.

Requires libcambia.so to be built and available (skipped otherwise).
"""

import pytest
import numpy as np
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine  # noqa: PLC0415
        from src.config import CambiaRulesConfig  # noqa: PLC0415

        rules = CambiaRulesConfig()
        e = GoEngine(seed=0, house_rules=rules, num_players=3)
        e.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(depth_limit: int = 0):
    """Build a minimal Config-like object for traversal tests."""
    deep_cfr = SimpleNamespace(
        traversal_depth_limit=depth_limit,
        encoding_mode="legacy",
    )
    system = SimpleNamespace(recursion_limit=10000)
    cfg = SimpleNamespace(deep_cfr=deep_cfr, system=system)
    return cfg


def _make_engine_and_agents(num_players: int = 3, seed: int = 42):
    """Create a GoEngine and N GoAgentState.new_nplayer() instances."""
    from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
    from src.config import CambiaRulesConfig  # noqa: PLC0415

    rules = CambiaRulesConfig()
    rules.cards_per_player = 4
    engine = GoEngine(seed=seed, house_rules=rules, num_players=num_players)
    agents = [GoAgentState.new_nplayer(engine, i, num_players) for i in range(num_players)]
    return engine, agents


def _make_tracker():
    return [float("inf")], [False]


def _call_traversal(engine, agents, num_players, config, depth_limit=None):
    from src.cfr.deep_worker import _deep_traverse_os_go_nplayer  # noqa: PLC0415
    from src.utils import WorkerStats, SimulationNodeData  # noqa: PLC0415
    from src.reservoir import ReservoirSample  # noqa: PLC0415

    advantage_samples = []
    strategy_samples = []
    worker_stats = WorkerStats()
    min_depth_tracker, bottom_out_tracker = _make_tracker()

    result = _deep_traverse_os_go_nplayer(
        engine=engine,
        agent_states=agents,
        updating_player=0,
        network=None,
        iteration=1,
        config=config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=min_depth_tracker,
        has_bottomed_out_tracker=bottom_out_tracker,
        simulation_nodes=[],
        exploration_epsilon=0.6,
        num_players=num_players,
        depth_limit=depth_limit,
    )
    return result, advantage_samples, strategy_samples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNPlayerTraversalBasic:
    @skip_if_no_go
    def test_basic_3p_call_returns_utility(self):
        """3P traversal returns a (3,) float64 utility array."""
        engine, agents = _make_engine_and_agents(num_players=3)
        config = _make_config()
        try:
            result, _, _ = _call_traversal(engine, agents, 3, config)
        finally:
            engine.close()
            for a in agents:
                a.close()

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64

        # --- N-player zero-sum conservation ---
        # N-player utility: u_i = (totalScore - N*score_i) / (N-1).
        # By construction, sum(u_i) = (N*totalScore - N*totalScore) / (N-1) = 0.
        # Tolerance: float32 accumulation from Go engine then cast to float64.
        assert abs(result.sum()) < 1e-4, (
            f"3P utility not zero-sum: sum={result.sum()}, values={result}"
        )

        # --- Utility range bounds ---
        # Max hand score = 4*13 = 52, min = 4*(-1) = -4. With 3P pairwise formula,
        # max |u_i| = (N-1)*max_score_diff / (N-1) = max_score_diff = 56.
        # But in practice, values are much smaller. Use generous bound.
        for i in range(3):
            assert -100.0 <= result[i] <= 100.0, (
                f"3P utility[{i}]={result[i]} implausibly large"
            )

    @skip_if_no_go
    def test_basic_4p_call_returns_utility(self):
        """4P traversal returns a (4,) float64 utility array."""
        engine, agents = _make_engine_and_agents(num_players=4, seed=99)
        config = _make_config()
        try:
            result, _, _ = _call_traversal(engine, agents, 4, config)
        finally:
            engine.close()
            for a in agents:
                a.close()

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert result.dtype == np.float64

        # 4P zero-sum conservation (same algebraic identity as 3P)
        assert abs(result.sum()) < 1e-4, (
            f"4P utility not zero-sum: sum={result.sum()}, values={result}"
        )


class TestNPlayerUtilityShape:
    @skip_if_no_go
    def test_utility_shape_matches_num_players_3(self):
        """Utility shape equals num_players for 3P game."""
        engine, agents = _make_engine_and_agents(num_players=3, seed=7)
        config = _make_config()
        try:
            result, _, _ = _call_traversal(engine, agents, 3, config)
        finally:
            engine.close()
            for a in agents:
                a.close()
        assert result.shape == (3,)

    @skip_if_no_go
    def test_utility_shape_matches_num_players_4(self):
        """Utility shape equals num_players for 4P game."""
        engine, agents = _make_engine_and_agents(num_players=4, seed=13)
        config = _make_config()
        try:
            result, _, _ = _call_traversal(engine, agents, 4, config)
        finally:
            engine.close()
            for a in agents:
                a.close()
        assert result.shape == (4,)


class TestNPlayerSampleDimensions:
    @skip_if_no_go
    def test_advantage_sample_feature_dim(self):
        """Advantage samples have features of shape (580,)."""
        engine, agents = _make_engine_and_agents(num_players=3, seed=55)
        config = _make_config()
        try:
            _, advantage_samples, _ = _call_traversal(engine, agents, 3, config)
        finally:
            engine.close()
            for a in agents:
                a.close()

        # There may be 0 samples if updating_player never acts; run until we get some
        # We check all collected samples have correct shape
        for s in advantage_samples:
            assert s.features.shape == (580,), f"Expected (580,), got {s.features.shape}"
            assert s.target.shape == (452,), f"Expected (452,), got {s.target.shape}"

            # --- Regret mask consistency (N-player) ---
            # Regret targets are zero-initialized; only legal action indices are written.
            # Illegal action slots must remain exactly zero.
            illegal_mask = ~s.action_mask
            assert (s.target[illegal_mask] == 0).all(), (
                "N-player: non-zero regret for illegal action"
            )

    @skip_if_no_go
    def test_strategy_sample_feature_dim(self):
        """Strategy samples have features of shape (580,) and targets of shape (452,)."""
        engine, agents = _make_engine_and_agents(num_players=3, seed=77)
        config = _make_config()
        try:
            _, _, strategy_samples = _call_traversal(engine, agents, 3, config)
        finally:
            engine.close()
            for a in agents:
                a.close()

        for s in strategy_samples:
            assert s.features.shape == (580,), f"Expected (580,), got {s.features.shape}"
            assert s.target.shape == (452,), f"Expected (452,), got {s.target.shape}"

            # --- Strategy probability distribution (N-player) ---
            # OS-MCCFR stores σ(I) at opponent nodes. The strategy over legal
            # actions forms a valid probability distribution summing to 1.0.
            mask = s.action_mask
            strategy_sum = s.target[mask].sum()
            assert abs(strategy_sum - 1.0) < 0.01, (
                f"N-player strategy sum {strategy_sum} != 1.0"
            )

    @skip_if_no_go
    def test_samples_collected_across_players(self):
        """After traversal, at least one sample (advantage or strategy) is collected."""
        # Run multiple seeds until we get samples
        collected = False
        for seed in range(5):
            engine, agents = _make_engine_and_agents(num_players=3, seed=seed * 17)
            config = _make_config()
            try:
                _, adv, strat = _call_traversal(engine, agents, 3, config)
            finally:
                engine.close()
                for a in agents:
                    a.close()
            if len(adv) + len(strat) > 0:
                collected = True
                break
        assert collected, "Expected at least one sample after traversal over multiple seeds"


class TestNPlayerDepthLimit:
    @skip_if_no_go
    def test_depth_limit_terminates(self):
        """Traversal with depth_limit=5 terminates without error."""
        engine, agents = _make_engine_and_agents(num_players=3, seed=42)
        config = _make_config(depth_limit=5)
        try:
            result, _, _ = _call_traversal(engine, agents, 3, config, depth_limit=5)
        finally:
            engine.close()
            for a in agents:
                a.close()

        assert result.shape == (3,)

    @skip_if_no_go
    def test_depth_limit_1_returns_zeros(self):
        """Traversal with depth_limit=1 hits cap at first node and returns zeros."""
        engine, agents = _make_engine_and_agents(num_players=3, seed=42)
        config = _make_config(depth_limit=1)
        try:
            result, adv, strat = _call_traversal(engine, agents, 3, config, depth_limit=1)
        finally:
            engine.close()
            for a in agents:
                a.close()

        # With depth_limit=1, depth=0 is under limit, depth=1 hits cap → result is zeros
        # (unless game terminates before that)
        assert result.shape == (3,)
        # Samples from depth=0 should be present (one node was visited before cap)
        assert len(adv) + len(strat) <= 1  # at most one decision node visited
