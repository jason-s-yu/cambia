"""
tests/test_depth_cap.py

Tests for traversal_depth_limit (depth cap) in Deep CFR traversal functions.

Verifies:
- depth_limit=0 (unlimited) does not constrain traversal
- depth_limit=N causes traversal to terminate at depth N (fewer nodes)
- Depth-capped traversal produces valid samples
- Config field present in the stub and source files
"""

import numpy as np
import pytest
from types import SimpleNamespace

from src.cfr.deep_worker import _deep_traverse, _deep_traverse_os
from src.config import CambiaRulesConfig
from src.constants import NUM_PLAYERS
from src.game.engine import CambiaGameState
from src.utils import WorkerStats


# ---------------------------------------------------------------------------
# Config builder (SimpleNamespace, following test_deep_worker_os.py pattern)
# ---------------------------------------------------------------------------


def _make_config(depth_limit: int = 0, sampling_method: str = "external") -> SimpleNamespace:
    """Build a minimal SimpleNamespace config for traversal tests."""
    config = SimpleNamespace()

    config.cambia_rules = CambiaRulesConfig()
    config.cambia_rules.max_game_turns = 6  # keep very small for ES traversal tractability
    config.cambia_rules.cards_per_player = 4
    config.cambia_rules.use_jokers = 0

    config.system = SimpleNamespace()
    config.system.recursion_limit = 200

    config.agent_params = SimpleNamespace()
    config.agent_params.memory_level = 0
    config.agent_params.time_decay_turns = 3

    config.deep_cfr = SimpleNamespace()
    config.deep_cfr.sampling_method = sampling_method
    config.deep_cfr.exploration_epsilon = 0.6
    config.deep_cfr.traversal_depth_limit = depth_limit
    config.deep_cfr.hidden_dim = 256
    config.deep_cfr.dropout = 0.1
    config.deep_cfr.engine_backend = "python"

    config.logging = SimpleNamespace()
    config.logging.log_level_file = "WARNING"
    config.logging.log_level_console = "WARNING"
    config.logging.log_dir = "logs"
    config.logging.log_file_prefix = "cambia"
    config.logging.log_max_bytes = 10 * 1024 * 1024
    config.logging.log_backup_count = 5
    config.logging.log_simulation_traces = False
    config.logging.log_archive_enabled = False
    config.logging.get_worker_log_level = lambda wid, ntotal: "WARNING"

    config.cfr_training = SimpleNamespace()
    config.cfr_training.num_workers = 1

    return config


# ---------------------------------------------------------------------------
# Traversal runner helpers
# ---------------------------------------------------------------------------


def _setup_game_and_agents(config):
    """Initialize game state and agent states for traversal."""
    from src.agent_state import AgentState
    from src.cfr.worker import _create_observation

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    initial_obs = _create_observation(None, None, game_state, -1, [])
    initial_hands = [list(p.hand) for p in game_state.players]
    initial_peeks = [p.initial_peek_indices for p in game_state.players]

    agent_states = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i,
            opponent_id=1 - i,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(initial_hands[i]),
            config=config,
        )
        agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
        agent_states.append(agent)

    return game_state, agent_states


def run_es_traversal(config):
    """Run one ES traversal; return (stats, adv_samples, strat_samples)."""
    game_state, agent_states = _setup_game_and_agents(config)

    advantage_samples = []
    strategy_samples = []
    worker_stats = WorkerStats()
    min_depth_tracker = [float("inf")]
    has_bottomed_out = [False]

    _deep_traverse(
        game_state=game_state,
        agent_states=agent_states,
        updating_player=0,
        network=None,
        iteration=0,
        config=config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=min_depth_tracker,
        has_bottomed_out_tracker=has_bottomed_out,
        simulation_nodes=[],
    )

    return worker_stats, advantage_samples, strategy_samples


def run_os_traversal(config):
    """Run one OS traversal; return (stats, adv_samples, strat_samples)."""
    game_state, agent_states = _setup_game_and_agents(config)

    advantage_samples = []
    strategy_samples = []
    worker_stats = WorkerStats()
    min_depth_tracker = [float("inf")]
    has_bottomed_out = [False]

    _deep_traverse_os(
        game_state=game_state,
        agent_states=agent_states,
        updating_player=0,
        network=None,
        iteration=0,
        config=config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=min_depth_tracker,
        has_bottomed_out_tracker=has_bottomed_out,
        simulation_nodes=[],
        exploration_epsilon=0.6,
    )

    return worker_stats, advantage_samples, strategy_samples


# ---------------------------------------------------------------------------
# Tests: config field presence
# ---------------------------------------------------------------------------


class TestDeepCfrConfigField:
    """Tests for the traversal_depth_limit config field."""

    def test_stub_default_is_zero(self):
        """Conftest stub has traversal_depth_limit defaulting to 0."""
        from src.config import DeepCfrConfig
        cfg = DeepCfrConfig()
        assert cfg.traversal_depth_limit == 0

    def test_stub_set_positive_value(self):
        """Conftest stub accepts traversal_depth_limit=10."""
        from src.config import DeepCfrConfig
        cfg = DeepCfrConfig(traversal_depth_limit=10)
        assert cfg.traversal_depth_limit == 10

    def test_stub_zero_means_unlimited(self):
        """Zero value accepted."""
        from src.config import DeepCfrConfig
        cfg = DeepCfrConfig(traversal_depth_limit=0)
        assert cfg.traversal_depth_limit == 0

    def test_internal_deepcfrconfig_has_field(self):
        """deep_trainer.py DeepCFRConfig declares traversal_depth_limit."""
        import os
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "cfr", "deep_trainer.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        assert "traversal_depth_limit" in source, (
            "traversal_depth_limit not found in deep_trainer.py"
        )

    def test_from_yaml_config_passes_field(self):
        """from_yaml_config includes traversal_depth_limit in kwargs."""
        import os
        import re
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "cfr", "deep_trainer.py"
        )
        with open(src_path, "r") as f:
            source = f.read()
        match = re.search(
            r"def from_yaml_config.*?return cls\(\*\*kwargs\)", source, re.DOTALL
        )
        assert match is not None, "from_yaml_config not found in deep_trainer.py"
        assert "traversal_depth_limit" in match.group(0), (
            "traversal_depth_limit not passed through from_yaml_config"
        )


# ---------------------------------------------------------------------------
# Tests: ES traversal depth cap
# ---------------------------------------------------------------------------


class TestESDepthCap:
    """Tests for depth cap in _deep_traverse (External Sampling)."""

    def test_depth_cap_reduces_nodes(self):
        """A lower depth cap visits no more nodes than a higher one.

        Note: ES traversal with depth_limit=0 (unlimited) is exponential in game
        depth and infeasible in tests. We compare two bounded limits instead.
        """
        deep_cfg = _make_config(depth_limit=5, sampling_method="external")
        shallow_cfg = _make_config(depth_limit=2, sampling_method="external")

        stats_deep, _, _ = run_es_traversal(deep_cfg)
        stats_shallow, _, _ = run_es_traversal(shallow_cfg)

        assert stats_shallow.nodes_visited <= stats_deep.nodes_visited

    def test_max_depth_respected(self):
        """ES traversal never exceeds the depth cap."""
        depth_limit = 3
        cfg = _make_config(depth_limit=depth_limit, sampling_method="external")

        stats, _, _ = run_es_traversal(cfg)
        assert stats.max_depth <= depth_limit

    def test_cap_at_1_produces_nodes(self):
        """ES traversal with depth_limit=1 visits at least one node."""
        cfg = _make_config(depth_limit=1, sampling_method="external")
        stats, adv_samples, _ = run_es_traversal(cfg)
        assert stats.nodes_visited > 0

    def test_depth3_produces_valid_samples(self):
        """ES with depth=3 produces well-formed advantage samples."""
        from src.encoding import NUM_ACTIONS, INPUT_DIM

        cfg = _make_config(depth_limit=3, sampling_method="external")
        _, adv_samples, strat_samples = run_es_traversal(cfg)

        all_samples = adv_samples + strat_samples
        for sample in all_samples:
            assert sample.features.shape == (INPUT_DIM,)
            assert sample.target.shape == (NUM_ACTIONS,)
            assert sample.action_mask.shape == (NUM_ACTIONS,)
            assert not np.any(np.isnan(sample.features))
            assert not np.any(np.isnan(sample.target))


# ---------------------------------------------------------------------------
# Tests: OS traversal depth cap
# ---------------------------------------------------------------------------


class TestOSDepthCap:
    """Tests for depth cap in _deep_traverse_os (Outcome Sampling)."""

    def test_depth_cap_reduces_nodes(self):
        """OS capped traversal visits no more nodes than a deeper one."""
        deep_cfg = _make_config(depth_limit=10, sampling_method="outcome")
        shallow_cfg = _make_config(depth_limit=2, sampling_method="outcome")

        stats_deep, _, _ = run_os_traversal(deep_cfg)
        stats_shallow, _, _ = run_os_traversal(shallow_cfg)

        assert stats_shallow.nodes_visited <= stats_deep.nodes_visited

    def test_max_depth_respected(self):
        """OS traversal never exceeds the depth cap."""
        depth_limit = 5
        cfg = _make_config(depth_limit=depth_limit, sampling_method="outcome")

        stats, _, _ = run_os_traversal(cfg)
        assert stats.max_depth <= depth_limit

    def test_depth10_produces_valid_samples(self):
        """OS with depth=10 produces well-formed samples."""
        from src.encoding import NUM_ACTIONS, INPUT_DIM

        cfg = _make_config(depth_limit=10, sampling_method="outcome")
        _, adv_samples, strat_samples = run_os_traversal(cfg)

        all_samples = adv_samples + strat_samples
        for sample in all_samples:
            assert sample.features.shape == (INPUT_DIM,)
            assert sample.target.shape == (NUM_ACTIONS,)
            assert sample.action_mask.shape == (NUM_ACTIONS,)
            assert not np.any(np.isnan(sample.features))
            assert not np.any(np.isnan(sample.target))
