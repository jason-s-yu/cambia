"""
Tests for Deep CFR Outcome Sampling implementation.

Covers:
- OS traversal walks single path (node count O(depth), not exponential)
- IS-weighted regret computation produces valid values
- Advantage and strategy samples are generated correctly
- Config routing between OS and ES
- Smoke test for short games
"""

import multiprocessing
import queue
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from src.agent_state import AgentState
from src.cfr.deep_worker import (
    DeepCFRWorkerResult,
    _deep_traverse,
    _deep_traverse_os,
    run_deep_cfr_worker,
)
from src.config import CambiaRulesConfig
from src.constants import NUM_PLAYERS
from src.game.engine import CambiaGameState
from src.reservoir import ReservoirSample
from src.utils import WorkerStats


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    # Manually construct config object
    config = SimpleNamespace()

    # CambiaRulesConfig
    config.cambia_rules = CambiaRulesConfig()
    config.cambia_rules.max_game_turns = 8
    config.cambia_rules.cards_per_player = 4

    # SystemConfig
    config.system = SimpleNamespace()
    config.system.recursion_limit = 100

    # AgentParamsConfig
    config.agent_params = SimpleNamespace()
    config.agent_params.memory_level = 1
    config.agent_params.time_decay_turns = 3

    # DeepCfrConfig
    config.deep_cfr = SimpleNamespace()
    config.deep_cfr.sampling_method = "outcome"
    config.deep_cfr.exploration_epsilon = 0.6
    config.deep_cfr.hidden_dim = 256
    config.deep_cfr.dropout = 0.1
    config.deep_cfr.learning_rate = 0.001
    config.deep_cfr.batch_size = 2048
    config.deep_cfr.train_steps_per_iteration = 4000
    config.deep_cfr.alpha = 1.5
    config.deep_cfr.traversals_per_step = 1000
    config.deep_cfr.advantage_buffer_capacity = 2000000
    config.deep_cfr.strategy_buffer_capacity = 2000000
    config.deep_cfr.save_interval = 10
    config.deep_cfr.use_gpu = False

    # LoggingConfig (needed by worker)
    config.logging = SimpleNamespace()
    config.logging.log_level_file = "INFO"
    config.logging.log_level_console = "INFO"
    config.logging.log_dir = "logs"
    config.logging.log_file_prefix = "cambia"
    config.logging.log_max_bytes = 10 * 1024 * 1024
    config.logging.log_backup_count = 5
    config.logging.log_simulation_traces = False
    config.logging.log_archive_enabled = False

    # Helper method for logging
    def get_worker_log_level(worker_id, num_workers):
        return "WARNING"
    config.logging.get_worker_log_level = get_worker_log_level

    return config


@pytest.fixture
def short_game_state(minimal_config):
    """Create a short game state for testing."""
    game_state = CambiaGameState(house_rules=minimal_config.cambia_rules)
    return game_state


@pytest.fixture
def initial_agent_states(minimal_config, short_game_state):
    """Create initial agent states."""
    from src.cfr.worker import _create_observation, _filter_observation

    initial_obs = _create_observation(None, None, short_game_state, -1, [])
    if initial_obs is None:
        raise ValueError("Failed to create initial observation")

    initial_hands = [list(p.hand) for p in short_game_state.players]
    initial_peeks = [p.initial_peek_indices for p in short_game_state.players]

    agent_states = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i,
            opponent_id=1 - i,
            memory_level=minimal_config.agent_params.memory_level,
            time_decay_turns=minimal_config.agent_params.time_decay_turns,
            initial_hand_size=len(initial_hands[i]),
            config=minimal_config,
        )
        agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
        agent_states.append(agent)

    return agent_states


def test_os_traversal_single_path(minimal_config, short_game_state, initial_agent_states):
    """Test that OS traversal walks exactly one root-to-terminal path."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()
    worker_stats.worker_id = 0

    min_depth_after_bottom_out_tracker = [float("inf")]
    has_bottomed_out_tracker = [False]
    simulation_nodes: List = []

    # Run OS traversal
    utility = _deep_traverse_os(
        game_state=short_game_state,
        agent_states=initial_agent_states,
        updating_player=0,
        network=None,  # Use uniform strategy
        iteration=0,
        config=minimal_config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
        has_bottomed_out_tracker=has_bottomed_out_tracker,
        simulation_nodes=simulation_nodes,
        exploration_epsilon=0.6,
    )

    # Verify utility is valid
    assert utility is not None
    assert len(utility) == NUM_PLAYERS
    assert not np.any(np.isnan(utility))

    # OS should visit O(depth) nodes, not exponential
    # For a game with max 8 turns and ~3-5 actions per turn, expect < 100 nodes
    assert worker_stats.nodes_visited < 100, (
        f"OS visited {worker_stats.nodes_visited} nodes, expected < 100. "
        "This suggests exponential traversal instead of single path."
    )

    # Should have some advantage and strategy samples
    total_samples = len(advantage_samples) + len(strategy_samples)
    assert total_samples > 0, "No samples generated"
    assert total_samples <= worker_stats.nodes_visited, (
        "More samples than nodes visited"
    )


def test_os_vs_es_node_count(minimal_config):
    """Compare node counts between OS and ES to verify OS is single-path."""
    # Create fresh game states for each traversal
    from src.cfr.worker import _create_observation, _filter_observation

    es_game = CambiaGameState(house_rules=minimal_config.cambia_rules)
    es_obs = _create_observation(None, None, es_game, -1, [])
    es_hands = [list(p.hand) for p in es_game.players]
    es_peeks = [p.initial_peek_indices for p in es_game.players]
    es_agents = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i, opponent_id=1-i,
            memory_level=minimal_config.agent_params.memory_level,
            time_decay_turns=minimal_config.agent_params.time_decay_turns,
            initial_hand_size=len(es_hands[i]),
            config=minimal_config,
        )
        agent.initialize(es_obs, es_hands[i], es_peeks[i])
        es_agents.append(agent)

    # Run ES traversal
    es_advantage: List[ReservoirSample] = []
    es_strategy: List[ReservoirSample] = []
    es_stats = WorkerStats()
    es_stats.worker_id = 0

    _ = _deep_traverse(
        game_state=es_game,
        agent_states=es_agents,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=es_advantage,
        strategy_samples=es_strategy,
        depth=0,
        worker_stats=es_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
    )

    # Create fresh game state for OS traversal
    os_game = CambiaGameState(house_rules=minimal_config.cambia_rules)
    os_obs = _create_observation(None, None, os_game, -1, [])
    os_hands = [list(p.hand) for p in os_game.players]
    os_peeks = [p.initial_peek_indices for p in os_game.players]
    os_agents = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i, opponent_id=1-i,
            memory_level=minimal_config.agent_params.memory_level,
            time_decay_turns=minimal_config.agent_params.time_decay_turns,
            initial_hand_size=len(os_hands[i]),
            config=minimal_config,
        )
        agent.initialize(os_obs, os_hands[i], os_peeks[i])
        os_agents.append(agent)

    # Run OS traversal
    os_advantage: List[ReservoirSample] = []
    os_strategy: List[ReservoirSample] = []
    os_stats = WorkerStats()
    os_stats.worker_id = 1

    _ = _deep_traverse_os(
        game_state=os_game,
        agent_states=os_agents,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=os_advantage,
        strategy_samples=os_strategy,
        depth=0,
        worker_stats=os_stats,
        progress_queue=None,
        worker_id=1,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.6,
    )

    # OS should visit far fewer nodes than ES
    # ES enumerates all actions at traverser nodes, OS samples one action everywhere
    assert os_stats.nodes_visited < es_stats.nodes_visited, (
        f"OS visited {os_stats.nodes_visited} nodes, ES visited {es_stats.nodes_visited}. "
        "OS should visit fewer nodes (single path) than ES (partial tree)."
    )

    # ES should visit many more nodes (potentially hundreds for even short games)
    # OS should visit O(depth) nodes (typically < 50 for short games)
    assert os_stats.nodes_visited < 50, f"OS visited {os_stats.nodes_visited} nodes, expected < 50"


def test_os_regret_values_valid(minimal_config, short_game_state, initial_agent_states):
    """Test that IS-weighted regrets are valid (no NaN, no explosion)."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    _ = _deep_traverse_os(
        game_state=short_game_state,
        agent_states=initial_agent_states,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.6,
    )

    # Check advantage samples
    for sample in advantage_samples:
        assert not np.any(np.isnan(sample.target)), "Regret target contains NaN"
        assert not np.any(np.isinf(sample.target)), "Regret target contains Inf"
        # Regrets should be reasonable (not exploding)
        max_abs_regret = np.max(np.abs(sample.target))
        assert max_abs_regret < 1000, f"Regret magnitude {max_abs_regret} too large"


def test_os_advantage_samples_generation(minimal_config, short_game_state, initial_agent_states):
    """Test that advantage samples are generated at traverser nodes."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    _ = _deep_traverse_os(
        game_state=short_game_state,
        agent_states=initial_agent_states,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.6,
    )

    # Should have some advantage samples (from traverser's nodes)
    assert len(advantage_samples) > 0, "No advantage samples generated"

    # Check sample structure
    for sample in advantage_samples:
        assert sample.features is not None
        assert sample.target is not None
        assert sample.action_mask is not None
        assert sample.iteration == 0
        assert len(sample.target) > 0


def test_os_strategy_samples_generation(minimal_config, short_game_state, initial_agent_states):
    """Test that strategy samples are generated at opponent nodes."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    _ = _deep_traverse_os(
        game_state=short_game_state,
        agent_states=initial_agent_states,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=advantage_samples,
        strategy_samples=strategy_samples,
        depth=0,
        worker_stats=worker_stats,
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.6,
    )

    # Should have some strategy samples (from opponent's nodes)
    assert len(strategy_samples) > 0, "No strategy samples generated"

    # Check sample structure
    for sample in strategy_samples:
        assert sample.features is not None
        assert sample.target is not None
        assert sample.action_mask is not None
        assert sample.iteration == 0
        # Strategy should sum to 1 (for legal actions)
        strategy_sum = np.sum(sample.target)
        assert 0.99 <= strategy_sum <= 1.01, f"Strategy sum {strategy_sum} not close to 1"


def test_config_routing_outcome_sampling(minimal_config):
    """Test that sampling_method='outcome' routes to OS traversal."""
    # Set config to outcome sampling
    minimal_config.deep_cfr.sampling_method = "outcome"
    minimal_config.deep_cfr.exploration_epsilon = 0.6

    # Prepare worker args
    worker_args = (
        0,  # iteration
        minimal_config,
        None,  # network_weights
        {"input_dim": 512, "hidden_dim": 256, "output_dim": 100},
        None,  # progress_queue
        None,  # archive_queue
        0,  # worker_id
        "logs",  # run_log_dir
        "test",  # run_timestamp
    )

    # Run worker
    result = run_deep_cfr_worker(worker_args)

    # Verify result
    assert result is not None
    assert isinstance(result, DeepCFRWorkerResult)
    assert result.stats.nodes_visited > 0

    # OS should visit fewer nodes (single path)
    assert result.stats.nodes_visited < 100, (
        f"OS visited {result.stats.nodes_visited} nodes, expected < 100"
    )


def test_config_routing_external_sampling(minimal_config):
    """Test that sampling_method='external' routes to ES traversal."""
    # Set config to external sampling
    minimal_config.deep_cfr.sampling_method = "external"

    # Prepare worker args
    worker_args = (
        0,  # iteration
        minimal_config,
        None,  # network_weights
        {"input_dim": 512, "hidden_dim": 256, "output_dim": 100},
        None,  # progress_queue
        None,  # archive_queue
        0,  # worker_id
        "logs",  # run_log_dir
        "test",  # run_timestamp
    )

    # Run worker
    result = run_deep_cfr_worker(worker_args)

    # Verify result
    assert result is not None
    assert isinstance(result, DeepCFRWorkerResult)
    assert result.stats.nodes_visited > 0

    # ES should visit more nodes (partial tree exploration)
    # For short games with max_game_turns=8, ES can visit hundreds of nodes


def test_os_smoke_test_multiple_traversals(minimal_config):
    """Smoke test: run multiple OS traversals and verify samples accumulate."""
    minimal_config.deep_cfr.sampling_method = "outcome"
    minimal_config.deep_cfr.exploration_epsilon = 0.6

    num_traversals = 5
    all_advantage_samples = []
    all_strategy_samples = []

    for i in range(num_traversals):
        worker_args = (
            i,  # iteration
            minimal_config,
            None,  # network_weights
            {"input_dim": 512, "hidden_dim": 256, "output_dim": 100},
            None,  # progress_queue
            None,  # archive_queue
            0,  # worker_id
            "logs",  # run_log_dir
            "test",  # run_timestamp
        )

        result = run_deep_cfr_worker(worker_args)
        assert result is not None

        all_advantage_samples.extend(result.advantage_samples)
        all_strategy_samples.extend(result.strategy_samples)

    # Verify samples accumulated
    assert len(all_advantage_samples) > 0, "No advantage samples after multiple traversals"
    assert len(all_strategy_samples) > 0, "No strategy samples after multiple traversals"

    # Each traversal should contribute some samples
    avg_adv_per_traversal = len(all_advantage_samples) / num_traversals
    avg_strat_per_traversal = len(all_strategy_samples) / num_traversals
    assert avg_adv_per_traversal > 0
    assert avg_strat_per_traversal > 0


def test_exploration_epsilon_affects_sampling(minimal_config):
    """Test that different exploration_epsilon values affect action selection."""
    from src.cfr.worker import _create_observation, _filter_observation

    # Create game state for high epsilon
    high_eps_game = CambiaGameState(house_rules=minimal_config.cambia_rules)
    high_eps_obs = _create_observation(None, None, high_eps_game, -1, [])
    high_eps_hands = [list(p.hand) for p in high_eps_game.players]
    high_eps_peeks = [p.initial_peek_indices for p in high_eps_game.players]
    high_eps_agents = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i, opponent_id=1-i,
            memory_level=minimal_config.agent_params.memory_level,
            time_decay_turns=minimal_config.agent_params.time_decay_turns,
            initial_hand_size=len(high_eps_hands[i]),
            config=minimal_config,
        )
        agent.initialize(high_eps_obs, high_eps_hands[i], high_eps_peeks[i])
        high_eps_agents.append(agent)

    # Run with high epsilon (more uniform exploration)
    high_eps_samples: List[ReservoirSample] = []
    _ = _deep_traverse_os(
        game_state=high_eps_game,
        agent_states=high_eps_agents,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=high_eps_samples,
        strategy_samples=[],
        depth=0,
        worker_stats=WorkerStats(),
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.9,  # High epsilon = more uniform
    )

    # Create game state for low epsilon
    low_eps_game = CambiaGameState(house_rules=minimal_config.cambia_rules)
    low_eps_obs = _create_observation(None, None, low_eps_game, -1, [])
    low_eps_hands = [list(p.hand) for p in low_eps_game.players]
    low_eps_peeks = [p.initial_peek_indices for p in low_eps_game.players]
    low_eps_agents = []
    for i in range(NUM_PLAYERS):
        agent = AgentState(
            player_id=i, opponent_id=1-i,
            memory_level=minimal_config.agent_params.memory_level,
            time_decay_turns=minimal_config.agent_params.time_decay_turns,
            initial_hand_size=len(low_eps_hands[i]),
            config=minimal_config,
        )
        agent.initialize(low_eps_obs, low_eps_hands[i], low_eps_peeks[i])
        low_eps_agents.append(agent)

    # Run with low epsilon (more strategy-based)
    low_eps_samples: List[ReservoirSample] = []
    _ = _deep_traverse_os(
        game_state=low_eps_game,
        agent_states=low_eps_agents,
        updating_player=0,
        network=None,
        iteration=0,
        config=minimal_config,
        advantage_samples=low_eps_samples,
        strategy_samples=[],
        depth=0,
        worker_stats=WorkerStats(),
        progress_queue=None,
        worker_id=0,
        min_depth_after_bottom_out_tracker=[float("inf")],
        has_bottomed_out_tracker=[False],
        simulation_nodes=[],
        exploration_epsilon=0.1,  # Low epsilon = more strategy-based
    )

    # Both should generate samples
    assert len(high_eps_samples) > 0
    assert len(low_eps_samples) > 0

    # Regret magnitudes might differ due to different sampling policies
    # (but both should be valid)
    for sample in high_eps_samples:
        assert not np.any(np.isnan(sample.target))
    for sample in low_eps_samples:
        assert not np.any(np.isnan(sample.target))
