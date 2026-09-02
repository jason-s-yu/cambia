"""
Tests for Deep CFR Outcome Sampling on the Go engine.

Covers:
- OS traversal walks single path (node count O(depth), not exponential)
- IS-weighted regret computation produces valid values
- Advantage and strategy samples are generated correctly
- Config routing between OS and ES
- Smoke test for short games

These were Python-engine tests until cambia-1783 retired that backend; each one
now asserts the same property against _deep_traverse_os_go / _deep_traverse_go.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from src.cfr.deep_worker import (
    DeepCFRWorkerResult,
    _deep_traverse_go,
    _deep_traverse_os_go,
    run_deep_cfr_worker,
)
from src.config import CambiaRulesConfig
from src.constants import NUM_PLAYERS
from src.reservoir import ReservoirSample
from src.utils import WorkerStats


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine  # noqa: PLC0415

        e = GoEngine(house_rules=CambiaRulesConfig())
        e.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _go_available(), reason="libcambia.so not available")


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
    config.deep_cfr.engine_backend = "go"
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
    config.deep_cfr.device = "cpu"

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

    config.cfr_training = SimpleNamespace()
    config.cfr_training.num_workers = 1

    return config


@contextmanager
def go_engine_and_agents(config, seed=None):
    """Yield (engine, agent_states) on the Go backend and close them after."""
    from src.ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415

    engine = GoEngine(house_rules=config.cambia_rules, seed=seed)
    agents = [
        GoAgentState(
            engine,
            pid,
            config.agent_params.memory_level,
            config.agent_params.time_decay_turns,
        )
        for pid in range(NUM_PLAYERS)
    ]
    try:
        yield engine, agents
    finally:
        for a in agents:
            a.close()
        engine.close()


def test_os_traversal_single_path(minimal_config):
    """Test that OS traversal walks exactly one root-to-terminal path."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()
    worker_stats.worker_id = 0

    min_depth_after_bottom_out_tracker = [float("inf")]
    has_bottomed_out_tracker = [False]
    simulation_nodes: List = []

    with go_engine_and_agents(minimal_config, seed=11) as (engine, agents):
        utility, _tail_ratio = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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
    assert total_samples <= worker_stats.nodes_visited, "More samples than nodes visited"


# The two ES cases in this module. Outcome sampling walks one path and costs
# under a second; external sampling walks the whole tree, and these two measured
# 221.6s of the module's roughly 223s, so they carry the mark and the rest of
# the file stays in the pull-request run (cambia-1920).
@pytest.mark.slow
def test_os_vs_es_node_count(minimal_config):
    """Compare node counts between OS and ES to verify OS is single-path."""
    es_advantage: List[ReservoirSample] = []
    es_strategy: List[ReservoirSample] = []
    es_stats = WorkerStats()
    es_stats.worker_id = 0

    with go_engine_and_agents(minimal_config, seed=23) as (engine, agents):
        _ = _deep_traverse_go(
            engine=engine,
            agent_states=agents,
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

    os_advantage: List[ReservoirSample] = []
    os_strategy: List[ReservoirSample] = []
    os_stats = WorkerStats()
    os_stats.worker_id = 1

    with go_engine_and_agents(minimal_config, seed=23) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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
    assert (
        os_stats.nodes_visited < 50
    ), f"OS visited {os_stats.nodes_visited} nodes, expected < 50"


def test_os_regret_values_valid(minimal_config):
    """Test that IS-weighted regrets are valid (no NaN, no explosion)."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    with go_engine_and_agents(minimal_config, seed=31) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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


def test_os_advantage_samples_generation(minimal_config):
    """Test that advantage samples are generated at traverser nodes."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    with go_engine_and_agents(minimal_config, seed=37) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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


def test_os_strategy_samples_generation(minimal_config):
    """Test that strategy samples are generated at opponent nodes."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    worker_stats = WorkerStats()

    with go_engine_and_agents(minimal_config, seed=41) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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
    assert (
        result.stats.nodes_visited < 100
    ), f"OS visited {result.stats.nodes_visited} nodes, expected < 100"


@pytest.mark.slow
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

    # ES enumerates every action at traverser nodes, so it walks a partial tree
    # rather than the single root-to-leaf path OS takes (which visits exactly
    # max_depth + 1 nodes). A node-count threshold would be seed-dependent:
    # an early Cambia call ends the game in under 20 nodes.
    assert result.stats.nodes_visited > result.stats.max_depth + 1


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
    assert (
        len(all_advantage_samples) > 0
    ), "No advantage samples after multiple traversals"
    assert len(all_strategy_samples) > 0, "No strategy samples after multiple traversals"

    # Each traversal should contribute some samples
    avg_adv_per_traversal = len(all_advantage_samples) / num_traversals
    avg_strat_per_traversal = len(all_strategy_samples) / num_traversals
    assert avg_adv_per_traversal > 0
    assert avg_strat_per_traversal > 0


def test_exploration_epsilon_affects_sampling(minimal_config):
    """Test that different exploration_epsilon values affect action selection."""
    # Run with high epsilon (more uniform exploration)
    high_eps_samples: List[ReservoirSample] = []
    with go_engine_and_agents(minimal_config, seed=53) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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

    # Run with low epsilon (more strategy-based)
    low_eps_samples: List[ReservoirSample] = []
    with go_engine_and_agents(minimal_config, seed=53) as (engine, agents):
        _ = _deep_traverse_os_go(
            engine=engine,
            agent_states=agents,
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


# ---------------------------------------------------------------------------
# Network round-trip contract tests
# ---------------------------------------------------------------------------

import torch
from src.networks import (
    AdvantageNetwork,
    ResidualAdvantageNetwork,
    build_advantage_network,
)
from src.encoding import INPUT_DIM, NUM_ACTIONS


@pytest.mark.parametrize("use_residual", [False, True], ids=["plain", "residual"])
def test_worker_network_roundtrip(minimal_config, use_residual):
    """Trainer serializes weights → worker deserializes into matching architecture.

    This is the contract that was broken for the EP-PBS run: the trainer built
    a ResidualAdvantageNetwork but _get_network_config() didn't forward
    use_residual, so the worker built a plain AdvantageNetwork and
    load_state_dict() failed silently (falling back to uniform strategy).
    """
    input_dim = INPUT_DIM
    hidden_dim = 64  # small for speed
    output_dim = NUM_ACTIONS
    num_hidden_layers = 2

    # 1. Build the "trainer-side" network
    trainer_net = build_advantage_network(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_residual=use_residual,
        num_hidden_layers=num_hidden_layers,
    )

    # 2. Serialize weights the same way _get_network_weights_for_workers() does
    weights_numpy = {k: v.cpu().numpy() for k, v in trainer_net.state_dict().items()}

    # 3. Build network_config the same way _get_network_config() does
    network_config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "validate_inputs": True,
        "use_residual": use_residual,
        "num_hidden_layers": num_hidden_layers,
    }

    # 4. Reconstruct in the worker the same way run_deep_cfr_worker() does
    worker_net = build_advantage_network(
        input_dim=network_config["input_dim"],
        hidden_dim=network_config["hidden_dim"],
        output_dim=network_config["output_dim"],
        validate_inputs=network_config.get("validate_inputs", True),
        use_residual=network_config.get("use_residual", False),
        num_hidden_layers=network_config.get("num_hidden_layers", 2),
    )
    weights_tensors = {
        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in weights_numpy.items()
    }
    # This is the line that failed in the EP-PBS run: must not raise
    worker_net.load_state_dict(weights_tensors)

    # 5. Verify the architectures match
    if use_residual:
        assert isinstance(worker_net, ResidualAdvantageNetwork)
    else:
        assert isinstance(worker_net, AdvantageNetwork)
        assert not isinstance(worker_net, ResidualAdvantageNetwork)

    # 6. Verify forward pass produces identical output
    worker_net.eval()
    trainer_net.eval()
    features = torch.randn(1, input_dim)
    mask = torch.ones(1, output_dim, dtype=torch.bool)
    with torch.no_grad():
        trainer_out = trainer_net(features, mask)
        worker_out = worker_net(features, mask)
    assert torch.allclose(
        trainer_out, worker_out, atol=1e-6
    ), f"Output mismatch: max diff={torch.max(torch.abs(trainer_out - worker_out))}"


@pytest.mark.parametrize("use_residual", [False, True], ids=["plain", "residual"])
def test_worker_network_mismatch_raises(use_residual):
    """Loading weights from architecture A into architecture B must raise."""
    input_dim = INPUT_DIM
    hidden_dim = 64
    output_dim = NUM_ACTIONS

    # Build network with one architecture
    source_net = build_advantage_network(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_residual=use_residual,
    )
    weights_numpy = {k: v.cpu().numpy() for k, v in source_net.state_dict().items()}

    # Build network with the OTHER architecture
    target_net = build_advantage_network(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_residual=not use_residual,
    )
    weights_tensors = {k: torch.tensor(v) for k, v in weights_numpy.items()}

    # Must fail: this is what the EP-PBS bug looked like
    with pytest.raises(RuntimeError, match="(Missing key|Unexpected key)"):
        target_net.load_state_dict(weights_tensors)
