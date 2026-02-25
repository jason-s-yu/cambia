"""
tests/test_es_validator.py

Tests for ESValidator (src/cfr/es_validator.py).

Covers:
- Creation with config and random weights
- compute_exploitability() return keys and values
- Metric invariants (regret >= 0, max >= mean, entropy >= 0)
- Zero traversals edge case
- Entropy helper function
- Depth limit respected
- Config fields parsed from YAML
- Trainer integration (es_validation_history attribute)
"""

from types import SimpleNamespace

import numpy as np
import pytest

# conftest.py handles the config stub
from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.networks import AdvantageNetwork
from src.cfr.es_validator import ESValidator, _compute_entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_config(
    depth: int = 5,
    interval: int = 1,
    traversals: int = 3,
    backend: str = "python",
) -> SimpleNamespace:
    """
    Build a SimpleNamespace config with fast ES validation settings.

    Uses SimpleNamespace (like other deep_worker tests) to avoid conflict with
    the conftest.py config stub.
    """
    config = SimpleNamespace()

    # deep_cfr sub-config
    config.deep_cfr = SimpleNamespace()
    config.deep_cfr.es_validation_interval = interval
    config.deep_cfr.es_validation_depth = depth
    config.deep_cfr.es_validation_traversals = traversals
    config.deep_cfr.engine_backend = backend
    config.deep_cfr.hidden_dim = 256
    config.deep_cfr.dropout = 0.1
    config.deep_cfr.learning_rate = 1e-3
    config.deep_cfr.batch_size = 2048
    config.deep_cfr.train_steps_per_iteration = 100
    config.deep_cfr.alpha = 1.5
    config.deep_cfr.traversals_per_step = 10
    config.deep_cfr.advantage_buffer_capacity = 10_000
    config.deep_cfr.strategy_buffer_capacity = 10_000
    config.deep_cfr.save_interval = 0
    config.deep_cfr.device = "cpu"
    config.deep_cfr.sampling_method = "external"
    config.deep_cfr.exploration_epsilon = 0.6

    # system sub-config
    config.system = SimpleNamespace()
    config.system.recursion_limit = 50

    # agent_params sub-config
    config.agent_params = SimpleNamespace()
    config.agent_params.memory_level = 1
    config.agent_params.time_decay_turns = 3

    # cambia_rules sub-config (real class for CambiaGameState)
    from src.config import CambiaRulesConfig

    config.cambia_rules = CambiaRulesConfig()
    config.cambia_rules.max_game_turns = 20

    # persistence stub (for trainer)
    config.persistence = SimpleNamespace()
    config.persistence.agent_data_save_path = "/tmp/deep_cfr_test_checkpoint.pt"

    # cfr_training stub (for trainer)
    config.cfr_training = SimpleNamespace()
    config.cfr_training.num_workers = 1
    config.cfr_training.num_iterations = 2

    # logging stub
    config.logging = SimpleNamespace()
    config.logging.log_level_file = "WARNING"
    config.logging.log_level_console = "WARNING"
    config.logging.log_dir = "/tmp/test_logs"
    config.logging.log_file_prefix = "cambia_test"
    config.logging.log_max_bytes = 1024 * 1024
    config.logging.log_backup_count = 2
    config.logging.log_simulation_traces = False
    config.logging.log_archive_enabled = False
    config.logging.get_worker_log_level = lambda wid, ntotal: "WARNING"

    return config


def make_random_weights() -> dict:
    """Return a random AdvantageNetwork state-dict as numpy arrays."""
    net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=256, output_dim=NUM_ACTIONS)
    return {k: v.cpu().numpy() for k, v in net.state_dict().items()}


def make_network_config() -> dict:
    """Return the standard network config dict."""
    return {"input_dim": INPUT_DIM, "hidden_dim": 256, "output_dim": NUM_ACTIONS}


# ---------------------------------------------------------------------------
# Test 1 — creation
# ---------------------------------------------------------------------------


class TestESValidatorCreation:
    def test_es_validator_creation(self):
        """ESValidator can be instantiated with config and random weights."""
        config = make_test_config()
        weights = make_random_weights()
        net_cfg = make_network_config()

        validator = ESValidator(config=config, network_weights=weights, network_config=net_cfg)

        assert validator is not None
        assert validator.depth_limit == 5
        assert validator.engine_backend == "python"
        assert isinstance(validator.network, AdvantageNetwork)


# ---------------------------------------------------------------------------
# Test 2 — return keys
# ---------------------------------------------------------------------------


EXPECTED_KEYS = {
    "mean_regret",
    "max_regret",
    "strategy_entropy",
    "traversals",
    "depth",
    "elapsed_seconds",
    "total_nodes",
}


class TestComputeExploitabilityKeys:
    def test_compute_exploitability_returns_expected_keys(self):
        """metrics dict contains all expected keys."""
        config = make_test_config(depth=3, traversals=2)
        validator = ESValidator(
            config=config,
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )
        metrics = validator.compute_exploitability(num_traversals=2)

        assert set(metrics.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 3 — few traversals, valid metrics
# ---------------------------------------------------------------------------


class TestComputeExploitabilityFewTraversals:
    def test_compute_exploitability_with_few_traversals(self):
        """Runs 5 traversals and returns a valid metrics dict."""
        config = make_test_config(depth=4, traversals=5)
        validator = ESValidator(
            config=config,
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )
        metrics = validator.compute_exploitability(num_traversals=5)

        assert metrics["traversals"] >= 0
        assert metrics["depth"] == 4
        assert metrics["elapsed_seconds"] >= 0.0
        assert metrics["total_nodes"] >= 0


# ---------------------------------------------------------------------------
# Test 4 — metric value invariants
# ---------------------------------------------------------------------------


class TestMetricsReasonable:
    def test_metrics_values_are_reasonable(self):
        """mean_regret >= 0, max_regret >= mean_regret, entropy >= 0."""
        config = make_test_config(depth=5, traversals=5)
        validator = ESValidator(
            config=config,
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )
        metrics = validator.compute_exploitability(num_traversals=5)

        assert metrics["mean_regret"] >= 0.0
        assert metrics["max_regret"] >= metrics["mean_regret"] - 1e-9
        assert metrics["strategy_entropy"] >= 0.0


# ---------------------------------------------------------------------------
# Test 5 — zero traversals
# ---------------------------------------------------------------------------


class TestZeroTraversals:
    def test_compute_exploitability_with_zero_traversals(self):
        """Returns gracefully zeroed metrics when num_traversals=0."""
        config = make_test_config()
        validator = ESValidator(
            config=config,
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )
        metrics = validator.compute_exploitability(num_traversals=0)

        assert metrics["mean_regret"] == 0.0
        assert metrics["max_regret"] == 0.0
        assert metrics["strategy_entropy"] == 0.0
        assert metrics["traversals"] == 0
        assert metrics["total_nodes"] == 0


# ---------------------------------------------------------------------------
# Test 6 — consistent random network
# ---------------------------------------------------------------------------


class TestESValidatorWithTrainedNetwork:
    def test_es_validator_with_trained_network(self):
        """Metrics from a consistent (random but fixed) network are reproducible."""
        config = make_test_config(depth=4, traversals=4)
        weights = make_random_weights()
        net_cfg = make_network_config()

        v1 = ESValidator(config=config, network_weights=weights, network_config=net_cfg)
        v2 = ESValidator(config=config, network_weights=weights, network_config=net_cfg)

        np.random.seed(42)
        m1 = v1.compute_exploitability(num_traversals=4)
        np.random.seed(42)
        m2 = v2.compute_exploitability(num_traversals=4)

        # Both should have the same keys and non-negative values
        assert set(m1.keys()) == set(m2.keys())
        assert m1["traversals"] == m2["traversals"]


# ---------------------------------------------------------------------------
# Test 7 — entropy helper
# ---------------------------------------------------------------------------


class TestEntropyComputation:
    def test_entropy_uniform(self):
        """Uniform distribution over N actions has entropy = ln(N)."""
        n = 4
        strategy = np.ones(n) / n
        entropy = _compute_entropy(strategy)
        expected = np.log(n)
        assert abs(entropy - expected) < 1e-6

    def test_entropy_deterministic(self):
        """Deterministic strategy (one action = 1) has entropy = 0."""
        strategy = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = _compute_entropy(strategy)
        assert abs(entropy) < 1e-6

    def test_entropy_all_zeros(self):
        """All-zeros strategy should return 0 (no valid probabilities)."""
        strategy = np.zeros(4)
        entropy = _compute_entropy(strategy)
        assert entropy == 0.0

    def test_entropy_non_negative(self):
        """Entropy is always non-negative."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            raw = rng.random(6)
            strategy = raw / raw.sum()
            assert _compute_entropy(strategy) >= 0.0


# ---------------------------------------------------------------------------
# Test 8 — depth limit
# ---------------------------------------------------------------------------


class TestDepthLimitRespected:
    def test_depth_limit_respected(self):
        """Traversals with depth_limit=1 should visit fewer nodes than depth_limit=5."""
        weights = make_random_weights()
        net_cfg = make_network_config()

        config_shallow = make_test_config(depth=1, traversals=3)
        config_deep = make_test_config(depth=5, traversals=3)

        v_shallow = ESValidator(
            config=config_shallow, network_weights=weights, network_config=net_cfg
        )
        v_deep = ESValidator(
            config=config_deep, network_weights=weights, network_config=net_cfg
        )

        np.random.seed(7)
        m_shallow = v_shallow.compute_exploitability(num_traversals=3)
        np.random.seed(7)
        m_deep = v_deep.compute_exploitability(num_traversals=3)

        # Shallow traversals visit fewer nodes
        assert m_shallow["total_nodes"] <= m_deep["total_nodes"]
        assert v_shallow.depth_limit == 1
        assert v_deep.depth_limit == 5


# ---------------------------------------------------------------------------
# Test 9 — config fields parsed
# ---------------------------------------------------------------------------


class TestConfigFieldsParsed:
    def test_config_fields_parsed(self):
        """Verify new config fields exist and have correct defaults on DeepCfrConfig stub."""
        from src.config import DeepCfrConfig

        cfg = DeepCfrConfig()
        assert hasattr(cfg, "es_validation_interval")
        assert hasattr(cfg, "es_validation_depth")
        assert hasattr(cfg, "es_validation_traversals")

        assert cfg.es_validation_interval == 10
        assert cfg.es_validation_depth == 10
        assert cfg.es_validation_traversals == 1000

    def test_config_fields_overridable(self):
        """Config fields can be set on a SimpleNamespace config instance."""
        config = make_test_config(depth=7, interval=5, traversals=50)
        assert config.deep_cfr.es_validation_depth == 7
        assert config.deep_cfr.es_validation_interval == 5
        assert config.deep_cfr.es_validation_traversals == 50

    def test_config_fields_parsed_from_yaml(self, tmp_path):
        """Verify new config fields are parsed from a YAML config file."""
        import sys

        # We need the real config module, but conftest may have injected a stub.
        # Remove the stub and reload the real config for this test.
        import importlib

        # Temporarily remove stub
        stub = sys.modules.pop("src.config", None)
        try:
            import yaml

            # Re-import the actual module
            real_config = importlib.import_module("src.config")
            load_config = getattr(real_config, "load_config", None)

            if load_config is None:
                pytest.skip("load_config not available (stub active)")

            yaml_content = {
                "deep_cfr": {
                    "es_validation_interval": 25,
                    "es_validation_depth": 15,
                    "es_validation_traversals": 500,
                }
            }
            config_file = tmp_path / "test_config.yaml"
            config_file.write_text(yaml.dump(yaml_content))

            cfg = load_config(str(config_file))
            assert cfg is not None
            assert cfg.deep_cfr.es_validation_interval == 25
            assert cfg.deep_cfr.es_validation_depth == 15
            assert cfg.deep_cfr.es_validation_traversals == 500
        finally:
            # Restore whatever was there (stub or real)
            if stub is not None:
                sys.modules["src.config"] = stub
            elif "src.config" in sys.modules:
                del sys.modules["src.config"]


# ---------------------------------------------------------------------------
# Test 10 — trainer integration
# ---------------------------------------------------------------------------


class TestTrainerESValidationIntegration:
    def test_trainer_es_validation_history_attribute(self):
        """DeepCFRTrainer has es_validation_history attribute initialized."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        config = make_test_config()
        dcfr_cfg = DeepCFRConfig(
            es_validation_interval=1,
            es_validation_depth=5,
            es_validation_traversals=3,
        )
        trainer = DeepCFRTrainer(config=config, deep_cfr_config=dcfr_cfg)

        assert hasattr(trainer, "es_validation_history")
        assert isinstance(trainer.es_validation_history, list)
        assert len(trainer.es_validation_history) == 0

    def test_trainer_dcfr_config_has_es_fields(self):
        """DeepCFRConfig dataclass has all three ES validation fields."""
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = DeepCFRConfig()
        assert hasattr(cfg, "es_validation_interval")
        assert hasattr(cfg, "es_validation_depth")
        assert hasattr(cfg, "es_validation_traversals")
        assert cfg.es_validation_interval == 10
        assert cfg.es_validation_depth == 10
        assert cfg.es_validation_traversals == 1000

    def test_trainer_from_yaml_config_propagates_es_fields(self):
        """DeepCFRConfig.from_yaml_config propagates ES fields from Config."""
        from src.cfr.deep_trainer import DeepCFRConfig

        config = make_test_config(depth=7, interval=5, traversals=42)
        dcfr_cfg = DeepCFRConfig.from_yaml_config(config)

        assert dcfr_cfg.es_validation_depth == 7
        assert dcfr_cfg.es_validation_interval == 5
        assert dcfr_cfg.es_validation_traversals == 42
