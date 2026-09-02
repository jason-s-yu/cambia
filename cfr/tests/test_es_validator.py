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


def _go_available() -> bool:
    try:
        from src.config import CambiaRulesConfig  # noqa: PLC0415
        from src.ffi.bridge import GoEngine  # noqa: PLC0415

        e = GoEngine(house_rules=CambiaRulesConfig())
        e.close()
        return True
    except Exception:
        return False


# The validator traverses on the Go engine only (cambia-1783); without
# libcambia.so every traversal raises, which is the point of the guard.
needs_go = pytest.mark.skipif(not _go_available(), reason="libcambia.so not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_config(
    depth: int = 5,
    interval: int = 1,
    traversals: int = 3,
    backend: str = "go",
) -> SimpleNamespace:
    """
    Build a SimpleNamespace config with fast ES validation settings.

    Uses SimpleNamespace (like other deep_worker tests) to avoid conflict with
    the conftest.py config stub.
    """
    config = SimpleNamespace()

    # deep_cfr sub-config: use real Pydantic model with test overrides
    from src.config import DeepCfrConfig

    config.deep_cfr = DeepCfrConfig(
        es_validation_interval=interval,
        es_validation_depth=depth,
        es_validation_traversals=traversals,
        engine_backend=backend,
        train_steps_per_iteration=100,
        traversals_per_step=10,
        advantage_buffer_capacity=10_000,
        strategy_buffer_capacity=10_000,
        save_interval=0,
        device="cpu",
        sampling_method="external",
    )

    # system sub-config
    config.system = SimpleNamespace()
    config.system.recursion_limit = 50

    # agent_params sub-config
    config.agent_params = SimpleNamespace()
    config.agent_params.memory_level = 1
    config.agent_params.time_decay_turns = 3

    # cambia_rules sub-config (real class, as GoEngine requires)
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
# Test 1: creation
# ---------------------------------------------------------------------------


class TestESValidatorCreation:
    def test_es_validator_creation(self):
        """ESValidator can be instantiated with config and random weights."""
        config = make_test_config()
        weights = make_random_weights()
        net_cfg = make_network_config()

        validator = ESValidator(
            config=config, network_weights=weights, network_config=net_cfg
        )

        assert validator is not None
        assert validator.depth_limit == 5
        assert isinstance(validator.network, AdvantageNetwork)


# ---------------------------------------------------------------------------
# Test 2: return keys
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
    @needs_go
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
# Test 3: few traversals, valid metrics
# ---------------------------------------------------------------------------


class TestComputeExploitabilityFewTraversals:
    @needs_go
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
# Test 4: metric value invariants
# ---------------------------------------------------------------------------


class TestMetricsReasonable:
    @needs_go
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
# Test 5: zero traversals
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
# Test 6: consistent random network
# ---------------------------------------------------------------------------


class TestESValidatorWithTrainedNetwork:
    @needs_go
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
# Test 7: entropy helper
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
# Test 8: depth limit
# ---------------------------------------------------------------------------


class TestDepthLimitRespected:
    @needs_go
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
# Test 9: config fields parsed
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
# Test 10: trainer integration
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


# ---------------------------------------------------------------------------
# Test 11: validator network is built through the trainer's factory
# ---------------------------------------------------------------------------


def make_residual_weights(hidden_dim: int = 32, num_hidden_layers: int = 3) -> dict:
    """Return a ResidualAdvantageNetwork state-dict as numpy arrays."""
    from src.networks import ResidualAdvantageNetwork

    net = ResidualAdvantageNetwork(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        output_dim=NUM_ACTIONS,
    )
    return {k: v.cpu().numpy() for k, v in net.state_dict().items()}


def make_residual_network_config(
    hidden_dim: int = 32, num_hidden_layers: int = 3
) -> dict:
    """Network config as the trainer emits it for the residual default."""
    return {
        "input_dim": INPUT_DIM,
        "hidden_dim": hidden_dim,
        "output_dim": NUM_ACTIONS,
        "network_type": "residual",
        "use_residual": True,
        "num_hidden_layers": num_hidden_layers,
        "validate_inputs": True,
        "use_pos_embed": True,
    }


SUPPORTED_NETWORKS = [
    ("mlp", {}, "AdvantageNetwork"),
    (
        "residual",
        {"use_residual": True, "num_hidden_layers": 3},
        "ResidualAdvantageNetwork",
    ),
    ("slot_film", {"use_pos_embed": True}, "SlotFiLMAdvantageNetwork"),
    ("slot_multiply", {"use_pos_embed": True}, "SlotFiLMAdvantageNetwork"),
]


class TestESValidatorNetworkFactory:
    @pytest.mark.parametrize("network_type,extra,expected", SUPPORTED_NETWORKS)
    def test_every_supported_network_loads(self, network_type, extra, expected):
        """Weights from any supported network load into the validator."""
        from src.networks import build_advantage_network

        net_cfg = {
            "input_dim": INPUT_DIM,
            "hidden_dim": 32,
            "output_dim": NUM_ACTIONS,
            "network_type": network_type,
            **extra,
        }
        trained = build_advantage_network(**net_cfg)
        weights = {k: v.cpu().numpy() for k, v in trained.state_dict().items()}

        validator = ESValidator(
            config=make_test_config(),
            network_weights=weights,
            network_config=dict(net_cfg),
        )

        assert type(validator.network).__name__ == expected

    def test_bare_network_config_still_builds_the_mlp(self):
        """A config carrying only dimensions keeps the historical MLP shape."""
        validator = ESValidator(
            config=make_test_config(),
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )

        assert isinstance(validator.network, AdvantageNetwork)

    def test_unknown_network_type_raises_named_error(self):
        """An unbuildable network_type raises the named error, not ValueError."""
        from src.cfr.es_validator import ESValidatorNetworkError

        net_cfg = make_residual_network_config()
        net_cfg["network_type"] = "nonesuch"

        with pytest.raises(ESValidatorNetworkError):
            ESValidator(
                config=make_test_config(),
                network_weights=make_residual_weights(),
                network_config=net_cfg,
            )

    def test_mismatched_weights_raise_named_error(self):
        """Residual weights against an mlp network_config raise, not warn."""
        from src.cfr.es_validator import ESValidatorNetworkError

        net_cfg = make_residual_network_config()
        net_cfg["network_type"] = "mlp"

        with pytest.raises(ESValidatorNetworkError) as exc_info:
            ESValidator(
                config=make_test_config(),
                network_weights=make_residual_weights(),
                network_config=net_cfg,
            )

        assert "ES validation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 12: a validator network failure is fatal, not a log line
# ---------------------------------------------------------------------------


def _residual_dcfr_config():
    """DeepCFRConfig at the residual default, sized for a one-step CPU run."""
    from src.cfr.deep_trainer import DeepCFRConfig

    return DeepCFRConfig(
        engine_backend="go",
        sampling_method="outcome",
        device="cpu",
        hidden_dim=32,
        batch_size=4,
        train_steps_per_iteration=1,
        traversals_per_step=2,
        traversal_depth_limit=6,
        advantage_buffer_capacity=1000,
        strategy_buffer_capacity=1000,
        save_interval=0,
        pipeline_training=False,
        num_traversal_threads=1,
        es_validation_interval=1,
        es_validation_depth=3,
        es_validation_traversals=2,
    )


def _small_trainer_config():
    """Config for a one-step trainer run: short games, single sampled path.

    The worker reads sampling_method and traversal_depth_limit off the Config,
    not off DeepCFRConfig, so both are pinned here. External sampling over a
    20-turn game enumerates a tree far too large for a unit test.
    """
    config = make_test_config(depth=3, interval=1, traversals=2)
    config.deep_cfr.sampling_method = "outcome"
    config.deep_cfr.traversal_depth_limit = 6
    config.cambia_rules.max_game_turns = 6
    return config


class TestTrainerESValidationIsNotSilent:
    @needs_go
    def test_residual_default_run_reports_mean_regret(self):
        """One training step at the residual default records an ES metric."""
        from src.cfr.deep_trainer import DeepCFRTrainer

        dcfr_cfg = _residual_dcfr_config()
        assert dcfr_cfg.use_residual is True
        assert dcfr_cfg.network_type == "residual"

        trainer = DeepCFRTrainer(config=_small_trainer_config(), deep_cfr_config=dcfr_cfg)
        trainer.train(num_training_steps=1)

        assert len(trainer.es_validation_history) == 1
        step, metrics = trainer.es_validation_history[0]
        assert step == 1
        assert "mean_regret" in metrics
        assert metrics["mean_regret"] >= 0.0
        assert metrics["traversals"] > 0

    @needs_go
    def test_mismatched_validator_network_aborts_training(self, monkeypatch):
        """A validator network that cannot load raises out of the training loop."""
        from src.cfr.deep_trainer import DeepCFRTrainer
        from src.cfr.es_validator import ESValidatorNetworkError

        trainer = DeepCFRTrainer(
            config=_small_trainer_config(), deep_cfr_config=_residual_dcfr_config()
        )

        # The trainer trains a residual net; declare an mlp so the load fails.
        real_get = trainer._get_network_config

        def mismatched():
            cfg = dict(real_get())
            cfg["network_type"] = "mlp"
            return cfg

        monkeypatch.setattr(trainer, "_get_network_config", mismatched)

        with pytest.raises(ESValidatorNetworkError):
            trainer.train(num_training_steps=1)

    @needs_go
    def test_disabled_validation_ignores_a_broken_validator_network(self):
        """es_validation_interval=0 trains to completion, mismatch or not."""
        from src.cfr.deep_trainer import DeepCFRTrainer

        dcfr_cfg = _residual_dcfr_config()
        dcfr_cfg.es_validation_interval = 0

        trainer = DeepCFRTrainer(config=_small_trainer_config(), deep_cfr_config=dcfr_cfg)
        trainer._get_network_config = lambda: {"network_type": "nonesuch"}

        trainer.train(num_training_steps=1)

        assert trainer.es_validation_history == []


# ---------------------------------------------------------------------------
# Test 13: a validation step that completes no traversal is fatal
# ---------------------------------------------------------------------------


class TestESValidatorTraversalFailures:
    def _validator(self):
        return ESValidator(
            config=make_test_config(depth=3, traversals=4),
            network_weights=make_random_weights(),
            network_config=make_network_config(),
        )

    def test_every_traversal_failing_raises(self):
        """Zero completed traversals is never a measurement, so it raises."""
        from src.cfr.es_validator import ESValidatorError

        validator = self._validator()

        def boom(updating_player):
            raise RuntimeError("libcambia.so is gone")

        validator._traverse_go = boom

        with pytest.raises(ESValidatorError) as exc_info:
            validator.compute_exploitability(num_traversals=4)

        assert "0 of 4" in str(exc_info.value)

    @needs_go
    def test_one_failed_traversal_of_four_still_reports(self, caplog):
        """A partial failure keeps its warning and still reports metrics."""
        import logging

        validator = self._validator()
        real = validator._traverse_go
        calls = []

        def flaky(updating_player):
            calls.append(updating_player)
            if len(calls) == 2:
                raise RuntimeError("transient hiccup")
            return real(updating_player)

        validator._traverse_go = flaky

        with caplog.at_level(logging.WARNING, logger="src.cfr.es_validator"):
            metrics = validator.compute_exploitability(num_traversals=4)

        assert metrics["traversals"] == 3
        assert metrics["mean_regret"] >= 0.0
        assert any("traversal 1 failed" in r.getMessage() for r in caplog.records)


class TestTrainerAbortsOnDeadValidation:
    @needs_go
    def test_all_traversals_failing_aborts_training(self, monkeypatch):
        """A validation step that completes nothing stops the run."""
        from src.cfr.deep_trainer import DeepCFRTrainer
        from src.cfr.es_validator import ESValidatorError

        def boom(self, updating_player):
            raise RuntimeError("libcambia.so is gone")

        monkeypatch.setattr(ESValidator, "_traverse_go", boom)

        trainer = DeepCFRTrainer(
            config=_small_trainer_config(), deep_cfr_config=_residual_dcfr_config()
        )

        with pytest.raises(ESValidatorError):
            trainer.train(num_training_steps=1)
