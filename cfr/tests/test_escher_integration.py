"""
tests/test_escher_integration.py

Checkpoint roundtrip tests for ESCHER Phase 0.

Covers:
- ESCHER config fields load correctly from YAML
- Checkpoint save contains value_net when traversal_method="escher"
- Checkpoint roundtrip (save + load) preserves all three networks
- OS checkpoint -> ESCHER load (value net initializes fresh)
- ESCHER checkpoint -> OS load (value fields ignored)

NOTE: These tests instantiate DeepCFRTrainer with a minimal Config mock so they
can run without the full Python game engine (no Go FFI required).
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.networks import HistoryValueNetwork


# ---------------------------------------------------------------------------
# Helpers to build minimal Config + DeepCFRConfig without full YAML loading
# ---------------------------------------------------------------------------


def _make_minimal_config(agent_data_save_path: str) -> "MagicMock":
    """Return a minimal Config mock sufficient for DeepCFRTrainer.__init__."""
    cfg = MagicMock()
    cfg.persistence.agent_data_save_path = agent_data_save_path
    cfg.cfr_training.num_iterations = 1
    cfg.cfr_training.num_workers = 1
    return cfg


def _make_dcfr_config(traversal_method: str = "outcome", **kwargs):
    """Build internal DeepCFRConfig directly from the trainer module."""
    from src.cfr.deep_trainer import DeepCFRConfig

    return DeepCFRConfig(
        pipeline_training=False,
        num_traversal_threads=1,
        traversal_method=traversal_method,
        **kwargs,
    )


def _build_trainer(tmp_path: str, traversal_method: str = "outcome", **kwargs):
    """Instantiate DeepCFRTrainer with a minimal config."""
    from src.cfr.deep_trainer import DeepCFRTrainer

    checkpoint_path = os.path.join(tmp_path, "checkpoint.pt")
    config = _make_minimal_config(checkpoint_path)
    dcfr_config = _make_dcfr_config(traversal_method=traversal_method, **kwargs)
    trainer = DeepCFRTrainer(config=config, deep_cfr_config=dcfr_config)
    return trainer, checkpoint_path


# ---------------------------------------------------------------------------
# ESCHER config fields
# ---------------------------------------------------------------------------


class TestEscherConfigFields:
    def test_deep_cfr_config_has_traversal_method(self):
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = DeepCFRConfig(pipeline_training=False)
        assert hasattr(cfg, "traversal_method")
        assert cfg.traversal_method == "outcome"

    def test_deep_cfr_config_escher_mode(self):
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = DeepCFRConfig(pipeline_training=False, traversal_method="escher")
        assert cfg.traversal_method == "escher"
        assert cfg.value_hidden_dim == 512
        assert cfg.value_learning_rate == 1e-3
        assert cfg.value_buffer_capacity == 2_000_000
        assert cfg.batch_counterfactuals is True

    def test_yaml_config_escher_fields_load(self, tmp_path):
        """Test that ESCHER fields round-trip through YAML -> load_config."""
        yaml_content = {
            "deep_cfr": {
                "traversal_method": "escher",
                "value_hidden_dim": 256,
                "value_learning_rate": 5e-4,
                "value_buffer_capacity": 500_000,
                "batch_counterfactuals": False,
            }
        }
        yaml_path = str(tmp_path / "test_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f)

        # Use the real load_config
        import importlib
        import sys

        # Remove stub if present, load real config
        real_config_mod = None
        try:
            # Try importing real config directly
            spec = importlib.util.spec_from_file_location(
                "src_config_real",
                str(Path(__file__).resolve().parent.parent / "src" / "config.py"),
            )
            real_config_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(real_config_mod)
        except Exception:
            pytest.skip("Cannot load real config module for YAML test")

        cfg = real_config_mod.load_config(yaml_path)
        assert cfg is not None
        assert cfg.deep_cfr.traversal_method == "escher"
        assert cfg.deep_cfr.value_hidden_dim == 256
        assert cfg.deep_cfr.value_learning_rate == 5e-4
        assert cfg.deep_cfr.value_buffer_capacity == 500_000
        assert cfg.deep_cfr.batch_counterfactuals is False


# ---------------------------------------------------------------------------
# Checkpoint save contains value_net when traversal_method="escher"
# ---------------------------------------------------------------------------


class TestCheckpointSaveEscher:
    def test_escher_checkpoint_contains_value_net(self, tmp_path):
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="escher")
        trainer.save_checkpoint(checkpoint_path)

        assert os.path.exists(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "value_net_state_dict" in ckpt
        assert ckpt["value_net_state_dict"] is not None
        assert len(ckpt["value_net_state_dict"]) > 0

    def test_os_checkpoint_value_net_is_none(self, tmp_path):
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="outcome")
        trainer.save_checkpoint(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "value_net_state_dict" in ckpt
        assert ckpt["value_net_state_dict"] is None

    def test_escher_checkpoint_has_value_buffer_path(self, tmp_path):
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="escher")
        trainer.save_checkpoint(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "value_buffer_path" in ckpt
        assert ckpt["value_buffer_path"] is not None


# ---------------------------------------------------------------------------
# Checkpoint roundtrip preserves all three networks
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_escher_roundtrip_preserves_value_net(self, tmp_path):
        """Save ESCHER checkpoint, load into new trainer, verify value_net weights match."""
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="escher")

        # Capture original value_net weights
        original_sd = {
            k: v.clone() for k, v in trainer.value_net.state_dict().items()
        }

        trainer.save_checkpoint(checkpoint_path)

        # Load into a fresh trainer
        trainer2, _ = _build_trainer(str(tmp_path), traversal_method="escher")
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.value_net is not None
        loaded_sd = trainer2.value_net.state_dict()
        for k in original_sd:
            assert k in loaded_sd, f"Missing key in loaded value_net: {k}"
            assert torch.allclose(original_sd[k], loaded_sd[k]), (
                f"Value net weight mismatch for {k}"
            )

    def test_escher_roundtrip_preserves_advantage_net(self, tmp_path):
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="escher")
        original_sd = {k: v.clone() for k, v in trainer.advantage_net.state_dict().items()}
        trainer.save_checkpoint(checkpoint_path)

        trainer2, _ = _build_trainer(str(tmp_path), traversal_method="escher")
        trainer2.load_checkpoint(checkpoint_path)

        for k in original_sd:
            assert torch.allclose(original_sd[k], trainer2.advantage_net.state_dict()[k])

    def test_escher_roundtrip_preserves_strategy_net(self, tmp_path):
        trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="escher")
        original_sd = {k: v.clone() for k, v in trainer.strategy_net.state_dict().items()}
        trainer.save_checkpoint(checkpoint_path)

        trainer2, _ = _build_trainer(str(tmp_path), traversal_method="escher")
        trainer2.load_checkpoint(checkpoint_path)

        for k in original_sd:
            assert torch.allclose(original_sd[k], trainer2.strategy_net.state_dict()[k])


# ---------------------------------------------------------------------------
# OS checkpoint -> ESCHER load (value net initializes fresh)
# ---------------------------------------------------------------------------


class TestCrossModeSafety:
    def test_os_checkpoint_to_escher_load(self, tmp_path):
        """Loading an OS checkpoint into ESCHER trainer: value_net initializes fresh."""
        # Save an OS checkpoint
        os_trainer, checkpoint_path = _build_trainer(str(tmp_path), traversal_method="outcome")
        os_trainer.save_checkpoint(checkpoint_path)

        # Load into ESCHER trainer
        escher_trainer, _ = _build_trainer(str(tmp_path), traversal_method="escher")
        escher_trainer.load_checkpoint(checkpoint_path)

        # value_net should exist and have valid weights (freshly initialized)
        assert escher_trainer.value_net is not None
        # Check that at least some parameters are non-zero (Kaiming init sets weights
        # non-zero; biases are zero-initialized, which is correct)
        total_nonzero = sum(
            p.abs().sum().item() for p in escher_trainer.value_net.parameters()
        )
        assert total_nonzero > 0
        for param in escher_trainer.value_net.parameters():
            assert not torch.isnan(param).any()

    def test_escher_checkpoint_to_os_load(self, tmp_path):
        """Loading an ESCHER checkpoint into OS trainer: value fields are ignored."""
        # Save an ESCHER checkpoint
        escher_trainer, checkpoint_path = _build_trainer(
            str(tmp_path), traversal_method="escher"
        )
        escher_trainer.save_checkpoint(checkpoint_path)

        # Load into OS trainer — should not raise
        os_trainer, _ = _build_trainer(str(tmp_path), traversal_method="outcome")
        os_trainer.load_checkpoint(checkpoint_path)  # should not raise

        # OS trainer should not have a value_net
        assert os_trainer.value_net is None
        assert os_trainer.value_optimizer is None


# ---------------------------------------------------------------------------
# Full training loop integration test (Phase 1)
# ---------------------------------------------------------------------------


class TestEscherTrainingLoop:
    """Integration tests for ESCHER training with the Go engine backend."""

    def _make_go_escher_config(self, agent_data_save_path: str):
        """Build a minimal real Config mock for Go-backend ESCHER training."""
        from types import SimpleNamespace
        from src.config import CambiaRulesConfig

        cfg = MagicMock()
        cfg.persistence.agent_data_save_path = agent_data_save_path
        cfg.cfr_training.num_iterations = 2
        cfg.cfr_training.num_workers = 1

        rules = CambiaRulesConfig()
        rules.max_game_turns = 4
        rules.cards_per_player = 4
        rules.initial_view_count = 2
        rules.cambia_allowed_round = 1
        cfg.cambia_rules = rules

        # Provide real SimpleNamespace for system (not MagicMock — avoids
        # TypeError: '>=' not supported between int and MagicMock)
        system = SimpleNamespace()
        system.recursion_limit = 200
        cfg.system = system

        # Provide real agent_params
        agent_params = SimpleNamespace()
        agent_params.memory_level = 1
        agent_params.time_decay_turns = 3
        cfg.agent_params = agent_params

        # Provide real deep_cfr config (not MagicMock — avoids comparison errors)
        deep_cfr = SimpleNamespace()
        deep_cfr.traversal_method = "escher"
        deep_cfr.sampling_method = "outcome"
        deep_cfr.exploration_epsilon = 0.6
        deep_cfr.traversal_depth_limit = 0
        deep_cfr.engine_backend = "go"
        deep_cfr.batch_counterfactuals = True
        deep_cfr.hidden_dim = 64
        deep_cfr.value_hidden_dim = 64
        deep_cfr.validate_inputs = False
        cfg.deep_cfr = deep_cfr

        # Provide real logging config
        logging_cfg = SimpleNamespace()
        logging_cfg.log_level_file = "WARNING"
        logging_cfg.log_level_console = "WARNING"
        logging_cfg.log_dir = "/tmp"
        logging_cfg.log_file_prefix = "cambia"
        logging_cfg.log_max_bytes = 1 * 1024 * 1024
        logging_cfg.log_backup_count = 1
        logging_cfg.log_simulation_traces = False
        logging_cfg.log_archive_enabled = False

        def get_worker_log_level(worker_id, num_workers):
            return "WARNING"

        logging_cfg.get_worker_log_level = get_worker_log_level
        cfg.logging = logging_cfg

        return cfg

    def _make_go_dcfr_config(self):
        """Build DeepCFRConfig for ESCHER with Go engine, small TPS."""
        from src.cfr.deep_trainer import DeepCFRConfig

        return DeepCFRConfig(
            pipeline_training=False,
            num_traversal_threads=1,
            traversal_method="escher",
            engine_backend="go",
            traversals_per_step=3,
            train_steps_per_iteration=1,
            batch_size=4,
            hidden_dim=64,
            value_hidden_dim=64,
            validate_inputs=False,
            batch_counterfactuals=True,
            value_buffer_capacity=1000,
            advantage_buffer_capacity=1000,
            strategy_buffer_capacity=1000,
        )

    @pytest.mark.skipif(
        not __import__("os").path.exists(
            __import__("os").path.join(
                __import__("os").path.dirname(__file__),
                "..",
                "libcambia.so",
            )
        ),
        reason="libcambia.so not found; skipping Go-backend integration test",
    )
    def test_escher_training_loop_2_iterations(self, tmp_path):
        """Full ESCHER training: 2 iterations, TPS=3, verify samples and checkpoint."""
        import os

        from src.cfr.deep_trainer import DeepCFRTrainer

        checkpoint_path = os.path.join(str(tmp_path), "escher_ckpt.pt")
        config = self._make_go_escher_config(checkpoint_path)
        dcfr_config = self._make_go_dcfr_config()

        trainer = DeepCFRTrainer(config=config, deep_cfr_config=dcfr_config)

        # Run 2 training steps
        trainer.train(num_training_steps=2)

        # 1. Verify checkpoint contains value_net state dict
        trainer.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)

        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "value_net_state_dict" in ckpt
        assert ckpt["value_net_state_dict"] is not None

        # 2. Verify value_buffer has samples
        assert trainer.value_buffer is not None
        assert len(trainer.value_buffer) > 0, (
            "Expected value buffer to have samples after 2 ESCHER iterations"
        )

        # 3. Verify value buffer samples have 444-dim features
        from src.encoding import INPUT_DIM

        batch = trainer.value_buffer.sample_batch(min(4, len(trainer.value_buffer)))
        assert batch.features.shape[1] == INPUT_DIM * 2, (
            f"Expected feature dim {INPUT_DIM * 2}, got {batch.features.shape[1]}"
        )
        assert batch.targets.shape[1] == 1, (
            f"Expected target dim 1, got {batch.targets.shape[1]}"
        )

        # 4. Verify all three networks have non-NaN weights
        import torch

        for net_name, net in [
            ("advantage_net", trainer.advantage_net),
            ("strategy_net", trainer.strategy_net),
            ("value_net", trainer.value_net),
        ]:
            for param in net.parameters():
                assert not torch.isnan(param).any(), (
                    f"{net_name} has NaN parameters after training"
                )

        # 5. Verify training completed without error
        assert trainer.total_traversals > 0
