"""Tests for device configuration: auto-resolution, backward compat, and CLI mapping."""

import os
import re
import tempfile
from dataclasses import dataclass

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers: extract real dataclass definitions from source files
# ---------------------------------------------------------------------------

def _read_source(relative_path: str) -> str:
    """Read a source file relative to cfr/src/."""
    base = os.path.join(os.path.dirname(__file__), "..", "src")
    with open(os.path.join(base, relative_path)) as f:
        return f.read()


def _extract_deep_cfr_config_from_config_py() -> type:
    """Extract the DeepCfrConfig dataclass from config.py (real module, not stub)."""
    source = _read_source("config.py")
    match = re.search(
        r"(@dataclass\nclass DeepCfrConfig:.*?)(?=\n@|\ndef |\nclass )",
        source,
        re.DOTALL,
    )
    assert match, "Could not find DeepCfrConfig class in config.py"
    class_src = match.group(1)
    from typing import Optional, Union
    namespace = {
        "dataclass": dataclass,
        "Optional": Optional,
        "Union": Union,
    }
    exec(compile(class_src, "<DeepCfrConfig>", "exec"), namespace)
    return namespace["DeepCfrConfig"]


def _extract_deep_cfr_config_from_trainer() -> type:
    """Extract the DeepCFRConfig dataclass from deep_trainer.py."""
    source = _read_source("cfr/deep_trainer.py")
    match = re.search(
        r"(@dataclass\nclass DeepCFRConfig:.*?)(?=\n@|\ndef |\nclass )",
        source,
        re.DOTALL,
    )
    assert match, "Could not find DeepCFRConfig class in deep_trainer.py"
    class_src = match.group(1)
    from src.encoding import INPUT_DIM, NUM_ACTIONS
    from typing import Optional, Union
    namespace = {
        "dataclass": dataclass,
        "INPUT_DIM": INPUT_DIM,
        "NUM_ACTIONS": NUM_ACTIONS,
        "Optional": Optional,
        "Union": Union,
    }
    exec(compile(class_src, "<DeepCFRConfig>", "exec"), namespace)
    return namespace["DeepCFRConfig"]


# ---------------------------------------------------------------------------
# Tests: _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    """Tests for _resolve_device() in deep_trainer.py."""

    def test_cpu_passthrough(self):
        from src.cfr.deep_trainer import _resolve_device
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        from src.cfr.deep_trainer import _resolve_device
        assert _resolve_device("cuda") == "cuda"

    def test_xpu_passthrough(self):
        from src.cfr.deep_trainer import _resolve_device
        assert _resolve_device("xpu") == "xpu"

    def test_cuda_colon_zero_passthrough(self):
        from src.cfr.deep_trainer import _resolve_device
        assert _resolve_device("cuda:0") == "cuda:0"

    def test_auto_returns_valid_device(self):
        from src.cfr.deep_trainer import _resolve_device
        result = _resolve_device("auto")
        assert isinstance(result, str)
        assert result in ("cpu", "cuda", "xpu")


# ---------------------------------------------------------------------------
# Tests: config.py DeepCfrConfig (source-level, not stub)
# ---------------------------------------------------------------------------

class TestDeepCfrConfigDevice:
    """Tests for config.py DeepCfrConfig device field (real source, not conftest stub)."""

    @pytest.fixture(scope="class")
    def DeepCfrConfig(self):
        return _extract_deep_cfr_config_from_config_py()

    def test_default_is_auto(self, DeepCfrConfig):
        cfg = DeepCfrConfig()
        assert cfg.device == "auto"

    def test_no_use_gpu_field(self, DeepCfrConfig):
        """use_gpu field should no longer exist on DeepCfrConfig."""
        cfg = DeepCfrConfig()
        assert not hasattr(cfg, "use_gpu")

    def test_accepts_explicit_device(self, DeepCfrConfig):
        cfg = DeepCfrConfig(device="xpu")
        assert cfg.device == "xpu"


# ---------------------------------------------------------------------------
# Tests: deep_trainer.py DeepCFRConfig (source-level)
# ---------------------------------------------------------------------------

class TestDeepCFRConfigDevice:
    """Tests for deep_trainer.py DeepCFRConfig device field."""

    @pytest.fixture(scope="class")
    def DeepCFRConfig(self):
        return _extract_deep_cfr_config_from_trainer()

    def test_default_is_auto(self, DeepCFRConfig):
        cfg = DeepCFRConfig()
        assert cfg.device == "auto"

    def test_explicit_cpu(self, DeepCFRConfig):
        cfg = DeepCFRConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_explicit_xpu(self, DeepCFRConfig):
        cfg = DeepCFRConfig(device="xpu")
        assert cfg.device == "xpu"

    def test_no_use_gpu_field(self, DeepCFRConfig):
        cfg = DeepCFRConfig()
        assert not hasattr(cfg, "use_gpu")


# ---------------------------------------------------------------------------
# Tests: YAML backward compatibility (use_gpu -> device)
# ---------------------------------------------------------------------------

class TestYamlBackwardCompat:
    """Tests that YAML configs with use_gpu are loaded correctly."""

    def _write_and_load(self, deep_cfr_dict):
        import importlib
        # Reload the real config module (conftest stub returns None from load_config)
        real_config = importlib.reload(importlib.import_module("src.config"))

        config_dict = {"deep_cfr": deep_cfr_dict}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_dict, f)
            f.flush()
            try:
                cfg = real_config.load_config(f.name)
            finally:
                os.unlink(f.name)
        return cfg

    def test_use_gpu_true_maps_to_cuda(self):
        cfg = self._write_and_load({"use_gpu": True})
        assert cfg.deep_cfr.device == "cuda"

    def test_use_gpu_false_maps_to_cpu(self):
        cfg = self._write_and_load({"use_gpu": False})
        assert cfg.deep_cfr.device == "cpu"

    def test_explicit_device_takes_precedence(self):
        cfg = self._write_and_load({"use_gpu": True, "device": "xpu"})
        assert cfg.deep_cfr.device == "xpu"

    def test_device_auto_without_use_gpu(self):
        cfg = self._write_and_load({"sampling_method": "outcome"})
        assert cfg.deep_cfr.device == "auto"

    def test_device_cpu_explicit(self):
        cfg = self._write_and_load({"device": "cpu"})
        assert cfg.deep_cfr.device == "cpu"


# ---------------------------------------------------------------------------
# Tests: DeepCFRConfig.from_yaml_config
# ---------------------------------------------------------------------------

class TestFromYamlConfig:
    """Tests for DeepCFRConfig.from_yaml_config device propagation."""

    def _make_config(self, device="auto"):
        from types import SimpleNamespace

        config = SimpleNamespace()
        config.deep_cfr = SimpleNamespace()
        config.deep_cfr.hidden_dim = 256
        config.deep_cfr.dropout = 0.1
        config.deep_cfr.learning_rate = 1e-3
        config.deep_cfr.batch_size = 2048
        config.deep_cfr.train_steps_per_iteration = 4000
        config.deep_cfr.alpha = 1.5
        config.deep_cfr.traversals_per_step = 1000
        config.deep_cfr.advantage_buffer_capacity = 2_000_000
        config.deep_cfr.strategy_buffer_capacity = 2_000_000
        config.deep_cfr.save_interval = 10
        config.deep_cfr.device = device
        config.deep_cfr.sampling_method = "outcome"
        config.deep_cfr.exploration_epsilon = 0.6
        config.deep_cfr.engine_backend = "python"
        config.deep_cfr.es_validation_interval = 10
        config.deep_cfr.es_validation_depth = 10
        config.deep_cfr.es_validation_traversals = 1000
        config.deep_cfr.pipeline_training = True
        config.deep_cfr.use_amp = False
        config.deep_cfr.use_compile = False
        config.deep_cfr.num_traversal_threads = 1
        config.deep_cfr.validate_inputs = True
        config.deep_cfr.traversal_depth_limit = 0
        config.deep_cfr.max_tasks_per_child = "auto"
        config.deep_cfr.worker_memory_budget_pct = 0.10
        return config

    def test_propagates_auto(self):
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = self._make_config(device="auto")
        dcfr = DeepCFRConfig.from_yaml_config(cfg)
        assert dcfr.device == "auto"

    def test_propagates_xpu(self):
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = self._make_config(device="xpu")
        dcfr = DeepCFRConfig.from_yaml_config(cfg)
        assert dcfr.device == "xpu"

    def test_cli_override_device(self):
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = self._make_config(device="auto")
        dcfr = DeepCFRConfig.from_yaml_config(cfg, device="cpu")
        assert dcfr.device == "cpu"

    def test_missing_device_defaults_to_auto(self):
        """If deep_cfg has no device attr, from_yaml_config defaults to auto."""
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = self._make_config(device="auto")
        del cfg.deep_cfr.device
        dcfr = DeepCFRConfig.from_yaml_config(cfg)
        assert dcfr.device == "auto"


# ---------------------------------------------------------------------------
# Tests: DeepCFRTrainer device init
# ---------------------------------------------------------------------------

class TestTrainerDeviceInit:
    """Tests that DeepCFRTrainer resolves device correctly."""

    def test_trainer_uses_cpu_when_device_cpu(self):
        from src.cfr.deep_trainer import DeepCFRConfig, DeepCFRTrainer
        from types import SimpleNamespace
        import torch

        config = SimpleNamespace()
        config.cfr_training = SimpleNamespace(num_iterations=1, num_workers=1)
        config.persistence = SimpleNamespace(agent_data_save_path="test_ckpt.pt")

        dcfr = DeepCFRConfig(device="cpu")
        trainer = DeepCFRTrainer(config=config, deep_cfr_config=dcfr)
        assert trainer.device == torch.device("cpu")

    def test_trainer_amp_disabled_on_cpu(self):
        """AMP should be disabled when device is cpu even if use_amp=True."""
        from src.cfr.deep_trainer import DeepCFRConfig, DeepCFRTrainer
        from types import SimpleNamespace

        config = SimpleNamespace()
        config.cfr_training = SimpleNamespace(num_iterations=1, num_workers=1)
        config.persistence = SimpleNamespace(agent_data_save_path="test_ckpt.pt")

        dcfr = DeepCFRConfig(device="cpu", use_amp=True)
        trainer = DeepCFRTrainer(config=config, deep_cfr_config=dcfr)
        assert trainer.use_amp is False


# ---------------------------------------------------------------------------
# Tests: source-level verification of _resolve_device function
# ---------------------------------------------------------------------------

class TestResolveDeviceSource:
    """Verify _resolve_device exists and handles xpu in source."""

    def test_function_exists_in_source(self):
        source = _read_source("cfr/deep_trainer.py")
        assert "def _resolve_device(" in source

    def test_xpu_branch_in_source(self):
        source = _read_source("cfr/deep_trainer.py")
        assert "torch.xpu" in source

    def test_auto_branch_in_source(self):
        source = _read_source("cfr/deep_trainer.py")
        assert 'device == "auto"' in source
