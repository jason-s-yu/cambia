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
    """Import DeepCfrConfig from the real config.py (not conftest stub)."""
    import importlib
    real_config = importlib.reload(importlib.import_module("src.config"))
    cls = getattr(real_config, "DeepCfrConfig", None)
    assert cls is not None, "Could not find DeepCfrConfig in src.config"
    return cls


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
    from typing import List, Optional, Union
    namespace = {
        "dataclass": dataclass,
        "INPUT_DIM": INPUT_DIM,
        "NUM_ACTIONS": NUM_ACTIONS,
        "List": List,
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
        import importlib
        from types import SimpleNamespace

        real_config = importlib.reload(importlib.import_module("src.config"))
        pydantic_cfg = real_config.DeepCfrConfig(device=device)
        # Use SimpleNamespace so tests can del attributes for missing-field testing
        deep_cfr = SimpleNamespace(**pydantic_cfg.model_dump())

        config = SimpleNamespace()
        config.deep_cfr = deep_cfr
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

    def test_default_device_is_auto(self):
        """DeepCfrConfig device defaults to auto when not explicitly overridden."""
        from src.cfr.deep_trainer import DeepCFRConfig

        cfg = self._make_config(device="auto")
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


# ---------------------------------------------------------------------------
# Tests: ESCHER value net input dim (Bug 3 validation)
# ---------------------------------------------------------------------------


class TestValueNetInputDim:
    """Tests that HistoryValueNetwork uses correct input dim for each encoding mode.

    Bug 3: deep_trainer.py hardcoded INPUT_DIM * 2 = 444. For EP-PBS mode,
    the value net must use EP_PBS_INPUT_DIM * 2 = 448 instead.
    """

    def test_legacy_value_net_input_dim(self):
        """Legacy encoding: value net input_dim = INPUT_DIM * 2 = 444."""
        from src.networks import HistoryValueNetwork
        from src.encoding import INPUT_DIM

        assert INPUT_DIM == 222
        net = HistoryValueNetwork(
            input_dim=INPUT_DIM * 2, hidden_dim=512, validate_inputs=False
        )
        assert net._input_dim == 444

    def test_ep_pbs_value_net_input_dim(self):
        """EP-PBS encoding: value net input_dim = EP_PBS_INPUT_DIM * 2 = 448."""
        from src.networks import HistoryValueNetwork
        from src.constants import EP_PBS_INPUT_DIM

        assert EP_PBS_INPUT_DIM == 224
        net = HistoryValueNetwork(
            input_dim=EP_PBS_INPUT_DIM * 2, hidden_dim=512, validate_inputs=False
        )
        assert net._input_dim == 448

    def test_value_net_accepts_both_dims(self):
        """HistoryValueNetwork must accept both 444 and 448 input dims without error."""
        import torch
        from src.networks import HistoryValueNetwork

        for dim in (444, 448):
            net = HistoryValueNetwork(input_dim=dim, hidden_dim=512, validate_inputs=False)
            x = torch.randn(4, dim)
            out = net(x)
            assert out.shape == (4, 1), f"Expected (4,1) output for dim={dim}, got {out.shape}"

    @pytest.mark.xfail(
        strict=False,
        reason="Bug 3 fix pending impl-1: deep_trainer.py should use dynamic value_input_dim",
    )
    def test_trainer_uses_dynamic_value_input_dim(self):
        """deep_trainer.py must compute value_input_dim dynamically (not hardcode INPUT_DIM * 2)."""
        source = _read_source("cfr/deep_trainer.py")
        # After Bug 3 fix, the trainer should compute base_dim * 2 based on encoding_mode
        assert "value_input_dim" in source
