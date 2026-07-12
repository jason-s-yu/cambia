"""
tests/test_desca_cli.py

Tests for DESCA CLI commands, config schema, eval wrapper registry, and F2
carry-forward (sd-cfr / os-mccfr top-level commands).
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner


def _real_config_module():
    """Return the real src.config module, bypassing any conftest stub."""
    saved = sys.modules.pop("src.config", None)
    mod = importlib.import_module("src.config")
    if saved is not None:
        sys.modules["src.config"] = saved
    return mod


# ---------------------------------------------------------------------------
# CLI runner fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cambia_app():
    from src.cli import app

    return app


# ---------------------------------------------------------------------------
# DESCA command - --help (no crash)
# ---------------------------------------------------------------------------


def test_train_desca_help(runner, cambia_app):
    result = runner.invoke(cambia_app, ["train", "desca", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--iterations" in result.output


def test_train_dense_escher_help(runner, cambia_app):
    """Alias dense-escher resolves to the same command."""
    result = runner.invoke(cambia_app, ["train", "dense-escher", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_train_dense_escher_underscore_help(runner, cambia_app):
    """Alias dense_escher resolves to the same command."""
    result = runner.invoke(cambia_app, ["train", "dense_escher", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


# ---------------------------------------------------------------------------
# DESCA command - missing desca config section
# ---------------------------------------------------------------------------


def test_train_desca_missing_desca_section(runner, cambia_app, tmp_path):
    """Invoking train desca with a config that has no [desca] section exits 1."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("cambia_rules:\n" "  use_jokers: 2\n" "deep_cfr:\n" "  device: cpu\n")
    result = runner.invoke(cambia_app, ["train", "desca", "--config", str(cfg)])
    assert result.exit_code == 1
    assert "desca" in result.output.lower()


# ---------------------------------------------------------------------------
# F2 carry-forward: sd-cfr / os-mccfr --help
# ---------------------------------------------------------------------------


def test_train_sd_cfr_help(runner, cambia_app):
    result = runner.invoke(cambia_app, ["train", "sd-cfr", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_train_sdcfr_alias_help(runner, cambia_app):
    """sdcfr and sd_cfr aliases also work."""
    for alias in ("sdcfr", "sd_cfr"):
        result = runner.invoke(cambia_app, ["train", alias, "--help"])
        assert result.exit_code == 0, f"alias '{alias}' failed"


def test_train_os_mccfr_help(runner, cambia_app):
    result = runner.invoke(cambia_app, ["train", "os-mccfr", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_train_osmccfr_alias_help(runner, cambia_app):
    """osmccfr and os_mccfr aliases also work."""
    for alias in ("osmccfr", "os_mccfr"):
        result = runner.invoke(cambia_app, ["train", alias, "--help"])
        assert result.exit_code == 0, f"alias '{alias}' failed"


# ---------------------------------------------------------------------------
# F2: sd-cfr dispatches train_deep with use_sd_cfr=True
# ---------------------------------------------------------------------------


def test_train_sd_cfr_dispatches_with_use_sd_cfr(tmp_path):
    """sd-cfr command passes use_sd_cfr=True to DeepCFRConfig.from_yaml_config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("deep_cfr:\n  device: cpu\n")

    captured_overrides = {}

    class FakeDeepCFRConfig:
        @classmethod
        def from_yaml_config(cls, config, **overrides):
            captured_overrides.update(overrides)
            return MagicMock()

    # DeepCFRConfig is imported inside train_sd_cfr from .cfr.deep_trainer
    with patch("src.cfr.deep_trainer.DeepCFRConfig", FakeDeepCFRConfig):
        with patch("src.main_train.run_deep_training", return_value=0):
            with patch("src.main_train.create_infrastructure", return_value=MagicMock()):
                runner = CliRunner()
                from src.cli import app

                runner.invoke(
                    app, ["train", "sd-cfr", "--config", str(cfg), "--steps", "1"]
                )

    assert captured_overrides.get("use_sd_cfr") is True


# ---------------------------------------------------------------------------
# F2: os-mccfr dispatches train_deep with sampling_method="os"
# ---------------------------------------------------------------------------


def test_train_os_mccfr_dispatches_with_sampling_method_os(tmp_path):
    """os-mccfr command passes sampling_method='os' to DeepCFRConfig.from_yaml_config."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("deep_cfr:\n  device: cpu\n")

    captured_overrides = {}

    class FakeDeepCFRConfig:
        @classmethod
        def from_yaml_config(cls, config, **overrides):
            captured_overrides.update(overrides)
            return MagicMock()

    with patch("src.cfr.deep_trainer.DeepCFRConfig", FakeDeepCFRConfig):
        with patch("src.main_train.run_deep_training", return_value=0):
            with patch("src.main_train.create_infrastructure", return_value=MagicMock()):
                runner = CliRunner()
                from src.cli import app

                runner.invoke(
                    app, ["train", "os-mccfr", "--config", str(cfg), "--steps", "1"]
                )

    assert captured_overrides.get("sampling_method") == "os"


# ---------------------------------------------------------------------------
# Config schema: DESCAConfig validates ablation YAMLs
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dir():
    return Path(__file__).resolve().parent.parent / "config"


def test_desca_phase1_base_yaml_validates(config_dir):
    real_config = _real_config_module()
    cfg = real_config.load_config(str(config_dir / "desca_phase1_base.yaml"))
    assert cfg is not None
    assert cfg.desca is not None
    assert isinstance(cfg.desca, real_config.DESCAConfig)
    assert cfg.desca.encoding_version == 2
    assert cfg.desca.num_abstract_actions == 32
    assert cfg.desca.inner_update == "apcfr_plus"


def test_desca_phase1_apcfr_yaml_validates(config_dir):
    real_config = _real_config_module()
    cfg = real_config.load_config(str(config_dir / "desca_phase1_apcfr.yaml"))
    assert cfg is not None
    assert cfg.desca is not None
    assert cfg.desca.inner_update == "apcfr_plus"
    assert cfg.desca.apcfr_asymmetry == 0.9
    assert cfg.desca.encoding_version == 2


def test_desca_phase1_apcfr_mild_yaml_validates(config_dir):
    real_config = _real_config_module()
    cfg = real_config.load_config(str(config_dir / "desca_phase1_apcfr_mild.yaml"))
    assert cfg is not None
    assert cfg.desca is not None
    assert cfg.desca.inner_update == "apcfr_plus"
    assert cfg.desca.apcfr_asymmetry == 0.5
    assert cfg.desca.encoding_version == 2


def test_desca_phase1_rmplus_yaml_validates(config_dir):
    real_config = _real_config_module()
    cfg = real_config.load_config(str(config_dir / "desca_phase1_rmplus.yaml"))
    assert cfg is not None
    assert cfg.desca is not None
    assert cfg.desca.inner_update == "rm_plus"
    assert cfg.desca.encoding_version == 2


def test_desca_three_way_ablation_configs_all_validate(config_dir):
    """All three ablation configs load without error and have distinct run identities."""
    real_config = _real_config_module()
    cfgs = {
        name: real_config.load_config(str(config_dir / f"desca_phase1_{name}.yaml"))
        for name in ("apcfr", "apcfr_mild", "rmplus")
    }
    for name, cfg in cfgs.items():
        assert cfg is not None, f"{name} config failed to load"
        assert cfg.desca is not None, f"{name} missing desca section"
        assert cfg.desca.encoding_version == 2, f"{name} encoding_version != 2"

    # Distinct asymmetry values for the two APCFR+ variants
    assert cfgs["apcfr"].desca.apcfr_asymmetry != cfgs["apcfr_mild"].desca.apcfr_asymmetry
    # Both APCFR+ variants use apcfr_plus; RM+ uses rm_plus
    assert cfgs["apcfr"].desca.inner_update == "apcfr_plus"
    assert cfgs["apcfr_mild"].desca.inner_update == "apcfr_plus"
    assert cfgs["rmplus"].desca.inner_update == "rm_plus"


def test_desca_stall_detection_defaults(config_dir):
    real_config = _real_config_module()
    cfg = real_config.load_config(str(config_dir / "desca_phase1_base.yaml"))
    sd = cfg.desca.stall_detection
    assert sd.window_size_iters == 50
    assert sd.num_windows == 5
    assert sd.max_iter_abs == 3000


# ---------------------------------------------------------------------------
# AGENT_REGISTRY: desca and dense-escher keys resolve to DESCAAgentWrapper
# ---------------------------------------------------------------------------


def test_agent_registry_desca_key():
    from src.evaluate_agents import AGENT_REGISTRY, DESCAAgentWrapper

    assert "desca" in AGENT_REGISTRY
    assert AGENT_REGISTRY["desca"] is DESCAAgentWrapper


def test_agent_registry_dense_escher_key():
    from src.evaluate_agents import AGENT_REGISTRY, DESCAAgentWrapper

    assert "dense-escher" in AGENT_REGISTRY
    assert AGENT_REGISTRY["dense-escher"] is DESCAAgentWrapper


# ---------------------------------------------------------------------------
# DESCAConfig: num_abstract_actions must equal NUM_ABSTRACT_ACTIONS_2P
# ---------------------------------------------------------------------------


def test_desca_num_abstract_actions_matches_abstraction_layer():
    real_config = _real_config_module()
    from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P

    cfg = real_config.DESCAConfig()
    assert cfg.num_abstract_actions == NUM_ABSTRACT_ACTIONS_2P


def test_desca_inner_update_literal_validation():
    real_config = _real_config_module()
    DESCAConfig = real_config.DESCAConfig

    cfg_apcfr = DESCAConfig(inner_update="apcfr_plus")
    assert cfg_apcfr.inner_update == "apcfr_plus"

    cfg_rm = DESCAConfig(inner_update="rm_plus")
    assert cfg_rm.inner_update == "rm_plus"

    with pytest.raises(Exception):
        DESCAConfig(inner_update="invalid_update")
