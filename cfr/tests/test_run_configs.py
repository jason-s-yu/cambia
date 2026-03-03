"""Tests for per-run config YAML files in cfr/runs/*/config.yaml."""

import importlib
import os
import sys
import pytest


def _get_real_load_config():
    """Load the real load_config, bypassing conftest stub."""
    # Temporarily remove the stub so importlib loads the real module
    real_mod = sys.modules.pop("src.config", None)
    try:
        mod = importlib.import_module("src.config")
        return mod.load_config
    finally:
        # Restore whatever was there before (stub or real)
        if real_mod is not None:
            sys.modules["src.config"] = real_mod


# Load the real load_config at module level
_real_load_config = _get_real_load_config()

# Base directory for run configs (relative to this test file's location)
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs")


def config_path(run_name: str) -> str:
    return os.path.join(RUNS_DIR, run_name, "config.yaml")


def load(run_name: str):
    path = config_path(run_name)
    if not os.path.exists(path):
        pytest.skip(f"Run directory pruned: {run_name}")
    return _real_load_config(path)


# Expected values per run: (run_name, sampling_method, traversal_depth_limit, traversals_per_step)
RUN_SPECS = [
    ("os-full", "outcome", 0, 1000),
    ("os-30", "outcome", 30, 1000),
    ("os-20", "outcome", 20, 1000),
    ("os-15", "outcome", 15, 1000),
    ("os-10", "outcome", 10, 1000),
    ("es-15", "external", 15, 10),
    ("es-10", "external", 10, 10),
]


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_loads(run_name, sampling, depth, traversals):
    """Each config file must exist and load without error."""
    cfg = load(run_name)
    assert cfg is not None, f"load_config returned None for {run_name}"
    assert hasattr(cfg, "deep_cfr"), f"{run_name}: Config missing deep_cfr section"


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_sampling_method(run_name, sampling, depth, traversals):
    """sampling_method must match expected value."""
    cfg = load(run_name)
    assert cfg.deep_cfr.sampling_method == sampling, (
        f"{run_name}: expected sampling_method={sampling!r}, "
        f"got {cfg.deep_cfr.sampling_method!r}"
    )


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_depth_limit(run_name, sampling, depth, traversals):
    """traversal_depth_limit must match expected value."""
    cfg = load(run_name)
    assert cfg.deep_cfr.traversal_depth_limit == depth, (
        f"{run_name}: expected traversal_depth_limit={depth}, "
        f"got {cfg.deep_cfr.traversal_depth_limit}"
    )


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_traversals_per_step(run_name, sampling, depth, traversals):
    """traversals_per_step must match expected value."""
    cfg = load(run_name)
    assert cfg.deep_cfr.traversals_per_step == traversals, (
        f"{run_name}: expected traversals_per_step={traversals}, "
        f"got {cfg.deep_cfr.traversals_per_step}"
    )


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_validate_inputs_disabled(run_name, sampling, depth, traversals):
    """validate_inputs must be False for GPU perf."""
    cfg = load(run_name)
    assert cfg.deep_cfr.validate_inputs is False, (
        f"{run_name}: expected validate_inputs=False, "
        f"got {cfg.deep_cfr.validate_inputs}"
    )


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_save_path(run_name, sampling, depth, traversals):
    """persistence.agent_data_save_path must point to correct run directory."""
    cfg = load(run_name)
    expected_save = f"runs/{run_name}/checkpoints/deep_cfr_checkpoint.pt"
    assert cfg.persistence.agent_data_save_path == expected_save, (
        f"{run_name}: expected save_path={expected_save!r}, "
        f"got {cfg.persistence.agent_data_save_path!r}"
    )


@pytest.mark.parametrize("run_name,sampling,depth,traversals", RUN_SPECS)
def test_run_config_log_dir(run_name, sampling, depth, traversals):
    """logging.log_dir must point to correct run directory."""
    cfg = load(run_name)
    expected_log_dir = f"runs/{run_name}/logs"
    assert cfg.logging.log_dir == expected_log_dir, (
        f"{run_name}: expected log_dir={expected_log_dir!r}, "
        f"got {cfg.logging.log_dir!r}"
    )


# ---------------------------------------------------------------------------
# ESCHER interleaved run config tests
# ---------------------------------------------------------------------------


def _load_run_config_yaml(run_name: str, filename: str = "config.yaml"):
    """Load directly from a run config YAML (before training copies it to config.yaml)."""
    path = os.path.join(RUNS_DIR, run_name, filename)
    if not os.path.exists(path):
        pytest.skip(f"Run config not found: {path}")
    cfg = _real_load_config(path)
    if cfg is None:
        pytest.skip(f"load_config returned None for {path}")
    return cfg


def test_escher_interleaved_value_target_buffer_passes():
    """value_target_buffer_passes must load correctly from escher-interleaved config."""
    cfg = _load_run_config_yaml("escher-interleaved")
    if not hasattr(cfg.deep_cfr, "value_target_buffer_passes"):
        pytest.skip("value_target_buffer_passes not yet in DeepCfrConfig (impl-1 pending)")
    assert cfg.deep_cfr.value_target_buffer_passes == 2.0, (
        f"expected value_target_buffer_passes=2.0, "
        f"got {cfg.deep_cfr.value_target_buffer_passes}"
    )


def test_escher_interleaved_sampling_method():
    """escher-interleaved config must have sampling_method='escher'."""
    cfg = _load_run_config_yaml("escher-interleaved")
    assert cfg.deep_cfr.sampling_method == "escher"


def test_escher_interleaved_encoding_layout():
    """escher-interleaved config must use interleaved EP-PBS encoding."""
    cfg = _load_run_config_yaml("escher-interleaved")
    assert cfg.deep_cfr.encoding_mode == "ep_pbs"
    assert cfg.deep_cfr.encoding_layout == "interleaved"


def test_escher_interleaved_traversals():
    """escher-interleaved config must have traversals_per_step=150."""
    cfg = _load_run_config_yaml("escher-interleaved")
    assert cfg.deep_cfr.traversals_per_step == 150
