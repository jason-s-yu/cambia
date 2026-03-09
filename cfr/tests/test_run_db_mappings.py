"""Tests for infer_algorithm gtcfr/sog detection and algo mapping tables."""

import pytest

from src.run_db import (
    ALGO_TO_AGENT_TYPE,
    ALGO_TO_CHECKPOINT_PREFIX,
    algo_to_agent_type,
    algo_to_checkpoint_prefix,
    infer_algorithm,
)


# --- infer_algorithm: checkpoint_keys detection ---


def test_infer_algorithm_gtcfr_from_checkpoint_keys():
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "optimizer_state_dict", "epoch"},
    )
    assert result == "gtcfr"


def test_infer_algorithm_sog_from_checkpoint_keys():
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "sog_metadata", "optimizer_state_dict"},
    )
    assert result == "sog"


def test_infer_algorithm_sog_over_gtcfr():
    """When both cvpn_state_dict and sog_metadata present, sog wins."""
    result = infer_algorithm(
        config_dict={},
        checkpoint_keys={"cvpn_state_dict", "sog_metadata"},
    )
    assert result == "sog"


# --- infer_algorithm: config dict detection ---


def test_infer_algorithm_gtcfr_from_config():
    result = infer_algorithm(
        config_dict={"deep_cfr": {"gtcfr_epochs": 200}},
    )
    assert result == "gtcfr"


def test_infer_algorithm_sog_from_config():
    result = infer_algorithm(
        config_dict={"deep_cfr": {"sog_epochs": 500}},
    )
    assert result == "sog"


# --- algo_to_agent_type ---


def test_algo_to_agent_type_all_entries():
    expected = {
        "rebel": "rebel",
        "os-mccfr": "deep_cfr",
        "es-mccfr": "deep_cfr",
        "escher": "escher",
        "sd-cfr": "sd_cfr",
        "gtcfr": "gtcfr",
        "sog": "sog_inference",
        "psro": "deep_cfr",
    }
    for algo, agent_type in expected.items():
        assert algo_to_agent_type(algo) == agent_type, f"Failed for {algo}"
        assert ALGO_TO_AGENT_TYPE[algo] == agent_type, f"Dict mismatch for {algo}"


def test_algo_to_agent_type_fallback():
    assert algo_to_agent_type("unknown-algo") == "deep_cfr"


# --- algo_to_checkpoint_prefix ---


def test_algo_to_checkpoint_prefix_all_entries():
    expected = {
        "rebel": "rebel_checkpoint",
        "os-mccfr": "deep_cfr_checkpoint",
        "es-mccfr": "deep_cfr_checkpoint",
        "escher": "deep_cfr_checkpoint",
        "sd-cfr": "deep_cfr_checkpoint",
        "gtcfr": "gtcfr_checkpoint",
        "sog": "sog_checkpoint",
        "psro": "deep_cfr_checkpoint",
    }
    for algo, prefix in expected.items():
        assert algo_to_checkpoint_prefix(algo) == prefix, f"Failed for {algo}"
        assert ALGO_TO_CHECKPOINT_PREFIX[algo] == prefix, f"Dict mismatch for {algo}"


def test_algo_to_checkpoint_prefix_fallback():
    assert algo_to_checkpoint_prefix("unknown-algo") == "deep_cfr_checkpoint"
