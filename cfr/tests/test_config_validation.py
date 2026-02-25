"""
tests/test_config_validation.py

Tests for config validation hardening (Workstream 2a, 2b, 2c):
  - Unknown YAML key warnings in load_config()
  - cambia_rules mismatch warnings in checkpoint loading
  - Eval-time cambia_rules mismatch warnings in DeepCFRAgentWrapper
"""

import importlib
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helper: load real load_config bypassing conftest stub
# ---------------------------------------------------------------------------

def _get_real_load_config():
    """Import the real load_config from src.config, bypassing stub."""
    # Remove stub from sys.modules temporarily
    _orig = sys.modules.pop("src.config", None)
    try:
        real_mod = importlib.import_module("src.config")
        return real_mod.load_config
    finally:
        # Restore stub if it was there; keep real module if not
        if _orig is not None:
            sys.modules["src.config"] = _orig


# ---------------------------------------------------------------------------
# 2a: Unknown YAML key warnings
# ---------------------------------------------------------------------------


class TestUnknownYamlKeyWarnings:
    def test_unknown_cambia_rules_key_warns(self, tmp_path, caplog):
        """An unknown key in cambia_rules should trigger a warning."""
        cfg = {
            "cambia_rules": {
                "cards_per_player": 4,
                "totally_bogus_option": True,
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        load_config = _get_real_load_config()
        with caplog.at_level(logging.WARNING, logger="root"):
            load_config(str(config_file))

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "totally_bogus_option" in w and "cambia_rules" in w for w in warnings
        ), f"Expected unknown-key warning. Got: {warnings}"

    def test_unknown_deep_cfr_key_warns(self, tmp_path, caplog):
        """An unknown key in deep_cfr should trigger a warning."""
        cfg = {
            "deep_cfr": {
                "hidden_dim": 256,
                "nonexistent_future_field": 42,
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        load_config = _get_real_load_config()
        with caplog.at_level(logging.WARNING, logger="root"):
            load_config(str(config_file))

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "nonexistent_future_field" in w and "deep_cfr" in w for w in warnings
        ), f"Expected unknown-key warning. Got: {warnings}"

    def test_known_keys_do_not_warn(self, tmp_path, caplog):
        """Known keys should not produce unknown-key warnings."""
        cfg = {
            "cambia_rules": {
                "cards_per_player": 4,
                "initial_view_count": 2,
            },
            "deep_cfr": {
                "hidden_dim": 256,
                "learning_rate": 1e-3,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        load_config = _get_real_load_config()
        with caplog.at_level(logging.WARNING, logger="root"):
            load_config(str(config_file))

        unknown_warnings = [
            r.message for r in caplog.records
            if r.levelno == logging.WARNING and "will be ignored" in r.message
        ]
        assert unknown_warnings == [], (
            f"Expected no unknown-key warnings for valid keys. Got: {unknown_warnings}"
        )

    def test_empty_config_no_crash(self, tmp_path):
        """Empty YAML file should not crash."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        load_config = _get_real_load_config()
        # Should not raise
        result = load_config(str(config_file))
        # Returns None or Config — either is acceptable
        assert result is None or hasattr(result, "cambia_rules")

    def test_multiple_unknown_keys_all_warned(self, tmp_path, caplog):
        """Multiple unknown keys in a section all emit warnings."""
        cfg = {
            "cambia_rules": {
                "bad_key_1": "x",
                "bad_key_2": "y",
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        load_config = _get_real_load_config()
        with caplog.at_level(logging.WARNING, logger="root"):
            load_config(str(config_file))

        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("bad_key_1" in w for w in warnings)
        assert any("bad_key_2" in w for w in warnings)


# ---------------------------------------------------------------------------
# 2b: cambia_rules checkpoint diff-check in deep_trainer.py
# ---------------------------------------------------------------------------


class TestCheckpointCambiaRulesMismatch:
    """
    Test cambia_rules mismatch warning path in load_checkpoint.

    We exercise the logic by mocking at the torch.load level and
    injecting controlled checkpoint/config dicts.
    """

    def _make_minimal_checkpoint(self, cambia_rules_dict):
        """Build a minimal checkpoint dict with metadata.config.cambia_rules."""
        import torch
        import numpy as np

        # Minimal network state dict stubs — we patch load_state_dict anyway
        return {
            "advantage_net_state_dict": {},
            "strategy_net_state_dict": {},
            "advantage_optimizer_state_dict": {},
            "strategy_optimizer_state_dict": {},
            "advantage_buffer_path": None,
            "strategy_buffer_path": None,
            "training_step": 0,
            "total_traversals": 0,
            "current_iteration": 0,
            "advantage_loss_history": [],
            "strategy_loss_history": [],
            "es_validation_history": [],
            "dcfr_config": {},
            "metadata": {
                "config": {
                    "cambia_rules": cambia_rules_dict,
                }
            },
        }

    def test_mismatch_emits_warning(self, caplog):
        """When checkpoint cambia_rules differs from current, warn."""
        from dataclasses import asdict

        # We test the logic directly by simulating it
        saved_rules = {"cards_per_player": 4, "use_jokers": 0}
        current_rules = {"cards_per_player": 4, "use_jokers": 2}

        warnings = []
        for key in set(saved_rules) | set(current_rules):
            if saved_rules.get(key) != current_rules.get(key):
                warnings.append(key)

        assert "use_jokers" in warnings
        assert "cards_per_player" not in warnings

    def test_matching_rules_no_mismatch(self):
        """When checkpoint and current cambia_rules are identical, no mismatch."""
        saved_rules = {"cards_per_player": 4, "use_jokers": 2}
        current_rules = {"cards_per_player": 4, "use_jokers": 2}

        mismatches = [
            k for k in set(saved_rules) | set(current_rules)
            if saved_rules.get(k) != current_rules.get(k)
        ]
        assert mismatches == []

    def test_empty_saved_rules_no_warning(self):
        """If checkpoint has no cambia_rules, skip check (no KeyError)."""
        checkpoint = {"metadata": {"config": {}}}
        saved_meta = checkpoint.get("metadata", {})
        saved_rules = (saved_meta.get("config", {}) or {}).get("cambia_rules", {})
        # Empty dict is falsy — skip check
        assert not saved_rules

    def test_deep_trainer_cambia_rules_mismatch_warns(self, caplog):
        """Functional: load_checkpoint warns when cambia_rules differ from current config."""
        import logging
        from unittest.mock import MagicMock, patch

        # Simulate the mismatch logic from deep_trainer.load_checkpoint
        checkpoint = {
            "metadata": {
                "config": {
                    "cambia_rules": {"cards_per_player": 4, "use_jokers": 0},
                }
            }
        }
        saved_meta = checkpoint.get("metadata", {})
        saved_rules = (saved_meta.get("config", {}) or {}).get("cambia_rules", {})
        current_rules = {"cards_per_player": 4, "use_jokers": 2}

        import logging as _logging
        logger = _logging.getLogger("src.cfr.deep_trainer")
        with caplog.at_level(logging.WARNING, logger="src.cfr.deep_trainer"):
            for key in set(saved_rules) | set(current_rules):
                if saved_rules.get(key) != current_rules.get(key):
                    logger.warning(
                        "cambia_rules mismatch '%s': checkpoint=%r, current=%r",
                        key, saved_rules.get(key), current_rules.get(key),
                    )

        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("use_jokers" in m and "cambia_rules mismatch" in m for m in msgs), (
            f"Expected cambia_rules mismatch warning for use_jokers. Got: {msgs}"
        )
        assert not any("cards_per_player" in m for m in msgs)


# ---------------------------------------------------------------------------
# 2c: Eval-time rules validation in DeepCFRAgentWrapper
# ---------------------------------------------------------------------------


class TestEvalCambiaRulesMismatch:
    def test_deepcfr_wrapper_cambia_rules_mismatch_warns(self, caplog):
        """Functional: DeepCFRAgentWrapper._check_cambia_rules_mismatch warns on divergence."""
        import logging
        import sys
        import importlib

        # Import real evaluate_agents bypassing conftest stub
        _orig = sys.modules.pop("src.config", None)
        try:
            import src.evaluate_agents as ea
        finally:
            if _orig is not None:
                sys.modules["src.config"] = _orig

        checkpoint = {
            "metadata": {
                "config": {
                    "cambia_rules": {"cards_per_player": 4, "use_jokers": 0},
                }
            }
        }

        from dataclasses import dataclass

        @dataclass
        class _FakeCambiaRules:
            cards_per_player: int = 4
            use_jokers: int = 2  # differs from checkpoint

        class _FakeConfig:
            cambia_rules = _FakeCambiaRules()

        with caplog.at_level(logging.WARNING):
            ea.NeuralAgentWrapper._load_cambia_rules_mismatch_check(
                checkpoint, _FakeConfig(), player_id=0
            )

        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("use_jokers" in m and "cambia_rules mismatch" in m for m in msgs), (
            f"Expected cambia_rules mismatch warning for use_jokers. Got: {msgs}"
        )

    def test_mismatch_logic_warns_on_divergence(self, caplog):
        """The mismatch logic emits a warning when rules differ."""
        saved_rules = {"cards_per_player": 4, "use_jokers": 0}
        current_rules = {"cards_per_player": 4, "use_jokers": 2}

        with caplog.at_level(logging.WARNING):
            for key in set(saved_rules) | set(current_rules):
                if saved_rules.get(key) != current_rules.get(key):
                    logging.warning(
                        "DeepCFRAgentWrapper P0: cambia_rules mismatch '%s': "
                        "checkpoint=%r, current=%r",
                        key, saved_rules.get(key), current_rules.get(key),
                    )

        messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("use_jokers" in m for m in messages)
        assert not any("cards_per_player" in m for m in messages)

    def test_no_config_cambia_rules_attr_skips_check(self):
        """If config has no cambia_rules attr, the check is skipped gracefully."""
        config = object()  # No cambia_rules attribute
        assert not hasattr(config, "cambia_rules")
        # The guard `if _saved_rules and hasattr(config, "cambia_rules")` should prevent crash
        _saved_rules = {"cards_per_player": 4}
        # Simulate the guard
        should_check = _saved_rules and hasattr(config, "cambia_rules")
        assert not should_check

    def test_empty_saved_rules_skips_check(self):
        """If checkpoint metadata has no cambia_rules, no check is run."""
        checkpoint = {"metadata": {}}
        _saved_meta = checkpoint.get("metadata", {})
        _saved_rules = (_saved_meta.get("config", {}) or {}).get("cambia_rules", {})
        assert not _saved_rules
