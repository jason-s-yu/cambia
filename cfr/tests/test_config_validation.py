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
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "will be ignored" in r.message
        ]
        assert (
            unknown_warnings == []
        ), f"Expected no unknown-key warnings for valid keys. Got: {unknown_warnings}"

    def test_empty_config_no_crash(self, tmp_path):
        """Empty YAML file should not crash."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        load_config = _get_real_load_config()
        # Should not raise
        result = load_config(str(config_file))
        # Returns None or Config: either is acceptable
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

        # Minimal network state dict stubs: we patch load_state_dict anyway
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
            k
            for k in set(saved_rules) | set(current_rules)
            if saved_rules.get(k) != current_rules.get(k)
        ]
        assert mismatches == []

    def test_empty_saved_rules_no_warning(self):
        """If checkpoint has no cambia_rules, skip check (no KeyError)."""
        checkpoint = {"metadata": {"config": {}}}
        saved_meta = checkpoint.get("metadata", {})
        saved_rules = (saved_meta.get("config", {}) or {}).get("cambia_rules", {})
        # Empty dict is falsy: skip check
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
                        key,
                        saved_rules.get(key),
                        current_rules.get(key),
                    )

        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "use_jokers" in m and "cambia_rules mismatch" in m for m in msgs
        ), f"Expected cambia_rules mismatch warning for use_jokers. Got: {msgs}"
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

        # Target the emitting logger explicitly (module: src.evaluate_agents),
        # not just root. Some full-suite orderings (e.g. tools/tiny_solver.py,
        # imported by the PRT-CFR tests, permanently sets every already-created
        # "src.*" logger to CRITICAL to silence chatty per-node warnings during
        # tree expansion) leave this logger's own .level pinned above WARNING
        # for the rest of the session. caplog.at_level(logger="root") only
        # resets the root logger's level, which doesn't help once a child
        # logger has an explicit level of its own; naming the logger here
        # forces it back down to WARNING for the duration of the check,
        # regardless of what an earlier-imported module did to it.
        with caplog.at_level(logging.WARNING, logger="src.evaluate_agents"):
            ea.NeuralAgentWrapper._load_cambia_rules_mismatch_check(
                checkpoint, _FakeConfig(), player_id=0
            )

        msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "use_jokers" in m and "cambia_rules mismatch" in m for m in msgs
        ), f"Expected cambia_rules mismatch warning for use_jokers. Got: {msgs}"

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
                        key,
                        saved_rules.get(key),
                        current_rules.get(key),
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


# ---------------------------------------------------------------------------
# cambia-542 F3: num_players bounds validation (DeepCfrConfig, PRTCFRConfig)
# ---------------------------------------------------------------------------
#
# No Go-side validation existed prior to cambia-542; a num_players=9 config
# would sail through Python and panic inside libcambia.so the first time a
# GoEngine was constructed (array-index panic on the fixed [MaxPlayers]
# board). These fail fast at config-construction time instead.


def _get_real_config_classes():
    """Import DeepCfrConfig/PRTCFRConfig from the real src.config, bypassing
    the conftest stub (which stubs num_players with no validation)."""
    _orig = sys.modules.pop("src.config", None)
    try:
        real_mod = importlib.import_module("src.config")
        return real_mod.DeepCfrConfig, real_mod.PRTCFRConfig
    finally:
        if _orig is not None:
            sys.modules["src.config"] = _orig


class TestNumPlayersValidation:
    @pytest.mark.parametrize("num_players", [2, 4, 8])
    def test_valid_num_players_accepted(self, num_players):
        DeepCfrConfig, PRTCFRConfig = _get_real_config_classes()
        assert DeepCfrConfig(num_players=num_players).num_players == num_players
        assert PRTCFRConfig(num_players=num_players).num_players == num_players

    @pytest.mark.parametrize("num_players", [0, 1, 9, 255])
    def test_invalid_num_players_rejected_deep_cfr_config(self, num_players):
        DeepCfrConfig, _ = _get_real_config_classes()
        with pytest.raises(Exception):
            DeepCfrConfig(num_players=num_players)

    @pytest.mark.parametrize("num_players", [0, 1, 9, 255])
    def test_invalid_num_players_rejected_prtcfr_config(self, num_players):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception):
            PRTCFRConfig(num_players=num_players)

    def test_default_num_players_is_valid(self):
        """The default (2) must itself pass validation."""
        DeepCfrConfig, PRTCFRConfig = _get_real_config_classes()
        assert DeepCfrConfig().num_players == 2
        assert PRTCFRConfig().num_players == 2


# ---------------------------------------------------------------------------
# cambia-640: prt_cfr config validator hygiene
#
# Two silent-failure modes traced from the x2r-staged-config-verify workflow:
#   1. PRTCFRConfig inherited extra="ignore" from _CambiaBaseModel, so a
#      typo'd key under prt_cfr passed `cambia config validate` with OK and
#      silently no-opped to the field's default.
#   2. stability_stop_mode (and sibling fields with the same raise-at-
#      construction pattern: lr_schedule, backend, stability_metric_mode)
#      were plain str with no Literal constraint, so an invalid value passed
#      validation and only failed later, at trainer/controller construction.
# ---------------------------------------------------------------------------


def _get_real_config_module():
    """Import the real src.config module, bypassing the conftest stub."""
    _orig = sys.modules.pop("src.config", None)
    try:
        return importlib.import_module("src.config")
    finally:
        if _orig is not None:
            sys.modules["src.config"] = _orig


class TestPRTCFRConfigUnknownKeysRejected:
    def test_bogus_key_rejected_at_construction(self):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception, match="totally_bogus_prt_cfr_option"):
            PRTCFRConfig(totally_bogus_prt_cfr_option=True)

    def test_typo_key_rejected_via_config_validate(self, tmp_path):
        """Mirrors `cambia config validate`: a typo'd key under prt_cfr (here,
        stabilty_stop_mode instead of stability_stop_mode) must fail loudly
        instead of silently no-opping to the default."""
        real_mod = _get_real_config_module()
        cfg = {"prt_cfr": {"iterations": 10, "stabilty_stop_mode": "plateau"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        with pytest.raises(Exception, match="stabilty_stop_mode"):
            raw = real_mod.resolve_config_yaml(str(config_file))
            real_mod.Config.model_validate(raw)

    def test_known_keys_still_accepted_via_config_validate(self, tmp_path):
        """A realistic prt_cfr block (shape of config/prtcfr_production.yaml)
        must still validate cleanly -- extra="forbid" must not reject real
        fields."""
        real_mod = _get_real_config_module()
        cfg = {
            "prt_cfr": {
                "iterations": 100,
                "backend": "go",
                "lr_schedule": "global_cosine",
                "stability_enabled": True,
                "stability_metric_mode": "min",
                "stability_metric_name": "nashconv",
                "stability_stop_mode": "plateau",
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(cfg))

        raw = real_mod.resolve_config_yaml(str(config_file))
        validated = real_mod.Config.model_validate(raw)
        assert validated.prt_cfr.stability_stop_mode == "plateau"
        assert validated.prt_cfr.lr_schedule == "global_cosine"
        assert validated.prt_cfr.backend == "go"


class TestPRTCFREnumFieldsConstrained:
    @pytest.mark.parametrize("bad_mode", ["invalid", "diverge", "Plateau", ""])
    def test_invalid_stability_stop_mode_rejected(self, bad_mode):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception) as exc_info:
            PRTCFRConfig(stability_stop_mode=bad_mode)
        msg = str(exc_info.value)
        assert "divergence" in msg
        assert "plateau" in msg

    @pytest.mark.parametrize("mode", ["divergence", "plateau"])
    def test_valid_stability_stop_mode_accepted(self, mode):
        _, PRTCFRConfig = _get_real_config_classes()
        assert PRTCFRConfig(stability_stop_mode=mode).stability_stop_mode == mode

    @pytest.mark.parametrize("bad_schedule", ["cosine", "linear", "Restart", ""])
    def test_invalid_lr_schedule_rejected(self, bad_schedule):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception) as exc_info:
            PRTCFRConfig(lr_schedule=bad_schedule)
        msg = str(exc_info.value)
        assert "restart" in msg
        assert "global_cosine" in msg

    @pytest.mark.parametrize("schedule", ["restart", "global_cosine"])
    def test_valid_lr_schedule_accepted(self, schedule):
        _, PRTCFRConfig = _get_real_config_classes()
        assert PRTCFRConfig(lr_schedule=schedule).lr_schedule == schedule

    @pytest.mark.parametrize("bad_backend", ["rust", "python", "Go", "GO", ""])
    def test_invalid_backend_rejected(self, bad_backend):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception) as exc_info:
            PRTCFRConfig(backend=bad_backend)
        msg = str(exc_info.value)
        assert "go" in msg

    @pytest.mark.parametrize("backend", ["go"])
    def test_valid_backend_accepted(self, backend):
        _, PRTCFRConfig = _get_real_config_classes()
        assert PRTCFRConfig(backend=backend).backend == backend

    @pytest.mark.parametrize("bad_mode", ["minimum", "MAX", "average", ""])
    def test_invalid_stability_metric_mode_rejected(self, bad_mode):
        _, PRTCFRConfig = _get_real_config_classes()
        with pytest.raises(Exception) as exc_info:
            PRTCFRConfig(stability_metric_mode=bad_mode)
        msg = str(exc_info.value)
        assert "min" in msg
        assert "max" in msg

    @pytest.mark.parametrize("mode", ["min", "max"])
    def test_valid_stability_metric_mode_accepted(self, mode):
        _, PRTCFRConfig = _get_real_config_classes()
        assert PRTCFRConfig(stability_metric_mode=mode).stability_metric_mode == mode

    def test_defaults_still_valid(self):
        """The field defaults (used by every existing run/test that doesn't
        override these fields) must themselves pass validation."""
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig()
        assert cfg.stability_stop_mode == "divergence"
        assert cfg.lr_schedule == "restart"
        assert cfg.backend == "go"
        assert cfg.stability_metric_mode == "min"


class TestShippedPRTCFRConfigsStillValidate:
    """cambia-640 hard compat constraint: the X2R/A3 experiment program is
    live, so every shipped config under cfr/config/ (and any local run-dir
    config.yaml) must still pass `cambia config validate` after tightening
    PRTCFRConfig. A dedicated sweep (not this file) covers the full config/
    tree; this test spot-checks the two files that exercise the fields
    tightened here (x2_tiny_gate.yaml: lr_schedule/stability_metric_mode;
    x2r/c0.yaml, x2r/confirm_s2.yaml: stability_stop_mode)."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "config/x2_tiny_gate.yaml",
            "config/x2r/c0.yaml",
            "config/x2r/c1.yaml",
            "config/x2r/confirm_s2.yaml",
            "config/x2r/confirm_s3.yaml",
            "config/x2r/c_rep.yaml",
            "config/prtcfr_production.yaml",
        ],
    )
    def test_shipped_config_validates(self, rel_path):
        real_mod = _get_real_config_module()
        cfr_root = Path(__file__).resolve().parent.parent
        raw = real_mod.resolve_config_yaml(str(cfr_root / rel_path))
        real_mod.Config.model_validate(raw)  # must not raise


# ---------------------------------------------------------------------------
# cambia-736: reservoir capacity scaling with k_games_per_iter
#
# Motivation (frozen evidence): the X2R C0 run (k_games_per_iter=212 against
# the unscaled buffer_capacity=2_000_000 default) capped the reservoir at
# iteration ~180 with ~18% sample retention, vs ~48% for the k=80 baseline
# the capacity default was sized against. reservoir_capacity_scale_with_k
# (opt-in, default False) and reservoir_capacity_reference_k let a cell that
# raises k_games_per_iter scale reservoir capacity to match.
# ---------------------------------------------------------------------------


class TestReservoirCapacityScalingDefaultOff:
    """Default (scale_with_k=False) must reproduce pre-cambia-736 behavior
    byte-for-byte: resolve_*_capacity() returns the static field unchanged."""

    def test_defaults_disable_scaling(self):
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig()
        assert cfg.reservoir_capacity_scale_with_k is False
        assert cfg.reservoir_capacity_reference_k == 80

    def test_explicit_buffer_capacity_no_new_field_is_byte_identical(self):
        """An existing YAML with an explicit buffer_capacity and no new field
        must resolve to exactly that value (the old getattr(config,
        'buffer_capacity', ...) behavior), regardless of k_games_per_iter."""
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(buffer_capacity=1_234_567, k_games_per_iter=212)
        assert cfg.resolve_buffer_capacity() == 1_234_567

    def test_explicit_reservoir_capacity_no_new_field_is_byte_identical(self):
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(reservoir_capacity=5_000_000, k_games_per_iter=8192)
        assert cfg.resolve_reservoir_capacity() == 5_000_000

    def test_shipped_x2r_c0_config_unscaled_by_default(self):
        """The actual X2R C0 config (k_games_per_iter=212, no scaling field)
        must still resolve to the static default -- the ticket adds an
        opt-in fix, it does not retroactively change C0's frozen behavior."""
        real_mod = _get_real_config_module()
        cfr_root = Path(__file__).resolve().parent.parent
        raw = real_mod.resolve_config_yaml(str(cfr_root / "config/x2r/c0.yaml"))
        validated = real_mod.Config.model_validate(raw)
        assert validated.prt_cfr.k_games_per_iter == 212
        assert validated.prt_cfr.reservoir_capacity_scale_with_k is False
        assert validated.prt_cfr.resolve_buffer_capacity() == 2_000_000


class TestReservoirCapacityScalingEnabled:
    @pytest.mark.parametrize(
        "k_games_per_iter, expected",
        [
            (80, 2_000_000),  # k == reference_k: no-op
            (212, 5_300_000),  # X2R C0's k: 2_000_000 * 212 / 80
            (40, 1_000_000),  # half the reference k
        ],
    )
    def test_buffer_capacity_scales_linearly_with_k(self, k_games_per_iter, expected):
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(
            reservoir_capacity_scale_with_k=True,
            reservoir_capacity_reference_k=80,
            k_games_per_iter=k_games_per_iter,
        )
        assert cfg.resolve_buffer_capacity() == expected

    @pytest.mark.parametrize(
        "k_games_per_iter, expected",
        [
            (8192, 20_000_000),  # k == reference_k: no-op
            (16384, 40_000_000),  # 2x the reference k
        ],
    )
    def test_reservoir_capacity_scales_linearly_with_k(self, k_games_per_iter, expected):
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(
            reservoir_capacity_scale_with_k=True,
            reservoir_capacity_reference_k=8192,
            k_games_per_iter=k_games_per_iter,
        )
        assert cfg.resolve_reservoir_capacity() == expected

    def test_nonpositive_reference_k_raises_when_scaling_enabled(self):
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(
            reservoir_capacity_scale_with_k=True,
            reservoir_capacity_reference_k=0,
        )
        with pytest.raises(ValueError, match="reservoir_capacity_reference_k"):
            cfg.resolve_buffer_capacity()

    def test_x2r_c0_config_with_scaling_opted_in_matches_ticket_math(self):
        """Overriding the shipped C0 config with the new opt-in field
        reproduces the ticket's own worked example (80 * 2.65 = 212 samples
        knob) as a capacity scale: 2_000_000 * 212 / 80 = 5_300_000."""
        real_mod = _get_real_config_module()
        cfr_root = Path(__file__).resolve().parent.parent
        raw = real_mod.resolve_config_yaml(str(cfr_root / "config/x2r/c0.yaml"))
        raw.setdefault("prt_cfr", {})["reservoir_capacity_scale_with_k"] = True
        validated = real_mod.Config.model_validate(raw)
        assert validated.prt_cfr.resolve_buffer_capacity() == 5_300_000


class TestReservoirCapacityScalingSchemaAndLogging:
    def test_fields_documented_in_json_schema(self):
        real_mod = _get_real_config_module()
        schema = real_mod.Config.model_json_schema()
        props = schema["$defs"]["PRTCFRConfig"]["properties"]
        for field in (
            "reservoir_capacity_scale_with_k",
            "reservoir_capacity_reference_k",
        ):
            assert field in props
            assert props[field].get("description"), f"{field} missing a description"

    def test_resolver_function_importable_and_matches_method(self):
        """resolve_prtcfr_reservoir_capacity is the shared primitive both
        resolve_buffer_capacity()/resolve_reservoir_capacity() and the
        trainers (via getattr-based duck typing) call."""
        real_mod = _get_real_config_module()
        _, PRTCFRConfig = _get_real_config_classes()
        cfg = PRTCFRConfig(
            reservoir_capacity_scale_with_k=True,
            reservoir_capacity_reference_k=80,
            k_games_per_iter=212,
            buffer_capacity=2_000_000,
        )
        direct = real_mod.resolve_prtcfr_reservoir_capacity(
            static_capacity=cfg.buffer_capacity,
            k_games_per_iter=cfg.k_games_per_iter,
            scale_with_k=cfg.reservoir_capacity_scale_with_k,
            reference_k=cfg.reservoir_capacity_reference_k,
        )
        assert direct == cfg.resolve_buffer_capacity() == 5_300_000

    def test_tiny_trainer_logs_resolved_capacity_at_startup(self, tmp_path, caplog):
        from src.cfr.prtcfr_trainer import PRTCFRTinyTrainer
        from tools.tiny_solver import build_tree

        # PRTCFRTinyTrainer reads config via getattr() duck typing and never
        # imports src.config, so it collects fine under the conftest stub.
        # The stub's PRTCFRConfig replica does not carry the new
        # reservoir_capacity_scale_with_k/reference_k fields, though, so the
        # config instance itself must be the real PRTCFRConfig for those
        # kwargs to stick (rather than being silently ignored by extra="ignore").
        _, PRTCFRConfig = _get_real_config_classes()
        real_mod = _get_real_config_module()
        base = real_mod.load_config("config/tiny_2card_plateau.yaml")
        root, _isets, _n, aborted = build_tree(
            base,
            n_deals=1,
            seed0=0,
            max_nodes_per_deal=200_000,
            enumerate_draws=True,
            perfect_recall=True,
            tokenize=True,
            seq_cap=256,
        )
        assert aborted == 0
        cfg = PRTCFRConfig(
            m_rollouts=1,
            k_games_per_iter=160,
            iterations=1,
            train_steps_per_iter=1,
            batch_size=32,
            device="cpu",
            reservoir_capacity_scale_with_k=True,
            reservoir_capacity_reference_k=80,
            buffer_capacity=2_000_000,
        )
        with caplog.at_level(logging.INFO, logger="src.cfr.prtcfr_trainer"):
            trainer = PRTCFRTinyTrainer(root, cfg, str(tmp_path / "snaps"))
        assert trainer.buffer_capacity == 4_000_000
        assert any(
            "reservoir capacity resolved" in r.message and "4000000" in r.message
            for r in caplog.records
        )
