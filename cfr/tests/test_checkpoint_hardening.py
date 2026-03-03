"""
tests/test_checkpoint_hardening.py

Unit tests for Workstream A: Checkpoint Hardening.

Covers:
  - atomic_torch_save: write-then-rename atomicity, crash safety
  - atomic_npz_save: write-then-rename atomicity
  - weights_only=True at all torch.load sites
  - DeepCFRConfig mutual exclusion: pipeline_training + num_traversal_threads
"""

import inspect
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# atomic_torch_save
# ---------------------------------------------------------------------------


class TestAtomicTorchSave:
    def test_file_created(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        data = {"x": torch.tensor([1, 2, 3])}
        atomic_torch_save(data, path)
        assert os.path.exists(path)

    def test_content_correct(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        data = {"key": torch.tensor([42.0])}
        atomic_torch_save(data, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.allclose(loaded["key"], data["key"])

    def test_no_tmp_file_left_on_success(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        atomic_torch_save({"v": torch.zeros(1)}, path)
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_no_partial_file_on_failure(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")

        # Simulate torch.save raising mid-write
        with patch("torch.save", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_torch_save({"v": torch.zeros(1)}, path)

        # The target file must not exist
        assert not os.path.exists(path)
        # No stray .tmp files either
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_creates_parent_directory(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "nested" / "dir" / "checkpoint.pt")
        atomic_torch_save({"v": torch.zeros(1)}, path)
        assert os.path.exists(path)

    def test_overwrites_existing_file_atomically(self, tmp_path):
        from src.persistence import atomic_torch_save

        path = str(tmp_path / "checkpoint.pt")
        atomic_torch_save({"v": torch.tensor([1.0])}, path)
        atomic_torch_save({"v": torch.tensor([2.0])}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["v"].item() == 2.0


# ---------------------------------------------------------------------------
# atomic_npz_save
# ---------------------------------------------------------------------------


class TestAtomicNpzSave:
    def _dummy_save_fn(self, arr, base_path):
        """Mimics ReservoirBuffer.save: writes <base_path>.npz."""
        np.savez(base_path, data=arr)

    def test_file_created(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([1, 2, 3])
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        assert os.path.exists(base + ".npz")

    def test_content_correct(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([10.0, 20.0])
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        loaded = np.load(base + ".npz")
        np.testing.assert_array_equal(loaded["data"], arr)

    def test_no_tmp_file_left_on_success(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.zeros(4)
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base)
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp") or f.endswith(".tmp.npz")]
        assert tmp_files == []

    def test_no_partial_file_on_failure(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")

        def failing_save(p):
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            atomic_npz_save(failing_save, base)

        assert not os.path.exists(base + ".npz")

    def test_accepts_path_with_npz_extension(self, tmp_path):
        from src.persistence import atomic_npz_save

        base = str(tmp_path / "buffer")
        arr = np.array([5])
        # Pass path that already has .npz
        atomic_npz_save(lambda p: self._dummy_save_fn(arr, p), base + ".npz")
        assert os.path.exists(base + ".npz")


# ---------------------------------------------------------------------------
# weights_only=True at load sites — source-level inspection (no module import)
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def _read_source(rel_path: str) -> str:
    return (_SRC_ROOT / rel_path).read_text()


def _find_torch_load_calls(source: str):
    """Return list of torch.load(...) calls from a source string."""
    import re
    # Match multi-line torch.load(...) by finding the opening and scanning to close paren
    calls = re.findall(r"torch\.load\((?:[^()]*|\([^()]*\))*\)", source)
    return calls


class TestWeightsOnlyAtLoadSites:
    def test_deep_trainer_load_checkpoint_uses_weights_only_true(self):
        source = _read_source("cfr/deep_trainer.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in deep_trainer.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )

    def test_main_train_inspect_checkpoint_uses_weights_only_true(self):
        source = _read_source("main_train.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in main_train.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )

    def test_cli_uses_weights_only_true(self):
        source = _read_source("cli.py")
        calls = _find_torch_load_calls(source)
        assert calls, "Expected at least one torch.load call in cli.py"
        for call in calls:
            assert "weights_only=True" in call, (
                f"torch.load call does not use weights_only=True: {call}"
            )


# ---------------------------------------------------------------------------
# Mutual exclusion guard — source-level inspection + minimal dataclass exec
# ---------------------------------------------------------------------------


def _extract_deep_cfr_config_class() -> type:
    """
    Extract and eval the DeepCFRConfig dataclass from deep_trainer.py without
    importing the full module (which has unavailable deps like rich, typer).
    """
    import re
    from dataclasses import dataclass

    source = _read_source("cfr/deep_trainer.py")

    # Extract class body: from @dataclass\nclass DeepCFRConfig: to the next top-level def/class
    match = re.search(
        r"(@dataclass\nclass DeepCFRConfig:.*?)(?=\n@|\ndef |\nclass )",
        source,
        re.DOTALL,
    )
    assert match, "Could not find DeepCFRConfig class in deep_trainer.py"
    class_src = match.group(1)

    # Build a minimal namespace to exec the class definition
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
# ESCHER: value net worker weights key + EP-PBS checkpoint loading
# ---------------------------------------------------------------------------


class TestValueNetWorkerKey:
    """Verify deep_trainer.py serializes value_net under __value_net__ key."""

    @pytest.mark.xfail(
        strict=False,
        reason="Bug 2 fix pending impl-1: _get_network_weights_for_workers must add __value_net__",
    )
    def test_worker_weights_include_value_net_key(self):
        """_get_network_weights_for_workers must serialize __value_net__ for ESCHER."""
        source = _read_source("cfr/deep_trainer.py")
        assert "__value_net__" in source, (
            "_get_network_weights_for_workers must include __value_net__ key for ESCHER"
        )


def _make_escher_ep_pbs_checkpoint(path: str):
    """Save a new-format ESCHER checkpoint with advantage_net_state_dict + EP-PBS config."""
    from src.networks import build_advantage_network
    from src.constants import EP_PBS_INPUT_DIM
    from src.encoding import NUM_ACTIONS

    adv_net = build_advantage_network(
        input_dim=EP_PBS_INPUT_DIM,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        dropout=0.1,
        validate_inputs=False,
        num_hidden_layers=3,
        use_residual=True,
        network_type="residual",
    )
    checkpoint = {
        "advantage_net_state_dict": adv_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {
            "hidden_dim": 256,
            "num_hidden_layers": 3,
            "use_residual": True,
            "network_type": "residual",
            "encoding_mode": "ep_pbs",
            "encoding_layout": "interleaved",
            "input_dim": EP_PBS_INPUT_DIM,
        },
    }
    torch.save(checkpoint, path)


class TestESCHEREPBSCheckpoint:
    """ESCHERAgentWrapper loads new-format (advantage_net) EP-PBS checkpoint correctly."""

    def _make_config(self):
        config = type("Config", (), {})()
        rules = type("CambiaRulesConfig", (), {})()
        rules.allowDrawFromDiscardPile = False
        rules.allowReplaceAbilities = False
        rules.snapRace = False
        rules.penaltyDrawCount = 2
        rules.use_jokers = 0
        rules.cards_per_player = 4
        rules.initial_view_count = 2
        rules.cambia_allowed_round = 0
        rules.allowOpponentSnapping = False
        rules.max_game_turns = 100
        config.cambia_rules = rules
        agent_params = type("AgentParamsConfig", (), {})()
        agent_params.memory_level = 1
        agent_params.time_decay_turns = 10
        config.agent_params = agent_params
        agents_cfg = type("AgentsConfig", (), {})()
        agents_cfg.cambia_call_threshold = 10
        agents_cfg.greedy_cambia_threshold = 5
        config.agents = agents_cfg
        return config

    def test_loads_ep_pbs_advantage_checkpoint(self, tmp_path):
        """ESCHERAgentWrapper loads new-format checkpoint with advantage_net_state_dict."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.networks import ResidualAdvantageNetwork

        ckpt_path = str(tmp_path / "escher_ep_pbs.pt")
        _make_escher_ep_pbs_checkpoint(ckpt_path)

        config = self._make_config()
        agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

        assert agent.advantage_net is not None, "advantage_net must be loaded for new-format checkpoint"
        assert agent.policy_net is None, "policy_net must be None for new-format checkpoint"
        assert agent._encoding_mode == "ep_pbs"
        assert agent._encoding_layout == "interleaved"

    def test_ep_pbs_checkpoint_produces_valid_action(self, tmp_path):
        """ESCHERAgentWrapper with EP-PBS checkpoint chooses a valid legal action."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.game.engine import CambiaGameState

        ckpt_path = str(tmp_path / "escher_ep_pbs.pt")
        _make_escher_ep_pbs_checkpoint(ckpt_path)

        config = self._make_config()
        agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

        game_state = CambiaGameState(house_rules=config.cambia_rules)
        agent.initialize_state(game_state)

        legal_actions = game_state.get_legal_actions()
        assert len(legal_actions) > 0
        chosen = agent.choose_action(game_state, legal_actions)
        assert chosen in legal_actions


class TestDeepCFRConfigMutualExclusion:
    @pytest.fixture(scope="class")
    def DeepCFRConfig(self):
        return _extract_deep_cfr_config_class()

    def test_pipeline_true_threads_1_ok(self, DeepCFRConfig):
        cfg = DeepCFRConfig(pipeline_training=True, num_traversal_threads=1)
        assert cfg.pipeline_training is True

    def test_pipeline_false_threads_many_ok(self, DeepCFRConfig):
        cfg = DeepCFRConfig(pipeline_training=False, num_traversal_threads=4)
        assert cfg.num_traversal_threads == 4

    def test_pipeline_true_threads_many_raises(self, DeepCFRConfig):
        with pytest.raises(ValueError, match="mutually exclusive"):
            DeepCFRConfig(pipeline_training=True, num_traversal_threads=2)

    def test_pipeline_true_threads_many_raises_message_helpful(self, DeepCFRConfig):
        with pytest.raises(ValueError) as exc_info:
            DeepCFRConfig(pipeline_training=True, num_traversal_threads=4)
        assert "pipeline_training" in str(exc_info.value)
        assert "num_traversal_threads" in str(exc_info.value)

    def test_default_config_valid(self, DeepCFRConfig):
        """Default config must not raise (pipeline_training=True, threads=1)."""
        cfg = DeepCFRConfig()
        assert cfg.pipeline_training is True
        assert cfg.num_traversal_threads == 1
