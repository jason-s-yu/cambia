"""
tests/test_evaluate.py

End-to-end evaluation tests for evaluation infrastructure.

Tests:
- DeepCFRAgentWrapper loads a freshly-initialized checkpoint and plays a full game
- run_evaluation completes without errors and returns valid win rates
- run_evaluation_multi_baseline returns per-baseline results
"""

import sys
import copy
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
import torch

# conftest.py handles the config stub

from src.networks import AdvantageNetwork, StrategyNetwork
from src.encoding import INPUT_DIM, NUM_ACTIONS

# --- Helpers ---


def _make_checkpoint(path: str):
    """Save a freshly-initialized Deep CFR checkpoint to path."""
    advantage_net = AdvantageNetwork(validate_inputs=False)
    strategy_net = StrategyNetwork(validate_inputs=False)

    checkpoint = {
        "advantage_net_state_dict": advantage_net.state_dict(),
        "strategy_net_state_dict": strategy_net.state_dict(),
        "advantage_optimizer_state_dict": {},
        "strategy_optimizer_state_dict": {},
        "training_step": 0,
        "total_traversals": 0,
        "current_iteration": 0,
        "dcfr_config": {
            "hidden_dim": 256,
            "learning_rate": 1e-3,
            "batch_size": 2048,
        },
        "advantage_loss_history": [],
        "strategy_loss_history": [],
        "es_validation_history": [],
        "advantage_buffer_path": None,
        "strategy_buffer_path": None,
        "grad_scaler_state_dict": {},
    }
    torch.save(checkpoint, path)


def _make_config():
    """Return a minimal config object suitable for evaluation."""
    config = type("Config", (), {})()

    rules = type("CambiaRulesConfig", (), {})()
    rules.allowDrawFromDiscardPile = False
    rules.allowReplaceAbilities = False
    rules.snapRace = False
    rules.penaltyDrawCount = 2
    rules.use_jokers = 0  # No jokers for simpler games
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
    greedy_cfg = type("GreedyAgentConfig", (), {})()
    greedy_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_agent = greedy_cfg
    config.agents = agents_cfg

    return config


# --- Tests ---


class TestDeepCFRAgentWrapper:
    def test_loads_checkpoint(self, tmp_path):
        """DeepCFRAgentWrapper should load a .pt checkpoint without error."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import DeepCFRAgentWrapper

        agent = DeepCFRAgentWrapper(player_id=0, config=config, checkpoint_path=ckpt_path)
        assert agent.advantage_net is not None

    def test_chooses_legal_action(self, tmp_path):
        """DeepCFRAgentWrapper.choose_action returns a legal action."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agents import action_codec
        from src.ffi.bridge import GoEngine

        agent = DeepCFRAgentWrapper(player_id=0, config=config, checkpoint_path=ckpt_path)
        # The Go engine is what the eval loop plays on (cambia-1426); the
        # wrapper's belief is a GoAgentState bound to this game.
        with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
            agent.initialize_state(game)
            legal = action_codec.actions_from_mask(game.legal_actions_mask())
            assert legal
            action = agent.choose_action(game, legal)
            assert action in legal
            agent.release_belief()

    def test_choose_action_without_state_init_falls_back(self, tmp_path):
        """choose_action on uninitialized agent falls back to random without crashing."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.constants import ActionDrawStockpile

        agent = DeepCFRAgentWrapper(player_id=0, config=config, checkpoint_path=ckpt_path)
        # Don't call initialize_state: agent_state is None
        legal = {ActionDrawStockpile()}
        action = agent.choose_action(None, legal)  # type: ignore
        assert action in legal

    def test_update_state_resilient(self, tmp_path):
        """update_state doesn't crash on None agent_state."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentObservation

        agent = DeepCFRAgentWrapper(player_id=0, config=config, checkpoint_path=ckpt_path)
        # Should not raise
        agent.update_state(None)  # type: ignore


class TestRunEvaluation:
    def test_random_vs_random(self, tmp_path):
        """run_evaluation with two RandomAgents completes without errors."""
        config = _make_config()

        from src.evaluate_agents import run_evaluation

        with patch("src.evaluate_agents.load_config", return_value=config):
            results = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="random",
                agent2_type="random",
                num_games=5,
                strategy_path=None,
            )

        assert isinstance(results, Counter)
        total = (
            results.get("P0 Wins", 0)
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
            + results.get("Errors", 0)
        )
        assert total == 5, f"Expected 5 game outcomes, got {total}: {dict(results)}"

    def test_deep_cfr_vs_random(self, tmp_path):
        """run_evaluation with DeepCFRAgentWrapper vs RandomAgent completes without errors."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import run_evaluation

        with patch("src.evaluate_agents.load_config", return_value=config):
            results = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="deep_cfr",
                agent2_type="random",
                num_games=5,
                strategy_path=None,
                checkpoint_path=ckpt_path,
                device="cpu",
            )

        assert isinstance(results, Counter)
        total = (
            results.get("P0 Wins", 0)
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
            + results.get("Errors", 0)
        )
        assert total == 5, f"Expected 5 game outcomes, got {total}: {dict(results)}"
        # deep_cfr wins should be in valid range
        assert 0 <= results.get("P0 Wins", 0) <= 5

    def test_win_rates_are_valid(self, tmp_path):
        """Win rates from evaluation are in [0, 1] range."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import run_evaluation

        with patch("src.evaluate_agents.load_config", return_value=config):
            results = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="deep_cfr",
                agent2_type="random",
                num_games=10,
                strategy_path=None,
                checkpoint_path=ckpt_path,
            )

        p0_wins = results.get("P0 Wins", 0)
        p1_wins = results.get("P1 Wins", 0)
        ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        total_scored = p0_wins + p1_wins + ties
        if total_scored > 0:
            win_rate = p0_wins / total_scored
            assert 0.0 <= win_rate <= 1.0


class TestRunEvaluationMultiBaseline:
    def test_multi_baseline(self, tmp_path):
        """run_evaluation_multi_baseline returns results for each baseline."""
        ckpt_path = str(tmp_path / "test.pt")
        _make_checkpoint(ckpt_path)
        config = _make_config()

        from src.evaluate_agents import run_evaluation_multi_baseline

        with patch("src.evaluate_agents.load_config", return_value=config):
            results_map = run_evaluation_multi_baseline(
                config_path="dummy.yaml",
                checkpoint_path=ckpt_path,
                num_games=3,
                baselines=["random"],
            )

        assert "random" in results_map
        results = results_map["random"]
        total = (
            results.get("P0 Wins", 0)
            + results.get("P1 Wins", 0)
            + results.get("Ties", 0)
            + results.get("MaxTurnTies", 0)
            + results.get("Errors", 0)
        )
        assert total == 3, f"Expected 3 outcomes for 'random', got {total}"


# --- Served policy: strategy net vs last iterate (cambia-721) ---


def _make_checkpoint_without_strategy_net(path: str):
    """A Deep CFR checkpoint carrying only the advantage net (SD-CFR style)."""
    checkpoint = {
        "advantage_net_state_dict": AdvantageNetwork(validate_inputs=False).state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {"hidden_dim": 256},
    }
    torch.save(checkpoint, path)


class _FixedPolicy(torch.nn.Module):
    """Puts all mass on one action index, whatever the features."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, features, action_mask):
        out = torch.zeros(features.shape[0], NUM_ACTIONS, dtype=torch.float32)
        out[:, self.index] = 1.0
        return out


class _FixedAdvantages(torch.nn.Module):
    """Regret matching on this yields all mass on one action index."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, features, action_mask):
        out = torch.full((features.shape[0], NUM_ACTIONS), -1.0, dtype=torch.float32)
        out[:, self.index] = 1.0
        return out


def _two_distinct_legal_indices(legal):
    from src.encoding import encode_action_mask

    mask = encode_action_mask(list(legal))
    indices = list(np.where(mask)[0])
    assert len(indices) >= 2, "need two legal actions to tell the nets apart"
    return int(indices[0]), int(indices[1])


class TestDeepCFRServedPolicy:
    def _agent(self, tmp_path, with_strategy_net=True):
        from src.evaluate_agents import DeepCFRAgentWrapper

        ckpt_path = str(tmp_path / "served.pt")
        if with_strategy_net:
            _make_checkpoint(ckpt_path)
        else:
            _make_checkpoint_without_strategy_net(ckpt_path)
        return DeepCFRAgentWrapper(
            player_id=0,
            config=_make_config(),
            checkpoint_path=ckpt_path,
            use_argmax=True,
        )

    def test_strategy_net_is_loaded_and_labelled(self, tmp_path):
        agent = self._agent(tmp_path, with_strategy_net=True)

        assert agent.strategy_net is not None
        assert agent.served_policy == "average_strategy"

    def test_strategy_net_drives_choose_action(self, tmp_path):
        from src.agents import action_codec
        from src.ffi.bridge import GoEngine
        from src.encoding import index_to_action

        config = _make_config()
        agent = self._agent(tmp_path, with_strategy_net=True)

        with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
            agent.initialize_state(game)
            legal = action_codec.actions_from_mask(game.legal_actions_mask())
            strategy_idx, advantage_idx = _two_distinct_legal_indices(legal)

            agent.strategy_net = _FixedPolicy(strategy_idx)
            agent.advantage_net = _FixedAdvantages(advantage_idx)

            chosen = agent.choose_action(game, legal)

            assert chosen == index_to_action(strategy_idx, list(legal))
            agent.release_belief()

    def test_missing_strategy_net_falls_back_to_the_last_iterate(self, tmp_path):
        from src.agents import action_codec
        from src.ffi.bridge import GoEngine
        from src.encoding import index_to_action

        config = _make_config()
        agent = self._agent(tmp_path, with_strategy_net=False)

        assert agent.strategy_net is None
        assert agent.served_policy == "last_iterate"

        with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
            agent.initialize_state(game)
            legal = action_codec.actions_from_mask(game.legal_actions_mask())
            _unused, advantage_idx = _two_distinct_legal_indices(legal)

            agent.advantage_net = _FixedAdvantages(advantage_idx)

            chosen = agent.choose_action(game, legal)

            assert chosen == index_to_action(advantage_idx, list(legal))
            agent.release_belief()
