"""
tests/test_escher_agent.py

Tests for ESCHER Phase 0 implementation:
  - NeuralAgentWrapper base class (shared state management)
  - ESCHERAgentWrapper (PolicyNetwork inference)
  - DeepCFRAgentWrapper regression (still works after refactor)
  - Registry integration
  - run_head_to_head_typed() function
"""

import copy
import tempfile
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# conftest.py handles config stub and sys.path injection
from src.networks import AdvantageNetwork, StrategyNetwork
from src.encoding import INPUT_DIM, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config():
    """Return a minimal config object for evaluation tests."""
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


def _make_deep_cfr_checkpoint(path: str):
    """Save a freshly-initialized Deep CFR checkpoint."""
    advantage_net = AdvantageNetwork(validate_inputs=False)
    strategy_net = StrategyNetwork(validate_inputs=False)
    checkpoint = {
        "advantage_net_state_dict": advantage_net.state_dict(),
        "strategy_net_state_dict": strategy_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {"hidden_dim": 256},
    }
    torch.save(checkpoint, path)


def _make_escher_checkpoint(path: str):
    """Save a freshly-initialized ESCHER checkpoint (strategy_net_state_dict key)."""
    strategy_net = StrategyNetwork(validate_inputs=False)
    checkpoint = {
        "strategy_net_state_dict": strategy_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {"hidden_dim": 256},
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# NeuralAgentWrapper base class tests
# ---------------------------------------------------------------------------


class TestNeuralAgentWrapperBase:
    def test_base_class_is_abstract(self):
        """NeuralAgentWrapper cannot be instantiated directly."""
        import abc
        from src.evaluate_agents import NeuralAgentWrapper

        assert hasattr(NeuralAgentWrapper, "__abstractmethods__")
        assert "choose_action" in NeuralAgentWrapper.__abstractmethods__

    def test_initialize_state_creates_agent_state(self):
        """initialize_state() should create an AgentState on the wrapper."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentState
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            assert agent.agent_state is None

            game_state = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game_state)

            assert agent.agent_state is not None
            assert isinstance(agent.agent_state, AgentState)
        finally:
            os.unlink(ckpt_path)

    def test_update_state_updates_agent_state(self):
        """update_state() should not raise and should keep agent_state intact."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentObservation
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game_state)

            obs = agent._create_observation(game_state, None, 0)
            assert obs is not None

            # update_state must not raise
            agent.update_state(obs)
            assert agent.agent_state is not None
        finally:
            os.unlink(ckpt_path)

    def test_update_state_no_op_when_not_initialized(self):
        """update_state() is silent (no raise) when agent_state is None."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentObservation
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            game_state = CambiaGameState(house_rules=config.cambia_rules)

            obs = AgentObservation(
                acting_player=-1,
                action=None,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[4, 4],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=[],
                did_cambia_get_called=False,
                who_called_cambia=None,
                is_game_over=False,
                current_turn=0,
            )
            # Should not raise
            agent.update_state(obs)
            assert agent.agent_state is None
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# ESCHERAgentWrapper tests
# ---------------------------------------------------------------------------


class TestESCHERAgentWrapper:
    def test_loads_mock_escher_checkpoint(self):
        """ESCHERAgentWrapper loads StrategyNetwork from mock checkpoint."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
            assert hasattr(agent, "policy_net")
            assert isinstance(agent.policy_net, StrategyNetwork)
        finally:
            os.unlink(ckpt_path)

    def test_choose_action_returns_valid_legal_action(self):
        """choose_action() returns one of the legal actions after initialization."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game_state)

            legal_actions = game_state.get_legal_actions()
            assert len(legal_actions) > 0

            chosen = agent.choose_action(game_state, legal_actions)
            assert chosen in legal_actions
        finally:
            os.unlink(ckpt_path)

    def test_choose_action_fallback_when_no_agent_state(self):
        """choose_action() falls back to random when agent_state is None."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
            # Do NOT call initialize_state — agent_state remains None

            game_state = CambiaGameState(house_rules=config.cambia_rules)
            legal_actions = game_state.get_legal_actions()
            assert len(legal_actions) > 0

            # Should not raise; falls back to random
            chosen = agent.choose_action(game_state, legal_actions)
            assert chosen in legal_actions
        finally:
            os.unlink(ckpt_path)

    def test_registered_in_agent_registry(self):
        """ESCHERAgentWrapper is registered as 'escher' in AGENT_REGISTRY."""
        from src.evaluate_agents import AGENT_REGISTRY, ESCHERAgentWrapper

        assert "escher" in AGENT_REGISTRY
        assert AGENT_REGISTRY["escher"] is ESCHERAgentWrapper

    def test_get_agent_escher_returns_wrapper(self):
        """get_agent('escher', ...) with checkpoint_path returns ESCHERAgentWrapper."""
        from src.evaluate_agents import get_agent, ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = get_agent("escher", 0, config, checkpoint_path=ckpt_path, device="cpu")
            assert isinstance(agent, ESCHERAgentWrapper)
        finally:
            os.unlink(ckpt_path)

    def test_get_agent_escher_raises_without_checkpoint(self):
        """get_agent('escher', ...) without checkpoint_path raises ValueError."""
        from src.evaluate_agents import get_agent

        config = _make_config()
        with pytest.raises(ValueError, match="checkpoint_path"):
            get_agent("escher", 0, config)

    def test_invalid_checkpoint_raises_error(self):
        """Loading from a non-existent path raises an appropriate error."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with pytest.raises(Exception):
            ESCHERAgentWrapper(0, config, "/nonexistent/path.pt", device="cpu")

    def test_invalid_checkpoint_missing_key_raises_error(self):
        """Checkpoint missing 'strategy_net_state_dict' raises KeyError."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            # Save checkpoint without the required key
            torch.save({"dcfr_config": {"hidden_dim": 256}}, ckpt_path)
            with pytest.raises(KeyError):
                ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# DeepCFRAgentWrapper regression tests
# ---------------------------------------------------------------------------


class TestDeepCFRAgentWrapperRegression:
    def test_registered_as_deep_cfr(self):
        """DeepCFRAgentWrapper is still registered as 'deep_cfr' in AGENT_REGISTRY."""
        from src.evaluate_agents import AGENT_REGISTRY, DeepCFRAgentWrapper

        assert "deep_cfr" in AGENT_REGISTRY
        assert AGENT_REGISTRY["deep_cfr"] is DeepCFRAgentWrapper

    def test_choose_action_returns_valid_action(self):
        """DeepCFRAgentWrapper.choose_action() returns a legal action (regression)."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game_state)

            legal_actions = game_state.get_legal_actions()
            chosen = agent.choose_action(game_state, legal_actions)
            assert chosen in legal_actions
        finally:
            os.unlink(ckpt_path)

    def test_inherits_from_neural_agent_wrapper(self):
        """DeepCFRAgentWrapper is a subclass of NeuralAgentWrapper (refactor check)."""
        from src.evaluate_agents import DeepCFRAgentWrapper, NeuralAgentWrapper

        assert issubclass(DeepCFRAgentWrapper, NeuralAgentWrapper)

    def test_get_agent_deep_cfr_returns_wrapper(self):
        """get_agent('deep_cfr', ...) still works correctly after refactor."""
        from src.evaluate_agents import get_agent, DeepCFRAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = get_agent("deep_cfr", 0, config, checkpoint_path=ckpt_path, device="cpu")
            assert isinstance(agent, DeepCFRAgentWrapper)
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# Head-to-head typed tests
# ---------------------------------------------------------------------------


class TestRunHeadToHeadTyped:
    def test_completes_10_games(self):
        """run_head_to_head_typed() completes 10 games without error."""
        from src.evaluate_agents import run_head_to_head_typed

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name
        try:
            _make_escher_checkpoint(ckpt_a)
            _make_deep_cfr_checkpoint(ckpt_b)

            result = run_head_to_head_typed(
                agent_a_type="escher",
                checkpoint_a=ckpt_a,
                agent_b_type="deep_cfr",
                checkpoint_b=ckpt_b,
                num_games=10,
                config=config,
                device="cpu",
            )

            assert result["num_games"] == 10
            total = result["wins_a"] + result["wins_b"] + result["draws"]
            assert total + result.get("errors", 0) >= 10
            assert 0.0 <= result["win_rate_a"] <= 1.0
            assert 0.0 <= result["win_rate_b"] <= 1.0
            assert "avg_game_turns" in result
            assert "std_game_turns" in result
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)

    def test_seat_alternation(self):
        """Seat assignment alternates: odd games have A as P0, even games have B as P0."""
        from src.evaluate_agents import run_head_to_head_typed, ESCHERAgentWrapper

        # Track which checkpoint each P0 uses across 4 games
        p0_types: list = []

        original_init = ESCHERAgentWrapper.__init__
        original_deep_init = None

        from src.evaluate_agents import DeepCFRAgentWrapper

        original_deep_init = DeepCFRAgentWrapper.__init__

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name

        try:
            _make_escher_checkpoint(ckpt_a)
            _make_escher_checkpoint(ckpt_b)

            instantiated: list = []

            real_get_agent = None
            from src import evaluate_agents as ea

            real_get_agent_fn = ea.get_agent

            def mock_get_agent(agent_type, player_id, cfg, **kwargs):
                agent = real_get_agent_fn(agent_type, player_id, cfg, **kwargs)
                # Record (player_id, checkpoint_path) pairs
                instantiated.append((player_id, kwargs.get("checkpoint_path", "")))
                return agent

            import unittest.mock as mock

            with mock.patch("src.evaluate_agents.get_agent", side_effect=mock_get_agent):
                run_head_to_head_typed(
                    agent_a_type="escher",
                    checkpoint_a=ckpt_a,
                    agent_b_type="escher",
                    checkpoint_b=ckpt_b,
                    num_games=4,
                    config=config,
                    device="cpu",
                )

            # Games 1 and 3 (odd): A is P0 (ckpt_a assigned to player_id=0)
            # Games 2 and 4 (even): B is P0 (ckpt_b assigned to player_id=0)
            # Each game creates 2 agents, so instantiated has 8 entries for 4 games
            assert len(instantiated) == 8

            # Game 1 (index 0,1): P0 should have ckpt_a
            assert instantiated[0] == (0, ckpt_a)
            # Game 2 (index 2,3): P0 should have ckpt_b
            assert instantiated[2] == (0, ckpt_b)
            # Game 3 (index 4,5): P0 should have ckpt_a
            assert instantiated[4] == (0, ckpt_a)
            # Game 4 (index 6,7): P0 should have ckpt_b
            assert instantiated[6] == (0, ckpt_b)
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)

    def test_returns_correct_keys(self):
        """run_head_to_head_typed() result dict has all required keys."""
        from src.evaluate_agents import run_head_to_head_typed

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name
        try:
            _make_escher_checkpoint(ckpt_a)
            _make_escher_checkpoint(ckpt_b)

            result = run_head_to_head_typed(
                agent_a_type="escher",
                checkpoint_a=ckpt_a,
                agent_b_type="escher",
                checkpoint_b=ckpt_b,
                num_games=5,
                config=config,
                device="cpu",
            )

            required_keys = {
                "wins_a", "wins_b", "draws", "win_rate_a", "win_rate_b",
                "num_games", "errors", "avg_game_turns", "std_game_turns",
            }
            assert required_keys.issubset(result.keys())
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)


# ---------------------------------------------------------------------------
# ESCHER agent state reset tests
# ---------------------------------------------------------------------------


class TestESCHERAgentStateReset:
    """Verify ESCHER agent properly reinitializes state for new games."""

    def test_agent_state_fresh_per_game(self):
        """Agent state must be freshly initialized for each new game.

        After initialize_state() with a second game, the wrapper must hold a
        brand-new AgentState that reflects the new game — not the old one.
        Checks:
          1. agent_state is not None after second init.
          2. agent_state is a different object from the first game's state.
          3. own_active_mask is reset (only initial-peek slots are active).
          4. own_hand buckets reflect the new game's peek indices.
        """
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.agent_state import AgentState, CardBucket
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

            # --- Game 1 ---
            game1 = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game1)

            assert agent.agent_state is not None
            state_after_game1 = agent.agent_state

            # Dirty the state: mark all own slots as HIGH_KING bucket
            # (simulates knowledge accumulated during a real game)
            for slot in range(len(game1.players[0].hand)):
                state_after_game1.own_hand[slot].bucket = CardBucket.HIGH_KING
            dirty_id = id(state_after_game1)

            # --- Game 2 ---
            game2 = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game2)

            assert agent.agent_state is not None, "agent_state must not be None after re-init"

            # 1. Must be a fresh object, not the game-1 state
            assert id(agent.agent_state) != dirty_id, (
                "initialize_state() must create a new AgentState object, not reuse the old one"
            )

            # 2. own_active_mask must only contain the initial-peek slots for game 2
            peek_indices = set(game2.players[0].initial_peek_indices)
            active = set(agent.agent_state.own_active_mask)
            # Every active slot must be one that was peeked at start
            assert active <= peek_indices, (
                f"own_active_mask {active} contains non-peeked slots; expected subset of {peek_indices}"
            )

            # 3. Peeked slots must have a non-UNKNOWN bucket (concrete knowledge)
            for slot in peek_indices:
                bucket = agent.agent_state.own_hand[slot].bucket
                assert bucket != CardBucket.UNKNOWN, (
                    f"Slot {slot} was peeked at init but has UNKNOWN bucket after initialize_state()"
                )

            # 4. Non-peeked slots must be UNKNOWN (no bleed-over from game 1)
            non_peeked = set(range(len(game2.players[0].hand))) - peek_indices
            for slot in non_peeked:
                bucket = agent.agent_state.own_hand[slot].bucket
                assert bucket == CardBucket.UNKNOWN, (
                    f"Slot {slot} was not peeked but has bucket {bucket} after re-init "
                    f"(possible stale-state bleed from game 1)"
                )
        finally:
            os.unlink(ckpt_path)

    def test_agent_state_object_replaced_not_mutated(self):
        """initialize_state() must replace agent_state, not mutate it in place."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

            game1 = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game1)
            state1 = agent.agent_state

            game2 = CambiaGameState(house_rules=config.cambia_rules)
            agent.initialize_state(game2)
            state2 = agent.agent_state

            assert state2 is not state1, (
                "initialize_state() must assign a new AgentState instance, not reuse the existing one"
            )
        finally:
            os.unlink(ckpt_path)
