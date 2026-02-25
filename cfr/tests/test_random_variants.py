"""Tests for RandomNoCambiaAgent and RandomLateCambiaAgent."""

import pytest
from dataclasses import dataclass, field
from typing import Set
from unittest.mock import MagicMock

from src.agents.baseline_agents import RandomNoCambiaAgent, RandomLateCambiaAgent
from src.constants import (
    GameAction,
    ActionCallCambia,
    ActionDrawStockpile,
    ActionDiscard,
)
from src.evaluate_agents import AGENT_REGISTRY


# ---------------------------------------------------------------------------
# Config stubs (matching pattern from test_baseline_agents.py)
# ---------------------------------------------------------------------------


@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(default_factory=_GreedyAgentConfig)


@dataclass
class _CambiaRulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 300


@dataclass
class _TestConfig:
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)
    cambia_rules: _CambiaRulesConfig = field(default_factory=_CambiaRulesConfig)


def _make_game_state(turn_number: int = 0):
    """Create a mock game state with a given turn number."""
    gs = MagicMock()
    gs._turn_number = turn_number
    return gs


@pytest.fixture
def config():
    return _TestConfig()


# ---------------------------------------------------------------------------
# RandomNoCambiaAgent tests
# ---------------------------------------------------------------------------


class TestRandomNoCambiaAgent:
    def test_never_calls_cambia_when_in_legal_actions(self, config):
        agent = RandomNoCambiaAgent(player_id=0, config=config)
        gs = _make_game_state()
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        for _ in range(50):
            action = agent.choose_action(gs, legal)
            assert not isinstance(action, ActionCallCambia)

    def test_only_cambia_falls_back_to_cambia(self, config):
        """If Cambia is the only legal action, agent is forced to return it."""
        agent = RandomNoCambiaAgent(player_id=0, config=config)
        gs = _make_game_state()
        legal = {ActionCallCambia()}
        action = agent.choose_action(gs, legal)
        assert isinstance(action, ActionCallCambia)

    def test_non_cambia_actions_still_chosen(self, config):
        agent = RandomNoCambiaAgent(player_id=0, config=config)
        gs = _make_game_state()
        legal = {ActionDrawStockpile()}
        action = agent.choose_action(gs, legal)
        assert isinstance(action, ActionDrawStockpile)

    def test_registered_in_agent_registry(self):
        assert "random_no_cambia" in AGENT_REGISTRY
        assert AGENT_REGISTRY["random_no_cambia"] is RandomNoCambiaAgent


# ---------------------------------------------------------------------------
# RandomLateCambiaAgent tests
# ---------------------------------------------------------------------------


class TestRandomLateCambiaAgent:
    def test_suppresses_cambia_before_n_turns(self, config):
        agent = RandomLateCambiaAgent(player_id=0, config=config, n_turns=8)
        gs = _make_game_state(turn_number=3)
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        for _ in range(50):
            action = agent.choose_action(gs, legal)
            assert not isinstance(action, ActionCallCambia)

    def test_suppresses_cambia_at_turn_zero(self, config):
        agent = RandomLateCambiaAgent(player_id=0, config=config, n_turns=8)
        gs = _make_game_state(turn_number=0)
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        for _ in range(50):
            action = agent.choose_action(gs, legal)
            assert not isinstance(action, ActionCallCambia)

    def test_allows_cambia_at_n_turns(self, config):
        agent = RandomLateCambiaAgent(player_id=0, config=config, n_turns=8)
        gs = _make_game_state(turn_number=8)
        legal = {ActionCallCambia()}
        action = agent.choose_action(gs, legal)
        assert isinstance(action, ActionCallCambia)

    def test_allows_cambia_after_n_turns(self, config):
        agent = RandomLateCambiaAgent(player_id=0, config=config, n_turns=8)
        gs = _make_game_state(turn_number=20)
        legal = {ActionCallCambia(), ActionDrawStockpile()}
        # With both options legal, cambia should appear at least sometimes over many samples
        actions = {agent.choose_action(gs, legal) for _ in range(200)}
        assert any(isinstance(a, ActionCallCambia) for a in actions)

    def test_default_n_turns_is_8(self, config):
        agent = RandomLateCambiaAgent(player_id=0, config=config)
        assert agent.n_turns == 8

    def test_fallback_when_only_cambia_before_n_turns(self, config):
        """Before n_turns, if only Cambia is legal, falls back to it."""
        agent = RandomLateCambiaAgent(player_id=0, config=config, n_turns=8)
        gs = _make_game_state(turn_number=3)
        legal = {ActionCallCambia()}
        action = agent.choose_action(gs, legal)
        assert isinstance(action, ActionCallCambia)

    def test_registered_in_agent_registry(self):
        assert "random_late_cambia" in AGENT_REGISTRY
        assert AGENT_REGISTRY["random_late_cambia"] is RandomLateCambiaAgent
