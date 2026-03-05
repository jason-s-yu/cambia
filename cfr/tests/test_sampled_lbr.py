"""Tests for sampled LBR exploitability measurement."""

import pytest
from dataclasses import dataclass, field
from typing import Optional

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import RandomAgent
from src.cfr.sampled_lbr import sampled_lbr


# ---------------------------------------------------------------------------
# Minimal config stubs
# ---------------------------------------------------------------------------


@dataclass
class _RulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 200
    lockCallerHand: bool = True
    num_decks: int = 1


@dataclass
class _Config:
    cambia_rules: _RulesConfig = field(default_factory=_RulesConfig)


# ---------------------------------------------------------------------------
# Helper: RandomAgent wrapper that has the BaseAgent interface
# ---------------------------------------------------------------------------


class _RandomAgentWrapper:
    """Thin wrapper exposing RandomAgent as an agent_wrapper for LBR."""

    def __init__(self, config):
        self._config = config
        self._agent = RandomAgent(0, config)

    def choose_action(self, game_state, legal_actions):
        return self._agent.choose_action(game_state, legal_actions)

    # No initialize_state needed; LBR handles this gracefully


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lbr_valid_result():
    """sampled_lbr returns a dict with the expected keys and correct types."""
    config = _Config()
    agent = _RandomAgentWrapper(config)
    result = sampled_lbr(
        agent, config, num_infosets=20, br_rollouts_per_infoset=5, seed=42
    )

    assert isinstance(result, dict), "Result must be a dict"
    assert "exploitability" in result
    assert "num_infosets_sampled" in result
    assert "std_err" in result
    assert isinstance(result["exploitability"], float)
    assert isinstance(result["num_infosets_sampled"], int)
    assert isinstance(result["std_err"], float)
    assert result["num_infosets_sampled"] >= 0


def test_lbr_non_negative_exploitability():
    """Exploitability is always >= 0 (BR value >= agent value by construction)."""
    config = _Config()
    agent = _RandomAgentWrapper(config)
    result = sampled_lbr(
        agent, config, num_infosets=50, br_rollouts_per_infoset=5, seed=7
    )

    assert result["exploitability"] >= 0.0, (
        f"Exploitability must be >= 0, got {result['exploitability']}"
    )
    assert result["std_err"] >= 0.0


def test_lbr_deterministic_seed():
    """Same seed produces identical results across two calls."""
    config = _Config()
    agent_a = _RandomAgentWrapper(config)
    agent_b = _RandomAgentWrapper(config)

    result_a = sampled_lbr(
        agent_a, config, num_infosets=30, br_rollouts_per_infoset=5, seed=99
    )
    result_b = sampled_lbr(
        agent_b, config, num_infosets=30, br_rollouts_per_infoset=5, seed=99
    )

    assert result_a["exploitability"] == result_b["exploitability"], (
        f"Same seed should give same exploitability: {result_a} vs {result_b}"
    )
    assert result_a["num_infosets_sampled"] == result_b["num_infosets_sampled"]
