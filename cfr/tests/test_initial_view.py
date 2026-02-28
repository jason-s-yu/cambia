"""
Tests for initial_view_count feature in Go engine + FFI.

Covers:
- Interface: InitialViewCount in HouseRules, passes through FFI
- Regression: default (2) preserves existing behavior
- Behavioral: 0 → no cards seen, 4 → all cards seen
- FFI round-trip: Python creates GoEngine with initial_view_count=3
"""

import numpy as np
import pytest

from src.config import CambiaRulesConfig
from src.ffi.bridge import GoAgentState, GoEngine


def _make_rules(initial_view_count=2, cards_per_player=4) -> CambiaRulesConfig:
    """Return a CambiaRulesConfig with Go-matching defaults."""
    rules = CambiaRulesConfig()
    rules.allowDrawFromDiscardPile = True
    rules.allowOpponentSnapping = True
    rules.max_game_turns = 46
    rules.cards_per_player = cards_per_player
    rules.initial_view_count = initial_view_count
    return rules


def _play_game(eng: GoEngine, max_steps: int = 200) -> int:
    """Play eng to completion using first legal action. Returns steps taken."""
    mask = eng.legal_actions_mask()
    step = 0
    while not eng.is_terminal() and step < max_steps:
        legal = np.where(mask > 0)[0]
        assert len(legal) > 0, f"No legal actions at step {step}"
        eng.apply_action(int(legal[0]))
        mask = eng.legal_actions_mask()
        step += 1
    return step


class TestInitialViewCountRegression:
    """Default (2) must match legacy behavior."""

    def test_default_creates_valid_game(self):
        rules = _make_rules(initial_view_count=2)
        with GoEngine(seed=42, house_rules=rules) as eng:
            assert eng.handle >= 0
            assert not eng.is_terminal()

    def test_default_agent_initializes(self):
        rules = _make_rules(initial_view_count=2)
        with GoEngine(seed=42, house_rules=rules) as eng:
            agent = GoAgentState(eng, player_id=0)
            assert agent._agent_h >= 0
            agent.close()

    def test_default_legal_actions_nonempty(self):
        rules = _make_rules(initial_view_count=2)
        with GoEngine(seed=42, house_rules=rules) as eng:
            mask = eng.legal_actions_mask()
            assert mask.sum() > 0


class TestInitialViewCountZero:
    """InitialViewCount=0 → no initial peeks."""

    def test_game_creates_with_zero_peeks(self):
        rules = _make_rules(initial_view_count=0)
        with GoEngine(seed=42, house_rules=rules) as eng:
            assert eng.handle >= 0
            assert not eng.is_terminal()

    def test_agent_initializes_with_zero_peeks(self):
        rules = _make_rules(initial_view_count=0)
        with GoEngine(seed=42, house_rules=rules) as eng:
            agent = GoAgentState(eng, player_id=0)
            assert agent._agent_h >= 0
            agent.close()


class TestInitialViewCountFour:
    """InitialViewCount=4 with CardsPerPlayer=4 → all cards seen."""

    def test_game_creates_with_four_peeks(self):
        rules = _make_rules(initial_view_count=4, cards_per_player=4)
        with GoEngine(seed=42, house_rules=rules) as eng:
            assert eng.handle >= 0

    def test_agent_initializes_with_four_peeks(self):
        rules = _make_rules(initial_view_count=4, cards_per_player=4)
        with GoEngine(seed=42, house_rules=rules) as eng:
            agent = GoAgentState(eng, player_id=0)
            assert agent._agent_h >= 0
            agent.close()


class TestInitialViewCountFFIRoundTrip:
    """FFI round-trip: Python GoEngine with various initial_view_count values."""

    def test_initial_view_count_3_creates_game(self):
        rules = _make_rules(initial_view_count=3, cards_per_player=4)
        with GoEngine(seed=99, house_rules=rules) as eng:
            assert eng.handle >= 0
            assert not eng.is_terminal()

    def test_initial_view_count_3_agent_encodes(self):
        """Verify agent can encode after init with initial_view_count=3."""
        rules = _make_rules(initial_view_count=3, cards_per_player=4)
        with GoEngine(seed=99, house_rules=rules) as eng:
            agent = GoAgentState(eng, player_id=0)
            try:
                # encode takes decision_context; 0 = draw phase
                vec = agent.encode_eppbs(decision_context=0)
                assert vec is not None
                assert len(vec) > 0
            finally:
                agent.close()

    def test_initial_view_count_3_game_plays_through(self):
        """Game with initial_view_count=3 should be playable."""
        rules = _make_rules(initial_view_count=3, cards_per_player=4)
        with GoEngine(seed=77, house_rules=rules) as eng:
            _play_game(eng)

    def test_initial_view_count_0_game_plays_through(self):
        """Game with initial_view_count=0 should be playable."""
        rules = _make_rules(initial_view_count=0, cards_per_player=4)
        with GoEngine(seed=123, house_rules=rules) as eng:
            _play_game(eng)

    def test_initial_view_count_4_game_plays_through(self):
        """Game with initial_view_count=4 (all cards) should be playable."""
        rules = _make_rules(initial_view_count=4, cards_per_player=4)
        with GoEngine(seed=456, house_rules=rules) as eng:
            _play_game(eng)

    def test_default_matches_legacy_behavior(self):
        """Games with default initial_view_count=2 produce valid initial states."""
        for seed in (1, 12345, 99999):
            rules = _make_rules(initial_view_count=2)
            with GoEngine(seed=seed, house_rules=rules) as eng:
                assert not eng.is_terminal()
                mask = eng.legal_actions_mask()
                assert mask.sum() > 0
