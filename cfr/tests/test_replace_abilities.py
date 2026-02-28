"""
tests/test_replace_abilities.py

Tests for AllowReplaceAbilities house rule:
- Interface: field exists on HouseRules, passes through config
- Regression: with default (false), replace NEVER triggers abilities
- Behavioral: with true, replacing from stockpile triggers old card's ability;
  replacing from discard does NOT; replacing non-ability card does NOT
"""

import pytest
from dataclasses import dataclass

from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.card import Card
from src.constants import (
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    JACK,
    QUEEN,
    KING,
    ACE,
    TWO,
    THREE,
    HEARTS,
    CLUBS,
    DIAMONDS,
    SPADES,
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionReplace,
    ActionDiscard,
    ActionPassSnap,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
)
from src.game._ability_mixin import AbilityMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rules(**kwargs):
    """Create a CambiaRulesConfig from the stub or real module."""
    from src.config import CambiaRulesConfig as _RC
    try:
        obj = _RC(**kwargs)
    except TypeError:
        obj = _RC()
        for k, v in kwargs.items():
            setattr(obj, k, v)
    return obj


def _make_game(rules_kwargs=None) -> CambiaGameState:
    """Create a minimal 2-player game with 4 cards each in hand."""
    if rules_kwargs is None:
        rules_kwargs = {}
    rules = _make_rules(**rules_kwargs)

    # Build simple hands: 4 known cards per player
    p0_hand = [Card(rank=ACE, suit=HEARTS), Card(rank=TWO, suit=CLUBS),
               Card(rank=THREE, suit=DIAMONDS), Card(rank=TWO, suit=SPADES)]
    p1_hand = [Card(rank=ACE, suit=CLUBS), Card(rank=TWO, suit=HEARTS),
               Card(rank=THREE, suit=SPADES), Card(rank=ACE, suit=DIAMONDS)]

    # Build stockpile (ability card on top — last in list)
    stockpile = [Card(rank=THREE, suit=CLUBS), Card(rank=ACE, suit=SPADES)]

    players = [
        PlayerState(hand=p0_hand, initial_peek_indices=(0, 1)),
        PlayerState(hand=p1_hand, initial_peek_indices=(0, 1)),
    ]
    return CambiaGameState(
        players=players,
        stockpile=stockpile,
        discard_pile=[Card(rank=TWO, suit=DIAMONDS)],
        current_player_index=0,
        house_rules=rules,
    )


def _make_game_with_ability_card_in_hand(hand_rank: str, rules_kwargs=None) -> CambiaGameState:
    """Make a game where p0 hand[0] is an ability card, non-ability on stockpile top."""
    if rules_kwargs is None:
        rules_kwargs = {}
    rules = _make_rules(**rules_kwargs)

    ability_card = Card(rank=hand_rank, suit=HEARTS)
    p0_hand = [ability_card, Card(rank=TWO, suit=CLUBS),
               Card(rank=THREE, suit=DIAMONDS), Card(rank=TWO, suit=SPADES)]
    p1_hand = [Card(rank=ACE, suit=CLUBS), Card(rank=TWO, suit=HEARTS),
               Card(rank=THREE, suit=SPADES), Card(rank=ACE, suit=DIAMONDS)]

    # Non-ability card on top of stockpile (last element)
    stockpile = [Card(rank=THREE, suit=CLUBS), Card(rank=ACE, suit=SPADES)]

    players = [
        PlayerState(hand=p0_hand, initial_peek_indices=(0, 1)),
        PlayerState(hand=p1_hand, initial_peek_indices=(0, 1)),
    ]
    return CambiaGameState(
        players=players,
        stockpile=stockpile,
        discard_pile=[Card(rank=TWO, suit=DIAMONDS)],
        current_player_index=0,
        house_rules=rules,
    )


def _pass_snap(game: CambiaGameState):
    """Pass through all active snap windows."""
    legal = game.get_legal_actions()
    while any(isinstance(a, ActionPassSnap) for a in legal):
        game.apply_action(ActionPassSnap())
        legal = game.get_legal_actions()


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestAllowReplaceAbilitiesInterface:
    def test_field_exists_on_rules(self):
        """allowReplaceAbilities field exists on CambiaRulesConfig."""
        rules = _make_rules()
        assert hasattr(rules, "allowReplaceAbilities")

    def test_default_is_false(self):
        """AllowReplaceAbilities defaults to False."""
        rules = _make_rules()
        assert rules.allowReplaceAbilities is False

    def test_can_be_set_true(self):
        """AllowReplaceAbilities can be set to True."""
        rules = _make_rules(allowReplaceAbilities=True)
        assert rules.allowReplaceAbilities is True

    def test_game_reads_rule(self):
        """CambiaGameState exposes the house_rules with allowReplaceAbilities."""
        game = _make_game({"allowReplaceAbilities": True})
        assert game.house_rules.allowReplaceAbilities is True


# ---------------------------------------------------------------------------
# Regression tests — default False, no ability should fire
# ---------------------------------------------------------------------------

class TestReplaceAbilitiesDefaultOff:
    """With AllowReplaceAbilities=False (default), replace never triggers ability."""

    @pytest.mark.parametrize("rank", [SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING])
    def test_no_ability_on_replace_default(self, rank):
        """Replacing any ability card from stockpile does NOT trigger ability (default)."""
        game = _make_game_with_ability_card_in_hand(rank, {})  # default: False

        # Draw non-ability card from stockpile.
        game.apply_action(ActionDrawStockpile())

        # Replace hand[0] (ability card) with drawn card.
        game.apply_action(ActionReplace(target_hand_index=0))

        # Legal actions should NOT include any ability resolution actions.
        legal = game.get_legal_actions()
        ability_types = (
            ActionAbilityPeekOwnSelect,
            ActionAbilityPeekOtherSelect,
            ActionAbilityBlindSwapSelect,
            ActionAbilityKingLookSelect,
        )
        for a in legal:
            assert not isinstance(a, ability_types), (
                f"rank={rank}: ability action {a!r} should not be legal after replace "
                f"with AllowReplaceAbilities=False"
            )

    def test_no_ability_non_ability_card(self):
        """Replacing a non-ability card (default or enabled) never triggers ability."""
        game = _make_game({"allowReplaceAbilities": True})

        # Draw from stockpile (top = ACE, no ability).
        game.apply_action(ActionDrawStockpile())

        # Replace hand[0] (also non-ability card).
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        ability_types = (
            ActionAbilityPeekOwnSelect,
            ActionAbilityPeekOtherSelect,
            ActionAbilityBlindSwapSelect,
            ActionAbilityKingLookSelect,
        )
        for a in legal:
            assert not isinstance(a, ability_types), (
                f"non-ability card replace should not trigger ability, got {a!r}"
            )


# ---------------------------------------------------------------------------
# Behavioral tests — AllowReplaceAbilities=True
# ---------------------------------------------------------------------------

class TestReplaceAbilitiesEnabled:
    """With AllowReplaceAbilities=True, replacing ability cards from stockpile triggers ability."""

    def test_replace_seven_triggers_peek_own(self):
        """Replacing a 7 from stockpile triggers PeekOwn (ActionAbilityPeekOwnSelect)."""
        game = _make_game_with_ability_card_in_hand(SEVEN, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityPeekOwnSelect) for a in legal), (
            f"Expected PeekOwn ability action after replacing 7 from stockpile. Got: {legal}"
        )

    def test_replace_eight_triggers_peek_own(self):
        """Replacing an 8 from stockpile triggers PeekOwn."""
        game = _make_game_with_ability_card_in_hand(EIGHT, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityPeekOwnSelect) for a in legal)

    def test_replace_nine_triggers_peek_other(self):
        """Replacing a 9 from stockpile triggers PeekOther."""
        game = _make_game_with_ability_card_in_hand(NINE, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityPeekOtherSelect) for a in legal)

    def test_replace_ten_triggers_peek_other(self):
        """Replacing a T from stockpile triggers PeekOther."""
        game = _make_game_with_ability_card_in_hand(TEN, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityPeekOtherSelect) for a in legal)

    def test_replace_jack_triggers_blind_swap(self):
        """Replacing a J from stockpile triggers BlindSwap."""
        game = _make_game_with_ability_card_in_hand(JACK, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityBlindSwapSelect) for a in legal)

    def test_replace_queen_triggers_blind_swap(self):
        """Replacing a Q from stockpile triggers BlindSwap."""
        game = _make_game_with_ability_card_in_hand(QUEEN, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityBlindSwapSelect) for a in legal)

    def test_replace_king_triggers_king_look(self):
        """Replacing a K from stockpile triggers KingLook."""
        game = _make_game_with_ability_card_in_hand(KING, {"allowReplaceAbilities": True})
        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        assert any(isinstance(a, ActionAbilityKingLookSelect) for a in legal)

    def test_replace_from_discard_no_ability(self):
        """Replacing with a card drawn from the discard pile does NOT trigger ability,
        even when AllowReplaceAbilities=True."""
        rules = _make_rules(allowReplaceAbilities=True, allowDrawFromDiscardPile=True)

        # Place an ability card (7) as the top of the discard pile.
        ability_card = Card(rank=SEVEN, suit=CLUBS)
        p0_hand = [Card(rank=ACE, suit=HEARTS), Card(rank=TWO, suit=CLUBS),
                   Card(rank=THREE, suit=DIAMONDS), Card(rank=TWO, suit=SPADES)]
        p1_hand = [Card(rank=ACE, suit=CLUBS), Card(rank=TWO, suit=HEARTS),
                   Card(rank=THREE, suit=SPADES), Card(rank=ACE, suit=DIAMONDS)]
        players = [
            PlayerState(hand=p0_hand, initial_peek_indices=(0, 1)),
            PlayerState(hand=p1_hand, initial_peek_indices=(0, 1)),
        ]
        game = CambiaGameState(
            players=players,
            stockpile=[Card(rank=THREE, suit=CLUBS)],
            discard_pile=[ability_card],  # 7 on top of discard
            current_player_index=0,
            house_rules=rules,
        )

        # Draw the 7 from DISCARD pile.
        game.apply_action(ActionDrawDiscard())

        # Replace hand[0] (non-ability Ace) with the drawn 7.
        # The replaced card (Ace) has no ability, and the 7 was drawn from discard.
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        ability_types = (
            ActionAbilityPeekOwnSelect,
            ActionAbilityPeekOtherSelect,
            ActionAbilityBlindSwapSelect,
            ActionAbilityKingLookSelect,
        )
        for a in legal:
            assert not isinstance(a, ability_types), (
                f"No ability should trigger when replacing with discard-drawn card. Got: {a!r}"
            )

    def test_replace_seven_full_ability_resolution(self):
        """Full flow: replace 7 → PeekOwn → select → snap phase → turn advances."""
        game = _make_game_with_ability_card_in_hand(SEVEN, {"allowReplaceAbilities": True})
        turn_before = game._turn_number

        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        # Should be in ability selection phase.
        legal = game.get_legal_actions()
        peek_actions = [a for a in legal if isinstance(a, ActionAbilityPeekOwnSelect)]
        assert peek_actions, "Expected PeekOwn actions"

        # Resolve peek at own hand[1].
        game.apply_action(ActionAbilityPeekOwnSelect(target_hand_index=1))

        # Pass snap phase.
        _pass_snap(game)

        # Turn should have advanced.
        assert game._turn_number > turn_before, (
            "Turn number should advance after complete replace+ability resolution"
        )

    def test_replace_queen_full_ability_resolution(self):
        """Full flow: replace Q → BlindSwap → select own+opp → snap → turn advances."""
        game = _make_game_with_ability_card_in_hand(QUEEN, {"allowReplaceAbilities": True})
        turn_before = game._turn_number

        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        swap_actions = [a for a in legal if isinstance(a, ActionAbilityBlindSwapSelect)]
        assert swap_actions, "Expected BlindSwap actions"

        # Do the swap: own[1] ↔ opp[0].
        game.apply_action(ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=0))

        _pass_snap(game)

        assert game._turn_number > turn_before

    def test_replace_king_full_ability_resolution_no_swap(self):
        """Full flow: replace K → KingLook → KingSwapDecision(no) → snap → turn advances."""
        game = _make_game_with_ability_card_in_hand(KING, {"allowReplaceAbilities": True})
        turn_before = game._turn_number

        game.apply_action(ActionDrawStockpile())
        game.apply_action(ActionReplace(target_hand_index=0))

        legal = game.get_legal_actions()
        look_actions = [a for a in legal if isinstance(a, ActionAbilityKingLookSelect)]
        assert look_actions, "Expected KingLook actions"

        # Look at own[1] and opp[0].
        game.apply_action(ActionAbilityKingLookSelect(own_hand_index=1, opponent_hand_index=0))

        # After look, should be in KingSwapDecision.
        legal = game.get_legal_actions()
        swap_decisions = [a for a in legal if isinstance(a, ActionAbilityKingSwapDecision)]
        assert swap_decisions, "Expected KingSwapDecision after KingLook"

        # Decide not to swap.
        game.apply_action(ActionAbilityKingSwapDecision(perform_swap=False))

        _pass_snap(game)

        assert game._turn_number > turn_before
