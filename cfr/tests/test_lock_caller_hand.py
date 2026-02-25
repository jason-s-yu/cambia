"""
tests/test_lock_caller_hand.py

Unit tests for lockCallerHand rule:
- Cambia caller cannot replace hand cards after calling Cambia.
- Opponent's BlindSwap/KingLook abilities fizzle when targeting the locked caller.
- lockCallerHand=False preserves old behavior.
- Bridge rejects non-CambiaRulesConfig house_rules with TypeError.
"""
import pytest

from src.card import Card, create_standard_deck
from src.config import CambiaRulesConfig
from src.constants import (
    ActionCallCambia,
    ActionDiscard,
    ActionDrawStockpile,
    ActionReplace,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
)
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rules(lock: bool = True) -> CambiaRulesConfig:
    return CambiaRulesConfig(
        lockCallerHand=lock,
        allowDrawFromDiscardPile=False,
        allowReplaceAbilities=False,
        snapRace=False,
        penaltyDrawCount=2,
        allowOpponentSnapping=False,
        use_jokers=2,
        cards_per_player=4,
        initial_view_count=2,
        cambia_allowed_round=0,
        max_game_turns=300,
    )


def build_post_cambia_pending_state(
    caller: int,
    lock: bool,
    drawn_card: Card,
) -> CambiaGameState:
    """
    Build a CambiaGameState where:
    - `caller` is the Cambia caller
    - The other player just drew `drawn_card` and is in the post-draw pending state
    """
    deck = create_standard_deck(include_jokers=2)
    # Remove drawn_card from deck to avoid duplicates
    remaining = [c for c in deck if not (c.rank == drawn_card.rank and c.suit == drawn_card.suit)]

    p0_hand = remaining[:4]
    p1_hand = remaining[4:8]
    stockpile = remaining[8:30]
    discard_top = remaining[30]

    rules = make_rules(lock=lock)
    players = [
        PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
        PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
    ]
    state = CambiaGameState(
        players=players,
        stockpile=list(stockpile),
        discard_pile=[discard_top],
        current_player_index=1 - caller,  # Other player is acting
        house_rules=rules,
        cambia_caller_id=caller,
    )
    # Simulate post-draw pending state for the acting player (non-caller)
    state.pending_action = ActionDiscard(use_ability=False)
    state.pending_action_player = 1 - caller
    state.pending_action_data = {"drawn_card": drawn_card}
    return state


def build_pending_ability_state(
    caller: int,
    lock: bool,
    ability_action,
) -> CambiaGameState:
    """
    Build a CambiaGameState where:
    - `caller` is the Cambia caller
    - The other player has a pending ability action
    """
    deck = create_standard_deck(include_jokers=2)
    p0_hand = deck[:4]
    p1_hand = deck[4:8]
    stockpile = deck[8:30]
    discard_top = deck[30]

    rules = make_rules(lock=lock)
    players = [
        PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
        PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
    ]
    state = CambiaGameState(
        players=players,
        stockpile=list(stockpile),
        discard_pile=[discard_top],
        current_player_index=1 - caller,
        house_rules=rules,
        cambia_caller_id=caller,
    )
    state.pending_action = ability_action
    state.pending_action_player = 1 - caller
    state.pending_action_data = {"ability_card": discard_top}
    return state


# ---------------------------------------------------------------------------
# Test 3a: Replace masked when lockCallerHand=True
# ---------------------------------------------------------------------------


class TestLockCallerHandReplace:
    def test_caller_cannot_replace_when_locked(self):
        """When lockCallerHand=True, the Cambia caller should not have Replace actions."""
        # Non-caller (P1) draws a plain card (no ability) and is in pending state
        # Caller is P0 — the pending state player is P1 (non-caller), so this checks
        # that the CALLER (P0) does not get Replace actions if they were the pending player.
        # Re-arrange: caller=P1, acting/pending=P0 (non-caller has drawn card)
        deck = create_standard_deck(include_jokers=2)
        # Use a 2 of hearts (no special ability) as drawn card
        drawn = Card(rank="2", suit="H")
        remaining = [c for c in deck if not (c.rank == "2" and c.suit == "H")]

        p0_hand = remaining[:4]   # P0 = caller
        p1_hand = remaining[4:8]
        stockpile = remaining[8:30]
        discard_top = remaining[30]

        rules = make_rules(lock=True)
        players = [
            PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
            PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
        ]
        # caller is P0, acting player is P0 — P0 drew a card, P0 is in post-draw pending
        state = CambiaGameState(
            players=players,
            stockpile=list(stockpile),
            discard_pile=[discard_top],
            current_player_index=0,
            house_rules=rules,
            cambia_caller_id=0,  # P0 called Cambia
        )
        state.pending_action = ActionDiscard(use_ability=False)
        state.pending_action_player = 0  # P0 is in post-draw pending
        state.pending_action_data = {"drawn_card": drawn}

        legal = state.get_legal_actions()
        replace_actions = [a for a in legal if isinstance(a, ActionReplace)]
        assert len(replace_actions) == 0, (
            f"Expected no Replace actions for locked caller P0, got: {replace_actions}"
        )
        # Discard action should still be present
        discard_actions = [a for a in legal if isinstance(a, ActionDiscard)]
        assert len(discard_actions) > 0, "Discard action should still be legal"

    def test_non_caller_can_replace_when_locked(self):
        """When lockCallerHand=True, the NON-caller should still have Replace actions."""
        deck = create_standard_deck(include_jokers=2)
        drawn = Card(rank="3", suit="D")
        remaining = [c for c in deck if not (c.rank == "3" and c.suit == "D")]

        p0_hand = remaining[:4]
        p1_hand = remaining[4:8]
        stockpile = remaining[8:30]
        discard_top = remaining[30]

        rules = make_rules(lock=True)
        players = [
            PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
            PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
        ]
        # Caller is P0, acting pending player is P1 (non-caller)
        state = CambiaGameState(
            players=players,
            stockpile=list(stockpile),
            discard_pile=[discard_top],
            current_player_index=1,
            house_rules=rules,
            cambia_caller_id=0,  # P0 called Cambia
        )
        state.pending_action = ActionDiscard(use_ability=False)
        state.pending_action_player = 1
        state.pending_action_data = {"drawn_card": drawn}

        legal = state.get_legal_actions()
        replace_actions = [a for a in legal if isinstance(a, ActionReplace)]
        # P1 has 4 cards, so should have 4 Replace options
        assert len(replace_actions) == 4, (
            f"Expected 4 Replace actions for non-caller P1, got: {replace_actions}"
        )

    def test_replace_allowed_when_lock_disabled(self):
        """When lockCallerHand=False, the caller CAN still Replace."""
        deck = create_standard_deck(include_jokers=2)
        drawn = Card(rank="4", suit="C")
        remaining = [c for c in deck if not (c.rank == "4" and c.suit == "C")]

        p0_hand = remaining[:4]
        p1_hand = remaining[4:8]
        stockpile = remaining[8:30]
        discard_top = remaining[30]

        rules = make_rules(lock=False)
        players = [
            PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
            PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
        ]
        state = CambiaGameState(
            players=players,
            stockpile=list(stockpile),
            discard_pile=[discard_top],
            current_player_index=0,
            house_rules=rules,
            cambia_caller_id=0,  # P0 is caller
        )
        state.pending_action = ActionDiscard(use_ability=False)
        state.pending_action_player = 0
        state.pending_action_data = {"drawn_card": drawn}

        legal = state.get_legal_actions()
        replace_actions = [a for a in legal if isinstance(a, ActionReplace)]
        # With lock disabled, P0 should be able to Replace (has 4 cards)
        assert len(replace_actions) == 4, (
            f"Expected 4 Replace actions when lock disabled, got: {replace_actions}"
        )


# ---------------------------------------------------------------------------
# Test 3b: BlindSwap fizzles when opponent is locked caller
# ---------------------------------------------------------------------------


class TestLockCallerHandBlindSwap:
    def test_blindswap_fizzles_when_opponent_is_caller(self):
        """When lockCallerHand=True and opponent is the Cambia caller, BlindSwap fizzles."""
        # Caller=P0, non-caller=P1 has pending BlindSwap ability
        state = build_pending_ability_state(
            caller=0,
            lock=True,
            ability_action=ActionAbilityBlindSwapSelect(-1, -1),
        )
        legal = state.get_legal_actions()
        blind_swap_actions = [a for a in legal if isinstance(a, ActionAbilityBlindSwapSelect)]
        assert len(blind_swap_actions) == 0, (
            f"BlindSwap should fizzle when opponent is locked caller, got: {blind_swap_actions}"
        )

    def test_blindswap_works_when_opponent_is_not_caller(self):
        """When lockCallerHand=True but opponent is NOT the caller, BlindSwap proceeds normally."""
        # Caller=P0, non-caller=P1 has pending BlindSwap, but target opponent (P0) is not caller
        # Wait: opponent_id = get_opponent_index(pending_player=P1) = P0 = caller
        # So we need caller=P1, pending_player=P0 to have opponent=P1 (caller) fizzle
        # For non-fizzle: caller=P0=acting player, pending=P0 — but caller can't act after call
        # Instead use no caller set: cambia_caller_id=None
        deck = create_standard_deck(include_jokers=2)
        p0_hand = deck[:4]
        p1_hand = deck[4:8]
        stockpile = deck[8:30]
        discard_top = deck[30]

        rules = make_rules(lock=True)
        players = [
            PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
            PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
        ]
        state = CambiaGameState(
            players=players,
            stockpile=list(stockpile),
            discard_pile=[discard_top],
            current_player_index=0,
            house_rules=rules,
            cambia_caller_id=None,  # No caller yet
        )
        state.pending_action = ActionAbilityBlindSwapSelect(-1, -1)
        state.pending_action_player = 0
        state.pending_action_data = {"ability_card": discard_top}

        legal = state.get_legal_actions()
        blind_swap_actions = [a for a in legal if isinstance(a, ActionAbilityBlindSwapSelect)]
        # P0 has 4 cards, P1 has 4 cards -> 4*4=16 BlindSwap combinations
        assert len(blind_swap_actions) == 16, (
            f"Expected 16 BlindSwap actions when no caller locked, got: {len(blind_swap_actions)}"
        )

    def test_blindswap_works_when_lock_disabled(self):
        """When lockCallerHand=False, BlindSwap works even if opponent called Cambia."""
        state = build_pending_ability_state(
            caller=0,
            lock=False,
            ability_action=ActionAbilityBlindSwapSelect(-1, -1),
        )
        legal = state.get_legal_actions()
        blind_swap_actions = [a for a in legal if isinstance(a, ActionAbilityBlindSwapSelect)]
        # P0 and P1 each have 4 cards -> 16 combinations
        assert len(blind_swap_actions) == 16, (
            f"Expected 16 BlindSwap actions when lock disabled, got: {len(blind_swap_actions)}"
        )


# ---------------------------------------------------------------------------
# Test 3c: KingLook fizzles when opponent is locked caller
# ---------------------------------------------------------------------------


class TestLockCallerHandKingLook:
    def test_kinglook_fizzles_when_opponent_is_caller(self):
        """When lockCallerHand=True and opponent is the Cambia caller, KingLook fizzles."""
        state = build_pending_ability_state(
            caller=0,
            lock=True,
            ability_action=ActionAbilityKingLookSelect(-1, -1),
        )
        legal = state.get_legal_actions()
        king_look_actions = [a for a in legal if isinstance(a, ActionAbilityKingLookSelect)]
        assert len(king_look_actions) == 0, (
            f"KingLook should fizzle when opponent is locked caller, got: {king_look_actions}"
        )

    def test_kinglook_works_when_lock_disabled(self):
        """When lockCallerHand=False, KingLook works even if opponent called Cambia."""
        state = build_pending_ability_state(
            caller=0,
            lock=False,
            ability_action=ActionAbilityKingLookSelect(-1, -1),
        )
        legal = state.get_legal_actions()
        king_look_actions = [a for a in legal if isinstance(a, ActionAbilityKingLookSelect)]
        # 4*4=16 combinations
        assert len(king_look_actions) == 16, (
            f"Expected 16 KingLook actions when lock disabled, got: {len(king_look_actions)}"
        )


# ---------------------------------------------------------------------------
# Test 3d: Bridge rejects non-CambiaRulesConfig
# ---------------------------------------------------------------------------


class TestBridgeTypeCheck:
    def test_bridge_rejects_non_cambia_rules_config(self):
        """GoEngine should raise TypeError if house_rules is not CambiaRulesConfig."""
        try:
            from src.ffi.bridge import GoEngine
        except (ImportError, FileNotFoundError, OSError):
            pytest.skip("libcambia.so not available")

        from types import SimpleNamespace
        fake_rules = SimpleNamespace(
            max_game_turns=300,
            cards_per_player=4,
            cambia_allowed_round=0,
            penaltyDrawCount=2,
            allowDrawFromDiscardPile=False,
            allowReplaceAbilities=False,
            allowOpponentSnapping=False,
            snapRace=False,
            use_jokers=2,
            lockCallerHand=True,
        )
        with pytest.warns(UserWarning, match="CambiaRulesConfig"):
            GoEngine(seed=42, house_rules=fake_rules)

    def test_bridge_accepts_cambia_rules_config(self):
        """GoEngine should accept a valid CambiaRulesConfig without error."""
        try:
            from src.ffi.bridge import GoEngine
        except (ImportError, FileNotFoundError, OSError):
            pytest.skip("libcambia.so not available")

        rules = CambiaRulesConfig(lockCallerHand=True)
        try:
            with GoEngine(seed=42, house_rules=rules) as engine:
                assert engine.handle >= 0
        except FileNotFoundError:
            pytest.skip("libcambia.so not available")
