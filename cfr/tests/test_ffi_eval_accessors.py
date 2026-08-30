"""
tests/test_ffi_eval_accessors.py

Tests for the cambia-1425 evaluation-surface accessors: the GoEngine methods
wrapping the read-only cambia_game_get_* exports, the GameView protocol they
satisfy, and the PythonGameView adapter over the Python reference engine.

Every case runs on a scripted deck so the assertions pin exact card identities
rather than whatever a seeded shuffle happened to deal.

Requires libcambia.so to be built and available (skipped otherwise).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.card import Card
from src.config import CambiaRulesConfig
from src.constants import ActionDrawStockpile
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState


def _go_available() -> bool:
    try:
        from src.ffi.bridge import GoEngine

        engine = GoEngine.from_deck(list(range(54)))
        engine.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")

if go_available:
    from src.agents.game_view import GameView, PythonGameView
    from src.ffi.bridge import (
        CARD_INDEX_NONE,
        DRAWN_FROM_DISCARD,
        DRAWN_FROM_STOCKPILE,
        NUM_CARD_INDICES,
        PENDING_DISCARD,
        PENDING_KING_DECISION,
        PENDING_NONE,
        PENDING_SNAP_MOVE,
        GoEngine,
        card_from_index,
        python_card_to_go_index,
        rank_from_index,
    )

# ---------------------------------------------------------------------------
# Scripted deck
# ---------------------------------------------------------------------------
#
# Canonical card indices (suit*13 + rank; C=0, D=1, H=2, S=3; ranks A=0..K=12):
#
#   0  = C-A   1  = C-2   2  = C-3   3  = C-4
#   13 = D-A   14 = D-2   15 = D-3   16 = D-4
#   25 = D-K (red King)   26 = H-A   27 = H-2

CARD_CA, CARD_C2, CARD_C3, CARD_C4 = 0, 1, 2, 3
CARD_DA, CARD_D2, CARD_D3, CARD_D4 = 13, 14, 15, 16
CARD_DK, CARD_HA, CARD_H2 = 25, 26, 27

# Raw action indices (engine/types.go). The FFI applies actions by index; the
# Python-side GameAction types do not carry one.
ACTION_DRAW_STOCKPILE = 0
ACTION_DRAW_DISCARD = 1
ACTION_DISCARD_NO_ABILITY = 3
ACTION_DISCARD_WITH_ABILITY = 4
ACTION_BASE_KING_LOOK = 59
ACTION_BASE_SNAP_OPPONENT = 104

CARDS_PER_PLAYER = 4


def scripted_deck(next_draw: int) -> List[int]:
    """Deal-order deck fixing both hands, the discard flip and the next draw.

    deck[0] goes to seat 0 slot 0, deck[1] to seat 1 slot 0, and so on
    round-robin; deck[8] is the discard flip; deck[9] is the first card drawn
    from the stockpile. The rest is filler in ascending index order.
    """
    head = [
        CARD_CA,
        CARD_DA,
        CARD_C2,
        CARD_D2,
        CARD_C3,
        CARD_D3,
        CARD_C4,
        CARD_D4,
        CARD_HA,
        next_draw,
    ]
    used = set(head)
    return head + [i for i in range(NUM_CARD_INDICES) if i not in used]


def go_game(next_draw: int, rules=None) -> "GoEngine":
    """A GoEngine dealt from the scripted deck."""
    return GoEngine.from_deck(scripted_deck(next_draw), 0, rules)


def scripted_rules() -> CambiaRulesConfig:
    """House rules pinned to explicit values on every field the view reports."""
    rules = CambiaRulesConfig()
    rules.max_game_turns = 46
    rules.cards_per_player = CARDS_PER_PLAYER
    rules.cambia_allowed_round = 0
    rules.penaltyDrawCount = 2
    rules.allowDrawFromDiscardPile = True
    rules.allowReplaceAbilities = False
    rules.allowOpponentSnapping = True
    rules.snapRace = False
    rules.use_jokers = 2
    rules.initial_view_count = 2
    return rules


def python_game(next_draw: int, rules=None) -> CambiaGameState:
    """A Python CambiaGameState dealt from the same scripted deck."""
    deck = scripted_deck(next_draw)
    hands: List[List[Card]] = [[], []]
    for c in range(CARDS_PER_PLAYER):
        for p in range(2):
            hands[p].append(card_from_index(deck[c * 2 + p]))
    discard = [card_from_index(deck[8])]
    # Python pops the stockpile from the end, so stockpile[-1] is the next draw.
    stockpile = [card_from_index(i) for i in reversed(deck[9:])]
    return CambiaGameState(
        players=[
            PlayerState(hand=hands[p], initial_peek_indices=(0, 1)) for p in range(2)
        ],
        stockpile=stockpile,
        discard_pile=discard,
        current_player_index=0,
        house_rules=rules if rules is not None else scripted_rules(),
    )


def ident(card: Optional[Card]) -> Optional[Tuple[str, Optional[str]]]:
    """(rank, suit) identity of a card.

    Card.__eq__ compares rank only (suit is compare=False), so `==` cannot tell
    a red King from a black one. Every card assertion here goes through this.
    """
    return None if card is None else (card.rank, card.suit)


def idents(cards) -> List[Tuple[str, Optional[str]]]:
    return [ident(c) for c in cards]


# ---------------------------------------------------------------------------
# Card index decoding
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestCardIndexDecoding:
    def test_round_trips_through_python_card(self):
        """Every non-joker index decodes to the card that re-encodes to it."""
        for idx in range(52):
            card = card_from_index(idx)
            assert card is not None
            assert python_card_to_go_index(card) == idx

    def test_jokers_decode_suitless(self):
        for idx in (52, 53):
            card = card_from_index(idx)
            assert card is not None
            assert card.rank == "R"
            assert card.suit is None
            assert card.value == 0

    def test_absent_marker_and_out_of_range_decode_to_none(self):
        assert card_from_index(CARD_INDEX_NONE) is None
        assert card_from_index(-1) is None
        assert card_from_index(NUM_CARD_INDICES) is None

    def test_red_and_black_kings_keep_their_values(self):
        """The full card index is what separates a -1 King from a 13 King."""
        assert card_from_index(CARD_DK).value == -1  # D-K
        assert card_from_index(12).value == 13  # C-K

    def test_rank_from_index(self):
        assert rank_from_index(0) == "A"
        assert rank_from_index(1) == "2"
        assert rank_from_index(9) == "T"
        assert rank_from_index(12) == "K"
        assert rank_from_index(13) == "R"
        assert rank_from_index(14) is None
        assert rank_from_index(-1) is None


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestGetPlayerHand:
    def test_returns_scripted_hands(self):
        with go_game(CARD_H2) as engine:
            assert idents(engine.get_player_hand(0)) == [
                ("A", "C"),
                ("2", "C"),
                ("3", "C"),
                ("4", "C"),
            ]
            assert idents(engine.get_player_hand(1)) == [
                ("A", "D"),
                ("2", "D"),
                ("3", "D"),
                ("4", "D"),
            ]

    def test_indices_flavour_matches_card_flavour(self):
        with go_game(CARD_H2) as engine:
            assert engine.get_hand_indices(0) == [CARD_CA, CARD_C2, CARD_C3, CARD_C4]
            assert engine.get_hand_indices(1) == [CARD_DA, CARD_D2, CARD_D3, CARD_D4]

    def test_truncates_to_hand_length_without_markers(self):
        """The list stops at the hand length, so no absent marker leaks out."""
        with go_game(CARD_H2) as engine:
            hand = engine.get_hand_indices(0)
            assert len(hand) == CARDS_PER_PLAYER
            assert CARD_INDEX_NONE not in hand

    def test_rejects_seat_past_the_table(self):
        with go_game(CARD_H2) as engine:
            with pytest.raises(RuntimeError, match="cambia_game_get_hand"):
                engine.get_player_hand(2)

    def test_rejects_negative_seat(self):
        """Caught before the uint8_t conversion, so it reads as a seat error."""
        with go_game(CARD_H2) as engine:
            with pytest.raises(ValueError, match="seat -1 out of range"):
                engine.get_player_hand(-1)


# ---------------------------------------------------------------------------
# Discard pile
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestDiscardPile:
    def test_initial_pile_is_the_flip_card(self):
        with go_game(CARD_H2) as engine:
            assert engine.discard_len() == 1
            assert idents(engine.get_discard_pile()) == [("A", "H")]
            assert ident(engine.get_discard_top()) == ("A", "H")

    def test_pile_grows_bottom_to_top(self):
        with go_game(CARD_H2) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)
            assert engine.discard_len() == 2
            assert idents(engine.get_discard_pile()) == [("A", "H"), ("2", "H")]
            assert ident(engine.get_discard_top()) == ("2", "H")

    def test_top_card_keeps_the_suit_the_bucket_drops(self):
        """discard_top() buckets; get_discard_top() keeps rank and suit."""
        with go_game(CARD_DK) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)
            top = engine.get_discard_top()
            assert ident(top) == ("K", "D")
            assert top.value == -1
            # The bucket only says "a King of some colour".
            assert isinstance(engine.discard_top(), int)


# ---------------------------------------------------------------------------
# Pending record
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestPending:
    def test_no_pending_after_the_deal(self):
        with go_game(CARD_H2) as engine:
            pending = engine.get_pending()
            assert pending.type == PENDING_NONE
            assert not pending.is_pending
            assert pending.seat is None
            assert pending.drawn_card is None
            assert pending.drawn_from is None
            assert pending.own_slot is None
            assert pending.target_slot is None
            assert pending.target_seat is None
            assert pending.own_card is None
            assert pending.target_card is None

    def test_draw_from_stockpile_carries_the_drawn_card(self):
        with go_game(CARD_DK) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            pending = engine.get_pending()
            assert pending.type == PENDING_DISCARD
            assert pending.is_pending
            assert pending.seat == 0
            assert ident(pending.drawn_card) == ("K", "D")
            assert pending.drawn_card.value == -1
            assert pending.drawn_from == DRAWN_FROM_STOCKPILE
            assert pending.own_slot is None
            assert pending.target_slot is None
            assert pending.target_seat is None

    def test_draw_from_discard_reports_its_source(self):
        with go_game(CARD_H2) as engine:
            engine.apply_action(ACTION_DRAW_DISCARD)
            pending = engine.get_pending()
            assert pending.type == PENDING_DISCARD
            assert ident(pending.drawn_card) == ("A", "H")
            assert pending.drawn_from == DRAWN_FROM_DISCARD

    def test_king_decision_carries_both_looked_cards(self):
        with go_game(CARD_DK) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_WITH_ABILITY)
            # Look at own slot 0 (C-A) and the opponent's slot 1 (D-2).
            engine.apply_action(ACTION_BASE_KING_LOOK + 0 * 6 + 1)

            pending = engine.get_pending()
            assert pending.type == PENDING_KING_DECISION
            assert pending.seat == 0
            assert pending.own_slot == 0
            assert pending.target_slot == 1
            assert pending.target_seat == 1
            assert ident(pending.own_card) == ("A", "C")
            assert ident(pending.target_card) == ("2", "D")
            assert pending.drawn_card is None
            assert pending.drawn_from is None

    def test_snap_move_names_the_snapped_seat_and_slot(self):
        with go_game(CARD_H2) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)
            # Seat 0 snaps the opponent's slot 1 (D-2), and now owes a card to
            # the vacated slot.
            engine.apply_action(ACTION_BASE_SNAP_OPPONENT + 1)

            pending = engine.get_pending()
            assert pending.type == PENDING_SNAP_MOVE
            assert pending.seat == 0
            assert pending.target_seat == 1
            assert pending.target_slot == 1
            assert pending.drawn_card is None
            assert pending.own_slot is None


# ---------------------------------------------------------------------------
# Snap state
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestSnapState:
    def test_inactive_after_the_deal(self):
        with go_game(CARD_H2) as engine:
            snap = engine.get_snap_state()
            assert snap.active is False
            assert snap.rank is None
            assert snap.card is None
            assert snap.snapper_count == 0
            assert snap.snapper_seat is None

    def test_active_after_a_matched_discard(self):
        # H-2 discarded opens a rank-2 window: both seats hold a 2.
        with go_game(CARD_H2) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)

            snap = engine.get_snap_state()
            assert snap.active is True
            assert snap.rank == "2"
            assert ident(snap.card) == ("2", "H")
            assert snap.snapper_count == 2
            assert snap.snapper_cursor == 0
            assert snap.snapper_seat == 0

    def test_no_window_when_nothing_matches(self):
        # A red King discarded matches nothing in either scripted hand.
        with go_game(CARD_DK) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)
            assert engine.get_snap_state().active is False


# ---------------------------------------------------------------------------
# House rules
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestHouseRules:
    def test_reads_back_what_the_game_was_built_with(self):
        rules = scripted_rules()
        with go_game(CARD_H2, rules) as engine:
            view = engine.get_house_rules()
            assert view.max_game_turns == 46
            assert view.cards_per_player == CARDS_PER_PLAYER
            assert view.cambia_allowed_round == 0
            assert view.penalty_draw_count == 2
            assert view.allow_draw_from_discard is True
            assert view.allow_replace_abilities is False
            assert view.allow_opponent_snapping is True
            assert view.snap_race is False
            assert view.num_jokers == 2
            assert view.num_players == 2
            assert view.initial_view_count == 2
            assert view.num_decks == 1

    def test_flags_track_the_rules_they_were_built_with(self):
        rules = scripted_rules()
        rules.allowDrawFromDiscardPile = False
        rules.allowReplaceAbilities = True
        rules.allowOpponentSnapping = False
        rules.use_jokers = 0
        rules.initial_view_count = 1
        rules.cambia_allowed_round = 3
        with go_game(CARD_H2, rules) as engine:
            view = engine.get_house_rules()
            assert view.allow_draw_from_discard is False
            assert view.allow_replace_abilities is True
            assert view.allow_opponent_snapping is False
            assert view.num_jokers == 0
            assert view.initial_view_count == 1
            assert view.cambia_allowed_round == 3

    def test_num_players_agrees_with_the_seat_count(self):
        with go_game(CARD_H2) as engine:
            assert engine.get_house_rules().num_players == engine.num_players()


# ---------------------------------------------------------------------------
# GameView protocol
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestGameViewProtocol:
    def test_go_engine_satisfies_the_protocol(self):
        with go_game(CARD_H2) as engine:
            assert isinstance(engine, GameView)

    def test_python_adapter_satisfies_the_protocol(self):
        assert isinstance(PythonGameView(python_game(CARD_H2)), GameView)

    def test_adapters_agree_on_the_dealt_state(self):
        rules = scripted_rules()
        py_view = PythonGameView(python_game(CARD_H2, rules))
        with go_game(CARD_H2, rules) as engine:
            assert py_view.num_players() == engine.num_players()
            assert py_view.acting_player() == engine.acting_player()
            assert py_view.turn_number() == engine.turn_number()
            assert py_view.is_terminal() == engine.is_terminal()
            assert py_view.stock_len() == engine.stock_len()
            assert py_view.discard_len() == engine.discard_len()
            for seat in range(2):
                assert idents(py_view.get_player_hand(seat)) == idents(
                    engine.get_player_hand(seat)
                )
            assert idents(py_view.get_discard_pile()) == idents(engine.get_discard_pile())
            assert ident(py_view.get_discard_top()) == ident(engine.get_discard_top())
            assert py_view.get_pending() == engine.get_pending()
            assert py_view.get_snap_state() == engine.get_snap_state()
            assert py_view.get_house_rules() == engine.get_house_rules()

    def test_adapters_agree_on_a_post_draw_pending(self):
        rules = scripted_rules()
        state = python_game(CARD_DK, rules)
        py_view = PythonGameView(state)
        state.apply_action(ActionDrawStockpile())
        with go_game(CARD_DK, rules) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)

            go_pending = engine.get_pending()
            py_pending = py_view.get_pending()
            assert py_pending.type == go_pending.type == PENDING_DISCARD
            assert py_pending.seat == go_pending.seat == 0
            assert (
                ident(py_pending.drawn_card) == ident(go_pending.drawn_card) == ("K", "D")
            )
            assert py_pending.drawn_from == go_pending.drawn_from == DRAWN_FROM_STOCKPILE

    def test_adapters_agree_on_an_open_snap_window(self):
        rules = scripted_rules()
        state = python_game(CARD_H2, rules)
        py_view = PythonGameView(state)
        state.apply_action(ActionDrawStockpile())
        state.apply_action(_python_discard_no_ability(state))
        with go_game(CARD_H2, rules) as engine:
            engine.apply_action(ACTION_DRAW_STOCKPILE)
            engine.apply_action(ACTION_DISCARD_NO_ABILITY)

            py_snap = py_view.get_snap_state()
            go_snap = engine.get_snap_state()
            assert py_snap.active is go_snap.active is True
            assert py_snap.rank == go_snap.rank == "2"
            assert ident(py_snap.card) == ident(go_snap.card) == ("2", "H")
            assert py_snap.snapper_count == go_snap.snapper_count
            assert py_snap.snapper_seat == go_snap.snapper_seat


def _python_discard_no_ability(state: CambiaGameState):
    """The ActionDiscard(use_ability=False) instance from the legal set."""
    from src.constants import ActionDiscard

    action = ActionDiscard(use_ability=False)
    assert action in state.get_legal_actions()
    return action
