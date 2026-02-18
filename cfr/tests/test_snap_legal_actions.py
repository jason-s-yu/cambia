"""
tests/test_snap_legal_actions.py

Unit tests for snap-phase legal action generation.
Constructs known card configurations and verifies that the engine
produces the correct set of legal snap actions.
"""
import pytest
from types import SimpleNamespace

from src.card import Card, create_standard_deck
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.constants import ActionPassSnap, ActionSnapOwn, ActionSnapOpponent
from src.encoding import action_to_index


def make_house_rules(allow_opponent_snapping: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        allowDrawFromDiscardPile=True,
        allowReplaceAbilities=False,
        snapRace=False,
        penaltyDrawCount=2,
        allowOpponentSnapping=allow_opponent_snapping,
        use_jokers=2,
        cards_per_player=4,
        initial_view_count=2,
        cambia_allowed_round=0,
        max_game_turns=300,
    )


def build_snap_state(
    p0_hand: list,
    p1_hand: list,
    snap_card: Card,
    allow_opponent_snapping: bool = True,
    snap_current_snapper: int = 0,
) -> CambiaGameState:
    """
    Construct a CambiaGameState in snap phase with the given hands and discard card.
    snap_current_snapper: which player in snap_potential_snappers is currently acting (0=P0, 1=P1).
    """
    house_rules = make_house_rules(allow_opponent_snapping=allow_opponent_snapping)

    used_cards = set(id(c) for c in p0_hand + p1_hand + [snap_card])
    remaining = [
        c for c in create_standard_deck(include_jokers=2)
        if id(c) not in used_cards
    ]

    players = [
        PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
        PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
    ]

    state = CambiaGameState(
        players=players,
        stockpile=remaining[:20],
        discard_pile=[snap_card],
        current_player_index=0,
        house_rules=house_rules,
        cambia_caller_id=None,
    )

    # Manually activate snap phase
    state.snap_phase_active = True
    state.snap_discarded_card = snap_card
    state.snap_potential_snappers = [0, 1]
    state.snap_current_snapper_idx = snap_current_snapper
    state.snap_results_log = []

    return state


# ---------------------------------------------------------------------------
# Test 1: Player has matching cards — snap own and snap opponent should be legal
# ---------------------------------------------------------------------------

def test_player_has_matching_own_cards():
    """P0 has two 5s at indices 0 and 2; P1 has a 5 at index 1.
    With allowOpponentSnapping=True, P0 should be able to snap own(0), own(2),
    and opponent(1)."""
    p0_hand = [
        Card(rank="5", suit="H"),
        Card(rank="7", suit="D"),
        Card(rank="5", suit="C"),
        Card(rank="K", suit="S"),
    ]
    p1_hand = [
        Card(rank="3", suit="H"),
        Card(rank="5", suit="S"),
        Card(rank="Q", suit="D"),
        Card(rank="9", suit="C"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=True)
    legal = state.get_legal_actions()

    assert ActionPassSnap() in legal
    assert ActionSnapOwn(own_card_hand_index=0) in legal
    assert ActionSnapOwn(own_card_hand_index=2) in legal
    assert ActionSnapOpponent(opponent_target_hand_index=1) in legal

    # Non-matching own cards should not appear
    assert ActionSnapOwn(own_card_hand_index=1) not in legal
    assert ActionSnapOwn(own_card_hand_index=3) not in legal

    # Non-matching opponent cards should not appear
    assert ActionSnapOpponent(opponent_target_hand_index=0) not in legal
    assert ActionSnapOpponent(opponent_target_hand_index=2) not in legal
    assert ActionSnapOpponent(opponent_target_hand_index=3) not in legal


# ---------------------------------------------------------------------------
# Test 2: Player has no matching cards — only pass snap should be legal
# ---------------------------------------------------------------------------

def test_player_has_no_matching_cards():
    """P0 has no 5s; snap discard is 5D. Only PassSnap should be legal."""
    p0_hand = [
        Card(rank="7", suit="D"),
        Card(rank="8", suit="H"),
        Card(rank="Q", suit="C"),
        Card(rank="3", suit="S"),
    ]
    p1_hand = [
        Card(rank="2", suit="H"),
        Card(rank="6", suit="S"),
        Card(rank="J", suit="D"),
        Card(rank="A", suit="C"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=True)
    legal = state.get_legal_actions()

    assert legal == {ActionPassSnap()}


# ---------------------------------------------------------------------------
# Test 3: Multiple matching cards in own hand
# ---------------------------------------------------------------------------

def test_multiple_matching_own_cards():
    """P0 has four 5s; snap discard is 5H. All own indices should be snap-legal."""
    p0_hand = [
        Card(rank="5", suit="H"),
        Card(rank="5", suit="D"),
        Card(rank="5", suit="C"),
        Card(rank="5", suit="S"),
    ]
    p1_hand = [
        Card(rank="2", suit="H"),
        Card(rank="6", suit="S"),
        Card(rank="J", suit="D"),
        Card(rank="A", suit="C"),
    ]
    # Use a different 5 as the snap trigger (same rank, different object)
    snap_card = Card(rank="5", suit="D")

    # Build state manually (snap card is a separate object with same rank)
    house_rules = make_house_rules(allow_opponent_snapping=False)
    players = [
        PlayerState(hand=list(p0_hand), initial_peek_indices=(0, 1)),
        PlayerState(hand=list(p1_hand), initial_peek_indices=(0, 1)),
    ]
    remaining = [
        Card(rank=r, suit=s)
        for r in ["7", "8", "9", "T", "J", "Q", "K", "A", "2", "3", "4"]
        for s in ["H", "D", "C", "S"]
    ][:20]

    state = CambiaGameState(
        players=players,
        stockpile=remaining,
        discard_pile=[snap_card],
        current_player_index=0,
        house_rules=house_rules,
        cambia_caller_id=None,
    )
    state.snap_phase_active = True
    state.snap_discarded_card = snap_card
    state.snap_potential_snappers = [0, 1]
    state.snap_current_snapper_idx = 0
    state.snap_results_log = []

    legal = state.get_legal_actions()

    assert ActionPassSnap() in legal
    assert ActionSnapOwn(own_card_hand_index=0) in legal
    assert ActionSnapOwn(own_card_hand_index=1) in legal
    assert ActionSnapOwn(own_card_hand_index=2) in legal
    assert ActionSnapOwn(own_card_hand_index=3) in legal

    # No opponent snap since allowOpponentSnapping=False
    assert not any(isinstance(a, ActionSnapOpponent) for a in legal)


# ---------------------------------------------------------------------------
# Test 4: Opponent snapping disabled
# ---------------------------------------------------------------------------

def test_opponent_snapping_disabled():
    """Same hands as Test 1 but allowOpponentSnapping=False. No SnapOpponent actions."""
    p0_hand = [
        Card(rank="5", suit="H"),
        Card(rank="7", suit="D"),
        Card(rank="5", suit="C"),
        Card(rank="K", suit="S"),
    ]
    p1_hand = [
        Card(rank="3", suit="H"),
        Card(rank="5", suit="S"),
        Card(rank="Q", suit="D"),
        Card(rank="9", suit="C"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=False)
    legal = state.get_legal_actions()

    assert ActionPassSnap() in legal
    assert ActionSnapOwn(own_card_hand_index=0) in legal
    assert ActionSnapOwn(own_card_hand_index=2) in legal

    # No SnapOpponent actions at all
    assert not any(isinstance(a, ActionSnapOpponent) for a in legal)


# ---------------------------------------------------------------------------
# Test 5: Opponent has matching cards, player can snap opponent
# ---------------------------------------------------------------------------

def test_snap_opponent_cards_no_own_match():
    """P0 has no own matches but P1 has two 5s. With allowOpponentSnapping=True,
    P0 should be able to snap opponent at indices 0 and 1."""
    p0_hand = [
        Card(rank="7", suit="D"),
        Card(rank="8", suit="H"),
        Card(rank="Q", suit="C"),
        Card(rank="3", suit="S"),
    ]
    p1_hand = [
        Card(rank="5", suit="H"),
        Card(rank="5", suit="C"),
        Card(rank="9", suit="D"),
        Card(rank="K", suit="S"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=True)
    legal = state.get_legal_actions()

    assert ActionPassSnap() in legal
    assert ActionSnapOpponent(opponent_target_hand_index=0) in legal
    assert ActionSnapOpponent(opponent_target_hand_index=1) in legal

    # No own snaps (P0 has no 5s)
    assert not any(isinstance(a, ActionSnapOwn) for a in legal)

    # Non-matching opponent positions not legal
    assert ActionSnapOpponent(opponent_target_hand_index=2) not in legal
    assert ActionSnapOpponent(opponent_target_hand_index=3) not in legal


# ---------------------------------------------------------------------------
# Test 6: Action indices map correctly
# ---------------------------------------------------------------------------

def test_action_index_mapping_pass_snap():
    """PassSnap should map to index 97."""
    assert action_to_index(ActionPassSnap()) == 97


def test_action_index_mapping_snap_own():
    """SnapOwn(i) should map to index 98+i."""
    for i in range(6):
        assert action_to_index(ActionSnapOwn(own_card_hand_index=i)) == 98 + i


def test_action_index_mapping_snap_opponent():
    """SnapOpponent(i) should map to index 104+i."""
    for i in range(6):
        assert action_to_index(ActionSnapOpponent(opponent_target_hand_index=i)) == 104 + i


def test_action_indices_in_legal_set_test1():
    """Verify action indices from Test 1 scenario match expected values."""
    p0_hand = [
        Card(rank="5", suit="H"),
        Card(rank="7", suit="D"),
        Card(rank="5", suit="C"),
        Card(rank="K", suit="S"),
    ]
    p1_hand = [
        Card(rank="3", suit="H"),
        Card(rank="5", suit="S"),
        Card(rank="Q", suit="D"),
        Card(rank="9", suit="C"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=True)
    legal = state.get_legal_actions()
    legal_indices = {action_to_index(a) for a in legal}

    assert 97 in legal_indices   # PassSnap
    assert 98 in legal_indices   # SnapOwn(0)
    assert 100 in legal_indices  # SnapOwn(2)
    assert 105 in legal_indices  # SnapOpponent(1)

    # Indices that should NOT be present
    assert 99 not in legal_indices   # SnapOwn(1)
    assert 101 not in legal_indices  # SnapOwn(3)
    assert 104 not in legal_indices  # SnapOpponent(0)
    assert 106 not in legal_indices  # SnapOpponent(2)
    assert 107 not in legal_indices  # SnapOpponent(3)


def test_action_indices_in_legal_set_test4():
    """Verify Test 4 (no opponent snapping) action indices."""
    p0_hand = [
        Card(rank="5", suit="H"),
        Card(rank="7", suit="D"),
        Card(rank="5", suit="C"),
        Card(rank="K", suit="S"),
    ]
    p1_hand = [
        Card(rank="3", suit="H"),
        Card(rank="5", suit="S"),
        Card(rank="Q", suit="D"),
        Card(rank="9", suit="C"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=False)
    legal = state.get_legal_actions()
    legal_indices = {action_to_index(a) for a in legal}

    assert 97 in legal_indices   # PassSnap
    assert 98 in legal_indices   # SnapOwn(0)
    assert 100 in legal_indices  # SnapOwn(2)

    # No SnapOpponent indices (104-109)
    for idx in range(104, 110):
        assert idx not in legal_indices


def test_action_indices_in_legal_set_test5():
    """Verify Test 5 (opponent cards match, no own match) action indices."""
    p0_hand = [
        Card(rank="7", suit="D"),
        Card(rank="8", suit="H"),
        Card(rank="Q", suit="C"),
        Card(rank="3", suit="S"),
    ]
    p1_hand = [
        Card(rank="5", suit="H"),
        Card(rank="5", suit="C"),
        Card(rank="9", suit="D"),
        Card(rank="K", suit="S"),
    ]
    snap_card = Card(rank="5", suit="D")

    state = build_snap_state(p0_hand, p1_hand, snap_card, allow_opponent_snapping=True)
    legal = state.get_legal_actions()
    legal_indices = {action_to_index(a) for a in legal}

    assert 97 in legal_indices    # PassSnap
    assert 104 in legal_indices   # SnapOpponent(0)
    assert 105 in legal_indices   # SnapOpponent(1)

    # No SnapOwn actions (98-103)
    for idx in range(98, 104):
        assert idx not in legal_indices

    # Non-matching opponent positions
    assert 106 not in legal_indices  # SnapOpponent(2)
    assert 107 not in legal_indices  # SnapOpponent(3)
