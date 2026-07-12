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
        c for c in create_standard_deck(include_jokers=2) if id(c) not in used_cards
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

    Per RULES.md Sec.5, a snap targets ANY known card -- a rank mismatch is a
    legal-but-penalized attempt, not an illegal action (S1W11: matches the Go
    engine's legalSnapDecision, which offers every hand slot). So with
    allowOpponentSnapping=True, ALL of P0's own indices and ALL of P1's
    opponent indices are legal targets, not just the rank-matching ones.
    """
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
    for i in range(4):
        assert ActionSnapOwn(own_card_hand_index=i) in legal
        assert ActionSnapOpponent(opponent_target_hand_index=i) in legal


# ---------------------------------------------------------------------------
# Test 2: Player has no matching cards — only pass snap should be legal
# ---------------------------------------------------------------------------


def test_player_has_no_matching_cards():
    """P0 has no 5s and P1 has no 5s; snap discard is 5D.

    Per RULES.md Sec.5 (S1W11), a snap is legal on ANY known card -- absence
    of a rank match doesn't shrink the legal set, it just means every
    SnapOwn/SnapOpponent attempt will be a (legal, penalized) miss. PassSnap
    plus every own/opponent hand slot is legal.
    """
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

    expected = {ActionPassSnap()}
    expected.update(ActionSnapOwn(own_card_hand_index=i) for i in range(4))
    expected.update(ActionSnapOpponent(opponent_target_hand_index=i) for i in range(4))
    # get_legal_actions() returns a canonically-ordered list (not a set); compare
    # as a set here since this test only cares about membership, not order.
    assert set(legal) == expected


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
    """P0 has no own matches but P1 has two 5s.

    Per RULES.md Sec.5 (S1W11), P0 can still attempt SnapOwn on any of their
    own 4 slots (a legal, penalized miss, since none match) in addition to
    SnapOpponent on any of P1's 4 slots.
    """
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
    for i in range(4):
        assert ActionSnapOwn(own_card_hand_index=i) in legal
        assert ActionSnapOpponent(opponent_target_hand_index=i) in legal


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
        assert (
            action_to_index(ActionSnapOpponent(opponent_target_hand_index=i)) == 104 + i
        )


def test_action_indices_in_legal_set_test1():
    """Verify action indices from Test 1 scenario match expected values.

    Per RULES.md Sec.5 (S1W11), every own/opponent slot is a legal snap
    target (rank mismatch is a penalized miss, not illegal), so indices
    98-103 (SnapOwn 0-5, only 0-3 exist for a 4-card hand) and 104-107
    (SnapOpponent 0-3) are ALL present.
    """
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

    assert 97 in legal_indices  # PassSnap
    for idx in list(range(98, 102)) + list(
        range(104, 108)
    ):  # SnapOwn(0-3) + SnapOpponent(0-3)
        assert idx in legal_indices, f"expected idx {idx} legal"


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

    assert 97 in legal_indices  # PassSnap
    assert 98 in legal_indices  # SnapOwn(0)
    assert 100 in legal_indices  # SnapOwn(2)

    # No SnapOpponent indices (104-109)
    for idx in range(104, 110):
        assert idx not in legal_indices


def test_action_indices_in_legal_set_test5():
    """Verify Test 5 (opponent cards match, no own match) action indices.

    Per RULES.md Sec.5 (S1W11), P0's own slots (98-101) are legal too (a
    penalized miss, since P0 has no matching rank), in addition to all of
    P1's opponent slots (104-107).
    """
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

    assert 97 in legal_indices  # PassSnap
    for idx in list(range(98, 102)) + list(
        range(104, 108)
    ):  # SnapOwn(0-3) + SnapOpponent(0-3)
        assert idx in legal_indices, f"expected idx {idx} legal"
