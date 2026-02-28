"""
tests/test_snap_race.py

Tests for SnapRace house rule: when snapRace=True, the first successful snap
ends the snap phase immediately (remaining eligible snappers get no turn).
"""
import pytest
from types import SimpleNamespace

from src.card import Card, create_standard_deck
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.constants import ActionPassSnap, ActionSnapOwn, ActionSnapOpponent


def make_house_rules(snap_race: bool = False, allow_opponent_snapping: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        allowDrawFromDiscardPile=True,
        allowReplaceAbilities=False,
        snapRace=snap_race,
        penaltyDrawCount=2,
        allowOpponentSnapping=allow_opponent_snapping,
        use_jokers=2,
        cards_per_player=4,
        initial_view_count=2,
        cambia_allowed_round=0,
        max_game_turns=300,
        lockCallerHand=True,
    )


def build_snap_state(
    p0_hand: list,
    p1_hand: list,
    snap_card: Card,
    snap_race: bool = False,
    allow_opponent_snapping: bool = False,
    current_snapper_idx: int = 0,
    potential_snappers: list = None,
) -> CambiaGameState:
    """Build a CambiaGameState already in snap phase with the given setup."""
    house_rules = make_house_rules(snap_race=snap_race, allow_opponent_snapping=allow_opponent_snapping)

    deck = create_standard_deck(include_jokers=2)
    used_ids = set(id(c) for c in p0_hand + p1_hand + [snap_card])
    remaining = [c for c in deck if id(c) not in used_ids]

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

    state.snap_phase_active = True
    state.snap_discarded_card = snap_card
    state.snap_potential_snappers = potential_snappers if potential_snappers is not None else [0, 1]
    state.snap_current_snapper_idx = current_snapper_idx
    state.snap_results_log = []

    return state


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

def test_snap_race_default_false():
    """CambiaRulesConfig-like SimpleNamespace has snapRace=False by default."""
    rules = make_house_rules()
    assert rules.snapRace is False


def test_snap_race_true_accessible():
    """snapRace=True is accessible on house_rules."""
    rules = make_house_rules(snap_race=True)
    assert rules.snapRace is True


# ---------------------------------------------------------------------------
# Regression: default (snapRace=False) — sequential snapper behavior preserved
# ---------------------------------------------------------------------------

def test_no_snap_race_first_snapper_passes_second_gets_turn():
    """With snapRace=False, P0 passing snap gives P1 their turn."""
    snap_card = Card("5", "S")
    filler = Card("7", "D")
    matching = Card("5", "H")

    p0_hand = [filler]         # P0 has no match
    p1_hand = [matching, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=False)

    # P0 is at index 0 in snap_potential_snappers — they pass
    state.apply_action(ActionPassSnap())

    assert state.snap_phase_active, "Snap phase should still be active after P0 pass"
    assert state.snap_current_snapper_idx == 1, "Index should advance to P1"


def test_no_snap_race_first_snapper_succeeds_second_still_gets_turn():
    """With snapRace=False, P0 successfully snapping does NOT end phase — P1 still acts."""
    snap_card = Card("5", "S")
    p0_match = Card("5", "H")
    p1_match = Card("5", "D")

    p0_hand = [p0_match, Card("7", "C")]
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=False)

    # P0 successfully snaps own card (index 0)
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    # Snap phase should still be active (P1 has not had their turn)
    assert state.snap_phase_active, "Snap phase should remain active for P1 with snapRace=False"
    assert state.snap_current_snapper_idx == 1, "Index should advance to P1"


def test_no_snap_race_both_snappers_exhaust_ends_phase():
    """With snapRace=False, after both P0 and P1 pass, snap phase ends."""
    snap_card = Card("5", "S")

    p0_hand = [Card("7", "D")]
    p1_hand = [Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=False)

    # P0 passes
    state.apply_action(ActionPassSnap())
    assert state.snap_phase_active

    # P1 passes — all snappers exhausted, phase ends
    state.apply_action(ActionPassSnap())
    assert not state.snap_phase_active, "Snap phase should end after all snappers acted"


# ---------------------------------------------------------------------------
# Behavioral: snapRace=True — first successful snap ends the phase immediately
# ---------------------------------------------------------------------------

def test_snap_race_own_success_ends_phase_immediately():
    """With snapRace=True, P0 snapping own card ends snap phase; P1 never gets turn."""
    snap_card = Card("5", "S")
    p0_match = Card("5", "H")
    p1_match = Card("5", "D")

    p0_hand = [p0_match, Card("7", "C")]
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    # P0 successfully snaps own card
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    assert not state.snap_phase_active, "Snap phase should end immediately after snap with snapRace=True"
    # The matched card should have been removed from P0's hand
    assert p0_match not in state.players[0].hand, "Snapped card should leave P0's hand"


def test_snap_race_pass_does_not_end_phase_early():
    """With snapRace=True, a pass (no snap) still advances to next snapper as normal."""
    snap_card = Card("5", "S")
    p1_match = Card("5", "D")

    p0_hand = [Card("7", "C")]  # P0 has no match
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    # P0 passes — no success, so snap phase continues
    state.apply_action(ActionPassSnap())

    assert state.snap_phase_active, "Snap phase should remain active after pass with snapRace=True"
    assert state.snap_current_snapper_idx == 1, "Should advance to P1"


def test_snap_race_second_snapper_success_also_ends_phase():
    """With snapRace=True, P1 (second snapper) snapping own card ends phase immediately."""
    snap_card = Card("5", "S")
    p1_match = Card("5", "D")

    p0_hand = [Card("7", "C")]  # P0 has no match
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    # P0 passes (no match)
    state.apply_action(ActionPassSnap())
    assert state.snap_phase_active

    # P1 snaps own card — phase should end
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))
    assert not state.snap_phase_active, "Snap phase should end after P1 snaps with snapRace=True"


def test_snap_race_failed_snap_does_not_end_phase():
    """With snapRace=True, a failed (penalty) snap does NOT end phase early."""
    snap_card = Card("5", "S")
    wrong_card = Card("7", "H")  # Wrong rank
    p1_match = Card("5", "D")

    p0_hand = [wrong_card]  # P0 will attempt wrong card
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    initial_p0_hand_size = len(state.players[0].hand)

    # P0 attempts snap with wrong card (penalty)
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    # Snap phase should still be active for P1 (snap_success was False)
    assert state.snap_phase_active, "Snap phase should remain active after failed snap"
    # P0 should have received penalty cards
    assert len(state.players[0].hand) > initial_p0_hand_size, "P0 should have penalty cards"


def test_snap_race_single_snapper_success_ends_phase():
    """With snapRace=True and only one eligible snapper, success ends phase."""
    snap_card = Card("5", "S")
    p0_match = Card("5", "H")

    p0_hand = [p0_match, Card("7", "D")]
    p1_hand = [Card("9", "C")]

    state = build_snap_state(
        p0_hand, p1_hand, snap_card, snap_race=True,
        potential_snappers=[0],  # Only P0 eligible
    )

    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    assert not state.snap_phase_active


# ---------------------------------------------------------------------------
# ActionSnapOpponent with snapRace=True
# ---------------------------------------------------------------------------

def test_snap_race_opponent_snap_deactivates_snap_phase():
    """With snapRace=True and opponent snapping, snap phase is deactivated for pending move."""
    snap_card = Card("5", "S")
    p1_match = Card("5", "D")
    filler = Card("7", "C")

    # P0 snaps opponent — P0 needs a card to move (filler), P1 has matching card
    p0_hand = [filler]
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(
        p0_hand, p1_hand, snap_card, snap_race=True, allow_opponent_snapping=True,
        potential_snappers=[0],  # Only P0 acts
    )

    # P0 snaps opponent's card at index 0
    state.apply_action(ActionSnapOpponent(opponent_target_hand_index=0))

    # Snap phase deactivated while pending move resolves
    assert not state.snap_phase_active
    # Pending move should be set
    assert state.pending_action is not None, "Pending move should be set for card transfer"
    # The opponent's matched card should be removed
    assert p1_match not in state.players[1].hand


# ---------------------------------------------------------------------------
# Contrast test: snapRace=False vs True produce different outcomes
# ---------------------------------------------------------------------------

def test_snap_race_true_vs_false_different_outcomes():
    """
    With 2 eligible snappers and P0 snapping successfully:
    - snapRace=False: P1 still gets their turn (snap_phase_active=True, idx=1)
    - snapRace=True: phase ends immediately (snap_phase_active=False)
    """
    # Build two independent states
    snap_card_false = Card("5", "S")
    p0_match_false = Card("5", "H")
    p1_match_false = Card("5", "D")

    state_false = build_snap_state(
        [p0_match_false, Card("7", "C")],
        [p1_match_false, Card("9", "C")],
        snap_card_false,
        snap_race=False,
    )

    snap_card_true = Card("5", "S")
    p0_match_true = Card("5", "H")
    p1_match_true = Card("5", "D")

    state_true = build_snap_state(
        [p0_match_true, Card("7", "C")],
        [p1_match_true, Card("9", "C")],
        snap_card_true,
        snap_race=True,
    )

    # P0 snaps own card in both
    state_false.apply_action(ActionSnapOwn(own_card_hand_index=0))
    state_true.apply_action(ActionSnapOwn(own_card_hand_index=0))

    # snapRace=False: P1 should still have their turn
    assert state_false.snap_phase_active, "snapRace=False: P1 should still get their turn"
    assert state_false.snap_current_snapper_idx == 1

    # snapRace=True: phase ended, P1 never acts
    assert not state_true.snap_phase_active, "snapRace=True: phase ended after first snap"
