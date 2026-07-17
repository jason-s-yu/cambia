"""
tests/test_snap_race.py

Tests for the snapRace house rule (cambia-564). Race-OFF (snapRace=False, the
default) is the sequential discarder-first model. Race-ON (snapRace=True) is the
true N-way race: every eligible snapper commits simultaneously (a commit mutates
no hand), then one uniform-random winner among the willing committers is drawn
and every losing willing committer draws the snap penalty; at most one snap
succeeds per discard. Mirrors the Go engine's snap_race.go.
"""

import random

import pytest
from types import SimpleNamespace

from src.card import Card, create_standard_deck
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.constants import ActionPassSnap, ActionSnapOwn, ActionSnapOpponent


def make_house_rules(
    snap_race: bool = False, allow_opponent_snapping: bool = False
) -> SimpleNamespace:
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
    house_rules = make_house_rules(
        snap_race=snap_race, allow_opponent_snapping=allow_opponent_snapping
    )

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
    state.snap_potential_snappers = (
        potential_snappers if potential_snappers is not None else [0, 1]
    )
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

    p0_hand = [filler]  # P0 has no match
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
    assert (
        state.snap_phase_active
    ), "Snap phase should remain active for P1 with snapRace=False"
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
# Behavioral: snapRace=True — simultaneous imperfect-info commit + N-way race
# ---------------------------------------------------------------------------


def test_snap_race_own_commit_defers_and_does_not_mutate():
    """With snapRace=True, a snap-own is a COMMIT: it advances to the next snapper
    without ending the phase or mutating any hand (imperfect info)."""
    snap_card = Card("5", "S")
    p0_match = Card("5", "H")
    p1_match = Card("5", "D")

    p0_hand = [p0_match, Card("7", "C")]
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    # P0 commits a snap-own; the phase must NOT resolve yet (P1 still to commit).
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    assert state.snap_phase_active, "commit must not end the phase before all commit"
    assert state.snap_current_snapper_idx == 1, "index should advance to P1"
    assert p0_match in state.players[0].hand, "a commit must not mutate the hand"
    assert len(state.players[0].hand) == 2 and len(state.players[1].hand) == 2


def test_snap_race_pass_does_not_end_phase_early():
    """With snapRace=True, a pass (no snap) still advances to next snapper as normal."""
    snap_card = Card("5", "S")
    p1_match = Card("5", "D")

    p0_hand = [Card("7", "C")]  # P0 has no match
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)

    # P0 passes — no success, so snap phase continues
    state.apply_action(ActionPassSnap())

    assert (
        state.snap_phase_active
    ), "Snap phase should remain active after pass with snapRace=True"
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
    assert (
        not state.snap_phase_active
    ), "Snap phase should end after P1 snaps with snapRace=True"


def test_snap_race_commit_defers_penalty_until_resolution():
    """With snapRace=True, a wrong-card snap commit applies NO penalty until the
    race resolves; the commit itself only advances to the next snapper."""
    snap_card = Card("5", "S")
    wrong_card = Card("7", "H")  # Wrong rank
    p1_match = Card("5", "D")

    p0_hand = [wrong_card]  # P0 commits a wrong card
    p1_hand = [p1_match, Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)
    initial_p0_hand_size = len(state.players[0].hand)

    # P0 commits a wrong-card snap: no resolution, no penalty yet.
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))

    assert state.snap_phase_active, "phase should remain active pending P1's commit"
    assert state.snap_current_snapper_idx == 1
    assert (
        len(state.players[0].hand) == initial_p0_hand_size
    ), "penalty must be deferred to resolution, not applied at commit time"


def test_snap_race_exactly_one_success_and_loser_penalized():
    """With two willing committers, exactly one snap succeeds (window closes on the
    single winner) and the losing willing committer draws the penalty."""
    snap_card = Card("5", "S")
    p0_hand = [Card("5", "H"), Card("7", "C")]
    p1_hand = [Card("5", "D"), Card("9", "C")]

    state = build_snap_state(p0_hand, p1_hand, snap_card, snap_race=True)
    state._rng = random.Random(0)

    state.apply_action(ActionSnapOwn(own_card_hand_index=0))  # P0 commit
    state.apply_action(ActionSnapOwn(own_card_hand_index=0))  # P1 commit -> resolve

    assert not state.snap_phase_active, "phase should end after race resolution"
    sizes = sorted(len(p.hand) for p in state.players)
    # Winner snapped away card 0 (2 -> 1); loser drew penaltyDrawCount=2 (2 -> 4).
    assert sizes == [1, 4], f"expected one winner (1) and one penalized loser (4), got {sizes}"


def test_snap_race_winner_not_fixed_priority():
    """The winner is a uniform chance draw, not the fixed discarder-first priority
    of race-OFF: across seeds, both committers win at least once."""
    p0_wins = p1_wins = 0
    for seed in range(200):
        snap_card = Card("5", "S")
        state = build_snap_state(
            [Card("5", "H"), Card("7", "C")],
            [Card("5", "D"), Card("9", "C")],
            snap_card,
            snap_race=True,
        )
        state._rng = random.Random(seed)
        state.apply_action(ActionSnapOwn(own_card_hand_index=0))
        state.apply_action(ActionSnapOwn(own_card_hand_index=0))
        if len(state.players[0].hand) == 1:
            p0_wins += 1
        elif len(state.players[1].hand) == 1:
            p1_wins += 1
        else:
            pytest.fail(f"seed {seed}: no clear winner")
    assert p0_wins > 0 and p1_wins > 0, f"winner is fixed: p0={p0_wins} p1={p1_wins}"


def test_snap_race_all_pass_no_penalty():
    """When every snapper passes, the window closes with no snap and no penalty."""
    snap_card = Card("5", "S")
    state = build_snap_state(
        [Card("7", "D"), Card("8", "C")],
        [Card("9", "C"), Card("T", "H")],
        snap_card,
        snap_race=True,
    )
    sizes_before = [len(p.hand) for p in state.players]
    state.apply_action(ActionPassSnap())  # P0 pass
    assert state.snap_phase_active
    state.apply_action(ActionPassSnap())  # P1 pass -> resolve
    assert not state.snap_phase_active, "phase should end after all-pass"
    assert [len(p.hand) for p in state.players] == sizes_before, "all-pass must not penalize"


def test_snap_race_single_snapper_success_ends_phase():
    """With snapRace=True and only one eligible snapper, success ends phase."""
    snap_card = Card("5", "S")
    p0_match = Card("5", "H")

    p0_hand = [p0_match, Card("7", "D")]
    p1_hand = [Card("9", "C")]

    state = build_snap_state(
        p0_hand,
        p1_hand,
        snap_card,
        snap_race=True,
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
        p0_hand,
        p1_hand,
        snap_card,
        snap_race=True,
        allow_opponent_snapping=True,
        potential_snappers=[0],  # Only P0 acts
    )

    # P0 snaps opponent's card at index 0
    state.apply_action(ActionSnapOpponent(opponent_target_hand_index=0))

    # Snap phase deactivated while pending move resolves
    assert not state.snap_phase_active
    # Pending move should be set
    assert (
        state.pending_action is not None
    ), "Pending move should be set for card transfer"
    # The opponent's matched card should be removed
    assert p1_match not in state.players[1].hand


# ---------------------------------------------------------------------------
# Contrast test: snapRace=False vs True produce different outcomes
# ---------------------------------------------------------------------------


def test_snap_race_true_vs_false_different_outcomes():
    """Full-window contrast when both snappers hold a matching card and both snap:
    - snapRace=False (sequential): BOTH snaps succeed -> both hands shrink by one.
    - snapRace=True (race): only ONE succeeds -> one hand shrinks, the loser is
      penalized instead.
    """
    # Race-OFF: P0 snaps, then P1 snaps; both remove their matching card.
    state_false = build_snap_state(
        [Card("5", "H"), Card("7", "C")],
        [Card("5", "D"), Card("9", "C")],
        Card("5", "S"),
        snap_race=False,
    )
    state_false.apply_action(ActionSnapOwn(own_card_hand_index=0))  # P0 succeeds
    assert state_false.snap_phase_active and state_false.snap_current_snapper_idx == 1
    state_false.apply_action(ActionSnapOwn(own_card_hand_index=0))  # P1 succeeds
    assert not state_false.snap_phase_active
    assert sorted(len(p.hand) for p in state_false.players) == [1, 1], (
        "snapRace=False: both snaps succeed"
    )

    # Race-ON: both commit; exactly one wins, the loser is penalized.
    state_true = build_snap_state(
        [Card("5", "H"), Card("7", "C")],
        [Card("5", "D"), Card("9", "C")],
        Card("5", "S"),
        snap_race=True,
    )
    state_true._rng = random.Random(0)
    state_true.apply_action(ActionSnapOwn(own_card_hand_index=0))  # commit
    assert state_true.snap_phase_active, "race-ON: first snap is a commit, not a resolve"
    state_true.apply_action(ActionSnapOwn(own_card_hand_index=0))  # commit -> resolve
    assert not state_true.snap_phase_active
    assert sorted(len(p.hand) for p in state_true.players) == [1, 4], (
        "snapRace=True: exactly one wins; the loser is penalized"
    )
