"""
tests/test_br_state_cross_engine.py

Cross-engine equivalence for the Go-backed best-response substrate (cambia-1428).

``src.cfr.br_state.GoBrState`` replaced the Python ``CambiaGameState`` under the
tabular exploitability search while keeping the Python ``AgentState`` belief
layer above it. That only holds if the substrate hands the belief layer exactly
what the Python engine did, so this walks both engines over the same pinned deals
and compares, at every step: the repr-sorted legal-action list (which is what
indexes a stored strategy vector), the decision context, and the whole
``AgentObservation`` frame, ``snap_results`` included.

The snap fields are the reason this file exists. The Go engine exports no
snap-results log, so ``GoBrState`` reconstructs one, and a reconstruction is only
worth having if it is checked against the thing it reconstructs --
``test_walk_exercises_the_snap_reconstruction`` fails if the walk stops reaching
successful snaps, penalty snaps and the SnapOpponent follow-up move, so the
comparison cannot quietly degrade into checking nothing.

cambia-1782 put the tabular training traversal on the same substrate, and it
needs the fuller frame the production builder produced: the peeked cards and the
King swap's slot pair. ``observation(ability_reveals=True)`` is that frame, and
the walk below compares it against ``src.cfr.worker._create_observation`` off the
Python engine, with a guard that the walk keeps reaching peeks and King swaps.
"""

import random
from typing import List, Tuple

import pytest

from src.agents.action_codec import index_to_action
from src.config import CambiaRulesConfig
from src.constants import (
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionSnapOpponentMove,
    GameAction,
)


def _go_available() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.ffi.bridge import GoEngine

            e = GoEngine(seed=0)
            e.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


# Opponent snapping ON so the reconstructed snap-results log is exercised,
# including the SnapOpponent path where the accumulated block stays visible
# across the follow-up move. The full standard deck is in play, so the ability
# ranks are live too. snapRace stays off: GoBrState refuses it, and a test below
# pins that refusal.
_RULES = CambiaRulesConfig(
    allowDrawFromDiscardPile=False,
    allowReplaceAbilities=False,
    allowOpponentSnapping=True,
    snapRace=False,
    penaltyDrawCount=1,
    use_jokers=0,
    cards_per_player=4,
    initial_view_count=2,
    cambia_allowed_round=1,
    max_game_turns=40,
)

_SEEDS = list(range(40))
_MAX_STEPS = 400


def _py_legal_sorted(pygame) -> List[GameAction]:
    return sorted(list(pygame.get_legal_actions()), key=repr)


def _obs_fields(obs) -> Tuple:
    """An observation flattened to comparable primitives, every field included."""
    return (
        obs.acting_player,
        obs.action,
        str(obs.discard_top_card),
        tuple(obs.player_hand_sizes),
        obs.stockpile_size,
        str(obs.drawn_card),
        (
            None
            if obs.peeked_cards is None
            else tuple(sorted((k, str(v)) for k, v in obs.peeked_cards.items()))
        ),
        tuple(
            tuple(sorted((k, repr(v)) for k, v in entry.items()))
            for entry in obs.snap_results
        ),
        obs.did_cambia_get_called,
        obs.who_called_cambia,
        obs.is_game_over,
        obs.current_turn,
        obs.king_swap_indices,
        obs.is_race_commit,
        obs.race_resolution,
    )


def _py_pre_apply_king_swap(pygame, action):
    """The King swap slots ``src.cfr.worker`` captured before applying."""
    from src.constants import ActionAbilityKingSwapDecision

    if not isinstance(action, ActionAbilityKingSwapDecision) or not action.perform_swap:
        return None
    pad = pygame.pending_action_data
    if pad and "own_idx" in pad and "opp_idx" in pad:
        return (pad["own_idx"], pad["opp_idx"])
    return None


def _new_pair(seed: int):
    """A Python game and a GoBrState dealt from that game's exact deck."""
    from src.cfr.br_state import GoBrState, deal_spec_from_python_game
    from src.game.engine import CambiaGameState

    pygame = CambiaGameState(house_rules=_RULES, _rng=random.Random(seed))
    deal = deal_spec_from_python_game(pygame)
    return pygame, GoBrState.new(_RULES, deal)


@skip_if_no_go
@pytest.mark.parametrize("seed", _SEEDS)
def test_observation_stream_matches_python_engine(seed: int):
    """Every frame the belief layer sees is identical on both engines."""
    from src.analysis_tools import AnalysisTools

    pygame, state = _new_pair(seed)
    try:
        # The deal itself has to match before any step comparison means anything.
        for p in range(2):
            assert [str(c) for c in state.hand(p)] == [
                str(c) for c in pygame.players[p].hand
            ], f"seat {p} initial hand differs"

        assert _obs_fields(state.initial_observation()) == _obs_fields(
            AnalysisTools._create_observation_for_br(pygame, None, -1)
        ), "initial observation differs"

        rng = random.Random(0xBEEF ^ seed)
        for step in range(_MAX_STEPS):
            if state.is_terminal() or pygame.is_terminal():
                assert (
                    state.is_terminal() == pygame.is_terminal()
                ), f"step {step}: terminal disagreement"
                break

            go_pairs = state.legal_actions()
            go_actions = [a for _, a in go_pairs]
            assert go_actions == _py_legal_sorted(
                pygame
            ), f"step {step}: legal action order differs"
            assert state.decision_context() == AnalysisTools._get_decision_context(
                pygame
            ), f"step {step}: decision context differs"

            actor = state.acting_player()
            assert actor == pygame.get_acting_player(), f"step {step}: actor differs"

            action_idx, action = go_pairs[rng.randrange(len(go_pairs))]

            state.apply(action_idx)
            pygame.apply_action(action)

            assert _obs_fields(state.observation(action, actor)) == _obs_fields(
                AnalysisTools._create_observation_for_br(pygame, action, actor)
            ), f"step {step}: observation after {action} differs"

        assert state.is_terminal() == pygame.is_terminal()
        if state.is_terminal():
            for p in range(2):
                assert state.utility(p) == pytest.approx(pygame.get_utility(p))
    finally:
        state.close()


@skip_if_no_go
def test_walk_exercises_the_snap_reconstruction():
    """The comparison walk really does reach the reconstructed snap outcomes.

    Without this the observation comparison above could pass while never once
    producing a non-empty snap_results, which is exactly the field the Go engine
    does not export and this module invents.
    """
    successes = penalties = moves = frames = 0
    for seed in _SEEDS:
        _, state = _new_pair(seed)
        try:
            rng = random.Random(0xBEEF ^ seed)
            for _ in range(_MAX_STEPS):
                if state.is_terminal():
                    break
                pairs = state.legal_actions()
                actor = state.acting_player()
                action_idx, action = pairs[rng.randrange(len(pairs))]
                if isinstance(action, ActionSnapOpponentMove):
                    moves += 1
                state.apply(action_idx)
                results = state.observation(action, actor).snap_results
                if results:
                    frames += 1
                successes += sum(1 for e in results if e["success"])
                penalties += sum(1 for e in results if e["penalty"])
        finally:
            state.close()

    assert frames > 0, "no observation carried snap results"
    assert successes > 0, "no successful snap was reconstructed"
    assert penalties > 0, "no penalty snap was reconstructed"
    assert moves > 0, "the SnapOpponent follow-up move was never reached"


@skip_if_no_go
def test_checkpoint_rewind_round_trips():
    """Rewinding restores the game, the action prefix and the snap-log mirror."""
    _, state = _new_pair(23)
    try:
        rng = random.Random(99)
        for _ in range(6):
            if state.is_terminal():
                break
            pairs = state.legal_actions()
            state.apply(pairs[rng.randrange(len(pairs))][0])

        if state.is_terminal():
            pytest.skip("walked into a terminal state before branching")

        cp = state.checkpoint()
        try:
            before_prefix = state.action_prefix
            actor = state.acting_player()
            before = _obs_fields(state.observation(None, actor))

            for action_idx, action in state.legal_actions():
                state.apply(action_idx)
                state.rewind(cp)
                assert (
                    state.action_prefix == before_prefix
                ), f"prefix not rewound after {action}"
                assert (
                    _obs_fields(state.observation(None, actor)) == before
                ), f"state not rewound after {action}"
        finally:
            state.release(cp)
    finally:
        state.close()


@skip_if_no_go
def test_replayed_prefix_reconstructs_the_same_node():
    """A pool worker's (deal, prefix) replay lands on the parent's exact state."""
    from src.cfr.br_state import GoBrState

    _, state = _new_pair(101)
    try:
        rng = random.Random(5)
        for _ in range(8):
            if state.is_terminal():
                break
            pairs = state.legal_actions()
            state.apply(pairs[rng.randrange(len(pairs))][0])

        replayed = GoBrState.new(_RULES, state.deal, state.action_prefix)
        try:
            actor = state.acting_player()
            assert _obs_fields(replayed.observation(None, actor)) == _obs_fields(
                state.observation(None, actor)
            )
            assert replayed.legal_actions() == state.legal_actions()
            assert replayed.decision_context() == state.decision_context()
        finally:
            replayed.close()
    finally:
        state.close()


@skip_if_no_go
def test_snap_race_rules_are_refused():
    """snapRace has a different resolution path, so it is refused, not guessed."""
    from src.cfr.br_state import DealSpec, GoBrState

    race_rules = CambiaRulesConfig(**{**_RULES.model_dump(), "snapRace": True})
    with pytest.raises(ValueError, match="snapRace"):
        GoBrState.new(race_rules, DealSpec(seed=1))


@skip_if_no_go
def test_legal_action_indices_decode_to_their_actions():
    """The (index, action) pairing the search applies is the codec's own."""
    _, state = _new_pair(7)
    try:
        for action_idx, action in state.legal_actions():
            assert index_to_action(action_idx) == action
    finally:
        state.close()


# The ability ranks are the rare ones: a uniform walk over 40 deals reached a
# single King look and never a performed swap, so the frame comparison below
# would have been checking nothing on exactly the two fields cambia-1782 adds.
# This walk takes an ability action when one is legal 85% of the time, which
# reaches all three reveals; _ABILITY_WALK_PREFERRED names what it favours.
_ABILITY_WALK_PREFERRED = (
    ActionAbilityKingLookSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
)


def _pick_ability_first(rng, pairs):
    """One legal (index, action) pair, biased towards the ability paths."""
    preferred = [
        pair
        for pair in pairs
        if isinstance(pair[1], _ABILITY_WALK_PREFERRED)
        or (isinstance(pair[1], ActionAbilityKingSwapDecision) and pair[1].perform_swap)
    ]
    pool = preferred if preferred and rng.random() < 0.85 else pairs
    return pool[rng.randrange(len(pool))]


@skip_if_no_go
@pytest.mark.parametrize("seed", _SEEDS)
def test_production_frame_matches_python_engine(seed: int):
    """The fuller frame the tabular traversal consumes is the production frame.

    ``observation(ability_reveals=True)`` has to equal what
    ``src.cfr.worker._create_observation`` built off the Python engine, or the
    ported traversal feeds the belief layer a different stream than the tables
    were written with.
    """
    from src.cfr.worker import _create_observation

    pygame, state = _new_pair(seed)
    try:
        assert _obs_fields(
            state.initial_observation(ability_reveals=True)
        ) == _obs_fields(
            _create_observation(None, None, pygame, -1, [])
        ), "initial observation differs"

        rng = random.Random(0xF00D ^ seed)
        for step in range(_MAX_STEPS):
            if state.is_terminal() or pygame.is_terminal():
                break
            actor = state.acting_player()
            action_idx, action = _pick_ability_first(rng, state.legal_actions())
            king_swap = _py_pre_apply_king_swap(pygame, action)

            state.apply(action_idx)
            pygame.apply_action(action)

            assert _obs_fields(
                state.observation(action, actor, ability_reveals=True)
            ) == _obs_fields(
                _create_observation(
                    None,
                    action,
                    pygame,
                    actor,
                    pygame.snap_results_log,
                    king_swap_indices=king_swap,
                )
            ), f"step {step}: production frame after {action} differs"
    finally:
        state.close()


@skip_if_no_go
def test_production_frame_walk_reaches_the_ability_reveals():
    """The frame comparison above really does reach a peek and a King swap.

    Both fields are new in cambia-1782 and both are rare, so without this the
    parametrized walk could pass while never producing either.
    """
    peeks = king_swaps = king_looks = 0
    for seed in _SEEDS:
        _, state = _new_pair(seed)
        try:
            rng = random.Random(0xF00D ^ seed)
            for _ in range(_MAX_STEPS):
                if state.is_terminal():
                    break
                actor = state.acting_player()
                action_idx, action = _pick_ability_first(rng, state.legal_actions())
                state.apply(action_idx)
                obs = state.observation(action, actor, ability_reveals=True)
                if isinstance(
                    action, (ActionAbilityPeekOwnSelect, ActionAbilityPeekOtherSelect)
                ):
                    assert obs.peeked_cards, f"{action} surfaced no peek"
                    peeks += 1
                if isinstance(action, ActionAbilityKingLookSelect):
                    assert obs.peeked_cards, "the King look surfaced no faces"
                    king_looks += 1
                if (
                    isinstance(action, ActionAbilityKingSwapDecision)
                    and action.perform_swap
                ):
                    assert (
                        obs.king_swap_indices is not None
                    ), "a performed King swap named no slots"
                    king_swaps += 1
        finally:
            state.close()

    assert peeks > 0, "no peek ability was reached"
    assert king_looks > 0, "the King look was never reached"
    assert king_swaps > 0, "a performed King swap was never reached"


@skip_if_no_go
def test_search_frame_stays_reduced():
    """The default frame is still the one the best-response search was verified on.

    ``AnalysisTools._filter_observation_for_br`` nulls peeked_cards but not
    king_swap_indices, so handing the search the fuller frame would quietly move
    its belief updates and its exploitability numbers.
    """
    reduced_swaps = 0
    for seed in _SEEDS:
        _, state = _new_pair(seed)
        try:
            rng = random.Random(0xF00D ^ seed)
            for _ in range(_MAX_STEPS):
                if state.is_terminal():
                    break
                actor = state.acting_player()
                action_idx, action = _pick_ability_first(rng, state.legal_actions())
                state.apply(action_idx)
                assert state.observation(action, actor).peeked_cards is None
                assert state.observation(action, actor).king_swap_indices is None
                if (
                    isinstance(action, ActionAbilityKingSwapDecision)
                    and action.perform_swap
                ):
                    reduced_swaps += 1
                    assert (
                        state.observation(
                            action, actor, ability_reveals=True
                        ).king_swap_indices
                        is not None
                    )
        finally:
            state.close()
    assert reduced_swaps > 0, "the reduced frame was never checked on a King swap"
