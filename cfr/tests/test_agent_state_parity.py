"""Tier 1 tests for Phase 0 AgentState parity fields (Stream B B1, B2).

Covers:
- observation_ages: ``slot_last_seen_turn`` is stamped on peek actions and reset on swaps.
- dead-card histogram: ``discard_bucket_counts`` + ``total_discards_seen`` update on
  Discard and Replace actions; initial discard bumps once.
- turn_progress: derived from ``_current_game_turn`` / ``max_game_turns``.
- action_history: ring buffer updates oldest-first on ``update()``, per-player keyed.
- clone() copies all new fields.
"""

from __future__ import annotations

import pytest

from src.agent_state import (
    AgentObservation,
    AgentState,
    KnownCardInfo,
)
from src.card import Card
from src.config import Config
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionDiscard,
    ActionDrawStockpile,
    ActionReplace,
    ActionSnapOwn,
    CardBucket,
    V2_ACTION_CATEGORY_ABILITY_SNAP,
    V2_ACTION_CATEGORY_DISCARD,
    V2_ACTION_CATEGORY_DRAW,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_state(*, player_id: int = 0, opponent_id: int = 1) -> AgentState:
    from src.config import CambiaRulesConfig, DeepCfrConfig

    cfg = Config(cambia_rules=CambiaRulesConfig(), deep_cfr=DeepCfrConfig())
    st = AgentState(
        player_id=player_id,
        opponent_id=opponent_id,
        memory_level=0,
        time_decay_turns=3,
        initial_hand_size=4,
        config=cfg,
    )
    # Initialize with a minimal observation.
    obs = AgentObservation(
        acting_player=-1,
        action=None,
        discard_top_card=Card(rank="5", suit="S"),  # MID_NUM bucket
        player_hand_sizes=[4, 4],
        stockpile_size=40,
        snap_results=[],
        current_turn=0,
    )
    peeks = (0, 1)
    hand = [
        Card(rank="A", suit="S"),  # Ace bucket (2)
        Card(rank="2", suit="H"),  # LowNum bucket (3)
        Card(rank="7", suit="D"),
        Card(rank="K", suit="S"),
    ]
    st.initialize(obs, hand, peeks)
    return st


def _make_obs(
    *,
    action,
    acting_player: int,
    turn: int,
    discard_top: Card,
    hand_sizes=(4, 4),
    stockpile_size: int = 40,
    peeked=None,
    snap_results=None,
) -> AgentObservation:
    return AgentObservation(
        acting_player=acting_player,
        action=action,
        discard_top_card=discard_top,
        player_hand_sizes=list(hand_sizes),
        stockpile_size=stockpile_size,
        snap_results=list(snap_results or []),
        peeked_cards=peeked,
        did_cambia_get_called=False,
        who_called_cambia=None,
        is_game_over=False,
        current_turn=turn,
    )


# ---------------------------------------------------------------------------
# Parity fields B1: observation_ages / discard histogram / turn_progress
# ---------------------------------------------------------------------------


def test_initialize_stamps_last_seen_for_peeked_slots():
    st = _mk_state()
    # Slots 0 and 1 were peeked at turn 0.
    assert st.slot_last_seen_turn[0] == 0
    assert st.slot_last_seen_turn[1] == 0
    # Slots 2, 3 remain unseen.
    assert st.slot_last_seen_turn[2] == -1
    assert st.slot_last_seen_turn[3] == -1
    # Opponent slots all unseen.
    for i in range(6, 12):
        assert st.slot_last_seen_turn[i] == -1


def test_initial_discard_tracked_once():
    st = _mk_state()
    # Initial discard_top was a 5 (MID_NUM bucket = 4).
    assert st.total_discards_seen == 1
    assert st.discard_bucket_counts[CardBucket.MID_NUM.value] == 1


def test_discard_action_increments_histogram():
    st = _mk_state()
    # Opponent performs Discard: discard top becomes an Ace.
    obs = _make_obs(
        action=ActionDiscard(use_ability=False),
        acting_player=1,
        turn=1,
        discard_top=Card(rank="A", suit="H"),
    )
    st.update(obs)
    assert st.discard_bucket_counts[CardBucket.ACE.value] == 1
    assert st.total_discards_seen == 2  # initial + this discard


def test_replace_action_increments_histogram():
    st = _mk_state()
    obs = _make_obs(
        action=ActionReplace(target_hand_index=0),
        acting_player=1,
        turn=1,
        discard_top=Card(rank="K", suit="S"),  # HIGH_KING = 8
    )
    st.update(obs)
    assert st.discard_bucket_counts[CardBucket.HIGH_KING.value] == 1


def test_non_discard_actions_do_not_bump_histogram():
    st = _mk_state()
    before = list(st.discard_bucket_counts)
    before_total = st.total_discards_seen
    obs = _make_obs(
        action=ActionDrawStockpile(),
        acting_player=1,
        turn=1,
        discard_top=Card(rank="5", suit="S"),  # unchanged top card
    )
    st.update(obs)
    assert st.discard_bucket_counts == before
    assert st.total_discards_seen == before_total


def test_peek_own_updates_last_seen_turn():
    st = _mk_state()
    # Self peeks own slot 2. Slot 2 was not initially peeked, so last_seen stays -1.
    card_peeked = Card(rank="7", suit="D")
    obs = _make_obs(
        action=ActionAbilityPeekOwnSelect(target_hand_index=2),
        acting_player=0,
        turn=3,
        discard_top=Card(rank="5", suit="S"),  # unchanged (peek does not discard)
        peeked={(0, 2): card_peeked},
    )
    # Note: the action's public discard-top update occurs first, then peek is processed.
    # PeekOwn does NOT push to discard, so pass the same top card.
    st.update(obs)
    # Slot 2 should now have last_seen_turn == 3.
    assert st.slot_last_seen_turn[2] == 3


def test_turn_progress_derivation():
    """turn_progress = current_turn / max_game_turns; max_game_turns from config."""
    st = _mk_state()
    # Config default max_game_turns is 300.
    st._current_game_turn = 60
    assert st.max_game_turns == 300
    # Computed externally by encoder: age = current_turn / max_turns.
    assert (st._current_game_turn / st.max_game_turns) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Action history ring buffer B2
# ---------------------------------------------------------------------------


def test_action_history_records_draw_category():
    st = _mk_state()
    obs = _make_obs(
        action=ActionDrawStockpile(),
        acting_player=0,
        turn=1,
        discard_top=Card(rank="5", suit="S"),
    )
    st.update(obs)
    own_ring = st.action_history[0]
    # Go-parity semantics: fill from index 0 outward. First push lands at index 0;
    # indices 1 and 2 remain empty until later pushes.
    assert own_ring[0] == (V2_ACTION_CATEGORY_DRAW, 0.0)
    assert own_ring[1] is None
    assert own_ring[2] is None


def test_action_history_ring_oldest_first_overflow():
    st = _mk_state()
    # Push 4 actions for the same actor; ring keeps the last 3 with oldest at index 0.
    actions = [
        ActionDrawStockpile(),  # category 0, slot 0.0
        ActionDiscard(use_ability=False),  # category 1, slot 0.0
        ActionAbilityPeekOwnSelect(target_hand_index=2),  # category 2, slot 2/5 = 0.4
        ActionSnapOwn(own_card_hand_index=3),  # category 2, slot 3/5 = 0.6
    ]
    # Send actions at increasing turns. Discard and SnapOwn imply a discard pile update;
    # SnapOwn needs snap_results to not crash reconciliation: feed a minimal result.
    st.update(
        _make_obs(
            action=actions[0],
            acting_player=0,
            turn=1,
            discard_top=Card(rank="5", suit="S"),
        )
    )
    st.update(
        _make_obs(
            action=actions[1],
            acting_player=0,
            turn=2,
            discard_top=Card(rank="A", suit="H"),
        )
    )
    # Peek: discard pile unchanged (previous top persists).
    st.update(
        _make_obs(
            action=actions[2],
            acting_player=0,
            turn=3,
            discard_top=Card(rank="A", suit="H"),
            peeked={(0, 2): Card(rank="7", suit="D")},
        )
    )
    # SnapOwn with successful snap removing own slot 3. Provide snap_results so the
    # reconciliation keeps hand size consistent: snap removes 1 from own.
    st.update(
        _make_obs(
            action=actions[3],
            acting_player=0,
            turn=4,
            discard_top=Card(rank="K", suit="S"),
            hand_sizes=(3, 4),
            snap_results=[
                {
                    "snapper": 0,
                    "success": True,
                    "penalty": False,
                    "action_type": "ActionSnapOwn",
                    "removed_own_index": 3,
                }
            ],
        )
    )
    ring = st.action_history[0]
    # After 4 pushes the oldest surviving is the 2nd action (DISCARD).
    assert ring[0][0] == V2_ACTION_CATEGORY_DISCARD
    assert ring[1][0] == V2_ACTION_CATEGORY_ABILITY_SNAP
    assert ring[2][0] == V2_ACTION_CATEGORY_ABILITY_SNAP
    # slot_norm for ring[1] (Peek slot 2) == 2/5 == 0.4.
    assert ring[1][1] == pytest.approx(0.4)
    # slot_norm for ring[2] (SnapOwn slot 3) == 3/5 == 0.6.
    assert ring[2][1] == pytest.approx(0.6)


def test_action_history_keyed_by_actor():
    st = _mk_state()
    # Opponent draws; own ring stays empty.
    st.update(
        _make_obs(
            action=ActionDrawStockpile(),
            acting_player=1,
            turn=1,
            discard_top=Card(rank="5", suit="S"),
        )
    )
    assert all(x is None for x in st.action_history[0])
    # Go-parity: first push lands at index 0, not index -1.
    assert st.action_history[1][0] == (V2_ACTION_CATEGORY_DRAW, 0.0)
    assert st.action_history[1][1] is None
    assert st.action_history[1][2] is None


def test_action_history_ignores_system_actor():
    st = _mk_state()
    obs = _make_obs(
        action=ActionDrawStockpile(),
        acting_player=-1,
        turn=1,
        discard_top=Card(rank="5", suit="S"),
    )
    st.update(obs)
    # Neither ring should have received an entry.
    assert all(x is None for x in st.action_history[0])
    assert all(x is None for x in st.action_history[1])


# ---------------------------------------------------------------------------
# clone() preserves new fields
# ---------------------------------------------------------------------------


def test_clone_copies_parity_fields():
    st = _mk_state()
    st.update(
        _make_obs(
            action=ActionDiscard(use_ability=False),
            acting_player=0,
            turn=1,
            discard_top=Card(rank="A", suit="H"),
        )
    )
    clone = st.clone()
    assert clone.slot_last_seen_turn == st.slot_last_seen_turn
    assert clone.discard_bucket_counts == st.discard_bucket_counts
    assert clone.total_discards_seen == st.total_discards_seen
    assert clone.max_game_turns == st.max_game_turns
    assert clone.action_history == st.action_history
    # Mutating clone must not affect original.
    clone.slot_last_seen_turn[0] = 999
    clone.discard_bucket_counts[0] = 999
    clone.action_history[0][0] = (V2_ACTION_CATEGORY_DISCARD, 0.5)
    assert st.slot_last_seen_turn[0] != 999
    assert st.discard_bucket_counts[0] != 999
    assert st.action_history[0][0] is None or st.action_history[0][0] != (
        V2_ACTION_CATEGORY_DISCARD,
        0.5,
    )
