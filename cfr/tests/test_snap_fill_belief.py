"""The RULES.md 5 fill keeps the mover's knowledge of the card it moves (cambia-1552).

After a successful opponent snap the snapper pays a card of their own into the slot the
snap emptied. The card keeps its identity, so what the snapper knew about it is what they
know about its new home; the reference tracker used to drop that knowledge, handing a card
it had peeked back to itself as UNKNOWN. The same rule covers the card a blind swap sends
away: the swap is blind in the card RECEIVED, not the one given.

The Go half of this pair lives in engine/agent/snap_fill_test.go and asserts the same
belief about the same destination slot.
"""

from __future__ import annotations

from src.agent_state import AgentObservation, AgentState
from src.card import Card
from src.config import Config
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    CardBucket,
    EpistemicTag,
)
from src.abstraction import get_card_bucket

OPP_SLOTS_START = 6

# The card the agent peeks and then pays away.
PAID = Card(rank="7", suit="S")
HAND = [Card(rank="2", suit="H"), PAID, Card(rank="3", suit="C"), Card(rank="4", suit="D")]


def _agent(peeked: tuple = (1,)) -> AgentState:
    """A 2-player agent that has peeked its own slot 1 at the deal."""
    from src.config import CambiaRulesConfig, DeepCfrConfig

    cfg = Config(cambia_rules=CambiaRulesConfig(), deep_cfr=DeepCfrConfig())
    st = AgentState(
        player_id=0,
        opponent_id=1,
        memory_level=1,
        time_decay_turns=3,
        initial_hand_size=4,
        config=cfg,
    )
    st.initialize(
        AgentObservation(
            acting_player=-1,
            action=None,
            discard_top_card=Card(rank="K", suit="S"),
            player_hand_sizes=[4, 4],
            stockpile_size=40,
        ),
        HAND,
        peeked,
    )
    return st


def _obs(action, hand_sizes, snap_results=None, turn=1) -> AgentObservation:
    return AgentObservation(
        acting_player=0,
        action=action,
        discard_top_card=Card(rank="9", suit="H"),
        player_hand_sizes=hand_sizes,
        stockpile_size=38,
        snap_results=snap_results or [],
        current_turn=turn,
    )


def test_snap_fill_carries_the_movers_knowledge():
    st = _agent()
    paid_bucket = get_card_bucket(PAID)
    assert st.own_hand[1].bucket == paid_bucket
    peek_turn = st.own_hand[1].last_seen_turn

    # We snap the opponent's slot 0, which empties it and owes them a card.
    st.update(
        _obs(
            ActionSnapOpponent(opponent_target_hand_index=0),
            [4, 3],
            snap_results=[
                {
                    "snapper": 0,
                    "success": True,
                    "penalty": False,
                    "action_type": "ActionSnapOpponent",
                    "removed_opponent_index": 0,
                }
            ],
        )
    )

    # We pay the fill with the card we peeked.
    st.update(_obs(ActionSnapOpponentMove(1, 0), [3, 4]))

    assert st.opponent_belief[0] == paid_bucket
    assert st.opponent_last_seen_turn[0] == peek_turn
    assert st.slot_tags[OPP_SLOTS_START] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[OPP_SLOTS_START] == paid_bucket.value
    # The card left our hand.
    assert len(st.own_hand) == 3
    assert all(info.card != PAID for info in st.own_hand.values())


def test_snap_fill_with_an_unknown_card_stays_unknown():
    st = _agent(peeked=())
    st.update(
        _obs(
            ActionSnapOpponent(opponent_target_hand_index=0),
            [4, 3],
            snap_results=[
                {
                    "snapper": 0,
                    "success": True,
                    "penalty": False,
                    "action_type": "ActionSnapOpponent",
                    "removed_opponent_index": 0,
                }
            ],
        )
    )
    st.update(_obs(ActionSnapOpponentMove(1, 0), [3, 4]))

    assert st.opponent_belief[0] == CardBucket.UNKNOWN
    assert st.slot_tags[OPP_SLOTS_START] == EpistemicTag.UNK


def test_blind_swap_carries_the_swappers_knowledge():
    st = _agent()
    paid_bucket = get_card_bucket(PAID)
    peek_turn = st.own_hand[1].last_seen_turn

    st.update(
        _obs(
            ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=2),
            [4, 4],
        )
    )

    # The card we sent keeps its identity in the slot it landed in.
    assert st.opponent_belief[2] == paid_bucket
    assert st.opponent_last_seen_turn[2] == peek_turn
    assert st.slot_tags[OPP_SLOTS_START + 2] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[OPP_SLOTS_START + 2] == paid_bucket.value
    # The card we received is blind.
    assert st.own_hand[1].bucket == CardBucket.UNKNOWN
    assert st.slot_tags[1] == EpistemicTag.UNK
