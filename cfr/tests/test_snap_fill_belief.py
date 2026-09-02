"""A card that changes hands keeps whatever its holders knew about it.

After a successful opponent snap the snapper pays a card of their own into the slot the
snap emptied (RULES.md 5). The card keeps its identity, so what the snapper knew about it
is what they know about its new home, and what the RECEIVER knew about it is what they now
know about their own slot; the reference tracker used to drop both (cambia-1552 for the
mover, cambia-1690 for the receiver). The same rule governs a swap: a blind or King swap
moves two cards and reveals nothing, so each face travels to the slot its card landed in
(cambia-1553). "Blind" names what the swap itself shows, not a licence to forget a face
already seen.

The Go half of this pair lives in engine/agent/snap_fill_test.go and engine/agent/
state_test.go and asserts the same beliefs about the same destination slots.
"""

from __future__ import annotations

from src.agent_state import AgentObservation, AgentState
from src.card import Card
from src.config import Config
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    CardBucket,
    EpistemicTag,
)
from src.abstraction import get_card_bucket

OPP_SLOTS_START = 6

# The card the agent peeks and then pays away.
PAID = Card(rank="7", suit="S")
HAND = [
    Card(rank="2", suit="H"),
    PAID,
    Card(rank="3", suit="C"),
    Card(rank="4", suit="D"),
]


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


def _obs(
    action,
    hand_sizes,
    snap_results=None,
    turn=1,
    actor=0,
    peeked=None,
    king_swap_indices=None,
) -> AgentObservation:
    return AgentObservation(
        acting_player=actor,
        action=action,
        discard_top_card=Card(rank="9", suit="H"),
        player_hand_sizes=hand_sizes,
        stockpile_size=38,
        snap_results=snap_results or [],
        current_turn=turn,
        peeked_cards=peeked,
        king_swap_indices=king_swap_indices,
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


# The card the opponent holds and pays to us, and the one we peek in their hand.
GIVEN = Card(rank="8", suit="D")


def _peek_opponent_slot(st: AgentState, slot: int, card: Card, turn: int = 1):
    """Drive a 9/T peek of the opponent's `slot` so the agent knows that card."""
    from src.constants import ActionAbilityPeekOtherSelect

    st.update(
        _obs(
            ActionAbilityPeekOtherSelect(target_opponent_hand_index=slot),
            [4, 4],
            turn=turn,
            peeked={(1, slot): card},
        )
    )


def test_the_receiver_keeps_what_it_knew_about_the_fill():
    """The mirror of the case above: the seat handed the fill keeps its own knowledge."""
    st = _agent(peeked=())
    _peek_opponent_slot(st, 1, GIVEN)
    given_bucket = get_card_bucket(GIVEN)
    assert st.opponent_belief[1] == given_bucket
    peek_turn = st.opponent_last_seen_turn[1]

    # The opponent snaps our slot 0, which empties it and owes us a card.
    st.update(
        _obs(
            ActionSnapOpponent(opponent_target_hand_index=0),
            [3, 4],
            actor=1,
            turn=2,
            snap_results=[
                {
                    "snapper": 1,
                    "success": True,
                    "penalty": False,
                    "action_type": "ActionSnapOpponent",
                    "removed_opponent_index": 0,
                }
            ],
        )
    )
    # They pay the fill with the card we had peeked.
    st.update(_obs(ActionSnapOpponentMove(1, 0), [4, 3], actor=1, turn=2))

    assert st.own_hand[0].bucket == given_bucket
    assert st.own_hand[0].last_seen_turn == peek_turn
    assert st.slot_tags[0] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[0] == given_bucket.value


def test_the_fill_lands_in_the_slot_the_snap_emptied():
    """The fill inserts at the vacated slot, so every later slot keeps its belief."""
    st = _agent()  # our slot 1 holds PAID and is peeked at the deal
    # Learn the opponent's slots 1 and 2 so a misplaced insert shows up as a shift.
    two = Card(rank="2", suit="S")
    nine = Card(rank="9", suit="C")
    _peek_opponent_slot(st, 1, two)
    _peek_opponent_slot(st, 2, nine)

    st.update(
        _obs(
            ActionSnapOpponent(opponent_target_hand_index=0),
            [4, 3],
            turn=2,
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
    # The snap shifted their hand left: what was slot 1 is now slot 0.
    assert st.opponent_belief[0] == get_card_bucket(two)
    assert st.opponent_belief[1] == get_card_bucket(nine)

    st.update(_obs(ActionSnapOpponentMove(1, 0), [3, 4], turn=2))

    # The fill goes in at slot 0 and pushes the rest back to where they were.
    assert st.opponent_belief[0] == get_card_bucket(PAID)
    assert st.opponent_belief[1] == get_card_bucket(two)
    assert st.opponent_belief[2] == get_card_bucket(nine)


def test_king_swap_carries_both_looked_at_faces():
    st = _agent(peeked=())
    own_card = HAND[2]
    opp_card = GIVEN
    st.update(
        _obs(
            ActionAbilityKingLookSelect(own_hand_index=2, opponent_hand_index=1),
            [4, 4],
            peeked={(0, 2): own_card, (1, 1): opp_card},
        )
    )
    assert st.own_hand[2].bucket == get_card_bucket(own_card)
    assert st.opponent_belief[1] == get_card_bucket(opp_card)

    st.update(
        _obs(
            ActionAbilityKingSwapDecision(perform_swap=True),
            [4, 4],
            turn=2,
            king_swap_indices=(2, 1),
        )
    )

    # Each face followed its card.
    assert st.own_hand[2].bucket == get_card_bucket(opp_card)
    assert st.opponent_belief[1] == get_card_bucket(own_card)
    assert st.slot_tags[2] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[2] == get_card_bucket(opp_card).value
    assert st.slot_tags[OPP_SLOTS_START + 1] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[OPP_SLOTS_START + 1] == get_card_bucket(own_card).value


def test_blind_swap_carries_the_face_we_had_peeked_into_our_hand():
    """We know the card a blind swap hands us when we had already peeked its slot."""
    st = _agent(peeked=())
    _peek_opponent_slot(st, 2, GIVEN)
    given_bucket = get_card_bucket(GIVEN)

    st.update(
        _obs(
            ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=2),
            [4, 4],
            turn=2,
        )
    )

    assert st.own_hand[1].bucket == given_bucket
    assert st.slot_tags[1] == EpistemicTag.PRIV_OWN
    assert st.slot_buckets[1] == given_bucket.value


def test_observer_of_a_blind_swap_keeps_the_face_it_gave_away():
    """From the far seat the same rule applies to the card the swap took from us."""
    st = _agent()  # our slot 1 is peeked at the deal
    paid_bucket = get_card_bucket(PAID)
    peek_turn = st.own_hand[1].last_seen_turn

    # The opponent swaps their slot 0 with our slot 1.
    st.update(
        _obs(
            ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=1),
            [4, 4],
            actor=1,
            turn=2,
        )
    )

    assert st.opponent_belief[0] == paid_bucket
    assert st.opponent_last_seen_turn[0] == peek_turn
    assert st.own_hand[1].bucket == CardBucket.UNKNOWN
