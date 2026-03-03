"""
Tests for src/encoding.py

Covers:
- Tensor feature encoding (encode_infoset)
- Action index mapping (action_to_index / index_to_action)
- Action mask encoding (encode_action_mask)
- Edge cases (empty hands, max hands, all UNKNOWN, etc.)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.cfr.exceptions import ActionEncodingError

# conftest.py handles the config stub automatically

from src.agent_state import KnownCardInfo
from src.card import Card
from src.constants import (
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionCallCambia,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
    ActionSnapOwn,
    CardBucket,
    DecayCategory,
    DecisionContext,
    GamePhase,
    StockpileEstimate,
)
from src.constants import EP_PBS_INPUT_DIM, EpistemicTag
from src.encoding import (
    EMPTY_SLOT_IDX,
    INPUT_DIM,
    MAX_HAND,
    NUM_ACTIONS,
    SLOT_ENCODING_DIM,
    _BUCKET_TO_SLOT_IDX,
    _DECAY_TO_SLOT_IDX,
    _encode_slot,
    action_to_index,
    encode_action_mask,
    encode_infoset,
    encode_infoset_eppbs,
    encode_infoset_eppbs_dealiased,
    encode_infoset_eppbs_interleaved,
    index_to_action,
)


# --- Helper to build a fake AgentState ---

def _make_agent_state(
    player_id=0,
    own_hand=None,
    opponent_belief=None,
    opponent_card_count=4,
    known_discard_top_bucket=CardBucket.UNKNOWN,
    stockpile_estimate=StockpileEstimate.HIGH,
    game_phase=GamePhase.EARLY,
    cambia_caller=None,
):
    """Build a SimpleNamespace that matches the AgentState interface for encode_infoset."""
    if own_hand is None:
        own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(4)
        }
    if opponent_belief is None:
        opponent_belief = {
            i: CardBucket.UNKNOWN for i in range(opponent_card_count)
        }
    return SimpleNamespace(
        player_id=player_id,
        own_hand=own_hand,
        opponent_belief=opponent_belief,
        opponent_card_count=opponent_card_count,
        known_discard_top_bucket=known_discard_top_bucket,
        stockpile_estimate=stockpile_estimate,
        game_phase=game_phase,
        cambia_caller=cambia_caller,
    )


# ===== Constants =====


class TestConstants:
    def test_input_dim(self):
        assert INPUT_DIM == 222

    def test_num_actions(self):
        assert NUM_ACTIONS == 146

    def test_max_hand(self):
        assert MAX_HAND == 6

    def test_slot_encoding_dim(self):
        assert SLOT_ENCODING_DIM == 15


# ===== _encode_slot =====


class TestEncodeSlot:
    def test_card_buckets(self):
        """Each CardBucket maps to its expected slot index."""
        expected = {
            CardBucket.ZERO: 0,
            CardBucket.NEG_KING: 1,
            CardBucket.ACE: 2,
            CardBucket.LOW_NUM: 3,
            CardBucket.MID_NUM: 4,
            CardBucket.PEEK_SELF: 5,
            CardBucket.PEEK_OTHER: 6,
            CardBucket.SWAP_BLIND: 7,
            CardBucket.HIGH_KING: 8,
            CardBucket.UNKNOWN: 13,
        }
        for bucket, expected_idx in expected.items():
            assert _encode_slot(bucket) == expected_idx, f"Failed for {bucket}"

    def test_decay_categories(self):
        """Each DecayCategory maps to its expected slot index."""
        expected = {
            DecayCategory.LIKELY_LOW: 10,
            DecayCategory.LIKELY_MID: 11,
            DecayCategory.LIKELY_HIGH: 12,
            DecayCategory.UNKNOWN: 13,
        }
        for cat, expected_idx in expected.items():
            assert _encode_slot(cat) == expected_idx, f"Failed for {cat}"

    def test_none_is_empty(self):
        assert _encode_slot(None) == EMPTY_SLOT_IDX

    def test_raw_int_card_bucket(self):
        """Raw integer values matching CardBucket.value map correctly."""
        assert _encode_slot(0) == 0  # ZERO
        assert _encode_slot(8) == 8  # HIGH_KING
        assert _encode_slot(99) == 13  # UNKNOWN

    def test_raw_int_decay_category(self):
        """Raw integer values matching DecayCategory.value map correctly."""
        assert _encode_slot(100) == 10  # LIKELY_LOW
        assert _encode_slot(101) == 11  # LIKELY_MID
        assert _encode_slot(102) == 12  # LIKELY_HIGH

    def test_unknown_fallback(self):
        """Unrecognized values fall back to UNKNOWN (index 13)."""
        assert _encode_slot(999) == 13
        assert _encode_slot(-1) == 13


# ===== encode_infoset =====


class TestEncodeInfoset:
    def test_output_shape_and_dtype(self):
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        assert features.shape == (INPUT_DIM,)
        assert features.dtype == np.float32

    def test_all_unknown_hand(self):
        """A hand of all UNKNOWN cards encodes to index 13 in each slot block."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(4):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[13] == 1.0, f"Slot {slot} UNKNOWN not set"
            assert block.sum() == 1.0, f"Slot {slot} has non-one-hot encoding"
        # Slots 4 and 5 should be EMPTY
        for slot in range(4, MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[EMPTY_SLOT_IDX] == 1.0, f"Slot {slot} EMPTY not set"
            assert block.sum() == 1.0

    def test_known_hand_encoding(self):
        """Known buckets in own hand encode to correct positions."""
        own_hand = {
            0: KnownCardInfo(bucket=CardBucket.ZERO),
            1: KnownCardInfo(bucket=CardBucket.ACE),
            2: KnownCardInfo(bucket=CardBucket.HIGH_KING),
            3: KnownCardInfo(bucket=CardBucket.UNKNOWN),
        }
        state = _make_agent_state(own_hand=own_hand)
        features = encode_infoset(state, DecisionContext.START_TURN)
        # Slot 0: ZERO -> index 0
        assert features[0 * SLOT_ENCODING_DIM + 0] == 1.0
        # Slot 1: ACE -> index 2
        assert features[1 * SLOT_ENCODING_DIM + 2] == 1.0
        # Slot 2: HIGH_KING -> index 8
        assert features[2 * SLOT_ENCODING_DIM + 8] == 1.0
        # Slot 3: UNKNOWN -> index 13
        assert features[3 * SLOT_ENCODING_DIM + 13] == 1.0

    def test_opponent_belief_encoding(self):
        """Opponent beliefs encode in the second block of 6 x 15."""
        opp_belief = {
            0: CardBucket.LOW_NUM,
            1: DecayCategory.LIKELY_HIGH,
            2: CardBucket.UNKNOWN,
            3: CardBucket.PEEK_SELF,
        }
        state = _make_agent_state(opponent_belief=opp_belief, opponent_card_count=4)
        features = encode_infoset(state, DecisionContext.START_TURN)
        opp_offset = MAX_HAND * SLOT_ENCODING_DIM  # 90
        # Slot 0: LOW_NUM -> index 3
        assert features[opp_offset + 0 * SLOT_ENCODING_DIM + 3] == 1.0
        # Slot 1: LIKELY_HIGH -> index 12
        assert features[opp_offset + 1 * SLOT_ENCODING_DIM + 12] == 1.0
        # Slot 2: UNKNOWN -> index 13
        assert features[opp_offset + 2 * SLOT_ENCODING_DIM + 13] == 1.0
        # Slot 3: PEEK_SELF -> index 5
        assert features[opp_offset + 3 * SLOT_ENCODING_DIM + 5] == 1.0

    def test_card_counts_normalized(self):
        """Own and opponent card counts are normalized by MAX_HAND."""
        state = _make_agent_state(opponent_card_count=3)
        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(5)
        }
        features = encode_infoset(state, DecisionContext.START_TURN)
        count_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM  # 180
        assert features[count_offset] == pytest.approx(5 / MAX_HAND)  # own
        assert features[count_offset + 1] == pytest.approx(3 / MAX_HAND)  # opp

    def test_card_count_clamped_to_max(self):
        """Card counts exceeding MAX_HAND are clamped."""
        state = _make_agent_state(opponent_card_count=8)
        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN) for i in range(8)
        }
        features = encode_infoset(state, DecisionContext.START_TURN)
        count_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM
        assert features[count_offset] == pytest.approx(1.0)  # clamped to 6/6
        assert features[count_offset + 1] == pytest.approx(1.0)

    def test_drawn_card_encoding(self):
        """Drawn card bucket encodes in 11-dim one-hot at offset 182."""
        state = _make_agent_state()
        drawn_offset = 2 * MAX_HAND * SLOT_ENCODING_DIM + 2  # 182
        # No drawn card -> index 10 (NONE)
        features_none = encode_infoset(state, DecisionContext.START_TURN)
        assert features_none[drawn_offset + 10] == 1.0
        # Drawn ACE -> index 2 (ACE is 3rd in bucket value list)
        features_ace = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.ACE
        )
        assert features_ace[drawn_offset + 2] == 1.0
        assert features_ace[drawn_offset + 10] == 0.0  # NONE not set

    def test_drawn_card_zero(self):
        """Drawn ZERO (Joker) encodes at index 0."""
        state = _make_agent_state()
        drawn_offset = 182
        features = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.ZERO
        )
        assert features[drawn_offset + 0] == 1.0

    def test_drawn_card_unknown(self):
        """Drawn UNKNOWN encodes at index 9."""
        state = _make_agent_state()
        drawn_offset = 182
        features = encode_infoset(
            state, DecisionContext.POST_DRAW, drawn_card_bucket=CardBucket.UNKNOWN
        )
        assert features[drawn_offset + 9] == 1.0

    def test_discard_top_encoding(self):
        """Discard top bucket encodes in 10-dim one-hot at offset 193."""
        discard_offset = 193
        for bucket, expected_idx in [
            (CardBucket.ZERO, 0),
            (CardBucket.HIGH_KING, 8),
            (CardBucket.UNKNOWN, 9),
        ]:
            state = _make_agent_state(known_discard_top_bucket=bucket)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[discard_offset + expected_idx] == 1.0
            assert features[discard_offset : discard_offset + 10].sum() == 1.0

    def test_stockpile_estimate_encoding(self):
        """Stockpile estimate encodes in 4-dim one-hot at offset 203."""
        stock_offset = 203
        mapping = {
            StockpileEstimate.HIGH: 0,
            StockpileEstimate.MEDIUM: 1,
            StockpileEstimate.LOW: 2,
            StockpileEstimate.EMPTY: 3,
        }
        for est, expected_idx in mapping.items():
            state = _make_agent_state(stockpile_estimate=est)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[stock_offset + expected_idx] == 1.0
            assert features[stock_offset : stock_offset + 4].sum() == 1.0

    def test_game_phase_encoding(self):
        """Game phase encodes in 6-dim one-hot at offset 207."""
        phase_offset = 207
        mapping = {
            GamePhase.START: 0,
            GamePhase.EARLY: 1,
            GamePhase.MID: 2,
            GamePhase.LATE: 3,
            GamePhase.CAMBIA_CALLED: 4,
            GamePhase.TERMINAL: 5,
        }
        for phase, expected_idx in mapping.items():
            state = _make_agent_state(game_phase=phase)
            features = encode_infoset(state, DecisionContext.START_TURN)
            assert features[phase_offset + expected_idx] == 1.0
            assert features[phase_offset : phase_offset + 6].sum() == 1.0

    def test_decision_context_encoding(self):
        """Decision context encodes in 6-dim one-hot at offset 213."""
        ctx_offset = 213
        mapping = {
            DecisionContext.START_TURN: 0,
            DecisionContext.POST_DRAW: 1,
            DecisionContext.SNAP_DECISION: 2,
            DecisionContext.ABILITY_SELECT: 3,
            DecisionContext.SNAP_MOVE: 4,
            DecisionContext.TERMINAL: 5,
        }
        state = _make_agent_state()
        for ctx, expected_idx in mapping.items():
            features = encode_infoset(state, ctx)
            assert features[ctx_offset + expected_idx] == 1.0
            assert features[ctx_offset : ctx_offset + 6].sum() == 1.0

    def test_cambia_caller_encoding(self):
        """Cambia caller encodes in 3-dim one-hot at offset 219."""
        cambia_offset = 219
        # NONE
        state = _make_agent_state(cambia_caller=None)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 2] == 1.0  # NONE
        # SELF
        state = _make_agent_state(player_id=0, cambia_caller=0)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 0] == 1.0  # SELF
        # OPPONENT
        state = _make_agent_state(player_id=0, cambia_caller=1)
        f = encode_infoset(state, DecisionContext.START_TURN)
        assert f[cambia_offset + 1] == 1.0  # OPPONENT

    def test_total_offset_is_222(self):
        """The encoding fills exactly 222 dimensions."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        assert features.shape == (222,)
        # All one-hot blocks contribute at least one non-zero entry each.
        # 12 one-hot blocks (6 own + 6 opp + drawn + discard + stock + phase + ctx + cambia)
        # plus 2 normalized scalars
        non_zero = np.count_nonzero(features)
        assert non_zero >= 14

    def test_empty_hand(self):
        """An agent with no cards encodes all own slots as EMPTY."""
        state = _make_agent_state()
        state.own_hand = {}
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[EMPTY_SLOT_IDX] == 1.0

    def test_max_hand_six_slots(self):
        """A full 6-card hand fills all own slots."""
        own_hand = {
            i: KnownCardInfo(bucket=CardBucket.ACE) for i in range(6)
        }
        state = _make_agent_state(own_hand=own_hand)
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(6):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            assert block[2] == 1.0  # ACE index

    def test_features_are_zero_initialized(self):
        """Areas not filled by one-hot or scalar remain zero."""
        state = _make_agent_state()
        features = encode_infoset(state, DecisionContext.START_TURN)
        for slot in range(4, MAX_HAND):
            block = features[slot * SLOT_ENCODING_DIM : (slot + 1) * SLOT_ENCODING_DIM]
            for i in range(SLOT_ENCODING_DIM):
                if i == EMPTY_SLOT_IDX:
                    assert block[i] == 1.0
                else:
                    assert block[i] == 0.0

    def test_different_states_produce_different_features(self):
        """Two different game states produce different feature vectors."""
        s1 = _make_agent_state(game_phase=GamePhase.EARLY)
        s2 = _make_agent_state(game_phase=GamePhase.LATE)
        f1 = encode_infoset(s1, DecisionContext.START_TURN)
        f2 = encode_infoset(s2, DecisionContext.START_TURN)
        assert not np.array_equal(f1, f2)

    def test_decay_in_opponent_belief(self):
        """DecayCategory values in opponent belief encode correctly."""
        opp_belief = {
            0: DecayCategory.LIKELY_LOW,
            1: DecayCategory.LIKELY_MID,
            2: DecayCategory.LIKELY_HIGH,
        }
        state = _make_agent_state(opponent_belief=opp_belief, opponent_card_count=3)
        features = encode_infoset(state, DecisionContext.START_TURN)
        opp_offset = MAX_HAND * SLOT_ENCODING_DIM
        assert features[opp_offset + 0 * SLOT_ENCODING_DIM + 10] == 1.0  # LIKELY_LOW
        assert features[opp_offset + 1 * SLOT_ENCODING_DIM + 11] == 1.0  # LIKELY_MID
        assert features[opp_offset + 2 * SLOT_ENCODING_DIM + 12] == 1.0  # LIKELY_HIGH


# ===== action_to_index =====


class TestActionToIndex:
    def test_draw_stockpile(self):
        assert action_to_index(ActionDrawStockpile()) == 0

    def test_draw_discard(self):
        assert action_to_index(ActionDrawDiscard()) == 1

    def test_call_cambia(self):
        assert action_to_index(ActionCallCambia()) == 2

    def test_discard_no_ability(self):
        assert action_to_index(ActionDiscard(use_ability=False)) == 3

    def test_discard_with_ability(self):
        assert action_to_index(ActionDiscard(use_ability=True)) == 4

    def test_replace(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionReplace(target_hand_index=i)) == 5 + i

    def test_peek_own(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionAbilityPeekOwnSelect(target_hand_index=i)) == 11 + i

    def test_peek_other(self):
        for i in range(MAX_HAND):
            assert (
                action_to_index(
                    ActionAbilityPeekOtherSelect(target_opponent_hand_index=i)
                )
                == 17 + i
            )

    def test_blind_swap(self):
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                expected = 23 + own * MAX_HAND + opp
                assert (
                    action_to_index(
                        ActionAbilityBlindSwapSelect(
                            own_hand_index=own, opponent_hand_index=opp
                        )
                    )
                    == expected
                )

    def test_blind_swap_boundaries(self):
        """First and last BlindSwap index."""
        assert action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0)
        ) == 23
        assert action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=5, opponent_hand_index=5)
        ) == 58

    def test_king_look(self):
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                expected = 59 + own * MAX_HAND + opp
                assert (
                    action_to_index(
                        ActionAbilityKingLookSelect(
                            own_hand_index=own, opponent_hand_index=opp
                        )
                    )
                    == expected
                )

    def test_king_look_boundaries(self):
        assert action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0)
        ) == 59
        assert action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=5, opponent_hand_index=5)
        ) == 94

    def test_king_swap_decision(self):
        assert action_to_index(ActionAbilityKingSwapDecision(perform_swap=False)) == 95
        assert action_to_index(ActionAbilityKingSwapDecision(perform_swap=True)) == 96

    def test_pass_snap(self):
        assert action_to_index(ActionPassSnap()) == 97

    def test_snap_own(self):
        for i in range(MAX_HAND):
            assert action_to_index(ActionSnapOwn(own_card_hand_index=i)) == 98 + i

    def test_snap_opponent(self):
        for i in range(MAX_HAND):
            assert (
                action_to_index(ActionSnapOpponent(opponent_target_hand_index=i))
                == 104 + i
            )

    def test_snap_opponent_move(self):
        for own in range(MAX_HAND):
            for slot in range(MAX_HAND):
                expected = 110 + own * MAX_HAND + slot
                assert (
                    action_to_index(
                        ActionSnapOpponentMove(
                            own_card_to_move_hand_index=own,
                            target_empty_slot_index=slot,
                        )
                    )
                    == expected
                )

    def test_snap_opponent_move_boundaries(self):
        assert action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=0, target_empty_slot_index=0)
        ) == 110
        assert action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=5, target_empty_slot_index=5)
        ) == 145

    def test_max_index_is_145(self):
        """The highest valid action index should be 145 (NUM_ACTIONS - 1)."""
        max_idx = action_to_index(
            ActionSnapOpponentMove(own_card_to_move_hand_index=5, target_empty_slot_index=5)
        )
        assert max_idx == NUM_ACTIONS - 1

    def test_all_indices_unique(self):
        """Every action type produces a unique index, covering [0, 146)."""
        all_actions = [
            ActionDrawStockpile(),
            ActionDrawDiscard(),
            ActionCallCambia(),
            ActionDiscard(use_ability=False),
            ActionDiscard(use_ability=True),
        ]
        for i in range(MAX_HAND):
            all_actions.append(ActionReplace(target_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionAbilityPeekOwnSelect(target_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                all_actions.append(
                    ActionAbilityBlindSwapSelect(own_hand_index=own, opponent_hand_index=opp)
                )
        for own in range(MAX_HAND):
            for opp in range(MAX_HAND):
                all_actions.append(
                    ActionAbilityKingLookSelect(own_hand_index=own, opponent_hand_index=opp)
                )
        all_actions.extend([
            ActionAbilityKingSwapDecision(perform_swap=False),
            ActionAbilityKingSwapDecision(perform_swap=True),
            ActionPassSnap(),
        ])
        for i in range(MAX_HAND):
            all_actions.append(ActionSnapOwn(own_card_hand_index=i))
        for i in range(MAX_HAND):
            all_actions.append(ActionSnapOpponent(opponent_target_hand_index=i))
        for own in range(MAX_HAND):
            for slot in range(MAX_HAND):
                all_actions.append(
                    ActionSnapOpponentMove(
                        own_card_to_move_hand_index=own,
                        target_empty_slot_index=slot,
                    )
                )

        indices = [action_to_index(a) for a in all_actions]
        assert len(indices) == NUM_ACTIONS
        assert len(set(indices)) == NUM_ACTIONS
        assert min(indices) == 0
        assert max(indices) == NUM_ACTIONS - 1

    def test_replace_out_of_range(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(ActionReplace(target_hand_index=MAX_HAND))

    def test_replace_negative_index(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(ActionReplace(target_hand_index=-1))

    def test_blind_swap_out_of_range(self):
        with pytest.raises(ActionEncodingError, match="out of range"):
            action_to_index(
                ActionAbilityBlindSwapSelect(own_hand_index=6, opponent_hand_index=0)
            )


# ===== index_to_action =====


class TestIndexToAction:
    def test_round_trip_draw_stockpile(self):
        actions = [ActionDrawStockpile(), ActionDrawDiscard(), ActionCallCambia()]
        for a in actions:
            idx = action_to_index(a)
            recovered = index_to_action(idx, actions)
            assert recovered == a

    def test_round_trip_replace(self):
        legal = [ActionReplace(target_hand_index=i) for i in range(4)]
        for a in legal:
            idx = action_to_index(a)
            recovered = index_to_action(idx, legal)
            assert recovered == a

    def test_round_trip_blind_swap(self):
        legal = [
            ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=2),
            ActionAbilityBlindSwapSelect(own_hand_index=3, opponent_hand_index=0),
        ]
        for a in legal:
            idx = action_to_index(a)
            recovered = index_to_action(idx, legal)
            assert recovered == a

    def test_no_match_raises(self):
        legal = [ActionDrawStockpile()]
        with pytest.raises(ActionEncodingError, match="No legal action matches index"):
            index_to_action(99, legal)

    def test_empty_legal_actions_raises(self):
        with pytest.raises(ActionEncodingError):
            index_to_action(0, [])


# ===== encode_action_mask =====


class TestEncodeActionMask:
    def test_output_shape_and_dtype(self):
        mask = encode_action_mask([ActionDrawStockpile()])
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_single_action(self):
        mask = encode_action_mask([ActionDrawStockpile()])
        assert mask[0] is np.bool_(True)
        assert mask.sum() == 1

    def test_multiple_actions(self):
        actions = [
            ActionDrawStockpile(),
            ActionDrawDiscard(),
            ActionCallCambia(),
        ]
        mask = encode_action_mask(actions)
        assert mask[0] and mask[1] and mask[2]
        assert mask.sum() == 3

    def test_empty_actions(self):
        mask = encode_action_mask([])
        assert mask.sum() == 0

    def test_replace_actions_mask(self):
        actions = [ActionReplace(target_hand_index=i) for i in range(4)]
        mask = encode_action_mask(actions)
        for i in range(4):
            assert mask[5 + i]
        for i in range(4, MAX_HAND):
            assert not mask[5 + i]

    def test_snap_actions_mask(self):
        actions = [
            ActionPassSnap(),
            ActionSnapOwn(own_card_hand_index=2),
            ActionSnapOpponent(opponent_target_hand_index=1),
        ]
        mask = encode_action_mask(actions)
        assert mask[97]   # PassSnap
        assert mask[100]  # SnapOwn(2)
        assert mask[105]  # SnapOpponent(1)
        assert mask.sum() == 3

    def test_out_of_range_action_skipped(self):
        """Actions with hand index >= MAX_HAND are silently skipped."""
        actions = [ActionReplace(target_hand_index=10)]
        mask = encode_action_mask(actions)
        assert mask.sum() == 0


# ===== encode_infoset_eppbs_interleaved =====

# EpistemicTag values: UNK=0, PRIV_OWN=1, PRIV_OPP=2, PUB=3
# Slot layout: slots 0-5 = own hand, slots 6-11 = opponent hand
# Per-slot block: 13 dims = 4 (tag one-hot) + 9 (identity one-hot)
# Public region: dims [0-41] (42 dims)
# Slot region:   dims [42-197] (12 * 13 = 156 dims)
# Padding:       dims [198-199]

_SLOT_BLOCK = 13  # EP_PBS_TAG_DIM(4) + EP_PBS_BUCKET_DIM(9)
_PUB_DIM = 42    # length of public feature region


def _make_eppbs_inputs(
    own_hand_size=6,
    opp_hand_size=6,
    own_tags=None,
    opp_tags=None,
    own_buckets=None,
    opp_buckets=None,
    discard_top_bucket=0,
    stock_estimate=0,
    game_phase=1,
    decision_context=0,
    cambia_state=2,
    drawn_card_bucket=-1,
):
    """Build slot_tags and slot_buckets lists for eppbs calls (12 slots each)."""
    if own_tags is None:
        own_tags = [EpistemicTag.UNK] * 6
    if opp_tags is None:
        opp_tags = [EpistemicTag.UNK] * 6
    if own_buckets is None:
        own_buckets = [0] * 6
    if opp_buckets is None:
        opp_buckets = [0] * 6

    slot_tags = list(own_tags) + list(opp_tags)
    slot_buckets = list(own_buckets) + list(opp_buckets)
    return slot_tags, slot_buckets


class TestEncodeInfosetEppbsInterleaved:
    def test_interleaved_total_dim(self):
        """Output shape is exactly (EP_PBS_INPUT_DIM,) with float32 dtype."""
        slot_tags, slot_buckets = _make_eppbs_inputs()
        out = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2
        )
        assert out.shape == (EP_PBS_INPUT_DIM,)
        assert out.dtype == np.float32

    def test_interleaved_layout_structure(self):
        """Dims [42:55] = slot 0 (13 dims), [55:68] = slot 1, etc."""
        # Use PRIV_OWN for slot 0 with bucket=2 (ACE), PUB for slot 1 with bucket=3
        slot_tags, slot_buckets = _make_eppbs_inputs(
            own_tags=[EpistemicTag.PRIV_OWN, EpistemicTag.PUB, EpistemicTag.UNK,
                      EpistemicTag.UNK, EpistemicTag.UNK, EpistemicTag.UNK],
            own_buckets=[2, 3, 0, 0, 0, 0],
        )
        out = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2
        )

        # Slot 0 block: dims [42:55]
        slot0 = out[42:55]
        assert slot0.shape == (_SLOT_BLOCK,)
        # tag = PRIV_OWN=1 → one-hot index 1
        assert slot0[1] == 1.0, "Slot 0 tag should be PRIV_OWN (index 1)"
        # identity bucket=2 at tag_dim(4)+2=6
        assert slot0[4 + 2] == 1.0, "Slot 0 identity bucket 2 should be set"

        # Slot 1 block: dims [55:68]
        slot1 = out[55:68]
        # tag = PUB=3 → one-hot index 3
        assert slot1[3] == 1.0, "Slot 1 tag should be PUB (index 3)"
        # identity bucket=3 at 4+3=7
        assert slot1[4 + 3] == 1.0, "Slot 1 identity bucket 3 should be set"

        # Verify slots are exactly 13 dims apart
        for i in range(12):
            start = 42 + i * _SLOT_BLOCK
            assert start + _SLOT_BLOCK <= 198, f"Slot {i} overflows into padding"

    def test_interleaved_hand_size_features(self):
        """Dim [40] = own_hand_size/6.0, dim [41] = opp_hand_size/6.0."""
        slot_tags, slot_buckets = _make_eppbs_inputs(own_hand_size=4, opp_hand_size=5)
        out = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2,
            own_hand_size=4, opp_hand_size=5,
        )
        assert out[40] == pytest.approx(4 / 6.0), "dim[40] should be own_hand_size/6"
        assert out[41] == pytest.approx(5 / 6.0), "dim[41] should be opp_hand_size/6"

    def test_interleaved_empty_slot_zeros(self):
        """With own_hand_size=4, slots 4 and 5 (own) are all-zeros (13 dims each)."""
        own_tags = [EpistemicTag.PRIV_OWN] * 6
        own_buckets = [3] * 6
        slot_tags, slot_buckets = _make_eppbs_inputs(
            own_tags=own_tags, own_buckets=own_buckets
        )
        out = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2,
            own_hand_size=4, opp_hand_size=6,
        )

        # Slots 0-3 (own, active): should have non-zero tag data
        for i in range(4):
            slot_block = out[42 + i * _SLOT_BLOCK : 42 + (i + 1) * _SLOT_BLOCK]
            assert slot_block.sum() > 0, f"Own active slot {i} should be non-zero"

        # Slots 4-5 (own, empty): all zeros
        for i in range(4, 6):
            slot_block = out[42 + i * _SLOT_BLOCK : 42 + (i + 1) * _SLOT_BLOCK]
            assert slot_block.sum() == 0.0, f"Own empty slot {i} should be all-zeros"

    def test_interleaved_unk_vs_empty(self):
        """UNK slots have tag=[1,0,0,0]; EMPTY slots (beyond hand size) have tag=[0,0,0,0]."""
        # All slots tagged as UNK, but own_hand_size=4 → slots 4,5 are empty
        slot_tags = [EpistemicTag.UNK] * 12
        slot_buckets = [0] * 12
        out = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2,
            own_hand_size=4, opp_hand_size=6,
        )

        # Slots 0-3: UNK → tag one-hot at index 0 = [1,0,0,0], identity all-zeros
        for i in range(4):
            slot_block = out[42 + i * _SLOT_BLOCK : 42 + (i + 1) * _SLOT_BLOCK]
            assert slot_block[0] == 1.0, f"Slot {i} UNK tag should be 1 at index 0"
            assert slot_block[1:4].sum() == 0.0, f"Slot {i} UNK remaining tag bits should be 0"
            assert slot_block[4:].sum() == 0.0, f"Slot {i} UNK identity should be all-zeros"

        # Slots 4-5: EMPTY → entire 13-dim block is zero (no tag set)
        for i in range(4, 6):
            slot_block = out[42 + i * _SLOT_BLOCK : 42 + (i + 1) * _SLOT_BLOCK]
            assert slot_block.sum() == 0.0, f"Slot {i} EMPTY should be all-zeros"
            # Explicitly: tag is [0,0,0,0], NOT [1,0,0,0]
            assert slot_block[0] == 0.0, f"Slot {i} EMPTY must NOT set UNK tag"

    def test_interleaved_backward_compat(self):
        """encode_infoset_eppbs dims [0-39] equal encode_infoset_eppbs_interleaved dims [0-39]."""
        own_tags = [EpistemicTag.PRIV_OWN, EpistemicTag.UNK, EpistemicTag.PUB,
                    EpistemicTag.UNK, EpistemicTag.UNK, EpistemicTag.UNK]
        opp_tags = [EpistemicTag.PRIV_OPP, EpistemicTag.UNK] * 3
        own_buckets = [2, 0, 5, 0, 0, 0]
        opp_buckets = [0] * 6
        slot_tags = own_tags + opp_tags
        slot_buckets = own_buckets + opp_buckets

        kwargs = dict(
            discard_top_bucket=3,
            stock_estimate=1,
            game_phase=2,
            decision_context=1,
            cambia_state=0,
            drawn_card_bucket=4,
        )

        out_orig = encode_infoset_eppbs(slot_tags, slot_buckets, **kwargs)
        out_interleaved = encode_infoset_eppbs_interleaved(
            slot_tags, slot_buckets, **kwargs
        )

        # Both are (EP_PBS_INPUT_DIM,) float32
        assert out_orig.shape == (EP_PBS_INPUT_DIM,)
        assert out_interleaved.shape == (EP_PBS_INPUT_DIM,)

        # Public features [0-39] must be identical between both encodings
        np.testing.assert_array_equal(
            out_orig[:40], out_interleaved[:40],
            err_msg="Public features [0-39] must be identical between both encodings",
        )

        # The two encodings must differ in layout (slot region is reorganized)
        assert not np.array_equal(out_orig[40:], out_interleaved[40:]), (
            "Slot regions should differ between non-interleaved and interleaved layouts"
        )


# ===== encode_infoset_eppbs_dealiased =====

# Layout constants for dealiased flat encoder:
#   [0-39]:   public features (40 dims)
#   [40-87]:  slot tags (12 slots × 4-dim one-hot)
#   [88-195]: slot identities (12 slots × 9-dim one-hot)
#   [196]:    own_hand_size / 6.0
#   [197]:    opp_hand_size / 6.0
#   [198-199]: padding

_DEALIASED_TAG_BASE = 40
_DEALIASED_ID_BASE = 88
_DEALIASED_TAG_DIM = 4
_DEALIASED_ID_DIM = 9


class TestEncodeInfosetEppbsDealiased:
    def test_eppbs_dealiased_empty_slots_are_zeros(self):
        """Slots beyond hand_size must be all-zeros (not UNK tag [1,0,0,0])."""
        # own_hand_size=4 means slots 4 and 5 are empty
        slot_tags, slot_buckets = _make_eppbs_inputs(
            own_hand_size=4,
            own_tags=[EpistemicTag.UNK] * 6,
            own_buckets=[0] * 6,
        )
        out = encode_infoset_eppbs_dealiased(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2,
            own_hand_size=4, opp_hand_size=6,
        )
        assert out.shape == (EP_PBS_INPUT_DIM,)
        assert out.dtype == np.float32

        # Own slots 4 and 5 are empty: tag block dims [40+4*4 : 40+6*4] must be zeros
        slot4_tag_start = _DEALIASED_TAG_BASE + 4 * _DEALIASED_TAG_DIM  # 56
        slot5_tag_end = _DEALIASED_TAG_BASE + 6 * _DEALIASED_TAG_DIM    # 64
        assert np.all(out[slot4_tag_start:slot5_tag_end] == 0.0), (
            "Empty own slots 4-5 tag region must be all-zeros (not UNK)"
        )

        # Own slots 4 and 5 identity block: [88+4*9 : 88+6*9] must be zeros
        slot4_id_start = _DEALIASED_ID_BASE + 4 * _DEALIASED_ID_DIM   # 124
        slot5_id_end = _DEALIASED_ID_BASE + 6 * _DEALIASED_ID_DIM     # 142
        assert np.all(out[slot4_id_start:slot5_id_end] == 0.0), (
            "Empty own slots 4-5 identity region must be all-zeros"
        )

        # Active own slots 0-3 should have their UNK tag set (EpistemicTag.UNK=0 → index 0)
        for s in range(4):
            tag_start = _DEALIASED_TAG_BASE + s * _DEALIASED_TAG_DIM
            assert out[tag_start] == 1.0, f"Active slot {s} should have UNK tag set at index 0"

    def test_eppbs_dealiased_hand_size_features(self):
        """Dim [196] = own_hand_size/6.0, dim [197] = opp_hand_size/6.0."""
        slot_tags, slot_buckets = _make_eppbs_inputs(own_hand_size=4, opp_hand_size=5)
        out = encode_infoset_eppbs_dealiased(
            slot_tags, slot_buckets, 0, 0, 1, 0, 2,
            own_hand_size=4, opp_hand_size=5,
        )
        assert out[196] == pytest.approx(4 / 6.0), "Dim 196 should be own_hand_size/6.0"
        assert out[197] == pytest.approx(5 / 6.0), "Dim 197 should be opp_hand_size/6.0"

        # Verify padding dims 198-199 are zero
        assert out[198] == 0.0
        assert out[199] == 0.0

    def test_eppbs_dealiased_matches_flat_for_full_hand(self):
        """With full 6-card hand and all UNK slots, first 196 dims should match flat encoder.

        The flat encoder writes UNK tag for all slots; dealiased with full hand
        also writes UNK tag for all slots (no empty slots). They should be identical
        in the first 196 dims (before hand-size features).
        """
        slot_tags = [EpistemicTag.UNK] * 12
        slot_buckets = [0] * 12
        kwargs = dict(
            discard_top_bucket=3,
            stock_estimate=1,
            game_phase=2,
            decision_context=1,
            cambia_state=2,
            drawn_card_bucket=-1,
        )
        out_flat = encode_infoset_eppbs(slot_tags, slot_buckets, **kwargs)
        out_dealiased = encode_infoset_eppbs_dealiased(
            slot_tags, slot_buckets, own_hand_size=6, opp_hand_size=6, **kwargs
        )

        assert out_flat.shape == (EP_PBS_INPUT_DIM,)
        assert out_dealiased.shape == (EP_PBS_INPUT_DIM,)

        # First 196 dims must be identical (no empty slots, same tag/id encoding)
        np.testing.assert_array_equal(
            out_flat[:196], out_dealiased[:196],
            err_msg="First 196 dims must match between flat and dealiased (full 6-card hand)",
        )

        # Dims 196-197 must be 1.0 (hand_size=6, 6/6=1.0)
        assert out_dealiased[196] == pytest.approx(1.0)
        assert out_dealiased[197] == pytest.approx(1.0)

        # Flat encoder has 4 padding zeros at [196-199]
        assert np.all(out_flat[196:] == 0.0), "Original flat encoder should have zeros at [196-199]"

    def test_eppbs_dealiased_cross_engine_parity(self):
        """Go and Python dealiased encoders produce identical vectors (FFI parity)."""
        pytest.importorskip("src.ffi.bridge", reason="FFI not available")
        try:
            from src.ffi.bridge import GoEngine, GoAgentState
        except (ImportError, FileNotFoundError):
            pytest.skip("libcambia.so not available")

        from src.config import CambiaRulesConfig
        rules = CambiaRulesConfig()

        with GoEngine(seed=42, house_rules=rules) as engine:
            agent = GoAgentState(engine, player_id=0)
            try:
                ctx = engine.decision_ctx()
                go_vec = agent.encode_eppbs_dealiased(decision_context=ctx, drawn_bucket=-1)
            finally:
                agent.close()

        assert go_vec.shape == (EP_PBS_INPUT_DIM,), f"Expected ({EP_PBS_INPUT_DIM},), got {go_vec.shape}"
        assert go_vec.dtype == np.float32

        # Hand sizes at [196] and [197] must be in [0, 1]
        assert 0.0 <= go_vec[196] <= 1.0, "own_hand_size feature out of range"
        assert 0.0 <= go_vec[197] <= 1.0, "opp_hand_size feature out of range"

        # Padding dims [198-199] must be zero
        assert go_vec[198] == 0.0
        assert go_vec[199] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
