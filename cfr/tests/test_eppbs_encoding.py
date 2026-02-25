"""Tests for EP-PBS encoding."""
import numpy as np
import pytest
from src.encoding import (
    EP_PBS_INPUT_DIM,
    EpistemicTag,
    encode_infoset_eppbs,
    bucket_saliency,
    EP_PBS_MAX_ACTIVE_MASK,
)


class TestEPPBSEncoding:
    def test_dimension(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        assert out.shape == (200,)
        assert out.dtype == np.float32

    def test_public_features(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 3, 1, 2, 0, 2, drawn_card_bucket=5)
        # discard bucket 3 → out[3] = 1
        assert out[3] == 1.0
        # stock estimate 1 → out[11] = 1
        assert out[11] == 1.0
        # phase 2 → out[16] = 1
        assert out[16] == 1.0
        # context 0 → out[20] = 1
        assert out[20] == 1.0
        # cambia NONE=2 → out[28] = 1
        assert out[28] == 1.0
        # drawn bucket 5 → out[34] = 1
        assert out[34] == 1.0

    def test_slot_tags_one_hot(self):
        tags = (
            [EpistemicTag.UNK, EpistemicTag.PRIV_OWN, EpistemicTag.PRIV_OPP, EpistemicTag.PUB]
            + [EpistemicTag.UNK] * 8
        )
        buckets = [0, 2, 0, 8] + [0] * 8
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Slot 0 tag (offset 40): UNK=0
        assert out[40] == 1.0
        # Slot 1 tag (offset 44): PRIV_OWN=1
        assert out[45] == 1.0
        # Slot 2 tag (offset 48): PRIV_OPP=2
        assert out[50] == 1.0
        # Slot 3 tag (offset 52): PUB=3
        assert out[55] == 1.0

    def test_slot_identity_zeroing(self):
        """TagPrivOpp and TagUnk slots should have zero bucket encoding."""
        tags = (
            [EpistemicTag.PRIV_OWN, EpistemicTag.PRIV_OPP, EpistemicTag.UNK, EpistemicTag.PUB]
            + [EpistemicTag.UNK] * 8
        )
        buckets = [2, 5, 0, 8] + [0] * 8
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Slot 0 (PRIV_OWN, bucket 2): identity at offset 88+2
        assert out[88 + 2] == 1.0
        # Slot 1 (PRIV_OPP): identity should be all zeros
        assert np.all(out[88 + 9 : 88 + 18] == 0.0)
        # Slot 2 (UNK): identity should be all zeros
        assert np.all(out[88 + 18 : 88 + 27] == 0.0)
        # Slot 3 (PUB, bucket 8): identity at offset 88+27+8
        assert out[88 + 27 + 8] == 1.0

    def test_saliency_values(self):
        # BucketMidNum (midpoint 5.5) → saliency 1.0
        assert abs(bucket_saliency(4) - 1.0) < 1e-6
        # BucketHighKing (midpoint 13.0) → saliency 8.5
        assert abs(bucket_saliency(8) - 8.5) < 1e-6
        # BucketNegKing (midpoint -1.0) → saliency 5.5
        assert abs(bucket_saliency(1) - 5.5) < 1e-6

    def test_no_drawn_card(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0, drawn_card_bucket=-1)
        # NONE → index 10 in drawn card section (offset 29+10=39)
        assert out[39] == 1.0

    def test_all_zeros_except_one_hot(self):
        """Sum of all one-hot bits in public section equals the number of features encoded."""
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        # Public section: 6 one-hot groups, each should have exactly 1 bit set
        # discard[0:10], stock[10:14], phase[14:20], ctx[20:26], cambia[26:29], drawn[29:40]
        assert out[0:10].sum() == 1.0
        assert out[10:14].sum() == 1.0
        assert out[14:20].sum() == 1.0
        assert out[20:26].sum() == 1.0
        assert out[26:29].sum() == 1.0
        assert out[29:40].sum() == 1.0

    def test_out_of_range_values_ignored(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        # discard_top_bucket=10 → out of range [0-9], no bit set
        out = encode_infoset_eppbs(tags, buckets, 10, 0, 0, 0, 0)
        assert out[0:10].sum() == 0.0

    def test_padding_zeros(self):
        tags = [EpistemicTag.UNK] * 12
        buckets = [0] * 12
        out = encode_infoset_eppbs(tags, buckets, 0, 0, 0, 0, 0)
        assert np.all(out[196:200] == 0.0)

    @pytest.mark.skip(reason="Requires rebuilt libcambia with EP-PBS exports")
    def test_cross_engine_parity(self):
        """Cross-engine parity test — enable after libcambia rebuilt."""
        pass


class TestEPPBSAgentStateTracking:
    """Tests for AgentState EP-PBS epistemic tag tracking."""

    @pytest.fixture
    def agent_state(self):
        """Minimal AgentState for EP-PBS testing (no game engine required)."""
        from unittest.mock import MagicMock
        from src.agent_state import AgentState, KnownCardInfo
        from src.constants import CardBucket, EpistemicTag

        cfg = MagicMock()
        cfg.cambia_rules.penaltyDrawCount = 2
        cfg.cambia_rules.use_jokers = 2

        state = AgentState(
            player_id=0,
            opponent_id=1,
            memory_level=0,
            time_decay_turns=0,
            initial_hand_size=4,
            config=cfg,
        )
        # Manually set up minimal hand state (bypass full initialize)
        from src.constants import GamePhase, StockpileEstimate
        state.own_hand = {
            i: KnownCardInfo(bucket=CardBucket.UNKNOWN, last_seen_turn=0)
            for i in range(4)
        }
        state.opponent_belief = {i: CardBucket.UNKNOWN for i in range(4)}
        state.opponent_last_seen_turn = {}
        state.opponent_card_count = 4
        state.game_phase = GamePhase.EARLY
        state.stockpile_estimate = StockpileEstimate.HIGH
        # EP-PBS is initialized by __post_init__
        return state

    def test_eppbs_fields_initialized(self, agent_state):
        """AgentState has EP-PBS fields after construction."""
        from src.constants import EpistemicTag
        assert len(agent_state.slot_tags) == 12
        assert all(t == EpistemicTag.UNK for t in agent_state.slot_tags)
        assert len(agent_state.slot_buckets) == 12
        assert len(agent_state.own_active_mask) == 0
        assert len(agent_state.opp_active_mask) == 0

    def test_eppbs_tag_transitions_unk_to_priv_own(self, agent_state):
        """UNK → PRIV_OWN when we learn a slot."""
        from src.constants import EpistemicTag
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)  # LOW_NUM bucket
        assert agent_state.slot_tags[0] == EpistemicTag.PRIV_OWN
        assert agent_state.slot_buckets[0] == 3
        assert 0 in agent_state.own_active_mask

    def test_eppbs_tag_transitions_priv_opp_to_pub(self, agent_state):
        """PRIV_OPP → PUB when we also learn a slot the opponent already knew."""
        from src.constants import EpistemicTag
        # First opponent learns slot 0
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OPP)
        assert agent_state.slot_tags[0] == EpistemicTag.PRIV_OPP
        assert 0 in agent_state.opp_active_mask
        # Now we learn it too → PUB
        agent_state._eppbs_set_tag(0, EpistemicTag.PUB, 5)
        assert agent_state.slot_tags[0] == EpistemicTag.PUB
        assert 0 not in agent_state.opp_active_mask  # removed from opp mask
        assert 0 not in agent_state.own_active_mask  # PUB not in own mask

    def test_eppbs_tag_transitions_priv_own_to_pub(self, agent_state):
        """PRIV_OWN → PUB when opponent learns a slot we already knew."""
        from src.constants import EpistemicTag
        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 4)
        assert 2 in agent_state.own_active_mask
        # Opponent learns it → PUB
        agent_state._eppbs_set_tag(2, EpistemicTag.PUB, 4)
        assert agent_state.slot_tags[2] == EpistemicTag.PUB
        assert 2 not in agent_state.own_active_mask

    def test_eppbs_saliency_eviction(self, agent_state):
        """Peeking 4 own cards evicts the lowest-saliency one."""
        from src.constants import EpistemicTag
        # Peek 3 low-saliency cards first: MID_NUM(4,sal=1.0), MID_NUM(4), LOW_NUM(3,sal=1.5)
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 4)   # MID_NUM, sal=1.0
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 3)   # LOW_NUM, sal=1.5
        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 5)   # PEEK_SELF, sal=3.0
        assert agent_state.own_active_mask == [0, 1, 2]
        # Now peek a high-saliency card: HIGH_KING(8,sal=8.5)
        agent_state._eppbs_set_tag(3, EpistemicTag.PRIV_OWN, 8)   # HIGH_KING, sal=8.5
        # Slot 0 (sal=1.0) should have been evicted
        assert 0 not in agent_state.own_active_mask
        assert agent_state.slot_tags[0] == EpistemicTag.UNK
        assert 3 in agent_state.own_active_mask
        assert len(agent_state.own_active_mask) == 3

    def test_eppbs_saliency_no_eviction_if_new_lower(self, agent_state):
        """New card with lower saliency than all existing is NOT added."""
        from src.constants import EpistemicTag
        # Fill mask with high-saliency cards
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 8)   # HIGH_KING, sal=8.5
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 1)   # NEG_KING, sal=5.5
        agent_state._eppbs_set_tag(2, EpistemicTag.PRIV_OWN, 6)   # PEEK_OTHER, sal=5.0
        # Now add very low saliency: MID_NUM(4, sal=1.0) — lower than minimum (5.0)
        agent_state._eppbs_set_tag(3, EpistemicTag.PRIV_OWN, 4)   # MID_NUM, sal=1.0
        # Slot 3 should NOT be in mask (lower saliency than slot 2's 5.0)
        assert 3 not in agent_state.own_active_mask
        # Slot 2 (min of {8.5,5.5,5.0}=5.0) should remain since new sal=1.0 < 5.0
        assert 2 in agent_state.own_active_mask

    def test_eppbs_fifo_eviction(self, agent_state):
        """Opponent peeking 4 slots evicts the oldest (FIFO)."""
        from src.constants import EpistemicTag
        # Opponent peeks slots 6,7,8 (3 opp slots)
        agent_state._eppbs_set_tag(6, EpistemicTag.PRIV_OPP)
        agent_state._eppbs_set_tag(7, EpistemicTag.PRIV_OPP)
        agent_state._eppbs_set_tag(8, EpistemicTag.PRIV_OPP)
        assert agent_state.opp_active_mask == [6, 7, 8]
        # Opponent peeks a 4th slot → FIFO evict slot 6
        agent_state._eppbs_set_tag(9, EpistemicTag.PRIV_OPP)
        assert 6 not in agent_state.opp_active_mask
        assert agent_state.slot_tags[6] == EpistemicTag.UNK
        assert 9 in agent_state.opp_active_mask
        assert agent_state.opp_active_mask == [7, 8, 9]

    def test_eppbs_clone_copies_state(self, agent_state):
        """clone() preserves EP-PBS state."""
        from src.constants import EpistemicTag
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)
        agent_state._eppbs_set_tag(6, EpistemicTag.PRIV_OPP)
        cloned = agent_state.clone()
        assert cloned.slot_tags[0] == EpistemicTag.PRIV_OWN
        assert cloned.slot_buckets[0] == 3
        assert cloned.slot_tags[6] == EpistemicTag.PRIV_OPP
        assert 0 in cloned.own_active_mask
        assert 6 in cloned.opp_active_mask
        # Mutations to original don't affect clone
        agent_state._eppbs_set_tag(1, EpistemicTag.PRIV_OWN, 5)
        assert cloned.slot_tags[1] == EpistemicTag.UNK

    def test_eppbs_forget_slot(self, agent_state):
        """Setting tag to UNK removes from active masks."""
        from src.constants import EpistemicTag
        agent_state._eppbs_set_tag(0, EpistemicTag.PRIV_OWN, 3)
        assert 0 in agent_state.own_active_mask
        agent_state._eppbs_set_tag(0, EpistemicTag.UNK)
        assert 0 not in agent_state.own_active_mask
        assert agent_state.slot_tags[0] == EpistemicTag.UNK
