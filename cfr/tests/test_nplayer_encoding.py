"""
tests/test_nplayer_encoding.py

Tests for N-player encoding functions, action index layout,
and FFI bridge N-player APIs.
"""

import numpy as np
import pytest

from src.constants import (
    N_PLAYER_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS,
    N_PLAYER_MAX_PLAYERS,
    N_PLAYER_MAX_SLOTS,
    N_PLAYER_POWERSET_DIM,
    N_PLAYER_IDENTITY_DIM,
    N_PLAYER_PUBLIC_DIM,
)
from src.encoding import (
    encode_infoset_nplayer,
    nplayer_action_to_index,
    encode_nplayer_action_mask,
)
from src.constants import (
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
)


# ---------------------------------------------------------------------------
# 1. encode_infoset_nplayer dimension test
# ---------------------------------------------------------------------------

class TestNPlayerEncodingDimension:
    def test_output_shape_is_580(self):
        """encode_infoset_nplayer must produce exactly 580-dim vector."""
        knowledge_masks = {}
        slot_buckets = {}
        out = encode_infoset_nplayer(
            knowledge_masks=knowledge_masks,
            slot_buckets=slot_buckets,
            encoding_player=0,
            num_players=2,
            discard_top_bucket=3,
            stock_estimate=0,
            game_phase=1,
            decision_context=0,
            cambia_state=2,
            drawn_card_bucket=-1,
        )
        assert out.shape == (N_PLAYER_INPUT_DIM,), f"Expected (580,), got {out.shape}"
        assert out.dtype == np.float32

    def test_output_shape_nplayer_6(self):
        """Works for N=6 players."""
        out = encode_infoset_nplayer(
            knowledge_masks={(0, 0): {0, 1, 2}},
            slot_buckets={(0, 0): 5},
            encoding_player=0,
            num_players=6,
            discard_top_bucket=0,
            stock_estimate=1,
            game_phase=2,
            decision_context=1,
            cambia_state=2,
        )
        assert out.shape == (580,)

    def test_all_zeros_when_no_knowledge(self):
        """Powerset and identity sections zero when no knowledge."""
        out = encode_infoset_nplayer(
            knowledge_masks={},
            slot_buckets={},
            encoding_player=0,
            num_players=2,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        # Powerset masks should all be zero
        assert np.all(out[:N_PLAYER_POWERSET_DIM] == 0.0)
        # Identity section should all be zero
        assert np.all(out[N_PLAYER_POWERSET_DIM:N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM] == 0.0)


# ---------------------------------------------------------------------------
# 2. N-player action index round-trip (all 452 indices)
# ---------------------------------------------------------------------------

class TestNPlayerActionIndices:
    def test_draw_stockpile(self):
        assert nplayer_action_to_index(ActionDrawStockpile()) == 0

    def test_draw_discard(self):
        assert nplayer_action_to_index(ActionDrawDiscard()) == 1

    def test_call_cambia(self):
        assert nplayer_action_to_index(ActionCallCambia()) == 2

    def test_discard_no_ability(self):
        assert nplayer_action_to_index(ActionDiscard(use_ability=False)) == 3

    def test_discard_ability(self):
        assert nplayer_action_to_index(ActionDiscard(use_ability=True)) == 4

    def test_replace_all_slots(self):
        for slot in range(6):
            idx = nplayer_action_to_index(ActionReplace(target_hand_index=slot))
            assert idx == 5 + slot, f"Replace({slot}) expected {5+slot}, got {idx}"

    def test_peek_own_all_slots(self):
        for slot in range(6):
            idx = nplayer_action_to_index(ActionAbilityPeekOwnSelect(target_hand_index=slot))
            assert idx == 11 + slot

    def test_peek_other_all_combinations(self):
        # 17 + slot*5 + opp_idx, slot in 0-5, opp_idx in 0-4
        for slot in range(6):
            for opp_idx in range(5):
                idx = nplayer_action_to_index(
                    ActionAbilityPeekOtherSelect(target_opponent_hand_index=slot),
                    opp_idx=opp_idx,
                )
                expected = 17 + slot * 5 + opp_idx
                assert idx == expected, f"PeekOther({slot},{opp_idx}): expected {expected}, got {idx}"

    def test_blind_swap_range(self):
        # 47 + own*30 + opp_slot*5 + opp_idx
        idx = nplayer_action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=0, opponent_hand_index=0),
            opp_idx=0,
        )
        assert idx == 47
        idx = nplayer_action_to_index(
            ActionAbilityBlindSwapSelect(own_hand_index=5, opponent_hand_index=5),
            opp_idx=4,
        )
        assert idx == 47 + 5*30 + 5*5 + 4  # 47 + 150 + 25 + 4 = 226

    def test_king_look_range(self):
        idx = nplayer_action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=0),
            opp_idx=0,
        )
        assert idx == 227
        idx = nplayer_action_to_index(
            ActionAbilityKingLookSelect(own_hand_index=5, opponent_hand_index=5),
            opp_idx=4,
        )
        assert idx == 227 + 5*30 + 5*5 + 4  # 227 + 150 + 25 + 4 = 406

    def test_king_swap_decision(self):
        assert nplayer_action_to_index(ActionAbilityKingSwapDecision(perform_swap=False)) == 407
        assert nplayer_action_to_index(ActionAbilityKingSwapDecision(perform_swap=True)) == 408

    def test_pass_snap(self):
        assert nplayer_action_to_index(ActionPassSnap()) == 409

    def test_snap_own_all_slots(self):
        for slot in range(6):
            idx = nplayer_action_to_index(ActionSnapOwn(own_card_hand_index=slot))
            assert idx == 410 + slot

    def test_snap_opponent_all_combinations(self):
        # 416 + slot*5 + opp_idx
        for slot in range(6):
            for opp_idx in range(5):
                idx = nplayer_action_to_index(
                    ActionSnapOpponent(opponent_target_hand_index=slot),
                    opp_idx=opp_idx,
                )
                expected = 416 + slot * 5 + opp_idx
                assert idx == expected

    def test_snap_opponent_move(self):
        for own in range(6):
            idx = nplayer_action_to_index(
                ActionSnapOpponentMove(
                    own_card_to_move_hand_index=own,
                    target_empty_slot_index=0,
                )
            )
            assert idx == 446 + own

    def test_all_indices_within_range(self):
        """Spot-check that all action indices are in [0, 452)."""
        test_actions = [
            (ActionDrawStockpile(), {}),
            (ActionDrawDiscard(), {}),
            (ActionCallCambia(), {}),
            (ActionDiscard(use_ability=False), {}),
            (ActionDiscard(use_ability=True), {}),
            (ActionReplace(target_hand_index=3), {}),
            (ActionAbilityPeekOwnSelect(target_hand_index=2), {}),
            (ActionAbilityPeekOtherSelect(target_opponent_hand_index=1), {"opp_idx": 2}),
            (ActionAbilityBlindSwapSelect(own_hand_index=1, opponent_hand_index=2), {"opp_idx": 1}),
            (ActionAbilityKingLookSelect(own_hand_index=0, opponent_hand_index=3), {"opp_idx": 0}),
            (ActionAbilityKingSwapDecision(perform_swap=True), {}),
            (ActionPassSnap(), {}),
            (ActionSnapOwn(own_card_hand_index=4), {}),
            (ActionSnapOpponent(opponent_target_hand_index=2), {"opp_idx": 3}),
            (ActionSnapOpponentMove(own_card_to_move_hand_index=1, target_empty_slot_index=2), {}),
        ]
        for action, kwargs in test_actions:
            idx = nplayer_action_to_index(action, **kwargs)
            assert 0 <= idx < N_PLAYER_NUM_ACTIONS, (
                f"{type(action).__name__} gave index {idx} out of range [0, {N_PLAYER_NUM_ACTIONS})"
            )


# ---------------------------------------------------------------------------
# 3. Action mask dimension test
# ---------------------------------------------------------------------------

class TestNPlayerActionMask:
    def test_encode_nplayer_action_mask_shape(self):
        uint8_input = np.zeros(N_PLAYER_NUM_ACTIONS, dtype=np.uint8)
        uint8_input[0] = 1
        uint8_input[409] = 1
        mask = encode_nplayer_action_mask(uint8_input)
        assert mask.shape == (N_PLAYER_NUM_ACTIONS,)
        assert mask.dtype == bool
        assert mask[0] is np.bool_(True)
        assert mask[409] is np.bool_(True)
        assert not mask[1]


# ---------------------------------------------------------------------------
# 4. Powerset mask encoding verification
# ---------------------------------------------------------------------------

class TestPowersetMaskEncoding:
    def test_single_knower(self):
        """Slot (0, 0) known only by player 0 → bit 0 set in first 6-bit block."""
        km = {(0, 0): {0}}
        out = encode_infoset_nplayer(
            knowledge_masks=km,
            slot_buckets={(0, 0): 3},
            encoding_player=0,
            num_players=3,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        # Slot (0,0) is global_slot=0, offset=0
        assert out[0] == 1.0   # bit 0 (player 0 knows)
        assert out[1] == 0.0   # bit 1 (player 1 doesn't know)
        assert out[2] == 0.0   # bit 2 (player 2 doesn't know)

    def test_multiple_knowers(self):
        """Slot (1, 2) known by players 0 and 2."""
        global_slot = 1 * 6 + 2  # = 8
        km = {(1, 2): {0, 2}}
        out = encode_infoset_nplayer(
            knowledge_masks=km,
            slot_buckets={(1, 2): 5},
            encoding_player=0,
            num_players=4,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        base = global_slot * N_PLAYER_MAX_PLAYERS
        assert out[base + 0] == 1.0   # player 0 knows
        assert out[base + 1] == 0.0   # player 1 doesn't know
        assert out[base + 2] == 1.0   # player 2 knows
        assert out[base + 3] == 0.0   # player 3 doesn't know

    def test_nonexistent_player_slots_are_zero(self):
        """Slots for players beyond num_players must remain zero."""
        out = encode_infoset_nplayer(
            knowledge_masks={(5, 0): {0}},  # player 5 doesn't exist in 2P game
            slot_buckets={(5, 0): 3},
            encoding_player=0,
            num_players=2,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        # Global slot for (5,0) = 5*6+0 = 30
        base = 30 * N_PLAYER_MAX_PLAYERS
        # All bits should be 0 since player 5 doesn't exist in a 2P game
        assert np.all(out[base:base + N_PLAYER_MAX_PLAYERS] == 0.0)


# ---------------------------------------------------------------------------
# 5. Slot identity encoding with known/unknown cards
# ---------------------------------------------------------------------------

class TestSlotIdentityEncoding:
    def test_known_card_sets_identity(self):
        """When encoding player knows a card, its bucket identity is set."""
        km = {(0, 0): {0}}  # player 0 knows slot (0,0)
        sb = {(0, 0): 5}    # bucket 5 = PEEK_SELF
        out = encode_infoset_nplayer(
            knowledge_masks=km,
            slot_buckets=sb,
            encoding_player=0,
            num_players=2,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        # Identity section starts at offset 216
        # Slot (0,0) = global_slot=0, identity offset = 216 + 0*9 = 216
        identity_base = N_PLAYER_POWERSET_DIM  # 216
        assert out[identity_base + 5] == 1.0  # bucket 5 is set
        for b in range(9):
            if b != 5:
                assert out[identity_base + b] == 0.0

    def test_unknown_card_zeros_identity(self):
        """When encoding player doesn't know a card, identity block is all zero."""
        km = {(0, 0): {1}}  # player 1 knows, but encoding player is 0
        sb = {(0, 0): 3}
        out = encode_infoset_nplayer(
            knowledge_masks=km,
            slot_buckets=sb,
            encoding_player=0,  # player 0 doesn't know
            num_players=2,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        identity_base = N_PLAYER_POWERSET_DIM
        # Slot (0,0) identity block should be all zero
        assert np.all(out[identity_base:identity_base + 9] == 0.0)

    def test_public_knowledge_sets_identity(self):
        """When a card is public knowledge, any encoding player sees the identity."""
        all_players = {0, 1, 2}
        km = {(1, 3): all_players}
        sb = {(1, 3): 7}  # bucket 7 = SWAP_BLIND
        out = encode_infoset_nplayer(
            knowledge_masks=km,
            slot_buckets=sb,
            encoding_player=2,  # player 2 also knows
            num_players=3,
            discard_top_bucket=0,
            stock_estimate=0,
            game_phase=0,
            decision_context=0,
            cambia_state=2,
        )
        global_slot = 1 * 6 + 3  # = 9
        identity_base = N_PLAYER_POWERSET_DIM + global_slot * 9
        assert out[identity_base + 7] == 1.0


# ---------------------------------------------------------------------------
# 6. Public features section
# ---------------------------------------------------------------------------

class TestPublicFeatures:
    def _pub_offset(self):
        return N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM  # 540

    def test_discard_bucket_onehot(self):
        for bucket in range(10):
            out = encode_infoset_nplayer(
                knowledge_masks={}, slot_buckets={},
                encoding_player=0, num_players=2,
                discard_top_bucket=bucket,
                stock_estimate=0, game_phase=0, decision_context=0, cambia_state=2,
            )
            pub = self._pub_offset()
            for b in range(10):
                expected = 1.0 if b == bucket else 0.0
                assert out[pub + b] == expected, f"discard_bucket={bucket} bit {b}"

    def test_drawn_card_none(self):
        out = encode_infoset_nplayer(
            knowledge_masks={}, slot_buckets={},
            encoding_player=0, num_players=2,
            discard_top_bucket=0, stock_estimate=0, game_phase=0,
            decision_context=0, cambia_state=2, drawn_card_bucket=-1,
        )
        pub = self._pub_offset()
        drawn_base = pub + 10 + 4 + 6 + 6 + 3  # after discard(10)+stock(4)+phase(6)+ctx(6)+cambia(3)
        assert out[drawn_base + 10] == 1.0  # NONE index

    def test_drawn_card_known(self):
        out = encode_infoset_nplayer(
            knowledge_masks={}, slot_buckets={},
            encoding_player=0, num_players=2,
            discard_top_bucket=0, stock_estimate=0, game_phase=0,
            decision_context=0, cambia_state=2, drawn_card_bucket=4,
        )
        pub = self._pub_offset()
        drawn_base = pub + 10 + 4 + 6 + 6 + 3
        assert out[drawn_base + 4] == 1.0
        assert out[drawn_base + 10] == 0.0


# ---------------------------------------------------------------------------
# 7. AgentState N-player knowledge mask methods
# ---------------------------------------------------------------------------

class TestAgentStateNPlayerMasks:
    def _make_agent_state(self):
        """Create a minimal AgentState for testing."""
        from unittest.mock import MagicMock
        from src.agent_state import AgentState
        config = MagicMock()
        config.cambia_rules.penaltyDrawCount = 1
        config.cambia_rules.use_jokers = 0
        state = AgentState(
            player_id=0,
            opponent_id=1,
            memory_level=0,
            time_decay_turns=0,
            initial_hand_size=4,
            config=config,
        )
        return state

    def test_default_num_players(self):
        state = self._make_agent_state()
        assert state.num_players == 2
        assert state.knowledge_masks == {}

    def test_nplayer_initialize(self):
        state = self._make_agent_state()
        state.nplayer_initialize(num_players=4, initial_peek_indices=(0, 1))
        assert state.num_players == 4
        # Player 0 knows their own slots 0 and 1
        assert 0 in state.knowledge_masks[(0, 0)]
        assert 0 in state.knowledge_masks[(0, 1)]

    def test_record_peek(self):
        state = self._make_agent_state()
        state.nplayer_initialize(3)
        state.nplayer_record_peek(target_player=1, card_slot=2, peeker=0)
        assert 0 in state.nplayer_get_knowledge_mask(1, 2)

    def test_record_reveal_all_players(self):
        state = self._make_agent_state()
        state.nplayer_initialize(3)
        state.nplayer_record_reveal(target_player=2, card_slot=0)
        mask = state.nplayer_get_knowledge_mask(2, 0)
        assert mask == {0, 1, 2}

    def test_record_swap_clears_knowledge(self):
        state = self._make_agent_state()
        state.nplayer_initialize(3)
        state.nplayer_record_peek(0, 1, 0)  # player 0 knows slot (0,1)
        state.nplayer_record_peek(1, 2, 1)  # player 1 knows slot (1,2)
        state.nplayer_record_swap(0, 1, 1, 2)
        assert state.nplayer_get_knowledge_mask(0, 1) == set()
        assert state.nplayer_get_knowledge_mask(1, 2) == set()

    def test_empty_knowledge_returns_empty_set(self):
        state = self._make_agent_state()
        state.nplayer_initialize(2)
        assert state.nplayer_get_knowledge_mask(0, 0) == set()

    def test_clone_copies_knowledge_masks(self):
        state = self._make_agent_state()
        state.nplayer_initialize(3, initial_peek_indices=(0,))
        cloned = state.clone()
        assert cloned.num_players == 3
        assert cloned.knowledge_masks == state.knowledge_masks
        # Ensure it's a deep copy
        cloned.knowledge_masks[(0, 0)].add(99)
        assert 99 not in state.knowledge_masks.get((0, 0), set())


# ---------------------------------------------------------------------------
# 8. GoEngine N-player construction (FFI — skip if libcambia unavailable)
# ---------------------------------------------------------------------------

class TestGoEngineNPlayer:
    @pytest.fixture(autouse=True)
    def skip_if_no_lib(self):
        try:
            from src.ffi.bridge import _get_lib
            _get_lib()
        except (FileNotFoundError, OSError):
            pytest.skip("libcambia.so not available")

    def test_nplayer_engine_construction(self):
        """GoEngine can be constructed with num_players=4 via house_rules."""
        import warnings
        from src.ffi.bridge import GoEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Use default (no rules) with num_players=4
            # This will use Go defaults but verifies the engine starts
            with GoEngine(seed=42, num_players=4) as game:
                assert game._num_players == 4
                assert game._game_h >= 0

    def test_nplayer_legal_actions_mask_shape(self):
        """nplayer_legal_actions_mask returns (452,) array."""
        import warnings
        from src.ffi.bridge import GoEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with GoEngine(seed=42, num_players=4) as game:
                mask = game.nplayer_legal_actions_mask()
                assert mask.shape == (452,)
                assert mask.dtype == np.uint8

    def test_nplayer_apply_action(self):
        """apply_nplayer_action raises ValueError for out-of-range index."""
        import warnings
        from src.ffi.bridge import GoEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with GoEngine(seed=42, num_players=3) as game:
                with pytest.raises(ValueError):
                    game.apply_nplayer_action(452)


# ---------------------------------------------------------------------------
# 9. GoAgentState N-player encode (FFI — skip if libcambia unavailable)
# ---------------------------------------------------------------------------

class TestGoAgentStateNPlayer:
    @pytest.fixture(autouse=True)
    def skip_if_no_lib(self):
        try:
            from src.ffi.bridge import _get_lib
            _get_lib()
        except (FileNotFoundError, OSError):
            pytest.skip("libcambia.so not available")

    def test_new_nplayer_creates_agent(self):
        """GoAgentState.new_nplayer() creates agent for N-player game."""
        import warnings
        from src.ffi.bridge import GoEngine, GoAgentState
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with GoEngine(seed=123, num_players=4) as game:
                with GoAgentState.new_nplayer(game, player_id=0, num_players=4) as agent:
                    assert agent._agent_h >= 0

    def test_encode_nplayer_returns_580_dims(self):
        """encode_nplayer returns (580,) float32 array."""
        import warnings
        from src.ffi.bridge import GoEngine, GoAgentState
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with GoEngine(seed=456, num_players=4) as game:
                with GoAgentState.new_nplayer(game, player_id=0, num_players=4) as agent:
                    enc = agent.encode_nplayer(decision_context=0, drawn_bucket=-1)
                    assert enc.shape == (580,)
                    assert enc.dtype == np.float32

    def test_nplayer_action_mask_shape(self):
        """nplayer_action_mask returns (452,) uint8 array."""
        import warnings
        from src.ffi.bridge import GoEngine, GoAgentState
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with GoEngine(seed=789, num_players=4) as game:
                with GoAgentState.new_nplayer(game, player_id=0, num_players=4) as agent:
                    mask = agent.nplayer_action_mask(game)
                    assert mask.shape == (452,)
                    assert mask.dtype == np.uint8
