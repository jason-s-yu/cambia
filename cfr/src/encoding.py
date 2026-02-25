"""
src/encoding.py

Converts AgentState + legal actions into fixed-size numpy tensors for Deep CFR.

Feature vector layout (222 dimensions total):
  - Own hand:           6 slots x 15-dim one-hot = 90
  - Opponent beliefs:   6 slots x 15-dim one-hot = 90
  - Own card count:     1 (normalized)
  - Opponent card count: 1 (normalized)
  - Drawn card bucket:  11-dim one-hot (10 buckets + NONE)
  - Discard top bucket: 10-dim one-hot
  - Stockpile estimate: 4-dim one-hot
  - Game phase:         6-dim one-hot
  - Decision context:   6-dim one-hot
  - Cambia caller:      3-dim one-hot (SELF/OPPONENT/NONE)

Action space: 146 fixed indices mapping all GameAction types.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .agent_state import AgentState, KnownCardInfo
from .constants import (
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
    EpistemicTag,
    GameAction,
    GamePhase,
    StockpileEstimate,
    EP_PBS_INPUT_DIM,
    EP_PBS_MAX_SLOTS,
    EP_PBS_TAG_DIM,
    EP_PBS_BUCKET_DIM,
    EP_PBS_PUBLIC_DIM,
    EP_PBS_MAX_ACTIVE_MASK,
    N_PLAYER_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS,
    N_PLAYER_MAX_PLAYERS,
    N_PLAYER_MAX_SLOTS,
    N_PLAYER_POWERSET_DIM,
    N_PLAYER_IDENTITY_DIM,
    N_PLAYER_PUBLIC_DIM,
    bucket_saliency,
)
from src.cfr.exceptions import InfosetEncodingError, ActionEncodingError

# --- Constants ---
MAX_HAND = 6
SLOT_ENCODING_DIM = 15  # 10 CardBucket + 3 DecayCategory + UNKNOWN + EMPTY
INPUT_DIM = 222
NUM_ACTIONS = 146

# EP-PBS constants imported from constants.py above

# Slot encoding indices within each 15-dim one-hot:
#   0-9:  CardBucket values (ZERO=0, NEG_KING=1, ACE=2, LOW_NUM=3, MID_NUM=4,
#          PEEK_SELF=5, PEEK_OTHER=6, SWAP_BLIND=7, HIGH_KING=8, UNKNOWN(bucket)=9)
#   10-12: DecayCategory values (LIKELY_LOW=10, LIKELY_MID=11, LIKELY_HIGH=12)
#   13:   UNKNOWN (used for CardBucket.UNKNOWN and DecayCategory.UNKNOWN)
#   14:   EMPTY (slot does not exist)

# Map CardBucket.value -> slot encoding index
_BUCKET_TO_SLOT_IDX = {
    CardBucket.ZERO.value: 0,
    CardBucket.NEG_KING.value: 1,
    CardBucket.ACE.value: 2,
    CardBucket.LOW_NUM.value: 3,
    CardBucket.MID_NUM.value: 4,
    CardBucket.PEEK_SELF.value: 5,
    CardBucket.PEEK_OTHER.value: 6,
    CardBucket.SWAP_BLIND.value: 7,
    CardBucket.HIGH_KING.value: 8,
    CardBucket.UNKNOWN.value: 13,  # UNKNOWN -> index 13
}

# Map DecayCategory.value -> slot encoding index
_DECAY_TO_SLOT_IDX = {
    DecayCategory.LIKELY_LOW.value: 10,
    DecayCategory.LIKELY_MID.value: 11,
    DecayCategory.LIKELY_HIGH.value: 12,
    DecayCategory.UNKNOWN.value: 13,  # DecayCategory.UNKNOWN shares index 13
}

EMPTY_SLOT_IDX = 14

# --- EP-PBS Constants ---
# EP_PBS_INPUT_DIM, EP_PBS_MAX_SLOTS, etc. imported from constants.py above.
# Layout:
#   [0-39]    public features (40)
#   [40-87]   slot tags: 12 slots x 4-dim one-hot (48)
#   [88-195]  slot identities: 12 slots x 9-dim one-hot (108)
#   [196-199] padding (4)
# Slot layout: slots 0-5 = own hand, slots 6-11 = opponent hand.

# 9-dim identity encoding: CardBucket value → identity index (0-8)
_BUCKET_TO_IDENTITY_IDX = {
    CardBucket.ZERO.value: 0,
    CardBucket.NEG_KING.value: 1,
    CardBucket.ACE.value: 2,
    CardBucket.LOW_NUM.value: 3,
    CardBucket.MID_NUM.value: 4,
    CardBucket.PEEK_SELF.value: 5,
    CardBucket.PEEK_OTHER.value: 6,
    CardBucket.SWAP_BLIND.value: 7,
    CardBucket.HIGH_KING.value: 8,
}

# eppbs_slot_saliency is an alias for bucket_saliency (imported from constants.py)
eppbs_slot_saliency = bucket_saliency

# Map StockpileEstimate.value -> one-hot index (4 values)
_STOCKPILE_TO_IDX = {
    StockpileEstimate.HIGH.value: 0,
    StockpileEstimate.MEDIUM.value: 1,
    StockpileEstimate.LOW.value: 2,
    StockpileEstimate.EMPTY.value: 3,
}

# Map GamePhase.value -> one-hot index (6 values)
_GAME_PHASE_TO_IDX = {
    GamePhase.START.value: 0,
    GamePhase.EARLY.value: 1,
    GamePhase.MID.value: 2,
    GamePhase.LATE.value: 3,
    GamePhase.CAMBIA_CALLED.value: 4,
    GamePhase.TERMINAL.value: 5,
}

# Map DecisionContext.value -> one-hot index (6 values)
_DECISION_CONTEXT_TO_IDX = {
    DecisionContext.START_TURN.value: 0,
    DecisionContext.POST_DRAW.value: 1,
    DecisionContext.SNAP_DECISION.value: 2,
    DecisionContext.ABILITY_SELECT.value: 3,
    DecisionContext.SNAP_MOVE.value: 4,
    DecisionContext.TERMINAL.value: 5,
}

# Cambia caller one-hot: SELF=0, OPPONENT=1, NONE=2
_CAMBIA_CALLER_SELF = 0
_CAMBIA_CALLER_OPPONENT = 1
_CAMBIA_CALLER_NONE = 2

# CardBucket values for drawn card one-hot (10 buckets + NONE)
_DRAWN_CARD_BUCKET_VALUES = [
    CardBucket.ZERO.value,
    CardBucket.NEG_KING.value,
    CardBucket.ACE.value,
    CardBucket.LOW_NUM.value,
    CardBucket.MID_NUM.value,
    CardBucket.PEEK_SELF.value,
    CardBucket.PEEK_OTHER.value,
    CardBucket.SWAP_BLIND.value,
    CardBucket.HIGH_KING.value,
    CardBucket.UNKNOWN.value,
]
_DRAWN_CARD_NONE_IDX = 10  # Index for "no drawn card"

# Discard top bucket one-hot uses the 10 CardBucket values (0-9)
_DISCARD_BUCKET_TO_IDX = {
    CardBucket.ZERO.value: 0,
    CardBucket.NEG_KING.value: 1,
    CardBucket.ACE.value: 2,
    CardBucket.LOW_NUM.value: 3,
    CardBucket.MID_NUM.value: 4,
    CardBucket.PEEK_SELF.value: 5,
    CardBucket.PEEK_OTHER.value: 6,
    CardBucket.SWAP_BLIND.value: 7,
    CardBucket.HIGH_KING.value: 8,
    CardBucket.UNKNOWN.value: 9,
}


# --- Action Index Mapping ---
# Fixed action index layout (146 total):
#   0: DrawStockpile
#   1: DrawDiscard
#   2: CallCambia
#   3: Discard(use_ability=False)
#   4: Discard(use_ability=True)
#   5-10: Replace(0-5)
#   11-16: PeekOwn(0-5)
#   17-22: PeekOther(0-5)
#   23-58: BlindSwap(own*6 + opp) for own in 0-5, opp in 0-5
#   59-94: KingLook(own*6 + opp)
#   95-96: KingSwapDecision(False=95, True=96)
#   97: PassSnap
#   98-103: SnapOwn(0-5)
#   104-109: SnapOpponent(0-5)
#   110-145: SnapOpponentMove(own*6 + slot) for own in 0-5, slot in 0-5

_IDX_DRAW_STOCKPILE = 0
_IDX_DRAW_DISCARD = 1
_IDX_CALL_CAMBIA = 2
_IDX_DISCARD_NO_ABILITY = 3
_IDX_DISCARD_ABILITY = 4
_IDX_REPLACE_BASE = 5       # 5-10
_IDX_PEEK_OWN_BASE = 11     # 11-16
_IDX_PEEK_OTHER_BASE = 17   # 17-22
_IDX_BLIND_SWAP_BASE = 23   # 23-58 (6x6=36)
_IDX_KING_LOOK_BASE = 59    # 59-94 (6x6=36)
_IDX_KING_SWAP_FALSE = 95
_IDX_KING_SWAP_TRUE = 96
_IDX_PASS_SNAP = 97
_IDX_SNAP_OWN_BASE = 98     # 98-103
_IDX_SNAP_OPP_BASE = 104    # 104-109
_IDX_SNAP_OPP_MOVE_BASE = 110  # 110-145 (6x6=36)


def _encode_slot(value: Union[int, CardBucket, DecayCategory, None]) -> int:
    """Return the one-hot index (0-14) for a hand/belief slot value."""
    if value is None:
        return EMPTY_SLOT_IDX

    # Get the raw int value
    if isinstance(value, (CardBucket, DecayCategory)):
        raw = value.value
    else:
        raw = int(value)

    # Check CardBucket mapping first
    if raw in _BUCKET_TO_SLOT_IDX:
        return _BUCKET_TO_SLOT_IDX[raw]

    # Check DecayCategory mapping
    if raw in _DECAY_TO_SLOT_IDX:
        return _DECAY_TO_SLOT_IDX[raw]

    # Fallback to UNKNOWN
    return 13


def encode_infoset(
    agent_state: AgentState,
    decision_context: DecisionContext,
    drawn_card_bucket: Optional[CardBucket] = None,
) -> np.ndarray:
    """
    Encode an agent's information set into a fixed-size feature vector.

    Args:
        agent_state: The agent's current belief state.
        decision_context: The current decision context.
        drawn_card_bucket: The bucket of the drawn card (for POST_DRAW), or None.

    Returns:
        np.ndarray of shape (222,) with float32 dtype.

    Raises:
        InfosetEncodingError: If feature encoding fails.
    """
    if agent_state is None:
        raise InfosetEncodingError("Agent state cannot be None")
    if decision_context is None:
        raise InfosetEncodingError("Decision context cannot be None")
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    offset = 0

    # --- Own hand: 6 slots x 15-dim one-hot = 90 ---
    own_hand_size = len(agent_state.own_hand)
    for slot in range(MAX_HAND):
        if slot < own_hand_size and slot in agent_state.own_hand:
            info = agent_state.own_hand[slot]
            if isinstance(info, KnownCardInfo):
                idx = _encode_slot(info.bucket)
            else:
                idx = _encode_slot(info)
        else:
            idx = EMPTY_SLOT_IDX
        features[offset + idx] = 1.0
        offset += SLOT_ENCODING_DIM
    # offset = 90

    # --- Opponent beliefs: 6 slots x 15-dim one-hot = 90 ---
    opp_count = agent_state.opponent_card_count
    for slot in range(MAX_HAND):
        if slot < opp_count and slot in agent_state.opponent_belief:
            belief = agent_state.opponent_belief[slot]
            idx = _encode_slot(belief)
        else:
            idx = EMPTY_SLOT_IDX
        features[offset + idx] = 1.0
        offset += SLOT_ENCODING_DIM
    # offset = 180

    # --- Own card count: normalized scalar = 1 ---
    features[offset] = min(own_hand_size, MAX_HAND) / MAX_HAND
    offset += 1
    # offset = 181

    # --- Opponent card count: normalized scalar = 1 ---
    features[offset] = min(opp_count, MAX_HAND) / MAX_HAND
    offset += 1
    # offset = 182

    # --- Drawn card bucket: 11-dim one-hot = 11 ---
    if drawn_card_bucket is not None and drawn_card_bucket != CardBucket.UNKNOWN:
        # Map bucket value to drawn card one-hot index
        try:
            drawn_idx = _DRAWN_CARD_BUCKET_VALUES.index(drawn_card_bucket.value)
        except ValueError:
            drawn_idx = _DRAWN_CARD_NONE_IDX
    elif drawn_card_bucket == CardBucket.UNKNOWN:
        # UNKNOWN bucket gets index 9 (the UNKNOWN slot in the drawn card encoding)
        drawn_idx = 9
    else:
        drawn_idx = _DRAWN_CARD_NONE_IDX
    features[offset + drawn_idx] = 1.0
    offset += 11
    # offset = 193

    # --- Discard top bucket: 10-dim one-hot = 10 ---
    discard_val = agent_state.known_discard_top_bucket.value
    discard_idx = _DISCARD_BUCKET_TO_IDX.get(discard_val, 9)  # default UNKNOWN
    features[offset + discard_idx] = 1.0
    offset += 10
    # offset = 203

    # --- Stockpile estimate: 4-dim one-hot = 4 ---
    stock_idx = _STOCKPILE_TO_IDX.get(agent_state.stockpile_estimate.value, 0)
    features[offset + stock_idx] = 1.0
    offset += 4
    # offset = 207

    # --- Game phase: 6-dim one-hot = 6 ---
    phase_idx = _GAME_PHASE_TO_IDX.get(agent_state.game_phase.value, 0)
    features[offset + phase_idx] = 1.0
    offset += 6
    # offset = 213

    # --- Decision context: 6-dim one-hot = 6 ---
    ctx_idx = _DECISION_CONTEXT_TO_IDX.get(decision_context.value, 0)
    features[offset + ctx_idx] = 1.0
    offset += 6
    # offset = 219

    # --- Cambia caller: 3-dim one-hot = 3 ---
    if agent_state.cambia_caller is None:
        cambia_idx = _CAMBIA_CALLER_NONE
    elif agent_state.cambia_caller == agent_state.player_id:
        cambia_idx = _CAMBIA_CALLER_SELF
    else:
        cambia_idx = _CAMBIA_CALLER_OPPONENT
    features[offset + cambia_idx] = 1.0
    offset += 3
    # offset = 222

    return features


def encode_infoset_eppbs(
    slot_tags: list,          # list of 12 EpistemicTag values
    slot_buckets: list,       # list of 12 bucket values (0 if unknown)
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    decision_context: int,
    cambia_state: int,
    drawn_card_bucket: int = -1,
) -> np.ndarray:
    """EP-PBS encoding for 2-player games. Returns ndarray of shape (200,)."""
    out = np.zeros(EP_PBS_INPUT_DIM, dtype=np.float32)
    offset = 0

    # Public features (40 dims)
    # [0-9]: discard top bucket (10-dim one-hot)
    if 0 <= discard_top_bucket <= 9:
        out[offset + discard_top_bucket] = 1.0
    offset += 10

    # [10-13]: stockpile estimate (4-dim one-hot)
    if 0 <= stock_estimate <= 3:
        out[offset + stock_estimate] = 1.0
    offset += 4

    # [14-19]: game phase (6-dim one-hot)
    if 0 <= game_phase <= 5:
        out[offset + game_phase] = 1.0
    offset += 6

    # [20-25]: decision context (6-dim one-hot)
    if 0 <= decision_context <= 5:
        out[offset + decision_context] = 1.0
    offset += 6

    # [26-28]: cambia state (3-dim one-hot)
    # Python order: SELF=0, OPPONENT=1, NONE=2
    if 0 <= cambia_state <= 2:
        out[offset + cambia_state] = 1.0
    offset += 3

    # [29-39]: drawn card bucket (11-dim one-hot)
    if drawn_card_bucket < 0:
        out[offset + 10] = 1.0  # NONE
    elif 0 <= drawn_card_bucket <= 9:
        out[offset + drawn_card_bucket] = 1.0
    offset += 11
    # offset = 40

    # Slot tags (48 dims): 12 slots × 4-dim one-hot
    for i in range(EP_PBS_MAX_SLOTS):
        tag = slot_tags[i] if i < len(slot_tags) else EpistemicTag.UNK
        if 0 <= tag <= 3:
            out[offset + tag] = 1.0
        offset += EP_PBS_TAG_DIM
    # offset = 88

    # Slot identities (108 dims): 12 slots × 9-dim bucket
    for i in range(EP_PBS_MAX_SLOTS):
        tag = slot_tags[i] if i < len(slot_tags) else EpistemicTag.UNK
        if tag in (EpistemicTag.PRIV_OWN, EpistemicTag.PUB):
            bucket = slot_buckets[i] if i < len(slot_buckets) else 0
            if 0 <= bucket <= 8:
                out[offset + bucket] = 1.0
        offset += EP_PBS_BUCKET_DIM
    # offset = 196

    # 4 dims padding (already zero)
    return out


def action_to_index(action: GameAction) -> int:
    """
    Map a GameAction to its fixed index in the action space [0, 146).

    Args:
        action: A GameAction NamedTuple.

    Returns:
        Integer index in [0, NUM_ACTIONS).

    Raises:
        ActionEncodingError: If action type is unrecognized or index is out of range.
    """
    if action is None:
        raise ActionEncodingError("Action cannot be None")
    if isinstance(action, ActionDrawStockpile):
        return _IDX_DRAW_STOCKPILE

    if isinstance(action, ActionDrawDiscard):
        return _IDX_DRAW_DISCARD

    if isinstance(action, ActionCallCambia):
        return _IDX_CALL_CAMBIA

    if isinstance(action, ActionDiscard):
        return _IDX_DISCARD_ABILITY if action.use_ability else _IDX_DISCARD_NO_ABILITY

    if isinstance(action, ActionReplace):
        idx = action.target_hand_index
        if 0 <= idx < MAX_HAND:
            return _IDX_REPLACE_BASE + idx
        raise ActionEncodingError(f"Replace index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionAbilityPeekOwnSelect):
        idx = action.target_hand_index
        if 0 <= idx < MAX_HAND:
            return _IDX_PEEK_OWN_BASE + idx
        raise ActionEncodingError(f"PeekOwn index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionAbilityPeekOtherSelect):
        idx = action.target_opponent_hand_index
        if 0 <= idx < MAX_HAND:
            return _IDX_PEEK_OTHER_BASE + idx
        raise ActionEncodingError(f"PeekOther index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionAbilityBlindSwapSelect):
        own = action.own_hand_index
        opp = action.opponent_hand_index
        if 0 <= own < MAX_HAND and 0 <= opp < MAX_HAND:
            return _IDX_BLIND_SWAP_BASE + own * MAX_HAND + opp
        raise ActionEncodingError(
            f"BlindSwap indices ({own}, {opp}) out of range [0, {MAX_HAND})"
        )

    if isinstance(action, ActionAbilityKingLookSelect):
        own = action.own_hand_index
        opp = action.opponent_hand_index
        if 0 <= own < MAX_HAND and 0 <= opp < MAX_HAND:
            return _IDX_KING_LOOK_BASE + own * MAX_HAND + opp
        raise ActionEncodingError(
            f"KingLook indices ({own}, {opp}) out of range [0, {MAX_HAND})"
        )

    if isinstance(action, ActionAbilityKingSwapDecision):
        return _IDX_KING_SWAP_TRUE if action.perform_swap else _IDX_KING_SWAP_FALSE

    if isinstance(action, ActionPassSnap):
        return _IDX_PASS_SNAP

    if isinstance(action, ActionSnapOwn):
        idx = action.own_card_hand_index
        if 0 <= idx < MAX_HAND:
            return _IDX_SNAP_OWN_BASE + idx
        raise ActionEncodingError(f"SnapOwn index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionSnapOpponent):
        idx = action.opponent_target_hand_index
        if 0 <= idx < MAX_HAND:
            return _IDX_SNAP_OPP_BASE + idx
        raise ActionEncodingError(f"SnapOpponent index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionSnapOpponentMove):
        own = action.own_card_to_move_hand_index
        slot = action.target_empty_slot_index
        if 0 <= own < MAX_HAND and 0 <= slot < MAX_HAND:
            return _IDX_SNAP_OPP_MOVE_BASE + own * MAX_HAND + slot
        raise ActionEncodingError(
            f"SnapOpponentMove indices ({own}, {slot}) out of range [0, {MAX_HAND})"
        )

    raise ActionEncodingError(f"Unrecognized action type: {type(action).__name__}")


def index_to_action(index: int, legal_actions: List[GameAction]) -> GameAction:
    """
    Map an action index back to the corresponding GameAction from the legal actions list.

    This finds the legal action whose action_to_index matches the given index.

    Args:
        index: Action index in [0, NUM_ACTIONS).
        legal_actions: List of currently legal GameAction instances.

    Returns:
        The matching GameAction from legal_actions.

    Raises:
        ActionEncodingError: If no legal action matches the index.
    """
    for action in legal_actions:
        if action_to_index(action) == index:
            return action
    raise ActionEncodingError(
        f"No legal action matches index {index}. "
        f"Legal action indices: {[action_to_index(a) for a in legal_actions]}"
    )


def encode_action_mask(legal_actions: List[GameAction]) -> np.ndarray:
    """
    Create a boolean mask over the fixed action space indicating which actions are legal.

    Args:
        legal_actions: List of currently legal GameAction instances.

    Returns:
        np.ndarray of shape (146,) with dtype bool, True for legal actions.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in legal_actions:
        try:
            idx = action_to_index(action)
            mask[idx] = True
        except ActionEncodingError:
            # JUSTIFIED: Skip actions that can't be mapped (e.g., hand index >= MAX_HAND)
            pass
    return mask


# ---------------------------------------------------------------------------
# N-Player action index layout (452 total)
# ---------------------------------------------------------------------------
# 0: DrawStockpile
# 1: DrawDiscard
# 2: CallCambia
# 3: Discard(no ability)
# 4: Discard(ability)
# 5-10: Replace(slot=0-5)
# 11-16: PeekOwn(slot=0-5)
# 17-46: PeekOther(slot, opp_idx) = 17 + slot*5 + opp_idx  (6 slots × 5 opponents)
# 47-226: BlindSwap(own, opp_slot, opp_idx) = 47 + own*30 + opp_slot*5 + opp_idx
# 227-406: KingLook(own, opp_slot, opp_idx) = 227 + own*30 + opp_slot*5 + opp_idx
# 407: KingSwapDecision(No)
# 408: KingSwapDecision(Yes)
# 409: PassSnap
# 410-415: SnapOwn(slot=0-5)
# 416-445: SnapOpponent(slot, opp_idx) = 416 + slot*5 + opp_idx  (6 slots × 5 opponents)
# 446-451: SnapOpponentMove(own_card_idx=0-5)

_NP_IDX_DRAW_STOCKPILE = 0
_NP_IDX_DRAW_DISCARD = 1
_NP_IDX_CALL_CAMBIA = 2
_NP_IDX_DISCARD_NO_ABILITY = 3
_NP_IDX_DISCARD_ABILITY = 4
_NP_IDX_REPLACE_BASE = 5       # 5-10
_NP_IDX_PEEK_OWN_BASE = 11     # 11-16
_NP_IDX_PEEK_OTHER_BASE = 17   # 17-46 (6 slots × 5 opps)
_NP_IDX_BLIND_SWAP_BASE = 47   # 47-226 (6 own × 6 opp_slots × 5 opps = 180)
_NP_IDX_KING_LOOK_BASE = 227   # 227-406 (6 own × 6 opp_slots × 5 opps = 180)
_NP_IDX_KING_SWAP_FALSE = 407
_NP_IDX_KING_SWAP_TRUE = 408
_NP_IDX_PASS_SNAP = 409
_NP_IDX_SNAP_OWN_BASE = 410   # 410-415
_NP_IDX_SNAP_OPP_BASE = 416   # 416-445 (6 slots × 5 opps = 30)
_NP_IDX_SNAP_OPP_MOVE_BASE = 446  # 446-451


def nplayer_action_to_index(action: GameAction, opp_idx: int = 0) -> int:
    """
    Map a GameAction to its N-player action index [0, 452).

    For actions targeting a specific opponent, opp_idx is the relative opponent
    index (0-4, where 0 is the next player clockwise).

    Args:
        action: A GameAction NamedTuple.
        opp_idx: Relative opponent index (0-4) for multi-target actions.

    Returns:
        Integer index in [0, N_PLAYER_NUM_ACTIONS).

    Raises:
        ActionEncodingError: If action type is unrecognized or index is out of range.
    """
    from src.cfr.exceptions import ActionEncodingError

    if action is None:
        raise ActionEncodingError("Action cannot be None")

    if isinstance(action, ActionDrawStockpile):
        return _NP_IDX_DRAW_STOCKPILE

    if isinstance(action, ActionDrawDiscard):
        return _NP_IDX_DRAW_DISCARD

    if isinstance(action, ActionCallCambia):
        return _NP_IDX_CALL_CAMBIA

    if isinstance(action, ActionDiscard):
        return _NP_IDX_DISCARD_ABILITY if action.use_ability else _NP_IDX_DISCARD_NO_ABILITY

    if isinstance(action, ActionReplace):
        idx = action.target_hand_index
        if 0 <= idx < MAX_HAND:
            return _NP_IDX_REPLACE_BASE + idx
        raise ActionEncodingError(f"Replace index {idx} out of range [0, {MAX_HAND})")

    if isinstance(action, ActionAbilityPeekOwnSelect):
        idx = action.target_hand_index
        if 0 <= idx < MAX_HAND:
            return _NP_IDX_PEEK_OWN_BASE + idx
        raise ActionEncodingError(f"PeekOwn index {idx} out of range")

    if isinstance(action, ActionAbilityPeekOtherSelect):
        slot = action.target_opponent_hand_index
        if 0 <= slot < MAX_HAND and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1:
            return _NP_IDX_PEEK_OTHER_BASE + slot * (N_PLAYER_MAX_PLAYERS - 1) + opp_idx
        raise ActionEncodingError(
            f"PeekOther slot={slot} opp_idx={opp_idx} out of range"
        )

    if isinstance(action, ActionAbilityBlindSwapSelect):
        own = action.own_hand_index
        opp_slot = action.opponent_hand_index
        if 0 <= own < MAX_HAND and 0 <= opp_slot < MAX_HAND and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1:
            return (
                _NP_IDX_BLIND_SWAP_BASE
                + own * MAX_HAND * (N_PLAYER_MAX_PLAYERS - 1)
                + opp_slot * (N_PLAYER_MAX_PLAYERS - 1)
                + opp_idx
            )
        raise ActionEncodingError(
            f"BlindSwap own={own} opp_slot={opp_slot} opp_idx={opp_idx} out of range"
        )

    if isinstance(action, ActionAbilityKingLookSelect):
        own = action.own_hand_index
        opp_slot = action.opponent_hand_index
        if 0 <= own < MAX_HAND and 0 <= opp_slot < MAX_HAND and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1:
            return (
                _NP_IDX_KING_LOOK_BASE
                + own * MAX_HAND * (N_PLAYER_MAX_PLAYERS - 1)
                + opp_slot * (N_PLAYER_MAX_PLAYERS - 1)
                + opp_idx
            )
        raise ActionEncodingError(
            f"KingLook own={own} opp_slot={opp_slot} opp_idx={opp_idx} out of range"
        )

    if isinstance(action, ActionAbilityKingSwapDecision):
        return _NP_IDX_KING_SWAP_TRUE if action.perform_swap else _NP_IDX_KING_SWAP_FALSE

    if isinstance(action, ActionPassSnap):
        return _NP_IDX_PASS_SNAP

    if isinstance(action, ActionSnapOwn):
        idx = action.own_card_hand_index
        if 0 <= idx < MAX_HAND:
            return _NP_IDX_SNAP_OWN_BASE + idx
        raise ActionEncodingError(f"SnapOwn index {idx} out of range")

    if isinstance(action, ActionSnapOpponent):
        slot = action.opponent_target_hand_index
        if 0 <= slot < MAX_HAND and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1:
            return _NP_IDX_SNAP_OPP_BASE + slot * (N_PLAYER_MAX_PLAYERS - 1) + opp_idx
        raise ActionEncodingError(
            f"SnapOpponent slot={slot} opp_idx={opp_idx} out of range"
        )

    if isinstance(action, ActionSnapOpponentMove):
        own = action.own_card_to_move_hand_index
        if 0 <= own < MAX_HAND:
            return _NP_IDX_SNAP_OPP_MOVE_BASE + own
        raise ActionEncodingError(f"SnapOpponentMove own={own} out of range")

    raise ActionEncodingError(f"Unrecognized action type: {type(action).__name__}")


def encode_nplayer_action_mask(uint8_mask: np.ndarray) -> np.ndarray:
    """
    Convert a raw uint8 action mask from Go FFI to a boolean numpy array.

    Args:
        uint8_mask: np.ndarray of shape (452,) with dtype uint8 from GoEngine.

    Returns:
        np.ndarray of shape (452,) with dtype bool.
    """
    return uint8_mask.astype(bool)


def encode_infoset_nplayer(
    knowledge_masks: dict,    # (player_idx, slot_idx) → set of player IDs who know
    slot_buckets: dict,       # (player_idx, slot_idx) → bucket int (0-8) or -1 unknown
    encoding_player: int,     # which player is encoding
    num_players: int,         # total players (2-6)
    discard_top_bucket: int,  # 0-9
    stock_estimate: int,      # 0-3
    game_phase: int,          # 0-5
    decision_context: int,    # 0-5
    cambia_state: int,        # 0=self, 1=opponent, 2=none
    drawn_card_bucket: int = -1,  # -1=none, 0-9=bucket
) -> np.ndarray:
    """
    N-player EP-PBS encoding. Returns ndarray of shape (580,).

    Layout:
      [0-215]   Powerset masks: 36 slots × 6 bits (which players know each card)
      [216-539] Slot identities: 36 slots × 9-dim one-hot (zeroed if encoding player doesn't know)
      [540-579] Public features (40 dims)

    Slot layout: slot = player_idx * 6 + card_idx (row-major over players then cards).
    """
    out = np.zeros(N_PLAYER_INPUT_DIM, dtype=np.float32)

    # --- Powerset masks (216 dims): 36 slots × 6 bits ---
    offset = 0
    for p in range(N_PLAYER_MAX_PLAYERS):
        for c in range(MAX_HAND):
            global_slot = p * MAX_HAND + c
            if p < num_players:
                knowers = knowledge_masks.get((p, c), set())
                for bit in range(N_PLAYER_MAX_PLAYERS):
                    if bit in knowers:
                        out[offset + bit] = 1.0
            # else: non-existent player slots remain zero
            offset += N_PLAYER_MAX_PLAYERS
    # offset = 216

    # --- Slot identities (324 dims): 36 slots × 9-dim one-hot ---
    identity_offset = N_PLAYER_POWERSET_DIM  # 216
    for p in range(N_PLAYER_MAX_PLAYERS):
        for c in range(MAX_HAND):
            if p < num_players and encoding_player in knowledge_masks.get((p, c), set()):
                bucket = slot_buckets.get((p, c), -1)
                if 0 <= bucket <= 8:
                    out[identity_offset + bucket] = 1.0
            identity_offset += EP_PBS_BUCKET_DIM
    # identity_offset = 216 + 324 = 540

    # --- Public features (40 dims) ---
    pub_offset = N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM  # 540

    # [540-549]: discard top bucket (10-dim one-hot)
    if 0 <= discard_top_bucket <= 9:
        out[pub_offset + discard_top_bucket] = 1.0
    pub_offset += 10

    # [550-553]: stockpile estimate (4-dim one-hot)
    if 0 <= stock_estimate <= 3:
        out[pub_offset + stock_estimate] = 1.0
    pub_offset += 4

    # [554-559]: game phase (6-dim one-hot)
    if 0 <= game_phase <= 5:
        out[pub_offset + game_phase] = 1.0
    pub_offset += 6

    # [560-565]: decision context (6-dim one-hot)
    if 0 <= decision_context <= 5:
        out[pub_offset + decision_context] = 1.0
    pub_offset += 6

    # [566-568]: cambia state (3-dim one-hot)
    if 0 <= cambia_state <= 2:
        out[pub_offset + cambia_state] = 1.0
    pub_offset += 3

    # [569-579]: drawn card bucket (11-dim one-hot)
    if drawn_card_bucket < 0:
        out[pub_offset + 10] = 1.0  # NONE
    elif 0 <= drawn_card_bucket <= 9:
        out[pub_offset + drawn_card_bucket] = 1.0
    # pub_offset += 11 → total = 580

    return out
