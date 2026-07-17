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
from .pbs import DECK_COUNTS, NUM_BUCKETS
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
    EP_PBS_V2_INPUT_DIM,
    V2_CARD_COUNT_DIM,
    V2_ACTION_HISTORY_DIM,
    V2_ACTION_HISTORY_PER_PLAYER,
    V2_ACTION_HISTORY_SLOTS,
    V2_ACTION_CATEGORY_DIM,
    V2_ACTION_SLOT_FEATURE_DIM,
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
_IDX_REPLACE_BASE = 5  # 5-10
_IDX_PEEK_OWN_BASE = 11  # 11-16
_IDX_PEEK_OTHER_BASE = 17  # 17-22
_IDX_BLIND_SWAP_BASE = 23  # 23-58 (6x6=36)
_IDX_KING_LOOK_BASE = 59  # 59-94 (6x6=36)
_IDX_KING_SWAP_FALSE = 95
_IDX_KING_SWAP_TRUE = 96
_IDX_PASS_SNAP = 97
_IDX_SNAP_OWN_BASE = 98  # 98-103
_IDX_SNAP_OPP_BASE = 104  # 104-109
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


def _write_history_features(
    out: np.ndarray, own_obs_ages, opp_obs_ages, dead_card_histogram, turn_progress: float
):
    """Write the 24 new dims at offsets [200-223] into an EP-PBS output buffer."""
    if own_obs_ages is not None:
        for i in range(min(6, len(own_obs_ages))):
            out[200 + i] = own_obs_ages[i]
    if opp_obs_ages is not None:
        for i in range(min(6, len(opp_obs_ages))):
            out[206 + i] = opp_obs_ages[i]
    if dead_card_histogram is not None:
        for i in range(min(10, len(dead_card_histogram))):
            out[212 + i] = dead_card_histogram[i]
    out[222] = turn_progress
    # [223] padding (already zero)


def encode_infoset_eppbs(
    slot_tags: list,  # list of 12 EpistemicTag values
    slot_buckets: list,  # list of 12 bucket values (0 if unknown)
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    decision_context: int,
    cambia_state: int,
    drawn_card_bucket: int = -1,
    own_obs_ages=None,
    opp_obs_ages=None,
    dead_card_histogram=None,
    turn_progress: float = 0.0,
) -> np.ndarray:
    """EP-PBS encoding for 2-player games. Returns ndarray of shape (224,)."""
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

    # [196-199] were padding; now [200-223] are history features.
    _write_history_features(
        out, own_obs_ages, opp_obs_ages, dead_card_histogram, turn_progress
    )
    return out


def encode_infoset_eppbs_dealiased(
    slot_tags: list,  # list of 12 EpistemicTag values
    slot_buckets: list,  # list of 12 bucket values (0 if unknown)
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    decision_context: int,
    cambia_state: int,
    drawn_card_bucket: int = -1,
    own_hand_size: int = 6,
    opp_hand_size: int = 6,
    own_obs_ages=None,
    opp_obs_ages=None,
    dead_card_histogram=None,
    turn_progress: float = 0.0,
) -> np.ndarray:
    """De-aliased flat EP-PBS encoding for 2-player games. Returns ndarray of shape (224,).

    Layout (200 dims):
      [0-39]:   public features (40 dims, identical to encode_infoset_eppbs)
      [40-87]:  slot tags (48 dims = 12 slots × 4-dim one-hot)
                EMPTY slots (slot_in_hand >= hand_size): all zeros (NOT UNK)
      [88-195]: slot identities (108 dims = 12 slots × 9-dim one-hot)
                EMPTY slots: all zeros
      [196]:    own_hand_size / 6.0 (NEW — was padding)
      [197]:    opp_hand_size / 6.0 (NEW — was padding)
      [198-199]: padding (2 dims)

    De-aliasing fix: empty slots (beyond hand_size) are all-zeros in both tag
    and identity regions, distinguishing them from genuinely unknown cards
    (which have UNK tag [1,0,0,0]).
    """
    out = np.zeros(EP_PBS_INPUT_DIM, dtype=np.float32)
    offset = 0

    # Public features (40 dims) — identical to encode_infoset_eppbs
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
    # De-aliasing: skip empty slots (leave all-zeros instead of UNK tag)
    slots_per_player = EP_PBS_MAX_SLOTS // 2
    for i in range(EP_PBS_MAX_SLOTS):
        player_idx = i // slots_per_player
        slot_in_hand = i % slots_per_player
        hand_size = own_hand_size if player_idx == 0 else opp_hand_size

        if slot_in_hand < hand_size:
            tag = slot_tags[i] if i < len(slot_tags) else EpistemicTag.UNK
            if 0 <= tag <= 3:
                out[offset + tag] = 1.0
        # else: empty slot — leave all-zeros (de-aliasing fix)
        offset += EP_PBS_TAG_DIM
    # offset = 88

    # Slot identities (108 dims): 12 slots × 9-dim bucket
    # De-aliasing: skip empty slots (leave all-zeros)
    for i in range(EP_PBS_MAX_SLOTS):
        player_idx = i // slots_per_player
        slot_in_hand = i % slots_per_player
        hand_size = own_hand_size if player_idx == 0 else opp_hand_size

        if slot_in_hand < hand_size:
            tag = slot_tags[i] if i < len(slot_tags) else EpistemicTag.UNK
            if tag in (EpistemicTag.PRIV_OWN, EpistemicTag.PUB):
                bucket = slot_buckets[i] if i < len(slot_buckets) else 0
                if 0 <= bucket <= 8:
                    out[offset + bucket] = 1.0
        # else: empty slot — leave all-zeros (de-aliasing fix)
        offset += EP_PBS_BUCKET_DIM
    # offset = 196

    # [196]: own_hand_size normalized
    out[196] = own_hand_size / 6.0

    # [197]: opp_hand_size normalized
    out[197] = opp_hand_size / 6.0

    # [198-199] were padding; now [200-223] are history features.
    _write_history_features(
        out, own_obs_ages, opp_obs_ages, dead_card_histogram, turn_progress
    )
    return out


def encode_infoset_eppbs_interleaved(
    slot_tags: list,
    slot_buckets: list,
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    decision_context: int,
    cambia_state: int,
    drawn_card_bucket: int = -1,
    own_hand_size: int = 6,
    opp_hand_size: int = 6,
    own_obs_ages=None,
    opp_obs_ages=None,
    dead_card_histogram=None,
    turn_progress: float = 0.0,
    encoding_version: int = 1,
    card_counting_posterior=None,
    action_history_window=None,
) -> np.ndarray:
    """EP-PBS interleaved encoding for 2-player games.

    Output shape depends on ``encoding_version``:
      encoding_version == 1 (default): (224,) - v1 layout, unchanged.
      encoding_version == 2: (257,) - v2 layout with 9-dim card-counting posterior at
        [224:233] and 24-dim action history window at [233:257]. Callers must supply
        ``card_counting_posterior`` (np.ndarray[9]) and ``action_history_window``
        (np.ndarray[24]); both default to zeros if omitted.

    The high-level wrapper ``encode_infoset_eppbs_interleaved_v2(agent_state, ...)``
    computes the v2 extras from AgentState and is the recommended v2 entry point.

    Layout (224 dims):
      [0-41]   public features (42 dims):
               [0-9]   discard_top one-hot (10)
               [10-13] stock_estimate one-hot (4)
               [14-19] game_phase one-hot (6)
               [20-25] decision_context one-hot (6)
               [26-28] cambia_state one-hot (3)
               [29-39] drawn_card one-hot (11)
               [40]    own_hand_size / 6.0
               [41]    opp_hand_size / 6.0
      [42-197] 12 slots × 13 dims each (156 dims):
               per slot i: tag one-hot (4) then identity bucket one-hot (9)
               EMPTY slots: all zeros
               UNK slots: tag=[1,0,0,0], identity=zeros
      [198-199] padding (2 dims, zeros)
      [200-223] history features (24 dims):
               [200-205] own slot obs ages (6)
               [206-211] opp slot obs ages (6)
               [212-221] dead-card histogram (10)
               [222] turn progress
               [223] padding
    """
    if encoding_version not in (1, 2):
        raise InfosetEncodingError(
            f"Unsupported encoding_version={encoding_version}; expected 1 or 2"
        )
    output_dim = EP_PBS_V2_INPUT_DIM if encoding_version == 2 else EP_PBS_INPUT_DIM
    out = np.zeros(output_dim, dtype=np.float32)
    offset = 0

    # Public features (40 dims) — identical to encode_infoset_eppbs
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

    # [40]: own_hand_size normalized
    out[offset] = own_hand_size / 6.0
    offset += 1

    # [41]: opp_hand_size normalized
    out[offset] = opp_hand_size / 6.0
    offset += 1
    # offset = 42

    # Interleaved slot encoding: 12 slots × 13 dims (tag 4 + identity 9)
    slots_per_player = EP_PBS_MAX_SLOTS // 2
    for i in range(EP_PBS_MAX_SLOTS):
        player_idx = i // slots_per_player
        slot_in_hand = i % slots_per_player
        hand_size = own_hand_size if player_idx == 0 else opp_hand_size
        slot_base = 42 + i * (EP_PBS_TAG_DIM + EP_PBS_BUCKET_DIM)

        if slot_in_hand >= hand_size:
            # Empty slot — leave all zeros
            continue

        tag = slot_tags[i] if i < len(slot_tags) else EpistemicTag.UNK
        # Write tag one-hot (4 dims)
        if 0 <= tag <= 3:
            out[slot_base + tag] = 1.0

        # Write identity bucket one-hot (9 dims) — same condition as existing function
        if tag in (EpistemicTag.PRIV_OWN, EpistemicTag.PUB):
            bucket = slot_buckets[i] if i < len(slot_buckets) else 0
            if 0 <= bucket <= 8:
                out[slot_base + EP_PBS_TAG_DIM + bucket] = 1.0
    # offset after slots = 42 + 12*13 = 198

    # [198-199] were padding; now [200-223] are history features.
    _write_history_features(
        out, own_obs_ages, opp_obs_ages, dead_card_histogram, turn_progress
    )

    # Encoding v2: append card-counting posterior + action history window.
    if encoding_version == 2:
        if card_counting_posterior is not None:
            arr = np.asarray(card_counting_posterior, dtype=np.float32)
            if arr.shape[0] != V2_CARD_COUNT_DIM:
                raise InfosetEncodingError(
                    f"card_counting_posterior must have {V2_CARD_COUNT_DIM} entries, got {arr.shape[0]}"
                )
            out[EP_PBS_INPUT_DIM : EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM] = arr
        if action_history_window is not None:
            arr = np.asarray(action_history_window, dtype=np.float32)
            if arr.shape[0] != V2_ACTION_HISTORY_DIM:
                raise InfosetEncodingError(
                    f"action_history_window must have {V2_ACTION_HISTORY_DIM} entries, got {arr.shape[0]}"
                )
            out[EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM : EP_PBS_V2_INPUT_DIM] = arr
    return out


# ---------------------------------------------------------------------------
# Encoding v2 (Phase 0 DESCA foundation)
# ---------------------------------------------------------------------------
# Layout:
#   [0:224]   = v1 interleaved encoding (unchanged)
#   [224:233] = card-counting posterior over remaining deck, 9 buckets
#   [233:257] = action history window, 24 dims
#
# Action history window layout (24 dims total):
#   [233:245] own player's last 3 actions, oldest first, 4 dims each
#   [245:257] opponent's last 3 actions, oldest first, 4 dims each
# Per-action 4 dims:
#   [0:3] action_category one-hot (0=DRAW_PHASE, 1=DISCARD_PHASE, 2=ABILITY_OR_SNAP)
#   [3]   target_slot_norm scalar in [0, 1] (slot_idx / 5.0, or 0.0 if no target)


def compute_card_counting_posterior(agent_state: AgentState) -> np.ndarray:
    """Compute the 9-dim card-counting posterior over remaining deck cards.

    remaining[b] = DECK_COUNTS[b]
                 - known cards in own hand of bucket b (tag in {PRIV_OWN, PUB})
                 - known cards in opp hand of bucket b (tag == PUB only;
                   PRIV_OPP excludes: opp knows, but the agent does not)
                 - observed discard count of bucket b

    The result is clamped to >= 0 and normalized so entries sum to 1.0. If the
    remaining deck computes as all-zero (pathological inputs), returns uniform
    1/9 to preserve the sum-to-1 invariant.
    """
    remaining = np.array(DECK_COUNTS, dtype=np.float64)

    slot_tags = getattr(agent_state, "slot_tags", [])
    slot_buckets = getattr(agent_state, "slot_buckets", [])
    own_hand_size = len(getattr(agent_state, "own_hand", {}))
    opp_hand_size = int(getattr(agent_state, "opponent_card_count", 0))

    # Own hand (slots 0..5): PRIV_OWN or PUB means the bucket is known.
    for i in range(min(6, own_hand_size)):
        if i >= len(slot_tags):
            break
        tag = slot_tags[i]
        tag_val = tag.value if hasattr(tag, "value") else int(tag)
        if tag_val in (int(EpistemicTag.PRIV_OWN), int(EpistemicTag.PUB)):
            b = int(slot_buckets[i]) if i < len(slot_buckets) else 0
            if 0 <= b < NUM_BUCKETS:
                remaining[b] -= 1.0

    # Opponent hand (slots 6..11): only PUB reveals bucket identity.
    for j in range(min(6, opp_hand_size)):
        slot = 6 + j
        if slot >= len(slot_tags):
            break
        tag = slot_tags[slot]
        tag_val = tag.value if hasattr(tag, "value") else int(tag)
        if tag_val == int(EpistemicTag.PUB):
            b = int(slot_buckets[slot]) if slot < len(slot_buckets) else 0
            if 0 <= b < NUM_BUCKETS:
                remaining[b] -= 1.0

    # Discards seen.
    hist = getattr(agent_state, "discard_bucket_counts", None)
    if hist is not None:
        for b in range(min(NUM_BUCKETS, len(hist))):
            remaining[b] -= float(hist[b])

    np.clip(remaining, 0.0, None, out=remaining)
    total = remaining.sum()
    if total <= 0.0:
        # Defensive fallback: uniform posterior keeps sum == 1 for downstream code.
        return np.full(NUM_BUCKETS, 1.0 / NUM_BUCKETS, dtype=np.float32)
    return (remaining / total).astype(np.float32)


def compute_action_history_window(agent_state: AgentState) -> np.ndarray:
    """Build the 24-dim action history window for the encoding player.

    Layout:
      [0:12]  encoding player's (own) ring: 3 slots * 4 dims, oldest first.
      [12:24] opponent's ring: 3 slots * 4 dims, oldest first.

    Per slot: 3-dim one-hot category + 1-dim target_slot_norm. Empty slots
    are left as all zeros (which is distinguishable from an action whose
    target_slot_norm is 0.0 by the category one-hot being absent).
    """
    out = np.zeros(V2_ACTION_HISTORY_DIM, dtype=np.float32)
    history = getattr(agent_state, "action_history", {})
    player_id = int(getattr(agent_state, "player_id", 0))
    opponent_id = int(getattr(agent_state, "opponent_id", 1 - player_id))

    def _write_ring(offset: int, ring):
        if ring is None:
            return
        # Write up to V2_ACTION_HISTORY_SLOTS entries. Oldest first (index 0).
        for slot_idx in range(min(V2_ACTION_HISTORY_SLOTS, len(ring))):
            entry = ring[slot_idx]
            if entry is None:
                continue
            category, slot_norm = entry
            base = offset + slot_idx * V2_ACTION_SLOT_FEATURE_DIM
            if 0 <= int(category) < V2_ACTION_CATEGORY_DIM:
                out[base + int(category)] = 1.0
            # Clamp slot_norm defensively.
            sn = float(slot_norm)
            if sn < 0.0:
                sn = 0.0
            elif sn > 1.0:
                sn = 1.0
            out[base + V2_ACTION_CATEGORY_DIM] = sn

    _write_ring(0, history.get(player_id))
    _write_ring(V2_ACTION_HISTORY_PER_PLAYER, history.get(opponent_id))
    return out


def encode_infoset_v2(
    agent_state: AgentState,
    base_v1: np.ndarray,
) -> np.ndarray:
    """Produce the 257-dim v2 encoding given an AgentState and its 224-dim v1 encoding.

    This is the high-level dispatch helper for ``encoding_version=2`` callers that
    already computed the v1 output via ``encode_infoset_eppbs_interleaved(...)``.
    """
    if base_v1.shape[0] != EP_PBS_INPUT_DIM:
        raise InfosetEncodingError(
            f"Expected v1 base of size {EP_PBS_INPUT_DIM}, got {base_v1.shape[0]}"
        )
    out = np.zeros(EP_PBS_V2_INPUT_DIM, dtype=np.float32)
    out[:EP_PBS_INPUT_DIM] = base_v1
    posterior = compute_card_counting_posterior(agent_state)
    history = compute_action_history_window(agent_state)
    out[EP_PBS_INPUT_DIM : EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM] = posterior
    out[EP_PBS_INPUT_DIM + V2_CARD_COUNT_DIM : EP_PBS_V2_INPUT_DIM] = history
    return out


def encode_infoset_eppbs_interleaved_v2(
    agent_state: AgentState,
    decision_context,
    drawn_card_bucket: int = -1,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """High-level v2 entry point: derives all encoder inputs from AgentState.

    Returns a 257-dim float32 vector.

    This wrapper constructs the low-level kwargs for
    ``encode_infoset_eppbs_interleaved`` from ``agent_state`` and then appends
    the posterior + action history window. It handles the observation-age and
    dead-card histogram features that Python AgentState now tracks (Phase 0
    parity fields B1), producing bit-for-bit matching output with the Go v2
    encoder on equivalent states.

    When ``agent_state`` is backed by a Go FFI agent (carries ``_go_agent``
    attribute pointing at a ``GoAgentState``), the encoding short-circuits to
    the Go-side ``encode_eppbs_interleaved_v2`` FFI which is byte-equivalent
    by construction. Otherwise the slow Python fallback runs, mirroring
    bit-exactly with the Go encoder.

    If ``out`` is provided as a (257,) float32 ndarray, the result is written
    into it (T1-2 buffer reuse hook for callers that pre-allocate).
    """
    # Go FFI fast path: detect a GoAgentState directly or a thin Python adapter
    # carrying ``_go_agent`` (used by the DESCA Go env_factory in cli.py).
    go_agent = getattr(agent_state, "_go_agent", agent_state)
    if hasattr(go_agent, "_agent_h") and hasattr(go_agent, "encode_eppbs_interleaved_v2"):
        ctx = (
            decision_context.value
            if hasattr(decision_context, "value")
            else int(decision_context)
        )
        return go_agent.encode_eppbs_interleaved_v2(
            int(ctx), int(drawn_card_bucket), out=out
        )

    st = agent_state
    # Cambia state mapping (SELF=0, OPPONENT=1, NONE=2).
    if st.cambia_caller is None:
        cambia_state = 2
    elif st.cambia_caller == st.player_id:
        cambia_state = 0
    else:
        cambia_state = 1

    slot_tags = [t.value if hasattr(t, "value") else int(t) for t in st.slot_tags]
    slot_buckets = [int(b) for b in st.slot_buckets]

    discard_top = (
        st.known_discard_top_bucket.value
        if hasattr(st.known_discard_top_bucket, "value")
        else int(st.known_discard_top_bucket)
    )
    stock_estimate = (
        st.stockpile_estimate.value
        if hasattr(st.stockpile_estimate, "value")
        else int(st.stockpile_estimate)
    )
    game_phase = (
        st.game_phase.value if hasattr(st.game_phase, "value") else int(st.game_phase)
    )
    ctx = (
        decision_context.value
        if hasattr(decision_context, "value")
        else int(decision_context)
    )

    own_hand_size = len(st.own_hand)
    opp_hand_size = int(st.opponent_card_count)

    # History parity features: observation ages, dead-card histogram, turn progress.
    max_turns = max(1, int(getattr(st, "max_game_turns", 1) or 1))
    current_turn = int(getattr(st, "_current_game_turn", 0))
    slot_last_seen = getattr(st, "slot_last_seen_turn", [-1] * 12)
    own_obs_ages = [0.0] * 6
    opp_obs_ages = [0.0] * 6
    # Mirror Go ``writeHistoryFeatures``: gate on LastSeenTurn > 0 (strict).
    # The Go engine uses 0 as "never observed" sentinel, so initial peeks stamped
    # at turn 0 are not counted toward an observation age until a later update
    # refreshes the timestamp.
    for i in range(min(6, own_hand_size)):
        ls = slot_last_seen[i] if i < len(slot_last_seen) else -1
        if ls > 0:
            own_obs_ages[i] = (current_turn - int(ls)) / float(max_turns)
    for j in range(min(6, opp_hand_size)):
        ls = slot_last_seen[6 + j] if (6 + j) < len(slot_last_seen) else -1
        if ls > 0:
            opp_obs_ages[j] = (current_turn - int(ls)) / float(max_turns)

    hist_counts = list(getattr(st, "discard_bucket_counts", [0] * NUM_BUCKETS))
    total = int(getattr(st, "total_discards_seen", 0))
    dead_card_histogram = [0.0] * 10  # [212:222] = 10 entries; bucket 9 is unused
    if total > 0:
        for b in range(min(NUM_BUCKETS, len(hist_counts))):
            dead_card_histogram[b] = float(hist_counts[b]) / float(total)

    turn_progress = float(current_turn) / float(max_turns)

    posterior = compute_card_counting_posterior(st)
    history = compute_action_history_window(st)

    return encode_infoset_eppbs_interleaved(
        slot_tags=slot_tags,
        slot_buckets=slot_buckets,
        discard_top_bucket=discard_top,
        stock_estimate=stock_estimate,
        game_phase=game_phase,
        decision_context=ctx,
        cambia_state=cambia_state,
        drawn_card_bucket=int(drawn_card_bucket),
        own_hand_size=own_hand_size,
        opp_hand_size=opp_hand_size,
        own_obs_ages=own_obs_ages,
        opp_obs_ages=opp_obs_ages,
        dead_card_histogram=dead_card_histogram,
        turn_progress=turn_progress,
        encoding_version=2,
        card_counting_posterior=posterior,
        action_history_window=history,
    )


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
        raise ActionEncodingError(
            f"SnapOpponent index {idx} out of range [0, {MAX_HAND})"
        )

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
# N-Player action index layout (N_PLAYER_NUM_ACTIONS total = 620)
# ---------------------------------------------------------------------------
# Mirrors the Go ground truth (engine/types.go NPlayerActionBase* constants,
# engine.MaxOpponents = N_PLAYER_MAX_PLAYERS - 1 = 7). Base offsets below are
# *derived* from N_PLAYER_MAX_PLAYERS/MAX_HAND rather than hardcoded literals
# -- cambia-542 F8/F1 carry-forward: these were previously hardcoded assuming
# 5 opponents (452-action, pre-MaxPlayers-6->8-bump layout), which silently
# desynced from the Go action space (e.g. index 47 decoded as BlindSwap here
# but as a PeekOther slot on the Go side, which starts BlindSwap at 59).
#
# 0: DrawStockpile
# 1: DrawDiscard
# 2: CallCambia
# 3: Discard(no ability)
# 4: Discard(ability)
# 5-10: Replace(slot=0-5)
# 11-16: PeekOwn(slot=0-5)
# 17-58: PeekOther(slot, opp_idx) = 17 + slot*7 + opp_idx  (6 slots x 7 opponents)
# 59-310: BlindSwap(own, opp_slot, opp_idx) = 59 + own*42 + opp_slot*7 + opp_idx
# 311-562: KingLook(own, opp_slot, opp_idx) = 311 + own*42 + opp_slot*7 + opp_idx
# 563: KingSwapDecision(No)
# 564: KingSwapDecision(Yes)
# 565: PassSnap
# 566-571: SnapOwn(slot=0-5)
# 572-613: SnapOpponent(slot, opp_idx) = 572 + slot*7 + opp_idx  (6 slots x 7 opponents)
# 614-619: SnapOpponentMove(own_card_idx=0-5)

_NP_MAX_OPPONENTS = N_PLAYER_MAX_PLAYERS - 1  # 7

_NP_IDX_DRAW_STOCKPILE = 0
_NP_IDX_DRAW_DISCARD = 1
_NP_IDX_CALL_CAMBIA = 2
_NP_IDX_DISCARD_NO_ABILITY = 3
_NP_IDX_DISCARD_ABILITY = 4
_NP_IDX_REPLACE_BASE = 5  # 5-10
_NP_IDX_PEEK_OWN_BASE = 11  # 11-16
_NP_IDX_PEEK_OTHER_BASE = 17  # 6 slots x MaxOpponents opps
_NP_IDX_BLIND_SWAP_BASE = _NP_IDX_PEEK_OTHER_BASE + MAX_HAND * _NP_MAX_OPPONENTS
_NP_IDX_KING_LOOK_BASE = _NP_IDX_BLIND_SWAP_BASE + MAX_HAND * MAX_HAND * _NP_MAX_OPPONENTS
_NP_IDX_KING_SWAP_FALSE = _NP_IDX_KING_LOOK_BASE + MAX_HAND * MAX_HAND * _NP_MAX_OPPONENTS
_NP_IDX_KING_SWAP_TRUE = _NP_IDX_KING_SWAP_FALSE + 1
_NP_IDX_PASS_SNAP = _NP_IDX_KING_SWAP_TRUE + 1
_NP_IDX_SNAP_OWN_BASE = _NP_IDX_PASS_SNAP + 1  # 6 entries
_NP_IDX_SNAP_OPP_BASE = _NP_IDX_SNAP_OWN_BASE + MAX_HAND  # 6 slots x MaxOpponents opps
_NP_IDX_SNAP_OPP_MOVE_BASE = (
    _NP_IDX_SNAP_OPP_BASE + MAX_HAND * _NP_MAX_OPPONENTS
)  # 6 entries


def nplayer_action_to_index(action: GameAction, opp_idx: int = 0) -> int:
    """
    Map a GameAction to its N-player action index [0, N_PLAYER_NUM_ACTIONS).

    For actions targeting a specific opponent, opp_idx is the relative opponent
    index (0 to N_PLAYER_MAX_PLAYERS-2, ascending over every other player).

    Args:
        action: A GameAction NamedTuple.
        opp_idx: Relative opponent index (0 to N_PLAYER_MAX_PLAYERS-2) for
            multi-target actions.

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
        return (
            _NP_IDX_DISCARD_ABILITY if action.use_ability else _NP_IDX_DISCARD_NO_ABILITY
        )

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
        raise ActionEncodingError(f"PeekOther slot={slot} opp_idx={opp_idx} out of range")

    if isinstance(action, ActionAbilityBlindSwapSelect):
        own = action.own_hand_index
        opp_slot = action.opponent_hand_index
        if (
            0 <= own < MAX_HAND
            and 0 <= opp_slot < MAX_HAND
            and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1
        ):
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
        if (
            0 <= own < MAX_HAND
            and 0 <= opp_slot < MAX_HAND
            and 0 <= opp_idx < N_PLAYER_MAX_PLAYERS - 1
        ):
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
        uint8_mask: np.ndarray of shape (N_PLAYER_NUM_ACTIONS,) with dtype uint8 from GoEngine.

    Returns:
        np.ndarray of shape (N_PLAYER_NUM_ACTIONS,) with dtype bool.
    """
    return uint8_mask.astype(bool)


def encode_infoset_nplayer(
    knowledge_masks: dict,  # (player_idx, slot_idx) → set of player IDs who know
    slot_buckets: dict,  # (player_idx, slot_idx) → bucket int (0-8) or -1 unknown
    encoding_player: int,  # which player is encoding
    num_players: int,  # total players (2-N_PLAYER_MAX_PLAYERS)
    discard_top_bucket: int,  # 0-9
    stock_estimate: int,  # 0-3
    game_phase: int,  # 0-5
    decision_context: int,  # 0-5
    cambia_state: int,  # 0=self, 1=opponent, 2=none
    drawn_card_bucket: int = -1,  # -1=none, 0-9=bucket
) -> np.ndarray:
    """
    N-player EP-PBS encoding. Returns ndarray of shape (N_PLAYER_INPUT_DIM,).

    Layout (N_PLAYER_MAX_PLAYERS=8, MAX_HAND=6 -> 48 slots):
      [0-383]   Powerset masks: 48 slots x N_PLAYER_MAX_PLAYERS bits (which players know each card)
      [384-815] Slot identities: 48 slots x 9-dim one-hot (zeroed if encoding player doesn't know)
      [816-855] Public features (40 dims)

    Slot layout: slot = player_idx * MAX_HAND + card_idx (row-major over players then cards).
    """
    out = np.zeros(N_PLAYER_INPUT_DIM, dtype=np.float32)

    # --- Powerset masks (N_PLAYER_POWERSET_DIM dims): 48 slots x 8 bits ---
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
    # offset == N_PLAYER_POWERSET_DIM (384)

    # --- Slot identities (N_PLAYER_IDENTITY_DIM dims): 48 slots x 9-dim one-hot ---
    identity_offset = N_PLAYER_POWERSET_DIM  # 384
    for p in range(N_PLAYER_MAX_PLAYERS):
        for c in range(MAX_HAND):
            if p < num_players and encoding_player in knowledge_masks.get((p, c), set()):
                bucket = slot_buckets.get((p, c), -1)
                if 0 <= bucket <= 8:
                    out[identity_offset + bucket] = 1.0
            identity_offset += EP_PBS_BUCKET_DIM
    # identity_offset == N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM (384 + 432 = 816)

    # --- Public features (N_PLAYER_PUBLIC_DIM=40 dims) ---
    pub_offset = N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM  # 816

    # discard top bucket (10-dim one-hot)
    if 0 <= discard_top_bucket <= 9:
        out[pub_offset + discard_top_bucket] = 1.0
    pub_offset += 10

    # stockpile estimate (4-dim one-hot)
    if 0 <= stock_estimate <= 3:
        out[pub_offset + stock_estimate] = 1.0
    pub_offset += 4

    # game phase (6-dim one-hot)
    if 0 <= game_phase <= 5:
        out[pub_offset + game_phase] = 1.0
    pub_offset += 6

    # decision context (6-dim one-hot)
    if 0 <= decision_context <= 5:
        out[pub_offset + decision_context] = 1.0
    pub_offset += 6

    # cambia state (3-dim one-hot)
    if 0 <= cambia_state <= 2:
        out[pub_offset + cambia_state] = 1.0
    pub_offset += 3

    # drawn card bucket (11-dim one-hot)
    if drawn_card_bucket < 0:
        out[pub_offset + 10] = 1.0  # NONE
    elif 0 <= drawn_card_bucket <= 9:
        out[pub_offset + drawn_card_bucket] = 1.0
    # pub_offset += 11 -> total == N_PLAYER_INPUT_DIM (856)

    return out
