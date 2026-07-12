"""
src/action_abstraction.py

Semantic action abstraction for DESCA (Dense ESCHER with Semantic Action
Abstraction). Collapses Cambia's 146 concrete 2-player actions down to
NUM_ABSTRACT_ACTIONS_2P = 32 abstract classes.

Design follows v3.1 spec Section 6.1:
- Turn-start and phase-commit actions (draw, cambia) stay 1:1.
- Post-draw replace is clustered by slot bucket quantile (low / mid / high /
  unknown). Discard splits on use_ability to distinguish deliberate ability use.
- Peek-self and peek-other cluster by observation-age and belief-entropy proxies.
- Blind swap and king look cluster over (own_bucket_quantile x opp_known_flag).
- Snap responses collapse per kind: pass_snap, snap_own, snap_opp, snap_opp_move.

Decision-time un-abstraction:
- If exactly one concrete action maps to the class, return it.
- Otherwise pick deterministically using numpy.random.default_rng(seed) on the
  sorted filtered set. Sorting guarantees the same seed always returns the same
  concrete action for the same (legal_actions, agent_state) pair.

The abstraction function depends only on observable features at the current
infoset (own bucket beliefs, opponent belief-observability, observation ages),
so perfect recall is preserved and CFR convergence bounds apply per spec
Section 6.3.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

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
    GameAction,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Indexable list of (abstract_name, rule_description). Rule strings document
# the semantic collapse; the concrete mapping lives in _concrete_to_abstract().
abstract_action_semantics: List[dict] = [
    {"name": "draw_stockpile", "rule": "direct: ActionDrawStockpile"},
    {"name": "draw_discard", "rule": "direct: ActionDrawDiscard"},
    {"name": "call_cambia", "rule": "direct: ActionCallCambia"},
    {"name": "discard_drawn_no_ability", "rule": "ActionDiscard(use_ability=False)"},
    {"name": "discard_drawn_with_ability", "rule": "ActionDiscard(use_ability=True)"},
    {
        "name": "replace_slot_low",
        "rule": "ActionReplace targeting slot whose own bucket quantile is low"
        " (ZERO, NEG_KING, ACE, LOW_NUM)",
    },
    {
        "name": "replace_slot_mid",
        "rule": "ActionReplace targeting slot with mid bucket (MID_NUM, PEEK_SELF)",
    },
    {
        "name": "replace_slot_high",
        "rule": "ActionReplace targeting slot with high bucket"
        " (PEEK_OTHER, SWAP_BLIND, HIGH_KING)",
    },
    {
        "name": "replace_slot_unknown",
        "rule": "ActionReplace targeting slot with UNKNOWN bucket",
    },
    {
        "name": "peek_own_known_recent",
        "rule": "ActionAbilityPeekOwnSelect on own slot with known bucket and recent observation",
    },
    {
        "name": "peek_own_known_stale",
        "rule": "ActionAbilityPeekOwnSelect on own slot with known bucket and stale observation",
    },
    {
        "name": "peek_own_unknown",
        "rule": "ActionAbilityPeekOwnSelect on own slot with UNKNOWN bucket",
    },
    {
        "name": "peek_other_known",
        "rule": "ActionAbilityPeekOtherSelect on opponent slot whose belief is tracked",
    },
    {
        "name": "peek_other_unknown",
        "rule": "ActionAbilityPeekOtherSelect on opponent slot with no tracked belief",
    },
    {
        "name": "blind_swap_own_low_opp_known",
        "rule": "ActionAbilityBlindSwapSelect with own=low, opp tracked",
    },
    {
        "name": "blind_swap_own_low_opp_unknown",
        "rule": "ActionAbilityBlindSwapSelect with own=low, opp untracked",
    },
    {
        "name": "blind_swap_own_mid_opp_known",
        "rule": "ActionAbilityBlindSwapSelect with own=mid_or_unknown, opp tracked",
    },
    {
        "name": "blind_swap_own_mid_opp_unknown",
        "rule": "ActionAbilityBlindSwapSelect with own=mid_or_unknown, opp untracked",
    },
    {
        "name": "blind_swap_own_high_opp_known",
        "rule": "ActionAbilityBlindSwapSelect with own=high, opp tracked",
    },
    {
        "name": "blind_swap_own_high_opp_unknown",
        "rule": "ActionAbilityBlindSwapSelect with own=high, opp untracked",
    },
    {
        "name": "king_look_own_low_opp_known",
        "rule": "ActionAbilityKingLookSelect with own=low, opp tracked",
    },
    {
        "name": "king_look_own_low_opp_unknown",
        "rule": "ActionAbilityKingLookSelect with own=low, opp untracked",
    },
    {
        "name": "king_look_own_mid_opp_known",
        "rule": "ActionAbilityKingLookSelect with own=mid_or_unknown, opp tracked",
    },
    {
        "name": "king_look_own_mid_opp_unknown",
        "rule": "ActionAbilityKingLookSelect with own=mid_or_unknown, opp untracked",
    },
    {
        "name": "king_look_own_high_opp_known",
        "rule": "ActionAbilityKingLookSelect with own=high, opp tracked",
    },
    {
        "name": "king_look_own_high_opp_unknown",
        "rule": "ActionAbilityKingLookSelect with own=high, opp untracked",
    },
    {"name": "king_swap_yes", "rule": "ActionAbilityKingSwapDecision(perform_swap=True)"},
    {"name": "king_swap_no", "rule": "ActionAbilityKingSwapDecision(perform_swap=False)"},
    {"name": "pass_snap", "rule": "direct: ActionPassSnap"},
    {"name": "snap_own", "rule": "ActionSnapOwn over any own hand index"},
    {"name": "snap_opp", "rule": "ActionSnapOpponent over any opponent hand index"},
    {
        "name": "snap_opp_move",
        "rule": "ActionSnapOpponentMove over any (own_to_move, target)",
    },
]

NUM_ABSTRACT_ACTIONS_2P: int = len(abstract_action_semantics)
assert (
    NUM_ABSTRACT_ACTIONS_2P == 32
), f"Expected 32 abstract classes, got {NUM_ABSTRACT_ACTIONS_2P}"

# Name -> index lookup for internal helpers.
_NAME_TO_IDX = {entry["name"]: idx for idx, entry in enumerate(abstract_action_semantics)}

# Buckets considered low, mid, high, unknown for own-slot quantile rules.
_LOW_BUCKETS = {
    CardBucket.ZERO.value,
    CardBucket.NEG_KING.value,
    CardBucket.ACE.value,
    CardBucket.LOW_NUM.value,
}
_MID_BUCKETS = {
    CardBucket.MID_NUM.value,
    CardBucket.PEEK_SELF.value,
}
_HIGH_BUCKETS = {
    CardBucket.PEEK_OTHER.value,
    CardBucket.SWAP_BLIND.value,
    CardBucket.HIGH_KING.value,
}

# Age threshold (in turns) separating "recent" from "stale" own peeks.
_RECENT_AGE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _own_slot_bucket(agent_state: Any, slot_idx: int) -> int:
    """Return own slot bucket value or CardBucket.UNKNOWN.value if missing."""
    own_hand = getattr(agent_state, "own_hand", None)
    if not own_hand or slot_idx not in own_hand:
        return CardBucket.UNKNOWN.value
    info = own_hand[slot_idx]
    bucket = getattr(info, "bucket", None)
    if bucket is None:
        return CardBucket.UNKNOWN.value
    return int(bucket.value if hasattr(bucket, "value") else bucket)


def _own_slot_class(agent_state: Any, slot_idx: int) -> str:
    """Map own slot to one of: low, mid, high, unknown."""
    b = _own_slot_bucket(agent_state, slot_idx)
    if b in _LOW_BUCKETS:
        return "low"
    if b in _MID_BUCKETS:
        return "mid"
    if b in _HIGH_BUCKETS:
        return "high"
    return "unknown"


def _own_slot_last_seen(agent_state: Any, slot_idx: int) -> int:
    """Return last_seen_turn for own slot, or -1 if no info."""
    own_hand = getattr(agent_state, "own_hand", None)
    if not own_hand or slot_idx not in own_hand:
        return -1
    info = own_hand[slot_idx]
    seen = getattr(info, "last_seen_turn", None)
    return -1 if seen is None else int(seen)


def _current_turn(agent_state: Any) -> int:
    """Best-effort read of the current observed turn counter."""
    return int(getattr(agent_state, "_current_game_turn", 0) or 0)


def _peek_own_class(agent_state: Any, slot_idx: int) -> str:
    """Map own slot to peek-self class: known_recent, known_stale, unknown."""
    bucket = _own_slot_bucket(agent_state, slot_idx)
    if bucket == CardBucket.UNKNOWN.value:
        return "unknown"
    last_seen = _own_slot_last_seen(agent_state, slot_idx)
    cur = _current_turn(agent_state)
    if last_seen < 0:
        return "known_stale"
    age = max(0, cur - last_seen)
    return "known_recent" if age <= _RECENT_AGE_THRESHOLD else "known_stale"


def _opp_slot_tracked(agent_state: Any, slot_idx: int) -> bool:
    """Return True if the opponent slot has a tracked (non-UNKNOWN) belief."""
    belief = getattr(agent_state, "opponent_belief", None)
    if not belief or slot_idx not in belief:
        return False
    val = belief[slot_idx]
    raw = getattr(val, "value", val)
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        return False
    return raw != CardBucket.UNKNOWN.value


# ---------------------------------------------------------------------------
# Concrete -> abstract
# ---------------------------------------------------------------------------


def _concrete_to_abstract(action: GameAction, agent_state: Any) -> Optional[int]:
    """Return the abstract index for a concrete action, or None if unmappable."""
    tag = getattr(action, "tag", None)

    if tag == "draw_stockpile":
        return _NAME_TO_IDX["draw_stockpile"]
    if tag == "draw_discard":
        return _NAME_TO_IDX["draw_discard"]
    if tag == "call_cambia":
        return _NAME_TO_IDX["call_cambia"]

    if tag == "discard":
        use = bool(getattr(action, "use_ability", False))
        return _NAME_TO_IDX[
            "discard_drawn_with_ability" if use else "discard_drawn_no_ability"
        ]

    if tag == "replace":
        slot = int(getattr(action, "target_hand_index", 0))
        cls = _own_slot_class(agent_state, slot)
        return _NAME_TO_IDX[f"replace_slot_{cls}"]

    if tag == "peek_own":
        slot = int(getattr(action, "target_hand_index", 0))
        cls = _peek_own_class(agent_state, slot)
        return _NAME_TO_IDX[f"peek_own_{cls}"]

    if tag == "peek_other":
        slot = int(getattr(action, "target_opponent_hand_index", 0))
        cls = "known" if _opp_slot_tracked(agent_state, slot) else "unknown"
        return _NAME_TO_IDX[f"peek_other_{cls}"]

    if tag == "blind_swap":
        own_cls = _own_slot_class(agent_state, int(getattr(action, "own_hand_index", 0)))
        opp_cls = (
            "known"
            if _opp_slot_tracked(
                agent_state, int(getattr(action, "opponent_hand_index", 0))
            )
            else "unknown"
        )
        own_group = "mid" if own_cls == "unknown" else own_cls
        return _NAME_TO_IDX[f"blind_swap_own_{own_group}_opp_{opp_cls}"]

    if tag == "king_look":
        own_cls = _own_slot_class(agent_state, int(getattr(action, "own_hand_index", 0)))
        opp_cls = (
            "known"
            if _opp_slot_tracked(
                agent_state, int(getattr(action, "opponent_hand_index", 0))
            )
            else "unknown"
        )
        own_group = "mid" if own_cls == "unknown" else own_cls
        return _NAME_TO_IDX[f"king_look_own_{own_group}_opp_{opp_cls}"]

    if tag == "king_swap":
        perform = bool(getattr(action, "perform_swap", False))
        return _NAME_TO_IDX["king_swap_yes" if perform else "king_swap_no"]

    if tag == "pass_snap":
        return _NAME_TO_IDX["pass_snap"]
    if tag == "snap_own":
        return _NAME_TO_IDX["snap_own"]
    if tag == "snap_opp":
        return _NAME_TO_IDX["snap_opp"]
    if tag == "snap_opp_move":
        return _NAME_TO_IDX["snap_opp_move"]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def abstract_actions(
    legal_actions: Sequence[GameAction],
    agent_state: Any,
) -> np.ndarray:
    """
    Compute a boolean mask over abstract action classes.

    Returns an ndarray of shape [NUM_ABSTRACT_ACTIONS_2P] where entry i is True
    iff at least one concrete action in `legal_actions` maps to class i.
    """
    mask = np.zeros(NUM_ABSTRACT_ACTIONS_2P, dtype=bool)
    for action in legal_actions:
        idx = _concrete_to_abstract(action, agent_state)
        if idx is not None:
            mask[idx] = True
    return mask


def unabstract(
    abstract_idx: int,
    legal_actions: Sequence[GameAction],
    agent_state: Any,
    seed: int,
) -> GameAction:
    """
    Map an abstract action index to a concrete, legal GameAction.

    Deterministic given `seed`: identical (abstract_idx, legal_actions,
    agent_state, seed) tuples always return the same concrete action.

    Raises:
        ValueError: if no concrete action in `legal_actions` maps to
            `abstract_idx`, or if `abstract_idx` is out of range.
    """
    if not 0 <= int(abstract_idx) < NUM_ABSTRACT_ACTIONS_2P:
        raise ValueError(
            f"abstract_idx {abstract_idx} outside [0, {NUM_ABSTRACT_ACTIONS_2P})"
        )

    candidates: List[GameAction] = [
        a
        for a in legal_actions
        if _concrete_to_abstract(a, agent_state) == int(abstract_idx)
    ]
    if not candidates:
        raise ValueError(
            f"No legal concrete action maps to abstract class "
            f"{abstract_action_semantics[int(abstract_idx)]['name']} "
            f"({abstract_idx})"
        )
    if len(candidates) == 1:
        return candidates[0]

    # Sort for deterministic ordering before seeded selection. Actions are
    # NamedTuples, so standard repr-based sort works without custom comparators.
    candidates.sort(key=lambda a: repr(a))
    rng = np.random.default_rng(int(seed))
    return candidates[int(rng.integers(0, len(candidates)))]


def abstract_action_name(abstract_idx: int) -> str:
    """Return the human-readable name for an abstract action index."""
    if not 0 <= int(abstract_idx) < NUM_ABSTRACT_ACTIONS_2P:
        raise ValueError(
            f"abstract_idx {abstract_idx} outside [0, {NUM_ABSTRACT_ACTIONS_2P})"
        )
    return abstract_action_semantics[int(abstract_idx)]["name"]


__all__ = [
    "NUM_ABSTRACT_ACTIONS_2P",
    "abstract_action_semantics",
    "abstract_actions",
    "unabstract",
    "abstract_action_name",
]
