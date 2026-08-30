"""
src/agents/action_codec.py

The action-index <-> GameAction bijection, shared by every caller that has to
cross the FFI boundary.

The Go engine speaks action INDICES: ``cambia_game_legal_actions`` returns a
146-bit mask and ``cambia_game_apply_action`` takes an index. Python agents
speak ``src.constants`` ``GameAction`` NamedTuples. Anything that drives the Go
engine with Python policies (the LBR / ISMCTS-BR best-response estimators, the
ported evaluation wrappers, the ``cambia play`` Go adapter) needs the decode, so
the table lives here instead of being restated per caller.

The layout mirrors ``engine/types.go``'s EncodeXxx and
``src.encoding.action_to_index``, but this module does not import
``src.encoding``: that module pulls in ``AgentState`` and the PBS tables, which
a Go-path search has no use for.

Scope note (cambia-1427): this is the 2-player (146-action) half only. The
N-player 620-action helpers land with cambia-1426, which owns the full version
of this module; the names here are chosen to match it so that version simply
supersedes this one.

Ordering: ``actions_from_mask`` returns ascending index order, which is
process-stable. Do NOT put these NamedTuples in a set and iterate it -- their
string ``tag`` field is PYTHONHASHSEED-salted, so set order varies run to run
(cambia-444).
"""

from typing import List, Sequence

from ..constants import (
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
    GameAction,
)

__all__ = [
    "NUM_ACTIONS",
    "MAX_HAND",
    "index_to_action",
    "actions_from_mask",
]

# Global 2-player action-space size, matching src.encoding.NUM_ACTIONS and the
# Go engine's agent.NumActions.
NUM_ACTIONS = 146

# Maximum hand size the action space indexes over (engine.MaxHandSize).
MAX_HAND = 6

_IDX_DRAW_STOCKPILE = 0
_IDX_DRAW_DISCARD = 1
_IDX_CALL_CAMBIA = 2
_IDX_DISCARD_NO_ABILITY = 3
_IDX_DISCARD_ABILITY = 4
_IDX_REPLACE_BASE = 5  # 5-10
_IDX_PEEK_OWN_BASE = 11  # 11-16
_IDX_PEEK_OTHER_BASE = 17  # 17-22
_IDX_BLIND_SWAP_BASE = 23  # 23-58 (6x6)
_IDX_KING_LOOK_BASE = 59  # 59-94 (6x6)
_IDX_KING_SWAP_FALSE = 95
_IDX_KING_SWAP_TRUE = 96
_IDX_PASS_SNAP = 97
_IDX_SNAP_OWN_BASE = 98  # 98-103
_IDX_SNAP_OPP_BASE = 104  # 104-109
_IDX_SNAP_OPP_MOVE_BASE = 110  # 110-145 (6x6)


def index_to_action(idx: int) -> GameAction:
    """Decode an action index to its ``GameAction``. Total over [0, 146)."""
    if idx == _IDX_DRAW_STOCKPILE:
        return ActionDrawStockpile()
    if idx == _IDX_DRAW_DISCARD:
        return ActionDrawDiscard()
    if idx == _IDX_CALL_CAMBIA:
        return ActionCallCambia()
    if idx == _IDX_DISCARD_NO_ABILITY:
        return ActionDiscard(use_ability=False)
    if idx == _IDX_DISCARD_ABILITY:
        return ActionDiscard(use_ability=True)
    if _IDX_REPLACE_BASE <= idx < _IDX_PEEK_OWN_BASE:
        return ActionReplace(target_hand_index=idx - _IDX_REPLACE_BASE)
    if _IDX_PEEK_OWN_BASE <= idx < _IDX_PEEK_OTHER_BASE:
        return ActionAbilityPeekOwnSelect(target_hand_index=idx - _IDX_PEEK_OWN_BASE)
    if _IDX_PEEK_OTHER_BASE <= idx < _IDX_BLIND_SWAP_BASE:
        return ActionAbilityPeekOtherSelect(
            target_opponent_hand_index=idx - _IDX_PEEK_OTHER_BASE
        )
    if _IDX_BLIND_SWAP_BASE <= idx < _IDX_KING_LOOK_BASE:
        offset = idx - _IDX_BLIND_SWAP_BASE
        return ActionAbilityBlindSwapSelect(
            own_hand_index=offset // MAX_HAND,
            opponent_hand_index=offset % MAX_HAND,
        )
    if _IDX_KING_LOOK_BASE <= idx < _IDX_KING_SWAP_FALSE:
        offset = idx - _IDX_KING_LOOK_BASE
        return ActionAbilityKingLookSelect(
            own_hand_index=offset // MAX_HAND,
            opponent_hand_index=offset % MAX_HAND,
        )
    if idx == _IDX_KING_SWAP_FALSE:
        return ActionAbilityKingSwapDecision(perform_swap=False)
    if idx == _IDX_KING_SWAP_TRUE:
        return ActionAbilityKingSwapDecision(perform_swap=True)
    if idx == _IDX_PASS_SNAP:
        return ActionPassSnap()
    if _IDX_SNAP_OWN_BASE <= idx < _IDX_SNAP_OPP_BASE:
        return ActionSnapOwn(own_card_hand_index=idx - _IDX_SNAP_OWN_BASE)
    if _IDX_SNAP_OPP_BASE <= idx < _IDX_SNAP_OPP_MOVE_BASE:
        return ActionSnapOpponent(opponent_target_hand_index=idx - _IDX_SNAP_OPP_BASE)
    if _IDX_SNAP_OPP_MOVE_BASE <= idx < NUM_ACTIONS:
        offset = idx - _IDX_SNAP_OPP_MOVE_BASE
        return ActionSnapOpponentMove(
            own_card_to_move_hand_index=offset // MAX_HAND,
            target_empty_slot_index=offset % MAX_HAND,
        )
    raise ValueError(f"unrecognized action index {idx}")


def actions_from_mask(indices: Sequence[int]) -> List[GameAction]:
    """Decode legal action indices, preserving ascending order.

    Callers pass ``numpy.flatnonzero(mask)``, which is already ascending -- a
    canonical, process-stable ordering, unlike the Python reference engine's
    ``get_legal_actions()`` set.
    """
    return [index_to_action(int(i)) for i in indices]
