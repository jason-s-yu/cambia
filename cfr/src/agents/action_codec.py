"""
src/agents/action_codec.py

Legal-action decoding for agents driven by the Go engine.

The Go engine speaks action INDICES: it hands out a legal-action bitmask and
takes back an index. Agents speak ``GameAction`` NamedTuples. This module is
the translation, and it is the only thing an eval-time caller needs in order to
hand an agent a legal-action set without a Python ``CambiaGameState`` in the
loop.

Two action spaces exist and they are not interchangeable:

  - The 2-player space (146 actions, ``src.encoding.action_to_index``). It
    encodes exactly one opponent, so at a table with more than two seats the
    engine's APPLY path resolves that opponent as ``1 - acting``, which
    underflows from seat 2 and indexes off the end of the player array. Only
    ever drive a two-seat table with it.
  - The N-player space (``N_PLAYER_NUM_ACTIONS`` actions,
    ``src.encoding.nplayer_action_to_index``). Every opponent-targeting action
    carries a relative opponent index, so it is the safe space at any seat
    count, and the only correct one above two seats.

Both reverse tables are BUILT by inverting the forward encoders rather than
transcribed from their index arithmetic, so a change to either encoder is
picked up here instead of silently disagreeing with it.

Relative opponent indices follow the engine's ``Opponents(player)`` order:
ascending seat order with the acting seat removed, so relative index ``j`` at
seat ``p`` is the ``j``-th smallest seat that is not ``p``.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

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
    N_PLAYER_MAX_PLAYERS,
)
from ..encoding import (
    MAX_HAND,
    NUM_ACTIONS,
    N_PLAYER_NUM_ACTIONS,
    action_to_index,
    nplayer_action_to_index,
)

__all__ = [
    "NUM_ACTIONS",
    "N_PLAYER_NUM_ACTIONS",
    "MAX_HAND",
    "NPlayerAction",
    "index_to_action",
    "actions_from_mask",
    "nplayer_index_to_action",
    "nplayer_actions_from_mask",
    "nplayer_index_for",
    "relative_opponent_index",
    "absolute_opponent_seat",
]


class NPlayerAction(NamedTuple):
    """An N-player legal action: the move, and which opponent it targets.

    ``opp_idx`` is None for actions that name no opponent (draws, Cambia,
    discard, replace, peek-own, King swap decision, pass-snap, snap-own,
    snap-opponent-move).
    """

    action: GameAction
    opp_idx: Optional[int]


def _every_action_shape() -> List[GameAction]:
    """Every action shape in the game, opponent index left unbound.

    ``ActionSnapOpponentMove`` is enumerated over both of its 2-player fields;
    the N-player space keeps only the own-card index, so the pass over that
    space collapses the duplicates itself.
    """
    shapes: List[GameAction] = [
        ActionDrawStockpile(),
        ActionDrawDiscard(),
        ActionCallCambia(),
        ActionDiscard(use_ability=False),
        ActionDiscard(use_ability=True),
        ActionAbilityKingSwapDecision(perform_swap=False),
        ActionAbilityKingSwapDecision(perform_swap=True),
        ActionPassSnap(),
    ]
    for i in range(MAX_HAND):
        shapes.append(ActionReplace(target_hand_index=i))
        shapes.append(ActionAbilityPeekOwnSelect(target_hand_index=i))
        shapes.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
        shapes.append(ActionSnapOwn(own_card_hand_index=i))
        shapes.append(ActionSnapOpponent(opponent_target_hand_index=i))
    for own in range(MAX_HAND):
        for opp in range(MAX_HAND):
            shapes.append(
                ActionAbilityBlindSwapSelect(own_hand_index=own, opponent_hand_index=opp)
            )
            shapes.append(
                ActionAbilityKingLookSelect(own_hand_index=own, opponent_hand_index=opp)
            )
            shapes.append(
                ActionSnapOpponentMove(
                    own_card_to_move_hand_index=own, target_empty_slot_index=opp
                )
            )
    return shapes


def _build_two_player_table() -> Tuple[Optional[GameAction], ...]:
    """Invert ``action_to_index`` over every action shape."""
    table: List[Optional[GameAction]] = [None] * NUM_ACTIONS
    for action in _every_action_shape():
        idx = action_to_index(action)
        if table[idx] is not None and table[idx] != action:
            raise RuntimeError(
                f"action_to_index is not injective: {table[idx]!r} and {action!r} "
                f"both map to {idx}"
            )
        table[idx] = action
    missing = [i for i, a in enumerate(table) if a is None]
    if missing:
        raise RuntimeError(f"2-player action table has unfilled indices: {missing}")
    return tuple(table)


#: The relative opponent count the N-player space reserves room for. Read off
#: the same constant the encoder validates against, not restated arithmetic.
_NP_MAX_OPPONENTS = N_PLAYER_MAX_PLAYERS - 1


def _build_nplayer_table() -> Tuple[Optional[NPlayerAction], ...]:
    """Invert ``nplayer_action_to_index`` over every (shape, opponent) pair.

    An action whose index does not move with ``opp_idx`` names no opponent, and
    is recorded with ``opp_idx=None`` so a caller never has to guess whether the
    field is meaningful.
    """
    table: List[Optional[NPlayerAction]] = [None] * N_PLAYER_NUM_ACTIONS
    for action in _every_action_shape():
        base = nplayer_action_to_index(action, opp_idx=0)
        targets_opponent = any(
            nplayer_action_to_index(action, opp_idx=j) != base
            for j in range(1, _NP_MAX_OPPONENTS)
        )
        opp_range = range(_NP_MAX_OPPONENTS) if targets_opponent else (0,)
        for opp_idx in opp_range:
            idx = nplayer_action_to_index(action, opp_idx=opp_idx)
            entry = NPlayerAction(action, opp_idx if targets_opponent else None)
            existing = table[idx]
            if existing is not None and existing != entry:
                # SnapOpponentMove is the one many-to-one case: the N-player
                # space keeps only the own-card index, so every 2-player
                # target-slot variant lands on the same index. Keep the first
                # (target slot 0); the runner rebinds the slot from the pending
                # record before handing the action to an agent.
                if isinstance(action, ActionSnapOpponentMove):
                    continue
                raise RuntimeError(
                    f"nplayer_action_to_index collision at {idx}: "
                    f"{existing!r} vs {entry!r}"
                )
            table[idx] = entry
    return tuple(table)


_TWO_PLAYER_TABLE = _build_two_player_table()
_NPLAYER_TABLE = _build_nplayer_table()

#: Forward map for the N-player space, so a chosen action can be re-encoded
#: without re-deriving the arithmetic. Keyed by (action, opp_idx).
_NPLAYER_INDEX: Dict[Tuple[GameAction, Optional[int]], int] = {}
for _idx, _entry in enumerate(_NPLAYER_TABLE):
    if _entry is not None:
        _NPLAYER_INDEX.setdefault((_entry.action, _entry.opp_idx), _idx)


def index_to_action(index: int) -> GameAction:
    """The 2-player-space action at ``index``.

    Total over [0, 146): unlike ``src.encoding.index_to_action`` this needs no
    legal-action list to search, which is the whole point when the legal set
    arrives as an engine bitmask.
    """
    if not 0 <= index < NUM_ACTIONS:
        raise IndexError(f"action index {index} out of range [0, {NUM_ACTIONS})")
    return _TWO_PLAYER_TABLE[index]


def actions_from_mask(mask) -> List[GameAction]:
    """Decode a (146,) legal-action mask into actions, ascending by index.

    Ascending index order is a fixed, process-stable order, unlike iterating a
    ``set`` of these NamedTuples, whose ``tag`` str field puts the iteration
    order under PYTHONHASHSEED (cambia-444). Agents that read the first entry of
    the list, or return the first match while scanning it, therefore behave
    identically run to run.
    """
    arr = np.asarray(mask)
    return [_TWO_PLAYER_TABLE[int(i)] for i in np.flatnonzero(arr)]


def nplayer_index_to_action(index: int) -> NPlayerAction:
    """The N-player-space action at ``index``, with its relative opponent."""
    if not 0 <= index < N_PLAYER_NUM_ACTIONS:
        raise IndexError(
            f"n-player action index {index} out of range [0, {N_PLAYER_NUM_ACTIONS})"
        )
    entry = _NPLAYER_TABLE[index]
    if entry is None:
        raise ValueError(f"n-player action index {index} decodes to no action")
    return entry


def nplayer_actions_from_mask(mask) -> List[NPlayerAction]:
    """Decode an N-player legal mask, ascending by index. Unmapped bits are
    dropped rather than raising: the N-player space is sparse (its opponent axis
    is sized for MaxOpponents, not this table's seat count)."""
    arr = np.asarray(mask)
    out: List[NPlayerAction] = []
    for i in np.flatnonzero(arr):
        entry = _NPLAYER_TABLE[int(i)]
        if entry is not None:
            out.append(entry)
    return out


def nplayer_index_for(action: GameAction, opp_idx: Optional[int]) -> int:
    """Re-encode an agent's chosen action into an N-player action index."""
    key = (action, opp_idx)
    idx = _NPLAYER_INDEX.get(key)
    if idx is not None:
        return idx
    # SnapOpponentMove carries a target slot in the 2-player shape that the
    # N-player space drops; normalize it to the shape the table holds.
    if isinstance(action, ActionSnapOpponentMove):
        return nplayer_action_to_index(action, opp_idx=0)
    return nplayer_action_to_index(action, opp_idx=opp_idx or 0)


def relative_opponent_index(seat: int, opponent_seat: int, num_players: int) -> int:
    """Relative opponent index of ``opponent_seat`` as seen from ``seat``.

    Mirrors the engine's ``Opponents(player)``: ascending seat order with the
    acting seat removed.
    """
    if opponent_seat == seat:
        raise ValueError(f"seat {seat} is not its own opponent")
    return opponent_seat - 1 if opponent_seat > seat else opponent_seat


def absolute_opponent_seat(seat: int, opp_idx: int, num_players: int) -> int:
    """Inverse of :func:`relative_opponent_index`."""
    return opp_idx + 1 if opp_idx >= seat else opp_idx
