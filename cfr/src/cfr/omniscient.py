"""Training-only omniscient feature extraction for the CVPN critic head.

WARNING: TRAINING-ONLY MODULE.

This module exposes ground-truth card identities for every player's hand via
the FFI export ``cambia_game_get_all_cards``. It is used by the asymmetric
perfect-information critic during DESCA training (v3.1 Phase 1) to compute
baseline values for ESCHER-style MCCFR.

It MUST NOT be imported from any eval-time code path. Eval-time agents (play,
evaluate_agents, head-to-head, search-time inference, PBS rollouts) must read
only belief-conditioned features via ``src/encoding.py`` or ``src/agent_state.py``.

Maintainers: adding a new import edge from this module into eval code is a
safety bug even if it compiles. Treat it like training-only logs or teacher
forcing signals.
"""

from __future__ import annotations

import numpy as np

from src.ffi.bridge import GoEngine

# 9 card buckets (0..8) + 1 "absent/unknown" bit per slot.
_OMNISCIENT_PER_SLOT_DIM: int = 10

# Sentinel used by cambia_game_get_all_cards for empty or unknown slots.
_EMPTY_SENTINEL: int = 0xFF


def omniscient_dim(num_players: int, max_hand_size: int = GoEngine.MAX_HAND_SIZE) -> int:
    """Return the flat omniscient feature dimension for a given player count."""
    return int(num_players) * int(max_hand_size) * _OMNISCIENT_PER_SLOT_DIM


def compute_omniscient_features(engine: GoEngine) -> np.ndarray:
    """Return a flat float32 omniscient feature vector for the given game.

    Shape: ``(num_players * max_hand_size * 10,)``. For each slot, writes a
    10-dim one-hot: indices 0..8 for CardBucket 0..8 (BucketZero..BucketHighKing),
    index 9 for empty or unknown. Exactly one entry per slot is set.

    Args:
        engine: The training-side GoEngine wrapping the game handle.

    Returns:
        np.ndarray of dtype float32 and shape (omniscient_dim,).
    """
    cards = engine._get_all_cards_unsafe()
    total = cards.shape[0]
    feats = np.zeros(total * _OMNISCIENT_PER_SLOT_DIM, dtype=np.float32)
    for i in range(total):
        v = int(cards[i])
        base = i * _OMNISCIENT_PER_SLOT_DIM
        if v == _EMPTY_SENTINEL or v >= 9:
            feats[base + 9] = 1.0
        else:
            feats[base + v] = 1.0
    return feats


__all__ = ["compute_omniscient_features", "omniscient_dim"]
