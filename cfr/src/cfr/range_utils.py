"""
src/cfr/range_utils.py

Per-hand-type range update utilities for SoG/GT-CFR training and eval.

Computes P(action | hand_type) for active hand types by building
delta-range PBS encodings and batching them through the CVPN.
Replaces the broken np.tile(policy, (468, 1)) surrogate that made
Bayesian range updates a no-op.

Performance: uses sparse range support to skip hand types where
range[h] < threshold. Since update_range multiplies by range[h],
near-zero hand types contribute nothing regardless of their policy row.
Early game: ~468 active. Late game: ~20-50. Cost scales with support size.

Encoding is vectorized: identity matrix rows provide delta ranges,
numpy broadcast/concatenate builds the batch without Python loops.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..networks import CVPN
from ..pbs import (
    PBS,
    PBS_INPUT_DIM,
    NUM_HAND_TYPES,
    NUM_PUBLIC_FEATURES,
    make_public_features,
    PHASE_DRAW,
    PHASE_DISCARD,
    PHASE_ABILITY,
    PHASE_SNAP,
    PHASE_TERMINAL,
)
from ..encoding import NUM_ACTIONS

# Decision context integer -> PBS phase index (same mapping as sog_worker/gtcfr_worker)
_CTX_TO_PHASE = {
    0: PHASE_DRAW,
    1: PHASE_DISCARD,
    2: PHASE_ABILITY,
    3: PHASE_SNAP,
    4: PHASE_SNAP,
    5: PHASE_TERMINAL,
}

_MAX_TURNS: int = 46
_STOCK_TOTAL: int = 46

# Pre-computed identity matrix for delta ranges (shared across all calls)
_IDENTITY = np.eye(NUM_HAND_TYPES, dtype=np.float32)  # (468, 468)

# Threshold for sparse range support. Hand types with range[h] below this
# are skipped (policy row set to uniform). Since update_range multiplies
# by range[h], near-zero entries contribute nothing to the Bayesian update.
RANGE_SUPPORT_THRESHOLD: float = 1e-4


def _encode_sparse_delta_batch(
    acting_player: int,
    opponent_range: np.ndarray,
    public_features: np.ndarray,
    active_indices: np.ndarray,
) -> np.ndarray:
    """Build (n_active, PBS_INPUT_DIM) encoding batch for active hand types only.

    Args:
        acting_player: 0 or 1.
        opponent_range: (NUM_HAND_TYPES,) float32.
        public_features: (NUM_PUBLIC_FEATURES,) float32.
        active_indices: (n_active,) int array of hand type indices to encode.

    Returns:
        np.ndarray of shape (n_active, PBS_INPUT_DIM), dtype float32.
    """
    n = len(active_indices)
    # Delta ranges: rows of identity matrix at active indices
    deltas = _IDENTITY[active_indices]  # (n, 468)

    opp_tiled = np.broadcast_to(
        opponent_range[None, :], (n, NUM_HAND_TYPES)
    )
    pub_tiled = np.broadcast_to(
        public_features[None, :], (n, NUM_PUBLIC_FEATURES)
    )

    if acting_player == 0:
        return np.concatenate([deltas, opp_tiled, pub_tiled], axis=1).astype(np.float32)
    else:
        return np.concatenate([opp_tiled, deltas, pub_tiled], axis=1).astype(np.float32)


def _run_cvpn_policy(
    cvpn: CVPN,
    pbs_encs: np.ndarray,
    legal_mask_single: np.ndarray,
) -> np.ndarray:
    """Run CVPN on a batch of PBS encodings and return softmax policy.

    Args:
        cvpn: CVPN in eval mode.
        pbs_encs: (N, PBS_INPUT_DIM) float32.
        legal_mask_single: (NUM_ACTIONS,) bool.

    Returns:
        np.ndarray of shape (N, NUM_ACTIONS) float32 probabilities.
    """
    n = pbs_encs.shape[0]
    mask_t = torch.from_numpy(legal_mask_single).unsqueeze(0).expand(n, -1)
    pbs_t = torch.from_numpy(pbs_encs)
    _, policy_logits = cvpn(pbs_t, mask_t)
    probs = F.softmax(policy_logits, dim=-1)
    return torch.nan_to_num(probs, nan=0.0).detach().numpy()


def compute_policy_matrix_cvpn(
    cvpn: CVPN,
    game: Any,
    range_p0: np.ndarray,
    range_p1: np.ndarray,
) -> np.ndarray:
    """Compute P(action | hand_type) for active hand types using GoEngine state.

    Uses sparse range support: only evaluates hand types where the acting
    player's range exceeds RANGE_SUPPORT_THRESHOLD. Inactive rows are set
    to uniform over legal actions (irrelevant since range[h] ~ 0).

    Args:
        cvpn: CVPN in eval mode.
        game: GoEngine instance at the current decision point (BEFORE apply_action).
        range_p0: Current range for player 0, shape (NUM_HAND_TYPES,).
        range_p1: Current range for player 1, shape (NUM_HAND_TYPES,).

    Returns:
        np.ndarray of shape (NUM_HAND_TYPES, NUM_ACTIONS): policy_matrix[h, a]
        is the probability that a player with hand type h takes action a.
    """
    mask_u8 = game.legal_actions_mask()  # (146,) uint8
    legal_mask = mask_u8.astype(bool)
    acting = game.acting_player()

    # Sparse support: only evaluate hand types with non-negligible range
    acting_range = range_p0 if acting == 0 else range_p1
    active_mask = acting_range > RANGE_SUPPORT_THRESHOLD
    active_indices = np.where(active_mask)[0]

    # Initialize result with uniform over legal actions (for inactive hand types)
    n_legal = int(legal_mask.sum())
    result = np.zeros((NUM_HAND_TYPES, NUM_ACTIONS), dtype=np.float32)
    if n_legal > 0:
        result[:, legal_mask] = 1.0 / n_legal

    if len(active_indices) == 0:
        return result

    ctx = game.decision_ctx()
    phase = _CTX_TO_PHASE.get(ctx, PHASE_DRAW)
    pub = make_public_features(
        turn=game.turn_number(),
        max_turns=_MAX_TURNS,
        phase=phase,
        discard_top_bucket=game.discard_top(),
        stockpile_remaining=game.stock_len(),
        stockpile_total=_STOCK_TOTAL,
    )

    opponent_range = range_p1 if acting == 0 else range_p0
    pbs_encs = _encode_sparse_delta_batch(acting, opponent_range, pub, active_indices)
    active_probs = _run_cvpn_policy(cvpn, pbs_encs, legal_mask)
    result[active_indices] = active_probs

    return result


def compute_policy_matrix_cvpn_from_pbs(
    cvpn: CVPN,
    pbs: PBS,
    legal_mask: np.ndarray,
    acting_player: int,
    range_p0: np.ndarray,
    range_p1: np.ndarray,
) -> np.ndarray:
    """Compute P(action | hand_type) for active hand types from an existing PBS.

    Like compute_policy_matrix_cvpn but works without GoEngine. Uses the
    same sparse range support optimization.

    Args:
        cvpn: CVPN in eval mode.
        pbs: PBS at the current decision point (public features are reused).
        legal_mask: (NUM_ACTIONS,) bool array, True for legal actions.
        acting_player: 0 or 1, the player whose turn it is.
        range_p0: Current range for player 0, shape (NUM_HAND_TYPES,).
        range_p1: Current range for player 1, shape (NUM_HAND_TYPES,).

    Returns:
        np.ndarray of shape (NUM_HAND_TYPES, NUM_ACTIONS): policy_matrix[h, a]
        is the probability that a player with hand type h takes action a.
    """
    legal_mask_bool = legal_mask.astype(bool)

    # Sparse support: only evaluate hand types with non-negligible range
    acting_range = range_p0 if acting_player == 0 else range_p1
    active_mask = acting_range > RANGE_SUPPORT_THRESHOLD
    active_indices = np.where(active_mask)[0]

    # Initialize result with uniform over legal actions
    n_legal = int(legal_mask_bool.sum())
    result = np.zeros((NUM_HAND_TYPES, NUM_ACTIONS), dtype=np.float32)
    if n_legal > 0:
        result[:, legal_mask_bool] = 1.0 / n_legal

    if len(active_indices) == 0:
        return result

    pub = pbs.public_features.astype(np.float32)
    opponent_range = range_p1 if acting_player == 0 else range_p0
    pbs_encs = _encode_sparse_delta_batch(acting_player, opponent_range, pub, active_indices)
    active_probs = _run_cvpn_policy(cvpn, pbs_encs, legal_mask_bool)
    result[active_indices] = active_probs

    return result
