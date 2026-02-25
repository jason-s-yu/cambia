"""
src/pbs.py

Public Belief State (PBS) module for ReBeL Phase 1.

A PBS captures the public information available to both players at any point in the
game, plus per-player range distributions over possible hand types (multisets of card
buckets). It is the primary input representation for the PBS value and policy networks.

Hand Types
----------
A hand type is a 9-tuple (c0, c1, ..., c8) where ci is the number of cards in bucket i
and sum(ci) == hand_size (default 4). The 9 buckets align with CardBucket enum values:

    0: Joker         (deck count: 2)
    1: RedKing       (deck count: 2)
    2: Ace           (deck count: 4)
    3: LowNum 2-4    (deck count: 12)
    4: MidNum 5-6    (deck count: 8)
    5: PeekSelf 7-8  (deck count: 8)
    6: PeekOther 9-T (deck count: 8)
    7: SwapBlind J-Q (deck count: 8)
    8: HighKing      (deck count: 2)

Public Features (NUM_PUBLIC_FEATURES = 20)
------------------------------------------
Index  Field                       Size  Description
-----  --------------------------  ----  ------------------------------------------
0      turn_norm                   1     turn number / MAX_TURNS (0..1)
1-6    game_phase_onehot           6     one-hot: draw, discard, ability,
                                          snap, cambia_round, terminal
7-16   discard_top_bucket_onehot   10    one-hot: 9 buckets + 1 for empty/unknown
17-19  stockpile_size_triclass     3     soft triclass: [high, medium, low]
                                          stockpile fraction thresholds 2/3, 1/3
"""

from __future__ import annotations

import functools
# DEPRECATED: ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games
# with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.
import warnings

warnings.warn(
    "pbs is DEPRECATED and will be removed. "
    "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
    "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Bucket metadata
# ---------------------------------------------------------------------------

BUCKET_NAMES: list[str] = [
    "Joker",
    "RedKing",
    "Ace",
    "LowNum",
    "MidNum",
    "PeekSelf",
    "PeekOther",
    "SwapBlind",
    "HighKing",
]

DECK_COUNTS: tuple[int, ...] = (2, 2, 4, 12, 8, 8, 8, 8, 2)
"""Maximum cards available per bucket across the full 54-card deck."""

NUM_BUCKETS: int = len(BUCKET_NAMES)
assert NUM_BUCKETS == 9
assert len(DECK_COUNTS) == NUM_BUCKETS

HAND_SIZE: int = 4
"""Default number of cards in a hand for 4-card games."""

# ---------------------------------------------------------------------------
# Hand type enumeration
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def enumerate_hand_types(hand_size: int = HAND_SIZE) -> list[tuple[int, ...]]:
    """
    Return a sorted list of all valid hand types.

    A hand type is a 9-tuple of non-negative integers where:
      - The tuple sums to ``hand_size``
      - Each element does not exceed the corresponding bucket deck count

    The result is cached after the first call.

    Parameters
    ----------
    hand_size:
        Total number of cards in the hand (default 4).

    Returns
    -------
    list[tuple[int, ...]]
        Sorted list of valid hand type tuples.
    """
    result: list[tuple[int, ...]] = []

    # Recursive generator for compositions of ``remaining`` into ``n_bins`` bins
    # with per-bin upper bounds.
    def _generate(bin_idx: int, remaining: int, current: list[int]) -> None:
        if bin_idx == NUM_BUCKETS:
            if remaining == 0:
                result.append(tuple(current))
            return
        max_here = min(remaining, DECK_COUNTS[bin_idx])
        for count in range(0, max_here + 1):
            current.append(count)
            _generate(bin_idx + 1, remaining - count, current)
            current.pop()

    _generate(0, hand_size, [])
    result.sort()
    return result


# Eagerly compute top-level constants so they are available at import time.
_ALL_HAND_TYPES: list[tuple[int, ...]] = enumerate_hand_types()

NUM_HAND_TYPES: int = len(_ALL_HAND_TYPES)
"""Total number of distinct valid 4-card hand types (verified: 468)."""

hand_type_to_index: dict[tuple[int, ...], int] = {
    ht: idx for idx, ht in enumerate(_ALL_HAND_TYPES)
}
"""Mapping from hand type tuple to its integer index."""

index_to_hand_type: list[tuple[int, ...]] = _ALL_HAND_TYPES
"""Mapping from integer index to hand type tuple."""

# ---------------------------------------------------------------------------
# Public feature encoding constants
# ---------------------------------------------------------------------------

NUM_PUBLIC_FEATURES: int = 20
"""
Breakdown of the 20 public features:
  [0]     turn_norm                  (1 float)
  [1-6]   game_phase_onehot          (6 floats)
  [7-16]  discard_top_bucket_onehot  (10 floats: 9 buckets + 1 for empty/unknown)
  [17-19] stockpile_size_triclass    (3 floats: high / medium / low soft membership)
"""

PBS_INPUT_DIM: int = 2 * NUM_HAND_TYPES + NUM_PUBLIC_FEATURES
"""Total dimension of a flattened PBS vector: [range_p0 | range_p1 | public_features]."""

# Game phase indices for the one-hot slice [1..6]
PHASE_DRAW = 0
PHASE_DISCARD = 1
PHASE_ABILITY = 2
PHASE_SNAP = 3
PHASE_CAMBIA_ROUND = 4
PHASE_TERMINAL = 5

# ---------------------------------------------------------------------------
# PBS dataclass
# ---------------------------------------------------------------------------


@dataclass
class PBS:
    """
    Public Belief State for a 2-player Cambia game.

    Attributes
    ----------
    range_p0:
        Probability distribution over hand types for player 0.
        Shape: (NUM_HAND_TYPES,), dtype float32, sums to 1.
    range_p1:
        Probability distribution over hand types for player 1.
        Shape: (NUM_HAND_TYPES,), dtype float32, sums to 1.
    public_features:
        Encoded public information visible to both players.
        Shape: (NUM_PUBLIC_FEATURES,), dtype float32.
        See module docstring for field breakdown.
    """

    range_p0: np.ndarray  # float32[NUM_HAND_TYPES]
    range_p1: np.ndarray  # float32[NUM_HAND_TYPES]
    public_features: np.ndarray  # float32[NUM_PUBLIC_FEATURES]

    def __post_init__(self) -> None:
        """Validate shapes and dtypes."""
        if self.range_p0.shape != (NUM_HAND_TYPES,):
            raise ValueError(
                f"range_p0 must have shape ({NUM_HAND_TYPES},), "
                f"got {self.range_p0.shape}"
            )
        if self.range_p1.shape != (NUM_HAND_TYPES,):
            raise ValueError(
                f"range_p1 must have shape ({NUM_HAND_TYPES},), "
                f"got {self.range_p1.shape}"
            )
        if self.public_features.shape != (NUM_PUBLIC_FEATURES,):
            raise ValueError(
                f"public_features must have shape ({NUM_PUBLIC_FEATURES},), "
                f"got {self.public_features.shape}"
            )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_pbs(pbs: PBS) -> np.ndarray:
    """
    Flatten a PBS into a single float32 vector.

    Concatenates ``[range_p0, range_p1, public_features]``.

    Parameters
    ----------
    pbs:
        The PBS to encode.

    Returns
    -------
    np.ndarray
        Shape ``(PBS_INPUT_DIM,)``, dtype float32.
    """
    return np.concatenate(
        [
            pbs.range_p0.astype(np.float32),
            pbs.range_p1.astype(np.float32),
            pbs.public_features.astype(np.float32),
        ]
    )


def encode_pbs_batch(pbs_list: Sequence[PBS]) -> np.ndarray:
    """
    Encode a batch of PBS instances.

    Parameters
    ----------
    pbs_list:
        Sequence of PBS objects.

    Returns
    -------
    np.ndarray
        Shape ``(B, PBS_INPUT_DIM)``, dtype float32, where B = len(pbs_list).
    """
    if not pbs_list:
        return np.empty((0, PBS_INPUT_DIM), dtype=np.float32)
    return np.stack([encode_pbs(p) for p in pbs_list], axis=0)


# ---------------------------------------------------------------------------
# Bayesian range updater
# ---------------------------------------------------------------------------


def update_range(
    range_vec: np.ndarray,
    action: int,
    policy_matrix: np.ndarray,
) -> np.ndarray:
    """
    Bayesian update of a range distribution after observing an action.

    Applies Bayes' rule:
        range_new[h] ∝ range_vec[h] * policy_matrix[h, action]

    then normalises the result to sum to 1.

    Parameters
    ----------
    range_vec:
        Current range distribution. Shape ``(NUM_HAND_TYPES,)``, sums to 1.
    action:
        The observed action index (column index into ``policy_matrix``).
    policy_matrix:
        Per-hand-type action probabilities.
        Shape ``(NUM_HAND_TYPES, num_actions)``.
        ``policy_matrix[h, a]`` is the probability that a player with hand
        type ``h`` takes action ``a``.

    Returns
    -------
    np.ndarray
        Updated range distribution. Shape ``(NUM_HAND_TYPES,)``, dtype float32,
        sums to 1.

    Notes
    -----
    If the normalisation constant Z is zero (all hand types assign zero
    probability to the observed action), the function returns a uniform range
    rather than a division-by-zero result.
    """
    likelihood = policy_matrix[:, action]  # (NUM_HAND_TYPES,)
    unnorm = range_vec.astype(np.float32) * likelihood.astype(np.float32)
    z = float(unnorm.sum())
    if z == 0.0:
        return uniform_range()
    return (unnorm / z).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def uniform_range() -> np.ndarray:
    """
    Uniform distribution over all hand types.

    Returns
    -------
    np.ndarray
        Shape ``(NUM_HAND_TYPES,)``, dtype float32, with every element equal to
        ``1 / NUM_HAND_TYPES``.
    """
    return np.full(NUM_HAND_TYPES, 1.0 / NUM_HAND_TYPES, dtype=np.float32)


def make_public_features(
    *,
    turn: int = 0,
    max_turns: int = 50,
    phase: int = PHASE_DRAW,
    discard_top_bucket: int | None = None,
    stockpile_remaining: int = 46,
    stockpile_total: int = 46,
    acting_player: int = 0,
) -> np.ndarray:
    """
    Build the ``public_features`` vector for a PBS.

    This is a convenience helper for constructing PBS objects in tests and
    solvers.  The returned array layout matches ``NUM_PUBLIC_FEATURES = 20``:

    Indices  Field
    -------  -----------------------------------
    0        turn_norm = turn / max_turns
    1-6      game_phase one-hot (6 phases)
    7-16     discard_top_bucket one-hot (9 buckets + 1 empty/unknown)
    17-19    stockpile triclass [high, medium, low]

    Parameters
    ----------
    turn:
        Current turn number (0-indexed).
    max_turns:
        Maximum turns used for normalisation (default 50).
    phase:
        One of the ``PHASE_*`` constants (0-5).
    discard_top_bucket:
        Bucket index (0-8) of the top discard card, or ``None`` for
        empty/unknown (mapped to index 9).
    stockpile_remaining:
        Number of cards remaining in the stockpile.
    stockpile_total:
        Total stockpile cards at game start (used for normalisation).
    acting_player:
        Index of the player whose turn it is (0 or 1).
    """
    features = np.zeros(NUM_PUBLIC_FEATURES, dtype=np.float32)

    # [0] turn normalised
    features[0] = turn / max(max_turns, 1)

    # [1-6] game phase one-hot
    if 0 <= phase <= 5:
        features[1 + phase] = 1.0

    # [7-16] discard top bucket one-hot (10 slots: buckets 0-8 + slot 9 for empty)
    if discard_top_bucket is None:
        features[7 + 9] = 1.0  # empty/unknown slot
    elif 0 <= discard_top_bucket <= 8:
        features[7 + discard_top_bucket] = 1.0

    # [17-19] stockpile soft triclass
    frac = stockpile_remaining / max(stockpile_total, 1)
    features[17] = float(frac > 2.0 / 3.0)   # high
    features[18] = float(1.0 / 3.0 < frac <= 2.0 / 3.0)  # medium
    features[19] = float(frac <= 1.0 / 3.0)  # low

    return features
