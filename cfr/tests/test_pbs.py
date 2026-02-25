"""
tests/test_pbs.py

Tests for the Public Belief State (PBS) module (cfr/src/pbs.py).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.pbs import (
    DECK_COUNTS,
    NUM_BUCKETS,
    NUM_HAND_TYPES,
    NUM_PUBLIC_FEATURES,
    PBS,
    PBS_INPUT_DIM,
    PHASE_ABILITY,
    PHASE_CAMBIA_ROUND,
    PHASE_DISCARD,
    PHASE_DRAW,
    PHASE_SNAP,
    PHASE_TERMINAL,
    encode_pbs,
    encode_pbs_batch,
    enumerate_hand_types,
    hand_type_to_index,
    index_to_hand_type,
    make_public_features,
    uniform_range,
    update_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pbs(phase: int = PHASE_DRAW) -> PBS:
    """Construct a valid PBS with uniform ranges and minimal public features."""
    return PBS(
        range_p0=uniform_range(),
        range_p1=uniform_range(),
        public_features=make_public_features(phase=phase),
    )


# ---------------------------------------------------------------------------
# Hand type enumeration
# ---------------------------------------------------------------------------


def test_num_hand_types() -> None:
    """NUM_HAND_TYPES must equal 468 per the ReBeL design analysis."""
    assert NUM_HAND_TYPES == 468, (
        f"Expected 468 hand types, got {NUM_HAND_TYPES}. "
        "Check bucket deck counts or hand size."
    )


def test_hand_types_valid() -> None:
    """Every hand type must sum to HAND_SIZE and respect per-bucket deck limits."""
    from src.pbs import HAND_SIZE

    hand_types = enumerate_hand_types()
    for ht in hand_types:
        assert sum(ht) == HAND_SIZE, f"Hand type {ht} does not sum to {HAND_SIZE}"
        for bucket_idx, (count, cap) in enumerate(zip(ht, DECK_COUNTS)):
            assert count <= cap, (
                f"Hand type {ht}: bucket {bucket_idx} has count {count} "
                f"exceeding deck cap {cap}"
            )
            assert count >= 0, f"Hand type {ht}: negative count at bucket {bucket_idx}"


def test_hand_types_unique_sorted() -> None:
    """The enumerated list must contain no duplicates and be sorted."""
    hand_types = enumerate_hand_types()
    as_set = set(hand_types)
    assert len(hand_types) == len(as_set), "Duplicate hand types found"
    assert hand_types == sorted(hand_types), "Hand types are not in sorted order"


def test_bidirectional_mapping() -> None:
    """hand_type_to_index and index_to_hand_type must be inverses of each other."""
    assert len(index_to_hand_type) == NUM_HAND_TYPES
    assert len(hand_type_to_index) == NUM_HAND_TYPES

    for idx, ht in enumerate(index_to_hand_type):
        assert hand_type_to_index[ht] == idx, (
            f"index_to_hand_type[{idx}] = {ht}, "
            f"but hand_type_to_index[{ht}] = {hand_type_to_index[ht]}"
        )

    for ht, idx in hand_type_to_index.items():
        assert index_to_hand_type[idx] == ht, (
            f"hand_type_to_index[{ht}] = {idx}, "
            f"but index_to_hand_type[{idx}] = {index_to_hand_type[idx]}"
        )


# ---------------------------------------------------------------------------
# PBS dataclass validation
# ---------------------------------------------------------------------------


def test_pbs_creation_valid() -> None:
    """A correctly shaped PBS should be created without errors."""
    pbs = _make_pbs()
    assert pbs.range_p0.shape == (NUM_HAND_TYPES,)
    assert pbs.range_p1.shape == (NUM_HAND_TYPES,)
    assert pbs.public_features.shape == (NUM_PUBLIC_FEATURES,)


def test_pbs_creation_invalid_range_shape() -> None:
    """PBS __post_init__ must reject wrong range shape."""
    with pytest.raises(ValueError, match="range_p0"):
        PBS(
            range_p0=np.zeros(10, dtype=np.float32),
            range_p1=uniform_range(),
            public_features=make_public_features(),
        )


def test_pbs_creation_invalid_features_shape() -> None:
    """PBS __post_init__ must reject wrong public_features shape."""
    with pytest.raises(ValueError, match="public_features"):
        PBS(
            range_p0=uniform_range(),
            range_p1=uniform_range(),
            public_features=np.zeros(5, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# encode_pbs / encode_pbs_batch
# ---------------------------------------------------------------------------


def test_encode_pbs_shape_dtype() -> None:
    """encode_pbs output must have shape (PBS_INPUT_DIM,) and dtype float32."""
    pbs = _make_pbs()
    encoded = encode_pbs(pbs)
    assert encoded.shape == (PBS_INPUT_DIM,), (
        f"Expected shape ({PBS_INPUT_DIM},), got {encoded.shape}"
    )
    assert encoded.dtype == np.float32, f"Expected float32, got {encoded.dtype}"


def test_encode_pbs_dim_constant() -> None:
    """PBS_INPUT_DIM must equal 2 * NUM_HAND_TYPES + NUM_PUBLIC_FEATURES."""
    expected = 2 * NUM_HAND_TYPES + NUM_PUBLIC_FEATURES
    assert PBS_INPUT_DIM == expected, (
        f"PBS_INPUT_DIM {PBS_INPUT_DIM} != 2*{NUM_HAND_TYPES}+{NUM_PUBLIC_FEATURES}"
    )


def test_encode_pbs_content() -> None:
    """Encoded vector must match manual concatenation of the three fields."""
    pbs = _make_pbs()
    encoded = encode_pbs(pbs)
    expected = np.concatenate([pbs.range_p0, pbs.range_p1, pbs.public_features]).astype(
        np.float32
    )
    np.testing.assert_array_equal(encoded, expected)


def test_encode_pbs_batch() -> None:
    """encode_pbs_batch must return shape (B, PBS_INPUT_DIM) for B PBS objects."""
    pbs_list = [_make_pbs(phase=p) for p in range(6)]
    batch = encode_pbs_batch(pbs_list)
    assert batch.shape == (6, PBS_INPUT_DIM), (
        f"Expected shape (6, {PBS_INPUT_DIM}), got {batch.shape}"
    )
    assert batch.dtype == np.float32

    # Each row must match the individual encode
    for i, pbs in enumerate(pbs_list):
        np.testing.assert_array_equal(batch[i], encode_pbs(pbs))


def test_encode_pbs_batch_empty() -> None:
    """encode_pbs_batch on an empty list returns shape (0, PBS_INPUT_DIM)."""
    batch = encode_pbs_batch([])
    assert batch.shape == (0, PBS_INPUT_DIM)


# ---------------------------------------------------------------------------
# update_range
# ---------------------------------------------------------------------------


def test_update_range_preserves_distribution() -> None:
    """Updated range must sum to 1 (is a valid probability distribution)."""
    rng = np.random.default_rng(42)
    range_vec = uniform_range()
    num_actions = 10
    # Uniform policy — all actions equally likely
    policy_matrix = np.full(
        (NUM_HAND_TYPES, num_actions), 1.0 / num_actions, dtype=np.float32
    )
    updated = update_range(range_vec, action=3, policy_matrix=policy_matrix)
    assert updated.shape == (NUM_HAND_TYPES,)
    np.testing.assert_allclose(updated.sum(), 1.0, atol=1e-5)


def test_update_range_zeros_impossible() -> None:
    """Hand types with zero policy for the observed action must get zero weight."""
    num_actions = 5
    policy_matrix = np.zeros((NUM_HAND_TYPES, num_actions), dtype=np.float32)
    # Only hand type 0 can take action 2
    policy_matrix[0, 2] = 1.0
    policy_matrix[0, 3] = 0.5

    range_vec = uniform_range()
    updated = update_range(range_vec, action=2, policy_matrix=policy_matrix)

    # All hand types except index 0 must have zero probability
    assert updated[0] == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(updated[1:], 0.0, atol=1e-6)
    np.testing.assert_allclose(updated.sum(), 1.0, atol=1e-5)


def test_update_range_all_zero_policy() -> None:
    """When Z=0, update_range must return a uniform distribution."""
    num_actions = 5
    # No hand type ever takes action 4
    policy_matrix = np.zeros((NUM_HAND_TYPES, num_actions), dtype=np.float32)
    range_vec = uniform_range()
    updated = update_range(range_vec, action=4, policy_matrix=policy_matrix)

    expected = uniform_range()
    np.testing.assert_allclose(updated, expected, atol=1e-6)


def test_update_range_dtype() -> None:
    """update_range must return float32 output."""
    num_actions = 3
    policy_matrix = np.ones((NUM_HAND_TYPES, num_actions), dtype=np.float64)
    range_vec = uniform_range().astype(np.float64)
    updated = update_range(range_vec, action=0, policy_matrix=policy_matrix)
    assert updated.dtype == np.float32


def test_update_range_bayesian_proportionality() -> None:
    """Update must be proportional to range * likelihood (Bayes' rule)."""
    num_actions = 4
    rng = np.random.default_rng(7)
    policy_matrix = rng.random((NUM_HAND_TYPES, num_actions)).astype(np.float32)
    # Normalise rows so they are valid policies
    row_sums = policy_matrix.sum(axis=1, keepdims=True)
    policy_matrix = policy_matrix / row_sums

    range_vec = uniform_range()
    action = 1
    updated = update_range(range_vec, action=action, policy_matrix=policy_matrix)

    # Manually compute expected
    likelihood = policy_matrix[:, action]
    unnorm = range_vec * likelihood
    z = unnorm.sum()
    expected = (unnorm / z).astype(np.float32)

    np.testing.assert_allclose(updated, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# uniform_range
# ---------------------------------------------------------------------------


def test_uniform_range() -> None:
    """uniform_range must return float32[NUM_HAND_TYPES] summing to 1."""
    ur = uniform_range()
    assert ur.shape == (NUM_HAND_TYPES,)
    assert ur.dtype == np.float32
    np.testing.assert_allclose(ur.sum(), 1.0, atol=1e-5)
    # All entries equal
    np.testing.assert_allclose(ur, 1.0 / NUM_HAND_TYPES, atol=1e-7)


# ---------------------------------------------------------------------------
# make_public_features helper
# ---------------------------------------------------------------------------


def test_make_public_features_shape() -> None:
    """make_public_features must return float32[NUM_PUBLIC_FEATURES]."""
    f = make_public_features()
    assert f.shape == (NUM_PUBLIC_FEATURES,)
    assert f.dtype == np.float32


def test_make_public_features_phase_onehot() -> None:
    """Only the correct phase slot should be set in the one-hot region."""
    for phase in range(6):
        f = make_public_features(phase=phase)
        phase_slice = f[1:7]
        expected = np.zeros(6, dtype=np.float32)
        expected[phase] = 1.0
        np.testing.assert_array_equal(phase_slice, expected)


def test_make_public_features_discard_empty() -> None:
    """When discard_top_bucket is None, the 'empty' slot (index 9 of 10) is set."""
    f = make_public_features(discard_top_bucket=None)
    discard_slice = f[7:17]
    assert discard_slice[9] == 1.0
    assert discard_slice[:9].sum() == 0.0


def test_make_public_features_discard_bucket() -> None:
    """When a bucket index is given, the correct slot is set."""
    for bucket in range(9):
        f = make_public_features(discard_top_bucket=bucket)
        discard_slice = f[7:17]
        assert discard_slice[bucket] == 1.0
        other = [i for i in range(10) if i != bucket]
        assert discard_slice[other].sum() == 0.0


def test_make_public_features_turn_norm() -> None:
    """turn_norm must equal turn / max_turns."""
    f = make_public_features(turn=10, max_turns=50)
    np.testing.assert_allclose(f[0], 10.0 / 50.0, atol=1e-6)


def test_make_public_features_stockpile_triclass() -> None:
    """Stockpile triclass soft-bins must be mutually exclusive and cover all fractions."""
    # High: > 2/3
    f_high = make_public_features(stockpile_remaining=40, stockpile_total=46)
    assert f_high[17] == 1.0 and f_high[18] == 0.0 and f_high[19] == 0.0

    # Medium: 1/3 < frac <= 2/3
    f_mid = make_public_features(stockpile_remaining=23, stockpile_total=46)
    assert f_mid[17] == 0.0 and f_mid[18] == 1.0 and f_mid[19] == 0.0

    # Low: <= 1/3
    f_low = make_public_features(stockpile_remaining=10, stockpile_total=46)
    assert f_low[17] == 0.0 and f_low[18] == 0.0 and f_low[19] == 1.0
