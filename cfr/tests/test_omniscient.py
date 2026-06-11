"""Tier 2 tests for the training-only omniscient feature helper.

Verifies that compute_omniscient_features produces byte-identical output when
invoked on the same game state through the Python wrapper vs. computing the
one-hot encoding directly from the raw FFI byte array.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cfr.omniscient import (
    _OMNISCIENT_PER_SLOT_DIM,
    compute_omniscient_features,
    omniscient_dim,
)
from src.ffi.bridge import GoEngine


def _direct_one_hot(cards: np.ndarray) -> np.ndarray:
    """Reference path: one-hot encode a packed card-bucket array directly."""
    total = cards.shape[0]
    feats = np.zeros(total * _OMNISCIENT_PER_SLOT_DIM, dtype=np.float32)
    for i in range(total):
        v = int(cards[i])
        base = i * _OMNISCIENT_PER_SLOT_DIM
        if v == 0xFF or v >= 9:
            feats[base + 9] = 1.0
        else:
            feats[base + v] = 1.0
    return feats


class TestOmniscientCrossPath:
    @pytest.mark.parametrize("seed", [1, 17, 101, 9999])
    def test_python_wrapper_matches_direct_encoding(self, seed: int) -> None:
        with GoEngine(seed=seed) as engine:
            py_feats = compute_omniscient_features(engine)
            direct_feats = _direct_one_hot(engine._get_all_cards_unsafe())

        assert py_feats.dtype == np.float32
        assert py_feats.shape == direct_feats.shape
        assert py_feats.shape == (omniscient_dim(2),)
        np.testing.assert_array_equal(py_feats, direct_feats)

    def test_per_slot_onehot_is_exactly_one(self) -> None:
        with GoEngine(seed=2026) as engine:
            feats = compute_omniscient_features(engine)

        reshaped = feats.reshape(-1, _OMNISCIENT_PER_SLOT_DIM)
        sums = reshaped.sum(axis=1)
        np.testing.assert_array_equal(sums, np.ones_like(sums))

    def test_dim_matches_constant(self) -> None:
        # 2P: 2 * 6 * 10 = 120.
        assert omniscient_dim(2) == 120
        assert omniscient_dim(4) == 240


class TestOmniscientTrainingOnlyImportBoundary:
    """Guardrail: eval-time modules must not import this helper.

    Static enforcement is not possible in Python, but the helper must live in
    the training subpackage and its docstring must declare the constraint.
    """

    def test_module_has_training_only_warning(self) -> None:
        import src.cfr.omniscient as mod

        assert "TRAINING-ONLY" in (mod.__doc__ or "")

    def test_module_lives_under_cfr_training_subpackage(self) -> None:
        import src.cfr.omniscient as mod

        assert mod.__name__.startswith("src.cfr.")
