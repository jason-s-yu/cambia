"""Regression test for DESCA production env_factory omniscient pipe.

Catches the case where the production env_factory does not produce a
working omniscient feature pipe, causing `desca_worker._encode_omniscient`
to silently fall through to a zero vector and degenerating the asymmetric
perfect-info critic to a fair-info critic. This bug was live in the Apr
2026 ablation launch and discovered after ~19h of training.

Go FFI engine backend only: `compute_omniscient_features` reads card
identities via `cambia_game_get_all_cards` directly (T1-1). The Python
backend's `_Engine._omniscient_features` counterpart was retired at
cambia-1784 along with `_build_desca_python_env_factory`.
"""

from __future__ import annotations

import numpy as np

from src.cli import _build_desca_env_factory_for_test_go  # type: ignore[attr-defined]
from src.cfr.desca_worker import _encode_omniscient
from src.cfr.omniscient import omniscient_dim


def test_production_env_factory_emits_nonzero_omniscient_features():
    """The production env_factory must produce non-zero, well-formed omni features.

    A zero output indicates the silent zero-fallback path triggered.
    """
    factory = _build_desca_env_factory_for_test_go()
    engine, _agents = factory(rng=np.random.default_rng(0))
    feats = _encode_omniscient(engine)

    expected_dim = omniscient_dim(2)
    assert feats.shape == (expected_dim,), (
        f"omniscient features wrong shape: got {feats.shape}, "
        f"expected ({expected_dim},)"
    )
    assert feats.dtype == np.float32

    # Each 10-dim slot must be a valid one-hot (exactly one 1.0).
    per_slot = 10
    assert expected_dim % per_slot == 0
    num_slots = expected_dim // per_slot
    for s in range(num_slots):
        slot = feats[s * per_slot : (s + 1) * per_slot]
        assert np.isclose(slot.sum(), 1.0), f"slot {s} not a valid one-hot: {slot}"

    # At least some slots must encode real card buckets (indices 0..8), not
    # all empty/unknown. A fresh Cambia game has 4 cards in each player's
    # hand, so we expect 8 real-bucket slots and 4 empty-flag slots in 2P.
    real_bucket_slots = 0
    for s in range(num_slots):
        slot = feats[s * per_slot : (s + 1) * per_slot]
        if slot[:9].sum() > 0:
            real_bucket_slots += 1
    assert real_bucket_slots >= 4, (
        f"expected at least 4 slots with real card buckets in a fresh 2P "
        f"game, got {real_bucket_slots} - silent zero-fallback likely fired"
    )


def test_engine_adapter_exposes_omniscient_pipe():
    """The production env_factory engine must expose `_get_all_cards_unsafe`.

    Without it, `_encode_omniscient` falls through to the silent
    zero-fallback path.
    """
    factory = _build_desca_env_factory_for_test_go()
    engine, _agents = factory(rng=np.random.default_rng(1))
    has_go_method = callable(getattr(engine, "_get_all_cards_unsafe", None))
    assert has_go_method, (
        f"production env_factory engine type {type(engine).__name__} does not "
        f"expose `_get_all_cards_unsafe` - silent zero-fallback would trigger "
        "in `_encode_omniscient`."
    )
