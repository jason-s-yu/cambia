"""tests/test_prtcfr_ffi_observe.py

Byte-identity gate for the additive fused-observe FFI export (X3 ladder step
(d1), cambia-607): ``bridge.observe_games_batch`` reads terminal flag +
acting player + legal mask + acting-player token body for a batch of games in
ONE cgo crossing (``cambia_games_observe_batch``), and must return EXACTLY what
the four separate per-game exports return (is_terminal / acting_player /
legal_actions_mask / agent.tokens), across a mixed batch of depths incl.
terminals.

This export is an ADDITIVE primitive: it is byte-identity proven here but is NOT
wired into the generation hot path in this branch. Both a cross-stream batched
consumer (which fragments the scheduler's single-drain inference batch) and a
per-stream n=1 consumer regressed gen wall-clock; see the cambia-607 report for
the investigation. The primitive is kept for a future, correctly-scheduled
consumer.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("cffi")

from src.cfr.prtcfr_worker import PRODUCTION_SEQ_CAP, new_production_driver

try:
    from src.ffi import bridge

    _HAVE_LIB = True
except Exception:  # pragma: no cover - lib unavailable
    _HAVE_LIB = False

pytestmark = pytest.mark.skipif(not _HAVE_LIB, reason="libcambia.so unavailable")


def _advance_random(driver, steps: int, rng: random.Random) -> None:
    """Advance a Go driver up to ``steps`` legal moves (stops early at terminal)."""
    for _ in range(steps):
        if driver.engine.is_terminal():
            return
        mask = driver.engine.legal_actions_mask()
        nz = np.nonzero(mask)[0]
        if len(nz) == 0:
            return
        action = int(rng.choice(nz))
        for _try in range(20):
            try:
                bridge.apply_games_batch(
                    [driver.engine.handle], [driver.a0.handle], [driver.a1.handle],
                    [action],
                )
                break
            except RuntimeError as e:
                if "overflow" in str(e):
                    return
                action = int(rng.choice(nz))


def test_observe_games_batch_matches_per_call():
    """observe_games_batch == (is_terminal, acting_player, legal mask, tokens)
    per game, byte-for-byte, across a mixed batch of depths incl. terminals."""
    rng = random.Random(20250717)
    drivers = []
    for k in range(24):
        d = new_production_driver(1000 + k, backend="go")
        _advance_random(d, rng.randint(0, 60), rng)
        drivers.append(d)

    term, actor, masks, tok_flat, offsets, lens = bridge.observe_games_batch(
        [d.engine.handle for d in drivers],
        [d.a0.handle for d in drivers],
        [d.a1.handle for d in drivers],
        PRODUCTION_SEQ_CAP,
    )

    saw_live = False
    for i, d in enumerate(drivers):
        exp_term = d.engine.is_terminal()
        assert bool(term[i]) == exp_term, f"game {i} terminal mismatch"
        body = tok_flat[int(offsets[i]) : int(offsets[i]) + int(lens[i])]
        if exp_term:
            assert actor[i] == 255
            assert int(lens[i]) == 0
            assert masks[i].sum() == 0
            continue
        saw_live = True
        exp_actor = d.engine.acting_player()
        assert int(actor[i]) == exp_actor, f"game {i} actor mismatch"
        exp_mask = d.engine.legal_actions_mask()
        np.testing.assert_array_equal(masks[i], exp_mask, err_msg=f"game {i} mask")
        acting_agent = d.a0 if exp_actor == 0 else d.a1
        exp_body = acting_agent.tokens()  # raw frame body (no BOS/EOS)
        np.testing.assert_array_equal(body, exp_body, err_msg=f"game {i} token body")

    assert saw_live, "test batch had no live games"
    for d in drivers:
        d.close()


def test_observe_games_batch_empty_and_packed_layout():
    """Empty batch returns empty arrays; a non-trivial batch's packed token
    layout stays consistent with per-game offsets/lengths."""
    empty = bridge.observe_games_batch([], [], [], PRODUCTION_SEQ_CAP)
    assert len(empty[0]) == 0 and len(empty[1]) == 0 and empty[2].shape == (0, 146)

    rng = random.Random(7)
    drivers = [new_production_driver(500 + k, backend="go") for k in range(8)]
    for d in drivers:
        _advance_random(d, rng.randint(20, 50), rng)
    term, actor, masks, tok_flat, offsets, lens = bridge.observe_games_batch(
        [d.engine.handle for d in drivers],
        [d.a0.handle for d in drivers],
        [d.a1.handle for d in drivers],
        PRODUCTION_SEQ_CAP,
    )
    for i in range(len(drivers)):
        if not term[i]:
            assert int(offsets[i]) + int(lens[i]) <= len(tok_flat)
    for d in drivers:
        d.close()
