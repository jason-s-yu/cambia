"""tests/test_prtcfr_go_bridge_integration.py

Integration tests against the REAL S1W2 Go FFI bridge (cfr/src/ffi/bridge.py)
for the PRT-CFR production sampler seam (S1W3 stage 2). Requires
cfr/libcambia.so to be built (GOPATH=/tmp/gopath GOMODCACHE=/tmp/gopath/pkg/mod
make libcambia from the repo root); skips cleanly if the .so is absent.

Per the coordinator's stage-2 FFI update: Go and Python engines diverge on
real snap resolution and stockpile-reshuffle order (pre-existing, ticketed
S1W11). These tests therefore validate Go-SIDE SELF-CONSISTENCY only --
save/restore round-trips and CRN-style replay determinism -- never comparing
values against the Python engine across a snap/reshuffle boundary.

IMPORTANT CROSS-LANE FINDING (see also prtcfr_worker.py's PRODUCTION_SEQ_CAP
docstring): the Go engine hard-caps per-agent token storage at
MaxTokenStream=4096 (engine/agent/tokens.go), independent of and BELOW the
v0.4 Phase 2 window-semantics decision's P100-derived PRODUCTION_SEQ_CAP=12288
(measured worst-case avoid_cambia token length 7284 at n=10000, still rising).
This is flagged, not fixed, here (engine/agent/* is S1W2's owned file; raising
it needs a Go source change + libcambia.so rebuild + S1W2's own overflow/parity
tests re-verified) -- see test_token_stream_overflow_hard_errors below, which
demonstrates the hard-error contract IS honored (never silent truncation) at
whatever the current cap is, and reproduces the gap.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CambiaRulesConfig  # noqa: E402

try:
    from src.ffi import bridge
    from src.ffi.bridge import GoAgentState, GoEngine

    _SO_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libcambia.so"
    )
    _HAS_SO = os.path.exists(_SO_PATH)
except Exception:  # pragma: no cover - import-time FFI load failure
    _HAS_SO = False

pytestmark = pytest.mark.skipif(
    not _HAS_SO, reason="libcambia.so not built; run `make libcambia` first"
)


def _production_rules() -> CambiaRulesConfig:
    hr = CambiaRulesConfig()
    hr.allowDrawFromDiscardPile = True
    hr.allowReplaceAbilities = True
    hr.allowOpponentSnapping = True
    hr.max_game_turns = 300
    hr.lockCallerHand = False
    return hr


def _new_game_and_agents(seed: int):
    engine = GoEngine(seed=seed, house_rules=_production_rules())
    a0 = GoAgentState(engine, player_id=0)
    a1 = GoAgentState(engine, player_id=1)
    return engine, a0, a1


def _first_legal_index(engine: GoEngine) -> int:
    mask = engine.legal_actions_mask()
    nz = mask.nonzero()[0]
    assert len(nz) > 0
    return int(nz[0])


# ---------------------------------------------------------------------------
# Basic playthrough + token accumulation (apply_games_batch is the only path
# that actually appends tokens -- cambia_agent_update alone does not, per the
# Go source: exports.go's single-agent update export never calls
# tokenPool[ah].Observe()).
# ---------------------------------------------------------------------------


def test_apply_games_batch_grows_token_streams():
    engine, a0, a1 = _new_game_and_agents(seed=1)
    try:
        len0 = a0.token_len()
        for _ in range(10):
            if engine.is_terminal():
                break
            action = _first_legal_index(engine)
            bridge.apply_games_batch([engine.handle], [a0.handle], [a1.handle], [action])
        len1 = a0.token_len()
        assert len1 > len0, "token stream did not grow after apply_games_batch steps"
    finally:
        a0.close()
        a1.close()
        engine.close()


def test_apply_games_batch_multiple_games_in_one_call():
    triples = [_new_game_and_agents(seed=100 + i) for i in range(3)]
    try:
        game_hs = [e.handle for e, _, _ in triples]
        a0_hs = [a0.handle for _, a0, _ in triples]
        a1_hs = [a1.handle for _, _, a1 in triples]
        actions = [_first_legal_index(e) for e, _, _ in triples]
        bridge.apply_games_batch(game_hs, a0_hs, a1_hs, actions)
        for _e, a0, _a1 in triples:
            assert a0.token_len() > 0
    finally:
        for e, a0, a1 in triples:
            a0.close()
            a1.close()
            e.close()


# ---------------------------------------------------------------------------
# state_save / state_restore round-trip (the invariant the coordinator asked
# for in place of cross-engine value validation on snap/reshuffle paths)
# ---------------------------------------------------------------------------


def test_state_save_restore_round_trip():
    engine, a0, a1 = _new_game_and_agents(seed=7)
    snap_h = None
    try:
        for _ in range(3):
            action = _first_legal_index(engine)
            bridge.apply_games_batch([engine.handle], [a0.handle], [a1.handle], [action])
        checkpoint_turn = engine.turn_number()
        checkpoint_token_len = a0.token_len()
        checkpoint_tokens = a0.tokens().copy()

        snap_h = bridge.state_save(engine.handle, a0.handle, a1.handle)

        # Mutate further past the checkpoint.
        for _ in range(5):
            if engine.is_terminal():
                break
            action = _first_legal_index(engine)
            bridge.apply_games_batch([engine.handle], [a0.handle], [a1.handle], [action])
        assert (
            a0.token_len() >= checkpoint_token_len
        )  # moved forward (or stayed if terminal hit)

        bridge.state_restore(engine.handle, snap_h, a0.handle, a1.handle)

        assert engine.turn_number() == checkpoint_turn
        assert a0.token_len() == checkpoint_token_len
        restored_tokens = a0.tokens()
        assert list(restored_tokens) == list(
            checkpoint_tokens
        ), "restored token stream does not match the checkpointed stream"
    finally:
        if snap_h is not None:
            bridge.state_snapshot_free(snap_h)
        a0.close()
        a1.close()
        engine.close()


def test_state_restore_replay_determinism_crn_invariant():
    """CRN-pairing precondition: restoring to the SAME checkpoint and applying
    the SAME action twice must produce IDENTICAL resulting token streams and
    turn numbers both times (Go-side self-consistency, not a Python-engine
    comparison -- exactly what the coordinator asked this task to validate)."""
    engine, a0, a1 = _new_game_and_agents(seed=13)
    snap_h = None
    try:
        for _ in range(2):
            action = _first_legal_index(engine)
            bridge.apply_games_batch([engine.handle], [a0.handle], [a1.handle], [action])
        snap_h = bridge.state_save(engine.handle, a0.handle, a1.handle)

        replay_action = _first_legal_index(engine)

        # Replicate 1.
        bridge.state_restore(engine.handle, snap_h, a0.handle, a1.handle)
        bridge.apply_games_batch(
            [engine.handle], [a0.handle], [a1.handle], [replay_action]
        )
        tokens_1 = list(a0.tokens())
        turn_1 = engine.turn_number()

        # Replicate 2: restore again, apply the SAME action again.
        bridge.state_restore(engine.handle, snap_h, a0.handle, a1.handle)
        bridge.apply_games_batch(
            [engine.handle], [a0.handle], [a1.handle], [replay_action]
        )
        tokens_2 = list(a0.tokens())
        turn_2 = engine.turn_number()

        assert (
            tokens_1 == tokens_2
        ), "replaying the identical action from the identical checkpoint diverged"
        assert turn_1 == turn_2
    finally:
        if snap_h is not None:
            bridge.state_snapshot_free(snap_h)
        a0.close()
        a1.close()
        engine.close()


# ---------------------------------------------------------------------------
# Token overflow: hard error, never silent truncation -- AND the cross-lane
# cap-mismatch reproduction (flagged, not fixed here).
# ---------------------------------------------------------------------------


def test_token_stream_overflow_hard_errors_never_silent_truncation():
    """Drive a single game far enough (skip CallCambia whenever a non-Cambia
    action exists, matching the P100 script's avoid_cambia stress cohort) to
    cross the Go engine's MaxTokenStream=4096 hard cap. Must raise RuntimeError
    (apply_games_batch's ret==-2 path), never silently truncate.

    This reproduces, against the REAL .so, the exact cross-lane finding in
    prtcfr_worker.py's PRODUCTION_SEQ_CAP docstring: MaxTokenStream=4096 is
    below the P100-measured production worst case (up to 7284+ tokens,
    avoid_cambia cohort) -- a real self-play trajectory of that shape WILL hit
    this and hard-error during training, not just in this synthetic test.
    """
    import random

    from src.constants import ActionCallCambia
    from src.encoding import action_to_index

    call_cambia_idx = action_to_index(ActionCallCambia())

    # A deterministic "always smallest legal index" policy is low-eventfulness
    # (few ability/snap triggers) and lands just under the cap (~3966 tokens
    # at seed 17, empirically) even over the full 300 turns; RANDOM play (like
    # the P100 script's avoid_cambia cohort) is what drives the higher,
    # overflow-crossing counts (mean 3217, p99 4476 at n=10000). Try a handful
    # of seeds with uniform-random non-Cambia play to reliably reproduce it.
    overflowed = False
    for seed in range(20):
        engine, a0, a1 = _new_game_and_agents(seed=1000 + seed)
        rng = random.Random(seed)
        try:
            for _ in range(3000):
                if engine.is_terminal():
                    break
                mask = engine.legal_actions_mask()
                legal = list(mask.nonzero()[0])
                if not legal:
                    break
                non_cambia = [a for a in legal if a != call_cambia_idx]
                pool = non_cambia if non_cambia else legal
                action = int(rng.choice(pool))
                try:
                    bridge.apply_games_batch(
                        [engine.handle], [a0.handle], [a1.handle], [action]
                    )
                except RuntimeError as e:
                    if "overflow" in str(e):
                        overflowed = True
                    else:
                        raise
                    break
        finally:
            a0.close()
            a1.close()
            engine.close()
        if overflowed:
            break

    assert overflowed, (
        "expected a token-stream overflow within 20 random-play seeds at the "
        "production rule profile's 300-turn cap; if this no longer fires, "
        "MaxTokenStream may have been raised -- update this test's "
        "expectation and the PRODUCTION_SEQ_CAP cross-lane note"
    )


def test_go_token_stream_cap_below_p100_production_cap():
    """Documents the cross-lane gap as a live, executable assertion (not just
    a docstring claim): the Go engine's hard token cap is currently BELOW the
    v0.4 Phase 2 P100-derived PRODUCTION_SEQ_CAP. If this test starts failing
    (GO_TOKEN_STREAM_CAP raised to >= PRODUCTION_SEQ_CAP), that is GOOD NEWS --
    delete this test and drop the cross-lane flag."""
    from src.cfr.prtcfr_worker import PRODUCTION_SEQ_CAP

    vocab = bridge.get_token_vocab()
    go_cap = vocab["GO_TOKEN_STREAM_CAP"]
    assert go_cap < PRODUCTION_SEQ_CAP, (
        f"expected the known cross-lane gap (Go cap {go_cap} < Python "
        f"PRODUCTION_SEQ_CAP {PRODUCTION_SEQ_CAP}); if this assertion now "
        f"fails, the gap has been resolved -- update the docstrings and "
        f"remove this guard test"
    )
