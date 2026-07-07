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

CROSS-LANE GAP RESOLVED (S1W12): the Go engine's per-agent token cap
(MaxTokenStream, engine/agent/tokens.go) was raised 4096 -> 12288 to match
prtcfr_worker.py's P100-derived PRODUCTION_SEQ_CAP=12288 (measured worst-case
avoid_cambia token length 7284 at n=10000). The durable invariant -- Go cap
>= PRODUCTION_SEQ_CAP, read live via bridge.get_token_stream_cap() (backed by
the cambia_token_stream_cap FFI export), never hardcoded -- is asserted by
test_go_token_stream_cap_at_least_production_cap below. See also
test_token_stream_overflow_hard_errors_never_silent_truncation, which still
demonstrates the hard-error contract (never silent truncation) at whatever
the current cap is, via a synthetic long-game stress profile since a real
300-turn production game no longer reaches the (now larger) cap.
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
# Token overflow: hard error, never silent truncation.
# ---------------------------------------------------------------------------


def test_token_stream_overflow_hard_errors_never_silent_truncation():
    """Drive a single game far enough to cross the Go engine's MaxTokenStream
    hard cap (currently 12288, S1W12). Must raise RuntimeError
    (apply_games_batch's ret==-2 path), never silently truncate.

    A REAL production game (max_game_turns=300) no longer reaches the cap
    (P100-measured worst case 7284 tokens, comfortably under 12288) -- that is
    the point of S1W12's cap raise. To still exercise the hard-error CONTRACT
    against the real .so, this uses a synthetic long-game rule profile
    (max_game_turns=900, 3x production) purely to accumulate enough tokens;
    it is not claiming that a real production trajectory overflows. The
    Go-side unit test (engine/agent/tokens_test.go::TestTokenOverflowIsHardError)
    covers the same contract at the unit level, cap-agnostically.
    """
    import random

    from src.constants import ActionCallCambia
    from src.encoding import action_to_index

    call_cambia_idx = action_to_index(ActionCallCambia())

    def _long_game_rules() -> CambiaRulesConfig:
        hr = _production_rules()
        hr.max_game_turns = 900  # synthetic: only to reliably exceed 12288 tokens
        return hr

    # Uniform-random non-Cambia play (matching the P100 script's avoid_cambia
    # cohort shape) reliably overflows within a handful of seeds at this
    # length (verified empirically: 10/10 seeds overflow within ~9000 steps).
    overflowed = False
    for seed in range(10):
        engine = GoEngine(seed=1000 + seed, house_rules=_long_game_rules())
        a0 = GoAgentState(engine, player_id=0)
        a1 = GoAgentState(engine, player_id=1)
        rng = random.Random(seed)
        try:
            for _ in range(9000):
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
        "expected a token-stream overflow within 10 random-play seeds at the "
        "synthetic 900-turn stress profile; if this no longer fires, "
        "MaxTokenStream may have been raised further -- update this test's "
        "stress profile to accumulate more tokens"
    )


def test_go_token_stream_cap_at_least_production_cap():
    """Durable invariant (inverted from the pre-S1W12 gap-reproduction guard):
    the Go engine's hard token cap must stay >= the P100-derived
    PRODUCTION_SEQ_CAP. Reads BOTH sides live -- the Go cap via the dedicated
    cambia_token_stream_cap FFI export (bridge.get_token_stream_cap()), never
    a hardcoded literal -- so this cannot silently drift out of sync with
    either constant's actual value."""
    from src.cfr.prtcfr_worker import PRODUCTION_SEQ_CAP

    go_cap = bridge.get_token_stream_cap()
    assert go_cap >= PRODUCTION_SEQ_CAP, (
        f"Go token-stream cap {go_cap} (engine/agent/tokens.go::MaxTokenStream) "
        f"is below PRODUCTION_SEQ_CAP {PRODUCTION_SEQ_CAP} "
        f"(cfr/src/cfr/prtcfr_worker.py) -- long self-play trajectories will "
        f"hard-error during training. Raise MaxTokenStream to at least "
        f"PRODUCTION_SEQ_CAP and rebuild libcambia.so."
    )
    # Cross-check: the vocab-embedded field must agree with the dedicated
    # single-value export (two independent Go-side read paths, one value).
    vocab_cap = bridge.get_token_vocab()["GO_TOKEN_STREAM_CAP"]
    assert vocab_cap == go_cap, (
        f"cambia_token_vocab's GO_TOKEN_STREAM_CAP ({vocab_cap}) disagrees "
        f"with cambia_token_stream_cap ({go_cap})"
    )
