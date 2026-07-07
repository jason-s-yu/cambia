"""tests/test_token_stream_parity.py

AC3 gate for the Go-side PRT-CFR event-stream token FFI (S1W2).

The Go engine tokenizes each player's observation-action event stream into an
append-only int32 buffer as a game advances (engine/agent/tokens.go), mirroring
cfr/src/sequence_encoding.py byte-for-byte. This test proves that with a LIVE
lockstep driver (the Phase 0 pattern, no golden fixtures): identical full
2-player games are played in the Python engine and the Go engine over shared
seeds, and at every in-lockstep decision point the Go token stream body is
asserted equal to Python encode_observation_sequence(..., add_bos_eos=False,
seq_cap=inf).

Coverage (asserted below):
- >= 100 live full-game decision points compared byte-for-byte.
- full-length games whose per-player streams EXCEED the 256-token SEQ_CAP
  (byte-equality on the FULL uncapped stream, per the window-semantics note).
- the frame-aligned keep-most-recent truncation helper (bridge.frame_aligned_
  window) reproduces encode_observation_sequence at a PARAMETERIZED cap (256 and
  the raised production window) byte-for-byte.
- constants cross-check: Go vocabulary layout + card/action/outcome id mappings
  assert equal to the Python module (no silent drift).
- cambia frame coverage (live lockstep).
- snap success/penalty + slot frames: Go-emitted, decoded by the Python codec.
- token-inclusive state save/restore round-trip.

Lockstep-narrowing note (per the AC3 rule "narrow the compare scope per-condition
and document; never skip wholesale"): the Go and Python engines have a KNOWN
pre-existing parity gap in snap-phase entry and successful-snap resolution
(engine core, out of this task's scope; see test_cross_engine_samples.py and
test_encoding_v2.py's ring_drift handling). The live driver therefore drains
snap phases via the common PassSnap action (which stays byte-identical) and stops
a game the instant the two engines' snap-phase state disagrees. Successful-snap
and ability-select frame *assembly* is byte-identical to any other snap/public
frame; the specific outcome/action tokens are covered by the exhaustive mapping
cross-check and (for snap success/penalty) a Go-emit -> Python-decode check.

Run (needs libcambia.so):
  python -m pytest tests/test_token_stream_parity.py -v -s
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card import Card
from src.constants import (
    NUM_PLAYERS,
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
)
from src.abstraction import get_card_bucket
from src.encoding import action_to_index, encode_action_mask
import src.sequence_encoding as se
from src.cfr.worker import _create_observation, _filter_observation

try:
    from tests.test_cross_engine_samples import (
        _setup_python_game_matching_go,
        _TEST_RULES,
        go_available,
        skip_if_no_go,
    )
except ImportError:  # pragma: no cover - path fallback
    from test_cross_engine_samples import (  # type: ignore
        _setup_python_game_matching_go,
        _TEST_RULES,
        go_available,
        skip_if_no_go,
    )

if go_available:
    from src.ffi.bridge import (
        GoAgentState,
        GoEngine,
        apply_games_batch,
        encode_action_token,
        encode_card_token,
        frame_aligned_window,
        get_token_vocab,
        python_card_to_go_index,
        state_restore,
        state_save,
        state_snapshot_free,
    )

_CAMBIA_IDX = 2
_DISCARD_WITH_ABILITY_IDX = 4  # engine legal-gen gap (py-only); excluded from picks
_SNAP_MIN = 97
_FULL_SEEDS = list(range(40))


# ---------------------------------------------------------------------------
# Constants + mapping cross-check (Go layout asserted against Python)
# ---------------------------------------------------------------------------


@skip_if_no_go
def test_token_vocab_layout_matches_python():
    """Go token vocabulary layout equals sequence_encoding.py, field for field."""
    v = get_token_vocab()
    expected = {
        "VOCAB_SIZE": se.VOCAB_SIZE,
        "PAD_ID": se.PAD_ID,
        "BOS_ID": se.BOS_ID,
        "EOS_ID": se.EOS_ID,
        "SEP_ID": se.SEP_ID,
        "NUM_SPECIAL": se.NUM_SPECIAL,
        "FRAME_BASE": se.FRAME_BASE,
        "NUM_FRAME_IDS": se.NUM_FRAME_IDS,
        "ACTOR_BASE": se.ACTOR_BASE,
        "MAX_ACTORS": se.MAX_ACTORS,
        "ACTION_BASE": se.ACTION_BASE,
        "NUM_ACTION_IDS": se.NUM_ACTION_IDS,
        "CARD_BASE": se.CARD_BASE,
        "NUM_CARD_IDS": se.NUM_CARD_IDS,
        "SLOT_BASE": se.SLOT_BASE,
        "NUM_SLOT_IDS": se.NUM_SLOT_IDS,
        "OUTCOME_BASE": se.OUTCOME_BASE,
        "NUM_SNAP_OUTCOME_IDS": se.NUM_SNAP_OUTCOME_IDS,
        "MAX_SLOTS": se.MAX_SLOTS,
        "SEQ_CAP": se.SEQ_CAP,
    }
    for k, want in expected.items():
        assert v[k] == want, f"vocab field {k}: Go={v[k]} Python={want}"
    assert v["GO_TOKEN_STREAM_CAP"] >= se.SEQ_CAP


@skip_if_no_go
def test_card_token_mapping_matches_python():
    """Every card identity maps to the same CARD token on both sides."""
    for rank, suit in se.CARD_IDENTITIES:
        card = Card(rank=rank, suit=suit)
        go_idx = python_card_to_go_index(card)
        assert encode_card_token(go_idx) == se._card_tok(card), (
            f"card ({rank},{suit}) go_idx={go_idx}"
        )
    assert encode_card_token(52) == encode_card_token(53)  # both jokers collapse


@skip_if_no_go
def test_action_token_mapping_matches_python():
    """Every 2-player action index maps to the same ACTION token on both sides."""
    actions: List[Any] = [
        ActionDrawStockpile(),
        ActionDrawDiscard(),
        ActionCallCambia(),
        ActionDiscard(use_ability=True),
        ActionDiscard(use_ability=False),
        ActionPassSnap(),
        ActionAbilityKingSwapDecision(perform_swap=False),
        ActionAbilityKingSwapDecision(perform_swap=True),
    ]
    for s in range(6):
        actions.append(ActionReplace(target_hand_index=s))
    for s in range(6):
        actions.append(ActionAbilityPeekOwnSelect(target_hand_index=s))
    for s in range(6):
        actions.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=s))
    for o in range(6):
        for p in range(6):
            actions.append(
                ActionAbilityBlindSwapSelect(own_hand_index=o, opponent_hand_index=p)
            )
    for o in range(6):
        for p in range(6):
            actions.append(
                ActionAbilityKingLookSelect(own_hand_index=o, opponent_hand_index=p)
            )
    for s in range(6):
        actions.append(ActionSnapOwn(own_card_hand_index=s))
    for s in range(6):
        actions.append(ActionSnapOpponent(opponent_target_hand_index=s))
    for o in range(6):
        for p in range(6):
            actions.append(
                ActionSnapOpponentMove(
                    own_card_to_move_hand_index=o, target_empty_slot_index=p
                )
            )
    assert len(actions) == 146
    for a in actions:
        idx = action_to_index(a)
        local, _name = se._action_local_id(a)
        assert encode_action_token(idx) == se._action_tok(local), (
            f"action {a} idx={idx}"
        )


# ---------------------------------------------------------------------------
# Live lockstep driver
# ---------------------------------------------------------------------------


def _go_legal_set(eng: "GoEngine") -> set:
    return set(np.where(eng.legal_actions_mask() > 0)[0].tolist())


def _py_legal_set(pygame) -> set:
    mask = encode_action_mask(list(pygame.get_legal_actions())).astype(np.uint8)
    return set(np.where(mask > 0)[0].tolist())


_DRAW_STOCKPILE_IDX = 0


def _is_snap_only(s: set) -> bool:
    return len(s) > 0 and all(i >= _SNAP_MIN for i in s)


def _states_diverged(eng: "GoEngine", pygame) -> bool:
    """Detect hidden-state divergence via observable signals.

    Catches the pre-existing engine reshuffle-order gap: when the stockpile
    empties and refills, Go and Python re-order it differently, so subsequent
    drawn/discarded cards differ. The discard-top bucket and stockpile size
    surface that without needing card-identity FFI.
    """
    if eng.is_terminal() != pygame.is_terminal():
        return True
    if eng.is_terminal():
        return False
    if eng.acting_player() != pygame.get_acting_player():
        return True
    if eng.stock_len() != len(pygame.stockpile):
        return True
    py_top = pygame.get_discard_top()
    py_bucket = -1 if py_top is None else get_card_bucket(py_top).value
    go_bucket = eng.discard_top()
    if go_bucket is None:
        go_bucket = -1
    return go_bucket != py_bucket


def _py_body(init_hand, init_peek, obs_stream, observer) -> List[int]:
    return se.encode_observation_sequence(
        init_hand, init_peek, obs_stream, observer, seq_cap=10**9, add_bos_eos=False
    )


class _GameResult:
    def __init__(self):
        self.compared = 0
        self.max_body = 0
        self.saw_cambia_frame = False
        self.saw_snap_frame = False
        self.init_hands: Dict[int, List[Card]] = {}
        self.init_peeks: Dict[int, Tuple[int, ...]] = {}
        self.obs_streams: Dict[int, List[Any]] = {}
        self.go_bodies: Dict[int, List[int]] = {}


def _play_lockstep(seed: int, call_cambia_after: int = -1) -> _GameResult:
    """Play one Python+Go lockstep game, asserting token-body byte-equality.

    Selection is deterministic lowest-common-action (the Phase-0 discipline):
    stays in engine lockstep for full games while draining snap phases via the
    common PassSnap. call_cambia_after >= 0 forces a CallCambia at the first
    start-turn after that many steps (covers the cambia frame). The game stops
    cleanly the instant the engines' snap-phase state disagrees (documented
    engine parity gap, out of scope).
    """
    res = _GameResult()
    pygame = _setup_python_game_matching_go(seed)
    eng = GoEngine(seed=seed, house_rules=_TEST_RULES)
    a0 = GoAgentState(eng, 0)
    a1 = GoAgentState(eng, 1)
    agents = {0: a0, 1: a1}

    obs_streams: Dict[int, List[Any]] = {p: [] for p in range(NUM_PLAYERS)}
    init_hands = {p: list(pygame.players[p].hand) for p in range(NUM_PLAYERS)}
    init_peeks = {p: tuple(pygame.players[p].initial_peek_indices) for p in range(NUM_PLAYERS)}
    called = False

    try:
        # Initial state: streams empty => body is exactly the init-peek frames.
        for observer in range(NUM_PLAYERS):
            go_body = agents[observer].tokens().tolist()
            py_body = _py_body(init_hands[observer], init_peeks[observer], [], observer)
            assert go_body == py_body, (
                f"seed {seed} obs {observer}: init-peek body mismatch\n"
                f"  Go={go_body}\n  Py={py_body}"
            )

        for step in range(400):
            if eng.is_terminal() or pygame.is_terminal():
                break
            go_set = _go_legal_set(eng)
            # Snap-phase agreement guard (the documented narrowing condition).
            if _is_snap_only(go_set) != bool(pygame.snap_phase_active):
                break
            common = sorted(go_set & _py_legal_set(pygame))
            if not common:
                break

            if (
                call_cambia_after >= 0
                and not called
                and step >= call_cambia_after
                and _CAMBIA_IDX in common
            ):
                action_idx = _CAMBIA_IDX
                called = True
            else:
                pool = [c for c in common if c not in (_CAMBIA_IDX, _DISCARD_WITH_ABILITY_IDX)]
                if not pool:
                    break
                action_idx = pool[0]

            # A stockpile draw from an empty stock triggers a reshuffle whose
            # order differs between the engines (pre-existing engine gap): stop
            # cleanly before it desyncs the deck.
            if action_idx == _DRAW_STOCKPILE_IDX and eng.stock_len() == 0:
                break

            actor = pygame.get_acting_player()
            py_action = None
            for a in pygame.get_legal_actions():
                if action_to_index(a) == action_idx:
                    py_action = a
                    break
            if py_action is None:
                break

            pygame.apply_action(py_action)
            apply_games_batch([eng.handle], [a0.handle], [a1.handle], [action_idx])

            snap_results = list(getattr(pygame, "snap_results_log", []) or [])
            full_obs = _create_observation(None, py_action, pygame, actor, snap_results)
            if full_obs is None:
                break
            for observer in range(NUM_PLAYERS):
                obs_streams[observer].append(_filter_observation(full_obs, observer))

            # Attribute any hidden-state divergence on this apply to the engine
            # gap, not the tokenizer: stop without asserting if state diverged.
            if _states_diverged(eng, pygame):
                break

            for observer in range(NUM_PLAYERS):
                go_body = agents[observer].tokens().tolist()
                py_body = _py_body(
                    init_hands[observer], init_peeks[observer], obs_streams[observer], observer
                )
                assert go_body == py_body, (
                    f"seed {seed} obs {observer} step {step}: token body mismatch\n"
                    f"  Go ({len(go_body)})={go_body}\n  Py ({len(py_body)})={py_body}"
                )
                res.max_body = max(res.max_body, len(go_body))
                if action_idx == _CAMBIA_IDX:
                    res.saw_cambia_frame = True
                if snap_results:
                    res.saw_snap_frame = True
            res.compared += 1
    finally:
        res.init_hands = init_hands
        res.init_peeks = init_peeks
        res.obs_streams = obs_streams
        res.go_bodies = {o: agents[o].tokens().tolist() for o in range(NUM_PLAYERS)}
        a0.close()
        a1.close()
        eng.close()
    return res


# ---------------------------------------------------------------------------
# Tests: live parity + long-game coverage + truncation helper
# ---------------------------------------------------------------------------


@skip_if_no_go
def test_live_lockstep_token_parity_full_games():
    """Go token body == Python encode_observation_sequence over 100+ live states."""
    total_compared = 0
    games_over_cap = 0
    overall_max = 0
    snap_frames = 0
    for seed in _FULL_SEEDS:
        res = _play_lockstep(seed)
        total_compared += res.compared
        overall_max = max(overall_max, res.max_body)
        if res.max_body > se.SEQ_CAP:
            games_over_cap += 1
        if res.saw_snap_frame:
            snap_frames += 1
    print(
        f"\n[token parity] seeds={len(_FULL_SEEDS)} compared_states={total_compared} "
        f"max_body={overall_max} (SEQ_CAP={se.SEQ_CAP}) games_over_cap={games_over_cap} "
        f"games_with_snap_frames={snap_frames}"
    )
    assert total_compared >= 100, (
        f"only {total_compared} live states compared byte-for-byte (need >= 100)"
    )
    # Full-length coverage: streams must exceed the 256 cap so the long-game /
    # truncation-boundary paths are exercised (mean natural length ~726).
    assert overall_max > se.SEQ_CAP, (
        f"max token body {overall_max} never exceeded SEQ_CAP={se.SEQ_CAP}"
    )
    assert games_over_cap >= 5, (
        f"only {games_over_cap} games exceeded the cap; long-game coverage too thin"
    )
    assert snap_frames > 0, "no snap frames exercised in any live game"


@skip_if_no_go
def test_cambia_frame_lockstep_byte_parity():
    """The cambia frame is emitted and byte-matches Python across live games."""
    covered = 0
    for seed in _FULL_SEEDS[:20]:
        res = _play_lockstep(seed, call_cambia_after=18)
        if res.saw_cambia_frame:
            covered += 1
    assert covered >= 3, f"cambia frame covered in only {covered} games"


@skip_if_no_go
@pytest.mark.parametrize("cap", [256, 2048])
def test_frame_aligned_window_matches_python(cap):
    """bridge.frame_aligned_window(go_body, cap) == encode_observation_sequence(cap)."""
    checked = 0
    over_cap_seen = 0
    for seed in _FULL_SEEDS[:20]:
        res = _play_lockstep(seed)
        for observer in range(NUM_PLAYERS):
            go_body = res.go_bodies[observer]
            windowed = frame_aligned_window(
                go_body,
                cap,
                add_bos_eos=True,
                frame_base=se.FRAME_BASE,
                num_frame_ids=se.NUM_FRAME_IDS,
                bos_id=se.BOS_ID,
                eos_id=se.EOS_ID,
            )
            py_seq = se.encode_observation_sequence(
                res.init_hands[observer],
                res.init_peeks[observer],
                res.obs_streams[observer],
                observer,
                seq_cap=cap,
                add_bos_eos=True,
            )
            assert windowed == py_seq, (
                f"seed {seed} obs {observer} cap {cap}: frame_aligned_window mismatch"
            )
            assert len(windowed) <= cap
            if len(go_body) > cap:
                over_cap_seen += 1
            checked += 1
    assert checked > 0
    if cap == 256:
        assert over_cap_seen > 0, "truncation branch never exercised at cap=256"


@skip_if_no_go
def test_go_snap_frames_decode_via_python_codec():
    """Go-emitted snap frames (success/penalty/fail + slots) decode via Python.

    Real snaps diverge the two engines (pre-existing engine snap gap), so instead
    of lockstep this drives a Go-ONLY game with random snapping (no divergence
    possible with one engine), then decodes the Go token stream with the Python
    codec (se.decode_sequence) and asserts the snap events are well-formed. This
    is the Go-emit -> Python-decode cross-check for the snap-success/penalty
    frames that lockstep cannot reach.
    """
    outcomes_seen = set()
    slots_seen = set()
    for seed in range(60):
        rng = random.Random(50_000 + seed)
        eng = GoEngine(seed=seed, house_rules=_TEST_RULES)
        a0 = GoAgentState(eng, 0)
        a1 = GoAgentState(eng, 1)
        try:
            for _ in range(300):
                if eng.is_terminal():
                    break
                legal = sorted(_go_legal_set(eng) - {_CAMBIA_IDX})
                if not legal:
                    break
                idx = rng.choice(legal)
                apply_games_batch([eng.handle], [a0.handle], [a1.handle], [idx])
            # Decode both players' Go token bodies with the Python codec.
            for ag in (a0, a1):
                body = ag.tokens().tolist()
                events = se.decode_sequence(body)  # raises on a malformed stream
                for ev in events:
                    if ev.kind == "snap":
                        outcomes_seen.add(ev.snap_outcome)
                        slots_seen.add(ev.snap_slot)
        finally:
            a0.close()
            a1.close()
            eng.close()
    # The Python codec decoded every Go stream without error, and the driver hit
    # a range of snap outcomes including at least one concrete-slot success.
    assert "fail" in outcomes_seen or "penalty" in outcomes_seen, outcomes_seen
    assert any(o in outcomes_seen for o in ("success_own", "success_opp")), (
        f"no successful-snap frame emitted/decoded; outcomes={outcomes_seen}"
    )
    assert any(s is not None and s >= 0 for s in slots_seen), (
        f"no concrete snap slot decoded; slots={slots_seen}"
    )


@skip_if_no_go
def test_state_save_restore_round_trip_tokens():
    """cambia_state_save/restore round-trips game + both agents' token streams."""
    seed = 3
    pygame = _setup_python_game_matching_go(seed)
    eng = GoEngine(seed=seed, house_rules=_TEST_RULES)
    a0 = GoAgentState(eng, 0)
    a1 = GoAgentState(eng, 1)

    def step_n(n: int) -> int:
        applied = 0
        for _ in range(n):
            if eng.is_terminal() or pygame.is_terminal():
                break
            go_set = _go_legal_set(eng)
            if _is_snap_only(go_set) != bool(pygame.snap_phase_active):
                break
            common = sorted(go_set & _py_legal_set(pygame))
            pool = [c for c in common if c not in (_CAMBIA_IDX, _DISCARD_WITH_ABILITY_IDX)]
            if not pool:
                break
            idx = pool[0]
            if idx == _DRAW_STOCKPILE_IDX and eng.stock_len() == 0:
                break
            py_action = next(
                a for a in pygame.get_legal_actions() if action_to_index(a) == idx
            )
            pygame.apply_action(py_action)
            apply_games_batch([eng.handle], [a0.handle], [a1.handle], [idx])
            if _states_diverged(eng, pygame):
                break
            applied += 1
        return applied

    try:
        assert step_n(14) >= 8
        saved0 = a0.tokens().tolist()
        saved1 = a1.tokens().tolist()
        saved_len0 = a0.token_len()
        assert saved_len0 > 0
        snap = state_save(eng.handle, a0.handle, a1.handle)
        try:
            assert step_n(8) >= 1  # mutate game + token streams
            assert a0.tokens().tolist() != saved0

            state_restore(eng.handle, snap, a0.handle, a1.handle)
            assert a0.tokens().tolist() == saved0, "player-0 token stream not restored"
            assert a1.tokens().tolist() == saved1, "player-1 token stream not restored"
            assert a0.token_len() == saved_len0
        finally:
            state_snapshot_free(snap)
    finally:
        a0.close()
        a1.close()
        eng.close()
