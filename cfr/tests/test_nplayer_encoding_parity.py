"""
tests/test_nplayer_encoding_parity.py

Cross-backend parity for the 936-dim N-player encoder (cambia-1750).

engine/agent/encoding.go's EncodeNPlayer (landed cambia-1551) is mirrored by hand in
cfr/src/encoding.py's encode_infoset_nplayer. Only the output dimension was
cross-checked through the FFI (test_eight_players.py's
test_go_python_input_dim_parity); a layout divergence between the two encoders --
an offset off by one block, a seat-order mismatch, a masking rule that drifted --
would go unnoticed until something consumed both.

There is no Python N-player AgentState that ingests real game observations the way
GoAgentState.update_nplayer does: src/agent_state.py's nplayer_* methods
(nplayer_record_peek/reveal/swap) are a manual bookkeeping API with no counterpart to
Go's UpdateNPlayer, in particular no equivalent of the slot-shift-on-snap logic
(nplayerRemoveSlot/nplayerInsertSlot, cambia-1550) that keeps tracked knowledge
aligned with a hand that grew or shrank. Reimplementing that here would duplicate
production belief-tracking logic under active development elsewhere (cambia-1548,
cambia-1552, cambia-1553) and is out of this ticket's scope.

Instead this drives real 4-seat games through the Go engine and the real
cambia_agent_new_nplayer / cambia_agent_encode_nplayer FFI path (GoEngine +
GoAgentState, matching the exact pattern src/cfr/deep_worker.py's
_deep_traverse_os_go_nplayer uses in production), and treats each resulting Go
vector as ground truth. At every decision point the vector is decoded back into the
facts it encodes -- per-seat hand lengths, which players know which slot, known
slot buckets, and the public sub-fields -- using ONLY the documented, fixed byte
layout (the block boundaries in engine/agent/encoding.go's and
cfr/src/encoding.py's docstrings), not encode_infoset_nplayer's own code. Feeding
those decoded facts back into encode_infoset_nplayer and asserting the result
equals the original Go vector element-for-element is therefore a genuine layout
parity check, not a round trip through the same logic: a Python offset bug changes
what the *re-encode* writes, not what the *decode* reads, so the comparison still
catches it.

This exercises the encoder LAYOUT (offsets, relative seat ordering, hand-length
masking, public sub-fields) across many real, naturally varying decision points; it
does not exercise Go's own belief-tracking correctness (that a card is *marked*
known/unknown is Go's decision and is taken as given here), and it does not exercise
a from-scratch Python belief tracker, since none suitable exists.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

from src.constants import (
    N_PLAYER_HAND_LEN_DIM,
    N_PLAYER_IDENTITY_DIM,
    N_PLAYER_MAX_PLAYERS,
    N_PLAYER_POWERSET_DIM,
    N_PLAYER_PUBLIC_DIM,
    N_PLAYER_SEAT_BLOCK_DIM,
    N_PLAYER_SEAT_COUNT_DIM,
    N_PLAYER_SEAT_DIM,
)
from src.encoding import MAX_HAND, encode_infoset_nplayer, nplayer_seat_order

NUM_SEEDS = 20
NUM_PLAYERS = 4
MAX_STEPS_PER_GAME = 250


def _go_available() -> bool:
    try:
        from src.config import CambiaRulesConfig  # noqa: PLC0415
        from src.ffi.bridge import GoEngine  # noqa: PLC0415

        rules = CambiaRulesConfig()
        eng = GoEngine(seed=0, house_rules=rules, num_players=NUM_PLAYERS)
        eng.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(not go_available, reason="libcambia.so not available")


# ---------------------------------------------------------------------------
# Named blocks (AC2: a divergence reports the block, not a bare index).
# ---------------------------------------------------------------------------

_OWN_SEAT_BASE = N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM + N_PLAYER_PUBLIC_DIM
_SEAT_COUNT_BASE = _OWN_SEAT_BASE + N_PLAYER_SEAT_DIM
_TABLE_BASE = _SEAT_COUNT_BASE + N_PLAYER_SEAT_COUNT_DIM

_NAMED_BLOCKS: List[Tuple[str, int, int]] = [
    ("powerset knower axis", 0, N_PLAYER_POWERSET_DIM),
    ("slot blocks", N_PLAYER_POWERSET_DIM, N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM),
    ("public", N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM, _OWN_SEAT_BASE),
    ("own seat", _OWN_SEAT_BASE, _SEAT_COUNT_BASE),
    ("seat count", _SEAT_COUNT_BASE, _TABLE_BASE),
    (
        "hand lengths",
        _TABLE_BASE,
        _TABLE_BASE + N_PLAYER_SEAT_DIM * N_PLAYER_SEAT_BLOCK_DIM,
    ),
]


def _block_name(idx: int) -> str:
    for name, lo, hi in _NAMED_BLOCKS:
        if lo <= idx < hi:
            return name
    return f"out-of-range index {idx}"


def _assert_vectors_equal(go_vec: np.ndarray, py_vec: np.ndarray, where: str) -> None:
    diff = np.nonzero(go_vec != py_vec)[0]
    if diff.size == 0:
        return
    blocks = sorted({_block_name(int(i)) for i in diff})
    sample = [(int(i), float(go_vec[i]), float(py_vec[i])) for i in diff[:8]]
    pytest.fail(
        f"{where}: N-player encoding diverged in block(s) {blocks} "
        f"({diff.size} of {go_vec.shape[0]} dims differ); "
        f"first mismatches (index, go, python) = {sample}"
    )


# ---------------------------------------------------------------------------
# Decode: invert the documented Go layout to recover the facts it wrote.
# ---------------------------------------------------------------------------


def _decode_go_nplayer_vector(
    vec: np.ndarray, encoding_player: int, num_players: int
) -> Tuple[
    Dict[Tuple[int, int], Set[int]],
    Dict[Tuple[int, int], int],
    List[int],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Recover (knowledge_masks, slot_buckets, hand_lens, discard, stock, phase,
    ctx, cambia_state, drawn_bucket) from a real Go-encoded vector, using only the
    fixed block layout both encoders document.
    """
    order = nplayer_seat_order(encoding_player, num_players)

    # -- table block: own seat / seat count / per-seat hand length --
    own_seat = int(np.argmax(vec[_OWN_SEAT_BASE : _OWN_SEAT_BASE + N_PLAYER_SEAT_DIM]))
    seats = (
        int(np.argmax(vec[_SEAT_COUNT_BASE : _SEAT_COUNT_BASE + N_PLAYER_SEAT_COUNT_DIM]))
        + 1
    )
    assert own_seat == encoding_player, f"own seat decode {own_seat} != {encoding_player}"
    assert seats == num_players, f"seat count decode {seats} != {num_players}"

    hand_lens = [0] * N_PLAYER_MAX_PLAYERS
    for r, seat in enumerate(order):
        if seat is None:
            continue
        base = _TABLE_BASE + r * N_PLAYER_SEAT_BLOCK_DIM
        in_play = vec[base + N_PLAYER_HAND_LEN_DIM] == 1.0
        if in_play:
            hand_lens[seat] = int(np.argmax(vec[base : base + N_PLAYER_HAND_LEN_DIM]))

    # -- powerset (knower axis) + slot identity blocks --
    knowledge_masks: Dict[Tuple[int, int], Set[int]] = {}
    slot_buckets: Dict[Tuple[int, int], int] = {}
    for r, seat in enumerate(order):
        if seat is None:
            continue
        for c in range(MAX_HAND):
            if c >= hand_lens[seat]:
                continue
            pbase = (r * MAX_HAND + c) * N_PLAYER_MAX_PLAYERS
            bits = vec[pbase : pbase + N_PLAYER_MAX_PLAYERS]
            knowers = {
                order[k]
                for k in range(N_PLAYER_MAX_PLAYERS)
                if bits[k] == 1.0 and order[k] is not None
            }
            if knowers:
                knowledge_masks[(seat, c)] = knowers  # type: ignore[arg-type]

            ibase = N_PLAYER_POWERSET_DIM + (r * MAX_HAND + c) * 9
            ident = vec[ibase : ibase + 9]
            if ident.any():
                slot_buckets[(seat, c)] = int(np.argmax(ident))

    # -- public sub-fields --
    off = N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM
    discard = int(np.argmax(vec[off : off + 10]))
    off += 10
    stock = int(np.argmax(vec[off : off + 4]))
    off += 4
    phase = int(np.argmax(vec[off : off + 6]))
    off += 6
    ctx = int(np.argmax(vec[off : off + 6]))
    off += 6
    cambia_state = int(np.argmax(vec[off : off + 3]))
    off += 3
    drawn_idx = int(np.argmax(vec[off : off + 11]))
    drawn_bucket = -1 if drawn_idx == 10 else drawn_idx

    return (
        knowledge_masks,
        slot_buckets,
        hand_lens,
        discard,
        stock,
        phase,
        ctx,
        cambia_state,
        drawn_bucket,
    )


def _reencode_via_python(
    decoded: tuple, encoding_player: int, num_players: int
) -> np.ndarray:
    (
        knowledge_masks,
        slot_buckets,
        hand_lens,
        discard,
        stock,
        phase,
        ctx,
        cambia_state,
        drawn_bucket,
    ) = decoded
    return encode_infoset_nplayer(
        knowledge_masks=knowledge_masks,
        slot_buckets=slot_buckets,
        encoding_player=encoding_player,
        num_players=num_players,
        discard_top_bucket=discard,
        stock_estimate=stock,
        game_phase=phase,
        decision_context=ctx,
        cambia_state=cambia_state,
        drawn_card_bucket=drawn_bucket,
        hand_lens=hand_lens,
    )


# ---------------------------------------------------------------------------
# Driver: replay a seeded game through the real FFI path, comparing at every
# decision (mirrors src/cfr/deep_worker.py's _deep_traverse_os_go_nplayer loop).
# ---------------------------------------------------------------------------


def _race_rules(snap_race: bool):
    from src.config import CambiaRulesConfig  # noqa: PLC0415

    rules = CambiaRulesConfig()
    rules.allowDrawFromDiscardPile = True
    rules.allowOpponentSnapping = True
    rules.snapRace = snap_race
    rules.max_game_turns = 40
    rules.cards_per_player = 4
    return rules


def _drive_and_compare(seed: int, snap_race: bool, num_players: int = NUM_PLAYERS) -> int:
    """Play one seeded N-player game, comparing the Go and re-encoded Python
    vectors at the acting player's every decision. Returns the number of
    decisions compared (0 means the seed produced no decision, a driver bug).
    """
    from src.ffi.bridge import GoAgentState, GoEngine  # noqa: PLC0415

    rules = _race_rules(snap_race)
    rng = random.Random((seed << 1) | int(snap_race))
    engine = GoEngine(seed=seed, house_rules=rules, num_players=num_players)
    agents = [
        GoAgentState.new_nplayer(engine, i, num_players) for i in range(num_players)
    ]
    decisions_compared = 0
    try:
        for _ in range(MAX_STEPS_PER_GAME):
            if engine.is_terminal():
                break
            legal_mask = engine.nplayer_legal_actions_mask()
            legal_indices = np.where(legal_mask > 0)[0]
            if legal_indices.size == 0:
                break

            ctx = engine.decision_ctx()
            player = engine.acting_player()
            drawn_bucket = engine.get_drawn_card_bucket() if ctx == 1 else -1

            go_vec = agents[player].encode_nplayer(ctx, drawn_bucket=drawn_bucket)
            decoded = _decode_go_nplayer_vector(go_vec, player, num_players)
            py_vec = _reencode_via_python(decoded, player, num_players)
            _assert_vectors_equal(
                go_vec,
                py_vec,
                f"seed={seed} snapRace={snap_race} step={decisions_compared} "
                f"player={player} ctx={ctx}",
            )
            decisions_compared += 1

            action = int(rng.choice(legal_indices.tolist()))
            engine.apply_nplayer_action(action)
            for a in agents:
                a.update_nplayer(engine)
    finally:
        engine.close()
        for a in agents:
            a.close()

    return decisions_compared


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_go
@pytest.mark.parametrize("snap_race", [False, True], ids=["snapRaceOff", "snapRaceOn"])
@pytest.mark.parametrize("seed", range(NUM_SEEDS))
def test_go_python_nplayer_encoding_matches_at_every_decision(seed: int, snap_race: bool):
    """The Go FFI encode and the Python mirror, fed the facts decoded from that
    same Go vector, must agree element-for-element at every decision of a real
    4-seat game (cambia-1750)."""
    decisions = _drive_and_compare(seed, snap_race)
    assert decisions > 0, (
        f"seed={seed} snapRace={snap_race}: game produced no decisions to compare "
        "(driver bug, not a parity result)"
    )
