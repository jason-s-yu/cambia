"""tests/test_parity_gate.py

The named Go-vs-Python lockstep parity gate (cambia-1234). Run it with:

    make parity-gate

Go (engine/) is the RULES.md reference implementation. The Python engine
(cfr/src/game/) is the reference mirror being retired under cambia-1424. This
gate is the MIGRATION ACCEPTANCE for that retirement: while both engines
exist, it holds them to identical numbers. It is not meant to be a permanent
CI job, and it retires with cambia-1430.

The bar is the cambia-225 40-seed lockstep. Over every seed in
tests/parity_seeds.PARITY_SEEDS the two engines play the same game in
lockstep, and after every action these are asserted equal:

  * acting player
  * legal-action set
  * pile lengths (stockpile length, discard-pile length, discard-top bucket)
  * hands (every player's slots, card bucket by card bucket)
  * token stream (each observer's full event-stream body, byte for byte)
  * terminal utilities (once both engines end)

The first divergence fails the gate, naming the seed, the step, and which of
those comparisons broke. The per-step machinery lives in
test_token_stream_parity._play_lockstep; this module is the gate that drives
it over the declared seed set at each table size and reports coverage.

FOUR-SEAT LEG: currently BLOCKED on the Python side by cambia-1419. See
test_parity_gate_four_seats -- the leg announces itself as a named skip
rather than passing silently, and turns into a FAILURE the moment the
blocker clears, so it cannot rot into a permanent green.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card import Card
from src.constants import JOKER_RANK_STR
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState

try:
    from tests.parity_seeds import (
        PARITY_SEED_COUNT,
        PARITY_SEED_COUNT_DEFAULT,
        PARITY_SEEDS,
    )
    from tests.test_cross_engine_samples import (
        XorShift64,
        _TEST_RULES,
        go_available,
        skip_if_no_go,
    )
    from tests.test_token_stream_parity import _play_lockstep
except ImportError:  # pragma: no cover - path fallback
    from parity_seeds import (  # type: ignore
        PARITY_SEED_COUNT,
        PARITY_SEED_COUNT_DEFAULT,
        PARITY_SEEDS,
    )
    from test_cross_engine_samples import (  # type: ignore
        XorShift64,
        _TEST_RULES,
        go_available,
        skip_if_no_go,
    )
    from test_token_stream_parity import _play_lockstep  # type: ignore

if go_available:
    from src.ffi.bridge import GoEngine

# The ticket that blocks the 4-seat leg, and the two source lines that cause
# it. Named here so the skip message points at code, not at folklore.
_FOUR_SEAT_BLOCKER = "cambia-1419"
_BLOCKER_RAISE_SITE = "cfr/src/game/_query_mixin.py get_opponent_index()"
_BLOCKER_SWALLOW_SITE = "cfr/src/game/_ability_mixin.py (bare `except Exception`)"

_FOUR_SEATS = 4


_BANNER_WIDTH = 78


def _banner(lines: List[str]) -> None:
    """Print a framed block that survives pytest's -q output, so a skipped or
    degraded leg is visible in the gate log rather than being one 's'."""
    print("\n" + "=" * _BANNER_WIDTH)
    for line in lines:
        if not line:
            print()
            continue
        indent = " " * (len(line) - len(line.lstrip()))
        for wrapped in textwrap.wrap(
            line.strip(),
            width=_BANNER_WIDTH - 4 - len(indent),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]:
            print("  " + indent + wrapped)
    print("=" * _BANNER_WIDTH)


# ---------------------------------------------------------------------------
# The declared seed constant
# ---------------------------------------------------------------------------


def test_parity_gate_seed_constant():
    """The gate's breadth is one declared, env-overridable constant."""
    assert PARITY_SEED_COUNT_DEFAULT == 40, (
        "the declared default is the cambia-225 40-seed bar; changing it "
        "changes the gate's acceptance breadth"
    )
    assert PARITY_SEED_COUNT >= 1
    assert PARITY_SEEDS == tuple(range(PARITY_SEED_COUNT))
    # Every cross-engine test must draw from this one constant, so the sweep
    # the gate reports is the sweep those tests actually ran.
    from tests import test_token_stream_parity as tsp

    assert list(tsp._FULL_SEEDS) == list(PARITY_SEEDS)


# ---------------------------------------------------------------------------
# Two-seat leg: strict
# ---------------------------------------------------------------------------


@skip_if_no_go
def test_parity_gate_two_seats():
    """Go == Python at 2 seats, in lockstep, over every declared seed.

    _play_lockstep asserts each comparison inline, so the first divergence
    raises here naming the seed, the step, and the comparison that broke.
    """
    states = 0
    terminal_games = 0
    utility_games = 0
    max_body = 0
    snap_games = 0

    for seed in PARITY_SEEDS:
        res = _play_lockstep(seed)
        states += res.compared
        max_body = max(max_body, res.max_body)
        if res.utilities_compared:
            terminal_games += 1
            utility_games += 1
        if res.saw_snap_frame:
            snap_games += 1

    print(
        f"\n[parity-gate 2p] seeds={len(PARITY_SEEDS)} states={states} "
        f"terminal_games={terminal_games} utility_compares={utility_games} "
        f"snap_games={snap_games} max_token_body={max_body}"
    )

    # Coverage floors: a gate that compared nothing must not read as green.
    assert states >= 100, (
        f"only {states} lockstep states compared across {len(PARITY_SEEDS)} seeds; "
        "the gate is not actually exercising the engines"
    )
    assert utility_games > 0, (
        "no seed reached a terminal state in both engines, so terminal "
        "utilities were never compared"
    )
    assert snap_games > 0, "no seed exercised a snap resolution"


# ---------------------------------------------------------------------------
# Four-seat leg: blocked on the Python side (cambia-1419)
# ---------------------------------------------------------------------------


def _setup_python_game_n(seed: int, num_players: int) -> CambiaGameState:
    """Build a Python game whose deal matches Go's Deal() at N seats.

    Mirrors engine/game.go Deal(): one Fisher-Yates pass over the 54-card
    deck with Go's xorshift64, cards dealt round-robin p0..p(n-1) per slot,
    the top card flipped to the discard, then the starting player drawn.
    """
    rng = XorShift64(seed)
    deck: List[Card] = [
        Card(rank=rank, suit=suit)
        for suit in ["H", "D", "C", "S"]
        for rank in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    ]
    deck.append(Card(rank=JOKER_RANK_STR))
    deck.append(Card(rank=JOKER_RANK_STR))
    rng.shuffle(deck)

    hands: List[List[Card]] = [[] for _ in range(num_players)]
    for _ in range(_TEST_RULES.cards_per_player):
        for p in range(num_players):
            hands[p].append(deck.pop())

    discard_card = deck.pop()
    starting_player = int(rng.rand_n(num_players))

    players = [
        PlayerState(hand=hands[p], initial_peek_indices=(0, 1))
        for p in range(num_players)
    ]
    return CambiaGameState(
        players=players,
        stockpile=deck,
        discard_pile=[discard_card],
        current_player_index=starting_player,
        house_rules=_TEST_RULES,
        num_players=num_players,
    )


def _python_four_seat_blocker(max_steps: int = 200) -> Optional[str]:
    """Probe whether the Python engine can still not play 4 seats.

    Returns a description of the live blocker, or None if Python now runs
    4-seat games (in which case the strict leg below should be enabled).
    """
    # Direct cause: the single-opponent query raises above 2 players.
    state = _setup_python_game_n(0, _FOUR_SEATS)
    try:
        state.get_opponent_index(0)
    except NotImplementedError as exc:
        raises = str(exc).strip()
    else:
        raises = ""

    # Demonstrated consequence: a bare `except Exception` upstream swallows
    # that raise and hands back an EMPTY legal set on a non-terminal state,
    # so the game wedges instead of failing loudly.
    wedged_at = None
    for step in range(max_steps):
        if state.is_terminal():
            break
        legal = state.get_legal_actions()
        if not legal:
            wedged_at = step
            break
        state.apply_action(sorted(legal, key=repr)[0])

    if not raises and wedged_at is None:
        return None
    detail = []
    if raises:
        detail.append(f"{_BLOCKER_RAISE_SITE} raises NotImplementedError: {raises}")
    if wedged_at is not None:
        detail.append(
            f"the 4-seat game wedges at step {wedged_at} with an empty legal "
            f"action set on a non-terminal state, because {_BLOCKER_SWALLOW_SITE} "
            "swallows that raise and returns no legal actions"
        )
    return "; ".join(detail)


@skip_if_no_go
def test_parity_gate_four_seats():
    """The 4-seat leg, or a loud named skip explaining why it cannot run.

    This test never passes quietly while blocked. Either it skips with the
    blocker named, or -- once the blocker clears -- it FAILS, telling whoever
    cleared it to turn the strict leg on. It does not silently start passing.
    """
    blocker = _python_four_seat_blocker()

    if blocker is None:
        pytest.fail(
            f"PARITY GATE: the {_FOUR_SEAT_BLOCKER} blocker appears to be CLEARED -- "
            "the Python engine now plays a 4-seat game. The 4-seat leg of this "
            "gate was skipped only because it could not run. Replace this guard "
            "with a strict 4-seat lockstep (mirror test_parity_gate_two_seats, "
            "driving Go via nplayer_legal_actions_mask/apply_nplayer_action and "
            "Python via _setup_python_game_n) and re-run `make parity-gate`."
        )

    # The gap is one-sided: prove Go plays 4 seats end to end, so the leg is
    # blocked by the engine being retired, not by the reference engine.
    go_terminal = 0
    go_seeds = list(PARITY_SEEDS)[:5]
    for seed in go_seeds:
        eng = GoEngine(seed=seed, house_rules=_TEST_RULES, num_players=_FOUR_SEATS)
        try:
            import numpy as np

            for _ in range(600):
                if eng.is_terminal():
                    break
                legal = np.where(eng.nplayer_legal_actions_mask() > 0)[0].tolist()
                if not legal:
                    break
                eng.apply_nplayer_action(legal[0])
            if eng.is_terminal():
                go_terminal += 1
                assert len(eng.get_nplayer_utility()) == _FOUR_SEATS
        finally:
            eng.close()

    assert go_terminal == len(go_seeds), (
        f"the Go engine failed to finish a 4-seat game on "
        f"{len(go_seeds) - go_terminal}/{len(go_seeds)} seeds; the 4-seat gap is "
        "supposed to be Python-only"
    )

    _banner(
        [
            f"PARITY GATE: 4-SEAT LEG SKIPPED -- BLOCKED ON {_FOUR_SEAT_BLOCKER.upper()}",
            "",
            "The Go engine plays 4-seat games end to end "
            f"({go_terminal}/{len(go_seeds)} seeds terminal, 4 utilities each).",
            "The Python reference engine cannot, so there is nothing to compare:",
            f"  {blocker}",
            "",
            "2-seat parity is still enforced strictly and is NOT affected.",
            f"This leg fails (not skips) as soon as {_FOUR_SEAT_BLOCKER} is fixed.",
        ]
    )
    pytest.skip(
        f"4-seat parity leg BLOCKED on {_FOUR_SEAT_BLOCKER}: the Python engine "
        f"cannot play 4 seats ({blocker}). Go plays 4 seats fine "
        f"({go_terminal}/{len(go_seeds)} seeds terminal). 2-seat parity is "
        "unaffected and still strict."
    )
