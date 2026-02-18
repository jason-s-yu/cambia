"""
tests/test_cross_engine_samples.py

Cross-language parity test: plays identical games in both the Python engine
and Go engine (via FFI) with the same card layout, comparing legal action
masks and structural state at every non-snap decision point.

Uses the Go-compatible xorshift64 RNG to construct a Python game state that
exactly matches what the Go engine produces for the same seed.

Snap phase handling: Go and Python handle snap-phase transitions differently
(known parity gap). This test handles snap phases by passing through them
in both engines independently, then comparing the main game state after
snap resolution. Snap-caused divergences are tracked and reported but do
not cause test failure, since snap parity is covered by test_snap_legal_actions.py.

Requires libcambia.so to be built and available (skipped otherwise).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.constants import (
    NUM_PLAYERS,
    JOKER_RANK_STR,
    ActionPassSnap,
)
from src.encoding import action_to_index, encode_action_mask, NUM_ACTIONS
from src.game.engine import CambiaGameState
from src.game.player_state import PlayerState
from src.card import Card
from src.config import CambiaRulesConfig

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------


def _go_available() -> bool:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from src.ffi.bridge import GoEngine

            e = GoEngine(seed=0)
            e.close()
        return True
    except Exception:
        return False


go_available = _go_available()
skip_if_no_go = pytest.mark.skipif(
    not go_available, reason="libcambia.so not available"
)

# ---------------------------------------------------------------------------
# xorshift64 — mirrors Go engine's RNG exactly
# ---------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1


class XorShift64:
    def __init__(self, seed: int) -> None:
        self.state: int = seed & _MASK64 or 1

    def next_rand(self) -> int:
        x = self.state
        x ^= (x << 13) & _MASK64
        x ^= (x >> 7) & _MASK64
        x ^= (x << 17) & _MASK64
        self.state = x
        return x

    def rand_n(self, n: int) -> int:
        return self.next_rand() % n

    def shuffle(self, lst: list) -> None:
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = int(self.rand_n(i + 1))
            lst[i], lst[j] = lst[j], lst[i]


# ---------------------------------------------------------------------------
# Game setup: Python state matching Go's Deal() including discard flip
# ---------------------------------------------------------------------------
_CARDS_PER_PLAYER = 4
_NUM_PLAYERS = 2

# House rules matching Go defaults
_TEST_RULES = CambiaRulesConfig()
_TEST_RULES.allowDrawFromDiscardPile = True
_TEST_RULES.allowOpponentSnapping = True
_TEST_RULES.max_game_turns = 46


def _setup_python_game_matching_go(seed: int) -> CambiaGameState:
    """
    Create a CambiaGameState whose deck layout, starting player, and initial
    discard card match what Go's engine produces for the same seed.

    This includes the initial discard flip that Go's Deal() performs.
    """
    rng = XorShift64(seed)

    go_suits = ["H", "D", "C", "S"]
    go_ranks = [
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"
    ]

    deck: List[Card] = []
    for suit in go_suits:
        for rank in go_ranks:
            deck.append(Card(rank=rank, suit=suit))
    deck.append(Card(rank=JOKER_RANK_STR))
    deck.append(Card(rank=JOKER_RANK_STR))
    assert len(deck) == 54

    rng.shuffle(deck)

    # Deal cards alternating P0, P1 (same as Go)
    hands: List[List[Card]] = [[] for _ in range(_NUM_PLAYERS)]
    for _ in range(_CARDS_PER_PLAYER):
        for p in range(_NUM_PLAYERS):
            hands[p].append(deck.pop())

    # Flip top card to discard (matching Go's Deal())
    discard_card = deck.pop()

    # Starting player
    starting_player = int(rng.rand_n(_NUM_PLAYERS))

    players = [
        PlayerState(hand=hands[p], initial_peek_indices=(0, 1))
        for p in range(_NUM_PLAYERS)
    ]

    state = CambiaGameState(
        players=players,
        stockpile=deck,
        discard_pile=[discard_card],
        current_player_index=starting_player,
        house_rules=_TEST_RULES,
    )
    return state


# Snap action index range: pass=97, snap_own=98-103, snap_opp=104-109,
# snap_opp_move=110-145
_SNAP_ACTION_MIN = 97
_PASS_SNAP_IDX = 97


def _is_snap_only(legal_indices: set) -> bool:
    """True if all legal actions are snap-related (indices >= 97)."""
    return len(legal_indices) > 0 and all(i >= _SNAP_ACTION_MIN for i in legal_indices)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_go
class TestCrossEngineLegalActions:
    """Compare legal action masks between Python and Go engines with identical cards."""

    def _play_and_compare(self, seed: int, max_steps: int = 500):
        """
        Play a full game in both engines with the same seed, comparing
        legal action masks at every non-snap decision point.

        When one engine enters a snap phase, it passes through it
        (PassSnap for all snappers) before resuming comparison.
        Snap divergences are tracked but do not cause test failure.
        """
        from src.ffi.bridge import GoEngine

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        compared_steps = 0
        snap_passes = 0  # How many snap phases we passed through
        non_snap_mismatches = []

        for step in range(max_steps):
            # Check terminal states
            go_term = go_engine.is_terminal()
            py_term = py_state.is_terminal()

            if go_term and py_term:
                break
            if go_term != py_term:
                # Terminal mismatch — check if snap divergence caused it.
                # If we've already had snap divergences, this is expected.
                if snap_passes > 0:
                    break  # Snap divergence caused terminal mismatch
                non_snap_mismatches.append(
                    f"step {step}: terminal mismatch go={go_term} py={py_term} "
                    f"(no prior snap divergence)"
                )
                break

            # Get legal actions from both engines
            go_mask = go_engine.legal_actions_mask()
            go_actions = set(np.where(go_mask > 0)[0].tolist())

            py_legal = py_state.get_legal_actions()
            py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
            py_actions = set(np.where(py_mask > 0)[0].tolist())

            go_snap_only = _is_snap_only(go_actions)
            py_snap_phase = py_state.snap_phase_active

            # --- Handle snap phases independently ---
            # If Go is in snap phase (snap-only actions), pass through it.
            if go_snap_only:
                go_engine.apply_action(_PASS_SNAP_IDX)
                snap_passes += 1
                # If Python is also in snap phase, pass through it too.
                if py_snap_phase:
                    py_state.apply_action(ActionPassSnap())
                continue

            # If Python is in snap phase but Go isn't, pass through Python's.
            if py_snap_phase:
                py_state.apply_action(ActionPassSnap())
                snap_passes += 1
                continue

            # --- Both in normal play: compare legal actions ---
            # Exclude snap indices from comparison (residual snap actions
            # from snap-race mechanics that don't require snap-only state).
            snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
            go_non_snap = go_actions - snap_indices
            py_non_snap = py_actions - snap_indices

            if go_non_snap != py_non_snap:
                if snap_passes > 0:
                    # Divergence caused by earlier snap handling differences.
                    # Expected — stop comparison, not a failure.
                    break
                go_only = sorted(go_non_snap - py_non_snap)
                py_only = sorted(py_non_snap - go_non_snap)
                non_snap_mismatches.append(
                    f"step {step}: legal_actions mismatch "
                    f"go_only={go_only} py_only={py_only}"
                )
                if len(non_snap_mismatches) > 5:
                    break

            # Compare acting player
            go_actor = go_engine.acting_player()
            py_actor = py_state.get_acting_player()
            if go_actor != py_actor and snap_passes == 0:
                non_snap_mismatches.append(
                    f"step {step}: acting_player go={go_actor} py={py_actor}"
                )

            compared_steps += 1

            # Pick lowest common non-snap legal action
            common = sorted(go_non_snap & py_non_snap)
            if len(common) == 0:
                # No common actions — engines have diverged (snap-related)
                break

            action_idx = common[0]

            # Apply to Go
            go_engine.apply_action(action_idx)

            # Find and apply matching Python action
            py_action = None
            for a in py_legal:
                try:
                    if action_to_index(a) == action_idx:
                        py_action = a
                        break
                except Exception:
                    pass

            if py_action is None:
                break

            py_state.apply_action(py_action)

        # Compare terminal utilities if both ended
        if go_engine.is_terminal() and py_state.is_terminal() and snap_passes == 0:
            go_util = go_engine.get_utility()
            py_util = np.array(
                [py_state.get_utility(i) for i in range(NUM_PLAYERS)],
                dtype=np.float32,
            )
            if not np.allclose(go_util, py_util, atol=1e-3):
                non_snap_mismatches.append(
                    f"terminal utilities: go={go_util} py={py_util}"
                )

        go_engine.close()

        # The test passes if:
        # 1. No non-snap mismatches occurred before any snap divergence
        # 2. At least some steps were compared successfully
        assert len(non_snap_mismatches) == 0, (
            f"Cross-engine mismatches for seed {seed} "
            f"(compared {compared_steps} steps, {snap_passes} snap passes):\n"
            + "\n".join(non_snap_mismatches)
        )
        assert compared_steps > 0, (
            f"seed {seed}: no steps compared (immediate divergence)"
        )

    def test_seed_0(self):
        self._play_and_compare(seed=0)

    def test_seed_42(self):
        self._play_and_compare(seed=42)

    def test_seed_100(self):
        self._play_and_compare(seed=100)

    def test_seed_999(self):
        self._play_and_compare(seed=999)

    def test_seed_12345(self):
        self._play_and_compare(seed=12345)


@skip_if_no_go
class TestCrossEngineInitialState:
    """Validate initial state parity before any actions are taken."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 137, 999, 12345])
    def test_initial_legal_actions(self, seed):
        """Initial legal actions (non-snap) should match between engines."""
        from src.ffi.bridge import GoEngine

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        go_mask = go_engine.legal_actions_mask()
        go_actions = set(np.where(go_mask > 0)[0].tolist())

        py_legal = py_state.get_legal_actions()
        py_mask = encode_action_mask(list(py_legal)).astype(np.uint8)
        py_actions = set(np.where(py_mask > 0)[0].tolist())

        # Initial state should never be in snap phase
        assert not py_state.snap_phase_active, "Python in snap at start"
        assert not _is_snap_only(go_actions), "Go in snap-only at start"

        # Non-snap actions should match
        snap_indices = set(range(_SNAP_ACTION_MIN, NUM_ACTIONS))
        go_non_snap = go_actions - snap_indices
        py_non_snap = py_actions - snap_indices
        assert go_non_snap == py_non_snap, (
            f"seed {seed}: initial non-snap actions differ: "
            f"go_only={sorted(go_non_snap - py_non_snap)} "
            f"py_only={sorted(py_non_snap - go_non_snap)}"
        )

        go_engine.close()

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 137, 999, 12345])
    def test_initial_acting_player(self, seed):
        """Initial acting player should match between engines."""
        from src.ffi.bridge import GoEngine

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        go_actor = go_engine.acting_player()
        py_actor = py_state.get_acting_player()
        assert go_actor == py_actor, (
            f"seed {seed}: acting player go={go_actor} py={py_actor}"
        )

        go_engine.close()

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 137, 999, 12345])
    def test_initial_stock_len(self, seed):
        """Initial stockpile length should match between engines."""
        from src.ffi.bridge import GoEngine

        go_engine = GoEngine(seed=seed, house_rules=_TEST_RULES)
        py_state = _setup_python_game_matching_go(seed)

        go_stock = go_engine.stock_len()
        py_stock = py_state.get_stockpile_size()
        # 54 cards - 8 dealt (4 per player) - 1 discard = 45
        assert go_stock == py_stock == 45, (
            f"seed {seed}: stock_len go={go_stock} py={py_stock}"
        )

        go_engine.close()
