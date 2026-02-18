"""
src/ffi/bridge.py

Python ctypes wrapper around libcambia.so, providing GoEngine and
GoAgentState as drop-in replacements for CambiaGameState and AgentState.

The shared library is loaded once at module import time.
"""

import ctypes
import os
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Library loading — module-level singleton
# ---------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None


def _load_library() -> ctypes.CDLL:
    """Load libcambia.so, trying several search paths in order."""
    candidates = []

    # 1. Explicit env var
    env_path = os.environ.get("LIBCAMBIA_PATH")
    if env_path:
        candidates.append(Path(env_path))

    # 2. Relative to this file: cfr/src/ffi/bridge.py → cfr/libcambia.so
    candidates.append(Path(__file__).resolve().parent.parent.parent / "libcambia.so")

    # 3. Current working directory
    candidates.append(Path.cwd() / "libcambia.so")

    for path in candidates:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            _setup_signatures(lib)
            return lib

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"libcambia.so not found. Searched:\n  {searched}\n"
        "Set LIBCAMBIA_PATH to the correct location."
    )


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Configure argtypes and restype for every exported C function."""

    # --- Game lifecycle ---
    lib.cambia_game_new.argtypes = [ctypes.c_uint64]
    lib.cambia_game_new.restype = ctypes.c_int32

    lib.cambia_game_new_with_rules.argtypes = [
        ctypes.c_uint64,   # seed
        ctypes.c_uint16,   # maxGameTurns
        ctypes.c_uint8,    # cardsPerPlayer
        ctypes.c_uint8,    # cambiaAllowedRound
        ctypes.c_uint8,    # penaltyDrawCount
        ctypes.c_uint8,    # allowDrawFromDiscard (bool as uint8)
        ctypes.c_uint8,    # allowReplaceAbilities (bool as uint8)
        ctypes.c_uint8,    # allowOpponentSnapping (bool as uint8)
        ctypes.c_uint8,    # snapRace (bool as uint8)
        ctypes.c_uint8,    # numJokers
    ]
    lib.cambia_game_new_with_rules.restype = ctypes.c_int32

    lib.cambia_game_free.argtypes = [ctypes.c_int32]
    lib.cambia_game_free.restype = None

    lib.cambia_game_apply_action.argtypes = [ctypes.c_int32, ctypes.c_uint16]
    lib.cambia_game_apply_action.restype = ctypes.c_int32

    lib.cambia_game_legal_actions.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.cambia_game_legal_actions.restype = ctypes.c_int32

    lib.cambia_game_is_terminal.argtypes = [ctypes.c_int32]
    lib.cambia_game_is_terminal.restype = ctypes.c_int32

    lib.cambia_game_get_utility.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.cambia_game_get_utility.restype = None

    lib.cambia_game_acting_player.argtypes = [ctypes.c_int32]
    lib.cambia_game_acting_player.restype = ctypes.c_uint8

    lib.cambia_game_save.argtypes = [ctypes.c_int32]
    lib.cambia_game_save.restype = ctypes.c_int32

    lib.cambia_game_restore.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.cambia_game_restore.restype = ctypes.c_int32

    lib.cambia_snapshot_free.argtypes = [ctypes.c_int32]
    lib.cambia_snapshot_free.restype = None

    # --- Agent lifecycle ---
    lib.cambia_agent_new.argtypes = [
        ctypes.c_int32,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
    ]
    lib.cambia_agent_new.restype = ctypes.c_int32

    lib.cambia_agent_free.argtypes = [ctypes.c_int32]
    lib.cambia_agent_free.restype = None

    lib.cambia_agent_clone.argtypes = [ctypes.c_int32]
    lib.cambia_agent_clone.restype = ctypes.c_int32

    lib.cambia_agent_update.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.cambia_agent_update.restype = ctypes.c_int32

    lib.cambia_agent_encode.argtypes = [
        ctypes.c_int32,
        ctypes.c_uint8,
        ctypes.c_int8,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.cambia_agent_encode.restype = ctypes.c_int32

    # --- Drawn card bucket ---
    lib.cambia_game_get_drawn_card_bucket.argtypes = [ctypes.c_int32]
    lib.cambia_game_get_drawn_card_bucket.restype = ctypes.c_int8

    # --- Utilities ---
    lib.cambia_game_turn_number.argtypes = [ctypes.c_int32]
    lib.cambia_game_turn_number.restype = ctypes.c_uint16

    lib.cambia_game_stock_len.argtypes = [ctypes.c_int32]
    lib.cambia_game_stock_len.restype = ctypes.c_uint8

    lib.cambia_agent_action_mask.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.cambia_agent_action_mask.restype = ctypes.c_int32


def _get_lib() -> ctypes.CDLL:
    """Return the module-level library singleton, loading it on first call."""
    global _LIB
    if _LIB is None:
        _LIB = _load_library()
    return _LIB


# ---------------------------------------------------------------------------
# GoEngine
# ---------------------------------------------------------------------------


class GoEngine:
    """
    Drop-in replacement for CambiaGameState using the Go engine via FFI.

    Manages a single game handle and delegates all operations to libcambia.so.
    Supports the context-manager protocol and is safe to close multiple times.
    """

    INPUT_DIM: int = 222
    NUM_ACTIONS: int = 146

    def __init__(self, seed: Optional[int] = None, house_rules=None) -> None:
        self._lib = _get_lib()
        self._closed = False

        if seed is None:
            seed = random.getrandbits(64)

        if house_rules is not None:
            self._game_h: int = self._lib.cambia_game_new_with_rules(
                ctypes.c_uint64(seed),
                ctypes.c_uint16(getattr(house_rules, "max_game_turns", 46)),
                ctypes.c_uint8(getattr(house_rules, "cards_per_player", 4)),
                ctypes.c_uint8(getattr(house_rules, "cambia_allowed_round", 0)),
                ctypes.c_uint8(getattr(house_rules, "penaltyDrawCount", 2)),
                ctypes.c_uint8(1 if getattr(house_rules, "allowDrawFromDiscardPile", False) else 0),
                ctypes.c_uint8(1 if getattr(house_rules, "allowReplaceAbilities", False) else 0),
                ctypes.c_uint8(1 if getattr(house_rules, "allowOpponentSnapping", False) else 0),
                ctypes.c_uint8(1 if getattr(house_rules, "snapRace", False) else 0),
                ctypes.c_uint8(getattr(house_rules, "use_jokers", 2)),
            )
        else:
            warnings.warn(
                "GoEngine() without house_rules uses Go defaults which differ "
                "from Python defaults (AllowDrawFromDiscard=true, "
                "AllowOpponentSnapping=true). Pass house_rules explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._game_h: int = self._lib.cambia_game_new(ctypes.c_uint64(seed))

        if self._game_h < 0:
            raise RuntimeError(
                f"cambia_game_new failed (returned {self._game_h}). "
                "Handle pool may be exhausted."
            )

    # --- Properties ---

    @property
    def handle(self) -> int:
        """Expose the raw game handle for agent construction."""
        return self._game_h

    # --- Core API ---

    def legal_actions_mask(self) -> np.ndarray:
        """
        Return a (146,) uint8 numpy array where 1 = legal action.

        Uses cambia_agent_action_mask (takes game handle, not agent handle).
        """
        buf = (ctypes.c_uint8 * self.NUM_ACTIONS)()
        ret = self._lib.cambia_agent_action_mask(
            ctypes.c_int32(self._game_h),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_action_mask failed (returned {ret}) on handle {self._game_h}"
            )
        return np.frombuffer(buf, dtype=np.uint8).copy()

    def apply_action(self, action_idx: int) -> None:
        """
        Apply an action by its integer index in [0, 146).

        Raises:
            ValueError: If action_idx is out of range.
            RuntimeError: If the engine rejects the action.
        """
        if not (0 <= action_idx < self.NUM_ACTIONS):
            raise ValueError(
                f"action_idx {action_idx} out of range [0, {self.NUM_ACTIONS})"
            )
        ret = self._lib.cambia_game_apply_action(
            ctypes.c_int32(self._game_h),
            ctypes.c_uint16(action_idx),
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_game_apply_action failed (returned {ret}) "
                f"for action {action_idx} on handle {self._game_h}"
            )

    def save(self) -> int:
        """
        Save the current game state to a snapshot.

        Returns:
            Snapshot handle (>= 0). Must be released with free_snapshot().

        Raises:
            RuntimeError: If the snapshot pool is exhausted.
        """
        snap_h = self._lib.cambia_game_save(ctypes.c_int32(self._game_h))
        if snap_h < 0:
            raise RuntimeError(
                f"cambia_game_save failed (returned {snap_h}) on handle {self._game_h}"
            )
        return int(snap_h)

    def restore(self, snap_h: int) -> None:
        """
        Restore the game state from a previously saved snapshot.

        Args:
            snap_h: Snapshot handle returned by save().

        Raises:
            RuntimeError: On engine error.
        """
        ret = self._lib.cambia_game_restore(
            ctypes.c_int32(self._game_h),
            ctypes.c_int32(snap_h),
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_game_restore failed (returned {ret}) "
                f"game={self._game_h} snap={snap_h}"
            )

    def free_snapshot(self, snap_h: int) -> None:
        """
        Release a snapshot handle.

        Args:
            snap_h: Snapshot handle to free.
        """
        self._lib.cambia_snapshot_free(ctypes.c_int32(snap_h))

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        ret = self._lib.cambia_game_is_terminal(ctypes.c_int32(self._game_h))
        if ret < 0:
            raise RuntimeError(
                f"cambia_game_is_terminal failed (returned {ret}) on handle {self._game_h}"
            )
        return bool(ret)

    def get_utility(self) -> np.ndarray:
        """
        Return a (2,) float32 array of utilities for both players.

        Only meaningful after the game is terminal.
        """
        buf = (ctypes.c_float * 2)()
        self._lib.cambia_game_get_utility(
            ctypes.c_int32(self._game_h),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
        )
        return np.frombuffer(buf, dtype=np.float32).copy()

    def acting_player(self) -> int:
        """
        Return the index (0 or 1) of the currently acting player.

        Returns:
            0 or 1.

        Raises:
            RuntimeError: If the sentinel error value (255) is returned.
        """
        result = self._lib.cambia_game_acting_player(ctypes.c_int32(self._game_h))
        if result == 255:
            raise RuntimeError(
                f"cambia_game_acting_player returned error sentinel 255 "
                f"on handle {self._game_h}"
            )
        return int(result)

    def turn_number(self) -> int:
        """Return the current turn number (starts at 0)."""
        return int(self._lib.cambia_game_turn_number(ctypes.c_int32(self._game_h)))

    def stock_len(self) -> int:
        """Return the number of cards remaining in the stockpile."""
        return int(self._lib.cambia_game_stock_len(ctypes.c_int32(self._game_h)))

    def get_drawn_card_bucket(self) -> int:
        """
        Return the card bucket of the pending drawn card, or -1 if none.

        Only returns a valid bucket (>= 0) when the game has a pending
        discard decision (i.e., a card has been drawn but not yet played).
        """
        return int(self._lib.cambia_game_get_drawn_card_bucket(ctypes.c_int32(self._game_h)))

    # --- Lifecycle ---

    def close(self) -> None:
        """Free the game handle. Safe to call multiple times."""
        if not self._closed and self._game_h >= 0:
            self._lib.cambia_game_free(ctypes.c_int32(self._game_h))
            self._game_h = -1
            self._closed = True

    def __enter__(self) -> "GoEngine":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# GoAgentState
# ---------------------------------------------------------------------------


class GoAgentState:
    """
    Drop-in replacement for AgentState using the Go engine via FFI.

    Tracks belief state for a single player and produces tensor encodings
    for use in Deep CFR training.
    """

    def __init__(
        self,
        engine: GoEngine,
        player_id: int,
        memory_level: int = 0,
        time_decay_turns: int = 0,
    ) -> None:
        self._lib = _get_lib()
        self._closed = False

        self._agent_h: int = self._lib.cambia_agent_new(
            ctypes.c_int32(engine.handle),
            ctypes.c_uint8(player_id),
            ctypes.c_uint8(memory_level),
            ctypes.c_uint8(time_decay_turns),
        )
        if self._agent_h < 0:
            raise RuntimeError(
                f"cambia_agent_new failed (returned {self._agent_h}) "
                f"for player {player_id}. Handle pool may be exhausted."
            )

    @classmethod
    def _from_handle(cls, agent_h: int) -> "GoAgentState":
        """
        Construct a GoAgentState wrapping an existing agent handle.

        Used by clone() to avoid calling cambia_agent_new again.
        """
        obj = object.__new__(cls)
        obj._lib = _get_lib()
        obj._agent_h = agent_h
        obj._closed = False
        return obj

    # --- Core API ---

    def update(self, engine: GoEngine) -> None:
        """
        Update agent beliefs based on the current game state.

        Args:
            engine: The GoEngine instance reflecting the latest game state.

        Raises:
            RuntimeError: On engine error.
        """
        ret = self._lib.cambia_agent_update(
            ctypes.c_int32(self._agent_h),
            ctypes.c_int32(engine.handle),
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_update failed (returned {ret}) "
                f"agent={self._agent_h} game={engine.handle}"
            )

    def clone(self) -> "GoAgentState":
        """
        Create an independent copy of this agent state.

        Returns:
            A new GoAgentState with a fresh handle mirroring this agent's
            current belief state.

        Raises:
            RuntimeError: If the agent handle pool is exhausted.
        """
        new_h = self._lib.cambia_agent_clone(ctypes.c_int32(self._agent_h))
        if new_h < 0:
            raise RuntimeError(
                f"cambia_agent_clone failed (returned {new_h}) "
                f"on agent handle {self._agent_h}"
            )
        return GoAgentState._from_handle(int(new_h))

    def encode(self, decision_context: int, drawn_bucket: int = -1) -> np.ndarray:
        """
        Encode agent belief state as a 222-dimensional float32 feature vector.

        Args:
            decision_context: Integer encoding of the current decision context.
            drawn_bucket: Bucket index of the drawn card, or -1 if none.

        Returns:
            np.ndarray of shape (222,) and dtype float32.

        Raises:
            RuntimeError: On engine error.
        """
        buf = (ctypes.c_float * 222)()
        ret = self._lib.cambia_agent_encode(
            ctypes.c_int32(self._agent_h),
            ctypes.c_uint8(decision_context),
            ctypes.c_int8(drawn_bucket),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_encode failed (returned {ret}) "
                f"on agent handle {self._agent_h}"
            )
        return np.frombuffer(buf, dtype=np.float32).copy()

    # --- Lifecycle ---

    def close(self) -> None:
        """Free the agent handle. Safe to call multiple times."""
        if not self._closed and self._agent_h >= 0:
            self._lib.cambia_agent_free(ctypes.c_int32(self._agent_h))
            self._agent_h = -1
            self._closed = True

    def __enter__(self) -> "GoAgentState":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
