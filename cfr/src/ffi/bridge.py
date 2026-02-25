"""
src/ffi/bridge.py

Python cffi ABI-mode wrapper around libcambia.so, providing GoEngine and
GoAgentState as drop-in replacements for CambiaGameState and AgentState.

The shared library is loaded once at module import time.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional

import cffi
import numpy as np

from src.config import CambiaRulesConfig

# ---------------------------------------------------------------------------
# Library loading — module-level singleton
# ---------------------------------------------------------------------------

_ffi = cffi.FFI()
_ffi.cdef("""
    /* Game lifecycle */
    int32_t cambia_game_new(uint64_t seed);
    int32_t cambia_game_new_with_rules(
        uint64_t seed,
        uint16_t maxGameTurns,
        uint8_t  cardsPerPlayer,
        uint8_t  cambiaAllowedRound,
        uint8_t  penaltyDrawCount,
        uint8_t  allowDrawFromDiscard,
        uint8_t  allowReplaceAbilities,
        uint8_t  allowOpponentSnapping,
        uint8_t  snapRace,
        uint8_t  numJokers,
        uint8_t  lockCallerHand,
        uint8_t  numPlayers
    );
    void    cambia_game_free(int32_t h);
    int32_t cambia_game_apply_action(int32_t h, uint16_t action);
    int32_t cambia_game_legal_actions(int32_t h, uint64_t *out);
    int32_t cambia_game_is_terminal(int32_t h);
    void    cambia_game_get_utility(int32_t h, float *out);
    uint8_t cambia_game_acting_player(int32_t h);
    int32_t cambia_game_save(int32_t h);
    int32_t cambia_game_restore(int32_t h, int32_t snap);
    void    cambia_snapshot_free(int32_t snap);

    /* Agent lifecycle */
    int32_t cambia_agent_new(int32_t game_h, uint8_t player_id,
                             uint8_t memory_level, uint8_t time_decay_turns);
    void    cambia_agent_free(int32_t h);
    int32_t cambia_agent_clone(int32_t h);
    int32_t cambia_agent_update(int32_t agent_h, int32_t game_h);
    int32_t cambia_agent_encode(int32_t h, uint8_t decision_context,
                                int8_t drawn_bucket, float *out);
    int32_t cambia_agent_encode_eppbs(int32_t h, uint8_t decision_context,
                                      int8_t drawn_bucket, float *out);
    int32_t cambia_agent_new_with_memory(int32_t game_h, uint8_t player_id,
                                         uint8_t memory_level, uint8_t time_decay_turns,
                                         uint8_t memory_archetype, double memory_decay_lambda,
                                         uint8_t memory_capacity);
    int32_t cambia_agent_apply_decay(int32_t agent_h, int64_t rng_seed);

    /* Drawn card bucket */
    int8_t  cambia_game_get_drawn_card_bucket(int32_t h);

    /* Decision context and batch agent update */
    uint8_t cambia_game_decision_ctx(int32_t h);
    int32_t cambia_agents_update_both(int32_t a0h, int32_t a1h, int32_t gh);

    /* Utilities */
    uint16_t cambia_game_turn_number(int32_t h);
    uint8_t  cambia_game_stock_len(int32_t h);
    int32_t  cambia_agent_action_mask(int32_t h, uint8_t *out);

    /* N-Player game and agent APIs */
    void    cambia_game_get_utility_n(int32_t h, float *out, uint8_t n);
    int32_t cambia_game_nplayer_legal_actions(int32_t h, uint64_t *out);
    int32_t cambia_game_apply_nplayer_action(int32_t h, uint16_t action);
    int32_t cambia_agent_new_nplayer(int32_t game_h, uint8_t player_id,
                                     uint8_t num_players, uint8_t memory_level,
                                     uint8_t time_decay_turns);
    int32_t cambia_agent_update_nplayer(int32_t agent_h, int32_t game_h);
    int32_t cambia_agent_encode_nplayer(int32_t agent_h, uint8_t decision_ctx,
                                        int8_t drawn_bucket, float *out);
    int32_t cambia_agent_nplayer_action_mask(int32_t agent_h, int32_t game_h,
                                             uint8_t *out);

    /* Subgame solver */
    int32_t cambia_subgame_build(int32_t game_h, int32_t max_depth);
    int32_t cambia_subgame_leaf_count(int32_t solver_h);
    int32_t cambia_subgame_export_leaves(int32_t solver_h, int32_t *game_handles_out);
    int32_t cambia_subgame_solve(int32_t solver_h, int32_t num_iterations,
                                 float *leaf_values, float *strategy_out,
                                 float *root_values_out);
    void    cambia_subgame_free(int32_t solver_h);

    /* Handle pool diagnostics */
    void    cambia_handle_pool_stats(int32_t *games_out, int32_t *agents_out, int32_t *snaps_out);
""")

_LIB = None


def _load_library():
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
            return _ffi.dlopen(str(path))

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"libcambia.so not found. Searched:\n  {searched}\n"
        "Set LIBCAMBIA_PATH to the correct location."
    )


def _get_lib():
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
    N_PLAYER_INPUT_DIM: int = 580
    N_PLAYER_NUM_ACTIONS: int = 452

    def __init__(
        self,
        seed: Optional[int] = None,
        house_rules=None,
        num_players: int = 2,
    ) -> None:
        self._closed = False
        self._game_h = -1
        self._owned = True
        self._num_players = max(2, int(num_players))

        # Validate house_rules type before loading the library
        if (
            house_rules is not None
            and not isinstance(house_rules, CambiaRulesConfig)
            and type(house_rules).__name__ != "CambiaRulesConfig"
            and type(house_rules).__name__ != "_CambiaRulesConfig"
        ):
            warnings.warn(
                f"house_rules should be CambiaRulesConfig, got {type(house_rules).__name__}. "
                "Attribute errors will be raised if fields are missing.",
                UserWarning,
                stacklevel=2,
            )

        self._lib = _get_lib()

        # Pre-allocated reusable buffers
        self._mask_buf = _ffi.new("uint8_t[146]")
        self._util_buf = _ffi.new("float[2]")
        # N-player buffers (allocated lazily via properties or always for simplicity)
        self._nplayer_mask_buf = _ffi.new("uint8_t[452]")
        self._nplayer_legal_buf = _ffi.new("uint64_t[8]")
        self._nplayer_util_buf = _ffi.new(f"float[{self._num_players}]")

        if seed is None:
            seed = random.getrandbits(64)

        if house_rules is not None:
            np_val = getattr(house_rules, "num_players", self._num_players)
            self._game_h: int = self._lib.cambia_game_new_with_rules(
                seed,
                house_rules.max_game_turns,
                house_rules.cards_per_player,
                house_rules.cambia_allowed_round,
                house_rules.penaltyDrawCount,
                1 if house_rules.allowDrawFromDiscardPile else 0,
                1 if house_rules.allowReplaceAbilities else 0,
                1 if house_rules.allowOpponentSnapping else 0,
                1 if house_rules.snapRace else 0,
                house_rules.use_jokers,
                1 if getattr(house_rules, "lockCallerHand", True) else 0,
                int(np_val),
            )
        else:
            warnings.warn(
                "GoEngine() without house_rules uses Go defaults which differ "
                "from Python defaults (AllowDrawFromDiscard=true, "
                "AllowOpponentSnapping=true). Pass house_rules explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._game_h: int = self._lib.cambia_game_new(seed)

        if self._game_h < 0:
            raise RuntimeError(
                f"cambia_game_new failed (returned {self._game_h}). "
                "Handle pool may be exhausted."
            )

    @classmethod
    def _from_handle(cls, game_h: int) -> "GoEngine":
        """Create a non-owning GoEngine view from an existing game handle.

        The returned object will NOT free the handle when closed or GC'd.
        Used by SubgameSolver.export_leaves() to wrap leaf game handles.
        """
        obj = object.__new__(cls)
        obj._lib = _get_lib()
        obj._game_h = game_h
        obj._num_players = 2
        obj._mask_buf = _ffi.new("uint8_t[146]")
        obj._util_buf = _ffi.new("float[2]")
        obj._nplayer_mask_buf = _ffi.new("uint8_t[452]")
        obj._nplayer_legal_buf = _ffi.new("uint64_t[8]")
        obj._nplayer_util_buf = _ffi.new("float[2]")
        obj._closed = False
        obj._owned = False
        return obj

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
        ret = self._lib.cambia_agent_action_mask(self._game_h, self._mask_buf)
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_action_mask failed (returned {ret}) on handle {self._game_h}"
            )
        return np.frombuffer(_ffi.buffer(self._mask_buf), dtype=np.uint8).copy()

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
        ret = self._lib.cambia_game_apply_action(self._game_h, action_idx)
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
        snap_h = self._lib.cambia_game_save(self._game_h)
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
        ret = self._lib.cambia_game_restore(self._game_h, snap_h)
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
        self._lib.cambia_snapshot_free(snap_h)

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        ret = self._lib.cambia_game_is_terminal(self._game_h)
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
        self._lib.cambia_game_get_utility(self._game_h, self._util_buf)
        return np.frombuffer(_ffi.buffer(self._util_buf), dtype=np.float32).copy()

    def acting_player(self) -> int:
        """
        Return the index (0 or 1) of the currently acting player.

        Returns:
            0 or 1.

        Raises:
            RuntimeError: If the sentinel error value (255) is returned.
        """
        result = self._lib.cambia_game_acting_player(self._game_h)
        if result == 255:
            raise RuntimeError(
                f"cambia_game_acting_player returned error sentinel 255 "
                f"on handle {self._game_h}"
            )
        return int(result)

    def turn_number(self) -> int:
        """Return the current turn number (starts at 0)."""
        return int(self._lib.cambia_game_turn_number(self._game_h))

    def stock_len(self) -> int:
        """Return the number of cards remaining in the stockpile."""
        return int(self._lib.cambia_game_stock_len(self._game_h))

    def get_drawn_card_bucket(self) -> int:
        """
        Return the card bucket of the pending drawn card, or -1 if none.

        Only returns a valid bucket (>= 0) when the game has a pending
        discard decision (i.e., a card has been drawn but not yet played).
        """
        return int(self._lib.cambia_game_get_drawn_card_bucket(self._game_h))

    def decision_ctx(self) -> int:
        """
        Return the current decision context as an integer.

        Values: 0=StartTurn, 1=PostDraw, 2=AbilitySelect,
                3=SnapDecision, 4=SnapMove, 5=Terminal.
        """
        return int(self._lib.cambia_game_decision_ctx(self._game_h))

    def update_both(self, a0: "GoAgentState", a1: "GoAgentState") -> None:
        """
        Update both agent belief states in a single FFI call.

        Args:
            a0: First agent (player 0).
            a1: Second agent (player 1).

        Raises:
            RuntimeError: On engine error.
        """
        ret = self._lib.cambia_agents_update_both(
            a0._agent_h, a1._agent_h, self._game_h
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_agents_update_both failed (returned {ret}) "
                f"a0={a0._agent_h} a1={a1._agent_h} game={self._game_h}"
            )

    # --- N-Player APIs ---

    def nplayer_legal_actions_mask(self) -> np.ndarray:
        """
        Return a (452,) uint8 numpy array where 1 = legal N-player action.

        Uses the 8×uint64 bitmask from Go, expanded to per-action bytes.
        Requires an agent handle; uses cambia_game_nplayer_legal_actions for
        the raw bitmask and returns it as a dense byte array.
        """
        ret = self._lib.cambia_game_nplayer_legal_actions(
            self._game_h, self._nplayer_legal_buf
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_game_nplayer_legal_actions failed (returned {ret}) "
                f"on handle {self._game_h}"
            )
        mask = np.zeros(self.N_PLAYER_NUM_ACTIONS, dtype=np.uint8)
        for i in range(self.N_PLAYER_NUM_ACTIONS):
            word = i // 64
            bit = i % 64
            if (int(self._nplayer_legal_buf[word]) >> bit) & 1:
                mask[i] = 1
        return mask

    def apply_nplayer_action(self, action_idx: int) -> None:
        """
        Apply an N-player action by its integer index in [0, 452).

        Raises:
            ValueError: If action_idx is out of range.
            RuntimeError: If the engine rejects the action.
        """
        if not (0 <= action_idx < self.N_PLAYER_NUM_ACTIONS):
            raise ValueError(
                f"action_idx {action_idx} out of range [0, {self.N_PLAYER_NUM_ACTIONS})"
            )
        ret = self._lib.cambia_game_apply_nplayer_action(self._game_h, action_idx)
        if ret < 0:
            raise RuntimeError(
                f"cambia_game_apply_nplayer_action failed (returned {ret}) "
                f"for action {action_idx} on handle {self._game_h}"
            )

    def get_nplayer_utility(self) -> np.ndarray:
        """
        Return a (num_players,) float32 array of utilities for all players.

        Only meaningful after the game is terminal.
        Uses the N-player FFI export that copies all N utility values.
        """
        util_buf = _ffi.new(f"float[{self._num_players}]")
        self._lib.cambia_game_get_utility_n(
            self._game_h, util_buf, self._num_players
        )
        return np.frombuffer(
            _ffi.buffer(util_buf, self._num_players * 4), dtype=np.float32
        ).copy()

    # --- Lifecycle ---

    def close(self) -> None:
        """Free the game handle. Safe to call multiple times."""
        if not self._closed and self._game_h >= 0:
            if getattr(self, "_owned", True):
                self._lib.cambia_game_free(self._game_h)
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

        # Pre-allocated reusable encode buffer
        self._encode_buf = _ffi.new("float[222]")

        self._agent_h: int = self._lib.cambia_agent_new(
            engine.handle,
            player_id,
            memory_level,
            time_decay_turns,
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
        obj._encode_buf = _ffi.new("float[222]")
        obj._closed = False
        return obj

    @classmethod
    def new_nplayer(
        cls,
        engine: "GoEngine",
        player_id: int,
        num_players: int = 2,
        memory_level: int = 0,
        time_decay_turns: int = 0,
    ) -> "GoAgentState":
        """
        Factory for N-player agent state.

        Creates an agent using cambia_agent_new_nplayer which initializes
        KnowledgeMask tracking for all num_players players.

        Args:
            engine: The GoEngine game instance.
            player_id: Player index (0 to num_players-1).
            num_players: Total number of players (2-6).
            memory_level: Memory abstraction level.
            time_decay_turns: Turns before time-based decay.

        Returns:
            A GoAgentState backed by an N-player agent handle.
        """
        lib = _get_lib()
        agent_h = lib.cambia_agent_new_nplayer(
            engine.handle,
            int(player_id),
            int(num_players),
            int(memory_level),
            int(time_decay_turns),
        )
        if agent_h < 0:
            raise RuntimeError(
                f"cambia_agent_new_nplayer failed (returned {agent_h}) "
                f"for player {player_id}, num_players={num_players}."
            )
        obj = object.__new__(cls)
        obj._lib = lib
        obj._agent_h = int(agent_h)
        obj._encode_buf = _ffi.new("float[222]")
        obj._closed = False
        return obj

    def update_nplayer(self, engine: "GoEngine") -> None:
        """Update N-player agent beliefs based on the current game state."""
        ret = self._lib.cambia_agent_update_nplayer(self._agent_h, engine.handle)
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_update_nplayer failed (returned {ret}) "
                f"agent={self._agent_h} game={engine.handle}"
            )

    @classmethod
    def new_with_memory(
        cls,
        engine: "GoEngine",
        player_id: int,
        memory_level: int = 0,
        time_decay_turns: int = 0,
        memory_archetype: int = 0,
        memory_decay_lambda: float = 0.1,
        memory_capacity: int = 3,
    ) -> "GoAgentState":
        """Factory for agent state with memory archetype configuration.

        Args:
            engine: The GoEngine game instance.
            player_id: Player index (0 or 1 for 2P).
            memory_level: Memory abstraction level.
            time_decay_turns: Turns before time-based decay.
            memory_archetype: 0=Perfect, 1=Decaying, 2=HumanLike.
            memory_decay_lambda: Decay rate λ for Decaying archetype.
            memory_capacity: Max active mask size for HumanLike archetype.

        Returns:
            A GoAgentState with the specified memory archetype.
        """
        lib = _get_lib()
        agent_h = lib.cambia_agent_new_with_memory(
            engine.handle,
            int(player_id),
            int(memory_level),
            int(time_decay_turns),
            int(memory_archetype),
            float(memory_decay_lambda),
            int(memory_capacity),
        )
        if agent_h < 0:
            raise RuntimeError(
                f"cambia_agent_new_with_memory failed (returned {agent_h}) "
                f"for player {player_id}, archetype={memory_archetype}."
            )
        obj = object.__new__(cls)
        obj._lib = lib
        obj._agent_h = int(agent_h)
        obj._encode_buf = _ffi.new("float[222]")
        obj._closed = False
        return obj

    def apply_decay(self, rng_seed: int = 0) -> None:
        """Apply memory decay/eviction using a seeded RNG.

        For MemoryDecaying: creates a fresh PCG RNG seeded with rng_seed
        and probabilistically decays PrivOwn slots.
        For MemoryHumanLike: evicts lowest-saliency slots until capacity
        is satisfied (rng_seed unused).
        For MemoryPerfect: no-op.
        """
        ret = self._lib.cambia_agent_apply_decay(self._agent_h, int(rng_seed))
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_apply_decay failed (returned {ret}) "
                f"on agent handle {self._agent_h}"
            )

    # --- Core API ---

    def update(self, engine: GoEngine) -> None:
        """
        Update agent beliefs based on the current game state.

        Args:
            engine: The GoEngine instance reflecting the latest game state.

        Raises:
            RuntimeError: On engine error.
        """
        ret = self._lib.cambia_agent_update(self._agent_h, engine.handle)
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
        new_h = self._lib.cambia_agent_clone(self._agent_h)
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
        ret = self._lib.cambia_agent_encode(
            self._agent_h,
            decision_context,
            drawn_bucket,
            self._encode_buf,
        )
        if ret < 0:
            raise RuntimeError(
                f"cambia_agent_encode failed (returned {ret}) "
                f"on agent handle {self._agent_h}"
            )
        return np.frombuffer(_ffi.buffer(self._encode_buf), dtype=np.float32).copy()

    def encode_eppbs(self, decision_context: int, drawn_bucket: int = -1) -> np.ndarray:
        """EP-PBS encoding via Go FFI. Returns ndarray of shape (200,)."""
        buf = _ffi.new("float[200]")
        rc = self._lib.cambia_agent_encode_eppbs(
            self._agent_h, int(decision_context), int(drawn_bucket), buf
        )
        if rc != 0:
            raise RuntimeError(f"EP-PBS encode failed: {rc}")
        return np.frombuffer(_ffi.buffer(buf, 200 * 4), dtype=np.float32).copy()

    def encode_nplayer(self, decision_context: int, drawn_bucket: int = -1) -> np.ndarray:
        """N-player encoding via Go FFI. Returns ndarray of shape (580,)."""
        buf = _ffi.new("float[580]")
        rc = self._lib.cambia_agent_encode_nplayer(
            self._agent_h, int(decision_context), int(drawn_bucket), buf
        )
        if rc != 0:
            raise RuntimeError(f"N-player encode failed: {rc}")
        return np.frombuffer(_ffi.buffer(buf, 580 * 4), dtype=np.float32).copy()

    def nplayer_action_mask(self, engine: GoEngine) -> np.ndarray:
        """
        Return a (452,) uint8 array where 1 = legal N-player action.

        Args:
            engine: The GoEngine instance for the current game state.

        Returns:
            np.ndarray of shape (452,) with dtype uint8.
        """
        mask_buf = _ffi.new("uint8_t[452]")
        rc = self._lib.cambia_agent_nplayer_action_mask(
            self._agent_h, engine.handle, mask_buf
        )
        if rc != 0:
            raise RuntimeError(f"N-player action mask failed: {rc}")
        return np.frombuffer(_ffi.buffer(mask_buf, 452), dtype=np.uint8).copy()

    # --- Lifecycle ---

    def close(self) -> None:
        """Free the agent handle. Safe to call multiple times."""
        if not self._closed and self._agent_h >= 0:
            self._lib.cambia_agent_free(self._agent_h)
            self._agent_h = -1
            self._closed = True

    def __enter__(self) -> "GoAgentState":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Subgame Solver
# ---------------------------------------------------------------------------


class SubgameSolver:
    """Python wrapper around the Go subgame CFR solver.

    Usage::

        with GoEngine(seed=42) as game:
            with SubgameSolver(game, max_depth=4) as solver:
                leaves = solver.export_leaves()
                leaf_values = np.zeros(solver.leaf_count * 2, dtype=np.float32)
                strategy, root_values = solver.solve(leaf_values, num_iterations=200)
    """

    def __init__(self, game: GoEngine, max_depth: int = 4) -> None:
        import warnings

        warnings.warn(
            "SubgameSolver is DEPRECATED and will be removed. "
            "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
            "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._lib = _get_lib()
        self._closed = False
        self._leaf_handles: list = []  # track exported leaf game handles for cleanup
        self._solver_h: int = self._lib.cambia_subgame_build(game._game_h, max_depth)
        if self._solver_h < 0:
            raise RuntimeError(
                f"cambia_subgame_build failed (returned {self._solver_h}). "
                "Solver pool may be exhausted."
            )
        self._leaf_count: int = int(self._lib.cambia_subgame_leaf_count(self._solver_h))

    @property
    def leaf_count(self) -> int:
        return self._leaf_count

    def export_leaves(self) -> list:
        """Export leaf game states as GoEngine views.

        The returned views are tracked by this solver and freed when the solver
        is freed (or when free_leaves() is called explicitly).  Callers should
        use the leaves for value-network evaluation and then either call
        free_leaves() or let the context manager handle cleanup.
        """
        # Free any previously exported leaves first
        self._free_leaves()
        buf = _ffi.new(f"int32_t[{self._leaf_count}]")
        rc = self._lib.cambia_subgame_export_leaves(self._solver_h, buf)
        if rc < 0:
            raise RuntimeError(f"cambia_subgame_export_leaves failed: {rc}")
        leaves = []
        for i in range(self._leaf_count):
            h = int(buf[i])
            self._leaf_handles.append(h)
            leaves.append(GoEngine._from_handle(h))
        return leaves

    def free_leaves(self) -> None:
        """Explicitly free exported leaf game handles to reclaim pool slots."""
        self._free_leaves()

    def _free_leaves(self) -> None:
        """Internal: free all tracked leaf game handles."""
        for h in self._leaf_handles:
            try:
                self._lib.cambia_game_free(h)
            except Exception:
                pass
        self._leaf_handles.clear()

    def solve(
        self,
        leaf_values: np.ndarray,
        num_iterations: int = 200,
    ) -> tuple:
        """Run CFR iterations and return (strategy, root_values).

        Args:
            leaf_values: float32 array of shape (leaf_count*2,) or (leaf_count, 2).
            num_iterations: Number of CFR iterations.

        Returns:
            Tuple of (strategy, root_values) as numpy float32 arrays.
            strategy has shape (146,), root_values has shape (2,).
        """
        leaf_values = np.ascontiguousarray(leaf_values.ravel(), dtype=np.float32)
        expected = self._leaf_count * 2
        if leaf_values.size != expected:
            raise ValueError(
                f"leaf_values must have {expected} elements, got {leaf_values.size}"
            )
        strategy_buf = np.zeros(146, dtype=np.float32)
        root_values_buf = np.zeros(2, dtype=np.float32)

        lv_ptr = _ffi.cast("float *", leaf_values.ctypes.data)
        st_ptr = _ffi.cast("float *", strategy_buf.ctypes.data)
        rv_ptr = _ffi.cast("float *", root_values_buf.ctypes.data)

        rc = self._lib.cambia_subgame_solve(
            self._solver_h, num_iterations, lv_ptr, st_ptr, rv_ptr
        )
        if rc < 0:
            raise RuntimeError(f"cambia_subgame_solve failed: {rc}")

        return strategy_buf, root_values_buf

    def free(self) -> None:
        """Release all resources: leaf game handles then solver handle."""
        self._free_leaves()
        if not self._closed and self._solver_h >= 0:
            self._lib.cambia_subgame_free(self._solver_h)
            self._solver_h = -1
            self._closed = True

    def __enter__(self) -> "SubgameSolver":
        return self

    def __exit__(self, *args: object) -> None:
        self.free()

    def __del__(self) -> None:
        self.free()


# ---------------------------------------------------------------------------
# Handle pool diagnostics
# ---------------------------------------------------------------------------


def get_handle_pool_stats() -> dict:
    """Return in-use slot counts for games, agents, and snapshots.

    Calls the Go-side cambia_handle_pool_stats (thread-safe via poolMu).

    Returns:
        dict with keys 'games', 'agents', 'snapshots' — each an int count of
        currently allocated handles in the respective pool.
    """
    lib = _get_lib()
    ffi = _ffi
    games_p = ffi.new("int32_t *")
    agents_p = ffi.new("int32_t *")
    snaps_p = ffi.new("int32_t *")
    lib.cambia_handle_pool_stats(games_p, agents_p, snaps_p)
    return {
        "games": int(games_p[0]),
        "agents": int(agents_p[0]),
        "snapshots": int(snaps_p[0]),
    }
