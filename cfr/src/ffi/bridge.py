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
from typing import List, Optional, Tuple

import cffi
import numpy as np

from src.config import CambiaRulesConfig
from src.constants import (
    N_PLAYER_INPUT_DIM as _GO_N_PLAYER_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS as _GO_N_PLAYER_NUM_ACTIONS,
)

# Word count for the Go N-player legal-actions bitmask: engine.NPlayerLegalActions()
# (engine/legal.go) returns a fixed [10]uint64 (640 bits, covering 620 actions).
# Derived here from the single-sourced N_PLAYER_NUM_ACTIONS rather than hardcoded
# so a future MaxPlayers bump can't silently under-allocate this buffer again the
# way the 580/452 dims did (cambia-542 F8).
_NPLAYER_LEGAL_WORDS = (_GO_N_PLAYER_NUM_ACTIONS + 63) // 64

# ---------------------------------------------------------------------------
# Card index mapping utilities
# ---------------------------------------------------------------------------

# Python suit → canonical index (C=0, D=1, H=2, S=3)
SUIT_OFFSET = {"C": 0, "D": 1, "H": 2, "S": 3}

# Python rank → canonical rank index (A=0, 2=1, ..., K=12)
RANK_VALUE = {
    "A": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "T": 9,
    "J": 10,
    "Q": 11,
    "K": 12,
}

# PRT-CFR per-agent token stream hard cap (must equal agent.MaxTokenStream in
# engine/agent/tokens.go, which is itself paired with prtcfr_worker.py's
# PRODUCTION_SEQ_CAP=12288). The Go side raises on overflow; this sizes the
# read buffer to hold the full stream. Raised 4096 -> 12288 in S1W12 alongside
# the Go constant; verify live via get_token_stream_cap() rather than trusting
# this literal to stay in sync (the FFI export is the source of truth).
TOKEN_STREAM_CAP = 12288


def python_card_to_go_index(card) -> int:
    """Convert a Python Card object to a canonical uint8 card index.

    Index encoding: suit*13 + rank (suits C=0,D=1,H=2,S=3; ranks A=0..K=12).
    Jokers (rank=='R', suit=None) map to index 52.
    Use extract_deck_from_python_game to distinguish two jokers (52 vs 53).
    """
    if card.rank == "R":  # JOKER_RANK_STR
        return 52
    return SUIT_OFFSET[card.suit] * 13 + RANK_VALUE[card.rank]


def extract_deck_from_python_game(game) -> Tuple[List[int], int]:
    """Reconstruct the deal-order deck from a fully-dealt Python CambiaGameState.

    Returns (deck_indices, starting_player) where:
    - deck_indices is a list of uint8 card indices in deal order:
      [p0c0, p1c0, ..., p0cN, p1cN, ..., discard_flip, stock_top, ..., stock_bottom]
    - starting_player is game.current_player_index

    This ordering matches what cambia_game_new_with_deck expects: deck[0] is the
    first card dealt (Player 0, slot 0) and is placed at Stockpile[deckLen-1]
    so that Deal's pop-from-end logic works correctly.
    """
    players = game.players
    num_players = len(players)
    cards_per_player = len(players[0].hand) if players else 4

    deck_indices: List[int] = []
    joker_count = 0

    def _card_index(card) -> int:
        nonlocal joker_count
        if card.rank == "R":
            idx = 52 + min(joker_count, 1)
            joker_count += 1
            return idx
        return SUIT_OFFSET[card.suit] * 13 + RANK_VALUE[card.rank]

    # Round-robin dealt cards: c=0..N-1, p=0..numPlayers-1
    for c in range(cards_per_player):
        for p in range(num_players):
            deck_indices.append(_card_index(players[p].hand[c]))

    # Discard flip card (top of discard pile at game start)
    if game.discard_pile:
        deck_indices.append(_card_index(game.discard_pile[0]))

    # Remaining stockpile: top (last element) to bottom (first element)
    for card in reversed(game.stockpile):
        deck_indices.append(_card_index(card))

    starting_player = getattr(game, "current_player_index", 0)
    return deck_indices, starting_player


# ---------------------------------------------------------------------------
# Library loading — module-level singleton
# ---------------------------------------------------------------------------

_ffi = cffi.FFI()
_ffi.cdef("""
    /* Game lifecycle */
    int32_t cambia_game_new(uint64_t seed);
    int32_t cambia_game_new_with_deck(
        uint8_t *deck, int32_t deck_len,
        uint8_t  num_players, uint8_t  cards_per_player,
        uint8_t  starting_player,
        uint16_t max_game_turns, uint8_t  cambia_allowed_round,
        uint8_t  penalty_draw_count, uint8_t  allow_draw_from_discard,
        uint8_t  allow_replace_abilities, uint8_t  allow_opponent_snapping,
        uint8_t  snap_race, uint8_t  num_jokers, uint8_t  lock_caller_hand,
        uint8_t  initial_view_count, uint8_t  num_decks
    );
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
        uint8_t  numPlayers,
        uint8_t  initialViewCount,
        uint8_t  numDecks
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
    int32_t cambia_agent_encode_eppbs_interleaved(int32_t h, uint8_t decision_context,
                                                   int8_t drawn_bucket, float *out);
    int32_t cambia_agent_encode_eppbs_dealiased(int32_t h, uint8_t decision_context,
                                                int8_t drawn_bucket, float *out);
    int32_t cambia_agent_encode_eppbs_interleaved_v2(int32_t h, uint8_t decision_context,
                                                      int8_t drawn_bucket, float *out);

    /* Training-only: omniscient card inspection */
    int32_t cambia_game_get_all_cards(int32_t game_h, uint8_t *out_buf, int32_t buf_len);
    uint8_t cambia_game_num_players(int32_t h);
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
    int32_t cambia_nplayer_input_dim(void);
    int32_t cambia_nplayer_num_actions(void);

    /* Subgame solver */
    int32_t cambia_subgame_build(int32_t game_h, int32_t max_depth);
    int32_t cambia_subgame_leaf_count(int32_t solver_h);
    int32_t cambia_subgame_export_leaves(int32_t solver_h, int32_t *game_handles_out);
    int32_t cambia_subgame_solve(int32_t solver_h, int32_t num_iterations,
                                 float *leaf_values, float *strategy_out,
                                 float *root_values_out);
    int32_t cambia_subgame_solve_ranged(int32_t solver_h, int32_t num_iterations,
                                        int32_t num_hand_types,
                                        float *leaf_values,
                                        float *range_p0, float *range_p1,
                                        float *strategy_out, float *root_cfvs_out);
    void    cambia_subgame_free(int32_t solver_h);

    /* Discard top */
    int32_t cambia_game_discard_top(int32_t game_h);

    /* Agent attribute getters (training-only; for Python action_abstraction) */
    int32_t cambia_agent_get_own_hand(int32_t agent_h, uint8_t *out_buf, int32_t buf_len);
    int32_t cambia_agent_get_opp_belief(int32_t agent_h, uint8_t *out_buf, int32_t buf_len);
    uint16_t cambia_agent_get_current_turn(int32_t agent_h);
    int32_t cambia_agent_get_hand_lens(int32_t agent_h, uint8_t *out_buf);

    /* Handle pool diagnostics */
    void    cambia_handle_pool_stats(int32_t *games_out, int32_t *agents_out, int32_t *snaps_out);

    /* PRT-CFR event-stream token FFI (S1W2, additive) */
    int32_t cambia_agent_token_len(int32_t agent_h);
    int32_t cambia_agent_tokens(int32_t agent_h, int32_t *out, int32_t max);
    int32_t cambia_agent_tokens_since(int32_t agent_h, int32_t since, int32_t *out, int32_t max);
    int32_t cambia_games_apply_batch(int32_t *game_hs, int32_t *a0s, int32_t *a1s,
                                     uint16_t *actions, int32_t n);
    int32_t cambia_state_save(int32_t game_h, int32_t a0_h, int32_t a1_h);
    int32_t cambia_state_restore(int32_t game_h, int32_t snap_h, int32_t a0_h, int32_t a1_h);
    void    cambia_state_snapshot_free(int32_t h);
    int32_t cambia_state_clone(int32_t game_h, int32_t a0_h, int32_t a1_h,
                               int32_t *out_game_h, int32_t *out_a0_h, int32_t *out_a1_h);
    int32_t cambia_token_vocab(int32_t *out, int32_t max);
    int32_t cambia_token_encode_card(uint8_t go_card_index);
    int32_t cambia_token_encode_action(uint16_t action_idx);
    int32_t cambia_token_stream_cap(void);
    int32_t cambia_tokenizer_version(void);
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
    # Single-sourced from cfr/src/constants.py (which mirrors the Go
    # engine/agent constants) rather than hardcoded -- cambia-542 F8: this
    # module previously hardcoded stale 580/452 values that drifted from the
    # 856/620 dims after the MaxPlayers 6->8 bump (commit 9073646), causing
    # malloc-crash buffer overflows in encode_nplayer/nplayer_action_mask/
    # nplayer_legal_actions_mask. get_nplayer_dims() below cross-checks these
    # against the live Go values through the FFI.
    N_PLAYER_INPUT_DIM: int = _GO_N_PLAYER_INPUT_DIM
    N_PLAYER_NUM_ACTIONS: int = _GO_N_PLAYER_NUM_ACTIONS
    # Encoding v2 (DESCA Phase 0) constants. Kept in sync with the Go
    # constants in engine/agent/constants.go (EPPBSV2InputDim etc.).
    EPPBS_V2_INPUT_DIM: int = 257
    MAX_HAND_SIZE: int = 6

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
        self._nplayer_mask_buf = _ffi.new(f"uint8_t[{_GO_N_PLAYER_NUM_ACTIONS}]")
        self._nplayer_legal_buf = _ffi.new(f"uint64_t[{_NPLAYER_LEGAL_WORDS}]")
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
                getattr(house_rules, "initial_view_count", 2),
                getattr(house_rules, "num_decks", 1),
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
    def _from_handle(cls, game_h: int, owned: bool = False) -> "GoEngine":
        """Create a GoEngine view from an existing game handle.

        By default (owned=False) the returned object will NOT free the handle
        when closed or GC'd -- the non-owning mode used by
        SubgameSolver.export_leaves() to wrap leaf game handles it manages
        separately. Pass owned=True when the handle is freshly allocated and
        this wrapper should own its lifecycle (e.g. state_clone()).
        """
        obj = object.__new__(cls)
        obj._lib = _get_lib()
        obj._game_h = game_h
        obj._num_players = 2
        obj._mask_buf = _ffi.new("uint8_t[146]")
        obj._util_buf = _ffi.new("float[2]")
        obj._nplayer_mask_buf = _ffi.new(f"uint8_t[{_GO_N_PLAYER_NUM_ACTIONS}]")
        obj._nplayer_legal_buf = _ffi.new(f"uint64_t[{_NPLAYER_LEGAL_WORDS}]")
        obj._nplayer_util_buf = _ffi.new("float[2]")
        obj._closed = False
        obj._owned = owned
        return obj

    @classmethod
    def from_deck(
        cls,
        deck_indices: List[int],
        starting_player: int = 0,
        house_rules=None,
    ) -> "GoEngine":
        """Create a GoEngine with a pre-determined deck order.

        The deck_indices list encodes the deal order: deck[0] is dealt to
        Player 0's first slot, deck[1] to Player 1's first slot, etc.
        Use extract_deck_from_python_game() to produce this list from a
        Python CambiaGameState.

        Args:
            deck_indices: List of uint8 card indices in deal order.
            starting_player: Which player acts first (0-indexed).
            house_rules: CambiaRulesConfig instance (or None for defaults).

        Returns:
            A new GoEngine instance owning the game handle.
        """
        lib = _get_lib()
        obj = object.__new__(cls)
        obj._lib = lib
        obj._closed = False
        obj._owned = True

        if house_rules is not None:
            np_val = int(getattr(house_rules, "num_players", 2) or 2)
        else:
            np_val = 2
        obj._num_players = max(2, np_val)

        obj._mask_buf = _ffi.new("uint8_t[146]")
        obj._util_buf = _ffi.new("float[2]")
        obj._nplayer_mask_buf = _ffi.new(f"uint8_t[{_GO_N_PLAYER_NUM_ACTIONS}]")
        obj._nplayer_legal_buf = _ffi.new(f"uint64_t[{_NPLAYER_LEGAL_WORDS}]")
        obj._nplayer_util_buf = _ffi.new(f"float[{obj._num_players}]")

        deck_arr = _ffi.new("uint8_t[]", deck_indices)

        if house_rules is not None:
            game_h = lib.cambia_game_new_with_deck(
                deck_arr,
                len(deck_indices),
                obj._num_players,
                int(house_rules.cards_per_player),
                int(starting_player),
                int(house_rules.max_game_turns),
                int(house_rules.cambia_allowed_round),
                int(house_rules.penaltyDrawCount),
                1 if house_rules.allowDrawFromDiscardPile else 0,
                1 if house_rules.allowReplaceAbilities else 0,
                1 if house_rules.allowOpponentSnapping else 0,
                1 if house_rules.snapRace else 0,
                int(house_rules.use_jokers),
                1 if getattr(house_rules, "lockCallerHand", True) else 0,
                int(getattr(house_rules, "initial_view_count", 2)),
                int(getattr(house_rules, "num_decks", 1)),
            )
        else:
            # Defaults: 2 players, 4 cards each, standard competitive rules
            game_h = lib.cambia_game_new_with_deck(
                deck_arr,
                len(deck_indices),
                2,
                4,
                int(starting_player),
                46,
                0,
                2,
                1,
                0,
                1,
                0,
                2,
                1,
                2,
                1,
            )

        if game_h < 0:
            raise RuntimeError(
                f"cambia_game_new_with_deck failed (returned {game_h}). "
                "Handle pool may be exhausted or deck length invalid."
            )
        obj._game_h = int(game_h)
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

    def discard_top(self) -> Optional[int]:
        """Return the bucket index of the top discard card, or None if empty."""
        result = int(self._lib.cambia_game_discard_top(self._game_h))
        return result if result >= 0 else None

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

    def _get_all_cards_unsafe(self) -> np.ndarray:
        """Return a packed uint8 array of bucket indices for every slot in
        every player's hand.

        Output shape: (num_players * MAX_HAND_SIZE,). Empty or unknown slots
        receive the sentinel value 0xFF. Known slots receive the CardBucket
        index (0..8) for BucketZero..BucketHighKing.

        WARNING: Training-only. Exposes ground-truth card identities for all
        players. The leading underscore and _unsafe suffix mark this as a
        deliberate omniscience leak; do not call from eval-time code paths.
        Use cfr.src.cfr.omniscient.compute_omniscient_features for the standard
        training wrapper.
        """
        n = int(self._num_players)
        size = n * self.MAX_HAND_SIZE
        buf = _ffi.new(f"uint8_t[{size}]")
        written = self._lib.cambia_game_get_all_cards(self._game_h, buf, size)
        if written < 0:
            raise RuntimeError(
                f"cambia_game_get_all_cards failed (returned {written}) "
                f"on handle {self._game_h} (buf_len={size})"
            )
        return np.frombuffer(_ffi.buffer(buf, size), dtype=np.uint8).copy()

    def update_both(self, a0: "GoAgentState", a1: "GoAgentState") -> None:
        """
        Update both agent belief states in a single FFI call.

        Args:
            a0: First agent (player 0).
            a1: Second agent (player 1).

        Raises:
            RuntimeError: On engine error.
        """
        ret = self._lib.cambia_agents_update_both(a0._agent_h, a1._agent_h, self._game_h)
        if ret < 0:
            raise RuntimeError(
                f"cambia_agents_update_both failed (returned {ret}) "
                f"a0={a0._agent_h} a1={a1._agent_h} game={self._game_h}"
            )

    # --- N-Player APIs ---

    def nplayer_legal_actions_mask(self) -> np.ndarray:
        """
        Return a (N_PLAYER_NUM_ACTIONS,) uint8 numpy array where 1 = legal N-player action.

        Uses the _NPLAYER_LEGAL_WORDS-word uint64 bitmask from Go, expanded to per-action bytes.
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
        Apply an N-player action by its integer index in [0, N_PLAYER_NUM_ACTIONS).

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
        self._lib.cambia_game_get_utility_n(self._game_h, util_buf, self._num_players)
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

        # Pre-allocated reusable encode buffers (T1-2 cffi buffer reuse).
        # encode (222), encode_eppbs (224), encode_eppbs_interleaved_v2 (257),
        # encode_nplayer (GoEngine.N_PLAYER_INPUT_DIM), plus agent-attr getter
        # buffers.
        self._encode_buf = _ffi.new("float[222]")
        self._encode_buf_224 = _ffi.new("float[224]")
        self._encode_buf_257 = _ffi.new(f"float[{GoEngine.EPPBS_V2_INPUT_DIM}]")
        self._encode_buf_nplayer = None  # lazy alloc only when N-player is used
        # Agent-attr getter buffers (training-only path for Python adapters):
        # 6*4 own hand triplets + 6 opp belief bytes + 2 hand-len bytes.
        self._own_hand_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE * 4}]")
        self._opp_belief_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE}]")
        self._hand_lens_buf = _ffi.new("uint8_t[2]")
        # PRT-CFR token stream read buffer (full Go per-agent hard cap).
        self._token_buf = _ffi.new(f"int32_t[{TOKEN_STREAM_CAP}]")

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
        # Pre-allocated reusable buffers (T1-2 cffi buffer reuse).
        obj._encode_buf = _ffi.new("float[222]")
        obj._encode_buf_224 = _ffi.new("float[224]")
        obj._encode_buf_257 = _ffi.new(f"float[{GoEngine.EPPBS_V2_INPUT_DIM}]")
        obj._encode_buf_nplayer = None
        obj._own_hand_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE * 4}]")
        obj._opp_belief_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE}]")
        obj._hand_lens_buf = _ffi.new("uint8_t[2]")
        obj._token_buf = _ffi.new(f"int32_t[{TOKEN_STREAM_CAP}]")
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
        # Pre-allocated reusable buffers (T1-2 cffi buffer reuse).
        obj._encode_buf = _ffi.new("float[222]")
        obj._encode_buf_224 = _ffi.new("float[224]")
        obj._encode_buf_257 = _ffi.new(f"float[{GoEngine.EPPBS_V2_INPUT_DIM}]")
        obj._encode_buf_nplayer = None
        obj._own_hand_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE * 4}]")
        obj._opp_belief_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE}]")
        obj._hand_lens_buf = _ffi.new("uint8_t[2]")
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
        # Pre-allocated reusable buffers (T1-2 cffi buffer reuse).
        obj._encode_buf = _ffi.new("float[222]")
        obj._encode_buf_224 = _ffi.new("float[224]")
        obj._encode_buf_257 = _ffi.new(f"float[{GoEngine.EPPBS_V2_INPUT_DIM}]")
        obj._encode_buf_nplayer = None
        obj._own_hand_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE * 4}]")
        obj._opp_belief_buf = _ffi.new(f"uint8_t[{GoEngine.MAX_HAND_SIZE}]")
        obj._hand_lens_buf = _ffi.new("uint8_t[2]")
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

    # --- PRT-CFR token stream (S1W2) ---

    @property
    def handle(self) -> int:
        """Expose the raw agent handle (for apply_games_batch / state_save)."""
        return self._agent_h

    def token_len(self) -> int:
        """Return the number of tokens in this agent's full stream body."""
        n = self._lib.cambia_agent_token_len(self._agent_h)
        if n < 0:
            raise RuntimeError(
                f"cambia_agent_token_len failed (returned {n}) "
                f"on agent handle {self._agent_h}"
            )
        return int(n)

    def tokens(self) -> np.ndarray:
        """Return the FULL token stream body as an int32 numpy array.

        No BOS/EOS and no truncation: this is the raw frame body. Use
        frame_aligned_window() to produce the byte-exact
        encode_observation_sequence output for a given cap.
        """
        n = self._lib.cambia_agent_tokens(
            self._agent_h, self._token_buf, TOKEN_STREAM_CAP
        )
        if n < 0:
            raise RuntimeError(
                f"cambia_agent_tokens failed (returned {n}) "
                f"on agent handle {self._agent_h}"
            )
        return np.frombuffer(
            _ffi.buffer(self._token_buf, int(n) * 4), dtype=np.int32
        ).copy()

    def tokens_since(self, since: int) -> np.ndarray:
        """Return the incremental tail tokens[since:] as an int32 numpy array."""
        n = self._lib.cambia_agent_tokens_since(
            self._agent_h, int(since), self._token_buf, TOKEN_STREAM_CAP
        )
        if n < 0:
            raise RuntimeError(
                f"cambia_agent_tokens_since failed (returned {n}) "
                f"agent={self._agent_h} since={since}"
            )
        return np.frombuffer(
            _ffi.buffer(self._token_buf, int(n) * 4), dtype=np.int32
        ).copy()

    def encode(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode agent belief state as a 222-dimensional float32 feature vector.

        Reuses a cached cffi buffer (T1-2). If ``out`` is supplied with matching
        shape/dtype, the encoded floats are copied into it.

        Args:
            decision_context: Integer encoding of the current decision context.
            drawn_bucket: Bucket index of the drawn card, or -1 if none.
            out: Optional pre-allocated float32 ndarray of shape (222,).

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
        view = np.frombuffer(_ffi.buffer(self._encode_buf), dtype=np.float32)
        if out is not None:
            if out.shape != (222,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected (222,)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    def encode_eppbs(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """EP-PBS encoding via Go FFI. Returns ndarray of shape (224,).

        Uses a pre-allocated cffi buffer cached on this GoAgentState instance
        to avoid per-call ``_ffi.new`` allocation (T1-2). If ``out`` is given
        and shape/dtype matches, the encoded floats are copied into it and
        returned; otherwise a fresh ndarray copy is returned.
        """
        rc = self._lib.cambia_agent_encode_eppbs(
            self._agent_h, int(decision_context), int(drawn_bucket), self._encode_buf_224
        )
        if rc != 0:
            raise RuntimeError(f"EP-PBS encode failed: {rc}")
        view = np.frombuffer(_ffi.buffer(self._encode_buf_224, 224 * 4), dtype=np.float32)
        if out is not None:
            if out.shape != (224,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected (224,)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    def encode_eppbs_interleaved(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """EP-PBS interleaved encoding via Go FFI. Returns ndarray of shape (224,).

        Uses interleaved slot layout: public(42) + 12×slot(13) + pad(2) + history(24) = 224.
        Public features include own/opp hand sizes at dims [40] and [41].
        Required for SlotFiLM and slot_multiply network architectures.

        Reuses a cached cffi buffer (T1-2). If ``out`` is supplied with matching
        shape/dtype, the encoded floats are copied into it.
        """
        rc = self._lib.cambia_agent_encode_eppbs_interleaved(
            self._agent_h, int(decision_context), int(drawn_bucket), self._encode_buf_224
        )
        if rc != 0:
            raise RuntimeError(f"EP-PBS interleaved encode failed: {rc}")
        view = np.frombuffer(_ffi.buffer(self._encode_buf_224, 224 * 4), dtype=np.float32)
        if out is not None:
            if out.shape != (224,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected (224,)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    def encode_eppbs_interleaved_v2(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encoding v2 (257-dim): base 224-dim interleaved EP-PBS + 9-dim
        card-counting posterior + 24-dim action history window.

        See engine/agent/encoding.go EncodeEPPBSInterleavedV2 for the full
        layout rationale. The Python binding in src/encoding.py dispatches on
        encoding_version; this method is the raw FFI bridge.

        Reuses a cached cffi buffer (T1-2). If ``out`` is supplied with matching
        shape/dtype, the encoded floats are copied into it.
        """
        dim = GoEngine.EPPBS_V2_INPUT_DIM
        rc = self._lib.cambia_agent_encode_eppbs_interleaved_v2(
            self._agent_h, int(decision_context), int(drawn_bucket), self._encode_buf_257
        )
        if rc != 0:
            raise RuntimeError(f"EP-PBS v2 interleaved encode failed: {rc}")
        view = np.frombuffer(_ffi.buffer(self._encode_buf_257, dim * 4), dtype=np.float32)
        if out is not None:
            if out.shape != (dim,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected ({dim},)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    def encode_eppbs_dealiased(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """De-aliased flat EP-PBS encoding via Go FFI. Returns ndarray of shape (224,).

        Uses flat layout (tags grouped, then identities grouped) with two de-aliasing fixes:
        - Empty slots (beyond hand_size) are all-zeros in both tag and identity regions
        - Hand sizes at dims [196] and [197]
        History features at dims [200-223].

        Reuses a cached cffi buffer (T1-2). If ``out`` is supplied with matching
        shape/dtype, the encoded floats are copied into it.
        """
        rc = self._lib.cambia_agent_encode_eppbs_dealiased(
            self._agent_h, int(decision_context), int(drawn_bucket), self._encode_buf_224
        )
        if rc != 0:
            raise RuntimeError(f"EP-PBS dealiased encode failed: {rc}")
        view = np.frombuffer(_ffi.buffer(self._encode_buf_224, 224 * 4), dtype=np.float32)
        if out is not None:
            if out.shape != (224,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected (224,)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    def encode_nplayer(
        self,
        decision_context: int,
        drawn_bucket: int = -1,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """N-player encoding via Go FFI. Returns ndarray of shape (N_PLAYER_INPUT_DIM,).

        Lazy-allocates a N_PLAYER_INPUT_DIM-float cffi buffer on first call
        (T1-2 buffer reuse for N-player path). If ``out`` is supplied with
        matching shape/dtype, the encoded floats are copied into it.
        """
        if self._encode_buf_nplayer is None:
            self._encode_buf_nplayer = _ffi.new(f"float[{_GO_N_PLAYER_INPUT_DIM}]")
        rc = self._lib.cambia_agent_encode_nplayer(
            self._agent_h,
            int(decision_context),
            int(drawn_bucket),
            self._encode_buf_nplayer,
        )
        if rc != 0:
            raise RuntimeError(f"N-player encode failed: {rc}")
        view = np.frombuffer(
            _ffi.buffer(self._encode_buf_nplayer, _GO_N_PLAYER_INPUT_DIM * 4),
            dtype=np.float32,
        )
        if out is not None:
            if out.shape != (_GO_N_PLAYER_INPUT_DIM,) or out.dtype != np.float32:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: got {out.shape}/{out.dtype}, "
                    f"expected ({_GO_N_PLAYER_INPUT_DIM},)/float32"
                )
            np.copyto(out, view)
            return out
        return view.copy()

    # --- Training-only attribute getters (T1-1) ---
    #
    # These wrap the cambia_agent_get_* FFI exports to expose minimal agent
    # state required by cfr/src/action_abstraction.py when running the DESCA
    # Go FFI env_factory. They are read-only views into the live agent state;
    # the returned ndarrays are copies safe to retain across FFI calls.

    def get_own_hand_buckets_and_seen(self) -> np.ndarray:
        """Return a (MaxHandSize, 3) int32 ndarray of (bucket, last_seen_turn, valid).

        Slot rows are 0..MaxHandSize-1; valid==1 means the slot is within
        OwnHandLen, else 0. ``bucket`` ranges 0..9 (CardBucket values; 9 is
        BucketUnknown). ``last_seen_turn`` is the encoded turn index packed
        from the Go uint16 (low byte | high byte << 8).
        """
        rc = self._lib.cambia_agent_get_own_hand(
            self._agent_h, self._own_hand_buf, GoEngine.MAX_HAND_SIZE * 4
        )
        if rc != 0:
            raise RuntimeError(f"cambia_agent_get_own_hand failed: {rc}")
        raw = np.frombuffer(
            _ffi.buffer(self._own_hand_buf, GoEngine.MAX_HAND_SIZE * 4),
            dtype=np.uint8,
        )
        out = np.zeros((GoEngine.MAX_HAND_SIZE, 3), dtype=np.int32)
        for s in range(GoEngine.MAX_HAND_SIZE):
            base = s * 4
            out[s, 0] = int(raw[base + 0])
            out[s, 1] = int(raw[base + 1]) | (int(raw[base + 2]) << 8)
            out[s, 2] = int(raw[base + 3])
        return out

    def get_opp_belief_buckets(self) -> np.ndarray:
        """Return a (MaxHandSize,) uint8 ndarray of opponent belief buckets.

        Indices 0..OppHandLen-1 hold CardBucket values 0..9 (9=Unknown);
        slots beyond OppHandLen receive sentinel 0xFF. Decay-encoded beliefs
        collapse to 9 (Unknown) per the Go-side getter contract.
        """
        rc = self._lib.cambia_agent_get_opp_belief(
            self._agent_h, self._opp_belief_buf, GoEngine.MAX_HAND_SIZE
        )
        if rc != 0:
            raise RuntimeError(f"cambia_agent_get_opp_belief failed: {rc}")
        return np.frombuffer(
            _ffi.buffer(self._opp_belief_buf, GoEngine.MAX_HAND_SIZE),
            dtype=np.uint8,
        ).copy()

    def get_current_turn(self) -> int:
        """Return the agent's CurrentTurn observation counter."""
        v = int(self._lib.cambia_agent_get_current_turn(self._agent_h))
        if v == 0xFFFF:
            raise RuntimeError(
                f"cambia_agent_get_current_turn failed on agent {self._agent_h}"
            )
        return v

    def get_hand_lens(self) -> Tuple[int, int]:
        """Return (own_hand_len, opp_hand_len)."""
        rc = self._lib.cambia_agent_get_hand_lens(self._agent_h, self._hand_lens_buf)
        if rc != 0:
            raise RuntimeError(f"cambia_agent_get_hand_lens failed: {rc}")
        return int(self._hand_lens_buf[0]), int(self._hand_lens_buf[1])

    def nplayer_action_mask(self, engine: GoEngine) -> np.ndarray:
        """
        Return a (N_PLAYER_NUM_ACTIONS,) uint8 array where 1 = legal N-player action.

        Args:
            engine: The GoEngine instance for the current game state.

        Returns:
            np.ndarray of shape (N_PLAYER_NUM_ACTIONS,) with dtype uint8.
        """
        mask_buf = _ffi.new(f"uint8_t[{_GO_N_PLAYER_NUM_ACTIONS}]")
        rc = self._lib.cambia_agent_nplayer_action_mask(
            self._agent_h, engine.handle, mask_buf
        )
        if rc != 0:
            raise RuntimeError(f"N-player action mask failed: {rc}")
        return np.frombuffer(
            _ffi.buffer(mask_buf, _GO_N_PLAYER_NUM_ACTIONS), dtype=np.uint8
        ).copy()

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

    def solve_ranged(
        self,
        leaf_values: np.ndarray,
        range_p0: np.ndarray,
        range_p1: np.ndarray,
        num_iterations: int = 200,
    ) -> tuple:
        """Run ranged CFR and return (strategy, root_cfvs).

        Args:
            leaf_values: float32 array of shape (leaf_count, 2, num_hand_types)
                or equivalently (leaf_count * 2 * num_hand_types,).
            range_p0: float32 array of shape (num_hand_types,).
            range_p1: float32 array of shape (num_hand_types,).
            num_iterations: Number of CFR iterations.

        Returns:
            Tuple of (strategy, root_cfvs) as numpy float32 arrays.
            strategy has shape (146,).
            root_cfvs has shape (2, num_hand_types) — per-hand-type CFVs for each player.
        """
        nht = len(range_p0)
        leaf_values = np.ascontiguousarray(leaf_values.ravel(), dtype=np.float32)
        expected = self._leaf_count * 2 * nht
        if leaf_values.size != expected:
            raise ValueError(
                f"leaf_values must have {expected} elements, got {leaf_values.size}"
            )
        range_p0 = np.ascontiguousarray(range_p0, dtype=np.float32)
        range_p1 = np.ascontiguousarray(range_p1, dtype=np.float32)
        strategy_buf = np.zeros(146, dtype=np.float32)
        root_cfvs_buf = np.zeros(2 * nht, dtype=np.float32)

        lv_ptr = _ffi.cast("float *", leaf_values.ctypes.data)
        r0_ptr = _ffi.cast("float *", range_p0.ctypes.data)
        r1_ptr = _ffi.cast("float *", range_p1.ctypes.data)
        st_ptr = _ffi.cast("float *", strategy_buf.ctypes.data)
        rv_ptr = _ffi.cast("float *", root_cfvs_buf.ctypes.data)

        rc = self._lib.cambia_subgame_solve_ranged(
            self._solver_h, num_iterations, nht, lv_ptr, r0_ptr, r1_ptr, st_ptr, rv_ptr
        )
        if rc < 0:
            raise RuntimeError(f"cambia_subgame_solve_ranged failed: {rc}")

        return strategy_buf, root_cfvs_buf.reshape(2, nht)

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


# ---------------------------------------------------------------------------
# PRT-CFR event-stream token FFI (S1W2, additive)
# ---------------------------------------------------------------------------


def apply_games_batch(
    game_handles: List[int],
    a0_handles: List[int],
    a1_handles: List[int],
    actions: List[int],
) -> None:
    """Apply one action to each game and update its two agents (belief + tokens).

    Vectorized single-FFI-call batch step: for game i, applies actions[i], then
    updates agents a0_handles[i] and a1_handles[i] (belief state + append-only
    token stream). An agent handle of -1 skips that agent. Per-call overhead is
    O(len(game_handles)) with no per-game Python roundtrip.

    Raises:
        ValueError: If the input lists differ in length.
        RuntimeError: On invalid handle / apply error, or token-stream overflow
            (the hard 4096-token cap is exceeded; never silently truncated).
    """
    n = len(game_handles)
    if not (len(a0_handles) == n and len(a1_handles) == n and len(actions) == n):
        raise ValueError(
            "apply_games_batch: game/a0/a1/action lists must be equal length"
        )
    if n == 0:
        return
    lib = _get_lib()
    gh = _ffi.new("int32_t[]", [int(x) for x in game_handles])
    a0 = _ffi.new("int32_t[]", [int(x) for x in a0_handles])
    a1 = _ffi.new("int32_t[]", [int(x) for x in a1_handles])
    act = _ffi.new("uint16_t[]", [int(x) for x in actions])
    ret = lib.cambia_games_apply_batch(gh, a0, a1, act, n)
    if ret == -2:
        raise RuntimeError(
            f"cambia_games_apply_batch: token stream overflow "
            f"(> {TOKEN_STREAM_CAP} tokens); hard cap exceeded, no silent "
            f"truncation"
        )
    if ret < 0:
        raise RuntimeError(
            f"cambia_games_apply_batch failed (returned {ret}); invalid handle "
            "or apply error"
        )


def state_save(game_h: int, a0_h: int, a1_h: int) -> int:
    """Snapshot a (game, both agents' belief + token) checkpoint. Returns handle."""
    lib = _get_lib()
    sh = lib.cambia_state_save(int(game_h), int(a0_h), int(a1_h))
    if sh < 0:
        raise RuntimeError(
            f"cambia_state_save failed (returned {sh}) "
            f"game={game_h} a0={a0_h} a1={a1_h}"
        )
    return int(sh)


def state_restore(game_h: int, snap_h: int, a0_h: int, a1_h: int) -> None:
    """Restore a (game, both agents' belief + token) checkpoint from a snapshot.

    a0_h/a1_h must match the handles passed to the paired state_save.
    """
    lib = _get_lib()
    ret = lib.cambia_state_restore(int(game_h), int(snap_h), int(a0_h), int(a1_h))
    if ret < 0:
        raise RuntimeError(
            f"cambia_state_restore failed (returned {ret}) "
            f"game={game_h} snap={snap_h} a0={a0_h} a1={a1_h}"
        )


def state_snapshot_free(snap_h: int) -> None:
    """Release a state snapshot handle from state_save."""
    _get_lib().cambia_state_snapshot_free(int(snap_h))


def state_clone(game_h: int, a0_h: int, a1_h: int) -> Tuple[int, int, int]:
    """Clone (game, both agents' belief + token state) onto FRESH handles.

    Independent clone for rollout fan-out (p2-redesign.md: "clone the engine
    state" at a decision node, then apply divergent playouts on each clone).
    Distinct from state_save/restore, which rewinds the SAME handles and so
    cannot back independent, simultaneously-live branches.

    Returns (game_h, a0_h, a1_h) for the new clone. The caller owns the new
    handles and must free them (GoEngine/GoAgentState.close(), or
    cambia_game_free/cambia_agent_free directly) when done.

    Raises:
        RuntimeError: On invalid source handle or pool exhaustion (game or
            agent pool full). No handles are leaked on failure -- the Go side
            frees any partially allocated handles before returning the error.
    """
    lib = _get_lib()
    out_g = _ffi.new("int32_t *")
    out_a0 = _ffi.new("int32_t *")
    out_a1 = _ffi.new("int32_t *")
    ret = lib.cambia_state_clone(int(game_h), int(a0_h), int(a1_h), out_g, out_a0, out_a1)
    if ret < 0:
        raise RuntimeError(
            f"cambia_state_clone failed (returned {ret}) "
            f"game={game_h} a0={a0_h} a1={a1_h}; pool may be exhausted"
        )
    return int(out_g[0]), int(out_a0[0]), int(out_a1[0])


def state_clone_wrapped(
    engine: "GoEngine", a0: "GoAgentState", a1: "GoAgentState"
) -> Tuple["GoEngine", "GoAgentState", "GoAgentState"]:
    """Object-wrapping convenience over state_clone().

    Returns owning GoEngine/GoAgentState objects for the clone (closing them
    frees the underlying handles, unlike the non-owning GoEngine._from_handle
    default used by SubgameSolver leaf export).
    """
    new_g, new_a0, new_a1 = state_clone(engine.handle, a0.handle, a1.handle)
    return (
        GoEngine._from_handle(new_g, owned=True),
        GoAgentState._from_handle(new_a0),
        GoAgentState._from_handle(new_a1),
    )


def get_token_stream_cap() -> int:
    """Return the live Go per-agent hard token-stream cap (agent.MaxTokenStream).

    Paired with cfr/src/cfr/prtcfr_worker.py::PRODUCTION_SEQ_CAP; callers that
    need to assert the invariant (Go cap >= PRODUCTION_SEQ_CAP) should read
    this live rather than trusting the module-level TOKEN_STREAM_CAP literal
    or PRODUCTION_SEQ_CAP to stay in sync by inspection.
    """
    return int(_get_lib().cambia_token_stream_cap())


def get_tokenizer_version() -> int:
    """Return the live Go tokenizer stream version (agent.TokenizerVersion).

    Paired with cfr/src/sequence_encoding.py::TOKENIZER_VERSION; the cross-check
    test reads this live rather than trusting the two to stay in sync by
    inspection. Bumped on every change to the produced token stream (cambia-612).
    """
    return int(_get_lib().cambia_tokenizer_version())


def get_nplayer_dims() -> Tuple[int, int]:
    """Return the live Go (N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS) pair.

    Paired with cfr/src/constants.py's N_PLAYER_INPUT_DIM/N_PLAYER_NUM_ACTIONS
    (which GoEngine.N_PLAYER_INPUT_DIM/N_PLAYER_NUM_ACTIONS and every N-player
    buffer allocation in this module derive from). The cross-check test reads
    this live rather than trusting the two to stay in sync by inspection --
    cambia-542 F8 was exactly that drift (bridge.py hardcoded 580/452 after
    the Go side moved to 856/620).
    """
    lib = _get_lib()
    return int(lib.cambia_nplayer_input_dim()), int(lib.cambia_nplayer_num_actions())


# Fixed field order of cambia_token_vocab (mirrors agent.TokenVocab). Consumers
# read positionally; the cross-check test asserts each against sequence_encoding.
TOKEN_VOCAB_FIELDS = [
    "VOCAB_SIZE",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "SEP_ID",
    "NUM_SPECIAL",
    "FRAME_BASE",
    "NUM_FRAME_IDS",
    "ACTOR_BASE",
    "MAX_ACTORS",
    "ACTION_BASE",
    "NUM_ACTION_IDS",
    "CARD_BASE",
    "NUM_CARD_IDS",
    "SLOT_BASE",
    "NUM_SLOT_IDS",
    "OUTCOME_BASE",
    "NUM_SNAP_OUTCOME_IDS",
    "MAX_SLOTS",
    "SEQ_CAP",
    "GO_TOKEN_STREAM_CAP",
    "PEEK_FRAME_BASE",
    "NUM_PEEK_FRAME_IDS",
    "RACE_FRAME_BASE",
    "NUM_RACE_FRAME_IDS",
]


def get_token_vocab() -> dict:
    """Return the Go token vocabulary layout as a {field: value} dict."""
    lib = _get_lib()
    n = len(TOKEN_VOCAB_FIELDS)
    buf = _ffi.new(f"int32_t[{n}]")
    written = lib.cambia_token_vocab(buf, n)
    if written < 0:
        raise RuntimeError(f"cambia_token_vocab failed (returned {written})")
    return {TOKEN_VOCAB_FIELDS[i]: int(buf[i]) for i in range(int(written))}


def encode_card_token(go_card_index: int) -> int:
    """Return the CARD-block token for a canonical Go card index (suit*13+rank)."""
    return int(_get_lib().cambia_token_encode_card(int(go_card_index)))


def encode_action_token(action_idx: int) -> int:
    """Return the ACTION-block token for a 2-player action index, or -1."""
    return int(_get_lib().cambia_token_encode_action(int(action_idx)))


# Frame widths keyed by frame-kind local id (order: init_peek, public, drawn,
# snap, cambia). Structural; matches the decoder in sequence_encoding.py.
_FRAME_WIDTHS = (3, 4, 2, 4, 2)

# Peek-result frame (cambia-529) lives in its own appended block: one marker id
# at PEEK_FRAME_BASE, width 4 [PEEK, OWNER, SLOT, CARD]. Paired with
# sequence_encoding.PEEK_FRAME_BASE / PEEK_FRAME_WIDTH; the constants cross-check
# test asserts the Go layout matches (get_token_vocab()["PEEK_FRAME_BASE"]).
_PEEK_FRAME_BASE = 325
_PEEK_FRAME_WIDTH = 4


def frame_aligned_window(
    token_body,
    seq_cap: int,
    add_bos_eos: bool = True,
    frame_base: int = 4,
    num_frame_ids: int = 5,
    bos_id: int = 1,
    eos_id: int = 2,
    peek_frame_base: int = _PEEK_FRAME_BASE,
    peek_frame_width: int = _PEEK_FRAME_WIDTH,
) -> List[int]:
    """Frame-aligned keep-most-recent window over a raw token body.

    Byte-exact reproduction of sequence_encoding.encode_observation_sequence's
    truncation + BOS/EOS wrapping, applied to a FULL token body returned by the
    Go side (or Python add_bos_eos=False path). Whole OLDEST frames are dropped
    until the body fits seq_cap (minus BOS/EOS), so the kept suffix always starts
    on a frame marker. ``seq_cap`` is a PARAMETER: production passes the raised
    window cap, tiny paths pass 256; the Go side itself never truncates.

    frame_base/num_frame_ids/bos_id/eos_id default to the tokenizer's layout but
    are overridable so the caller can pin them to the live sequence_encoding
    constants (the cross-check keeps Go and Python in lockstep).
    """
    body = [int(t) for t in token_body]
    # Segment the flat body into whole frames by walking frame markers.
    frames: List[List[int]] = []
    i = 0
    n = len(body)
    while i < n:
        tok = body[i]
        local = tok - frame_base
        if 0 <= local < num_frame_ids:
            width = _FRAME_WIDTHS[local]
        elif tok == peek_frame_base:
            # Peek-result frame (cambia-529): appended-block marker, fixed width.
            width = peek_frame_width
        else:
            raise ValueError(
                f"frame_aligned_window: expected a FRAME marker at pos {i}, got {tok}"
            )
        frames.append(body[i : i + width])
        i += width
    budget = seq_cap - (2 if add_bos_eos else 0)
    if budget < 0:
        budget = 0
    kept_rev: List[List[int]] = []
    used = 0
    for g in reversed(frames):
        if used + len(g) > budget:
            break
        kept_rev.append(g)
        used += len(g)
    kept = list(reversed(kept_rev))
    seq: List[int] = []
    if add_bos_eos:
        seq.append(bos_id)
    for g in kept:
        seq.extend(g)
    if add_bos_eos:
        seq.append(eos_id)
    return seq
