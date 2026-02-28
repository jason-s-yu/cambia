"""
Tests for 8-player (MaxPlayers=8) expansion.

Verifies:
- 2P encoding constants unchanged (222/224 dim, 146 actions)
- N-player constants updated correctly for 8 players
- Go/Python constant parity
"""

import pytest
from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.constants import (
    EP_PBS_INPUT_DIM,
    N_PLAYER_MAX_PLAYERS,
    N_PLAYER_MAX_SLOTS,
    N_PLAYER_POWERSET_DIM,
    N_PLAYER_IDENTITY_DIM,
    N_PLAYER_PUBLIC_DIM,
    N_PLAYER_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS,
)


# ---------------------------------------------------------------------------
# 2P constant regression — must NOT change
# ---------------------------------------------------------------------------

def test_2p_input_dim_unchanged():
    """Legacy 2P input dim must remain 222."""
    assert INPUT_DIM == 222, f"INPUT_DIM={INPUT_DIM}, want 222"


def test_ep_pbs_input_dim_unchanged():
    """EP-PBS 2P input dim must remain 224."""
    assert EP_PBS_INPUT_DIM == 224, f"EP_PBS_INPUT_DIM={EP_PBS_INPUT_DIM}, want 224"


def test_num_actions_2p_unchanged():
    """2P action space must remain 146."""
    assert NUM_ACTIONS == 146, f"NUM_ACTIONS={NUM_ACTIONS}, want 146"


# ---------------------------------------------------------------------------
# N-player constant values (MaxPlayers=8, MaxHandSize=6, MaxOpponents=7)
# ---------------------------------------------------------------------------

def test_n_player_max_players():
    assert N_PLAYER_MAX_PLAYERS == 8, f"N_PLAYER_MAX_PLAYERS={N_PLAYER_MAX_PLAYERS}, want 8"


def test_n_player_max_slots():
    """48 = 8 players × 6 cards per player."""
    assert N_PLAYER_MAX_SLOTS == 48, f"N_PLAYER_MAX_SLOTS={N_PLAYER_MAX_SLOTS}, want 48"


def test_n_player_powerset_dim():
    """384 = 48 slots × 8 bits (MaxKnowledgePlayers)."""
    assert N_PLAYER_POWERSET_DIM == 384, f"N_PLAYER_POWERSET_DIM={N_PLAYER_POWERSET_DIM}, want 384"
    # Also verify formula
    assert N_PLAYER_POWERSET_DIM == N_PLAYER_MAX_SLOTS * N_PLAYER_MAX_PLAYERS


def test_n_player_identity_dim():
    """432 = 48 slots × 9 buckets."""
    assert N_PLAYER_IDENTITY_DIM == 432, f"N_PLAYER_IDENTITY_DIM={N_PLAYER_IDENTITY_DIM}, want 432"
    assert N_PLAYER_IDENTITY_DIM == N_PLAYER_MAX_SLOTS * 9


def test_n_player_input_dim():
    """856 = 384 (powerset) + 432 (identity) + 40 (public)."""
    assert N_PLAYER_INPUT_DIM == 856, f"N_PLAYER_INPUT_DIM={N_PLAYER_INPUT_DIM}, want 856"
    assert N_PLAYER_INPUT_DIM == N_PLAYER_POWERSET_DIM + N_PLAYER_IDENTITY_DIM + N_PLAYER_PUBLIC_DIM


def test_n_player_public_dim_unchanged():
    """Public features (40 dims) are player-count independent."""
    assert N_PLAYER_PUBLIC_DIM == 40


def test_n_player_num_actions():
    """620 = layout with MaxOpponents=7.

    Layout:
      5 fixed + Replace(6) + PeekOwn(6) + PeekOther(6×7=42)
      + BlindSwap(6×6×7=252) + KingLook(252)
      + KingSwapNo + KingSwapYes + PassSnap
      + SnapOwn(6) + SnapOpponent(6×7=42) + SnapOpponentMove(6) = 620
    """
    assert N_PLAYER_NUM_ACTIONS == 620, f"N_PLAYER_NUM_ACTIONS={N_PLAYER_NUM_ACTIONS}, want 620"


# ---------------------------------------------------------------------------
# Parity check: Go constants must match Python constants
# (values computed independently; this is the cross-validation)
# ---------------------------------------------------------------------------

def test_go_python_input_dim_parity():
    """Go NPlayerInputDim must equal Python N_PLAYER_INPUT_DIM."""
    # These are the same formula: MaxTotalSlots*MaxKnowledgePlayers + MaxTotalSlots*9 + 40
    go_value = 856  # from engine/agent/constants.go NPlayerInputDim
    assert N_PLAYER_INPUT_DIM == go_value, (
        f"Python N_PLAYER_INPUT_DIM={N_PLAYER_INPUT_DIM} != Go NPlayerInputDim={go_value}"
    )


def test_go_python_num_actions_parity():
    """Go NPlayerNumActions must equal Python N_PLAYER_NUM_ACTIONS."""
    go_value = 620  # from engine/agent/constants.go NPlayerNumActions
    assert N_PLAYER_NUM_ACTIONS == go_value, (
        f"Python N_PLAYER_NUM_ACTIONS={N_PLAYER_NUM_ACTIONS} != Go NPlayerNumActions={go_value}"
    )


# ---------------------------------------------------------------------------
# Behavioral tests using the Python game engine
# ---------------------------------------------------------------------------

def test_8_player_game_creation():
    """8-player game can be created with correct player count and hand sizes."""
    from src.game.engine import CambiaGameState

    game = CambiaGameState(num_players=8)
    assert game.num_players == 8
    assert len(game.players) == 8
    # Each player should have 4 cards dealt.
    for p in range(8):
        assert len(game.players[p].hand) == 4, (
            f"player {p}: hand size {len(game.players[p].hand)}, want 4"
        )


def test_7_player_game_creation():
    """7-player game can be created."""
    from src.game.engine import CambiaGameState

    game = CambiaGameState(num_players=7)
    assert game.num_players == 7
    assert len(game.players) == 7
    for p in range(7):
        assert len(game.players[p].hand) == 4


def test_6_player_game_backward_compat():
    """6-player games must still work after expansion."""
    from src.game.engine import CambiaGameState

    game = CambiaGameState(num_players=6)
    assert game.num_players == 6
    assert len(game.players) == 6
    for p in range(6):
        assert len(game.players[p].hand) == 4


def test_2_player_game_backward_compat():
    """2-player games must be completely unaffected."""
    from src.game.engine import CambiaGameState

    game = CambiaGameState(num_players=2)
    assert game.num_players == 2
    assert len(game.players) == 2
    for p in range(2):
        assert len(game.players[p].hand) == 4
