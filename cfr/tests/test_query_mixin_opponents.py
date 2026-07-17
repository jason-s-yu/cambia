"""
tests/test_query_mixin_opponents.py

cambia-542 F2: get_opponent_index() silently returned 1 - player_index at
any num_players, which is only a well-defined "the opponent" at N=2 and
produces a plausible-but-wrong in-bounds index at N=3+ for player_index in
{0, 1} (only player_index >= 2 produced a negative, caught-by-bounds-check
index under the old code). These tests cover:
  - get_opponent_index() unchanged behavior at num_players == 2.
  - get_opponent_index() raises NotImplementedError at num_players != 2
    instead of returning a silently-wrong index.
  - get_opponent_index()'s existing invalid-player_index guard (-1) still
    fires before the num_players check.
  - get_opponents(), the new N-aware replacement API, at N=2..5.
"""

import pytest

from src.game.engine import CambiaGameState


def _make_state(num_players: int) -> CambiaGameState:
    """Minimal CambiaGameState for exercising QueryMixin methods that only
    read self.num_players / self.players -- no game setup needed."""
    return CambiaGameState(players=[], num_players=num_players)


class TestGetOpponentIndexTwoPlayer:
    def test_player_zero(self):
        state = _make_state(2)
        assert state.get_opponent_index(0) == 1

    def test_player_one(self):
        state = _make_state(2)
        assert state.get_opponent_index(1) == 0

    def test_invalid_player_index_returns_sentinel(self):
        """Out-of-range player_index is checked before the num_players guard
        and still returns the -1 error sentinel (unchanged behavior)."""
        state = _make_state(2)
        assert state.get_opponent_index(5) == -1
        assert state.get_opponent_index(-1) == -1


class TestGetOpponentIndexRaisesAtNPlayer:
    @pytest.mark.parametrize("num_players", [3, 4, 5, 8])
    @pytest.mark.parametrize("player_index", [0, 1, 2])
    def test_raises_not_implemented(self, num_players, player_index):
        if player_index >= num_players:
            pytest.skip("player_index must be valid for this num_players")
        state = _make_state(num_players)
        with pytest.raises(NotImplementedError):
            state.get_opponent_index(player_index)

    def test_does_not_silently_return_wrong_index(self):
        """Regression guard for the exact cambia-542 F2 bug: at N=3, players
        0 and 1 used to get a plausible-but-wrong in-bounds index (1, 0
        respectively) instead of an error."""
        state = _make_state(3)
        with pytest.raises(NotImplementedError):
            state.get_opponent_index(0)
        with pytest.raises(NotImplementedError):
            state.get_opponent_index(1)


class TestGetOpponents:
    def test_two_player(self):
        state = _make_state(2)
        assert state.get_opponents(0) == [1]
        assert state.get_opponents(1) == [0]

    def test_three_player(self):
        state = _make_state(3)
        assert state.get_opponents(0) == [1, 2]
        assert state.get_opponents(1) == [0, 2]
        assert state.get_opponents(2) == [0, 1]

    def test_five_player_ascending_order(self):
        state = _make_state(5)
        assert state.get_opponents(2) == [0, 1, 3, 4]

    def test_invalid_player_index_returns_empty_list(self):
        state = _make_state(3)
        assert state.get_opponents(5) == []
        assert state.get_opponents(-1) == []
