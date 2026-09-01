"""
tests/test_subgame_guard.py

Regression tests for cambia-1554: SubgameSolver (and the Go cgo exports it
wraps: cambia_subgame_build / cambia_subgame_solve / cambia_subgame_solve_ranged)
must refuse a table with more than two players instead of crashing the whole
interpreter.

Ticket reproduction this guards against: a 4-player GoEngine handed to
SubgameSolver used to panic inside libcambia.so ("panic: runtime error: index
out of range [255] with length 2" at subgame_solver.go:174, reached via
cambia_subgame_solve), aborting the Python process outright - opp := 1-player
underflows to 255 once ActingPlayer() is 2..7.

Unlike tests/test_subgame_bridge.py (module-skipped: "ReBeL is deprecated"),
this file is NOT skipped. It exercises exactly the guard this ticket adds,
and the whole point is proving the interpreter survives to report a normal
Python exception.
"""

import warnings

import pytest

from src.config import CambiaRulesConfig
from src.ffi.bridge import GoEngine, SubgameSolver


def _make_game(num_players: int, seed: int = 1) -> GoEngine:
    """A genuine N-player GoEngine (not just a Python-side attribute -
    house_rules routes through cambia_game_new_with_rules, so
    game.num_players() reflects the real Go-side seat count)."""
    rules = CambiaRulesConfig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return GoEngine(seed=seed, house_rules=rules, num_players=num_players)


class TestSubgameSolverRejectsNonTwoPlayer:
    """SubgameSolver must raise a clear Python error, not crash, at 3-8 seats
    (AC1/AC2)."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6, 7, 8])
    def test_construction_raises_value_error(self, num_players):
        game = _make_game(num_players)
        try:
            assert game.num_players() == num_players
            with pytest.raises(ValueError, match="2-player"):
                SubgameSolver(game, max_depth=2)
        finally:
            game.close()

    def test_four_player_reproduces_ticket_scenario(self):
        """Exact reproduction from cambia-1554: 4-player table, max_depth=2,
        matching the ticket's own repro steps."""
        game = _make_game(4)
        try:
            with pytest.raises(ValueError):
                SubgameSolver(game, max_depth=2)

            # The interpreter is still alive and the game handle is still
            # usable afterwards - this is the actual regression: before the
            # fix, the process aborted here and nothing below ever ran.
            assert game.num_players() == 4
            assert game.is_terminal() in (True, False)
        finally:
            game.close()

    def test_repeated_rejected_construction_does_not_leak_or_crash(self):
        """Constructing (and failing) several times in a row must not corrupt
        the solver pool or crash on a later, valid build."""
        for _ in range(5):
            game = _make_game(3)
            try:
                with pytest.raises(ValueError):
                    SubgameSolver(game, max_depth=1)
            finally:
                game.close()

        # A subsequent 2-player build must still succeed: the guard must not
        # have left the pool in a bad state.
        game2 = _make_game(2)
        try:
            with SubgameSolver(game2, max_depth=1) as solver:
                assert solver.leaf_count >= 0
        finally:
            game2.close()


class TestSubgameSolverAcceptsTwoPlayers:
    """Control: the guard must not reject the table it actually supports."""

    def test_two_player_construction_succeeds(self):
        game = _make_game(2)
        try:
            with SubgameSolver(game, max_depth=2) as solver:
                assert solver.leaf_count >= 0
        finally:
            game.close()
