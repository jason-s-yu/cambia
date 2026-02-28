"""Tests for the tournament infrastructure (src/tournament.py)."""

import pytest

from src.tournament import BracketRound, Matchup, PlayerStanding, TournamentConfig, TournamentState


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


def test_interface_creation():
    """All types and methods are callable."""
    cfg = TournamentConfig(
        mode="series",
        num_rounds=2,
        games_per_matchup=1,
        scoring_method="wins",
        seeding_method="random",
        player_ids=[1, 2, 3, 4],
    )
    ts = TournamentState(cfg)
    assert ts is not None
    assert not ts.is_complete()
    standings = ts.get_standings()
    assert len(standings) == 4
    p1, p2, is_bye = ts.next_matchup()
    assert p1 != -1


def test_dataclass_defaults():
    """TournamentConfig defaults are sane."""
    cfg = TournamentConfig(player_ids=[1, 2])
    assert cfg.mode == "series"
    assert cfg.games_per_matchup == 1
    assert cfg.scoring_method == "wins"


# ---------------------------------------------------------------------------
# Series tests
# ---------------------------------------------------------------------------


def test_series_4player_3round():
    """4-player 3-round series. Scores accumulate. Standings sorted."""
    cfg = TournamentConfig(
        mode="series",
        num_rounds=3,
        scoring_method="wins",
        player_ids=[1, 2, 3, 4],
    )
    ts = TournamentState(cfg)

    played = 0
    while not ts.is_complete():
        p1, p2, _ = ts.next_matchup()
        assert p1 != -1, f"incomplete but no matchup after {played}"
        # Player with lower ID always wins.
        scores = [5, 10] if p1 < p2 else [10, 5]
        ts.record_result(p1, p2, scores)
        played += 1

    # C(4,2)=6 matchups/round × 3 rounds = 18.
    assert played == 18, f"expected 18, got {played}"
    assert ts.is_complete()

    standings = ts.get_standings()
    assert len(standings) == 4
    # Player 1 wins all matchups where p1=1, which is all (1,2),(1,3),(1,4) = 3/round × 3 = 9.
    assert standings[0].player_id == 1
    assert standings[0].wins == 9


def test_series_cumulative_score():
    """cumulative_score: higher total score is ranked first."""
    cfg = TournamentConfig(
        mode="series",
        num_rounds=1,
        scoring_method="cumulative_score",
        player_ids=[10, 20, 30],
    )
    ts = TournamentState(cfg)
    # Give p20 the highest cumulative score.
    results = [(10, 20, [3, 15]), (10, 30, [3, 12]), (20, 30, [8, 5])]
    for p1, p2, scores in results:
        ts.record_result(p1, p2, scores)

    assert ts.is_complete()
    standings = ts.get_standings()
    # Scores: p10=3+3=6, p20=15+8=23, p30=12+5=17
    assert standings[0].player_id == 20
    assert standings[0].score >= standings[1].score


def test_series_lowest_cumulative():
    """lowest_cumulative: lower total score is better."""
    cfg = TournamentConfig(
        mode="series",
        num_rounds=1,
        scoring_method="lowest_cumulative",
        player_ids=[1, 2],
    )
    ts = TournamentState(cfg)
    p1, p2, _ = ts.next_matchup()
    ts.record_result(p1, p2, [3, 10])

    standings = ts.get_standings()
    assert standings[0].score <= standings[1].score


def test_series_2players():
    """2-player 1-round series completes after 1 game."""
    cfg = TournamentConfig(mode="series", num_rounds=1, player_ids=[1, 2])
    ts = TournamentState(cfg)
    p1, p2, _ = ts.next_matchup()
    ts.record_result(p1, p2, [3, 8])
    assert ts.is_complete()


def test_series_complete_flag():
    """Completed flag advances correctly over multiple rounds."""
    cfg = TournamentConfig(mode="series", num_rounds=2, player_ids=[1, 2])
    ts = TournamentState(cfg)

    assert not ts.is_complete()
    p1, p2, _ = ts.next_matchup()
    ts.record_result(p1, p2, [1, 2])
    assert not ts.is_complete()

    p1, p2, _ = ts.next_matchup()
    ts.record_result(p1, p2, [1, 2])
    assert ts.is_complete()


def test_series_complete_raises():
    """next_matchup raises RuntimeError after tournament ends."""
    cfg = TournamentConfig(mode="series", num_rounds=1, player_ids=[1, 2])
    ts = TournamentState(cfg)
    p1, p2, _ = ts.next_matchup()
    ts.record_result(p1, p2, [1, 2])
    with pytest.raises(RuntimeError, match="tournament is complete"):
        ts.next_matchup()


# ---------------------------------------------------------------------------
# Single elimination tests
# ---------------------------------------------------------------------------


def test_single_elim_8_player_structure():
    """8 players → 3 rounds, 4 matchups in round 1."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=list(range(1, 9)))
    ts = TournamentState(cfg)
    assert len(ts.bracket) == 3
    assert len(ts.bracket[0].matchups) == 4


def test_single_elim_6_player_byes():
    """6 players → 2 byes in round 1 (next power of 2 = 8, 2 byes)."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=list(range(1, 7)))
    ts = TournamentState(cfg)
    byes = sum(1 for m in ts.bracket[0].matchups if m.bye)
    assert byes == 2


def test_single_elim_5_player_byes():
    """5 players → 3 byes (next power of 2 = 8, 3 byes)."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=list(range(1, 6)))
    ts = TournamentState(cfg)
    byes = sum(1 for m in ts.bracket[0].matchups if m.bye)
    assert byes == 3


def test_single_elim_2_players():
    """2-player single elim completes after 1 game."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=[1, 2])
    ts = TournamentState(cfg)
    assert not ts.is_complete()
    p1, p2, is_bye = ts.next_matchup()
    assert not is_bye
    ts.record_result(p1, p2, [3, 7])
    assert ts.is_complete()


def test_single_elim_8_player_full():
    """8-player bracket plays to completion. Lower ID always wins."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=list(range(1, 9)))
    ts = TournamentState(cfg)

    for _ in range(100):
        if ts.is_complete():
            break
        p1, p2, is_bye = ts.next_matchup()
        if p1 == -1:
            break
        if is_bye:
            continue
        ts.record_result(p1, p2, [5, 10])  # p1 always wins

    assert ts.is_complete()
    standings = ts.get_standings()
    assert standings[0].player_id == 1


def test_single_elim_loser_eliminated():
    """Losers are marked eliminated in single elimination."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=[1, 2, 3, 4])
    ts = TournamentState(cfg)

    while not ts.is_complete():
        p1, p2, is_bye = ts.next_matchup()
        if p1 == -1 or is_bye:
            continue
        ts.record_result(p1, p2, [5, 10])

    eliminated = [s for s in ts.standings if s.eliminated]
    # 3 players eliminated in 4-player single elim.
    assert len(eliminated) == 3


def test_single_elim_winner_advances():
    """Round 2 matchups contain round 1 winners."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=[10, 20, 30, 40])
    ts = TournamentState(cfg)
    # Play round 1.
    for _ in range(2):
        p1, p2, is_bye = ts.next_matchup()
        if not is_bye:
            ts.record_result(p1, p2, [5, 10])  # p1 always wins

    if len(ts.bracket) > 1 and ts.bracket[1].matchups:
        r2_players = set()
        for m in ts.bracket[1].matchups:
            r2_players.add(m.player1)
            if m.player2 >= 0:
                r2_players.add(m.player2)
        # All round 2 players should have been round 1 winners.
        r1_winners = {m.winner for m in ts.bracket[0].matchups if m.winner >= 0}
        assert r2_players.issubset(r1_winners)


# ---------------------------------------------------------------------------
# Double elimination tests
# ---------------------------------------------------------------------------


def test_double_elim_two_losses():
    """4-player double elim: player needs 2 losses to be eliminated."""
    cfg = TournamentConfig(mode="double_elimination", player_ids=[1, 2, 3, 4])
    ts = TournamentState(cfg)

    for _ in range(100):
        if ts.is_complete():
            break
        p1, p2, is_bye = ts.next_matchup()
        if p1 == -1 or is_bye:
            continue
        try:
            ts.record_result(p1, p2, [5, 10])
        except ValueError:
            break

    for s in ts.standings:
        if s.eliminated:
            assert s.losses >= 2, (
                f"player {s.player_id} eliminated with only {s.losses} loss"
            )


def test_double_elim_loser_dropped():
    """Loser of winners bracket game goes to losers bracket (not immediately eliminated)."""
    cfg = TournamentConfig(mode="double_elimination", player_ids=[1, 2, 3, 4])
    ts = TournamentState(cfg)

    # Play one winners bracket matchup.
    p1, p2, is_bye = ts.next_matchup()
    if not is_bye:
        ts.record_result(p1, p2, [5, 10])
        loser = p2  # p2 gets higher score
        ls = ts._standing_for(loser)
        # After one loss, should NOT be eliminated in double elim.
        assert not ls.eliminated
        assert ls.losses == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_one_player_trivial():
    """1-player tournament is immediately complete."""
    cfg = TournamentConfig(mode="single_elimination", player_ids=[42])
    ts = TournamentState(cfg)
    assert ts.is_complete()


def test_odd_players_series():
    """3-player series works correctly."""
    cfg = TournamentConfig(mode="series", num_rounds=2, player_ids=[1, 2, 3])
    ts = TournamentState(cfg)
    # C(3,2)=3 matchups/round × 2 = 6 total.
    played = 0
    while not ts.is_complete():
        p1, p2, _ = ts.next_matchup()
        ts.record_result(p1, p2, [1, 2])
        played += 1
    assert played == 6


def test_standings_not_mutated():
    """get_standings returns a new list each call."""
    cfg = TournamentConfig(mode="series", num_rounds=1, player_ids=[1, 2])
    ts = TournamentState(cfg)
    s1 = ts.get_standings()
    s2 = ts.get_standings()
    assert s1 is not s2
