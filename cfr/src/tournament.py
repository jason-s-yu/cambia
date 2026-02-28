"""
Tournament infrastructure for Cambia.

Supports three modes:
  - "series": fixed number of rounds, all players vs all others, cumulative scoring.
  - "single_elimination": standard single-elimination bracket with byes.
  - "double_elimination": winners + losers bracket; two losses to eliminate.

Usage::

    from src.tournament import TournamentConfig, TournamentState

    cfg = TournamentConfig(
        mode="single_elimination",
        player_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    t = TournamentState(cfg)
    while not t.is_complete():
        p1, p2, is_bye = t.next_matchup()
        if is_bye:
            continue
        scores = run_game(p1, p2)  # returns [score_p1, score_p2]
        t.record_result(p1, p2, scores)
    standings = t.get_standings()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TournamentConfig:
    """Configuration for a tournament."""

    mode: str = "series"  # "series" | "single_elimination" | "double_elimination"
    num_rounds: int = 1  # for series mode
    games_per_matchup: int = 1  # best-of-N
    scoring_method: str = "wins"  # "wins" | "cumulative_score" | "lowest_cumulative"
    seeding_method: str = "random"  # "random" | "by_score"
    player_ids: List[int] = field(default_factory=list)


@dataclass
class PlayerStanding:
    """Tracks a player's tournament performance."""

    player_id: int
    score: int = 0
    wins: int = 0
    losses: int = 0
    eliminated: bool = False


@dataclass
class Matchup:
    """A single game between two players."""

    player1: int
    player2: int = -1  # -1 for bye
    winner: int = -1  # -1 until played
    scores: List[int] = field(default_factory=list)
    bye: bool = False
    played: bool = False


@dataclass
class BracketRound:
    """One round of bracket matchups."""

    matchups: List[Matchup] = field(default_factory=list)


class TournamentState:
    """Full mutable state of a tournament."""

    def __init__(self, config: TournamentConfig) -> None:
        if config.games_per_matchup <= 0:
            config = TournamentConfig(**{**config.__dict__, "games_per_matchup": 1})
        if not config.scoring_method:
            config = TournamentConfig(**{**config.__dict__, "scoring_method": "wins"})

        self.config = config
        self.bracket: List[BracketRound] = []
        self.standings: List[PlayerStanding] = [
            PlayerStanding(player_id=pid) for pid in config.player_ids
        ]
        self.current_round: int = 0
        self.completed: bool = False

        # Internal series schedule: list of rounds, each a list of (p1, p2) pairs.
        self._series_schedule: List[List[Tuple[int, int]]] = []
        # Internal double-elim losers bracket.
        self._losers_rounds: List[BracketRound] = []
        self._losers_round: int = 0

        if config.mode == "single_elimination":
            self._build_single_elim_bracket()
        elif config.mode == "double_elimination":
            self._build_double_elim_bracket()
        else:
            self._build_series_schedule()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_single_elim_bracket(self) -> None:
        players = self.config.player_ids
        n = len(players)
        if n <= 1:
            self.completed = True
            return

        # Next power of two >= n.
        slots = 1
        while slots < n:
            slots <<= 1
        num_byes = slots - n

        round1 = BracketRound()
        byes_assigned = 0
        lo, hi = 0, n - 1
        while lo < hi:
            m = Matchup(player1=players[lo])
            if byes_assigned < num_byes:
                m.bye = True
                m.player2 = -1
                m.winner = players[lo]
                m.played = True
                byes_assigned += 1
                lo += 1
            else:
                m.player2 = players[hi]
                hi -= 1
                lo += 1
            round1.matchups.append(m)
        if lo == hi:
            round1.matchups.append(
                Matchup(
                    player1=players[lo],
                    player2=-1,
                    bye=True,
                    winner=players[lo],
                    played=True,
                )
            )

        self.bracket = [round1]
        total_rounds = math.ceil(math.log2(n)) if n > 1 else 0
        for _ in range(1, total_rounds):
            self.bracket.append(BracketRound())

    def _build_double_elim_bracket(self) -> None:
        self._build_single_elim_bracket()
        self._losers_rounds = []
        self._losers_round = 0

    def _build_series_schedule(self) -> None:
        players = self.config.player_ids
        rounds = max(self.config.num_rounds, 1)
        schedule: List[List[Tuple[int, int]]] = []
        for _ in range(rounds):
            pairs: List[Tuple[int, int]] = []
            for i in range(len(players)):
                for j in range(i + 1, len(players)):
                    pairs.append((players[i], players[j]))
            schedule.append(pairs)
        self._series_schedule = schedule

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_matchup(self) -> Tuple[int, int, bool]:
        """Return (player1, player2, is_bye) for the next unplayed matchup.

        Returns (-1, -1, False) if all matchups in the current batch are done
        (check is_complete() or wait for more results).
        Raises RuntimeError if the tournament is complete.
        """
        if self.completed:
            raise RuntimeError("tournament is complete")
        if self.config.mode == "single_elimination":
            return self._next_single_elim_matchup()
        elif self.config.mode == "double_elimination":
            return self._next_double_elim_matchup()
        else:
            return self._next_series_matchup()

    def record_result(self, player1: int, player2: int, scores: List[int]) -> None:
        """Record the outcome of a matchup.

        scores[0] is player1's score, scores[1] is player2's score.
        Lower score wins in Cambia.
        """
        if self.completed:
            raise RuntimeError("tournament is complete")
        if self.config.mode == "single_elimination":
            self._record_single_elim(player1, player2, scores, in_losers=False)
        elif self.config.mode == "double_elimination":
            self._record_double_elim(player1, player2, scores)
        else:
            self._record_series(player1, player2, scores)

    def get_standings(self) -> List[PlayerStanding]:
        """Return standings sorted by the configured scoring method."""
        out = list(self.standings)

        def key(s: PlayerStanding):
            elim_key = 1 if s.eliminated else 0
            if self.config.scoring_method == "lowest_cumulative":
                return (elim_key, s.score, s.player_id)
            elif self.config.scoring_method == "cumulative_score":
                return (elim_key, -s.score, s.player_id)
            else:  # "wins"
                return (elim_key, -s.wins, s.player_id)

        out.sort(key=key)
        return out

    def is_complete(self) -> bool:
        """Return True when the tournament has finished."""
        return self.completed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _standing_for(self, pid: int) -> Optional[PlayerStanding]:
        for s in self.standings:
            if s.player_id == pid:
                return s
        return None

    def _determine_winner(self, p1: int, p2: int, scores: List[int]) -> int:
        """Lower score wins in Cambia."""
        if len(scores) < 2:
            return p1
        return p1 if scores[0] <= scores[1] else p2

    # -- Series --

    def _is_series_matchup_played(self, p1: int, p2: int) -> bool:
        if self.current_round >= len(self.bracket):
            return False
        for m in self.bracket[self.current_round].matchups:
            if (m.player1 == p1 and m.player2 == p2) or (
                m.player1 == p2 and m.player2 == p1
            ):
                return m.played
        return False

    def _next_series_matchup(self) -> Tuple[int, int, bool]:
        if self.current_round >= len(self._series_schedule):
            return -1, -1, False
        for p1, p2 in self._series_schedule[self.current_round]:
            if not self._is_series_matchup_played(p1, p2):
                return p1, p2, False
        return -1, -1, False

    def _is_series_round_complete(self, round_idx: int) -> bool:
        if round_idx >= len(self._series_schedule):
            return True
        expected = len(self._series_schedule[round_idx])
        if round_idx >= len(self.bracket):
            return False
        played = sum(1 for m in self.bracket[round_idx].matchups if m.played)
        return played >= expected

    def _record_series(self, p1: int, p2: int, scores: List[int]) -> None:
        winner = self._determine_winner(p1, p2, scores)
        loser = p2 if winner == p1 else p1

        s1 = self._standing_for(p1)
        s2 = self._standing_for(p2)
        if s1 and scores:
            s1.score += scores[0]
        if s2 and len(scores) > 1:
            s2.score += scores[1]
        ws = self._standing_for(winner)
        ls = self._standing_for(loser)
        if ws:
            ws.wins += 1
        if ls:
            ls.losses += 1

        while self.current_round >= len(self.bracket):
            self.bracket.append(BracketRound())
        self.bracket[self.current_round].matchups.append(
            Matchup(player1=p1, player2=p2, winner=winner, scores=scores, played=True)
        )

        if self._is_series_round_complete(self.current_round):
            self.current_round += 1
            if self.current_round >= self.config.num_rounds:
                self.completed = True

    # -- Single elimination --

    def _next_single_elim_matchup(self) -> Tuple[int, int, bool]:
        if self.current_round >= len(self.bracket):
            return -1, -1, False
        for m in self.bracket[self.current_round].matchups:
            if not m.played:
                return m.player1, m.player2, m.bye
        return -1, -1, False

    def _is_round_complete(self, br: BracketRound) -> bool:
        return all(m.played for m in br.matchups)

    def _record_single_elim(
        self, p1: int, p2: int, scores: List[int], in_losers: bool
    ) -> None:
        bracket = self._losers_rounds if in_losers else self.bracket
        current = self._losers_round if in_losers else self.current_round

        if current >= len(bracket):
            raise ValueError("no active round in bracket")

        br = bracket[current]
        idx = None
        for i, m in enumerate(br.matchups):
            if m.played:
                continue
            if (m.player1 == p1 and m.player2 == p2) or (
                m.player1 == p2 and m.player2 == p1
            ):
                idx = i
                break
        if idx is None:
            raise ValueError(f"matchup ({p1},{p2}) not found in current round")

        m = br.matchups[idx]
        winner = m.player1 if m.bye else self._determine_winner(m.player1, m.player2, scores)
        loser = m.player2 if winner == m.player1 else m.player1

        m.winner = winner
        m.scores = scores
        m.played = True

        ws = self._standing_for(winner)
        ls = self._standing_for(loser)
        if ws:
            ws.wins += 1
        if ls:
            ls.losses += 1
            if not in_losers:
                ls.eliminated = True

        if self._is_round_complete(br):
            if not in_losers:
                self._advance_single_elim_round()

    def _advance_single_elim_round(self) -> None:
        if self.current_round >= len(self.bracket):
            return
        br = self.bracket[self.current_round]
        winners = [m.winner for m in br.matchups if m.winner >= 0]
        self.current_round += 1

        if len(winners) <= 1:
            self.completed = True
            return

        if self.current_round >= len(self.bracket):
            self.bracket.append(BracketRound())

        next_br = self.bracket[self.current_round]
        i = 0
        while i + 1 < len(winners):
            next_br.matchups.append(
                Matchup(player1=winners[i], player2=winners[i + 1], winner=-1)
            )
            i += 2
        if len(winners) % 2 == 1:
            w = winners[-1]
            next_br.matchups.append(
                Matchup(player1=w, player2=-1, bye=True, winner=w, played=True)
            )

    # -- Double elimination --

    def _next_double_elim_matchup(self) -> Tuple[int, int, bool]:
        p1, p2, bye = self._next_single_elim_matchup()
        if p1 != -1:
            return p1, p2, bye
        if self._losers_round < len(self._losers_rounds):
            for m in self._losers_rounds[self._losers_round].matchups:
                if not m.played:
                    return m.player1, m.player2, m.bye
        return -1, -1, False

    def _drop_to_losers(self, pid: int) -> None:
        while self._losers_round >= len(self._losers_rounds):
            self._losers_rounds.append(BracketRound())
        lr = self._losers_rounds[self._losers_round]
        for m in lr.matchups:
            if m.player2 < 0 and not m.played:
                m.player2 = pid
                return
        lr.matchups.append(Matchup(player1=pid, player2=-1, winner=-1))

    def _advance_double_elim_losers(self) -> None:
        if self._losers_round >= len(self._losers_rounds):
            return
        lr = self._losers_rounds[self._losers_round]
        # Check all matchups with actual opponents are done.
        for m in lr.matchups:
            if not m.played and m.player2 >= 0:
                return
        survivors = [m.winner for m in lr.matchups if m.winner >= 0]
        if len(survivors) <= 1:
            self._check_double_elim_complete()
            return
        self._losers_round += 1
        while self._losers_round >= len(self._losers_rounds):
            self._losers_rounds.append(BracketRound())
        nr = self._losers_rounds[self._losers_round]
        i = 0
        while i + 1 < len(survivors):
            nr.matchups.append(
                Matchup(player1=survivors[i], player2=survivors[i + 1], winner=-1)
            )
            i += 2

    def _check_double_elim_complete(self) -> None:
        if self.current_round >= len(self.bracket) or not self.bracket[self.current_round].matchups:
            self.completed = True

    def _record_double_elim(self, p1: int, p2: int, scores: List[int]) -> None:
        # Try winners bracket first.
        if self.current_round < len(self.bracket):
            br = self.bracket[self.current_round]
            for m in br.matchups:
                if m.played:
                    continue
                if (m.player1 == p1 and m.player2 == p2) or (
                    m.player1 == p2 and m.player2 == p1
                ):
                    winner = self._determine_winner(m.player1, m.player2, scores)
                    loser = m.player2 if winner == m.player1 else m.player1
                    m.winner = winner
                    m.scores = scores
                    m.played = True

                    ws = self._standing_for(winner)
                    ls = self._standing_for(loser)
                    if ws:
                        ws.wins += 1
                    if ls:
                        ls.losses += 1
                        self._drop_to_losers(loser)

                    if self._is_round_complete(br):
                        self._advance_single_elim_round()
                        self._advance_double_elim_losers()
                    return

        # Try losers bracket.
        if self._losers_round < len(self._losers_rounds):
            lr = self._losers_rounds[self._losers_round]
            for m in lr.matchups:
                if m.played:
                    continue
                if (m.player1 == p1 and m.player2 == p2) or (
                    m.player1 == p2 and m.player2 == p1
                ):
                    winner = self._determine_winner(m.player1, m.player2, scores)
                    loser = m.player2 if winner == m.player1 else m.player1
                    m.winner = winner
                    m.scores = scores
                    m.played = True

                    ws = self._standing_for(winner)
                    ls = self._standing_for(loser)
                    if ws:
                        ws.wins += 1
                    if ls:
                        ls.losses += 1
                        ls.eliminated = True  # second loss

                    if self._is_round_complete(lr):
                        self._losers_round += 1
                        self._advance_double_elim_losers()
                    return

        raise ValueError(f"matchup ({p1},{p2}) not found in any active bracket")
