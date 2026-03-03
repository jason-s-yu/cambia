"""
Tournament Circuit Mode for Cambia.

N-player FFA rounds with cumulative scoring, aggression subsidies,
dealer rotation, disconnect handling, and OpenSkill rating.
Separate from the bracket-based tournament system in tournament.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── OpenSkill (Plackett-Luce) ────────────────────────────────────────


@dataclass
class OpenSkillRating:
    """Player skill rating using Plackett-Luce model."""

    mu: float = 25.0
    sigma: float = 25.0 / 3.0  # ≈ 8.333


def update_openskill(
    ratings: list[OpenSkillRating],
    ranks: list[int],
    beta: float = 8.0,
    tau: float = 0.0,
) -> list[OpenSkillRating]:
    """Update ratings after a match using Plackett-Luce.

    Args:
        ratings: Current ratings for each player.
        ranks: 1-indexed ordinal ranks (1=best). Equal ranks = tie.
        beta: Scale parameter (default 8.0).
        tau: Additive dynamics (default 0.0 for circuit mode).

    Returns:
        New ratings for each player.
    """
    n = len(ratings)
    # Optionally inflate sigma with dynamics
    working = []
    for r in ratings:
        if tau > 0:
            new_sigma = math.sqrt(r.sigma**2 + tau**2)
        else:
            new_sigma = r.sigma
        working.append(OpenSkillRating(mu=r.mu, sigma=new_sigma))

    delta_mu = [0.0] * n
    info = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            si = working[i].sigma
            sj = working[j].sigma
            c = math.sqrt(2 * beta**2 + si**2 + sj**2)

            # Plackett-Luce pairwise win probability
            exp_i = math.exp(working[i].mu / c)
            exp_j = math.exp(working[j].mu / c)
            p_ij = exp_i / (exp_i + exp_j)

            # Score: 1 if i beats j, 0 if j beats i, 0.5 if tie
            if ranks[i] < ranks[j]:
                s_ij = 1.0
            elif ranks[i] > ranks[j]:
                s_ij = 0.0
            else:
                s_ij = 0.5

            delta_mu[i] += (si**2 / c) * (s_ij - p_ij)
            gamma = si / c
            info[i] += gamma * (1 - gamma)

    new_ratings = []
    for i in range(n):
        new_mu = working[i].mu + delta_mu[i]
        si = working[i].sigma
        variance_reduction = si**2 * info[i]
        new_sigma = si * math.sqrt(max(1 - variance_reduction, 0.0001))
        new_ratings.append(OpenSkillRating(mu=new_mu, sigma=new_sigma))

    return new_ratings


def ranks_from_scores(scores: list[int], tie_margin: int = 3) -> list[int]:
    """Convert cumulative scores to ordinal ranks with tie margin.

    Players within tie_margin of the group minimum get the same rank.
    Lower score = better rank.
    """
    n = len(scores)
    if n == 0:
        return []

    # Sort indices by score ascending
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0] * n

    pos = 0
    while pos < n:
        group_min = scores[order[pos]]
        group_end = pos
        # Extend group while within tie_margin of group minimum
        while group_end < n and scores[order[group_end]] <= group_min + tie_margin:
            group_end += 1
        rank = pos + 1  # 1-indexed rank for this group
        for k in range(pos, group_end):
            ranks[order[k]] = rank
        pos = group_end

    return ranks


# ── Tournament House Rules ───────────────────────────────────────────


def tournament_house_rules() -> dict[str, object]:
    """Return enforced house rules for circuit/tournament mode.

    Per T1:
      - allowDrawFromDiscardPile = True
      - allowReplaceAbilities = True
      - lockCallerHand = False

    Returns a dict suitable for passing to CambiaRulesConfig or game engine.
    """
    return {
        "allowDrawFromDiscardPile": True,
        "allowReplaceAbilities": True,
        "lockCallerHand": False,
    }


# ── Circuit Types ────────────────────────────────────────────────────


@dataclass
class CircuitConfig:
    format: str = "standard"  # "quick"|"standard"|"championship"
    num_players: int = 4
    num_rounds: int = 0  # 0 = auto from format
    player_ids: list[int] = field(default_factory=list)
    missed_round_score: int = 41
    abandon_threshold: int = 2


@dataclass
class CircuitPlayerState:
    player_id: int
    cumulative_score: int = 0
    raw_cumulative: int = 0
    round_scores: list[int] = field(default_factory=list)
    round_placements: list[int] = field(default_factory=list)
    h2h_record: dict[int, list[int]] = field(default_factory=dict)  # {opp: [wins, losses]}
    best_round: int = 999_999
    consecutive_misses: int = 0
    abandoned: bool = False


@dataclass
class CircuitRoundResult:
    round_num: int
    player_scores: dict[int, int]
    placements: list[int]
    cambia_caller_id: int = -1
    subsidies: dict[int, int] = field(default_factory=dict)
    dealer_id: int = -1
    first_actor_id: int = -1
    forfeited: dict[int, bool] = field(default_factory=dict)


@dataclass
class CircuitResult:
    """Final result of a completed circuit tournament."""

    standings: list[CircuitPlayerState]
    rounds: list[CircuitRoundResult]
    rating_changes: dict[int, OpenSkillRating] = field(default_factory=dict)


# Subsidy tables by placement position (0-indexed)
_SUBSIDY_4P = [-5, -2, 0, 0]
_SUBSIDY_5P_PLUS = lambda n: [-5, -2, -1] + [0] * (n - 3)


def _get_subsidies_for_n(n: int) -> list[int]:
    """Return subsidy list for n players (indexed by 0-based placement)."""
    if n <= 4:
        return _SUBSIDY_4P[:n] if n < 4 else list(_SUBSIDY_4P)
    return _SUBSIDY_5P_PLUS(n)


class CircuitState:
    """Full mutable state of a circuit tournament."""

    def __init__(self, config: CircuitConfig) -> None:
        FORMAT_ROUNDS = {"quick": 8, "standard": 12, "championship": 20}
        if config.num_rounds == 0:
            config.num_rounds = FORMAT_ROUNDS.get(config.format, 12)

        if config.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if config.num_players < 2:
            raise ValueError("num_players must be >= 2")
        if config.num_rounds % config.num_players != 0:
            raise ValueError(
                f"num_rounds ({config.num_rounds}) must be a multiple of "
                f"num_players ({config.num_players})"
            )
        if len(config.player_ids) != config.num_players:
            raise ValueError("len(player_ids) must equal num_players")

        if config.missed_round_score == 0:
            config.missed_round_score = 41
        if config.abandon_threshold == 0:
            config.abandon_threshold = 2

        self.config = config
        self.players = [CircuitPlayerState(player_id=pid) for pid in config.player_ids]
        self._player_map: dict[int, CircuitPlayerState] = {
            p.player_id: p for p in self.players
        }
        self.rounds: list[CircuitRoundResult] = []
        self.current_round = 0
        self.dealer_seat = 0
        self.completed = False

    def record_round(self, scores: dict[int, int], cambia_caller_id: int = -1) -> None:
        """Record a completed round's results."""
        participating = {
            pid: s for pid, s in scores.items() if pid in self._player_map
        }
        n = len(participating)
        subsidy_table = _get_subsidies_for_n(n)

        # Sort player IDs by score ascending; Cambia caller wins ties (placed first)
        def sort_key(pid: int) -> tuple:
            s = participating[pid]
            is_caller = 0 if pid == cambia_caller_id else 1
            return (s, is_caller)

        sorted_pids = sorted(participating.keys(), key=sort_key)

        # Assign 1-indexed placements (tied scores share same placement)
        placement_map: dict[int, int] = {}
        pos = 0
        while pos < n:
            pid = sorted_pids[pos]
            score = participating[pid]
            group_end = pos + 1
            # Group players with same score
            while group_end < n and participating[sorted_pids[group_end]] == score:
                group_end += 1
            rank = pos + 1
            for k in range(pos, group_end):
                placement_map[sorted_pids[k]] = rank
            pos = group_end

        # Determine subsidy for each player by their sorted position
        # Ties: Cambia caller gets higher bonus; if no caller in tie, BOTH get higher bonus
        position_subsidy: dict[int, int] = {}
        pos = 0
        while pos < n:
            pid = sorted_pids[pos]
            score = participating[pid]
            # Find all players tied at this score
            group = [sorted_pids[pos]]
            k = pos + 1
            while k < n and participating[sorted_pids[k]] == score:
                group.append(sorted_pids[k])
                k += 1

            if len(group) == 1:
                position_subsidy[pid] = subsidy_table[pos]
                pos += 1
                continue

            # Tied group: positions pos..pos+len(group)-1
            # Cambia caller (if present) gets the best (lowest index) position's subsidy
            # Others get the worst position's subsidy in the group, UNLESS no caller in group
            caller_in_group = cambia_caller_id in group
            best_subsidy = subsidy_table[pos]  # highest bonus (most negative = best)
            worst_subsidy_idx = pos + len(group) - 1
            worst_subsidy = subsidy_table[min(worst_subsidy_idx, n - 1)]

            for gpid in group:
                if caller_in_group:
                    if gpid == cambia_caller_id:
                        position_subsidy[gpid] = best_subsidy
                    else:
                        position_subsidy[gpid] = worst_subsidy
                else:
                    # No caller in group: all get best subsidy
                    position_subsidy[gpid] = best_subsidy

            pos += len(group)

        subsidies: dict[int, int] = {}
        for pid in participating:
            subsidies[pid] = position_subsidy[pid]

        # Update player states
        for pid, raw_score in participating.items():
            p = self._player_map[pid]
            subsidy = subsidies[pid]
            p.raw_cumulative += raw_score
            p.cumulative_score += raw_score + subsidy
            p.round_scores.append(raw_score)
            p.round_placements.append(placement_map[pid])
            if raw_score < p.best_round:
                p.best_round = raw_score
            p.consecutive_misses = 0

        # Update H2H records
        pid_list = list(participating.keys())
        for i in range(len(pid_list)):
            for j in range(i + 1, len(pid_list)):
                a = pid_list[i]
                b = pid_list[j]
                sa = participating[a]
                sb = participating[b]
                if sa == sb:
                    # Tie: Cambia caller wins; neither caller = no update
                    if a == cambia_caller_id:
                        self._h2h_win(a, b)
                    elif b == cambia_caller_id:
                        self._h2h_win(b, a)
                    # else: true tie, no update
                elif sa < sb:
                    self._h2h_win(a, b)
                else:
                    self._h2h_win(b, a)

        dealer_id = self.next_dealer_seat()
        first_actor_id = self.next_first_actor()

        result = CircuitRoundResult(
            round_num=self.current_round + 1,
            player_scores=dict(participating),
            placements=list(placement_map.values()),
            cambia_caller_id=cambia_caller_id,
            subsidies=subsidies,
            dealer_id=dealer_id,
            first_actor_id=first_actor_id,
        )
        self.rounds.append(result)
        self.current_round += 1
        self.dealer_seat = (self.dealer_seat + 1) % self.config.num_players
        if self.current_round >= self.config.num_rounds:
            self.completed = True

    def _h2h_win(self, winner: int, loser: int) -> None:
        pw = self._player_map[winner]
        pl = self._player_map[loser]
        if loser not in pw.h2h_record:
            pw.h2h_record[loser] = [0, 0]
        if winner not in pl.h2h_record:
            pl.h2h_record[winner] = [0, 0]
        pw.h2h_record[loser][0] += 1
        pl.h2h_record[winner][1] += 1

    def record_missed_round(self, player_id: int) -> None:
        """Record a player missing a round (scores 41)."""
        p = self._player_map[player_id]
        miss_score = self.config.missed_round_score
        p.raw_cumulative += miss_score
        p.cumulative_score += miss_score
        p.round_scores.append(miss_score)
        p.consecutive_misses += 1

        if p.consecutive_misses >= self.config.abandon_threshold:
            p.abandoned = True
            remaining = self.config.num_rounds - len(p.round_scores)
            penalty = miss_score * remaining
            p.raw_cumulative += penalty
            p.cumulative_score += penalty

    def record_reconnection(self, player_id: int) -> None:
        """Reset consecutive miss counter on reconnect."""
        self._player_map[player_id].consecutive_misses = 0

    def next_dealer_seat(self) -> int:
        """Return player ID of current round's dealer."""
        return self.config.player_ids[self.dealer_seat]

    def next_first_actor(self) -> int:
        """Return player ID of first actor (left of dealer)."""
        return self.config.player_ids[
            (self.dealer_seat + 1) % len(self.config.player_ids)
        ]

    def get_standings(self) -> list[CircuitPlayerState]:
        """Return players sorted by tournament standing."""

        def h2h_total_wins(p: CircuitPlayerState) -> int:
            return sum(v[0] for v in p.h2h_record.values())

        return sorted(
            self.players,
            key=lambda p: (
                p.cumulative_score,
                p.raw_cumulative,
                -h2h_total_wins(p),
                p.best_round,
                p.player_id,
            ),
        )

    def is_complete(self) -> bool:
        return self.completed


class CircuitRunner:
    """Run a full circuit tournament with agent callables."""

    def __init__(
        self,
        config: CircuitConfig,
        agent_factory: Callable,
        game_factory: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.agent_factory = agent_factory
        self.game_factory = game_factory
        self.state = CircuitState(config)

    def run(self) -> CircuitResult:
        """Execute the full tournament and return results."""
        if self.game_factory is None:
            raise NotImplementedError("game_factory required for CircuitRunner.run()")

        while not self.state.is_complete():
            scores, cambia_caller_id = self.game_factory(
                player_ids=self.config.player_ids,
                dealer_id=self.state.next_dealer_seat(),
                first_actor_id=self.state.next_first_actor(),
            )
            self.state.record_round(scores, cambia_caller_id)

        standings = self.state.get_standings()
        return CircuitResult(standings=standings, rounds=self.state.rounds)
