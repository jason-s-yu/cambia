"""
PSRO (Policy-Space Response Oracles) meta-loop for robust N-player training.

Maintains a rolling population of checkpoint policies + heuristic baselines.
Opponents are sampled from the population during traversal. Population quality
is evaluated via round-robin tournaments using Plackett-Luce Δτ ranking.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PopulationMember:
    """A single policy in the PSRO population."""

    path: str  # Path to checkpoint file
    iteration: int  # Training iteration when added
    is_heuristic: bool = False  # True for built-in heuristic bots
    agent_type: str = ""  # Agent type name (for heuristics)
    rating: float = 0.0  # Current Plackett-Luce rating (τ)
    games_played: int = 0  # Total games played in evaluation


class PSROOracle:
    """
    Manages a PSRO population for N-player training.

    Population = rolling window of recent checkpoints + fixed heuristic baselines.

    Usage:
        oracle = PSROOracle(max_checkpoints=15, heuristic_types=["random", "greedy",
            "memory_heuristic"])
        oracle.add_checkpoint("/path/to/ckpt.pt", iteration=100)
        opponents = oracle.sample_opponents(num_opponents=3)
        oracle.evaluate_population(games_per_matchup=50, config=config)
    """

    def __init__(
        self,
        max_checkpoints: int = 15,
        heuristic_types: Optional[List[str]] = None,
    ):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[PopulationMember] = []
        self._heuristics: List[PopulationMember] = []

        # Initialize heuristic members
        if heuristic_types is None:
            heuristic_types = ["random", "greedy", "memory_heuristic"]
        for agent_type in heuristic_types:
            self._heuristics.append(
                PopulationMember(
                    path="",
                    iteration=-1,
                    is_heuristic=True,
                    agent_type=agent_type,
                )
            )

    @property
    def population(self) -> List[PopulationMember]:
        """All members: checkpoints + heuristics."""
        return self._checkpoints + self._heuristics

    @property
    def size(self) -> int:
        """Total population size."""
        return len(self._checkpoints) + len(self._heuristics)

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoint-based policies."""
        return len(self._checkpoints)

    def add_checkpoint(self, path: str, iteration: int) -> Optional[str]:
        """
        Add a checkpoint to the population.

        If population exceeds max_checkpoints, evicts the oldest checkpoint.

        Args:
            path: Path to checkpoint file.
            iteration: Training iteration when this checkpoint was created.

        Returns:
            Path of evicted checkpoint if one was removed, else None.
        """
        member = PopulationMember(path=path, iteration=iteration)
        self._checkpoints.append(member)

        evicted = None
        if len(self._checkpoints) > self.max_checkpoints:
            evicted_member = self._checkpoints.pop(0)  # FIFO eviction
            evicted = evicted_member.path
            logger.info(
                "PSRO: evicted checkpoint iter=%d (%s), population=%d",
                evicted_member.iteration,
                evicted_member.path,
                self.size,
            )

        logger.info(
            "PSRO: added checkpoint iter=%d, population=%d checkpoints + %d heuristics",
            iteration,
            len(self._checkpoints),
            len(self._heuristics),
        )
        return evicted

    def sample_opponents(self, num_opponents: int) -> List[PopulationMember]:
        """
        Sample opponents uniformly from the population.

        Args:
            num_opponents: Number of opponents to sample (N-1 for N-player game).

        Returns:
            List of PopulationMember instances to use as opponents.

        Raises:
            ValueError: If population is empty.
        """
        pop = self.population
        if not pop:
            raise ValueError("Cannot sample from empty PSRO population")

        # Sample with replacement if needed (population might be smaller than num_opponents)
        return [random.choice(pop) for _ in range(num_opponents)]

    def evaluate_population(
        self,
        games_per_matchup: int,
        num_players: int,
        config,
    ) -> Dict[str, float]:
        """
        Run round-robin tournament and compute Plackett-Luce Δτ ratings.

        Each matchup: sample `num_players` policies, play `games_per_matchup` games,
        record finish orderings. Compute Plackett-Luce ratings from orderings.

        Args:
            games_per_matchup: Games per player pairing.
            num_players: Number of players per game.
            config: Full Config object for game rules.

        Returns:
            Dict mapping member identifier to Δτ rating.
        """
        pop = self.population
        if len(pop) < num_players:
            logger.warning(
                "Population size %d < num_players %d, skipping evaluation",
                len(pop),
                num_players,
            )
            return {}

        # Collect finish orderings from round-robin games
        orderings = self._run_round_robin(pop, games_per_matchup, num_players, config)

        if not orderings:
            return {}

        # Compute Plackett-Luce ratings
        ratings = self._plackett_luce_ratings(pop, orderings)

        # Update member ratings
        result = {}
        for i, member in enumerate(pop):
            old_rating = member.rating
            member.rating = ratings[i]
            key = member.agent_type if member.is_heuristic else f"ckpt_{member.iteration}"
            result[key] = ratings[i] - old_rating  # Δτ

        return result

    def _run_round_robin(
        self,
        pop: List[PopulationMember],
        games_per_matchup: int,
        num_players: int,
        config,
    ) -> List[List[int]]:
        """
        Run round-robin tournament games.

        Returns list of orderings, where each ordering is a list of population
        indices sorted by finish position (first = winner).
        """
        from ..evaluate_agents import get_agent

        orderings = []
        n = len(pop)

        # Generate all combinations of num_players from population
        from itertools import combinations

        matchups = list(combinations(range(n), num_players))

        for matchup in matchups:
            for _ in range(games_per_matchup):
                try:
                    # Create agents for this matchup
                    agents = []
                    for seat, pop_idx in enumerate(matchup):
                        member = pop[pop_idx]
                        if member.is_heuristic:
                            agent = get_agent(member.agent_type, seat, config)
                        else:
                            agent = get_agent(
                                "nplayer" if num_players > 2 else "sd_cfr",
                                seat,
                                config,
                                checkpoint_path=member.path,
                                num_players=num_players,
                            )
                        agents.append(agent)

                    # Play game and get scores
                    scores = self._play_game(agents, num_players, config)

                    # Convert scores to ordering (lower score = better = earlier in list)
                    indexed = [(scores[i], matchup[i]) for i in range(num_players)]
                    indexed.sort(key=lambda x: x[0])  # Sort by score ascending
                    ordering = [idx for _, idx in indexed]
                    orderings.append(ordering)

                    # Update games_played
                    for pop_idx in matchup:
                        pop[pop_idx].games_played += 1

                except Exception as e:
                    logger.warning("PSRO eval game failed: %s", e)
                    continue

        return orderings

    def _play_game(self, agents, num_players: int, config) -> List[float]:
        """
        Play a single game and return scores for each seat.

        Returns list of raw scores (sum of card values).
        """
        from ..ffi.bridge import GoEngine

        engine = GoEngine(house_rules=config.cambia_rules, num_players=num_players)
        try:
            while not engine.is_terminal():
                player = engine.acting_player()

                if num_players > 2:
                    mask = engine.nplayer_legal_actions_mask()
                else:
                    mask = engine.legal_actions_mask()

                legal = np.where(mask > 0)[0]
                if len(legal) == 0:
                    break

                # Get agent's action choice
                agent = agents[player]
                action = self._agent_choose(agent, legal, engine, player)

                if num_players > 2:
                    engine.apply_nplayer_action(action)
                else:
                    engine.apply_action(action)

            if num_players > 2:
                utils = engine.get_nplayer_utility(num_players)
            else:
                utils = engine.get_utility()

            return utils.tolist()
        finally:
            engine.close()

    def _agent_choose(self, agent, legal_actions, engine, player_id) -> int:
        """Have an agent choose from legal actions. Falls back to random."""
        try:
            # BaseAgent subclasses have choose_action or similar
            if hasattr(agent, "choose_action_index"):
                return agent.choose_action_index(legal_actions, engine)
            # Simple fallback: random choice
            return int(np.random.choice(legal_actions))
        except Exception:
            return int(np.random.choice(legal_actions))

    def _plackett_luce_ratings(
        self,
        pop: List[PopulationMember],
        orderings: List[List[int]],
        num_iterations: int = 100,
    ) -> np.ndarray:
        """
        Compute Plackett-Luce model ratings via iterative MM algorithm.

        The Plackett-Luce model assigns a strength parameter τ_i to each player.
        The probability of an ordering (1st, 2nd, ..., kth) is:
            P = Π_{j=1}^{k-1} τ_{σ(j)} / Σ_{l=j}^{k} τ_{σ(l)}

        We use the iterative minorization-maximization (MM) algorithm of Hunter (2004)
        to find the MLE ratings.

        Args:
            pop: Population members.
            orderings: List of orderings (each is list of population indices, winner first).
            num_iterations: MM algorithm iterations.

        Returns:
            Array of ratings (higher = stronger).
        """
        n = len(pop)
        # Initialize ratings uniformly
        ratings = np.ones(n, dtype=np.float64)

        for _ in range(num_iterations):
            new_ratings = np.zeros(n, dtype=np.float64)

            for i in range(n):
                # W_i = number of times player i won a comparison
                wins = 0.0
                denominator = 0.0

                for ordering in orderings:
                    if i not in ordering:
                        continue
                    pos = ordering.index(i)
                    # W_i = number of orderings where i is not ranked last
                    wins += 1.0 if pos < len(ordering) - 1 else 0.0

                    # Denominator: sum of 1 / (sum of ratings of remaining players at pos j).
                    # The PL likelihood has k-1 terms (last choice is forced), so j goes
                    # from 0 to min(pos, k-2). Player i is in remaining set at pos j iff j <= pos.
                    for j in range(len(ordering) - 1):  # j from 0 to k-2
                        if j <= pos:
                            remaining_sum = sum(
                                ratings[ordering[k]] for k in range(j, len(ordering))
                            )
                            if remaining_sum > 1e-10:
                                denominator += 1.0 / remaining_sum

                if denominator > 1e-10:
                    new_ratings[i] = wins / denominator
                else:
                    new_ratings[i] = 0.0

            # Normalize so ratings sum to n
            total = new_ratings.sum()
            if total > 1e-10:
                new_ratings *= n / total

            ratings = new_ratings

        return ratings

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Return population sorted by rating (descending)."""
        rankings = []
        for member in self.population:
            key = member.agent_type if member.is_heuristic else f"ckpt_{member.iteration}"
            rankings.append((key, member.rating))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def save_state(self, path: str) -> None:
        """Save PSRO population state to a JSON file."""
        import json

        state = {
            "max_checkpoints": self.max_checkpoints,
            "checkpoints": [
                {
                    "path": m.path,
                    "iteration": m.iteration,
                    "rating": m.rating,
                    "games": m.games_played,
                }
                for m in self._checkpoints
            ],
            "heuristics": [
                {"type": m.agent_type, "rating": m.rating, "games": m.games_played}
                for m in self._heuristics
            ],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load PSRO population state from a JSON file."""
        import json

        with open(path) as f:
            state = json.load(f)

        self.max_checkpoints = state["max_checkpoints"]
        self._checkpoints = [
            PopulationMember(
                path=c["path"],
                iteration=c["iteration"],
                rating=c.get("rating", 0.0),
                games_played=c.get("games", 0),
            )
            for c in state["checkpoints"]
        ]
        self._heuristics = [
            PopulationMember(
                path="",
                iteration=-1,
                is_heuristic=True,
                agent_type=h["type"],
                rating=h.get("rating", 0.0),
                games_played=h.get("games", 0),
            )
            for h in state["heuristics"]
        ]
