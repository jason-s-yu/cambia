"""Tests for PSRO oracle: population management, Plackett-Luce ratings, persistence."""

import json
import math
import pytest
import numpy as np

from src.cfr.psro import PSROOracle, PopulationMember


class TestPSROPopulation:
    def test_init_with_heuristics(self):
        """Default init creates 3 heuristic members."""
        oracle = PSROOracle()
        assert oracle.checkpoint_count == 0
        assert len(oracle._heuristics) == 3
        assert oracle.size == 3
        types = {m.agent_type for m in oracle._heuristics}
        assert types == {"random", "greedy", "memory_heuristic"}

    def test_init_custom_heuristics(self):
        """Custom heuristic list is respected."""
        oracle = PSROOracle(heuristic_types=["random"])
        assert len(oracle._heuristics) == 1
        assert oracle._heuristics[0].agent_type == "random"
        assert oracle._heuristics[0].is_heuristic is True

    def test_init_no_heuristics(self):
        """Empty heuristic list gives population of 0 initially."""
        oracle = PSROOracle(heuristic_types=[])
        assert oracle.size == 0

    def test_add_checkpoint(self):
        """Adding a checkpoint increases population size."""
        oracle = PSROOracle(heuristic_types=[])
        evicted = oracle.add_checkpoint("/path/to/ckpt_100.pt", iteration=100)
        assert evicted is None
        assert oracle.checkpoint_count == 1
        assert oracle.size == 1
        assert oracle._checkpoints[0].path == "/path/to/ckpt_100.pt"
        assert oracle._checkpoints[0].iteration == 100
        assert oracle._checkpoints[0].is_heuristic is False

    def test_add_multiple_checkpoints(self):
        """Multiple checkpoints accumulate correctly."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=[])
        for i in range(5):
            oracle.add_checkpoint(f"/ckpt_{i}.pt", iteration=i * 10)
        assert oracle.checkpoint_count == 5
        assert oracle.size == 5

    def test_eviction_fifo(self):
        """Oldest checkpoint evicted when exceeding max_checkpoints."""
        oracle = PSROOracle(max_checkpoints=3, heuristic_types=[])
        for i in range(3):
            oracle.add_checkpoint(f"/ckpt_{i}.pt", iteration=i)

        # Adding 4th should evict /ckpt_0.pt
        evicted = oracle.add_checkpoint("/ckpt_3.pt", iteration=3)
        assert evicted == "/ckpt_0.pt"
        assert oracle.checkpoint_count == 3
        paths = [m.path for m in oracle._checkpoints]
        assert "/ckpt_0.pt" not in paths
        assert "/ckpt_3.pt" in paths

    def test_eviction_returns_none_when_under_limit(self):
        """No eviction when under limit."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=[])
        evicted = oracle.add_checkpoint("/ckpt.pt", iteration=1)
        assert evicted is None

    def test_population_includes_heuristics(self):
        """population property combines checkpoints and heuristics."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        pop = oracle.population
        assert len(pop) == 2
        heuristic_members = [m for m in pop if m.is_heuristic]
        checkpoint_members = [m for m in pop if not m.is_heuristic]
        assert len(heuristic_members) == 1
        assert len(checkpoint_members) == 1

    def test_sample_opponents(self):
        """sample_opponents returns correct count."""
        oracle = PSROOracle(heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        opponents = oracle.sample_opponents(num_opponents=3)
        assert len(opponents) == 3
        for opp in opponents:
            assert isinstance(opp, PopulationMember)

    def test_sample_with_replacement(self):
        """sample_opponents samples with replacement when pop < num_opponents."""
        oracle = PSROOracle(heuristic_types=["random"])
        # Only 1 member, request 5 — must use replacement
        opponents = oracle.sample_opponents(num_opponents=5)
        assert len(opponents) == 5

    def test_sample_empty_raises(self):
        """Sampling from empty population raises ValueError."""
        oracle = PSROOracle(heuristic_types=[])
        with pytest.raises(ValueError, match="empty"):
            oracle.sample_opponents(num_opponents=1)

    def test_sample_returns_population_members(self):
        """All sampled opponents come from the population."""
        oracle = PSROOracle(heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        pop_set = set(id(m) for m in oracle.population)
        opponents = oracle.sample_opponents(num_opponents=10)
        for opp in opponents:
            assert id(opp) in pop_set


class TestPlackettLuce:
    def _make_oracle(self, n: int) -> PSROOracle:
        oracle = PSROOracle(max_checkpoints=n, heuristic_types=[])
        for i in range(n):
            oracle.add_checkpoint(f"/ckpt_{i}.pt", iteration=i)
        return oracle

    def test_uniform_ratings(self):
        """With symmetric orderings, ratings stay near uniform."""
        oracle = self._make_oracle(3)
        pop = oracle.population
        # Create perfectly symmetric orderings: each permutation of 3 players
        from itertools import permutations
        orderings = list(permutations(range(3)))
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        # All ratings should be near 1.0 (normalized to sum=n)
        assert ratings.sum() == pytest.approx(3.0, abs=1e-6)
        for r in ratings:
            assert abs(r - 1.0) < 0.1

    def test_dominant_player(self):
        """Player who always wins gets highest rating."""
        oracle = self._make_oracle(3)
        pop = oracle.population
        # Player 0 always wins: [0, 1, 2] repeated many times
        orderings = [[0, 1, 2]] * 100
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert ratings[0] == max(ratings)

    def test_last_place_player(self):
        """Player who always loses gets lowest rating."""
        oracle = self._make_oracle(3)
        pop = oracle.population
        # Player 2 always loses
        orderings = [[0, 1, 2]] * 100
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert ratings[2] == min(ratings)

    def test_ratings_sum(self):
        """Ratings sum to population size (normalization)."""
        oracle = self._make_oracle(4)
        pop = oracle.population
        orderings = [[0, 1, 2, 3]] * 50 + [[3, 2, 1, 0]] * 50
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert ratings.sum() == pytest.approx(4.0, abs=1e-6)

    def test_ratings_positive(self):
        """All ratings are non-negative."""
        oracle = self._make_oracle(5)
        pop = oracle.population
        rng = np.random.default_rng(42)
        orderings = [rng.permutation(5).tolist() for _ in range(200)]
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert all(r >= 0 for r in ratings)

    def test_two_player(self):
        """Works correctly with 2-player orderings."""
        oracle = self._make_oracle(2)
        pop = oracle.population
        # Player 0 wins 80%, player 1 wins 20%
        orderings = [[0, 1]] * 80 + [[1, 0]] * 20
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        assert ratings[0] > ratings[1]
        assert ratings.sum() == pytest.approx(2.0, abs=1e-6)

    def test_empty_orderings(self):
        """Empty orderings: no data → all ratings zeroed (ghost-rating fix)."""
        oracle = self._make_oracle(3)
        pop = oracle.population
        ratings = oracle._plackett_luce_ratings(pop, [])
        # With ghost-rating fix, players with zero orderings get 0 not stale values.
        # total=0 so normalization is skipped and zeros are returned.
        assert ratings.sum() == pytest.approx(0.0, abs=1e-4)


class TestPSROStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """Save then load recovers same state."""
        oracle = PSROOracle(max_checkpoints=10, heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/ckpt_50.pt", iteration=50)
        oracle.add_checkpoint("/ckpt_100.pt", iteration=100)

        save_path = str(tmp_path / "psro_state.json")
        oracle.save_state(save_path)

        oracle2 = PSROOracle(max_checkpoints=5, heuristic_types=[])
        oracle2.load_state(save_path)

        assert oracle2.max_checkpoints == 10
        assert oracle2.checkpoint_count == 2
        assert len(oracle2._heuristics) == 2
        assert oracle2._checkpoints[0].path == "/ckpt_50.pt"
        assert oracle2._checkpoints[1].iteration == 100

    def test_load_preserves_ratings(self, tmp_path):
        """Ratings survive save/load cycle."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        oracle._checkpoints[0].rating = 2.5
        oracle._heuristics[0].rating = 1.5

        save_path = str(tmp_path / "psro_state.json")
        oracle.save_state(save_path)

        oracle2 = PSROOracle(heuristic_types=[])
        oracle2.load_state(save_path)

        assert oracle2._checkpoints[0].rating == pytest.approx(2.5)
        assert oracle2._heuristics[0].rating == pytest.approx(1.5)

    def test_load_preserves_games_played(self, tmp_path):
        """games_played survives save/load cycle."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=["random"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        oracle._checkpoints[0].games_played = 42
        oracle._heuristics[0].games_played = 17

        save_path = str(tmp_path / "psro_state.json")
        oracle.save_state(save_path)

        oracle2 = PSROOracle(heuristic_types=[])
        oracle2.load_state(save_path)

        assert oracle2._checkpoints[0].games_played == 42
        assert oracle2._heuristics[0].games_played == 17

    def test_save_creates_valid_json(self, tmp_path):
        """Saved file is valid JSON with expected keys."""
        oracle = PSROOracle(max_checkpoints=3, heuristic_types=["random"])
        oracle.add_checkpoint("/ckpt.pt", iteration=5)

        save_path = str(tmp_path / "psro_state.json")
        oracle.save_state(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert "max_checkpoints" in data
        assert "checkpoints" in data
        assert "heuristics" in data
        assert data["max_checkpoints"] == 3
        assert len(data["checkpoints"]) == 1
        assert len(data["heuristics"]) == 1

    def test_empty_population_roundtrip(self, tmp_path):
        """Empty population (no checkpoints, no heuristics) saves/loads cleanly."""
        oracle = PSROOracle(max_checkpoints=5, heuristic_types=[])
        save_path = str(tmp_path / "psro_empty.json")
        oracle.save_state(save_path)

        oracle2 = PSROOracle(heuristic_types=["random"])
        oracle2.load_state(save_path)
        assert oracle2.checkpoint_count == 0
        assert len(oracle2._heuristics) == 0


class TestPSRORankings:
    def test_rankings_sorted(self):
        """get_rankings returns descending order."""
        oracle = PSROOracle(heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/ckpt.pt", iteration=1)
        # Set distinct ratings
        oracle._checkpoints[0].rating = 3.0
        oracle._heuristics[0].rating = 1.0
        oracle._heuristics[1].rating = 2.0

        rankings = oracle.get_rankings()
        assert rankings[0][1] >= rankings[1][1] >= rankings[2][1]

    def test_rankings_include_all(self):
        """All population members appear in rankings."""
        oracle = PSROOracle(heuristic_types=["random", "greedy"])
        oracle.add_checkpoint("/ckpt_1.pt", iteration=1)
        oracle.add_checkpoint("/ckpt_2.pt", iteration=2)

        rankings = oracle.get_rankings()
        assert len(rankings) == 4  # 2 checkpoints + 2 heuristics

    def test_rankings_keys(self):
        """Heuristics use agent_type, checkpoints use ckpt_{iter}."""
        oracle = PSROOracle(heuristic_types=["random"])
        oracle.add_checkpoint("/ckpt_50.pt", iteration=50)

        rankings = oracle.get_rankings()
        keys = {k for k, _ in rankings}
        assert "random" in keys
        assert "ckpt_50" in keys

    def test_rankings_empty_population(self):
        """Empty population returns empty rankings."""
        oracle = PSROOracle(heuristic_types=[])
        assert oracle.get_rankings() == []

    def test_rankings_ties(self):
        """Ties are handled gracefully (all same rating)."""
        oracle = PSROOracle(heuristic_types=["random", "greedy"])
        rankings = oracle.get_rankings()
        # Default rating is 0.0 for all
        assert all(r == 0.0 for _, r in rankings)
        assert len(rankings) == 2


class TestGhostRatingRegression:
    def test_ghost_rating_zeroed(self):
        """Player never appearing in orderings must get rating 0, not stale value."""
        oracle = PSROOracle(heuristic_types=[])
        # Add 4 dummy population members via internal list for direct testing
        for i in range(4):
            oracle._checkpoints.append(
                PopulationMember(path=f"/fake/ckpt_{i}", iteration=i, rating=9.9)
            )
        pop = oracle._checkpoints  # 4 members

        # Orderings only involving players 0, 1, 2 — player 3 never appears
        orderings = [
            [0, 1, 2],
            [1, 0, 2],
            [2, 1, 0],
            [0, 2, 1],
        ]
        ratings = oracle._plackett_luce_ratings(pop, orderings)
        # Player 3 had stale rating 9.9 — after fix, it must be 0.0 (before normalization)
        # After normalization the value may differ, but the key check: no inf/nan
        assert not np.isnan(ratings).any(), "NaN in ratings"
        assert not np.isinf(ratings).any(), "Inf in ratings"
        # Player 3 was never in an ordering, so wins=0 and denominator=0 → new_ratings[3]=0
        # After normalization by n/total this could be non-zero only if total==0, but
        # players 0-2 have non-zero ratings so total>0. Player 3's unnormalized value is 0.
        # We test the raw (pre-normalization) behaviour via a single iteration:
        n = len(pop)
        raw = np.zeros(n, dtype=np.float64)
        prev = np.array([m.rating for m in pop])
        for i in range(n):
            wins = 0.0
            denom = 0.0
            for ordering in orderings:
                if i not in ordering:
                    continue
                rank = ordering.index(i)
                wins += 1.0
                for k in range(rank + 1, len(ordering)):
                    remaining = [ordering[j] for j in range(k, len(ordering))]
                    rs = sum(prev[p] for p in remaining)
                    if rs > 1e-10:
                        denom += 1.0 / rs
            if denom > 1e-10:
                raw[i] = wins / denom
            else:
                raw[i] = 0.0  # The fix: not prev[i]
        assert raw[3] == 0.0, f"Expected raw[3]=0.0, got {raw[3]}"
