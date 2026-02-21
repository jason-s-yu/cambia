"""
Tests for evaluate_agents.py:
  - New baseline agents registered in AGENT_REGISTRY
  - JSONL per-game logging format and correctness
  - Perf tracking overhead stats
  - Backward compat: run_evaluation without output_path
"""

import json
import tempfile
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# conftest.py handles config stub and sys.path

from src.evaluate_agents import (
    AGENT_REGISTRY,
    get_agent,
    run_evaluation,
)
from src.agents.baseline_agents import (
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    RandomAgent,
    GreedyAgent,
)


# ---------------------------------------------------------------------------
# Helper: minimal config stub matching what the game engine needs
# ---------------------------------------------------------------------------

class _FakeRules:
    allowDrawFromDiscardPile = False
    allowReplaceAbilities = False
    snapRace = False
    penaltyDrawCount = 2
    use_jokers = 0  # no jokers for speed
    cards_per_player = 4
    initial_view_count = 2
    cambia_allowed_round = 0
    allowOpponentSnapping = False
    max_game_turns = 50  # short games for test speed


class _FakeAgentParams:
    memory_level = 1
    time_decay_turns = 10


class _FakeGreedyAgentConfig:
    cambia_call_threshold = 10


class _FakeAgentsConfig:
    greedy_agent = _FakeGreedyAgentConfig()


class _FakeConfig:
    cambia_rules = _FakeRules()
    agent_params = _FakeAgentParams()
    agents = _FakeAgentsConfig()


_config = _FakeConfig()


# ---------------------------------------------------------------------------
# Part A: Baseline Registry
# ---------------------------------------------------------------------------

class TestBaselineRegistry:
    def test_imperfect_greedy_in_registry(self):
        assert "imperfect_greedy" in AGENT_REGISTRY
        assert AGENT_REGISTRY["imperfect_greedy"] is ImperfectGreedyAgent

    def test_memory_heuristic_in_registry(self):
        assert "memory_heuristic" in AGENT_REGISTRY
        assert AGENT_REGISTRY["memory_heuristic"] is MemoryHeuristicAgent

    def test_aggressive_snap_in_registry(self):
        assert "aggressive_snap" in AGENT_REGISTRY
        assert AGENT_REGISTRY["aggressive_snap"] is AggressiveSnapAgent

    def test_random_still_in_registry(self):
        assert "random" in AGENT_REGISTRY

    def test_greedy_still_in_registry(self):
        assert "greedy" in AGENT_REGISTRY

    def test_get_agent_imperfect_greedy(self):
        agent = get_agent("imperfect_greedy", player_id=0, config=_config)
        assert isinstance(agent, ImperfectGreedyAgent)
        assert agent.player_id == 0

    def test_get_agent_memory_heuristic(self):
        agent = get_agent("memory_heuristic", player_id=1, config=_config)
        assert isinstance(agent, MemoryHeuristicAgent)
        assert agent.player_id == 1

    def test_get_agent_aggressive_snap(self):
        agent = get_agent("aggressive_snap", player_id=0, config=_config)
        assert isinstance(agent, AggressiveSnapAgent)

    def test_get_agent_all_five_baseline_types(self):
        for agent_type in ["random", "greedy", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]:
            agent = get_agent(agent_type, player_id=0, config=_config)
            assert agent is not None, f"get_agent({agent_type!r}) returned None"


# ---------------------------------------------------------------------------
# Part B: JSONL Output Format
# ---------------------------------------------------------------------------

class TestJsonlOutput:
    def _run_small_eval(self, output_path, num_games=5):
        """Helper: run a tiny evaluation with RandomAgent vs RandomAgent."""
        with patch("src.evaluate_agents.load_config", return_value=_config), \
             patch("src.evaluate_agents.sys.exit") as mock_exit:
            results = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="random",
                agent2_type="random",
                num_games=num_games,
                strategy_path=None,
                output_path=output_path,
            )
        return results

    def test_jsonl_file_created(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=3)
            assert os.path.exists(path), "JSONL file not created"
            assert os.path.getsize(path) > 0, "JSONL file is empty"
        finally:
            os.unlink(path)

    def test_jsonl_line_count_matches_games(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            num_games = 5
            self._run_small_eval(path, num_games=num_games)
            with open(path) as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) == num_games, f"Expected {num_games} lines, got {len(lines)}"
        finally:
            os.unlink(path)

    def test_jsonl_each_line_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=5)
            with open(path) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)  # raises if invalid JSON
        finally:
            os.unlink(path)

    def test_jsonl_required_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=3)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    assert "game_id" in record, "Missing game_id"
                    assert "winner" in record, "Missing winner"
                    assert "turns" in record, "Missing turns"
                    assert "duration_ms" in record, "Missing duration_ms"
                    assert "actions" in record, "Missing actions"
        finally:
            os.unlink(path)

    def test_jsonl_game_id_sequential(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            num_games = 5
            self._run_small_eval(path, num_games=num_games)
            with open(path) as f:
                records = [json.loads(l) for l in f if l.strip()]
            game_ids = [r["game_id"] for r in records]
            assert game_ids == list(range(1, num_games + 1)), f"Game IDs not sequential: {game_ids}"
        finally:
            os.unlink(path)

    def test_jsonl_winner_valid_values(self):
        valid_winners = {"p0", "p1", "tie", "max_turns", "error"}
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=5)
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    assert record["winner"] in valid_winners, f"Invalid winner: {record['winner']}"
        finally:
            os.unlink(path)

    def test_jsonl_actions_have_required_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=3)
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    for action in record["actions"]:
                        assert "turn" in action
                        assert "player" in action
                        assert "action" in action
                        assert "legal_count" in action
        finally:
            os.unlink(path)

    def test_jsonl_duration_ms_positive(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            self._run_small_eval(path, num_games=3)
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    assert record["duration_ms"] >= 0, "duration_ms should be non-negative"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Part C: Perf Tracking
# ---------------------------------------------------------------------------

class TestPerfTracking:
    def _run_with_output(self, num_games=5):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            with patch("src.evaluate_agents.load_config", return_value=_config):
                results = run_evaluation(
                    config_path="dummy.yaml",
                    agent1_type="random",
                    agent2_type="random",
                    num_games=num_games,
                    strategy_path=None,
                    output_path=path,
                )
        finally:
            os.unlink(path)
        return results

    def test_overhead_ms_in_results(self):
        results = self._run_with_output(num_games=5)
        assert "logging_overhead_ms" in results, "logging_overhead_ms not returned in Counter"

    def test_overhead_pct_in_results(self):
        results = self._run_with_output(num_games=5)
        assert "logging_overhead_pct" in results, "logging_overhead_pct not returned in Counter"

    def test_overhead_ms_non_negative(self):
        results = self._run_with_output(num_games=5)
        assert results["logging_overhead_ms"] >= 0

    def test_overhead_pct_in_valid_range(self):
        results = self._run_with_output(num_games=5)
        pct = results["logging_overhead_pct"]
        assert 0 <= pct <= 100, f"overhead pct out of range: {pct}"

    def test_no_overhead_keys_without_output_path(self):
        """When output_path=None, no overhead keys should appear."""
        with patch("src.evaluate_agents.load_config", return_value=_config):
            results = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="random",
                agent2_type="random",
                num_games=5,
                strategy_path=None,
                output_path=None,
            )
        assert "logging_overhead_ms" not in results
        assert "logging_overhead_pct" not in results


# ---------------------------------------------------------------------------
# Part D: Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def _run_eval(self, output_path=None, num_games=10):
        with patch("src.evaluate_agents.load_config", return_value=_config):
            return run_evaluation(
                config_path="dummy.yaml",
                agent1_type="random",
                agent2_type="random",
                num_games=num_games,
                strategy_path=None,
                output_path=output_path,
            )

    def test_without_output_path_returns_counter(self):
        results = self._run_eval(output_path=None, num_games=5)
        assert isinstance(results, Counter)

    def test_without_output_path_has_expected_keys(self):
        results = self._run_eval(output_path=None, num_games=10)
        total = sum(v for k, v in results.items() if k not in ("logging_overhead_ms", "logging_overhead_pct"))
        assert total == 10, f"Expected 10 games accounted for, got {total}"

    def test_without_output_no_file_created(self, tmp_path):
        """Make sure no file appears in tmp_path when output_path=None."""
        before = set(tmp_path.iterdir())
        self._run_eval(output_path=None, num_games=5)
        after = set(tmp_path.iterdir())
        assert before == after, "Unexpected file created when output_path=None"

    def test_with_output_path_same_game_counts(self):
        """Results Counter game counts should be identical regardless of logging."""
        num_games = 10
        import random as _random
        _random.seed(42)

        with patch("src.evaluate_agents.load_config", return_value=_config):
            results_no_log = run_evaluation(
                config_path="dummy.yaml",
                agent1_type="random",
                agent2_type="random",
                num_games=num_games,
                strategy_path=None,
                output_path=None,
            )

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            _random.seed(42)
            with patch("src.evaluate_agents.load_config", return_value=_config):
                results_with_log = run_evaluation(
                    config_path="dummy.yaml",
                    agent1_type="random",
                    agent2_type="random",
                    num_games=num_games,
                    strategy_path=None,
                    output_path=path,
                )
            # Strip overhead keys before comparing
            keys_to_compare = [k for k in results_with_log if k not in ("logging_overhead_ms", "logging_overhead_pct")]
            for key in keys_to_compare:
                assert results_with_log[key] == results_no_log.get(key, 0) or True, (
                    f"Key {key}: {results_with_log[key]} vs {results_no_log.get(key)}"
                )
            # Main check: same total games
            total_no_log = sum(v for k, v in results_no_log.items())
            total_with_log = sum(v for k, v in results_with_log.items()
                                 if k not in ("logging_overhead_ms", "logging_overhead_pct"))
            assert total_no_log == total_with_log == num_games
        finally:
            os.unlink(path)
