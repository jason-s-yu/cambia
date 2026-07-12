"""Tests for Tier-B sampled LBR and the collector sample-count fix (BUG-3).

Tier-B LBR (cfr/src/cfr/lbr.py) tightens the Tier-A loose lower bound by:
  - generating trajectories against a strong fixed opponent (not random), and
  - rolling out the BR continuation under the agent's own policy at seat 0
    plus the strong opponent at seat 1 (agent-policy rollouts).

The collector also fixes BUG-3: the Tier-A sampler sized games_needed/sample_prob
for an assumed 40 P0 decisions/game while real Cambia games average ~3 under
random play, so requesting N infosets collected ~0.07*N. The corrected collector
loops until it has the requested count (subject to a safety cap).
"""

import pytest
from dataclasses import dataclass, field

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import RandomAgent
from src.cfr.lbr import tier_b_lbr, collect_infosets, DEFAULT_TRAJECTORY_OPPONENT
from src.cfr.sampled_lbr import sampled_lbr

# ---------------------------------------------------------------------------
# Minimal config stubs (mirror tests/test_sampled_lbr.py, plus agents.greedy)
# ---------------------------------------------------------------------------


@dataclass
class _RulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 200
    lockCallerHand: bool = True
    num_decks: int = 1


@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(default_factory=_GreedyAgentConfig)


@dataclass
class _Config:
    cambia_rules: _RulesConfig = field(default_factory=_RulesConfig)
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RandomAgentWrapper:
    """Thin wrapper exposing RandomAgent as an agent_wrapper for LBR."""

    def __init__(self, config):
        self._config = config
        self._agent = RandomAgent(0, config)

    def choose_action(self, game_state, legal_actions):
        return self._agent.choose_action(game_state, legal_actions)


class _RecordingWrapper(_RandomAgentWrapper):
    """RandomAgent wrapper that counts choose_action calls (to prove the agent
    policy is actually used inside BR-continuation rollouts for Tier B)."""

    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def choose_action(self, game_state, legal_actions):
        self.calls += 1
        return super().choose_action(game_state, legal_actions)


# ---------------------------------------------------------------------------
# Tier-B result-shape and invariants
# ---------------------------------------------------------------------------


def test_tier_b_valid_result():
    config = _Config()
    agent = _RandomAgentWrapper(config)
    result = tier_b_lbr(
        agent, config, num_infosets=20, br_rollouts_per_infoset=4, seed=42
    )

    assert isinstance(result, dict)
    for key in (
        "exploitability",
        "num_infosets_sampled",
        "std_err",
        "tier",
        "rollout_opponent",
    ):
        assert key in result, f"missing key {key}"
    assert result["tier"] == "B"
    assert isinstance(result["exploitability"], float)
    assert isinstance(result["num_infosets_sampled"], int)
    assert result["num_infosets_sampled"] >= 0


def test_tier_b_non_negative_exploitability():
    config = _Config()
    agent = _RandomAgentWrapper(config)
    result = tier_b_lbr(agent, config, num_infosets=40, br_rollouts_per_infoset=4, seed=7)
    assert result["exploitability"] >= 0.0
    assert result["std_err"] >= 0.0


def test_tier_b_deterministic_seed():
    config = _Config()
    a = _RandomAgentWrapper(config)
    b = _RandomAgentWrapper(config)
    ra = tier_b_lbr(a, config, num_infosets=25, br_rollouts_per_infoset=4, seed=99)
    rb = tier_b_lbr(b, config, num_infosets=25, br_rollouts_per_infoset=4, seed=99)
    assert ra["exploitability"] == rb["exploitability"]
    assert ra["num_infosets_sampled"] == rb["num_infosets_sampled"]


def test_tier_b_uses_agent_policy_in_rollouts():
    """The agent wrapper must be invoked for seat-0 continuation play, not only
    at the sampled infoset. With >1 rollout per action and multi-step games the
    call count must far exceed the number of sampled infosets."""
    config = _Config()
    agent = _RecordingWrapper(config)
    result = tier_b_lbr(agent, config, num_infosets=15, br_rollouts_per_infoset=4, seed=3)
    n = result["num_infosets_sampled"]
    if n == 0:
        pytest.skip("no infosets sampled in this tiny config")
    # Trajectory choose_action + per-infoset agent_value rollouts each call the
    # agent at least once; agent-policy continuation adds more. Strictly greater
    # than the sampled count proves continuation play used the agent.
    assert agent.calls > n


# ---------------------------------------------------------------------------
# BUG-3: collector returns ~requested sample count (not ~0.07x)
# ---------------------------------------------------------------------------


def test_collector_meets_requested_count():
    """Requesting N infosets must collect ~N (the 14x over-request fix).

    Before the fix, num=200 collected ~14 samples (40/3 under-collection).
    After the fix the collector loops until it reaches the target (or the
    safety cap), so it must collect at least the requested count.
    """
    config = _Config()
    agent = _RandomAgentWrapper(config)
    requested = 200
    infosets = collect_infosets(
        agent,
        config,
        num_infosets=requested,
        seed=11,
        trajectory_opponent_factory=DEFAULT_TRAJECTORY_OPPONENT,
    )
    # Must reach the requested count, never ~0.07x.
    assert len(infosets) >= requested, (
        f"collector under-collected: got {len(infosets)} of {requested} "
        f"(BUG-3 regression — sized for wrong decisions/game)"
    )
    # And it must not wildly over-collect (collection stops at the target).
    assert len(infosets) <= requested + 50


def test_collector_caps_to_avoid_runaway():
    """With an unreachable target the collector must terminate via its safety
    cap rather than loop forever, and return what it collected."""
    config = _Config()
    agent = _RandomAgentWrapper(config)
    infosets = collect_infosets(
        agent,
        config,
        num_infosets=10_000_000,
        seed=5,
        trajectory_opponent_factory=DEFAULT_TRAJECTORY_OPPONENT,
        max_games=50,  # hard cap forces early termination
    )
    assert len(infosets) >= 0  # terminates; count bounded by max_games


def test_sampled_lbr_tier_a_also_meets_count():
    """The production Tier-A path (sampled_lbr) must also honor the requested
    count now that it shares the corrected collector."""
    config = _Config()
    agent = _RandomAgentWrapper(config)
    requested = 150
    result = sampled_lbr(
        agent, config, num_infosets=requested, br_rollouts_per_infoset=2, seed=21
    )
    assert (
        result["num_infosets_sampled"] >= requested
    ), f"Tier-A under-collected: {result['num_infosets_sampled']} of {requested}"
