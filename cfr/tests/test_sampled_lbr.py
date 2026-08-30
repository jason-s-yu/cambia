"""Tests for sampled LBR exploitability measurement (Tier A) on the Go engine.

The estimator was ported off the Python reference engine in cambia-1427: it now
searches a ``GoSearchState`` and branches by rewinding the token-inclusive
state_save/state_restore checkpoint. These tests therefore drive it with a
Go-native policy (``choose_action(view, legal_actions)`` over a ``GameView``)
rather than a Python-engine ``RandomAgent``.
"""

import random
from dataclasses import dataclass, field

from src.cfr.sampled_lbr import sampled_lbr

# ---------------------------------------------------------------------------
# Minimal config stubs
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
class _Config:
    cambia_rules: _RulesConfig = field(default_factory=_RulesConfig)


# ---------------------------------------------------------------------------
# Helper: a Go-native uniform policy (the target under test)
# ---------------------------------------------------------------------------


class _UniformWrapper:
    """Uniform-random target with its own RNG, so a run is seed-deterministic.

    Written against the cambia-1427 policy boundary: it receives a ``GameView``
    and a list of ``GameAction`` NamedTuples in the engine's ascending index
    order, and never touches the game object.
    """

    accepts_game_view = True

    def __init__(self, config, seed: int = 0):
        self._config = config
        self.player_id = 0
        self._rng = random.Random(seed)

    def choose_action(self, view, legal_actions):
        actions = list(legal_actions)
        return actions[self._rng.randrange(len(actions))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lbr_valid_result():
    """sampled_lbr returns a dict with the expected keys and correct types."""
    config = _Config()
    agent = _UniformWrapper(config)
    result = sampled_lbr(
        agent, config, num_infosets=20, br_rollouts_per_infoset=5, seed=42
    )

    assert isinstance(result, dict), "Result must be a dict"
    assert "exploitability" in result
    assert "num_infosets_sampled" in result
    assert "std_err" in result
    assert isinstance(result["exploitability"], float)
    assert isinstance(result["num_infosets_sampled"], int)
    assert isinstance(result["std_err"], float)
    assert result["num_infosets_sampled"] >= 0


def test_lbr_reports_tier_and_seed():
    """The result carries its tier and the seed that produced it, so a persisted
    row is reproducible (cambia-1427 AC3)."""
    config = _Config()
    agent = _UniformWrapper(config)
    result = sampled_lbr(
        agent, config, num_infosets=10, br_rollouts_per_infoset=2, seed=1234
    )
    assert result["tier"] == "A"
    assert result["seed"] == 1234


def test_lbr_non_negative_exploitability():
    """Exploitability is always >= 0 (BR value >= agent value by construction)."""
    config = _Config()
    agent = _UniformWrapper(config)
    result = sampled_lbr(
        agent, config, num_infosets=50, br_rollouts_per_infoset=5, seed=7
    )

    assert (
        result["exploitability"] >= 0.0
    ), f"Exploitability must be >= 0, got {result['exploitability']}"
    assert result["std_err"] >= 0.0


def test_lbr_deterministic_seed():
    """Same seed produces identical results across two calls."""
    config = _Config()
    agent_a = _UniformWrapper(config)
    agent_b = _UniformWrapper(config)

    result_a = sampled_lbr(
        agent_a, config, num_infosets=30, br_rollouts_per_infoset=5, seed=99
    )
    result_b = sampled_lbr(
        agent_b, config, num_infosets=30, br_rollouts_per_infoset=5, seed=99
    )

    assert (
        result_a["exploitability"] == result_b["exploitability"]
    ), f"Same seed should give same exploitability: {result_a} vs {result_b}"
    assert result_a["num_infosets_sampled"] == result_b["num_infosets_sampled"]


def test_lbr_seed_changes_the_estimate():
    """Different seeds must actually re-roll the measurement, or --lbr-seed would
    be decorative."""
    config = _Config()
    results = {
        seed: sampled_lbr(
            _UniformWrapper(config),
            config,
            num_infosets=30,
            br_rollouts_per_infoset=4,
            seed=seed,
        )["exploitability"]
        for seed in (1, 2, 3)
    }
    assert len(set(results.values())) > 1, f"seed had no effect: {results}"
