"""Deal seeding for the evaluation loops and the estimators (cambia-1974).

The defect these pin, measured in cambia-1807: constructing a PPOAgentWrapper
loads a Stable-Baselines3 model, which reseeds Python's global ``random``
module to a fixed state. ``run_head_to_head_typed`` built both agents inside
its per-game loop and then constructed its game with a ``None`` deck seed,
which ``GoEngine`` filled from ``random.getrandbits(64)``. That draw therefore
returned the same value on every iteration and the whole match was one deal,
played over and over: PPO won 150 of 150 against uniform random on a single
byte-identical action sequence.

Covered here:

1. A match whose agents reseed the global module on construction still deals a
   different hand every game, and the agents are built once for the match
   rather than once per game.
2. The seat-swap pair shares its deal, and a match run with the two sides
   exchanged meets the identical deals, so deck luck cancels.
3. The estimators' uniform opponents are unmoved by anything that reseeds the
   global module part-way through a measurement.
4. The run seed rides on the result, so a match given no seed can still be
   replayed from the number it reports.
"""

import random
from dataclasses import dataclass, field
from typing import List

import pytest

from src.cfr.lbr import tier_b_lbr
from src.evaluate_agents import (
    AGENT_REGISTRY,
    _h2h_deck_seed,
    run_head_to_head_typed,
)
from src.utils import resolve_run_seed

# ---------------------------------------------------------------------------
# Config stub (mirrors tests/test_lbr_tier_b.py)
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
# A stub that reseeds the global module on construction, as an SB3 load does
# ---------------------------------------------------------------------------


class _ReseedingAgent:
    """Uniform play, plus the side effect that caused the defect.

    ``MaskablePPO.load`` reseeds the global ``random`` module; this reproduces
    that in one line, without sb3 (an optional extra) or a checkpoint. Its own
    choices come from a private stream, so the reseed changes nothing about how
    this agent plays: anything that moves because of it moved for some other
    reason.
    """

    accepts_game_view = True
    constructions = 0

    def __init__(self, player_id, config):
        type(self).constructions += 1
        random.seed(20260902)
        self.player_id = player_id
        self._rng = random.Random(11 + player_id)

    def choose_action(self, game_state, legal_actions):
        actions = sorted(legal_actions, key=repr)
        return actions[self._rng.randrange(len(actions))]


@pytest.fixture
def reseeding_agent_type(monkeypatch):
    """Register the stub so get_agent builds it, and reset its counter."""
    _ReseedingAgent.constructions = 0
    monkeypatch.setitem(AGENT_REGISTRY, "reseeding_stub", _ReseedingAgent)
    return "reseeding_stub"


@pytest.fixture
def recorded_deals(monkeypatch):
    """Every deck seed and seat-0 opening hand a match was dealt."""
    import src.evaluate_agents as ea

    real = ea._GoEvalGame
    seeds: List = []
    hands: List = []

    class _RecordingGame(real):
        def __init__(self, house_rules, seed, num_players, agents):
            super().__init__(house_rules, seed, num_players, agents)
            seeds.append(seed)
            hands.append(tuple(repr(c) for c in self.engine.get_player_hand(0)))

    monkeypatch.setattr(ea, "_GoEvalGame", _RecordingGame)
    return seeds, hands


# ---------------------------------------------------------------------------
# The loops
# ---------------------------------------------------------------------------


def test_a_match_with_a_reseeding_agent_deals_a_different_hand_each_game(
    reseeding_agent_type, recorded_deals
):
    """The regression itself: two games, two deals.

    Before the fix both games were dealt from ``random.getrandbits(64)`` taken
    right after the stub reseeded the module, so both seeds, and both hands,
    were identical.
    """
    seeds, hands = recorded_deals
    run_head_to_head_typed(
        agent_a_type=reseeding_agent_type,
        checkpoint_a="",
        agent_b_type=reseeding_agent_type,
        checkpoint_b="",
        num_games=2,
        config=_Config(),
        seed=4242,
    )
    assert len(seeds) == 2
    assert all(s is not None for s in seeds), "a game was dealt without a seed"
    # Games 1 and 2 are the two halves of one seat rotation, so they SHARE a
    # deal by design; the point is that neither was drawn from the global
    # module. A third game opens the next rotation and must move.
    assert seeds[0] == seeds[1], "the seat-swap pair must share its deal"
    assert hands[0] == hands[1]

    seeds.clear()
    hands.clear()
    run_head_to_head_typed(
        agent_a_type=reseeding_agent_type,
        checkpoint_a="",
        agent_b_type=reseeding_agent_type,
        checkpoint_b="",
        num_games=4,
        config=_Config(),
        seed=4242,
    )
    assert (
        len({tuple(seeds[:2]), tuple(seeds[2:])}) == 2
    ), f"both seat rotations were dealt the same cards: {seeds}"
    assert hands[0] != hands[2], "consecutive rotations dealt an identical hand"


def test_agents_are_built_once_for_the_match(reseeding_agent_type):
    """Four agents for the match (each side at each seat), not four per game.

    A model load between two games is what put the reseed inside the loop in
    the first place; building outside it is the half of the fix that keeps any
    future library side effect from landing mid-match.
    """
    run_head_to_head_typed(
        agent_a_type=reseeding_agent_type,
        checkpoint_a="",
        agent_b_type=reseeding_agent_type,
        checkpoint_b="",
        num_games=6,
        config=_Config(),
        seed=7,
    )
    assert (
        _ReseedingAgent.constructions == 4
    ), f"built {_ReseedingAgent.constructions} agents for a 6-game match"


def test_swapping_the_sides_meets_the_same_deals(reseeding_agent_type, recorded_deals):
    """Common random numbers across the seat swap: game i of A-versus-B and
    game i of B-versus-A are dealt the same cards, so a win rate carries the
    difference between the sides and not the difference between their decks."""
    seeds, hands = recorded_deals
    common = dict(
        checkpoint_a="",
        checkpoint_b="",
        num_games=4,
        config=_Config(),
        seed=99,
    )
    run_head_to_head_typed(
        agent_a_type=reseeding_agent_type,
        agent_b_type="imperfect_greedy",
        **common,
    )
    forward_seeds, forward_hands = list(seeds), list(hands)

    seeds.clear()
    hands.clear()
    run_head_to_head_typed(
        agent_a_type="imperfect_greedy",
        agent_b_type=reseeding_agent_type,
        **common,
    )
    assert seeds == forward_seeds, "the swapped match was dealt different cards"
    assert hands == forward_hands


def test_the_deck_seed_is_symmetric_in_the_two_sides():
    """The property the swap test rests on, pinned directly."""
    assert _h2h_deck_seed(5, "a", "b", 3) == _h2h_deck_seed(5, "b", "a", 3)
    assert _h2h_deck_seed(5, "a", "b", 3) != _h2h_deck_seed(5, "a", "b", 4)
    assert _h2h_deck_seed(5, "a", "b", 3) != _h2h_deck_seed(6, "a", "b", 3)


def test_the_result_reports_the_seed_it_dealt_from(reseeding_agent_type):
    """A match given no seed still says which one it used, so it can be replayed."""
    result = run_head_to_head_typed(
        agent_a_type=reseeding_agent_type,
        checkpoint_a="",
        agent_b_type=reseeding_agent_type,
        checkpoint_b="",
        num_games=2,
        config=_Config(),
    )
    assert isinstance(result["run_seed"], int)
    assert result["run_seed"] > 0


def test_an_unset_run_seed_is_not_drawn_from_the_global_module():
    """Two resolutions taken after the same global reseed must still differ.

    This is the defect one level up: a run seed drawn with ``random`` would be
    identical for every match that followed a model load, and the fix would
    have moved the single deal from the game to the match.
    """
    random.seed(1234)
    first = resolve_run_seed(None)
    random.seed(1234)
    second = resolve_run_seed(None)
    assert first != second
    # An explicit seed is still honoured exactly.
    assert resolve_run_seed(77) == 77


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


class _UniformWrapper:
    """Uniform play off a private stream."""

    accepts_game_view = True

    def __init__(self, config, seed: int = 0):
        self.player_id = 0
        self._rng = random.Random(seed)

    def choose_action(self, view, legal_actions):
        actions = list(legal_actions)
        return actions[self._rng.randrange(len(actions))]


class _ReseedingWrapper(_UniformWrapper):
    """Uniform play that reseeds the global module at every decision.

    Stands in for a measurement that loads a model part-way through. Its own
    choices are unaffected, so a moved result can only have come from something
    that was reading the global module.
    """

    def choose_action(self, view, legal_actions):
        random.seed(20260902)
        return super().choose_action(view, legal_actions)


def test_a_mid_run_global_reseed_does_not_move_a_tier_b_estimate():
    """The estimators' uniform opponents draw from a stream this module owns.

    Before cambia-1974 they were seeded from ``random.getrandbits``, so every
    opponent built after a model load started from the same state and the
    estimate moved with something that has nothing to do with the game.
    """
    config = _Config()
    quiet = tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=8, br_rollouts_per_infoset=2, seed=5
    )
    noisy = tier_b_lbr(
        _ReseedingWrapper(config),
        config,
        num_infosets=8,
        br_rollouts_per_infoset=2,
        seed=5,
    )
    assert quiet["num_infosets_sampled"] == noisy["num_infosets_sampled"]
    assert quiet["exploitability"] == noisy["exploitability"]


def test_the_estimator_reproduces_from_its_seed_across_a_reseed():
    """Same seed, same number, whatever the global module did in between."""
    config = _Config()
    first = tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=8, br_rollouts_per_infoset=2, seed=3
    )
    random.seed(999)
    second = tier_b_lbr(
        _UniformWrapper(config), config, num_infosets=8, br_rollouts_per_infoset=2, seed=3
    )
    assert first["exploitability"] == second["exploitability"]
