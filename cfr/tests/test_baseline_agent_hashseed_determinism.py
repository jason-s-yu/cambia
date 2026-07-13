"""
tests/test_baseline_agent_hashseed_determinism.py

Cross-process determinism regression tests for the baseline heuristic agents
(cambia-444).

Background: several ``choose_action`` implementations in
``src.agents.baseline_agents`` rebuilt a fresh ``set()`` from the (already
canonically ordered) ``legal_actions`` list to filter by action type --
e.g. ``snap_own_actions = {a for a in legal_actions if isinstance(a,
ActionSnapOwn)}`` -- then iterated that set directly, or fed it to
``random.choice(list(...))``. ``GameAction`` NamedTuples carry a ``tag: str``
type-discriminator field (src/constants.py), so their hash -- and hence
Python's ``set`` iteration order -- depends on ``PYTHONHASHSEED``, which is
randomized per-process by default. That made the affected agents'
decisions (and therefore any downstream measurement fed by them, e.g. the
Tier-B LBR and ISMCTS-BR exploitability estimators, whose default strong
opponent is ``ImperfectGreedyAgent``) non-reproducible across processes even
under an identical RNG seed, despite the resulting action DISTRIBUTION being
unbiased (this is a reproducibility bug, not a bias bug).

Fix (src/agents/baseline_agents.py): the rebuilt-and-filtered collections are
now list comprehensions, which preserve ``legal_actions``' existing
canonical order instead of re-exposing hash-salted set order. Affected:
``GreedyAgent`` (already safe via ``min()``/``sorted()``, converted for
consistency), ``ImperfectGreedyAgent``, ``MemoryHeuristicAgent``,
``RandomNoCambiaAgent``, ``RandomLateCambiaAgent``, ``AggressiveSnapAgent``,
``HumanPlayerAgent``.

These tests verify the fix two ways:

1. Cross-process: the SAME seeded self-play games, for each affected agent
   class at seat 0, run in fresh subprocesses under deliberately different
   ``PYTHONHASHSEED`` values, asserting the recorded action trace is
   byte-identical. This is a general safety net, but only actually exercises
   the fix when a game happens to reach a snap decision with 2+
   simultaneously-matching candidates -- which real self-play may or may not
   hit within a small seed budget.

2. Direct order-fidelity (single-process, no subprocess needed): forces a
   snap phase with multiple simultaneously-matching ``ActionSnapOwn`` /
   ``ActionSnapOpponent`` candidates and asserts ``choose_action`` returns
   whichever candidate is FIRST in the ``legal_actions`` list -- proving the
   implementation now respects input list order instead of re-deriving an
   arbitrary (set-hash-driven) order. This directly and reliably pins the
   causal mechanism of the fix, independent of whether a given
   PYTHONHASHSEED happens to produce a visible reordering.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.agents.baseline_agents import (
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    HumanPlayerAgent,
)
from src.card import Card
from src.config import CambiaRulesConfig
from src.constants import ActionSnapOwn, ActionSnapOpponent, ActionPassSnap
from src.game.engine import CambiaGameState

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------------------
# Subprocess script: plays a handful of tiny seeded games with each affected
# agent class at seat 0 (a seeded RandomAgent at seat 1), recording the
# applied-action trace. Kept self-contained (no test-module imports) since it
# runs as a fresh ``python -c`` process.
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = """
import random
from dataclasses import dataclass, field

from src.config import CambiaRulesConfig
from src.game.engine import CambiaGameState
from src.agents.baseline_agents import (
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    HumanPlayerAgent,
    RandomNoCambiaAgent,
    RandomLateCambiaAgent,
    RandomAgent,
)


@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(default_factory=_GreedyAgentConfig)


@dataclass
class _Config:
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)
    cambia_rules: CambiaRulesConfig = field(default_factory=CambiaRulesConfig)


AGENT_CLASSES = [
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    HumanPlayerAgent,
    RandomNoCambiaAgent,
    RandomLateCambiaAgent,
]

SEEDS = [11, 137]
MAX_TURNS = 60

trace = []
for AgentClass in AGENT_CLASSES:
    for seed in SEEDS:
        # RandomAgent (and RandomNoCambiaAgent/RandomLateCambiaAgent, which
        # extend it) draw from the GLOBAL random module, not an instance RNG
        # -- matching src.cfr.lbr.collect_infosets' own convention of
        # reseeding the shared global module before a deterministic run.
        random.seed(seed * 31 + 7)
        config = _Config()
        agent = AgentClass(player_id=0, config=config)
        opponent = RandomAgent(player_id=1, config=config)
        rules = CambiaRulesConfig(max_game_turns=MAX_TURNS)
        game = CambiaGameState(house_rules=rules, seed=seed)
        opp_rng = random.Random(seed * 31 + 7)

        turns = 0
        while not game.is_terminal() and turns < MAX_TURNS:
            actor = game.get_acting_player()
            if actor == -1:
                break
            legal = game.get_legal_actions()
            if not legal:
                break
            if actor == 0:
                action = agent.choose_action(game, legal)
            else:
                action = opp_rng.choice(legal)
            trace.append(repr(action))
            game.apply_action(action)
            turns += 1
        trace.append(f"---END {AgentClass.__name__} seed={seed} turns={turns}---")

print("|".join(trace))
"""


def _run_with_hashseed(hashseed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess with PYTHONHASHSEED={hashseed} failed: "
        f"stdout={result.stdout!r} stderr={result.stderr[-4000:]!r}"
    )
    assert (
        result.stdout.strip()
    ), f"subprocess with PYTHONHASHSEED={hashseed} produced no trace"
    return result.stdout


def test_baseline_agent_action_traces_identical_across_pythonhashseed():
    """Same seeded games, same affected agent classes, run in fresh
    subprocesses with deliberately different PYTHONHASHSEED values. The
    recorded action trace must be byte-identical -- the direct regression
    test for cambia-444 (set-comprehension rebuilds of legal_actions inside
    choose_action re-exposing hash-salted iteration order)."""
    out_a = _run_with_hashseed("0")
    out_b = _run_with_hashseed("1")
    out_c = _run_with_hashseed("2147483647")

    assert out_a == out_b, (
        "baseline agent action trace differs between PYTHONHASHSEED=0 and "
        "PYTHONHASHSEED=1 -- hash-seed dependence regressed (cambia-444)"
    )
    assert out_a == out_c, (
        "baseline agent action trace differs between PYTHONHASHSEED=0 and "
        "PYTHONHASHSEED=2147483647 -- hash-seed dependence regressed (cambia-444)"
    )


# ---------------------------------------------------------------------------
# Direct order-fidelity tests (single-process, deterministic).
# ---------------------------------------------------------------------------


class _GreedyAgentConfig:
    cambia_call_threshold = 5


class _AgentsConfig:
    greedy_agent = _GreedyAgentConfig()


class _Config:
    agents = _AgentsConfig()
    cambia_rules = CambiaRulesConfig()


def _make_game() -> CambiaGameState:
    return CambiaGameState(house_rules=CambiaRulesConfig())


_SNAP_OWN_AGENT_CLASSES = [
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    HumanPlayerAgent,
]


@pytest.mark.parametrize("AgentClass", _SNAP_OWN_AGENT_CLASSES)
def test_snap_own_selection_respects_legal_actions_list_order(AgentClass):
    """With three own slots simultaneously known to match the discard rank,
    choose_action must return whichever ActionSnapOwn is FIRST in the
    legal_actions list -- not whichever a set() would have hashed first.
    Checked in both an order and its exact reverse, so a regression to a
    set()-based rebuild (whose iteration order is insertion-order-independent)
    would need to coincidentally reproduce list order twice in a row to slip
    past this test undetected."""
    config = _Config()
    game = _make_game()
    agent = AgentClass(player_id=0, config=config)
    agent._ensure_initialized(game)

    # All three slots known and matching rank "5"; slot 3 stays unknown.
    agent.own_memory = {0: 5, 1: 5, 2: 5, 3: None}
    agent.own_rank_memory = {0: "5", 1: "5", 2: "5", 3: None}
    game.discard_pile = [Card(rank="5", suit="H")]
    game.snap_phase_active = True

    candidates = [
        ActionSnapOwn(own_card_hand_index=0),
        ActionSnapOwn(own_card_hand_index=1),
        ActionSnapOwn(own_card_hand_index=2),
        ActionPassSnap(),
    ]

    forward = agent.choose_action(game, list(candidates))
    assert forward == candidates[0], (
        f"{AgentClass.__name__} chose {forward!r} instead of the first listed "
        f"match {candidates[0]!r} (cambia-444 order-fidelity regression)"
    )

    reversed_candidates = list(reversed(candidates))
    expected_backward = next(
        a for a in reversed_candidates if isinstance(a, ActionSnapOwn)
    )
    backward = agent.choose_action(game, reversed_candidates)
    assert backward == expected_backward, (
        f"{AgentClass.__name__} chose {backward!r} instead of the first listed "
        f"match {expected_backward!r} on the reversed input (cambia-444 "
        f"order-fidelity regression)"
    )

    game.snap_phase_active = False


def test_aggressive_snap_opponent_selection_respects_legal_actions_list_order():
    """AggressiveSnapAgent additionally snaps opponent cards; same
    order-fidelity requirement applies to its snap_opp_actions rebuild."""
    config = _Config()
    game = _make_game()
    agent = AggressiveSnapAgent(player_id=0, config=config)
    agent._ensure_initialized(game)

    # No own-slot matches, so the agent falls through to opponent snaps.
    agent.own_memory = {0: None, 1: None, 2: None, 3: None}
    agent.own_rank_memory = {0: None, 1: None, 2: None, 3: None}
    agent.opponent_memory = {0: 5, 1: 5, 2: 5, 3: None}
    agent.opponent_rank_memory = {0: "5", 1: "5", 2: "5", 3: None}
    game.discard_pile = [Card(rank="5", suit="H")]
    game.snap_discarded_card = Card(rank="5", suit="H")
    game.snap_phase_active = True

    candidates = [
        ActionSnapOwn(own_card_hand_index=0),  # unmatched (unknown), falls through
        ActionSnapOpponent(opponent_target_hand_index=0),
        ActionSnapOpponent(opponent_target_hand_index=1),
        ActionSnapOpponent(opponent_target_hand_index=2),
        ActionPassSnap(),
    ]

    forward = agent.choose_action(game, list(candidates))
    assert forward == candidates[1], (
        f"AggressiveSnapAgent chose {forward!r} instead of the first listed "
        f"opponent match {candidates[1]!r} (cambia-444 order-fidelity regression)"
    )

    reversed_candidates = list(reversed(candidates))
    backward = agent.choose_action(game, reversed_candidates)
    # reversed_candidates[0] is ActionPassSnap; the first ActionSnapOpponent
    # in the reversed list is opponent_target_hand_index=2.
    expected_backward = ActionSnapOpponent(opponent_target_hand_index=2)
    assert backward == expected_backward, (
        f"AggressiveSnapAgent chose {backward!r} instead of the first listed "
        f"opponent match {expected_backward!r} on the reversed input "
        f"(cambia-444 order-fidelity regression)"
    )

    game.snap_phase_active = False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
