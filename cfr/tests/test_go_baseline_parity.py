"""The Go-backed baselines decide exactly what the Python ones decide.

mean_imp is a historical metric, so moving a baseline's decision inside the
engine (cambia-1487) is only admissible if the decision itself is unchanged.
This module is the gate on that: for each baseline it replays whole seeded
games twice, once with the pure-Python reference from baseline_agents.py in
both seats and once with the Go-backed subclass from go_baselines.py, and
requires the two applied-action sequences to be identical.

Replaying rather than interleaving is deliberate. The random baselines draw
from a CPython RNG whose stream is part of the metric, and a replay run
consumes that stream from the same starting state in the same order, so a fast
path that drew a different number of words would desynchronise and show up as a
diverged action rather than being masked by a shared generator.

The heavy legs are marked slow; the default run keeps a smaller pass that still
covers every baseline and both rule profiles.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import action_codec  # noqa: E402
from src.agents.baseline_agents import (  # noqa: E402
    AggressiveSnapAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    RandomAgent,
    RandomLateCambiaAgent,
    RandomNoCambiaAgent,
)
from src.agents.go_baselines import (  # noqa: E402
    GoAggressiveSnapAgent,
    GoImperfectGreedyAgent,
    GoMemoryHeuristicAgent,
    GoRandomAgent,
    GoRandomLateCambiaAgent,
    GoRandomNoCambiaAgent,
)
from src.config import load_config  # noqa: E402
from src.ffi.bridge import GoEngine  # noqa: E402

#: reference class, Go-backed class, and whether the policy draws randomly.
BASELINE_PAIRS = {
    "random": (RandomAgent, GoRandomAgent, True),
    "random_no_cambia": (RandomNoCambiaAgent, GoRandomNoCambiaAgent, True),
    "random_late_cambia": (RandomLateCambiaAgent, GoRandomLateCambiaAgent, True),
    "imperfect_greedy": (ImperfectGreedyAgent, GoImperfectGreedyAgent, False),
    "memory_heuristic": (MemoryHeuristicAgent, GoMemoryHeuristicAgent, False),
    "aggressive_snap": (AggressiveSnapAgent, GoAggressiveSnapAgent, False),
}

#: Both rule profiles the battery is run under: the production ruleset (opponent
#: snapping, discard draws and replace abilities all on, so every branch of every
#: policy is reachable) and the tighter one.
RULE_CONFIGS = ("config/prtcfr_production.yaml", "config/ppo_encoding_v2.yaml")

#: Deck seeds are drawn from here so both runs of a game deal identically.
SEED_BASE = 0x51875EED

#: Action-count safety valve, well above any legal game's length.
MAX_ACTIONS = 6000


def _config(path: str):
    cfg = load_config(str(Path(__file__).resolve().parent.parent / path))
    assert cfg is not None, f"failed to load {path}"
    return cfg


def _build(cls, config, seed, random_draw):
    """Build both seats of one policy, seeding the drawing ones per seat."""
    if random_draw:
        return [cls(seat, config, seed=seed + seat) for seat in (0, 1)]
    return [cls(seat, config) for seat in (0, 1)]


def _play(agents, config, deck_seed):
    """Play one seeded game with these agents in both seats.

    Returns the applied action indices, in order.
    """
    for agent in agents:
        agent._last_game_id = None
    applied = []
    with GoEngine(
        seed=deck_seed, house_rules=config.cambia_rules, num_players=2
    ) as engine:
        while not engine.is_terminal() and len(applied) < MAX_ACTIONS:
            legal = action_codec.LazyLegalActions(engine.legal_action_indices())
            if not legal:
                break
            action = agents[engine.acting_player()].choose_action(engine, legal)
            idx = action_codec.TWO_PLAYER_INDEX[action]
            applied.append(idx)
            engine.apply_action(idx)
    return applied


def _compare(baseline: str, config_path: str, games: int) -> int:
    """Replay `games` seeded games under both implementations.

    Returns the number of decisions compared. Fails on the first game whose
    action sequences differ, naming the seed and the diverging step.
    """
    ref_cls, go_cls, random_draw = BASELINE_PAIRS[baseline]
    config = _config(config_path)
    decisions = 0

    for game in range(games):
        deck_seed = SEED_BASE + game * 7919
        agent_seed = SEED_BASE ^ (game * 2654435761 & 0xFFFF_FFFF)

        # Reseed the global module too, so a policy that fell back to it would
        # still start both runs from the same point rather than from entropy.
        random.seed(agent_seed)
        ref = _play(_build(ref_cls, config, agent_seed, random_draw), config, deck_seed)
        random.seed(agent_seed)
        fast = _play(_build(go_cls, config, agent_seed, random_draw), config, deck_seed)

        assert len(ref) == len(fast), (
            f"{baseline} on {config_path}: game seed {deck_seed} ran "
            f"{len(ref)} reference actions and {len(fast)} engine-side ones"
        )
        for step, (a, b) in enumerate(zip(ref, fast)):
            assert a == b, (
                f"{baseline} on {config_path}: game seed {deck_seed} diverged at "
                f"step {step}: reference chose action {a}, engine-side chose {b}"
            )
        decisions += len(ref)

    assert decisions > 0, f"{baseline} on {config_path} compared no decisions"
    return decisions


@pytest.mark.integration
@pytest.mark.parametrize("baseline", sorted(BASELINE_PAIRS))
@pytest.mark.parametrize("config_path", RULE_CONFIGS)
def test_go_baseline_matches_python_smoke(baseline, config_path):
    """Every baseline agrees with its reference over a short pass."""
    assert _compare(baseline, config_path, games=60) > 0


@pytest.mark.slow
@pytest.mark.parametrize("baseline", sorted(BASELINE_PAIRS))
def test_go_baseline_matches_python_full(baseline):
    """Every baseline agrees with its reference over the full 2000-game pass.

    Run under the production ruleset, which is the one the mean_imp battery
    plays and the only one under which opponent snapping, discard draws and
    replace abilities are all reachable.
    """
    decisions = _compare(baseline, RULE_CONFIGS[0], games=2000)
    assert decisions >= 2000, f"{baseline} compared only {decisions} decisions"


def test_randrange_draws_what_choice_draws():
    """`randrange(n)` picks what `choice(seq)` picks, off the same stream.

    This is the identity the random baselines' fast path rests on: the engine
    returns the candidate sequence and the wrapper draws a position into it,
    rather than drawing from a decoded action list. Both go through
    `Random._randbelow(n)`, so the position and the consumed words match.
    """
    for seed in range(200):
        for n in range(1, 40):
            seq = list(range(n))
            direct = random.Random(seed)
            indexed = random.Random(seed)
            for _ in range(5):
                assert direct.choice(seq) == indexed.randrange(n)
            assert direct.getstate() == indexed.getstate()
