"""
cfr/src/cfr/lbr.py

Tier-B sampled Local Best Response (LBR) for approximate exploitability, plus
the shared infoset collector that fixes the Tier-A 14x sample-request bug
(BUG-3 in the CFR-ceiling E3 report).

Background
----------
Tier-A LBR (``src.cfr.sampled_lbr.sampled_lbr``) estimates exploitability with:
  - trajectories generated against a uniform-random opponent, and
  - best-response continuation rollouts where BOTH seats play uniform-random.

Random rollouts make Tier-A a LOOSE lower bound: a real adversary plays well
after a deviation, not randomly, so Tier-A understates the true exploitability.
The relative ordering across agents (measured identically) is trustworthy; the
absolute number is not.

Tier-B tightens the bound by making the continuation realistic:
  - trajectories are generated against a STRONG fixed opponent (heuristic), and
  - BR continuation rollouts roll seat 0 under the AGENT'S OWN policy and seat 1
    under the strong opponent ("agent-policy rollouts, strong trajectory
    opponent", per the E3 report / report.md build list).

Because the agent under test plays on after its own one-step deviation rather
than reverting to random, the best-response value is measured against a
realistic continuation, yielding a higher (tighter) exploitability estimate.

The 14x sampler fix
-------------------
The Tier-A collector sized ``games_needed`` and the per-decision sample
probability for an assumed 40 P0 decisions/game, but real Cambia games average
~3 P0 decisions/game under random play. Requesting N infosets therefore
collected only ~0.07*N. ``collect_infosets`` removes the brittle
decisions/game assumption: it plays games and samples P0 decisions until it has
the requested count (or hits a safety cap on games played), so requesting N
collects ~N. ``src.cfr.sampled_lbr.sampled_lbr`` now consumes this same
collector, so the production ``cambia evaluate --lbr`` path is fixed too.

Both tiers are pure eval-time measurement: no network or harness state is
mutated. The agent wrapper's ``choose_action`` is read-only on game state, so
it is safe to call on deep-copied branch states without forking the wrapper.
"""

import copy
import logging
import math
import random as _random_module
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import RandomAgent

logger = logging.getLogger(__name__)

# Exploitability is always measured from P0 (the agent under test).
_PLAYER_ID = 0
_OPPONENT_ID = 1

# Measured under random play (E3 report, 200-game sample): ~3 P0 decisions/game.
# Used only to size the games-played safety cap, never to gate collection.
_EST_P0_DECISIONS_PER_GAME = 3.0

# A sampled infoset: (pre-decision state copy, legal-action list, agent action idx).
SampledInfoset = Tuple[CambiaGameState, list, int]

# Factory signature: (player_id, config) -> agent exposing choose_action.
OpponentFactory = Callable[[int, Any], Any]


def _make_strong_opponent(player_id: int, config: Any):
    """Construct the default strong fixed opponent (ImperfectGreedyAgent).

    Falls back to RandomAgent if the heuristic cannot be built from this config
    (e.g. a minimal config without ``agents.greedy_agent``), so the collector
    never hard-fails on config shape.
    """
    try:
        from src.agents.baseline_agents import ImperfectGreedyAgent

        return ImperfectGreedyAgent(player_id, config)
    except Exception as exc:  # JUSTIFIED: eval resilience to minimal configs
        logger.warning(
            "lbr: strong opponent unavailable (%s); falling back to RandomAgent.",
            exc,
        )
        return RandomAgent(player_id, config)


def _make_random_opponent(player_id: int, config: Any):
    return RandomAgent(player_id, config)


# Default Tier-B opponents: strong both for trajectory generation and for the
# adversary seat during agent-policy continuation rollouts.
DEFAULT_TRAJECTORY_OPPONENT: OpponentFactory = _make_strong_opponent
DEFAULT_ROLLOUT_OPPONENT: OpponentFactory = _make_strong_opponent


def _resolve_max_turns(config: Any) -> int:
    house_rules = config.cambia_rules
    max_turns = getattr(house_rules, "max_game_turns", 0)
    if max_turns <= 0:
        max_turns = 500
    return max_turns


def collect_infosets(
    agent_wrapper,
    config,
    num_infosets: int,
    seed: int = 42,
    trajectory_opponent_factory: OpponentFactory = _make_random_opponent,
    sample_prob: float = 1.0,
    max_games: Optional[int] = None,
) -> List[SampledInfoset]:
    """Collect pre-decision P0 infosets by self-play against a trajectory opponent.

    Plays games with ``agent_wrapper`` at seat 0 and a fresh
    ``trajectory_opponent_factory(1, config)`` opponent at seat 1, recording P0
    decision points until ``num_infosets`` are collected or ``max_games`` games
    have been played.

    This is the BUG-3 fix: collection is bounded by the requested COUNT, not by
    a games budget sized for a wrong decisions/game assumption. Requesting N
    infosets yields ~N (subject to the safety cap).

    Args:
        agent_wrapper: agent under test; ``choose_action(state, legal)`` and an
            optional ``initialize_state(state)``.
        config: config exposing ``cambia_rules`` (and ``agents`` for strong opps).
        num_infosets: target number of P0 infosets to collect.
        seed: RNG seed for reproducibility.
        trajectory_opponent_factory: builds the seat-1 opponent per game.
        sample_prob: per-eligible-P0-decision sampling probability. Defaults to
            1.0 (take every decision) so the target count is reached quickly.
        max_games: hard cap on games played (safety against unreachable targets).
            Defaults to a generous multiple of the games implied by the measured
            decisions/game, with an absolute ceiling.

    Returns:
        list of (state_copy, legal_actions_list, agent_action_idx).
    """
    rng = np.random.default_rng(seed)
    _random_module.seed(seed)

    house_rules = config.cambia_rules
    max_turns = _resolve_max_turns(config)

    if max_games is None:
        implied = int(math.ceil(num_infosets / _EST_P0_DECISIONS_PER_GAME))
        # 4x margin over the implied games, floor 200, absolute ceiling so an
        # agent that never reaches a P0 decision cannot loop unbounded.
        max_games = min(max(200, implied * 4), 2_000_000)

    sampled: List[SampledInfoset] = []
    games_played = 0

    while len(sampled) < num_infosets and games_played < max_games:
        games_played += 1

        game_seed = int(rng.integers(0, 2**31))
        game_rng = _random_module.Random(game_seed)
        game_state = CambiaGameState(house_rules=house_rules, _rng=game_rng)
        opp_agent = trajectory_opponent_factory(_OPPONENT_ID, config)

        if hasattr(agent_wrapper, "initialize_state"):
            try:
                agent_wrapper.initialize_state(game_state)
            except Exception:  # JUSTIFIED: eval resilience
                pass

        turn = 0
        while not game_state.is_terminal() and turn < max_turns:
            turn += 1
            ap = game_state.get_acting_player()
            if ap == -1:
                break
            legal_actions = game_state.get_legal_actions()
            if not legal_actions:
                break

            if ap == _PLAYER_ID:
                take = len(sampled) < num_infosets and rng.random() < sample_prob
                if take:
                    state_copy = copy.deepcopy(game_state)
                    actions_list = list(legal_actions)
                    agent_action = agent_wrapper.choose_action(
                        game_state, legal_actions
                    )
                    try:
                        action_idx = actions_list.index(agent_action)
                    except ValueError:
                        action_idx = 0
                    sampled.append((state_copy, actions_list, action_idx))
                    chosen_action = agent_action
                else:
                    chosen_action = agent_wrapper.choose_action(
                        game_state, legal_actions
                    )
            else:
                chosen_action = opp_agent.choose_action(game_state, legal_actions)

            try:
                game_state.apply_action(chosen_action)
            except Exception:  # JUSTIFIED: eval resilience
                break

            # Full-recall token-stream feed for agents that consume the engine
            # token stream (PRT-CFR): every applied action, both seats. Generic
            # and guarded by hasattr, so only such agents are affected; without
            # it the PRT-CFR prefix would stay empty across the game and its
            # policy (hence the LBR exploitability) would be measured wrong.
            if hasattr(agent_wrapper, "observe_transition"):
                try:
                    agent_wrapper.observe_transition(game_state, chosen_action, ap)
                except Exception:  # JUSTIFIED: eval resilience
                    pass

            if len(sampled) >= num_infosets:
                break

    if len(sampled) < num_infosets:
        logger.warning(
            "collect_infosets: collected %d of %d requested after %d games "
            "(safety cap hit).",
            len(sampled),
            num_infosets,
            games_played,
        )
    return sampled


def _rollout_terminal_utility(game_state: CambiaGameState) -> float:
    """Terminal utility for P0, or a hand-score estimate on turn-cap timeout.

    Matches the Tier-A tie/timeout convention so the two tiers are comparable.
    """
    if game_state.is_terminal():
        return game_state._utilities[_PLAYER_ID]
    try:
        my_score = sum(c.value for c in game_state.players[_PLAYER_ID].hand)
        opp_score = sum(c.value for c in game_state.players[_OPPONENT_ID].hand)
        if my_score < opp_score:
            return 1.0
        if my_score > opp_score:
            return -1.0
        return 0.0
    except Exception:  # JUSTIFIED: eval resilience
        return 0.0


def _agent_policy_rollout(
    branch_state: CambiaGameState,
    agent_wrapper,
    rollout_opponent,
    max_turns: int,
) -> float:
    """Roll out a branch under agent-policy play.

    Seat 0 plays the agent's own policy (via ``agent_wrapper``); seat 1 plays
    the strong ``rollout_opponent``. ``choose_action`` is read-only on game
    state, so the shared wrapper is safe to reuse here without forking.
    """
    turns = 0
    while not branch_state.is_terminal() and turns < max_turns:
        turns += 1
        ap = branch_state.get_acting_player()
        if ap == -1:
            break
        la = branch_state.get_legal_actions()
        if not la:
            break
        try:
            if ap == _PLAYER_ID:
                act = agent_wrapper.choose_action(branch_state, la)
            else:
                act = rollout_opponent.choose_action(branch_state, la)
        except Exception:  # JUSTIFIED: eval resilience
            break
        try:
            branch_state.apply_action(act)
        except Exception:  # JUSTIFIED: eval resilience
            break
    return _rollout_terminal_utility(branch_state)


def tier_b_lbr(
    agent_wrapper,
    config,
    num_infosets: int = 10000,
    br_rollouts_per_infoset: int = 100,
    seed: int = 42,
    trajectory_opponent_factory: OpponentFactory = DEFAULT_TRAJECTORY_OPPONENT,
    rollout_opponent_factory: OpponentFactory = DEFAULT_ROLLOUT_OPPONENT,
    max_games: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the Tier-B sampled LBR exploitability estimate.

    Algorithm:
      1. Collect P0 infosets along trajectories where the agent (seat 0) faces a
         strong fixed opponent (seat 1).
      2. At each sampled infoset, for each legal action:
           deep-copy the pre-decision state, apply the candidate action, then
           roll out the continuation under agent-policy play (seat 0 = agent,
           seat 1 = strong opponent). Average over ``br_rollouts_per_infoset``.
      3. BR value = max over actions of mean continuation utility.
         Agent value = mean continuation utility of the action the agent chose.
      4. Exploitability = mean(BR value - agent value) over infosets (>= 0).

    Returns a dict:
        exploitability: float (mean BR gap; >= 0 by construction)
        num_infosets_sampled: int
        std_err: float (standard error of the per-infoset gap mean)
        tier: "B"
        rollout_opponent: str (label of the strong opponent class, for the row)
    """
    max_turns = _resolve_max_turns(config)

    sampled = collect_infosets(
        agent_wrapper,
        config,
        num_infosets=num_infosets,
        seed=seed,
        trajectory_opponent_factory=trajectory_opponent_factory,
        max_games=max_games,
    )

    opp_label = type(rollout_opponent_factory(_OPPONENT_ID, config)).__name__

    if not sampled:
        logger.warning("tier_b_lbr: no infosets sampled.")
        return {
            "exploitability": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "tier": "B",
            "rollout_opponent": opp_label,
        }

    gaps: List[float] = []
    for state_copy, legal_actions, agent_action_idx in sampled:
        action_mean_utils: List[float] = []
        for action in legal_actions:
            utils: List[float] = []
            for _ in range(br_rollouts_per_infoset):
                branch = copy.deepcopy(state_copy)
                try:
                    branch.apply_action(action)
                except Exception:  # JUSTIFIED: eval resilience
                    utils.append(0.0)
                    continue
                rollout_opp = rollout_opponent_factory(_OPPONENT_ID, config)
                utils.append(
                    _agent_policy_rollout(
                        branch, agent_wrapper, rollout_opp, max_turns
                    )
                )
            action_mean_utils.append(float(np.mean(utils)) if utils else 0.0)

        if not action_mean_utils:
            continue
        br_value = max(action_mean_utils)
        safe_idx = min(agent_action_idx, len(action_mean_utils) - 1)
        agent_value = action_mean_utils[safe_idx]
        gaps.append(br_value - agent_value)

    if not gaps:
        return {
            "exploitability": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "tier": "B",
            "rollout_opponent": opp_label,
        }

    gaps_arr = np.array(gaps)
    exploitability = float(np.mean(gaps_arr))
    std_err = (
        float(np.std(gaps_arr, ddof=1) / np.sqrt(len(gaps_arr)))
        if len(gaps_arr) > 1
        else 0.0
    )
    return {
        "exploitability": exploitability,
        "num_infosets_sampled": len(sampled),
        "std_err": std_err,
        "tier": "B",
        "rollout_opponent": opp_label,
    }
