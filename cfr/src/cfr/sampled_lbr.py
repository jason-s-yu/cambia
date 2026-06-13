"""
cfr/src/cfr/sampled_lbr.py

Sampled Local Best Response (LBR) for approximate exploitability measurement.

Exploitability is estimated by:
  1. Playing self-play games under the agent's policy.
  2. Randomly sampling decision points for the agent (P0).
  3. At each sampled infoset, computing the BR value (max over actions of
     mean rollout utility) and the agent's value (utility of agent's chosen action).
  4. Exploitability = mean(BR_value - agent_value) across sampled infosets.
"""

import copy
import logging
from typing import Any, Dict

import numpy as np

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import RandomAgent
from src.cfr.lbr import collect_infosets, _make_random_opponent

logger = logging.getLogger(__name__)

# Exploitability is always measured from P0 (the agent under test).
_PLAYER_ID = 0


def _rollout(game_state: CambiaGameState, agents: list, max_turns: int) -> float:
    """Roll out a game from the current state under the given agents.

    Returns the final utility for _PLAYER_ID, or a hand-score estimate on timeout.
    """
    turns = 0
    while not game_state.is_terminal() and turns < max_turns:
        turns += 1
        ap = game_state.get_acting_player()
        if ap == -1:
            break
        la = game_state.get_legal_actions()
        if not la:
            break
        rollout_act = agents[ap].choose_action(game_state, la)
        try:
            game_state.apply_action(rollout_act)
        except Exception:
            break

    if game_state.is_terminal():
        return game_state._utilities[_PLAYER_ID]

    # Timeout: approximate from hand scores (lower hand score = better)
    try:
        my_score = sum(c.value for c in game_state.players[_PLAYER_ID].hand)
        opp_score = sum(c.value for c in game_state.players[1 - _PLAYER_ID].hand)
        if my_score < opp_score:
            return 1.0
        elif my_score > opp_score:
            return -1.0
        return 0.0
    except Exception:
        return 0.0


def sampled_lbr(
    agent_wrapper,
    config,
    num_infosets: int = 10000,
    br_rollouts_per_infoset: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute the Tier-A sampled LBR exploitability estimate.

    Tier A: trajectories are generated against a uniform-random opponent and BR
    continuation rollouts play both seats uniform-random. This is a LOOSE lower
    bound (a real adversary plays well after a deviation); the relative ordering
    across agents is trustworthy, the absolute number is not. For the tighter
    agent-policy variant see ``src.cfr.lbr.tier_b_lbr``.

    Algorithm:
    1. Play games under agent's policy (as P0) vs RandomAgent (as P1).
    2. Collect P0 decision points until ``num_infosets`` are gathered
       (via ``src.cfr.lbr.collect_infosets``; fixes the old 14x over-request).
    3. For each sampled infoset:
       a. For each legal action: deep-copy state, apply action, rollout, record utility.
       b. BR value = max over actions of mean rollout utility.
       c. Agent value = mean rollout utility for agent's chosen action.
    4. Exploitability = mean(BR_value - agent_value) across infosets.

    Args:
        agent_wrapper: Agent with a `choose_action(game_state, legal_actions)` method.
                       Optionally has `initialize_state(game_state)`.
        config: Config with `config.cambia_rules` (CambiaRulesConfig).
        num_infosets: Target number of infosets to sample.
        br_rollouts_per_infoset: Rollouts per action per infoset.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
          - exploitability: float (mean BR gap; always >= 0)
          - num_infosets_sampled: int
          - std_err: float (standard error of the mean)
    """
    house_rules = config.cambia_rules
    max_turns = getattr(house_rules, "max_game_turns", 0)
    if max_turns <= 0:
        max_turns = 500

    # Collect P0 infosets via the shared collector. Tier A uses a uniform-random
    # trajectory opponent (and random rollouts below). Routing through
    # collect_infosets fixes BUG-3: the old inline collector sized games_needed
    # and the sample rate for an assumed 40 P0 decisions/game while real games
    # average ~3, so requesting N infosets collected ~0.07*N. The shared
    # collector loops until the requested count is met (subject to a safety cap).
    sampled_infosets = collect_infosets(
        agent_wrapper,
        config,
        num_infosets=num_infosets,
        seed=seed,
        trajectory_opponent_factory=_make_random_opponent,
    )

    if not sampled_infosets:
        logger.warning("sampled_lbr: No infosets sampled.")
        return {"exploitability": 0.0, "num_infosets_sampled": 0, "std_err": 0.0}

    exploitability_gaps = []

    for state_copy, legal_actions, agent_action_idx in sampled_infosets:
        action_mean_utils = []

        for action in legal_actions:
            utilities = []
            for _ in range(br_rollouts_per_infoset):
                branch = copy.deepcopy(state_copy)
                try:
                    branch.apply_action(action)
                except Exception:
                    utilities.append(0.0)
                    continue

                rollout_agents = [RandomAgent(0, config), RandomAgent(1, config)]
                util = _rollout(branch, rollout_agents, max_turns)
                utilities.append(util)

            action_mean_utils.append(float(np.mean(utilities)) if utilities else 0.0)

        if not action_mean_utils:
            continue

        br_value = max(action_mean_utils)
        safe_idx = min(agent_action_idx, len(action_mean_utils) - 1)
        agent_value = action_mean_utils[safe_idx]
        exploitability_gaps.append(br_value - agent_value)

    if not exploitability_gaps:
        return {"exploitability": 0.0, "num_infosets_sampled": 0, "std_err": 0.0}

    gaps_arr = np.array(exploitability_gaps)
    exploitability = float(np.mean(gaps_arr))
    std_err = (
        float(np.std(gaps_arr, ddof=1) / np.sqrt(len(gaps_arr)))
        if len(gaps_arr) > 1
        else 0.0
    )

    return {
        "exploitability": exploitability,
        "num_infosets_sampled": len(sampled_infosets),
        "std_err": std_err,
    }
