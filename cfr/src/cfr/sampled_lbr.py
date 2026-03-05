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
import random as _random_module
from typing import Any, Dict

import numpy as np

from src.game.engine import CambiaGameState
from src.agents.baseline_agents import RandomAgent

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
    """Compute sampled LBR exploitability estimate.

    Algorithm:
    1. Play games under agent's policy (as P0) vs RandomAgent (as P1).
    2. At P0 decision points, sample infosets with a target rate.
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
    rng = np.random.default_rng(seed)
    _random_module.seed(seed)

    house_rules = config.cambia_rules
    max_turns = getattr(house_rules, "max_game_turns", 0)
    if max_turns <= 0:
        max_turns = 500

    # Estimate how many games to play to collect ~num_infosets samples.
    # Typical game: ~40 P0 decisions. We plan for ~10% sample rate.
    expected_p0_decisions = 40
    games_needed = max(200, (num_infosets * 10) // expected_p0_decisions)
    sample_prob = min(1.0, num_infosets / max(1, expected_p0_decisions * games_needed))

    # Each entry: (state_copy, legal_actions_list, agent_action_idx)
    sampled_infosets = []

    for game_idx in range(games_needed):
        if len(sampled_infosets) >= num_infosets:
            break

        # Seed the per-game RNG deterministically from the numpy rng.
        game_seed = int(rng.integers(0, 2**31))
        game_rng = _random_module.Random(game_seed)
        game_state = CambiaGameState(house_rules=house_rules, _rng=game_rng)
        opp_agent = RandomAgent(1, config)

        if hasattr(agent_wrapper, "initialize_state"):
            try:
                agent_wrapper.initialize_state(game_state)
            except Exception:
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
                # Deepcopy BEFORE choose_action so we capture pre-decision state.
                if (
                    len(sampled_infosets) < num_infosets
                    and rng.random() < sample_prob
                ):
                    state_copy = copy.deepcopy(game_state)
                    actions_list = list(legal_actions)
                    agent_action = agent_wrapper.choose_action(game_state, legal_actions)
                    try:
                        action_idx = actions_list.index(agent_action)
                    except ValueError:
                        action_idx = 0
                    sampled_infosets.append(
                        (state_copy, actions_list, action_idx)
                    )
                    chosen_action = agent_action
                else:
                    chosen_action = agent_wrapper.choose_action(game_state, legal_actions)
            else:
                chosen_action = opp_agent.choose_action(game_state, legal_actions)

            try:
                game_state.apply_action(chosen_action)
            except Exception:
                break

    if not sampled_infosets:
        logger.warning(
            "sampled_lbr: No infosets sampled after %d games.", games_needed
        )
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
