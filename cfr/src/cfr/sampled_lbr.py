"""
cfr/src/cfr/sampled_lbr.py

Sampled Local Best Response (LBR) for approximate exploitability measurement.

Exploitability is estimated by:
  1. Playing games under the agent's policy.
  2. Sampling decision points for the agent (P0).
  3. At each sampled infoset, computing the BR value (max over actions of
     mean rollout utility) and the agent's value (utility of agent's chosen action).
  4. Exploitability = mean(BR_value - agent_value) across sampled infosets.

Runs on the Go engine (cambia-1427): the search substrate, the rewind-based
branching and the policy boundary all live in ``src.cfr.lbr`` and are shared
with Tier B and with ISMCTS-BR. See that module's docstring for the
``GoSearchState`` contract and the policy hooks.
"""

import logging
import random as _random_module
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.cfr.lbr import (
    GoSearchState,
    UniformRandomPolicy,
    _make_random_opponent,
    _resolve_max_turns,
    collect_infosets,
    replay_infoset,
    terminal_utility,
)
from src.agents.action_codec import actions_from_mask

logger = logging.getLogger(__name__)

# Exploitability is always measured from P0 (the agent under test).
_PLAYER_ID = 0


def _rollout(state: GoSearchState, policies: list, max_turns: int) -> float:
    """Roll out from the current state under ``policies`` (indexed by seat).

    Returns the final utility for _PLAYER_ID, or a hand-score estimate on
    timeout.
    """
    turns = 0
    while not state.is_terminal() and turns < max_turns:
        turns += 1
        ap = state.acting_player()
        if ap == -1:
            break
        legal_indices = state.legal_indices()
        if not legal_indices:
            break
        legal_actions = actions_from_mask(legal_indices)
        try:
            act = policies[ap].choose_action(state.view(), legal_actions)
            pos = legal_actions.index(act)
        except Exception:  # JUSTIFIED: eval resilience
            break
        if not state.apply_index(legal_indices[pos]):
            break
    return terminal_utility(state)


def sampled_lbr(
    agent_wrapper,
    config,
    num_infosets: int = 10000,
    br_rollouts_per_infoset: int = 100,
    seed: int = 42,
    rollout_seed: Optional[int] = None,
    deal_decks: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Compute the Tier-A sampled LBR exploitability estimate.

    Tier A: trajectories are generated against a uniform-random opponent and BR
    continuation rollouts play both seats uniform-random. This is a LOOSE lower
    bound (a real adversary plays well after a deviation); the relative ordering
    across agents is trustworthy, the absolute number is not. For the tighter
    agent-policy variant see ``src.cfr.lbr.tier_b_lbr``.

    Algorithm:
    1. Play games with the agent as P0 against a uniform-random P1.
    2. Collect P0 decision points until ``num_infosets`` are gathered
       (via ``src.cfr.lbr.collect_infosets``; fixes the old 14x over-request).
    3. For each sampled infoset:
       a. Replay the state, then for each legal action: rewind, apply the
          action, roll out, record utility.
       b. BR value = max over actions of mean rollout utility.
       c. Agent value = mean rollout utility for the agent's chosen action.
    4. Exploitability = mean(BR_value - agent_value) across infosets.

    Args:
        agent_wrapper: policy under test; ``choose_action(view, legal_actions)``
            plus the optional hooks documented in ``src.cfr.lbr``.
        config: Config with `config.cambia_rules` (CambiaRulesConfig).
        num_infosets: Target number of infosets to sample.
        br_rollouts_per_infoset: Rollouts per action per infoset.
        seed: Random seed for reproducibility.
        rollout_seed: Seed for the rollout policies' RNG. Defaults to ``seed``,
            so a run is fully determined by ``seed`` alone; pass it separately
            only to re-roll the continuation noise over a fixed infoset sample.
        deal_decks: optional pool of explicit deck orders to deal from, required
            for configs whose deck the Go FFI rules struct cannot express (e.g.
            a ``deck_ranks`` tiny game). See ``src.cfr.lbr``.

    Returns:
        dict with keys:
          - exploitability: float (mean BR gap; always >= 0)
          - num_infosets_sampled: int
          - std_err: float (standard error of the mean)
          - tier: "A"
          - seed: int (echoed, so a persisted row records what produced it)
    """
    house_rules = config.cambia_rules
    max_turns = _resolve_max_turns(config)

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
        deal_decks=deal_decks,
    )

    def _empty(reason: str) -> Dict[str, Any]:
        logger.warning("sampled_lbr: %s", reason)
        return {
            "exploitability": 0.0,
            "num_infosets_sampled": 0,
            "std_err": 0.0,
            "tier": "A",
            "seed": seed,
        }

    if not sampled_infosets:
        return _empty("No infosets sampled.")

    # One RNG for every rollout policy in the run, so the whole Tier-A estimate
    # is a deterministic function of (seed, rollout_seed) and does not depend on
    # the global `random` module's state.
    rollout_rng = _random_module.Random(seed if rollout_seed is None else rollout_seed)

    exploitability_gaps: List[float] = []

    for infoset in sampled_infosets:
        try:
            state = replay_infoset(house_rules, infoset, agent_wrapper)
        except Exception as exc:  # JUSTIFIED: eval resilience
            logger.warning("sampled_lbr: infoset replay failed (%s); skipping.", exc)
            continue

        snap_h: Optional[int] = None
        action_mean_utils: List[float] = []
        try:
            snap_h = state.save()
            for action_idx in infoset.legal_indices:
                utilities: List[float] = []
                for _ in range(br_rollouts_per_infoset):
                    state.restore(snap_h)
                    if not state.apply_index(action_idx):
                        utilities.append(0.0)
                        continue
                    rollout_policies = [
                        UniformRandomPolicy(
                            0, _random_module.Random(rollout_rng.getrandbits(63))
                        ),
                        UniformRandomPolicy(
                            1, _random_module.Random(rollout_rng.getrandbits(63))
                        ),
                    ]
                    utilities.append(_rollout(state, rollout_policies, max_turns))
                action_mean_utils.append(float(np.mean(utilities)) if utilities else 0.0)
        finally:
            if snap_h is not None:
                GoSearchState.free_snapshot(snap_h)
            state.close()

        if not action_mean_utils:
            continue

        br_value = max(action_mean_utils)
        safe_pos = min(infoset.agent_action_pos, len(action_mean_utils) - 1)
        agent_value = action_mean_utils[safe_pos]
        exploitability_gaps.append(br_value - agent_value)

    if not exploitability_gaps:
        return _empty("no infoset produced a measurable gap.")

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
        "tier": "A",
        "seed": seed,
    }
