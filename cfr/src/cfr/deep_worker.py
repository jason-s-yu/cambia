"""
src/cfr/deep_worker.py

Implements the Deep CFR worker process using External Sampling MCCFR.

Key differences from worker.py (tabular outcome sampling):
- External Sampling: enumerate ALL actions at traverser nodes, sample ONE at opponent/chance nodes
- No importance sampling correction (exact regrets from enumeration)
- Returns ReservoirSamples instead of regret/strategy dict updates
- Uses neural network for strategy computation (AdvantageNetwork -> ReLU -> normalize)
- Encodes infosets from the Go agent state via encoding.py helpers
"""

import logging
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from .exceptions import NetworkError
from ..constants import (
    NUM_PLAYERS,
    EP_PBS_INPUT_DIM,
    N_PLAYER_NUM_ACTIONS,
    N_PLAYER_INPUT_DIM,
)
from ..serial_rotating_handler import SerialRotatingFileHandler
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import (
    AdvantageNetwork,
    HistoryValueNetwork,
    build_advantage_network,
    get_strategy_from_advantages,
)
from ..reservoir import ReservoirSample
from ..utils import WorkerStats, SimulationNodeData

logger = logging.getLogger(__name__)

# Progress update interval (nodes)
PROGRESS_UPDATE_NODE_INTERVAL = 2500

# Importance sampling weight clipping bound (OS-MCCFR variance reduction).
# Applied to the FULL weight pi^sigma(h a, z) / q(h -> z), not to the local 1/q(a|h)
# alone: clipping only the local factor leaves the tail uncorrected, which is the
# defect this bound used to hide.
#
# BIAS: the clip is a truncated importance sampling estimator. Weights above the
# bound are pulled down to it, so trajectories that sigma strongly prefers but the
# epsilon-mixed sampler rarely draws are under-counted, and the regret target is
# biased low on exactly those actions. The estimator is unbiased only on nodes
# where the clip does not bind. The bound trades that bias for a finite second
# moment: the unclipped weight is a product of per-node ratios over the traverser's
# whole remaining decision horizon, and its variance grows without bound in the
# horizon. See docs/sampling.md.
MAX_IS_WEIGHT = 20.0


def _clipped_is_weight(tail_ratio: float, sampling_prob: float) -> float:
    """
    Full outcome-sampling importance weight for one traverser decision, clipped.

    ``tail_ratio`` is pi^sigma(h a, z) / q(h a -> z), the ratio accumulated below the
    sampled action; ``sampling_prob`` is q(a|h) at this node. The product is the
    weight that makes ``u(z) * w`` an unbiased estimate of the counterfactual value
    v^sigma(h a). Non-finite products (over/underflow over a long horizon) collapse
    to the bound.
    """
    weight = tail_ratio / sampling_prob
    if not np.isfinite(weight):
        return MAX_IS_WEIGHT
    return min(weight, MAX_IS_WEIGHT)


@dataclass
class DeepCFRWorkerResult:
    """Results from a single Deep CFR worker traversal."""

    advantage_samples: List[ReservoirSample] = field(default_factory=list)
    strategy_samples: List[ReservoirSample] = field(default_factory=list)
    value_samples: List[ReservoirSample] = field(default_factory=list)
    stats: WorkerStats = field(default_factory=WorkerStats)
    simulation_nodes: List[SimulationNodeData] = field(default_factory=list)
    final_utility: Optional[List[float]] = None
    # ESCHER regret telemetry: mean(abs(sampled_regret)) and mean(abs(cf_regret)) across traverser nodes
    escher_sampled_regret_mag: Optional[float] = None
    escher_cf_regret_mag: Optional[float] = None


def _get_strategy_from_network(
    network: AdvantageNetwork,
    features: np.ndarray,
    action_mask: np.ndarray,
    _feat_buf: Optional[torch.Tensor] = None,
    _mask_buf: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute strategy from advantage network.

    Uses a pre-built AdvantageNetwork (created once per worker), runs forward pass,
    then applies ReLU + normalize (regret matching on predicted advantages).

    If _feat_buf and _mask_buf are provided (pre-allocated tensors of shape
    (1, INPUT_DIM) and (1, NUM_ACTIONS)), they are filled in-place with copy_()
    to avoid per-call tensor allocation.

    Returns numpy array of shape (NUM_ACTIONS,) with strategy probabilities.

    Raises:
        NetworkError: If network inference fails
    """
    try:
        with torch.inference_mode():
            if _feat_buf is not None and _mask_buf is not None:
                _feat_buf.copy_(
                    torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
                )
                _mask_buf.copy_(torch.from_numpy(action_mask).bool().unsqueeze(0))
                features_tensor = _feat_buf
                mask_tensor = _mask_buf
            else:
                features_tensor = torch.as_tensor(
                    features, dtype=torch.float32
                ).unsqueeze(0)
                mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0)
            raw_advantages = network(features_tensor, mask_tensor).squeeze(0)
            strategy_tensor = get_strategy_from_advantages(
                raw_advantages.unsqueeze(0), mask_tensor
            )
            return strategy_tensor.squeeze(0).numpy()
    except Exception as e:
        raise NetworkError(f"Network inference failed: {e}") from e


def _infer_decision_context(legal_mask: np.ndarray) -> int:
    """
    Infer DecisionContext integer from a legal action mask (Go engine).

    Action index ranges (from engine/types.go):
      0-2:    StartTurn (DrawStockpile, DrawDiscard, CallCambia)
      3-10:   PostDraw (DiscardNoAbility, DiscardWithAbility, Replace 0-5)
      11-96:  AbilitySelect (PeekOwn, PeekOther, BlindSwap, KingLook, KingSwapNo/Yes)
      97-109: SnapDecision (PassSnap, SnapOwn 0-5, SnapOpponent 0-5)
      110-145: SnapMove (SnapOpponentMove)

    Returns:
        Integer decision context (0=StartTurn, 1=PostDraw, 2=SnapDecision,
        3=AbilitySelect, 4=SnapMove), resolved through bridge.DecisionCtx
        rather than hand-written literals so a future engine renumbering
        cannot silently reintroduce the SnapDecision/AbilitySelect swap
        (cambia-1688; see cambia-1484 for the same fix on the test-side map).
    """
    from ..ffi.bridge import DecisionCtx  # noqa: PLC0415

    if legal_mask[0] or legal_mask[1] or legal_mask[2]:
        return int(DecisionCtx.START_TURN)
    if legal_mask[3] or legal_mask[4] or any(legal_mask[5:11]):
        return int(DecisionCtx.POST_DRAW)
    if any(legal_mask[11:97]):
        return int(DecisionCtx.ABILITY_SELECT)
    if legal_mask[97] or any(legal_mask[98:110]):
        return int(DecisionCtx.SNAP_DECISION)
    if any(legal_mask[110:146]):
        return int(DecisionCtx.SNAP_MOVE)
    return int(DecisionCtx.START_TURN)  # fallback


_INTERLEAVED_NETWORK_TYPES = frozenset({"slot_film", "slot_multiply"})


def _encode_ep_pbs(
    agent_state: "GoAgentState",
    decision_context: int,
    drawn_bucket: int,
    network_type: str,
    encoding_layout: str = "auto",
) -> "np.ndarray":
    """Route EP-PBS encoding to interleaved or flat layout.

    Uses interleaved layout when: encoding_layout="interleaved", or
    network_type is in _INTERLEAVED_NETWORK_TYPES (slot_film, slot_multiply).
    All other cases use the flat encode_eppbs() layout.
    """
    if encoding_layout == "flat_dealiased":
        return agent_state.encode_eppbs_dealiased(
            decision_context, drawn_bucket=drawn_bucket
        )
    if encoding_layout == "interleaved" or network_type in _INTERLEAVED_NETWORK_TYPES:
        return agent_state.encode_eppbs_interleaved(
            decision_context, drawn_bucket=drawn_bucket
        )
    return agent_state.encode_eppbs(decision_context, drawn_bucket=drawn_bucket)


def _deep_traverse_go(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],
    updating_player: int,
    network: Optional[AdvantageNetwork],
    iteration: int,
    config: Config,
    advantage_samples: List[ReservoirSample],
    strategy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[queue.Queue],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
    _feat_buf: Optional[torch.Tensor] = None,
    _mask_buf: Optional[torch.Tensor] = None,
) -> Optional[np.ndarray]:
    """
    Recursive External Sampling traversal for Deep CFR using the Go engine backend.

    Uses GoEngine (save/restore instead of undo) and GoAgentState (encode directly).

    At traverser's node: enumerate ALL legal actions, recurse on each, compute exact regrets.
    At opponent's node: sample ONE action from strategy (network), recurse.

    Returns the utility vector (shape (2,) float64) for both players, or None when an
    engine call failed anywhere at or below this node. None is the failure channel,
    not a zero utility: 0 is inside the legal utility range, so a fabricated zero is
    indistinguishable from a genuine draw once it reaches the reservoir. A node that
    returns None has emitted no sample whose target depends on the failed subtree,
    and its caller masks the action leading here out of its own target.
    """
    logger = logging.getLogger(__name__)

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    # Progress update
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    try:
        if engine.is_terminal():
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            util = engine.get_utility().astype(np.float64)
            return util
    except Exception as e_term:
        logger.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return None

    # Depth limit check (system recursion limit)
    if depth >= config.system.recursion_limit:
        logger.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        # A depth rail, not an engine failure: the subtree is cut short, so this
        # returns the truncated-subtree value of 0 exactly as traversal_depth_limit
        # does. It stays counted as an error because reaching it means the recursion
        # ran away, but the samples above it are real.
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Traversal depth cap (0 = unlimited)
    depth_limit = getattr(getattr(config, "deep_cfr", None), "traversal_depth_limit", 0)
    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get legal actions mask
    try:
        legal_mask = engine.legal_actions_mask()
    except Exception as e_legal:
        logger.error(
            "W%d D%d: Error getting legal action mask: %s", worker_id, depth, e_legal
        )
        worker_stats.error_count += 1
        return None

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger.error("W%d D%d: No legal actions but non-terminal!", worker_id, depth)
        worker_stats.error_count += 1
        return None

    # Get decision context directly from Go engine
    current_context = engine.decision_ctx()

    # Get acting player
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return None

    # Get drawn card bucket for POST_DRAW encoding
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode infoset using Go agent
    _encoding_mode = getattr(getattr(config, "deep_cfr", None), "encoding_mode", "legacy")
    _network_type = getattr(getattr(config, "deep_cfr", None), "network_type", "residual")
    _encoding_layout = getattr(
        getattr(config, "deep_cfr", None), "encoding_layout", "auto"
    )
    try:
        if _encoding_mode == "ep_pbs":
            features = _encode_ep_pbs(
                agent_states[player],
                current_context,
                drawn_bucket,
                _network_type,
                _encoding_layout,
            )
        else:
            features = agent_states[player].encode(
                current_context, drawn_bucket=drawn_bucket
            )
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger.error("W%d D%d: Error encoding infoset: %s", worker_id, depth, e_encode)
        worker_stats.error_count += 1
        return None

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy_full = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience - fallback to uniform
            logger.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
    else:
        strategy_full = None

    # Extract local strategy over legal actions
    if strategy_full is not None and len(strategy_full) == NUM_ACTIONS:
        local_strategy = strategy_full[legal_indices].astype(np.float64)
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- External Sampling Logic ---
    if player == updating_player:
        # TRAVERSER'S NODE: enumerate ALL legal actions
        action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

        # Save engine state and clone agent states before enumerating
        try:
            snap = engine.save()
        except Exception as e_save:
            logger.error(
                "W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save
            )
            worker_stats.error_count += 1
            return None

        try:
            agent_clones = [a.clone() for a in agent_states]
        except Exception as e_clone:
            logger.error(
                "W%d D%d: Failed to clone agent states: %s", worker_id, depth, e_clone
            )
            worker_stats.error_count += 1
            engine.free_snapshot(snap)
            return None

        # Per-action outcome. A slot stays False unless the action was applied and
        # its subtree returned a real value, so a failed action is never read as a
        # utility of 0 (which is inside the legal [-1, +1] range and therefore
        # indistinguishable from a genuine draw downstream).
        action_ok = [False] * num_actions

        for i, action_idx in enumerate(legal_indices):
            if i > 0:
                # Restore engine and agent states for next action
                try:
                    engine.restore(snap)
                except Exception as e_restore:
                    logger.error(
                        "W%d D%d: Failed to restore engine: %s",
                        worker_id,
                        depth,
                        e_restore,
                    )
                    worker_stats.error_count += 1
                    break
                agents_restored = True
                for j, a in enumerate(agent_states):
                    a.close()
                    try:
                        agent_states[j] = agent_clones[j].clone()
                    except Exception as e_clone2:
                        logger.error(
                            "W%d D%d: Failed to clone agent %d: %s",
                            worker_id,
                            depth,
                            j,
                            e_clone2,
                        )
                        worker_stats.error_count += 1
                        agents_restored = False
                        break
                if not agents_restored:
                    # The agent states are half-restored; applying the next action
                    # against them would produce a plausible but wrong subtree.
                    break

            try:
                engine.apply_action(int(action_idx))
                engine.update_both(agent_states[0], agent_states[1])
            except Exception as e_apply:
                logger.error(
                    "W%d D%d: Error applying action %d: %s",
                    worker_id,
                    depth,
                    action_idx,
                    e_apply,
                )
                worker_stats.error_count += 1
                continue

            try:
                child_value = _deep_traverse_go(
                    engine,
                    agent_states,
                    updating_player,
                    network,
                    iteration,
                    config,
                    advantage_samples,
                    strategy_samples,
                    depth + 1,
                    worker_stats,
                    progress_queue,
                    worker_id,
                    min_depth_after_bottom_out_tracker,
                    has_bottomed_out_tracker,
                    simulation_nodes,
                    _feat_buf,
                    _mask_buf,
                )
                if child_value is None:
                    # The subtree reported a failure; leave the slot unusable.
                    continue
                action_values[i] = child_value
                action_ok[i] = True
            except Exception as e_recurse:  # JUSTIFIED: worker resilience
                logger.error(
                    "W%d D%d: Recursion error after action %d: %s",
                    worker_id,
                    depth,
                    action_idx,
                    e_recurse,
                    exc_info=True,
                )
                worker_stats.error_count += 1

        # Final restore of engine and agent states
        final_restore_ok = True
        try:
            engine.restore(snap)
        except Exception as e_restore_final:
            logger.error(
                "W%d D%d: Failed final engine restore: %s",
                worker_id,
                depth,
                e_restore_final,
            )
            worker_stats.error_count += 1
            final_restore_ok = False
        for j, a in enumerate(agent_states):
            a.close()
            agent_states[j] = agent_clones[j]  # swap in backup directly
        engine.free_snapshot(snap)

        if not all(action_ok) or not final_restore_ok:
            # v(I) = sum_a sigma(a) v(I a) needs every action's value, so one missing
            # action contaminates the baseline and with it every regret at this node,
            # not just the failed action's. Emit nothing and report the failure so the
            # caller masks the action that leads here out of its own target.
            logger.error(
                "W%d D%d: Discarding advantage sample: %d of %d actions unusable%s.",
                worker_id,
                depth,
                num_actions - sum(action_ok),
                num_actions,
                "" if final_restore_ok else ", final restore failed",
            )
            return None

        # Compute exact counterfactual values
        node_value = local_strategy @ action_values  # shape: (NUM_PLAYERS,)

        # Compute regrets: regret(a) = v(a)[player] - node_value[player]
        regrets = action_values[:, player] - node_value[player]

        # Build full-size regret target vector (NUM_ACTIONS)
        regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for i, action_idx in enumerate(legal_indices):
            regret_target[int(action_idx)] = regrets[i]

        # Store advantage sample
        advantage_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=regret_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

        return node_value

    else:
        # OPPONENT'S NODE: sample ONE action from strategy
        # Store strategy sample for this infoset
        strategy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for i, action_idx in enumerate(legal_indices):
            strategy_target[int(action_idx)] = local_strategy[i]

        strategy_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

        # Sample one action
        if np.sum(local_strategy) > 1e-9:
            try:
                chosen_local_idx = np.random.choice(num_actions, p=local_strategy)
            except ValueError:
                chosen_local_idx = np.random.choice(num_actions)
                worker_stats.warning_count += 1
        else:
            chosen_local_idx = np.random.choice(num_actions)
            worker_stats.warning_count += 1

        chosen_action_idx = int(legal_indices[chosen_local_idx])

        # Save, apply, recurse, restore
        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            snap = engine.save()
        except Exception as e_save:
            logger.error(
                "W%d D%d: Failed to save engine state for opponent: %s",
                worker_id,
                depth,
                e_save,
            )
            worker_stats.error_count += 1
            return None

        try:
            agent_clones = [a.clone() for a in agent_states]
        except Exception as e_clone:
            logger.error(
                "W%d D%d: Failed to clone agent states for opponent: %s",
                worker_id,
                depth,
                e_clone,
            )
            worker_stats.error_count += 1
            engine.free_snapshot(snap)
            return None

        apply_ok = False
        try:
            engine.apply_action(chosen_action_idx)
            engine.update_both(agent_states[0], agent_states[1])
            apply_ok = True
        except Exception as e_apply:
            logger.error(
                "W%d D%d: Error applying sampled action %d: %s",
                worker_id,
                depth,
                chosen_action_idx,
                e_apply,
            )
            worker_stats.error_count += 1

        subtree_ok = False
        if apply_ok:
            try:
                child_value = _deep_traverse_go(
                    engine,
                    agent_states,
                    updating_player,
                    network,
                    iteration,
                    config,
                    advantage_samples,
                    strategy_samples,
                    depth + 1,
                    worker_stats,
                    progress_queue,
                    worker_id,
                    min_depth_after_bottom_out_tracker,
                    has_bottomed_out_tracker,
                    simulation_nodes,
                    _feat_buf,
                    _mask_buf,
                )
                if child_value is not None:
                    node_value = child_value
                    subtree_ok = True
            except Exception as e_recurse:  # JUSTIFIED: worker resilience
                logger.error(
                    "W%d D%d: Recursion error after sampled action %d: %s",
                    worker_id,
                    depth,
                    chosen_action_idx,
                    e_recurse,
                    exc_info=True,
                )
                worker_stats.error_count += 1

        # Restore engine and agent states
        try:
            engine.restore(snap)
        except Exception as e_restore:
            logger.error(
                "W%d D%d: Failed to restore engine after opponent: %s",
                worker_id,
                depth,
                e_restore,
            )
            worker_stats.error_count += 1
            subtree_ok = False
        for j, a in enumerate(agent_states):
            a.close()
            agent_states[j] = agent_clones[j]  # swap in backup directly
        engine.free_snapshot(snap)

        # The strategy sample above was taken before the recursion and its target is
        # sigma at this infoset, so a failure below it does not fabricate anything in
        # that sample; it stays. The utility does not: report the failure upward.
        return node_value if subtree_ok else None


def _deep_traverse_os_go(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],
    updating_player: int,
    network: Optional[AdvantageNetwork],
    iteration: int,
    config: Config,
    advantage_samples: List[ReservoirSample],
    strategy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[queue.Queue],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
    exploration_epsilon: float,
    _feat_buf: Optional[torch.Tensor] = None,
    _mask_buf: Optional[torch.Tensor] = None,
    depth_limit: Optional[int] = None,
    recursion_limit: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Recursive Outcome Sampling traversal for Deep CFR using the Go engine backend.

    Uses GoEngine (save/restore) and GoAgentState (encode directly).

    At traverser nodes the action is sampled from the exploration policy
    q(a|h) = epsilon * uniform + (1-epsilon) * sigma(a|h); at opponent nodes it is
    sampled from sigma itself.

    Because the traverser explores off sigma at every node of the sampled path, the
    raw utility that comes back from the recursion is the value of the epsilon-mixed
    continuation, not of sigma's. The correction is the tail importance ratio
    pi^sigma(h, z) / q(h -> z) (Lanctot et al. 2009, "Monte Carlo Sampling for
    Regret Minimization in Extensive Games"): each node multiplies in
    sigma(a|h) / q(a|h) for the action it sampled and hands the running product to
    its parent. Opponent nodes contribute 1 on the normal path, since they sample
    from sigma. The traverser then weights the sampled utility by
    tail_ratio(child) / q(a|h) rather than by 1 / q(a|h).

    Returns (utility vector of shape (2,) float64, tail importance ratio from this
    node down to the sampled terminal). The ratio is returned unclipped: the
    MAX_IS_WEIGHT bound is applied once, where the weight is consumed, so it does
    not compound down the recursion.

    The utility is None when an engine call failed at or below this node. None is
    the failure channel rather than a zero utility, since 0 is inside the legal
    utility range and a fabricated zero is indistinguishable from a genuine draw
    once it reaches the reservoir.
    """
    # Resolve config values once at the root call; propagated via params on recursion
    if recursion_limit is None:
        recursion_limit = getattr(
            getattr(config, "system", None), "recursion_limit", 10000
        )
    if depth_limit is None:
        depth_limit = getattr(
            getattr(config, "deep_cfr", None), "traversal_depth_limit", 0
        )

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    # Progress update
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    try:
        if engine.is_terminal():
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            util = engine.get_utility().astype(np.float64)
            return util, 1.0
    except Exception as e_term:
        logger.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return None, 1.0

    # Depth limit check (system recursion limit)
    if depth >= recursion_limit:
        logger.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        # A depth rail, not an engine failure: the subtree is cut short, so this
        # returns the truncated-subtree value of 0 exactly as traversal_depth_limit
        # does. It stays counted as an error because reaching it means the recursion
        # ran away, but the samples above it are real.
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    # Traversal depth cap (0 = unlimited)
    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    # Get legal actions mask
    try:
        legal_mask = engine.legal_actions_mask()
    except Exception as e_legal:
        logger.error(
            "W%d D%d: Error getting legal action mask: %s", worker_id, depth, e_legal
        )
        worker_stats.error_count += 1
        return None, 1.0

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger.error("W%d D%d: No legal actions but non-terminal!", worker_id, depth)
        worker_stats.error_count += 1
        return None, 1.0

    # Get decision context directly from Go engine
    current_context = engine.decision_ctx()

    # Get acting player
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return None, 1.0

    # Get drawn card bucket for POST_DRAW encoding
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode infoset using Go agent
    _encoding_mode = getattr(getattr(config, "deep_cfr", None), "encoding_mode", "legacy")
    _network_type = getattr(getattr(config, "deep_cfr", None), "network_type", "residual")
    _encoding_layout = getattr(
        getattr(config, "deep_cfr", None), "encoding_layout", "auto"
    )
    try:
        if _encoding_mode == "ep_pbs":
            features = _encode_ep_pbs(
                agent_states[player],
                current_context,
                drawn_bucket,
                _network_type,
                _encoding_layout,
            )
        else:
            features = agent_states[player].encode(
                current_context, drawn_bucket=drawn_bucket
            )
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger.error("W%d D%d: Error encoding infoset: %s", worker_id, depth, e_encode)
        worker_stats.error_count += 1
        return None, 1.0

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy_full = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience
            logger.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
    else:
        strategy_full = None

    # Extract local strategy over legal actions
    if strategy_full is not None and len(strategy_full) == NUM_ACTIONS:
        local_strategy = strategy_full[legal_indices].astype(np.float64)
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- Outcome Sampling Logic ---
    if player == updating_player:
        # Traverser: use exploration policy q(a) = epsilon * uniform + (1-epsilon) * sigma(a)
        uniform_prob = 1.0 / num_actions
        exploration_policy = (
            exploration_epsilon * uniform_prob
            + (1.0 - exploration_epsilon) * local_strategy
        )
    else:
        # Opponent: sample from pure strategy (no exploration)
        # This ensures unbiased counterfactual value estimation: see Lanctot et al. 2009
        exploration_policy = local_strategy.copy()

    # Normalize
    total_prob = exploration_policy.sum()
    if total_prob > 1e-9:
        exploration_policy /= total_prob
    else:
        exploration_policy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Sample ONE action from exploration policy. q(a|h) has to be the probability the
    # action was actually drawn with, the degenerate fallback included, or the
    # importance ratio is taken against a distribution that was never sampled from.
    try:
        chosen_local_idx = np.random.choice(num_actions, p=exploration_policy)
        sampling_prob = float(exploration_policy[chosen_local_idx])
    except ValueError:
        chosen_local_idx = np.random.choice(num_actions)
        sampling_prob = 1.0 / num_actions
        worker_stats.warning_count += 1

    chosen_action_idx = int(legal_indices[chosen_local_idx])

    # sigma(a|h) / q(a|h) for the action just sampled: this node's factor of the tail
    # importance ratio handed to the parent. Exactly 1.0 at opponent nodes on the
    # normal path, where the exploration policy is sigma itself.
    node_ratio = (
        float(local_strategy[chosen_local_idx]) / sampling_prob
        if sampling_prob > 1e-12
        else 0.0
    )

    # Save engine state and clone agent states
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    try:
        snap = engine.save()
    except Exception as e_save:
        logger.error("W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save)
        worker_stats.error_count += 1
        return None, 1.0

    try:
        agent_clones = [a.clone() for a in agent_states]
    except Exception as e_clone:
        logger.error(
            "W%d D%d: Failed to clone agent states: %s", worker_id, depth, e_clone
        )
        worker_stats.error_count += 1
        engine.free_snapshot(snap)
        return None, 1.0

    apply_ok = False
    try:
        engine.apply_action(chosen_action_idx)
        engine.update_both(agent_states[0], agent_states[1])
        apply_ok = True
    except Exception as e_apply:
        logger.error(
            "W%d D%d: Error applying sampled action %d: %s",
            worker_id,
            depth,
            chosen_action_idx,
            e_apply,
        )
        worker_stats.error_count += 1

    # pi^sigma(h a, z) / q(h a -> z) below the sampled action. Stays 1.0 when the
    # subtree was never entered, in which case node_value is zero anyway.
    child_tail_ratio = 1.0
    subtree_ok = False

    if apply_ok:
        try:
            child_value, child_tail_ratio = _deep_traverse_os_go(
                engine,
                agent_states,
                updating_player,
                network,
                iteration,
                config,
                advantage_samples,
                strategy_samples,
                depth + 1,
                worker_stats,
                progress_queue,
                worker_id,
                min_depth_after_bottom_out_tracker,
                has_bottomed_out_tracker,
                simulation_nodes,
                exploration_epsilon,
                _feat_buf,
                _mask_buf,
                depth_limit,
                recursion_limit,
            )
            if child_value is not None:
                node_value = child_value
                subtree_ok = True
        except Exception as e_recurse:  # JUSTIFIED: worker resilience
            logger.error(
                "W%d D%d: Recursion error after sampled action %d: %s",
                worker_id,
                depth,
                chosen_action_idx,
                e_recurse,
                exc_info=True,
            )
            worker_stats.error_count += 1

    # Restore engine and agent states
    try:
        engine.restore(snap)
    except Exception as e_restore:
        logger.error("W%d D%d: Failed to restore engine: %s", worker_id, depth, e_restore)
        worker_stats.error_count += 1
        subtree_ok = False
    for j, a in enumerate(agent_states):
        a.close()
        agent_states[j] = agent_clones[j]  # swap in backup directly
    engine.free_snapshot(snap)

    # --- Compute IS-corrected regrets and store samples ---
    if player == updating_player:
        # TRAVERSER: compute IS-weighted regrets. A failed subtree leaves node_value
        # at zeros, which is a legal utility rather than a missing one, so the sample
        # is dropped rather than carried into the reservoir.
        sampled_utility = node_value[player]

        if subtree_ok and sampling_prob > 1e-9:
            # u(z) * pi^sigma(h a*, z) / q(h a* -> z) is the unbiased estimate of
            # sigma's continuation value v^sigma(h a*); dividing by q(a*|h) turns it
            # into the unbiased estimate of the counterfactual value of the sampled
            # action. Every unsampled action estimates to zero, and the baseline
            # sigma(a*|h) * utility_estimate estimates v^sigma(h).
            utility_estimate = sampled_utility * _clipped_is_weight(
                child_tail_ratio, sampling_prob
            )

            # Compute IS-corrected regrets (constant baseline from sampled action)
            regrets = np.zeros(num_actions, dtype=np.float64)
            baseline = local_strategy[chosen_local_idx] * utility_estimate
            for a_idx in range(num_actions):
                action_value_estimate = (
                    1.0 if a_idx == chosen_local_idx else 0.0
                ) * utility_estimate
                regrets[a_idx] = action_value_estimate - baseline

            # Build full-size regret target vector (NUM_ACTIONS)
            regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for i, action_idx in enumerate(legal_indices):
                regret_target[int(action_idx)] = regrets[i]

            # Store advantage sample
            advantage_samples.append(
                ReservoirSample(
                    features=features,
                    target=regret_target,
                    action_mask=action_mask.astype(np.bool_),
                    iteration=iteration,
                )
            )
    else:
        # OPPONENT: store strategy sample
        strategy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for i, action_idx in enumerate(legal_indices):
            strategy_target[int(action_idx)] = local_strategy[i]

        strategy_samples.append(
            ReservoirSample(
                features=features,
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

    # The opponent-node strategy sample above targets sigma at this infoset and does
    # not depend on the subtree, so it stands even when the subtree failed. The
    # utility does not: hand the failure to the caller.
    return (node_value if subtree_ok else None), node_ratio * child_tail_ratio


# ---------------------------------------------------------------------------
# N-Player Outcome Sampling traversal (Go engine backend)
# ---------------------------------------------------------------------------


def _deep_traverse_os_go_nplayer(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],  # Length = num_players
    updating_player: int,
    network: Optional[AdvantageNetwork],
    iteration: int,
    config: Config,
    advantage_samples: List[ReservoirSample],
    strategy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[queue.Queue],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
    exploration_epsilon: float,
    num_players: int,
    _feat_buf: Optional[torch.Tensor] = None,
    _mask_buf: Optional[torch.Tensor] = None,
    depth_limit: Optional[int] = None,
    recursion_limit: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Recursive Outcome Sampling traversal for Deep CFR using the Go engine backend
    with N-player support (2-6 players).

    Uses GoEngine (save/restore) and GoAgentState (encode_nplayer directly).

    Sampling and the importance correction match the two-player variant
    (_deep_traverse_os_go): traverser nodes sample from the epsilon-mixed policy,
    every other seat samples from sigma, and each node passes the running tail
    importance ratio pi^sigma(h, z) / q(h -> z) up to its parent so the traverser's
    regret target estimates sigma's counterfactual value rather than the value of
    the epsilon-mixed continuation.

    Returns (utility vector of shape (num_players,) float64, tail importance ratio
    from this node down to the sampled terminal, unclipped). The utility is None
    when an engine call failed at or below this node, for the reason given on
    _deep_traverse_os_go.
    """
    # Resolve config values once at root; propagated via params on recursion
    if recursion_limit is None:
        recursion_limit = getattr(
            getattr(config, "system", None), "recursion_limit", 10000
        )
    if depth_limit is None:
        depth_limit = getattr(
            getattr(config, "deep_cfr", None), "traversal_depth_limit", 0
        )

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    # Progress update
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    try:
        if engine.is_terminal():
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            util = engine.get_nplayer_utility().astype(np.float64)
            return util, 1.0
    except Exception as e_term:
        logger.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return None, 1.0

    # Depth limit check (system recursion limit)
    if depth >= recursion_limit:
        logger.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        # A depth rail, not an engine failure: the subtree is cut short, so this
        # returns the truncated-subtree value of 0 exactly as traversal_depth_limit
        # does. It stays counted as an error because reaching it means the recursion
        # ran away, but the samples above it are real.
        return np.zeros(num_players, dtype=np.float64), 1.0

    # Traversal depth cap (0 = unlimited)
    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(num_players, dtype=np.float64), 1.0

    # Get legal actions mask (N-player 452-action space)
    try:
        legal_mask = engine.nplayer_legal_actions_mask()
    except Exception as e_legal:
        logger.error(
            "W%d D%d: Error getting N-player legal action mask: %s",
            worker_id,
            depth,
            e_legal,
        )
        worker_stats.error_count += 1
        return None, 1.0

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger.error(
            "W%d D%d: No legal N-player actions but non-terminal!", worker_id, depth
        )
        worker_stats.error_count += 1
        return None, 1.0

    # Get decision context directly from Go engine
    current_context = engine.decision_ctx()

    # Get acting player
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return None, 1.0

    # Get drawn card bucket for POST_DRAW encoding
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode infoset using N-player Go agent (580-dim)
    try:
        features = agent_states[player].encode_nplayer(
            current_context, drawn_bucket=drawn_bucket
        )
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger.error(
            "W%d D%d: Error encoding N-player infoset: %s", worker_id, depth, e_encode
        )
        worker_stats.error_count += 1
        return None, 1.0

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy_full = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience
            logger.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
    else:
        strategy_full = None

    # Extract local strategy over legal actions
    if strategy_full is not None and len(strategy_full) == N_PLAYER_NUM_ACTIONS:
        local_strategy = strategy_full[legal_indices].astype(np.float64)
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- Outcome Sampling Logic ---
    if player == updating_player:
        # Traverser: use exploration policy q(a) = epsilon * uniform + (1-epsilon) * sigma(a)
        uniform_prob = 1.0 / num_actions
        exploration_policy = (
            exploration_epsilon * uniform_prob
            + (1.0 - exploration_epsilon) * local_strategy
        )
    else:
        # Opponent: sample from pure strategy (no exploration)
        # This ensures unbiased counterfactual value estimation: see Lanctot et al. 2009
        exploration_policy = local_strategy.copy()

    # Normalize
    total_prob = exploration_policy.sum()
    if total_prob > 1e-9:
        exploration_policy /= total_prob
    else:
        exploration_policy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Sample ONE action from exploration policy. q(a|h) has to be the probability the
    # action was actually drawn with, the degenerate fallback included, or the
    # importance ratio is taken against a distribution that was never sampled from.
    try:
        chosen_local_idx = np.random.choice(num_actions, p=exploration_policy)
        sampling_prob = float(exploration_policy[chosen_local_idx])
    except ValueError:
        chosen_local_idx = np.random.choice(num_actions)
        sampling_prob = 1.0 / num_actions
        worker_stats.warning_count += 1

    chosen_action_idx = int(legal_indices[chosen_local_idx])

    # sigma(a|h) / q(a|h) for the action just sampled: this node's factor of the tail
    # importance ratio handed to the parent. Exactly 1.0 at opponent nodes on the
    # normal path, where the exploration policy is sigma itself.
    node_ratio = (
        float(local_strategy[chosen_local_idx]) / sampling_prob
        if sampling_prob > 1e-12
        else 0.0
    )

    # Save engine state and clone all agent states
    node_value = np.zeros(num_players, dtype=np.float64)

    try:
        snap = engine.save()
    except Exception as e_save:
        logger.error("W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save)
        worker_stats.error_count += 1
        return None, 1.0

    try:
        agent_clones = [a.clone() for a in agent_states]
    except Exception as e_clone:
        logger.error(
            "W%d D%d: Failed to clone N-player agent states: %s",
            worker_id,
            depth,
            e_clone,
        )
        worker_stats.error_count += 1
        engine.free_snapshot(snap)
        return None, 1.0

    apply_ok = False
    try:
        engine.apply_nplayer_action(chosen_action_idx)
        for a in agent_states:
            a.update_nplayer(engine)
        apply_ok = True
    except Exception as e_apply:
        logger.error(
            "W%d D%d: Error applying N-player sampled action %d: %s",
            worker_id,
            depth,
            chosen_action_idx,
            e_apply,
        )
        worker_stats.error_count += 1

    # pi^sigma(h a, z) / q(h a -> z) below the sampled action. Stays 1.0 when the
    # subtree was never entered, in which case node_value is zero anyway.
    child_tail_ratio = 1.0
    subtree_ok = False

    if apply_ok:
        try:
            child_value, child_tail_ratio = _deep_traverse_os_go_nplayer(
                engine,
                agent_states,
                updating_player,
                network,
                iteration,
                config,
                advantage_samples,
                strategy_samples,
                depth + 1,
                worker_stats,
                progress_queue,
                worker_id,
                min_depth_after_bottom_out_tracker,
                has_bottomed_out_tracker,
                simulation_nodes,
                exploration_epsilon,
                num_players,
                _feat_buf,
                _mask_buf,
                depth_limit,
                recursion_limit,
            )
            if child_value is not None:
                node_value = child_value
                subtree_ok = True
        except Exception as e_recurse:  # JUSTIFIED: worker resilience
            logger.error(
                "W%d D%d: Recursion error after N-player sampled action %d: %s",
                worker_id,
                depth,
                chosen_action_idx,
                e_recurse,
                exc_info=True,
            )
            worker_stats.error_count += 1

    # Restore engine and agent states
    try:
        engine.restore(snap)
    except Exception as e_restore:
        logger.error("W%d D%d: Failed to restore engine: %s", worker_id, depth, e_restore)
        worker_stats.error_count += 1
        subtree_ok = False
    for j, a in enumerate(agent_states):
        a.close()
        agent_states[j] = agent_clones[j]  # swap in backup directly
    engine.free_snapshot(snap)

    # --- Compute IS-corrected regrets and store samples ---
    if player == updating_player:
        # TRAVERSER: compute IS-weighted regrets. A failed subtree leaves node_value
        # at zeros, which is a legal utility rather than a missing one, so the sample
        # is dropped rather than carried into the reservoir.
        sampled_utility = node_value[player]

        if subtree_ok and sampling_prob > 1e-9:
            # u(z) * pi^sigma(h a*, z) / q(h a* -> z) is the unbiased estimate of
            # sigma's continuation value v^sigma(h a*); dividing by q(a*|h) turns it
            # into the unbiased estimate of the counterfactual value of the sampled
            # action. Every unsampled action estimates to zero, and the baseline
            # sigma(a*|h) * utility_estimate estimates v^sigma(h).
            utility_estimate = sampled_utility * _clipped_is_weight(
                child_tail_ratio, sampling_prob
            )

            # Compute IS-corrected regrets (constant baseline from sampled action)
            regrets = np.zeros(num_actions, dtype=np.float64)
            baseline = local_strategy[chosen_local_idx] * utility_estimate
            for a_idx in range(num_actions):
                action_value_estimate = (
                    1.0 if a_idx == chosen_local_idx else 0.0
                ) * utility_estimate
                regrets[a_idx] = action_value_estimate - baseline

            # Build full-size regret target vector (N_PLAYER_NUM_ACTIONS)
            regret_target = np.zeros(N_PLAYER_NUM_ACTIONS, dtype=np.float32)
            for i, action_idx in enumerate(legal_indices):
                regret_target[int(action_idx)] = regrets[i]

            # Store advantage sample
            advantage_samples.append(
                ReservoirSample(
                    features=features,
                    target=regret_target,
                    action_mask=action_mask.astype(np.bool_),
                    iteration=iteration,
                )
            )
    else:
        # OPPONENT: store strategy sample
        strategy_target = np.zeros(N_PLAYER_NUM_ACTIONS, dtype=np.float32)
        for i, action_idx in enumerate(legal_indices):
            strategy_target[int(action_idx)] = local_strategy[i]

        strategy_samples.append(
            ReservoirSample(
                features=features,
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

    # The opponent-node strategy sample above targets sigma at this infoset and does
    # not depend on the subtree, so it stands even when the subtree failed. The
    # utility does not: hand the failure to the caller.
    return (node_value if subtree_ok else None), node_ratio * child_tail_ratio


# ---------------------------------------------------------------------------
# ESCHER traversal helpers
# ---------------------------------------------------------------------------


def _value_net_predict(
    value_net: HistoryValueNetwork, features_both: np.ndarray, device: "torch.device"
) -> float:
    """Single-sample value prediction. Returns scalar float."""
    with torch.inference_mode():
        feat_t = torch.from_numpy(features_both).unsqueeze(0).float().to(device)
        return value_net(feat_t).item()


def _value_net_batch_predict(
    value_net: HistoryValueNetwork, features_batch: np.ndarray, device: "torch.device"
) -> np.ndarray:
    """Batched value prediction. features_batch: (N, 444) numpy array. Returns (N,) array."""
    with torch.inference_mode():
        feat_t = torch.from_numpy(features_batch).float().to(device)
        return value_net(feat_t).squeeze(-1).cpu().numpy()


def _escher_traverse_go(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],
    updating_player: int,
    regret_net: Optional[AdvantageNetwork],
    value_net: Optional[HistoryValueNetwork],
    iteration: int,
    config: Config,
    regret_samples: List[ReservoirSample],
    value_samples: List[ReservoirSample],
    policy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[queue.Queue],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
    value_net_device: Optional["torch.device"] = None,
    batch_counterfactuals: bool = True,
    _feat_buf: Optional[torch.Tensor] = None,
    _mask_buf: Optional[torch.Tensor] = None,
    depth_limit: Optional[int] = None,
    recursion_limit: Optional[int] = None,
    _sampled_regret_track: Optional[List[float]] = None,
    _cf_regret_track: Optional[List[float]] = None,
) -> Optional[np.ndarray]:
    """
    ESCHER traversal for Deep CFR using Go engine backend.

    Key differences from OS-MCCFR (_deep_traverse_os_go):
    - Samples directly from strategy (no epsilon mixing / importance weights)
    - Encodes BOTH players for the value network (444-dim concatenation)
    - Stores value samples at ALL non-terminal nodes
    - Computes counterfactual regrets via value network for unsampled actions
    - Stores regret samples only at traverser nodes (not IS-weighted)
    - Stores policy samples only at opponent nodes

    Returns the utility vector (shape (2,) float64) for both players, or None when an
    engine call failed at or below this node. A node that returns None has stored no
    value sample (its target would have been a fabricated 0.0) and no regret sample
    whose baseline or entries rest on the failure; entries for individual actions that
    could not be evaluated are masked out of the regret sample's action mask instead
    of being left at a fabricated regret of zero.
    """
    # Resolve config values once at the root call
    if recursion_limit is None:
        recursion_limit = getattr(
            getattr(config, "system", None), "recursion_limit", 10000
        )
    if depth_limit is None:
        depth_limit = getattr(
            getattr(config, "deep_cfr", None), "traversal_depth_limit", 0
        )
    if value_net_device is None:
        value_net_device = torch.device("cpu")

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

    # Progress update
    if progress_queue and (
        worker_stats.nodes_visited % PROGRESS_UPDATE_NODE_INTERVAL == 0
    ):
        try:
            progress_update = (
                worker_id,
                depth,
                worker_stats.max_depth,
                worker_stats.nodes_visited,
                (
                    int(min_depth_after_bottom_out_tracker[0])
                    if min_depth_after_bottom_out_tracker[0] != float("inf")
                    else 0
                ),
            )
            progress_queue.put_nowait(progress_update)
        except queue.Full:
            pass
        except Exception as pq_e:
            logger.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    try:
        if engine.is_terminal():
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return engine.get_utility().astype(np.float64)
    except Exception as e_term:
        logger.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return None

    # Depth limit checks
    if depth >= recursion_limit:
        logger.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        # A depth rail, not an engine failure: the subtree is cut short, so this
        # returns the truncated-subtree value of 0 exactly as traversal_depth_limit
        # does. It stays counted as an error because reaching it means the recursion
        # ran away, but the samples above it are real.
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get legal actions mask
    try:
        legal_mask = engine.legal_actions_mask()
    except Exception as e_legal:
        logger.error(
            "W%d D%d: Error getting legal action mask: %s", worker_id, depth, e_legal
        )
        worker_stats.error_count += 1
        return None

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger.error("W%d D%d: No legal actions but non-terminal!", worker_id, depth)
        worker_stats.error_count += 1
        return None

    # Get decision context and acting player
    current_context = engine.decision_ctx()
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return None

    # Get drawn card bucket for POST_DRAW encoding (acting player only)
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode acting player's infoset
    _encoding_mode = getattr(getattr(config, "deep_cfr", None), "encoding_mode", "legacy")
    _network_type = getattr(getattr(config, "deep_cfr", None), "network_type", "residual")
    _encoding_layout = getattr(
        getattr(config, "deep_cfr", None), "encoding_layout", "auto"
    )
    try:
        if _encoding_mode == "ep_pbs":
            features_player = _encode_ep_pbs(
                agent_states[player],
                current_context,
                drawn_bucket,
                _network_type,
                _encoding_layout,
            )
        else:
            features_player = agent_states[player].encode(
                current_context, drawn_bucket=drawn_bucket
            )
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger.error(
            "W%d D%d: Error encoding acting player infoset: %s",
            worker_id,
            depth,
            e_encode,
        )
        worker_stats.error_count += 1
        return None

    # Compute strategy from regret network (no epsilon mixing - pure strategy sampling)
    if regret_net is not None:
        try:
            strategy_full = _get_strategy_from_network(
                regret_net, features_player, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger.warning(
                "W%d D%d: Regret net inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience
            logger.warning(
                "W%d D%d: Regret net inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
    else:
        strategy_full = None

    # Extract local strategy over legal actions
    if strategy_full is not None and len(strategy_full) == NUM_ACTIONS:
        local_strategy = strategy_full[legal_indices].astype(np.float64)
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- ESCHER: Sample ONE action directly from strategy (no epsilon mixing) ---
    try:
        chosen_local_idx = np.random.choice(num_actions, p=local_strategy)
    except ValueError:
        chosen_local_idx = np.random.choice(num_actions)
        worker_stats.warning_count += 1

    chosen_action_idx = int(legal_indices[chosen_local_idx])

    # Save engine state and clone agent states before recursion
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    try:
        snap = engine.save()
    except Exception as e_save:
        logger.error("W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save)
        worker_stats.error_count += 1
        return None

    try:
        agent_clones = [a.clone() for a in agent_states]
    except Exception as e_clone:
        logger.error(
            "W%d D%d: Failed to clone agent states: %s", worker_id, depth, e_clone
        )
        worker_stats.error_count += 1
        engine.free_snapshot(snap)
        return None

    apply_ok = False
    try:
        engine.apply_action(chosen_action_idx)
        engine.update_both(agent_states[0], agent_states[1])
        apply_ok = True
    except Exception as e_apply:
        logger.error(
            "W%d D%d: Error applying sampled action %d: %s",
            worker_id,
            depth,
            chosen_action_idx,
            e_apply,
        )
        worker_stats.error_count += 1

    subtree_ok = False
    if apply_ok:
        try:
            child_value = _escher_traverse_go(
                engine,
                agent_states,
                updating_player,
                regret_net,
                value_net,
                iteration,
                config,
                regret_samples,
                value_samples,
                policy_samples,
                depth + 1,
                worker_stats,
                progress_queue,
                worker_id,
                min_depth_after_bottom_out_tracker,
                has_bottomed_out_tracker,
                simulation_nodes,
                value_net_device,
                batch_counterfactuals,
                _feat_buf,
                _mask_buf,
                depth_limit,
                recursion_limit,
                _sampled_regret_track,
                _cf_regret_track,
            )
            if child_value is not None:
                node_value = child_value
                subtree_ok = True
        except Exception as e_recurse:  # JUSTIFIED: worker resilience
            logger.error(
                "W%d D%d: Recursion error after sampled action %d: %s",
                worker_id,
                depth,
                chosen_action_idx,
                e_recurse,
                exc_info=True,
            )
            worker_stats.error_count += 1

    # Restore engine and agent states
    restore_ok = True
    try:
        engine.restore(snap)
    except Exception as e_restore:
        logger.error("W%d D%d: Failed to restore engine: %s", worker_id, depth, e_restore)
        worker_stats.error_count += 1
        subtree_ok = False
        # Everything below reads the engine at this node: the opponent encode, v_hat,
        # and every counterfactual apply. A failed restore leaves the engine at the
        # child, so all of it would describe the wrong state.
        restore_ok = False
    for j, a in enumerate(agent_states):
        a.close()
        agent_states[j] = agent_clones[j]  # swap in backup directly
    engine.free_snapshot(snap)

    # --- Encode BOTH players for value network (444-dim concatenation) ---
    # Non-acting player uses context 0 (START_TURN) and drawn_bucket=-1 (not in POST_DRAW)
    opponent = 1 - player
    opp_context = engine.decision_ctx()
    opp_encode_ok = True
    try:
        if _encoding_mode == "ep_pbs":
            features_opp = _encode_ep_pbs(
                agent_states[opponent], opp_context, -1, _network_type, _encoding_layout
            )
        else:
            features_opp = agent_states[opponent].encode(opp_context, drawn_bucket=-1)
    except Exception as e_enc_opp:
        logger.error(
            "W%d D%d: Error encoding opponent infoset: %s. Dropping value sample and "
            "regret sample at this node.",
            worker_id,
            depth,
            e_enc_opp,
        )
        worker_stats.error_count += 1
        opp_encode_ok = False
        _opp_dim = EP_PBS_INPUT_DIM if _encoding_mode == "ep_pbs" else INPUT_DIM
        features_opp = np.zeros(_opp_dim, dtype=np.float32)

    if player == 0:
        features_both = np.concatenate([features_player, features_opp]).astype(np.float32)
    else:
        features_both = np.concatenate([features_opp, features_player]).astype(np.float32)

    # --- Store value sample at every non-terminal node whose target is real ---
    # Target = realized utility for updating_player from this subtree. A failed
    # subtree leaves node_value at zeros, and zeros over the concatenated features of
    # a failed encode are not an observation of anything, so neither is stored.
    if subtree_ok and opp_encode_ok and restore_ok:
        value_target = float(node_value[updating_player])
        value_samples.append(
            ReservoirSample(
                features=features_both,
                target=np.array([value_target], dtype=np.float32),
                action_mask=np.empty(0, dtype=np.bool_),
                iteration=iteration,
            )
        )

    # --- Per-player sample logic ---
    if player == updating_player:
        # TRAVERSER NODE: compute value-based counterfactual regrets
        # First, get V(h) prediction from value network at current (pre-action) state
        v_hat = 0.0
        # v_hat is the baseline of every entry in the regret vector, so a fabricated
        # one contaminates the whole sample, not one action. It is real when there is
        # no value net (0.0 is then the defined baseline) and when the prediction
        # succeeds on features that were encoded successfully.
        v_hat_ok = opp_encode_ok and restore_ok
        if value_net is not None:
            try:
                v_hat = _value_net_predict(value_net, features_both, value_net_device)
            except Exception as e_vnet:
                logger.error(
                    "W%d D%d: Value net predict failed: %s. Dropping regret sample.",
                    worker_id,
                    depth,
                    e_vnet,
                )
                worker_stats.error_count += 1
                v_hat_ok = False

        # Regret vector: regret[a] = V(h,a) - V(h)
        # For sampled action: V(h, chosen) = actual child_value[player]
        # For other actions: V(h, a) = value_net(features_both after applying a)
        #
        # regret_mask starts at the legal set and loses every action whose entry could
        # not be computed. A masked entry is left at 0 but the loss no longer reads it,
        # so a failed action is not trained toward a regret of exactly zero.
        regret_full = np.zeros(NUM_ACTIONS, dtype=np.float32)
        regret_mask = action_mask.astype(np.bool_).copy()
        if subtree_ok:
            sampled_regret = float(node_value[player]) - v_hat
            regret_full[chosen_action_idx] = sampled_regret
            if _sampled_regret_track is not None:
                _sampled_regret_track.append(abs(sampled_regret))
        else:
            regret_mask[chosen_action_idx] = False

        if num_actions > 1 and value_net is not None:
            if batch_counterfactuals:
                # Collect all counterfactual features first, then batch-predict
                cf_features_list: List[np.ndarray] = []
                cf_action_indices: List[int] = []

                for local_idx, action_idx in enumerate(legal_indices):
                    if int(action_idx) == chosen_action_idx:
                        continue

                    try:
                        cf_snap = engine.save()
                    except Exception as e_cf_save:
                        logger.warning(
                            "W%d D%d: CF save failed for action %d: %s",
                            worker_id,
                            depth,
                            action_idx,
                            e_cf_save,
                        )
                        worker_stats.warning_count += 1
                        regret_mask[int(action_idx)] = False
                        continue

                    try:
                        cf_agent_clones = [a.clone() for a in agent_states]
                    except Exception as e_cf_clone:
                        logger.warning(
                            "W%d D%d: CF agent clone failed: %s",
                            worker_id,
                            depth,
                            e_cf_clone,
                        )
                        worker_stats.warning_count += 1
                        regret_mask[int(action_idx)] = False
                        engine.free_snapshot(cf_snap)
                        continue

                    cf_ok = False
                    cf_feat_both = None
                    try:
                        engine.apply_action(int(action_idx))
                        engine.update_both(agent_states[0], agent_states[1])
                        # Encode both players after applying counterfactual action
                        cf_ctx = engine.decision_ctx()
                        cf_drawn = -1
                        cf_next_player = engine.acting_player()
                        if cf_ctx == 1:  # CtxPostDraw for the next acting player
                            cf_drawn = engine.get_drawn_card_bucket()
                        if _encoding_mode == "ep_pbs":
                            cf_feat_p0 = _encode_ep_pbs(
                                agent_states[0],
                                cf_ctx if cf_next_player == 0 else 0,
                                cf_drawn if cf_next_player == 0 else -1,
                                _network_type,
                                _encoding_layout,
                            )
                            cf_feat_p1 = _encode_ep_pbs(
                                agent_states[1],
                                cf_ctx if cf_next_player == 1 else 0,
                                cf_drawn if cf_next_player == 1 else -1,
                                _network_type,
                                _encoding_layout,
                            )
                        else:
                            cf_feat_p0 = agent_states[0].encode(
                                cf_ctx if cf_next_player == 0 else 0,
                                drawn_bucket=cf_drawn if cf_next_player == 0 else -1,
                            )
                            cf_feat_p1 = agent_states[1].encode(
                                cf_ctx if cf_next_player == 1 else 0,
                                drawn_bucket=cf_drawn if cf_next_player == 1 else -1,
                            )
                        cf_feat_both = np.concatenate([cf_feat_p0, cf_feat_p1]).astype(
                            np.float32
                        )
                        cf_ok = True
                    except Exception as e_cf_apply:
                        logger.warning(
                            "W%d D%d: CF apply/encode failed for action %d: %s",
                            worker_id,
                            depth,
                            action_idx,
                            e_cf_apply,
                        )
                        worker_stats.warning_count += 1

                    # Restore
                    try:
                        engine.restore(cf_snap)
                    except Exception as e_cf_restore:
                        logger.warning(
                            "W%d D%d: CF restore failed: %s",
                            worker_id,
                            depth,
                            e_cf_restore,
                        )
                        worker_stats.error_count += 1
                    for j, a in enumerate(agent_states):
                        a.close()
                        agent_states[j] = cf_agent_clones[j]
                    engine.free_snapshot(cf_snap)

                    if cf_ok and cf_feat_both is not None:
                        cf_features_list.append(cf_feat_both)
                        cf_action_indices.append(int(action_idx))
                    else:
                        regret_mask[int(action_idx)] = False

                # Batch predict all counterfactual values
                if cf_features_list:
                    try:
                        cf_batch = np.stack(cf_features_list)
                        cf_values = _value_net_batch_predict(
                            value_net, cf_batch, value_net_device
                        )
                        for i, cf_action_idx in enumerate(cf_action_indices):
                            regret_full[cf_action_idx] = float(cf_values[i]) - v_hat
                    except Exception as e_batch:
                        logger.warning(
                            "W%d D%d: Batch CF predict failed: %s",
                            worker_id,
                            depth,
                            e_batch,
                        )
                        worker_stats.warning_count += 1
                        for cf_action_idx in cf_action_indices:
                            regret_mask[cf_action_idx] = False
            else:
                # Unbatched: one-at-a-time counterfactual evaluation
                for local_idx, action_idx in enumerate(legal_indices):
                    if int(action_idx) == chosen_action_idx:
                        continue

                    try:
                        cf_snap = engine.save()
                    except Exception as e_cf_save:
                        logger.warning(
                            "W%d D%d: CF save failed for action %d: %s",
                            worker_id,
                            depth,
                            action_idx,
                            e_cf_save,
                        )
                        worker_stats.warning_count += 1
                        regret_mask[int(action_idx)] = False
                        continue

                    try:
                        cf_agent_clones = [a.clone() for a in agent_states]
                    except Exception as e_cf_clone:
                        logger.warning(
                            "W%d D%d: CF agent clone failed: %s",
                            worker_id,
                            depth,
                            e_cf_clone,
                        )
                        worker_stats.warning_count += 1
                        regret_mask[int(action_idx)] = False
                        engine.free_snapshot(cf_snap)
                        continue

                    # Defaulting to v_hat would write a regret of exactly 0 for this
                    # action, which is a fabricated estimate, not a missing one.
                    cf_val = 0.0
                    cf_val_ok = False
                    try:
                        engine.apply_action(int(action_idx))
                        engine.update_both(agent_states[0], agent_states[1])
                        cf_ctx = engine.decision_ctx()
                        cf_drawn = -1
                        cf_next_player = engine.acting_player()
                        if cf_ctx == 1:
                            cf_drawn = engine.get_drawn_card_bucket()
                        if _encoding_mode == "ep_pbs":
                            cf_feat_p0 = _encode_ep_pbs(
                                agent_states[0],
                                cf_ctx if cf_next_player == 0 else 0,
                                cf_drawn if cf_next_player == 0 else -1,
                                _network_type,
                                _encoding_layout,
                            )
                            cf_feat_p1 = _encode_ep_pbs(
                                agent_states[1],
                                cf_ctx if cf_next_player == 1 else 0,
                                cf_drawn if cf_next_player == 1 else -1,
                                _network_type,
                                _encoding_layout,
                            )
                        else:
                            cf_feat_p0 = agent_states[0].encode(
                                cf_ctx if cf_next_player == 0 else 0,
                                drawn_bucket=cf_drawn if cf_next_player == 0 else -1,
                            )
                            cf_feat_p1 = agent_states[1].encode(
                                cf_ctx if cf_next_player == 1 else 0,
                                drawn_bucket=cf_drawn if cf_next_player == 1 else -1,
                            )
                        cf_feat_both = np.concatenate([cf_feat_p0, cf_feat_p1]).astype(
                            np.float32
                        )
                        cf_val = _value_net_predict(
                            value_net, cf_feat_both, value_net_device
                        )
                        cf_val_ok = True
                    except Exception as e_cf_apply:
                        logger.warning(
                            "W%d D%d: CF apply/encode/predict failed for action %d: %s",
                            worker_id,
                            depth,
                            action_idx,
                            e_cf_apply,
                        )
                        worker_stats.warning_count += 1

                    # Restore
                    try:
                        engine.restore(cf_snap)
                    except Exception as e_cf_restore:
                        logger.warning(
                            "W%d D%d: CF restore failed: %s",
                            worker_id,
                            depth,
                            e_cf_restore,
                        )
                        worker_stats.error_count += 1
                    for j, a in enumerate(agent_states):
                        a.close()
                        agent_states[j] = cf_agent_clones[j]
                    engine.free_snapshot(cf_snap)

                    if cf_val_ok:
                        regret_full[int(action_idx)] = cf_val - v_hat
                    else:
                        regret_mask[int(action_idx)] = False

        # Track CF regret magnitudes for telemetry, over the entries that exist
        if _cf_regret_track is not None and num_actions > 1:
            for _cf_idx in legal_indices:
                if int(_cf_idx) != chosen_action_idx and regret_mask[int(_cf_idx)]:
                    _cf_regret_track.append(abs(float(regret_full[int(_cf_idx)])))

        # Store regret sample. A contaminated baseline takes the whole vector with
        # it, and a vector with no surviving action carries no signal.
        if v_hat_ok and bool(regret_mask.any()):
            regret_samples.append(
                ReservoirSample(
                    features=features_player.astype(np.float32),
                    target=regret_full,
                    action_mask=regret_mask,
                    iteration=iteration,
                )
            )
        else:
            logger.error(
                "W%d D%d: Discarding ESCHER regret sample (baseline usable: %s, "
                "actions surviving: %d).",
                worker_id,
                depth,
                v_hat_ok,
                int(regret_mask.sum()),
            )

    else:
        # OPPONENT NODE: store policy sample (average strategy)
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for i, action_idx in enumerate(legal_indices):
            policy_target[int(action_idx)] = local_strategy[i]

        policy_samples.append(
            ReservoirSample(
                features=features_player.astype(np.float32),
                target=policy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

    # The policy sample targets sigma at this infoset and does not depend on the
    # subtree, so it stands regardless. The utility does not.
    return node_value if subtree_ok else None


def run_deep_cfr_worker(
    worker_args: Tuple[
        int,  # iteration
        Config,
        Optional[Dict[str, Any]],  # network_weights (serialized state_dict)
        Dict[str, int],  # network_config
        Optional[queue.Queue],  # progress_queue
        Optional[Any],  # archive_queue
        int,  # worker_id
        str,  # run_log_dir
        str,  # run_timestamp
    ],
    file_handler_override: Optional[logging.Handler] = None,
) -> Optional[DeepCFRWorkerResult]:
    """
    Top-level function executed by each Deep CFR worker process.
    Sets up logging, initializes game, runs external sampling traversal,
    returns advantage and strategy samples.
    """
    logger_instance: Optional[logging.Logger] = None
    worker_stats = WorkerStats()
    (
        iteration,
        config,
        network_weights_serialized,
        network_config,
        progress_queue,
        archive_queue,
        worker_id,
        run_log_dir,
        run_timestamp,
    ) = worker_args

    worker_stats.worker_id = worker_id
    simulation_nodes_this_sim: List[SimulationNodeData] = []
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    value_samples: List[ReservoirSample] = []
    # ESCHER regret telemetry accumulation (populated only during ESCHER traversal)
    _escher_sampled_track: List[float] = []
    _escher_cf_track: List[float] = []

    # --- Logging setup (same pattern as tabular worker) ---
    worker_root_logger = logging.getLogger()
    try:
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except Exception:
                    pass

        # Set root logger level to match the worker's configured level
        # (avoids creating expensive LogRecords that just get filtered by handler)
        worker_log_level_str = config.logging.get_worker_log_level(
            worker_id, config.cfr_training.num_workers
        )
        effective_level = getattr(logging, worker_log_level_str.upper(), logging.WARNING)
        worker_root_logger.setLevel(effective_level)
        null_handler = logging.NullHandler()
        worker_root_logger.addHandler(null_handler)
        worker_root_logger.propagate = False

        if file_handler_override is not None:
            # Reuse pre-created handler: avoids glob.glob() on every traversal.
            file_handler_override.setLevel(effective_level)
            worker_root_logger.addHandler(file_handler_override)
        else:
            # Standalone call (multiprocessing path or tests): create handler here.
            worker_log_dir = os.path.join(run_log_dir, f"w{worker_id}")
            os.makedirs(worker_log_dir, exist_ok=True)
            log_pattern = os.path.join(
                worker_log_dir,
                f"{config.logging.log_file_prefix}_run_{run_timestamp}-w{worker_id}",
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
            )
            file_handler = SerialRotatingFileHandler(
                filename_pattern=log_pattern,
                maxBytes=config.logging.log_max_bytes,
                backupCount=config.logging.log_backup_count,
                encoding="utf-8",
                archive_queue=archive_queue,
                logging_config_snapshot=config.logging,
            )
            file_handler.setLevel(effective_level)
            file_handler.setFormatter(formatter)
            worker_root_logger.addHandler(file_handler)

        logger_instance = logging.getLogger(__name__)
        logger_instance.info(
            "Deep CFR Worker %d logging initialized (dir: %s).", worker_id, run_log_dir
        )
    except Exception as log_setup_e:
        print(
            f"!!! CRITICAL Error setting up logging W{worker_id}: {log_setup_e} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        worker_stats.error_count += 1
        if not worker_root_logger.hasHandlers():
            worker_root_logger.addHandler(logging.NullHandler())
        logger_instance = logging.getLogger(__name__)

    # --- Main simulation logic ---
    try:
        # Build advantage network once for this worker, load weights
        advantage_network: Optional[AdvantageNetwork] = None
        if network_weights_serialized is not None:
            try:
                input_dim = network_config.get("input_dim", INPUT_DIM)
                hidden_dim = network_config.get("hidden_dim", 256)
                output_dim = network_config.get("output_dim", NUM_ACTIONS)

                advantage_network = build_advantage_network(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    validate_inputs=network_config.get("validate_inputs", True),
                    use_residual=network_config.get("use_residual", False),
                    num_hidden_layers=network_config.get("num_hidden_layers", 2),
                    network_type=network_config.get("network_type", "residual"),
                    use_pos_embed=network_config.get("use_pos_embed", True),
                )
                weights_tensors = {
                    k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in network_weights_serialized.items()
                    if k != "__value_net__"  # Value net weights stored separately
                }
                advantage_network.load_state_dict(weights_tensors)
                advantage_network.eval()
                try:
                    example_features = torch.zeros(1, input_dim, dtype=torch.float32)
                    example_mask = torch.ones(1, output_dim, dtype=torch.bool)
                    advantage_network = torch.jit.trace(
                        advantage_network, (example_features, example_mask)
                    )
                    # logger.debug("TorchScript tracing succeeded for worker inference")
                except Exception as e:
                    if logger_instance:
                        logger_instance.warning(
                            "TorchScript tracing failed, using eager mode: %s", e
                        )
            except Exception as e_deserialize:
                if logger_instance:
                    logger_instance.warning(
                        "W%d: Failed to build/load network: %s. Using uniform strategy.",
                        worker_id,
                        e_deserialize,
                    )
                worker_stats.warning_count += 1
                advantage_network = None

        # Pre-allocate inference buffers once per worker (reused across all traversal calls)
        _feat_buf: Optional[torch.Tensor] = None
        _mask_buf: Optional[torch.Tensor] = None
        if advantage_network is not None:
            _worker_input_dim = network_config.get("input_dim", INPUT_DIM)
            _feat_buf = torch.zeros(1, _worker_input_dim, dtype=torch.float32)
            _mask_buf = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)

        # Build value network for ESCHER (if applicable)
        deep_cfr_cfg = getattr(config, "deep_cfr", None)
        traversal_method = getattr(deep_cfr_cfg, "traversal_method", "outcome")
        value_network: Optional[HistoryValueNetwork] = None
        value_net_device = torch.device("cpu")
        batch_counterfactuals = getattr(deep_cfr_cfg, "batch_counterfactuals", True)

        if traversal_method == "escher" and network_weights_serialized is not None:
            value_weights_serialized = network_weights_serialized.get("__value_net__")
            if value_weights_serialized is not None:
                try:
                    value_hidden_dim = network_config.get("value_hidden_dim", 512)
                    _vnet_base_dim = network_config.get("input_dim", INPUT_DIM)
                    value_network = HistoryValueNetwork(
                        input_dim=_vnet_base_dim * 2,
                        hidden_dim=value_hidden_dim,
                        validate_inputs=network_config.get("validate_inputs", True),
                    )
                    value_weights_tensors = {
                        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                        for k, v in value_weights_serialized.items()
                    }
                    value_network.load_state_dict(value_weights_tensors)
                    value_network.eval()
                except Exception as e_vnet:
                    if logger_instance:
                        logger_instance.warning(
                            "W%d: Failed to build/load value network: %s. Using zero estimates.",
                            worker_id,
                            e_vnet,
                        )
                    worker_stats.warning_count += 1
                    value_network = None

        # Alternate updating player each iteration
        updating_player = iteration % NUM_PLAYERS
        min_depth_after_bottom_out_tracker = [float("inf")]
        has_bottomed_out_tracker = [False]

        from ..ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415

        go_engine = None
        go_agents = []
        _profiling = getattr(
            getattr(config, "deep_cfr", None), "enable_traversal_profiling", False
        )
        _setup_t0 = time.time() if _profiling else 0.0
        try:
            go_engine = GoEngine(house_rules=config.cambia_rules)
            go_agents = [
                GoAgentState(
                    go_engine,
                    pid,
                    config.agent_params.memory_level,
                    config.agent_params.time_decay_turns,
                )
                for pid in range(NUM_PLAYERS)
            ]
        except Exception as go_init_e:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Failed Go engine/agent init: %s",
                    worker_id,
                    iteration,
                    go_init_e,
                    exc_info=True,
                )
            worker_stats.error_count += 1
            if go_engine is not None:
                go_engine.close()
            return DeepCFRWorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
            )
        if _profiling:
            _setup_elapsed = time.time() - _setup_t0
            if logger_instance:
                logger_instance.debug(
                    "W%d Iter %d: engine_setup=%.4fs",
                    worker_id,
                    iteration,
                    _setup_elapsed,
                )

        _traversal_t0 = time.time() if _profiling else 0.0
        try:
            sampling_method = getattr(config.deep_cfr, "sampling_method", "external")
            if traversal_method == "escher":
                depth_limit = getattr(
                    getattr(config, "deep_cfr", None), "traversal_depth_limit", 0
                )
                recursion_limit = getattr(
                    getattr(config, "system", None), "recursion_limit", 10000
                )
                final_utility_value = _escher_traverse_go(
                    engine=go_engine,
                    agent_states=go_agents,
                    updating_player=updating_player,
                    regret_net=advantage_network,
                    value_net=value_network,
                    iteration=iteration,
                    config=config,
                    regret_samples=advantage_samples,
                    value_samples=value_samples,
                    policy_samples=strategy_samples,
                    depth=0,
                    worker_stats=worker_stats,
                    progress_queue=progress_queue,
                    worker_id=worker_id,
                    min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
                    has_bottomed_out_tracker=has_bottomed_out_tracker,
                    simulation_nodes=simulation_nodes_this_sim,
                    value_net_device=value_net_device,
                    batch_counterfactuals=batch_counterfactuals,
                    _feat_buf=_feat_buf,
                    _mask_buf=_mask_buf,
                    depth_limit=depth_limit,
                    recursion_limit=recursion_limit,
                    _sampled_regret_track=_escher_sampled_track,
                    _cf_regret_track=_escher_cf_track,
                )
            elif sampling_method == "outcome":
                exploration_epsilon = getattr(config.deep_cfr, "exploration_epsilon", 0.6)
                depth_limit = getattr(
                    getattr(config, "deep_cfr", None), "traversal_depth_limit", 0
                )
                recursion_limit = getattr(
                    getattr(config, "system", None), "recursion_limit", 10000
                )
                final_utility_value, _root_tail_ratio = _deep_traverse_os_go(
                    engine=go_engine,
                    agent_states=go_agents,
                    updating_player=updating_player,
                    network=advantage_network,
                    iteration=iteration,
                    config=config,
                    advantage_samples=advantage_samples,
                    strategy_samples=strategy_samples,
                    depth=0,
                    worker_stats=worker_stats,
                    progress_queue=progress_queue,
                    worker_id=worker_id,
                    min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
                    has_bottomed_out_tracker=has_bottomed_out_tracker,
                    simulation_nodes=simulation_nodes_this_sim,
                    exploration_epsilon=exploration_epsilon,
                    _feat_buf=_feat_buf,
                    _mask_buf=_mask_buf,
                    depth_limit=depth_limit,
                    recursion_limit=recursion_limit,
                )
            else:
                final_utility_value = _deep_traverse_go(
                    engine=go_engine,
                    agent_states=go_agents,
                    updating_player=updating_player,
                    network=advantage_network,
                    iteration=iteration,
                    config=config,
                    advantage_samples=advantage_samples,
                    strategy_samples=strategy_samples,
                    depth=0,
                    worker_stats=worker_stats,
                    progress_queue=progress_queue,
                    worker_id=worker_id,
                    min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
                    has_bottomed_out_tracker=has_bottomed_out_tracker,
                    simulation_nodes=simulation_nodes_this_sim,
                    _feat_buf=_feat_buf,
                    _mask_buf=_mask_buf,
                )
        finally:
            _cleanup_t0 = time.time() if _profiling else 0.0
            for a in go_agents:
                a.close()
            go_engine.close()
            if _profiling:
                _cleanup_elapsed = time.time() - _cleanup_t0
                _traversal_elapsed = time.time() - _traversal_t0
                if logger_instance:
                    logger_instance.debug(
                        "W%d Iter %d: traversal=%.4fs cleanup=%.4fs nodes=%d max_depth=%d",
                        worker_id,
                        iteration,
                        _traversal_elapsed,
                        _cleanup_elapsed,
                        worker_stats.nodes_visited,
                        worker_stats.max_depth,
                    )

        # Release network and inference buffers to reduce memory pressure
        # in the reused subprocess (pipeline_training with max_tasks_per_child>1).
        del advantage_network, value_network, _feat_buf, _mask_buf

        if final_utility_value is None or len(final_utility_value) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Traversal returned invalid utility: %s.",
                    worker_id,
                    iteration,
                    final_utility_value,
                )
            worker_stats.error_count += 1
            final_utility_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        worker_stats.min_depth_after_bottom_out = (
            int(min_depth_after_bottom_out_tracker[0])
            if min_depth_after_bottom_out_tracker[0] != float("inf")
            else 0
        )

        if logger_instance:
            logger_instance.info(
                "W%d Iter %d: Traversal complete. Adv samples: %d, Strat samples: %d, "
                "Val samples: %d, Nodes: %d",
                worker_id,
                iteration,
                len(advantage_samples),
                len(strategy_samples),
                len(value_samples),
                worker_stats.nodes_visited,
            )

        _escher_sampled_mag = (
            float(np.mean(_escher_sampled_track)) if _escher_sampled_track else None
        )
        _escher_cf_mag = float(np.mean(_escher_cf_track)) if _escher_cf_track else None
        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            value_samples=value_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=final_utility_value.tolist(),
            escher_sampled_regret_mag=_escher_sampled_mag,
            escher_cf_regret_mag=_escher_cf_mag,
        )

    except KeyboardInterrupt:
        if logger_instance:
            logger_instance.warning(
                "W%d Iter %d received KeyboardInterrupt.", worker_id, iteration
            )
        worker_stats.error_count += 1
        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            value_samples=value_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
        )
    except (
        Exception
    ) as e_inner:  # JUSTIFIED: worker resilience - top-level worker catch to prevent pool crash
        worker_stats.error_count += 1
        if logger_instance:
            logger_instance.critical(
                "!!! Unhandled Error W%d Iter %d: %s !!!",
                worker_id,
                iteration,
                e_inner,
                exc_info=True,
            )
        print(
            f"!!! FATAL DEEP WORKER ERROR W{worker_id} Iter {iteration}: {e_inner} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            value_samples=value_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
        )
    finally:
        if logger_instance:
            for handler in logger_instance.handlers[:]:
                if hasattr(handler, "flush"):
                    try:
                        handler.flush()
                    except Exception:
                        pass
                if hasattr(handler, "close"):
                    try:
                        handler.close()
                    except Exception:
                        pass
