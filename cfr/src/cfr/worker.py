"""
src/cfr/worker.py

Implements the worker process logic for CFR+ training using outcome-sampling
Monte Carlo CFR (OS-MCCFR, Lanctot et al. 2009). Each worker plays one sampled
trajectory against a strategy snapshot provided by the main process, accumulates
local regret and average-strategy updates along it, and returns them.

The module described itself as external sampling (ES-MCCFR) through cambia-719
while sampling a single action at every node, the updating player's included.
That is outcome sampling, and it is what the estimator below implements: an
epsilon-mixed behaviour policy at the traverser's nodes, the full 1/q correction
carried down the trajectory, and counterfactual reaches threaded so the regret
of an infoset is weighted by the opponents' probability of reaching it.

Engine (cambia-1782)
--------------------
The rules run on the Go engine, through ``src.cfr.br_state.GoBrState`` -- the
same substrate the tabular best-response search moved onto in cambia-1428, so
training and exploitability read one implementation of the rules. Only the
rules moved. The policy table is keyed by ``InfosetKey`` built from the Python
``AgentState`` belief machinery, so the belief layer stays exactly where it
was and tables written before the port stay addressable; ``GoBrState`` hands
that unchanged machinery the observation stream it always consumed, read off a
``GoEngine``.

The Python engine's ``apply_action`` returned an undo callable. The Go engine
has none, so the traversal brackets each sampled action with
``GoBrState.checkpoint`` / ``rewind`` / ``release``, which restores the same
state the undo did.

``_create_observation`` and ``_filter_observation`` below are the production
observation builders that several other lanes import. They are duck-typed on
whatever game state they are handed rather than importing one, so this module
carries no dependency on the retiring Python engine.
"""

import copy
import logging
import multiprocessing
import os
import queue
import random
import sys
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypeAlias, Union, Any

import numpy as np

from ..agent_state import AgentObservation, AgentState
from ..config import CfrPlusParamsConfig, Config
from .exceptions import (
    GameStateError,
    AgentStateError,
    ObservationUpdateError,
    EncodingError,
    InfosetEncodingError,
    ActionEncodingError,
    TraversalError,
)

from ..card import Card
from ..constants import (
    NUM_PLAYERS,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionDrawDiscard,
    ActionDrawStockpile,
    ActionPassSnap,
    ActionReplace,
    ActionSnapOpponent,
    ActionSnapOwn,
    GameAction,
)
from .br_state import Checkpoint, DealSpec, GoBrState
from ..serial_rotating_handler import SerialRotatingFileHandler
from ..utils import (
    InfosetKey,
    LocalReachProbUpdateDict,
    LocalRegretUpdateDict,
    LocalStrategyUpdateDict,
    WorkerResult,
    WorkerStats,
    get_rm_plus_strategy,
    normalize_probabilities,
    SimulationNodeData,
)

# Type Aliases
RegretSnapshotDict: TypeAlias = Dict[InfosetKey, np.ndarray]
ProgressQueueWorker: TypeAlias = queue.Queue
ArchiveQueueWorker: TypeAlias = Union[queue.Queue, "multiprocessing.Queue"]

# Tuned progress update interval
PROGRESS_UPDATE_NODE_INTERVAL = 2500

logger = logging.getLogger(__name__)  # Get logger instance at module level


def _serialize_action_for_history(action: GameAction) -> Any:
    """Simple serialization for history log."""
    if hasattr(action, "_asdict"):
        # Attempt to serialize card objects within the action dict if possible
        action_dict = action._asdict()
        serialized_dict = {}
        for k, v in action_dict.items():
            # Use the imported Card class directly for the check
            if isinstance(v, Card):
                serialized_dict[k] = str(v)
            else:
                # Basic serialization for other types
                serialized_dict[k] = (
                    repr(v) if not isinstance(v, (int, float, bool, str)) else v
                )
        return {type(action).__name__: serialized_dict}  # Include type name
    elif isinstance(action, Card):  # Check against Card directly
        return str(action)
    elif action is None:
        return None
    else:  # Fallback for simple actions or unexpected types
        return type(action).__name__


def averaging_weight(iteration: int, params: CfrPlusParamsConfig) -> float:
    """The CFR+ delayed linear averaging weight for a 0-based iteration index.

    Tammelin et al. (2015) weight the average-strategy accumulation by
    ``max(0, t - d)`` for 1-based iteration ``t`` and delay ``d``, discarding the
    early iterations whose strategies are still arbitrary. The weight belongs to
    that accumulation alone: regret updates in CFR+ are unweighted, and applying
    the delay to them instead zeroes every update until the delay elapses
    (cambia-718).
    """
    if not params.weighted_averaging_enabled:
        return 1.0
    return float(max(0, (iteration + 1) - params.averaging_delay))


def sampling_policy(strategy: np.ndarray, epsilon: float) -> np.ndarray:
    """The epsilon-mixed behaviour policy outcome sampling explores with.

    Sampling from the current strategy alone can never revisit an action regret
    matching has driven to zero, so one unlucky early sample freezes that action
    out for the rest of the run. Mixing epsilon of a uniform policy into the
    sampling distribution keeps every legal action reachable; the 1/q correction
    in the regret estimate removes the bias this introduces (Lanctot et al.
    2009). The strategy itself is untouched, only what gets sampled from it.
    """
    num_actions = len(strategy)
    if num_actions == 0:
        return np.array([], dtype=np.float64)
    probs = np.asarray(strategy, dtype=np.float64)
    if epsilon <= 0.0:
        return probs
    uniform = np.ones(num_actions, dtype=np.float64) / num_actions
    return (1.0 - epsilon) * probs + epsilon * uniform


def sampled_regrets(
    strategy: np.ndarray, chosen_index: int, action_value: float
) -> np.ndarray:
    """Instantaneous sampled counterfactual regrets at one infoset.

    Outcome sampling estimates the value of the sampled action alone, so the
    estimate is ``action_value`` for that action and zero for every other. The
    node value is the strategy's own expectation over those estimates, which is
    ``sigma[a*] * action_value``, and each action's regret is that baseline
    subtracted from its estimate. The regrets are therefore orthogonal to the
    strategy: ``sum_a sigma(a) r(a) == 0``.

    Subtracting a per-action ``sigma[a] * action_value`` instead, as the code
    did before cambia-719, understates every unsampled action's regret by a
    factor that varies with the strategy, and the RM+ floor then rectifies the
    resulting drift asymmetrically.
    """
    chosen_prob = float(strategy[chosen_index])
    regrets = np.full(len(strategy), -chosen_prob * action_value, dtype=np.float64)
    regrets[chosen_index] = action_value * (1.0 - chosen_prob)
    return regrets


def suffix_reach_ratio(
    child_ratio: float, own_action_prob: float, sampling_prob: float
) -> float:
    """The suffix factor a node hands its caller, composed one level up.

    The counterfactual value of a sampled action is the leaf utility weighted by
    the updating player's own probability of playing the suffix and divided by
    the behaviour policy's probability of sampling it. Both are products over
    the same nodes, so the recursion carries their ratio rather than either
    alone: a terminal returns 1, and each node multiplies in its own action
    probability (1 for a node the updating player does not own) over the
    probability the sampler drew that action.

    Dividing by ``sampling_prob`` is the part that is easy to lose. Without it
    the trajectory's prefix is corrected for and its suffix is not, which biases
    every value estimated above a node that had more than one continuation.
    """
    if sampling_prob <= 0.0:
        return 0.0
    return child_ratio * own_action_prob / sampling_prob


def _traverse_game_for_worker(
    game_state: GoBrState,
    agent_states: List[AgentState],
    my_reach: float,
    opp_reach: float,
    sample_reach: float,
    iteration: int,
    updating_player: int,  # The player whose regret/strategy is being updated this iteration
    averaging_weight: float,  # CFR+ delayed averaging weight; strategy sum only
    regret_sum_snapshot: RegretSnapshotDict,
    config: Config,
    local_regret_updates: LocalRegretUpdateDict,
    local_strategy_sum_updates: LocalStrategyUpdateDict,
    local_reach_prob_updates: LocalReachProbUpdateDict,
    depth: int,
    worker_stats: WorkerStats,
    progress_queue: Optional[ProgressQueueWorker],
    worker_id: int,
    min_depth_after_bottom_out_tracker: List[float],
    has_bottomed_out_tracker: List[bool],
    simulation_nodes: List[SimulationNodeData],
) -> Tuple[np.ndarray, float]:
    """One node of the outcome-sampling traversal (Lanctot et al. 2009).

    Samples a single action and recurses on it, then updates the average
    strategy at every node and the sampled counterfactual regrets at the
    updating player's nodes.

    Three reach probabilities are threaded down, all measured from the root to
    this node: ``my_reach`` is the updating player's own probability of playing
    here, ``opp_reach`` the probability everyone else does, and ``sample_reach``
    the probability the behaviour policy had of sampling this trajectory prefix.
    Dividing by ``sample_reach`` is what makes the single-trajectory estimate
    unbiased; before cambia-719 all three were held at the constant 1, so the
    correction was inert and no counterfactual weighting was applied.

    Returns the utility vector of the sampled leaf together with the suffix
    factor described in ``suffix_reach_ratio``: the updating player's own
    probability of playing from this node to that leaf, over the behaviour
    policy's probability of having sampled it. The caller needs that ratio to
    weight the sampled action's counterfactual value.
    """
    # Use the module-level logger
    logger_traverse = logging.getLogger(__name__)

    worker_stats.nodes_visited += 1
    worker_stats.max_depth = max(worker_stats.max_depth, depth)

    if has_bottomed_out_tracker[0]:
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )

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
            pass  # Ignore if queue is full
        except Exception as pq_e:
            logger_traverse.error(
                "W%d D%d: Error putting progress on queue: %s", worker_id, depth, pq_e
            )
            worker_stats.error_count += 1

    if game_state.is_terminal():
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return (
            np.array(
                [game_state.utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            ),
            1.0,
        )

    if depth >= config.system.recursion_limit:
        logger_traverse.error(
            "W%d D%d: Max recursion depth reached. Returning 0.", worker_id, depth
        )
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    # Determine context. engine/legal.go's DecisionCtx partitions states exactly
    # as the Python pending-record walk did and its values are numerically
    # identical to DecisionContext's, so this is the mapping rather than a
    # re-derivation -- and it is total, so there is no unknown-pending fallback
    # left to warn about.
    current_context = game_state.decision_context()

    player = game_state.acting_player()
    if player == -1:
        logger_traverse.error(
            "W%d D%d: Could not determine acting player. State: %s Context: %s",
            worker_id,
            depth,
            game_state,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    try:
        current_agent_state = agent_states[player]
        if not callable(current_agent_state.get_infoset_key):
            logger_traverse.error(
                "W%d D%d: Agent state P%d missing get_infoset_key. State: %s",
                worker_id,
                depth,
                player,
                current_agent_state,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

        base_infoset_tuple = current_agent_state.get_infoset_key()
        if not isinstance(base_infoset_tuple, tuple):
            logger_traverse.error(
                "W%d D%d: get_infoset_key did not return a tuple for P%d. Got %s.",
                worker_id,
                depth,
                player,
                type(base_infoset_tuple).__name__,
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

        infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        infoset_key_tuple = infoset_key.astuple()
    except (AgentStateError, EncodingError, InfosetEncodingError) as e_key:
        logger_traverse.warning(
            "W%d D%d: Agent/encoding error getting infoset key P%d: %s. Context: %s",
            worker_id,
            depth,
            player,
            e_key,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0
    except (
        Exception
    ) as e_key:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_traverse.error(
            "W%d D%d: Error getting infoset key P%d: %s. AgentState: %s Context: %s",
            worker_id,
            depth,
            player,
            e_key,
            current_agent_state,
            current_context.name,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    try:
        # Already sorted by repr, which is what indexes the stored strategy
        # vector; each action carries the engine index that applies it.
        legal_pairs = game_state.legal_actions()
        legal_actions = [action for _, action in legal_pairs]
    except GameStateError as e_legal:
        logger_traverse.warning(
            "W%d D%d: Game state error getting legal actions P%d: %s. Context: %s",
            worker_id,
            depth,
            player,
            e_legal,
            current_context.name,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0
    except (
        Exception
    ) as e_legal:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_traverse.error(
            "W%d D%d: Error getting legal actions P%d: %s. Prefix: %s Context: %s",
            worker_id,
            depth,
            player,
            e_legal,
            game_state.action_prefix,
            current_context.name,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0

    num_actions = len(legal_actions)

    if num_actions == 0:
        if not game_state.is_terminal():
            logger_traverse.error(
                "W%d D%d: No legal actions P%d, but state non-terminal! Prefix: %s Context: %s",
                worker_id,
                depth,
                player,
                game_state.action_prefix,
                current_context.name,
            )
            worker_stats.error_count += 1
            # Log stall node
            stall_node = SimulationNodeData(
                depth=depth,
                player=player,
                infoset_key=infoset_key_tuple,
                context=current_context.name,
                strategy=[],
                chosen_action="STALLED_NO_LEGAL_ACTIONS",
                state_delta=[],
            )
            if config.logging.log_simulation_traces:
                simulation_nodes.append(stall_node)
            return np.zeros(NUM_PLAYERS, dtype=np.float64), 1.0
        else:  # Terminal due to no legal actions
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return (
                np.array(
                    [game_state.utility(i) for i in range(NUM_PLAYERS)],
                    dtype=np.float64,
                ),
                1.0,
            )

    # Get strategy from regrets
    current_regrets: Optional[np.ndarray] = regret_sum_snapshot.get(infoset_key)
    strategy = np.array([])
    if current_regrets is not None and len(current_regrets) == num_actions:
        strategy = get_rm_plus_strategy(current_regrets)
    else:
        if current_regrets is not None:
            logger_traverse.warning(
                "W%d D%d: Regret dim mismatch key %s. Snap:%d Need:%d. Using uniform.",
                worker_id,
                depth,
                infoset_key,
                len(current_regrets),
                num_actions,
            )
            worker_stats.warning_count += 1
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
        # Initialize local updates if key is new or has wrong dimension
        if (
            infoset_key not in local_regret_updates
            or len(local_regret_updates.get(infoset_key, np.array([])))
            != num_actions  # Check against np.array([])
        ):
            local_regret_updates[infoset_key] = np.zeros(num_actions, dtype=np.float64)
        if (
            infoset_key not in local_strategy_sum_updates
            or len(local_strategy_sum_updates.get(infoset_key, np.array([])))
            != num_actions  # Check against np.array([])
        ):
            local_strategy_sum_updates[infoset_key] = np.zeros(
                num_actions, dtype=np.float64
            )

    # --- Strategy Sum Update (Common to all CFR variants) ---
    # The acting player's own reach to this infoset. In a two-player game the
    # opponents' reach is exactly the other seat's, so opp_reach is that seat's
    # own reach whenever it is the one acting.
    player_reach = my_reach if player == updating_player else opp_reach
    # Averaging an outcome-sampled trajectory needs the same 1/q correction the
    # regrets get, or infosets the behaviour policy reaches often are
    # over-represented in the average strategy (cambia-719).
    strategy_sum_weight = (
        averaging_weight * player_reach / sample_reach if sample_reach > 0.0 else 0.0
    )
    if strategy_sum_weight > 0 and player_reach > 1e-9:
        if len(strategy) == num_actions:
            # Ensure local update entry exists and has correct dimension
            if (
                len(local_strategy_sum_updates.get(infoset_key, np.array([])))
                != num_actions
            ):
                local_strategy_sum_updates[infoset_key] = np.zeros(
                    num_actions, dtype=np.float64
                )
            local_strategy_sum_updates[infoset_key] += strategy_sum_weight * strategy
            local_reach_prob_updates[infoset_key] += strategy_sum_weight
        else:
            logger_traverse.error(
                "W%d D%d: Strategy len %d != num_actions %d for key %s. Skip strat update.",
                worker_id,
                depth,
                len(strategy),
                num_actions,
                infoset_key,
            )
            worker_stats.error_count += 1

    # --- Outcome Sampling: Sample one action and recurse ---
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)
    chosen_action_index: Optional[int] = None
    chosen_action: Optional[GameAction] = None
    sampling_prob_chosen = 0.0
    # The sampling reach and suffix reach of the sampled continuation, set once
    # an action is drawn; the regret update below reads both.
    next_sample_reach = 0.0
    tail_prob = 1.0

    if num_actions > 0 and len(strategy) == num_actions and np.sum(strategy) > 1e-9:
        # Normalize strategy before sampling just in case
        if not np.isclose(np.sum(strategy), 1.0):
            strategy = normalize_probabilities(strategy)

        # Explore only at the updating player's nodes: the other seat's actions
        # are chance from this traversal's point of view and are sampled on
        # policy, which is what makes opp_reach a counterfactual weight.
        sampling_probs = (
            sampling_policy(strategy, config.cfr_plus_params.outcome_sampling_epsilon)
            if player == updating_player
            else strategy
        )

        try:
            chosen_action_index = np.random.choice(num_actions, p=sampling_probs)
            chosen_action = legal_actions[chosen_action_index]
            sampling_prob_chosen = float(sampling_probs[chosen_action_index])
        except (
            ValueError
        ) as e_choice:  # Catch potential errors from invalid probabilities
            logger_traverse.error(
                "W%d D%d P%d: Error sampling action with strategy %s: %s. Using uniform.",
                worker_id,
                depth,
                player,
                strategy,
                e_choice,
            )
            worker_stats.error_count += 1
            if num_actions > 0:
                chosen_action_index = np.random.choice(num_actions)
                chosen_action = legal_actions[chosen_action_index]
                sampling_prob_chosen = 1.0 / num_actions
            else:  # Should not happen if num_actions > 0
                chosen_action_index = None
                chosen_action = None
                sampling_prob_chosen = 0.0

    elif num_actions > 0:  # Strategy was invalid (e.g., all zeros or length mismatch)
        logger_traverse.warning(
            "W%d D%d P%d: Invalid strategy %s for %d actions. Sampling uniformly.",
            worker_id,
            depth,
            player,
            strategy,
            num_actions,
        )
        worker_stats.warning_count += 1
        chosen_action_index = np.random.choice(num_actions)
        chosen_action = legal_actions[chosen_action_index]
        sampling_prob_chosen = 1.0 / num_actions
        # Use uniform strategy for regret update as well
        strategy = np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])

    # If an action was chosen (either via strategy or fallback uniform)
    if chosen_action is not None and chosen_action_index is not None:
        # Log the decision node (including the sampled action)
        node_data = SimulationNodeData(
            depth=depth,
            player=player,
            infoset_key=infoset_key_tuple,
            context=current_context.name,
            strategy=strategy.tolist() if strategy.size > 0 else [],
            chosen_action=_serialize_action_for_history(
                chosen_action
            ),  # Log sampled action
            state_delta=[],  # Will be set after apply_action
        )
        if config.logging.log_simulation_traces:
            simulation_nodes.append(node_data)

        # The reaches the sampled action produces. A player's own action
        # multiplies its own reach; the sampling reach takes the behaviour
        # policy's probability, which differs from the strategy's wherever
        # exploration was mixed in.
        chosen_strategy_prob = (
            float(strategy[chosen_action_index])
            if len(strategy) == num_actions
            else 1.0 / num_actions
        )
        if player == updating_player:
            next_my_reach = my_reach * chosen_strategy_prob
            next_opp_reach = opp_reach
        else:
            next_my_reach = my_reach
            next_opp_reach = opp_reach * chosen_strategy_prob
        next_sample_reach = sample_reach * sampling_prob_chosen

        # Apply the sampled action under a checkpoint. The Go engine has no
        # undo callable, so the checkpoint taken here is what the recursion
        # rewinds to on the way back up (cambia-1782).
        chosen_engine_index = legal_pairs[chosen_action_index][0]
        checkpoint: Optional[Checkpoint] = None
        apply_success = False
        try:
            checkpoint = game_state.checkpoint()
            game_state.apply(chosen_engine_index)
            apply_success = True
            if (
                config.logging.log_simulation_traces
                and simulation_nodes
                and simulation_nodes[-1] is node_data
            ):
                # The Go engine exports no per-action state delta. The applied
                # action index is the replayable record in its place: a deal
                # spec plus the action prefix reconstructs the node exactly,
                # which is the substrate's own record-and-replay contract.
                node_data["state_delta"] = [("action_index", str(chosen_engine_index))]
        except (
            Exception
        ) as apply_err:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            logger_traverse.error(
                "W%d D%d P%d: Error applying sampled action %s (index %d): %s. Prefix:%s Context:%s",
                worker_id,
                depth,
                player,
                chosen_action,
                chosen_engine_index,
                apply_err,
                game_state.action_prefix,
                current_context.name,
                exc_info=True,
            )
            worker_stats.error_count += 1
            if (
                config.logging.log_simulation_traces
                and simulation_nodes
                and simulation_nodes[-1] is node_data
            ):
                node_data["state_delta"] = [("apply_error", str(apply_err))]

        try:
            if apply_success:
                next_agent_states = []
                agent_update_failed = False
                player_specific_obs_for_log = None
                agent_idx = -1
                try:
                    # ability_reveals: the production frame, so a peek reaches
                    # the peeker's belief and a King swap moves the faces both
                    # beliefs had already seen.
                    observation = game_state.observation(
                        chosen_action, player, ability_reveals=True
                    )
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        if agent_idx == player:
                            player_specific_obs_for_log = player_specific_obs
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except (AgentStateError, ObservationUpdateError) as e_update:
                    logger_traverse.warning(
                        "W%d D%d: Agent state update error P%d after action %s: %s",
                        worker_id,
                        depth,
                        agent_idx,
                        chosen_action,
                        e_update,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                except (
                    Exception
                ) as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_traverse.error(
                        "W%d D%d: Error updating agent P%d after action %s: %s. Prefix:%s FilteredObs:%s",
                        worker_id,
                        depth,
                        agent_idx,
                        chosen_action,
                        e_update,
                        game_state.action_prefix,
                        player_specific_obs_for_log,  # May be None if error happened before P0 update
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True

                if not agent_update_failed:
                    try:
                        # Single recursive call for the sampled action
                        node_value, tail_prob = _traverse_game_for_worker(
                            game_state,
                            next_agent_states,
                            my_reach=next_my_reach,
                            opp_reach=next_opp_reach,
                            sample_reach=next_sample_reach,
                            iteration=iteration,
                            updating_player=updating_player,
                            averaging_weight=averaging_weight,
                            regret_sum_snapshot=regret_sum_snapshot,
                            config=config,
                            local_regret_updates=local_regret_updates,
                            local_strategy_sum_updates=local_strategy_sum_updates,
                            local_reach_prob_updates=local_reach_prob_updates,
                            depth=depth + 1,
                            worker_stats=worker_stats,
                            progress_queue=progress_queue,
                            worker_id=worker_id,
                            min_depth_after_bottom_out_tracker=(
                                min_depth_after_bottom_out_tracker
                            ),
                            has_bottomed_out_tracker=has_bottomed_out_tracker,
                            simulation_nodes=simulation_nodes,
                        )
                    except TraversalError as recursive_err:
                        logger_traverse.warning(
                            "W%d D%d: Traversal error in recursive call after action %s: %s",
                            worker_id,
                            depth,
                            chosen_action,
                            recursive_err,
                        )
                        worker_stats.error_count += 1
                        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)
                    except (
                        Exception
                    ) as recursive_err:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                        logger_traverse.error(
                            "W%d D%d: Error in recursive call after action %s: %s. Prefix:%s Context:%s",
                            worker_id,
                            depth,
                            chosen_action,
                            recursive_err,
                            game_state.action_prefix,
                            current_context.name,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        node_value = np.zeros(
                            NUM_PLAYERS, dtype=np.float64
                        )  # Set to zero on error
        finally:
            # Rewind once, on every path out. The Python engine's undo ran
            # twice when the recursive call raised (once in the handler and
            # again after it), which a checkpoint restore makes both
            # unnecessary and unsafe -- the snapshot handle is freed here.
            if checkpoint is not None:
                try:
                    if apply_success:
                        game_state.rewind(checkpoint)
                finally:
                    game_state.release(checkpoint)

    # --- Outcome Sampling Regret Update ---
    if player == updating_player and chosen_action_index is not None:
        if len(strategy) == num_actions:  # Check strategy validity
            if next_sample_reach > 0.0:
                # The sampled action's counterfactual value: the leaf utility,
                # weighted by the opponents' reach to this infoset and by the
                # updating player's own reach over the suffix, and corrected by
                # the behaviour policy's probability of the whole trajectory
                # prefix through this action (Lanctot et al. 2009). Every
                # unsampled action's estimate is zero.
                action_value = (
                    opp_reach * tail_prob * node_value[player] / next_sample_reach
                )

                # Ensure local regret update entry exists and has correct dimension
                if (
                    len(local_regret_updates.get(infoset_key, np.array([])))
                    != num_actions
                ):
                    local_regret_updates[infoset_key] = np.zeros(
                        num_actions, dtype=np.float64
                    )

                # CFR+ regret updates carry no iteration weight. The delayed
                # linear weight applies to the average strategy only; folding it
                # in here zeroed every update up to the delay (cambia-718).
                local_regret_updates[infoset_key] += sampled_regrets(
                    strategy, chosen_action_index, action_value
                )

            else:  # Sampling reach underflowed to zero
                # The reach is a product over the trajectory prefix, so it gets
                # small with depth by construction. Only an actual underflow to
                # zero is skipped: clipping at some epsilon instead would drop
                # deep updates and bias the estimate. Before cambia-719 this
                # branch tested a single node's probability and was where an
                # action at strategy probability zero got frozen out for good.
                logger_traverse.debug(
                    "W%d D%d P%d: Sampling reach %.3e underflowed for action %s at key %s. Skipping regret update.",
                    worker_id,
                    depth,
                    player,
                    next_sample_reach,
                    chosen_action,
                    infoset_key,
                )
        else:  # Strategy was invalid when sampling occurred
            logger_traverse.warning(
                "W%d D%d P%d: Cannot perform regret update, strategy was invalid during sampling.",
                worker_id,
                depth,
                player,
            )

    # The suffix factor handed to the caller. Only the updating player's own
    # actions enter the numerator, so a node belonging to anyone else contributes
    # 1 there while still dividing out the probability its action was sampled
    # with.
    if chosen_action_index is None:
        return node_value, 1.0
    own_action_prob = 1.0
    if player == updating_player and len(strategy) == num_actions:
        own_action_prob = float(strategy[chosen_action_index])
    return node_value, suffix_reach_ratio(
        tail_prob, own_action_prob, sampling_prob_chosen
    )


def run_cfr_simulation_worker(
    worker_args: Tuple[
        int,
        Config,
        RegretSnapshotDict,
        Optional[ProgressQueueWorker],
        Optional[ArchiveQueueWorker],
        int,
        str,
        str,
    ],
    deal: Optional[DealSpec] = None,
) -> Optional[WorkerResult]:
    """Top-level function executed by each worker process. Sets up per-worker logging.

    ``deal`` pins the deal instead of drawing a fresh one, which is what the
    table-equality gate needs to put two engines on the same game. The training
    pool maps this function over a single argument and never passes it, so a
    production run keeps dealing from process entropy exactly as the Python
    engine's unseeded shuffle did.
    """
    # Initialize logger_instance to None
    logger_instance: Optional[logging.Logger] = None
    worker_stats = WorkerStats()
    (
        iteration,
        config,
        regret_sum_snapshot,
        progress_queue,
        archive_queue,
        worker_id,
        run_log_dir,
        run_timestamp,
    ) = worker_args

    # Store worker ID in stats
    worker_stats.worker_id = worker_id

    # Initialize trace list for this simulation
    simulation_nodes_this_sim: List[SimulationNodeData] = []
    final_utility_value: Optional[np.ndarray] = None
    game_state: Optional[GoBrState] = None

    worker_root_logger = logging.getLogger()
    try:
        # Clear existing handlers for this process
        for handler in worker_root_logger.handlers[:]:
            worker_root_logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except Exception:
                    pass

        # Set root level BEFORE adding handlers
        worker_root_logger.setLevel(logging.DEBUG)

        # Add NullHandler to prevent defaults and stop propagation
        null_handler = logging.NullHandler()
        worker_root_logger.addHandler(null_handler)
        worker_root_logger.propagate = False  # Prevent logs reaching main process root

        # Setup per-worker logging to file
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
        worker_log_level_str = config.logging.get_worker_log_level(
            worker_id, config.cfr_training.num_workers
        )
        file_log_level = getattr(logging, worker_log_level_str.upper(), logging.DEBUG)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        worker_root_logger.addHandler(file_handler)  # Add specific file handler

        # Now get the logger instance for this module AFTER setup
        logger_instance = logging.getLogger(__name__)
        logger_instance.info(
            "Worker %d logging initialized (dir: %s, file_level: %s). Root handlers: %s",
            worker_id,
            worker_log_dir,
            logging.getLevelName(file_log_level),
            [type(h).__name__ for h in worker_root_logger.handlers],
        )

    except Exception as log_setup_e:
        # Fallback logging if setup fails
        print(
            f"!!! CRITICAL Error setting up logging W{worker_id}: {log_setup_e} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        worker_stats.error_count += 1
        # Ensure the root logger doesn't propagate setup errors if handlers fail
        if not worker_root_logger.hasHandlers():
            worker_root_logger.addHandler(logging.NullHandler())
        logger_instance = logging.getLogger(__name__)  # Still try to get logger

    # Main simulation logic
    try:
        # --- Game and Agent State Initialization ---
        try:
            game_state = GoBrState.new(
                config.cambia_rules,
                deal if deal is not None else DealSpec(seed=random.getrandbits(63)),
            )
        except GameStateError as game_init_e:
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game state initialization error: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                )
            worker_stats.error_count += 1
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )
        except (
            Exception
        ) as game_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Failed GameState init: %s",
                    worker_id,
                    iteration,
                    game_init_e,
                    exc_info=True,
                )
            worker_stats.error_count += 1
            # Return minimal result indicating failure
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )

        initial_agent_states = []
        if not game_state.is_terminal():
            try:
                # Create observation needed for AgentState initialization
                initial_obs = game_state.initial_observation(ability_reveals=True)
                initial_hands = [
                    game_state.hand(i) for i in range(game_state.num_players())
                ]
                initial_peeks = [
                    game_state.initial_peek_indices()
                    for _ in range(game_state.num_players())
                ]
                for i in range(NUM_PLAYERS):
                    agent = AgentState(
                        player_id=i,
                        opponent_id=1 - i,
                        memory_level=config.agent_params.memory_level,
                        time_decay_turns=config.agent_params.time_decay_turns,
                        initial_hand_size=len(initial_hands[i]),
                        config=config,
                    )
                    agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
                    initial_agent_states.append(agent)
            except (
                AgentStateError,
                ObservationUpdateError,
                EncodingError,
            ) as agent_init_e:
                if logger_instance:
                    logger_instance.warning(
                        "W%d Iter %d: Agent state initialization error: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                    )
                worker_stats.error_count += 1
                return WorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=None,
                )
            except (
                Exception
            ) as agent_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Failed AgentStates init: %s. Deal: %s",
                        worker_id,
                        iteration,
                        agent_init_e,
                        game_state.deal,
                        exc_info=True,
                    )
                worker_stats.error_count += 1
                return WorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=None,
                )
        else:  # Game terminal at start
            if logger_instance:
                logger_instance.warning(
                    "W%d Iter %d: Game terminal at init. Deal: %s",
                    worker_id,
                    iteration,
                    game_state.deal,
                )
            final_utility_value = np.array(
                [game_state.utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=final_utility_value.tolist(),
            )

        if len(initial_agent_states) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Incorrect agent states initialized (%d).",
                    worker_id,
                    iteration,
                    len(initial_agent_states),
                )
            worker_stats.error_count += 1
            return WorkerResult(
                stats=worker_stats,
                simulation_nodes=simulation_nodes_this_sim,
                final_utility=None,
            )

        # --- Traversal ---
        updating_player = iteration % NUM_PLAYERS
        iteration_averaging_weight = averaging_weight(iteration, config.cfr_plus_params)
        local_regret_updates: LocalRegretUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_strategy_sum_updates: LocalStrategyUpdateDict = defaultdict(
            lambda: np.array([], dtype=np.float64)
        )
        local_reach_prob_updates: LocalReachProbUpdateDict = defaultdict(float)
        min_depth_after_bottom_out_tracker = [float("inf")]
        has_bottomed_out_tracker = [False]

        # Run the outcome-sampling traversal
        final_utility_value, _root_tail = _traverse_game_for_worker(
            game_state=game_state,
            agent_states=initial_agent_states,
            my_reach=1.0,
            opp_reach=1.0,
            sample_reach=1.0,
            iteration=iteration,
            updating_player=updating_player,
            averaging_weight=iteration_averaging_weight,
            regret_sum_snapshot=regret_sum_snapshot,
            config=config,
            local_regret_updates=local_regret_updates,
            local_strategy_sum_updates=local_strategy_sum_updates,
            local_reach_prob_updates=local_reach_prob_updates,
            depth=0,
            worker_stats=worker_stats,
            progress_queue=progress_queue,
            worker_id=worker_id,
            min_depth_after_bottom_out_tracker=min_depth_after_bottom_out_tracker,
            has_bottomed_out_tracker=has_bottomed_out_tracker,
            simulation_nodes=simulation_nodes_this_sim,
        )

        # Note: final_utility_value here is the return from the recursive call,
        # which represents the utility obtained from the single sampled path.
        if final_utility_value is None or len(final_utility_value) != NUM_PLAYERS:
            if logger_instance:
                logger_instance.error(
                    "W%d Iter %d: Traversal returned invalid utility: %s. Setting final to zero.",
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

        # --- Return Result ---
        return WorkerResult(
            regret_updates=dict(local_regret_updates),
            strategy_updates=dict(local_strategy_sum_updates),
            reach_prob_updates=dict(local_reach_prob_updates),
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=(
                final_utility_value.tolist() if final_utility_value is not None else None
            ),
        )

    except KeyboardInterrupt:
        if logger_instance:
            logger_instance.warning(
                "W%d Iter %d received KeyboardInterrupt.", worker_id, iteration
            )
        worker_stats.error_count += 1
        # Return partial results if interrupted
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    except (
        Exception
    ) as e_inner:  # JUSTIFIED: worker resilience - top-level worker catch to prevent pool crash
        worker_stats.error_count += 1
        if logger_instance:
            logger_instance.critical(
                "!!! Unhandled Error W%d Iter %d simulation: %s !!!",
                worker_id,
                iteration,
                e_inner,
                exc_info=True,
            )
        # Also print to stderr for visibility if logging fails
        print(
            f"!!! FATAL WORKER ERROR W{worker_id} Iter {iteration}: {e_inner} !!!",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        return WorkerResult(
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=None,
        )
    finally:
        # The engine handle pool is finite and a worker process is reused across
        # iterations, so the state is closed on every exit path, early returns
        # included.
        if game_state is not None:
            game_state.close()
        # Ensure logs are flushed and handlers closed on worker exit
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


# --- Observation Helpers ---
#
# The tabular traversal above no longer calls these: it reads its frames off the
# engine in GoBrState.observation. They stay because the deep-CFR, PRT-CFR and
# evaluation lanes import them, and they are duck-typed on the game state they
# are handed rather than importing one, so this module carries no dependency on
# the retiring Python engine (cambia-1782).
def _create_observation(
    prev_state: Any,  # Kept for signature consistency if needed elsewhere
    action: Optional[GameAction],
    next_state: Any,
    acting_player: int,
    snap_results: List[Dict],
    king_swap_indices: Optional[tuple] = None,  # (own_idx, opp_idx) for king swap
) -> Optional[AgentObservation]:
    """Creates the AgentObservation object based on the state *after* the action.

    ``next_state`` is any game state exposing the Python engine's read surface
    (get_discard_top, get_player_card_count, get_stockpile_size, players,
    pending_action_data, cambia_caller_id, is_terminal, get_turn_number).
    """
    logger_obs = logging.getLogger(__name__)  # Use module logger
    try:
        discard_top = next_state.get_discard_top()
        hand_sizes = [next_state.get_player_card_count(i) for i in range(NUM_PLAYERS)]
        stock_size = next_state.get_stockpile_size()
        cambia_called = next_state.cambia_caller_id is not None
        who_called = next_state.cambia_caller_id
        game_over = next_state.is_terminal()
        turn_num = next_state.get_turn_number()

        # Determine drawn card based on action and NEXT state
        drawn_card_for_obs = None
        if isinstance(action, (ActionDrawStockpile, ActionDrawDiscard)):
            # Surface the freshly drawn card at the post-draw decision node. The
            # card lives in the post-draw pending state (engine sets
            # pending_action_data["drawn_card"]); the tokenizer emits it as the
            # actor-private drawn frame BEFORE the discard/replace decision, so
            # that decision's legal-action mask is determined by the infoset
            # (cambia-528). Belief tracking still reads drawn_card on Replace
            # (populated below); the tokenizer gates the drawn frame on the draw
            # action, so the two uses do not double-emit.
            pad = getattr(next_state, "pending_action_data", None)
            if (
                pad
                and getattr(next_state, "pending_action_player", None) == acting_player
            ):
                drawn_card_for_obs = pad.get("drawn_card")
        elif isinstance(action, ActionDiscard):
            # The card just discarded *was* the drawn card. It's now the top of discard.
            drawn_card_for_obs = next_state.get_discard_top()
            if drawn_card_for_obs is None:
                logger_obs.error("Create Obs: ActionDiscard but discard pile empty?")
        elif isinstance(action, ActionReplace):
            # The card just placed into the hand *was* the drawn card.
            if acting_player != -1 and action.target_hand_index < len(
                next_state.players[acting_player].hand
            ):
                drawn_card_for_obs = next_state.players[acting_player].hand[
                    action.target_hand_index
                ]
            else:
                logger_obs.error(
                    "Create Obs: ActionReplace index %d invalid for actor %d hand size %d",
                    action.target_hand_index if action else -1,
                    acting_player,
                    (
                        len(next_state.players[acting_player].hand)
                        if acting_player != -1
                        else -1
                    ),
                )

        # Populate peeked cards *only* if the action caused a peek
        peeked_cards_dict = None
        if isinstance(action, ActionAbilityPeekOwnSelect) and acting_player != -1:
            hand = next_state.get_player_hand(acting_player)
            target_idx = action.target_hand_index
            if hand and 0 <= target_idx < len(hand):
                card = hand[target_idx]
                peeked_cards_dict = {(acting_player, target_idx): card}
            else:
                logger_obs.warning(
                    "Create Obs: PeekOwn index %d invalid for hand size %d.",
                    target_idx,
                    len(hand) if hand else 0,
                )
        elif isinstance(action, ActionAbilityPeekOtherSelect) and acting_player != -1:
            opp_idx = next_state.get_opponent_index(acting_player)
            opp_hand = next_state.get_player_hand(opp_idx)
            target_opp_idx = action.target_opponent_hand_index
            if opp_hand and 0 <= target_opp_idx < len(opp_hand):
                card = opp_hand[target_opp_idx]
                peeked_cards_dict = {(opp_idx, target_opp_idx): card}
            else:
                logger_obs.warning(
                    "Create Obs: PeekOther index %d invalid for opp hand size %d.",
                    target_opp_idx,
                    len(opp_hand) if opp_hand else 0,
                )
        elif isinstance(action, ActionAbilityKingLookSelect) and acting_player != -1:
            # Re-fetch cards looked at for the observation
            own_idx, opp_look_idx = action.own_hand_index, action.opponent_hand_index
            opp_real_idx = next_state.get_opponent_index(acting_player)
            own_hand = next_state.get_player_hand(acting_player)
            opp_hand = next_state.get_player_hand(opp_real_idx)
            card1, card2 = None, None
            if own_hand and 0 <= own_idx < len(own_hand):
                card1 = own_hand[own_idx]
            if opp_hand and 0 <= opp_look_idx < len(opp_hand):
                card2 = opp_hand[opp_look_idx]
            if card1 and card2:
                peeked_cards_dict = {
                    (acting_player, own_idx): card1,
                    (opp_real_idx, opp_look_idx): card2,
                }
            else:
                logger_obs.warning(
                    "Create Obs: KingLook indices invalid/cards missing. Own %s/%s, Opp %s/%s",
                    own_idx,
                    len(own_hand) if own_hand else "N/A",
                    opp_look_idx,
                    len(opp_hand) if opp_hand else "N/A",
                )

        final_snap_results = snap_results if snap_results else []

        # Race-ON snap (cambia-564). A live race_resolution record on next_state
        # means this action just resolved a race window: emit the public race frames
        # (and suppress the normal frames). Otherwise, an intermediate race-ON snap
        # commit (pass/own/opp) is suppressed entirely (imperfect info). Race-OFF and
        # non-snap actions leave both cleared.
        race_resolution = getattr(next_state, "race_resolution", None)
        is_race_commit = False
        if race_resolution is None and getattr(next_state.house_rules, "snapRace", False):
            if isinstance(action, (ActionPassSnap, ActionSnapOwn, ActionSnapOpponent)):
                is_race_commit = True

        # Populate king_swap_indices if this action is a performed king swap
        obs_king_swap_indices = king_swap_indices
        if (
            obs_king_swap_indices is None
            and isinstance(action, ActionAbilityKingSwapDecision)
            and action.perform_swap
        ):
            # Try to read from next_state pending_action_data (may already be cleared)
            pad = next_state.pending_action_data
            if pad and "own_idx" in pad and "opp_idx" in pad:
                obs_king_swap_indices = (pad["own_idx"], pad["opp_idx"])

        obs = AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=discard_top,
            player_hand_sizes=hand_sizes,
            stockpile_size=stock_size,
            drawn_card=drawn_card_for_obs,  # Determined from next_state based on action
            peeked_cards=peeked_cards_dict,
            snap_results=final_snap_results,
            closing_snap_results=list(
                getattr(next_state, "snap_results_at_close", []) or []
            ),
            did_cambia_get_called=cambia_called,
            who_called_cambia=who_called,
            is_game_over=game_over,
            current_turn=turn_num,
            king_swap_indices=obs_king_swap_indices,
            is_race_commit=is_race_commit,
            race_resolution=race_resolution,
        )
        logger_obs.debug("Created observation: %s", obs)
        return obs
    except GameStateError as e:
        logger_obs.warning("Game state error creating observation: %s", e)
        return None
    except (
        Exception
    ) as e:  # JUSTIFIED: worker resilience - observation creation must not crash worker
        logger_obs.error("Error creating observation: %s", e, exc_info=True)
        return None


def _filter_observation(obs: AgentObservation, observer_id: int) -> AgentObservation:
    """Creates a player-specific view of the observation, masking private info."""
    # Shallow copy is usually sufficient as AgentState doesn't modify observation fields
    filtered_obs = copy.copy(obs)

    # Mask drawn card unless observer is the actor AND the action requires the drawn card info
    if obs.drawn_card and obs.acting_player != observer_id:
        filtered_obs.drawn_card = None
    elif obs.drawn_card and obs.acting_player == observer_id:
        # Keep drawn_card for the actor on the draw action (the tokenizer's
        # post-draw private frame, cambia-528) and on Discard/Replace (belief
        # tracking reads it there). Every other action nulls it.
        if not isinstance(
            obs.action,
            (ActionDiscard, ActionReplace, ActionDrawStockpile, ActionDrawDiscard),
        ):
            filtered_obs.drawn_card = None

    # Filter peeked cards - only pass if observer was the actor performing the peek/look
    if obs.peeked_cards:
        is_observer_peek_action = (
            (
                isinstance(obs.action, ActionAbilityPeekOwnSelect)
                and obs.acting_player == observer_id
            )
            or (
                isinstance(obs.action, ActionAbilityPeekOtherSelect)
                and obs.acting_player == observer_id
            )
            or (
                isinstance(obs.action, ActionAbilityKingLookSelect)
                and obs.acting_player == observer_id
            )
        )
        if not is_observer_peek_action:
            filtered_obs.peeked_cards = None
        # Note: KingSwapDecision doesn't reveal peek info in *this* observation's action
    else:
        filtered_obs.peeked_cards = None

    # Snap results are public information derived from game state deltas
    filtered_obs.snap_results = obs.snap_results
    # Public for the same reason, and carried on its own field so the tokenizer
    # channel above stays byte-identical to Go's (cambia-1985).
    filtered_obs.closing_snap_results = obs.closing_snap_results

    return filtered_obs
