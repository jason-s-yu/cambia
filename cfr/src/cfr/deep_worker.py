"""
src/cfr/deep_worker.py

Implements the Deep CFR worker process using External Sampling MCCFR.

Key differences from worker.py (tabular outcome sampling):
- External Sampling: enumerate ALL actions at traverser nodes, sample ONE at opponent/chance nodes
- No importance sampling correction (exact regrets from enumeration)
- Returns ReservoirSamples instead of regret/strategy dict updates
- Uses neural network for strategy computation (AdvantageNetwork -> ReLU -> normalize)
- Encodes infosets via encode_infoset() from encoding.py
"""

import logging
import os
import queue
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..agent_state import AgentState
from ..config import Config
from .exceptions import (
    GameStateError,
    ActionApplicationError,
    UndoFailureError,
    AgentStateError,
    ObservationUpdateError,
    EncodingError,
    NetworkError,
    TraversalError,
)
from ..constants import (
    NUM_PLAYERS,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionAbilityPeekOtherSelect,
    ActionAbilityPeekOwnSelect,
    ActionDiscard,
    ActionSnapOpponentMove,
    DecisionContext,
)
from ..game.engine import CambiaGameState
from ..serial_rotating_handler import SerialRotatingFileHandler
from ..abstraction import get_card_bucket
from ..encoding import encode_infoset, encode_action_mask, action_to_index, INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, get_strategy_from_advantages
from ..reservoir import ReservoirSample
from ..utils import WorkerStats, SimulationNodeData

# Re-use observation helpers from worker.py
from .worker import _create_observation, _filter_observation

logger = logging.getLogger(__name__)

# Progress update interval (nodes)
PROGRESS_UPDATE_NODE_INTERVAL = 2500


@dataclass
class DeepCFRWorkerResult:
    """Results from a single Deep CFR worker traversal."""

    advantage_samples: List[ReservoirSample] = field(default_factory=list)
    strategy_samples: List[ReservoirSample] = field(default_factory=list)
    stats: WorkerStats = field(default_factory=WorkerStats)
    simulation_nodes: List[SimulationNodeData] = field(default_factory=list)
    final_utility: Optional[List[float]] = None


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
                _feat_buf.copy_(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0))
                _mask_buf.copy_(torch.from_numpy(action_mask).bool().unsqueeze(0))
                features_tensor = _feat_buf
                mask_tensor = _mask_buf
            else:
                features_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0)
            raw_advantages = network(features_tensor, mask_tensor).squeeze(0)
            strategy_tensor = get_strategy_from_advantages(
                raw_advantages.unsqueeze(0), mask_tensor
            )
            return strategy_tensor.squeeze(0).numpy()
    except Exception as e:
        raise NetworkError(f"Network inference failed: {e}") from e


def _deep_traverse(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
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
) -> np.ndarray:
    """
    Recursive External Sampling traversal for Deep CFR.

    At traverser's node: enumerate ALL legal actions, recurse on each, compute exact regrets.
    At opponent's node: sample ONE action from strategy (network), recurse.
    At chance node: sample ONE outcome, recurse.

    Returns utility vector for both players.
    """
    logger_t = logging.getLogger(__name__)

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
            logger_t.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    if game_state.is_terminal():
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Depth limit check (system recursion limit)
    if depth >= config.system.recursion_limit:
        logger_t.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Traversal depth cap (0 = unlimited)
    depth_limit = getattr(getattr(config, "deep_cfr", None), "traversal_depth_limit", 0)
    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Determine decision context
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    elif game_state.pending_action:
        pending = game_state.pending_action
        if isinstance(pending, ActionDiscard):
            current_context = DecisionContext.POST_DRAW
        elif isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            current_context = DecisionContext.ABILITY_SELECT
        elif isinstance(pending, ActionSnapOpponentMove):
            current_context = DecisionContext.SNAP_MOVE
        else:
            logger_t.warning(
                "W%d D%d: Unknown pending action type (%s).",
                worker_id,
                depth,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger_t.error("W%d D%d: Could not determine acting player.", worker_id, depth)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    opponent = 1 - player

    # Get legal actions
    try:
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except GameStateError as e_legal:
        logger_t.warning(
            "W%d D%d: Game state error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_legal:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    num_actions = len(legal_actions)
    if num_actions == 0:
        if not game_state.is_terminal():
            logger_t.error(
                "W%d D%d: No legal actions but non-terminal!", worker_id, depth
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Encode infoset and action mask
    current_agent_state = agent_states[player]

    # Get drawn card bucket for POST_DRAW encoding
    drawn_card_bucket = None
    if current_context == DecisionContext.POST_DRAW:
        drawn_card_obj = game_state.pending_action_data.get("drawn_card")
        if drawn_card_obj is not None:
            drawn_card_bucket = get_card_bucket(drawn_card_obj)

    try:
        features = encode_infoset(
            current_agent_state, current_context, drawn_card_bucket=drawn_card_bucket
        )
        action_mask = encode_action_mask(legal_actions)
    except (EncodingError, AgentStateError) as e_encode:
        logger_t.warning(
            "W%d D%d: Encoding/agent state error for infoset: %s",
            worker_id,
            depth,
            e_encode,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_encode:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error encoding infoset/mask: %s",
            worker_id,
            depth,
            e_encode,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger_t.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        except Exception as e_net:  # JUSTIFIED: worker resilience - fallback to uniform strategy on unexpected errors
            logger_t.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        # No network weights yet (first iteration) - use uniform
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Map network strategy (NUM_ACTIONS) to local strategy (num_actions)
    # The strategy from get_strategy_from_advantages is already over legal actions only
    # if action_mask was used correctly. However, the network outputs NUM_ACTIONS dims.
    # We need to extract only the legal action probabilities.
    if len(strategy) == NUM_ACTIONS:
        local_strategy = np.zeros(num_actions, dtype=np.float64)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            local_strategy[a_idx] = strategy[global_idx]
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        strategy = local_strategy

    # Ensure strategy length matches
    if len(strategy) != num_actions:
        logger_t.warning(
            "W%d D%d: Strategy len %d != num_actions %d. Using uniform.",
            worker_id,
            depth,
            len(strategy),
            num_actions,
        )
        worker_stats.warning_count += 1
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- External Sampling Logic ---
    if player == updating_player:
        # TRAVERSER'S NODE: enumerate ALL legal actions
        action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

        for a_idx, action in enumerate(legal_actions):
            apply_success = False
            try:
                state_delta, undo_info = game_state.apply_action(action)
                if callable(undo_info):
                    apply_success = True
                else:
                    logger_t.error(
                        "W%d D%d: apply_action for %s returned invalid undo.",
                        worker_id,
                        depth,
                        action,
                    )
                    worker_stats.error_count += 1
            except ActionApplicationError as e_apply:
                logger_t.warning(
                    "W%d D%d: Action application error for %s: %s",
                    worker_id,
                    depth,
                    action,
                    e_apply,
                )
                worker_stats.error_count += 1
            except Exception as e_apply:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                logger_t.error(
                    "W%d D%d: Error applying action %s: %s",
                    worker_id,
                    depth,
                    action,
                    e_apply,
                    exc_info=True,
                )
                worker_stats.error_count += 1

            if apply_success:
                # Create observation and update agent states
                observation = _create_observation(
                    None, action, game_state, player, game_state.snap_results_log
                )
                next_agent_states = []
                agent_update_failed = False

                if observation is None:
                    logger_t.error(
                        "W%d D%d: Failed to create observation after %s.",
                        worker_id,
                        depth,
                        action,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass
                else:
                    try:
                        for agent_idx, agent_state in enumerate(agent_states):
                            cloned_agent = agent_state.clone()
                            player_specific_obs = _filter_observation(
                                observation, agent_idx
                            )
                            cloned_agent.update(player_specific_obs)
                            next_agent_states.append(cloned_agent)
                    except (AgentStateError, ObservationUpdateError) as e_update:
                        logger_t.warning(
                            "W%d D%d: Agent state update error after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_update,
                        )
                        worker_stats.error_count += 1
                        agent_update_failed = True
                        try:
                            undo_info()
                        except UndoFailureError:
                            pass
                        except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            pass
                    except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                        logger_t.error(
                            "W%d D%d: Error updating agents after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_update,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        agent_update_failed = True
                        try:
                            undo_info()
                        except UndoFailureError:
                            pass
                        except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                            pass

                if not agent_update_failed:
                    try:
                        action_values[a_idx] = _deep_traverse(
                            game_state,
                            next_agent_states,
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
                    except TraversalError as e_recurse:
                        logger_t.warning(
                            "W%d D%d: Traversal error in recursion after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_recurse,
                        )
                        worker_stats.error_count += 1
                    except Exception as e_recurse:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                        logger_t.error(
                            "W%d D%d: Recursion error after %s: %s",
                            worker_id,
                            depth,
                            action,
                            e_recurse,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1

                    # Undo after recursion
                    try:
                        undo_info()
                    except UndoFailureError as e_undo:
                        logger_t.error(
                            "W%d D%d: Undo failure for %s: %s. State corrupt.",
                            worker_id,
                            depth,
                            action,
                            e_undo,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)
                    except Exception as e_undo:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                        logger_t.error(
                            "W%d D%d: Error undoing %s: %s. State corrupt.",
                            worker_id,
                            depth,
                            action,
                            e_undo,
                            exc_info=True,
                        )
                        worker_stats.error_count += 1
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Compute exact counterfactual values
        # node_value = sum over actions: strategy[a] * action_values[a]
        node_value = strategy @ action_values  # shape: (NUM_PLAYERS,)

        # Compute regrets: regret(a) = v(a)[player] - node_value[player]
        regrets = action_values[:, player] - node_value[player]

        # Build full-size regret target vector (NUM_ACTIONS)
        regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            regret_target[global_idx] = regrets[a_idx]

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
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            strategy_target[global_idx] = strategy[a_idx]

        strategy_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

        # Sample one action
        if np.sum(strategy) > 1e-9:
            try:
                chosen_idx = np.random.choice(num_actions, p=strategy)
            except ValueError:
                chosen_idx = np.random.choice(num_actions)
                worker_stats.warning_count += 1
        else:
            chosen_idx = np.random.choice(num_actions)
            worker_stats.warning_count += 1

        chosen_action = legal_actions[chosen_idx]

        # Apply action, recurse, undo
        apply_success = False
        node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            state_delta, undo_info = game_state.apply_action(chosen_action)
            if callable(undo_info):
                apply_success = True
            else:
                logger_t.error(
                    "W%d D%d: apply_action for sampled %s returned invalid undo.",
                    worker_id,
                    depth,
                    chosen_action,
                )
                worker_stats.error_count += 1
        except ActionApplicationError as e_apply:
            logger_t.warning(
                "W%d D%d: Action application error for sampled %s: %s",
                worker_id,
                depth,
                chosen_action,
                e_apply,
            )
            worker_stats.error_count += 1
        except Exception as e_apply:  # JUSTIFIED: worker resilience - workers must not crash the training pool
            logger_t.error(
                "W%d D%d: Error applying sampled %s: %s",
                worker_id,
                depth,
                chosen_action,
                e_apply,
                exc_info=True,
            )
            worker_stats.error_count += 1

        if apply_success:
            observation = _create_observation(
                None, chosen_action, game_state, player, game_state.snap_results_log
            )
            next_agent_states = []
            agent_update_failed = False

            if observation is None:
                logger_t.error(
                    "W%d D%d: Failed to create observation after sampled %s.",
                    worker_id,
                    depth,
                    chosen_action,
                )
                worker_stats.error_count += 1
                agent_update_failed = True
                try:
                    undo_info()
                except UndoFailureError:
                    pass
                except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                    pass
            else:
                try:
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except (AgentStateError, ObservationUpdateError) as e_update:
                    logger_t.warning(
                        "W%d D%d: Agent state update error after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_update,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass
                except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_t.error(
                        "W%d D%d: Error updating agents after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_update,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    agent_update_failed = True
                    try:
                        undo_info()
                    except UndoFailureError:
                        pass
                    except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                        pass

            if not agent_update_failed:
                try:
                    node_value = _deep_traverse(
                        game_state,
                        next_agent_states,
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
                except TraversalError as e_recurse:
                    logger_t.warning(
                        "W%d D%d: Traversal error in recursion after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_recurse,
                    )
                    worker_stats.error_count += 1
                except Exception as e_recurse:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    logger_t.error(
                        "W%d D%d: Recursion error after sampled %s: %s",
                        worker_id,
                        depth,
                        chosen_action,
                        e_recurse,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1

                try:
                    undo_info()
                except UndoFailureError as e_undo:
                    logger_t.error(
                        "W%d D%d: Undo failure for sampled %s: %s. State corrupt.",
                        worker_id,
                        depth,
                        chosen_action,
                        e_undo,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    return np.zeros(NUM_PLAYERS, dtype=np.float64)
                except Exception as e_undo:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                    logger_t.error(
                        "W%d D%d: Error undoing sampled %s: %s. State corrupt.",
                        worker_id,
                        depth,
                        chosen_action,
                        e_undo,
                        exc_info=True,
                    )
                    worker_stats.error_count += 1
                    return np.zeros(NUM_PLAYERS, dtype=np.float64)

        return node_value


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
        Integer decision context (0=StartTurn, 1=PostDraw, 2=AbilitySelect,
        3=SnapDecision, 4=SnapMove).
    """
    if legal_mask[0] or legal_mask[1] or legal_mask[2]:
        return 0  # CtxStartTurn
    if legal_mask[3] or legal_mask[4] or any(legal_mask[5:11]):
        return 1  # CtxPostDraw
    if any(legal_mask[11:97]):
        return 2  # CtxAbilitySelect
    if legal_mask[97] or any(legal_mask[98:110]):
        return 3  # CtxSnapDecision
    if any(legal_mask[110:146]):
        return 4  # CtxSnapMove
    return 0  # fallback


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
) -> np.ndarray:
    """
    Recursive External Sampling traversal for Deep CFR using the Go engine backend.

    Uses GoEngine (save/restore instead of undo) and GoAgentState (encode directly).

    At traverser's node: enumerate ALL legal actions, recurse on each, compute exact regrets.
    At opponent's node: sample ONE action from strategy (network), recurse.

    Returns utility vector (shape (2,) float64) for both players.
    """
    logger_t = logging.getLogger(__name__)

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
            logger_t.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
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
        logger_t.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Depth limit check (system recursion limit)
    if depth >= config.system.recursion_limit:
        logger_t.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
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
        logger_t.error(
            "W%d D%d: Error getting legal action mask: %s", worker_id, depth, e_legal
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger_t.error("W%d D%d: No legal actions but non-terminal!", worker_id, depth)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Infer decision context from legal mask
    current_context = _infer_decision_context(legal_mask)

    # Get acting player
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger_t.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get drawn card bucket for POST_DRAW encoding
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode infoset using Go agent
    try:
        features = agent_states[player].encode(current_context, drawn_bucket=drawn_bucket)
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger_t.error(
            "W%d D%d: Error encoding infoset: %s", worker_id, depth, e_encode
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy_full = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger_t.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience - fallback to uniform
            logger_t.warning(
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
            logger_t.error(
                "W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            agent_clones = [a.clone() for a in agent_states]
        except Exception as e_clone:
            logger_t.error(
                "W%d D%d: Failed to clone agent states: %s", worker_id, depth, e_clone
            )
            worker_stats.error_count += 1
            engine.free_snapshot(snap)
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        for i, action_idx in enumerate(legal_indices):
            if i > 0:
                # Restore engine and agent states for next action
                try:
                    engine.restore(snap)
                except Exception as e_restore:
                    logger_t.error(
                        "W%d D%d: Failed to restore engine: %s", worker_id, depth, e_restore
                    )
                    worker_stats.error_count += 1
                    break
                for j, a in enumerate(agent_states):
                    a.close()
                    try:
                        agent_states[j] = agent_clones[j].clone()
                    except Exception as e_clone2:
                        logger_t.error(
                            "W%d D%d: Failed to clone agent %d: %s",
                            worker_id, depth, j, e_clone2,
                        )
                        worker_stats.error_count += 1
                        break

            try:
                engine.apply_action(int(action_idx))
                for a in agent_states:
                    a.update(engine)
            except Exception as e_apply:
                logger_t.error(
                    "W%d D%d: Error applying action %d: %s",
                    worker_id, depth, action_idx, e_apply,
                )
                worker_stats.error_count += 1
                continue

            try:
                action_values[i] = _deep_traverse_go(
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
            except Exception as e_recurse:  # JUSTIFIED: worker resilience
                logger_t.error(
                    "W%d D%d: Recursion error after action %d: %s",
                    worker_id, depth, action_idx, e_recurse, exc_info=True,
                )
                worker_stats.error_count += 1

        # Final restore of engine and agent states
        try:
            engine.restore(snap)
        except Exception as e_restore_final:
            logger_t.error(
                "W%d D%d: Failed final engine restore: %s", worker_id, depth, e_restore_final
            )
            worker_stats.error_count += 1
        for j, a in enumerate(agent_states):
            a.close()
            agent_states[j] = agent_clones[j]  # swap in backup directly
        engine.free_snapshot(snap)

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
            logger_t.error(
                "W%d D%d: Failed to save engine state for opponent: %s",
                worker_id, depth, e_save,
            )
            worker_stats.error_count += 1
            return node_value

        try:
            agent_clones = [a.clone() for a in agent_states]
        except Exception as e_clone:
            logger_t.error(
                "W%d D%d: Failed to clone agent states for opponent: %s",
                worker_id, depth, e_clone,
            )
            worker_stats.error_count += 1
            engine.free_snapshot(snap)
            return node_value

        apply_ok = False
        try:
            engine.apply_action(chosen_action_idx)
            for a in agent_states:
                a.update(engine)
            apply_ok = True
        except Exception as e_apply:
            logger_t.error(
                "W%d D%d: Error applying sampled action %d: %s",
                worker_id, depth, chosen_action_idx, e_apply,
            )
            worker_stats.error_count += 1

        if apply_ok:
            try:
                node_value = _deep_traverse_go(
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
            except Exception as e_recurse:  # JUSTIFIED: worker resilience
                logger_t.error(
                    "W%d D%d: Recursion error after sampled action %d: %s",
                    worker_id, depth, chosen_action_idx, e_recurse, exc_info=True,
                )
                worker_stats.error_count += 1

        # Restore engine and agent states
        try:
            engine.restore(snap)
        except Exception as e_restore:
            logger_t.error(
                "W%d D%d: Failed to restore engine after opponent: %s",
                worker_id, depth, e_restore,
            )
            worker_stats.error_count += 1
        for j, a in enumerate(agent_states):
            a.close()
            agent_states[j] = agent_clones[j]  # swap in backup directly
        engine.free_snapshot(snap)

        return node_value


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
) -> np.ndarray:
    """
    Recursive Outcome Sampling traversal for Deep CFR using the Go engine backend.

    Uses GoEngine (save/restore) and GoAgentState (encode directly).

    At ALL nodes (both traverser and opponent): sample ONE action using exploration policy
    q(a|h) = epsilon * uniform + (1-epsilon) * sigma(a|h), then apply importance sampling
    correction to compute regrets.

    Returns utility vector (shape (2,) float64) for both players.
    """
    logger_t = logging.getLogger(__name__)

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
            logger_t.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
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
        logger_t.error("W%d D%d: Error checking terminal: %s", worker_id, depth, e_term)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Depth limit check (system recursion limit)
    if depth >= config.system.recursion_limit:
        logger_t.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
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
        logger_t.error(
            "W%d D%d: Error getting legal action mask: %s", worker_id, depth, e_legal
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    legal_indices = np.where(legal_mask > 0)[0]
    num_actions = len(legal_indices)

    if num_actions == 0:
        logger_t.error("W%d D%d: No legal actions but non-terminal!", worker_id, depth)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Infer decision context from legal mask
    current_context = _infer_decision_context(legal_mask)

    # Get acting player
    try:
        player = engine.acting_player()
    except Exception as e_player:
        logger_t.error(
            "W%d D%d: Error getting acting player: %s", worker_id, depth, e_player
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Get drawn card bucket for POST_DRAW encoding
    drawn_bucket = -1
    if current_context == 1:  # CtxPostDraw
        drawn_bucket = engine.get_drawn_card_bucket()

    # Encode infoset using Go agent
    try:
        features = agent_states[player].encode(current_context, drawn_bucket=drawn_bucket)
        action_mask = legal_mask.copy()
    except Exception as e_encode:
        logger_t.error(
            "W%d D%d: Error encoding infoset: %s", worker_id, depth, e_encode
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy_full = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger_t.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy_full = None
        except Exception as e_net:  # JUSTIFIED: worker resilience
            logger_t.warning(
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
    # Compute exploration policy: q(a) = epsilon * uniform + (1-epsilon) * sigma(a)
    uniform_prob = 1.0 / num_actions
    exploration_policy = (
        exploration_epsilon * uniform_prob + (1.0 - exploration_epsilon) * local_strategy
    )

    # Normalize
    total_prob = exploration_policy.sum()
    if total_prob > 1e-9:
        exploration_policy /= total_prob
    else:
        exploration_policy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Sample ONE action from exploration policy
    try:
        chosen_local_idx = np.random.choice(num_actions, p=exploration_policy)
    except ValueError:
        chosen_local_idx = np.random.choice(num_actions)
        worker_stats.warning_count += 1

    chosen_action_idx = int(legal_indices[chosen_local_idx])
    sampling_prob = exploration_policy[chosen_local_idx]

    # Save engine state and clone agent states
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    try:
        snap = engine.save()
    except Exception as e_save:
        logger_t.error(
            "W%d D%d: Failed to save engine state: %s", worker_id, depth, e_save
        )
        worker_stats.error_count += 1
        return node_value

    try:
        agent_clones = [a.clone() for a in agent_states]
    except Exception as e_clone:
        logger_t.error(
            "W%d D%d: Failed to clone agent states: %s", worker_id, depth, e_clone
        )
        worker_stats.error_count += 1
        engine.free_snapshot(snap)
        return node_value

    apply_ok = False
    try:
        engine.apply_action(chosen_action_idx)
        for a in agent_states:
            a.update(engine)
        apply_ok = True
    except Exception as e_apply:
        logger_t.error(
            "W%d D%d: Error applying sampled action %d: %s",
            worker_id, depth, chosen_action_idx, e_apply,
        )
        worker_stats.error_count += 1

    if apply_ok:
        try:
            node_value = _deep_traverse_os_go(
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
            )
        except Exception as e_recurse:  # JUSTIFIED: worker resilience
            logger_t.error(
                "W%d D%d: Recursion error after sampled action %d: %s",
                worker_id, depth, chosen_action_idx, e_recurse, exc_info=True,
            )
            worker_stats.error_count += 1

    # Restore engine and agent states
    try:
        engine.restore(snap)
    except Exception as e_restore:
        logger_t.error(
            "W%d D%d: Failed to restore engine: %s", worker_id, depth, e_restore
        )
        worker_stats.error_count += 1
    for j, a in enumerate(agent_states):
        a.close()
        agent_states[j] = agent_clones[j]  # swap in backup directly
    engine.free_snapshot(snap)

    # --- Compute IS-corrected regrets and store samples ---
    if player == updating_player:
        # TRAVERSER: compute IS-weighted regrets
        sampled_utility = node_value[player]

        if sampling_prob > 1e-9:
            utility_estimate = sampled_utility / sampling_prob

            # Compute IS-corrected regrets
            regrets = np.zeros(num_actions, dtype=np.float64)
            for a_idx in range(num_actions):
                action_value_estimate = (
                    1.0 if a_idx == chosen_local_idx else 0.0
                ) * utility_estimate
                regrets[a_idx] = action_value_estimate - local_strategy[a_idx] * utility_estimate

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
    else:
        # OPPONENT: store strategy sample
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

    return node_value


def _deep_traverse_os(
    game_state: CambiaGameState,
    agent_states: List[AgentState],
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
) -> np.ndarray:
    """
    Recursive Outcome Sampling traversal for Deep CFR.

    At ALL nodes (both traverser and opponent): sample ONE action using exploration policy
    q(a|h) = epsilon * uniform + (1-epsilon) * sigma(a|h), then apply importance sampling
    correction to compute regrets.

    Returns utility vector for both players.
    """
    logger_t = logging.getLogger(__name__)

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
            logger_t.error("W%d D%d: Error putting progress: %s", worker_id, depth, pq_e)
            worker_stats.error_count += 1

    # Terminal check
    if game_state.is_terminal():
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.array(
            [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
        )

    # Depth limit check (system recursion limit)
    if depth >= config.system.recursion_limit:
        logger_t.error("W%d D%d: Max recursion depth reached.", worker_id, depth)
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Traversal depth cap (0 = unlimited)
    depth_limit = getattr(getattr(config, "deep_cfr", None), "traversal_depth_limit", 0)
    if depth_limit > 0 and depth >= depth_limit:
        has_bottomed_out_tracker[0] = True
        min_depth_after_bottom_out_tracker[0] = min(
            min_depth_after_bottom_out_tracker[0], float(depth)
        )
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Determine decision context
    if game_state.snap_phase_active:
        current_context = DecisionContext.SNAP_DECISION
    elif game_state.pending_action:
        pending = game_state.pending_action
        if isinstance(pending, ActionDiscard):
            current_context = DecisionContext.POST_DRAW
        elif isinstance(
            pending,
            (
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
            ),
        ):
            current_context = DecisionContext.ABILITY_SELECT
        elif isinstance(pending, ActionSnapOpponentMove):
            current_context = DecisionContext.SNAP_MOVE
        else:
            logger_t.warning(
                "W%d D%d: Unknown pending action type (%s).",
                worker_id,
                depth,
                type(pending).__name__,
            )
            worker_stats.warning_count += 1
            current_context = DecisionContext.START_TURN
    else:
        current_context = DecisionContext.START_TURN

    player = game_state.get_acting_player()
    if player == -1:
        logger_t.error("W%d D%d: Could not determine acting player.", worker_id, depth)
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    opponent = 1 - player

    # Get legal actions
    try:
        legal_actions_set = game_state.get_legal_actions()
        legal_actions = sorted(list(legal_actions_set), key=repr)
    except GameStateError as e_legal:
        logger_t.warning(
            "W%d D%d: Game state error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_legal:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error getting legal actions: %s",
            worker_id,
            depth,
            e_legal,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    num_actions = len(legal_actions)
    if num_actions == 0:
        if not game_state.is_terminal():
            logger_t.error(
                "W%d D%d: No legal actions but non-terminal!", worker_id, depth
            )
            worker_stats.error_count += 1
            return np.zeros(NUM_PLAYERS, dtype=np.float64)
        else:
            has_bottomed_out_tracker[0] = True
            min_depth_after_bottom_out_tracker[0] = min(
                min_depth_after_bottom_out_tracker[0], float(depth)
            )
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

    # Encode infoset and action mask
    current_agent_state = agent_states[player]

    # Get drawn card bucket for POST_DRAW encoding
    drawn_card_bucket = None
    if current_context == DecisionContext.POST_DRAW:
        drawn_card_obj = game_state.pending_action_data.get("drawn_card")
        if drawn_card_obj is not None:
            drawn_card_bucket = get_card_bucket(drawn_card_obj)

    try:
        features = encode_infoset(
            current_agent_state, current_context, drawn_card_bucket=drawn_card_bucket
        )
        action_mask = encode_action_mask(legal_actions)
    except (EncodingError, AgentStateError) as e_encode:
        logger_t.warning(
            "W%d D%d: Encoding/agent state error for infoset: %s",
            worker_id,
            depth,
            e_encode,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)
    except Exception as e_encode:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error encoding infoset/mask: %s",
            worker_id,
            depth,
            e_encode,
            exc_info=True,
        )
        worker_stats.error_count += 1
        return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # Compute strategy from advantage network
    if network is not None:
        try:
            strategy = _get_strategy_from_network(
                network, features, action_mask, _feat_buf, _mask_buf
            )
        except NetworkError as e_net:
            logger_t.warning(
                "W%d D%d: Network inference error: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        except Exception as e_net:  # JUSTIFIED: worker resilience - fallback to uniform strategy on unexpected errors
            logger_t.warning(
                "W%d D%d: Network inference failed: %s. Using uniform.",
                worker_id,
                depth,
                e_net,
            )
            worker_stats.warning_count += 1
            strategy = np.ones(num_actions, dtype=np.float64) / num_actions
    else:
        # No network weights yet (first iteration) - use uniform
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Map network strategy (NUM_ACTIONS) to local strategy (num_actions)
    if len(strategy) == NUM_ACTIONS:
        local_strategy = np.zeros(num_actions, dtype=np.float64)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            local_strategy[a_idx] = strategy[global_idx]
        total = local_strategy.sum()
        if total > 1e-9:
            local_strategy /= total
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        strategy = local_strategy

    # Ensure strategy length matches
    if len(strategy) != num_actions:
        logger_t.warning(
            "W%d D%d: Strategy len %d != num_actions %d. Using uniform.",
            worker_id,
            depth,
            len(strategy),
            num_actions,
        )
        worker_stats.warning_count += 1
        strategy = np.ones(num_actions, dtype=np.float64) / num_actions

    # --- Outcome Sampling Logic ---
    # Compute exploration policy: q(a) = epsilon * uniform + (1-epsilon) * sigma(a)
    uniform_prob = 1.0 / num_actions
    exploration_policy = (
        exploration_epsilon * uniform_prob + (1.0 - exploration_epsilon) * strategy
    )

    # Normalize just in case
    total_prob = exploration_policy.sum()
    if total_prob > 1e-9:
        exploration_policy /= total_prob
    else:
        exploration_policy = np.ones(num_actions, dtype=np.float64) / num_actions

    # Sample ONE action from exploration policy
    try:
        chosen_idx = np.random.choice(num_actions, p=exploration_policy)
    except ValueError:
        chosen_idx = np.random.choice(num_actions)
        worker_stats.warning_count += 1

    chosen_action = legal_actions[chosen_idx]
    sampling_prob = exploration_policy[chosen_idx]

    # Apply action, recurse, undo
    apply_success = False
    node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

    try:
        state_delta, undo_info = game_state.apply_action(chosen_action)
        if callable(undo_info):
            apply_success = True
        else:
            logger_t.error(
                "W%d D%d: apply_action for %s returned invalid undo.",
                worker_id,
                depth,
                chosen_action,
            )
            worker_stats.error_count += 1
    except ActionApplicationError as e_apply:
        logger_t.warning(
            "W%d D%d: Action application error for %s: %s",
            worker_id,
            depth,
            chosen_action,
            e_apply,
        )
        worker_stats.error_count += 1
    except Exception as e_apply:  # JUSTIFIED: worker resilience - workers must not crash the training pool
        logger_t.error(
            "W%d D%d: Error applying action %s: %s",
            worker_id,
            depth,
            chosen_action,
            e_apply,
            exc_info=True,
        )
        worker_stats.error_count += 1

    if apply_success:
        observation = _create_observation(
            None, chosen_action, game_state, player, game_state.snap_results_log
        )
        next_agent_states = []
        agent_update_failed = False

        if observation is None:
            logger_t.error(
                "W%d D%d: Failed to create observation after %s.",
                worker_id,
                depth,
                chosen_action,
            )
            worker_stats.error_count += 1
            agent_update_failed = True
            try:
                undo_info()
            except UndoFailureError:
                pass
            except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                pass
        else:
            try:
                for agent_idx, agent_state in enumerate(agent_states):
                    cloned_agent = agent_state.clone()
                    player_specific_obs = _filter_observation(observation, agent_idx)
                    cloned_agent.update(player_specific_obs)
                    next_agent_states.append(cloned_agent)
            except (AgentStateError, ObservationUpdateError) as e_update:
                logger_t.warning(
                    "W%d D%d: Agent state update error after %s: %s",
                    worker_id,
                    depth,
                    chosen_action,
                    e_update,
                )
                worker_stats.error_count += 1
                agent_update_failed = True
                try:
                    undo_info()
                except UndoFailureError:
                    pass
                except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                    pass
            except Exception as e_update:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                logger_t.error(
                    "W%d D%d: Error updating agents after %s: %s",
                    worker_id,
                    depth,
                    chosen_action,
                    e_update,
                    exc_info=True,
                )
                worker_stats.error_count += 1
                agent_update_failed = True
                try:
                    undo_info()
                except UndoFailureError:
                    pass
                except Exception:  # JUSTIFIED: worker resilience - must attempt cleanup even after undo errors
                    pass

        if not agent_update_failed:
            try:
                node_value = _deep_traverse_os(
                    game_state,
                    next_agent_states,
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
                )
            except TraversalError as e_recurse:
                logger_t.warning(
                    "W%d D%d: Traversal error in recursion after %s: %s",
                    worker_id,
                    depth,
                    chosen_action,
                    e_recurse,
                )
                worker_stats.error_count += 1
            except Exception as e_recurse:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                logger_t.error(
                    "W%d D%d: Recursion error after %s: %s",
                    worker_id,
                    depth,
                    chosen_action,
                    e_recurse,
                    exc_info=True,
                )
                worker_stats.error_count += 1

            try:
                undo_info()
            except UndoFailureError as e_undo:
                logger_t.error(
                    "W%d D%d: Undo failure for %s: %s. State corrupt.",
                    worker_id,
                    depth,
                    chosen_action,
                    e_undo,
                    exc_info=True,
                )
                worker_stats.error_count += 1
                return np.zeros(NUM_PLAYERS, dtype=np.float64)
            except Exception as e_undo:  # JUSTIFIED: worker resilience - must not crash on undo, state likely corrupt
                logger_t.error(
                    "W%d D%d: Error undoing %s: %s. State corrupt.",
                    worker_id,
                    depth,
                    chosen_action,
                    e_undo,
                    exc_info=True,
                )
                worker_stats.error_count += 1
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

    # --- Compute IS-corrected regrets and store samples ---
    if player == updating_player:
        # TRAVERSER: compute IS-weighted regrets
        sampled_utility = node_value[player]

        if sampling_prob > 1e-9:
            utility_estimate = sampled_utility / sampling_prob

            # Compute IS-corrected regrets
            regrets = np.zeros(num_actions, dtype=np.float64)
            for a_idx in range(num_actions):
                action_value_estimate = (
                    1.0 if a_idx == chosen_idx else 0.0
                ) * utility_estimate
                regrets[a_idx] = action_value_estimate - strategy[a_idx] * utility_estimate

            # Build full-size regret target vector (NUM_ACTIONS)
            regret_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for a_idx, action in enumerate(legal_actions):
                global_idx = action_to_index(action)
                regret_target[global_idx] = regrets[a_idx]

            # Store advantage sample
            advantage_samples.append(
                ReservoirSample(
                    features=features.astype(np.float32),
                    target=regret_target,
                    action_mask=action_mask.astype(np.bool_),
                    iteration=iteration,
                )
            )
    else:
        # OPPONENT: store strategy sample (same as ES)
        strategy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a_idx, action in enumerate(legal_actions):
            global_idx = action_to_index(action)
            strategy_target[global_idx] = strategy[a_idx]

        strategy_samples.append(
            ReservoirSample(
                features=features.astype(np.float32),
                target=strategy_target,
                action_mask=action_mask.astype(np.bool_),
                iteration=iteration,
            )
        )

    return node_value


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
            "Deep CFR Worker %d logging initialized (dir: %s).", worker_id, worker_log_dir
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

                advantage_network = AdvantageNetwork(
                    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                    validate_inputs=network_config.get("validate_inputs", True),
                )
                weights_tensors = {
                    k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in network_weights_serialized.items()
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
            _feat_buf = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
            _mask_buf = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)

        # Alternate updating player each iteration
        updating_player = iteration % NUM_PLAYERS
        min_depth_after_bottom_out_tracker = [float("inf")]
        has_bottomed_out_tracker = [False]

        deep_cfr_cfg = getattr(config, "deep_cfr", None)
        use_go = getattr(deep_cfr_cfg, "engine_backend", "python") == "go"

        if use_go:
            # --- Go engine backend ---
            from ..ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415

            go_engine = None
            go_agents = []
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

            try:
                sampling_method = getattr(config.deep_cfr, "sampling_method", "external")
                if sampling_method == "outcome":
                    exploration_epsilon = getattr(config.deep_cfr, "exploration_epsilon", 0.6)
                    final_utility_value = _deep_traverse_os_go(
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
                for a in go_agents:
                    a.close()
                go_engine.close()

        else:
            # --- Python engine backend ---
            # Initialize game state
            try:
                game_state = CambiaGameState(house_rules=config.cambia_rules)
            except GameStateError as game_init_e:
                if logger_instance:
                    logger_instance.warning(
                        "W%d Iter %d: Game state initialization error: %s",
                        worker_id,
                        iteration,
                        game_init_e,
                    )
                worker_stats.error_count += 1
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                )
            except Exception as game_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Failed GameState init: %s",
                        worker_id,
                        iteration,
                        game_init_e,
                        exc_info=True,
                    )
                worker_stats.error_count += 1
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                )

            # Initialize agent states
            initial_agent_states = []
            if not game_state.is_terminal():
                try:
                    initial_obs = _create_observation(None, None, game_state, -1, [])
                    if initial_obs is None:
                        raise ValueError("Failed to create initial observation.")

                    initial_hands = [list(p.hand) for p in game_state.players]
                    initial_peeks = [p.initial_peek_indices for p in game_state.players]
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
                except (AgentStateError, ObservationUpdateError, EncodingError) as agent_init_e:
                    if logger_instance:
                        logger_instance.warning(
                            "W%d Iter %d: Agent state initialization error: %s",
                            worker_id,
                            iteration,
                            agent_init_e,
                        )
                    worker_stats.error_count += 1
                    return DeepCFRWorkerResult(
                        stats=worker_stats,
                        simulation_nodes=simulation_nodes_this_sim,
                    )
                except Exception as agent_init_e:  # JUSTIFIED: worker resilience - workers must not crash the training pool
                    if logger_instance:
                        logger_instance.error(
                            "W%d Iter %d: Failed AgentStates init: %s",
                            worker_id,
                            iteration,
                            agent_init_e,
                            exc_info=True,
                        )
                    worker_stats.error_count += 1
                    return DeepCFRWorkerResult(
                        stats=worker_stats,
                        simulation_nodes=simulation_nodes_this_sim,
                    )
            else:
                if logger_instance:
                    logger_instance.warning(
                        "W%d Iter %d: Game terminal at init.",
                        worker_id,
                        iteration,
                    )
                final_utility = np.array(
                    [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
                )
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                    final_utility=final_utility.tolist(),
                )

            if len(initial_agent_states) != NUM_PLAYERS:
                if logger_instance:
                    logger_instance.error(
                        "W%d Iter %d: Incorrect agent states (%d).",
                        worker_id,
                        iteration,
                        len(initial_agent_states),
                    )
                worker_stats.error_count += 1
                return DeepCFRWorkerResult(
                    stats=worker_stats,
                    simulation_nodes=simulation_nodes_this_sim,
                )

            # Run traversal (external or outcome sampling based on config)
            sampling_method = config.deep_cfr.sampling_method
            if sampling_method == "outcome":
                # Run outcome sampling traversal
                exploration_epsilon = config.deep_cfr.exploration_epsilon
                final_utility_value = _deep_traverse_os(
                    game_state=game_state,
                    agent_states=initial_agent_states,
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
                )
            else:
                # Run external sampling traversal (default)
                final_utility_value = _deep_traverse(
                    game_state=game_state,
                    agent_states=initial_agent_states,
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
                "W%d Iter %d: Traversal complete. Adv samples: %d, Strat samples: %d, Nodes: %d",
                worker_id,
                iteration,
                len(advantage_samples),
                len(strategy_samples),
                worker_stats.nodes_visited,
            )

        return DeepCFRWorkerResult(
            advantage_samples=advantage_samples,
            strategy_samples=strategy_samples,
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
            final_utility=final_utility_value.tolist(),
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
            stats=worker_stats,
            simulation_nodes=simulation_nodes_this_sim,
        )
    except Exception as e_inner:  # JUSTIFIED: worker resilience - top-level worker catch to prevent pool crash
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
