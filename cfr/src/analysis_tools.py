"""src/analysis_tools.py"""

# WARNING: LEGACY TABULAR CFR CODE
# This module implements best-response and exploitability analysis for TABULAR CFR only.
# It is NOT compatible with the Deep CFR pipeline (neural network checkpoints).
# For Deep CFR evaluation, use evaluate_agents.py and es_validator.py.
#
# cambia-1428 (Python engine retirement): the recursive best-response tree
# search below runs on the Go engine. Only the rules substrate moved: the
# search still keys the opponent's average strategy by InfosetKey built from
# the Python AgentState, because src.cfr.worker writes that table with the same
# belief machinery during training and a key rebuilt off GoAgentState would
# silently look up different entries. src.cfr.br_state.GoBrState is the
# adapter -- it deals a GoEngine, branches under checkpoint/rewind instead of
# the retired apply/undo pair, and rebuilds the AgentObservation stream off the
# engine's eval surface. Pool workers cannot receive an engine handle across a
# fork, so a parallel branch travels as (deal spec, action prefix, action) and
# the worker replays it onto its own engine.

import logging
import json
import os
import copy
import queue
import random
import time
import threading
import multiprocessing
import multiprocessing.pool
import traceback
from typing import Any, Optional, List, Tuple
from dataclasses import asdict, is_dataclass
import numpy as np

from .card import Card
from .agent_state import AgentState, AgentObservation
from .constants import (
    ActionReplace,
    ActionDrawStockpile,
    GameAction,
    DecisionContext,
    NUM_PLAYERS,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)
from .config import Config
from .utils import InfosetKey, PolicyDict, normalize_probabilities, SimulationTrace
from .cfr.br_state import DealSpec, GoBrState

from .cfr.exceptions import (
    GracefulShutdownException,
    GameStateError,
    AgentStateError,
    ObservationUpdateError,
)

# Conditional imports for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .live_display import LiveDisplayManager

logger = logging.getLogger(__name__)


# --- Top-level functions for parallel BR calculation ---


def _br_action_worker(
    house_rules: Any,
    deal: DealSpec,
    action_prefix: Tuple[int, ...],
    action_idx: int,
    br_agent_state_copy: AgentState,
    opp_view_agent_state_copy: AgentState,
    action: GameAction,
    opponent_avg_strategy: PolicyDict,
    br_player: int,
    depth: int,
) -> float:
    """
    Target function for the BR pool. Applies one action and calls node logic.

    The node arrives as (deal spec, action prefix) rather than as a copy of the
    game: an engine handle cannot cross a fork, so this rebuilds its own engine
    and replays the prefix onto it. The replay is exact -- the deal is a pure
    function of the spec and the engine is deterministic given an action
    sequence -- so the worker starts from the identical state its parent was on.
    """
    # No direct access to the main shutdown event here.
    # Relies on the pool being terminated if shutdown is triggered.
    state: Optional[GoBrState] = None
    try:
        state = GoBrState.new(house_rules, deal, action_prefix)
        state.apply(action_idx)

        # Create observation AFTER action
        obs_after_action = state.observation(action, br_agent_state_copy.player_id)

        # Update agent states using static helpers
        br_obs_filtered = AnalysisTools._filter_observation_for_br(
            obs_after_action, br_player
        )
        opp_view_obs_filtered = AnalysisTools._filter_observation_for_br(
            obs_after_action, br_agent_state_copy.opponent_id
        )
        br_agent_state_copy.update(br_obs_filtered)
        opp_view_agent_state_copy.update(opp_view_obs_filtered)

        # Recursive call to node logic (serially within this worker)
        action_value = AnalysisTools._best_response_node_logic(
            state,
            opponent_avg_strategy,
            br_player,
            br_agent_state_copy,
            opp_view_agent_state_copy,
            depth + 1,
            pool=None,  # No pool for recursive calls within worker
        )
        return action_value
    except GameStateError as e:
        logger.error(
            "BR ActionWorker(D%d, P%d): Game state error processing action %s: %s",
            depth,
            br_player,
            action,
            e,
        )
        return -float("inf")  # Indicate failure
    except (AgentStateError, ObservationUpdateError) as e:
        logger.error(
            "BR ActionWorker(D%d, P%d): Agent state error processing action %s: %s",
            depth,
            br_player,
            action,
            e,
        )
        return -float("inf")  # Indicate failure
    except Exception as e:  # JUSTIFIED: BR calculation resilience
        # Log error from within the worker process
        logger.error(
            "BR ActionWorker(D%d, P%d): Error processing action %s: %s\n%s",
            depth,
            br_player,
            action,
            e,
            traceback.format_exc(),
            exc_info=False,  # Keep log cleaner
        )
        return -float("inf")  # Indicate failure
    finally:
        # The FFI handle pool is finite and this worker process is reused for
        # every task the pool hands it, so the engine has to go back now.
        if state is not None:
            state.close()


def _best_response_recursive_entry(
    game_state: GoBrState,
    opponent_avg_strategy: PolicyDict,
    br_player: int,
    br_agent_state: AgentState,
    opp_view_agent_state: AgentState,
    pool: Optional[multiprocessing.pool.Pool],  # Pool for parallelizing actions
    depth: int = 0,
) -> float:
    """Entry point for the recursive best response calculation."""
    return AnalysisTools._best_response_node_logic(
        game_state,
        opponent_avg_strategy,
        br_player,
        br_agent_state,
        opp_view_agent_state,
        depth,
        pool,
    )


def build_br_start_state(
    config: Config,
    br_player: int,
    deal: DealSpec,
) -> Tuple[GoBrState, AgentState, AgentState]:
    """Deal the root of one best-response search and both belief states.

    Returns the engine-backed state plus the BR seat's belief and the BR seat's
    model of the opponent's belief, both initialized from the pre-first-action
    frame. Split out of _run_br_calculation_process so the cross-engine equality
    harness can start a search on a pinned deal without a process and a queue.

    The caller owns the returned state and must close it: the FFI handle pool is
    finite.
    """
    opponent_player = 1 - br_player
    state = GoBrState.new(config.cambia_rules, deal)
    try:
        initial_obs = state.initial_observation()
        peeks = state.initial_peek_indices()

        br_agent_state = AgentState(
            player_id=br_player,
            opponent_id=opponent_player,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(state.hand(br_player)),
            config=config,
        )
        br_agent_state.initialize(initial_obs, state.hand(br_player), peeks)

        opp_view_agent_state = AgentState(
            player_id=opponent_player,
            opponent_id=br_player,
            memory_level=config.agent_params.memory_level,
            time_decay_turns=config.agent_params.time_decay_turns,
            initial_hand_size=len(state.hand(opponent_player)),
            config=config,
        )
        opp_view_agent_state.initialize(initial_obs, state.hand(opponent_player), peeks)
    except Exception:
        state.close()
        raise
    return state, br_agent_state, opp_view_agent_state


def _run_br_calculation_process(
    avg_strat: PolicyDict,
    config: Config,
    br_player: int,
    result_queue: multiprocessing.Queue,
    # Add shutdown_event (will be a copy, but can be checked)
    # Note: Modifying the *original* event won't work across processes.
    # This check is primarily for reacting if the main process signalled shutdown
    # *before* this process started its main work.
    shutdown_event_copy: Optional[threading.Event],
    deal: Optional[DealSpec] = None,
):
    """
    Function executed by each of the two main BR calculation processes.
    Sets up a local pool and calls the BR entry point. Includes basic error handling.

    The engine is dealt inside this process, after the pool is forked: an engine
    handle cannot cross a fork, and forking a process that has already started
    the Go runtime is not safe either. `deal` defaults to a fresh random seed,
    which is the unseeded per-process deal the Python-engine search had.
    """
    pool: Optional[multiprocessing.pool.Pool] = None  # Define pool here for finally block
    state: Optional[GoBrState] = None
    try:
        # --- Check for immediate shutdown ---
        if shutdown_event_copy is not None and shutdown_event_copy.is_set():
            logger.warning(
                "BR Process P%d: Shutdown detected at startup. Exiting.", br_player
            )
            result_queue.put((br_player, float("inf")))  # Indicate failure/abort
            return

        # Basic logging setup for this process (optional, could log to specific file)
        # configure_logging_for_process(f"br_process_p{br_player}") # Example

        logger.info("BR Process P%d: Starting calculation...", br_player)
        exploit_workers = config.analysis.exploitability_num_workers
        logger.info(
            "BR Process P%d: Using %d pool workers for action evaluation.",
            br_player,
            exploit_workers,
        )

        # The pool is forked BEFORE the engine is dealt, deliberately: a worker
        # forked from a process that has already loaded libcambia would inherit
        # a Go runtime it cannot safely use.
        if exploit_workers > 1:
            pool = multiprocessing.Pool(processes=exploit_workers)
            logger.debug("BR Process P%d: Worker pool created.", br_player)

        # Initialize game state and agent states *within this process*
        if deal is None:
            deal = DealSpec(seed=random.getrandbits(63))
        state, br_agent_state, opp_view_agent_state = build_br_start_state(
            config, br_player, deal
        )
        logger.debug("BR Process P%d: Game and Agent states initialized.", br_player)

        # Call the entry point for BR calculation
        br_value = _best_response_recursive_entry(
            state,
            avg_strat,
            br_player,
            br_agent_state,
            opp_view_agent_state,
            pool,  # Pass the pool
            depth=0,
        )
        logger.info(
            "BR Process P%d: Calculation complete. Value: %.6f", br_player, br_value
        )

        # Put result on the queue
        result_queue.put((br_player, br_value))

    except GameStateError as e:
        logger.error(
            "!!! BR Process P%d: Game state error during calculation: %s",
            br_player,
            e,
        )
        try:
            result_queue.put((br_player, float("inf")))
        except Exception as q_err:  # JUSTIFIED: BR calculation resilience
            logger.error(
                "BR Process P%d: Failed to put error signal on queue: %s",
                br_player,
                q_err,
            )
    except (AgentStateError, ObservationUpdateError) as e:
        logger.error(
            "!!! BR Process P%d: Agent state error during calculation: %s",
            br_player,
            e,
        )
        try:
            result_queue.put((br_player, float("inf")))
        except Exception as q_err:  # JUSTIFIED: BR calculation resilience
            logger.error(
                "BR Process P%d: Failed to put error signal on queue: %s",
                br_player,
                q_err,
            )
    except Exception as e:  # JUSTIFIED: BR calculation resilience
        # Log the error from within the BR process
        logger.error(
            "!!! BR Process P%d: Unhandled exception during calculation: %s\n%s",
            br_player,
            e,
            traceback.format_exc(),  # Log full traceback from process
            exc_info=False,  # Prevent duplicate logging by root logger if propagated
        )
        try:
            # Attempt to put failure signal on queue
            result_queue.put((br_player, float("inf")))
        except Exception as q_err:  # JUSTIFIED: BR calculation resilience
            logger.error(
                "BR Process P%d: Failed to put error signal on queue: %s",
                br_player,
                q_err,
            )
    finally:
        # Clean up the local pool associated with *this* BR process
        if pool:
            try:
                logger.debug("BR Process P%d: Closing worker pool...", br_player)
                pool.close()
                pool.join()
                logger.debug("BR Process P%d: Worker pool closed and joined.", br_player)
            except Exception as e_pool_close:  # JUSTIFIED: BR calculation resilience
                logger.error(
                    "BR Process P%d: Error closing pool: %s", br_player, e_pool_close
                )
        if state is not None:
            state.close()
        logger.info("BR Process P%d: Exiting.", br_player)


# Helper function for default serialization in JSON dump
def default_serializer(obj):
    """Default JSON serializer for objects not directly serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.void)):
        return None
    if isinstance(obj, InfosetKey):
        return obj.astuple()  # Use the tuple representation
    if isinstance(obj, Card):
        # Inlined from the retired src.game.helpers.serialize_card (identical
        # behavior): obj is already known non-None here.
        return str(obj)
    # Handle GameAction NamedTuples and other dataclasses explicitly
    if hasattr(obj, "_asdict") and callable(obj._asdict):  # Check for NamedTuple
        action_dict = obj._asdict()
        serialized_dict = {}
        for k, v in action_dict.items():
            # Recursive call for nested objects/cards
            serialized_dict[k] = default_serializer(v)
        return {type(obj).__name__: serialized_dict}  # Include type name
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)  # Use dataclasses.asdict for regular dataclasses
    # Fallback for other types
    try:
        # If it's a simple object instance, return its class name
        if hasattr(obj, "__class__") and not isinstance(obj, type):
            return obj.__class__.__name__
        return str(obj)  # Try string representation
    except TypeError:
        return repr(obj)  # Final fallback


class AnalysisTools:
    """Provides tools for analyzing CFR training progress and game history."""

    def __init__(
        self,
        config: Config,
        log_dir: Optional[str] = None,
        log_file_prefix: Optional[str] = None,
    ):
        self.config = config
        self.delta_log_file_path = None  # (Deprecated?) Path for detailed delta logs
        self.simulation_trace_log_path = None  # Path for simulation traces

        if log_dir and log_file_prefix:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Setup path for the delta log file (keep for now if needed elsewhere)
                self.delta_log_file_path = os.path.join(
                    log_dir, f"{log_file_prefix}_game_deltas.jsonl"
                )
                # Setup path for the new simulation trace log file
                sim_trace_prefix = getattr(
                    config.logging,
                    "simulation_trace_filename_prefix",
                    "simulation_traces",
                )
                self.simulation_trace_log_path = os.path.join(
                    log_dir, f"{sim_trace_prefix}_simulation_traces.jsonl"
                )
                logger.info(
                    "AnalysisTools: Simulation trace log path: %s",
                    self.simulation_trace_log_path,
                )
            except OSError as e_mkdir:
                logger.error(
                    "AnalysisTools: Failed to create log directory '%s': %s",
                    log_dir,
                    e_mkdir,
                )
            except AttributeError as e_attr:
                logger.error(
                    "AnalysisTools: Missing config attribute for logging setup: %s",
                    e_attr,
                )
        else:
            logger.warning(
                "AnalysisTools: Log directory/prefix not provided. Trace/Delta logging disabled."
            )

    def calculate_exploitability(
        self,
        average_strategy: PolicyDict,
        config: Config,
        live_display_manager: Optional["LiveDisplayManager"] = None,
        # Add shutdown_event from trainer
        shutdown_event: Optional[threading.Event] = None,
    ) -> float:
        """Calculates the exploitability of the agent's average strategy using parallel processes."""
        if not average_strategy:
            logger.warning("Cannot calculate exploitability: Average strategy is empty.")
            return float("inf")

        exploitability = float("inf")  # Default to infinity
        start_time = time.time()
        br_processes: List[multiprocessing.Process] = []  # Keep track of processes

        # Update display status
        if live_display_manager and hasattr(
            live_display_manager, "update_main_process_status"
        ):
            live_display_manager.update_main_process_status(
                "Calculating exploitability..."
            )

        try:
            logger.info("Starting parallel exploitability calculation...")
            # Use a standard multiprocessing Queue
            result_queue = multiprocessing.Queue()

            # Spawn two processes, one for each BR player
            for br_player in range(NUM_PLAYERS):
                logger.info(
                    "Spawning Best Response calculation process for Player %d...",
                    br_player,
                )
                # Pass the shutdown event (it will be a copy, but process can check initial state)
                process = multiprocessing.Process(
                    target=_run_br_calculation_process,
                    args=(
                        average_strategy,
                        config,
                        br_player,
                        result_queue,
                        shutdown_event,
                    ),
                    name=f"BR_Calc_P{br_player}",
                )
                br_processes.append(process)
                process.start()

            # Wait for both processes to finish and collect results
            results_dict = {}
            processes_completed = 0
            while processes_completed < NUM_PLAYERS:
                # Check shutdown event periodically while waiting
                if shutdown_event and shutdown_event.is_set():
                    logger.warning(
                        "Exploitability calculation interrupted by shutdown signal."
                    )
                    # Terminate running BR processes
                    for p in br_processes:
                        if p.is_alive():
                            logger.warning(
                                "Terminating BR process %s due to shutdown.", p.name
                            )
                            p.terminate()
                    raise GracefulShutdownException(
                        "Shutdown during exploitability calculation"
                    )

                try:
                    # Use timeout to allow periodic checks
                    p_id, value = result_queue.get(timeout=0.5)  # Shorter timeout
                    results_dict[p_id] = value
                    processes_completed += 1
                    logger.info("BR Process P%d finished. Result: %.6f", p_id, value)
                except queue.Empty:
                    # Check if any process terminated unexpectedly ONLY if not shutting down
                    if not (shutdown_event and shutdown_event.is_set()):
                        for i, p in enumerate(br_processes):
                            # Check if process is dead AND we haven't received its result yet
                            if not p.is_alive() and i not in results_dict:
                                logger.error(
                                    "BR Process P%d terminated unexpectedly (exit code %s). Assigning inf.",
                                    i,
                                    p.exitcode,
                                )
                                results_dict[i] = float("inf")
                                processes_completed += 1  # Mark as completed (failed)
                    continue  # Continue waiting or checking shutdown/dead processes
                except Exception as q_get_err:
                    logger.error("Error getting result from BR queue: %s", q_get_err)
                    # Treat as failure? For now, continue waiting/checking.

            # Join processes after collecting results or shutdown
            logger.debug("Joining BR processes...")
            for process in br_processes:
                try:
                    process.join(timeout=5.0)  # Keep timeout for process join
                    if process.is_alive():
                        logger.warning(
                            "Process %s did not join within timeout. Terminating.",
                            process.name,
                        )
                        process.terminate()
                        process.join(timeout=1.0)  # Wait briefly after terminate
                except Exception as e_join:
                    logger.error("Error joining BR process %s: %s", process.name, e_join)
            logger.debug("BR processes joined.")

            # Calculate final exploitability
            br_value_p0 = results_dict.get(0, float("inf"))
            br_value_p1 = results_dict.get(1, float("inf"))

            if br_value_p0 == float("inf") or br_value_p1 == float("inf"):
                logger.warning(
                    "Exploitability calculation resulted in infinity (BR failed/aborted for at least one player)."
                )
                exploitability = float("inf")
            else:
                exploitability = (br_value_p0 + br_value_p1) / 2.0
                logger.info("Calculated Exploitability: %.6f", exploitability)

        except GracefulShutdownException:
            logger.warning("Exploitability calculation aborted due to shutdown signal.")
            exploitability = float("inf")  # Mark as aborted/failed
            # Ensure any remaining processes are cleaned up if possible
            for p in br_processes:
                if p.is_alive():
                    try:
                        p.terminate()
                    except Exception as e:  # JUSTIFIED: BR calculation resilience
                        logger.debug("Error terminating BR process during cleanup: %s", e)
                    try:
                        p.join(0.5)
                    except Exception as e:  # JUSTIFIED: BR calculation resilience
                        logger.debug("Error joining BR process during cleanup: %s", e)
        except GameStateError as e_exploit:
            logger.error(
                "Game state error during exploitability calculation: %s",
                e_exploit,
            )
            exploitability = float("inf")  # Indicate error
        except (AgentStateError, ObservationUpdateError) as e_exploit:
            logger.error(
                "Agent state error during exploitability calculation: %s",
                e_exploit,
            )
            exploitability = float("inf")  # Indicate error
        except Exception as e_exploit:  # JUSTIFIED: BR calculation resilience
            logger.exception(
                "Error during parallel exploitability calculation setup/coordination: %s",
                e_exploit,
            )
            exploitability = float("inf")  # Indicate error
        finally:
            # Ensure status is reset even on error/shutdown
            if live_display_manager and hasattr(
                live_display_manager, "update_main_process_status"
            ):
                live_display_manager.update_main_process_status("Idle / Waiting...")
            logger.info(
                "Total Exploitability calculation attempt time: %.2f seconds",
                time.time() - start_time,
            )

        return exploitability

    @staticmethod
    def _best_response_node_logic(
        game_state: GoBrState,
        opponent_avg_strategy: PolicyDict,
        br_player: int,
        br_agent_state: AgentState,
        opp_view_agent_state: AgentState,
        depth: int,
        pool: Optional[multiprocessing.pool.Pool],  # Pool for parallelizing actions
    ) -> float:
        """Recursive logic for a node in the Best Response calculation.

        Branching is checkpoint / apply / rewind rather than the retired
        apply/undo pair: the Go engine has no in-process undo, so the node takes
        one checkpoint and rewinds to it after each child. Belief rewinds by
        cloning the Python AgentState per branch, exactly as before.
        """
        try:
            if game_state.is_terminal():
                return game_state.utility(br_player)

            acting_player = game_state.acting_player()
            if acting_player == -1:
                logger.error(
                    "BR NodeLogic(D%d): Invalid acting player at prefix %s.",
                    depth,
                    game_state.action_prefix,
                )
                return 0.0

            opponent_player = 1 - br_player
            # Sorted by repr: src.cfr.worker indexes each stored strategy vector
            # in that order, so reading one in the engine's ascending index
            # order would read the probability of a different action.
            legal_pairs: List[Tuple[int, GameAction]] = game_state.legal_actions()
            num_actions = len(legal_pairs)

            if num_actions == 0:
                if not game_state.is_terminal():
                    logger.error(
                        "BR NodeLogic(D%d): No legal actions but non-terminal at "
                        "prefix %s!",
                        depth,
                        game_state.action_prefix,
                    )
                return game_state.utility(br_player)  # Return current utility

            current_context = game_state.decision_context()

            # --- Node Logic ---
            if acting_player == br_player:
                # Maximize value for BR player

                # Parallel execution for BR player's actions
                if pool and num_actions > 1:
                    tasks = []
                    deal = game_state.deal
                    prefix = game_state.action_prefix
                    house_rules = game_state.house_rules
                    for action_idx, action in legal_pairs:
                        # The worker rebuilds the node from (deal, prefix): an
                        # engine handle cannot cross a fork. Only the belief
                        # states travel, and those are plain Python objects.
                        try:
                            br_agent_state_copy = br_agent_state.clone()
                            opp_view_agent_state_copy = opp_view_agent_state.clone()
                        except Exception as e_copy:
                            logger.error(
                                "BR NodeLogic(D%d): Error cloning belief states for "
                                "parallel task: %s",
                                depth,
                                e_copy,
                            )
                            pool = None  # Fallback to serial
                            break

                        tasks.append(
                            (
                                house_rules,
                                deal,
                                prefix,
                                action_idx,
                                br_agent_state_copy,
                                opp_view_agent_state_copy,
                                action,
                                opponent_avg_strategy,
                                br_player,
                                depth,  # Pass depth for logging within worker
                            )
                        )

                    if pool:  # Check if pool wasn't disabled by clone error
                        try:
                            results = pool.starmap(_br_action_worker, tasks)
                            valid_results = [r for r in results if r != -float("inf")]
                            if not valid_results:
                                return 0.0
                            return max(valid_results)
                        except Exception as e_starmap:
                            logger.error(
                                "BR NodeLogic(D%d): Error in pool.starmap: %s",
                                depth,
                                e_starmap,
                            )
                            pool = None  # Fallback to serial

                # Serial execution for BR player's actions
                max_value = -float("inf")
                actions_attempted_serially = 0
                checkpoint = game_state.checkpoint()
                try:
                    for action_idx, action in legal_pairs:
                        try:
                            game_state.apply(action_idx)
                        except Exception as e_apply:
                            logger.error(
                                "BR NodeLogic(D%d): Engine rejected BR action %s: %s",
                                depth,
                                action,
                                e_apply,
                            )
                            game_state.rewind(checkpoint)
                            continue

                        obs_after_action = game_state.observation(action, acting_player)

                        next_br_agent_state = br_agent_state.clone()
                        next_opp_view_agent_state = opp_view_agent_state.clone()
                        try:
                            br_obs_filtered = AnalysisTools._filter_observation_for_br(
                                obs_after_action, br_player
                            )
                            next_br_agent_state.update(br_obs_filtered)
                            opp_obs_filtered = AnalysisTools._filter_observation_for_br(
                                obs_after_action, opponent_player
                            )
                            next_opp_view_agent_state.update(opp_obs_filtered)
                        except (AgentStateError, ObservationUpdateError) as e_update:
                            logger.error(
                                "BR NodeLogic(D%d): Agent state error updating after "
                                "BR action %s: %s",
                                depth,
                                action,
                                e_update,
                            )
                            game_state.rewind(checkpoint)
                            continue
                        except (
                            Exception
                        ) as e_update:  # JUSTIFIED: BR calculation resilience
                            logger.error(
                                "BR NodeLogic(D%d): Error updating agent states after "
                                "BR action %s: %s",
                                depth,
                                action,
                                e_update,
                                exc_info=False,
                            )
                            game_state.rewind(checkpoint)
                            continue

                        action_value = AnalysisTools._best_response_node_logic(
                            game_state,
                            opponent_avg_strategy,
                            br_player,
                            next_br_agent_state,
                            next_opp_view_agent_state,
                            depth + 1,
                            pool=None,
                        )
                        game_state.rewind(checkpoint)
                        max_value = max(max_value, action_value)
                        actions_attempted_serially += 1
                finally:
                    game_state.release(checkpoint)

                if max_value == -float("inf") and actions_attempted_serially == 0:
                    return 0.0
                return max_value

            else:  # Opponent's turn (always serial)
                try:
                    base_infoset_tuple = opp_view_agent_state.get_infoset_key()
                    if not isinstance(base_infoset_tuple, tuple):
                        raise TypeError("Infoset key not tuple")
                    infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
                except AgentStateError as e_key:
                    logger.error(
                        "BR NodeLogic(D%d): Agent state error getting Opponent P%d "
                        "infoset key: %s",
                        depth,
                        acting_player,
                        e_key,
                    )
                    return 0.0
                except Exception as e_key:  # JUSTIFIED: BR calculation resilience
                    logger.error(
                        "BR NodeLogic(D%d): Error getting Opponent P%d infoset key: "
                        "%s. OppView State: %s",
                        depth,
                        acting_player,
                        e_key,
                        opp_view_agent_state,
                        exc_info=False,
                    )
                    return 0.0

                opponent_strategy = opponent_avg_strategy.get(infoset_key)
                strategy_was_missing = opponent_strategy is None
                dim_mismatch = False

                if opponent_strategy is None:
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                elif len(opponent_strategy) != num_actions:
                    logger.warning(
                        "BR NodeLogic(D%d): Dim mismatch Opp P%d strategy at OppView "
                        "key %s. Have %d, need %d. Using uniform.",
                        depth,
                        acting_player,
                        infoset_key,
                        len(opponent_strategy),
                        num_actions,
                    )
                    opponent_strategy = (
                        np.ones(num_actions) / num_actions
                        if num_actions > 0
                        else np.array([])
                    )
                    dim_mismatch = True

                expected_value = 0.0
                strategy_sum = (
                    opponent_strategy.sum() if opponent_strategy is not None else 0.0
                )

                if num_actions > 0 and strategy_sum > 1e-9:
                    if not np.isclose(strategy_sum, 1.0):
                        if not strategy_was_missing and not dim_mismatch:
                            pass
                        opponent_strategy = normalize_probabilities(opponent_strategy)
                        if len(opponent_strategy) == 0 or not np.isclose(
                            opponent_strategy.sum(), 1.0
                        ):
                            logger.error(
                                "BR NodeLogic(D%d): Failed to normalize opp strategy "
                                "for %s. Using uniform.",
                                depth,
                                infoset_key,
                            )
                            opponent_strategy = (
                                np.ones(num_actions) / num_actions
                                if num_actions > 0
                                else np.array([])
                            )

                    checkpoint = game_state.checkpoint()
                    try:
                        for i, (action_idx, action) in enumerate(legal_pairs):
                            action_prob = opponent_strategy[i]
                            if action_prob < 1e-9:
                                continue

                            try:
                                game_state.apply(action_idx)
                            except Exception as e_apply:
                                logger.error(
                                    "BR NodeLogic(D%d): Engine rejected Opponent "
                                    "action %s: %s",
                                    depth,
                                    action,
                                    e_apply,
                                )
                                game_state.rewind(checkpoint)
                                continue

                            obs_after_action = game_state.observation(
                                action, acting_player
                            )

                            next_br_agent_state = br_agent_state.clone()
                            next_opp_view_agent_state = opp_view_agent_state.clone()
                            try:
                                br_obs_filtered = (
                                    AnalysisTools._filter_observation_for_br(
                                        obs_after_action, br_player
                                    )
                                )
                                next_br_agent_state.update(br_obs_filtered)
                                opp_obs_filtered = (
                                    AnalysisTools._filter_observation_for_br(
                                        obs_after_action, opponent_player
                                    )
                                )
                                next_opp_view_agent_state.update(opp_obs_filtered)
                            except (AgentStateError, ObservationUpdateError) as e_update:
                                logger.error(
                                    "BR NodeLogic(D%d): Agent state error updating "
                                    "after Opp action %s: %s",
                                    depth,
                                    action,
                                    e_update,
                                )
                                game_state.rewind(checkpoint)
                                continue
                            except (
                                Exception
                            ) as e_update:  # JUSTIFIED: BR calculation resilience
                                logger.error(
                                    "BR NodeLogic(D%d): Error updating agent states "
                                    "after Opp action %s: %s",
                                    depth,
                                    action,
                                    e_update,
                                    exc_info=False,
                                )
                                game_state.rewind(checkpoint)
                                continue

                            recursive_value = AnalysisTools._best_response_node_logic(
                                game_state,
                                opponent_avg_strategy,
                                br_player,
                                next_br_agent_state,
                                next_opp_view_agent_state,
                                depth + 1,
                                pool=None,
                            )
                            game_state.rewind(checkpoint)
                            expected_value += action_prob * recursive_value
                    finally:
                        game_state.release(checkpoint)
                else:
                    return game_state.utility(br_player)

                return expected_value

        except GameStateError as e_br_rec:
            logger.error(
                "BR NodeLogic(D%d): Game state error in recursion: %s",
                depth,
                e_br_rec,
            )
            return 0.0
        except (AgentStateError, ObservationUpdateError) as e_br_rec:
            logger.error(
                "BR NodeLogic(D%d): Agent state error in recursion: %s",
                depth,
                e_br_rec,
            )
            return 0.0
        except Exception as e_br_rec:  # JUSTIFIED: BR calculation resilience
            logger.exception(
                "BR NodeLogic(D%d): Unhandled error in recursion at prefix %s: %s",
                depth,
                game_state.action_prefix,
                e_br_rec,
            )
            return 0.0

    # --- Python-engine helpers (tools/ only) ---
    #
    # The best-response search no longer calls these two: it reads the decision
    # context straight off the engine (engine/legal.go's DecisionCtx) and builds
    # its observations in GoBrState. They stay because the reduced-deck research
    # tools -- tools/tiny_solver.py's python backend, tools/tiny_probe.py,
    # tools/tiny_exploit.py -- key their infosets through them and still recurse
    # a Python CambiaGameState. They are duck-typed on that state rather than
    # importing it, so this module carries no dependency on the retiring engine.

    @staticmethod
    def _get_decision_context(game_state) -> Optional[DecisionContext]:
        """Determine DecisionContext from a Python CambiaGameState."""
        try:
            if game_state.snap_phase_active:
                return DecisionContext.SNAP_DECISION
            pending = game_state.pending_action
            if pending:
                # Use isinstance for type checking
                if isinstance(pending, ActionDiscard):
                    return DecisionContext.POST_DRAW
                if isinstance(
                    pending,
                    (
                        ActionAbilityPeekOwnSelect,
                        ActionAbilityPeekOtherSelect,
                        ActionAbilityBlindSwapSelect,
                        ActionAbilityKingLookSelect,
                        ActionAbilityKingSwapDecision,
                    ),
                ):
                    return DecisionContext.ABILITY_SELECT
                if isinstance(pending, ActionSnapOpponentMove):
                    return DecisionContext.SNAP_MOVE
                logger.warning(
                    "BR Context: Unknown pending action type: %s", type(pending).__name__
                )
                return DecisionContext.START_TURN  # Fallback
            if game_state.is_terminal():
                return DecisionContext.TERMINAL
            return DecisionContext.START_TURN
        except AttributeError as e_attr:
            logger.error(
                "Error determining decision context due to missing attribute: %s", e_attr
            )
            return None
        except Exception as e_ctx:
            logger.error("Error determining decision context: %s", e_ctx, exc_info=True)
            return None

    @staticmethod
    def _create_observation_for_br(
        game_state,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Observation for belief updates, off a Python CambiaGameState.

        GoBrState.observation is the engine-backed counterpart and the one the
        best-response search uses; this stays for the reduced-deck tools noted
        above, and the two must keep producing the same frame for the same
        position.
        """
        try:
            drawn_card_for_obs = None
            if isinstance(action, ActionDiscard):
                drawn_card_for_obs = game_state.get_discard_top()
            elif isinstance(action, ActionReplace):
                if acting_player != -1 and 0 <= action.target_hand_index < len(
                    game_state.players[acting_player].hand
                ):
                    drawn_card_for_obs = game_state.players[acting_player].hand[
                        action.target_hand_index
                    ]
                else:
                    logger.error(
                        "BR Create Obs: ActionReplace index %d invalid for actor %d hand size %d",
                        action.target_hand_index,
                        acting_player,
                        (
                            len(game_state.players[acting_player].hand)
                            if acting_player != -1
                            else -1
                        ),
                    )
            elif isinstance(action, ActionDrawStockpile):
                # Surface the freshly-drawn stockpile card to the actor at the
                # post-draw decision node. The card lives in pending_action_data
                # after the draw is applied (engine.py sets pending_action_player +
                # pending_action_data={"drawn_card": ...}); mirrors the X1 pkey path
                # in tools/tiny_solver.py. Guarded so production callers where the
                # pending state is absent get drawn_card=None (no behavior change).
                try:
                    if game_state.pending_action_player == acting_player:
                        drawn_card_for_obs = game_state.pending_action_data.get(
                            "drawn_card"
                        )
                except (AttributeError, KeyError):
                    # Pending state absent/None (production callers) -> drawn_card
                    # is simply unavailable. Narrow: real errors still propagate.
                    drawn_card_for_obs = None

            peeked_cards_for_obs = None

            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=drawn_card_for_obs,
                peeked_cards=peeked_cards_for_obs,
                snap_results=copy.deepcopy(game_state.snap_results_log),
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
            return obs
        except GameStateError as e_obs:
            logger.error("Game state error creating observation for BR: %s", e_obs)
            return None
        except Exception as e_obs:  # JUSTIFIED: BR calculation resilience
            logger.error("Error creating observation for BR: %s", e_obs, exc_info=True)
            return None

    @staticmethod
    def _filter_observation_for_br(
        obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """
        Filters observation for BR agent state updates.
        Keeps drawn_card info *if* the observer is the acting player
        and the action was DrawStockpile/Replace/Discard. The DrawStockpile case
        lets the actor's perfect-recall stream carry the freshly-drawn card at the
        post-draw decision node; the opponent never sees it.
        """
        filtered_obs = copy.copy(obs)

        if obs.drawn_card and obs.acting_player == observer_id:
            if not isinstance(
                obs.action, (ActionDiscard, ActionReplace, ActionDrawStockpile)
            ):
                filtered_obs.drawn_card = None
        elif obs.drawn_card and obs.acting_player != observer_id:
            filtered_obs.drawn_card = None

        filtered_obs.peeked_cards = None

        return filtered_obs

    # Simulation Trace Logging

    def log_simulation_trace(self, trace_data: SimulationTrace):
        """Logs the detailed trace of a worker simulation to a JSON Lines file."""
        if not self.simulation_trace_log_path:
            if getattr(self.config.logging, "log_simulation_traces", False):
                logger.warning(
                    "Simulation trace logging enabled but path not set. Skipping."
                )
            return

        try:
            if (
                not isinstance(trace_data, dict)
                or "metadata" not in trace_data
                or "history" not in trace_data
            ):
                logger.warning(
                    "Attempted to log invalid simulation trace data structure: %s",
                    trace_data,
                )
                return

            with open(self.simulation_trace_log_path, "a", encoding="utf-8") as f:
                json_record = json.dumps(trace_data, default=default_serializer)
                f.write(json_record + "\n")
        except IOError as e_io:
            logger.error(
                "Error writing simulation trace to %s: %s",
                self.simulation_trace_log_path,
                e_io,
            )
        except TypeError as e_type:
            logger.error(
                "Error serializing simulation trace details to JSON: %s.", e_type
            )
            try:
                problematic_part = {
                    k: repr(v)[:200] for k, v in trace_data.get("metadata", {}).items()
                }
                problematic_part["history_len"] = len(trace_data.get("history", []))
                logger.debug("Problematic trace data (repr): %s", problematic_part)
            except Exception as e:  # JUSTIFIED: BR calculation resilience
                logger.debug("Error creating debug trace data: %s", e)
        except Exception as e_log:  # JUSTIFIED: BR calculation resilience
            logger.error(
                "Unexpected error logging simulation trace: %s", e_log, exc_info=True
            )
