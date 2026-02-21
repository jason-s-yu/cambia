"""
src/cfr/es_validator.py

ES Validation — runs short-depth External Sampling traversals to measure
exploitability metrics during Deep CFR training.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..agent_state import AgentState
from ..config import Config
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
from ..encoding import INPUT_DIM, NUM_ACTIONS, encode_infoset, encode_action_mask, action_to_index
from ..abstraction import get_card_bucket
from ..game.engine import CambiaGameState
from ..networks import AdvantageNetwork, get_strategy_from_advantages
from ..reservoir import ReservoirSample

# Re-use observation helpers from tabular worker
from .worker import _create_observation, _filter_observation

logger = logging.getLogger(__name__)


def _compute_entropy(strategy: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    pos = strategy[strategy > 1e-10]  # avoid log(0)
    if len(pos) == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


def _get_strategy_from_network(
    network: AdvantageNetwork,
    features: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute strategy from advantage network.

    Returns numpy array of shape (NUM_ACTIONS,) with strategy probabilities.
    """
    with torch.no_grad():
        features_t = torch.from_numpy(features).float().unsqueeze(0)
        mask_t = torch.from_numpy(action_mask).bool().unsqueeze(0)
        advantages = network(features_t, mask_t).squeeze(0).numpy()
    return get_strategy_from_advantages(
        torch.from_numpy(advantages).unsqueeze(0),
        mask_t,
    ).squeeze(0).numpy()


def _infer_decision_context_python(game_state: "CambiaGameState") -> DecisionContext:
    """Infer DecisionContext from CambiaGameState (Python engine)."""
    if game_state.snap_phase_active:
        return DecisionContext.SNAP_DECISION
    if game_state.pending_action:
        pending = game_state.pending_action
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
    return DecisionContext.START_TURN


class ESValidator:
    """Runs short-depth ES validation passes to check convergence."""

    def __init__(
        self,
        config: Config,
        network_weights: Dict[str, np.ndarray],
        network_config: Dict[str, int],
    ):
        self.config = config
        self.depth_limit = getattr(
            getattr(config, "deep_cfr", None), "es_validation_depth", 10
        )
        self.engine_backend = getattr(
            getattr(config, "deep_cfr", None), "engine_backend", "python"
        )

        # Reconstruct advantage network from weights
        input_dim = network_config.get("input_dim", INPUT_DIM)
        hidden_dim = network_config.get("hidden_dim", 256)
        output_dim = network_config.get("output_dim", NUM_ACTIONS)

        self.network = AdvantageNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        state_dict = {k: torch.from_numpy(v) for k, v in network_weights.items()}
        self.network.load_state_dict(state_dict)
        self.network.eval()

        self._network_config = network_config

    def compute_exploitability(self, num_traversals: int = 1000) -> Dict[str, Any]:
        """
        Run ES at short depth, return exploitability metrics.

        For each traversal:
        1. Create a new game
        2. Run external sampling traversal (updating player alternates)
        3. Collect regret samples
        4. Compute metrics from collected regrets

        Returns dict with:
            mean_regret: float — mean absolute regret across all samples
            max_regret: float — maximum absolute regret
            strategy_entropy: float — mean entropy of strategy at visited nodes
            traversals: int — how many traversals completed
            depth: int — configured depth limit
            elapsed_seconds: float
            total_nodes: int
        """
        start_time = time.time()

        all_regrets: List[float] = []
        all_entropies: List[float] = []
        completed_traversals = 0
        total_nodes = 0

        for t in range(num_traversals):
            updating_player = t % NUM_PLAYERS

            try:
                regrets, entropies, nodes = self._run_single_traversal(updating_player)
                all_regrets.extend(regrets)
                all_entropies.extend(entropies)
                total_nodes += nodes
                completed_traversals += 1
            except Exception as e:
                logger.warning("ES validation traversal %d failed: %s", t, e)
                continue

        elapsed = time.time() - start_time

        if not all_regrets:
            return {
                "mean_regret": 0.0,
                "max_regret": 0.0,
                "strategy_entropy": 0.0,
                "traversals": completed_traversals,
                "depth": self.depth_limit,
                "elapsed_seconds": elapsed,
                "total_nodes": total_nodes,
            }

        regret_array = np.array(all_regrets)
        entropy_array = np.array(all_entropies) if all_entropies else np.array([0.0])

        return {
            "mean_regret": float(np.mean(np.abs(regret_array))),
            "max_regret": float(np.max(np.abs(regret_array))),
            "strategy_entropy": float(np.mean(entropy_array)),
            "traversals": completed_traversals,
            "depth": self.depth_limit,
            "elapsed_seconds": elapsed,
            "total_nodes": total_nodes,
        }

    def _run_single_traversal(
        self, updating_player: int
    ) -> Tuple[List[float], List[float], int]:
        """Run one ES traversal, returns (regrets_list, entropies_list, nodes_count)."""
        if self.engine_backend == "go":
            return self._traverse_go(updating_player)
        else:
            return self._traverse_python(updating_player)

    def _traverse_python(
        self, updating_player: int
    ) -> Tuple[List[float], List[float], int]:
        """
        Run one ES traversal using the Python game engine.

        Returns (regrets_list, entropies_list, nodes_count).
        """
        regrets: List[float] = []
        entropies: List[float] = []
        nodes_counter = [0]

        # Initialise game
        game_state = CambiaGameState(house_rules=getattr(self.config, "cambia_rules", None))

        if game_state.is_terminal():
            return regrets, entropies, 0

        # Initialise agent states (mirror deep_worker pattern)
        initial_obs = _create_observation(None, None, game_state, -1, [])
        if initial_obs is None:
            return regrets, entropies, 0

        initial_hands = [list(p.hand) for p in game_state.players]
        initial_peeks = [p.initial_peek_indices for p in game_state.players]
        agent_states: List[AgentState] = []
        for i in range(NUM_PLAYERS):
            agent = AgentState(
                player_id=i,
                opponent_id=1 - i,
                memory_level=getattr(
                    getattr(self.config, "agent_params", None), "memory_level", 1
                ),
                time_decay_turns=getattr(
                    getattr(self.config, "agent_params", None), "time_decay_turns", 3
                ),
                initial_hand_size=len(initial_hands[i]),
                config=self.config,
            )
            agent.initialize(initial_obs, initial_hands[i], initial_peeks[i])
            agent_states.append(agent)

        self._traverse_python_recursive(
            game_state=game_state,
            agent_states=agent_states,
            updating_player=updating_player,
            depth=0,
            regrets=regrets,
            entropies=entropies,
            nodes_counter=nodes_counter,
        )

        return regrets, entropies, nodes_counter[0]

    def _traverse_python_recursive(
        self,
        game_state: "CambiaGameState",
        agent_states: List[AgentState],
        updating_player: int,
        depth: int,
        regrets: List[float],
        entropies: List[float],
        nodes_counter: List[int],
    ) -> np.ndarray:
        """
        Recursive ES traversal (Python engine).

        Returns utility vector for both players.
        """
        nodes_counter[0] += 1

        if game_state.is_terminal():
            return np.array(
                [game_state.get_utility(i) for i in range(NUM_PLAYERS)], dtype=np.float64
            )

        if depth >= self.depth_limit:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Legal actions
        try:
            legal_actions_set = game_state.get_legal_actions()
            legal_actions = sorted(list(legal_actions_set), key=repr)
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        num_actions = len(legal_actions)
        if num_actions == 0:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        player = game_state.get_acting_player()
        if player == -1:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Determine context and encode
        current_context = _infer_decision_context_python(game_state)
        current_agent_state = agent_states[player]

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
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        # Compute strategy
        try:
            strategy_full = _get_strategy_from_network(self.network, features, action_mask)
        except Exception:
            strategy_full = None

        # Extract local strategy over legal actions
        if strategy_full is not None and len(strategy_full) == NUM_ACTIONS:
            local_strategy = np.zeros(num_actions, dtype=np.float64)
            for a_idx, action in enumerate(legal_actions):
                global_idx = action_to_index(action)
                local_strategy[a_idx] = strategy_full[global_idx]
            total = local_strategy.sum()
            if total > 1e-9:
                local_strategy /= total
            else:
                local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

        # Record entropy for this node
        entropies.append(_compute_entropy(local_strategy))

        # --- External Sampling Logic ---
        if player == updating_player:
            # TRAVERSER'S NODE: enumerate ALL legal actions
            action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

            for a_idx, action in enumerate(legal_actions):
                try:
                    state_delta, undo_info = game_state.apply_action(action)
                    if not callable(undo_info):
                        continue
                except Exception:
                    continue

                observation = _create_observation(
                    None, action, game_state, player, game_state.snap_results_log
                )
                if observation is None:
                    try:
                        undo_info()
                    except Exception:
                        pass
                    continue

                next_agent_states = []
                agent_update_failed = False
                try:
                    for agent_idx, agent_state in enumerate(agent_states):
                        cloned_agent = agent_state.clone()
                        player_specific_obs = _filter_observation(observation, agent_idx)
                        cloned_agent.update(player_specific_obs)
                        next_agent_states.append(cloned_agent)
                except Exception:
                    agent_update_failed = True
                    try:
                        undo_info()
                    except Exception:
                        pass

                if not agent_update_failed:
                    try:
                        action_values[a_idx] = self._traverse_python_recursive(
                            game_state=game_state,
                            agent_states=next_agent_states,
                            updating_player=updating_player,
                            depth=depth + 1,
                            regrets=regrets,
                            entropies=entropies,
                            nodes_counter=nodes_counter,
                        )
                    except Exception:
                        pass
                    try:
                        undo_info()
                    except Exception:
                        return np.zeros(NUM_PLAYERS, dtype=np.float64)

            # Compute counterfactual values and regrets
            node_value = local_strategy @ action_values  # shape: (NUM_PLAYERS,)
            action_regrets = action_values[:, player] - node_value[player]

            # Record per-action regrets
            for r in action_regrets:
                regrets.append(float(r))

            return node_value

        else:
            # OPPONENT'S NODE: sample ONE action from strategy
            if np.sum(local_strategy) > 1e-9:
                try:
                    chosen_idx = np.random.choice(num_actions, p=local_strategy)
                except ValueError:
                    chosen_idx = np.random.choice(num_actions)
            else:
                chosen_idx = np.random.choice(num_actions)

            chosen_action = legal_actions[chosen_idx]

            try:
                state_delta, undo_info = game_state.apply_action(chosen_action)
                if not callable(undo_info):
                    return np.zeros(NUM_PLAYERS, dtype=np.float64)
            except Exception:
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            observation = _create_observation(
                None, chosen_action, game_state, player, game_state.snap_results_log
            )
            if observation is None:
                try:
                    undo_info()
                except Exception:
                    pass
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            next_agent_states = []
            try:
                for agent_idx, agent_state in enumerate(agent_states):
                    cloned_agent = agent_state.clone()
                    player_specific_obs = _filter_observation(observation, agent_idx)
                    cloned_agent.update(player_specific_obs)
                    next_agent_states.append(cloned_agent)
            except Exception:
                try:
                    undo_info()
                except Exception:
                    pass
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            try:
                node_value = self._traverse_python_recursive(
                    game_state=game_state,
                    agent_states=next_agent_states,
                    updating_player=updating_player,
                    depth=depth + 1,
                    regrets=regrets,
                    entropies=entropies,
                    nodes_counter=nodes_counter,
                )
            except Exception:
                node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

            try:
                undo_info()
            except Exception:
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            return node_value

    def _traverse_go(
        self, updating_player: int
    ) -> Tuple[List[float], List[float], int]:
        """
        Run one ES traversal using the Go engine backend.

        Falls back to Python if the Go engine cannot be imported.
        """
        try:
            from ..ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "Go engine (ffi.bridge) not available; falling back to Python backend."
            )
            return self._traverse_python(updating_player)

        regrets: List[float] = []
        entropies: List[float] = []
        nodes_counter = [0]

        go_engine = None
        go_agents: List["GoAgentState"] = []
        try:
            go_engine = GoEngine(house_rules=getattr(self.config, "cambia_rules", None))
            go_agents = [
                GoAgentState(
                    go_engine,
                    pid,
                    getattr(
                        getattr(self.config, "agent_params", None), "memory_level", 1
                    ),
                    getattr(
                        getattr(self.config, "agent_params", None), "time_decay_turns", 3
                    ),
                )
                for pid in range(NUM_PLAYERS)
            ]
            self._traverse_go_recursive(
                engine=go_engine,
                agent_states=go_agents,
                updating_player=updating_player,
                depth=0,
                regrets=regrets,
                entropies=entropies,
                nodes_counter=nodes_counter,
            )
        except Exception as e:
            logger.warning("Go traversal error: %s", e)
        finally:
            for a in go_agents:
                try:
                    a.close()
                except Exception:
                    pass
            if go_engine is not None:
                try:
                    go_engine.close()
                except Exception:
                    pass

        return regrets, entropies, nodes_counter[0]

    def _traverse_go_recursive(
        self,
        engine: Any,
        agent_states: List[Any],
        updating_player: int,
        depth: int,
        regrets: List[float],
        entropies: List[float],
        nodes_counter: List[int],
    ) -> np.ndarray:
        """Recursive ES traversal (Go engine). Returns utility vector."""
        from .deep_worker import _infer_decision_context  # noqa: PLC0415

        nodes_counter[0] += 1

        try:
            if engine.is_terminal():
                return engine.get_utility().astype(np.float64)
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        if depth >= self.depth_limit:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            legal_mask = engine.legal_actions_mask()
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        legal_indices = np.where(legal_mask > 0)[0]
        num_actions = len(legal_indices)
        if num_actions == 0:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            player = engine.acting_player()
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        current_context = _infer_decision_context(legal_mask)

        # Get drawn card bucket for POST_DRAW encoding
        drawn_bucket = -1
        if current_context == 1:  # CtxPostDraw
            drawn_bucket = engine.get_drawn_card_bucket()

        try:
            features = agent_states[player].encode(current_context, drawn_bucket=drawn_bucket)
            action_mask = legal_mask.copy()
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            strategy_full = _get_strategy_from_network(self.network, features, action_mask)
        except Exception:
            strategy_full = None

        if strategy_full is not None and len(strategy_full) == NUM_ACTIONS:
            local_strategy = strategy_full[legal_indices].astype(np.float64)
            total = local_strategy.sum()
            if total > 1e-9:
                local_strategy /= total
            else:
                local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions
        else:
            local_strategy = np.ones(num_actions, dtype=np.float64) / num_actions

        entropies.append(_compute_entropy(local_strategy))

        if player == updating_player:
            # TRAVERSER'S NODE
            action_values = np.zeros((num_actions, NUM_PLAYERS), dtype=np.float64)

            try:
                snap = engine.save()
            except Exception:
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            try:
                agent_clones = [a.clone() for a in agent_states]
            except Exception:
                engine.free_snapshot(snap)
                return np.zeros(NUM_PLAYERS, dtype=np.float64)

            for i, action_idx in enumerate(legal_indices):
                if i > 0:
                    try:
                        engine.restore(snap)
                    except Exception:
                        break
                    for j in range(len(agent_states)):
                        agent_states[j].close()
                        try:
                            agent_states[j] = agent_clones[j].clone()
                        except Exception:
                            break

                try:
                    engine.apply_action(int(action_idx))
                    for a in agent_states:
                        a.update(engine)
                except Exception:
                    continue

                try:
                    action_values[i] = self._traverse_go_recursive(
                        engine=engine,
                        agent_states=agent_states,
                        updating_player=updating_player,
                        depth=depth + 1,
                        regrets=regrets,
                        entropies=entropies,
                        nodes_counter=nodes_counter,
                    )
                except Exception:
                    pass

            # Final restore
            try:
                engine.restore(snap)
            except Exception:
                pass
            for j in range(len(agent_states)):
                agent_states[j].close()
                try:
                    agent_states[j] = agent_clones[j].clone()
                except Exception:
                    pass

            for c in agent_clones:
                try:
                    c.close()
                except Exception:
                    pass
            engine.free_snapshot(snap)

            node_value = local_strategy @ action_values
            action_regrets = action_values[:, player] - node_value[player]
            for r in action_regrets:
                regrets.append(float(r))

            return node_value

        else:
            # OPPONENT'S NODE: sample one action
            if np.sum(local_strategy) > 1e-9:
                try:
                    chosen_local_idx = np.random.choice(num_actions, p=local_strategy)
                except ValueError:
                    chosen_local_idx = np.random.choice(num_actions)
            else:
                chosen_local_idx = np.random.choice(num_actions)

            chosen_action_idx = int(legal_indices[chosen_local_idx])

            node_value = np.zeros(NUM_PLAYERS, dtype=np.float64)

            try:
                snap = engine.save()
            except Exception:
                return node_value

            try:
                agent_clones = [a.clone() for a in agent_states]
            except Exception:
                engine.free_snapshot(snap)
                return node_value

            apply_ok = False
            try:
                engine.apply_action(chosen_action_idx)
                for a in agent_states:
                    a.update(engine)
                apply_ok = True
            except Exception:
                pass

            if apply_ok:
                try:
                    node_value = self._traverse_go_recursive(
                        engine=engine,
                        agent_states=agent_states,
                        updating_player=updating_player,
                        depth=depth + 1,
                        regrets=regrets,
                        entropies=entropies,
                        nodes_counter=nodes_counter,
                    )
                except Exception:
                    pass

            try:
                engine.restore(snap)
            except Exception:
                pass
            for j in range(len(agent_states)):
                agent_states[j].close()
                try:
                    agent_states[j] = agent_clones[j].clone()
                except Exception:
                    pass
            for c in agent_clones:
                try:
                    c.close()
                except Exception:
                    pass
            engine.free_snapshot(snap)

            return node_value
