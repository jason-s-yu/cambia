"""
src/cfr/es_validator.py

ES Validation: runs short-depth External Sampling traversals to measure
exploitability metrics during Deep CFR training.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..constants import NUM_PLAYERS
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import build_advantage_network, get_strategy_from_advantages

logger = logging.getLogger(__name__)


class ESValidatorError(RuntimeError):
    """A validation step that produced no measurement at all.

    The caller must not downgrade this to a log line: every subclass means the
    step reported nothing while training carried on, which is the silence this
    class of error exists to end (cambia-1880).
    """


class ESValidatorNetworkError(ESValidatorError):
    """Raised when the validator cannot build or load its advantage network."""


class ESValidatorTraversalError(ESValidatorError):
    """Raised when no traversal in a validation step completed.

    A partial failure is not this: some traversals completing still yields a
    measurement, so those keep their per-traversal warning.
    """


def _compute_entropy(strategy: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    pos = strategy[strategy > 1e-10]  # avoid log(0)
    if len(pos) == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


def _get_strategy_from_network(
    network: torch.nn.Module,
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
    return (
        get_strategy_from_advantages(
            torch.from_numpy(advantages).unsqueeze(0),
            mask_t,
        )
        .squeeze(0)
        .numpy()
    )


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

        # Rebuild the trainer's advantage network. Going through the same
        # factory with the same settings is what makes the state_dict load for
        # every supported network_type; hardcoding AdvantageNetwork here made
        # every residual run's validation a silent no-op (cambia-1880).
        factory_kwargs = {
            "input_dim": network_config.get("input_dim", INPUT_DIM),
            "hidden_dim": network_config.get("hidden_dim", 256),
            "output_dim": network_config.get("output_dim", NUM_ACTIONS),
        }
        for key in (
            "dropout",
            "validate_inputs",
            "num_hidden_layers",
            "use_residual",
            "network_type",
            "use_pos_embed",
            "num_players",
        ):
            if key in network_config:
                factory_kwargs[key] = network_config[key]

        try:
            self.network = build_advantage_network(**factory_kwargs)
        except Exception as e:
            raise ESValidatorNetworkError(
                f"ES validation could not build its advantage network from "
                f"{factory_kwargs!r}: {e}"
            ) from e

        state_dict = {k: torch.from_numpy(v) for k, v in network_weights.items()}
        try:
            self.network.load_state_dict(state_dict)
        except Exception as e:
            raise ESValidatorNetworkError(
                f"ES validation could not load the trainer's weights into a "
                f"{type(self.network).__name__} built from {factory_kwargs!r}: {e}"
            ) from e
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
            mean_regret: float - mean absolute regret across all samples
            max_regret: float - maximum absolute regret
            strategy_entropy: float - mean entropy of strategy at visited nodes
            traversals: int - how many traversals completed
            depth: int - configured depth limit
            elapsed_seconds: float
            total_nodes: int
        """
        start_time = time.time()

        all_regrets: List[float] = []
        all_entropies: List[float] = []
        completed_traversals = 0
        total_nodes = 0
        first_error: Optional[Exception] = None

        for t in range(num_traversals):
            updating_player = t % NUM_PLAYERS

            try:
                regrets, entropies, nodes = self._traverse_go(updating_player)
                all_regrets.extend(regrets)
                all_entropies.extend(entropies)
                total_nodes += nodes
                completed_traversals += 1
            except Exception as e:
                if first_error is None:
                    first_error = e
                logger.warning("ES validation traversal %d failed: %s", t, e)
                continue

        elapsed = time.time() - start_time

        # Zeroed metrics read as "converged", so a run where every traversal
        # failed (most often a missing libcambia.so) must raise rather than
        # report zeros (cambia-1783). The caller treats this like an unusable
        # network and stops the run: nothing completed is never a healthy
        # measurement, and the usual cause does not clear on its own
        # (cambia-1880).
        if num_traversals > 0 and completed_traversals == 0:
            raise ESValidatorTraversalError(
                f"ES validation completed 0 of {num_traversals} traversals on the "
                f"Go engine; first error: {first_error}"
            ) from first_error

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

    def _traverse_go(self, updating_player: int) -> Tuple[List[float], List[float], int]:
        """
        Run one ES traversal on the Go engine.

        Returns (regrets_list, entropies_list, nodes_count).
        """
        from ..ffi.bridge import GoEngine, GoAgentState  # noqa: PLC0415

        regrets: List[float] = []
        entropies: List[float] = []
        nodes_counter = [0]

        # Engine and agent construction stay outside the tolerant block: a
        # missing or mismatched libcambia.so must surface, not be logged away
        # as one more failed traversal.
        go_engine = GoEngine(house_rules=getattr(self.config, "cambia_rules", None))
        go_agents: List["GoAgentState"] = []
        try:
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
        except Exception:
            go_engine.close()
            raise

        try:
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
            features = agent_states[player].encode(
                current_context, drawn_bucket=drawn_bucket
            )
            action_mask = legal_mask.copy()
        except Exception:
            return np.zeros(NUM_PLAYERS, dtype=np.float64)

        try:
            strategy_full = _get_strategy_from_network(
                self.network, features, action_mask
            )
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
