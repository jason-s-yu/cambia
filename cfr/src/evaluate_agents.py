"""Script to evaluate different Cambia agents against each other."""

import argparse
import json
import logging
import sys
import random
import time
import copy
from typing import Optional, Dict, List, Type, Set
from collections import Counter
import numpy as np
import math
from tqdm import tqdm

from src.config import load_config, Config
from src.game.engine import CambiaGameState
from src.agents.baseline_agents import (
    BaseAgent,
    RandomAgent,
    GreedyAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
    RandomNoCambiaAgent,
    RandomLateCambiaAgent,
    HumanPlayerAgent,
)
from src.agent_state import AgentState, AgentObservation
from src.cfr.trainer import CFRTrainer
from src.utils import (
    InfosetKey,
    normalize_probabilities,
)

from src.constants import (
    NUM_PLAYERS,
    GameAction,
    DecisionContext,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
)
from src.cfr.exceptions import GameStateError, AgentStateError, ObservationUpdateError

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CFR Agent Wrapper ---


class CFRAgentWrapper(BaseAgent):
    """Wraps a computed average strategy for use in evaluation."""

    def __init__(
        self,
        player_id: int,
        config: Config,
        average_strategy: Dict[InfosetKey, np.ndarray],
    ):
        super().__init__(player_id, config)
        if not isinstance(average_strategy, dict):
            raise TypeError(
                "CFRAgentWrapper requires average_strategy to be a dictionary."
            )
        self.average_strategy = average_strategy
        self.agent_state: Optional[AgentState] = None  # Internal state

    def initialize_state(self, initial_game_state: CambiaGameState):
        """Initialize the internal AgentState."""
        # FIX: Call internal observation creation method
        initial_obs = self._create_observation(initial_game_state, None, -1)
        if initial_obs is None:
            raise RuntimeError(
                f"CFRAgent P{self.player_id} failed to create initial observation."
            )

        self.agent_state = AgentState(
            player_id=self.player_id,
            opponent_id=self.opponent_id,
            memory_level=self.config.agent_params.memory_level,
            time_decay_turns=self.config.agent_params.time_decay_turns,
            initial_hand_size=len(initial_game_state.players[self.player_id].hand),
            config=self.config,
        )
        # Ensure initial hands/peeks are valid before passing
        initial_hand = initial_game_state.players[self.player_id].hand
        initial_peeks = initial_game_state.players[self.player_id].initial_peek_indices
        if not isinstance(initial_hand, list) or not isinstance(initial_peeks, tuple):
            raise TypeError(
                f"CFRAgent P{self.player_id}: Invalid initial hand/peek data."
            )

        self.agent_state.initialize(initial_obs, initial_hand, initial_peeks)
        logger.debug("CFRAgent P%d initialized state.", self.player_id)

    def update_state(self, observation: AgentObservation):
        """Update internal state based on observation."""
        if self.agent_state:
            # FIX: Call internal filtering method
            filtered_obs = self._filter_observation(observation, self.player_id)
            try:
                self.agent_state.update(filtered_obs)
            except (AgentStateError, ObservationUpdateError) as e_update:
                logger.error(
                    "CFRAgent P%d agent state error updating state: %s",
                    self.player_id,
                    e_update,
                )
                # Continue with old state
            except Exception as e_update:  # JUSTIFIED: evaluation resilience
                logger.error(
                    "CFRAgent P%d failed to update state: %s. Obs: %s",
                    self.player_id,
                    e_update,
                    filtered_obs,
                    exc_info=True,
                )
                # Decide how to handle state update failure - continue with old state? Raise?
                # For now, log and continue.
        else:
            logger.error(
                "CFRAgent P%d cannot update state, not initialized.", self.player_id
            )

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Chooses an action based on the learned average strategy."""
        if not self.agent_state:
            raise RuntimeError(
                f"CFRAgent P{self.player_id} state not initialized before choose_action."
            )
        if not legal_actions:
            raise ValueError(
                f"CFRAgent P{self.player_id} cannot choose from empty legal actions."
            )

        # Determine context (copied from analysis_tools/worker)
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
                current_context = DecisionContext.START_TURN  # Fallback
        else:
            current_context = DecisionContext.START_TURN

        try:
            base_infoset_tuple = self.agent_state.get_infoset_key()
            # Ensure base_infoset_tuple is actually a tuple before unpacking
            if not isinstance(base_infoset_tuple, tuple):
                raise TypeError(
                    f"get_infoset_key returned {type(base_infoset_tuple).__name__}, expected tuple"
                )
            infoset_key = InfosetKey(*base_infoset_tuple, current_context.value)
        except AgentStateError as e_key:
            logger.error(
                "CFRAgent P%d agent state error getting infoset key: %s",
                self.player_id,
                e_key,
            )
            return random.choice(list(legal_actions))
        except Exception as e_key:  # JUSTIFIED: evaluation resilience
            logger.error(
                "CFRAgent P%d Error getting infoset key: %s. State: %s",
                self.player_id,
                e_key,
                self.agent_state,
                exc_info=True,
            )
            return random.choice(list(legal_actions))

        strategy = self.average_strategy.get(infoset_key)
        action_list = sorted(list(legal_actions), key=repr)
        num_actions = len(action_list)

        # Handle missing strategy or dimension mismatch
        if strategy is None or len(strategy) != num_actions:
            if strategy is not None:
                logger.warning(
                    "CFRAgent P%d strategy dim mismatch key %s (Have %d, Need %d). Using uniform.",
                    self.player_id,
                    infoset_key,
                    len(strategy),
                    num_actions,
                )
            # else: logger.debug("CFRAgent P%d strategy not found for key %s. Using uniform.", self.player_id, infoset_key) # Reduce noise
            strategy = (
                np.ones(num_actions) / num_actions if num_actions > 0 else np.array([])
            )

        # Handle zero-length strategy (should only happen if num_actions is 0, caught earlier)
        if len(strategy) == 0:
            logger.error(
                "CFRAgent P%d: Zero strategy length despite %d legal actions exist.",
                self.player_id,
                num_actions,
            )
            return random.choice(action_list)  # Fallback

        # Normalize strategy if needed (defensive programming)
        strategy_sum = np.sum(strategy)
        if not np.isclose(strategy_sum, 1.0):
            logger.warning(
                "CFRAgent P%d: Normalizing strategy for key %s (Sum: %f)",
                self.player_id,
                infoset_key,
                strategy_sum,
            )
            strategy = normalize_probabilities(strategy)
            if len(strategy) == 0 or not np.isclose(
                np.sum(strategy), 1.0
            ):  # Check normalization result
                logger.error(
                    "CFRAgent P%d: Failed to normalize strategy for %s. Using uniform.",
                    self.player_id,
                    infoset_key,
                )
                strategy = (
                    np.ones(num_actions) / num_actions
                    if num_actions > 0
                    else np.array([])
                )

        # Sample action
        try:
            chosen_index = np.random.choice(num_actions, p=strategy)
            chosen_action = action_list[chosen_index]
        except (
            ValueError
        ) as e_choice:  # Catch errors from np.random.choice (e.g., probabilities don't sum to 1)
            logger.error(
                "CFRAgent P%d: Error choosing action for key %s (strategy %s): %s. Choosing random.",
                self.player_id,
                infoset_key,
                strategy,
                e_choice,
            )
            chosen_action = random.choice(action_list)  # Fallback

        # logger.debug("CFRAgent P%d chose action: %s (Prob: %.3f, Key: %s)", self.player_id, chosen_action, strategy[chosen_index], infoset_key)
        return chosen_action

    # --- Observation helpers moved into CFRAgentWrapper ---
    def _create_observation(
        self,
        game_state: CambiaGameState,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Creates observation needed *by this agent* after an action."""
        try:
            # Simplified for evaluation: Assume agent state uses public info + own known cards
            obs = AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,  # Don't pass private draw info during evaluation obs
                peeked_cards=None,  # Don't pass private peek info during evaluation obs
                snap_results=copy.deepcopy(game_state.snap_results_log),  # Public
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
            return obs
        except GameStateError as e_obs:
            logger.error(
                "CFRAgent P%d: Game state error creating observation: %s",
                self.player_id,
                e_obs,
            )
            return None
        except Exception as e_obs:  # JUSTIFIED: evaluation resilience
            logger.error(
                "CFRAgent P%d: Error creating observation: %s",
                self.player_id,
                e_obs,
                exc_info=True,
            )
            return None

    def _filter_observation(
        self, obs: AgentObservation, observer_id: int
    ) -> AgentObservation:
        """Filters observation for the agent's own perspective (minimal filtering needed here)."""
        # Since _create_observation doesn't include sensitive info, filtering is simpler
        filtered_obs = copy.copy(obs)
        # Ensure fields intended to be private for updates are None
        filtered_obs.drawn_card = None
        filtered_obs.peeked_cards = None
        return filtered_obs


# --- Neural Agent Wrapper Base ---


import abc


class NeuralAgentWrapper(BaseAgent, abc.ABC):
    """
    Abstract base class for neural-network-backed agent wrappers.

    Shared state management (AgentState init/update) and inference helpers
    used by DeepCFRAgentWrapper, ESCHERAgentWrapper, and ReBeLAgentWrapper.
    """

    def __init__(self, player_id: int, config, device: str = "cpu"):
        super().__init__(player_id, config)
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.agent_state: Optional[AgentState] = None

    @abc.abstractmethod
    def choose_action(
        self, game_state, legal_actions: Set[GameAction]
    ) -> GameAction:
        pass

    def initialize_state(self, initial_game_state):
        """Initialize internal AgentState."""
        initial_hand = initial_game_state.players[self.player_id].hand
        initial_peeks = initial_game_state.players[self.player_id].initial_peek_indices
        self.agent_state = AgentState(
            player_id=self.player_id,
            opponent_id=self.opponent_id,
            memory_level=self.config.agent_params.memory_level,
            time_decay_turns=self.config.agent_params.time_decay_turns,
            initial_hand_size=len(initial_hand),
            config=self.config,
        )
        initial_obs = AgentObservation(
            acting_player=-1,
            action=None,
            discard_top_card=initial_game_state.get_discard_top(),
            player_hand_sizes=[
                initial_game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
            ],
            stockpile_size=initial_game_state.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=copy.deepcopy(initial_game_state.snap_results_log),
            did_cambia_get_called=initial_game_state.cambia_caller_id is not None,
            who_called_cambia=initial_game_state.cambia_caller_id,
            is_game_over=initial_game_state.is_terminal(),
            current_turn=initial_game_state.get_turn_number(),
        )
        self.agent_state.initialize(initial_obs, initial_hand, initial_peeks)

    def update_state(self, observation: AgentObservation):
        """Update internal AgentState after an action."""
        if not self.agent_state:
            return
        filtered = copy.copy(observation)
        filtered.drawn_card = None
        filtered.peeked_cards = None
        try:
            from src.cfr.exceptions import AgentStateError, ObservationUpdateError
            self.agent_state.update(filtered)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error(
                "%s P%d state update error: %s", self.__class__.__name__, self.player_id, e
            )

    def _get_decision_context(self, game_state) -> DecisionContext:
        """Determine the current decision context from game state."""
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

    def _encode_eppbs(self, decision_context: DecisionContext) -> np.ndarray:
        """Encode agent state using EP-PBS encoding for evaluation."""
        from src.encoding import encode_infoset_eppbs

        st = self.agent_state
        # Determine cambia_state: 0=self called, 1=opponent called, 2=none
        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == self.player_id:
            cambia_state = 0
        else:
            cambia_state = 1

        return encode_infoset_eppbs(
            slot_tags=[t.value if hasattr(t, 'value') else int(t) for t in st.slot_tags],
            slot_buckets=[int(b) for b in st.slot_buckets],
            discard_top_bucket=st.known_discard_top_bucket.value if hasattr(st.known_discard_top_bucket, 'value') else int(st.known_discard_top_bucket),
            stock_estimate=st.stockpile_estimate.value if hasattr(st.stockpile_estimate, 'value') else int(st.stockpile_estimate),
            game_phase=st.game_phase.value if hasattr(st.game_phase, 'value') else int(st.game_phase),
            decision_context=decision_context.value if hasattr(decision_context, 'value') else int(decision_context),
            cambia_state=cambia_state,
        )

    def _create_observation(
        self,
        game_state,
        action: Optional[GameAction],
        acting_player: int,
    ) -> Optional[AgentObservation]:
        """Creates a public observation from game state after an action."""
        try:
            return AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=copy.deepcopy(game_state.snap_results_log),
                did_cambia_get_called=game_state.cambia_caller_id is not None,
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error(
                "%s P%d observation error: %s", self.__class__.__name__, self.player_id, e
            )
            return None

    @classmethod
    def _load_cambia_rules_mismatch_check(cls, checkpoint, config, player_id):
        """Warn on cambia_rules mismatch between checkpoint and eval config."""
        _saved_meta = checkpoint.get("metadata", {})
        _saved_rules = (_saved_meta.get("config", {}) or {}).get("cambia_rules", {})
        if _saved_rules and hasattr(config, "cambia_rules"):
            try:
                from dataclasses import asdict as _asdict
                _current_rules = _asdict(config.cambia_rules)
                for _key in set(_saved_rules) | set(_current_rules):
                    if _saved_rules.get(_key) != _current_rules.get(_key):
                        logger.warning(
                            "%s P%d: cambia_rules mismatch '%s': checkpoint=%r, current=%r",
                            cls.__name__, player_id, _key,
                            _saved_rules.get(_key), _current_rules.get(_key),
                        )
            except Exception:
                pass


# --- Deep CFR Agent Wrapper ---


class DeepCFRAgentWrapper(NeuralAgentWrapper):
    """
    Wraps a trained Deep CFR AdvantageNetwork for use in evaluation.

    Loads a .pt checkpoint, reconstructs the AdvantageNetwork, and uses
    get_strategy_from_advantages() to sample actions during play.
    """

    def __init__(
        self,
        player_id: int,
        config: Config,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        super().__init__(player_id, config, device=device)
        from src.networks import AdvantageNetwork, get_strategy_from_advantages, build_advantage_network
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        self._get_strategy_from_advantages = get_strategy_from_advantages
        self._INPUT_DIM = INPUT_DIM
        self._NUM_ACTIONS = NUM_ACTIONS

        # Load checkpoint
        torch = self._torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        dcfr_config = checkpoint.get("dcfr_config", {})

        # Warn on cambia_rules mismatch between checkpoint and eval config
        self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

        hidden_dim = dcfr_config.get("hidden_dim", 256)
        num_hidden_layers = dcfr_config.get("num_hidden_layers", 2)
        use_residual = dcfr_config.get("use_residual", False)
        self._encoding_mode = dcfr_config.get("encoding_mode", "legacy")

        from src.constants import EP_PBS_INPUT_DIM
        net_input_dim = EP_PBS_INPUT_DIM if self._encoding_mode == "ep_pbs" else INPUT_DIM

        self.advantage_net = build_advantage_network(
            input_dim=net_input_dim,
            hidden_dim=hidden_dim,
            output_dim=NUM_ACTIONS,
            dropout=0.1,
            validate_inputs=False,
            num_hidden_layers=num_hidden_layers,
            use_residual=use_residual,
        )
        self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
        self.advantage_net.to(self.device)
        self.advantage_net.eval()

        logger.info(
            "DeepCFRAgent P%d loaded checkpoint (step=%s, traversals=%s)",
            self.player_id,
            checkpoint.get("training_step", "N/A"),
            checkpoint.get("total_traversals", "N/A"),
        )

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose an action using the AdvantageNetwork via regret-matching strategy."""
        from src.encoding import encode_infoset, encode_infoset_eppbs, encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            if self._encoding_mode == "ep_pbs":
                features = self._encode_eppbs(decision_context)
            else:
                features = encode_infoset(self.agent_state, decision_context)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("DeepCFRAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            advantages = self.advantage_net(feat_t, mask_t)
            strategy = self._get_strategy_from_advantages(advantages, mask_t)
            probs = strategy.squeeze(0).cpu().numpy()

        # Sample from legal action probabilities
        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


# --- ESCHER Agent Wrapper ---


class ESCHERAgentWrapper(NeuralAgentWrapper):
    """
    Wraps a trained ESCHER StrategyNetwork for use in evaluation.

    Loads a .pt checkpoint, reconstructs the StrategyNetwork, and samples
    actions directly from the strategy probabilities (no regret matching).
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        super().__init__(player_id, config, device=device)
        from src.networks import StrategyNetwork
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        self._INPUT_DIM = INPUT_DIM
        self._NUM_ACTIONS = NUM_ACTIONS

        torch = self._torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        dcfr_config = checkpoint.get("dcfr_config", {})

        # Warn on cambia_rules mismatch
        self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

        hidden_dim = dcfr_config.get("hidden_dim", 256)

        self.policy_net = StrategyNetwork(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            output_dim=NUM_ACTIONS,
            validate_inputs=False,
        )
        self.policy_net.load_state_dict(checkpoint["strategy_net_state_dict"])
        self.policy_net.to(self.device)
        self.policy_net.eval()

        logger.info(
            "ESCHERAgent P%d loaded checkpoint (step=%s, traversals=%s)",
            self.player_id,
            checkpoint.get("training_step", "N/A"),
            checkpoint.get("total_traversals", "N/A"),
        )

    def choose_action(
        self, game_state, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose an action using the StrategyNetwork probabilities."""
        from src.encoding import encode_infoset, encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            features = encode_infoset(self.agent_state, decision_context)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("ESCHERAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            probs = self.policy_net(feat_t, mask_t)
            probs_np = probs.squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


# --- ReBeL Agent Wrapper ---


class ReBeLAgentWrapper(NeuralAgentWrapper):
    """
    Wraps trained PBSValueNetwork + PBSPolicyNetwork for ReBeL evaluation.

    Loads a .pt checkpoint, reconstructs both networks, and selects actions
    via PBSPolicyNetwork inference with a PBS-encoded game state.
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        import warnings

        warnings.warn(
            "ReBeLAgentWrapper is DEPRECATED and will be removed. "
            "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
            "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(player_id, config, device=device)
        from src.networks import PBSValueNetwork, PBSPolicyNetwork
        from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES

        torch = self._torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        dcfr_config = checkpoint.get("rebel_config", checkpoint.get("dcfr_config", {}))

        # Warn on cambia_rules mismatch
        self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

        value_hidden_dim = dcfr_config.get("rebel_value_hidden_dim", 1024)
        policy_hidden_dim = dcfr_config.get("rebel_policy_hidden_dim", 512)
        self.rebel_depth = dcfr_config.get("rebel_subgame_depth", 4)
        self.rebel_cfr_iterations = dcfr_config.get("rebel_cfr_iterations", 200)

        self.value_net = PBSValueNetwork(
            input_dim=PBS_INPUT_DIM,
            hidden_dim=value_hidden_dim,
            output_dim=2 * NUM_HAND_TYPES,
            validate_inputs=False,
        )
        self.value_net.load_state_dict(checkpoint["rebel_value_net_state_dict"])
        self.value_net.to(self.device)
        self.value_net.eval()

        self.policy_net = PBSPolicyNetwork(
            input_dim=PBS_INPUT_DIM,
            hidden_dim=policy_hidden_dim,
            validate_inputs=False,
        )
        self.policy_net.load_state_dict(checkpoint["rebel_policy_net_state_dict"])
        self.policy_net.to(self.device)
        self.policy_net.eval()

        logger.info(
            "ReBeLAgent P%d loaded checkpoint (step=%s, traversals=%s)",
            self.player_id,
            checkpoint.get("training_step", "N/A"),
            checkpoint.get("total_traversals", "N/A"),
        )

    def _build_pbs(self, game_state):
        """Build a PBS from the current game state using uniform ranges."""
        from src.pbs import (
            PBS, uniform_range, make_public_features,
            PHASE_DRAW, PHASE_DISCARD, PHASE_ABILITY, PHASE_SNAP, PHASE_TERMINAL,
        )

        range_p0 = uniform_range()
        range_p1 = uniform_range()
        try:
            # Determine phase from game state
            if game_state.is_terminal():
                phase = PHASE_TERMINAL
            elif game_state.snap_phase_active:
                phase = PHASE_SNAP
            elif game_state.pending_action is not None:
                if isinstance(game_state.pending_action, ActionDiscard):
                    phase = PHASE_DISCARD
                elif isinstance(
                    game_state.pending_action,
                    (
                        ActionAbilityPeekOwnSelect,
                        ActionAbilityPeekOtherSelect,
                        ActionAbilityBlindSwapSelect,
                        ActionAbilityKingLookSelect,
                        ActionAbilityKingSwapDecision,
                    ),
                ):
                    phase = PHASE_ABILITY
                else:
                    phase = PHASE_DRAW
            else:
                phase = PHASE_DRAW

            stockpile_remaining = game_state.get_stockpile_size()
            stockpile_total = 46  # Standard initial stockpile (54 - 8 dealt - 1 discard + jokers)

            public_features = make_public_features(
                turn=game_state.get_turn_number(),
                max_turns=getattr(self.config.cambia_rules, "max_game_turns", 100),
                phase=phase,
                stockpile_remaining=stockpile_remaining,
                stockpile_total=stockpile_total,
            )
        except Exception:
            from src.pbs import NUM_PUBLIC_FEATURES
            public_features = np.zeros(NUM_PUBLIC_FEATURES, dtype=np.float32)

        return PBS(range_p0=range_p0, range_p1=range_p1, public_features=public_features)

    def choose_action(
        self, game_state, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose an action using PBSPolicyNetwork inference."""
        from src.encoding import encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError
        from src.pbs import encode_pbs

        if not self.agent_state or game_state is None:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        try:
            pbs = self._build_pbs(game_state)
            pbs_enc = encode_pbs(pbs)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("ReBeLAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            pbs_t = torch.from_numpy(pbs_enc).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            probs = self.policy_net(pbs_t, mask_t)
            probs_np = probs.squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


class SDCFRAgentWrapper(NeuralAgentWrapper):
    """
    Wraps SD-CFR advantage network snapshots for evaluation.

    Loads all snapshots, runs each through regret matching, and averages
    the resulting strategies with linear or uniform weighting.
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str,
        device: str = "cpu",
        use_ema: bool = True,
    ):
        super().__init__(player_id, config, device=device)
        from src.networks import build_advantage_network, get_strategy_from_advantages
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        self._get_strategy_from_advantages = get_strategy_from_advantages
        self._INPUT_DIM = INPUT_DIM
        self._NUM_ACTIONS = NUM_ACTIONS

        torch = self._torch
        import os
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        dcfr_config = checkpoint.get("dcfr_config", {})
        self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

        hidden_dim = dcfr_config.get("hidden_dim", 256)
        num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
        use_residual = dcfr_config.get("use_residual", True)
        self._weighting = dcfr_config.get("sd_cfr_snapshot_weighting", "linear")
        self._encoding_mode = dcfr_config.get("encoding_mode", "legacy")

        from src.constants import EP_PBS_INPUT_DIM
        net_input_dim = EP_PBS_INPUT_DIM if self._encoding_mode == "ep_pbs" else INPUT_DIM

        base_path = os.path.splitext(checkpoint_path)[0]

        # Try EMA fast path: single network, O(1) inference
        ema_path = f"{base_path}_ema.pt"
        ema_enabled = use_ema and dcfr_config.get("use_ema", True) and os.path.exists(ema_path)

        if ema_enabled:
            ema_data = torch.load(ema_path, map_location=self.device, weights_only=True)
            ema_net = build_advantage_network(
                input_dim=net_input_dim,
                hidden_dim=hidden_dim,
                output_dim=NUM_ACTIONS,
                dropout=0.0,
                validate_inputs=False,
                num_hidden_layers=num_hidden_layers,
                use_residual=use_residual,
            )
            ema_state = {k: v for k, v in ema_data["ema_state_dict"].items()}
            ema_net.load_state_dict(ema_state)
            ema_net.to(self.device)
            ema_net.eval()
            self._ema_net = ema_net
            self._snapshot_nets = []
            self._snapshot_iterations = []
            logger.info(
                "SDCFRAgent P%d loaded EMA weights (step=%s) for O(1) inference.",
                self.player_id, checkpoint.get("training_step", "N/A"),
            )
        else:
            self._ema_net = None
            # Fall back to full snapshot averaging
            sd_snapshots_path = checkpoint.get("sd_snapshots_path", f"{base_path}_sd_snapshots.pt")
            if not os.path.exists(sd_snapshots_path):
                raise FileNotFoundError(f"SD-CFR snapshots not found: {sd_snapshots_path}")

            snapshot_data = torch.load(sd_snapshots_path, map_location="cpu", weights_only=True)
            num_snapshots = int(snapshot_data["num_snapshots"].item())
            self._snapshot_iterations = snapshot_data["iterations"].tolist()

            if num_snapshots == 0:
                raise ValueError("SD-CFR snapshot file contains no snapshots")

            # Reconstruct snapshot state dicts from flattened tensor format
            snapshots = []
            for i in range(num_snapshots):
                prefix = f"snap_{i}_"
                snap = {k[len(prefix):]: v for k, v in snapshot_data.items() if k.startswith(prefix)}
                snapshots.append(snap)

            # Build networks from snapshots
            self._snapshot_nets = []
            for snap_weights in snapshots:
                net = build_advantage_network(
                    input_dim=net_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=NUM_ACTIONS,
                    dropout=0.0,  # No dropout at eval
                    validate_inputs=False,
                    num_hidden_layers=num_hidden_layers,
                    use_residual=use_residual,
                )
                # snap_weights are already tensors from torch.load
                net.load_state_dict(snap_weights)
                net.to(self.device)
                net.eval()
                self._snapshot_nets.append(net)

            logger.info(
                "SDCFRAgent P%d loaded %d snapshots (step=%s, weighting=%s)",
                self.player_id, len(self._snapshot_nets),
                checkpoint.get("training_step", "N/A"), self._weighting,
            )

    def choose_action(
        self, game_state, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose action by averaging regret-matched strategies across all snapshots."""
        from src.encoding import encode_infoset, encode_infoset_eppbs, encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            if self._encoding_mode == "ep_pbs":
                features = self._encode_eppbs(decision_context)
            else:
                features = encode_infoset(self.agent_state, decision_context)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:
            logger.error("SDCFRAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

            if self._ema_net is not None:
                # O(1) EMA fast path: single forward pass
                advantages = self._ema_net(feat_t, mask_t)
                avg_strategy = self._get_strategy_from_advantages(advantages, mask_t)
            else:
                # Full snapshot averaging fallback
                avg_strategy = torch.zeros(1, self._NUM_ACTIONS, device=self.device)
                total_weight = 0.0

                for i, net in enumerate(self._snapshot_nets):
                    advantages = net(feat_t, mask_t)
                    strategy = self._get_strategy_from_advantages(advantages, mask_t)

                    if self._weighting == "linear":
                        w = float(self._snapshot_iterations[i] + 1) if i < len(self._snapshot_iterations) else 1.0
                    else:
                        w = 1.0

                    avg_strategy += w * strategy
                    total_weight += w

                if total_weight > 0:
                    avg_strategy /= total_weight

            probs = avg_strategy.squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


# --- N-Player Agent Wrapper ---


class NPlayerAgentWrapper(NeuralAgentWrapper):
    """
    Wrapper for N-player trained models using QRE strategy.

    Loads a checkpoint trained with N-player encoding (580-dim input, 452 actions)
    and uses QRE (softmax) strategy at eval time with the final lambda temperature.
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str = None,
        device: str = "cpu",
        num_players: int = 2,
        qre_lambda: float = 0.05,
        **kwargs,
    ):
        super().__init__(player_id, config, device=device)
        from src.networks import build_advantage_network
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        self.num_players = kwargs.get('num_players', num_players)
        self.qre_lambda = kwargs.get('qre_lambda', qre_lambda)
        self._N_PLAYER_INPUT_DIM = N_PLAYER_INPUT_DIM
        self._N_PLAYER_NUM_ACTIONS = N_PLAYER_NUM_ACTIONS

        torch = self._torch

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            dcfr_config = checkpoint.get("dcfr_config", {})

            self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

            hidden_dim = dcfr_config.get("hidden_dim", 256)
            num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
            use_residual = dcfr_config.get("use_residual", True)

            self.advantage_net = build_advantage_network(
                input_dim=N_PLAYER_INPUT_DIM,
                hidden_dim=hidden_dim,
                output_dim=N_PLAYER_NUM_ACTIONS,
                dropout=0.0,
                validate_inputs=False,
                num_hidden_layers=num_hidden_layers,
                use_residual=use_residual,
            )
            self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
            self.advantage_net.to(self.device)
            self.advantage_net.eval()

            logger.info(
                "NPlayerAgent P%d loaded checkpoint (step=%s)",
                self.player_id,
                checkpoint.get("training_step", "N/A"),
            )
        else:
            # No checkpoint: build a fresh network (used in tests / warm-start)
            self.advantage_net = build_advantage_network(
                input_dim=N_PLAYER_INPUT_DIM,
                hidden_dim=256,
                output_dim=N_PLAYER_NUM_ACTIONS,
                dropout=0.0,
                validate_inputs=False,
                num_hidden_layers=3,
                use_residual=True,
            )
            self.advantage_net.to(self.device)
            self.advantage_net.eval()

    def choose_action(
        self, game_state, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose an action using N-player encoding and QRE strategy."""
        from src.cfr.deep_trainer import qre_strategy
        from src.cfr.exceptions import ActionEncodingError
        from src.encoding import encode_action_mask, index_to_action
        import numpy as np

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            # Attempt N-player encoding; fall back to legacy if not available
            try:
                from src.encoding import encode_infoset_nplayer
                features = encode_infoset_nplayer(self.agent_state, decision_context)
            except (ImportError, AttributeError):
                from src.encoding import encode_infoset
                features = encode_infoset(self.agent_state, decision_context)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("NPlayerAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            # Build a mask of shape (1, N_PLAYER_NUM_ACTIONS) — pad legacy mask if needed
            mask_np = action_mask
            if mask_np.shape[0] < self._N_PLAYER_NUM_ACTIONS:
                padded = np.zeros(self._N_PLAYER_NUM_ACTIONS, dtype=np.float32)
                padded[:mask_np.shape[0]] = mask_np
                mask_np = padded

            mask_t = torch.from_numpy(mask_np.astype(bool)).unsqueeze(0).to(self.device)

            # Ensure input dim matches
            if feat_t.shape[-1] != self._N_PLAYER_INPUT_DIM:
                padded_feat = torch.zeros(1, self._N_PLAYER_INPUT_DIM, device=self.device)
                padded_feat[0, :feat_t.shape[-1]] = feat_t[0]
                feat_t = padded_feat

            advantages = self.advantage_net(feat_t, mask_t)
            strategy = qre_strategy(advantages, mask_t, self.qre_lambda)
            probs = strategy.squeeze(0).cpu().numpy()

        legal_indices = np.where(mask_np)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except (ActionEncodingError, Exception):
            return random.choice(legal_list)


# --- Agent Factory ---

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "imperfect_greedy": ImperfectGreedyAgent,
    "memory_heuristic": MemoryHeuristicAgent,
    "aggressive_snap": AggressiveSnapAgent,
    "cfr": CFRAgentWrapper,
    "deep_cfr": DeepCFRAgentWrapper,
    "escher": ESCHERAgentWrapper,
    "sd_cfr": SDCFRAgentWrapper,
    "nplayer": NPlayerAgentWrapper,
    "random_no_cambia": RandomNoCambiaAgent,
    "random_late_cambia": RandomLateCambiaAgent,
    "human_player": HumanPlayerAgent,
}


def get_agent(agent_type: str, player_id: int, config, **kwargs) -> BaseAgent:
    """Instantiates an agent based on its type."""
    agent_class = AGENT_REGISTRY.get(agent_type.lower())
    if not agent_class:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Available: {list(AGENT_REGISTRY.keys())}"
        )

    if agent_type.lower() == "cfr":
        avg_strategy = kwargs.get("average_strategy")
        if not avg_strategy or not isinstance(avg_strategy, dict):  # Check type
            raise ValueError("CFRAgent requires 'average_strategy' dictionary.")
        return CFRAgentWrapper(player_id, config, avg_strategy)
    elif agent_type.lower() in ("deep_cfr", "escher", "sd_cfr"):
        checkpoint_path = kwargs.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                f"{agent_class.__name__} requires 'checkpoint_path'."
            )
        device = kwargs.get("device", "cpu")
        return agent_class(player_id, config, checkpoint_path, device=device)
    else:
        # Pass config to baseline agents as well
        return agent_class(player_id, config)


# --- Evaluation Loop ---


def run_evaluation(
    config_path: str,
    agent1_type: str,
    agent2_type: str,
    num_games: int,
    strategy_path: Optional[str],
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> Counter:
    """Runs head-to-head evaluation between two agents. Returns results Counter."""
    logger.info("--- Starting Agent Evaluation ---")
    logger.info("Config: %s", config_path)
    logger.info("Agent 1 (P0): %s", agent1_type.upper())
    logger.info("Agent 2 (P1): %s", agent2_type.upper())
    logger.info("Number of Games: %d", num_games)
    if agent1_type.lower() == "cfr" or agent2_type.lower() == "cfr":
        if not strategy_path:
            logger.error("Strategy file path (--strategy) required for CFR agent.")
            sys.exit(1)
        logger.info("CFR Strategy File: %s", strategy_path)
    _checkpoint_agent_types = {"deep_cfr", "escher", "sd_cfr", "nplayer"}
    if agent1_type.lower() in _checkpoint_agent_types or agent2_type.lower() in _checkpoint_agent_types:
        if not checkpoint_path:
            logger.error("Checkpoint path (--checkpoint) required for %s agent.", agent1_type)
            sys.exit(1)
        logger.info("Checkpoint: %s", checkpoint_path)

    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration from %s", config_path)
        sys.exit(1)

    average_strategy = None
    if agent1_type.lower() == "cfr" or agent2_type.lower() == "cfr":
        logger.info("Loading CFR agent data from %s...", strategy_path)
        # Use the CFRTrainer temporarily just for loading/computing strategy
        try:
            # Minimal init, avoids needing full trainer setup dependencies if possible
            temp_trainer = CFRTrainer(config=config)
            temp_trainer.load_data(strategy_path)  # Use load_data method
            logger.info("Computing average strategy...")
            average_strategy = temp_trainer.compute_average_strategy()
            if average_strategy is None:
                logger.error("Failed to compute average strategy from loaded data.")
                sys.exit(1)
            logger.info("Average strategy computed (%d infosets).", len(average_strategy))
        except Exception as e_load:  # JUSTIFIED: evaluation resilience
            logger.exception("Failed to load or process CFR strategy: %s", e_load)
            sys.exit(1)

    try:
        agent1_kwargs: Dict = {}
        agent2_kwargs: Dict = {}
        if agent1_type.lower() == "cfr":
            agent1_kwargs = {"average_strategy": average_strategy}
        elif agent1_type.lower() in _checkpoint_agent_types:
            agent1_kwargs = {"checkpoint_path": checkpoint_path, "device": device}
        if agent2_type.lower() == "cfr":
            agent2_kwargs = {"average_strategy": average_strategy}
        elif agent2_type.lower() in _checkpoint_agent_types:
            agent2_kwargs = {"checkpoint_path": checkpoint_path, "device": device}
        agent1 = get_agent(agent1_type, player_id=0, config=config, **agent1_kwargs)
        agent2 = get_agent(agent2_type, player_id=1, config=config, **agent2_kwargs)
        agents = [agent1, agent2]
        logger.info("Agents instantiated.")
    except ValueError as e:
        logger.error("Error creating agents: %s", e)
        sys.exit(1)

    results: Counter = Counter()
    start_time = time.perf_counter()
    jsonl_overhead_ms = 0.0
    # Per-game tracking for enhanced stats
    score_margins: List[float] = []
    game_turns_list: List[int] = []

    output_file = None
    if output_path is not None:
        output_file = open(output_path, "w", encoding="utf-8")

    try:
        for game_num in tqdm(range(1, num_games + 1), desc="Simulating Games", unit="game"):
            game_start = time.perf_counter()
            game_actions: List[Dict] = []
            game_winner = "error"
            game_error = False

            try:
                game_state = CambiaGameState(house_rules=config.cambia_rules)
                # Initialize stateful agents
                for agent in agents:
                    if isinstance(agent, (CFRAgentWrapper, NeuralAgentWrapper)):
                        agent.initialize_state(game_state)

                turn = 0
                max_turns = (
                    config.cambia_rules.max_game_turns
                    if config.cambia_rules.max_game_turns > 0
                    else 500
                )

                while not game_state.is_terminal() and turn < max_turns:
                    turn += 1
                    acting_player_id = game_state.get_acting_player()
                    if acting_player_id == -1:
                        logger.error(
                            "Game %d Turn %d: Invalid acting player (-1). State: %s",
                            game_num,
                            turn,
                            game_state,
                        )
                        results["Errors"] += 1
                        game_error = True
                        break

                    current_agent = agents[acting_player_id]
                    try:
                        legal_actions = game_state.get_legal_actions()
                        if not legal_actions:
                            # Check again if terminal, might have become terminal after last action
                            if game_state.is_terminal():
                                break
                            logger.error(
                                "Game %d Turn %d: No legal actions but non-terminal? State: %s",
                                game_num,
                                turn,
                                game_state,
                            )
                            results["Errors"] += 1
                            game_error = True
                            break

                        # Choose action
                        chosen_action = current_agent.choose_action(game_state, legal_actions)

                        # Collect action record if logging
                        if output_file is not None:
                            game_actions.append({
                                "turn": turn,
                                "player": acting_player_id,
                                "action": type(chosen_action).__name__,
                                "legal_count": len(legal_actions),
                            })

                        # Apply action
                        state_delta, undo_info = game_state.apply_action(chosen_action)
                        if not callable(
                            undo_info
                        ):  # Should not happen if apply_action succeeds
                            logger.error(
                                "Game %d Turn %d: Action %s applied but returned invalid undo info.",
                                game_num,
                                turn,
                                chosen_action,
                            )
                            results["Errors"] += 1
                            game_error = True
                            break

                        # Create observation AFTER action (for stateful agents)
                        has_stateful = any(
                            isinstance(a, (CFRAgentWrapper, NeuralAgentWrapper)) for a in agents
                        )
                        observation = None
                        if has_stateful and hasattr(current_agent, "_create_observation"):
                            observation = current_agent._create_observation(
                                game_state, chosen_action, acting_player_id
                            )

                        # Update agent states (only stateful agents need it)
                        if observation:
                            for agent in agents:
                                if isinstance(agent, (CFRAgentWrapper, NeuralAgentWrapper)):
                                    agent.update_state(observation)

                    except GameStateError as e_turn:
                        logger.error(
                            "Game state error during game %d turn %d for P%d: %s",
                            game_num,
                            turn,
                            acting_player_id,
                            e_turn,
                        )
                        results["Errors"] += 1
                        game_error = True
                        break  # End game on error
                    except (AgentStateError, ObservationUpdateError) as e_turn:
                        logger.error(
                            "Agent state error during game %d turn %d for P%d: %s",
                            game_num,
                            turn,
                            acting_player_id,
                            e_turn,
                        )
                        results["Errors"] += 1
                        game_error = True
                        break  # End game on error
                    except Exception as e_turn:  # JUSTIFIED: evaluation resilience
                        logger.exception(
                            "Error during game %d turn %d for P%d: %s. State: %s",
                            game_num,
                            turn,
                            acting_player_id,
                            e_turn,
                            game_state,
                        )
                        results["Errors"] += 1
                        game_error = True
                        break  # End game on error

                # Game End
                if game_state.is_terminal():
                    winner = game_state._winner
                    if winner == 0:
                        results["P0 Wins"] += 1
                        game_winner = "p0"
                    elif winner == 1:
                        results["P1 Wins"] += 1
                        game_winner = "p1"
                    else:
                        results["Ties"] += 1
                        game_winner = "tie"
                    # Capture score margin: sum of card values per player hand
                    try:
                        hand_scores = [
                            sum(card.value for card in game_state.players[i].hand)
                            for i in range(len(game_state.players))
                        ]
                        if len(hand_scores) == 2:
                            margin = abs(hand_scores[0] - hand_scores[1])
                            score_margins.append(float(margin))
                    except Exception:
                        pass  # Skip margin if hand unavailable
                    game_turns_list.append(turn)
                elif turn >= max_turns:
                    logger.debug(
                        "Game %d reached max turns (%d). Scoring as tie.", game_num, max_turns
                    )
                    results["MaxTurnTies"] += 1
                    game_winner = "max_turns"
                    game_turns_list.append(turn)

            except GameStateError as e_game_loop:
                logger.error(
                    "Game state error during game simulation %d: %s",
                    game_num,
                    e_game_loop,
                )
                results["Errors"] += 1
                game_error = True
            except (AgentStateError, ObservationUpdateError) as e_game_loop:
                logger.error(
                    "Agent state error during game simulation %d: %s",
                    game_num,
                    e_game_loop,
                )
                results["Errors"] += 1
                game_error = True
            except Exception as e_game_loop:  # JUSTIFIED: evaluation resilience
                logger.exception(
                    "Critical error during game simulation %d setup or loop: %s",
                    game_num,
                    e_game_loop,
                )
                results["Errors"] += 1
                game_error = True

            game_end = time.perf_counter()
            game_duration_ms = (game_end - game_start) * 1000.0

            # Write JSONL record if logging enabled
            if output_file is not None:
                if game_error and game_winner == "error":
                    pass  # winner already set to "error"
                serialize_start = time.perf_counter()
                record = {
                    "game_id": game_num,
                    "winner": game_winner,
                    "turns": turn,
                    "duration_ms": round(game_duration_ms, 3),
                    "actions": game_actions,
                }
                output_file.write(json.dumps(record) + "\n")
                serialize_end = time.perf_counter()
                jsonl_overhead_ms += (serialize_end - serialize_start) * 1000.0

    finally:
        if output_file is not None:
            output_file.close()

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000.0
    total_time = total_time_ms / 1000.0
    games_played = sum(results.values())

    # Report JSONL overhead if logging was enabled
    if output_path is not None:
        avg_overhead_ms = jsonl_overhead_ms / num_games if num_games > 0 else 0.0
        pct_overhead = (jsonl_overhead_ms / total_time_ms * 100) if total_time_ms > 0 else 0.0
        logger.info(
            "JSONL logging overhead: total=%.1fms, avg=%.3fms/game, pct=%.2f%% of total time",
            jsonl_overhead_ms,
            avg_overhead_ms,
            pct_overhead,
        )
        results["logging_overhead_ms"] = int(jsonl_overhead_ms)
        results["logging_overhead_pct"] = int(pct_overhead)

    # Aggregate enhanced stats — stored as a plain dict attribute (.stats).
    # NOT stored in the Counter itself to preserve backward-compat sum invariants.
    enhanced_stats: Dict = {}
    if score_margins:
        avg_margin = sum(score_margins) / len(score_margins)
        enhanced_stats["avg_score_margin"] = avg_margin
    if game_turns_list:
        avg_turns = sum(game_turns_list) / len(game_turns_list)
        variance = sum((t - avg_turns) ** 2 for t in game_turns_list) / len(game_turns_list)
        std_turns = math.sqrt(variance)
        enhanced_stats["avg_game_turns"] = avg_turns
        enhanced_stats["std_game_turns"] = std_turns
    # Attach as attribute so CLI and tests can access enhanced stats without
    # polluting the Counter sum that existing tests rely on.
    results.stats = enhanced_stats  # type: ignore[attr-defined]

    # Report Results
    logger.info("--- Evaluation Results ---")
    logger.info("Agents: P0 = %s, P1 = %s", agent1_type.upper(), agent2_type.upper())
    logger.info("Games Simulated: %d", num_games)
    logger.info("Valid Games Completed: %d", games_played)
    logger.info("Total Time: %.2f seconds", total_time)
    if games_played > 0:
        logger.info("Time per Game: %.3f seconds", total_time / games_played)

    p0_wins = results.get("P0 Wins", 0)
    p1_wins = results.get("P1 Wins", 0)
    ties = results.get("Ties", 0)
    max_turn_ties = results.get("MaxTurnTies", 0)
    errors = results.get("Errors", 0)
    total_scored = p0_wins + p1_wins + ties + max_turn_ties  # Denominator for percentages

    logger.info(
        "Score: P0 Wins=%d (%.2f%%), P1 Wins=%d (%.2f%%), Ties=%d (%.2f%%), MaxTurnTies=%d, Errors=%d",
        p0_wins,
        (p0_wins / total_scored * 100) if total_scored else 0,
        p1_wins,
        (p1_wins / total_scored * 100) if total_scored else 0,
        ties,
        (ties / total_scored * 100) if total_scored else 0,
        max_turn_ties,
        errors,
    )
    if "avg_score_margin" in enhanced_stats:
        logger.info("Avg Score Margin: %.2f", enhanced_stats["avg_score_margin"])
    if "avg_game_turns" in enhanced_stats:
        logger.info(
            "Game Length: mean=%.1f turns, stdev=%.1f turns",
            enhanced_stats["avg_game_turns"],
            enhanced_stats.get("std_game_turns", 0.0),
        )
    logger.info("--------------------------")
    return results


def run_evaluation_multi_baseline(
    config_path: str,
    checkpoint_path: str,
    num_games: int,
    baselines: List[str],
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> Dict[str, Counter]:
    """
    Evaluate a Deep CFR checkpoint against multiple baseline agents.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to .pt checkpoint.
        num_games: Number of games per baseline matchup.
        baselines: List of baseline agent type strings.
        device: Torch device string for network inference.
        output_dir: If set, writes per-game JSONL to {output_dir}/{baseline}.jsonl.

    Returns:
        Dict mapping baseline name -> Counter with results.
    """
    all_results: Dict[str, Counter] = {}
    for baseline in baselines:
        logger.info("Evaluating deep_cfr vs %s (%d games)...", baseline, num_games)
        output_path = f"{output_dir}/{baseline}.jsonl" if output_dir is not None else None
        results = run_evaluation(
            config_path=config_path,
            agent1_type="deep_cfr",
            agent2_type=baseline,
            num_games=num_games,
            strategy_path=None,
            checkpoint_path=checkpoint_path,
            device=device,
            output_path=output_path,
        )
        all_results[baseline] = results
    return all_results


def run_head_to_head(
    checkpoint_a: str,
    checkpoint_b: str,
    num_games: int,
    config,
    device: str = "cpu",
) -> Dict:
    """
    Play two Deep CFR checkpoints against each other.

    Alternates which checkpoint goes first every game to reduce first-mover bias.

    Returns:
        Dict with checkpoint_a_wins, checkpoint_b_wins, ties, avg_game_turns,
        std_game_turns.
    """
    logger.info(
        "Head-to-head: %s vs %s (%d games)", checkpoint_a, checkpoint_b, num_games
    )

    checkpoint_a_wins = 0
    checkpoint_b_wins = 0
    ties_count = 0
    errors_count = 0
    turns_list: List[int] = []

    for game_num in range(1, num_games + 1):
        # Alternate who goes first
        a_is_p0 = (game_num % 2 == 1)
        if a_is_p0:
            agent0 = DeepCFRAgentWrapper(0, config, checkpoint_a, device=device)
            agent1 = DeepCFRAgentWrapper(1, config, checkpoint_b, device=device)
        else:
            agent0 = DeepCFRAgentWrapper(0, config, checkpoint_b, device=device)
            agent1 = DeepCFRAgentWrapper(1, config, checkpoint_a, device=device)

        agents = [agent0, agent1]

        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            for agent in agents:
                agent.initialize_state(game_state)

            max_turns = (
                config.cambia_rules.max_game_turns
                if config.cambia_rules.max_game_turns > 0
                else 500
            )
            turn = 0

            while not game_state.is_terminal() and turn < max_turns:
                turn += 1
                acting_player_id = game_state.get_acting_player()
                if acting_player_id == -1:
                    break
                current_agent = agents[acting_player_id]
                try:
                    legal_actions = game_state.get_legal_actions()
                    if not legal_actions:
                        if game_state.is_terminal():
                            break
                        break
                    chosen_action = current_agent.choose_action(game_state, legal_actions)
                    _, undo_info = game_state.apply_action(chosen_action)
                    if not callable(undo_info):
                        break
                    # Update agent states
                    obs = current_agent._create_observation(
                        game_state, chosen_action, acting_player_id
                    )
                    if obs:
                        for agent in agents:
                            agent.update_state(obs)
                except Exception as e_turn:
                    logger.error("Head-to-head game %d turn error: %s", game_num, e_turn)
                    break

            turns_list.append(turn)

            if game_state.is_terminal():
                winner = game_state._winner
                if winner is None:
                    ties_count += 1
                elif (a_is_p0 and winner == 0) or (not a_is_p0 and winner == 1):
                    checkpoint_a_wins += 1
                else:
                    checkpoint_b_wins += 1
            else:
                # Max turns reached without terminal — count as tie
                ties_count += 1

        except Exception as e_game:
            logger.error("Head-to-head game %d error: %s", game_num, e_game)
            errors_count += 1

    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0.0
    if turns_list:
        variance = sum((t - avg_turns) ** 2 for t in turns_list) / len(turns_list)
        std_turns = math.sqrt(variance)
    else:
        std_turns = 0.0

    total = checkpoint_a_wins + checkpoint_b_wins + ties_count
    logger.info(
        "Head-to-head results: A=%d (%.1f%%), B=%d (%.1f%%), ties=%d, avg_turns=%.1f",
        checkpoint_a_wins,
        checkpoint_a_wins / total * 100 if total else 0,
        checkpoint_b_wins,
        checkpoint_b_wins / total * 100 if total else 0,
        ties_count,
        avg_turns,
    )

    return {
        "checkpoint_a_wins": checkpoint_a_wins,
        "checkpoint_b_wins": checkpoint_b_wins,
        "ties": ties_count,
        "errors": errors_count,
        "avg_game_turns": avg_turns,
        "std_game_turns": std_turns,
        "total_games": num_games,
    }


def run_head_to_head_typed(
    agent_a_type: str,
    checkpoint_a: str,
    agent_b_type: str,
    checkpoint_b: str,
    num_games: int,
    config,
    device: str = "cpu",
) -> Dict:
    """
    Play two typed agent checkpoints against each other.

    Alternates seat assignment every game to reduce first-mover bias. Uses
    get_agent() to instantiate agents, enabling ESCHER, ReBeL, and Deep CFR.

    Returns:
        Dict with wins_a, wins_b, draws, win_rate_a, win_rate_b, num_games,
        errors, avg_game_turns, std_game_turns.
    """
    logger.info(
        "Head-to-head-typed: %s (%s) vs %s (%s) (%d games)",
        agent_a_type, checkpoint_a, agent_b_type, checkpoint_b, num_games,
    )

    wins_a = 0
    wins_b = 0
    draws = 0
    errors_count = 0
    turns_list: List[int] = []

    for game_num in range(1, num_games + 1):
        a_is_p0 = (game_num % 2 == 1)

        try:
            if a_is_p0:
                agent0 = get_agent(agent_a_type, 0, config, checkpoint_path=checkpoint_a, device=device)
                agent1 = get_agent(agent_b_type, 1, config, checkpoint_path=checkpoint_b, device=device)
            else:
                agent0 = get_agent(agent_b_type, 0, config, checkpoint_path=checkpoint_b, device=device)
                agent1 = get_agent(agent_a_type, 1, config, checkpoint_path=checkpoint_a, device=device)

            agents = [agent0, agent1]

            game_state = CambiaGameState(house_rules=config.cambia_rules)
            for agent in agents:
                if hasattr(agent, "initialize_state"):
                    agent.initialize_state(game_state)

            max_turns = (
                config.cambia_rules.max_game_turns
                if getattr(config.cambia_rules, "max_game_turns", 0) > 0
                else 500
            )
            turn = 0

            while not game_state.is_terminal() and turn < max_turns:
                turn += 1
                acting_player_id = game_state.get_acting_player()
                if acting_player_id == -1:
                    break
                current_agent = agents[acting_player_id]
                try:
                    legal_actions = game_state.get_legal_actions()
                    if not legal_actions:
                        if game_state.is_terminal():
                            break
                        break
                    chosen_action = current_agent.choose_action(game_state, legal_actions)
                    _, undo_info = game_state.apply_action(chosen_action)
                    if not callable(undo_info):
                        break
                    if hasattr(current_agent, "_create_observation"):
                        obs = current_agent._create_observation(
                            game_state, chosen_action, acting_player_id
                        )
                        if obs:
                            for agent in agents:
                                if hasattr(agent, "update_state"):
                                    agent.update_state(obs)
                except Exception as e_turn:
                    logger.error(
                        "Head-to-head-typed game %d turn error: %s", game_num, e_turn
                    )
                    break

            turns_list.append(turn)

            if game_state.is_terminal():
                winner = game_state._winner
                if winner is None:
                    draws += 1
                elif (a_is_p0 and winner == 0) or (not a_is_p0 and winner == 1):
                    wins_a += 1
                else:
                    wins_b += 1
            else:
                draws += 1

        except Exception as e_game:
            logger.error("Head-to-head-typed game %d error: %s", game_num, e_game)
            errors_count += 1

    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0.0
    if turns_list:
        variance = sum((t - avg_turns) ** 2 for t in turns_list) / len(turns_list)
        std_turns = math.sqrt(variance)
    else:
        std_turns = 0.0

    total_scored = wins_a + wins_b + draws
    logger.info(
        "Head-to-head-typed: A=%d (%.1f%%), B=%d (%.1f%%), draws=%d, errors=%d, avg_turns=%.1f",
        wins_a,
        wins_a / total_scored * 100 if total_scored else 0,
        wins_b,
        wins_b / total_scored * 100 if total_scored else 0,
        draws,
        errors_count,
        avg_turns,
    )

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / total_scored if total_scored else 0.0,
        "win_rate_b": wins_b / total_scored if total_scored else 0.0,
        "num_games": num_games,
        "errors": errors_count,
        "avg_game_turns": avg_turns,
        "std_game_turns": std_turns,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cambia Agents Head-to-Head")
    parser.add_argument(
        "agent1", help="Type of Agent 1 (P0)", choices=list(AGENT_REGISTRY.keys())
    )
    parser.add_argument(
        "agent2", help="Type of Agent 2 (P1)", choices=list(AGENT_REGISTRY.keys())
    )
    parser.add_argument(
        "-n", "--num_games", type=int, default=100, help="Number of games to simulate"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default=None,
        help="Path to saved CFR agent strategy file (required if agent type is 'cfr')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging for evaluation script",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src.game.engine").setLevel(logging.INFO)
        logging.getLogger("src.agents.baseline_agents").setLevel(logging.DEBUG)
        logging.getLogger("src.agent_state").setLevel(
            logging.INFO
        )  # Keep agent state less verbose unless debugging it
    else:
        # Silence logs below INFO from libraries if not verbose
        logging.getLogger("src.game.engine").setLevel(logging.WARNING)
        logging.getLogger("src.agents.baseline_agents").setLevel(logging.INFO)
        logging.getLogger("src.agent_state").setLevel(logging.WARNING)

    run_evaluation(
        config_path=args.config,
        agent1_type=args.agent1,
        agent2_type=args.agent2,
        num_games=args.num_games,
        strategy_path=args.strategy,
    )
