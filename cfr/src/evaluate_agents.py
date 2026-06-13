"""Script to evaluate different Cambia agents against each other."""

import argparse
import hashlib
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


def _drawn_card_bucket_from_game_state(game_state) -> int:
    """The acting player's own drawn-card bucket, or -1 if none is pending.

    Mirrors the Go FFI ``cambia_game_get_drawn_card_bucket`` the v2 trainer
    feeds ``encode_infoset_eppbs_interleaved_v2``: a bucket only exists while a
    discard decision is pending (POST_DRAW); otherwise -1. The drawn card is the
    acting player's own private information, legitimately known at decision time.
    Used by the v2 eval wrappers so their encoding matches the training
    distribution byte-for-byte on dims [0:11] (the drawn-card one-hot).
    """
    try:
        from src.abstraction import get_card_bucket
        from src.constants import CardBucket

        if getattr(game_state, "snap_phase_active", False):
            return -1
        if not isinstance(getattr(game_state, "pending_action", None), ActionDiscard):
            return -1
        drawn = (getattr(game_state, "pending_action_data", None) or {}).get("drawn_card")
        if drawn is None:
            return -1
        bucket = get_card_bucket(drawn)
        return int(bucket.value) if bucket != CardBucket.UNKNOWN else -1
    except Exception:  # JUSTIFIED: evaluation resilience -- fall back to "no draw"
        return -1


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

    def __init__(self, player_id: int, config, device: str = "cpu", use_argmax: bool = False):
        super().__init__(player_id, config)
        import torch
        self._torch = torch
        self.device = torch.device(device)
        self.agent_state: Optional[AgentState] = None
        self._use_argmax = use_argmax

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
        """Encode agent state using EP-PBS encoding for evaluation.

        Dispatches to the correct layout encoder based on encoding_layout and network_type,
        mirroring the logic in deep_worker.py:_encode_ep_pbs().
        """
        from src.encoding import (
            encode_infoset_eppbs,
            encode_infoset_eppbs_interleaved,
            encode_infoset_eppbs_dealiased,
        )

        _INTERLEAVED_NETWORK_TYPES = frozenset({"slot_film", "slot_multiply"})

        st = self.agent_state
        # Determine cambia_state: 0=self called, 1=opponent called, 2=none
        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == self.player_id:
            cambia_state = 0
        else:
            cambia_state = 1

        kwargs = dict(
            slot_tags=[t.value if hasattr(t, 'value') else int(t) for t in st.slot_tags],
            slot_buckets=[int(b) for b in st.slot_buckets],
            discard_top_bucket=st.known_discard_top_bucket.value if hasattr(st.known_discard_top_bucket, 'value') else int(st.known_discard_top_bucket),
            stock_estimate=st.stockpile_estimate.value if hasattr(st.stockpile_estimate, 'value') else int(st.stockpile_estimate),
            game_phase=st.game_phase.value if hasattr(st.game_phase, 'value') else int(st.game_phase),
            decision_context=decision_context.value if hasattr(decision_context, 'value') else int(decision_context),
            cambia_state=cambia_state,
        )

        layout = getattr(self, "_encoding_layout", "auto")
        network_type = getattr(self, "_network_type", "mlp")

        if layout == "flat_dealiased":
            kwargs["own_hand_size"] = len(st.own_hand)
            kwargs["opp_hand_size"] = st.opponent_card_count
            encoding = encode_infoset_eppbs_dealiased(**kwargs)
        elif layout == "interleaved" or network_type in _INTERLEAVED_NETWORK_TYPES:
            kwargs["own_hand_size"] = len(st.own_hand)
            kwargs["opp_hand_size"] = st.opponent_card_count
            encoding = encode_infoset_eppbs_interleaved(**kwargs)
        else:
            encoding = encode_infoset_eppbs(**kwargs)

        # Truncate to network input_dim for backward compat (200→224 migration)
        expected_dim = getattr(self, "_net_input_dim", len(encoding))
        if len(encoding) > expected_dim:
            encoding = encoding[:expected_dim]
        return encoding

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
        use_argmax: bool = False,
    ):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
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
        network_type = dcfr_config.get("network_type", "residual")
        self._encoding_mode = dcfr_config.get("encoding_mode", "legacy")
        self._encoding_layout = dcfr_config.get("encoding_layout", "auto")
        self._network_type = network_type

        from src.constants import EP_PBS_INPUT_DIM
        # Use checkpoint's input_dim if available (handles 200→224 migration)
        net_input_dim = dcfr_config.get(
            "input_dim",
            EP_PBS_INPUT_DIM if self._encoding_mode == "ep_pbs" else INPUT_DIM,
        )
        self._net_input_dim = net_input_dim

        self.advantage_net = build_advantage_network(
            input_dim=net_input_dim,
            hidden_dim=hidden_dim,
            output_dim=NUM_ACTIONS,
            dropout=0.1,
            validate_inputs=False,
            num_hidden_layers=num_hidden_layers,
            use_residual=use_residual,
            network_type=network_type,
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

        if self._use_argmax:
            chosen_local = np.argmax(legal_probs)
        else:
            chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


# --- ESCHER Agent Wrapper ---


class ESCHERAgentWrapper(NeuralAgentWrapper):
    """
    Wraps a trained ESCHER network for use in evaluation.

    Supports two checkpoint formats:
    - New format (SD-CFR + ESCHER): loads AdvantageNetwork (advantage_net_state_dict),
      uses get_strategy_from_advantages() for action selection. Supports EP-PBS encoding.
    - Legacy format: loads StrategyNetwork (strategy_net_state_dict) for backward compat.
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str,
        device: str = "cpu",
        use_argmax: bool = False,
    ):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        self._INPUT_DIM = INPUT_DIM
        self._NUM_ACTIONS = NUM_ACTIONS

        torch = self._torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        dcfr_config = checkpoint.get("dcfr_config", {})

        # Warn on cambia_rules mismatch
        self._load_cambia_rules_mismatch_check(checkpoint, config, player_id)

        hidden_dim = dcfr_config.get("hidden_dim", 256)
        num_hidden_layers = dcfr_config.get("num_hidden_layers", 3)
        use_residual = dcfr_config.get("use_residual", True)
        network_type = dcfr_config.get("network_type", "residual")
        self._encoding_mode = dcfr_config.get("encoding_mode", "legacy")
        self._encoding_layout = dcfr_config.get("encoding_layout", "auto")
        self._network_type = network_type

        from src.constants import EP_PBS_INPUT_DIM
        # Use checkpoint's input_dim if available (handles 200→224 migration)
        net_input_dim = dcfr_config.get(
            "input_dim",
            EP_PBS_INPUT_DIM if self._encoding_mode == "ep_pbs" else INPUT_DIM,
        )
        self._net_input_dim = net_input_dim

        if "advantage_net_state_dict" in checkpoint:
            # New format: AdvantageNetwork + regret matching (SD-CFR ESCHER)
            from src.networks import build_advantage_network, get_strategy_from_advantages
            self._get_strategy_from_advantages = get_strategy_from_advantages
            self.advantage_net = build_advantage_network(
                input_dim=net_input_dim,
                hidden_dim=hidden_dim,
                output_dim=NUM_ACTIONS,
                dropout=0.1,
                validate_inputs=False,
                num_hidden_layers=num_hidden_layers,
                use_residual=use_residual,
                network_type=network_type,
            )
            self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
            self.advantage_net.to(self.device)
            self.advantage_net.eval()
            self.policy_net = None
        else:
            # Legacy format: StrategyNetwork (backward compat for old checkpoints)
            from src.networks import StrategyNetwork
            self.advantage_net = None
            self._get_strategy_from_advantages = None
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
        """Choose an action using regret matching (new) or direct policy probabilities (legacy)."""
        from src.encoding import encode_infoset, encode_action_mask, index_to_action
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
            logger.error("ESCHERAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            if self.advantage_net is not None:
                # New format: advantages → regret matching → strategy
                advantages = self.advantage_net(feat_t, mask_t)
                strategy = self._get_strategy_from_advantages(advantages, mask_t)
                probs = strategy.squeeze(0).cpu().numpy()
            else:
                # Legacy format: direct strategy probabilities
                probs = self.policy_net(feat_t, mask_t)
                probs = probs.squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        if self._use_argmax:
            chosen_local = np.argmax(legal_probs)
        else:
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
        use_search: bool = True,
    ):
        super().__init__(player_id, config, device=device)
        from src.networks import PBSValueNetwork, PBSPolicyNetwork
        from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES, uniform_range

        self.use_search = use_search
        self._range_p0 = uniform_range()
        self._range_p1 = uniform_range()

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

    def initialize_state(self, initial_game_state):
        """Reset ranges to uniform and delegate to parent."""
        from src.pbs import uniform_range
        self._range_p0 = uniform_range()
        self._range_p1 = uniform_range()
        super().initialize_state(initial_game_state)

    def _build_pbs(self, game_state):
        """Build a PBS from the current game state using tracked ranges."""
        from src.pbs import (
            PBS, make_public_features,
            PHASE_DRAW, PHASE_DISCARD, PHASE_ABILITY, PHASE_SNAP, PHASE_TERMINAL,
        )

        range_p0 = self._range_p0
        range_p1 = self._range_p1
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

            discard_top_bucket = None
            try:
                from src.pbs import rank_to_bucket
                from src.constants import JOKER_RANK_STR, RED_SUITS, ALL_RANKS_STR
                discard_card = game_state.get_discard_top()
                if discard_card is not None:
                    rank_int = ALL_RANKS_STR.index(discard_card.rank)
                    bucket = rank_to_bucket(rank_int)
                    # Distinguish Red King (bucket 1) from Black King (bucket 8)
                    if discard_card.rank == "K" and discard_card.suit in RED_SUITS:
                        bucket = 1
                    discard_top_bucket = bucket
            except Exception:
                discard_top_bucket = None

            public_features = make_public_features(
                turn=game_state.get_turn_number(),
                max_turns=getattr(self.config.cambia_rules, "max_game_turns", 100),
                phase=phase,
                discard_top_bucket=discard_top_bucket,
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

        # Update own range using observed action and current strategy.
        # Policy matrix: broadcast probs_np uniformly across hand types.
        # (Full per-hand-type range update requires GoEngine + NUM_HAND_TYPES
        # forward passes; this trivial update maintains the interface.)
        try:
            from src.pbs import update_range, NUM_HAND_TYPES
            policy_matrix = np.tile(probs_np, (NUM_HAND_TYPES, 1))
            if self.player_id == 0:
                self._range_p0 = update_range(self._range_p0, int(chosen_global_idx), policy_matrix)
            else:
                self._range_p1 = update_range(self._range_p1, int(chosen_global_idx), policy_matrix)
        except Exception:
            pass

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


class GTCFRAgentWrapper(NeuralAgentWrapper):
    """Evaluation wrapper for GT-CFR agent.

    Uses CVPN direct inference at each decision point to select actions.
    Maintains range distributions across turns (same approach as ReBeLAgentWrapper).

    Note: Full GT-CFR search (GTCFRSearch.search()) requires GoEngine, which is
    not available in the Python eval harness. This wrapper uses CVPN direct
    inference on PBS-encoded state as a proxy for the search policy.
    """

    def __init__(self, player_id: int, config, checkpoint_path: str = "", **kwargs):
        device = kwargs.get("device", "cpu")
        self._deterministic = kwargs.get("deterministic", True)
        self._per_hand_ranges = kwargs.get("per_hand_ranges", False)
        super().__init__(player_id, config, device=device)
        import torch
        from src.networks import build_cvpn
        from src.pbs import uniform_range

        self._cvpn = build_cvpn(
            hidden_dim=config.deep_cfr.gtcfr_cvpn_hidden_dim,
            num_blocks=config.deep_cfr.gtcfr_cvpn_num_blocks,
            validate_inputs=False,
            detach_policy_grad=False,
        )
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self._cvpn.load_state_dict(ckpt["cvpn_state_dict"])
        self._cvpn.to(self.device)
        self._cvpn.eval()

        self._range_p0 = uniform_range()
        self._range_p1 = uniform_range()

        logger.info(
            "GTCFRAgent P%d initialized (checkpoint=%s)",
            self.player_id,
            checkpoint_path or "none",
        )

    def initialize_state(self, initial_game_state):
        """Reset ranges to uniform and delegate to parent."""
        from src.pbs import uniform_range
        self._range_p0 = uniform_range()
        self._range_p1 = uniform_range()
        super().initialize_state(initial_game_state)

    def _build_pbs(self, game_state):
        """Build a PBS from current game state using tracked ranges."""
        from src.pbs import (
            PBS, make_public_features,
            PHASE_DRAW, PHASE_DISCARD, PHASE_ABILITY, PHASE_SNAP, PHASE_TERMINAL,
        )

        range_p0 = self._range_p0
        range_p1 = self._range_p1
        try:
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
            stockpile_total = 46

            discard_top_bucket = None
            try:
                from src.pbs import rank_to_bucket
                from src.constants import RED_SUITS, ALL_RANKS_STR
                discard_card = game_state.get_discard_top()
                if discard_card is not None:
                    rank_int = ALL_RANKS_STR.index(discard_card.rank)
                    bucket = rank_to_bucket(rank_int)
                    if discard_card.rank == "K" and discard_card.suit in RED_SUITS:
                        bucket = 1
                    discard_top_bucket = bucket
            except Exception:
                discard_top_bucket = None

            public_features = make_public_features(
                turn=game_state.get_turn_number(),
                max_turns=getattr(self.config.cambia_rules, "max_game_turns", 100),
                phase=phase,
                discard_top_bucket=discard_top_bucket,
                stockpile_remaining=stockpile_remaining,
                stockpile_total=stockpile_total,
            )
        except Exception:
            from src.pbs import NUM_PUBLIC_FEATURES
            public_features = np.zeros(NUM_PUBLIC_FEATURES, dtype=np.float32)

        return PBS(range_p0=range_p0, range_p1=range_p1, public_features=public_features)

    def choose_action(self, game_state, legal_actions, **kwargs) -> GameAction:
        """Select action via CVPN policy inference."""
        from src.encoding import encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError
        from src.pbs import encode_pbs
        import torch.nn.functional as F

        if not self.agent_state or game_state is None:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        try:
            pbs = self._build_pbs(game_state)
            pbs_enc = encode_pbs(pbs)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("GTCFRAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            pbs_t = torch.from_numpy(pbs_enc).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            _, policy_logits = self._cvpn(pbs_t, mask_t)
            probs_np = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        if self._deterministic:
            chosen_local = np.argmax(legal_probs)
        else:
            chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        # Range update: per-hand-type (slow, correct) or tiled (fast, approximate)
        try:
            from src.pbs import update_range, NUM_HAND_TYPES
            if self._per_hand_ranges:
                from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs
                policy_matrix = compute_policy_matrix_cvpn_from_pbs(
                    self._cvpn, pbs, action_mask, self.player_id,
                    self._range_p0, self._range_p1,
                )
            else:
                policy_matrix = np.tile(probs_np, (NUM_HAND_TYPES, 1))
            if self.player_id == 0:
                self._range_p0 = update_range(self._range_p0, int(chosen_global_idx), policy_matrix)
            else:
                self._range_p1 = update_range(self._range_p1, int(chosen_global_idx), policy_matrix)
        except Exception:
            pass

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)

    def observe_action(self, action, acting_player: int, **kwargs):
        """Update opponent range after observing their action (uniform policy fallback)."""
        try:
            from src.pbs import update_range, NUM_HAND_TYPES
            from src.encoding import NUM_ACTIONS, encode_action_mask
            uniform_policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            policy_matrix = np.tile(uniform_policy, (NUM_HAND_TYPES, 1))
            action_mask = encode_action_mask([action])
            legal_indices = np.where(action_mask)[0]
            if len(legal_indices) > 0:
                action_idx = int(legal_indices[0])
                if acting_player == 0:
                    self._range_p0 = update_range(self._range_p0, action_idx, policy_matrix)
                else:
                    self._range_p1 = update_range(self._range_p1, action_idx, policy_matrix)
        except Exception:
            pass

    def reset(self):
        """Reset ranges for a new game."""
        from src.pbs import uniform_range
        self._range_p0 = uniform_range()
        self._range_p1 = uniform_range()


class SoGAgentWrapper(GTCFRAgentWrapper):
    """Search-at-eval wrapper using a shadow GoEngine and SoGSearch.

    Maintains a matched-deck GoEngine (via cambia_game_new_with_deck) as a
    shadow simulator. At each decision point, runs SoGSearch on the shadow
    engine to get a high-quality policy, then applies the chosen action to
    keep the shadow engine in sync with the Python game.

    Falls back to CVPN-only inference (parent class) if GoEngine is
    unavailable (e.g., libcambia.so not found).
    """

    def __init__(self, player_id: int, config, checkpoint_path: str = "", **kwargs):
        super().__init__(player_id, config, checkpoint_path, **kwargs)
        self._eval_budget = int(kwargs.get("eval_budget", 200))
        self._sog_c_puct = float(kwargs.get("c_puct", 2.0))
        self._sog_cfr_iters = int(kwargs.get("cfr_iters", 10))
        self._go_engine = None
        self._sog_search = None
        self._last_tree = None
        self._last_action_idx: Optional[int] = None

    def initialize_state(self, initial_game_state):
        """Reset ranges, create shadow GoEngine from Python game state."""
        super().initialize_state(initial_game_state)
        self._cleanup_search()
        try:
            from src.ffi.bridge import GoEngine, extract_deck_from_python_game
            from src.cfr.sog_search import SoGSearch
            deck_indices, starting_player = extract_deck_from_python_game(initial_game_state)
            house_rules = getattr(self.config, "cambia_rules", None)
            self._go_engine = GoEngine.from_deck(deck_indices, starting_player, house_rules)
            _exp_k = 3
            if hasattr(self.config, "deep_cfr") and hasattr(self.config.deep_cfr, "gtcfr_expansion_k"):
                _exp_k = self.config.deep_cfr.gtcfr_expansion_k
            self._sog_search = SoGSearch(
                self._cvpn,
                train_budget=self._eval_budget,
                eval_budget=self._eval_budget,
                c_puct=self._sog_c_puct,
                cfr_iters_per_expansion=self._sog_cfr_iters,
                expansion_k=_exp_k,
                device=str(self.device),
            )
        except Exception as e:
            logger.warning(
                "SoGAgentWrapper P%d: GoEngine init failed (%s). Using CVPN-only fallback.",
                self.player_id, e,
            )
            self._go_engine = None
            self._sog_search = None

    def choose_action(self, game_state, legal_actions, **kwargs) -> GameAction:
        """Run SoGSearch on shadow GoEngine; fall back to CVPN-only if unavailable."""
        from src.encoding import encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError

        if self._go_engine is None or self._sog_search is None:
            return super().choose_action(game_state, legal_actions, **kwargs)

        legal_list = list(legal_actions)
        if not legal_list:
            return super().choose_action(game_state, legal_actions, **kwargs)

        try:
            result = self._sog_search.search(
                self._go_engine,
                self._range_p0,
                self._range_p1,
                prior_tree=self._last_tree,
                action_taken=self._last_action_idx,
            )
            self._last_tree = self._sog_search.get_tree()

            policy = result.policy  # (NUM_ACTIONS,) float32
            action_mask = encode_action_mask(legal_list)
            legal_indices = np.where(action_mask)[0]
            if len(legal_indices) == 0:
                return random.choice(legal_list)

            legal_probs = policy[legal_indices].astype(np.float32)
            prob_sum = legal_probs.sum()
            if prob_sum <= 0:
                legal_probs = np.ones(len(legal_indices), dtype=np.float32) / len(legal_indices)
            else:
                legal_probs /= prob_sum

            if self._deterministic:
                chosen_local = np.argmax(legal_probs)
            else:
                chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
            chosen_global_idx = int(legal_indices[chosen_local])
            self._last_action_idx = chosen_global_idx

            # Per-hand-type policy matrix BEFORE apply_action (needs pre-action state)
            policy_matrix_np = None
            try:
                from src.cfr.range_utils import compute_policy_matrix_cvpn
                policy_matrix_np = compute_policy_matrix_cvpn(
                    self._cvpn, self._go_engine,
                    self._range_p0, self._range_p1,
                )
            except Exception:
                pass

            # Apply to shadow GoEngine to keep it in sync
            try:
                self._go_engine.apply_action(chosen_global_idx)
            except Exception as e_go:
                logger.warning(
                    "SoGAgent P%d: GoEngine apply_action failed: %s. Disabling shadow engine.",
                    self.player_id, e_go,
                )
                self._cleanup_search()

            # Bayesian range update with per-hand-type policies
            try:
                from src.pbs import update_range
                if policy_matrix_np is None:
                    from src.pbs import NUM_HAND_TYPES
                    policy_matrix_np = np.tile(policy, (NUM_HAND_TYPES, 1))
                if self.player_id == 0:
                    self._range_p0 = update_range(self._range_p0, chosen_global_idx, policy_matrix_np)
                else:
                    self._range_p1 = update_range(self._range_p1, chosen_global_idx, policy_matrix_np)
            except Exception:
                pass

            try:
                return index_to_action(chosen_global_idx, legal_list)
            except ActionEncodingError:
                return random.choice(legal_list)

        except Exception as e:
            logger.error("SoGAgentWrapper P%d search error: %s", self.player_id, e)
            return super().choose_action(game_state, legal_actions, **kwargs)

    def update_state(self, observation):
        """Update Python agent state and apply opponent actions to shadow GoEngine."""
        super().update_state(observation)
        if self._go_engine is None:
            return
        acting = getattr(observation, "acting_player", -1)
        if acting == -1 or acting == self.player_id:
            return
        action = getattr(observation, "action", None)
        if action is None:
            return
        try:
            from src.encoding import encode_action_mask
            action_mask = encode_action_mask([action])
            legal_indices = np.where(action_mask)[0]
            if len(legal_indices) > 0:
                opp_idx = int(legal_indices[0])
                self._go_engine.apply_action(opp_idx)
        except Exception as e:
            logger.warning(
                "SoGAgent P%d: opponent GoEngine apply failed: %s. Disabling shadow engine.",
                self.player_id, e,
            )
            self._cleanup_search()

    def reset(self):
        """Reset ranges and cleanup GoEngine."""
        super().reset()
        self._cleanup_search()

    def _cleanup_search(self):
        """Free GoEngine handle and SoGSearch tree."""
        if self._sog_search is not None:
            try:
                self._sog_search.cleanup()
            except Exception:
                pass
        self._sog_search = None
        self._last_tree = None
        self._last_action_idx = None
        if self._go_engine is not None:
            try:
                self._go_engine.close()
            except Exception:
                pass
            self._go_engine = None

    def __del__(self):
        self._cleanup_search()


class SoGInferenceAgentWrapper(GTCFRAgentWrapper):
    """CVPN-only eval for SoG checkpoints. Inherits all GTCFRAgentWrapper logic.

    Disables per-hand-type range updates (468 forward passes per decision)
    since the CVPN was trained with aggregate ranges. Uses the fast tiled
    surrogate instead. This keeps CVPN-only eval at ~50ms/game vs ~12-28s.
    """

    def __init__(self, player_id: int, config, checkpoint_path: str = "", **kwargs):
        super().__init__(player_id, config, checkpoint_path, **kwargs)
        self._skip_range_matrix = True

    def choose_action(self, game_state, legal_actions, **kwargs):
        """Override to use fast tiled range update instead of 468-pass matrix."""
        from src.encoding import encode_action_mask, index_to_action
        from src.cfr.exceptions import ActionEncodingError
        from src.pbs import encode_pbs
        import torch.nn.functional as F

        if not self.agent_state or game_state is None:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        try:
            pbs = self._build_pbs(game_state)
            pbs_enc = encode_pbs(pbs)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:
            logger.error("SoGInferenceAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            pbs_t = torch.from_numpy(pbs_enc).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            _, policy_logits = self._cvpn(pbs_t, mask_t)
            probs_np = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        legal_indices = np.where(action_mask)[0]
        if len(legal_indices) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_indices]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        if self._deterministic:
            chosen_local = np.argmax(legal_probs)
        else:
            chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        # Fast tiled range update (skip 468-pass matrix for inference-only eval)
        try:
            from src.pbs import update_range, NUM_HAND_TYPES
            policy_matrix = np.tile(probs_np, (NUM_HAND_TYPES, 1))
            if self.player_id == 0:
                self._range_p0 = update_range(self._range_p0, int(chosen_global_idx), policy_matrix)
            else:
                self._range_p1 = update_range(self._range_p1, int(chosen_global_idx), policy_matrix)
        except Exception:
            pass

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
        use_argmax: bool = False,
    ):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
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
        network_type = dcfr_config.get("network_type", "residual")
        self._weighting = dcfr_config.get("sd_cfr_snapshot_weighting", "linear")
        self._encoding_mode = dcfr_config.get("encoding_mode", "legacy")
        self._encoding_layout = dcfr_config.get("encoding_layout", "auto")
        self._network_type = network_type

        from src.constants import EP_PBS_INPUT_DIM
        # Use checkpoint's input_dim if available (handles 200→224 migration)
        net_input_dim = dcfr_config.get(
            "input_dim",
            EP_PBS_INPUT_DIM if self._encoding_mode == "ep_pbs" else INPUT_DIM,
        )
        self._net_input_dim = net_input_dim

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
                network_type=network_type,
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
                # Fall back to sibling file (handles moved run directories)
                sd_snapshots_path = f"{base_path}_sd_snapshots.pt"
            if not os.path.exists(sd_snapshots_path):
                # Try stripping _iter_NNN suffix (snapshot file is shared)
                import re
                canon = re.sub(r"_iter_\d+$", "", base_path)
                sd_snapshots_path = f"{canon}_sd_snapshots.pt"
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
                    network_type=network_type,
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

        if self._use_argmax:
            chosen_local = np.argmax(legal_probs)
        else:
            chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except ActionEncodingError:
            return random.choice(legal_list)


# --- PPO Agent Wrapper ---


class PPOAgentWrapper(BaseAgent):
    """Wraps a trained SB3 MaskablePPO model for evaluation."""

    def __init__(self, player_id: int, config, model_path: str, device: str = "cpu"):
        super().__init__(player_id, config)
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            raise ImportError(
                "sb3-contrib required for PPO agent. "
                "Install with: pip install -e '.[rl]'"
            )
        self._model = MaskablePPO.load(model_path, device=device)
        self._agent_state: Optional[AgentState] = None
        # The model's observation space dictates which encoding layout to feed.
        # 257-dim models were trained on EP-PBS v2 (encoding_version=2); 224-dim
        # models on the v1 base layout. Feeding the wrong width crashes
        # MaskablePPO.predict, which previously forced a random-action fallback.
        try:
            obs_dim = int(self._model.observation_space.shape[0])
        except Exception:
            obs_dim = 224
        self._encoding_version = 2 if obs_dim >= 257 else 1
        self._obs_dim = obs_dim

    def initialize_state(self, initial_game_state):
        """Initialize internal AgentState."""
        initial_hand = initial_game_state.players[self.player_id].hand
        initial_peeks = initial_game_state.players[self.player_id].initial_peek_indices
        self._agent_state = AgentState(
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
                initial_game_state.get_player_card_count(i)
                for i in range(NUM_PLAYERS)
            ],
            stockpile_size=initial_game_state.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=list(initial_game_state.snap_results_log),
            did_cambia_get_called=initial_game_state.cambia_caller_id is not None,
            who_called_cambia=initial_game_state.cambia_caller_id,
            is_game_over=initial_game_state.is_terminal(),
            current_turn=initial_game_state.get_turn_number(),
        )
        self._agent_state.initialize(initial_obs, initial_hand, initial_peeks)

    def update_state(self, observation: AgentObservation):
        """Update internal state based on observation."""
        if not self._agent_state:
            return
        filtered = copy.copy(observation)
        filtered.drawn_card = None
        filtered.peeked_cards = None
        try:
            self._agent_state.update(filtered)
        except Exception as e:
            logger.error("PPOAgent P%d state update error: %s", self.player_id, e)

    def _encode_obs(self, ctx, drawn_card_bucket: int = -1) -> np.ndarray:
        """Encode the PPO player's infoset, dispatching on the model's obs width.

        v2 (257-dim) models trained through ``ppo_env._get_obs`` on the canonical
        ``encode_infoset_eppbs_interleaved_v2`` (posterior + action-history block
        populated from AgentState). v1 (224-dim) models trained on the v1
        interleaved layout. Per-agent parity: each model is encoded at eval the
        same way it was trained, so the detected ``_encoding_version`` selects
        the path. Feeding the wrong width crashes ``MaskablePPO.predict``.
        """
        from src.encoding import (
            encode_infoset_eppbs_interleaved,
            encode_infoset_eppbs_interleaved_v2,
        )

        st = self._agent_state
        if getattr(self, "_encoding_version", 1) == 2:
            return encode_infoset_eppbs_interleaved_v2(
                st, ctx, int(drawn_card_bucket)
            ).astype(np.float32)

        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == self.player_id:
            cambia_state = 0
        else:
            cambia_state = 1

        return encode_infoset_eppbs_interleaved(
            slot_tags=[t.value if hasattr(t, "value") else int(t) for t in st.slot_tags],
            slot_buckets=[int(b) for b in st.slot_buckets],
            discard_top_bucket=(
                st.known_discard_top_bucket.value
                if hasattr(st.known_discard_top_bucket, "value")
                else int(st.known_discard_top_bucket)
            ),
            stock_estimate=(
                st.stockpile_estimate.value
                if hasattr(st.stockpile_estimate, "value")
                else int(st.stockpile_estimate)
            ),
            game_phase=(
                st.game_phase.value
                if hasattr(st.game_phase, "value")
                else int(st.game_phase)
            ),
            decision_context=ctx.value if hasattr(ctx, "value") else int(ctx),
            cambia_state=cambia_state,
            own_hand_size=len(st.own_hand),
            opp_hand_size=st.opponent_card_count,
        ).astype(np.float32)

    def choose_action(self, game_state, legal_actions) -> GameAction:
        """Choose action using the trained PPO model."""
        from src.encoding import encode_action_mask, index_to_action
        from src.constants import (
            DecisionContext,
            ActionDiscard,
            ActionAbilityPeekOwnSelect,
            ActionAbilityPeekOtherSelect,
            ActionAbilityBlindSwapSelect,
            ActionAbilityKingLookSelect,
            ActionAbilityKingSwapDecision,
            ActionSnapOpponentMove,
        )

        if not self._agent_state:
            import random as _random

            return _random.choice(list(legal_actions))

        # Decision context
        if game_state.snap_phase_active:
            ctx = DecisionContext.SNAP_DECISION
        elif game_state.pending_action:
            p = game_state.pending_action
            if isinstance(p, ActionDiscard):
                ctx = DecisionContext.POST_DRAW
            elif isinstance(
                p,
                (
                    ActionAbilityPeekOwnSelect,
                    ActionAbilityPeekOtherSelect,
                    ActionAbilityBlindSwapSelect,
                    ActionAbilityKingLookSelect,
                    ActionAbilityKingSwapDecision,
                ),
            ):
                ctx = DecisionContext.ABILITY_SELECT
            elif isinstance(p, ActionSnapOpponentMove):
                ctx = DecisionContext.SNAP_MOVE
            else:
                ctx = DecisionContext.START_TURN
        else:
            ctx = DecisionContext.START_TURN

        drawn_card_bucket = _drawn_card_bucket_from_game_state(game_state)
        obs = self._encode_obs(ctx, drawn_card_bucket)

        mask = encode_action_mask(list(legal_actions))
        import numpy as np

        action_idx, _ = self._model.predict(
            np.array(obs, dtype=np.float32),
            action_masks=mask,
            deterministic=True,
        )
        try:
            return index_to_action(int(action_idx), list(legal_actions))
        except (ValueError, IndexError):
            import random as _random

            return _random.choice(list(legal_actions))


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
        use_argmax: bool = False,
        **kwargs,
    ):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)
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
            network_type = dcfr_config.get("network_type", "residual")

            self.advantage_net = build_advantage_network(
                input_dim=N_PLAYER_INPUT_DIM,
                hidden_dim=hidden_dim,
                output_dim=N_PLAYER_NUM_ACTIONS,
                dropout=0.0,
                validate_inputs=False,
                num_hidden_layers=num_hidden_layers,
                use_residual=use_residual,
                network_type=network_type,
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

        if self._use_argmax:
            chosen_local = np.argmax(legal_probs)
        else:
            chosen_local = np.random.choice(len(legal_indices), p=legal_probs)
        chosen_global_idx = legal_indices[chosen_local]

        try:
            return index_to_action(int(chosen_global_idx), legal_list)
        except (ActionEncodingError, Exception):
            return random.choice(legal_list)


class MixedOpponentAgent(BaseAgent):
    """Agent that delegates to one of two sub-agents per action, sampled by weight.

    Used for H3 diagnostic: simulates the bugged opponent that played
    epsilon-random (e.g. 0.6 * random + 0.4 * baseline).
    """

    def __init__(
        self,
        player_id: int,
        config: Config,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        weight_a: float = 0.6,
    ):
        super().__init__(player_id, config)
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.weight_a = weight_a

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        if random.random() < self.weight_a:
            return self.agent_a.choose_action(game_state, legal_actions)
        return self.agent_b.choose_action(game_state, legal_actions)


# --- DESCA Agent Wrapper ---


class DESCAAgentWrapper(NeuralAgentWrapper):
    """
    Wraps a trained DESCA AvgStrategyNetwork for evaluation.

    At each decision: encode to 257-dim EP-PBS v2, compute abstract action mask,
    forward through avg-strategy network, sample or argmax, then unabstract to a
    concrete legal action.
    """

    def __init__(
        self,
        player_id: int,
        config,
        checkpoint_path: str,
        device: str = "cpu",
        use_argmax: bool = False,
    ):
        super().__init__(player_id, config, device=device, use_argmax=use_argmax)

        torch = self._torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        try:
            from src.desca_networks import AvgStrategyNetwork
        except ImportError as e:
            raise ImportError(
                f"DESCAAgentWrapper requires desca_networks.py (Stream A): {e}"
            ) from e

        from src.constants import EP_PBS_V2_INPUT_DIM
        from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P

        desca_cfg = checkpoint.get("desca_config", {})
        hidden_dim = desca_cfg.get("hidden_dim", 512)
        input_dim = desca_cfg.get("encoding_dim", EP_PBS_V2_INPUT_DIM)
        num_actions = desca_cfg.get("num_abstract_actions", NUM_ABSTRACT_ACTIONS_2P)

        self._num_abstract_actions = num_actions
        self._input_dim = input_dim

        self.avg_strategy_net = AvgStrategyNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
        self.avg_strategy_net.load_state_dict(checkpoint["avg_strategy_state_dict"])
        self.avg_strategy_net.to(self.device)
        self.avg_strategy_net.eval()

        logger.info(
            "DESCAAgent P%d loaded checkpoint (iter=%s)",
            self.player_id,
            checkpoint.get("iteration", "N/A"),
        )

    def _encode_v2(self, decision_context, drawn_card_bucket: int = -1) -> np.ndarray:
        """Encode agent state to 257-dim EP-PBS v2 via the canonical encoder.

        Routes through ``encode_infoset_eppbs_interleaved_v2`` -- the same
        high-level entry point the DESCA trainer (``desca_worker._encode_state``)
        uses. That encoder derives the card-counting posterior (dims [224:233])
        and action-history window (dims [233:257]) from ``AgentState`` directly,
        plus the v1 history-parity features (observation ages, dead-card
        histogram, turn progress). The prior hand-rolled low-level call omitted
        all of those, zeroing ~57 input dims relative to training (RC-B).
        """
        from src.encoding import encode_infoset_eppbs_interleaved_v2

        return encode_infoset_eppbs_interleaved_v2(
            self.agent_state, decision_context, int(drawn_card_bucket)
        )

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose an action via DESCA avg-strategy network over the abstract action space."""
        from src.action_abstraction import abstract_actions, unabstract

        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)
        drawn_card_bucket = _drawn_card_bucket_from_game_state(game_state)

        try:
            features = self._encode_v2(decision_context, drawn_card_bucket)
            abstract_mask = abstract_actions(legal_list, self.agent_state)
        except Exception as e:  # JUSTIFIED: evaluation resilience
            logger.error("DESCAAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        torch = self._torch
        with torch.inference_mode():
            feat_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(abstract_mask).bool().unsqueeze(0).to(self.device)
            probs = self.avg_strategy_net(feat_t, mask_t)
            probs_np = probs.squeeze(0).cpu().numpy()

        legal_abstract = np.where(abstract_mask)[0]
        if len(legal_abstract) == 0:
            return random.choice(legal_list)

        legal_probs = probs_np[legal_abstract]
        prob_sum = legal_probs.sum()
        if prob_sum <= 0:
            legal_probs = np.ones(len(legal_abstract)) / len(legal_abstract)
        else:
            legal_probs = legal_probs / prob_sum

        if self._use_argmax:
            chosen_local = int(np.argmax(legal_probs))
        else:
            chosen_local = int(np.random.choice(len(legal_abstract), p=legal_probs))

        chosen_abstract_idx = int(legal_abstract[chosen_local])

        seed = hash((id(game_state), chosen_abstract_idx)) & 0xFFFF_FFFF
        try:
            return unabstract(chosen_abstract_idx, legal_list, self.agent_state, seed=seed)
        except (ValueError, Exception) as e:
            logger.error("DESCAAgent P%d unabstract error: %s", self.player_id, e)
            return random.choice(legal_list)


# --- Constants ---

# Canonical mean_imp baseline set. Import this from all scripts that compute mean_imp.
MEAN_IMP_BASELINES: tuple[str, ...] = (
    "random_no_cambia",
    "random_late_cambia",
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
)

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
    "ppo": PPOAgentWrapper,
    "rebel": ReBeLAgentWrapper,
    "gtcfr": GTCFRAgentWrapper,
    "sog": SoGAgentWrapper,
    "sog_inference": SoGInferenceAgentWrapper,
    "nplayer": NPlayerAgentWrapper,
    "desca": DESCAAgentWrapper,
    "dense-escher": DESCAAgentWrapper,
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
        use_argmax = kwargs.get("use_argmax", False)
        return agent_class(player_id, config, checkpoint_path, device=device, use_argmax=use_argmax)
    elif agent_type.lower() == "ppo":
        model_path = kwargs.get("model_path") or kwargs.get("checkpoint_path")
        if not model_path:
            raise ValueError("PPOAgentWrapper requires 'model_path' or 'checkpoint_path'.")
        device = kwargs.get("device", "cpu")
        return PPOAgentWrapper(player_id, config, model_path, device=device)
    elif agent_type.lower() == "rebel":
        checkpoint_path = kwargs.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("ReBeLAgentWrapper requires 'checkpoint_path'.")
        device = kwargs.get("device", "cpu")
        return ReBeLAgentWrapper(player_id, config, checkpoint_path, device=device)
    elif agent_type.lower() == "gtcfr":
        checkpoint_path = kwargs.get("checkpoint_path", "")
        device = kwargs.get("device", "cpu")
        deterministic = kwargs.get("use_argmax", True)
        return GTCFRAgentWrapper(player_id, config, checkpoint_path, device=device, deterministic=deterministic)
    elif agent_type.lower() == "sog":
        checkpoint_path = kwargs.get("checkpoint_path", "")
        device = kwargs.get("device", "cpu")
        eval_budget = int(kwargs.get("eval_budget", 200))
        c_puct = float(kwargs.get("c_puct", 2.0))
        cfr_iters = int(kwargs.get("cfr_iters", 10))
        deterministic = kwargs.get("use_argmax", True)
        return SoGAgentWrapper(
            player_id, config, checkpoint_path,
            device=device, eval_budget=eval_budget, c_puct=c_puct, cfr_iters=cfr_iters,
            deterministic=deterministic,
        )
    elif agent_type.lower() == "sog_inference":
        checkpoint_path = kwargs.get("checkpoint_path", "")
        device = kwargs.get("device", "cpu")
        deterministic = kwargs.get("use_argmax", True)
        return SoGInferenceAgentWrapper(player_id, config, checkpoint_path, device=device, deterministic=deterministic)
    elif agent_type.lower() in ("desca", "dense-escher"):
        checkpoint_path = kwargs.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("DESCAAgentWrapper requires 'checkpoint_path'.")
        device = kwargs.get("device", "cpu")
        use_argmax = kwargs.get("use_argmax", False)
        return DESCAAgentWrapper(player_id, config, checkpoint_path, device=device, use_argmax=use_argmax)
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
    use_argmax: bool = False,
    seat_scheme: str = "alternated",
    crn_seed_base: Optional[int] = None,
) -> Counter:
    """Runs head-to-head evaluation between two agents. Returns results Counter.

    Win attribution is agent-relative, not seat-relative: results["P0 Wins"]
    counts wins by the agent under test (agent1) and results["P1 Wins"] counts
    wins by the opponent (agent2), regardless of which physical seat each
    occupied in a given game. This keeps the Counter's meaning stable as "the
    evaluated agent's record" across both seat schemes, so downstream win-rate
    math (p0_wins / total) and the baseline strength-ordering invariants remain
    correct after alternation.

    Args:
        seat_scheme: "alternated" (default) swaps which seat the agent under test
            occupies every game to cancel first-mover bias; "fixed" keeps agent1
            in seat 0 for every game (legacy behavior).
        crn_seed_base: When set, each seat-swap pair of games shares a
            deterministic deck seed so the agent under test faces an identical
            deal from both seats (common random numbers). None leaves the deck
            unseeded (process entropy), preserving prior behavior.
    """
    seat_scheme = (seat_scheme or "alternated").lower()
    if seat_scheme not in ("alternated", "fixed"):
        logger.warning("Unknown seat_scheme %r; falling back to 'alternated'.", seat_scheme)
        seat_scheme = "alternated"
    alternate_seats = seat_scheme == "alternated"
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
    _checkpoint_agent_types = {"deep_cfr", "escher", "sd_cfr", "nplayer", "ppo", "rebel", "gtcfr", "sog", "sog_inference", "desca", "dense-escher"}
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
            agent1_kwargs = {"checkpoint_path": checkpoint_path, "device": device, "use_argmax": use_argmax}
        if agent2_type.lower() == "cfr":
            agent2_kwargs = {"average_strategy": average_strategy}
        elif agent2_type.lower() in _checkpoint_agent_types:
            agent2_kwargs = {"checkpoint_path": checkpoint_path, "device": device, "use_argmax": use_argmax}

        # Build each side at the seat(s) it will occupy. A wrapper's player_id is
        # baked in at construction and threaded into its AgentState, so swapping
        # seats requires a distinct instance per seat. Construct once up front
        # (loading any checkpoint at most twice total) rather than per game, then
        # reset per-game belief via initialize_state inside the loop.
        #
        # Seat layout: in odd games the agent under test (agent1) sits at seat 0;
        # in even games it sits at seat 1. Under the fixed scheme only the seat-0
        # agent1 / seat-1 agent2 pair is ever used, so the mirror seats are built
        # lazily to avoid redundant checkpoint loads.
        agent1_by_seat: Dict[int, BaseAgent] = {
            0: get_agent(agent1_type, player_id=0, config=config, **agent1_kwargs)
        }
        agent2_by_seat: Dict[int, BaseAgent] = {
            1: get_agent(agent2_type, player_id=1, config=config, **agent2_kwargs)
        }
        if alternate_seats:
            agent1_by_seat[1] = get_agent(
                agent1_type, player_id=1, config=config, **agent1_kwargs
            )
            agent2_by_seat[0] = get_agent(
                agent2_type, player_id=0, config=config, **agent2_kwargs
            )
        logger.info(
            "Agents instantiated (seat_scheme=%s, crn=%s).",
            seat_scheme,
            "on" if crn_seed_base is not None else "off",
        )
    except ValueError as e:
        logger.error("Error creating agents: %s", e)
        sys.exit(1)

    # Stable string identifying the agent under test, used in the CRN seed hash so
    # the deck sequence is deterministic per (agent, baseline) matchup.
    crn_identity = checkpoint_path or agent1_type

    results: Counter = Counter()
    start_time = time.perf_counter()
    jsonl_overhead_ms = 0.0
    # Per-game tracking for enhanced stats
    score_margins: List[float] = []
    game_turns_list: List[int] = []
    t1_cambia_count = 0
    # Track how many distinct seats the agent under test actually occupied across
    # decisive games. seat_balanced is reported True only if alternation ran and
    # the agent played from both seats (so the win rate is genuinely seat-fair).
    agent_under_test_seats_used: Set[int] = set()

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
                # Seat assignment: agent under test (agent1) sits at seat 0 on odd
                # games, seat 1 on even games when alternating; always seat 0 under
                # the fixed scheme. Win attribution below maps physical seats back
                # to agent-relative wins, so the Counter stays agent-indexed.
                a_is_p0 = (game_num % 2 == 1) if alternate_seats else True
                if a_is_p0:
                    agents = [agent1_by_seat[0], agent2_by_seat[1]]
                else:
                    agents = [agent2_by_seat[0], agent1_by_seat[1]]
                # Seat index occupied by the agent under test this game.
                agent_seat = 0 if a_is_p0 else 1

                # Common-random-numbers seat pairing: each seat-swap pair shares one
                # deck seed so the agent under test faces the identical deal from
                # both seats, canceling deck-luck variance. Pair index groups games
                # (1,2), (3,4), ... under one seed; the seat differs within a pair.
                deck_seed: Optional[int] = None
                if crn_seed_base is not None:
                    pair_index = (game_num - 1) // 2 if alternate_seats else (game_num - 1)
                    seed_key = f"{crn_seed_base}|{crn_identity}|{agent2_type}|{pair_index}"
                    deck_seed = int(
                        hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16
                    )

                game_state = CambiaGameState(
                    house_rules=config.cambia_rules, seed=deck_seed
                )
                # Reset per-game agent state. Neural/CFR wrappers reinitialize
                # belief explicitly; baseline agents detect a new game lazily by
                # the game_state object id (_needs_reinit). That id check is unsafe
                # when an agent instance is reused across non-consecutive games
                # (as under seat alternation, where each instance plays every other
                # game): Python can recycle a freed game_state's address, so the new
                # game collides with a stale _last_game_id and the agent keeps the
                # prior game's memory, immediately calling Cambia (games collapse to
                # ~4 turns and win rates regress to coin-flips). Invalidating the id
                # sentinel here forces a fresh reinit every game for every agent.
                for agent in agents:
                    if isinstance(agent, (CFRAgentWrapper, NeuralAgentWrapper)):
                        agent.initialize_state(game_state)
                    elif hasattr(agent, "_last_game_id"):
                        agent._last_game_id = None

                turn = 0
                # Safety valve in action-units. The engine itself enforces the
                # max_game_turns cap on its turn-number scale via _check_game_end
                # (becoming terminal and scoring normally), so this local cap is a
                # runaway guard, not the primary termination scale. Sized well above
                # the engine cap in action terms: a single turn may span several
                # actions (draw, ability sub-selects, snap responses), so multiply
                # the turn-number cap by a per-turn action factor.
                engine_turn_cap = (
                    config.cambia_rules.max_game_turns
                    if config.cambia_rules.max_game_turns > 0
                    else 500
                )
                max_turns = engine_turn_cap * 16

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

                        # Track first-turn Cambia calls by the agent under test
                        # (which may sit at either seat under alternation).
                        if turn == 1 and acting_player_id == agent_seat and type(chosen_action).__name__ == "ActionCallCambia":
                            t1_cambia_count += 1

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
                    # Counter is agent-relative: "P0 Wins" = agent under test (agent1),
                    # "P1 Wins" = opponent (agent2), regardless of physical seat. The
                    # JSONL game_winner label below stays seat-relative ("p0"/"p1")
                    # to match the seat-indexed per-action trace.
                    if winner == agent_seat:
                        results["P0 Wins"] += 1
                        agent_under_test_seats_used.add(agent_seat)
                    elif winner is not None and winner != agent_seat:
                        results["P1 Wins"] += 1
                        agent_under_test_seats_used.add(agent_seat)
                    else:
                        results["Ties"] += 1

                    if winner == 0:
                        game_winner = "p0"
                    elif winner == 1:
                        game_winner = "p1"
                    else:
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
                        "Game %d hit the action-count safety valve (%d actions) without "
                        "engine termination. Scoring as MaxTurnTie.", game_num, max_turns
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
    enhanced_stats["t1_cambia_rate"] = t1_cambia_count / num_games if num_games > 0 else 0.0
    # Eval-hygiene provenance: seat scheme actually used, whether the agent under
    # test genuinely played both seats (seat_balanced), and the CRN seed root.
    enhanced_stats["seat_scheme"] = seat_scheme
    enhanced_stats["seat_balanced"] = bool(
        alternate_seats and len(agent_under_test_seats_used) >= 2
    )
    enhanced_stats["selection_mode"] = "argmax" if use_argmax else "stochastic"
    enhanced_stats["crn_seed"] = crn_seed_base
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


def _run_single_baseline(args: tuple) -> tuple:
    """Worker function for parallel baseline evaluation.

    Runs in a spawned process so must re-import everything from scratch.
    Takes a tuple to work with ProcessPoolExecutor.map().

    Returns:
        (baseline_name, Counter_results, stats_dict). The stats dict is returned
        separately because a Counter's dynamically-attached `.stats` attribute is
        NOT preserved by pickling across the ProcessPoolExecutor boundary; the
        parent reattaches it so enhanced/hygiene stats survive parallel runs.
    """
    (
        config_path, agent_type, baseline, num_games, checkpoint_path, device,
        output_path, use_argmax, seat_scheme, crn_seed_base,
    ) = args
    results = run_evaluation(
        config_path=config_path,
        agent1_type=agent_type,
        agent2_type=baseline,
        num_games=num_games,
        strategy_path=None,
        checkpoint_path=checkpoint_path,
        device=device,
        output_path=output_path,
        use_argmax=use_argmax,
        seat_scheme=seat_scheme,
        crn_seed_base=crn_seed_base,
    )
    return baseline, results, getattr(results, "stats", {})


def run_evaluation_multi_baseline(
    config_path: str,
    checkpoint_path: str,
    num_games: int,
    baselines: List[str],
    device: str = "cpu",
    output_dir: Optional[str] = None,
    use_argmax: bool = False,
    agent_type: str = "deep_cfr",
    max_workers: Optional[int] = None,
    seat_scheme: str = "alternated",
    crn_seed_base: Optional[int] = None,
) -> Dict[str, Counter]:
    """
    Evaluate a checkpoint against multiple baseline agents.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to .pt checkpoint.
        num_games: Number of games per baseline matchup.
        baselines: List of baseline agent type strings.
        device: Torch device string for network inference.
        output_dir: If set, writes per-game JSONL to {output_dir}/{baseline}.jsonl.
        use_argmax: If True, use argmax instead of stochastic sampling for action selection.
        agent_type: Agent type string (default: "deep_cfr"). Use "rebel" for ReBeL agents.
        max_workers: Number of parallel baseline workers. None = auto (min of baseline
            count and cpu_count/2, capped at 7). Set to 1 for sequential execution.
        seat_scheme: "alternated" (default) or "fixed". Passed through to run_evaluation.
        crn_seed_base: Common-random-numbers seed root for deck pairing. When None
            (default), a deterministic root is derived from the checkpoint path so
            seat-swap pairs share decks reproducibly across runs while distinct
            checkpoints get distinct deck sequences. Set explicitly to override.

    Returns:
        Dict mapping baseline name -> Counter with results.
    """
    import multiprocessing as mp
    import os as _os

    if max_workers is None:
        cpu_count = _os.cpu_count() or 1
        max_workers = min(len(baselines), max(1, cpu_count // 2), 7)

    # Default CRN on: derive a stable seed root from the checkpoint identity so the
    # deck sequence is reproducible per checkpoint and the agent under test faces
    # identical deals from both seats within each seat-swap pair.
    if crn_seed_base is None:
        ident = str(checkpoint_path or agent_type)
        crn_seed_base = int(hashlib.sha256(ident.encode("utf-8")).hexdigest()[:8], 16)

    # Build argument tuples
    work_items = []
    for baseline in baselines:
        output_path = f"{output_dir}/{baseline}.jsonl" if output_dir is not None else None
        work_items.append((
            config_path, agent_type, baseline, num_games,
            checkpoint_path, device, output_path, use_argmax,
            seat_scheme, crn_seed_base,
        ))

    def _reattach(results: Counter, stats: Dict) -> Counter:
        # Counter pickling drops dynamic attributes, so the worker returns stats
        # separately; reattach so persist/CLI see enhanced + hygiene stats.
        results.stats = stats or {}  # type: ignore[attr-defined]
        return results

    if max_workers <= 1 or len(baselines) <= 1:
        # Sequential fallback
        all_results: Dict[str, Counter] = {}
        for item in work_items:
            baseline, results, stats = _run_single_baseline(item)
            logger.info("Completed %s vs %s (%d games)", agent_type, baseline, num_games)
            all_results[baseline] = _reattach(results, stats)
        return all_results

    logger.info(
        "Parallel eval: %d baselines across %d workers", len(baselines), max_workers,
    )
    all_results = {}
    ctx = mp.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        future_to_baseline = {
            pool.submit(_run_single_baseline, item): item[2]
            for item in work_items
        }
        for future in as_completed(future_to_baseline):
            baseline = future_to_baseline[future]
            try:
                _, results, stats = future.result()
                all_results[baseline] = _reattach(results, stats)
                logger.info("Completed %s vs %s (%d games)", agent_type, baseline, num_games)
            except Exception:
                logger.exception("Failed evaluating vs %s", baseline)
                all_results[baseline] = Counter()

    return all_results


def run_head_to_head(
    checkpoint_a: str,
    checkpoint_b: str,
    num_games: int,
    config,
    device: str = "cpu",
    agent_type: str = "deep_cfr",
) -> Dict:
    """
    Play two checkpoints of the same agent type against each other.

    Alternates which checkpoint goes first every game to reduce first-mover bias.
    Supports deep_cfr, rebel, sd_cfr, escher, and other NeuralAgentWrapper types.

    Returns:
        Dict with checkpoint_a_wins, checkpoint_b_wins, ties, avg_game_turns,
        std_game_turns.
    """
    logger.info(
        "Head-to-head (%s): %s vs %s (%d games)", agent_type, checkpoint_a, checkpoint_b, num_games
    )

    _agent_class = AGENT_REGISTRY.get(agent_type.lower(), DeepCFRAgentWrapper)

    checkpoint_a_wins = 0
    checkpoint_b_wins = 0
    ties_count = 0
    errors_count = 0
    turns_list: List[int] = []

    for game_num in range(1, num_games + 1):
        # Alternate who goes first
        a_is_p0 = (game_num % 2 == 1)
        if a_is_p0:
            agent0 = _agent_class(0, config, checkpoint_a, device=device)
            agent1 = _agent_class(1, config, checkpoint_b, device=device)
        else:
            agent0 = _agent_class(0, config, checkpoint_b, device=device)
            agent1 = _agent_class(1, config, checkpoint_a, device=device)

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
    use_argmax_a: bool = False,
    use_argmax_b: bool = False,
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
                agent0 = get_agent(agent_a_type, 0, config, checkpoint_path=checkpoint_a, device=device, use_argmax=use_argmax_a)
                agent1 = get_agent(agent_b_type, 1, config, checkpoint_path=checkpoint_b, device=device, use_argmax=use_argmax_b)
            else:
                agent0 = get_agent(agent_b_type, 0, config, checkpoint_path=checkpoint_b, device=device, use_argmax=use_argmax_b)
                agent1 = get_agent(agent_a_type, 1, config, checkpoint_path=checkpoint_a, device=device, use_argmax=use_argmax_a)

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


def persist_eval_results(
    run_dir: str,
    iteration: int,
    results_map: Dict[str, Counter],
    run_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    adv_loss: Optional[float] = None,
    strat_loss: Optional[float] = None,
    selection_mode: Optional[str] = None,
    crn_seed: Optional[int] = None,
    seat_scheme: Optional[str] = None,
) -> None:
    """Dual-write eval results to metrics.jsonl and SQLite.

    Args:
        run_dir: Path to run directory (e.g., runs/v2.2-sog-v3/).
        iteration: Checkpoint iteration/epoch number.
        results_map: Dict mapping baseline name -> Counter with P0 Wins, P1 Wins, Ties, stats.
        run_name: Run name for JSONL rows. Auto-derived from run_dir basename if None.
        checkpoint_path: Path to checkpoint file (for SQLite registration).
        adv_loss: Advantage/value loss from training logs (optional).
        strat_loss: Strategy/policy loss from training logs (optional).
        selection_mode: "argmax" / "stochastic" override. Per-baseline values from
            results.stats take precedence when present (they reflect the actual run).
        crn_seed: Common-random-numbers seed root override (fallback for stats).
        seat_scheme: "alternated" / "fixed" override (fallback for stats).
    """
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    run_dir_path = _Path(run_dir).resolve()
    if run_name is None:
        run_name = run_dir_path.name

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build rows
    all_rows = []
    for baseline, results in results_map.items():
        p0_wins = results.get("P0 Wins", 0)
        p1_wins = results.get("P1 Wins", 0)
        ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        total = p0_wins + p1_wins + ties
        win_rate = p0_wins / total if total > 0 else 0.0

        # Wilson CI
        ci_low, ci_high = 0.0, 0.0
        if total > 0:
            z = 1.96
            p = win_rate
            denom = 1 + z**2 / total
            center = (p + z**2 / (2 * total)) / denom
            margin = (z / denom) * math.sqrt(
                p * (1 - p) / total + z**2 / (4 * total**2)
            )
            ci_low = max(0.0, center - margin)
            ci_high = min(1.0, center + margin)

        stats = getattr(results, "stats", {})
        avg_game_turns = stats.get("avg_game_turns")
        t1_cambia_rate = stats.get("t1_cambia_rate")
        avg_score_margin = stats.get("avg_score_margin")

        # Eval-hygiene provenance. Per-baseline values produced by run_evaluation
        # (in results.stats) take precedence; the function-level overrides serve as
        # a fallback for callers that pass hand-built Counters without stats.
        row_seat_scheme = stats.get("seat_scheme", seat_scheme)
        row_selection_mode = stats.get("selection_mode", selection_mode)
        row_crn_seed = stats.get("crn_seed", crn_seed)
        # seat_balanced is only true when alternation actually ran and the agent
        # under test played both seats; default 0 keeps legacy/fixed runs honest.
        row_seat_balanced = int(bool(stats.get("seat_balanced", False)))

        row = {
            "run": run_name,
            "iter": iteration,
            "baseline": baseline,
            "win_rate": round(win_rate, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "games_played": total,
            "p0_wins": p0_wins,
            "p1_wins": p1_wins,
            "ties": ties,
            "adv_loss": None if adv_loss is None or adv_loss != adv_loss else round(adv_loss, 6),
            "strat_loss": None if strat_loss is None or strat_loss != strat_loss else round(strat_loss, 6),
            "timestamp": timestamp,
            "avg_game_turns": round(avg_game_turns, 2) if avg_game_turns is not None else None,
            "t1_cambia_rate": round(t1_cambia_rate, 4) if t1_cambia_rate is not None else None,
            "avg_score_margin": round(avg_score_margin, 2) if avg_score_margin is not None else None,
            "seat_scheme": row_seat_scheme,
            "selection_mode": row_selection_mode,
            "crn_seed": None if row_crn_seed is None else str(row_crn_seed),
            "seat_balanced": row_seat_balanced,
        }
        all_rows.append(row)

    # Append to metrics.jsonl
    if all_rows:
        run_dir_path.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir_path / "metrics.jsonl"
        with open(metrics_path, "a", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

    # Create evaluations directory
    eval_dir = run_dir_path / "evaluations" / f"iter_{iteration}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # SQLite dual-write (non-fatal)
    if all_rows:
        try:
            import src.run_db as run_db

            db = run_db.get_db()
            # Upsert run
            config_path = run_dir_path / "config.yaml"
            yaml_text = None
            config_dict = {}
            if config_path.exists():
                try:
                    yaml_text = config_path.read_text(encoding="utf-8")
                    import yaml

                    config_dict = yaml.safe_load(yaml_text) or {}
                except Exception:
                    pass

            ckpt_keys = None
            if checkpoint_path:
                try:
                    import torch

                    ckpt_data = torch.load(
                        str(checkpoint_path), map_location="cpu", weights_only=True
                    )
                    ckpt_keys = set(ckpt_data.keys())
                    del ckpt_data
                except Exception:
                    pass

            algorithm = run_db.infer_algorithm(
                config_dict, checkpoint_keys=ckpt_keys
            )
            run_id = run_db.upsert_run(
                db,
                name=run_name,
                algorithm=algorithm,
                config_yaml=yaml_text,
                config_dict=config_dict,
                status="running",
            )
            ckpt_id = None
            if checkpoint_path:
                ckpt_id = run_db.register_checkpoint(
                    db, run_id, iteration, str(checkpoint_path)
                )
            for row in all_rows:
                run_db.insert_eval_result(db, run_id, ckpt_id, row)
            run_db.write_eval_summary_jsonl(db, run_id, str(run_dir_path))
            # Refresh best_metric_* from the freshly written eval rows so the
            # run's recorded best never lags behind its eval history.
            try:
                run_db.recompute_best_metric(db, run_id)
            except Exception:
                pass
            db.close()
        except Exception:
            logger.warning(
                "SQLite dual-write failed for %s iter %d", run_name, iteration
            )


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
