"""Gymnasium environment wrapping the Cambia Python engine for PPO training."""

import copy
import logging

import gymnasium
import numpy as np

from src.game.engine import CambiaGameState
from src.agent_state import AgentState, AgentObservation
from src.encoding import (
    encode_infoset_eppbs_interleaved,
    encode_action_mask,
    action_to_index,
    index_to_action,
    EP_PBS_INPUT_DIM,
    NUM_ACTIONS,
)
from src.constants import (
    DecisionContext,
    ActionDiscard,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionSnapOpponentMove,
    NUM_PLAYERS,
)
from src.evaluate_agents import get_agent, NeuralAgentWrapper

logger = logging.getLogger(__name__)


class CambiaEnv(gymnasium.Env):
    """Single-agent Gymnasium environment for Cambia.

    The PPO agent controls one seat; an internal fixed opponent handles all other
    turns. The episode ends when the game reaches a terminal state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_type: str = "imperfect_greedy",
        seed: int | None = None,
        agent_seat: int = 0,
        config_path: str = "config.yaml",
    ):
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(
            -5.0, 5.0, (EP_PBS_INPUT_DIM,), np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self._opponent_type = opponent_type
        self._agent_seat = agent_seat
        self._opponent_seat = 1 - agent_seat
        self._config_path = config_path
        self._config = None
        self._game_state: CambiaGameState | None = None
        self._agent_states: list[AgentState] | None = None
        self._opponent = None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Lazy config loader
    # ------------------------------------------------------------------

    def _load_config(self):
        if self._config is None:
            from src.config import load_config

            self._config = load_config(self._config_path)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._load_config()

        self._game_state = CambiaGameState(house_rules=self._config.cambia_rules)

        agent_states = []
        for pid in range(NUM_PLAYERS):
            st = AgentState(
                player_id=pid,
                opponent_id=1 - pid,
                memory_level=self._config.agent_params.memory_level,
                time_decay_turns=self._config.agent_params.time_decay_turns,
                initial_hand_size=len(self._game_state.players[pid].hand),
                config=self._config,
            )
            obs = AgentObservation(
                acting_player=-1,
                action=None,
                discard_top_card=self._game_state.get_discard_top(),
                player_hand_sizes=[
                    self._game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)
                ],
                stockpile_size=self._game_state.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=list(self._game_state.snap_results_log),
                did_cambia_get_called=self._game_state.cambia_caller_id is not None,
                who_called_cambia=self._game_state.cambia_caller_id,
                is_game_over=self._game_state.is_terminal(),
                current_turn=self._game_state.get_turn_number(),
            )
            st.initialize(
                obs,
                self._game_state.players[pid].hand,
                self._game_state.players[pid].initial_peek_indices,
            )
            agent_states.append(st)
        self._agent_states = agent_states

        self._opponent = get_agent(
            self._opponent_type,
            player_id=self._opponent_seat,
            config=self._config,
        )
        if hasattr(self._opponent, "initialize_state"):
            self._opponent.initialize_state(self._game_state)

        self._advance_opponent()
        return self._get_obs(), {}

    def step(self, action: int):
        gs = self._game_state
        if gs.is_terminal():
            return self._get_obs(), 0.0, True, False, {}

        legal = gs.get_legal_actions()

        # Resolve int action → GameAction; fall back to random if not legal.
        game_action = None
        for a in legal:
            if action_to_index(a) == action:
                game_action = a
                break
        if game_action is None:
            game_action = list(legal)[int(self._rng.integers(len(legal)))]
            logger.debug("PPO chose illegal action %d; falling back to random.", action)

        acting_player = gs.get_acting_player()
        gs.apply_action(game_action)
        obs = self._make_observation(game_action, acting_player)
        self._update_states(obs)

        self._advance_opponent()

        terminated = gs.is_terminal()
        reward = float(gs.get_utility(self._agent_seat)) if terminated else 0.0
        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        """SB3 MaskablePPO protocol: return bool mask over all actions."""
        return encode_action_mask(list(self._game_state.get_legal_actions()))

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_opponent(self):
        """Run opponent turns until it's the PPO agent's turn or game ends."""
        gs = self._game_state
        while not gs.is_terminal() and gs.get_acting_player() != self._agent_seat:
            acting_player = gs.get_acting_player()
            legal = gs.get_legal_actions()
            action = self._opponent.choose_action(gs, legal)
            gs.apply_action(action)
            obs = self._make_observation(action, acting_player)
            self._update_states(obs)
            if hasattr(self._opponent, "update_state"):
                self._opponent.update_state(obs)

    def _update_states(self, obs: AgentObservation):
        """Fan-out a public observation to all agent states (strip private fields)."""
        for st in self._agent_states:
            filtered = copy.copy(obs)
            filtered.drawn_card = None
            filtered.peeked_cards = None
            try:
                st.update(filtered)
            except Exception as e:
                logger.debug("AgentState update error: %s", e)

    def _get_obs(self) -> np.ndarray:
        """Encode the PPO player's infoset as a float32 vector."""
        gs = self._game_state
        ctx = self._get_decision_context(gs)
        st = self._agent_states[self._agent_seat]

        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == self._agent_seat:
            cambia_state = 0
        else:
            cambia_state = 1

        encoding = encode_infoset_eppbs_interleaved(
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
                st.game_phase.value if hasattr(st.game_phase, "value") else int(st.game_phase)
            ),
            decision_context=ctx.value if hasattr(ctx, "value") else int(ctx),
            cambia_state=cambia_state,
            own_hand_size=len(st.own_hand),
            opp_hand_size=st.opponent_card_count,
        )
        return encoding.astype(np.float32)

    def _get_decision_context(self, gs: CambiaGameState) -> DecisionContext:
        if gs.snap_phase_active:
            return DecisionContext.SNAP_DECISION
        if gs.pending_action:
            p = gs.pending_action
            if isinstance(p, ActionDiscard):
                return DecisionContext.POST_DRAW
            if isinstance(
                p,
                (
                    ActionAbilityPeekOwnSelect,
                    ActionAbilityPeekOtherSelect,
                    ActionAbilityBlindSwapSelect,
                    ActionAbilityKingLookSelect,
                    ActionAbilityKingSwapDecision,
                ),
            ):
                return DecisionContext.ABILITY_SELECT
            if isinstance(p, ActionSnapOpponentMove):
                return DecisionContext.SNAP_MOVE
        return DecisionContext.START_TURN

    def _make_observation(self, action, acting_player: int) -> AgentObservation:
        gs = self._game_state
        return AgentObservation(
            acting_player=acting_player,
            action=action,
            discard_top_card=gs.get_discard_top(),
            player_hand_sizes=[gs.get_player_card_count(i) for i in range(NUM_PLAYERS)],
            stockpile_size=gs.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=list(gs.snap_results_log),
            did_cambia_get_called=gs.cambia_caller_id is not None,
            who_called_cambia=gs.cambia_caller_id,
            is_game_over=gs.is_terminal(),
            current_turn=gs.get_turn_number(),
        )


def make_env(
    opponent_type: str = "imperfect_greedy",
    seed: int = 0,
    agent_seat: int = 0,
    config_path: str = "config.yaml",
):
    """Factory for SubprocVecEnv compatibility."""

    def _init():
        return CambiaEnv(
            opponent_type=opponent_type,
            seed=seed,
            agent_seat=agent_seat,
            config_path=config_path,
        )

    return _init
