"""Gymnasium environment wrapping the Cambia Python engine for PPO training.

Two opponent regimes are supported, selected by ``opponent_type``:

- A baseline string (e.g. ``imperfect_greedy``): the PPO seat trains as a
  best-response to a fixed opponent. This is the diagnostic regime and is NOT a
  trustworthy equilibrium anchor: the policy learns to exploit one weak fixed
  opponent rather than to play well in general.
- ``"self_play"``: fair self-play. The opponent seat is driven by a
  frozen-periodic snapshot of the learning policy, refreshed from disk every K
  timesteps by a training-side callback (see
  ``ppo_train.SelfPlaySnapshotCallback``). Both seats are the same policy class;
  the agent seat is randomized per episode so the learning policy plays both P0
  and P1. This is the E2 anchor regime: PPO improves only by beating copies of
  itself, so its mean_imp re-derives the metric's reachable headroom rather than
  scoring a best-response to one weak opponent.
"""

import copy
import logging
import os
import threading

import gymnasium
import numpy as np

from src.game.engine import CambiaGameState
from src.agent_state import AgentState, AgentObservation
from src.encoding import (
    encode_infoset_eppbs_interleaved,
    encode_infoset_eppbs_interleaved_v2,
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
    EP_PBS_V2_INPUT_DIM,
)
from src.evaluate_agents import get_agent, NeuralAgentWrapper

logger = logging.getLogger(__name__)


def _peek_encoding_version(config_path: str) -> int:
    """Read encoding_version from YAML without full Pydantic validation.

    Used by CambiaEnv.__init__ to set observation_space shape before the
    config is fully loaded. Keeps _config lazy (None until first reset).
    Returns 1 on any read or parse error.
    """
    try:
        import yaml

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return int(raw.get("deep_cfr", {}).get("encoding_version", 1))
    except Exception:
        return 1


# Sentinel opponent_type that selects the fair self-play regime instead of a
# fixed baseline agent from evaluate_agents.get_agent.
SELF_PLAY_OPPONENT = "self_play"


class SelfPlayPolicyOpponent:
    """Frozen-periodic-snapshot self-play opponent for MaskablePPO.

    The opponent seat is driven by a snapshot of the learning policy persisted
    to ``snapshot_path`` (an SB3 ``.zip``). A training-side callback overwrites
    that file every K timesteps; this opponent watches the file's mtime and
    reloads on change, so the opponent strength tracks the learner with a lag of
    at most one refresh interval. Frozen-periodic (not a live mirror) is the
    standard self-play recipe: it keeps the opponent stationary within a rollout,
    which PPO's on-policy advantage estimates require, while still climbing as
    the learner improves.

    Until the first snapshot exists (the opening refresh interval of training),
    the opponent plays uniform-random legal actions so episodes still terminate
    and produce reward signal. This warm-up window is small relative to a 30M
    step run and does not bias the converged anchor: once snapshots exist, every
    opponent move comes from a copy of the learning policy.

    The opponent is constructed inside each SubprocVecEnv worker process. Model
    loading is lazy and guarded by a per-instance lock so a mid-rollout reload
    cannot race the predict call.
    """

    def __init__(
        self,
        snapshot_path: str,
        device: str = "cpu",
        deterministic: bool = False,
        rng: np.random.Generator | None = None,
    ):
        self._snapshot_path = snapshot_path
        self._device = device
        self._deterministic = deterministic
        self._rng = rng if rng is not None else np.random.default_rng()
        self._model = None
        self._loaded_mtime: float | None = None
        self._lock = threading.Lock()

    def _snapshot_file(self) -> str:
        # SB3 model.save(path) writes path + ".zip" when path lacks the suffix.
        if self._snapshot_path.endswith(".zip"):
            return self._snapshot_path
        return self._snapshot_path + ".zip"

    def _maybe_reload(self):
        """Load or hot-reload the snapshot model when the file changes."""
        path = self._snapshot_file()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            # No snapshot yet -> uniform-random warm-up.
            self._model = None
            return
        if self._model is not None and self._loaded_mtime == mtime:
            return
        try:
            from sb3_contrib import MaskablePPO

            self._model = MaskablePPO.load(path, device=self._device)
            self._loaded_mtime = mtime
        except Exception as e:  # JUSTIFIED: a partial/locked snapshot file mid-write
            logger.debug("Self-play snapshot reload failed (%s); keeping prior model.", e)

    def predict_index(self, obs: np.ndarray, action_mask: np.ndarray) -> int | None:
        """Return the chosen action index, or None to signal random fallback."""
        with self._lock:
            self._maybe_reload()
            model = self._model
        if model is None:
            return None
        idx, _ = model.predict(
            obs, action_masks=action_mask, deterministic=self._deterministic
        )
        return int(idx)


class CambiaEnv(gymnasium.Env):
    """Single-agent Gymnasium environment for Cambia.

    The PPO agent controls one seat; the opponent seat is handled either by a
    fixed baseline agent (best-response diagnostic) or by a frozen-periodic
    snapshot of the learning policy (fair self-play, ``opponent_type ==
    "self_play"``). The episode ends when the game reaches a terminal state.

    Under self-play the agent seat is randomized each episode so the learning
    policy plays both P0 and P1; otherwise the agent stays in ``agent_seat``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_type: str = "imperfect_greedy",
        seed: int | None = None,
        agent_seat: int = 0,
        config_path: str = "config.yaml",
        selfplay_snapshot_path: str | None = None,
        selfplay_deterministic: bool = False,
    ):
        super().__init__()
        # Peek at encoding_version from raw YAML to set obs dim.
        # _config stays None until _load_config() is called on first reset().
        self._encoding_version: int = _peek_encoding_version(config_path)
        obs_dim = EP_PBS_V2_INPUT_DIM if self._encoding_version == 2 else EP_PBS_INPUT_DIM
        self.observation_space = gymnasium.spaces.Box(
            -5.0, 5.0, (obs_dim,), np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        self._opponent_type = opponent_type
        self._self_play = opponent_type == SELF_PLAY_OPPONENT
        self._selfplay_snapshot_path = selfplay_snapshot_path
        self._selfplay_deterministic = selfplay_deterministic
        if self._self_play and not selfplay_snapshot_path:
            raise ValueError(
                "opponent_type='self_play' requires selfplay_snapshot_path "
                "(the SB3 .zip the snapshot callback writes)."
            )
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
            full_version = self._config.deep_cfr.encoding_version
            if full_version != self._encoding_version:
                raise RuntimeError(
                    f"encoding_version mismatch between YAML peek ({self._encoding_version}) "
                    f"and fully-resolved config ({full_version}). The peek in "
                    f"_peek_encoding_version only reads the child YAML and does not follow "
                    f"_base: inheritance or rule-profile defaults. observation_space was "
                    f"sized from the peek value and cannot change after __init__, so a "
                    f"mismatch would make the gym obs incompatible with the actual encoder "
                    f"output. Fix by setting deep_cfr.encoding_version explicitly in "
                    f"{self._config_path} (not inherited)."
                )
            self._encoding_version = full_version

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._load_config()

        # Fair self-play: randomize which seat the learning policy occupies each
        # episode so it experiences both P0 (acts first) and P1 over the run.
        # The fixed-opponent regime keeps the agent in its configured seat.
        if self._self_play:
            self._agent_seat = int(self._rng.integers(NUM_PLAYERS))
            self._opponent_seat = 1 - self._agent_seat

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

        if self._self_play:
            self._opponent = SelfPlayPolicyOpponent(
                snapshot_path=self._selfplay_snapshot_path,
                device="cpu",
                deterministic=self._selfplay_deterministic,
                rng=self._rng,
            )
        else:
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
            if self._self_play:
                action = self._select_selfplay_action(legal)
            else:
                action = self._opponent.choose_action(gs, legal)
            gs.apply_action(action)
            obs = self._make_observation(action, acting_player)
            self._update_states(obs)
            if not self._self_play and hasattr(self._opponent, "update_state"):
                self._opponent.update_state(obs)

    def _select_selfplay_action(self, legal):
        """Pick the opponent move from the frozen snapshot policy.

        Encodes the opponent seat's infoset, builds its legal-action mask, and
        queries the snapshot. Falls back to a uniform-random legal action when
        no snapshot exists yet (warm-up) or the predicted index is not legal
        (mask/predict edge case), mirroring the agent-seat fallback in step().
        """
        legal_list = list(legal)
        opp_obs = self._get_obs(seat=self._opponent_seat)
        mask = encode_action_mask(legal_list)
        idx = self._opponent.predict_index(opp_obs, mask)
        if idx is not None:
            for a in legal_list:
                if action_to_index(a) == idx:
                    return a
        return legal_list[int(self._rng.integers(len(legal_list)))]

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

    def _get_obs(self, seat: int | None = None) -> np.ndarray:
        """Encode a seat's infoset as a float32 vector.

        ``seat`` defaults to the PPO agent seat (the gym observation). The
        self-play opponent passes its own seat so it acts on its private belief
        state, never the agent's, keeping both seats on the same encoder.
        """
        if seat is None:
            seat = self._agent_seat
        gs = self._game_state
        ctx = self._get_decision_context(gs)
        st = self._agent_states[seat]

        if self._encoding_version == 2:
            return encode_infoset_eppbs_interleaved_v2(st, ctx).astype(np.float32)

        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == seat:
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
    selfplay_snapshot_path: str | None = None,
    selfplay_deterministic: bool = False,
):
    """Factory for SubprocVecEnv compatibility."""

    def _init():
        # Pin torch intra-op threads per SubprocVecEnv worker: env stepping and
        # the self-play opponent's MaskablePPO inference both use torch, and an
        # unpinned per-worker pool thrashes cores at high n_envs.
        import torch

        torch.set_num_threads(1)
        return CambiaEnv(
            opponent_type=opponent_type,
            seed=seed,
            agent_seat=agent_seat,
            config_path=config_path,
            selfplay_snapshot_path=selfplay_snapshot_path,
            selfplay_deterministic=selfplay_deterministic,
        )

    return _init
