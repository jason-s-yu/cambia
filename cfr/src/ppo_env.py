"""Gymnasium environment wrapping the Go Cambia engine for PPO training.

The environment steps on ``GoEngine`` and reads its observation off
``GoAgentState`` (cfr/src/ffi/bridge.py), so the game rules, the belief update
and the tensor encoding all come from libcambia. Nothing here imports the
Python reference engine (cambia-1376, retirement sprint cambia-1424).

Seat count
----------
``num_players`` selects the seat count and, with it, the engine's action and
encoding space, because the Go engine has two of them:

- 2 seats use the 2-player space: 146 actions, EP-PBS interleaved observations
  (224-dim v1 / 257-dim v2). This is the space every existing PPO checkpoint
  was trained in, so a 2-seat run stays weight-compatible with them.
- 3+ seats use the N-player space: ``GoEngine.N_PLAYER_NUM_ACTIONS`` actions
  and ``GoEngine.N_PLAYER_INPUT_DIM`` observations from ``encode_nplayer``.

The two spaces have different widths, so a checkpoint does not transfer across
that boundary. The env reports whichever space its seat count selects through
``observation_space`` / ``action_space``, which is what MaskablePPO reads.

Opponent regimes
----------------
- ``"self_play"``: fair self-play. Every seat other than the learner is driven
  by a frozen-periodic snapshot of the learning policy, refreshed from disk
  every K timesteps by a training-side callback (see
  ``ppo_train.SelfPlaySnapshotCallback``). The learner's seat is randomized per
  episode so it plays every seat over a run. This is the E2 anchor regime: PPO
  improves only by beating copies of itself.
- ``"random_legal"``: opponent seats play a uniform-random legal action. Cheap
  control, and the regime the throughput benchmark uses.
- a ``src.agents.baseline_agents`` name (``"imperfect_greedy"``,
  ``"memory_heuristic"``, ``"aggressive_snap"``, ``"random"``, ``"greedy"``,
  ``"random_no_cambia"``, ``"random_late_cambia"``, ``"human_player"``): the
  fixed best-response diagnostic. Every opponent seat is driven by a fresh
  instance of that baseline (cambia-1426's GameView port), rebuilt every
  episode so a baseline's own-game memory sentinel never survives a reset.
  Only names resolving to that module qualify (see ``is_baseline_opponent``):
  checkpoint-backed wrappers and the tabular ``CFRAgentWrapper`` are not
  accepted here. ``cli.py``'s ``train ppo`` defaults to ``"imperfect_greedy"``
  (cambia-1482), matching PPO-200k's original training opponent.

Observation contract
--------------------
The pre-port env fed every seat's belief state a public-only observation, with
``drawn_card`` and ``peeked_cards`` stripped for the actor as well (the
contract ratified in note cambia-1074). ``GoAgentState`` has no such mode:
``cambia_agent_update`` reads the game state directly and records what the
rules reveal to the acting seat, which is what every other Go-backed trainer
already gets. Moving the belief onto GoAgentState therefore changes what the
policy observes; the divergence is measured, not incidental. See the
concerns recorded with cambia-1376.
"""

import logging
import os
import threading

import gymnasium
import numpy as np

from src.agents import action_codec
from src.agents.game_view import tracked_opponent_seat
from src.constants import EP_PBS_V2_INPUT_DIM, ActionSnapOpponentMove
from src.encoding import EP_PBS_INPUT_DIM, NUM_ACTIONS, action_to_index
from src.ffi.bridge import GoAgentState, GoEngine

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


# Sentinel opponent_type that selects the fair self-play regime.
SELF_PLAY_OPPONENT = "self_play"

# Sentinel opponent_type for uniform-random legal play on the opponent seats.
RANDOM_LEGAL_OPPONENT = "random_legal"

SUPPORTED_OPPONENTS = (SELF_PLAY_OPPONENT, RANDOM_LEGAL_OPPONENT)

#: Module a registered agent class must live in to be accepted as a fixed
#: opponent here. These are cambia-1426's GameView port: checkpoint-free,
#: undo-free, and reading the game only through GoEngine/GameView, so
#: accepting one cannot revive the Python engine behind this env.
_BASELINE_MODULE = "src.agents.baseline_agents"


def is_baseline_opponent(name: str) -> bool:
    """True if ``name`` names a checkpoint-free GameView baseline agent.

    Excludes checkpoint-backed wrappers (deep_cfr, ppo, rebel, ...) and
    ``CFRAgentWrapper`` (tabular, Python-only -- see its docstring in
    evaluate_agents.py) even though they share the same agent registry.
    """
    from src.evaluate_agents import AGENT_REGISTRY

    agent_class = AGENT_REGISTRY.get(name.lower())
    return agent_class is not None and agent_class.__module__ == _BASELINE_MODULE


class SelfPlayPolicyOpponent:
    """Frozen-periodic-snapshot self-play opponent for MaskablePPO.

    The opponent seats are driven by a snapshot of the learning policy
    persisted to ``snapshot_path`` (an SB3 ``.zip``). A training-side callback
    overwrites that file every K timesteps; this opponent watches the file's
    mtime and reloads on change, so opponent strength tracks the learner with a
    lag of at most one refresh interval. Frozen-periodic (not a live mirror) is
    the standard self-play recipe: it keeps the opponent stationary within a
    rollout, which PPO's on-policy advantage estimates require, while still
    climbing as the learner improves.

    Until the first snapshot exists (the opening refresh interval of training),
    the opponent plays uniform-random legal actions so episodes still terminate
    and produce reward signal. This warm-up window is small relative to a long
    run and does not bias the converged anchor: once snapshots exist, every
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
    """Single-agent Gymnasium environment for Cambia on the Go engine.

    The PPO agent controls one seat; every other seat is played by the
    configured opponent regime. The episode ends when the game reaches a
    terminal state. Under self-play the agent seat is randomized each episode
    so the learning policy plays every seat.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_type: str = SELF_PLAY_OPPONENT,
        seed: int | None = None,
        agent_seat: int = 0,
        config_path: str = "config.yaml",
        selfplay_snapshot_path: str | None = None,
        selfplay_deterministic: bool = False,
        num_players: int = 2,
    ):
        super().__init__()
        num_players = int(num_players)
        if num_players < 2:
            raise ValueError(f"num_players must be at least 2, got {num_players}")
        self._num_players = num_players
        # 3+ seats need the engine's N-player action and encoding space; 2 seats
        # stay in the 2-player space every existing PPO checkpoint was fit in.
        self._nplayer_space = num_players > 2

        # Peek at encoding_version from raw YAML to set obs dim.
        # _config stays None until _load_config() is called on first reset().
        self._encoding_version: int = _peek_encoding_version(config_path)
        if self._nplayer_space:
            obs_dim = GoEngine.N_PLAYER_INPUT_DIM
            n_actions = GoEngine.N_PLAYER_NUM_ACTIONS
        else:
            obs_dim = (
                EP_PBS_V2_INPUT_DIM if self._encoding_version == 2 else EP_PBS_INPUT_DIM
            )
            n_actions = NUM_ACTIONS
        self.observation_space = gymnasium.spaces.Box(-5.0, 5.0, (obs_dim,), np.float32)
        self.action_space = gymnasium.spaces.Discrete(n_actions)

        self._opponent_type = opponent_type
        self._self_play = opponent_type == SELF_PLAY_OPPONENT
        self._fixed_baseline = (
            opponent_type not in SUPPORTED_OPPONENTS
            and is_baseline_opponent(opponent_type)
        )
        if opponent_type not in SUPPORTED_OPPONENTS and not self._fixed_baseline:
            raise NotImplementedError(
                f"opponent_type={opponent_type!r} is not available on the Go-backed "
                f"env. Supported: {', '.join(SUPPORTED_OPPONENTS)}, or a "
                "src.agents.baseline_agents name (e.g. 'imperfect_greedy', "
                "'memory_heuristic', 'aggressive_snap', 'random', 'greedy', "
                "'random_no_cambia', 'random_late_cambia', 'human_player'). "
                "Checkpoint-backed wrappers and the tabular CFRAgentWrapper are "
                "not supported here."
            )
        self._baseline_agents: dict | None = None
        self._selfplay_snapshot_path = selfplay_snapshot_path
        self._selfplay_deterministic = selfplay_deterministic
        if self._self_play and not selfplay_snapshot_path:
            raise ValueError(
                "opponent_type='self_play' requires selfplay_snapshot_path "
                "(the SB3 .zip the snapshot callback writes)."
            )

        self._agent_seat = int(agent_seat) % num_players
        self._config_path = config_path
        self._config = None
        self._engine: GoEngine | None = None
        self._agents: list[GoAgentState] | None = None
        self._opponent = None
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Lazy config loader
    # ------------------------------------------------------------------

    def _load_config(self):
        if self._config is not None:
            return
        from src.config import load_config

        self._config = load_config(self._config_path)
        full_version = self._config.deep_cfr.encoding_version
        if not self._nplayer_space and full_version != self._encoding_version:
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

    # A deal whose opponents finish the game before the learner ever acts would
    # be an episode with no decision and no reward. It should not be reachable
    # under the shipped rule sets, so a handful of retries is generous; the
    # alternative is silently handing PPO a zero-reward terminal transition.
    _MAX_DEGENERATE_DEALS = 16

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._load_config()

        for _ in range(self._MAX_DEGENERATE_DEALS):
            self._deal()
            self._advance_opponent()
            if not self._engine.is_terminal():
                return self._get_obs(), {}
            logger.debug("discarding a deal that ended before the agent's first turn")

        self._release()
        raise RuntimeError(
            f"{self._MAX_DEGENERATE_DEALS} consecutive deals ended before the "
            f"agent's first turn at {self._num_players} seats. Check "
            f"cambia_rules in {self._config_path}: max_game_turns or "
            "cambia_allowed_round may make games terminate immediately."
        )

    def _deal(self):
        """Free the previous episode and set up a fresh game and belief states.

        Handles come from fixed-size Go pools and a training run resets
        thousands of times per worker, so the release is explicit here rather
        than left to __del__.
        """
        self._release()

        # Fair self-play: randomize which seat the learning policy occupies each
        # episode so it experiences every seat over the run.
        if self._self_play:
            self._agent_seat = int(self._rng.integers(self._num_players))

        game_seed = int(self._rng.integers(1 << 62))
        self._engine = GoEngine(
            seed=game_seed,
            house_rules=self._config.cambia_rules,
            num_players=self._num_players,
        )
        seats = self._engine.num_players()
        if seats != self._num_players:
            self._release()
            raise RuntimeError(
                f"engine dealt {seats} seats, env was configured for "
                f"{self._num_players}. cambia_rules in {self._config_path} may pin "
                f"num_players; the env's num_players must agree with it."
            )

        mem = self._config.agent_params.memory_level
        decay = self._config.agent_params.time_decay_turns
        # Built one at a time and assigned as we go: if the agent pool runs dry
        # partway through, _release has to be able to reach the handles already
        # taken, which a list comprehension that never returns would not allow.
        self._agents = []
        try:
            for pid in range(self._num_players):
                if self._nplayer_space:
                    agent = GoAgentState.new_nplayer(
                        self._engine, pid, self._num_players, mem, decay
                    )
                else:
                    agent = GoAgentState(self._engine, pid, mem, decay)
                self._agents.append(agent)
        except Exception:
            self._release()
            raise

        if self._self_play:
            self._opponent = SelfPlayPolicyOpponent(
                snapshot_path=self._selfplay_snapshot_path,
                device="cpu",
                deterministic=self._selfplay_deterministic,
                rng=self._rng,
            )
        else:
            self._opponent = None

        if self._fixed_baseline:
            # Fresh instances every episode: a baseline detects "new game" by
            # id() of what it is handed (game_view.as_game_view's cache), and
            # Go handles come from a pool an address can be reused in, so a
            # rebuilt-not-reset instance sidesteps that sentinel entirely
            # rather than relying on invalidating it correctly every reset.
            from src.evaluate_agents import get_agent

            self._baseline_agents = {
                seat: get_agent(self._opponent_type, player_id=seat, config=self._config)
                for seat in range(self._num_players)
                if seat != self._agent_seat
            }
        else:
            self._baseline_agents = None

    def step(self, action: int):
        engine = self._engine
        if engine is None:
            raise RuntimeError("step() before reset()")
        if engine.is_terminal():
            return self._get_obs(), 0.0, True, False, {}

        mask = self._legal_mask()
        idx = int(action)
        if idx < 0 or idx >= mask.shape[0] or not mask[idx]:
            idx = self._random_legal(mask)
            logger.debug("PPO chose illegal action %s; falling back to random.", action)

        self._apply(idx)
        self._update_agents()
        self._advance_opponent()

        terminated = engine.is_terminal()
        reward = float(self._utility()[self._agent_seat]) if terminated else 0.0
        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        """SB3 MaskablePPO protocol: return a bool mask over all actions."""
        return self._legal_mask().astype(bool)

    def render(self):
        pass

    def close(self):
        self._release()

    def __del__(self):
        try:
            self._release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _release(self):
        """Free the Go game and agent handles this episode holds."""
        for agent in self._agents or ():
            agent.close()
        self._agents = None
        self._baseline_agents = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def _legal_mask(self) -> np.ndarray:
        """Return the acting seat's legal-action mask as uint8, engine-side."""
        if self._nplayer_space:
            return self._engine.nplayer_legal_actions_mask()
        return self._engine.legal_actions_mask()

    def _apply(self, action_idx: int) -> None:
        if self._nplayer_space:
            self._engine.apply_nplayer_action(action_idx)
        else:
            self._engine.apply_action(action_idx)

    def _utility(self) -> np.ndarray:
        if self._nplayer_space:
            return self._engine.get_nplayer_utility()
        return self._engine.get_utility()

    def _update_agents(self) -> None:
        """Refresh every seat's belief state from the post-action game state."""
        if self._nplayer_space:
            for agent in self._agents:
                agent.update_nplayer(self._engine)
        else:
            # One FFI call for both seats on the 2-player path.
            self._engine.update_both(self._agents[0], self._agents[1])

    def _random_legal(self, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError(
                f"no legal action at a non-terminal state "
                f"(seat {self._engine.acting_player()}, ctx {self._engine.decision_ctx()})"
            )
        return int(legal[int(self._rng.integers(legal.size))])

    def _advance_opponent(self):
        """Run opponent turns until it is the PPO agent's turn or the game ends."""
        engine = self._engine
        while not engine.is_terminal() and engine.acting_player() != self._agent_seat:
            seat = engine.acting_player()
            if self._fixed_baseline:
                idx = self._select_baseline_action(seat)
            else:
                mask = self._legal_mask()
                if self._self_play:
                    idx = self._select_selfplay_action(seat, mask)
                else:
                    idx = self._random_legal(mask)
            self._apply(idx)
            self._update_agents()

    def _select_baseline_action(self, seat: int) -> int:
        """Ask the fixed baseline occupying ``seat`` for its move.

        The baseline agents' GameView contract speaks ``GameAction`` objects,
        not engine indices, so the legal set is decoded through action_codec
        and the agent's choice is re-encoded back to the index the engine's
        apply path expects. This mirrors
        ``evaluate_agents._GoEvalGame.legal_actions()`` / ``.apply()`` for the
        same two action spaces; duplicated here (via the same action_codec /
        game_view primitives, not re-derived arithmetic) rather than imported
        because that class owns its own engine and agent list end to end and
        this env owns its own separately.
        """
        engine = self._engine
        agent = self._baseline_agents[seat]

        if not self._nplayer_space:
            mask = engine.legal_actions_mask()
            legal_actions = action_codec.actions_from_mask(mask)
            chosen = agent.choose_action(engine, legal_actions)
            return int(action_to_index(chosen))

        pending = engine.get_pending()
        if pending.target_seat is not None and pending.target_seat != seat:
            target = int(pending.target_seat)
        else:
            target = tracked_opponent_seat(seat, self._num_players)
        rel = action_codec.relative_opponent_index(seat, target, self._num_players)

        preferred, preferred_index = [], {}
        fallback, fallback_index = [], {}
        mask = engine.nplayer_legal_actions_mask()
        for idx in np.flatnonzero(np.asarray(mask)):
            entry = action_codec.nplayer_index_to_action(int(idx))
            action = entry.action
            if isinstance(action, ActionSnapOpponentMove):
                # The N-player space keeps only the own-card index; the
                # target slot lives on the pending record.
                action = ActionSnapOpponentMove(
                    own_card_to_move_hand_index=action.own_card_to_move_hand_index,
                    target_empty_slot_index=int(pending.target_slot or 0),
                )
            if entry.opp_idx is None or entry.opp_idx == rel:
                if action not in preferred_index:
                    preferred.append(action)
                    preferred_index[action] = int(idx)
            if action not in fallback_index:
                fallback.append(action)
                fallback_index[action] = int(idx)

        legal_actions = preferred if preferred else fallback
        index_map = preferred_index if preferred else fallback_index
        chosen = agent.choose_action(engine, legal_actions)
        idx = index_map.get(chosen)
        if idx is None:
            idx = action_codec.nplayer_index_for(chosen, rel)
        return int(idx)

    def _select_selfplay_action(self, seat: int, mask: np.ndarray) -> int:
        """Pick an opponent move from the frozen snapshot policy.

        Encodes the opponent seat's own belief state, so the snapshot acts on
        that seat's information and never the learner's. Falls back to a
        uniform-random legal action when no snapshot exists yet (warm-up) or
        the predicted index is not legal, mirroring the agent-seat fallback in
        step().
        """
        obs = self._get_obs(seat=seat)
        idx = self._opponent.predict_index(obs, mask.astype(bool))
        if idx is not None and 0 <= idx < mask.shape[0] and mask[idx]:
            return int(idx)
        return self._random_legal(mask)

    def _get_obs(self, seat: int | None = None) -> np.ndarray:
        """Encode a seat's infoset as a float32 vector.

        ``seat`` defaults to the PPO agent seat (the gym observation). The
        self-play opponent passes its own seat so both seats run through the
        same encoder on their own belief state.
        """
        if seat is None:
            seat = self._agent_seat
        engine = self._engine
        ctx = engine.decision_ctx()
        drawn = engine.get_drawn_card_bucket()
        agent = self._agents[seat]

        if self._nplayer_space:
            return agent.encode_nplayer(ctx, drawn)
        if self._encoding_version == 2:
            return agent.encode_eppbs_interleaved_v2(ctx, drawn)
        return agent.encode_eppbs_interleaved(ctx, drawn)


def make_env(
    opponent_type: str = SELF_PLAY_OPPONENT,
    seed: int = 0,
    agent_seat: int = 0,
    config_path: str = "config.yaml",
    selfplay_snapshot_path: str | None = None,
    selfplay_deterministic: bool = False,
    num_players: int = 2,
):
    """Factory for SubprocVecEnv compatibility."""

    def _init():
        # Pin torch intra-op threads per SubprocVecEnv worker: the self-play
        # opponent's MaskablePPO inference uses torch, and an unpinned
        # per-worker pool thrashes cores at high n_envs.
        import torch

        torch.set_num_threads(1)
        return CambiaEnv(
            opponent_type=opponent_type,
            seed=seed,
            agent_seat=agent_seat,
            config_path=config_path,
            selfplay_snapshot_path=selfplay_snapshot_path,
            selfplay_deterministic=selfplay_deterministic,
            num_players=num_players,
        )

    return _init
