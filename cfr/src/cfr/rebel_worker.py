"""
src/cfr/rebel_worker.py

ReBeL self-play episode runner for Phase 1c training.

At each decision point during self-play, constructs a depth-limited subgame,
evaluates leaf nodes with the PBS value network, runs CFR iterations, and
records (PBS_encoding, value_target, policy_target, action_mask) tuples.

Training data shapes:
    features:      (PBS_INPUT_DIM,) = (956,)  float32
    value_target:  (VALUE_DIM,)    = (936,)  float32  — tiled from solver's 2-dim root values
    policy_target: (NUM_ACTIONS,)  = (146,)  float32  — averaged strategy from subgame CFR
    action_mask:   (NUM_ACTIONS,)  = (146,)  bool
"""

# DEPRECATED: ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games
# with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.
import warnings

warnings.warn(
    "rebel_worker is DEPRECATED and will be removed. "
    "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
    "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import multiprocessing as mp
import queue
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import DeepCfrConfig
from ..encoding import NUM_ACTIONS
from ..networks import PBSValueNetwork, PBSPolicyNetwork
from ..pbs import (
    PBS,
    PBS_INPUT_DIM,
    NUM_HAND_TYPES,
    encode_pbs,
    encode_pbs_batch,
    uniform_range,
    make_public_features,
    PHASE_DRAW,
    PHASE_DISCARD,
    PHASE_ABILITY,
    PHASE_SNAP,
    PHASE_TERMINAL,
)

logger = logging.getLogger(__name__)

# Value target dimension: 2 * NUM_HAND_TYPES = 936
VALUE_DIM: int = 2 * NUM_HAND_TYPES

# Decision context integer → PBS phase index
# ctx: 0=StartTurn, 1=PostDraw, 2=AbilitySelect, 3=SnapDecision, 4=SnapMove, 5=Terminal
_CTX_TO_PHASE: Dict[int, int] = {
    0: PHASE_DRAW,
    1: PHASE_DISCARD,
    2: PHASE_ABILITY,
    3: PHASE_SNAP,
    4: PHASE_SNAP,
    5: PHASE_TERMINAL,
}

_MAX_TURNS: int = 46
_STOCK_TOTAL: int = 46


@dataclass
class EpisodeSample:
    """One training sample from a ReBeL self-play episode decision point."""

    features: np.ndarray      # (PBS_INPUT_DIM,) = (956,) float32
    value_target: np.ndarray  # (VALUE_DIM,)     = (936,) float32
    policy_target: np.ndarray # (NUM_ACTIONS,)   = (146,) float32
    action_mask: np.ndarray   # (NUM_ACTIONS,)   = (146,) bool


def _build_pbs(game: Any, range_p0: np.ndarray, range_p1: np.ndarray) -> PBS:
    """Construct a PBS from a GoEngine state and range distributions."""
    ctx = game.decision_ctx()
    phase = _CTX_TO_PHASE.get(ctx, PHASE_DRAW)
    pub = make_public_features(
        turn=game.turn_number(),
        max_turns=_MAX_TURNS,
        phase=phase,
        discard_top_bucket=None,  # not exposed via current FFI
        stockpile_remaining=game.stock_len(),
        stockpile_total=_STOCK_TOTAL,
    )
    return PBS(
        range_p0=range_p0.copy(),
        range_p1=range_p1.copy(),
        public_features=pub,
    )


def _get_leaf_values(
    value_net: PBSValueNetwork,
    leaf_engines: List[Any],
    range_p0: np.ndarray,
    range_p1: np.ndarray,
) -> np.ndarray:
    """
    Encode each leaf's PBS through the value network and collapse to per-player scalars.

    Args:
        value_net: PBSValueNetwork in eval mode.
        leaf_engines: List of GoEngine views for leaf states.
        range_p0: Current range distribution for player 0 (NUM_HAND_TYPES,).
        range_p1: Current range distribution for player 1 (NUM_HAND_TYPES,).

    Returns:
        float32 array of shape (n_leaves, 2) — per-player scalar value at each leaf.
        Player 0 value = dot(range_p0, V[0:468]); Player 1 = dot(range_p1, V[468:936]).
    """
    n = len(leaf_engines)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)

    leaf_pbss = [_build_pbs(eng, range_p0, range_p1) for eng in leaf_engines]
    leaf_enc = encode_pbs_batch(leaf_pbss)  # (n, 956)

    enc_t = torch.from_numpy(leaf_enc)
    leaf_cf = value_net(enc_t).detach().numpy()  # (n, 936)

    # Collapse per-hand-type CFs to per-player scalars via range dot product
    v_p0 = leaf_cf[:, :NUM_HAND_TYPES] @ range_p0   # (n,)
    v_p1 = leaf_cf[:, NUM_HAND_TYPES:] @ range_p1   # (n,)

    return np.column_stack([v_p0, v_p1]).astype(np.float32)  # (n, 2)


def rebel_self_play_episode(
    game_config: Any,
    value_net: PBSValueNetwork,
    policy_net: PBSPolicyNetwork,
    rebel_config: DeepCfrConfig,
    exploration_epsilon: float = 0.05,
) -> List[EpisodeSample]:
    """
    Run one ReBeL self-play episode.

    At each decision point:
      1. Construct PBS from current range distributions + public state
      2. Build depth-limited subgame (Go solver)
      3. Evaluate leaf nodes with value network → per-player scalars
      4. Solve subgame with CFR → (averaged_strategy, root_values_2d)
      5. Record (pbs_enc, value_target_936, strategy_146, mask_146)
      6. Sample action with epsilon-greedy exploration
      7. Apply action and update both agent belief states

    Range vectors are initialised as uniform and kept uniform throughout this
    initial implementation (full Bayesian updating requires per-hand-type policy
    queries and is deferred to a later phase).

    Args:
        game_config: House rules object passed to GoEngine (or None for Go defaults).
        value_net: PBSValueNetwork in eval mode, on CPU.
        policy_net: PBSPolicyNetwork in eval mode, on CPU (reserved for future use).
        rebel_config: DeepCfrConfig carrying rebel_* hyper-parameters.
        exploration_epsilon: Probability of choosing a uniformly random action.

    Returns:
        List of EpisodeSample, one per decision point encountered during the game.
    """
    from ..ffi.bridge import GoEngine, GoAgentState, SubgameSolver  # deferred for worker spawn

    samples: List[EpisodeSample] = []
    seed = random.getrandbits(64)

    with GoEngine(seed=seed, house_rules=game_config) as game:
        a0 = GoAgentState(game, player_id=0)
        a1 = GoAgentState(game, player_id=1)

        # Range vectors start uniform (no private information observed yet)
        range_p0 = uniform_range()
        range_p1 = uniform_range()

        try:
            with torch.inference_mode():
                while not game.is_terminal():
                    mask_u8 = game.legal_actions_mask()  # (146,) uint8
                    mask_bool = mask_u8.astype(bool)

                    if not mask_bool.any():
                        break

                    # --- Construct root PBS ---
                    root_pbs = _build_pbs(game, range_p0, range_p1)
                    root_pbs_enc = encode_pbs(root_pbs)  # (956,)

                    # --- Default strategy: uniform over legal actions ---
                    n_legal = int(mask_bool.sum())
                    strategy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    strategy[mask_bool] = 1.0 / n_legal
                    root_values_2d = np.zeros(2, dtype=np.float32)

                    # --- Build subgame and solve ---
                    with SubgameSolver(
                        game, max_depth=rebel_config.rebel_subgame_depth
                    ) as solver:
                        leaf_engines = solver.export_leaves()
                        n_leaves = len(leaf_engines)

                        if n_leaves > 0:
                            leaf_values = _get_leaf_values(
                                value_net, leaf_engines, range_p0, range_p1
                            )  # (n_leaves, 2)
                            # Free leaf game handles eagerly — they consumed
                            # game pool slots and are no longer needed.
                            del leaf_engines
                            solver.free_leaves()
                            strategy, root_values_2d = solver.solve(
                                leaf_values,
                                num_iterations=rebel_config.rebel_cfr_iterations,
                            )

                    # --- Build 936-dim value target ---
                    # Tile per-player scalar root values across all hand-type slots.
                    # Full per-hand-type CFs would require the Go solver to propagate
                    # range-weighted values — deferred to a later implementation phase.
                    value_target = np.empty(VALUE_DIM, dtype=np.float32)
                    value_target[:NUM_HAND_TYPES] = root_values_2d[0]
                    value_target[NUM_HAND_TYPES:] = root_values_2d[1]

                    samples.append(
                        EpisodeSample(
                            features=root_pbs_enc.astype(np.float32),
                            value_target=value_target,
                            policy_target=strategy.astype(np.float32),
                            action_mask=mask_bool.copy(),
                        )
                    )

                    # --- Sample action with epsilon-greedy exploration ---
                    legal_indices = np.where(mask_bool)[0]
                    if random.random() < exploration_epsilon:
                        action = int(np.random.choice(legal_indices))
                    else:
                        probs = strategy.copy()
                        probs[~mask_bool] = 0.0
                        total = float(probs.sum())
                        if total <= 0.0:
                            action = int(np.random.choice(legal_indices))
                        else:
                            probs /= total
                            action = int(np.random.choice(NUM_ACTIONS, p=probs))

                    # --- Apply action and update agent beliefs ---
                    game.apply_action(action)
                    game.update_both(a0, a1)

        finally:
            a0.close()
            a1.close()

    return samples


def run_rebel_episodes(
    num_episodes: int,
    value_net: PBSValueNetwork,
    policy_net: PBSPolicyNetwork,
    rebel_config: DeepCfrConfig,
    game_config: Any = None,
    exploration_epsilon: float = 0.05,
) -> List[List[EpisodeSample]]:
    """
    Run multiple ReBeL episodes in a single process (for testing and sequential use).

    Args:
        num_episodes: Number of self-play episodes to run.
        value_net: PBSValueNetwork (will be set to eval mode).
        policy_net: PBSPolicyNetwork (will be set to eval mode).
        rebel_config: DeepCfrConfig carrying rebel_* hyper-parameters.
        game_config: House rules passed to GoEngine (None = Go defaults).
        exploration_epsilon: Probability of uniform random action.

    Returns:
        List of episode sample lists, one inner list per episode.
    """
    value_net.eval()
    policy_net.eval()

    results: List[List[EpisodeSample]] = []
    with torch.inference_mode():
        for _ in range(num_episodes):
            ep = rebel_self_play_episode(
                game_config, value_net, policy_net, rebel_config, exploration_epsilon
            )
            results.append(ep)
    return results


def rebel_worker_fn(
    task_queue: "mp.Queue[Optional[Tuple[int, float]]]",
    result_queue: "mp.Queue",
    value_net_state: Dict[str, Any],
    policy_net_state: Dict[str, Any],
    rebel_config: DeepCfrConfig,
    game_config: Any,
) -> None:
    """
    Worker process target for parallel ReBeL episode generation.

    Reconstructs value and policy networks from state dicts, then loops:
      - Pull (episode_id, epsilon) from task_queue
      - Run rebel_self_play_episode
      - Push (episode_id, samples, error_or_None) to result_queue

    Shutdown: push None onto task_queue to signal termination.

    Args:
        task_queue: Each item is (episode_id: int, epsilon: float), or None to stop.
        result_queue: Each item is (episode_id, list[EpisodeSample], error_str_or_None).
        value_net_state: state_dict for PBSValueNetwork.
        policy_net_state: state_dict for PBSPolicyNetwork.
        rebel_config: DeepCfrConfig carrying rebel_* hyper-parameters.
        game_config: House rules passed to GoEngine (None = Go defaults).
    """
    value_net = PBSValueNetwork()
    value_net.load_state_dict(value_net_state)
    value_net.eval()

    policy_net = PBSPolicyNetwork()
    policy_net.load_state_dict(policy_net_state)
    policy_net.eval()

    with torch.inference_mode():
        while True:
            try:
                task = task_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            if task is None:  # shutdown sentinel
                break

            ep_id, epsilon = task
            try:
                samples = rebel_self_play_episode(
                    game_config, value_net, policy_net, rebel_config, epsilon
                )
                result_queue.put((ep_id, samples, None))
            except Exception as exc:
                logger.exception("rebel_worker_fn: episode %d failed", ep_id)
                result_queue.put((ep_id, [], str(exc)))
