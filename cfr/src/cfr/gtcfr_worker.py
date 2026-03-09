"""
src/cfr/gtcfr_worker.py

GT-CFR self-play episode runner for Phase 2 training.

At each decision point during self-play, constructs a PBS, runs GTCFRSearch,
and records (PBS_encoding, value_target, policy_target, action_mask) tuples.

Training data shapes:
    features:      (PBS_INPUT_DIM,) = (956,)  float32
    value_target:  (VALUE_DIM,)    = (936,)  float32  — from search root CFVs
    policy_target: (NUM_ACTIONS,)  = (146,)  float32  — average strategy from search
    action_mask:   (NUM_ACTIONS,)  = (146,)  bool
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from ..config import DeepCfrConfig
from ..encoding import NUM_ACTIONS
from ..networks import CVPN
from ..pbs import (
    PBS,
    PBS_INPUT_DIM,
    NUM_HAND_TYPES,
    encode_pbs,
    uniform_range,
    update_range,
    make_public_features,
    PHASE_DRAW,
    PHASE_DISCARD,
    PHASE_ABILITY,
    PHASE_SNAP,
    PHASE_TERMINAL,
)
from .gtcfr_search import GTCFRSearch

logger = logging.getLogger(__name__)

# Value target dimension: 2 * NUM_HAND_TYPES = 936
VALUE_DIM: int = 2 * NUM_HAND_TYPES

# Decision context integer → PBS phase index
# (copied from rebel_worker.py — coexistence policy)
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
_MAX_DECISIONS: int = 200  # safety cap: abort episode if stuck in loops


@dataclass
class EpisodeSample:
    """One training sample from a GT-CFR self-play episode decision point."""

    features: np.ndarray      # (PBS_INPUT_DIM,) = (956,) float32
    value_target: np.ndarray  # (VALUE_DIM,)     = (936,) float32
    policy_target: np.ndarray # (NUM_ACTIONS,)   = (146,) float32
    action_mask: np.ndarray   # (NUM_ACTIONS,)   = (146,) bool


def _build_pbs(game: Any, range_p0: np.ndarray, range_p1: np.ndarray) -> PBS:
    """Construct a PBS from a GoEngine state and range distributions.

    Copied from rebel_worker.py (coexistence policy: do not import from rebel).
    """
    ctx = game.decision_ctx()
    phase = _CTX_TO_PHASE.get(ctx, PHASE_DRAW)
    pub = make_public_features(
        turn=game.turn_number(),
        max_turns=_MAX_TURNS,
        phase=phase,
        discard_top_bucket=game.discard_top(),
        stockpile_remaining=game.stock_len(),
        stockpile_total=_STOCK_TOTAL,
    )
    return PBS(
        range_p0=range_p0.copy(),
        range_p1=range_p1.copy(),
        public_features=pub,
    )


def gtcfr_self_play_episode(
    game_config: Any,
    cvpn: CVPN,
    config: DeepCfrConfig,
    exploration_epsilon: float = 0.05,
) -> List[EpisodeSample]:
    """
    Run one GT-CFR self-play episode.

    At each decision point:
      1. Construct PBS from current range distributions + public state
      2. Run GTCFRSearch.search() → (policy, root_values)
      3. Record (pbs_enc, value_target_936, policy_146, mask_146)
      4. Sample action with epsilon-greedy exploration from search policy
      5. Apply action and update ranges (simplified Bayesian update via policy)

    Range vectors are initialised as uniform and updated after each action
    using the search policy as a simplified surrogate for P(action | hand_type).

    Args:
        game_config: House rules object passed to GoEngine (or None for Go defaults).
        cvpn: CVPN in eval mode, on CPU.
        config: DeepCfrConfig carrying gtcfr_* hyper-parameters.
        exploration_epsilon: Probability of choosing a uniformly random action.

    Returns:
        List of EpisodeSample, one per decision point encountered during the game.
    """
    from ..ffi.bridge import GoEngine, GoAgentState  # deferred for worker spawn

    samples: List[EpisodeSample] = []
    seed = random.getrandbits(64)

    search = GTCFRSearch(
        cvpn=cvpn,
        expansion_budget=config.gtcfr_expansion_budget,
        c_puct=config.gtcfr_c_puct,
        cfr_iters_per_expansion=config.gtcfr_cfr_iters_per_expansion,
        expansion_k=config.gtcfr_expansion_k,
        device="cpu",
    )

    with GoEngine(seed=seed, house_rules=game_config) as game:
        a0 = GoAgentState(game, player_id=0)
        a1 = GoAgentState(game, player_id=1)

        # Range vectors start uniform
        range_p0 = uniform_range()
        range_p1 = uniform_range()

        _decision_count: int = 0

        try:
            with torch.inference_mode():
                while not game.is_terminal():
                    if _decision_count >= _MAX_DECISIONS:
                        logger.warning(
                            "gtcfr_episode: hit %d decision cap, aborting.",
                            _MAX_DECISIONS,
                        )
                        break
                    _decision_count += 1

                    mask_u8 = game.legal_actions_mask()  # (146,) uint8
                    mask_bool = mask_u8.astype(bool)

                    if not mask_bool.any():
                        break

                    # --- Construct root PBS and encode ---
                    root_pbs = _build_pbs(game, range_p0, range_p1)
                    root_pbs_enc = encode_pbs(root_pbs)  # (956,)

                    # --- GT-CFR search ---
                    result = search.search(game, range_p0, range_p1)
                    # result.policy: (146,) average strategy
                    # result.root_values: (936,) flattened (2, NUM_HAND_TYPES) CFVs

                    policy = result.policy.astype(np.float32)
                    root_values = result.root_values.astype(np.float32)

                    samples.append(
                        EpisodeSample(
                            features=root_pbs_enc.astype(np.float32),
                            value_target=root_values,
                            policy_target=policy,
                            action_mask=mask_bool.copy(),
                        )
                    )

                    # --- Capture acting player BEFORE apply_action ---
                    acting_player = game.acting_player()

                    # --- Sample action with epsilon-greedy exploration ---
                    legal_indices = np.where(mask_bool)[0]
                    if random.random() < exploration_epsilon:
                        action = int(np.random.choice(legal_indices))
                    else:
                        probs = policy.copy()
                        probs[~mask_bool] = 0.0
                        total = float(probs.sum())
                        if total <= 0.0:
                            action = int(np.random.choice(legal_indices))
                        else:
                            probs /= total
                            action = int(np.random.choice(NUM_ACTIONS, p=probs))

                    # --- Per-hand-type policy matrix BEFORE apply_action ---
                    from .range_utils import compute_policy_matrix_cvpn
                    policy_matrix = compute_policy_matrix_cvpn(
                        cvpn, game, range_p0, range_p1
                    )

                    # --- Apply action and update agent beliefs ---
                    game.apply_action(action)
                    game.update_both(a0, a1)

                    # --- Bayesian range update with per-hand-type policies ---
                    if acting_player == 0:
                        range_p0 = update_range(range_p0, action, policy_matrix)
                    else:
                        range_p1 = update_range(range_p1, action, policy_matrix)

        finally:
            a0.close()
            a1.close()

    logger.info(
        "gtcfr_episode done: samples=%d",
        len(samples),
    )

    return samples
