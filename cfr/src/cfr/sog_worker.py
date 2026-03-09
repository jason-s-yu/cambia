"""
src/cfr/sog_worker.py

SoG self-play episode runner, Phase 3.

Extends GT-CFR worker with continual re-solving via SoGSearch.
Tree is threaded across sequential decisions within an episode.

Training data shapes match GT-CFR (same CVPN, same EpisodeSample):
    features:      (PBS_INPUT_DIM,) = (956,)  float32
    value_target:  (VALUE_DIM,)    = (936,)  float32
    policy_target: (NUM_ACTIONS,)  = (146,)  float32
    action_mask:   (NUM_ACTIONS,)  = (146,)  bool
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

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
from .gtcfr_worker import EpisodeSample  # reuse from Phase 2, do not redefine
from .sog_search import SoGSearch

logger = logging.getLogger(__name__)

VALUE_DIM: int = 2 * NUM_HAND_TYPES

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


def _build_pbs(game: Any, range_p0: np.ndarray, range_p1: np.ndarray) -> PBS:
    """Construct a PBS from a GoEngine state and range distributions."""
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


def sog_self_play_episode(
    game_config: Any,
    cvpn: CVPN,
    config: DeepCfrConfig,
    exploration_epsilon: float = 0.05,
) -> List[EpisodeSample]:
    """
    Run one SoG self-play episode with continual re-solving.

    At each decision point:
      1. Construct PBS from current range distributions + public state
      2. Run SoGSearch.search() with prior_tree from previous step
      3. Record (pbs_enc, value_target_936, policy_146, mask_146)
      4. Sample action with epsilon-greedy exploration
      5. Pass tree to next decision as prior_tree

    Args:
        game_config: House rules for GoEngine (None = Go defaults).
        cvpn:        CVPN in eval mode on CPU.
        config:      DeepCfrConfig with sog_* hyper-parameters.
        exploration_epsilon: Probability of random action.

    Returns:
        List of EpisodeSample, one per decision point.
    """
    from ..ffi.bridge import GoEngine, GoAgentState  # deferred for worker spawn

    samples: List[EpisodeSample] = []
    seed = random.getrandbits(64)

    sog_search = SoGSearch(
        cvpn=cvpn,
        train_budget=config.sog_train_budget,
        eval_budget=config.sog_eval_budget,
        c_puct=config.sog_c_puct,
        cfr_iters_per_expansion=config.sog_cfr_iters_per_expansion,
        expansion_k=config.gtcfr_expansion_k,
        device="cpu",
        max_persist_depth=config.sog_max_persist_depth,
        max_persist_handles=config.sog_max_persist_handles,
        safety_margin=config.sog_safety_margin,
        safety_check_enabled=False,  # training: low budget can't beat CVPN estimates
    )
    sog_search.use_train_budget()

    prior_tree = None
    prev_action: Optional[int] = None

    with GoEngine(seed=seed, house_rules=game_config) as game:
        a0 = GoAgentState(game, player_id=0)
        a1 = GoAgentState(game, player_id=1)

        range_p0 = uniform_range()
        range_p1 = uniform_range()

        # Diagnostic accumulators
        _entropy_p0: List[float] = []
        _entropy_p1: List[float] = []
        _policy_entropies: List[float] = []
        _tree_depths: List[int] = []
        _decision_count: int = 0

        try:
            with torch.inference_mode():
                while not game.is_terminal():
                    if _decision_count >= _MAX_DECISIONS:
                        logger.warning(
                            "sog_episode: hit %d decision cap, aborting.",
                            _MAX_DECISIONS,
                        )
                        break
                    _decision_count += 1

                    mask_u8 = game.legal_actions_mask()
                    mask_bool = mask_u8.astype(bool)

                    if not mask_bool.any():
                        break

                    # Construct root PBS and encode
                    root_pbs = _build_pbs(game, range_p0, range_p1)
                    root_pbs_enc = encode_pbs(root_pbs)

                    # SoG search with tree persistence
                    result = sog_search.search(
                        game,
                        range_p0,
                        range_p1,
                        prior_tree=prior_tree,
                        action_taken=prev_action,
                    )
                    prior_tree = sog_search.get_tree()

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

                    acting_player = game.acting_player()

                    # Epsilon-greedy action selection
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

                    # Per-hand-type policy matrix BEFORE apply_action
                    # (state must reflect pre-action decision point)
                    from .range_utils import compute_policy_matrix_cvpn
                    policy_matrix = compute_policy_matrix_cvpn(
                        cvpn, game, range_p0, range_p1
                    )

                    game.apply_action(action)
                    game.update_both(a0, a1)

                    # Bayesian range update with per-hand-type policies
                    if acting_player == 0:
                        range_p0 = update_range(range_p0, action, policy_matrix)
                    else:
                        range_p1 = update_range(range_p1, action, policy_matrix)

                    # Diagnostics: range entropy, policy entropy, tree depth
                    _h0 = float(-np.sum(range_p0 * np.log(range_p0 + 1e-10)))
                    _h1 = float(-np.sum(range_p1 * np.log(range_p1 + 1e-10)))
                    _entropy_p0.append(_h0)
                    _entropy_p1.append(_h1)

                    legal_p = policy[mask_bool]
                    legal_p = legal_p / (legal_p.sum() + 1e-10)
                    _pe = float(-np.sum(legal_p * np.log(legal_p + 1e-10)))
                    _policy_entropies.append(_pe)

                    _tree_depths.append(result.depth_stats.get("max", 0))

                    prev_action = action

        finally:
            a0.close()
            a1.close()
            sog_search.cleanup()

    if _entropy_p0:
        logger.info(
            "sog_episode done: samples=%d "
            "range_entropy p0=%.3f->%.3f p1=%.3f->%.3f "
            "policy_entropy mean=%.3f tree_depth_max mean=%.1f",
            len(samples),
            _entropy_p0[0], _entropy_p0[-1],
            _entropy_p1[0], _entropy_p1[-1],
            float(np.mean(_policy_entropies)),
            float(np.mean(_tree_depths)),
        )
    else:
        logger.info("sog_episode done: samples=%d", len(samples))
    return samples


def _sog_batch_worker(args: Tuple) -> List:
    """
    ProcessPoolExecutor worker that runs N SoG self-play episodes.

    Must be module-level (not a closure/method) for ProcessPoolExecutor pickle.

    Args:
        args: (num_episodes, cvpn_state_numpy, config, game_config)
            cvpn_state_numpy: {str: np.ndarray} state_dict for CVPN
            config:           DeepCfrConfig with sog_* hyper-parameters
            game_config:      House rules for GoEngine (None = defaults)

    Returns:
        Flat list of EpisodeSample from all completed episodes.
    """
    num_episodes, cvpn_state_numpy, config, game_config = args

    from ..networks import build_cvpn

    cvpn = build_cvpn(
        hidden_dim=config.gtcfr_cvpn_hidden_dim,
        num_blocks=config.gtcfr_cvpn_num_blocks,
        validate_inputs=False,
        detach_policy_grad=config.cvpn_detach_policy_grad,
    )
    weights = {
        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in cvpn_state_numpy.items()
    }
    cvpn.load_state_dict(weights)
    cvpn.eval()

    all_samples = []
    with torch.inference_mode():
        for _ in range(num_episodes):
            try:
                samples = sog_self_play_episode(
                    game_config,
                    cvpn,
                    config,
                    exploration_epsilon=config.sog_exploration_epsilon,
                )
                all_samples.extend(samples)
            except Exception as e:
                logger.warning("SoG episode failed: %s", e)

    return all_samples
