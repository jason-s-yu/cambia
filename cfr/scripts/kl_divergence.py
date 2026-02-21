"""
scripts/kl_divergence.py

Compute KL-divergence between two Deep CFR checkpoints' strategy distributions
over a set of randomly-sampled game states.

Usage:
    python scripts/kl_divergence.py \
        --checkpoint-a cfr/runs/os-20/checkpoints/deep_cfr_checkpoint_iter_25.pt \
        --checkpoint-b cfr/runs/os-20/checkpoints/deep_cfr_checkpoint_iter_50.pt \
        [--num-states 500]

Reports mean/max/median KL(a||b) over the sampled states.
"""

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

import torch

from src.networks import AdvantageNetwork, get_strategy_from_advantages
from src.encoding import INPUT_DIM, NUM_ACTIONS, encode_infoset, encode_action_mask
from src.agent_state import AgentState, AgentObservation
from src.config import Config, CambiaRulesConfig, AgentParamsConfig
from src.constants import DecisionContext, NUM_PLAYERS
from src.game.engine import CambiaGameState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_EPS = 1e-10


def _load_network(checkpoint_path: str, device: torch.device) -> AdvantageNetwork:
    """Load an AdvantageNetwork from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dcfr_config = checkpoint.get("dcfr_config", {})
    hidden_dim = dcfr_config.get("hidden_dim", 256)

    net = AdvantageNetwork(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=NUM_ACTIONS,
        validate_inputs=False,
    )
    net.load_state_dict(checkpoint["advantage_net_state_dict"])
    net.to(device)
    net.eval()
    return net


def _make_minimal_config() -> Config:
    """Create a minimal Config object for agent state initialization."""
    config = Config()
    # Instantiate sub-configs — construct without kwargs to handle stub environments.
    rules = CambiaRulesConfig()
    rules.use_jokers = 0  # No jokers for speed in state sampling.
    config.cambia_rules = rules

    agent_params = AgentParamsConfig()
    agent_params.memory_level = 1
    agent_params.time_decay_turns = 3
    config.agent_params = agent_params
    return config


def _init_agent_state(game_state: CambiaGameState, player_id: int, config: Config) -> AgentState:
    """Initialize an AgentState from a fresh game state."""
    opponent_id = 1 - player_id
    hand = game_state.players[player_id].hand
    peeks = game_state.players[player_id].initial_peek_indices

    agent_state = AgentState(
        player_id=player_id,
        opponent_id=opponent_id,
        memory_level=config.agent_params.memory_level,
        time_decay_turns=config.agent_params.time_decay_turns,
        initial_hand_size=len(hand),
        config=config,
    )
    obs = AgentObservation(
        acting_player=-1,
        action=None,
        discard_top_card=game_state.get_discard_top(),
        player_hand_sizes=[game_state.get_player_card_count(i) for i in range(NUM_PLAYERS)],
        stockpile_size=game_state.get_stockpile_size(),
        drawn_card=None,
        peeked_cards=None,
        snap_results=copy.deepcopy(game_state.snap_results_log),
        did_cambia_get_called=game_state.cambia_caller_id is not None,
        who_called_cambia=game_state.cambia_caller_id,
        is_game_over=game_state.is_terminal(),
        current_turn=game_state.get_turn_number(),
    )
    agent_state.initialize(obs, hand, peeks)
    return agent_state


def _get_strategy(
    net: AdvantageNetwork,
    features: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run the network and return a strategy distribution (numpy, shape NUM_ACTIONS)."""
    with torch.inference_mode():
        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask.astype(np.float32)).bool().unsqueeze(0).to(device)
        advantages = net(feat_t, mask_t)
        strategy = get_strategy_from_advantages(advantages, mask_t)
        return strategy.squeeze(0).cpu().numpy()


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence KL(p || q) over legal-action support."""
    support = (p > 0)
    if not support.any():
        return 0.0
    p_s = p[support]
    q_s = np.clip(q[support], _EPS, None)
    return float(np.sum(p_s * np.log(p_s / q_s)))


def _sample_game_states(
    num_states: int, config: Config, max_steps_per_game: int = 30
) -> list[tuple[AgentState, np.ndarray, DecisionContext]]:
    """
    Generate random game states by playing random legal actions.

    Returns a list of (agent_state, action_mask, decision_context) tuples
    corresponding to decision points where the network can be queried.
    """
    samples = []
    rng = np.random.default_rng(seed=42)
    max_attempts = num_states * 10

    attempts = 0
    while len(samples) < num_states and attempts < max_attempts:
        attempts += 1
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
        except Exception as e:
            logger.debug("Failed to create game state: %s", e)
            continue

        agent_state = _init_agent_state(game_state, player_id=0, config=config)

        for _ in range(rng.integers(1, max_steps_per_game + 1)):
            if game_state.is_terminal():
                break
            legal_actions = list(game_state.get_legal_actions())
            if not legal_actions:
                break

            action_mask = encode_action_mask(legal_actions)
            if action_mask.sum() == 0:
                break

            # Determine decision context.
            from src.constants import (
                ActionDiscard,
                ActionAbilityPeekOwnSelect,
                ActionAbilityPeekOtherSelect,
                ActionAbilityBlindSwapSelect,
                ActionAbilityKingLookSelect,
                ActionAbilityKingSwapDecision,
                ActionSnapOpponentMove,
            )

            if game_state.snap_phase_active:
                ctx = DecisionContext.SNAP_DECISION
            elif game_state.pending_action:
                pa = game_state.pending_action
                if isinstance(pa, ActionDiscard):
                    ctx = DecisionContext.POST_DRAW
                elif isinstance(
                    pa,
                    (
                        ActionAbilityPeekOwnSelect,
                        ActionAbilityPeekOtherSelect,
                        ActionAbilityBlindSwapSelect,
                        ActionAbilityKingLookSelect,
                        ActionAbilityKingSwapDecision,
                    ),
                ):
                    ctx = DecisionContext.ABILITY_SELECT
                elif isinstance(pa, ActionSnapOpponentMove):
                    ctx = DecisionContext.SNAP_MOVE
                else:
                    ctx = DecisionContext.START_TURN
            else:
                ctx = DecisionContext.START_TURN

            try:
                features = encode_infoset(agent_state, ctx)
                samples.append((agent_state, action_mask, ctx, features))
            except Exception:
                pass

            # Apply a random action to advance state.
            idx = rng.integers(0, len(legal_actions))
            chosen = legal_actions[idx]
            try:
                game_state.apply_action(chosen)
            except Exception:
                break

            # Update agent state briefly (public info only).
            try:
                obs = AgentObservation(
                    acting_player=game_state.get_acting_player(),
                    action=chosen,
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
                agent_state.update(obs)
            except Exception:
                pass

    return samples[:num_states]


def compute_kl_divergence(
    checkpoint_a: str,
    checkpoint_b: str,
    num_states: int = 500,
    device_str: str = "cpu",
) -> dict:
    """
    Compute KL(a||b) divergence over random game states.

    Returns dict with keys: mean_kl, max_kl, median_kl, num_states_sampled.
    """
    device = torch.device(device_str)

    logger.info("Loading checkpoint A: %s", checkpoint_a)
    net_a = _load_network(checkpoint_a, device)

    logger.info("Loading checkpoint B: %s", checkpoint_b)
    net_b = _load_network(checkpoint_b, device)

    config = _make_minimal_config()

    logger.info("Sampling %d game states...", num_states)
    samples = _sample_game_states(num_states, config)
    logger.info("Got %d samples", len(samples))

    if not samples:
        logger.warning("No samples collected.")
        return {"mean_kl": float("nan"), "max_kl": float("nan"), "median_kl": float("nan"), "num_states_sampled": 0}

    kl_values = []
    for agent_state, action_mask, ctx, features in samples:
        try:
            strat_a = _get_strategy(net_a, features, action_mask, device)
            strat_b = _get_strategy(net_b, features, action_mask, device)
            kl = _kl_div(strat_a, strat_b)
            kl_values.append(kl)
        except Exception as e:
            logger.debug("Skipping sample due to error: %s", e)

    if not kl_values:
        return {"mean_kl": float("nan"), "max_kl": float("nan"), "median_kl": float("nan"), "num_states_sampled": 0}

    kl_arr = np.array(kl_values)
    return {
        "mean_kl": float(np.mean(kl_arr)),
        "max_kl": float(np.max(kl_arr)),
        "median_kl": float(np.median(kl_arr)),
        "num_states_sampled": len(kl_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute KL-divergence between two Deep CFR checkpoint strategy distributions."
    )
    parser.add_argument("--checkpoint-a", required=True, help="Path to first .pt checkpoint")
    parser.add_argument("--checkpoint-b", required=True, help="Path to second .pt checkpoint")
    parser.add_argument(
        "--num-states",
        type=int,
        default=500,
        help="Number of random game states to sample (default: 500)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device string (default: cpu)")

    args = parser.parse_args()

    result = compute_kl_divergence(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        num_states=args.num_states,
        device_str=args.device,
    )

    print(f"KL-divergence KL(a||b) over {result['num_states_sampled']} states:")
    print(f"  Mean:   {result['mean_kl']:.6f}")
    print(f"  Max:    {result['max_kl']:.6f}")
    print(f"  Median: {result['median_kl']:.6f}")


if __name__ == "__main__":
    main()
