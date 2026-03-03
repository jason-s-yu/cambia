#!/usr/bin/env python3
"""
Diagnostic: evaluate a RANDOMLY INITIALIZED ResNet against all baselines.

Answers the question: "Does training actively hurt the agent, or is the
encoding/action-space biased even before training?"

If random-init network achieves ~50% WR -> training actively hurts.
If <50% -> structural bias in action mapping or encoding.

Uses the same architecture as Phase 2 (interleaved-resnet-adaptive):
  ResidualAdvantageNetwork(input_dim=200, hidden_dim=256, 3 layers, dropout=0.1)
  Interleaved EP-PBS encoding, regret-matching action selection.
"""

import sys
import os
import time
import random
import copy
import logging
from collections import Counter
from typing import Optional, Set, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Ensure cfr/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
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
)
from src.agent_state import AgentState, AgentObservation
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
from src.encoding import encode_action_mask, index_to_action
from src.networks import build_advantage_network, get_strategy_from_advantages
from src.evaluate_agents import NeuralAgentWrapper, AGENT_REGISTRY, get_agent
from src.cfr.exceptions import GameStateError, AgentStateError, ObservationUpdateError, ActionEncodingError

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class RandomInitAgentWrapper(NeuralAgentWrapper):
    """
    Wraps a randomly initialized ResNet for evaluation.
    Uses interleaved EP-PBS encoding + regret matching, identical to DeepCFRAgentWrapper.
    """

    def __init__(
        self,
        player_id: int,
        config,
        input_dim: int = 200,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__(player_id, config, device=device, use_argmax=False)
        from src.encoding import NUM_ACTIONS

        self._NUM_ACTIONS = NUM_ACTIONS
        self._encoding_mode = "ep_pbs"
        self._encoding_layout = "interleaved"
        self._network_type = "residual"
        self._net_input_dim = input_dim

        self.advantage_net = build_advantage_network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=NUM_ACTIONS,
            dropout=dropout,
            validate_inputs=False,
            num_hidden_layers=num_hidden_layers,
            use_residual=True,
            network_type="residual",
        )
        self.advantage_net.to(self.device)
        self.advantage_net.eval()

        # Count params
        total = sum(p.numel() for p in self.advantage_net.parameters())
        print(f"  RandomInitAgent P{player_id}: {total:,} params, input_dim={input_dim}")

    def choose_action(
        self, game_state: CambiaGameState, legal_actions: Set[GameAction]
    ) -> GameAction:
        """Choose action via regret matching on random network outputs."""
        if not self.agent_state:
            return random.choice(list(legal_actions))

        legal_list = list(legal_actions)
        decision_context = self._get_decision_context(game_state)

        try:
            features = self._encode_eppbs(decision_context)
            action_mask = encode_action_mask(legal_list)
        except Exception as e:
            logger.error("RandomInitAgent P%d encoding error: %s", self.player_id, e)
            return random.choice(legal_list)

        with torch.inference_mode():
            feat_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            advantages = self.advantage_net(feat_t, mask_t)
            strategy = get_strategy_from_advantages(advantages, mask_t)
            probs = strategy.squeeze(0).cpu().numpy()

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


def run_single_matchup(
    agent_factory,
    baseline_type: str,
    config,
    num_games: int,
) -> Dict:
    """Run num_games between agent_factory (P0) and baseline_type (P1)."""
    results = Counter()
    score_margins = []
    game_turns_list = []
    t1_cambia_count = 0

    for game_num in tqdm(range(1, num_games + 1), desc=f"vs {baseline_type}", unit="game", leave=False):
        try:
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            agent0 = agent_factory(player_id=0)
            agent1 = get_agent(baseline_type, player_id=1, config=config)
            agents = [agent0, agent1]

            # Initialize stateful agents
            for agent in agents:
                if isinstance(agent, NeuralAgentWrapper):
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
                    results["Errors"] += 1
                    break

                current_agent = agents[acting_player_id]
                try:
                    legal_actions = game_state.get_legal_actions()
                    if not legal_actions:
                        if game_state.is_terminal():
                            break
                        results["Errors"] += 1
                        break

                    chosen_action = current_agent.choose_action(game_state, legal_actions)

                    # Track T1 Cambia
                    if turn == 1 and acting_player_id == 0 and type(chosen_action).__name__ == "ActionCallCambia":
                        t1_cambia_count += 1

                    state_delta, undo_info = game_state.apply_action(chosen_action)
                    if not callable(undo_info):
                        results["Errors"] += 1
                        break

                    # Update stateful agents
                    has_stateful = any(isinstance(a, NeuralAgentWrapper) for a in agents)
                    if has_stateful and hasattr(current_agent, "_create_observation"):
                        observation = current_agent._create_observation(
                            game_state, chosen_action, acting_player_id
                        )
                        if observation:
                            for agent in agents:
                                if isinstance(agent, NeuralAgentWrapper):
                                    agent.update_state(observation)

                except (GameStateError, AgentStateError, ObservationUpdateError) as e:
                    logger.error("Game %d error: %s", game_num, e)
                    results["Errors"] += 1
                    break
                except Exception as e:
                    logger.error("Game %d unexpected error: %s", game_num, e)
                    results["Errors"] += 1
                    break

            # Determine winner
            if game_state.is_terminal():
                winner = game_state._winner
                game_turns_list.append(turn)
                if winner == 0:
                    results["P0_Wins"] += 1
                elif winner == 1:
                    results["P1_Wins"] += 1
                else:
                    results["Ties"] += 1
                # Capture score margin
                try:
                    hand_scores = [
                        sum(card.value for card in game_state.players[i].hand)
                        for i in range(len(game_state.players))
                    ]
                    if len(hand_scores) == 2:
                        margin = hand_scores[1] - hand_scores[0]
                        score_margins.append(float(margin))
                except Exception:
                    pass
            else:
                results["Timeout"] += 1

        except Exception as e:
            logger.error("Game %d setup error: %s", game_num, e)
            results["Errors"] += 1

    total_decided = results["P0_Wins"] + results["P1_Wins"] + results["Ties"]
    wr = results["P0_Wins"] / total_decided * 100 if total_decided > 0 else 0.0
    avg_turns = np.mean(game_turns_list) if game_turns_list else 0.0
    avg_margin = np.mean(score_margins) if score_margins else 0.0
    t1c_rate = t1_cambia_count / num_games * 100 if num_games > 0 else 0.0

    return {
        "baseline": baseline_type,
        "wr": wr,
        "p0_wins": results["P0_Wins"],
        "p1_wins": results["P1_Wins"],
        "ties": results["Ties"],
        "errors": results["Errors"],
        "timeouts": results.get("Timeout", 0),
        "avg_turns": avg_turns,
        "avg_margin": avg_margin,
        "t1c_rate": t1c_rate,
        "total_games": num_games,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Random-init network diagnostic")
    parser.add_argument(
        "-c", "--config", type=str,
        default="runs/interleaved-resnet-adaptive/config.yaml",
        help="Config YAML path",
    )
    parser.add_argument("-n", "--num-games", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    if not config:
        print(f"ERROR: Failed to load config from {args.config}")
        sys.exit(1)

    print(f"Config: {args.config}")
    print(f"Architecture: ResidualAdvantageNetwork(input={args.input_dim}, hidden={args.hidden_dim}, layers={args.num_layers}, dropout={args.dropout})")
    print(f"Encoding: interleaved EP-PBS, truncated to {args.input_dim} dims")
    print(f"Action selection: regret matching (ReLU + normalize)")
    print(f"Games per baseline: {args.num_games}")
    print(f"Seed: {args.seed}")
    print()

    # Create a single random network (shared weights for all games)
    def agent_factory(player_id):
        return RandomInitAgentWrapper(
            player_id=player_id,
            config=config,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_layers,
            dropout=args.dropout,
            device="cpu",
        )

    # Test with a quick game first
    print("Smoke test (1 game vs random_no_cambia)...")
    smoke = run_single_matchup(agent_factory, "random_no_cambia", config, 1)
    print(f"  Smoke test passed. Turns: {smoke['avg_turns']:.0f}")
    print()

    baselines = [
        "random",
        "greedy",
        "random_no_cambia",
        "random_late_cambia",
        "imperfect_greedy",
        "memory_heuristic",
        "aggressive_snap",
    ]

    MI3_BASELINES = {"imperfect_greedy", "memory_heuristic", "aggressive_snap"}
    MI5_BASELINES = {"random_no_cambia", "random_late_cambia", "imperfect_greedy", "memory_heuristic", "aggressive_snap"}

    all_results = []
    start = time.time()

    for bl in baselines:
        result = run_single_matchup(agent_factory, bl, config, args.num_games)
        all_results.append(result)
        print(f"  {bl:25s}  WR={result['wr']:5.1f}%  wins={result['p0_wins']}/{result['total_games']}  "
              f"avg_turns={result['avg_turns']:5.1f}  T1C={result['t1c_rate']:.1f}%  "
              f"margin={result['avg_margin']:+.1f}  err={result['errors']}")

    elapsed = time.time() - start

    # Compute mi(3) and mi(5)
    mi3_wrs = [r["wr"] for r in all_results if r["baseline"] in MI3_BASELINES]
    mi5_wrs = [r["wr"] for r in all_results if r["baseline"] in MI5_BASELINES]
    mi3 = np.mean(mi3_wrs) if mi3_wrs else 0.0
    mi5 = np.mean(mi5_wrs) if mi5_wrs else 0.0

    print()
    print("=" * 70)
    print("RANDOM-INIT NETWORK DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"{'Baseline':25s} {'WR%':>6s} {'Wins':>6s} {'Losses':>6s} {'Ties':>5s} {'AvgTurns':>8s} {'T1C%':>5s}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['baseline']:25s} {r['wr']:5.1f}% {r['p0_wins']:6d} {r['p1_wins']:6d} {r['ties']:5d} {r['avg_turns']:8.1f} {r['t1c_rate']:5.1f}")
    print("-" * 70)
    print(f"mean_imp(3) = {mi3:.1f}%")
    print(f"mean_imp(5) = {mi5:.1f}%")
    print(f"Elapsed: {elapsed:.0f}s")
    print()

    # Interpretation
    print("INTERPRETATION:")
    if mi5 > 47:
        print("  Random-init network achieves ~50% => training ACTIVELY HURTS the agent.")
        print("  The network bias + regret matching produces a near-uniform strategy that")
        print("  outperforms the trained agent. Training is converging to a worse strategy.")
    elif mi5 > 38:
        print("  Random-init network achieves 38-47% => moderate structural bias exists,")
        print("  but training adds some value (trained agent at ~41% mi(5)).")
        print("  The encoding/action-mapping has some inherent bias.")
    elif mi5 > 30:
        print("  Random-init network achieves 30-38% => similar to trained agent!")
        print("  Training barely improves over random init. The function approximation")
        print("  is failing to learn meaningful strategies.")
    else:
        print("  Random-init network achieves <30% => training DOES help significantly.")
        print("  The trained agent (35% mi(3) / 41% mi(5)) meaningfully improves over")
        print("  random initialization.")


if __name__ == "__main__":
    main()
