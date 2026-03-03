"""H3 Diagnostic: Evaluate Phase 2 agent vs mixed opponent (0.6*random + 0.4*baseline).

If the agent wins >55-60%, it proves the H3-bugged training produced an agent that
perfectly solved the wrong objective: best-response to a 60%-random opponent.

Usage:
    python scripts/h3_diagnostic.py <checkpoint> [--games 5000] [--config <yaml>]
"""

import argparse
import random
import time
from collections import Counter
from typing import List, Set

from tqdm import tqdm

from src.config import load_config
from src.game.engine import CambiaGameState
from src.constants import GameAction
from src.evaluate_agents import (
    MixedOpponentAgent,
    NeuralAgentWrapper,
    SDCFRAgentWrapper,
    DeepCFRAgentWrapper,
    get_agent,
)
from src.agents.baseline_agents import (
    RandomAgent,
    RandomNoCambiaAgent,
    ImperfectGreedyAgent,
    MemoryHeuristicAgent,
    AggressiveSnapAgent,
)

# Baselines to mix with random at 0.6/0.4
BASELINES_TO_TEST = [
    ("random", RandomAgent),
    ("random_no_cambia", RandomNoCambiaAgent),
    ("imperfect_greedy", ImperfectGreedyAgent),
    ("memory_heuristic", MemoryHeuristicAgent),
    ("aggressive_snap", AggressiveSnapAgent),
]

EPSILON = 0.6  # The H3 bug's epsilon value


def run_h3_diagnostic(
    checkpoint_path: str,
    config_path: str,
    num_games: int = 5000,
    device: str = "cpu",
) -> dict:
    config = load_config(config_path)
    results = {}

    for baseline_name, baseline_cls in BASELINES_TO_TEST:
        label = f"mixed(0.6*random + 0.4*{baseline_name})"
        print(f"\n--- {label} ({num_games} games) ---")

        # Create P0: trained agent (SD-CFR uses EMA wrapper)
        p0 = get_agent(
            "sd_cfr", player_id=0, config=config,
            checkpoint_path=checkpoint_path, device=device,
        )

        # Create P1: mixed opponent
        agent_random = RandomAgent(player_id=1, config=config)
        agent_baseline = baseline_cls(player_id=1, config=config)
        p1 = MixedOpponentAgent(
            player_id=1, config=config,
            agent_a=agent_random, agent_b=agent_baseline,
            weight_a=EPSILON,
        )

        counts = Counter()
        game_turns_list: List[int] = []
        t1_cambia_count = 0

        for game_num in tqdm(range(num_games), desc=f"vs {baseline_name}", unit="game"):
            game_state = CambiaGameState(house_rules=config.cambia_rules)
            if isinstance(p0, NeuralAgentWrapper):
                p0.initialize_state(game_state)

            agents = [p0, p1]
            turn = 0
            max_turns = config.cambia_rules.max_game_turns if config.cambia_rules.max_game_turns > 0 else 500

            while not game_state.is_terminal() and turn < max_turns:
                turn += 1
                acting_player = game_state.get_acting_player()
                if acting_player == -1:
                    counts["errors"] += 1
                    break

                legal_actions = game_state.get_legal_actions()
                if not legal_actions:
                    if game_state.is_terminal():
                        break
                    counts["errors"] += 1
                    break

                current = agents[acting_player]
                chosen = current.choose_action(game_state, legal_actions)

                if turn == 1 and acting_player == 0 and type(chosen).__name__ == "ActionCallCambia":
                    t1_cambia_count += 1

                state_delta, undo_info = game_state.apply_action(chosen)

                # Update P0 state
                if isinstance(p0, NeuralAgentWrapper) and hasattr(p0, "_create_observation"):
                    obs = p0._create_observation(game_state, chosen, acting_player)
                    if obs:
                        p0.update_state(obs)

            if game_state.is_terminal():
                winner = game_state._winner
                if winner == 0:
                    counts["p0_wins"] += 1
                elif winner == 1:
                    counts["p1_wins"] += 1
                else:
                    counts["ties"] += 1
                game_turns_list.append(turn)
            else:
                counts["timeouts"] += 1

        total = counts["p0_wins"] + counts["p1_wins"] + counts["ties"]
        wr = counts["p0_wins"] / total * 100 if total > 0 else 0
        avg_turns = sum(game_turns_list) / len(game_turns_list) if game_turns_list else 0
        t1_rate = t1_cambia_count / num_games * 100

        results[baseline_name] = {
            "win_rate": wr,
            "p0_wins": counts["p0_wins"],
            "p1_wins": counts["p1_wins"],
            "ties": counts["ties"],
            "errors": counts["errors"],
            "avg_turns": avg_turns,
            "t1_cambia_pct": t1_rate,
        }

        print(f"  WR: {wr:.1f}%  P0:{counts['p0_wins']} P1:{counts['p1_wins']} "
              f"Ties:{counts['ties']}  AvgTurns:{avg_turns:.1f}  T1Cambia:{t1_rate:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("H3 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"{'Opponent Mix':<45} {'WR':>6} {'Turns':>6} {'T1%':>5}")
    print("-" * 70)
    for name, r in results.items():
        label = f"0.6*random + 0.4*{name}"
        print(f"{label:<45} {r['win_rate']:>5.1f}% {r['avg_turns']:>5.1f} {r['t1_cambia_pct']:>4.1f}%")

    print("-" * 70)
    avg_wr = sum(r["win_rate"] for r in results.values()) / len(results)
    print(f"{'Mean WR across all mixes':<45} {avg_wr:>5.1f}%")
    print()
    if avg_wr > 55:
        print("DIAGNOSIS: Agent WR > 55% against mixed opponents.")
        print("This confirms the H3 bug: the agent was trained as best-response")
        print("to a 60%-random opponent and solves that objective well.")
    elif avg_wr > 45:
        print("DIAGNOSIS: Agent WR 45-55% — inconclusive.")
        print("The agent may have partially adapted to random opponents.")
    else:
        print("DIAGNOSIS: Agent WR < 45% — H3 not the primary issue.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3 Diagnostic Prediction Test")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--config", "-c", default="runs/interleaved-resnet-adaptive/config.yaml",
                        help="Config YAML path")
    parser.add_argument("--games", "-n", type=int, default=5000, help="Games per mix")
    parser.add_argument("--device", "-d", default="cpu", help="Torch device")
    args = parser.parse_args()

    run_h3_diagnostic(args.checkpoint, args.config, args.games, args.device)
