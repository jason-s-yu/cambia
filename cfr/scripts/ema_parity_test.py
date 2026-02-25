#!/usr/bin/env python3
"""
EMA Parity Test (Phase 1.2)

Validates that EMA serving weights produce equivalent play to full snapshot
averaging by running head-to-head games between:
  Agent A: SDCFRAgentWrapper(use_ema=True)  — O(1) EMA inference
  Agent B: SDCFRAgentWrapper(use_ema=False) — full snapshot averaging

Target: 50% +/- 1.5% win rate over 5,000 games.
"""

import argparse
import logging
import os
import sys

# Add cfr/ to path so `from src.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config
from src.evaluate_agents import SDCFRAgentWrapper
from src.game.engine import CambiaGameState

logger = logging.getLogger(__name__)


def run_ema_parity(
    checkpoint_path: str,
    config_path: str,
    num_games: int = 5000,
    device: str = "cpu",
) -> dict:
    """Run EMA vs snapshot-averaging head-to-head."""
    config = load_config(config_path)

    # Verify _ema.pt file exists
    base_path = os.path.splitext(checkpoint_path)[0]
    ema_path = f"{base_path}_ema.pt"
    if not os.path.exists(ema_path):
        print(f"FATAL: EMA file not found: {ema_path}")
        sys.exit(1)
    print(f"EMA file found: {ema_path}")

    # Verify snapshots file exists (needed for non-EMA agent)
    snap_path = f"{base_path}_sd_snapshots.pt"
    if not os.path.exists(snap_path):
        print(f"FATAL: SD-CFR snapshots file not found: {snap_path}")
        sys.exit(1)
    print(f"Snapshots file found: {snap_path}")

    wins_ema = 0
    wins_snap = 0
    draws = 0
    errors = 0

    for game_num in range(1, num_games + 1):
        ema_is_p0 = game_num % 2 == 1

        try:
            if ema_is_p0:
                agent0 = SDCFRAgentWrapper(0, config, checkpoint_path, device=device, use_ema=True)
                agent1 = SDCFRAgentWrapper(1, config, checkpoint_path, device=device, use_ema=False)
            else:
                agent0 = SDCFRAgentWrapper(0, config, checkpoint_path, device=device, use_ema=False)
                agent1 = SDCFRAgentWrapper(1, config, checkpoint_path, device=device, use_ema=True)

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
                legal_actions = game_state.get_legal_actions()
                if not legal_actions:
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

            if game_state.is_terminal():
                winner = game_state._winner
                if winner is None:
                    draws += 1
                elif ema_is_p0:
                    if winner == 0:
                        wins_ema += 1
                    else:
                        wins_snap += 1
                else:
                    if winner == 1:
                        wins_ema += 1
                    else:
                        wins_snap += 1
            else:
                errors += 1

        except Exception as e:
            logger.warning("Game %d error: %s", game_num, e)
            errors += 1

        if game_num % 500 == 0:
            decided = wins_ema + wins_snap + draws
            wr = wins_ema / decided * 100 if decided > 0 else 0
            print(f"[{game_num}/{num_games}] EMA WR: {wr:.1f}% ({wins_ema}W/{wins_snap}L/{draws}D, {errors} errors)")

    decided = wins_ema + wins_snap + draws
    ema_wr = wins_ema / decided * 100 if decided > 0 else 0

    result = {
        "wins_ema": wins_ema,
        "wins_snap": wins_snap,
        "draws": draws,
        "errors": errors,
        "decided_games": decided,
        "ema_win_rate_pct": ema_wr,
    }

    print("\n=== EMA Parity Test Results ===")
    print(f"Games: {num_games} ({decided} decided, {errors} errors)")
    print(f"EMA wins: {wins_ema}, Snapshot wins: {wins_snap}, Draws: {draws}")
    print(f"EMA win rate: {ema_wr:.2f}%")

    if abs(ema_wr - 50.0) <= 1.5:
        print("PASS: EMA parity within 50% +/- 1.5%")
    else:
        print(f"FAIL: EMA win rate {ema_wr:.2f}% outside tolerance (48.5-51.5%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="EMA Parity Test (Phase 1.2)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/sys-bench/checkpoints/deep_cfr_checkpoint.pt",
        help="Path to checkpoint (must have companion _ema.pt and _sd_snapshots.pt files)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sys_bench.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=5000,
        help="Number of head-to-head games",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu/xpu/auto)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_ema_parity(args.checkpoint, args.config, args.num_games, args.device)


if __name__ == "__main__":
    main()
