#!/usr/bin/env python3
"""
scripts/eval_watcher.py

Polling script that auto-evaluates every new checkpoint found in one or more run dirs.
Runs CPU-only to avoid GPU contention with training.

Usage:
    python cfr/scripts/eval_watcher.py \
        --run-dirs cfr/runs/prod-full-333 cfr/runs/prod-full-500 \
        --games 5000 --poll-interval 30
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

# Ensure cfr/ is importable when run directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
if str(_CFR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CFR_ROOT))

import torch

from src.evaluate_agents import run_evaluation_multi_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_BASELINES = [
    "random",
    "greedy",
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
]

STATE_FILENAME = "eval_watcher_state.json"


def load_state(run_dir: str) -> dict:
    state_path = Path(run_dir) / STATE_FILENAME
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read state file %s, starting fresh.", state_path)
    return {"evaluated": []}


def save_state(run_dir: str, state: dict) -> None:
    state_path = Path(run_dir) / STATE_FILENAME
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _infer_iter(checkpoint_path: str) -> int:
    match = re.search(r"_iter_(\d+)\.pt$", checkpoint_path)
    return int(match.group(1)) if match else -1


def evaluate_checkpoint(
    run_dir: str,
    checkpoint_path: str,
    num_games: int,
    config_path: str,
) -> dict:
    """Evaluate a single checkpoint, write metrics.jsonl rows and per-game JSONL."""
    run_dir_path = Path(run_dir).resolve()
    checkpoint_path = str(Path(checkpoint_path).resolve())
    iter_num = _infer_iter(checkpoint_path)

    # Per-game JSONL output dir
    eval_dir = run_dir_path / "evaluations" / f"iter_{iter_num}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint for loss info
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adv_hist = checkpoint.get("adv_loss_history", [])
    strat_hist = checkpoint.get("strat_loss_history", [])
    adv_loss = float(adv_hist[-1]) if adv_hist else float("nan")
    strat_loss = float(strat_hist[-1]) if strat_hist else float("nan")

    # Run eval once with per-game JSONL output
    all_results = run_evaluation_multi_baseline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        num_games=num_games,
        baselines=ALL_BASELINES,
        device="cpu",
        output_dir=str(eval_dir),
    )

    # Append metrics.jsonl rows
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics_path = run_dir_path / "metrics.jsonl"
    with open(metrics_path, "a", encoding="utf-8") as f:
        for baseline, results in all_results.items():
            p0_wins = results.get("P0 Wins", 0)
            p1_wins = results.get("P1 Wins", 0)
            ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
            total = p0_wins + p1_wins + ties
            win_rate = p0_wins / total if total > 0 else 0.0
            row = {
                "run": run_dir_path.name,
                "iter": iter_num,
                "baseline": baseline,
                "win_rate": round(win_rate, 6),
                "games_played": total,
                "p0_wins": p0_wins,
                "p1_wins": p1_wins,
                "ties": ties,
                "adv_loss": None if adv_loss != adv_loss else round(adv_loss, 6),
                "strat_loss": None if strat_loss != strat_loss else round(strat_loss, 6),
                "timestamp": timestamp,
            }
            f.write(json.dumps(row) + "\n")

    return all_results, iter_num


def format_results(run_dir: str, iter_num: int, all_results: dict) -> str:
    run_name = Path(run_dir).name
    ts = datetime.now().strftime("%H:%M:%S")
    parts = []
    for baseline, results in all_results.items():
        p0_wins = results.get("P0 Wins", 0)
        total = p0_wins + results.get("P1 Wins", 0) + results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        win_rate = p0_wins / total if total > 0 else 0.0
        parts.append(f"{baseline}={win_rate:.2f}")
    return f"[{ts}] Evaluated {run_name} iter {iter_num}: {', '.join(parts)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll run dirs and auto-evaluate new checkpoints (CPU-only)."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories to watch.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5000,
        help="Games per baseline per checkpoint (default: 5000).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polling cycles (default: 30).",
    )
    args = parser.parse_args()

    run_dirs = [str(Path(d).resolve()) for d in args.run_dirs]
    logger.info("Watching %d run dir(s): %s", len(run_dirs), run_dirs)
    logger.info("Games per eval: %d, poll interval: %ds", args.games, args.poll_interval)

    try:
        while True:
            for run_dir in run_dirs:
                config_path = os.path.join(run_dir, "config.yaml")
                if not os.path.exists(config_path):
                    logger.warning("No config.yaml in %s, skipping.", run_dir)
                    continue

                state = load_state(run_dir)
                ckpt_pattern = os.path.join(run_dir, "checkpoints", "deep_cfr_checkpoint_iter_*.pt")
                checkpoints = sorted(glob(ckpt_pattern))

                for ckpt in checkpoints:
                    filename = os.path.basename(ckpt)
                    if filename in state["evaluated"]:
                        continue

                    logger.info("New checkpoint: %s", ckpt)
                    try:
                        all_results, iter_num = evaluate_checkpoint(
                            run_dir, ckpt, args.games, config_path
                        )
                        log_line = format_results(run_dir, iter_num, all_results)
                        print(log_line, flush=True)
                        state["evaluated"].append(filename)
                        save_state(run_dir, state)
                    except Exception:
                        logger.exception("Failed to evaluate %s", ckpt)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving state and exiting.")
        # State already saved after each successful eval; nothing more to do.
        sys.exit(0)


if __name__ == "__main__":
    main()
