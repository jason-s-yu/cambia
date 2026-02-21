"""
scripts/collect_metrics.py

Collect evaluation metrics for a Deep CFR checkpoint against all 5 baseline agents.

Usage:
    python scripts/collect_metrics.py --run-dir cfr/runs/os-20 \
        --checkpoint cfr/runs/os-20/checkpoints/deep_cfr_checkpoint_iter_50.pt \
        [--games 2000] [--config cfr/runs/os-20/config.yaml]

Writes one JSONL row per (checkpoint, baseline) pair to <run_dir>/metrics.jsonl.
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure cfr/src is importable when run directly.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_SRC = _SCRIPT_DIR.parent / "src"
if str(_CFR_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_CFR_SRC.parent))

import torch

from src.evaluate_agents import run_evaluation_multi_baseline
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_BASELINES = [
    "random",
    "greedy",
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
]


def _infer_iter_from_path(checkpoint_path: str) -> int:
    """Extract iteration number from checkpoint filename, e.g. deep_cfr_checkpoint_iter_50.pt -> 50."""
    match = re.search(r"_iter_(\d+)\.pt$", checkpoint_path)
    if match:
        return int(match.group(1))
    # Fall back to reading from checkpoint metadata.
    return -1


def _extract_loss_history(checkpoint: dict) -> tuple[float, float]:
    """
    Extract the most recent adv_loss and strat_loss from checkpoint loss history lists.

    Returns (adv_loss, strat_loss) as floats, or NaN if not available.
    """
    adv_loss = float("nan")
    strat_loss = float("nan")

    adv_hist = checkpoint.get("adv_loss_history", [])
    if adv_hist:
        adv_loss = float(adv_hist[-1])

    strat_hist = checkpoint.get("strat_loss_history", [])
    if strat_hist:
        strat_loss = float(strat_hist[-1])

    return adv_loss, strat_loss


def collect_metrics(
    run_dir: str,
    checkpoint_path: str,
    num_games: int,
    config_path: str,
    device: str = "cpu",
) -> None:
    """
    Run evaluation against all baselines and append JSONL rows to <run_dir>/metrics.jsonl.
    """
    run_dir_path = Path(run_dir).resolve()
    run_name = run_dir_path.name

    checkpoint_path = str(Path(checkpoint_path).resolve())

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Determine iteration number.
    iter_num = checkpoint.get("training_step", _infer_iter_from_path(checkpoint_path))
    if iter_num == -1:
        logger.warning("Could not determine iteration number from checkpoint path.")

    adv_loss, strat_loss = _extract_loss_history(checkpoint)

    logger.info(
        "Checkpoint: run=%s, iter=%s, adv_loss=%.4f, strat_loss=%.4f",
        run_name,
        iter_num,
        adv_loss,
        strat_loss,
    )

    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load config from %s", config_path)
        sys.exit(1)

    # Run evaluation against all baselines.
    all_results = run_evaluation_multi_baseline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        num_games=num_games,
        baselines=ALL_BASELINES,
        device=device,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics_path = run_dir_path / "metrics.jsonl"

    rows_written = 0
    with open(metrics_path, "a", encoding="utf-8") as f:
        for baseline, results in all_results.items():
            p0_wins = results.get("P0 Wins", 0)
            p1_wins = results.get("P1 Wins", 0)
            ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
            total_played = p0_wins + p1_wins + ties
            win_rate = p0_wins / total_played if total_played > 0 else 0.0

            row = {
                "run": run_name,
                "iter": iter_num,
                "baseline": baseline,
                "win_rate": round(win_rate, 6),
                "games_played": total_played,
                "p0_wins": p0_wins,
                "p1_wins": p1_wins,
                "ties": ties,
                "avg_game_turns": None,  # not tracked at this level; use per-game JSONL for this
                "adv_loss": None if adv_loss != adv_loss else round(adv_loss, 6),  # nan check
                "strat_loss": None if strat_loss != strat_loss else round(strat_loss, 6),
                "timestamp": timestamp,
            }
            f.write(json.dumps(row) + "\n")
            rows_written += 1
            logger.info(
                "  %-20s win_rate=%.3f  (%d/%d)",
                baseline,
                win_rate,
                p0_wins,
                total_played,
            )

    logger.info("Wrote %d rows to %s", rows_written, metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Deep CFR checkpoint evaluation metrics."
    )
    parser.add_argument("--run-dir", required=True, help="Path to run directory (e.g. cfr/runs/os-20)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--games", type=int, default=2000, help="Games per baseline (default: 2000)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to <run-dir>/config.yaml",
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device string (default: cpu)"
    )

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = str(Path(args.run_dir).resolve() / "config.yaml")

    collect_metrics(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        config_path=config_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
