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
import math
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

from src.evaluate_agents import run_evaluation_multi_baseline, run_head_to_head, MEAN_IMP_BASELINES
from src.config import load_config

try:
    import src.run_db as run_db
    _RUN_DB_AVAILABLE = True
except Exception:
    _RUN_DB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MEAN_IMP_BASELINES imported from src.evaluate_agents (canonical source)
# Full evaluation set — includes context-only baselines not in mean_imp
ALL_BASELINES = ["random", "greedy"] + list(MEAN_IMP_BASELINES)

STATE_FILENAME = "eval_watcher_state.json"


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Returns (ci_low, ci_high)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


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


def _get_run_id_for_dir(
    db,
    run_dir_path: Path,
    config_path: str,
    checkpoint_path: str = None,
) -> int:
    """Upsert a run record for the given run directory. Returns run_id."""
    run_name = run_dir_path.name
    yaml_text = None
    config_dict = {}
    try:
        with open(config_path, encoding="utf-8") as f:
            yaml_text = f.read()
        try:
            import yaml
            config_dict = yaml.safe_load(yaml_text) or {}
        except Exception:
            pass
    except Exception:
        pass
    # Pass checkpoint filename for ReBeL detection by filename pattern
    algorithm = run_db.infer_algorithm(
        config_dict,
        checkpoint_filename=checkpoint_path,
    )
    return run_db.upsert_run(
        db,
        name=run_name,
        algorithm=algorithm,
        config_yaml=yaml_text,
        config_dict=config_dict,
        status="running",
    )


def evaluate_checkpoint(
    run_dir: str,
    checkpoint_path: str,
    num_games: int,
    config_path: str,
    agent_type: str = "deep_cfr",
) -> dict:
    """Evaluate a single checkpoint, write metrics.jsonl rows and per-game JSONL."""
    run_dir_path = Path(run_dir).resolve()
    checkpoint_path = str(Path(checkpoint_path).resolve())
    iter_num = _infer_iter(checkpoint_path)

    # Per-game JSONL output dir
    eval_dir = run_dir_path / "evaluations" / f"iter_{iter_num}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint for loss info — guard for agent-specific key differences
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    is_rebel = "rebel_value_net_state_dict" in checkpoint or agent_type == "rebel"
    if is_rebel:
        # ReBeL uses value/policy loss histories
        try:
            val_hist = checkpoint.get("value_loss_history", [])
            pol_hist = checkpoint.get("policy_loss_history", [])
            adv_loss = float(val_hist[-1]) if val_hist else float("nan")
            strat_loss = float(pol_hist[-1]) if pol_hist else float("nan")
        except Exception:
            adv_loss = float("nan")
            strat_loss = float("nan")
    else:
        try:
            adv_hist = checkpoint.get("adv_loss_history", [])
            strat_hist = checkpoint.get("strat_loss_history", [])
            adv_loss = float(adv_hist[-1]) if adv_hist else float("nan")
            strat_loss = float(strat_hist[-1]) if strat_hist else float("nan")
        except Exception:
            adv_loss = float("nan")
            strat_loss = float("nan")

    # Run eval once with per-game JSONL output
    all_results = run_evaluation_multi_baseline(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        num_games=num_games,
        baselines=ALL_BASELINES,
        device="cpu",
        output_dir=str(eval_dir),
        agent_type=agent_type,
    )

    # Append metrics.jsonl rows
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics_path = run_dir_path / "metrics.jsonl"
    all_rows = []
    with open(metrics_path, "a", encoding="utf-8") as f:
        for baseline, results in all_results.items():
            p0_wins = results.get("P0 Wins", 0)
            p1_wins = results.get("P1 Wins", 0)
            ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
            total = p0_wins + p1_wins + ties
            win_rate = p0_wins / total if total > 0 else 0.0
            ci_low, ci_high = wilson_ci(p0_wins, total)
            row = {
                "run": run_dir_path.name,
                "iter": iter_num,
                "baseline": baseline,
                "win_rate": round(win_rate, 6),
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "games_played": total,
                "p0_wins": p0_wins,
                "p1_wins": p1_wins,
                "ties": ties,
                "adv_loss": None if adv_loss != adv_loss else round(adv_loss, 6),
                "strat_loss": None if strat_loss != strat_loss else round(strat_loss, 6),
                "timestamp": timestamp,
            }
            stats = getattr(results, 'stats', {})
            row["avg_game_turns"] = round(stats.get("avg_game_turns", 0), 2)
            row["t1_cambia_rate"] = round(stats.get("t1_cambia_rate", 0), 4)
            row["avg_score_margin"] = round(stats.get("avg_score_margin", 0), 2)
            f.write(json.dumps(row) + "\n")
            all_rows.append(row)

    # DB dual-write
    if _RUN_DB_AVAILABLE:
        try:
            db = run_db.get_db()
            db_run_id = _get_run_id_for_dir(db, run_dir_path, config_path)
            ckpt_id = run_db.register_checkpoint(db, db_run_id, iter_num, checkpoint_path)
            for row in all_rows:
                run_db.insert_eval_result(db, db_run_id, ckpt_id, row)
            db.close()
        except Exception:
            logger.debug("DB dual-write failed for %s iter %d", run_dir_path.name, iter_num)

    return all_results, iter_num


def evaluate_head_to_head(
    run_dir: str,
    checkpoint_path: str,
    iter_num: int,
    config_path: str,
    num_games: int = 2000,
    agent_type: str = "deep_cfr",
    checkpoint_prefix: str = "deep_cfr_checkpoint",
) -> None:
    """Pit checkpoint against earlier iterations and write head_to_head.jsonl."""
    if num_games <= 0:
        return

    config = load_config(config_path)
    if not config:
        logger.warning("Could not load config for H2H eval: %s", config_path)
        return

    run_dir_path = Path(run_dir).resolve()
    ckpt_dir = run_dir_path / "checkpoints"
    all_ckpts = sorted(glob(str(ckpt_dir / f"{checkpoint_prefix}_iter_*.pt")))

    if not all_ckpts:
        return

    # Find targets: previous (adjacent), earliest, and closest to T-500
    targets = []

    # Build iter -> path map
    iter_map = {_infer_iter(p): p for p in all_ckpts}
    sorted_iters = sorted(it for it in iter_map if it >= 0)

    # Previous adjacent checkpoint (T vs T-1)
    cur_idx = sorted_iters.index(iter_num) if iter_num in sorted_iters else -1
    if cur_idx > 0:
        prev_iter = sorted_iters[cur_idx - 1]
        targets.append(("previous", iter_map[prev_iter], prev_iter))

    # Earliest checkpoint
    earliest = all_ckpts[0]
    earliest_iter = _infer_iter(earliest)
    if earliest_iter >= 0 and earliest_iter != iter_num:
        # Skip if already covered by "previous"
        if not targets or targets[0][2] != earliest_iter:
            targets.append(("earliest", earliest, earliest_iter))

    # T-500: find closest available checkpoint to iter_num - 500
    target_iter = iter_num - 500
    if target_iter > earliest_iter:
        best_match = min(all_ckpts, key=lambda p: abs(_infer_iter(p) - target_iter))
        best_iter = _infer_iter(best_match)
        if best_iter != iter_num and best_iter != earliest_iter:
            targets.append(("t_minus_500", best_match, best_iter))

    if not targets:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    h2h_path = run_dir_path / "head_to_head.jsonl"

    with open(h2h_path, "a", encoding="utf-8") as f:
        for label, opponent_path, opp_iter in targets:
            logger.info("H2H: iter %d vs %s (iter %d), %d games", iter_num, label, opp_iter, num_games)
            try:
                h2h = run_head_to_head(
                    checkpoint_a=str(Path(checkpoint_path).resolve()),
                    checkpoint_b=str(Path(opponent_path).resolve()),
                    num_games=num_games,
                    config=config,
                    device="cpu",
                    agent_type=agent_type,
                )
                total_scored = h2h["checkpoint_a_wins"] + h2h["checkpoint_b_wins"] + h2h["ties"]
                a_win_rate = h2h["checkpoint_a_wins"] / total_scored if total_scored > 0 else 0.0
                row = {
                    "run": run_dir_path.name,
                    "iter_a": iter_num,
                    "iter_b": opp_iter,
                    "label": label,
                    "a_wins": h2h["checkpoint_a_wins"],
                    "b_wins": h2h["checkpoint_b_wins"],
                    "ties": h2h["ties"],
                    "a_win_rate": round(a_win_rate, 4),
                    "avg_game_turns": round(h2h["avg_game_turns"], 2),
                    "timestamp": timestamp,
                }
                f.write(json.dumps(row) + "\n")
                logger.info(
                    "H2H iter %d vs %d (%s): WR=%.1f%%",
                    iter_num, opp_iter, label, a_win_rate * 100,
                )
                if _RUN_DB_AVAILABLE:
                    try:
                        _db = run_db.get_db()
                        _run_id = _get_run_id_for_dir(_db, run_dir_path, config_path)
                        run_db.insert_head_to_head(_db, _run_id, row)
                        _db.close()
                    except Exception:
                        logger.debug("DB H2H write failed for %s iter %d", run_dir_path.name, iter_num)
            except Exception:
                logger.exception("H2H eval failed: iter %d vs %s", iter_num, label)


def format_results(run_dir: str, iter_num: int, all_results: dict) -> str:
    run_name = Path(run_dir).name
    ts = datetime.now().strftime("%H:%M:%S")
    parts = []
    for baseline, results in all_results.items():
        p0_wins = results.get("P0 Wins", 0)
        total = p0_wins + results.get("P1 Wins", 0) + results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        win_rate = p0_wins / total if total > 0 else 0.0
        stats = getattr(results, 'stats', {})
        t1_cambia = stats.get("t1_cambia_rate")
        suffix = f" [t1_cambia={t1_cambia:.3f}]" if t1_cambia is not None else ""
        parts.append(f"{baseline}={win_rate:.2f}{suffix}")
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
    parser.add_argument(
        "--h2h-games",
        type=int,
        default=2000,
        help="Games per head-to-head comparison (default: 2000, 0 to disable).",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="deep_cfr",
        help=(
            "Agent type to evaluate (default: deep_cfr). "
            "Choices: deep_cfr, rebel, sd_cfr, escher, nplayer, ppo."
        ),
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=None,
        help=(
            "Filename prefix for checkpoint glob (default: inferred from --agent-type). "
            "E.g. 'rebel_checkpoint' for ReBeL, 'deep_cfr_checkpoint' for Deep CFR."
        ),
    )
    args = parser.parse_args()

    # Infer checkpoint prefix from agent type if not explicitly provided
    _PREFIX_MAP = {
        "rebel": "rebel_checkpoint",
        "deep_cfr": "deep_cfr_checkpoint",
        "sd_cfr": "deep_cfr_checkpoint",
        "escher": "deep_cfr_checkpoint",
        "nplayer": "deep_cfr_checkpoint",
        "ppo": "ppo_checkpoint",
    }
    checkpoint_prefix = args.checkpoint_prefix or _PREFIX_MAP.get(args.agent_type, "deep_cfr_checkpoint")

    run_dirs = [str(Path(d).resolve()) for d in args.run_dirs]
    logger.info("Watching %d run dir(s): %s", len(run_dirs), run_dirs)
    logger.info(
        "Agent type: %s, checkpoint prefix: %s, games: %d, poll interval: %ds",
        args.agent_type, checkpoint_prefix, args.games, args.poll_interval,
    )

    try:
        while True:
            for run_dir in run_dirs:
                config_path = os.path.join(run_dir, "config.yaml")
                if not os.path.exists(config_path):
                    logger.warning("No config.yaml in %s, skipping.", run_dir)
                    continue

                state = load_state(run_dir)
                ckpt_pattern = os.path.join(
                    run_dir, "checkpoints", f"{checkpoint_prefix}_iter_*.pt"
                )
                checkpoints = sorted(glob(ckpt_pattern))

                for ckpt in checkpoints:
                    filename = os.path.basename(ckpt)
                    if filename in state["evaluated"]:
                        continue

                    logger.info("New checkpoint: %s", ckpt)
                    try:
                        all_results, iter_num = evaluate_checkpoint(
                            run_dir, ckpt, args.games, config_path,
                            agent_type=args.agent_type,
                        )
                        log_line = format_results(run_dir, iter_num, all_results)
                        print(log_line, flush=True)
                        state["evaluated"].append(filename)
                        save_state(run_dir, state)
                        # Head-to-head cross-iteration eval
                        if args.h2h_games > 0:
                            try:
                                evaluate_head_to_head(
                                    run_dir, ckpt, iter_num, config_path, args.h2h_games,
                                    agent_type=args.agent_type,
                                    checkpoint_prefix=checkpoint_prefix,
                                )
                            except Exception:
                                logger.exception("H2H evaluation failed for iter %d", iter_num)
                        # Write DB summary outputs after all evals complete
                        if _RUN_DB_AVAILABLE:
                            try:
                                _db = run_db.get_db()
                                _run_id = _get_run_id_for_dir(
                                    _db, Path(run_dir).resolve(), config_path
                                )
                                run_db.write_run_meta_json(_db, _run_id, run_dir)
                                run_db.write_eval_summary_jsonl(_db, _run_id, run_dir)
                                _db.close()
                            except Exception:
                                logger.debug("DB summary write failed for %s", run_dir)
                    except Exception:
                        logger.exception("Failed to evaluate %s", ckpt)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving state and exiting.")
        # State already saved after each successful eval; nothing more to do.
        sys.exit(0)


if __name__ == "__main__":
    main()
