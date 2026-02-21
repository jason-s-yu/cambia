#!/usr/bin/env python3
"""
Autonomous validation runner for Phase 1 training experiments.

Monitors 7 training runs for iteration-specific checkpoints, evaluates each
against 5 baselines, writes JSONL per-game logs, and tracks performance overhead.

Usage:
    python validation_runner.py [--poll-interval 30] [--dry-run]

Runs autonomously until all target evaluations complete or all training runs finish.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("runs/validation_runner.log"),
    ],
)
logger = logging.getLogger(__name__)

# === Configuration ===

RUNS = ["os-full", "os-30", "os-20", "os-15", "os-10", "es-15", "es-10"]
TARGET_ITERATIONS = [25, 50, 75, 250, 500, 750]
BASELINES = ["random", "greedy", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]
GAMES_INTERMEDIATE = 400
GAMES_FINAL = 500
MAX_ITERATION = 750


def checkpoint_iter_path(run: str, iteration: int) -> str:
    return f"runs/{run}/checkpoints/deep_cfr_checkpoint_iter_{iteration}.pt"


def eval_output_dir(run: str, iteration: int) -> str:
    return f"runs/{run}/evaluations/iter_{iteration}"


def is_training_done(run: str) -> bool:
    """Check if training has completed (final checkpoint exists or process exited)."""
    final_path = checkpoint_iter_path(run, MAX_ITERATION)
    return os.path.exists(final_path)


def get_latest_training_step(run: str) -> int:
    """Get the highest iteration checkpoint that exists for a run."""
    checkpoint_dir = f"runs/{run}/checkpoints"
    if not os.path.isdir(checkpoint_dir):
        return 0
    best = 0
    for f in os.listdir(checkpoint_dir):
        if f.startswith("deep_cfr_checkpoint_iter_") and f.endswith(".pt"):
            try:
                step = int(f.replace("deep_cfr_checkpoint_iter_", "").replace(".pt", ""))
                best = max(best, step)
            except ValueError:
                continue
    return best


def run_evaluation(run: str, iteration: int, dry_run: bool = False) -> dict:
    """Run evaluation for a checkpoint against all baselines."""
    ckpt_path = checkpoint_iter_path(run, iteration)
    output_dir = eval_output_dir(run, iteration)
    os.makedirs(output_dir, exist_ok=True)

    n_games = GAMES_FINAL if iteration == MAX_ITERATION else GAMES_INTERMEDIATE
    baselines_str = ",".join(BASELINES)

    config_path = f"runs/{run}/config.yaml"
    cmd = [
        sys.executable, "-m", "cambia", "evaluate",
        ckpt_path,
        "--config", config_path,
        "--games", str(n_games),
        "--baselines", baselines_str,
        "--output-dir", output_dir,
    ]

    logger.info("Evaluating %s iter %d: %d games x %d baselines", run, iteration, n_games, len(BASELINES))

    if dry_run:
        logger.info("  [DRY RUN] Would run: %s", " ".join(cmd))
        return {"run": run, "iteration": iteration, "status": "dry_run"}

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per evaluation
            cwd=str(Path(__file__).parent),
        )
        elapsed = time.perf_counter() - start

        eval_result = {
            "run": run,
            "iteration": iteration,
            "n_games": n_games,
            "n_baselines": len(BASELINES),
            "elapsed_sec": round(elapsed, 1),
            "returncode": result.returncode,
            "timestamp": datetime.now().isoformat(),
        }

        if result.returncode == 0:
            logger.info("  Completed in %.1fs (exit 0)", elapsed)
            eval_result["status"] = "success"
        else:
            logger.warning("  Failed (exit %d) in %.1fs", result.returncode, elapsed)
            logger.warning("  stderr: %s", result.stderr[-500:] if result.stderr else "")
            eval_result["status"] = "failed"
            eval_result["stderr_tail"] = result.stderr[-500:] if result.stderr else ""

        # Save stdout for parsing
        stdout_path = os.path.join(output_dir, "eval_stdout.txt")
        with open(stdout_path, "w") as f:
            f.write(result.stdout or "")

        return eval_result

    except subprocess.TimeoutExpired:
        logger.error("  Evaluation timed out after 30 minutes")
        return {"run": run, "iteration": iteration, "status": "timeout"}
    except Exception as e:
        logger.error("  Evaluation error: %s", e)
        return {"run": run, "iteration": iteration, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 validation runner")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polls")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    logger.info("=== Phase 1 Validation Runner ===")
    logger.info("Runs: %s", RUNS)
    logger.info("Target iterations: %s", TARGET_ITERATIONS)
    logger.info("Baselines: %s", BASELINES)
    logger.info("Games: %d intermediate, %d final", GAMES_INTERMEDIATE, GAMES_FINAL)
    logger.info("Poll interval: %ds", args.poll_interval)

    # Track evaluated (run, iteration) pairs
    evaluated = set()
    results = []

    # Check for already-existing checkpoints (resuming)
    for run in RUNS:
        for iteration in TARGET_ITERATIONS:
            output_dir = eval_output_dir(run, iteration)
            if os.path.exists(os.path.join(output_dir, "eval_stdout.txt")):
                evaluated.add((run, iteration))
                logger.info("Skipping %s iter %d — already evaluated", run, iteration)

    total_evals = len(RUNS) * len(TARGET_ITERATIONS)
    logger.info("Pending evaluations: %d / %d", total_evals - len(evaluated), total_evals)

    stale_count = 0
    max_stale = 120  # After 60 minutes of no new checkpoints, consider training stalled

    while len(evaluated) < total_evals:
        found_new = False

        for run in RUNS:
            for iteration in TARGET_ITERATIONS:
                if (run, iteration) in evaluated:
                    continue

                ckpt_path = checkpoint_iter_path(run, iteration)
                if os.path.exists(ckpt_path):
                    found_new = True
                    result = run_evaluation(run, iteration, dry_run=args.dry_run)
                    results.append(result)
                    evaluated.add((run, iteration))

                    # Write running results summary
                    summary_path = "runs/validation_results.jsonl"
                    with open(summary_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

        # Progress report
        done = len(evaluated)
        if found_new:
            stale_count = 0
            logger.info("Progress: %d / %d evaluations complete", done, total_evals)

            # Per-run status
            for run in RUNS:
                run_done = sum(1 for it in TARGET_ITERATIONS if (run, it) in evaluated)
                latest = get_latest_training_step(run)
                logger.info("  %s: %d/%d evals, latest step=%d", run, run_done, len(TARGET_ITERATIONS), latest)
        else:
            stale_count += 1
            if stale_count % 20 == 0:  # Every 10 minutes
                logger.info("Waiting for checkpoints... (%d/%d done, %d polls without new)",
                           done, total_evals, stale_count)
                for run in RUNS:
                    latest = get_latest_training_step(run)
                    run_done = sum(1 for it in TARGET_ITERATIONS if (run, it) in evaluated)
                    logger.info("  %s: step=%d, %d/%d evals", run, latest, run_done, len(TARGET_ITERATIONS))

        # Check if all OS runs are done (ES may take much longer)
        os_runs = [r for r in RUNS if r.startswith("os-")]
        os_all_done = all(is_training_done(r) for r in os_runs)
        os_all_evaluated = all((r, it) in evaluated for r in os_runs for it in TARGET_ITERATIONS)
        if os_all_done and os_all_evaluated:
            es_remaining = sum(1 for r in RUNS if r.startswith("es-")
                             for it in TARGET_ITERATIONS if (r, it) not in evaluated)
            if es_remaining > 0:
                logger.info("All OS runs evaluated. %d ES evaluations remaining.", es_remaining)

        if stale_count >= max_stale:
            logger.warning("No new checkpoints for %d polls. Training may have stalled.",
                          max_stale)
            # Don't exit — ES runs can take 10+ hours
            stale_count = 0  # Reset to avoid repeated warnings

        time.sleep(args.poll_interval)

    logger.info("=== All %d evaluations complete ===", total_evals)

    # Final summary
    logger.info("Results summary:")
    for run in RUNS:
        run_results = [r for r in results if r.get("run") == run]
        successes = sum(1 for r in run_results if r.get("status") == "success")
        failures = sum(1 for r in run_results if r.get("status") != "success")
        logger.info("  %s: %d success, %d failed", run, successes, failures)


if __name__ == "__main__":
    main()
