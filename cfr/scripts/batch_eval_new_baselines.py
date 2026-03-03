#!/usr/bin/env python3
"""Batch evaluate new random baselines (random_no_cambia, random_late_cambia) against
all key checkpoints referenced in the review request."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate_agents import run_evaluation_multi_baseline

NEW_BASELINES = ["random_no_cambia", "random_late_cambia"]
GAMES = 5000

# All checkpoints referenced in the review request
EVALS = [
    # Phase 1b Exp A (interleaved+ResNet, best result)
    {
        "config": "runs/ablation-interleaved-resnet/config.yaml",
        "checkpoints": "runs/ablation-interleaved-resnet/checkpoints",
        "iters": [25, 50, 75, 100, 125, 150, 175, 200],
        "run_name": "ablation-interleaved-resnet",
    },
    # De-aliased flat ablation
    {
        "config": "runs/ablation-dealiased-flat/config.yaml",
        "checkpoints": "runs/ablation-dealiased-flat/checkpoints",
        "iters": [25, 50, 75, 100, 125, 150, 175, 200],
        "run_name": "ablation-dealiased-flat",
    },
    # prod-full-333 (legacy OS-dCFR)
    {
        "config": "runs/prod-full-333/config.yaml",
        "checkpoints": "runs/prod-full-333/checkpoints",
        "iters": [1075],
        "run_name": "prod-full-333",
    },
    # sd-cfr-500k (legacy SD-CFR) — closest to iter 500 is 550
    {
        "config": "runs/sd-cfr-500k/config.yaml",
        "checkpoints": "runs/sd-cfr-500k/checkpoints",
        "iters": [550],
        "run_name": "sd-cfr-500k",
    },
    # eppbs-2p (flat EP-PBS, 1500 iter run) — key milestones
    {
        "config": "runs/eppbs-2p/config.yaml",
        "checkpoints": "runs/eppbs-2p/checkpoints",
        "iters": [100, 200, 400, 600, 800, 950, 1000, 1200, 1500],
        "run_name": "eppbs-2p",
    },
]

OUTPUT_FILE = Path("runs/new_baseline_evals.jsonl")


def find_checkpoint(ckpt_dir: str, iter_num: int) -> str | None:
    d = Path(ckpt_dir)
    candidates = [
        d / f"deep_cfr_checkpoint_iter_{iter_num}.pt",
        d / f"checkpoint_iter_{iter_num}.pt",
        d / f"sd_cfr_checkpoint_iter_{iter_num}.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Glob fallback
    matches = list(d.glob(f"*iter_{iter_num}.pt"))
    return str(matches[0]) if matches else None


def main():
    total = sum(len(e["iters"]) for e in EVALS)
    done = 0
    print(f"Evaluating {total} checkpoints x {len(NEW_BASELINES)} baselines x {GAMES} games")

    with open(OUTPUT_FILE, "a") as f:
        for eval_spec in EVALS:
            for it in eval_spec["iters"]:
                ckpt = find_checkpoint(eval_spec["checkpoints"], it)
                if ckpt is None:
                    print(f"  SKIP {eval_spec['run_name']} iter {it}: checkpoint not found")
                    continue

                t0 = time.time()
                print(f"  [{done+1}/{total}] {eval_spec['run_name']} iter {it}...", end="", flush=True)

                results = run_evaluation_multi_baseline(
                    config_path=eval_spec["config"],
                    checkpoint_path=ckpt,
                    num_games=GAMES,
                    baselines=NEW_BASELINES,
                    device="cpu",
                )

                for bl, res in results.items():
                    p0 = res.get("P0 Wins", 0)
                    p1 = res.get("P1 Wins", 0)
                    ties = res.get("Ties", 0) + res.get("MaxTurnTies", 0)
                    total_games = p0 + p1 + ties
                    wr = p0 / total_games if total_games > 0 else 0.0
                    row = {
                        "run": eval_spec["run_name"],
                        "iter": it,
                        "baseline": bl,
                        "win_rate": round(wr, 6),
                        "games_played": total_games,
                        "p0_wins": p0,
                        "p1_wins": p1,
                        "ties": ties,
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()

                elapsed = time.time() - t0
                # Print summary
                parts = []
                for bl, res in results.items():
                    p0 = res.get("P0 Wins", 0)
                    total_games = p0 + res.get("P1 Wins", 0) + res.get("Ties", 0) + res.get("MaxTurnTies", 0)
                    parts.append(f"{bl}={p0/total_games:.3f}")
                done += 1
                print(f" {', '.join(parts)} ({elapsed:.1f}s)")

    print(f"\nDone. Results appended to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
