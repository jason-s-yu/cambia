"""R0: Compare EMA vs raw checkpoint weights against baselines.

Phase 2 iter checkpoints contain both:
  - advantage_net_state_dict: raw network weights at that iteration
  - ema_state_dict: t^1.5-weighted EMA of all snapshots up to that iteration

Existing eval_summary.jsonl data used raw weights (DeepCFRAgentWrapper).
This script evaluates EMA weights from the same checkpoints for comparison.

Usage:
    cd cfr
    python scripts/r0_ema_vs_raw_eval.py [--iters 50 100 200 300 450 600] [--games 2000]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CFR_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_CFR_ROOT / "src"))
os.chdir(str(_CFR_ROOT))

import torch
from src.config import load_config
from src.evaluate_agents import (
    MEAN_IMP_BASELINES,
    run_evaluation,
)

RUN_DIR = Path("runs/interleaved-resnet-adaptive")
CKPT_DIR = RUN_DIR / "checkpoints"
CONFIG_PATH = str(RUN_DIR / "config.yaml")

DEFAULT_ITERS = [50, 100, 200, 300, 450, 600]
DEFAULT_GAMES = 2000  # 1000 per seat


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def eval_with_weights(
    weight_source: str,
    checkpoint_path: str,
    config_path: str,
    baselines: list,
    games: int,
    ema_override_state: dict = None,
):
    """Evaluate a checkpoint against baselines.

    If ema_override_state is provided, temporarily patches the checkpoint
    so that advantage_net_state_dict = ema_state_dict, then evaluates
    using the standard deep_cfr agent type (which loads advantage_net_state_dict).
    """
    if ema_override_state is not None:
        # Create temp checkpoint with EMA weights as the advantage net
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        ckpt["advantage_net_state_dict"] = ema_override_state
        tmp_path = checkpoint_path + ".ema_tmp.pt"
        torch.save(ckpt, tmp_path)
        eval_path = tmp_path
    else:
        eval_path = checkpoint_path
        tmp_path = None

    results = {}
    half = games // 2
    try:
        for bl in baselines:
            c1 = run_evaluation(
                config_path=config_path,
                agent1_type="deep_cfr",
                agent2_type=bl,
                num_games=half,
                strategy_path=None,
                checkpoint_path=eval_path,
                device="cpu",
            )
            c2 = run_evaluation(
                config_path=config_path,
                agent1_type=bl,
                agent2_type="deep_cfr",
                num_games=half,
                strategy_path=None,
                checkpoint_path=eval_path,
                device="cpu",
            )
            model_wins = c1.get("P0 Wins", 0) + c2.get("P1 Wins", 0)
            bl_wins = c1.get("P1 Wins", 0) + c2.get("P0 Wins", 0)
            decided = model_wins + bl_wins
            wr, ci_lo, ci_hi = wilson_ci(model_wins, decided) if decided else (0, 0, 0)

            results[bl] = {
                "win_rate": round(wr, 4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "model_wins": model_wins,
                "bl_wins": bl_wins,
                "decided": decided,
            }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return results


def load_existing_raw_evals(run_dir: Path) -> dict:
    """Load existing eval_summary.jsonl (raw-weight evaluations)."""
    path = run_dir / "eval_summary.jsonl"
    if not path.exists():
        return {}
    raw_data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            raw_data[rec["iter"]] = rec.get("baselines", {})
    return raw_data


def mean_imp(baseline_results: dict) -> float:
    """Compute mean_imp from baseline win rates."""
    vals = [baseline_results[bl]["win_rate"] for bl in MEAN_IMP_BASELINES if bl in baseline_results]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    parser = argparse.ArgumentParser(description="R0: EMA vs raw checkpoint evaluation")
    parser.add_argument("--iters", nargs="+", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="Games per baseline (seat-balanced)")
    parser.add_argument("--baselines", nargs="+", default=MEAN_IMP_BASELINES)
    parser.add_argument("--output", default=str(RUN_DIR / "r0_ema_vs_raw.json"))
    parser.add_argument("--skip-raw", action="store_true",
                        help="Skip raw eval (use existing eval_summary.jsonl data)")
    args = parser.parse_args()

    print(f"R0: EMA vs Raw evaluation — {len(args.iters)} checkpoints × "
          f"{len(args.baselines)} baselines × {args.games} games")
    print(f"Config: {CONFIG_PATH}")
    print(f"Checkpoints: {CKPT_DIR}")
    print()

    # Load existing raw eval data
    existing_raw = load_existing_raw_evals(RUN_DIR)

    all_results = {}
    t0 = time.perf_counter()

    for it in args.iters:
        ckpt_path = CKPT_DIR / f"deep_cfr_checkpoint_iter_{it}.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] iter {it}: checkpoint not found")
            continue

        print(f"--- iter {it} ---")

        # Load checkpoint to extract EMA weights
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        ema_state = ckpt.get("ema_state_dict")
        if ema_state is None:
            print(f"  [SKIP] No ema_state_dict in checkpoint")
            continue

        # EMA evaluation
        print(f"  Evaluating EMA weights ({args.games} games × {len(args.baselines)} baselines)...")
        mt = time.perf_counter()
        ema_results = eval_with_weights(
            "ema", str(ckpt_path), CONFIG_PATH, args.baselines, args.games,
            ema_override_state=ema_state,
        )
        ema_time = time.perf_counter() - mt
        ema_mi = mean_imp(ema_results)
        print(f"  EMA  mean_imp = {ema_mi*100:.1f}%  ({ema_time:.0f}s)")

        # Raw evaluation — use existing data or re-run
        if args.skip_raw and it in existing_raw:
            raw_results = existing_raw[it]
            raw_mi = mean_imp(raw_results)
            print(f"  RAW  mean_imp = {raw_mi*100:.1f}%  (from eval_summary.jsonl)")
        else:
            print(f"  Evaluating RAW weights ({args.games} games × {len(args.baselines)} baselines)...")
            mt = time.perf_counter()
            raw_results = eval_with_weights(
                "raw", str(ckpt_path), CONFIG_PATH, args.baselines, args.games,
            )
            raw_time = time.perf_counter() - mt
            raw_mi = mean_imp(raw_results)
            print(f"  RAW  mean_imp = {raw_mi*100:.1f}%  ({raw_time:.0f}s)")

        delta = ema_mi - raw_mi
        print(f"  DELTA (ema - raw) = {delta*100:+.1f}pp")

        all_results[str(it)] = {
            "ema": ema_results,
            "raw": raw_results,
            "ema_mean_imp": round(ema_mi, 4),
            "raw_mean_imp": round(raw_mi, 4),
            "delta_pp": round(delta * 100, 2),
        }
        print()

    total = time.perf_counter() - t0

    # Summary table
    print("=" * 80)
    print("R0 SUMMARY: EMA vs Raw weights (mean_imp)")
    print("=" * 80)
    header = f"{'iter':>6s}  {'RAW':>8s}  {'EMA':>8s}  {'Δ(pp)':>8s}"
    print(header)
    print("-" * len(header))
    for it_str in sorted(all_results, key=int):
        r = all_results[it_str]
        print(f"{it_str:>6s}  {r['raw_mean_imp']*100:>7.1f}%  {r['ema_mean_imp']*100:>7.1f}%  {r['delta_pp']:>+7.1f}")

    print()
    print("Per-baseline breakdown:")
    for it_str in sorted(all_results, key=int):
        r = all_results[it_str]
        print(f"\n  iter {it_str}:")
        for bl in args.baselines:
            raw_wr = r["raw"].get(bl, {}).get("win_rate", 0) * 100
            ema_wr = r["ema"].get(bl, {}).get("win_rate", 0) * 100
            d = ema_wr - raw_wr
            print(f"    {bl:<20s}  RAW={raw_wr:5.1f}%  EMA={ema_wr:5.1f}%  Δ={d:+5.1f}pp")

    print(f"\nTotal time: {total:.0f}s")

    # Save results
    output = {
        "description": "R0: EMA vs raw checkpoint weight evaluation for Phase 2 (interleaved-resnet-adaptive)",
        "config": CONFIG_PATH,
        "games_per_baseline": args.games,
        "baselines": args.baselines,
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
