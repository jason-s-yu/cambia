"""Re-evaluate all prod-full-333 checkpoints against fixed baselines.

The stale-memory bug (agents not resetting between games) invalidated all prior
evaluations against imperfect agents. This script re-runs seat-balanced evals
for key checkpoints and outputs corrected metrics.

Greedy and random evals are NOT re-run — they were unaffected (no memory state).
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from evaluate_agents import run_evaluation

CONFIG = "runs/eppbs-2p/config.yaml"
# Use legacy config for prod-full-333 (it used legacy encoding)
LEGACY_CONFIG = "runs/prod-full-333/config.yaml"
CHECKPOINT_DIR = Path("runs/prod-full-333/checkpoints")
GAMES_PER_BASELINE = 5000  # 2500 per seat

# Only re-eval imperfect baselines (greedy/random were unaffected)
BASELINES = ["imperfect_greedy", "memory_heuristic", "aggressive_snap"]

# Also include greedy and random for comparison (unaffected but useful to have)
ALL_BASELINES = ["random", "greedy"] + BASELINES

# Key checkpoints: every 100 iters + final
KEY_ITERS = list(range(100, 1100, 100)) + [1075]
# Remove duplicates and sort
KEY_ITERS = sorted(set(KEY_ITERS))


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def eval_checkpoint(ckpt_path: str, config_path: str, baselines: list, games: int):
    """Evaluate a checkpoint against baselines with seat balancing."""
    results = {}
    half = games // 2
    for bl in baselines:
        c1 = run_evaluation(
            config_path=config_path, agent1_type="deep_cfr", agent2_type=bl,
            num_games=half, strategy_path=None, checkpoint_path=ckpt_path, device="cpu",
        )
        c2 = run_evaluation(
            config_path=config_path, agent1_type=bl, agent2_type="deep_cfr",
            num_games=half, strategy_path=None, checkpoint_path=ckpt_path, device="cpu",
        )
        model_wins = c1.get("P0 Wins", 0) + c2.get("P1 Wins", 0)
        bl_wins = c1.get("P1 Wins", 0) + c2.get("P0 Wins", 0)
        decided = model_wins + bl_wins
        wr, ci_lo, ci_hi = wilson_ci(model_wins, decided) if decided else (0, 0, 0)

        s1, s2 = getattr(c1, "stats", {}), getattr(c2, "stats", {})
        avg_turns = (s1.get("avg_game_turns", 0) + s2.get("avg_game_turns", 0)) / 2

        results[bl] = {
            "model_wins": model_wins, "bl_wins": bl_wins,
            "win_rate": round(wr, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "avg_turns": round(avg_turns, 1), "decided": decided,
        }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=LEGACY_CONFIG)
    parser.add_argument("--games", type=int, default=GAMES_PER_BASELINE)
    parser.add_argument("--baselines", nargs="+", default=ALL_BASELINES)
    parser.add_argument("--iters", nargs="+", type=int, default=KEY_ITERS)
    parser.add_argument("--output", default="runs/prod-full-333/corrected_eval.json")
    args = parser.parse_args()

    # Check config exists
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        # Try eppbs config as fallback (different encoding but same game rules)
        config_path = CONFIG
        print(f"Falling back to: {config_path}")

    all_results = {}
    t0 = time.perf_counter()

    for it in args.iters:
        ckpt = CHECKPOINT_DIR / f"deep_cfr_checkpoint_iter_{it}.pt"
        if not ckpt.exists():
            print(f"[SKIP] iter {it}: {ckpt} not found")
            continue

        print(f"\n[iter {it}] Evaluating against {len(args.baselines)} baselines "
              f"({args.games} games each, seat-balanced)...")
        mt = time.perf_counter()

        results = eval_checkpoint(str(ckpt), config_path, args.baselines, args.games)
        elapsed = time.perf_counter() - mt

        row = f"  iter {it:5d}:"
        for bl in args.baselines:
            r = results[bl]
            row += f"  {bl[:8]}={r['win_rate']*100:.1f}%"
        row += f"  ({elapsed:.0f}s)"
        print(row)
        all_results[str(it)] = results

    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.0f}s")

    # Print summary table
    print("\n=== Corrected Eval Summary ===")
    header = f"{'iter':>6s}"
    for bl in args.baselines:
        header += f"  {bl[:12]:>12s}"
    print(header)
    for it_str in sorted(all_results, key=int):
        row = f"{it_str:>6s}"
        for bl in args.baselines:
            r = all_results[it_str].get(bl, {})
            wr = r.get("win_rate", 0)
            row += f"  {wr*100:>11.1f}%"
        print(row)

    # Save
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": config_path,
            "games_per_baseline": args.games,
            "baselines": args.baselines,
            "note": "Corrected eval: agents now properly reset memory between games",
            "bug": "ImperfectMemoryMixin._ensure_initialized() never re-initialized on new game",
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
