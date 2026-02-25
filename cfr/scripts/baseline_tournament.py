"""Round-robin tournament between all baseline agents.

Produces a full pairwise win-rate matrix with confidence intervals.
No neural networks — pure heuristic agents only.
"""

import itertools
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

# Add cfr/src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from evaluate_agents import run_evaluation, AGENT_REGISTRY

BASELINES = [
    "random",
    "random_no_cambia",
    "random_late_cambia",
    "greedy",
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
    "human_player",
]

# Agent types that need checkpoints — skip these
CHECKPOINT_AGENTS = {"cfr", "deep_cfr", "escher", "sd_cfr", "nplayer"}


def wilson_ci(wins, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline agent round-robin tournament")
    parser.add_argument(
        "--config", type=str, default="runs/eppbs-2p/config.yaml",
        help="Config YAML (for game rules)",
    )
    parser.add_argument(
        "--games", type=int, default=10000,
        help="Games per matchup (alternates seats)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for full results",
    )
    args = parser.parse_args()

    config_path = args.config
    games_per_matchup = args.games

    # Verify all baselines exist
    for b in BASELINES:
        assert b in AGENT_REGISTRY, f"Unknown baseline: {b}"
        assert b not in CHECKPOINT_AGENTS, f"{b} requires checkpoint"

    matchups = list(itertools.combinations(BASELINES, 2))
    total_games = len(matchups) * games_per_matchup
    print(f"Tournament: {len(BASELINES)} baselines, {len(matchups)} matchups, "
          f"{games_per_matchup} games each = {total_games:,} total games")
    print(f"Config: {config_path}")
    print()

    results = {}
    t0 = time.perf_counter()

    for i, (a1, a2) in enumerate(matchups, 1):
        half = games_per_matchup // 2
        print(f"[{i}/{len(matchups)}] {a1} vs {a2} ({games_per_matchup} games, seat-balanced)...",
              end=" ", flush=True)
        mt = time.perf_counter()

        # Play half with a1 as P0, half with a2 as P0 to eliminate first-mover bias
        c1 = run_evaluation(
            config_path=config_path, agent1_type=a1, agent2_type=a2,
            num_games=half, strategy_path=None, checkpoint_path=None, device="cpu",
        )
        c2 = run_evaluation(
            config_path=config_path, agent1_type=a2, agent2_type=a1,
            num_games=half, strategy_path=None, checkpoint_path=None, device="cpu",
        )
        elapsed = time.perf_counter() - mt

        # a1 wins: P0 wins in c1 (a1 was P0) + P1 wins in c2 (a1 was P1)
        a1_wins = c1.get("P0 Wins", 0) + c2.get("P1 Wins", 0)
        a2_wins = c1.get("P1 Wins", 0) + c2.get("P0 Wins", 0)
        ties = c1.get("Ties", 0) + c2.get("Ties", 0)
        mtt = c1.get("MaxTurnTies", 0) + c2.get("MaxTurnTies", 0)
        errs = c1.get("Errors", 0) + c2.get("Errors", 0)
        decided = a1_wins + a2_wins

        wr, ci_lo, ci_hi = wilson_ci(a1_wins, decided) if decided > 0 else (0, 0, 0)

        # Merge enhanced stats
        s1 = getattr(c1, "stats", {})
        s2 = getattr(c2, "stats", {})
        avg_turns = (s1.get("avg_game_turns", 0) + s2.get("avg_game_turns", 0)) / 2
        avg_margin = (s1.get("avg_score_margin", 0) + s2.get("avg_score_margin", 0)) / 2

        print(f"{a1} {wr*100:.1f}% [{ci_lo*100:.1f}-{ci_hi*100:.1f}] | "
              f"turns={avg_turns:.0f} margin={avg_margin:.1f} | "
              f"{elapsed:.1f}s ({games_per_matchup/elapsed:.0f} games/s)"
              f"{f' errs={errs}' if errs else ''}")

        results[f"{a1}_vs_{a2}"] = {
            "agent1": a1,
            "agent2": a2,
            "games": games_per_matchup,
            "a1_wins": a1_wins,
            "a2_wins": a2_wins,
            "ties": ties,
            "max_turn_ties": mtt,
            "errors": errs,
            "a1_win_rate": round(wr, 4),
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
            "avg_turns": round(avg_turns, 1),
            "avg_score_margin": round(avg_margin, 1),
            "elapsed_s": round(elapsed, 1),
        }

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal: {total_elapsed:.0f}s ({total_games/total_elapsed:.0f} games/s)")

    # Print summary matrix
    print("\n=== Win Rate Matrix (row = P0, col = P1) ===")
    short = {b: b[:8] for b in BASELINES}
    header = f"{'':>12s}" + "".join(f"{short[b]:>10s}" for b in BASELINES)
    print(header)
    for a1 in BASELINES:
        row = f"{short[a1]:>12s}"
        for a2 in BASELINES:
            if a1 == a2:
                row += f"{'---':>10s}"
            else:
                key = f"{a1}_vs_{a2}"
                alt_key = f"{a2}_vs_{a1}"
                if key in results:
                    wr = results[key]["a1_win_rate"]
                    row += f"{wr*100:>9.1f}%"
                elif alt_key in results:
                    # a1 was a2 in that matchup, so win rate = 1 - a1_win_rate
                    wr = 1 - results[alt_key]["a1_win_rate"]
                    row += f"{wr*100:>9.1f}%"
                else:
                    row += f"{'?':>10s}"
        print(row)

    # Save full results
    output_path = args.output or "runs/eppbs-2p/baseline_tournament.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": config_path,
            "games_per_matchup": games_per_matchup,
            "baselines": BASELINES,
            "total_games": total_games,
            "total_elapsed_s": round(total_elapsed, 1),
            "matchups": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
