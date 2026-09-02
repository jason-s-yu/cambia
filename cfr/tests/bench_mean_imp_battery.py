"""Throughput benchmark for the mean_imp evaluation battery (cambia-1487).

Not collected by pytest (no ``test_`` prefix): this is the instrument behind
the before/after numbers on the baseline-heuristic port, kept next to the
tests so the measurement is reproducible rather than a one-off script.

Each leg plays ``--games`` games of the agent under test against one mean_imp
baseline through ``run_evaluation``, which is the loop the real battery runs.
The agent under test defaults to ``random``, whose per-decision cost is a
uniform draw over the legal set, so the number tracks the baselines and the
engine rather than a policy forward pass.

Usage (from cfr/, with PYTHONPATH and LIBCAMBIA_PATH pinned):

    python tests/bench_mean_imp_battery.py --games 25000 \
        --config config/ppo_encoding_v2.yaml
"""

import argparse
import json
import logging
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=25000)
    parser.add_argument("--config", type=str, default="config/ppo_encoding_v2.yaml")
    parser.add_argument("--agent", type=str, default="random")
    parser.add_argument("--seats", type=int, default=2)
    parser.add_argument("--crn-seed", type=int, default=12345)
    parser.add_argument("--baselines", type=str, default="")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    from src.evaluate_agents import MEAN_IMP_BASELINES, run_evaluation

    baselines = (
        [b.strip() for b in args.baselines.split(",") if b.strip()]
        if args.baselines
        else list(MEAN_IMP_BASELINES)
    )

    def leg(baseline: str, games: int, identity: str):
        return run_evaluation(
            config_path=args.config,
            agent1_type=args.agent,
            agent2_type=baseline,
            num_games=games,
            strategy_path=None,
            crn_seed_base=args.crn_seed,
            crn_identity=identity,
            num_players=args.seats,
        )

    def battery():
        out = {}
        for baseline in baselines:
            t0 = time.perf_counter()
            c0 = time.process_time()
            leg(baseline, args.games, "cambia-1487-bench")
            dt = time.perf_counter() - t0
            cpu = time.process_time() - c0
            out[baseline] = (dt, cpu)
            print(
                f"{baseline:22s} {dt:8.2f}s {args.games / dt:8.1f} games/s"
                f"   cpu {cpu:8.2f}s {args.games / cpu:8.1f} games/cpu-s",
                flush=True,
            )
        return out

    if args.profile:
        import cProfile
        import pstats

        prof = cProfile.Profile()
        prof.enable()
        per_leg = battery()
        prof.disable()
        prof.dump_stats(args.profile)
        pstats.Stats(prof).sort_stats("tottime").print_stats(35)
    else:
        # One untimed warm-up leg: the first game pays lazy config load and the
        # library dlopen.
        leg(baselines[0], 20, "cambia-1487-warmup")
        per_leg = battery()

    total_s = sum(wall for wall, _ in per_leg.values())
    total_cpu = sum(cpu for _, cpu in per_leg.values())
    total_games = args.games * len(baselines)
    # CPU time as well as wall: this measurement is single-threaded and CPU
    # bound, so on a box with co-tenants the wall number moves with their load
    # while the CPU number does not.
    print(
        f"\nBATTERY {total_games} games in {total_s:.2f}s "
        f"= {total_games / total_s:.1f} games/s"
        f"   cpu {total_cpu:.2f}s = {total_games / total_cpu:.1f} games/cpu-s"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "games_per_leg": args.games,
                    "agent": args.agent,
                    "per_leg_seconds": {k: v[0] for k, v in per_leg.items()},
                    "per_leg_cpu_seconds": {k: v[1] for k, v in per_leg.items()},
                    "total_seconds": total_s,
                    "total_cpu_seconds": total_cpu,
                    "games_per_second": total_games / total_s,
                    "games_per_cpu_second": total_games / total_cpu,
                },
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
