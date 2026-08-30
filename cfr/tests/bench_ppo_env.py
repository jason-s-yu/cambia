"""Throughput benchmark for the PPO gym environment (cambia-1376 AC2).

Not collected by pytest (no ``test_`` prefix): this is the instrument behind
the before/after numbers on the GoEngine port, kept next to the tests so the
measurement is reproducible rather than a one-off script.

The environment is driven by a uniform-random legal action at every agent
decision, which is the same work per step the PPO rollout does minus the
policy forward pass, so the number isolates environment cost.

Usage (from cfr/, with PYTHONPATH and LIBCAMBIA_PATH pinned):

    python tests/bench_ppo_env.py --steps 20000 --config config/ppo_encoding_v2.yaml
"""

import argparse
import time

import numpy as np


def bench(
    steps: int,
    config_path: str,
    opponent: str,
    num_players: int,
    seed: int,
    snapshot_path: str | None = None,
) -> dict:
    """Step the env with uniform-random legal actions and time it.

    Returns a dict with agent steps per second, episodes completed, and the
    mean number of agent decisions per episode.
    """
    from src.ppo_env import CambiaEnv

    kwargs = dict(
        opponent_type=opponent,
        seed=seed,
        config_path=config_path,
        selfplay_snapshot_path=snapshot_path,
    )
    try:
        env = CambiaEnv(num_players=num_players, **kwargs)
    except TypeError:
        # Pre-port env has no num_players parameter; 2 seats is all it does.
        if num_players != 2:
            raise
        env = CambiaEnv(**kwargs)

    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    # One untimed warm-up episode: first reset pays lazy config load, library
    # dlopen and (under self-play) the first snapshot read.
    for _ in range(200):
        mask = np.asarray(env.action_masks()).astype(bool)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            env.reset()
            continue
        _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
        if term or trunc:
            env.reset()
            break

    episodes = 0
    done_steps = 0
    t0 = time.perf_counter()
    while done_steps < steps:
        mask = np.asarray(env.action_masks()).astype(bool)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            env.reset()
            episodes += 1
            continue
        _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
        done_steps += 1
        if term or trunc:
            env.reset()
            episodes += 1
    elapsed = time.perf_counter() - t0
    env.close()

    return {
        "steps": done_steps,
        "seconds": elapsed,
        "steps_per_sec": done_steps / elapsed,
        "episodes": episodes,
        "steps_per_episode": done_steps / episodes if episodes else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--config", default="config/ppo_encoding_v2.yaml")
    ap.add_argument("--opponent", default="imperfect_greedy")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshot-path", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    rates = []
    for r in range(args.repeats):
        res = bench(
            args.steps,
            args.config,
            args.opponent,
            args.num_players,
            args.seed + r,
            args.snapshot_path,
        )
        rates.append(res["steps_per_sec"])
        print(
            f"run {r}: {res['steps']} agent steps in {res['seconds']:.2f}s "
            f"-> {res['steps_per_sec']:,.0f} steps/s "
            f"({res['episodes']} episodes, {res['steps_per_episode']:.1f} steps/ep)"
        )
    arr = np.array(rates)
    print(
        f"\nopponent={args.opponent} seats={args.num_players} "
        f"config={args.config}\n"
        f"median {np.median(arr):,.0f} steps/s  "
        f"(min {arr.min():,.0f}, max {arr.max():,.0f}, n={len(arr)})"
    )


if __name__ == "__main__":
    main()
