"""PPO learnability diagnostic - standalone script."""

import os
import sys
import time
import multiprocessing

os.chdir("/home/agent/dev/cambia/cfr")


class WinRateCallback:
    """Lightweight eval: track WR every N timesteps by playing 200 games."""

    def __init__(self, eval_freq=10000, n_eval_games=200, verbose=1):
        from stable_baselines3.common.callbacks import BaseCallback

        self._base_cls = BaseCallback
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.results = []


def _make_callback(eval_freq=2500, n_eval_games=200):
    from stable_baselines3.common.callbacks import BaseCallback
    from src.ppo_env import CambiaEnv

    class _WRCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=1)
            self.eval_freq = eval_freq
            self.n_eval_games = n_eval_games
            self.results = []

        def _on_step(self):
            if self.n_calls % self.eval_freq == 0:
                env = CambiaEnv(
                    opponent_type="imperfect_greedy", seed=self.n_calls
                )
                wins = 0
                for g in range(self.n_eval_games):
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        mask = env.action_masks()
                        action, _ = self.model.predict(
                            obs, deterministic=True, action_masks=mask
                        )
                        obs, reward, terminated, truncated, _ = env.step(
                            int(action)
                        )
                        done = terminated or truncated
                    if reward > 0:
                        wins += 1
                wr = wins / self.n_eval_games
                self.results.append((self.num_timesteps, wr))
                env.close()
                print(
                    f"  [{self.num_timesteps:>7,} steps] "
                    f"WR vs imperfect_greedy = {wr:.1%}  "
                    f"({wins}/{self.n_eval_games})",
                    flush=True,
                )
            return True

    return _WRCallback()


def main():
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from src.ppo_env import make_env, CambiaEnv

    TIMESTEPS = 200_000
    N_ENVS = 4
    SEED = 42
    SAVE_PATH = "runs/ppo-diagnostic/model"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("=" * 60)
    print("PPO Learnability Diagnostic")
    print(f"  Timesteps: {TIMESTEPS:,}")
    print(f"  Envs: {N_ENVS}")
    print(f"  Opponent: imperfect_greedy")
    print(f"  Eval: 200 games every 10K steps")
    print("=" * 60, flush=True)

    t0 = time.time()

    envs = SubprocVecEnv(
        [make_env("imperfect_greedy", seed=SEED + i) for i in range(N_ENVS)]
    )

    model = MaskablePPO(
        "MlpPolicy",
        envs,
        verbose=0,
        device="cpu",
        seed=SEED,
        policy_kwargs={"net_arch": [256, 256]},
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=1.0,
        ent_coef=0.01,
    )

    wr_callback = _make_callback(eval_freq=2500, n_eval_games=200)
    # eval_freq is per-env, so 2500 * 4 = every 10K global steps

    model.learn(total_timesteps=TIMESTEPS, callback=wr_callback)
    model.save(SAVE_PATH)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min. Model: {SAVE_PATH}")

    envs.close()

    # ---- Final eval ----
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (5000 games each)")
    print("=" * 60, flush=True)

    BASELINES = [
        "random",
        "greedy",
        "random_no_cambia",
        "random_late_cambia",
        "imperfect_greedy",
        "memory_heuristic",
        "aggressive_snap",
    ]

    MI3_BASELINES = {"imperfect_greedy", "memory_heuristic", "aggressive_snap"}
    MI5_BASELINES = {
        "random_no_cambia",
        "random_late_cambia",
        "imperfect_greedy",
        "memory_heuristic",
        "aggressive_snap",
    }

    wr_results = {}
    for opp in BASELINES:
        env = CambiaEnv(opponent_type=opp, seed=9999)
        wins = 0
        draws = 0
        n_games = 5000
        for g in range(n_games):
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=mask
                )
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            if reward > 0:
                wins += 1
            elif reward == 0:
                draws += 1
        wr = wins / n_games
        wr_results[opp] = wr
        print(
            f"  vs {opp:<25s}: {wr:.1%}  ({wins}/{n_games}, {draws} draws)",
            flush=True,
        )
        env.close()

    mi3 = np.mean([wr_results[b] for b in MI3_BASELINES])
    mi5 = np.mean([wr_results[b] for b in MI5_BASELINES])
    print(f"\n--- Summary ---")
    print(f"mi(3) = {mi3:.1%}")
    print(f"mi(5) = {mi5:.1%}")

    # Training curve
    print(f"\n--- Training Curve ---")
    for ts, wr in wr_callback.results:
        print(f"  {ts:>7,} steps: {wr:.1%}")

    print(f"\nTotal training + eval time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
