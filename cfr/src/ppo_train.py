"""PPO best-response diagnostic trainer for Cambia."""

import logging
import os

logger = logging.getLogger(__name__)


def train_ppo(
    opponent: str,
    timesteps: int,
    save_path: str,
    n_envs: int,
    eval_freq: int,
    net_arch: list,
    seed: int = 42,
    config_path: str = "config.yaml",
):
    """Train a MaskablePPO agent against a fixed opponent."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for PPO training. "
            "Install with: pip install -e '.[rl]'"
        )
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    from src.ppo_env import make_env, CambiaEnv

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"PPO Diagnostic Training")
    print(f"  Opponent: {opponent}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Net arch: {net_arch}")
    print(f"  Save path: {save_path}")

    envs = SubprocVecEnv(
        [make_env(opponent, seed=seed + i, config_path=config_path) for i in range(n_envs)]
    )

    eval_env = CambiaEnv(
        opponent_type=opponent, seed=seed + 1000, config_path=config_path
    )

    model = MaskablePPO(
        "MlpPolicy",
        envs,
        verbose=1,
        device="cpu",
        seed=seed,
        policy_kwargs={"net_arch": net_arch},
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=1.0,
        ent_coef=0.01,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path) or ".",
        log_path=os.path.dirname(save_path) or ".",
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=100,
        deterministic=True,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    envs.close()
    eval_env.close()
