"""PPO trainer for Cambia, on the Go-backed environment (cambia-1376).

Two regimes, selected by ``opponent``:

- ``"self_play"``: fair self-play (E2). Every seat other than the learner is a
  frozen-periodic snapshot of the learning policy, refreshed every
  ``selfplay_snapshot_freq`` timesteps. The learner's seat is randomized per
  episode. PPO improves only by beating copies of itself, so its measurement
  re-derives reachable headroom. This run is the equilibrium anchor: never
  optimized toward, only measured.
- ``"random_legal"``: opponent seats play uniform-random legal actions. A cheap
  control and the regime the env benchmark uses.
- a ``src.agents.baseline_agents`` name (``"imperfect_greedy"`` and friends):
  the fixed best-response diagnostic, on cambia-1426's GameView port
  (cambia-1482). ``imperfect_greedy`` is the default opponent (``cli.py``'s
  ``train ppo``), matching PPO-200k's original training opponent.
  Checkpoint-backed wrappers and the tabular ``CFRAgentWrapper`` are not
  accepted; see ``ppo_env.is_baseline_opponent``.

``num_players`` sets the seat count. Two seats use the engine's 2-player action
and encoding space; three or more use its N-player space, which has different
widths, so checkpoints do not transfer across that boundary.

Eval-during-training persists per baseline to ``metrics.jsonl`` and the SQLite
run_db through the shared ``persist_eval_results`` path. That eval still runs
on the Python-engine battery in ``evaluate_agents``; it is 2-seat only and is
skipped at higher seat counts, and ``eval_freq=0`` turns it off entirely.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _run_dir_from_save_path(save_path: str) -> str:
    """Derive the run directory from a checkpoint save path.

    Convention (matches ppo_encoding_v2.yaml and the DESCA trainers):
    ``runs/<run>/checkpoints/<model>`` -> ``runs/<run>``. Falls back to the
    parent directory when there is no ``checkpoints`` segment.
    """
    from pathlib import Path

    p = Path(save_path).resolve()
    parts = p.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        return str(Path(*parts[:idx]))
    return str(p.parent)


def _mean_imp(results_map, baselines) -> float:
    """P0 win rate averaged over the baselines that have data."""
    rates = []
    for bl in baselines:
        c = results_map.get(bl)
        if not c:
            continue
        p0 = c.get("P0 Wins", 0)
        p1 = c.get("P1 Wins", 0)
        ties = c.get("Ties", 0) + c.get("MaxTurnTies", 0)
        total = p0 + p1 + ties
        if total > 0:
            rates.append(p0 / total)
    return sum(rates) / len(rates) if rates else 0.0


def _register_run(run_dir: str, run_name: str, config_path: str):
    """Upsert the run into run_db on training start, and materialize the run-dir
    config so eval persistence finds runs/<run>/config.yaml. Non-fatal."""
    try:
        from pathlib import Path
        import yaml
        import src.run_db as run_db

        cfg_path = Path(config_path)
        yaml_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else None
        config_dict = yaml.safe_load(yaml_text) if yaml_text else {}
        run_cfg = Path(run_dir) / "config.yaml"
        if not run_cfg.exists() and yaml_text is not None:
            run_cfg.parent.mkdir(parents=True, exist_ok=True)
            run_cfg.write_text(yaml_text, encoding="utf-8")
        db = run_db.get_db()
        algorithm = run_db.infer_algorithm(config_dict or {})
        run_db.upsert_run(
            db,
            name=run_name,
            algorithm=algorithm,
            config_yaml=yaml_text,
            config_dict=config_dict or {},
            status="running",
        )
        db.close()
    except Exception:
        logger.warning("run_db upsert_run failed for %s (non-fatal).", run_name)


def _mark_run_status(run_name: str, status: str):
    try:
        import src.run_db as run_db

        db = run_db.get_db()
        row = db.execute("SELECT id FROM runs WHERE name=?", (run_name,)).fetchone()
        if row is not None:
            run_db.update_run_status(db, row["id"], status)
        db.close()
    except Exception:
        logger.warning("run_db status update failed for %s (non-fatal).", run_name)


def _build_callbacks(
    *,
    self_play: bool,
    snapshot_stem: str,
    selfplay_snapshot_freq: int,
    config_path: str,
    run_dir: str,
    run_name: str,
    save_path: str,
    eval_freq: int,
    n_envs: int,
    eval_games: int,
    eval_max_workers,
    checkpoint_freq,
    run_eval: bool,
):
    """Construct the SB3 callback list. SB3 is imported here so the module stays
    importable without sb3-contrib installed (the diagnostic path also defers)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class SelfPlaySnapshotCallback(BaseCallback):
        """Refresh the frozen self-play opponent snapshot on disk.

        Saving overwrites ``snapshot_stem + .zip``; env workers detect the mtime
        change and hot-reload (see ppo_env.SelfPlayPolicyOpponent). The opponent
        tracks the learner with a lag of at most one refresh interval while
        staying stationary within each PPO rollout.
        """

        def __init__(self, stem: str, freq: int, envs: int):
            super().__init__(verbose=1)
            self._stem = stem
            self._every = max(freq // max(envs, 1), 1)
            self._last = 0

        def _on_step(self) -> bool:
            if self.n_calls - self._last >= self._every:
                self._last = self.n_calls
                self.model.save(self._stem)
                if self.verbose:
                    print(
                        f"[self-play] refreshed opponent snapshot at "
                        f"{self.num_timesteps:,} timesteps"
                    )
            return True

    class PeriodicCheckpointCallback(BaseCallback):
        """Save a timestamped checkpoint every checkpoint_freq timesteps."""

        def __init__(self, path: str, freq: int, envs: int):
            super().__init__()
            self._every = max(freq // max(envs, 1), 1)
            self._last = 0
            self._dir = os.path.dirname(path) or "."

        def _on_step(self) -> bool:
            if self.n_calls - self._last >= self._every:
                self._last = self.n_calls
                stem = os.path.join(self._dir, f"ppo_model_steps_{self.num_timesteps}")
                self.model.save(stem)
                if self.verbose:
                    print(f"[checkpoint] saved {stem}.zip")
            return True

    class MeanImpEvalCallback(BaseCallback):
        """Per-baseline mean_imp eval during training with run_db persistence.

        Every eval_freq timesteps: save the current policy to a checkpoint .zip,
        evaluate it as the ``ppo`` agent against the five MEAN_IMP_BASELINES via
        the shared multi-baseline driver, then dual-write per-baseline rows to
        metrics.jsonl + SQLite through persist_eval_results. The iteration is the
        global timestep count, so the anchor's trajectory is queryable by step.
        """

        def __init__(self):
            super().__init__(verbose=1)
            self._every = max(eval_freq // max(n_envs, 1), 1)
            self._last = 0
            self._ckpt_dir = os.path.dirname(save_path) or "."

        def _on_step(self) -> bool:
            if self.n_calls - self._last < self._every:
                return True
            self._last = self.n_calls
            self._run_eval()
            return True

        def _run_eval(self):
            from src.evaluate_agents import (
                run_evaluation_multi_baseline,
                persist_eval_results,
                MEAN_IMP_BASELINES,
            )

            steps = int(self.num_timesteps)
            ckpt_stem = os.path.join(self._ckpt_dir, f"ppo_model_eval_{steps}")
            self.model.save(ckpt_stem)
            ckpt_zip = ckpt_stem + ".zip"

            try:
                results_map = run_evaluation_multi_baseline(
                    config_path=config_path,
                    checkpoint_path=ckpt_zip,
                    num_games=eval_games,
                    baselines=list(MEAN_IMP_BASELINES),
                    device="cpu",
                    agent_type="ppo",
                    max_workers=eval_max_workers,
                )
            except Exception:
                logger.exception("PPO eval failed at %d steps (non-fatal).", steps)
                return

            try:
                persist_eval_results(
                    run_dir=run_dir,
                    iteration=steps,
                    results_map=results_map,
                    run_name=run_name,
                    checkpoint_path=ckpt_zip,
                )
            except Exception:
                logger.exception(
                    "persist_eval_results failed at %d steps (non-fatal).", steps
                )
                return

            if self.verbose:
                mi = _mean_imp(results_map, MEAN_IMP_BASELINES)
                print(f"[eval] step {steps:,}: mean_imp5 = {mi:.4f}")

    callbacks = []
    if self_play:
        callbacks.append(
            SelfPlaySnapshotCallback(snapshot_stem, selfplay_snapshot_freq, n_envs)
        )
    if run_eval:
        callbacks.append(MeanImpEvalCallback())
    if checkpoint_freq:
        callbacks.append(PeriodicCheckpointCallback(save_path, checkpoint_freq, n_envs))
    return callbacks


def train_ppo(
    opponent: str,
    timesteps: int,
    save_path: str,
    n_envs: int,
    eval_freq: int,
    net_arch: list,
    seed: int = 42,
    config_path: str = "config.yaml",
    run_name: str | None = None,
    eval_games: int = 5000,
    selfplay_snapshot_freq: int = 200_000,
    eval_max_workers: int | None = None,
    checkpoint_freq: int | None = None,
    num_players: int = 2,
):
    """Train a MaskablePPO agent.

    Args:
        opponent: "self_play" for fair self-play (E2), or "random_legal".
        timesteps: Total training timesteps.
        save_path: Checkpoint path stem (SB3 appends .zip). The run directory is
            derived from this (runs/<run>/checkpoints/<model> -> runs/<run>).
        n_envs: Parallel SubprocVecEnv workers.
        eval_freq: Run the per-baseline mean_imp eval every N timesteps. 0
            disables the eval callback (the smoke and benchmark path).
        net_arch: MLP hidden sizes.
        seed: Base RNG seed.
        config_path: Path to config YAML.
        run_name: Run name for metrics.jsonl / run_db rows. Defaults to the run
            directory basename.
        eval_games: Games per baseline in each eval cycle.
        selfplay_snapshot_freq: Refresh the self-play opponent snapshot every N
            timesteps (self-play only).
        eval_max_workers: Parallel baseline eval workers (None = auto).
        checkpoint_freq: Save a periodic timestamped checkpoint every N
            timesteps. None disables periodic checkpoints (eval still saves one).
        num_players: Seat count. 2 uses the engine's 2-player action/encoding
            space; 3+ uses its N-player space. The mean_imp eval battery is
            2-seat only and is skipped above that.
    """
    # Thread pinning: SubprocVecEnv spawns n_envs workers, each importing torch +
    # numpy; the self-play opponent also loads a MaskablePPO per worker. Without
    # pinning, every worker opens a full-core BLAS/OMP pool, so n_envs x ncores
    # threads oversubscribe the box and thrash (load ~100 at n_envs=16). setdefault
    # lets a launch-time override win; spawn workers inherit the env at import.
    import torch as _torch

    for _tvar in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_tvar, "1")
    _torch.set_num_threads(1)

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for PPO training. "
            "Install with: pip install -e '.[rl]'"
        )
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CallbackList
    from src.ppo_env import (
        make_env,
        SELF_PLAY_OPPONENT,
        RANDOM_LEGAL_OPPONENT,
        SUPPORTED_OPPONENTS,
        is_baseline_opponent,
    )

    self_play = opponent == SELF_PLAY_OPPONENT
    num_players = int(num_players)
    if num_players < 2:
        raise ValueError(f"num_players must be at least 2, got {num_players}")
    if opponent not in SUPPORTED_OPPONENTS and not is_baseline_opponent(opponent):
        # Fail here rather than inside every SubprocVecEnv worker, where the
        # traceback arrives as an opaque connection reset.
        raise NotImplementedError(
            f"opponent={opponent!r} is not available on the Go-backed env. "
            f"Supported: {', '.join(SUPPORTED_OPPONENTS)}, or a "
            "src.agents.baseline_agents name (e.g. 'imperfect_greedy'). "
            "Checkpoint-backed wrappers and the tabular CFRAgentWrapper are "
            "not supported here."
        )

    # The mean_imp battery in evaluate_agents drives 2-player games; it has no
    # meaning at a 3+ seat table, so the callback is 2-seat only.
    run_eval = bool(eval_freq) and num_players == 2
    if eval_freq and num_players != 2:
        print(
            f"  Note: mean_imp eval disabled at {num_players} seats "
            "(the battery is 2-player only)."
        )
    if run_eval:
        # The battery scores through PPOAgentWrapper, which still strips
        # drawn_card and peeked_cards from the agent's own observation (the
        # pre-port public-only contract). This env feeds the policy
        # GoAgentState's belief instead, which keeps what the rules reveal to
        # the acting seat, so the eval runs the policy off its training
        # distribution and the trajectory is not comparable to pre-port runs.
        # cambia-1426 moves the eval side onto GoAgentState; until it lands,
        # say so rather than letting a mean_imp trajectory look authoritative.
        msg = (
            "mean_imp eval scores this policy through the pre-port public-only "
            "observation contract while training now uses GoAgentState's "
            "belief; the trajectory is off-distribution and not comparable to "
            "pre-port runs until cambia-1426 lands."
        )
        print(f"  WARNING: {msg}")
        logger.warning(msg)

    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    run_dir = _run_dir_from_save_path(save_path)
    os.makedirs(run_dir, exist_ok=True)
    if run_name is None:
        run_name = os.path.basename(run_dir)

    # Live policy snapshot the self-play env workers reload from.
    snapshot_stem = os.path.join(save_dir, "selfplay_opponent")
    snapshot_path = snapshot_stem + ".zip"

    if self_play:
        regime_label = "FAIR SELF-PLAY (E2 anchor)"
    elif opponent == RANDOM_LEGAL_OPPONENT:
        regime_label = "uniform-random opponent"
    else:
        regime_label = "fixed-baseline best-response diagnostic"

    print("PPO Training")
    print(f"  Regime: {regime_label}")
    print(f"  Opponent: {opponent}")
    print(
        f"  Seats: {num_players} "
        f"({'N-player' if num_players > 2 else '2-player'} action space)"
    )
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Net arch: {net_arch}")
    print(f"  Save path: {save_path}")
    print(f"  Run dir: {run_dir}  (run_name={run_name})")
    if self_play:
        print(
            f"  Self-play snapshot: {snapshot_path} "
            f"(refresh every {selfplay_snapshot_freq:,} steps)"
        )
    if run_eval:
        print(f"  Eval: {eval_games} games/baseline every {eval_freq:,} steps")
    else:
        print("  Eval: disabled")

    envs = SubprocVecEnv(
        [
            make_env(
                opponent,
                seed=seed + i,
                config_path=config_path,
                selfplay_snapshot_path=snapshot_path if self_play else None,
                num_players=num_players,
            )
            for i in range(n_envs)
        ]
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

    # Write an initial snapshot so self-play workers exit warm-up after the
    # first refresh interval (they reload on file change).
    if self_play:
        model.save(snapshot_stem)

    callbacks = _build_callbacks(
        self_play=self_play,
        snapshot_stem=snapshot_stem,
        selfplay_snapshot_freq=selfplay_snapshot_freq,
        config_path=config_path,
        run_dir=run_dir,
        run_name=run_name,
        save_path=save_path,
        eval_freq=eval_freq,
        n_envs=n_envs,
        eval_games=eval_games,
        eval_max_workers=eval_max_workers,
        checkpoint_freq=checkpoint_freq,
        run_eval=run_eval,
    )

    _register_run(run_dir, run_name, config_path)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    finally:
        model.save(save_path)
        print(f"\nModel saved to {save_path}")
        envs.close()
        _mark_run_status(run_name, "completed")
