"""src/cli.py - Typer-based CLI for Cambia CFR Training Suite."""

import os
import sys
import signal
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union, get_args, get_origin

import typer

# Main app
app = typer.Typer(
    name="cambia",
    help="Cambia CFR Training Suite",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Train subcommand group
train_app = typer.Typer(
    help="Train a CFR agent",
    no_args_is_help=True,
)
app.add_typer(train_app, name="train")


def register_with_aliases(app, names, fn, *, help: Optional[str] = None) -> None:
    """Register a single Typer callback under multiple command names.

    Typer/Click lacks native command aliases, so the same callable is
    registered once per alias. Used to let users invoke identical behaviour via
    dashed, undashed, or underscored forms (e.g. `gt-cfr` <-> `gtcfr`).
    """
    for name in names:
        app.command(name, help=help)(fn)


def setup_multiprocessing():
    """Set up multiprocessing start method for stability."""
    try:
        preferred_method = "forkserver" if sys.platform != "win32" else "spawn"
        available_methods = multiprocessing.get_all_start_methods()
        current_method = multiprocessing.get_start_method(allow_none=True)

        method_to_set = None
        if preferred_method in available_methods:
            method_to_set = preferred_method
        elif "spawn" in available_methods:
            method_to_set = "spawn"
        elif "fork" in available_methods and sys.platform != "win32":
            method_to_set = "fork"

        if method_to_set and (current_method is None or current_method != method_to_set):
            force_set = current_method is None
            multiprocessing.set_start_method(method_to_set, force=force_set)
    except RuntimeError:
        pass


@train_app.command("tabular", help="Train using tabular CFR+")
def train_tabular(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Number of iterations to run (overrides config)",
    ),
    load: bool = typer.Option(
        False,
        "--load",
        help="Load existing agent data before training",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override save path for agent data",
    ),
):
    """Train a tabular CFR+ agent."""
    from .config import load_config
    from .main_train import create_infrastructure, run_tabular_training, handle_sigint

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    total_iterations = (
        iterations if iterations is not None else cfg.cfr_training.num_iterations
    )

    try:
        infra = create_infrastructure(cfg, total_iterations)
        exit_code = run_tabular_training(
            cfg,
            infra,
            iterations=iterations,
            load=load,
            save_path=str(save_path) if save_path else None,
        )
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@train_app.command("deep", help="Train using Deep CFR")
def train_deep(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Number of training steps to run",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    # Deep CFR overrides
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate",
        rich_help_panel="Deep CFR Overrides",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size",
        rich_help_panel="Deep CFR Overrides",
    ),
    train_steps: Optional[int] = typer.Option(
        None,
        "--train-steps",
        help="SGD steps per iteration",
        rich_help_panel="Deep CFR Overrides",
    ),
    traversals: Optional[int] = typer.Option(
        None,
        "--traversals",
        help="Traversals per training step",
        rich_help_panel="Deep CFR Overrides",
    ),
    alpha: Optional[float] = typer.Option(
        None,
        "--alpha",
        help="Iteration weighting exponent",
        rich_help_panel="Deep CFR Overrides",
    ),
    buffer_capacity: Optional[int] = typer.Option(
        None,
        "--buffer-capacity",
        help="Reservoir buffer capacity",
        rich_help_panel="Deep CFR Overrides",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
        rich_help_panel="Deep CFR Overrides",
    ),
    gpu: Optional[bool] = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="[deprecated: use --device] Use GPU if available",
        rich_help_panel="Deep CFR Overrides",
        hidden=True,
    ),
    amp: Optional[bool] = typer.Option(
        None,
        "--amp/--no-amp",
        help="Enable/disable automatic mixed precision (AMP) on CUDA",
        rich_help_panel="Deep CFR Overrides",
    ),
    compile: Optional[bool] = typer.Option(
        None,
        "--compile/--no-compile",
        help="Enable/disable torch.compile on CUDA",
        rich_help_panel="Deep CFR Overrides",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Disable Rich TUI; print plain-text progress to stdout. "
        "Auto-enabled when stdout is not a TTY (e.g. nohup, pipes, log redirection).",
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Fix all random seeds (42) and force num_traversal_threads=1 for reproducible runs.",
        rich_help_panel="Deep CFR Overrides",
    ),
    profile_step: Optional[int] = typer.Option(
        None,
        "--profile-step",
        help="Run torch.profiler on this step number and export Chrome trace to the run directory.",
        rich_help_panel="Deep CFR Overrides",
    ),
):
    """Train a Deep CFR agent."""
    from .config import load_config
    from .main_train import create_infrastructure, run_deep_training, handle_sigint
    from .cfr.deep_trainer import DeepCFRConfig

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    # Auto-enable headless mode when stdout is not a TTY
    if not headless:
        headless = not sys.stdout.isatty()

    if deterministic:
        import random as _random
        import numpy as _np
        import torch as _torch

        _torch.manual_seed(42)
        _np.random.seed(42)
        _random.seed(42)
        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    # Build overrides dict from CLI options
    overrides = {}
    if deterministic:
        overrides["num_traversal_threads"] = 1
    if lr is not None:
        overrides["learning_rate"] = lr
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if train_steps is not None:
        overrides["train_steps_per_iteration"] = train_steps
    if traversals is not None:
        overrides["traversals_per_step"] = traversals
    if alpha is not None:
        overrides["alpha"] = alpha
    if buffer_capacity is not None:
        overrides["advantage_buffer_capacity"] = buffer_capacity
        overrides["strategy_buffer_capacity"] = buffer_capacity
    if device is not None:
        overrides["device"] = device
    elif gpu is not None:
        # Backward compat: --gpu/--no-gpu maps to device
        overrides["device"] = "cuda" if gpu else "cpu"
    if amp is not None:
        overrides["use_amp"] = amp
    if compile is not None:
        overrides["use_compile"] = compile
    if profile_step is not None:
        overrides["profile_step"] = profile_step

    # Bridge config.py DeepCfrConfig -> deep_trainer.py DeepCFRConfig with overrides
    dcfr_config = DeepCFRConfig.from_yaml_config(cfg, **overrides)

    total_steps = steps if steps is not None else 100

    try:
        infra = create_infrastructure(cfg, total_steps, headless=headless)
        exit_code = run_deep_training(
            cfg,
            dcfr_config,
            infra,
            steps=steps,
            checkpoint=str(checkpoint) if checkpoint else None,
            save_path=str(save_path) if save_path else None,
            headless=headless,
        )
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@train_app.command("psro", help="Train using Deep CFR with PSRO meta-loop")
def train_psro(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Number of training steps to run",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    population_size: Optional[int] = typer.Option(
        None,
        "--population-size",
        help="PSRO population size (rolling checkpoint window)",
    ),
    eval_games: Optional[int] = typer.Option(
        None,
        "--eval-games",
        help="Total games per PSRO population evaluation",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Disable Rich TUI; print plain-text progress to stdout.",
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Fix all random seeds (42) and force num_traversal_threads=1 for reproducible runs.",
    ),
):
    """Train with PSRO (Policy-Space Response Oracles) meta-loop."""
    from .config import load_config
    from .main_train import create_infrastructure, run_deep_training, handle_sigint
    from .cfr.deep_trainer import DeepCFRConfig

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    if not headless:
        headless = not sys.stdout.isatty()

    if deterministic:
        import random as _random
        import numpy as _np
        import torch as _torch

        _torch.manual_seed(42)
        _np.random.seed(42)
        _random.seed(42)
        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    # Force PSRO mode on
    overrides: dict = {"use_psro": True}
    if deterministic:
        overrides["num_traversal_threads"] = 1
    if population_size is not None:
        overrides["psro_population_size"] = population_size
    if eval_games is not None:
        overrides["psro_eval_games"] = eval_games

    dcfr_config = DeepCFRConfig.from_yaml_config(cfg, **overrides)

    total_steps = steps if steps is not None else 100

    try:
        infra = create_infrastructure(cfg, total_steps, headless=headless)
        exit_code = run_deep_training(
            cfg,
            dcfr_config,
            infra,
            steps=steps,
            checkpoint=str(checkpoint) if checkpoint else None,
            headless=headless,
        )
    except Exception as e:
        print(f"FATAL: Error during PSRO training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@train_app.command("rebel", help="Train using ReBeL (Recursive Belief-based Learning)")
def train_rebel(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Number of training iterations (overrides config rebel_epochs)",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
):
    """Train a ReBeL agent (2-player)."""
    from .cfr.rebel_trainer import ReBeLTrainer
    from .config import load_config

    setup_multiprocessing()

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    rebel_config = cfg.deep_cfr
    if device is not None:
        rebel_config.device = device

    ckpt_path = (
        str(save_path)
        if save_path
        else cfg.persistence.agent_data_save_path.replace(".joblib", "_rebel.pt")
    )

    trainer = ReBeLTrainer(
        config=rebel_config,
        checkpoint_path=ckpt_path,
        game_config=cfg.cambia_rules,
    )

    if checkpoint:
        trainer.load_checkpoint(str(checkpoint))

    total_iters = iterations if iterations is not None else rebel_config.rebel_epochs

    try:
        trainer.train(num_iterations=total_iters)
    except Exception as e:
        print(f"FATAL: Error during ReBeL training: {e}", file=sys.stderr)
        raise typer.Exit(1)


@train_app.command("gtcfr", help="Train using GT-CFR (Growing-Tree CFR, Phase 2)")
def train_gtcfr(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of training epochs (overrides config gtcfr_epochs)",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from GT-CFR checkpoint",
    ),
    warm_start: Optional[Path] = typer.Option(
        None,
        "--warm-start",
        help="Warm-start CVPN from a Phase 1 ReBeL checkpoint",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
):
    """Train a GT-CFR agent (Phase 2 search-based training)."""
    from .cfr.gtcfr_trainer import GTCFRTrainer
    from .config import load_config

    setup_multiprocessing()

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    gtcfr_config = cfg.deep_cfr
    if device is not None:
        gtcfr_config.device = device

    ckpt_path = (
        str(save_path)
        if save_path
        else str(Path(cfg.persistence.agent_data_save_path).parent / "gtcfr_checkpoint.pt")
    )

    trainer = GTCFRTrainer(
        config=gtcfr_config,
        game_config=cfg.cambia_rules,
        checkpoint_path=ckpt_path,
    )

    if warm_start:
        trainer.warm_start_from_rebel(str(warm_start))

    if checkpoint:
        trainer.load_checkpoint(str(checkpoint))

    total_epochs = epochs if epochs is not None else gtcfr_config.gtcfr_epochs

    try:
        trainer.train(num_epochs=total_epochs)
    except Exception as e:
        print(f"FATAL: Error during GT-CFR training: {e}", file=sys.stderr)
        raise typer.Exit(1)


@train_app.command("sog", help="Train using SoG (Student of Games, Phase 3)")
def train_sog(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Number of training epochs (overrides config sog_epochs)",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from SoG checkpoint",
    ),
    warm_start: Optional[Path] = typer.Option(
        None,
        "--warm-start",
        help="Warm-start CVPN from a GT-CFR or ReBeL checkpoint",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
):
    """Train a SoG agent (Phase 3 continual re-solving + budget decoupling)."""
    from .cfr.sog_trainer import SoGTrainer
    from .config import load_config

    setup_multiprocessing()

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    sog_config = cfg.deep_cfr
    if device is not None:
        sog_config.device = device

    ckpt_path = (
        str(save_path)
        if save_path
        else str(Path(cfg.persistence.agent_data_save_path).parent / "sog_checkpoint.pt")
    )

    trainer = SoGTrainer(
        config=sog_config,
        game_config=cfg.cambia_rules,
        checkpoint_path=ckpt_path,
    )

    if warm_start:
        warm_start_str = str(warm_start)
        # Detect checkpoint type by presence of sog_metadata or cvpn_state_dict
        try:
            import torch as _torch
            ckpt_peek = _torch.load(warm_start_str, map_location="cpu", weights_only=True)
            if "rebel_policy_net_state_dict" in ckpt_peek:
                trainer.warm_start_from_rebel(warm_start_str)
            else:
                trainer.warm_start_from_gtcfr(warm_start_str)
        except Exception:
            # Fallback: try GT-CFR format
            trainer.warm_start_from_gtcfr(warm_start_str)

    if checkpoint:
        trainer.load_checkpoint(str(checkpoint))

    total_epochs = epochs if epochs is not None else sog_config.sog_epochs

    try:
        trainer.train(num_epochs=total_epochs)
    except Exception as e:
        print(f"FATAL: Error during SoG training: {e}", file=sys.stderr)
        raise typer.Exit(1)


@train_app.command(
    "ppo", help="Train PPO: fair self-play (--self-play, E2 anchor) or best-response diagnostic"
)
def train_ppo_cmd(
    opponent: str = typer.Option(
        "imperfect_greedy",
        "--opponent",
        "-o",
        help="Fixed opponent agent type (best-response diagnostic). Ignored when --self-play is set.",
    ),
    self_play: bool = typer.Option(
        False,
        "--self-play",
        help="Fair self-play: opponent is a frozen-periodic snapshot of the learning policy (E2 anchor).",
    ),
    timesteps: int = typer.Option(
        500_000,
        "--timesteps",
        "-t",
        help="Total training timesteps",
    ),
    save_path: Path = typer.Option(
        "runs/ppo-diagnostic/model",
        "--save-path",
        "-s",
        help="Path to save trained model (runs/<run>/checkpoints/<model> derives the run dir)",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name for metrics.jsonl / run_db rows. Defaults to the run-dir basename.",
    ),
    n_envs: int = typer.Option(
        4,
        "--n-envs",
        help="Number of parallel training environments",
    ),
    eval_freq: int = typer.Option(
        10_000,
        "--eval-freq",
        help="Run per-baseline mean_imp eval every N timesteps",
    ),
    eval_games: int = typer.Option(
        5000,
        "--eval-games",
        help="Games per baseline in each eval cycle",
    ),
    eval_workers: Optional[int] = typer.Option(
        None,
        "--eval-workers",
        help="Parallel baseline eval workers (None = auto)",
    ),
    snapshot_freq: int = typer.Option(
        200_000,
        "--snapshot-freq",
        help="Refresh the self-play opponent snapshot every N timesteps (self-play only)",
    ),
    checkpoint_freq: Optional[int] = typer.Option(
        None,
        "--checkpoint-freq",
        help="Save a timestamped checkpoint every N timesteps (None disables periodic checkpoints)",
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
):
    """Train a PPO agent. With --self-play, the opponent is a frozen snapshot of
    the learning policy (the E2 equilibrium anchor, per-baseline persisted);
    otherwise the agent best-responds to a fixed baseline (diagnostic)."""
    from .ppo_train import train_ppo
    from .ppo_env import SELF_PLAY_OPPONENT

    opponent_arg = SELF_PLAY_OPPONENT if self_play else opponent

    train_ppo(
        opponent=opponent_arg,
        timesteps=timesteps,
        save_path=str(save_path),
        n_envs=n_envs,
        eval_freq=eval_freq,
        net_arch=[256, 256],
        seed=seed,
        config_path=str(config),
        run_name=run_name,
        eval_games=eval_games,
        selfplay_snapshot_freq=snapshot_freq,
        eval_max_workers=eval_workers,
        checkpoint_freq=checkpoint_freq,
    )


def _build_desca_python_env_factory(cfg):
    """Build the production DESCA env_factory backed by the Python CambiaGameState.

    Factored out of `train_desca` so the regression test in
    `tests/test_desca_env_factory_omniscient.py` can exercise the same code
    path. The classes are defined inside this builder (not at module scope)
    so that `_Agent.update`'s `isinstance(engine, _Engine)` check binds to
    the same class object the factory produces.
    """
    import copy
    from .game.engine import CambiaGameState
    from .agent_state import AgentState, AgentObservation
    from .constants import DecisionContext, ActionDiscard
    from .abstraction import get_card_bucket

    _counter = [0]

    class _Engine:
        def __init__(self, game):
            self._game = game
            self._last_actor = -1
            self._last_action = None

        def legal_actions(self):
            return sorted(self._game.get_legal_actions(), key=repr)

        def is_terminal(self):
            return self._game.is_terminal()

        def get_utility(self):
            if not self._game.is_terminal():
                return [0.0] * len(self._game.players)
            return [float(self._game.get_utility(i)) for i in range(len(self._game.players))]

        def get_acting_player(self):
            return int(self._game.current_player_index)

        def apply_action(self, action):
            self._last_actor = int(self._game.current_player_index)
            self._last_action = action
            try:
                self._game.apply_action(action)
            except Exception:
                pass

        def save(self):
            return copy.deepcopy(self._game)

        def restore(self, snap):
            self._game.__dict__.update(snap.__dict__)
            self._last_actor = -1
            self._last_action = None

        def free_snapshot(self, snap):
            pass

        def get_decision_context(self):
            if getattr(self._game, "snap_phase_active", False):
                return DecisionContext.SNAP_DECISION.value
            pending = getattr(self._game, "pending_action", None)
            if pending is not None:
                if isinstance(pending, ActionDiscard):
                    return DecisionContext.POST_DRAW.value
                return DecisionContext.ABILITY_SELECT.value
            return DecisionContext.START_TURN.value

        def get_drawn_card_bucket(self):
            return -1

        def _omniscient_features(self):
            """Return 120-dim (2P) omniscient feature vector reading Python game cards.

            Format mirrors `cfr.src.cfr.omniscient.compute_omniscient_features`:
            10-dim per slot (one-hot 0..8 for CardBucket, 9 for empty/unknown);
            slot order is `p * MaxHandSize + s` for p in [0, num_players),
            s in [0, MaxHandSize). Avoids the silent zero-fallback at
            `desca_worker._encode_omniscient` for the Python backend.
            """
            import numpy as _np
            _MAX_HAND_SIZE = 6  # matches engine.MaxHandSize on the Go side
            _PER_SLOT = 10
            num_players = len(self._game.players)
            feats = _np.zeros(num_players * _MAX_HAND_SIZE * _PER_SLOT, dtype=_np.float32)
            for p in range(num_players):
                hand = self._game.players[p].hand
                for s in range(_MAX_HAND_SIZE):
                    base = (p * _MAX_HAND_SIZE + s) * _PER_SLOT
                    if s >= len(hand) or hand[s] is None:
                        feats[base + 9] = 1.0
                        continue
                    v = get_card_bucket(hand[s]).value
                    if v >= 9:
                        feats[base + 9] = 1.0
                    else:
                        feats[base + v] = 1.0
            return feats

    class _Agent:
        def __init__(self, agent_state):
            object.__setattr__(self, "_agent", agent_state)

        def update(self, engine):
            if not isinstance(engine, _Engine):
                return
            if engine._last_action is None:
                return
            game = engine._game
            try:
                obs = AgentObservation(
                    acting_player=engine._last_actor,
                    action=engine._last_action,
                    discard_top_card=game.get_discard_top(),
                    player_hand_sizes=[
                        game.get_player_card_count(i)
                        for i in range(len(game.players))
                    ],
                    stockpile_size=game.get_stockpile_size(),
                    drawn_card=None,
                    peeked_cards=None,
                    snap_results=[],
                    did_cambia_get_called=False,
                    who_called_cambia=None,
                    is_game_over=game.is_terminal(),
                    current_turn=game.get_turn_number(),
                )
                object.__getattribute__(self, "_agent").update(obs)
            except Exception:
                pass

        def clone(self):
            return _Agent(copy.deepcopy(object.__getattribute__(self, "_agent")))

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_agent"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_agent"), name, value)

    memory_level = getattr(getattr(cfg, "agent_params", None), "memory_level", 1)
    time_decay_turns = getattr(getattr(cfg, "agent_params", None), "time_decay_turns", 3)

    def factory(rng=None):
        _counter[0] += 1
        game = CambiaGameState(house_rules=cfg.cambia_rules)
        engine = _Engine(game)
        num_players = len(game.players)
        init_obs = AgentObservation(
            acting_player=-1,
            action=None,
            discard_top_card=game.get_discard_top(),
            player_hand_sizes=[game.get_player_card_count(i) for i in range(num_players)],
            stockpile_size=game.get_stockpile_size(),
            drawn_card=None,
            peeked_cards=None,
            snap_results=[],
            did_cambia_get_called=False,
            who_called_cambia=None,
            is_game_over=False,
            current_turn=0,
        )
        agents = []
        for pid in range(num_players):
            initial_hand = list(game.players[pid].hand)
            initial_peeks = getattr(
                game.players[pid], "initial_peek_indices", tuple(range(len(initial_hand)))
            )
            agent_state = AgentState(
                player_id=pid,
                opponent_id=1 - pid,
                memory_level=memory_level,
                time_decay_turns=time_decay_turns,
                initial_hand_size=len(initial_hand),
                config=cfg,
            )
            agent_state.initialize(init_obs, initial_hand, initial_peeks)
            agents.append(_Agent(agent_state))
        return engine, agents

    return factory


def _build_desca_env_factory_for_test():
    """Test-only convenience wrapper. Builds a minimal DESCA config and returns
    the production env_factory. Used by `tests/test_desca_env_factory_omniscient.py`
    to verify the omniscient pipe end-to-end."""
    from .config import load_config
    from pathlib import Path as _Path
    cfg_path = _Path(__file__).parent.parent / "config" / "desca_phase1_rmplus.yaml"
    cfg = load_config(str(cfg_path))
    return _build_desca_python_env_factory(cfg)


def _build_desca_go_env_factory(cfg):
    """Build a DESCA env_factory backed by the Go FFI engine + agent.

    Production path. Eliminates `copy.deepcopy` of Python `CambiaGameState`
    in the hot loop and routes the omniscient critic through
    `GoEngine._get_all_cards_unsafe` (compute_omniscient_features uses this
    directly, no fallback needed).

    The adapter classes mirror the surface contract documented on
    `desca_worker.run_desca_iteration`:
      - engine: legal_actions, is_terminal, get_utility, get_acting_player,
                apply_action, save, restore, free_snapshot,
                get_decision_context, get_drawn_card_bucket. The FFI engine
                also exposes `_get_all_cards_unsafe` so
                `compute_omniscient_features` works directly without the
                `_omniscient_features` fallback.
      - agent:  update(engine), clone(); plus Python AgentState attrs needed
                by `cfr/src/action_abstraction.py`: own_hand (dict[slot ->
                _OwnInfo(bucket, last_seen_turn)]), opponent_belief (dict[slot
                -> CardBucket]), _current_game_turn (int).
    """
    import numpy as np
    from dataclasses import dataclass
    from .ffi.bridge import GoEngine, GoAgentState
    from .constants import (
        ActionAbilityBlindSwapSelect,
        ActionAbilityKingLookSelect,
        ActionAbilityKingSwapDecision,
        ActionAbilityPeekOtherSelect,
        ActionAbilityPeekOwnSelect,
        ActionCallCambia,
        ActionDiscard,
        ActionDrawDiscard,
        ActionDrawStockpile,
        ActionPassSnap,
        ActionReplace,
        ActionSnapOpponent,
        ActionSnapOpponentMove,
        ActionSnapOwn,
        CardBucket,
    )

    # --- 146-action index decoder (mirror of engine/types.go EncodeXxx) ---
    _IDX_DRAW_STOCKPILE = 0
    _IDX_DRAW_DISCARD = 1
    _IDX_CALL_CAMBIA = 2
    _IDX_DISCARD_NO_ABILITY = 3
    _IDX_DISCARD_ABILITY = 4
    _IDX_REPLACE_BASE = 5  # 5-10
    _IDX_PEEK_OWN_BASE = 11  # 11-16
    _IDX_PEEK_OTHER_BASE = 17  # 17-22
    _IDX_BLIND_SWAP_BASE = 23  # 23-58 (6x6=36)
    _IDX_KING_LOOK_BASE = 59  # 59-94 (6x6=36)
    _IDX_KING_SWAP_FALSE = 95
    _IDX_KING_SWAP_TRUE = 96
    _IDX_PASS_SNAP = 97
    _IDX_SNAP_OWN_BASE = 98  # 98-103
    _IDX_SNAP_OPP_BASE = 104  # 104-109
    _IDX_SNAP_OPP_MOVE_BASE = 110  # 110-145 (6x6=36)
    _MAX_HAND = 6

    def _index_to_named_action(idx: int):
        """Decode an action index in [0,146) to its NamedTuple."""
        if idx == _IDX_DRAW_STOCKPILE:
            return ActionDrawStockpile()
        if idx == _IDX_DRAW_DISCARD:
            return ActionDrawDiscard()
        if idx == _IDX_CALL_CAMBIA:
            return ActionCallCambia()
        if idx == _IDX_DISCARD_NO_ABILITY:
            return ActionDiscard(use_ability=False)
        if idx == _IDX_DISCARD_ABILITY:
            return ActionDiscard(use_ability=True)
        if _IDX_REPLACE_BASE <= idx < _IDX_PEEK_OWN_BASE:
            return ActionReplace(target_hand_index=idx - _IDX_REPLACE_BASE)
        if _IDX_PEEK_OWN_BASE <= idx < _IDX_PEEK_OTHER_BASE:
            return ActionAbilityPeekOwnSelect(target_hand_index=idx - _IDX_PEEK_OWN_BASE)
        if _IDX_PEEK_OTHER_BASE <= idx < _IDX_BLIND_SWAP_BASE:
            return ActionAbilityPeekOtherSelect(
                target_opponent_hand_index=idx - _IDX_PEEK_OTHER_BASE
            )
        if _IDX_BLIND_SWAP_BASE <= idx < _IDX_KING_LOOK_BASE:
            offset = idx - _IDX_BLIND_SWAP_BASE
            return ActionAbilityBlindSwapSelect(
                own_hand_index=offset // _MAX_HAND,
                opponent_hand_index=offset % _MAX_HAND,
            )
        if _IDX_KING_LOOK_BASE <= idx < _IDX_KING_SWAP_FALSE:
            offset = idx - _IDX_KING_LOOK_BASE
            return ActionAbilityKingLookSelect(
                own_hand_index=offset // _MAX_HAND,
                opponent_hand_index=offset % _MAX_HAND,
            )
        if idx == _IDX_KING_SWAP_FALSE:
            return ActionAbilityKingSwapDecision(perform_swap=False)
        if idx == _IDX_KING_SWAP_TRUE:
            return ActionAbilityKingSwapDecision(perform_swap=True)
        if idx == _IDX_PASS_SNAP:
            return ActionPassSnap()
        if _IDX_SNAP_OWN_BASE <= idx < _IDX_SNAP_OPP_BASE:
            return ActionSnapOwn(own_card_hand_index=idx - _IDX_SNAP_OWN_BASE)
        if _IDX_SNAP_OPP_BASE <= idx < _IDX_SNAP_OPP_MOVE_BASE:
            return ActionSnapOpponent(opponent_target_hand_index=idx - _IDX_SNAP_OPP_BASE)
        if _IDX_SNAP_OPP_MOVE_BASE <= idx < 146:
            offset = idx - _IDX_SNAP_OPP_MOVE_BASE
            return ActionSnapOpponentMove(
                own_card_to_move_hand_index=offset // _MAX_HAND,
                target_empty_slot_index=offset % _MAX_HAND,
            )
        raise ValueError(f"unrecognized action index {idx}")

    def _named_action_to_index(action) -> int:
        """Encode a NamedTuple action back to its 146-action index."""
        tag = getattr(action, "tag", None)
        if tag == "draw_stockpile":
            return _IDX_DRAW_STOCKPILE
        if tag == "draw_discard":
            return _IDX_DRAW_DISCARD
        if tag == "call_cambia":
            return _IDX_CALL_CAMBIA
        if tag == "discard":
            return _IDX_DISCARD_ABILITY if action.use_ability else _IDX_DISCARD_NO_ABILITY
        if tag == "replace":
            return _IDX_REPLACE_BASE + int(action.target_hand_index)
        if tag == "peek_own":
            return _IDX_PEEK_OWN_BASE + int(action.target_hand_index)
        if tag == "peek_other":
            return _IDX_PEEK_OTHER_BASE + int(action.target_opponent_hand_index)
        if tag == "blind_swap":
            return (
                _IDX_BLIND_SWAP_BASE
                + int(action.own_hand_index) * _MAX_HAND
                + int(action.opponent_hand_index)
            )
        if tag == "king_look":
            return (
                _IDX_KING_LOOK_BASE
                + int(action.own_hand_index) * _MAX_HAND
                + int(action.opponent_hand_index)
            )
        if tag == "king_swap":
            return _IDX_KING_SWAP_TRUE if action.perform_swap else _IDX_KING_SWAP_FALSE
        if tag == "pass_snap":
            return _IDX_PASS_SNAP
        if tag == "snap_own":
            return _IDX_SNAP_OWN_BASE + int(action.own_card_hand_index)
        if tag == "snap_opp":
            return _IDX_SNAP_OPP_BASE + int(action.opponent_target_hand_index)
        if tag == "snap_opp_move":
            return (
                _IDX_SNAP_OPP_MOVE_BASE
                + int(action.own_card_to_move_hand_index) * _MAX_HAND
                + int(action.target_empty_slot_index)
            )
        raise ValueError(f"unrecognized action tag {tag!r}")

    @dataclass
    class _OwnInfo:
        """Lightweight stand-in for Python AgentState.KnownCardInfo.

        action_abstraction.py only reads ``.bucket`` and ``.last_seen_turn``;
        we deliberately avoid exposing a Card attribute because we don't have
        actual card identity (only bucket) on the Go path. This is byte-equal
        to the inputs action_abstraction needs.
        """

        bucket: CardBucket
        last_seen_turn: int

    class _GoEngineAdapter:
        """Adapter mirroring the duck-typed contract on `desca_worker._traverse`.

        Wraps a `GoEngine`. Exposes:
          legal_actions, is_terminal, get_utility, get_acting_player,
          apply_action (NamedTuple), save, restore, free_snapshot,
          get_decision_context, get_drawn_card_bucket, _get_all_cards_unsafe.
        """

        def __init__(self, go_engine: GoEngine) -> None:
            self._engine = go_engine
            # `compute_omniscient_features` checks for `num_players`; expose it.
            self.num_players = int(getattr(go_engine, "_num_players", 2))

        def legal_actions(self):
            mask = self._engine.legal_actions_mask()
            indices = np.flatnonzero(mask)
            return [_index_to_named_action(int(i)) for i in indices]

        def is_terminal(self) -> bool:
            return self._engine.is_terminal()

        def get_utility(self):
            return self._engine.get_utility()

        def get_acting_player(self) -> int:
            return self._engine.acting_player()

        def apply_action(self, action) -> None:
            self._engine.apply_action(_named_action_to_index(action))

        def save(self) -> int:
            return self._engine.save()

        def restore(self, snap_h: int) -> None:
            self._engine.restore(snap_h)

        def free_snapshot(self, snap_h: int) -> None:
            self._engine.free_snapshot(snap_h)

        def get_decision_context(self) -> int:
            return self._engine.decision_ctx()

        def get_drawn_card_bucket(self) -> int:
            return self._engine.get_drawn_card_bucket()

        def _get_all_cards_unsafe(self) -> np.ndarray:
            """Bridge to GoEngine for compute_omniscient_features."""
            return self._engine._get_all_cards_unsafe()

        def close(self) -> None:
            try:
                self._engine.close()
            except Exception:
                pass

        # Expose the underlying GoEngine for tests / parity checks.
        @property
        def go_engine(self) -> GoEngine:
            return self._engine

    class _GoAgentStateAdapter:
        """Adapter wrapping a GoAgentState for DESCA traversal.

        Exposes the Python AgentState attribute surface that
        `cfr/src/action_abstraction.py` reads:
          own_hand: Dict[int, _OwnInfo]
          opponent_belief: Dict[int, CardBucket]
          _current_game_turn: int

        These are refreshed on every `update(engine)` call. `_go_agent` is
        also published so `cfr/src/encoding.py:encode_infoset_eppbs_interleaved_v2`
        can fast-path through the FFI v2 encoder (byte-equivalent to the
        slow Python path on equivalent state, by Phase 0 cross-validation).
        """

        def __init__(self, go_agent: GoAgentState) -> None:
            object.__setattr__(self, "_go_agent", go_agent)
            object.__setattr__(self, "own_hand", {})
            object.__setattr__(self, "opponent_belief", {})
            object.__setattr__(self, "_current_game_turn", 0)
            self._refresh()

        def _refresh(self) -> None:
            """Pull own_hand, opponent_belief, current_turn from the Go agent.

            Bucket value mapping (Go -> Python CardBucket):
              0..8 -> CardBucket(0..8) (ZERO..HIGH_KING)
              9    -> CardBucket.UNKNOWN (Python uses sentinel value 99 for UNKNOWN)
              0xFF -> CardBucket.UNKNOWN (Go sentinel for empty/out-of-range slot)
            """
            ga: GoAgentState = self._go_agent
            own_arr = ga.get_own_hand_buckets_and_seen()
            opp_arr = ga.get_opp_belief_buckets()
            own_len, opp_len = ga.get_hand_lens()
            current_turn = ga.get_current_turn()

            def _go_bucket_to_py(v: int) -> CardBucket:
                if v == 0xFF or v == 9:
                    return CardBucket.UNKNOWN
                if 0 <= v <= 8:
                    return CardBucket(v)
                return CardBucket.UNKNOWN

            new_own = {}
            for s in range(own_len):
                bucket = _go_bucket_to_py(int(own_arr[s, 0]))
                last_seen = int(own_arr[s, 1])
                new_own[s] = _OwnInfo(bucket=bucket, last_seen_turn=last_seen)

            new_opp = {}
            for s in range(opp_len):
                new_opp[s] = _go_bucket_to_py(int(opp_arr[s]))

            object.__setattr__(self, "own_hand", new_own)
            object.__setattr__(self, "opponent_belief", new_opp)
            object.__setattr__(self, "_current_game_turn", int(current_turn))

        def update(self, engine_adapter) -> None:
            """Update the Go-side belief state, then refresh cached attrs.

            Accepts either a `_GoEngineAdapter` or a raw GoEngine for
            flexibility; the GoAgentState.update API needs the raw GoEngine.
            """
            go_engine = getattr(engine_adapter, "go_engine", engine_adapter)
            self._go_agent.update(go_engine)
            self._refresh()

        def clone(self) -> "_GoAgentStateAdapter":
            new_go_agent = self._go_agent.clone()
            new_obj = _GoAgentStateAdapter.__new__(_GoAgentStateAdapter)
            object.__setattr__(new_obj, "_go_agent", new_go_agent)
            object.__setattr__(new_obj, "own_hand", dict(self.own_hand))
            object.__setattr__(new_obj, "opponent_belief", dict(self.opponent_belief))
            object.__setattr__(
                new_obj, "_current_game_turn", int(self._current_game_turn)
            )
            return new_obj

        def close(self) -> None:
            try:
                self._go_agent.close()
            except Exception:
                pass

    memory_level = getattr(getattr(cfg, "agent_params", None), "memory_level", 1)
    time_decay_turns = getattr(getattr(cfg, "agent_params", None), "time_decay_turns", 3)

    def factory(rng=None):
        seed = None
        if rng is not None:
            try:
                seed = int(rng.integers(0, 2**63 - 1))
            except Exception:
                seed = None
        go_engine = GoEngine(seed=seed, house_rules=cfg.cambia_rules)
        engine = _GoEngineAdapter(go_engine)
        num_players = int(go_engine._num_players)
        agents = []
        for pid in range(num_players):
            ga = GoAgentState(
                go_engine,
                pid,
                memory_level=memory_level,
                time_decay_turns=time_decay_turns,
            )
            agents.append(_GoAgentStateAdapter(ga))
        return engine, agents

    return factory


def _build_desca_env_factory_for_test_go():
    """Test-only convenience wrapper for the Go-backed env_factory.

    Mirrors `_build_desca_env_factory_for_test` so the parametrized
    regression tests can exercise both backends from one harness.
    """
    from .config import load_config
    from pathlib import Path as _Path
    cfg_path = _Path(__file__).parent.parent / "config" / "desca_phase1_rmplus.yaml"
    cfg = load_config(str(cfg_path))
    return _build_desca_go_env_factory(cfg)


def train_desca(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Number of training iterations (overrides config desca.iterations)",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from DESCA checkpoint",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save directory",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
    backend: str = typer.Option(
        "go",
        "--backend",
        help=(
            "DESCA env_factory backend: 'go' (default; Go FFI engine + agent) "
            "or 'python' (legacy CambiaGameState + AgentState; kept for "
            "fallback comparisons)."
        ),
    ),
):
    """Train a DESCA agent (v3.1 Dense ESCHER with Semantic Action Abstraction)."""
    import copy
    import logging as _logging

    from .config import load_config
    from .action_abstraction import NUM_ABSTRACT_ACTIONS_2P

    setup_multiprocessing()

    # Surface per-iter progress from the trainer's logger. Without this, the
    # default logging config suppresses INFO messages and training runs dark.
    _root_logger = _logging.getLogger()
    if not any(isinstance(h, _logging.StreamHandler) for h in _root_logger.handlers):
        _h = _logging.StreamHandler(sys.stderr)
        _h.setLevel(_logging.INFO)
        _h.setFormatter(_logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
        _root_logger.addHandler(_h)
    _logging.getLogger("src.cfr.desca_trainer").setLevel(_logging.INFO)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    if cfg.desca is None:
        print(
            "ERROR: Config missing [desca] section. "
            "Add a `desca:` block to your YAML or use a DESCA-specific config.",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    desca_cfg = cfg.desca

    if desca_cfg.num_abstract_actions != NUM_ABSTRACT_ACTIONS_2P:
        print(
            f"ERROR: config.desca.num_abstract_actions={desca_cfg.num_abstract_actions} "
            f"does not match action_abstraction.NUM_ABSTRACT_ACTIONS_2P={NUM_ABSTRACT_ACTIONS_2P}. "
            "Update your config to match the landed abstraction layer.",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    if iterations is not None:
        desca_cfg.iterations = iterations

    # Resolve device: CLI > "cpu". Avoid "auto" - resolve it concretely here.
    _raw_device = device or getattr(getattr(cfg, "deep_cfr", None), "device", None) or "cpu"
    if _raw_device == "auto":
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _device = "cuda"
            elif hasattr(_torch, "xpu") and _torch.xpu.is_available():
                _device = "xpu"
            else:
                _device = "cpu"
        except Exception:
            _device = "cpu"
    else:
        _device = _raw_device

    try:
        from .desca_networks import (
            AvgStrategyNetwork,
            HistoryValueNetwork,
            RegretNetwork,
        )
    except ImportError as e:
        print(f"ERROR: Could not import DESCA networks: {e}", file=sys.stderr)
        raise typer.Exit(1)

    try:
        from .cfr.desca_trainer import DESCATrainer
    except ImportError as e:
        print(f"ERROR: Could not import DESCATrainer: {e}", file=sys.stderr)
        print("Stream B (desca_trainer.py) must be merged before training.", file=sys.stderr)
        raise typer.Exit(1)

    # Build networks
    _hidden = desca_cfg.hidden_dim
    _n_abs = desca_cfg.num_abstract_actions
    regret_net = RegretNetwork(input_dim=257, hidden_dim=_hidden, num_actions=_n_abs)
    avg_strategy_net = AvgStrategyNetwork(input_dim=257, hidden_dim=_hidden, num_actions=_n_abs)
    history_value_net = HistoryValueNetwork(input_dim=257, omniscient_dim=120, hidden_dim=_hidden)

    # Build the production env_factory. Default backend is Go FFI: eliminates
    # `copy.deepcopy` of Python game/agent state in the worker hot loop and
    # routes the omniscient critic through GoEngine._get_all_cards_unsafe
    # directly. The Python backend stays callable via `--backend python` for
    # fallback comparisons (used by `tests/test_desca_env_factory_omniscient.py`).
    _backend = (backend or "go").strip().lower()
    if _backend not in ("go", "python"):
        print(
            f"ERROR: --backend must be 'go' or 'python', got {backend!r}.",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    if _backend == "go":
        env_factory = _build_desca_go_env_factory(cfg)
    else:
        env_factory = _build_desca_python_env_factory(cfg)

    # Checkpoint path: CLI --save-path > config.persistence.agent_data_save_path
    _ckpt_path = (
        str(save_path) if save_path else cfg.persistence.agent_data_save_path
    )
    _ckpt_dir = Path(_ckpt_path).parent
    _ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Write materialized config.yaml to run dir (resolves _base) so the run dir is
    # self-contained for `cambia evaluate runs/X/ --latest`.
    _run_dir = _ckpt_dir.parent if _ckpt_dir.name == "checkpoints" else _ckpt_dir
    _run_config_dst = _run_dir / "config.yaml"
    if not _run_config_dst.exists():
        try:
            from .config import resolve_config_yaml
            import yaml as _yaml
            _merged = resolve_config_yaml(str(config))
            with open(_run_config_dst, "w", encoding="utf-8") as _f:
                _yaml.safe_dump(_merged, _f, sort_keys=False)
        except Exception as _e:
            print(f"WARNING: could not write materialized config.yaml to run dir: {_e}", file=sys.stderr)

    trainer = DESCATrainer(
        desca_cfg,
        regret_net,
        avg_strategy_net,
        history_value_net,
        env_factory,
        device=_device,
        checkpoint_path=_ckpt_path,
    )

    if checkpoint:
        trainer.load_checkpoint(str(checkpoint))

    try:
        trainer.train(num_iterations=None)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint before exit...", file=sys.stderr)
        try:
            trainer.save_checkpoint()
        except Exception as e_save:
            print(f"WARNING: checkpoint save failed: {e_save}", file=sys.stderr)
        raise typer.Exit(0)
    except Exception as e:
        print(f"FATAL: Error during DESCA training: {e}", file=sys.stderr)
        raise typer.Exit(1)


# Register DESCA under all three official aliases (spec Section 5 / contract
# item 5). These share the same callback so `--help` is identical across names.
register_with_aliases(
    train_app,
    ["desca", "dense-escher", "dense_escher"],
    train_desca,
    help="Train using DESCA (Dense ESCHER + Semantic Action Abstraction, Phase 1)",
)


def train_prtcfr(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file (must carry a `prt_cfr:` block)",
        exists=True,
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Run name (e.g. v0.4-prtcfr-pilot); run dir is runs/<run-name>",
    ),
    iterations: Optional[int] = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Number of training iterations (overrides config prt_cfr.iterations)",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override the run directory (defaults to runs/<run-name>)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda (default: config prt_cfr.device)",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Production GameDriver backend: 'go' (default) or 'python'",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Warm-start from resume_state.json + the rolling checkpoint in the run dir",
    ),
):
    """Train using PRT-CFR (Perfect-Recall Trajectory CFR, v0.4 Phase 2)."""
    import logging as _logging

    from .config import load_config, resolve_config_yaml
    from .cfr import gpu_safety

    setup_multiprocessing()

    # Surface per-iter progress from the trainer's logger.
    _root_logger = _logging.getLogger()
    if not any(isinstance(h, _logging.StreamHandler) for h in _root_logger.handlers):
        _h = _logging.StreamHandler(sys.stderr)
        _h.setLevel(_logging.INFO)
        _h.setFormatter(
            _logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
        )
        _root_logger.addHandler(_h)
    _logging.getLogger("src.cfr.prtcfr_trainer").setLevel(_logging.INFO)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)
    if getattr(cfg, "prt_cfr", None) is None:
        print(
            "ERROR: Config missing [prt_cfr] section. Add a `prt_cfr:` block "
            "(see config/prtcfr_production.yaml) or use a PRT-CFR config.",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    prt_cfg = cfg.prt_cfr

    if iterations is not None:
        prt_cfg.iterations = iterations
    if backend is not None:
        _b = backend.strip().lower()
        if _b not in ("go", "python"):
            print(
                f"ERROR: --backend must be 'go' or 'python', got {backend!r}.",
                file=sys.stderr,
            )
            raise typer.Exit(1)
        prt_cfg.backend = _b

    # Resolve device: CLI > config > cpu. Resolve "auto" concretely.
    _raw_device = device or getattr(prt_cfg, "device", None) or "cpu"
    if _raw_device == "auto":
        try:
            import torch as _torch
            _device = "cuda" if _torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    else:
        _device = _raw_device
    prt_cfg.device = _device

    # Run directory: --save-path > runs/<run-name>.
    _name = run_name or "v0.4-prtcfr-run"
    _run_dir = Path(save_path) if save_path else (Path("runs") / _name)
    _run_dir.mkdir(parents=True, exist_ok=True)

    # Materialized config.yaml (resolves _base) so the run dir is self-contained
    # for `cambia evaluate runs/X/ --latest`.
    _config_dict = None
    _config_yaml = None
    _run_config_dst = _run_dir / "config.yaml"
    try:
        import yaml as _yaml
        _config_dict = resolve_config_yaml(str(config))
        _config_yaml = _yaml.safe_dump(_config_dict, sort_keys=False)
        if not _run_config_dst.exists():
            with open(_run_config_dst, "w", encoding="utf-8") as _f:
                _f.write(_config_yaml)
    except Exception as _e:
        print(f"WARNING: could not materialize config.yaml: {_e}", file=sys.stderr)

    # GPU failsafes for the run path (no-ops on CPU; keeps the caching allocator).
    if _device.startswith("cuda") and gpu_safety.cuda_available():
        try:
            gpu_safety.require_free_vram(2.0, label="prtcfr")
        except RuntimeError as _e:
            print(f"ERROR: {_e}", file=sys.stderr)
            raise typer.Exit(1)

    from .cfr.prtcfr_trainer import PRTCFRProductionTrainer

    trainer = PRTCFRProductionTrainer(
        prt_cfg,
        str(_run_dir),
        run_name=_name,
        config_yaml=_config_yaml,
        config_dict=_config_dict,
    )

    _iters = iterations if iterations is not None else prt_cfg.iterations

    # Thread --resume into trainer.train(resume=...) only if the trainer
    # actually accepts it. Guards against a hard TypeError while the trainer
    # side of resume-from-disk (a separate task) hasn't landed yet; --resume
    # without support fails with a clear message instead of a raw crash.
    import inspect as _inspect

    _train_kwargs = {"iterations": _iters}
    if "resume" in _inspect.signature(trainer.train).parameters:
        _train_kwargs["resume"] = resume
    elif resume:
        print(
            "ERROR: --resume requires a PRT-CFR trainer build with resume "
            "support (trainer.train() has no 'resume' parameter).",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    try:
        trainer.train(**_train_kwargs)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. The last completed iteration's snapshot + rolling "
            "checkpoint are on disk; reservoirs flushed.",
            file=sys.stderr,
        )
        raise typer.Exit(0)
    except Exception as e:
        print(f"FATAL: Error during PRT-CFR training: {e}", file=sys.stderr)
        raise typer.Exit(1)
    finally:
        trainer.close()


register_with_aliases(
    train_app,
    ["prtcfr", "prt-cfr", "prt_cfr"],
    train_prtcfr,
    help="Train using PRT-CFR (Perfect-Recall Trajectory CFR, v0.4 Phase 2)",
)

# Dashed aliases for existing primary commands.
register_with_aliases(
    train_app,
    ["gt-cfr"],
    train_gtcfr,
    help="Alias for `gtcfr`.",
)


def train_sd_cfr(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Number of training steps to run",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Disable Rich TUI; print plain-text progress to stdout.",
    ),
):
    """Train using SD-CFR (Single Deep CFR; no strategy network)."""
    from .config import load_config
    from .main_train import create_infrastructure, run_deep_training, handle_sigint
    from .cfr.deep_trainer import DeepCFRConfig

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    if not headless:
        headless = not sys.stdout.isatty()

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    overrides: dict = {"use_sd_cfr": True}
    if device is not None:
        overrides["device"] = device

    dcfr_config = DeepCFRConfig.from_yaml_config(cfg, **overrides)
    total_steps = steps if steps is not None else 100

    try:
        infra = create_infrastructure(cfg, total_steps, headless=headless)
        exit_code = run_deep_training(
            cfg,
            dcfr_config,
            infra,
            steps=steps,
            checkpoint=str(checkpoint) if checkpoint else None,
            save_path=str(save_path) if save_path else None,
            headless=headless,
        )
    except Exception as e:
        print(f"FATAL: Error during SD-CFR training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


def train_os_mccfr(
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Number of training steps to run",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Resume from checkpoint",
        exists=True,
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        "-s",
        help="Override checkpoint save path",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Training device: auto, cpu, cuda, xpu (default: auto)",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Disable Rich TUI; print plain-text progress to stdout.",
    ),
):
    """Train using Outcome-Sampling MCCFR."""
    from .config import load_config
    from .main_train import create_infrastructure, run_deep_training, handle_sigint
    from .cfr.deep_trainer import DeepCFRConfig

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    if not headless:
        headless = not sys.stdout.isatty()

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    overrides: dict = {"sampling_method": "os"}
    if device is not None:
        overrides["device"] = device

    dcfr_config = DeepCFRConfig.from_yaml_config(cfg, **overrides)
    total_steps = steps if steps is not None else 100

    try:
        infra = create_infrastructure(cfg, total_steps, headless=headless)
        exit_code = run_deep_training(
            cfg,
            dcfr_config,
            infra,
            steps=steps,
            checkpoint=str(checkpoint) if checkpoint else None,
            save_path=str(save_path) if save_path else None,
            headless=headless,
        )
    except Exception as e:
        print(f"FATAL: Error during OS-MCCFR training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


register_with_aliases(
    train_app,
    ["sd-cfr", "sdcfr", "sd_cfr"],
    train_sd_cfr,
    help="Train using SD-CFR (Single Deep CFR; no strategy network)",
)

register_with_aliases(
    train_app,
    ["os-mccfr", "osmccfr", "os_mccfr"],
    train_os_mccfr,
    help="Train using Outcome-Sampling MCCFR",
)


@app.command("resume", help="Resume training from a checkpoint")
def resume(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
        exists=True,
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-n",
        help="Additional steps/iterations to run",
    ),
):
    """Resume training from a checkpoint (auto-detects type)."""
    from .config import load_config
    from .main_train import (
        create_infrastructure,
        run_tabular_training,
        run_deep_training,
        handle_sigint,
    )

    setup_multiprocessing()
    signal.signal(signal.SIGINT, handle_sigint)

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    suffix = checkpoint.suffix.lower()

    try:
        if suffix == ".pt":
            # Deep CFR checkpoint
            from .cfr.deep_trainer import DeepCFRConfig

            dcfr_config = DeepCFRConfig.from_yaml_config(cfg)
            total_steps = steps if steps is not None else 100
            infra = create_infrastructure(cfg, total_steps)
            exit_code = run_deep_training(
                cfg,
                dcfr_config,
                infra,
                steps=steps,
                checkpoint=str(checkpoint),
            )
        elif suffix == ".joblib":
            # Tabular CFR checkpoint
            total_iterations = (
                steps if steps is not None else cfg.cfr_training.num_iterations
            )
            infra = create_infrastructure(cfg, total_iterations)
            exit_code = run_tabular_training(
                cfg,
                infra,
                iterations=steps,
                load=True,
                save_path=str(checkpoint),
            )
        else:
            print(f"ERROR: Unknown checkpoint type: {suffix}", file=sys.stderr)
            print(
                "Expected .pt (Deep CFR) or .joblib (Tabular CFR)", file=sys.stderr
            )
            raise typer.Exit(1)
    except Exception as e:
        print(f"FATAL: Error during training: {e}", file=sys.stderr)
        raise typer.Exit(1)

    raise typer.Exit(exit_code)


@app.command("info", help="Display checkpoint metadata")
def info(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint file",
        exists=True,
    ),
):
    """Display checkpoint metadata."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    suffix = checkpoint.suffix.lower()

    if suffix == ".pt":
        import torch

        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

            table = Table(title=f"Deep CFR Checkpoint: {checkpoint.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Training Step", str(ckpt.get("training_step", "N/A")))
            table.add_row(
                "Total Traversals", str(ckpt.get("total_traversals", "N/A"))
            )

            if "dcfr_config" in ckpt:
                dcfr = ckpt["dcfr_config"]
                table.add_row("Learning Rate", str(dcfr.get("learning_rate", "N/A")))
                table.add_row("Batch Size", str(dcfr.get("batch_size", "N/A")))
                table.add_row("Hidden Dim", str(dcfr.get("hidden_dim", "N/A")))
                table.add_row("Alpha", str(dcfr.get("alpha", "N/A")))

            adv_history = ckpt.get("advantage_loss_history", [])
            if adv_history:
                _, last_loss = adv_history[-1]
                table.add_row("Last Advantage Loss", f"{last_loss:.6f}")

            strat_history = ckpt.get("strategy_loss_history", [])
            if strat_history:
                _, last_loss = strat_history[-1]
                table.add_row("Last Strategy Loss", f"{last_loss:.6f}")

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading checkpoint:[/red] {e}")
            raise typer.Exit(1)

    elif suffix == ".joblib":
        import joblib

        try:
            data = joblib.load(checkpoint)

            table = Table(title=f"Tabular CFR Checkpoint: {checkpoint.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            if isinstance(data, dict):
                table.add_row("Iteration", str(data.get("iteration", "N/A")))
                table.add_row(
                    "Infoset Count", str(len(data.get("regret_sum", {})))
                )

                if "exploitability_history" in data:
                    history = data["exploitability_history"]
                    if history:
                        table.add_row(
                            "Recent Exploitability", f"{history[-1]:.6f}"
                        )
            else:
                table.add_row("Type", str(type(data).__name__))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error loading checkpoint:[/red] {e}")
            raise typer.Exit(1)

    else:
        console.print(f"[red]ERROR: Unknown checkpoint type:[/red] {suffix}")
        console.print("Expected .pt (Deep CFR) or .joblib (Tabular CFR)")
        raise typer.Exit(1)


# Benchmark subcommand group
benchmark_app = typer.Typer(
    help="Run performance benchmarks",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("all")
def benchmark_all(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda/xpu)",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Also save raw JSON output",
    ),
):
    """Run all benchmarks."""
    from .benchmarks.runner import BenchmarkSuite
    from .benchmarks import network_bench
    from .benchmarks.traversal_bench import benchmark_traversal
    from .benchmarks.worker_scaling import benchmark_worker_scaling
    from .benchmarks.memory_bench import benchmark_memory
    from .benchmarks.e2e_bench import benchmark_e2e

    suite = BenchmarkSuite()
    suite.register(network_bench.benchmark_network_performance, "network")
    suite.register(benchmark_traversal, "traversal")
    suite.register(benchmark_worker_scaling, "scaling")
    suite.register(benchmark_memory, "memory")
    suite.register(benchmark_e2e, "e2e")

    results = suite.run_all(
        output_dir=str(output_dir),
        device=device,
        config_path=str(config),
    )

    print(f"\nBenchmark suite complete. Results saved to {output_dir}")


@benchmark_app.command("network")
def benchmark_network_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda/xpu)",
    ),
    batch_sizes: Optional[str] = typer.Option(
        None,
        "--batch-sizes",
        help="Comma-separated list of batch sizes (e.g., 256,512,1024)",
    ),
):
    """Run network performance benchmarks."""
    from datetime import datetime
    from .benchmarks import network_bench
    from .benchmarks.reporting import print_result
    import json

    batch_size_list = None
    if batch_sizes:
        batch_size_list = [int(x.strip()) for x in batch_sizes.split(",")]

    result = network_bench.benchmark_network_performance(
        device=device, batch_sizes=batch_size_list
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "network"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("traversal")
def benchmark_traversal_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    num_traversals: int = typer.Option(
        20,
        "--num-traversals",
        "-n",
        help="Number of traversals to run",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run traversal benchmarks."""
    from datetime import datetime
    from .benchmarks.traversal_bench import benchmark_traversal
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_traversal(
        num_traversals=num_traversals,
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "traversal"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("scaling")
def benchmark_scaling_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    worker_counts: Optional[str] = typer.Option(
        None,
        "--worker-counts",
        help="Comma-separated list of worker counts (e.g., 1,2,4,8)",
    ),
    traversals: int = typer.Option(
        100,
        "--traversals",
        "-t",
        help="Traversals per test",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run worker scaling benchmarks."""
    from datetime import datetime
    from .benchmarks.worker_scaling import benchmark_worker_scaling
    from .benchmarks.reporting import print_result
    import json

    worker_count_list = None
    if worker_counts:
        worker_count_list = [int(x.strip()) for x in worker_counts.split(",")]

    result = benchmark_worker_scaling(
        worker_counts=worker_count_list,
        traversals_per_test=traversals,
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "scaling"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("memory")
def benchmark_memory_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run memory profiling benchmarks."""
    from datetime import datetime
    from .benchmarks.memory_bench import benchmark_memory
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_memory(
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "memory"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("e2e")
def benchmark_e2e_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device to run on (cpu/cuda/xpu)",
    ),
    num_steps: int = typer.Option(
        2,
        "--num-steps",
        "-n",
        help="Number of training steps to benchmark",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="Number of parallel workers",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run end-to-end training step benchmark."""
    from datetime import datetime
    from .benchmarks.e2e_bench import benchmark_e2e
    from .benchmarks.reporting import print_result
    import json

    result = benchmark_e2e(
        num_steps=num_steps,
        device=device,
        num_workers=num_workers,
        config_path=str(config),
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "e2e"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


@benchmark_app.command("es")
def benchmark_es_cmd(
    output_dir: Path = typer.Option(
        "/workspace/benchmarks",
        "--output-dir",
        "-o",
        help="Output directory for benchmark results",
    ),
    depths: Optional[str] = typer.Option(
        "5,8,10,12",
        "--depths",
        help="Comma-separated list of depth limits (e.g., 5,8,10,12)",
    ),
    num_traversals: int = typer.Option(
        10,
        "--num-traversals",
        "-n",
        help="Number of traversals per (backend, depth) combination",
    ),
    backends: Optional[str] = typer.Option(
        "python,go",
        "--backends",
        help="Comma-separated list of backends to test (python,go)",
    ),
    config: Path = typer.Option(
        "/workspace/config/parallel.config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Run ES exploitability benchmarks across backends and depth limits."""
    from datetime import datetime
    from .benchmarks.es_bench import benchmark_es
    from .benchmarks.reporting import print_result
    import json

    depth_list = [int(d.strip()) for d in depths.split(",")] if depths else None
    backend_list = [b.strip() for b in backends.split(",")] if backends else None

    result = benchmark_es(
        depths=depth_list,
        num_traversals=num_traversals,
        backends=backend_list,
        config_path=str(config),
        device="cpu",
    )

    print_result(result)

    # Save to timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = output_dir / timestamp / "es"
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {json_path}")


# Run-dir checkpoint discovery (used by `evaluate` below in --latest/--epoch
# mode). Most algorithms (CFR family) write epoch/iter-numbered .pt files.
# PPO (sb3-contrib) is the exception: two independent training callbacks
# (see ppo_train.py) save step-numbered .zip archives off the same global
# timestep counter: "<prefix>_steps_<N>.zip" (periodic) and
# "<prefix>_eval_<N>.zip" (post-eval; already has a persisted mean_imp row
# in run_db/metrics.jsonl at that step).
_ZIP_CHECKPOINT_ALGOS = {"ppo"}

# Legacy prefix aliases for pre-2026-03-09 .pt-naming runs.
_LEGACY_CHECKPOINT_PREFIXES = {
    "gtcfr_checkpoint": ["checkpoint_gtcfr"],
    "sog_checkpoint": ["checkpoint_sog_sog", "checkpoint_sog"],
}


def _checkpoint_iteration_from_name(name: str) -> int:
    """Extract the epoch/iter/steps/eval training-step number from a checkpoint name.

    Shared by run-dir checkpoint discovery (--latest/--epoch) and eval-results
    persistence (the `iteration` column), so PPO's steps_/eval_ naming and the
    CFR-family epoch_/iter_ naming both parse into the same field.
    """
    import re

    m = re.search(r"(?:epoch|iter|steps|eval)_(\d+)", name)
    return int(m.group(1)) if m else 0


def _find_run_dir_checkpoints(ckpt_dir: Path, prefix: str, algorithm: str) -> List[Path]:
    """Glob run_dir/checkpoints/ for files matching an algorithm's naming convention.

    CFR-family algorithms: "<prefix>*epoch_<N>.pt" or "<prefix>*iter_<N>.pt",
    with legacy prefix aliases. PPO: "<prefix>*steps_<N>.zip" or
    "<prefix>*eval_<N>.zip" (sb3-contrib .zip saves).
    """
    if algorithm == "prt-cfr":
        # PRT-CFR writes per-iteration regret-net SNAPSHOTS (the SD-CFR mixture
        # inputs), not epoch/iter-numbered rolling checkpoints. They live in
        # run_dir/snapshots as prtcfr_snapshot_iter_{t}.pt.
        patterns = ["prtcfr_snapshot_iter_*.pt"]
    elif algorithm in _ZIP_CHECKPOINT_ALGOS:
        patterns = [f"*{prefix}*steps_*.zip", f"*{prefix}*eval_*.zip"]
    else:
        prefixes = [prefix] + _LEGACY_CHECKPOINT_PREFIXES.get(prefix, [])
        patterns = [
            f"*{p}*{suffix}"
            for p in prefixes
            for suffix in ["epoch_*.pt", "iter_*.pt"]
        ]
    return sorted(set(match for pattern in patterns for match in ckpt_dir.glob(pattern)))


def _extract_checkpoint_num(p: Path) -> tuple:
    """Sort key for --latest checkpoint selection: (step number, tie-break).

    On an exact numeric tie between a PPO "eval_" and "steps_" checkpoint (the
    two callbacks can coincide if eval_freq divides checkpoint_freq evenly),
    the eval_ checkpoint wins: it already carries a persisted mean_imp row in
    run_db/metrics.jsonl from training, so re-evaluating it corroborates a
    known result rather than blindly evaluating an unlogged snapshot.
    """
    tie_break = 1 if "_eval_" in p.name else 0
    return (_checkpoint_iteration_from_name(p.name), tie_break)


def _checkpoint_matches_epoch(p: Path, epoch: int, algorithm: str) -> bool:
    """Check whether a checkpoint filename matches the requested --epoch/N target."""
    if algorithm in _ZIP_CHECKPOINT_ALGOS:
        return f"steps_{epoch}.zip" in p.name or f"eval_{epoch}.zip" in p.name
    return f"epoch_{epoch}.pt" in p.name or f"iter_{epoch}.pt" in p.name


@app.command("evaluate", help="Evaluate a Deep CFR checkpoint against baseline agents")
def evaluate(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to .pt checkpoint file, or path to a run directory",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file (auto-detected in run-dir mode)",
    ),
    games: int = typer.Option(
        100,
        "--games",
        "-n",
        help="Number of games per baseline matchup (default: 5000 in run-dir mode)",
    ),
    baselines: str = typer.Option(
        None,
        "--baselines",
        "-b",
        help="Comma-separated list of baseline agents to evaluate against "
        "(choices: random, random_no_cambia, random_late_cambia, greedy, imperfect_greedy, memory_heuristic, aggressive_snap, cfr). "
        "Defaults to the full MEAN_IMP_BASELINES set.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Torch device for inference (cpu, cuda, or xpu)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for per-baseline JSONL game logs",
    ),
    argmax: bool = typer.Option(
        False,
        "--argmax",
        help="Use argmax action selection instead of stochastic sampling during evaluation",
    ),
    agent_type: str = typer.Option(
        "deep_cfr",
        "--agent-type",
        help="Agent type: deep_cfr, rebel, sd_cfr, escher, gtcfr, sog, sog_inference",
    ),
    latest: bool = typer.Option(
        False,
        "--latest",
        help="Auto-select the highest-numbered checkpoint in the run directory",
    ),
    epoch: Optional[int] = typer.Option(
        None,
        "--epoch",
        help="Select a specific checkpoint by epoch/iteration number",
    ),
    lbr: bool = typer.Option(
        False,
        "--lbr",
        help="Run sampled LBR exploitability measurement after evaluation",
    ),
    lbr_infosets: int = typer.Option(
        10000,
        "--lbr-infosets",
        help="Number of infosets to sample for LBR",
    ),
    lbr_rollouts: int = typer.Option(
        100,
        "--lbr-rollouts",
        help="Rollouts per infoset for LBR",
    ),
    lbr_tier: str = typer.Option(
        "A",
        "--lbr-tier",
        help=(
            "LBR tier: 'A' (random rollouts, loose lower bound) or 'B' "
            "(agent-policy rollouts vs a strong opponent, tighter bound)."
        ),
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        "-j",
        help="Parallel baseline workers (default: auto, set 1 for sequential)",
    ),
):
    """Evaluate a checkpoint against baseline agents and print a win-rate table."""
    from rich.console import Console
    from rich.table import Table
    from .evaluate_agents import run_evaluation_multi_baseline, MEAN_IMP_BASELINES

    run_dir = None
    checkpoint_path = None
    games_was_default = (games == 100)

    if checkpoint.is_dir():
        # Run-dir mode
        run_dir = checkpoint
        run_config = run_dir / "config.yaml"
        if not run_config.exists():
            print(f"ERROR: No config.yaml found in {run_dir}", file=sys.stderr)
            raise typer.Exit(1)

        if config is None:
            config = run_config

        # Auto-detect algorithm and agent type
        from .run_db import infer_algorithm, algo_to_agent_type, algo_to_checkpoint_prefix
        from .config import resolve_config_yaml

        try:
            config_dict = resolve_config_yaml(run_config)
        except FileNotFoundError as e:
            # _base unresolvable (legacy run dir without co-located base or fallback hit).
            # Fall back to raw load; algorithm detection will rely on checkpoint filename below.
            print(f"WARNING: {e}", file=sys.stderr)
            import yaml as _yaml
            with open(run_config, encoding="utf-8") as f:
                config_dict = _yaml.safe_load(f) or {}

        # Initial algorithm guess from config; refined below using checkpoint filename.
        algorithm = infer_algorithm(config_dict)

        # Sample any checkpoint filename to refine algorithm detection (handles the case
        # where config is unmaterialized and `algorithm` field is hidden behind _base).
        chk_ckpt_dir = run_dir / "checkpoints"
        if chk_ckpt_dir.exists():
            sample_ckpts = sorted(chk_ckpt_dir.glob("*_checkpoint*.pt"))
            if sample_ckpts:
                algorithm = infer_algorithm(config_dict, checkpoint_filename=sample_ckpts[0].name)

        # Use auto-detected values unless user explicitly overrode
        if agent_type == "deep_cfr":
            agent_type = algo_to_agent_type(algorithm)

        prefix = algo_to_checkpoint_prefix(algorithm)

        # PRT-CFR writes per-iteration regret-net snapshots to run_dir/snapshots
        # (the SD-CFR mixture inputs), not to run_dir/checkpoints; every other
        # algorithm uses checkpoints/.
        ckpt_dir = run_dir / ("snapshots" if algorithm == "prt-cfr" else "checkpoints")

        # Default to 5000 games in run-dir mode (but respect explicit user override)
        if games_was_default:
            games = 5000

        # Resolve checkpoint
        if not ckpt_dir.exists():
            print(f"ERROR: No {ckpt_dir.name}/ directory in {run_dir}", file=sys.stderr)
            raise typer.Exit(1)

        all_ckpts = _find_run_dir_checkpoints(ckpt_dir, prefix, algorithm)

        if latest:
            if not all_ckpts:
                print(f"ERROR: No checkpoints matching prefix '{prefix}' found", file=sys.stderr)
                raise typer.Exit(1)
            all_ckpts.sort(key=_extract_checkpoint_num)
            checkpoint_path = all_ckpts[-1]
        elif epoch is not None:
            matches = [p for p in all_ckpts if _checkpoint_matches_epoch(p, epoch, algorithm)]
            if not matches:
                print(f"ERROR: No checkpoint for epoch/iter {epoch} found", file=sys.stderr)
                raise typer.Exit(1)
            checkpoint_path = matches[0]
        else:
            print("ERROR: Run-dir mode requires --latest or --epoch N", file=sys.stderr)
            raise typer.Exit(1)

        # Refine algorithm detection with checkpoint keys
        try:
            import torch

            ckpt_data = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            ckpt_keys = set(ckpt_data.keys())
            refined_algo = infer_algorithm(config_dict, checkpoint_keys=ckpt_keys)
            if agent_type == algo_to_agent_type(algorithm):
                agent_type = algo_to_agent_type(refined_algo)
            del ckpt_data
        except Exception:
            pass

        checkpoint = checkpoint_path
    else:
        # File mode
        if not checkpoint.exists():
            print(f"ERROR: Checkpoint file not found: {checkpoint}", file=sys.stderr)
            raise typer.Exit(1)
        checkpoint_path = checkpoint
        # Try to detect run_dir from checkpoint path
        if checkpoint.parent.name == "checkpoints" and (checkpoint.parent.parent / "config.yaml").exists():
            run_dir = checkpoint.parent.parent

        if config is None:
            # Fall back to config.yaml in cwd
            config = Path("config.yaml")
            if not config.exists():
                print("ERROR: No --config specified and no config.yaml in current directory.", file=sys.stderr)
                raise typer.Exit(1)

    if not config.exists():
        print(f"ERROR: Config file not found: {config}", file=sys.stderr)
        raise typer.Exit(1)

    if baselines is None:
        baseline_list = list(MEAN_IMP_BASELINES)
    else:
        baseline_list = [b.strip() for b in baselines.split(",") if b.strip()]
    if not baseline_list:
        print("ERROR: No baselines specified.", file=sys.stderr)
        raise typer.Exit(1)

    results_map = run_evaluation_multi_baseline(
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        num_games=games,
        baselines=baseline_list,
        device=device,
        output_dir=str(output_dir) if output_dir else None,
        use_argmax=argmax,
        agent_type=agent_type,
        max_workers=max_workers,
    )

    console = Console()
    table = Table(title=f"Evaluation: {checkpoint.name}")
    table.add_column("Baseline", style="cyan")
    table.add_column(f"{agent_type} Wins", style="green")
    table.add_column("Baseline Wins", style="red")
    table.add_column("Ties", style="yellow")
    table.add_column("Errors", style="dim")
    table.add_column("Win Rate", style="bold green")
    table.add_column("Avg Margin", style="blue")
    table.add_column("Avg Turns", style="magenta")

    for baseline, results in results_map.items():
        p0_wins = results.get("P0 Wins", 0)
        p1_wins = results.get("P1 Wins", 0)
        ties = results.get("Ties", 0) + results.get("MaxTurnTies", 0)
        errors = results.get("Errors", 0)
        total = p0_wins + p1_wins + ties
        win_rate = f"{p0_wins / total * 100:.1f}%" if total > 0 else "N/A"
        stats = getattr(results, "stats", {})
        avg_margin = stats.get("avg_score_margin")
        avg_turns = stats.get("avg_game_turns")
        std_turns = stats.get("std_game_turns")
        margin_str = f"{avg_margin:.1f}" if avg_margin is not None else "N/A"
        turns_str = (
            f"{avg_turns:.1f}±{std_turns:.1f}"
            if avg_turns is not None and std_turns is not None
            else "N/A"
        )
        table.add_row(
            baseline,
            str(p0_wins),
            str(p1_wins),
            str(ties),
            str(errors),
            win_rate,
            margin_str,
            turns_str,
        )

    console.print(table)

    # Eval-hygiene provenance line: seat scheme, selection mode, and whether the
    # agent under test genuinely played both seats (seat_balanced).
    _first_stats = next(
        (getattr(r, "stats", {}) for r in results_map.values()), {}
    )
    _seat_scheme = _first_stats.get("seat_scheme", "alternated")
    _selection_mode = _first_stats.get(
        "selection_mode", "argmax" if argmax else "stochastic"
    )
    _seat_balanced = bool(_first_stats.get("seat_balanced", False))
    console.print(
        f"[dim]seat_scheme={_seat_scheme}  seat_balanced={_seat_balanced}  "
        f"selection_mode={_selection_mode}[/dim]"
    )

    # Persist results if we have a run directory context
    if run_dir is not None:
        from .evaluate_agents import persist_eval_results

        iteration = _checkpoint_iteration_from_name(str(checkpoint))
        persist_eval_results(
            run_dir=str(run_dir),
            iteration=iteration,
            results_map=results_map,
            checkpoint_path=str(checkpoint),
            selection_mode="argmax" if argmax else "stochastic",
            seat_scheme="alternated",
        )
        print(f"\nResults persisted to {run_dir}/metrics.jsonl")

    if lbr:
        from .config import load_config as _load_config
        from .evaluate_agents import get_agent

        tier = (lbr_tier or "A").strip().upper()
        if tier not in ("A", "B"):
            print(f"[lbr] ERROR: unknown --lbr-tier {lbr_tier!r}", file=sys.stderr)
            tier = "A"

        lbr_config = _load_config(str(config))
        if lbr_config is None:
            print("[lbr] ERROR: could not load config for LBR.", file=sys.stderr)
        else:
            agent = get_agent(
                agent_type,
                player_id=0,
                config=lbr_config,
                checkpoint_path=str(checkpoint),
                device=device,
            )
            if tier == "B":
                from .cfr.lbr import tier_b_lbr

                result = tier_b_lbr(
                    agent,
                    lbr_config,
                    num_infosets=lbr_infosets,
                    br_rollouts_per_infoset=lbr_rollouts,
                )
                opp = result.get("rollout_opponent", "?")
                print(
                    f"[lbr] tier=B exploitability={result['exploitability']:.3f} "
                    f"({result['num_infosets_sampled']} infosets, "
                    f"stderr={result['std_err']:.3f}, opp={opp})"
                )
            else:
                from .cfr.sampled_lbr import sampled_lbr as run_lbr

                result = run_lbr(
                    agent,
                    lbr_config,
                    num_infosets=lbr_infosets,
                    br_rollouts_per_infoset=lbr_rollouts,
                )
                print(
                    f"[lbr] tier=A exploitability={result['exploitability']:.3f} "
                    f"({result['num_infosets_sampled']} infosets, "
                    f"stderr={result['std_err']:.3f})"
                )


@app.command(
    "eval-watch",
    help="Poll run dir(s) and auto-evaluate new checkpoints as they appear (CPU-only).",
)
def eval_watch(
    run_dirs: List[Path] = typer.Argument(
        ...,
        help="One or more run directories to watch for new checkpoints.",
    ),
    games: int = typer.Option(
        5000,
        "--games",
        "-n",
        help="Games per baseline per checkpoint (default: 5000).",
    ),
    poll_interval: int = typer.Option(
        30,
        "--poll-interval",
        help="Seconds between polling cycles (default: 30).",
    ),
    h2h_games: int = typer.Option(
        2000,
        "--h2h-games",
        help="Games per head-to-head comparison (default: 2000, 0 to disable).",
    ),
    agent_type: str = typer.Option(
        "deep_cfr",
        "--agent-type",
        help="Agent type to evaluate (default: deep_cfr). Choices: deep_cfr, rebel, sd_cfr, escher.",
    ),
    checkpoint_prefix: Optional[str] = typer.Option(
        None,
        "--checkpoint-prefix",
        help=(
            "Filename prefix for checkpoint glob (auto-inferred from --agent-type if omitted). "
            "E.g. 'rebel_checkpoint' for ReBeL."
        ),
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        "-j",
        help="Parallel baseline workers per checkpoint (default: auto, set 1 for sequential)",
    ),
):
    """
    Watch run directories and evaluate every new checkpoint against all baselines.

    Writes metrics.jsonl and (optionally) head_to_head.jsonl into each run directory.
    Supports all agent types via --agent-type (deep_cfr, rebel, gtcfr, sog, etc.).
    """
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parent.parent / "scripts" / "eval_watcher.py"
    cmd = [
        _sys.executable,
        str(script),
        "--run-dirs",
        *[str(d) for d in run_dirs],
        "--games",
        str(games),
        "--poll-interval",
        str(poll_interval),
        "--h2h-games",
        str(h2h_games),
        "--agent-type",
        agent_type,
    ]
    if checkpoint_prefix:
        cmd += ["--checkpoint-prefix", checkpoint_prefix]
    if max_workers is not None:
        cmd += ["--max-workers", str(max_workers)]
    raise SystemExit(subprocess.call(cmd))


@app.command("play", help="Play Cambia interactively against an AI opponent")
def play(
    opponent: str = typer.Option(
        "random_no_cambia",
        "--opponent",
        "-o",
        help="Opponent agent type (e.g., random, imperfect_greedy, sd_cfr)",
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint path (required for neural agents: deep_cfr, escher, sd_cfr)",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Torch device for neural agent inference",
    ),
    num_players: int = typer.Option(
        2,
        "--num-players",
        "-n",
        help="Number of players (2-6)",
    ),
    human_seats: str = typer.Option(
        "0",
        "--human-seats",
        help="Comma-separated seat indices for human players (e.g., '0' or '0,2')",
    ),
    your_name: str = typer.Option(
        "You",
        "--name",
        help="Your player name",
    ),
):
    """Play a game of Cambia against AI opponents interactively."""
    from .evaluate_agents import get_agent, load_config, AGENT_REGISTRY
    from .play import SeatConfig, play_game

    cfg = load_config(str(config))
    if not cfg:
        print(f"ERROR: Failed to load config from {config}", file=sys.stderr)
        raise typer.Exit(1)

    human_indices = {int(s.strip()) for s in human_seats.split(",") if s.strip()}
    _checkpoint_types = {"deep_cfr", "escher", "sd_cfr", "nplayer", "ppo"}

    seats = []
    human_count = 0
    for seat_id in range(num_players):
        if seat_id in human_indices:
            human_count += 1
            name = your_name if human_count == 1 else f"Player {seat_id}"
            seats.append(SeatConfig(seat_id=seat_id, is_human=True, name=name))
        else:
            agent_kwargs = {}
            if opponent.lower() in _checkpoint_types:
                if not checkpoint:
                    print(f"ERROR: --checkpoint required for {opponent} agent", file=sys.stderr)
                    raise typer.Exit(1)
                agent_kwargs["checkpoint_path"] = str(checkpoint)
                agent_kwargs["device"] = device
            agent = get_agent(opponent, player_id=seat_id, config=cfg, **agent_kwargs)
            seats.append(SeatConfig(
                seat_id=seat_id, is_human=False, name=f"{opponent} (P{seat_id})",
                agent_type=opponent, agent=agent,
            ))

    if not human_indices:
        print("ERROR: At least one seat must be human", file=sys.stderr)
        raise typer.Exit(1)

    available = list(AGENT_REGISTRY.keys())
    if opponent.lower() not in [a.lower() for a in available]:
        print(f"ERROR: Unknown opponent '{opponent}'. Available: {available}", file=sys.stderr)
        raise typer.Exit(1)

    play_game(seats, cfg.cambia_rules)


@app.command("head-to-head", help="Play two Deep CFR checkpoints against each other")
def head_to_head(
    checkpoint_a: Path = typer.Option(
        ...,
        "--checkpoint-a",
        "-a",
        help="Path to first Deep CFR .pt checkpoint",
        exists=True,
    ),
    checkpoint_b: Path = typer.Option(
        ...,
        "--checkpoint-b",
        "-b",
        help="Path to second Deep CFR .pt checkpoint",
        exists=True,
    ),
    games: int = typer.Option(
        2000,
        "--games",
        "-n",
        help="Number of games to play",
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Torch device for inference (cpu, cuda, or xpu)",
    ),
):
    """Play two Deep CFR checkpoints head-to-head and report win rates."""
    from rich.console import Console
    from rich.table import Table
    from .config import load_config
    from .evaluate_agents import run_head_to_head

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration.", file=sys.stderr)
        raise typer.Exit(1)

    results = run_head_to_head(
        checkpoint_a=str(checkpoint_a),
        checkpoint_b=str(checkpoint_b),
        num_games=games,
        config=cfg,
        device=device,
    )

    total = results["total_games"]
    a_wins = results["checkpoint_a_wins"]
    b_wins = results["checkpoint_b_wins"]
    ties = results["ties"]
    scored = a_wins + b_wins + ties

    console = Console()
    table = Table(title="Head-to-Head Results")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Wins", style="green")
    table.add_column("Win Rate", style="bold green")

    table.add_row(
        checkpoint_a.name,
        str(a_wins),
        f"{a_wins / scored * 100:.1f}%" if scored else "N/A",
    )
    table.add_row(
        checkpoint_b.name,
        str(b_wins),
        f"{b_wins / scored * 100:.1f}%" if scored else "N/A",
    )
    table.add_row("Ties", str(ties), f"{ties / scored * 100:.1f}%" if scored else "N/A")
    table.add_row(
        "Avg Turns",
        f"{results['avg_game_turns']:.1f} ± {results['std_game_turns']:.1f}",
        "",
    )

    console.print(table)


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------
config_app = typer.Typer(
    help="Config schema and diff utilities",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("schema")
def config_schema(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON schema to file (default: stdout)"
    ),
):
    """Dump the Config JSON schema."""
    from src.config import Config
    import json

    schema = Config.model_json_schema()
    text = json.dumps(schema, indent=2)
    if output:
        output.write_text(text)
        typer.echo(f"Schema written to {output}")
    else:
        typer.echo(text)


@config_app.command("diff")
def config_diff(
    file1: Path = typer.Argument(..., help="First config YAML"),
    file2: Path = typer.Argument(..., help="Second config YAML"),
    as_json: bool = typer.Option(False, "--json", help="Output diff as JSON"),
):
    """Show differences between two config files."""
    from src.config import load_config
    import json

    c1 = load_config(str(file1))
    c2 = load_config(str(file2))
    if c1 is None or c2 is None:
        typer.echo("Error: could not load one or both config files.", err=True)
        raise typer.Exit(1)

    d1 = c1.model_dump(exclude_defaults=True)
    d2 = c2.model_dump(exclude_defaults=True)

    def _diff(a, b, path=""):
        diffs = []
        all_keys = set(list(a.keys()) + list(b.keys()))
        for k in sorted(all_keys):
            p = f"{path}.{k}" if path else k
            va, vb = a.get(k), b.get(k)
            if isinstance(va, dict) and isinstance(vb, dict):
                diffs.extend(_diff(va, vb, p))
            elif va != vb:
                diffs.append((p, va, vb))
        return diffs

    diffs = _diff(d1, d2)
    if not diffs:
        typer.echo("No differences.")
        return

    if as_json:
        result = {p: {"file1": v1, "file2": v2} for p, v1, v2 in diffs}
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Config diff: {file1.name} vs {file2.name}")
        table.add_column("Field", style="cyan")
        table.add_column(file1.name, style="yellow")
        table.add_column(file2.name, style="green")
        for p, v1, v2 in diffs:
            table.add_row(p, str(v1) if v1 is not None else "(default)", str(v2) if v2 is not None else "(default)")
        console.print(table)


def _coerce_override_value(raw: str):
    """Coerce a `--set key=value` string to bool/int/float, else leave as str."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _resolve_nested_model_cls(annotation):
    """Unwrap Optional[...]/Union[...] to find a pydantic BaseModel subclass.

    Returns None if the annotation isn't (or doesn't wrap) a BaseModel, which
    means the corresponding field is a leaf (can't be dotted into further).
    """
    from pydantic import BaseModel

    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            resolved = _resolve_nested_model_cls(arg)
            if resolved is not None:
                return resolved
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _apply_dotted_override(merged: dict, dotted_key: str, raw_value: str, root_model_cls) -> None:
    """Validate a dotted `--set` key against the Config schema and apply it.

    Walks `root_model_cls.model_fields` one dotted segment at a time,
    descending into nested dicts (creating them as needed) only where the
    schema actually has a nested model at that position. Raises ValueError
    with a human-readable message on any unknown segment or non-object
    descent, so a bad key is caught here rather than silently ignored by
    Config's `extra="ignore"` model config.
    """
    parts = [p for p in dotted_key.split(".")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"malformed key '{dotted_key}'")

    cur_model_cls = root_model_cls
    cur_dict = merged
    for i, part in enumerate(parts):
        fields = getattr(cur_model_cls, "model_fields", None)
        if not fields or part not in fields:
            raise ValueError(
                f"unknown config key '{dotted_key}' (no field '{part}' on "
                f"{getattr(cur_model_cls, '__name__', cur_model_cls)!s})"
            )
        if i == len(parts) - 1:
            cur_dict[part] = _coerce_override_value(raw_value)
            return
        nested_model_cls = _resolve_nested_model_cls(fields[part].annotation)
        if nested_model_cls is None:
            raise ValueError(
                f"cannot descend into non-object field '{part}' in key '{dotted_key}'"
            )
        next_dict = cur_dict.setdefault(part, {})
        if not isinstance(next_dict, dict):
            raise ValueError(
                f"field '{part}' in key '{dotted_key}' is not an object in the base config"
            )
        cur_dict = next_dict
        cur_model_cls = nested_model_cls


@config_app.command("render")
def config_render(
    base: Path = typer.Argument(..., help="Base config YAML (may reference _base)", exists=True),
    set_: List[str] = typer.Option(
        [],
        "--set",
        help="Dotted-key override key=value, e.g. --set prt_cfr.iterations=5 (repeatable)",
    ),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Materialized output YAML path"
    ),
):
    """Resolve _base, apply --set overrides, validate, and write a materialized config."""
    from src.config import Config, resolve_config_yaml
    import yaml as _yaml

    try:
        merged = resolve_config_yaml(str(base))
    except Exception as e:
        typer.echo(f"Error: could not resolve '{base}': {e}", err=True)
        raise typer.Exit(1)

    for item in set_:
        if "=" not in item:
            typer.echo(f"Error: --set value must be key=value, got '{item}'", err=True)
            raise typer.Exit(1)
        key, _, value = item.partition("=")
        try:
            _apply_dotted_override(merged, key, value, Config)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    try:
        Config.model_validate(merged)
    except Exception as e:
        typer.echo(f"Error: materialized config is invalid: {e}", err=True)
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    typer.echo(f"Rendered config written to {output}")


@config_app.command("validate")
def config_validate(
    path: Path = typer.Argument(..., help="Config YAML to validate", exists=True),
):
    """Validate a config YAML: resolves _base, validates against the Config schema."""
    from src.config import Config, resolve_config_yaml

    try:
        raw = resolve_config_yaml(str(path))
        Config.model_validate(raw)
    except Exception as e:
        typer.echo(f"Error: '{path}' is invalid: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"OK: '{path}' is valid.")


# ---------------------------------------------------------------------------
# Runs subcommands
# ---------------------------------------------------------------------------
runs_app = typer.Typer(
    help="Run database management",
    no_args_is_help=True,
)
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    sort_by: str = typer.Option("name", "--sort-by", help="Sort by field (name, status, best_metric_value, created_at)"),
):
    """List all runs in the database."""
    from src.run_db import get_db
    import json

    db = get_db()
    query = "SELECT * FROM runs"
    conditions = []
    params = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if tag:
        conditions.append("tags LIKE ?")
        params.append(f"%{tag}%")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    valid_sorts = {"name", "status", "best_metric_value", "created_at", "updated_at"}
    if sort_by in valid_sorts:
        query += f" ORDER BY {sort_by}"
    else:
        query += " ORDER BY name"

    rows = db.execute(query, params).fetchall()
    db.close()

    if not rows:
        typer.echo("No runs found.")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Cambia CFR Runs")
    table.add_column("Name", style="cyan")
    table.add_column("Algorithm", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Best Metric")
    table.add_column("Tags")
    table.add_column("Updated")

    for row in rows:
        best = ""
        if row["best_metric_name"]:
            best = f"{row['best_metric_name']}={row['best_metric_value']:.3f} @{row['best_metric_iter']}"
        tags = json.loads(row["tags"]) if row["tags"] else []
        table.add_row(
            row["name"],
            row["algorithm"] or "",
            row["status"],
            best,
            ", ".join(tags) if tags else "",
            (row["updated_at"] or "")[:10],
        )
    console.print(table)


@runs_app.command("prune")
def runs_prune(
    run_name: str = typer.Argument(..., help="Run name to prune"),
    keep_every_n: int = typer.Option(50, "--keep-every-n", help="Keep every Nth checkpoint"),
    keep_latest: int = typer.Option(3, "--keep-latest", help="Keep N most recent checkpoints"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
):
    """Prune non-retained checkpoints from a run."""
    from src.run_db import get_db, compute_retention_flags

    db = get_db()
    row = db.execute("SELECT id FROM runs WHERE name = ?", (run_name,)).fetchone()
    if not row:
        typer.echo(f"Run '{run_name}' not found.")
        db.close()
        raise typer.Exit(1)

    run_id = row["id"]
    non_retained = compute_retention_flags(db, run_id, keep_every_n, keep_latest)

    if not non_retained:
        typer.echo("No checkpoints to prune.")
        db.close()
        return

    # Get file paths for non-retained
    placeholders = ",".join("?" * len(non_retained))
    ckpts = db.execute(
        f"SELECT id, iteration, file_path FROM checkpoints WHERE id IN ({placeholders})",
        non_retained,
    ).fetchall()

    typer.echo(f"Checkpoints to prune ({len(ckpts)}):")
    total_size = 0
    for c in ckpts:
        fpath = Path(c["file_path"])
        size = fpath.stat().st_size if fpath.exists() else 0
        total_size += size
        typer.echo(f"  iter {c['iteration']}: {fpath.name} ({size / 1024 / 1024:.1f} MB)")

    typer.echo(f"\nTotal space to reclaim: {total_size / 1024 / 1024:.1f} MB")

    if dry_run:
        typer.echo("(dry run — no files deleted)")
        db.close()
        return

    if not typer.confirm("Delete these checkpoints?"):
        typer.echo("Aborted.")
        db.close()
        return

    deleted = 0
    for c in ckpts:
        fpath = Path(c["file_path"])
        if fpath.exists():
            fpath.unlink()
            deleted += 1
    typer.echo(f"Deleted {deleted} checkpoint files.")
    db.close()


@runs_app.command("backfill")
def runs_backfill(
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Path to runs directory"
    ),
):
    """Backfill the run database from existing run directories."""
    import subprocess
    import sys

    script = Path(__file__).parent.parent / "scripts" / "backfill_db.py"
    if not script.exists():
        typer.echo(f"Backfill script not found: {script}", err=True)
        raise typer.Exit(1)

    result = subprocess.run(
        [sys.executable, str(script), "--runs-dir", str(runs_dir)],
        capture_output=False,
    )
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
