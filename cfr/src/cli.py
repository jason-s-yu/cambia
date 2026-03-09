"""src/cli.py - Typer-based CLI for Cambia CFR Training Suite."""

import os
import sys
import signal
import multiprocessing
from pathlib import Path
from typing import List, Optional

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
        else os.path.splitext(cfg.persistence.agent_data_save_path)[0] + "_gtcfr.pt"
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
        else os.path.splitext(cfg.persistence.agent_data_save_path)[0] + "_sog.pt"
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


@train_app.command("ppo", help="Train PPO best-response diagnostic agent")
def train_ppo_cmd(
    opponent: str = typer.Option(
        "imperfect_greedy",
        "--opponent",
        "-o",
        help="Opponent agent type to train against",
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
        help="Path to save trained model",
    ),
    n_envs: int = typer.Option(
        4,
        "--n-envs",
        help="Number of parallel training environments",
    ),
    eval_freq: int = typer.Option(
        10_000,
        "--eval-freq",
        help="Evaluate every N timesteps",
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
    """Train a PPO agent as best-response to a fixed opponent (diagnostic)."""
    from .ppo_train import train_ppo

    train_ppo(
        opponent=opponent,
        timesteps=timesteps,
        save_path=str(save_path),
        n_envs=n_envs,
        eval_freq=eval_freq,
        net_arch=[256, 256],
        seed=seed,
        config_path=str(config),
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
):
    """Evaluate a checkpoint against baseline agents and print a win-rate table."""
    import re
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
        import yaml

        with open(run_config, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        algorithm = infer_algorithm(config_dict)

        # Use auto-detected values unless user explicitly overrode
        if agent_type == "deep_cfr":
            agent_type = algo_to_agent_type(algorithm)

        prefix = algo_to_checkpoint_prefix(algorithm)

        # Default to 5000 games in run-dir mode (but respect explicit user override)
        if games_was_default:
            games = 5000

        # Resolve checkpoint
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"ERROR: No checkpoints/ directory in {run_dir}", file=sys.stderr)
            raise typer.Exit(1)

        def _find_checkpoints(pfx):
            found = sorted(ckpt_dir.glob(f"*{pfx}*epoch_*.pt"))
            if not found:
                found = sorted(ckpt_dir.glob(f"*{pfx}*iter_*.pt"))
            return found

        all_ckpts = _find_checkpoints(prefix)

        def extract_num(p):
            m = re.search(r"(?:epoch|iter)_(\d+)", p.name)
            return int(m.group(1)) if m else 0

        if latest:
            if not all_ckpts:
                print(f"ERROR: No checkpoints matching prefix '{prefix}' found", file=sys.stderr)
                raise typer.Exit(1)
            all_ckpts.sort(key=extract_num)
            checkpoint_path = all_ckpts[-1]
        elif epoch is not None:
            matches = [
                p for p in all_ckpts
                if f"epoch_{epoch}.pt" in p.name or f"iter_{epoch}.pt" in p.name
            ]
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

    # Persist results if we have a run directory context
    if run_dir is not None:
        from .evaluate_agents import persist_eval_results

        m = re.search(r"(?:epoch|iter)_(\d+)", str(checkpoint))
        iteration = int(m.group(1)) if m else 0
        persist_eval_results(
            run_dir=str(run_dir),
            iteration=iteration,
            results_map=results_map,
            checkpoint_path=str(checkpoint),
        )
        print(f"\nResults persisted to {run_dir}/metrics.jsonl")

    if lbr:
        from .cfr.sampled_lbr import sampled_lbr as run_lbr
        from .config import load_config as _load_config
        from .evaluate_agents import get_agent

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
            result = run_lbr(
                agent,
                lbr_config,
                num_infosets=lbr_infosets,
                br_rollouts_per_infoset=lbr_rollouts,
            )
            print(
                f"[lbr] exploitability={result['exploitability']:.3f} "
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
):
    """
    Watch run directories and evaluate every new checkpoint against all baselines.

    Writes metrics.jsonl and (optionally) head_to_head.jsonl into each run directory.
    Supports Deep CFR, ReBeL, SD-CFR, and ESCHER agent types via --agent-type.
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
