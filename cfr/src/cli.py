"""src/cli.py - Typer-based CLI for Cambia CFR Training Suite."""

import sys
import signal
import multiprocessing
from pathlib import Path
from typing import Optional

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

    cfg = load_config(str(config))
    if not cfg:
        print("ERROR: Failed to load configuration. Exiting.", file=sys.stderr)
        raise typer.Exit(1)

    # Build overrides dict from CLI options
    overrides = {}
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
        help="Path to Deep CFR .pt checkpoint file",
        exists=True,
    ),
    config: Path = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
    ),
    games: int = typer.Option(
        100,
        "--games",
        "-n",
        help="Number of games per baseline matchup",
    ),
    baselines: str = typer.Option(
        "random,greedy",
        "--baselines",
        "-b",
        help="Comma-separated list of baseline agents to evaluate against "
        "(choices: random, greedy, imperfect_greedy, memory_heuristic, aggressive_snap, cfr)",
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
):
    """Evaluate a Deep CFR checkpoint against baseline agents and print a win-rate table."""
    from rich.console import Console
    from rich.table import Table
    from .evaluate_agents import run_evaluation_multi_baseline

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
    )

    console = Console()
    table = Table(title=f"Evaluation: {checkpoint.name}")
    table.add_column("Baseline", style="cyan")
    table.add_column("DeepCFR Wins", style="green")
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
        100,
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


if __name__ == "__main__":
    app()
