#!/usr/bin/env python3
"""benchmark_concurrent.py - Concurrency benchmark for Deep CFR training.

Launches N independent training processes simultaneously and measures per-step
wall time to determine concurrency feasibility for Phase 3+4 training.

Each concurrency level N launches N independent `cambia train deep` processes
with unique checkpoint/profiling paths. Tier 2 (JSONL) profiling captures
per-step phase timing; Tier 3 (torch.profiler) captures one Chrome trace per
run at the configured `profile_step`. Statistics are computed over warm steps
only (steps > warmup).

Hardware context matters: GPU contention is the binding constraint for N>1.
Re-run this benchmark on each target system to establish its concurrency ceiling.

Prerequisites:
    pip install -e .          # from cfr/ (installs 'cambia' console script)
    make libcambia            # from repo root (builds Go FFI shared library)

Config:
    cfr/config/bench_concurrent.yaml
    Key settings: enable_traversal_profiling=true, profile_step=50,
    sd_cfr_mode=true, use_residual=true, encoding_mode=ep_pbs.

Usage:
    cd cfr
    python scripts/benchmark_concurrent.py                             # full N=1..4 sweep
    python scripts/benchmark_concurrent.py --max-concurrent 2          # N=1..2 only
    python scripts/benchmark_concurrent.py --keep-artifacts            # keep temp dirs
    python scripts/benchmark_concurrent.py --steps 120 --warmup 50     # longer runs

Output:
    - Console summary table with mean, std, 95% CI, P5, P95, throughput
    - JSON results at cfr/runs/benchmark_results_{timestamp}.json
    - Chrome traces at {temp_dir}/run_N/profile_step_50.json (with --keep-artifacts)
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Locate repo root relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CFR_DIR = REPO_ROOT / "cfr"
DEFAULT_CONFIG = CFR_DIR / "config" / "bench_concurrent.yaml"


def _collect_hardware_info() -> dict:
    """Collect system hardware info for benchmark reproducibility."""
    info = {"cpu": "unknown", "gpu": "unknown", "ram": "unknown",
            "python": sys.version.split()[0], "torch": "unknown"}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            info["gpu"] = torch.xpu.get_device_name(0)
        else:
            info["gpu"] = "CPU only"
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram"] = f"{kb / 1048576:.0f} GB"
                    break
    except Exception:
        pass
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrency benchmark for Deep CFR training (Phase 2.6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        metavar="N",
        help="Maximum concurrency level to test (default: 4). Tests N=1..max-concurrent.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=80,
        metavar="N",
        help="Total training steps per run (default: 80).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        metavar="N",
        help="Warmup steps to exclude from statistics (default: 30).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to benchmark config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Skip cleanup of temp directories (for debugging).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CFR_DIR / "runs",
        help="Directory to save benchmark results JSON (default: cfr/runs/).",
    )
    return parser.parse_args()


def compute_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of per-step times."""
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    sorted_vals = sorted(values)
    p5 = sorted_vals[max(0, int(0.05 * n) - 1)]
    p95 = sorted_vals[min(n - 1, int(0.95 * n))]
    median = sorted_vals[n // 2]
    # 95% CI half-width: 1.96 * std / sqrt(n)
    ci_half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    throughput = 1.0 / mean if mean > 0 else 0.0
    return {
        "n_samples": n,
        "mean": mean,
        "std": std,
        "median": median,
        "p5": p5,
        "p95": p95,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
        "throughput_steps_per_sec": throughput,
    }


def parse_profiling_jsonl(jsonl_path: Path, warmup: int) -> list[float]:
    """Parse per-step wall times from a profiling JSONL file.

    Expects records with a 'step' field and at least one timing field.
    Falls back to 'total_step_time', 'step_time', or sum of known phase fields.
    Returns warm-step times only (step > warmup).
    """
    step_times = []
    if not jsonl_path.exists():
        return step_times

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = record.get("step", record.get("iteration", None))
            if step is None or step <= warmup:
                continue

            # Try common field names for total step wall time
            wall = None
            for key in ("total_step_time", "step_time", "wall_time", "elapsed"):
                if key in record:
                    wall = float(record[key])
                    break

            if wall is None:
                # Sum known phase fields (in milliseconds, convert to seconds)
                phase_keys_ms = (
                    "traversal_ms",
                    "adv_train_ms",
                    "strat_train_ms",
                    "buffer_insert_ms",
                    "weights_copy_ms",
                )
                phase_sum_ms = sum(float(record[k]) for k in phase_keys_ms if k in record)
                if phase_sum_ms > 0:
                    wall = phase_sum_ms / 1000.0
                else:
                    # Fallback: try _time suffix (seconds)
                    phase_keys_s = (
                        "traversal_time",
                        "adv_train_time",
                        "strat_train_time",
                        "buffer_insert_time",
                        "weights_copy_time",
                    )
                    phase_sum = sum(float(record[k]) for k in phase_keys_s if k in record)
                    if phase_sum > 0:
                        wall = phase_sum

            if wall is not None and wall > 0:
                step_times.append(wall)

    return step_times


def launch_training_processes(
    n: int,
    base_tmp: Path,
    config: Path,
    steps: int,
) -> tuple[list[subprocess.Popen], list[Path]]:
    """Launch n independent training processes, each with a unique run directory."""
    processes = []
    run_dirs = []

    for i in range(n):
        run_dir = base_tmp / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        profiling_path = run_dir / "profiling.jsonl"
        checkpoint_path = run_dir / "checkpoint.pt"
        run_dirs.append(run_dir)

        # Build command. Use the 'cambia' console_script entry point
        # (installed via pyproject.toml: cambia = "src.cli:app").
        cmd = [
            "cambia",
            "train",
            "deep",
            "--config",
            str(config),
            "--steps",
            str(steps),
            "--save-path",
            str(checkpoint_path),
            "--headless",
        ]

        env = os.environ.copy()
        # Signal the trainer to write profiling output to our run dir.
        # The deep_trainer reads profiling_jsonl_path from config; we override via env
        # if supported, otherwise rely on the config default (empty = auto-derive from save_path dir).
        # Since save_path is per-run, profiling.jsonl should land in the same dir if the trainer
        # derives it from save_path. Set CAMBIA_PROFILING_PATH as a belt-and-suspenders override.
        env["CAMBIA_PROFILING_PATH"] = str(profiling_path)
        env["CAMBIA_RUN_DIR"] = str(run_dir)

        # Suppress output to avoid terminal noise; redirect to per-run log files.
        log_stdout = open(run_dir / "stdout.log", "w")
        log_stderr = open(run_dir / "stderr.log", "w")

        proc = subprocess.Popen(
            cmd,
            stdout=log_stdout,
            stderr=log_stderr,
            env=env,
            cwd=str(CFR_DIR),
        )
        processes.append(proc)
        print(f"  Launched process {i+1}/{n} (PID {proc.pid}) -> {run_dir}")

    return processes, run_dirs


def wait_for_processes(processes: list[subprocess.Popen], timeout_per_step: float = 120.0) -> list[int]:
    """Wait for all processes to complete, return list of return codes."""
    return_codes = []
    for i, proc in enumerate(processes):
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            print("\nInterrupted — terminating all processes.")
            for p in processes:
                p.terminate()
            sys.exit(1)
        return_codes.append(rc)
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  Process {i+1} finished: {status}")
    return return_codes


def print_results_table(results: list[dict], warmup: int, steps: int) -> None:
    """Print a plain-text summary table."""
    print(
        f"\nConcurrency Benchmark Results (warm steps {warmup+1}-{steps})\n"
        + "-" * 80
    )
    header = f"{'N':>4} | {'Mean (s)':>9} | {'Std (s)':>8} | {'95% CI':>16} | {'P5 (s)':>7} | {'P95 (s)':>7} | {'Steps/s':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        n = r["concurrency"]
        s = r.get("stats", {})
        if not s:
            print(f"{n:>4} | {'N/A':>9} | {'N/A':>8} | {'N/A':>16} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8}")
            continue
        ci = f"[{s['ci_low']:.2f}, {s['ci_high']:.2f}]"
        print(
            f"{n:>4} | {s['mean']:>9.2f} | {s['std']:>8.2f} | {ci:>16} | "
            f"{s['p5']:>7.2f} | {s['p95']:>7.2f} | {s['throughput_steps_per_sec']:>8.4f}"
        )
    print("-" * 80)


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.warmup >= args.steps:
        print(
            f"ERROR: warmup ({args.warmup}) must be < steps ({args.steps})",
            file=sys.stderr,
        )
        sys.exit(1)

    concurrency_levels = list(range(1, args.max_concurrent + 1))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"benchmark_results_{timestamp}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    tmp_dirs_to_clean = []

    # Collect system info for reproducibility
    hw_info = _collect_hardware_info()

    print("=" * 80)
    print("CONCURRENCY BENCHMARK — Deep CFR Training")
    print("=" * 80)
    print(f"\nSystem:  {hw_info['cpu']}")
    print(f"GPU:     {hw_info['gpu']}")
    print(f"RAM:     {hw_info['ram']}")
    print(f"Python:  {hw_info['python']}")
    print(f"PyTorch: {hw_info['torch']}")
    print(f"\nConfig:  {args.config}")
    print(f"Levels:  N in {concurrency_levels}, {args.steps} steps, {args.warmup} warmup")
    print(f"Output:  {output_path}\n")

    for n in concurrency_levels:
        print(f"=== N={n} concurrent instances ===")
        base_tmp = Path(tempfile.mkdtemp(prefix=f"cambia_bench_n{n}_"))
        tmp_dirs_to_clean.append(base_tmp)

        t_start = time.monotonic()
        processes, run_dirs = launch_training_processes(
            n=n,
            base_tmp=base_tmp,
            config=args.config,
            steps=args.steps,
        )
        return_codes = wait_for_processes(processes)
        wall_total = time.monotonic() - t_start
        print(f"  Wall time for N={n}: {wall_total:.1f}s")

        # Collect profiling data from all runs
        all_step_times: list[float] = []
        per_run_data = []
        for i, run_dir in enumerate(run_dirs):
            jsonl_path = run_dir / "profiling.jsonl"
            step_times = parse_profiling_jsonl(jsonl_path, warmup=args.warmup)
            per_run_data.append(
                {
                    "run_index": i,
                    "return_code": return_codes[i],
                    "run_dir": str(run_dir),
                    "warm_step_times": step_times,
                    "n_warm_steps": len(step_times),
                }
            )
            all_step_times.extend(step_times)
            print(f"  Run {i}: {len(step_times)} warm steps parsed from {jsonl_path}")

        stats = compute_stats(all_step_times) if all_step_times else {}
        if stats:
            print(
                f"  Stats (N={n}): mean={stats['mean']:.2f}s std={stats['std']:.2f}s "
                f"p5={stats['p5']:.2f}s p95={stats['p95']:.2f}s"
            )
        else:
            print(f"  WARNING: No warm-step timing data collected for N={n}.")
            print(f"  Check logs in {base_tmp} for errors.")

        all_results.append(
            {
                "concurrency": n,
                "steps": args.steps,
                "warmup": args.warmup,
                "wall_time_total": wall_total,
                "return_codes": return_codes,
                "stats": stats,
                "per_run": per_run_data,
            }
        )

    # Print summary table
    print_results_table(all_results, warmup=args.warmup, steps=args.steps)

    # Save JSON results
    output_data = {
        "timestamp": timestamp,
        "hardware": hw_info,
        "config": str(args.config),
        "max_concurrent": args.max_concurrent,
        "steps": args.steps,
        "warmup": args.warmup,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    if not args.keep_artifacts:
        for d in tmp_dirs_to_clean:
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"WARNING: Could not remove {d}: {e}", file=sys.stderr)
        print("Temp directories cleaned up.")
    else:
        print("Artifacts kept (--keep-artifacts):")
        for d in tmp_dirs_to_clean:
            print(f"  {d}")


if __name__ == "__main__":
    main()
