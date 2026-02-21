"""
scripts/plot_metrics.py

Generate matplotlib plots from collected metrics.jsonl files.

Plots produced:
  1. win_rate_by_run.png    — One subplot per run, one line per baseline.
  2. win_rate_by_baseline.png — One subplot per baseline, one line per run.
  3. win_rate_combined.png  — All runs on same axes, averaged across baselines.

Usage:
    python scripts/plot_metrics.py [--runs-dir PATH] [--output-dir PATH]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_BASELINES = [
    "random",
    "greedy",
    "imperfect_greedy",
    "memory_heuristic",
    "aggressive_snap",
]

# Distinct colors for up to 8 baselines / runs.
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _marker(i: int) -> str:
    return _MARKERS[i % len(_MARKERS)]


def load_all_metrics(runs_dir: Path) -> list[dict]:
    """Load all metrics.jsonl files from <runs_dir>/*/metrics.jsonl."""
    rows = []
    for metrics_file in sorted(runs_dir.glob("*/metrics.jsonl")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed JSONL row in %s: %s", metrics_file, e)
    return rows


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)
    return dict(groups)


def plot_win_rate_by_run(
    rows: list[dict], output_dir: Path, import_matplotlib
) -> None:
    """One subplot per run, one line per baseline."""
    plt = import_matplotlib()
    by_run = _group_by(rows, "run")
    if not by_run:
        return

    n_runs = len(by_run)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)

    for col, (run_name, run_rows) in enumerate(sorted(by_run.items())):
        ax = axes[0][col]
        by_baseline = _group_by(run_rows, "baseline")

        for i, baseline in enumerate(_BASELINES):
            baseline_rows = sorted(by_baseline.get(baseline, []), key=lambda r: r.get("iter", 0))
            if not baseline_rows:
                continue
            iters = [r["iter"] for r in baseline_rows]
            win_rates = [r["win_rate"] for r in baseline_rows]
            ax.plot(
                iters,
                win_rates,
                label=baseline,
                color=_color(i),
                marker=_marker(i),
                markersize=4,
                linewidth=1.5,
            )

        ax.set_title(run_name, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate (P0)")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Win Rate vs Iteration (by run)", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "win_rate_by_run.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_win_rate_by_baseline(
    rows: list[dict], output_dir: Path, import_matplotlib
) -> None:
    """One subplot per baseline, one line per run."""
    plt = import_matplotlib()
    all_runs = sorted(set(r.get("run", "?") for r in rows))
    if not all_runs:
        return

    n_baselines = len(_BASELINES)
    fig, axes = plt.subplots(1, n_baselines, figsize=(5 * n_baselines, 5), squeeze=False)

    by_run = _group_by(rows, "run")

    for col, baseline in enumerate(_BASELINES):
        ax = axes[0][col]

        for i, run_name in enumerate(all_runs):
            run_rows = [r for r in by_run.get(run_name, []) if r.get("baseline") == baseline]
            run_rows = sorted(run_rows, key=lambda r: r.get("iter", 0))
            if not run_rows:
                continue
            iters = [r["iter"] for r in run_rows]
            win_rates = [r["win_rate"] for r in run_rows]
            ax.plot(
                iters,
                win_rates,
                label=run_name,
                color=_color(i),
                marker=_marker(i),
                markersize=4,
                linewidth=1.5,
            )

        ax.set_title(baseline, fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate (P0)")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Win Rate vs Iteration (by baseline)", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "win_rate_by_baseline.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_win_rate_combined(
    rows: list[dict], output_dir: Path, import_matplotlib
) -> None:
    """All runs on the same axes, averaging win rate across all baselines per iteration."""
    plt = import_matplotlib()
    # Build (run, iter) -> [win_rates] mapping.
    agg: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        run = row.get("run", "?")
        it = row.get("iter", 0)
        wr = row.get("win_rate")
        if wr is not None:
            agg[(run, it)].append(wr)

    all_runs = sorted(set(k[0] for k in agg))
    if not all_runs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, run_name in enumerate(all_runs):
        run_data = {k[1]: v for k, v in agg.items() if k[0] == run_name}
        iters = sorted(run_data.keys())
        avg_wr = [float(sum(run_data[it]) / len(run_data[it])) for it in iters]
        ax.plot(
            iters,
            avg_wr,
            label=run_name,
            color=_color(i),
            marker=_marker(i),
            markersize=4,
            linewidth=2,
        )

    ax.set_title("Win Rate vs Iteration (averaged across baselines)", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Win Rate (P0)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "win_rate_combined.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def generate_plots(runs_dir: str, output_dir: str) -> None:
    runs_path = Path(runs_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    rows = load_all_metrics(runs_path)
    if not rows:
        logger.warning("No metrics data found in %s", runs_path)
        return

    logger.info("Loaded %d metric rows from %s", len(rows), runs_path)

    # Lazy import matplotlib so the module is importable without it at test time.
    def _import_plt():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    plot_win_rate_by_run(rows, out_path, _import_plt)
    plot_win_rate_by_baseline(rows, out_path, _import_plt)
    plot_win_rate_combined(rows, out_path, _import_plt)

    logger.info("All plots saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Deep CFR training metrics.")
    parser.add_argument(
        "--runs-dir",
        default="cfr/runs",
        help="Directory containing run subdirectories with metrics.jsonl (default: cfr/runs)",
    )
    parser.add_argument(
        "--output-dir",
        default="cfr/runs/plots",
        help="Directory to save plots (default: cfr/runs/plots)",
    )
    args = parser.parse_args()
    generate_plots(runs_dir=args.runs_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
