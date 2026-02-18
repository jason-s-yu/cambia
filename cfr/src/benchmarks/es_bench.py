"""
src/benchmarks/es_bench.py

ES (External Sampling) validation benchmark for Deep CFR.
Measures exploitability computation throughput across backends and depth limits.
"""

import logging
import time
from typing import Optional, List

import numpy as np

from ..config import load_config
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork
from ..cfr.es_validator import ESValidator
from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


def benchmark_es(
    depths: Optional[List[int]] = None,
    num_traversals: int = 10,
    backends: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Benchmark ES validation across backends and depth limits.

    For each (backend, depth) combination, creates an ESValidator with random
    network weights and runs compute_exploitability(), measuring nodes/sec,
    traversals/sec, and avg_nodes_per_traversal.

    Args:
        depths: List of depth limits to benchmark (default: [5, 8, 10, 12])
        num_traversals: Number of traversals per combination
        backends: List of backends to test (default: ["python", "go"])
        config_path: Path to config file (defaults to parallel.config.yaml)
        device: Device for network initialization (cpu/cuda)

    Returns:
        BenchmarkResult with per-(backend, depth) metrics
    """
    depths = depths or [5, 8, 10, 12]
    backends = backends or ["python", "go"]
    config_path = config_path or "/workspace/config/parallel.config.yaml"

    config = load_config(config_path)
    if not config:
        raise RuntimeError(f"Failed to load config from {config_path}")

    # Create random network weights once; reuse across all combinations
    net = AdvantageNetwork(input_dim=INPUT_DIM, hidden_dim=256, output_dim=NUM_ACTIONS)
    network_weights = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    network_config = {
        "input_dim": INPUT_DIM,
        "hidden_dim": 256,
        "output_dim": NUM_ACTIONS,
    }

    metrics: dict = {}
    timings: dict = {}
    combo_results = []

    for backend in backends:
        for depth in depths:
            label = f"{backend}_depth{depth}"
            logger.info(
                "ES benchmark: backend=%s depth=%d num_traversals=%d",
                backend,
                depth,
                num_traversals,
            )

            # Override config fields for this combination
            config.deep_cfr.es_validation_depth = depth
            config.deep_cfr.engine_backend = backend

            validator = ESValidator(
                config=config,
                network_weights=network_weights,
                network_config=network_config,
            )

            start = time.time()
            result = validator.compute_exploitability(num_traversals=num_traversals)
            elapsed = time.time() - start

            total_nodes = result.get("total_nodes", 0)
            completed = result.get("traversals", 0)

            nodes_per_sec = total_nodes / elapsed if elapsed > 0 else 0.0
            traversals_per_sec = completed / elapsed if elapsed > 0 else 0.0
            avg_nodes_per_traversal = total_nodes / completed if completed > 0 else 0.0

            metrics[f"{label}_nodes_per_sec"] = nodes_per_sec
            metrics[f"{label}_traversals_per_sec"] = traversals_per_sec
            metrics[f"{label}_avg_nodes_per_traversal"] = avg_nodes_per_traversal
            metrics[f"{label}_mean_regret"] = result.get("mean_regret", 0.0)
            metrics[f"{label}_max_regret"] = result.get("max_regret", 0.0)
            metrics[f"{label}_strategy_entropy"] = result.get("strategy_entropy", 0.0)
            metrics[f"{label}_total_nodes"] = total_nodes
            metrics[f"{label}_completed_traversals"] = completed

            timings[f"{label}_total_time"] = elapsed

            combo_results.append(
                {
                    "backend": backend,
                    "depth": depth,
                    "nodes_per_sec": nodes_per_sec,
                    "traversals_per_sec": traversals_per_sec,
                    "avg_nodes_per_traversal": avg_nodes_per_traversal,
                    "total_nodes": total_nodes,
                    "completed_traversals": completed,
                    "elapsed": elapsed,
                }
            )

            logger.info(
                "  %s: %.0f nodes/s, %.2f trav/s, %.0f nodes/trav",
                label,
                nodes_per_sec,
                traversals_per_sec,
                avg_nodes_per_traversal,
            )

    # Build comparison section: go vs python speedup for each depth
    comparison: dict = {}
    for depth in depths:
        py_key = f"python_depth{depth}_nodes_per_sec"
        go_key = f"go_depth{depth}_nodes_per_sec"
        if py_key in metrics and go_key in metrics:
            py_val = metrics[py_key]
            go_val = metrics[go_key]
            if py_val > 0:
                comparison[f"depth{depth}_go_speedup"] = go_val / py_val
            else:
                comparison[f"depth{depth}_go_speedup"] = 0.0

    metrics["comparison"] = comparison

    logger.info(
        "ES benchmark complete: %d backend/depth combinations",
        len(combo_results),
    )

    return BenchmarkResult(
        name="es_exploitability",
        config={
            "depths": depths,
            "num_traversals": num_traversals,
            "backends": backends,
            "config_path": config_path,
        },
        timings=timings,
        metrics=metrics,
        metadata={
            "device": device,
            "hidden_dim": 256,
            "combinations": combo_results,
        },
    )
