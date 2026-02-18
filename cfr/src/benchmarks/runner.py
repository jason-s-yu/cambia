"""
src/benchmarks/runner.py

Core benchmarking infrastructure for Deep CFR.

Provides BenchmarkResult dataclass for storing results and BenchmarkSuite
for organizing and running multiple benchmarks with consistent output management.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .reporting import print_result, save_summary

logger = logging.getLogger(__name__)

# Canonical timestamp format used for directory names and result timestamps.
TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


def _now_stamp() -> str:
    """Return current time in the canonical benchmark timestamp format."""
    return datetime.now().strftime(TIMESTAMP_FMT)


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.

    Attributes:
        name: Human-readable benchmark name
        config: Configuration/parameters used for the benchmark
        timings: Named timing measurements in seconds (e.g., {"forward_pass": 0.123})
        metrics: Derived metrics like throughput, utilization (e.g., {"samples_per_sec": 1000})
        metadata: System information and context (device, timestamp, etc.)
        timestamp: Timestamp when the benchmark was run (format: YYYY-MM-DD_HHMMSS)
        run_id: Shared identifier tying results from the same suite run
        notes: Free-text context about what state the code was in (e.g. "baseline before columnar buffer refactor")
    """
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_stamp)
    run_id: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Construct from dictionary (for loading from JSON)."""
        return cls(**data)


class BenchmarkSuite:
    """
    Framework for organizing and running benchmarks.

    Usage:
        suite = BenchmarkSuite(notes="baseline before columnar buffer refactor")
        suite.register(my_benchmark_fn)
        results = suite.run_all(output_dir="/workspace/benchmarks", device="cuda")
    """

    def __init__(self, notes: str = ""):
        self.benchmarks: Dict[str, Callable] = {}
        self.notes = notes
        logger.info("BenchmarkSuite initialized")

    def register(self, benchmark_fn: Callable, name: Optional[str] = None):
        """
        Register a benchmark function.

        Args:
            benchmark_fn: A callable that returns a BenchmarkResult.
                         Should accept (device, **kwargs) parameters.
            name: Optional name override. Defaults to function name.
        """
        bench_name = name or benchmark_fn.__name__
        if bench_name in self.benchmarks:
            logger.warning("Overwriting existing benchmark: %s", bench_name)
        self.benchmarks[bench_name] = benchmark_fn
        logger.info("Registered benchmark: %s", bench_name)

    def run_one(
        self,
        name: str,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        run_id: str = "",
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a single benchmark by name.

        Args:
            name: Name of the registered benchmark
            output_dir: Optional directory to save results
            device: Device to run on ("cpu", "cuda", etc.)
            run_id: Shared run identifier (set automatically by run_all)
            **kwargs: Additional arguments passed to the benchmark function

        Returns:
            BenchmarkResult from the benchmark

        Raises:
            KeyError: If benchmark name is not registered
        """
        if name not in self.benchmarks:
            raise KeyError(f"Benchmark '{name}' not registered. Available: {list(self.benchmarks.keys())}")

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Running benchmark: %s (device=%s)", name, device)

        benchmark_fn = self.benchmarks[name]
        result = benchmark_fn(device=device, **kwargs)

        # Stamp run_id and notes onto the result
        if run_id:
            result.run_id = run_id
        if self.notes and not result.notes:
            result.notes = self.notes

        # Add system metadata if not already present
        if "device" not in result.metadata:
            result.metadata["device"] = device
        if "cuda_available" not in result.metadata:
            result.metadata["cuda_available"] = torch.cuda.is_available()
        if device == "cuda" and torch.cuda.is_available():
            result.metadata["cuda_device_name"] = torch.cuda.get_device_name(0)

        # Print to console
        print_result(result)

        # Save to disk if output_dir specified
        if output_dir:
            bench_dir = Path(output_dir) / name
            bench_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = bench_dir / "results.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Saved results to %s", json_path)

        return result

    def run_all(
        self,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[BenchmarkResult]:
        """
        Run all registered benchmarks.

        Args:
            output_dir: Base directory for all benchmark results.
                       Creates timestamped subdirectory: /workspace/benchmarks/YYYY-MM-DD_HHMMSS/
            device: Device to run on ("cpu", "cuda", etc.)
            **kwargs: Additional arguments passed to all benchmark functions

        Returns:
            List of BenchmarkResult from all benchmarks
        """
        # Create timestamped output directory — use canonical format and share
        # the same run_id across all results in this suite run.
        run_id = _now_stamp()
        if output_dir:
            output_base = Path(output_dir) / run_id
            output_base.mkdir(parents=True, exist_ok=True)
            logger.info("Benchmark suite output directory: %s", output_base)
        else:
            output_base = None

        results = []

        for name in self.benchmarks:
            try:
                result = self.run_one(
                    name,
                    output_dir=str(output_base) if output_base else None,
                    device=device,
                    run_id=run_id,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.exception("Benchmark '%s' failed: %s", name, e)
                # Continue with other benchmarks

        # Save summary
        if output_base:
            summary_path = output_base / "summary.txt"
            save_summary(results, str(summary_path))
            logger.info("Saved summary to %s", summary_path)

        logger.info("Benchmark suite complete: %d/%d succeeded", len(results), len(self.benchmarks))
        return results
