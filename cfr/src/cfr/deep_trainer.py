"""
src/cfr/deep_trainer.py

Deep CFR Trainer — replaces the tabular CFRTrainer for neural network-based training.

Training loop:
1. Every K traversals, distribute advantage network weights to workers
2. Workers run external sampling traversals, return ReservoirSamples
3. Append samples to advantage buffer (Mv) and strategy buffer (Mpi)
4. Train advantage network on Mv with weighted MSE loss: (t^alpha) * MSE(pred, target)
5. Train strategy network on Mpi similarly

Uses multiprocessing Pool pattern similar to training_loop_mixin.py.
"""

import concurrent.futures
import contextlib
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
import time
import queue
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..constants import NUM_PLAYERS, EP_PBS_INPUT_DIM
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, StrategyNetwork, HistoryValueNetwork, build_advantage_network
from ..persistence import atomic_torch_save, atomic_npz_save
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..utils import LogQueue as ProgressQueue
from ..live_display import LiveDisplayManager
from ..log_archiver import LogArchiver

from .deep_worker import run_deep_cfr_worker, DeepCFRWorkerResult
from .es_validator import ESValidator
from .exceptions import GracefulShutdownException, CheckpointSaveError, CheckpointLoadError
from ..serial_rotating_handler import SerialRotatingFileHandler

logger = logging.getLogger(__name__)


@dataclass
class DeepCFRConfig:
    """Configuration for Deep CFR training."""
    # Network architecture
    input_dim: int = INPUT_DIM
    hidden_dim: int = 256
    output_dim: int = NUM_ACTIONS
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 2048
    train_steps_per_iteration: int = 4000
    alpha: float = 1.5  # Weighting exponent for t^alpha loss

    # Traversals per training step
    traversals_per_step: int = 1000

    # Buffer capacity
    advantage_buffer_capacity: int = 2_000_000
    strategy_buffer_capacity: int = 2_000_000

    # Checkpointing
    save_interval: int = 10  # Save every N training steps (not traversals)

    # Device for training: "auto" = cuda if available, else xpu if available, else cpu
    device: str = "auto"

    # Sampling method: "outcome" or "external"
    sampling_method: str = "outcome"

    # Exploration epsilon for outcome sampling (ignored for external sampling)
    exploration_epsilon: float = 0.6

    # Engine backend: "python" or "go"
    engine_backend: str = "python"

    # ES Validation
    es_validation_interval: int = 10  # validate every N training steps
    es_validation_depth: int = 10  # max turns for ES validation
    es_validation_traversals: int = 1000  # traversals per validation

    # Pipeline training: overlap traversals with strategy network training
    pipeline_training: bool = True

    # Automatic mixed precision (FP16) on CUDA
    use_amp: bool = False

    # torch.compile for CUDA graph optimization
    use_compile: bool = False

    # Threads for Go FFI traversals (requires engine_backend="go")
    num_traversal_threads: int = 1

    # Validate network inputs (NaN check). Disable for GPU perf (avoids GPU→CPU sync).
    validate_inputs: bool = True

    # Residual network architecture
    use_residual: bool = False
    num_hidden_layers: int = 2

    # Max traversal depth (0 = unlimited, backward compatible)
    traversal_depth_limit: int = 0

    # Worker recycling to prevent RSS growth from glibc malloc fragmentation.
    # Pipeline workers' RSS grows ~723 MB/step without recycling.
    # "auto": calculate from system RAM and worker_memory_budget_pct
    # int: explicit recycling interval (kill/respawn worker every N steps)
    # None: never recycle (WARNING: RSS grows unbounded, ~723 MB/step)
    max_tasks_per_child: Optional[Union[int, str]] = "auto"

    # Fraction of system RAM budgeted per worker process (for auto calculation).
    # Default 0.10 (10%) allows ~8 concurrent training runs on the same machine.
    worker_memory_budget_pct: float = 0.10

    # ESCHER traversal method: "outcome" (OS-dCFR), "external" (ES-dCFR), or "escher"
    traversal_method: str = "outcome"

    # ESCHER value network hidden dimension
    value_hidden_dim: int = 512

    # ESCHER value network learning rate
    value_learning_rate: float = 1e-3

    # ESCHER value buffer capacity
    value_buffer_capacity: int = 2_000_000

    # ESCHER: whether to use batched counterfactual value estimation
    batch_counterfactuals: bool = True

    # SD-CFR: drop strategy network, use advantage snapshot averaging
    use_sd_cfr: bool = False
    sd_cfr_max_snapshots: int = 200
    sd_cfr_snapshot_weighting: str = "linear"  # "linear" or "uniform"
    num_hidden_layers: int = 3
    use_residual: bool = True
    use_ema: bool = True  # EMA serving weights for O(1) SD-CFR inference

    # Profiling: gate traversal timing logs + handle pool stats behind this flag
    enable_traversal_profiling: bool = False
    # Path for structured JSONL profiling output (empty = auto-generate in run dir)
    profiling_jsonl_path: str = ""
    # Run torch.profiler on this step number and export Chrome trace (None = disabled)
    profile_step: Optional[int] = None

    # Encoding mode: "legacy" (222-dim) or "ep_pbs" (200-dim EP-PBS encoding)
    encoding_mode: str = "legacy"

    # N-player configuration
    num_players: int = 2  # Number of players (2-6)

    # QRE regularization
    qre_lambda_start: float = 0.5   # Initial QRE temperature
    qre_lambda_end: float = 0.05    # Final QRE temperature
    qre_anneal_fraction: float = 0.6  # Fraction of total iterations to anneal over

    # PSRO configuration
    use_psro: bool = False
    psro_population_size: int = 15
    psro_eval_games: int = 200
    psro_checkpoint_interval: int = 50  # Add to PSRO population every N iterations
    psro_heuristic_types: str = "random,greedy,memory_heuristic"  # Comma-separated

    def __post_init__(self):
        if self.pipeline_training and self.num_traversal_threads > 1:
            raise ValueError(
                "pipeline_training=True and num_traversal_threads>1 are mutually exclusive. "
                "Pipeline training already parallelises traversal and training in separate "
                "processes; combining it with multi-threaded traversal is not supported."
            )
        # Override input_dim/output_dim based on num_players or encoding_mode
        if self.num_players > 2:
            from ..constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS
            self.input_dim = N_PLAYER_INPUT_DIM
            self.output_dim = N_PLAYER_NUM_ACTIONS
        elif self.encoding_mode == "ep_pbs":
            self.input_dim = EP_PBS_INPUT_DIM
        elif self.encoding_mode != "legacy":
            raise ValueError(
                f"Unknown encoding_mode: {self.encoding_mode!r}. Must be 'legacy' or 'ep_pbs'."
            )

    @classmethod
    def from_yaml_config(cls, config: "Config", **overrides) -> "DeepCFRConfig":
        """Construct DeepCFRConfig from Config.deep_cfr, applying CLI overrides."""
        deep_cfg = config.deep_cfr
        kwargs = {
            "hidden_dim": deep_cfg.hidden_dim,
            "dropout": deep_cfg.dropout,
            "learning_rate": deep_cfg.learning_rate,
            "batch_size": deep_cfg.batch_size,
            "train_steps_per_iteration": deep_cfg.train_steps_per_iteration,
            "alpha": deep_cfg.alpha,
            "traversals_per_step": deep_cfg.traversals_per_step,
            "advantage_buffer_capacity": deep_cfg.advantage_buffer_capacity,
            "strategy_buffer_capacity": deep_cfg.strategy_buffer_capacity,
            "save_interval": deep_cfg.save_interval,
            "device": getattr(deep_cfg, "device", "auto"),
            "sampling_method": deep_cfg.sampling_method,
            "exploration_epsilon": deep_cfg.exploration_epsilon,
            "engine_backend": getattr(deep_cfg, "engine_backend", "python"),
            "es_validation_interval": getattr(deep_cfg, "es_validation_interval", 10),
            "es_validation_depth": getattr(deep_cfg, "es_validation_depth", 10),
            "es_validation_traversals": getattr(deep_cfg, "es_validation_traversals", 1000),
            "pipeline_training": getattr(deep_cfg, "pipeline_training", True),
            "use_amp": getattr(deep_cfg, "use_amp", False),
            "use_compile": getattr(deep_cfg, "use_compile", False),
            "num_traversal_threads": getattr(deep_cfg, "num_traversal_threads", 1),
            "validate_inputs": getattr(deep_cfg, "validate_inputs", True),
            "traversal_depth_limit": getattr(deep_cfg, "traversal_depth_limit", 0),
            "max_tasks_per_child": getattr(deep_cfg, "max_tasks_per_child", "auto"),
            "worker_memory_budget_pct": getattr(deep_cfg, "worker_memory_budget_pct", 0.10),
            "use_residual": getattr(deep_cfg, "use_residual", False),
            "num_hidden_layers": getattr(deep_cfg, "num_hidden_layers", 2),
            "traversal_method": getattr(deep_cfg, "traversal_method", "outcome"),
            "value_hidden_dim": getattr(deep_cfg, "value_hidden_dim", 512),
            "value_learning_rate": getattr(deep_cfg, "value_learning_rate", 1e-3),
            "value_buffer_capacity": getattr(deep_cfg, "value_buffer_capacity", 2_000_000),
            "batch_counterfactuals": getattr(deep_cfg, "batch_counterfactuals", True),
            "use_sd_cfr": getattr(deep_cfg, "use_sd_cfr", True),
            "sd_cfr_max_snapshots": getattr(deep_cfg, "sd_cfr_max_snapshots", 200),
            "sd_cfr_snapshot_weighting": getattr(deep_cfg, "sd_cfr_snapshot_weighting", "linear"),
            "num_hidden_layers": getattr(deep_cfg, "num_hidden_layers", 3),
            "use_residual": getattr(deep_cfg, "use_residual", True),
            "use_ema": getattr(deep_cfg, "use_ema", True),
            "enable_traversal_profiling": getattr(deep_cfg, "enable_traversal_profiling", False),
            "profiling_jsonl_path": getattr(deep_cfg, "profiling_jsonl_path", ""),
            "profile_step": getattr(deep_cfg, "profile_step", None),
            "encoding_mode": getattr(deep_cfg, "encoding_mode", "legacy"),
            "num_players": getattr(deep_cfg, "num_players", 2),
            "qre_lambda_start": getattr(deep_cfg, "qre_lambda_start", 0.5),
            "qre_lambda_end": getattr(deep_cfg, "qre_lambda_end", 0.05),
            "qre_anneal_fraction": getattr(deep_cfg, "qre_anneal_fraction", 0.6),
            "use_psro": getattr(deep_cfg, "use_psro", False),
            "psro_population_size": getattr(deep_cfg, "psro_population_size", 15),
            "psro_eval_games": getattr(deep_cfg, "psro_eval_games", 200),
            "psro_checkpoint_interval": getattr(deep_cfg, "psro_checkpoint_interval", 50),
            "psro_heuristic_types": getattr(deep_cfg, "psro_heuristic_types", "random,greedy,memory_heuristic"),
        }
        # Apply CLI overrides (only non-None values)
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return cls(**kwargs)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device string to a concrete device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"
    if device == "xpu":
        if not hasattr(torch, "xpu"):
            raise RuntimeError(
                "Device 'xpu' requested but intel_extension_for_pytorch is not installed. "
                "Install it with: pip install intel_extension_for_pytorch"
            )
        if not torch.xpu.is_available():
            raise RuntimeError(
                "Device 'xpu' requested but no XPU device is available. "
                "Ensure an Intel Arc GPU is present and IPEX is installed correctly."
            )
    return device


def _resolve_max_tasks_per_child(
    max_tasks_per_child: Optional[Union[int, str]],
    worker_memory_budget_pct: float,
) -> Optional[int]:
    """Resolve max_tasks_per_child for ProcessPoolExecutor.

    Pipeline worker RSS grows ~723 MB per training step due to glibc malloc
    fragmentation (Python frees objects but glibc retains heap pages). This
    function calculates how many steps a worker can run before being recycled.

    Empirical constants (measured Feb 2026, Phase 1 profiling):
        WORKER_WARMUP_MB = 1600   # fresh worker baseline RSS after model load
        WORKER_GROWTH_MB = 723    # RSS increase per training step (1000 traversals)

    Auto formula:
        budget_mb = system_ram_gb * worker_memory_budget_pct * 1024
        max_tasks = floor((budget_mb - WORKER_WARMUP_MB) / WORKER_GROWTH_MB)
        return clamp(max_tasks, 2, 100)

    Args:
        max_tasks_per_child: "auto", positive int, or None (no recycling).
        worker_memory_budget_pct: fraction of system RAM per worker (default 0.10).

    Returns:
        Resolved integer for ProcessPoolExecutor max_tasks_per_child, or None.
    """
    if max_tasks_per_child is None:
        logger.warning(
            "max_tasks_per_child=None: worker recycling disabled. "
            "RSS will grow ~723 MB/step unbounded. Only use for short test runs."
        )
        return None

    if isinstance(max_tasks_per_child, int):
        if max_tasks_per_child < 1:
            raise ValueError(f"max_tasks_per_child must be >= 1, got {max_tasks_per_child}")
        return max_tasks_per_child

    if isinstance(max_tasks_per_child, str) and max_tasks_per_child.lower() == "auto":
        WORKER_WARMUP_MB = 1600
        WORKER_GROWTH_MB = 723

        try:
            import psutil
            total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            # Fallback: read /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_mb = int(line.split()[1]) / 1024  # kB to MB
                            break
                    else:
                        total_ram_mb = 16 * 1024  # conservative 16 GB fallback
            except OSError:
                total_ram_mb = 16 * 1024

        budget_mb = total_ram_mb * worker_memory_budget_pct
        raw = int((budget_mb - WORKER_WARMUP_MB) / WORKER_GROWTH_MB)
        resolved = max(2, min(raw, 100))

        logger.info(
            "Auto max_tasks_per_child=%d (system_ram=%.1f GB, budget_pct=%.0f%%, "
            "budget=%.0f MB, warmup=%d MB, growth=%d MB/step)",
            resolved,
            total_ram_mb / 1024,
            worker_memory_budget_pct * 100,
            budget_mb,
            WORKER_WARMUP_MB,
            WORKER_GROWTH_MB,
        )
        return resolved

    raise ValueError(
        f"max_tasks_per_child must be 'auto', a positive int, or None. Got: {max_tasks_per_child!r}"
    )


def _create_worker_file_handler(
    config, worker_id: int, run_log_dir: str, run_timestamp: str, archive_queue=None,
) -> SerialRotatingFileHandler:
    """Create a single SerialRotatingFileHandler for a worker slot.

    Called once per batch/step so that glob.glob() runs once to find the
    correct serial number, then the handler increments in memory for
    subsequent log files within the same step.
    """
    worker_log_dir = os.path.join(run_log_dir, f"w{worker_id}")
    os.makedirs(worker_log_dir, exist_ok=True)
    log_pattern = os.path.join(
        worker_log_dir,
        f"{config.logging.log_file_prefix}_run_{run_timestamp}-w{worker_id}",
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - [%(processName)-20s] - %(name)-25s - %(message)s"
    )
    handler = SerialRotatingFileHandler(
        filename_pattern=log_pattern,
        maxBytes=config.logging.log_max_bytes,
        backupCount=config.logging.log_backup_count,
        encoding="utf-8",
        archive_queue=archive_queue,
        logging_config_snapshot=config.logging,
    )
    handler.setFormatter(formatter)
    return handler


def _run_traversals_batch(
    iteration_offset: int,
    total_traversals_offset: int,
    config,
    network_weights: Dict[str, Any],
    network_config: Dict[str, int],
    traversals_per_step: int,
    num_workers: int,
    run_log_dir: str,
    run_timestamp: str,
    progress_queue=None,
    archive_queue=None,
) -> Tuple[List[ReservoirSample], List[ReservoirSample], List[ReservoirSample], int, int, Dict[str, float]]:
    """Run all traversals for one training step.

    Returns (adv_samples, strat_samples, value_samples, traversals_done, total_nodes, timing_stats).
    timing_stats is a dict with keys: min_s, max_s, mean_s, total_s, count.
    """
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    value_samples: List[ReservoirSample] = []
    traversals_done = 0
    total_nodes = 0
    traversal_times: List[float] = []

    # Create file handler ONCE for the batch — avoids glob.glob() per traversal.
    file_handler = _create_worker_file_handler(
        config, 0, run_log_dir, run_timestamp, archive_queue,
    )

    try:
        for i in range(traversals_per_step):
            iter_num = iteration_offset + i
            args_tuple = (
                iter_num,
                config,
                network_weights,
                network_config,
                progress_queue,
                archive_queue,
                0,  # worker_id: constant slot (not per-traversal index)
                run_log_dir,
                run_timestamp,
            )
            _t0 = time.time()
            result = run_deep_cfr_worker(args_tuple, file_handler_override=file_handler)
            traversal_times.append(time.time() - _t0)
            if isinstance(result, DeepCFRWorkerResult):
                advantage_samples.extend(result.advantage_samples)
                strategy_samples.extend(result.strategy_samples)
                if hasattr(result, "value_samples") and result.value_samples:
                    value_samples.extend(result.value_samples)
                total_nodes += result.stats.nodes_visited
                if result.stats.error_count > 0:
                    logger.warning("Worker reported %d errors.", result.stats.error_count)
            traversals_done += 1
    finally:
        try:
            file_handler.close()
        except Exception:
            pass

    timing_stats: Dict[str, float] = {
        "min_s": min(traversal_times) if traversal_times else 0.0,
        "max_s": max(traversal_times) if traversal_times else 0.0,
        "mean_s": (sum(traversal_times) / len(traversal_times)) if traversal_times else 0.0,
        "total_s": sum(traversal_times),
        "count": float(len(traversal_times)),
    }
    return advantage_samples, strategy_samples, value_samples, traversals_done, total_nodes, timing_stats


def _run_single_traversal(args_tuple, file_handler_override=None):
    """Wrapper for a single traversal call, used by ThreadPoolExecutor."""
    result = run_deep_cfr_worker(args_tuple, file_handler_override=file_handler_override)
    return result


def _run_traversals_threaded(
    iteration_offset: int,
    config,
    network_weights: Dict[str, Any],
    network_config: Dict[str, int],
    traversals_per_step: int,
    num_threads: int,
    run_log_dir: str,
    run_timestamp: str,
    progress_queue=None,
    archive_queue=None,
) -> Tuple[List[ReservoirSample], List[ReservoirSample], List[ReservoirSample], int, int]:
    """Run traversals using ThreadPoolExecutor for Go FFI backend.

    Threads share process memory so the advantage network can be used read-only
    without serialization. Each thread creates its own GoEngine instance; the
    Go handle pool is mutex-protected (task #6).

    Returns (adv_samples, strat_samples, value_samples, traversals_done, total_nodes).
    """
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    value_samples: List[ReservoirSample] = []
    traversals_done = 0
    total_nodes = 0

    # Create one file handler per thread slot ONCE — avoids glob.glob() per traversal.
    worker_handlers: Dict[int, SerialRotatingFileHandler] = {}
    for slot in range(num_threads):
        worker_handlers[slot] = _create_worker_file_handler(
            config, slot, run_log_dir, run_timestamp, archive_queue,
        )

    worker_args_list = []
    handler_for_args = []
    for i in range(traversals_per_step):
        iter_num = iteration_offset + i
        slot = i % num_threads
        worker_args_list.append((
            iter_num,
            config,
            network_weights,
            network_config,
            progress_queue,
            archive_queue,
            slot,  # worker_id: thread pool slot
            run_log_dir,
            run_timestamp,
        ))
        handler_for_args.append(worker_handlers[slot])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_pool:
            futures = [
                thread_pool.submit(_run_single_traversal, args, fh)
                for args, fh in zip(worker_args_list, handler_for_args)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, DeepCFRWorkerResult):
                    advantage_samples.extend(result.advantage_samples)
                    strategy_samples.extend(result.strategy_samples)
                    if hasattr(result, "value_samples") and result.value_samples:
                        value_samples.extend(result.value_samples)
                    total_nodes += result.stats.nodes_visited
                    if result.stats.error_count > 0:
                        logger.warning("Worker reported %d errors.", result.stats.error_count)
                traversals_done += 1
    finally:
        for handler in worker_handlers.values():
            try:
                handler.close()
            except Exception:
                pass

    return advantage_samples, strategy_samples, value_samples, traversals_done, total_nodes


def qre_strategy(
    advantages: torch.Tensor,
    legal_mask: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute QRE (softmax) strategy from advantages with temperature lambda.

    CRITICAL IMPLEMENTATION NOTES (from researcher review _1c):
    - Mask illegal actions with -inf via masked_fill
    - Use per-row max via max(dim=-1, keepdim=True) for numerical stability
    - Do NOT use advantages[legal_mask].max() — this flattens across batch dim,
      computing a global max that causes cross-row underflow → NaN

    Args:
        advantages: (batch, num_actions) tensor
        legal_mask: (batch, num_actions) bool tensor, True = legal
        lam: QRE temperature (positive float). Higher = more uniform.

    Returns:
        (batch, num_actions) strategy tensor (sums to 1 per row over legal actions)
    """
    # Mask illegal actions to -inf
    masked_adv = advantages.masked_fill(~legal_mask, float('-inf'))

    # Per-row max for numerical stability (NOT global max!)
    row_max = masked_adv.max(dim=-1, keepdim=True).values

    # Stable softmax with temperature
    safe_adv = masked_adv - row_max
    logits = (safe_adv / max(lam, 1e-8)).clamp(-88, 88)
    sigma = torch.softmax(logits, dim=-1)

    # Zero out illegal actions (softmax might assign tiny probs due to -inf arithmetic)
    sigma = sigma * legal_mask.float()

    # Re-normalize
    row_sum = sigma.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sigma = sigma / row_sum

    return sigma


class DeepCFRTrainer:
    """
    Orchestrates Deep CFR training with external sampling workers.

    Replaces the tabular CFRTrainer. Uses:
    - AdvantageNetwork for regret prediction
    - StrategyNetwork for average strategy
    - ReservoirBuffers for training sample storage
    - Multiprocessing pool for parallel traversals
    """

    def __init__(
        self,
        config: Config,
        deep_cfr_config: Optional[DeepCFRConfig] = None,
        run_log_dir: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        progress_queue: Optional[ProgressQueue] = None,
        live_display_manager: Optional[LiveDisplayManager] = None,
        archive_queue: Optional[Union[queue.Queue, "multiprocessing.Queue"]] = None,
    ):
        self.config = config
        self.dcfr_config = deep_cfr_config or DeepCFRConfig()
        self.run_log_dir = run_log_dir
        self.run_timestamp = run_timestamp
        self.shutdown_event = shutdown_event or threading.Event()
        self.progress_queue = progress_queue
        self.live_display_manager = live_display_manager
        self.archive_queue = archive_queue
        self.log_archiver_global_ref: Optional[LogArchiver] = None

        # Device selection
        resolved_device = _resolve_device(self.dcfr_config.device)
        self.device = torch.device(resolved_device)
        logger.info("Deep CFR Trainer using device: %s", self.device)

        # Networks
        self.advantage_net = build_advantage_network(
            input_dim=self.dcfr_config.input_dim,
            hidden_dim=self.dcfr_config.hidden_dim,
            output_dim=self.dcfr_config.output_dim,
            dropout=self.dcfr_config.dropout,
            validate_inputs=self.dcfr_config.validate_inputs,
            num_hidden_layers=self.dcfr_config.num_hidden_layers,
            use_residual=self.dcfr_config.use_residual,
        ).to(self.device)

        if not self.dcfr_config.use_sd_cfr:
            self.strategy_net = StrategyNetwork(
                input_dim=self.dcfr_config.input_dim,
                hidden_dim=self.dcfr_config.hidden_dim,
                output_dim=self.dcfr_config.output_dim,
                dropout=self.dcfr_config.dropout,
                validate_inputs=self.dcfr_config.validate_inputs,
            ).to(self.device)
        else:
            self.strategy_net = None

        # B5: torch.compile — gated by config (effective on CUDA and XPU)
        if self.dcfr_config.use_compile and hasattr(torch, "compile") and self.device.type != "cpu":
            try:
                self.advantage_net = torch.compile(self.advantage_net, mode="reduce-overhead")
                if self.strategy_net is not None:
                    self.strategy_net = torch.compile(self.strategy_net, mode="reduce-overhead")
                logger.info("Networks compiled with torch.compile (reduce-overhead mode)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

        # Optimizers
        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=self.dcfr_config.learning_rate
        )
        if self.strategy_net is not None:
            self.strategy_optimizer = optim.Adam(
                self.strategy_net.parameters(), lr=self.dcfr_config.learning_rate
            )
        else:
            self.strategy_optimizer = None

        # ESCHER value network (only when traversal_method == "escher")
        self._is_escher = self.dcfr_config.traversal_method == "escher"
        if self._is_escher:
            self.value_net: Optional[HistoryValueNetwork] = HistoryValueNetwork(
                input_dim=INPUT_DIM * 2,
                hidden_dim=self.dcfr_config.value_hidden_dim,
                validate_inputs=self.dcfr_config.validate_inputs,
            ).to(self.device)
            self.value_optimizer: Optional[optim.Optimizer] = optim.Adam(
                self.value_net.parameters(), lr=self.dcfr_config.value_learning_rate
            )
            self.value_buffer: Optional[ReservoirBuffer] = ReservoirBuffer(
                capacity=self.dcfr_config.value_buffer_capacity,
                input_dim=INPUT_DIM * 2,
                target_dim=1,
                has_mask=False,
            )
        else:
            self.value_net = None
            self.value_optimizer = None
            self.value_buffer = None

        # B4: AMP scaler — gated by config (effective on CUDA and XPU)
        self.use_amp = self.dcfr_config.use_amp and self.device.type != "cpu"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        # Reservoir buffers
        self.advantage_buffer = ReservoirBuffer(
            capacity=self.dcfr_config.advantage_buffer_capacity,
            input_dim=self.dcfr_config.input_dim,
        )
        if not self.dcfr_config.use_sd_cfr:
            self.strategy_buffer = ReservoirBuffer(
                capacity=self.dcfr_config.strategy_buffer_capacity,
                input_dim=self.dcfr_config.input_dim,
            )
        else:
            self.strategy_buffer = None

        # Training state
        self.current_iteration = 0
        self.total_traversals = 0
        self.training_step = 0

        # Tracking
        self.advantage_loss_history: List[Tuple[int, float]] = []
        self.strategy_loss_history: List[Tuple[int, float]] = []
        self.es_validation_history: List[Tuple[int, Dict]] = []

        # SD-CFR snapshot storage
        self._sd_snapshots: List[Dict[str, np.ndarray]] = []
        self._sd_snapshot_iterations: List[int] = []
        self._snapshot_count: int = 0  # Total snapshots ever taken (for reservoir sampling)

        # EMA serving weights: O(1) inference alternative to full snapshot averaging.
        # Tracks the linearly t-weighted (alpha=1.5) ensemble as an online EMA.
        # Only active when use_sd_cfr=True and use_ema=True.
        self._ema_state_dict: Optional[Dict[str, np.ndarray]] = None
        self._ema_weight_sum: float = 0.0

        # PSRO population oracle (optional)
        if self.dcfr_config.use_psro:
            from .psro import PSROOracle
            heuristic_types = [
                t.strip() for t in self.dcfr_config.psro_heuristic_types.split(",") if t.strip()
            ]
            self._psro_oracle: Optional["PSROOracle"] = PSROOracle(
                max_checkpoints=self.dcfr_config.psro_population_size,
                heuristic_types=heuristic_types,
            )
            logger.info(
                "PSRO enabled: max_checkpoints=%d, heuristics=%s",
                self.dcfr_config.psro_population_size, heuristic_types,
            )
        else:
            self._psro_oracle = None

        adv_params = sum(p.numel() for p in self.advantage_net.parameters())
        strat_params = sum(p.numel() for p in self.strategy_net.parameters()) if self.strategy_net else 0
        logger.info(
            "DeepCFRTrainer initialized. Advantage net params: %d, Strategy net params: %d, SD-CFR: %s, PSRO: %s",
            adv_params, strat_params, self.dcfr_config.use_sd_cfr, self.dcfr_config.use_psro,
        )

    def _get_network_weights_for_workers(self) -> Dict[str, Any]:
        """
        Serialize advantage network weights for distribution to workers.
        Returns numpy arrays for pickle-friendly multiprocessing transfer.
        """
        state_dict = self.advantage_net.state_dict()
        return {k: v.cpu().numpy() for k, v in state_dict.items()}

    def _get_network_config(self) -> Dict[str, Any]:
        """Returns network configuration dict for workers."""
        return {
            "input_dim": self.dcfr_config.input_dim,
            "hidden_dim": self.dcfr_config.hidden_dim,
            "output_dim": self.dcfr_config.output_dim,
            "validate_inputs": self.dcfr_config.validate_inputs,
            "encoding_mode": self.dcfr_config.encoding_mode,
            "num_players": self.dcfr_config.num_players,
            "use_residual": self.dcfr_config.use_residual,
            "num_hidden_layers": self.dcfr_config.num_hidden_layers,
        }

    def _take_advantage_snapshot(self):
        """Serialize current advantage_net weights as a numpy snapshot for SD-CFR averaging."""
        snapshot = {k: v.cpu().numpy() for k, v in self.advantage_net.state_dict().items()}
        max_snaps = self.dcfr_config.sd_cfr_max_snapshots
        self._snapshot_count += 1

        if len(self._sd_snapshots) < max_snaps:
            self._sd_snapshots.append(snapshot)
            self._sd_snapshot_iterations.append(self.training_step)
        else:
            # Reservoir sampling (Vitter's Algorithm R)
            j = np.random.randint(0, self._snapshot_count)
            if j < max_snaps:
                self._sd_snapshots[j] = snapshot
                self._sd_snapshot_iterations[j] = self.training_step

    def _update_ema(self):
        """
        Update EMA serving weights after each advantage snapshot.

        Uses the corrected formula for non-uniform alpha weighting:
          w_T = (T+1)^alpha
          new_sum = old_sum + w_T
          θ_EMA = (old_sum / new_sum) * θ_EMA + (w_T / new_sum) * θ_current

        This tracks the same weighted ensemble as full snapshot averaging
        but in O(1) space and O(params) time per update.
        """
        if not self.dcfr_config.use_ema or not self.dcfr_config.use_sd_cfr:
            return

        w_T = float((self.training_step + 1) ** self.dcfr_config.alpha)
        new_sum = self._ema_weight_sum + w_T
        current_weights = {k: v.cpu().numpy() for k, v in self.advantage_net.state_dict().items()}

        if self._ema_state_dict is None:
            self._ema_state_dict = {k: v.copy() for k, v in current_weights.items()}
            self._ema_weight_sum = w_T
            return

        ratio_old = self._ema_weight_sum / new_sum
        ratio_new = w_T / new_sum
        for key in self._ema_state_dict:
            self._ema_state_dict[key] = (
                ratio_old * self._ema_state_dict[key] + ratio_new * current_weights[key]
            )
        self._ema_weight_sum = new_sum

    def _prefetch_batches(self, buffer, batch_size, num_steps, prefetch_queue):
        """Background thread: prepare batches and put them in the queue."""
        for _ in range(num_steps):
            batch = buffer.sample_batch(batch_size)
            if not batch:
                break
            features_t = torch.from_numpy(batch.features).float().pin_memory()
            targets_t = torch.from_numpy(batch.targets).float().pin_memory()
            masks_t = torch.from_numpy(batch.masks).pin_memory()
            iterations_t = torch.from_numpy(batch.iterations.astype(np.float32)).pin_memory()
            prefetch_queue.put((features_t, targets_t, masks_t, iterations_t))
        prefetch_queue.put(None)  # sentinel

    def _train_network(
        self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        buffer: ReservoirBuffer,
        alpha: float,
        num_steps: int,
        network_name: str,
    ) -> float:
        """
        Train a network on reservoir buffer samples using weighted MSE loss.

        Loss = (t^alpha) * MSE(network(features), target)
        where t is the iteration number stored in each sample.

        Returns average loss over all training steps.
        """
        if len(buffer) == 0:
            logger.warning("Cannot train %s: buffer is empty.", network_name)
            return 0.0

        network.train()
        total_loss = 0.0
        actual_steps = 0
        batch_size = self.dcfr_config.batch_size

        def _do_train_step(features_t, targets_t, masks_t, iterations_t):
            nonlocal total_loss, actual_steps
            # Compute iteration weights: (t + 1)^alpha to avoid 0^alpha for iteration 0
            weights = (iterations_t + 1.0).pow(alpha)
            # Normalize weights to prevent loss magnitude from growing with iterations
            weights = weights / weights.mean()

            optimizer.zero_grad()

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                predictions = network(features_t, masks_t)
                # Use masked_fill instead of multiply to avoid -inf * 0 = NaN
                # (AdvantageNetwork sets illegal actions to -inf)
                masked_preds = predictions.masked_fill(~masks_t, 0.0)
                masked_targets = targets_t.masked_fill(~masks_t, 0.0)
                num_legal = masks_t.float().sum(dim=1).clamp(min=1.0)
                per_sample_mse = ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
                weighted_loss = (weights * per_sample_mse).mean()

            self.scaler.scale(weighted_loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += weighted_loss.item()
            actual_steps += 1

        if self.device.type == "cuda":
            prefetch_queue = queue.Queue(maxsize=2)
            prefetch_thread = threading.Thread(
                target=self._prefetch_batches,
                args=(buffer, batch_size, num_steps, prefetch_queue),
                daemon=True,
            )
            prefetch_thread.start()

            for step in range(num_steps):
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected during %s training.", network_name)
                    break
                batch_data = prefetch_queue.get()
                if batch_data is None:
                    break
                features_t, targets_t, masks_t, iterations_t = batch_data
                features_t = features_t.to(self.device, non_blocking=True)
                targets_t = targets_t.to(self.device, non_blocking=True)
                masks_t = masks_t.to(self.device, non_blocking=True)
                iterations_t = iterations_t.to(self.device, non_blocking=True)
                _do_train_step(features_t, targets_t, masks_t, iterations_t)

            prefetch_thread.join(timeout=5.0)
        else:
            for step in range(num_steps):
                if self.shutdown_event.is_set():
                    logger.warning("Shutdown detected during %s training.", network_name)
                    break

                batch = buffer.sample_batch(batch_size)
                if not batch:
                    break

                # ColumnarBatch already provides pre-stacked numpy arrays
                features_t = torch.from_numpy(batch.features).float().to(self.device)
                targets_t = torch.from_numpy(batch.targets).float().to(self.device)
                masks_t = torch.from_numpy(batch.masks).to(self.device)
                iterations_t = torch.from_numpy(batch.iterations.astype(np.float32)).to(self.device)
                _do_train_step(features_t, targets_t, masks_t, iterations_t)

        avg_loss = total_loss / max(actual_steps, 1)
        logger.info(
            "%s training: %d steps, avg loss: %.6f (buffer size: %d)",
            network_name, actual_steps, avg_loss, len(buffer),
        )
        return avg_loss

    def _shutdown_pool(self, pool: Optional[multiprocessing.pool.Pool]):
        """Gracefully shuts down the multiprocessing pool."""
        if pool:
            pool_running = getattr(pool, "_state", -1) == multiprocessing.pool.RUN
            if pool_running:
                logger.info("Terminating worker pool...")
                try:
                    pool.terminate()
                    time.sleep(0.5)
                    pool.join()
                    logger.info("Worker pool terminated and joined.")
                except ValueError:
                    logger.warning("Pool already closed.")
                except Exception as e:
                    logger.error("Error during pool shutdown: %s", e, exc_info=True)
            else:
                logger.info("Worker pool already terminated.")

    def _get_qre_lambda(self) -> float:
        """Compute current QRE lambda based on training iteration and annealing schedule.

        Returns float('inf') when QRE is not configured, which results in standard
        regret matching (infinite temperature → uniform strategy).
        """
        if not hasattr(self.dcfr_config, 'qre_lambda_start'):
            return float('inf')  # No QRE → standard regret matching

        start = self.dcfr_config.qre_lambda_start
        end = self.dcfr_config.qre_lambda_end
        frac = self.dcfr_config.qre_anneal_fraction
        total_iters = self.config.cfr_training.num_iterations

        anneal_end = int(total_iters * frac)
        if self.training_step >= anneal_end or anneal_end == 0:
            return end

        progress = self.training_step / anneal_end
        return start + (end - start) * progress

    def train(self, num_training_steps: Optional[int] = None):
        """
        Main Deep CFR training loop.

        Each training step:
        1. Distribute advantage network weights to workers
        2. Run K traversals in parallel
        3. Collect samples into reservoir buffers
        4. Train advantage network on advantage buffer
        5. Train strategy network on strategy buffer
        """
        cfr_config = getattr(self.config, "cfr_training", None)
        if not cfr_config:
            logger.critical("CFRTrainingConfig not found. Cannot train.")
            return

        total_steps = num_training_steps or getattr(cfr_config, "num_iterations", 100)
        num_workers = getattr(cfr_config, "num_workers", 1)
        save_interval = self.dcfr_config.save_interval
        traversals_per_step = self.dcfr_config.traversals_per_step
        display = self.live_display_manager

        start_step = self.training_step + 1
        end_step = self.training_step + total_steps

        # Tier 2: open JSONL profiling file if enabled
        _jsonl_file = None
        if self.dcfr_config.enable_traversal_profiling:
            _jsonl_path = self.dcfr_config.profiling_jsonl_path
            if not _jsonl_path:
                # Auto-generate path alongside the checkpoint
                _ckpt_path = getattr(
                    self.config.persistence, "agent_data_save_path",
                    "strategy/deep_cfr_checkpoint.pt",
                )
                _run_dir = os.path.dirname(_ckpt_path) if os.path.dirname(_ckpt_path) else "."
                _jsonl_path = os.path.join(_run_dir, "profiling.jsonl")
            try:
                os.makedirs(os.path.dirname(_jsonl_path) if os.path.dirname(_jsonl_path) else ".", exist_ok=True)
                _jsonl_file = open(_jsonl_path, "a", encoding="utf-8")
                logger.info("Profiling JSONL output: %s", _jsonl_path)
            except Exception as _e:
                logger.warning("Failed to open profiling JSONL file %s: %s", _jsonl_path, _e)
                _jsonl_file = None

        sampling_method = self.dcfr_config.sampling_method
        logger.info(
            "Starting Deep CFR training from step %d to %d (%d workers, %d traversals/step, sampling=%s).",
            start_step, end_step, num_workers, traversals_per_step, sampling_method,
        )
        if sampling_method == "outcome":
            logger.info(
                "Using Outcome Sampling MCCFR with exploration_epsilon=%.2f",
                self.dcfr_config.exploration_epsilon,
            )
        if self.dcfr_config.engine_backend == "go" and self.dcfr_config.num_traversal_threads > 1:
            logger.info(
                "Using ThreadPoolExecutor with %d threads for Go FFI traversals.",
                self.dcfr_config.num_traversal_threads,
            )

        pool: Optional[multiprocessing.pool.Pool] = None
        pending_future: Optional[concurrent.futures.Future] = None
        executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

        if self.dcfr_config.pipeline_training:
            ctx = multiprocessing.get_context("spawn")
            _max_tasks = _resolve_max_tasks_per_child(
                self.dcfr_config.max_tasks_per_child,
                self.dcfr_config.worker_memory_budget_pct,
            )
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=ctx, max_tasks_per_child=_max_tasks
            )

        network_config = self._get_network_config()

        try:
            for step in range(start_step, end_step + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before training step")

                step_start_time = time.time()
                self.training_step = step
                phase_times = {}  # phase name -> elapsed seconds

                # Tier 3: enter torch.profiler for this step if configured
                _profiling_this_step = (
                    self.dcfr_config.profile_step is not None
                    and step == self.dcfr_config.profile_step
                )
                _prof = None
                if _profiling_this_step:
                    import torch.profiler as _torch_profiler
                    _prof_activities = [_torch_profiler.ProfilerActivity.CPU]
                    if self.device.type == "cuda":
                        _prof_activities.append(_torch_profiler.ProfilerActivity.CUDA)
                    _prof = _torch_profiler.profile(
                        activities=_prof_activities,
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                    )
                    _prof.__enter__()

                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Running traversals...")

                # Get current network weights for workers
                _t0 = time.time()
                network_weights = self._get_network_weights_for_workers()
                phase_times["weights_copy"] = time.time() - _t0

                # Collect traversal results — either from pipeline future or synchronously
                _trav_start = time.time()
                if pending_future is not None:
                    # B3: Collect results from pipelined traversal started after prev step's adv training
                    _future_result = pending_future.result()
                    step_advantage_samples, step_strategy_samples, step_value_samples, traversals_done, total_nodes, _trav_timing = _future_result
                    pending_future = None
                    if self.dcfr_config.enable_traversal_profiling:
                        logger.info(
                            "Step %d traversal timing (pipeline): min=%.3fs max=%.3fs mean=%.3fs total=%.3fs count=%d",
                            step,
                            _trav_timing["min_s"],
                            _trav_timing["max_s"],
                            _trav_timing["mean_s"],
                            _trav_timing["total_s"],
                            int(_trav_timing["count"]),
                        )
                elif num_workers > 1:
                    # Parallel pool-based traversals (non-pipelined path)
                    step_advantage_samples: List[ReservoirSample] = []
                    step_strategy_samples: List[ReservoirSample] = []
                    step_value_samples: List[ReservoirSample] = []
                    traversals_done = 0
                    total_nodes = 0

                    while traversals_done < traversals_per_step:
                        if self.shutdown_event.is_set():
                            raise GracefulShutdownException("Shutdown during traversals")

                        batch_size = min(num_workers, traversals_per_step - traversals_done)

                        worker_args_list = []
                        for i in range(batch_size):
                            iter_num = self.total_traversals + traversals_done + i
                            worker_args_list.append((
                                iter_num,
                                self.config,
                                network_weights,
                                network_config,
                                self.progress_queue,
                                self.archive_queue,
                                i % num_workers,  # worker_id: pool slot (not traversal index)
                                self.run_log_dir or "logs",
                                self.run_timestamp or "unknown",
                            ))

                        if not pool:
                            logger.info("Creating worker pool (size %d)...", num_workers)
                            pool = multiprocessing.Pool(processes=num_workers)

                        try:
                            async_results = pool.map_async(
                                run_deep_cfr_worker, worker_args_list
                            )
                            while not async_results.ready():
                                if self.shutdown_event.is_set():
                                    self._shutdown_pool(pool)
                                    pool = None
                                    raise GracefulShutdownException(
                                        "Shutdown during parallel workers"
                                    )
                                async_results.wait(timeout=0.5)
                            results = async_results.get()
                        except (GracefulShutdownException, KeyboardInterrupt):
                            self._shutdown_pool(pool)
                            pool = None
                            raise
                        except Exception as e_pool:
                            logger.exception("Worker pool error: %s", e_pool)
                            self._shutdown_pool(pool)
                            pool = None
                            raise

                        for result in results:
                            if isinstance(result, DeepCFRWorkerResult):
                                step_advantage_samples.extend(result.advantage_samples)
                                step_strategy_samples.extend(result.strategy_samples)
                                if hasattr(result, "value_samples") and result.value_samples:
                                    step_value_samples.extend(result.value_samples)
                                total_nodes += result.stats.nodes_visited
                                if result.stats.error_count > 0:
                                    logger.warning(
                                        "Worker reported %d errors.", result.stats.error_count
                                    )

                        traversals_done += batch_size
                elif (
                    self.dcfr_config.engine_backend == "go"
                    and self.dcfr_config.num_traversal_threads > 1
                ):
                    # Threaded path for Go FFI backend — threads share the
                    # advantage network read-only; each thread gets its own
                    # GoEngine instance (handle pool is mutex-protected).
                    step_advantage_samples, step_strategy_samples, step_value_samples, traversals_done, total_nodes = (
                        _run_traversals_threaded(
                            self.total_traversals,
                            self.config,
                            network_weights,
                            network_config,
                            traversals_per_step,
                            self.dcfr_config.num_traversal_threads,
                            self.run_log_dir or "logs",
                            self.run_timestamp or "unknown",
                            self.progress_queue,
                            self.archive_queue,
                        )
                    )
                else:
                    # Sequential path (single worker, no pipeline yet on first step)
                    step_advantage_samples, step_strategy_samples, step_value_samples, traversals_done, total_nodes, _trav_timing = (
                        _run_traversals_batch(
                            self.total_traversals,
                            self.total_traversals,
                            self.config,
                            network_weights,
                            network_config,
                            traversals_per_step,
                            1,
                            self.run_log_dir or "logs",
                            self.run_timestamp or "unknown",
                            self.progress_queue,
                            self.archive_queue,
                        )
                    )
                    if self.dcfr_config.enable_traversal_profiling:
                        logger.info(
                            "Step %d traversal timing: min=%.3fs max=%.3fs mean=%.3fs total=%.3fs count=%d",
                            step,
                            _trav_timing["min_s"],
                            _trav_timing["max_s"],
                            _trav_timing["mean_s"],
                            _trav_timing["total_s"],
                            int(_trav_timing["count"]),
                        )

                phase_times["traversal"] = time.time() - _trav_start
                self.total_traversals += traversals_done

                # Add samples to reservoir buffers
                _buf_start = time.time()
                for sample in step_advantage_samples:
                    self.advantage_buffer.add(sample)
                if self.strategy_buffer is not None:
                    for sample in step_strategy_samples:
                        self.strategy_buffer.add(sample)
                if self._is_escher and self.value_buffer is not None:
                    for sample in step_value_samples:
                        self.value_buffer.add(sample)
                phase_times["buffer_insert"] = time.time() - _buf_start

                logger.info(
                    "Step %d: %d traversals, %d advantage samples, %d strategy samples, %d nodes",
                    step, traversals_done, len(step_advantage_samples),
                    len(step_strategy_samples), total_nodes,
                )

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before network training")

                # Submit next traversals BEFORE training so both adv_train and
                # strat_train overlap with the worker's traversal.  The worker
                # uses pre-adv-training weights (one step staler for the
                # advantage net), which is an acceptable approximation — Deep
                # CFR already tolerates stale strategy weights with the prior
                # placement.  Saves ~4s/step by overlapping adv_train with
                # the worker traversal.
                if executor and step < end_step:
                    next_weights = self._get_network_weights_for_workers()
                    pending_future = executor.submit(
                        _run_traversals_batch,
                        self.total_traversals,  # iteration_offset
                        self.total_traversals,  # total_traversals_offset
                        self.config,
                        next_weights,
                        network_config,
                        traversals_per_step,
                        1,  # num_workers (within subprocess)
                        self.run_log_dir or "logs",
                        self.run_timestamp or "unknown",
                    )

                # Train advantage network
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Training advantage net...")

                _adv_start = time.time()
                adv_loss = self._train_network(
                    self.advantage_net, self.advantage_optimizer,
                    self.advantage_buffer, self.dcfr_config.alpha,
                    self.dcfr_config.train_steps_per_iteration,
                    "AdvantageNetwork",
                )
                phase_times["adv_train"] = time.time() - _adv_start
                self.advantage_loss_history.append((step, adv_loss))

                # SD-CFR: snapshot advantage network after training
                if self.dcfr_config.use_sd_cfr:
                    self._take_advantage_snapshot()
                    self._update_ema()

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before strategy training")

                if not self.dcfr_config.use_sd_cfr:
                    # Train strategy network (overlaps with async traversals if pipelining)
                    if display and hasattr(display, "update_main_process_status"):
                        display.update_main_process_status(f"Step {step}: Training strategy net...")

                    _strat_start = time.time()
                    strat_loss = self._train_network(
                        self.strategy_net, self.strategy_optimizer,
                        self.strategy_buffer, self.dcfr_config.alpha,
                        self.dcfr_config.train_steps_per_iteration,
                        "StrategyNetwork",
                    )
                    phase_times["strat_train"] = time.time() - _strat_start
                    self.strategy_loss_history.append((step, strat_loss))
                else:
                    strat_loss = 0.0

                step_time = time.time() - step_start_time

                # Update display
                if display and hasattr(display, "update_stats"):
                    buf_str = f"Adv:{len(self.advantage_buffer)}"
                    if self.strategy_buffer is not None:
                        buf_str += f" Str:{len(self.strategy_buffer)}"
                    if self.dcfr_config.use_sd_cfr:
                        buf_str += f" Snap:{len(self._sd_snapshots)}"
                    display.update_stats(
                        iteration=step,
                        infosets=buf_str,
                        exploitability=f"AdvL:{adv_loss:.4f}" + (f" StrL:{strat_loss:.4f}" if not self.dcfr_config.use_sd_cfr else f" Snaps:{len(self._sd_snapshots)}"),
                        last_iter_time=step_time,
                    )
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status("Idle / Waiting...")

                if self.dcfr_config.use_sd_cfr:
                    logger.info(
                        "Step %d complete in %.2fs. Adv loss: %.6f. "
                        "Buffer: Adv=%d. Snapshots: %d. Total traversals: %d",
                        step, step_time, adv_loss,
                        len(self.advantage_buffer), len(self._sd_snapshots),
                        self.total_traversals,
                    )
                else:
                    logger.info(
                        "Step %d complete in %.2fs. Adv loss: %.6f, Strat loss: %.6f. "
                        "Buffers: Adv=%d, Strat=%d. Total traversals: %d",
                        step, step_time, adv_loss, strat_loss,
                        len(self.advantage_buffer), len(self.strategy_buffer),
                        self.total_traversals,
                    )
                if display is None:
                    phase_str = " ".join(f"{k}={v:.2f}s" for k, v in phase_times.items())
                    if self.dcfr_config.use_sd_cfr:
                        print(
                            f"[step {step}/{end_step}] time={step_time:.1f}s "
                            f"adv_loss={adv_loss:.6f} "
                            f"adv_buf={len(self.advantage_buffer)} "
                            f"snapshots={len(self._sd_snapshots)} "
                            f"traversals={self.total_traversals} "
                            f"| {phase_str}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[step {step}/{end_step}] time={step_time:.1f}s "
                            f"adv_loss={adv_loss:.6f} strat_loss={strat_loss:.6f} "
                            f"adv_buf={len(self.advantage_buffer)} "
                            f"str_buf={len(self.strategy_buffer)} "
                            f"traversals={self.total_traversals} "
                            f"| {phase_str}",
                            flush=True,
                        )

                # Tier 3: exit profiler and export Chrome trace
                if _profiling_this_step and _prof is not None:
                    try:
                        _prof.__exit__(None, None, None)
                        _ckpt_base = getattr(
                            self.config.persistence, "agent_data_save_path",
                            "strategy/deep_cfr_checkpoint.pt",
                        )
                        _trace_dir = os.path.dirname(_ckpt_base) if os.path.dirname(_ckpt_base) else "."
                        _trace_path = os.path.join(_trace_dir, f"profile_step_{step}.json")
                        _prof.export_chrome_trace(_trace_path)
                        logger.info("Torch profiler trace exported to %s", _trace_path)
                    except Exception as _prof_e:
                        logger.warning("Torch profiler export failed: %s", _prof_e)

                # Tier 2: write JSONL profiling record
                if _jsonl_file is not None:
                    try:
                        _samples_this_step = len(step_advantage_samples) + len(step_strategy_samples)
                        _record = {
                            "step": step,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "traversal_ms": round(phase_times.get("traversal", 0.0) * 1000, 1),
                            "adv_train_ms": round(phase_times.get("adv_train", 0.0) * 1000, 1),
                            "strat_train_ms": round(phase_times.get("strat_train", 0.0) * 1000, 1),
                            "buffer_insert_ms": round(phase_times.get("buffer_insert", 0.0) * 1000, 1),
                            "weights_copy_ms": round(phase_times.get("weights_copy", 0.0) * 1000, 1),
                            "buffer_size": len(self.advantage_buffer),
                            "samples_this_step": _samples_this_step,
                        }
                        _jsonl_file.write(json.dumps(_record) + "\n")
                        _jsonl_file.flush()
                    except Exception as _jl_e:
                        logger.warning("Profiling JSONL write failed: %s", _jl_e)

                # Periodic save
                if save_interval > 0 and step % save_interval == 0:
                    # Tier 1: log handle pool stats at INFO level
                    try:
                        from ..ffi.bridge import get_handle_pool_stats
                        _pool_stats = get_handle_pool_stats()
                        logger.info(
                            "Handle pool: games=%s, agents=%s, snapshots=%s",
                            _pool_stats.get("games", "?"),
                            _pool_stats.get("agents", "?"),
                            _pool_stats.get("snapshots", "?"),
                        )
                    except Exception:
                        pass
                    self.save_checkpoint()
                    # PSRO: add checkpoint to population after save
                    if self._psro_oracle is not None and step % self.dcfr_config.psro_checkpoint_interval == 0:
                        checkpoint_path = getattr(
                            self.config.persistence, "agent_data_save_path",
                            "strategy/deep_cfr_checkpoint.pt",
                        )
                        self._psro_oracle.add_checkpoint(checkpoint_path, step)
                        logger.info("PSRO: added checkpoint at step %d (population size=%d)", step, self._psro_oracle.size)

                        # Run population evaluation if configured
                        if self.dcfr_config.psro_eval_games > 0:
                            try:
                                n = max(1, self._psro_oracle.size)
                                ratings = self._psro_oracle.evaluate_population(
                                    games_per_matchup=max(1, self.dcfr_config.psro_eval_games // n),
                                    num_players=self.dcfr_config.num_players,
                                    config=self.config,
                                )
                                if ratings:
                                    logger.info("PSRO population ratings at step %d: %s", step, ratings)
                                    if self.live_display_manager is None:
                                        print(f"[psro step={step}] ratings={ratings}", flush=True)
                            except Exception as e_psro:
                                logger.warning("PSRO evaluation failed at step %d: %s", step, e_psro)

                        # Persist PSRO state alongside the main checkpoint
                        try:
                            checkpoint_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else "."
                            psro_path = os.path.join(checkpoint_dir, "psro_state.json")
                            self._psro_oracle.save_state(psro_path)
                        except Exception as e_psro_save:
                            logger.warning("Failed to save PSRO state: %s", e_psro_save)

                # ES Validation
                es_interval = self.dcfr_config.es_validation_interval
                if es_interval > 0 and step % es_interval == 0:
                    if display and hasattr(display, "update_main_process_status"):
                        display.update_main_process_status(
                            f"Step {step}: Running ES validation..."
                        )
                    try:
                        validator = ESValidator(
                            config=self.config,
                            network_weights=network_weights,
                            network_config=network_config,
                        )
                        es_metrics = validator.compute_exploitability(
                            num_traversals=self.dcfr_config.es_validation_traversals
                        )
                        logger.info(
                            "ES Validation at step %d: mean_regret=%.6f, max_regret=%.6f, "
                            "entropy=%.4f, %d traversals in %.2fs (%d nodes)",
                            step,
                            es_metrics["mean_regret"],
                            es_metrics["max_regret"],
                            es_metrics["strategy_entropy"],
                            es_metrics["traversals"],
                            es_metrics["elapsed_seconds"],
                            es_metrics["total_nodes"],
                        )
                        if self.live_display_manager is None:
                            print(
                                f"[es_validation step={step}] "
                                f"mean_regret={es_metrics['mean_regret']:.6f} "
                                f"max_regret={es_metrics['max_regret']:.6f} "
                                f"entropy={es_metrics['strategy_entropy']:.4f} "
                                f"traversals={es_metrics['traversals']} "
                                f"elapsed={es_metrics['elapsed_seconds']:.2f}s",
                                flush=True,
                            )
                        self.es_validation_history.append((step, es_metrics))
                    except Exception as e_val:
                        logger.warning(
                            "ES validation failed at step %d: %s", step, e_val
                        )

            logger.info("Deep CFR training completed %d steps.", total_steps)

        except (GracefulShutdownException, KeyboardInterrupt) as e:
            logger.warning("Shutdown during training: %s. Saving checkpoint...", type(e).__name__)
            if pending_future:
                pending_future.cancel()
            if executor:
                executor.shutdown(wait=False)
            self._shutdown_pool(pool)
            pool = None
            if _jsonl_file is not None:
                try:
                    _jsonl_file.close()
                except Exception:
                    pass
            self.save_checkpoint()
            raise GracefulShutdownException("Shutdown processed in Deep CFR trainer") from e

        except Exception as e:
            logger.exception("Unhandled error in Deep CFR training loop.")
            if pending_future:
                pending_future.cancel()
            if executor:
                executor.shutdown(wait=False)
            self._shutdown_pool(pool)
            pool = None
            if _jsonl_file is not None:
                try:
                    _jsonl_file.close()
                except Exception:
                    pass
            self.save_checkpoint()
            raise

        # Close JSONL profiling file if open
        if _jsonl_file is not None:
            try:
                _jsonl_file.close()
            except Exception:
                pass

        # Final save
        if not self.shutdown_event.is_set():
            if executor:
                executor.shutdown(wait=True)
            self._shutdown_pool(pool)
            self.save_checkpoint()

    def save_checkpoint(self, filepath: Optional[str] = None):
        """
        Save training state: network weights, optimizer state, buffers, iteration count.

        Saves the main checkpoint as a .pt file and reservoir buffers as .npz files
        alongside it.

        Raises:
            CheckpointSaveError: If saving the checkpoint or buffers fails.
        """
        path = filepath or getattr(
            self.config.persistence, "agent_data_save_path",
            "strategy/deep_cfr_checkpoint.pt",
        )

        try:
            # Ensure directory exists
            checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Derive buffer file paths from the main checkpoint path
            base_path = os.path.splitext(path)[0]
            adv_buffer_path = f"{base_path}_advantage_buffer"
            strat_buffer_path = f"{base_path}_strategy_buffer"
            val_buffer_path = f"{base_path}_value_buffer" if self._is_escher else None
            sd_snapshots_path = f"{base_path}_sd_snapshots.pt" if self.dcfr_config.use_sd_cfr else None
            ema_path = f"{base_path}_ema.pt" if (self.dcfr_config.use_sd_cfr and self.dcfr_config.use_ema) else None

            checkpoint = {
                "advantage_net_state_dict": self.advantage_net.state_dict(),
                "strategy_net_state_dict": self.strategy_net.state_dict() if self.strategy_net is not None else None,
                "value_net_state_dict": self.value_net.state_dict() if self._is_escher else None,
                "advantage_optimizer_state_dict": self.advantage_optimizer.state_dict(),
                "strategy_optimizer_state_dict": self.strategy_optimizer.state_dict() if self.strategy_optimizer is not None else None,
                "value_optimizer_state_dict": (
                    self.value_optimizer.state_dict() if self._is_escher else None
                ),
                "training_step": self.training_step,
                "total_traversals": self.total_traversals,
                "current_iteration": self.current_iteration,
                "dcfr_config": asdict(self.dcfr_config),
                "advantage_loss_history": self.advantage_loss_history,
                "strategy_loss_history": self.strategy_loss_history,
                "es_validation_history": self.es_validation_history,
                "advantage_buffer_path": adv_buffer_path,
                "strategy_buffer_path": strat_buffer_path,
                "value_buffer_path": val_buffer_path,
                "sd_snapshots_path": sd_snapshots_path,
                "grad_scaler_state_dict": self.scaler.state_dict(),
                "ema_state_dict": {k: torch.from_numpy(v) for k, v in self._ema_state_dict.items()} if self._ema_state_dict is not None else None,
                "ema_weight_sum": self._ema_weight_sum if self._ema_state_dict is not None else None,
                "snapshot_count": self._snapshot_count,
            }

            atomic_torch_save(checkpoint, path)
            atomic_npz_save(self.advantage_buffer.save, adv_buffer_path)
            if self.strategy_buffer is not None:
                atomic_npz_save(self.strategy_buffer.save, strat_buffer_path)
            if self._is_escher and self.value_buffer is not None:
                atomic_npz_save(self.value_buffer.save, val_buffer_path)
            if self.dcfr_config.use_sd_cfr and self._sd_snapshots:
                # Flatten snapshots into a tensor-only dict for weights_only=True compat
                snapshot_data = {
                    "num_snapshots": torch.tensor(len(self._sd_snapshots)),
                    "iterations": torch.tensor(self._sd_snapshot_iterations, dtype=torch.long),
                }
                for i, snap in enumerate(self._sd_snapshots):
                    for k, v in snap.items():
                        snapshot_data[f"snap_{i}_{k}"] = torch.from_numpy(v)
                atomic_torch_save(snapshot_data, sd_snapshots_path)
            if ema_path is not None and self._ema_state_dict is not None:
                ema_data = {
                    "ema_state_dict": {k: torch.from_numpy(v) for k, v in self._ema_state_dict.items()},
                    "ema_weight_sum": torch.tensor(self._ema_weight_sum),
                }
                atomic_torch_save(ema_data, ema_path)
                logger.info("EMA state saved to %s (weight_sum=%.2f).", ema_path, self._ema_weight_sum)
            elif ema_path is None:
                logger.warning("EMA save skipped: ema_path is None (use_sd_cfr or use_ema not enabled).")
            else:
                logger.warning("EMA save skipped: _ema_state_dict is None (no EMA updates recorded yet).")

            # Save iteration-specific .pt copy (no buffers) for post-hoc evaluation
            iter_path = f"{base_path}_iter_{self.training_step}.pt"
            try:
                atomic_torch_save(checkpoint, iter_path)
            except Exception as e_iter:
                logger.warning("Failed to save iteration checkpoint %s: %s", iter_path, e_iter)

            # PSRO state (saved alongside the checkpoint directory)
            if self._psro_oracle is not None:
                psro_path = os.path.join(checkpoint_dir, "psro_state.json")
                try:
                    self._psro_oracle.save_state(psro_path)
                    logger.info("PSRO state saved to %s.", psro_path)
                except Exception as e_psro:
                    logger.warning("Failed to save PSRO state to %s: %s", psro_path, e_psro)

            logger.info(
                "Checkpoint saved to %s (step %d, %d traversals).",
                path, self.training_step, self.total_traversals,
            )
            if self.live_display_manager is None:
                print(
                    f"[checkpoint] saved to {path} "
                    f"(step {self.training_step}, {self.total_traversals} traversals)",
                    flush=True,
                )
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Failed to save checkpoint to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Unexpected error saving checkpoint to {path}: {e}") from e

    def load_checkpoint(self, filepath: Optional[str] = None):
        """
        Load training state from checkpoint.

        Raises:
            CheckpointLoadError: If loading the checkpoint or buffers fails.
        """
        path = filepath or getattr(
            self.config.persistence, "agent_data_save_path",
            "strategy/deep_cfr_checkpoint.pt",
        )

        if not os.path.exists(path):
            logger.info("No checkpoint found at %s. Starting fresh.", path)
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)

            self.advantage_net.load_state_dict(checkpoint["advantage_net_state_dict"])
            if self.strategy_net is not None and checkpoint.get("strategy_net_state_dict") is not None:
                self.strategy_net.load_state_dict(checkpoint["strategy_net_state_dict"])
            self.advantage_optimizer.load_state_dict(checkpoint["advantage_optimizer_state_dict"])
            if self.strategy_optimizer is not None and checkpoint.get("strategy_optimizer_state_dict") is not None:
                self.strategy_optimizer.load_state_dict(checkpoint["strategy_optimizer_state_dict"])

            # ESCHER value network — cross-mode safe
            value_net_sd = checkpoint.get("value_net_state_dict")
            if self._is_escher:
                if value_net_sd is not None:
                    # ESCHER checkpoint -> ESCHER trainer: load weights
                    self.value_net.load_state_dict(value_net_sd)
                    val_opt_sd = checkpoint.get("value_optimizer_state_dict")
                    if val_opt_sd is not None and self.value_optimizer is not None:
                        self.value_optimizer.load_state_dict(val_opt_sd)
                    logger.info("Loaded value_net from ESCHER checkpoint.")
                else:
                    # OS checkpoint -> ESCHER trainer: keep fresh init
                    logger.info(
                        "OS checkpoint loaded into ESCHER trainer: value_net initializes fresh."
                    )
            # else: ESCHER checkpoint -> OS trainer — value fields are ignored (value_net is None)

            # Load reservoir buffers from their saved .npz files
            adv_buffer_path = checkpoint.get("advantage_buffer_path")
            strat_buffer_path = checkpoint.get("strategy_buffer_path")

            # Fallback: if buffer paths not in checkpoint, derive them from checkpoint path
            if not adv_buffer_path and not strat_buffer_path:
                base_path = os.path.splitext(path)[0]
                adv_buffer_path = f"{base_path}_advantage_buffer"
                strat_buffer_path = f"{base_path}_strategy_buffer"

            adv_loaded = False
            strat_loaded = False

            if adv_buffer_path:
                npz_path = adv_buffer_path if adv_buffer_path.endswith(".npz") else adv_buffer_path + ".npz"
                if os.path.exists(npz_path):
                    self.advantage_buffer = ReservoirBuffer(
                        capacity=self.dcfr_config.advantage_buffer_capacity,
                        input_dim=self.dcfr_config.input_dim,
                    )
                    self.advantage_buffer.load(adv_buffer_path)
                    adv_loaded = True
                else:
                    logger.warning(
                        "Advantage buffer file not found: %s. Starting with empty buffer.", npz_path
                    )

            if self.strategy_buffer is not None and strat_buffer_path:
                npz_path = strat_buffer_path if strat_buffer_path.endswith(".npz") else strat_buffer_path + ".npz"
                if os.path.exists(npz_path):
                    self.strategy_buffer = ReservoirBuffer(
                        capacity=self.dcfr_config.strategy_buffer_capacity,
                        input_dim=self.dcfr_config.input_dim,
                    )
                    self.strategy_buffer.load(strat_buffer_path)
                    strat_loaded = True
                else:
                    logger.warning(
                        "Strategy buffer file not found: %s. Starting with empty buffer.", npz_path
                    )

            # ESCHER value buffer loading
            if self._is_escher:
                val_buffer_path = checkpoint.get("value_buffer_path")
                if val_buffer_path:
                    npz_path = val_buffer_path if val_buffer_path.endswith(".npz") else val_buffer_path + ".npz"
                    if os.path.exists(npz_path):
                        self.value_buffer = ReservoirBuffer(
                            capacity=self.dcfr_config.value_buffer_capacity,
                            input_dim=INPUT_DIM * 2,
                            target_dim=1,
                            has_mask=False,
                        )
                        self.value_buffer.load(val_buffer_path)
                        logger.info("Loaded value buffer from %s.", npz_path)
                    else:
                        logger.warning(
                            "Value buffer file not found: %s. Starting with empty buffer.", npz_path
                        )

            if not adv_loaded and not strat_loaded:
                logger.warning(
                    "No buffer files found alongside checkpoint; starting with cold buffer."
                )

            # SD-CFR snapshot loading
            sd_snapshots_path = checkpoint.get("sd_snapshots_path")
            if sd_snapshots_path and os.path.exists(sd_snapshots_path):
                snapshot_data = torch.load(sd_snapshots_path, map_location="cpu", weights_only=True)
                num_snapshots = int(snapshot_data["num_snapshots"].item())
                self._sd_snapshot_iterations = snapshot_data["iterations"].tolist()
                self._sd_snapshots = []
                for i in range(num_snapshots):
                    prefix = f"snap_{i}_"
                    snap = {k[len(prefix):]: v.numpy() for k, v in snapshot_data.items() if k.startswith(prefix)}
                    self._sd_snapshots.append(snap)
                logger.info("Loaded %d SD-CFR snapshots from %s.", len(self._sd_snapshots), sd_snapshots_path)

            # Restore snapshot count for correct reservoir sampling after warm-start
            if "snapshot_count" in checkpoint:
                self._snapshot_count = int(checkpoint["snapshot_count"])
            else:
                # Backward compat: fall back to training_step (old behavior)
                self._snapshot_count = self.training_step

            # EMA state loading: checkpoint dict → separate file → initialize fresh
            if self.dcfr_config.use_sd_cfr and self.dcfr_config.use_ema:
                ema_dict_in_checkpoint = checkpoint.get("ema_state_dict")
                if ema_dict_in_checkpoint is not None:
                    self._ema_state_dict = {
                        k: v.numpy() for k, v in ema_dict_in_checkpoint.items()
                    }
                    self._ema_weight_sum = float(checkpoint["ema_weight_sum"])
                    logger.info(
                        "Loaded EMA state from checkpoint dict (weight_sum=%.2f).", self._ema_weight_sum
                    )
                else:
                    # Backward compat: try separate _ema.pt file
                    base_path = os.path.splitext(path)[0]
                    ema_path = f"{base_path}_ema.pt"
                    if os.path.exists(ema_path):
                        ema_data = torch.load(ema_path, map_location="cpu", weights_only=True)
                        self._ema_state_dict = {
                            k: v.numpy() for k, v in ema_data["ema_state_dict"].items()
                        }
                        self._ema_weight_sum = float(ema_data["ema_weight_sum"].item())
                        logger.info(
                            "Loaded EMA state from separate file %s (weight_sum=%.2f).",
                            ema_path, self._ema_weight_sum,
                        )
                    else:
                        logger.info(
                            "No EMA state found in checkpoint dict or separate file; EMA will initialize on next snapshot."
                        )

            # PSRO state loading
            if self._psro_oracle is not None:
                checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
                psro_path = os.path.join(checkpoint_dir, "psro_state.json")
                if os.path.exists(psro_path):
                    try:
                        self._psro_oracle.load_state(psro_path)
                        logger.info("PSRO state loaded from %s.", psro_path)
                    except Exception as e_psro:
                        logger.warning("Failed to load PSRO state from %s: %s", psro_path, e_psro)
                else:
                    logger.info("No PSRO state file found at %s; starting fresh.", psro_path)

            self.training_step = checkpoint.get("training_step", 0)
            self.total_traversals = checkpoint.get("total_traversals", 0)
            self.current_iteration = checkpoint.get("current_iteration", 0)
            self.advantage_loss_history = checkpoint.get("advantage_loss_history", [])
            self.strategy_loss_history = checkpoint.get("strategy_loss_history", [])
            self.es_validation_history = checkpoint.get("es_validation_history", [])

            # Restore GradScaler state if present
            if "grad_scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

            # Warn on config mismatch between checkpoint and current config
            saved_config = checkpoint.get("dcfr_config", {})
            if saved_config:
                current_config = asdict(self.dcfr_config)
                for key in set(saved_config) | set(current_config):
                    saved_val = saved_config.get(key)
                    current_val = current_config.get(key)
                    if saved_val != current_val:
                        logger.warning(
                            "Config mismatch for '%s': checkpoint=%r, current=%r",
                            key, saved_val, current_val,
                        )

            # Warn on cambia_rules mismatch between checkpoint and current config
            saved_meta = checkpoint.get("metadata", {})
            saved_rules = (saved_meta.get("config", {}) or {}).get("cambia_rules", {})
            if saved_rules:
                current_rules = asdict(self.config.cambia_rules)
                for key in set(saved_rules) | set(current_rules):
                    if saved_rules.get(key) != current_rules.get(key):
                        logger.warning(
                            "cambia_rules mismatch '%s': checkpoint=%r, current=%r",
                            key,
                            saved_rules.get(key),
                            current_rules.get(key),
                        )

            strat_len = len(self.strategy_buffer) if self.strategy_buffer is not None else 0
            logger.info(
                "Checkpoint loaded from %s. Resuming at step %d (%d traversals). "
                "Buffers: Adv=%d, Strat=%d.",
                path, self.training_step, self.total_traversals,
                len(self.advantage_buffer), strat_len,
            )
        except FileNotFoundError as e:
            logger.error("Checkpoint file not found: %s", path)
            raise CheckpointLoadError(f"Checkpoint file not found: {path}") from e
        except (KeyError, ValueError) as e:
            logger.error("Corrupted or incompatible checkpoint file %s: %s", path, e)
            raise CheckpointLoadError(f"Corrupted or incompatible checkpoint file {path}: {e}") from e
        except (OSError, IOError) as e:
            logger.error("Failed to load checkpoint from %s: %s", path, e)
            raise CheckpointLoadError(f"Failed to load checkpoint from {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error loading checkpoint from %s: %s", path, e)
            raise CheckpointLoadError(f"Unexpected error loading checkpoint from {path}: {e}") from e

    def get_strategy_network(self) -> StrategyNetwork:
        """Returns the trained strategy network for deployment/evaluation."""
        return self.strategy_net

    def get_advantage_network(self) -> AdvantageNetwork:
        """Returns the trained advantage network."""
        return self.advantage_net
