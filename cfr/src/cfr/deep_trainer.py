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
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
import time
import queue
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import Config
from ..constants import NUM_PLAYERS
from ..encoding import INPUT_DIM, NUM_ACTIONS
from ..networks import AdvantageNetwork, StrategyNetwork
from ..persistence import atomic_torch_save, atomic_npz_save
from ..reservoir import ReservoirBuffer, ReservoirSample
from ..utils import LogQueue as ProgressQueue
from ..live_display import LiveDisplayManager
from ..log_archiver import LogArchiver

from .deep_worker import run_deep_cfr_worker, DeepCFRWorkerResult
from .es_validator import ESValidator
from .exceptions import GracefulShutdownException, CheckpointSaveError, CheckpointLoadError

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

    # Use GPU for training if available
    use_gpu: bool = False

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

    # Max traversal depth (0 = unlimited, backward compatible)
    traversal_depth_limit: int = 0

    def __post_init__(self):
        if self.pipeline_training and self.num_traversal_threads > 1:
            raise ValueError(
                "pipeline_training=True and num_traversal_threads>1 are mutually exclusive. "
                "Pipeline training already parallelises traversal and training in separate "
                "processes; combining it with multi-threaded traversal is not supported."
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
            "use_gpu": deep_cfg.use_gpu,
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
        }
        # Apply CLI overrides (only non-None values)
        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return cls(**kwargs)


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
) -> Tuple[List[ReservoirSample], List[ReservoirSample], int, int]:
    """Run all traversals for one training step. Returns (adv_samples, strat_samples, traversals_done, total_nodes)."""
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    traversals_done = 0
    total_nodes = 0

    for i in range(traversals_per_step):
        iter_num = iteration_offset + i
        args_tuple = (
            iter_num,
            config,
            network_weights,
            network_config,
            progress_queue,
            archive_queue,
            i,  # worker_id
            run_log_dir,
            run_timestamp,
        )
        result = run_deep_cfr_worker(args_tuple)
        if isinstance(result, DeepCFRWorkerResult):
            advantage_samples.extend(result.advantage_samples)
            strategy_samples.extend(result.strategy_samples)
            total_nodes += result.stats.nodes_visited
            if result.stats.error_count > 0:
                logger.warning("Worker reported %d errors.", result.stats.error_count)
        traversals_done += 1

    return advantage_samples, strategy_samples, traversals_done, total_nodes


def _run_single_traversal(args_tuple):
    """Wrapper for a single traversal call, used by ThreadPoolExecutor."""
    result = run_deep_cfr_worker(args_tuple)
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
) -> Tuple[List[ReservoirSample], List[ReservoirSample], int, int]:
    """Run traversals using ThreadPoolExecutor for Go FFI backend.

    Threads share process memory so the advantage network can be used read-only
    without serialization. Each thread creates its own GoEngine instance; the
    Go handle pool is mutex-protected (task #6).
    """
    advantage_samples: List[ReservoirSample] = []
    strategy_samples: List[ReservoirSample] = []
    traversals_done = 0
    total_nodes = 0

    worker_args_list = []
    for i in range(traversals_per_step):
        iter_num = iteration_offset + i
        worker_args_list.append((
            iter_num,
            config,
            network_weights,
            network_config,
            progress_queue,
            archive_queue,
            i,  # worker_id
            run_log_dir,
            run_timestamp,
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as thread_pool:
        futures = [thread_pool.submit(_run_single_traversal, args) for args in worker_args_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, DeepCFRWorkerResult):
                advantage_samples.extend(result.advantage_samples)
                strategy_samples.extend(result.strategy_samples)
                total_nodes += result.stats.nodes_visited
                if result.stats.error_count > 0:
                    logger.warning("Worker reported %d errors.", result.stats.error_count)
            traversals_done += 1

    return advantage_samples, strategy_samples, traversals_done, total_nodes


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
        if self.dcfr_config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info("Deep CFR Trainer using device: %s", self.device)

        # Networks
        self.advantage_net = AdvantageNetwork(
            input_dim=self.dcfr_config.input_dim,
            hidden_dim=self.dcfr_config.hidden_dim,
            output_dim=self.dcfr_config.output_dim,
            dropout=self.dcfr_config.dropout,
            validate_inputs=self.dcfr_config.validate_inputs,
        ).to(self.device)

        self.strategy_net = StrategyNetwork(
            input_dim=self.dcfr_config.input_dim,
            hidden_dim=self.dcfr_config.hidden_dim,
            output_dim=self.dcfr_config.output_dim,
            dropout=self.dcfr_config.dropout,
            validate_inputs=self.dcfr_config.validate_inputs,
        ).to(self.device)

        # B5: torch.compile — gated by config (only effective on CUDA)
        if self.dcfr_config.use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.advantage_net = torch.compile(self.advantage_net, mode="reduce-overhead")
                self.strategy_net = torch.compile(self.strategy_net, mode="reduce-overhead")
                logger.info("Networks compiled with torch.compile (reduce-overhead mode)")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

        # Optimizers
        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=self.dcfr_config.learning_rate
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=self.dcfr_config.learning_rate
        )

        # B4: AMP scaler — gated by config (only effective on CUDA)
        self.use_amp = self.dcfr_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Reservoir buffers
        self.advantage_buffer = ReservoirBuffer(
            capacity=self.dcfr_config.advantage_buffer_capacity
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=self.dcfr_config.strategy_buffer_capacity
        )

        # Training state
        self.current_iteration = 0
        self.total_traversals = 0
        self.training_step = 0

        # Tracking
        self.advantage_loss_history: List[Tuple[int, float]] = []
        self.strategy_loss_history: List[Tuple[int, float]] = []
        self.es_validation_history: List[Tuple[int, Dict]] = []

        logger.info(
            "DeepCFRTrainer initialized. Advantage net params: %d, Strategy net params: %d",
            sum(p.numel() for p in self.advantage_net.parameters()),
            sum(p.numel() for p in self.strategy_net.parameters()),
        )

    def _get_network_weights_for_workers(self) -> Dict[str, Any]:
        """
        Serialize advantage network weights for distribution to workers.
        Returns numpy arrays for pickle-friendly multiprocessing transfer.
        """
        state_dict = self.advantage_net.state_dict()
        return {k: v.cpu().numpy() for k, v in state_dict.items()}

    def _get_network_config(self) -> Dict[str, int]:
        """Returns network configuration dict for workers."""
        return {
            "input_dim": self.dcfr_config.input_dim,
            "hidden_dim": self.dcfr_config.hidden_dim,
            "output_dim": self.dcfr_config.output_dim,
            "validate_inputs": self.dcfr_config.validate_inputs,
        }

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

            with torch.amp.autocast("cuda", enabled=self.use_amp):
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
            # Use 'spawn' context to avoid CUDA fork deadlock
            ctx = multiprocessing.get_context("spawn")
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=ctx
            )

        network_config = self._get_network_config()

        try:
            for step in range(start_step, end_step + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before training step")

                step_start_time = time.time()
                self.training_step = step

                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Running traversals...")

                # Get current network weights for workers
                network_weights = self._get_network_weights_for_workers()

                # Collect traversal results — either from pipeline future or synchronously
                if pending_future is not None:
                    # B3: Collect results from pipelined traversal started after prev step's adv training
                    step_advantage_samples, step_strategy_samples, traversals_done, total_nodes = (
                        pending_future.result()
                    )
                    pending_future = None
                elif num_workers > 1:
                    # Parallel pool-based traversals (non-pipelined path)
                    step_advantage_samples: List[ReservoirSample] = []
                    step_strategy_samples: List[ReservoirSample] = []
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
                                i,  # worker_id
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
                    step_advantage_samples, step_strategy_samples, traversals_done, total_nodes = (
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
                    step_advantage_samples, step_strategy_samples, traversals_done, total_nodes = (
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

                self.total_traversals += traversals_done

                # Add samples to reservoir buffers
                for sample in step_advantage_samples:
                    self.advantage_buffer.add(sample)
                for sample in step_strategy_samples:
                    self.strategy_buffer.add(sample)

                logger.info(
                    "Step %d: %d traversals, %d advantage samples, %d strategy samples, %d nodes",
                    step, traversals_done, len(step_advantage_samples),
                    len(step_strategy_samples), total_nodes,
                )

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before network training")

                # Train advantage network
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Training advantage net...")

                adv_loss = self._train_network(
                    self.advantage_net, self.advantage_optimizer,
                    self.advantage_buffer, self.dcfr_config.alpha,
                    self.dcfr_config.train_steps_per_iteration,
                    "AdvantageNetwork",
                )
                self.advantage_loss_history.append((step, adv_loss))

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before strategy training")

                # B3: Submit next traversals async after adv training (if pipelining and not last step)
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

                # Train strategy network (overlaps with async traversals if pipelining)
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status(f"Step {step}: Training strategy net...")

                strat_loss = self._train_network(
                    self.strategy_net, self.strategy_optimizer,
                    self.strategy_buffer, self.dcfr_config.alpha,
                    self.dcfr_config.train_steps_per_iteration,
                    "StrategyNetwork",
                )
                self.strategy_loss_history.append((step, strat_loss))

                step_time = time.time() - step_start_time

                # Update display
                if display and hasattr(display, "update_stats"):
                    display.update_stats(
                        iteration=step,
                        infosets=f"Adv:{len(self.advantage_buffer)} Str:{len(self.strategy_buffer)}",
                        exploitability=f"AdvL:{adv_loss:.4f} StrL:{strat_loss:.4f}",
                        last_iter_time=step_time,
                    )
                if display and hasattr(display, "update_main_process_status"):
                    display.update_main_process_status("Idle / Waiting...")

                logger.info(
                    "Step %d complete in %.2fs. Adv loss: %.6f, Strat loss: %.6f. "
                    "Buffers: Adv=%d, Strat=%d. Total traversals: %d",
                    step, step_time, adv_loss, strat_loss,
                    len(self.advantage_buffer), len(self.strategy_buffer),
                    self.total_traversals,
                )
                if display is None:
                    print(
                        f"[step {step}/{end_step}] time={step_time:.1f}s "
                        f"adv_loss={adv_loss:.6f} strat_loss={strat_loss:.6f} "
                        f"adv_buf={len(self.advantage_buffer)} "
                        f"str_buf={len(self.strategy_buffer)} "
                        f"traversals={self.total_traversals}",
                        flush=True,
                    )

                # Periodic save
                if save_interval > 0 and step % save_interval == 0:
                    self.save_checkpoint()

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
            self.save_checkpoint()
            raise

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

            checkpoint = {
                "advantage_net_state_dict": self.advantage_net.state_dict(),
                "strategy_net_state_dict": self.strategy_net.state_dict(),
                "advantage_optimizer_state_dict": self.advantage_optimizer.state_dict(),
                "strategy_optimizer_state_dict": self.strategy_optimizer.state_dict(),
                "training_step": self.training_step,
                "total_traversals": self.total_traversals,
                "current_iteration": self.current_iteration,
                "dcfr_config": asdict(self.dcfr_config),
                "advantage_loss_history": self.advantage_loss_history,
                "strategy_loss_history": self.strategy_loss_history,
                "es_validation_history": self.es_validation_history,
                "advantage_buffer_path": adv_buffer_path,
                "strategy_buffer_path": strat_buffer_path,
                "grad_scaler_state_dict": self.scaler.state_dict(),
            }

            atomic_torch_save(checkpoint, path)
            atomic_npz_save(self.advantage_buffer.save, adv_buffer_path)
            atomic_npz_save(self.strategy_buffer.save, strat_buffer_path)
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
            self.strategy_net.load_state_dict(checkpoint["strategy_net_state_dict"])
            self.advantage_optimizer.load_state_dict(checkpoint["advantage_optimizer_state_dict"])
            self.strategy_optimizer.load_state_dict(checkpoint["strategy_optimizer_state_dict"])

            # Load reservoir buffers from their saved .npz files
            adv_buffer_path = checkpoint.get("advantage_buffer_path")
            strat_buffer_path = checkpoint.get("strategy_buffer_path")

            if adv_buffer_path:
                npz_path = adv_buffer_path if adv_buffer_path.endswith(".npz") else adv_buffer_path + ".npz"
                if os.path.exists(npz_path):
                    self.advantage_buffer = ReservoirBuffer(
                        capacity=self.dcfr_config.advantage_buffer_capacity
                    )
                    self.advantage_buffer.load(adv_buffer_path)
                else:
                    logger.warning(
                        "Advantage buffer file not found: %s. Starting with empty buffer.", npz_path
                    )

            if strat_buffer_path:
                npz_path = strat_buffer_path if strat_buffer_path.endswith(".npz") else strat_buffer_path + ".npz"
                if os.path.exists(npz_path):
                    self.strategy_buffer = ReservoirBuffer(
                        capacity=self.dcfr_config.strategy_buffer_capacity
                    )
                    self.strategy_buffer.load(strat_buffer_path)
                else:
                    logger.warning(
                        "Strategy buffer file not found: %s. Starting with empty buffer.", npz_path
                    )

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

            logger.info(
                "Checkpoint loaded from %s. Resuming at step %d (%d traversals). "
                "Buffers: Adv=%d, Strat=%d.",
                path, self.training_step, self.total_traversals,
                len(self.advantage_buffer), len(self.strategy_buffer),
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
