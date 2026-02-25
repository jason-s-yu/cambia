"""
src/cfr/rebel_trainer.py

ReBeL Trainer — training loop orchestrator for ReBeL (Recursive Belief-based Learning).

Architecture:
- PBSValueNetwork: predicts counterfactual values given PBS encoding (956 -> 936)
- PBSPolicyNetwork: predicts action probabilities given PBS encoding (956 -> 146)
- ReservoirBuffers: value_buffer (has_mask=False) and policy_buffer (has_mask=True)
- Self-play episodes via ProcessPoolExecutor (spawn context) for worker isolation

Training loop per iteration:
1. Copy network state_dicts for workers (numpy arrays for pickling efficiency)
2. Run rebel_games_per_epoch episodes via ProcessPoolExecutor
3. Insert episode samples into value/policy buffers
4. Train PBSValueNetwork on value_buffer (MSE loss, iteration-weighted)
5. Train PBSPolicyNetwork on policy_buffer (MSE loss, iteration-weighted)
6. Checkpoint save every save_interval iterations
7. Headless output: phase timing, buffer sizes, losses
"""

# DEPRECATED: ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games
# with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.
import warnings

warnings.warn(
    "rebel_trainer is DEPRECATED and will be removed. "
    "ReBeL/PBS-based subgame solving is mathematically unsound for N-player FFA games "
    "with continuous beliefs (Cambia). See docs-gen/current/research-brief-position-aware-pbs.md.",
    DeprecationWarning,
    stacklevel=2,
)

import concurrent.futures
import logging
import multiprocessing
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import DeepCfrConfig
from ..networks import PBSValueNetwork, PBSPolicyNetwork
from ..persistence import atomic_torch_save, atomic_npz_save
from ..reservoir import ReservoirBuffer, ReservoirSample
from .deep_trainer import _resolve_device, _resolve_max_tasks_per_child
from .exceptions import (
    CheckpointSaveError,
    CheckpointLoadError,
    GracefulShutdownException,
)

logger = logging.getLogger(__name__)

# PBS dimensions — imported from canonical sources
from ..pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from ..encoding import NUM_ACTIONS
VALUE_OUTPUT_DIM: int = 2 * NUM_HAND_TYPES
POLICY_OUTPUT_DIM: int = NUM_ACTIONS


# ---------------------------------------------------------------------------
# Top-level worker function (must be module-level for ProcessPoolExecutor pickle)
# ---------------------------------------------------------------------------


def _rebel_batch_worker(args: Tuple) -> List:
    """
    ProcessPoolExecutor worker — runs N self-play episodes and returns all samples.

    Must be at module level (not a closure or method) so ProcessPoolExecutor can
    pickle it for spawn-based worker processes.

    Args:
        args: (num_episodes, value_state_numpy, policy_state_numpy, config, game_config)
            value_state_numpy: {str: np.ndarray} state_dict for PBSValueNetwork
            policy_state_numpy: {str: np.ndarray} state_dict for PBSPolicyNetwork
            config: DeepCfrConfig carrying rebel_* hyper-parameters
            game_config: house rules for GoEngine (None = Go defaults)

    Returns:
        Flat list of EpisodeSample objects from all completed episodes.
    """
    num_episodes, value_state_numpy, policy_state_numpy, config, game_config = args

    from ..networks import PBSValueNetwork, PBSPolicyNetwork
    from .rebel_worker import rebel_self_play_episode

    # Reconstruct value network
    value_net = PBSValueNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=config.rebel_value_hidden_dim,
        output_dim=VALUE_OUTPUT_DIM,
        validate_inputs=False,
    )
    value_weights = {
        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in value_state_numpy.items()
    }
    value_net.load_state_dict(value_weights)
    value_net.eval()

    # Reconstruct policy network
    policy_net = PBSPolicyNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=config.rebel_policy_hidden_dim,
        output_dim=POLICY_OUTPUT_DIM,
        validate_inputs=False,
    )
    policy_weights = {
        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in policy_state_numpy.items()
    }
    policy_net.load_state_dict(policy_weights)
    policy_net.eval()

    all_samples = []
    with torch.inference_mode():
        for _ in range(num_episodes):
            try:
                samples = rebel_self_play_episode(
                    game_config, value_net, policy_net, config
                )
                all_samples.extend(samples)
            except Exception as e:
                logger.warning("ReBeL episode failed: %s", e)

    return all_samples


# ---------------------------------------------------------------------------
# ReBeLTrainer
# ---------------------------------------------------------------------------


class ReBeLTrainer:
    """
    Orchestrates ReBeL training with self-play episode workers.

    Each iteration:
    1. Run rebel_games_per_epoch episodes via ProcessPoolExecutor
    2. Insert episode samples into value and policy ReservoirBuffers
    3. Train PBSValueNetwork on value_buffer
    4. Train PBSPolicyNetwork on policy_buffer
    5. Save checkpoint every save_interval iterations
    """

    def __init__(
        self,
        config: DeepCfrConfig,
        checkpoint_path: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        game_config: Optional[Any] = None,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path or "rebel_checkpoint.pt"
        self.shutdown_event = shutdown_event or threading.Event()
        self.game_config = game_config  # CambiaRulesConfig for GoEngine

        # Device
        resolved_device = _resolve_device(config.device)
        self.device = torch.device(resolved_device)
        logger.info("ReBeLTrainer using device: %s", self.device)

        # Networks
        self.value_net = PBSValueNetwork(
            input_dim=PBS_INPUT_DIM,
            hidden_dim=config.rebel_value_hidden_dim,
            output_dim=VALUE_OUTPUT_DIM,
        ).to(self.device)

        self.policy_net = PBSPolicyNetwork(
            input_dim=PBS_INPUT_DIM,
            hidden_dim=config.rebel_policy_hidden_dim,
            output_dim=POLICY_OUTPUT_DIM,
        ).to(self.device)

        # Optimizers
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=config.rebel_value_learning_rate
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.rebel_policy_learning_rate
        )

        # Buffers
        # Value buffer: no action masks (PBSValueNetwork predicts 936-dim CFs without masking)
        self.value_buffer = ReservoirBuffer(
            capacity=config.rebel_value_buffer_capacity,
            input_dim=PBS_INPUT_DIM,
            target_dim=VALUE_OUTPUT_DIM,
            has_mask=False,
        )

        # Policy buffer: with action masks
        self.policy_buffer = ReservoirBuffer(
            capacity=config.rebel_policy_buffer_capacity,
            input_dim=PBS_INPUT_DIM,
            target_dim=POLICY_OUTPUT_DIM,
            has_mask=True,
        )

        # Training state
        self.current_iteration: int = 0
        self.value_loss_history: List[Tuple[int, float]] = []
        self.policy_loss_history: List[Tuple[int, float]] = []

        # Provenance
        self._loaded_from_checkpoint: Optional[str] = None
        self._loaded_from_metadata: Optional[Dict[str, Any]] = None

        logger.info(
            "ReBeLTrainer initialized. Value net: %d params, Policy net: %d params",
            sum(p.numel() for p in self.value_net.parameters()),
            sum(p.numel() for p in self.policy_net.parameters()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_network_state_dicts_numpy(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Serialize networks to CPU numpy state dicts for pickling to workers."""
        value_state = {
            k: v.cpu().numpy() for k, v in self.value_net.state_dict().items()
        }
        policy_state = {
            k: v.cpu().numpy() for k, v in self.policy_net.state_dict().items()
        }
        return value_state, policy_state

    def _insert_samples(self, samples: List, iteration: int) -> None:
        """Insert EpisodeSamples into value and policy buffers."""
        for sample in samples:
            v_sample = ReservoirSample(
                features=sample.features,
                target=sample.value_target,
                action_mask=np.empty(0, dtype=bool),  # not used (has_mask=False)
                iteration=iteration,
            )
            self.value_buffer.add(v_sample)

            p_sample = ReservoirSample(
                features=sample.features,
                target=sample.policy_target,
                action_mask=sample.action_mask,
                iteration=iteration,
            )
            self.policy_buffer.add(p_sample)

    def _train_value_network(self, num_steps: int) -> float:
        """
        Train PBSValueNetwork on value_buffer samples.

        Loss: (t^alpha) * MSE(V(pbs), value_target) — iteration-weighted, no masking.
        Returns average loss over all training steps.
        """
        if len(self.value_buffer) == 0:
            logger.warning("Cannot train value network: buffer is empty.")
            return 0.0

        self.value_net.train()
        total_loss = 0.0
        actual_steps = 0
        batch_size = self.config.batch_size
        alpha = self.config.alpha

        for _ in range(num_steps):
            if self.shutdown_event.is_set():
                logger.warning("Shutdown detected during value network training.")
                break

            batch = self.value_buffer.sample_batch(batch_size)
            if not batch:
                break

            features_t = torch.from_numpy(batch.features).float().to(self.device)
            targets_t = torch.from_numpy(batch.targets).float().to(self.device)
            iterations_t = torch.from_numpy(
                batch.iterations.astype(np.float32)
            ).to(self.device)

            weights = (iterations_t + 1.0).pow(alpha)
            weights = weights / weights.mean()

            self.value_optimizer.zero_grad()
            predictions = self.value_net(features_t)  # (batch, 936)
            per_sample_mse = ((predictions - targets_t) ** 2).mean(dim=1)
            loss = (weights * per_sample_mse).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.value_optimizer.step()

            total_loss += loss.item()
            actual_steps += 1

        avg_loss = total_loss / max(actual_steps, 1)
        logger.info(
            "Value net training: %d steps, avg loss: %.6f (buffer size: %d)",
            actual_steps, avg_loss, len(self.value_buffer),
        )
        return avg_loss

    def _train_policy_network(self, num_steps: int) -> float:
        """
        Train PBSPolicyNetwork on policy_buffer samples.

        Loss: (t^alpha) * MSE(pi(pbs), policy_target) — iteration-weighted, with masking.
        Returns average loss over all training steps.
        """
        if len(self.policy_buffer) == 0:
            logger.warning("Cannot train policy network: buffer is empty.")
            return 0.0

        self.policy_net.train()
        total_loss = 0.0
        actual_steps = 0
        batch_size = self.config.batch_size
        alpha = self.config.alpha

        for _ in range(num_steps):
            if self.shutdown_event.is_set():
                logger.warning("Shutdown detected during policy network training.")
                break

            batch = self.policy_buffer.sample_batch(batch_size)
            if not batch:
                break

            features_t = torch.from_numpy(batch.features).float().to(self.device)
            targets_t = torch.from_numpy(batch.targets).float().to(self.device)
            masks_t = torch.from_numpy(batch.masks).to(self.device)
            iterations_t = torch.from_numpy(
                batch.iterations.astype(np.float32)
            ).to(self.device)

            weights = (iterations_t + 1.0).pow(alpha)
            weights = weights / weights.mean()

            self.policy_optimizer.zero_grad()
            predictions = self.policy_net(features_t, masks_t)  # (batch, 146) softmax

            # MSE on strategy targets, restricted to legal actions
            masked_preds = predictions.masked_fill(~masks_t, 0.0)
            masked_targets = targets_t.masked_fill(~masks_t, 0.0)
            num_legal = masks_t.float().sum(dim=1).clamp(min=1.0)
            per_sample_mse = (
                ((masked_preds - masked_targets) ** 2).sum(dim=1) / num_legal
            )
            loss = (weights * per_sample_mse).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

            total_loss += loss.item()
            actual_steps += 1

        avg_loss = total_loss / max(actual_steps, 1)
        logger.info(
            "Policy net training: %d steps, avg loss: %.6f (buffer size: %d)",
            actual_steps, avg_loss, len(self.policy_buffer),
        )
        return avg_loss

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_iterations: Optional[int] = None) -> None:
        """
        Main ReBeL training loop.

        Args:
            num_iterations: Number of training iterations (overrides config.rebel_epochs
                if provided). Each iteration runs rebel_games_per_epoch self-play episodes.
        """
        total_iters = num_iterations if num_iterations is not None else self.config.rebel_epochs
        episodes_per_iter = self.config.rebel_games_per_epoch
        train_steps = self.config.train_steps_per_iteration
        save_interval = self.config.save_interval

        _max_tasks = _resolve_max_tasks_per_child(
            self.config.max_tasks_per_child,
            self.config.worker_memory_budget_pct,
        )
        ctx = multiprocessing.get_context("spawn")

        start_iter = self.current_iteration + 1
        end_iter = self.current_iteration + total_iters

        game_config = self.game_config

        logger.info(
            "Starting ReBeL training from iter %d to %d "
            "(%d episodes/iter, %d train_steps/iter).",
            start_iter, end_iter, episodes_per_iter, train_steps,
        )

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=ctx, max_tasks_per_child=_max_tasks
        )

        try:
            for iteration in range(start_iter, end_iter + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before iteration")

                self.current_iteration = iteration
                iter_start = time.time()
                phase_times: Dict[str, float] = {}

                # 1. Copy network weights to numpy for worker pickling
                value_state, policy_state = self._get_network_state_dicts_numpy()

                # 2. Run self-play episodes
                _sp_start = time.time()
                worker_args = (
                    episodes_per_iter,
                    value_state,
                    policy_state,
                    self.config,
                    game_config,
                )
                try:
                    future = executor.submit(_rebel_batch_worker, worker_args)
                    all_samples = future.result()
                except Exception as e:
                    logger.error(
                        "Self-play workers failed at iter %d: %s", iteration, e
                    )
                    all_samples = []
                phase_times["self_play"] = time.time() - _sp_start

                # 3. Insert into buffers
                self._insert_samples(all_samples, iteration)

                # 4. Train value network
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before value training")

                _vt_start = time.time()
                v_loss = self._train_value_network(train_steps)
                phase_times["value_train"] = time.time() - _vt_start
                self.value_loss_history.append((iteration, v_loss))

                # 5. Train policy network
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before policy training")

                _pt_start = time.time()
                p_loss = self._train_policy_network(train_steps)
                phase_times["policy_train"] = time.time() - _pt_start
                self.policy_loss_history.append((iteration, p_loss))

                iter_time = time.time() - iter_start

                # 6. Checkpoint save
                if save_interval > 0 and iteration % save_interval == 0:
                    self.save_checkpoint()

                # 7. Headless output
                phase_str = " ".join(
                    f"{k}={v:.1f}s" for k, v in phase_times.items()
                )
                print(
                    f"[rebel] iter {iteration} | episodes={len(all_samples)} "
                    f"{phase_str} | "
                    f"v_loss={v_loss:.2f} p_loss={p_loss:.2f} | "
                    f"v_buf={len(self.value_buffer)} p_buf={len(self.policy_buffer)}",
                    flush=True,
                )

                logger.info(
                    "Iter %d complete in %.2fs. v_loss=%.6f, p_loss=%.6f. "
                    "Buffers: v=%d, p=%d. Samples: %d",
                    iteration, iter_time, v_loss, p_loss,
                    len(self.value_buffer), len(self.policy_buffer),
                    len(all_samples),
                )

            logger.info("ReBeL training completed %d iterations.", total_iters)

        except (GracefulShutdownException, KeyboardInterrupt) as e:
            logger.warning("Shutdown during ReBeL training: %s", type(e).__name__)
            executor.shutdown(wait=False)
            self.save_checkpoint()
            raise GracefulShutdownException("Shutdown processed in ReBeLTrainer") from e

        except Exception as e:
            logger.exception("Unhandled error in ReBeL training loop.")
            executor.shutdown(wait=False)
            self.save_checkpoint()
            raise

        # Final save on clean exit
        if not self.shutdown_event.is_set():
            executor.shutdown(wait=True)
            self.save_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _build_checkpoint_metadata(self) -> Dict[str, Any]:
        """Build provenance metadata for checkpoint reproducibility."""
        import platform
        import subprocess

        metadata: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sampling_method": "rebel",
            "device": str(self.device),
            "iteration": self.current_iteration,
        }

        # Full config snapshot
        try:
            config_dict = asdict(self.config)
            import json
            json.dumps(config_dict, default=str)
            metadata["config"] = config_dict
        except Exception:
            metadata["config"] = None

        # Git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(__file__),
            )
            metadata["git_commit"] = (
                result.stdout.strip() if result.returncode == 0 else None
            )
        except Exception:
            metadata["git_commit"] = None

        # Hardware
        metadata["hardware"] = {
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "pytorch_version": str(torch.__version__),
            "xpu_available": bool(
                getattr(torch.xpu, "is_available", lambda: False)()
                if hasattr(torch, "xpu") else False
            ),
            "cuda_available": bool(torch.cuda.is_available()),
        }

        # Buffer sizes at save time
        metadata["buffer_sizes"] = {
            "value": len(self.value_buffer),
            "policy": len(self.policy_buffer),
        }

        # Network param counts
        metadata["network_params"] = {
            "value_net": sum(p.numel() for p in self.value_net.parameters()),
            "policy_net": sum(p.numel() for p in self.policy_net.parameters()),
        }

        # Warm-start lineage
        if self._loaded_from_checkpoint:
            metadata["warm_start"] = {
                "source_checkpoint": self._loaded_from_checkpoint,
                "source_metadata": self._loaded_from_metadata,
            }
        else:
            metadata["warm_start"] = None

        return metadata

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """
        Save training state: networks, optimizers, buffers, iteration counter.

        Atomic save pattern: writes to a temp file then renames, same as deep_trainer.
        Buffers are saved as .npz files alongside the main .pt checkpoint.

        Raises:
            CheckpointSaveError: If saving fails.
        """
        path = filepath or self.checkpoint_path

        try:
            checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
            os.makedirs(checkpoint_dir, exist_ok=True)

            base_path = os.path.splitext(path)[0]
            value_buffer_path = f"{base_path}_rebel_value_buffer"
            policy_buffer_path = f"{base_path}_rebel_policy_buffer"

            checkpoint = {
                "rebel_value_net_state_dict": self.value_net.state_dict(),
                "rebel_policy_net_state_dict": self.policy_net.state_dict(),
                "rebel_value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "rebel_policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "iteration": self.current_iteration,
                "rebel_config": asdict(self.config),
                "metadata": self._build_checkpoint_metadata(),
                "value_loss_history": self.value_loss_history,
                "policy_loss_history": self.policy_loss_history,
                "value_buffer_path": value_buffer_path,
                "policy_buffer_path": policy_buffer_path,
            }

            atomic_torch_save(checkpoint, path)
            atomic_npz_save(self.value_buffer.save, value_buffer_path)
            atomic_npz_save(self.policy_buffer.save, policy_buffer_path)

            # Iteration-specific .pt copy for post-hoc evaluation
            iter_path = f"{base_path}_iter_{self.current_iteration}.pt"
            try:
                atomic_torch_save(checkpoint, iter_path)
            except Exception as e_iter:
                logger.warning(
                    "Failed to save iter checkpoint %s: %s", iter_path, e_iter
                )

            logger.info(
                "Checkpoint saved to %s (iter %d).", path, self.current_iteration
            )
            print(
                f"[checkpoint] saved to {path} (iter {self.current_iteration})",
                flush=True,
            )

        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Failed to save checkpoint to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(
                f"Unexpected error saving checkpoint to {path}: {e}"
            ) from e

    def load_checkpoint(self, filepath: Optional[str] = None) -> None:
        """
        Load training state from checkpoint.

        Raises:
            CheckpointLoadError: If loading fails.
        """
        path = filepath or self.checkpoint_path

        if not os.path.exists(path):
            logger.info("No checkpoint found at %s. Starting fresh.", path)
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)

            self.value_net.load_state_dict(checkpoint["rebel_value_net_state_dict"])
            self.policy_net.load_state_dict(checkpoint["rebel_policy_net_state_dict"])
            self.value_optimizer.load_state_dict(
                checkpoint["rebel_value_optimizer_state_dict"]
            )
            self.policy_optimizer.load_state_dict(
                checkpoint["rebel_policy_optimizer_state_dict"]
            )

            self.current_iteration = checkpoint.get("iteration", 0)
            self.value_loss_history = checkpoint.get("value_loss_history", [])
            self.policy_loss_history = checkpoint.get("policy_loss_history", [])

            # Load buffers
            value_buffer_path = checkpoint.get("value_buffer_path")
            policy_buffer_path = checkpoint.get("policy_buffer_path")

            if value_buffer_path:
                npz_path = (
                    value_buffer_path
                    if value_buffer_path.endswith(".npz")
                    else value_buffer_path + ".npz"
                )
                if os.path.exists(npz_path):
                    self.value_buffer = ReservoirBuffer(
                        capacity=self.config.rebel_value_buffer_capacity,
                        input_dim=PBS_INPUT_DIM,
                        target_dim=VALUE_OUTPUT_DIM,
                        has_mask=False,
                    )
                    self.value_buffer.load(value_buffer_path)
                    logger.info(
                        "Loaded value buffer: %d samples.", len(self.value_buffer)
                    )
                else:
                    logger.warning("Value buffer file not found: %s", npz_path)

            if policy_buffer_path:
                npz_path = (
                    policy_buffer_path
                    if policy_buffer_path.endswith(".npz")
                    else policy_buffer_path + ".npz"
                )
                if os.path.exists(npz_path):
                    self.policy_buffer = ReservoirBuffer(
                        capacity=self.config.rebel_policy_buffer_capacity,
                        input_dim=PBS_INPUT_DIM,
                        target_dim=POLICY_OUTPUT_DIM,
                        has_mask=True,
                    )
                    self.policy_buffer.load(policy_buffer_path)
                    logger.info(
                        "Loaded policy buffer: %d samples.", len(self.policy_buffer)
                    )
                else:
                    logger.warning("Policy buffer file not found: %s", npz_path)

            # Record warm-start lineage
            self._loaded_from_checkpoint = os.path.abspath(path)
            self._loaded_from_metadata = checkpoint.get("metadata")

            logger.info(
                "Checkpoint loaded from %s. Resuming at iter %d. "
                "Buffers: v=%d, p=%d.",
                path, self.current_iteration,
                len(self.value_buffer), len(self.policy_buffer),
            )

        except FileNotFoundError as e:
            raise CheckpointLoadError(f"Checkpoint not found: {path}") from e
        except (KeyError, ValueError) as e:
            raise CheckpointLoadError(
                f"Corrupted or incompatible checkpoint {path}: {e}"
            ) from e
        except (OSError, IOError) as e:
            raise CheckpointLoadError(
                f"Failed to load checkpoint from {path}: {e}"
            ) from e
        except Exception as e:
            raise CheckpointLoadError(
                f"Unexpected error loading checkpoint from {path}: {e}"
            ) from e
