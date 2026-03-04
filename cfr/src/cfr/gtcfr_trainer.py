"""
src/cfr/gtcfr_trainer.py

GT-CFR Trainer — training loop orchestrator for Phase 2.

Architecture:
- CVPN (dual-head): predicts counterfactual values (936) + policy logits (146)
- Two ReservoirBuffers: value_buffer (has_mask=False) and policy_buffer (has_mask=True)
- Self-play episodes via ProcessPoolExecutor (spawn context)

Training loop per epoch:
1. Copy CVPN state_dict (numpy) for worker pickling
2. Run gtcfr_games_per_epoch episodes via ProcessPoolExecutor
3. Insert episode samples into value and policy buffers
4. Train CVPN: combined MSE(values) + CE(policy) loss from separate buffer samples
5. Checkpoint save every save_interval epochs
6. Headless output: epoch timing, buffer sizes, losses
"""

import concurrent.futures
import logging
import multiprocessing
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..config import DeepCfrConfig
from ..networks import CVPN, build_cvpn, warm_start_cvpn_from_rebel
from ..persistence import atomic_torch_save, atomic_npz_save
from ..reservoir import ReservoirBuffer, ReservoirSample
from .deep_trainer import _resolve_device, _resolve_max_tasks_per_child
from .exceptions import (
    CheckpointSaveError,
    CheckpointLoadError,
    GracefulShutdownException,
)

logger = logging.getLogger(__name__)

from ..pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from ..encoding import NUM_ACTIONS

VALUE_DIM: int = 2 * NUM_HAND_TYPES   # 936
POLICY_DIM: int = NUM_ACTIONS          # 146


# ---------------------------------------------------------------------------
# Module-level worker function (must be module-level for ProcessPoolExecutor pickle)
# ---------------------------------------------------------------------------


def _gtcfr_batch_worker(args: Tuple) -> List:
    """
    ProcessPoolExecutor worker — runs N GT-CFR self-play episodes and returns all samples.

    Must be at module level (not a closure or method) so ProcessPoolExecutor can
    pickle it for spawn-based worker processes.

    Args:
        args: (num_episodes, cvpn_state_numpy, config, game_config)
            cvpn_state_numpy: {str: np.ndarray} state_dict for CVPN
            config: DeepCfrConfig carrying gtcfr_* hyper-parameters
            game_config: house rules for GoEngine (None = Go defaults)

    Returns:
        Flat list of EpisodeSample objects from all completed episodes.
    """
    num_episodes, cvpn_state_numpy, config, game_config = args

    from ..networks import build_cvpn
    from .gtcfr_worker import gtcfr_self_play_episode

    cvpn = build_cvpn(
        hidden_dim=config.gtcfr_cvpn_hidden_dim,
        num_blocks=config.gtcfr_cvpn_num_blocks,
        validate_inputs=False,
    )
    weights = {
        k: torch.tensor(v) if isinstance(v, np.ndarray) else v
        for k, v in cvpn_state_numpy.items()
    }
    cvpn.load_state_dict(weights)
    cvpn.eval()

    all_samples = []
    with torch.inference_mode():
        for _ in range(num_episodes):
            try:
                samples = gtcfr_self_play_episode(
                    game_config,
                    cvpn,
                    config,
                    exploration_epsilon=config.gtcfr_exploration_epsilon,
                )
                all_samples.extend(samples)
            except Exception as e:
                logger.warning("GT-CFR episode failed: %s", e)

    return all_samples


# ---------------------------------------------------------------------------
# GTCFRTrainer
# ---------------------------------------------------------------------------


class GTCFRTrainer:
    """
    Orchestrates GT-CFR training with self-play episode workers.

    Each epoch:
    1. Run gtcfr_games_per_epoch episodes via ProcessPoolExecutor
    2. Insert episode samples into value_buffer and policy_buffer
    3. Train CVPN with combined value + policy loss (separate buffer samples)
    4. Save checkpoint every save_interval epochs
    """

    def __init__(
        self,
        config: DeepCfrConfig,
        game_config: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
    ):
        self.config = config
        self.game_config = game_config
        self.checkpoint_path = checkpoint_path or "gtcfr_checkpoint.pt"
        self.shutdown_event = shutdown_event or threading.Event()

        # Device
        resolved_device = _resolve_device(config.device)
        self.device = torch.device(resolved_device)
        logger.info("GTCFRTrainer using device: %s", self.device)

        # Network
        self.cvpn = build_cvpn(
            hidden_dim=config.gtcfr_cvpn_hidden_dim,
            num_blocks=config.gtcfr_cvpn_num_blocks,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.cvpn.parameters(), lr=config.gtcfr_cvpn_learning_rate
        )

        # Two buffers (like rebel): value (no mask), policy (with mask)
        self.value_buffer = ReservoirBuffer(
            capacity=config.gtcfr_buffer_capacity,
            input_dim=PBS_INPUT_DIM,
            target_dim=VALUE_DIM,
            has_mask=False,
        )
        self.policy_buffer = ReservoirBuffer(
            capacity=config.gtcfr_buffer_capacity,
            input_dim=PBS_INPUT_DIM,
            target_dim=POLICY_DIM,
            has_mask=True,
        )

        # Training state
        self.current_epoch: int = 0
        self.loss_history: List[Tuple[int, float, float]] = []

        logger.info(
            "GTCFRTrainer initialized. CVPN params: %d",
            sum(p.numel() for p in self.cvpn.parameters()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cvpn_state_numpy(self) -> Dict[str, Any]:
        """Serialize CVPN to CPU numpy state dict for pickling to workers."""
        return {k: v.cpu().numpy() for k, v in self.cvpn.state_dict().items()}

    def _insert_samples(self, samples: List, epoch: int) -> None:
        """Insert EpisodeSamples into value and policy buffers."""
        for sample in samples:
            v_sample = ReservoirSample(
                features=sample.features,
                target=sample.value_target,
                action_mask=np.empty(0, dtype=bool),  # not used (has_mask=False)
                iteration=epoch,
            )
            self.value_buffer.add(v_sample)

            p_sample = ReservoirSample(
                features=sample.features,
                target=sample.policy_target,
                action_mask=sample.action_mask,
                iteration=epoch,
            )
            self.policy_buffer.add(p_sample)

    def _generate_episodes(self, num_episodes: int) -> List:
        """Run self-play episodes via ProcessPoolExecutor."""
        cvpn_state = self._get_cvpn_state_numpy()
        worker_args = (num_episodes, cvpn_state, self.config, self.game_config)

        _max_tasks = _resolve_max_tasks_per_child(
            self.config.max_tasks_per_child,
            self.config.worker_memory_budget_pct,
        )
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=ctx, max_tasks_per_child=_max_tasks
        ) as executor:
            try:
                future = executor.submit(_gtcfr_batch_worker, worker_args)
                return future.result()
            except Exception as e:
                logger.error("GT-CFR self-play workers failed: %s", e)
                return []

    def _train_step(self, num_steps: int) -> Tuple[float, float]:
        """
        Train CVPN on buffer samples.

        Loss = gtcfr_value_loss_weight * MSE(values, value_targets)
             + gtcfr_policy_loss_weight * CE(policy_logits, policy_targets)

        Samples value and policy from their respective buffers separately.

        Returns:
            (avg_value_loss, avg_policy_loss)
        """
        if len(self.value_buffer) == 0 or len(self.policy_buffer) == 0:
            logger.warning("Cannot train CVPN: buffer(s) empty.")
            return 0.0, 0.0

        self.cvpn.train()
        total_v_loss = 0.0
        total_p_loss = 0.0
        actual_steps = 0

        batch_size = self.config.batch_size
        v_weight = self.config.gtcfr_value_loss_weight
        p_weight = self.config.gtcfr_policy_loss_weight

        for _ in range(num_steps):
            if self.shutdown_event.is_set():
                break

            v_batch = self.value_buffer.sample_batch(batch_size)
            p_batch = self.policy_buffer.sample_batch(batch_size)
            if not v_batch or not p_batch:
                break

            # Value head: no masks needed
            v_features = torch.from_numpy(v_batch.features).float().to(self.device)
            value_targets = torch.from_numpy(v_batch.targets).float().to(self.device)
            # Use all-true mask (value head ignores masking)
            v_mask = torch.ones(
                v_features.shape[0], POLICY_DIM, dtype=torch.bool, device=self.device
            )

            # Policy head
            p_features = torch.from_numpy(p_batch.features).float().to(self.device)
            policy_targets = torch.from_numpy(p_batch.targets).float().to(self.device)
            p_mask = torch.from_numpy(p_batch.masks).to(self.device)

            self.optimizer.zero_grad()

            # Forward value head
            values_pred, _ = self.cvpn(v_features, v_mask)
            v_loss = F.mse_loss(values_pred, value_targets)

            # Forward policy head
            _, policy_logits = self.cvpn(p_features, p_mask)
            # Cross-entropy with soft targets: -(target * log_softmax).sum(dim=-1).mean()
            # Mask illegal actions to 0 in log_probs to avoid 0 * -inf = nan
            log_probs = F.log_softmax(policy_logits, dim=-1)
            safe_log_probs = log_probs.masked_fill(~p_mask, 0.0)
            p_loss = -(policy_targets * safe_log_probs).sum(dim=-1).mean()

            loss = v_weight * v_loss + p_weight * p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cvpn.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            actual_steps += 1

        avg_v = total_v_loss / max(actual_steps, 1)
        avg_p = total_p_loss / max(actual_steps, 1)
        logger.info(
            "CVPN training: %d steps, value_loss=%.6f policy_loss=%.6f "
            "(v_buf=%d p_buf=%d)",
            actual_steps, avg_v, avg_p, len(self.value_buffer), len(self.policy_buffer),
        )
        return avg_v, avg_p

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        Main GT-CFR training loop.

        Args:
            num_epochs: Number of training epochs (overrides config.gtcfr_epochs if given).
        """
        total_epochs = num_epochs if num_epochs is not None else self.config.gtcfr_epochs
        episodes_per_epoch = self.config.gtcfr_games_per_epoch
        train_steps = self.config.train_steps_per_iteration
        save_interval = self.config.save_interval

        start_epoch = self.current_epoch + 1
        end_epoch = self.current_epoch + total_epochs

        logger.info(
            "Starting GT-CFR training from epoch %d to %d "
            "(%d episodes/epoch, %d train_steps/epoch).",
            start_epoch, end_epoch, episodes_per_epoch, train_steps,
        )

        try:
            for epoch in range(start_epoch, end_epoch + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before epoch")

                self.current_epoch = epoch
                epoch_start = time.time()

                # 1. Self-play
                _sp_start = time.time()
                all_samples = self._generate_episodes(episodes_per_epoch)
                sp_time = time.time() - _sp_start

                # 2. Insert into buffers
                self._insert_samples(all_samples, epoch)

                # 3. Train CVPN
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before training")

                _train_start = time.time()
                v_loss, p_loss = self._train_step(train_steps)
                train_time = time.time() - _train_start
                self.loss_history.append((epoch, v_loss, p_loss))

                epoch_time = time.time() - epoch_start

                # 4. Checkpoint
                if save_interval > 0 and epoch % save_interval == 0:
                    self.save_checkpoint()

                # 5. Headless output
                print(
                    f"[gtcfr] epoch {epoch} | samples={len(all_samples)} "
                    f"sp={sp_time:.1f}s train={train_time:.1f}s | "
                    f"v_loss={v_loss:.4f} p_loss={p_loss:.4f} | "
                    f"v_buf={len(self.value_buffer)} p_buf={len(self.policy_buffer)}",
                    flush=True,
                )

                logger.info(
                    "Epoch %d complete in %.2fs. v_loss=%.6f p_loss=%.6f "
                    "v_buf=%d p_buf=%d samples=%d",
                    epoch, epoch_time, v_loss, p_loss,
                    len(self.value_buffer), len(self.policy_buffer), len(all_samples),
                )

            logger.info("GT-CFR training completed %d epochs.", total_epochs)

        except (GracefulShutdownException, KeyboardInterrupt) as e:
            logger.warning("Shutdown during GT-CFR training: %s", type(e).__name__)
            self.save_checkpoint()
            raise GracefulShutdownException("Shutdown processed in GTCFRTrainer") from e

        except Exception as e:
            logger.exception("Unhandled error in GT-CFR training loop.")
            self.save_checkpoint()
            raise

        # Final save on clean exit
        if not self.shutdown_event.is_set():
            self.save_checkpoint()

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def warm_start_from_rebel(self, rebel_checkpoint_path: str) -> None:
        """Load compatible weights from a Phase 1 ReBeL checkpoint into CVPN."""
        if not os.path.exists(rebel_checkpoint_path):
            logger.warning(
                "ReBeL checkpoint not found at %s — skipping warm start.",
                rebel_checkpoint_path,
            )
            return

        try:
            ckpt = torch.load(rebel_checkpoint_path, map_location="cpu", weights_only=True)
            policy_state = ckpt.get("rebel_policy_net_state_dict", {})
            value_state = ckpt.get("rebel_value_net_state_dict", {})
            skipped = warm_start_cvpn_from_rebel(
                self.cvpn, policy_state_dict=policy_state, value_state_dict=value_state
            )
            logger.info(
                "Warm start from ReBeL checkpoint %s. Skipped %d keys.",
                rebel_checkpoint_path, len(skipped),
            )
            print(f"[gtcfr] warm start from {rebel_checkpoint_path} (skipped {len(skipped)} keys)")
        except Exception as e:
            logger.error("Failed to warm start from %s: %s", rebel_checkpoint_path, e)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """Save CVPN, optimizer, buffers, epoch counter.

        Raises:
            CheckpointSaveError: If saving fails.
        """
        path = filepath or self.checkpoint_path

        try:
            checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
            os.makedirs(checkpoint_dir, exist_ok=True)

            base_path = os.path.splitext(path)[0]
            value_buffer_path = f"{base_path}_gtcfr_value_buffer"
            policy_buffer_path = f"{base_path}_gtcfr_policy_buffer"

            checkpoint = {
                "cvpn_state_dict": self.cvpn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "buffer_size": len(self.value_buffer),
                "config": self.config.model_dump(),
                "loss_history": self.loss_history,
                "value_buffer_path": value_buffer_path,
                "policy_buffer_path": policy_buffer_path,
                "metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(self.device),
                },
            }

            atomic_torch_save(checkpoint, path)
            atomic_npz_save(self.value_buffer.save, value_buffer_path)
            atomic_npz_save(self.policy_buffer.save, policy_buffer_path)

            logger.info("GT-CFR checkpoint saved to %s (epoch %d).", path, self.current_epoch)
            print(f"[checkpoint] gtcfr saved to {path} (epoch {self.current_epoch})", flush=True)

        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(f"Failed to save checkpoint to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving checkpoint to %s: %s", path, e)
            raise CheckpointSaveError(
                f"Unexpected error saving checkpoint to {path}: {e}"
            ) from e

    def load_checkpoint(self, filepath: Optional[str] = None) -> None:
        """Load CVPN, optimizer, buffers, epoch counter.

        Raises:
            CheckpointLoadError: If loading fails.
        """
        path = filepath or self.checkpoint_path

        if not os.path.exists(path):
            logger.info("No checkpoint found at %s. Starting fresh.", path)
            return

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)

            self.cvpn.load_state_dict(checkpoint["cvpn_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint.get("epoch", 0)
            self.loss_history = checkpoint.get("loss_history", [])

            # Load value buffer
            vbp = checkpoint.get("value_buffer_path")
            if vbp:
                npz = vbp if vbp.endswith(".npz") else vbp + ".npz"
                if os.path.exists(npz):
                    self.value_buffer = ReservoirBuffer(
                        capacity=self.config.gtcfr_buffer_capacity,
                        input_dim=PBS_INPUT_DIM,
                        target_dim=VALUE_DIM,
                        has_mask=False,
                    )
                    self.value_buffer.load(vbp)
                    logger.info("Loaded value buffer: %d samples.", len(self.value_buffer))
                else:
                    logger.warning("Value buffer file not found: %s", npz)

            # Load policy buffer
            pbp = checkpoint.get("policy_buffer_path")
            if pbp:
                npz = pbp if pbp.endswith(".npz") else pbp + ".npz"
                if os.path.exists(npz):
                    self.policy_buffer = ReservoirBuffer(
                        capacity=self.config.gtcfr_buffer_capacity,
                        input_dim=PBS_INPUT_DIM,
                        target_dim=POLICY_DIM,
                        has_mask=True,
                    )
                    self.policy_buffer.load(pbp)
                    logger.info("Loaded policy buffer: %d samples.", len(self.policy_buffer))
                else:
                    logger.warning("Policy buffer file not found: %s", npz)

            logger.info(
                "GT-CFR checkpoint loaded from %s. Resuming at epoch %d.",
                path, self.current_epoch,
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
