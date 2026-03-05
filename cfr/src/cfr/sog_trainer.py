"""
src/cfr/sog_trainer.py

SoG Trainer: training loop coordinator for Phase 3.

Mirrors GTCFRTrainer closely. Differences:
- Uses _sog_batch_worker (SoG self-play with tree persistence)
- Buffer paths: *_sog_value_buffer.npz, *_sog_policy_buffer.npz
- Checkpoint key: cvpn_state_dict (same as GT-CFR for cross-loading) + sog_metadata
- Epoch checkpoint: *_sog_epoch_N.pt
- Uses sog_games_per_epoch and sog_epochs config fields
- Warm start from GT-CFR checkpoint or ReBeL
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
from .sog_worker import _sog_batch_worker

logger = logging.getLogger(__name__)

from ..pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from ..encoding import NUM_ACTIONS

VALUE_DIM: int = 2 * NUM_HAND_TYPES   # 936
POLICY_DIM: int = NUM_ACTIONS          # 146


class SoGTrainer:
    """
    Runs SoG training with self-play episode workers.

    Each epoch:
    1. Run sog_games_per_epoch episodes via ProcessPoolExecutor (SoG self-play)
    2. Insert episode samples into value_buffer and policy_buffer
    3. Train CVPN with combined value + policy loss (separate buffer samples)
    4. Save checkpoint every save_interval epochs

    Uses the same CVPN architecture and checkpoint key (cvpn_state_dict) as
    GTCFRTrainer, enabling cross-loading from GT-CFR checkpoints.
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
        self.checkpoint_path = checkpoint_path or "sog_checkpoint.pt"
        self.shutdown_event = shutdown_event or threading.Event()

        resolved_device = _resolve_device(config.device)
        self.device = torch.device(resolved_device)
        logger.info("SoGTrainer using device: %s", self.device)

        # Network (same CVPN architecture as GT-CFR, reuse gtcfr_* config fields)
        self.cvpn = build_cvpn(
            hidden_dim=config.gtcfr_cvpn_hidden_dim,
            num_blocks=config.gtcfr_cvpn_num_blocks,
            detach_policy_grad=config.cvpn_detach_policy_grad,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.cvpn.parameters(), lr=config.gtcfr_cvpn_learning_rate
        )

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

        self.current_epoch: int = 0
        self.loss_history: List[Tuple[int, float, float]] = []

        logger.info(
            "SoGTrainer initialized. CVPN params: %d, detach_policy_grad: %s",
            sum(p.numel() for p in self.cvpn.parameters()),
            config.cvpn_detach_policy_grad,
        )

        # Auto warm-start from config if specified and no explicit checkpoint
        if config.sog_warm_start_checkpoint and checkpoint_path and not os.path.exists(checkpoint_path):
            self.warm_start_from_gtcfr(config.sog_warm_start_checkpoint)

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
                action_mask=np.empty(0, dtype=bool),
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
        """Run SoG self-play episodes via ProcessPoolExecutor."""
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
                future = executor.submit(_sog_batch_worker, worker_args)
                return future.result()
            except Exception as e:
                logger.error("SoG self-play workers failed: %s", e)
                return []

    def _train_step(self, num_steps: int) -> Tuple[float, float]:
        """
        Train CVPN on buffer samples.

        Loss = gtcfr_value_loss_weight * MSE(values, value_targets)
             + gtcfr_policy_loss_weight * CE(policy_logits, policy_targets)

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

            v_features = torch.from_numpy(v_batch.features).float().to(self.device)
            value_targets = torch.from_numpy(v_batch.targets).float().to(self.device)
            v_mask = torch.ones(
                v_features.shape[0], POLICY_DIM, dtype=torch.bool, device=self.device
            )

            p_features = torch.from_numpy(p_batch.features).float().to(self.device)
            policy_targets = torch.from_numpy(p_batch.targets).float().to(self.device)
            p_mask = torch.from_numpy(p_batch.masks).to(self.device)

            self.optimizer.zero_grad()

            values_pred, _ = self.cvpn(v_features, v_mask)
            v_loss = F.mse_loss(values_pred, value_targets)

            _, policy_logits = self.cvpn(p_features, p_mask)
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
        Main SoG training loop.

        Args:
            num_epochs: Number of training epochs (overrides config.sog_epochs if given).
        """
        total_epochs = num_epochs if num_epochs is not None else self.config.sog_epochs
        episodes_per_epoch = self.config.sog_games_per_epoch
        train_steps = self.config.train_steps_per_iteration
        save_interval = self.config.save_interval

        start_epoch = self.current_epoch + 1
        end_epoch = self.current_epoch + total_epochs

        logger.info(
            "Starting SoG training from epoch %d to %d "
            "(%d episodes/epoch, %d train_steps/epoch).",
            start_epoch, end_epoch, episodes_per_epoch, train_steps,
        )

        try:
            for epoch in range(start_epoch, end_epoch + 1):
                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before epoch")

                self.current_epoch = epoch
                epoch_start = time.time()

                _sp_start = time.time()
                all_samples = self._generate_episodes(episodes_per_epoch)
                sp_time = time.time() - _sp_start

                self._insert_samples(all_samples, epoch)

                if self.shutdown_event.is_set():
                    raise GracefulShutdownException("Shutdown before training")

                _train_start = time.time()
                v_loss, p_loss = self._train_step(train_steps)
                train_time = time.time() - _train_start
                self.loss_history.append((epoch, v_loss, p_loss))

                epoch_time = time.time() - epoch_start

                if save_interval > 0 and epoch % save_interval == 0:
                    self.save_checkpoint()

                print(
                    f"[sog] epoch {epoch} | samples={len(all_samples)} "
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

            logger.info("SoG training completed %d epochs.", total_epochs)

        except (GracefulShutdownException, KeyboardInterrupt) as e:
            logger.warning("Shutdown during SoG training: %s", type(e).__name__)
            self.save_checkpoint()
            raise GracefulShutdownException("Shutdown processed in SoGTrainer") from e

        except Exception as e:
            logger.exception("Unhandled error in SoG training loop.")
            self.save_checkpoint()
            raise

        if not self.shutdown_event.is_set():
            self.save_checkpoint()

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def warm_start_from_gtcfr(self, gtcfr_checkpoint_path: str) -> None:
        """Load CVPN weights from a GT-CFR Phase 2 checkpoint."""
        if not os.path.exists(gtcfr_checkpoint_path):
            logger.warning(
                "GT-CFR checkpoint not found at %s, skipping warm start.",
                gtcfr_checkpoint_path,
            )
            return
        try:
            ckpt = torch.load(
                gtcfr_checkpoint_path, map_location="cpu", weights_only=True
            )
            self.cvpn.load_state_dict(ckpt["cvpn_state_dict"])
            epoch = ckpt.get("epoch", 0)
            logger.info(
                "Warm start from GT-CFR checkpoint %s (epoch %d).",
                gtcfr_checkpoint_path, epoch,
            )
            print(
                f"[sog] warm start from GT-CFR {gtcfr_checkpoint_path} (epoch {epoch})"
            )
        except Exception as e:
            logger.error(
                "Failed to warm start from GT-CFR %s: %s", gtcfr_checkpoint_path, e
            )

    def warm_start_from_rebel(self, rebel_checkpoint_path: str) -> None:
        """Load compatible weights from a Phase 1 ReBeL checkpoint."""
        if not os.path.exists(rebel_checkpoint_path):
            logger.warning(
                "ReBeL checkpoint not found at %s, skipping warm start.",
                rebel_checkpoint_path,
            )
            return
        try:
            ckpt = torch.load(
                rebel_checkpoint_path, map_location="cpu", weights_only=True
            )
            policy_state = ckpt.get("rebel_policy_net_state_dict", {})
            value_state = ckpt.get("rebel_value_net_state_dict", {})
            skipped = warm_start_cvpn_from_rebel(
                self.cvpn, policy_state_dict=policy_state, value_state_dict=value_state
            )
            logger.info(
                "Warm start from ReBeL checkpoint %s. Skipped %d keys.",
                rebel_checkpoint_path, len(skipped),
            )
            print(
                f"[sog] warm start from ReBeL {rebel_checkpoint_path} "
                f"(skipped {len(skipped)} keys)"
            )
        except Exception as e:
            logger.error(
                "Failed to warm start from ReBeL %s: %s", rebel_checkpoint_path, e
            )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """Save CVPN, optimizer, buffers, epoch counter.

        Uses cvpn_state_dict key (same as GT-CFR) for cross-loading.
        Adds sog_metadata key with SoG-specific info.

        Raises:
            CheckpointSaveError: If saving fails.
        """
        path = filepath or self.checkpoint_path

        try:
            checkpoint_dir = os.path.dirname(path) if os.path.dirname(path) else "."
            os.makedirs(checkpoint_dir, exist_ok=True)

            base_path = os.path.splitext(path)[0]
            value_buffer_path = f"{base_path}_sog_value_buffer"
            policy_buffer_path = f"{base_path}_sog_policy_buffer"

            checkpoint = {
                "cvpn_state_dict": self.cvpn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "buffer_size": len(self.value_buffer),
                "config": self.config.model_dump(),
                "loss_history": self.loss_history,
                "value_buffer_path": value_buffer_path,
                "policy_buffer_path": policy_buffer_path,
                "sog_metadata": {
                    "phase": 3,
                    "trainer": "SoGTrainer",
                    "train_budget": self.config.sog_train_budget,
                    "eval_budget": self.config.sog_eval_budget,
                },
                "metadata": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "device": str(self.device),
                },
            }

            atomic_torch_save(checkpoint, path)
            atomic_npz_save(self.value_buffer.save, value_buffer_path)
            atomic_npz_save(self.policy_buffer.save, policy_buffer_path)

            # Epoch-specific .pt copy for post-hoc evaluation
            epoch_path = f"{base_path}_sog_epoch_{self.current_epoch}.pt"
            try:
                atomic_torch_save(checkpoint, epoch_path)
            except Exception as e_epoch:
                logger.warning(
                    "Failed to save epoch checkpoint %s: %s", epoch_path, e_epoch
                )

            logger.info("SoG checkpoint saved to %s (epoch %d).", path, self.current_epoch)
            print(
                f"[checkpoint] sog saved to {path} (epoch {self.current_epoch})",
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
        """Load CVPN, optimizer, buffers, epoch counter.

        Compatible with both GT-CFR checkpoints (cvpn_state_dict key) and
        SoG checkpoints (same key + sog_metadata).

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
                    logger.info(
                        "Loaded policy buffer: %d samples.", len(self.policy_buffer)
                    )
                else:
                    logger.warning("Policy buffer file not found: %s", pbp)

            logger.info(
                "SoG checkpoint loaded from %s. Resuming at epoch %d.",
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
