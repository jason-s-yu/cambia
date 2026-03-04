"""
tests/test_gtcfr_trainer.py

Tests for gtcfr_trainer.py — training loop orchestrator.

Tests run without Go FFI by mocking self-play episode generation.
Network and buffer operations use synthetic data.
"""

import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import DeepCfrConfig
from src.cfr.gtcfr_trainer import (
    GTCFRTrainer,
    VALUE_DIM,
    POLICY_DIM,
    PBS_INPUT_DIM,
    NUM_ACTIONS,
    _gtcfr_batch_worker,
)
from src.cfr.gtcfr_worker import EpisodeSample
from src.networks import build_cvpn

if __name__ == "__main__":
    # __main__ guard: prevents ProcessPoolExecutor spawn workers from re-running tests.
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fast_config(**overrides) -> DeepCfrConfig:
    cfg = DeepCfrConfig()
    cfg.gtcfr_expansion_budget = 5
    cfg.gtcfr_cfr_iters_per_expansion = 2
    cfg.gtcfr_c_puct = 2.0
    cfg.gtcfr_cvpn_hidden_dim = 64
    cfg.gtcfr_cvpn_num_blocks = 1
    cfg.gtcfr_games_per_epoch = 1
    cfg.gtcfr_epochs = 1
    cfg.gtcfr_buffer_capacity = 1000
    cfg.gtcfr_exploration_epsilon = 0.05
    cfg.gtcfr_value_loss_weight = 1.0
    cfg.gtcfr_policy_loss_weight = 1.0
    cfg.gtcfr_cvpn_learning_rate = 1e-3
    cfg.batch_size = 4
    cfg.train_steps_per_iteration = 2
    cfg.save_interval = 0
    cfg.max_tasks_per_child = None
    cfg.worker_memory_budget_pct = 0.10
    cfg.device = "cpu"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_fake_episode_samples(n: int = 5) -> List[EpisodeSample]:
    """Generate n synthetic EpisodeSamples with correct shapes and dtypes."""
    samples = []
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[:5] = True
    for _ in range(n):
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[:5] = 0.2
        samples.append(
            EpisodeSample(
                features=np.random.rand(PBS_INPUT_DIM).astype(np.float32),
                value_target=np.random.rand(VALUE_DIM).astype(np.float32),
                policy_target=policy,
                action_mask=mask.copy(),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGTCFRTrainerInit:
    def test_trainer_init(self):
        """GTCFRTrainer creates CVPN and buffers without error."""
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config, game_config=None)

        assert trainer.cvpn is not None
        assert trainer.value_buffer is not None
        assert trainer.policy_buffer is not None
        assert trainer.current_epoch == 0
        assert len(trainer.value_buffer) == 0
        assert len(trainer.policy_buffer) == 0
        n_params = sum(p.numel() for p in trainer.cvpn.parameters())
        assert n_params > 0

    def test_trainer_buffer_has_correct_dims(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        # Value buffer: no mask, target=VALUE_DIM
        assert trainer.value_buffer._target_dim == VALUE_DIM
        assert not trainer.value_buffer._has_mask
        # Policy buffer: with mask, target=POLICY_DIM
        assert trainer.policy_buffer._target_dim == POLICY_DIM
        assert trainer.policy_buffer._has_mask


class TestInsertSamples:
    def test_insert_samples_grows_buffer(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        samples = make_fake_episode_samples(n=10)
        trainer._insert_samples(samples, epoch=1)
        assert len(trainer.value_buffer) == 10
        assert len(trainer.policy_buffer) == 10

    def test_insert_samples_value_target_shape(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        samples = make_fake_episode_samples(n=3)
        trainer._insert_samples(samples, epoch=1)
        batch = trainer.value_buffer.sample_batch(3)
        assert batch is not None
        assert batch.targets.shape == (3, VALUE_DIM)

    def test_insert_samples_policy_target_shape(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        samples = make_fake_episode_samples(n=3)
        trainer._insert_samples(samples, epoch=1)
        batch = trainer.policy_buffer.sample_batch(3)
        assert batch is not None
        assert batch.targets.shape == (3, POLICY_DIM)
        assert batch.masks.shape == (3, POLICY_DIM)


class TestTrainStep:
    def test_train_step_with_empty_buffer_returns_zeros(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        v_loss, p_loss = trainer._train_step(num_steps=2)
        assert v_loss == 0.0
        assert p_loss == 0.0

    def test_train_step_with_data_returns_finite_loss(self):
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        samples = make_fake_episode_samples(n=10)
        trainer._insert_samples(samples, epoch=1)
        v_loss, p_loss = trainer._train_step(num_steps=2)
        assert np.isfinite(v_loss), f"value_loss is not finite: {v_loss}"
        assert np.isfinite(p_loss), f"policy_loss is not finite: {p_loss}"


class TestTrainerOneEpoch:
    def test_trainer_one_epoch(self):
        """1 epoch (1 episode), buffer grows, losses are finite."""
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)

        fake_samples = make_fake_episode_samples(n=5)

        with patch.object(trainer, "_generate_episodes", return_value=fake_samples):
            trainer.train(num_epochs=1)

        assert trainer.current_epoch == 1
        assert len(trainer.value_buffer) == 5
        assert len(trainer.policy_buffer) == 5
        assert len(trainer.loss_history) == 1
        epoch_idx, v_loss, p_loss = trainer.loss_history[0]
        assert epoch_idx == 1
        assert np.isfinite(v_loss)
        assert np.isfinite(p_loss)


class TestCheckpointSaveLoad:
    def test_checkpoint_save_load(self):
        """Save and reload, verify CVPN weights match."""
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)

        # Set some non-default weights by doing one train step
        samples = make_fake_episode_samples(n=8)
        trainer._insert_samples(samples, epoch=1)
        trainer._train_step(num_steps=1)
        trainer.current_epoch = 3

        # Save original weights
        original_weights = {
            k: v.clone() for k, v in trainer.cvpn.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_gtcfr.pt")
            trainer.save_checkpoint(ckpt_path)

            # Create fresh trainer and load
            trainer2 = GTCFRTrainer(config=config)
            trainer2.load_checkpoint(ckpt_path)

            # Verify CVPN weights match
            for k, v in trainer2.cvpn.state_dict().items():
                assert torch.allclose(v, original_weights[k], atol=1e-6), (
                    f"CVPN weight mismatch for key {k}"
                )

            # Verify epoch
            assert trainer2.current_epoch == 3

    def test_load_nonexistent_checkpoint(self):
        """Loading a missing checkpoint should not raise — starts fresh."""
        config = make_fast_config()
        trainer = GTCFRTrainer(config=config)
        trainer.load_checkpoint("/nonexistent/path/checkpoint.pt")
        assert trainer.current_epoch == 0  # Still fresh
