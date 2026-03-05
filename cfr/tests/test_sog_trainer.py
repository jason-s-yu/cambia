"""
tests/test_sog_trainer.py

Tests for sog_trainer.py (training loop coordinator).

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
from src.cfr.sog_trainer import (
    SoGTrainer,
    VALUE_DIM,
    POLICY_DIM,
    PBS_INPUT_DIM,
    NUM_ACTIONS,
)
from src.cfr.gtcfr_worker import EpisodeSample
from src.networks import build_cvpn

if __name__ == "__main__":
    # Prevent ProcessPoolExecutor spawn workers from re-running tests.
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fast_config(**overrides) -> DeepCfrConfig:
    cfg = DeepCfrConfig()
    cfg.gtcfr_cvpn_hidden_dim = 32
    cfg.gtcfr_cvpn_num_blocks = 1
    cfg.gtcfr_cvpn_learning_rate = 1e-3
    cfg.gtcfr_value_loss_weight = 1.0
    cfg.gtcfr_policy_loss_weight = 1.0
    cfg.gtcfr_buffer_capacity = 1000
    cfg.sog_games_per_epoch = 1
    cfg.sog_epochs = 1
    cfg.sog_train_budget = 2
    cfg.sog_eval_budget = 5
    cfg.sog_exploration_epsilon = 0.5
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
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[:5] = True
    samples = []
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
# Tests: init
# ---------------------------------------------------------------------------


class TestSoGTrainerInit:
    def test_trainer_init(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config, game_config=None)

        assert trainer.cvpn is not None
        assert trainer.value_buffer is not None
        assert trainer.policy_buffer is not None
        assert trainer.current_epoch == 0
        assert len(trainer.value_buffer) == 0
        assert len(trainer.policy_buffer) == 0
        n_params = sum(p.numel() for p in trainer.cvpn.parameters())
        assert n_params > 0

    def test_trainer_buffer_dims(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
        assert trainer.value_buffer._target_dim == VALUE_DIM
        assert not trainer.value_buffer._has_mask
        assert trainer.policy_buffer._target_dim == POLICY_DIM
        assert trainer.policy_buffer._has_mask

    def test_trainer_uses_gtcfr_cvpn_config(self):
        """SoGTrainer should use gtcfr_cvpn_* fields (same CVPN architecture)."""
        config = make_fast_config()
        config.gtcfr_cvpn_hidden_dim = 64
        config.gtcfr_cvpn_num_blocks = 2
        trainer = SoGTrainer(config=config)
        n_params = sum(p.numel() for p in trainer.cvpn.parameters())

        # Larger CVPN should have more params
        config2 = make_fast_config()
        config2.gtcfr_cvpn_hidden_dim = 32
        config2.gtcfr_cvpn_num_blocks = 1
        trainer2 = SoGTrainer(config=config2)
        n_params2 = sum(p.numel() for p in trainer2.cvpn.parameters())

        assert n_params > n_params2


# ---------------------------------------------------------------------------
# Tests: insert + train step
# ---------------------------------------------------------------------------


class TestInsertAndTrain:
    def test_insert_samples_grows_buffer(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
        samples = make_fake_episode_samples(n=8)
        trainer._insert_samples(samples, epoch=1)
        assert len(trainer.value_buffer) == 8
        assert len(trainer.policy_buffer) == 8

    def test_train_step_empty_returns_zeros(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
        v, p = trainer._train_step(2)
        assert v == 0.0 and p == 0.0

    def test_train_step_with_data_finite(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
        samples = make_fake_episode_samples(n=10)
        trainer._insert_samples(samples, epoch=1)
        v, p = trainer._train_step(2)
        assert np.isfinite(v)
        assert np.isfinite(p)


# ---------------------------------------------------------------------------
# Tests: one epoch
# ---------------------------------------------------------------------------


class TestSoGTrainerOneEpoch:
    def test_trainer_one_epoch(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
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


# ---------------------------------------------------------------------------
# Tests: checkpoint roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_checkpoint_save_load(self):
        """Save and reload, verify CVPN weights and sog_metadata match."""
        config = make_fast_config()
        trainer = SoGTrainer(config=config)

        # Do one train step so weights are non-default
        samples = make_fake_episode_samples(n=8)
        trainer._insert_samples(samples, epoch=1)
        trainer._train_step(1)
        trainer.current_epoch = 3

        original_weights = {
            k: v.clone() for k, v in trainer.cvpn.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_sog.pt")
            trainer.save_checkpoint(ckpt_path)

            # Verify sog_metadata in checkpoint
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert "sog_metadata" in saved
            assert saved["sog_metadata"]["phase"] == 3
            assert saved["sog_metadata"]["trainer"] == "SoGTrainer"
            assert "cvpn_state_dict" in saved  # same key as GT-CFR

            # Load into fresh trainer
            trainer2 = SoGTrainer(config=config)
            trainer2.load_checkpoint(ckpt_path)

            # Verify CVPN weights match
            for k, v in trainer2.cvpn.state_dict().items():
                assert torch.allclose(v, original_weights[k], atol=1e-6), (
                    f"CVPN weight mismatch for key {k}"
                )

            assert trainer2.current_epoch == 3

    def test_load_nonexistent_starts_fresh(self):
        config = make_fast_config()
        trainer = SoGTrainer(config=config)
        trainer.load_checkpoint("/nonexistent/sog_checkpoint.pt")
        assert trainer.current_epoch == 0

    def test_cross_load_from_gtcfr_checkpoint(self):
        """SoGTrainer can load a GT-CFR checkpoint (same cvpn_state_dict key)."""
        config = make_fast_config()

        # Build a fake GT-CFR-style checkpoint (no sog_metadata)
        cvpn = build_cvpn(hidden_dim=32, num_blocks=1)
        gtcfr_ckpt = {
            "cvpn_state_dict": cvpn.state_dict(),
            "optimizer_state_dict": torch.optim.Adam(cvpn.parameters()).state_dict(),
            "epoch": 42,
            "loss_history": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "gtcfr_ckpt.pt")
            torch.save(gtcfr_ckpt, ckpt_path)

            trainer = SoGTrainer(config=config)
            trainer.load_checkpoint(ckpt_path)  # should not raise

            assert trainer.current_epoch == 42

    def test_warm_start_from_gtcfr(self):
        """warm_start_from_gtcfr loads CVPN weights from GT-CFR checkpoint."""
        config = make_fast_config()

        cvpn = build_cvpn(hidden_dim=32, num_blocks=1)
        # Set distinctive weights
        with torch.no_grad():
            for p in cvpn.parameters():
                p.fill_(0.42)
        gtcfr_ckpt = {
            "cvpn_state_dict": cvpn.state_dict(),
            "epoch": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "gtcfr.pt")
            torch.save(gtcfr_ckpt, ckpt_path)

            trainer = SoGTrainer(config=config)
            trainer.warm_start_from_gtcfr(ckpt_path)

            # CVPN weights should match the saved ones
            for k, v in trainer.cvpn.state_dict().items():
                assert torch.allclose(v, cvpn.state_dict()[k], atol=1e-6), (
                    f"warm start weight mismatch at {k}"
                )
