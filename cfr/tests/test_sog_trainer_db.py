"""
tests/test_sog_trainer_db.py

Tests for run_db integration in SoGTrainer.
Uses mocks to avoid SQLite and Go FFI dependencies.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import DeepCfrConfig
from src.cfr.sog_trainer import SoGTrainer, VALUE_DIM, POLICY_DIM, PBS_INPUT_DIM, NUM_ACTIONS
from src.cfr.gtcfr_worker import EpisodeSample


def _fast_config(**overrides) -> DeepCfrConfig:
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


def _fake_samples(n: int = 5):
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


class TestSoGTrainerCallsUpsertRun:
    """Verify SoGTrainer.__init__ calls run_db.upsert_run when DB is available."""

    def test_upsert_run_called_on_init(self):
        mock_run_db = MagicMock()
        mock_run_db.get_db.return_value = MagicMock()
        mock_run_db.infer_algorithm.return_value = "sog"
        mock_run_db.upsert_run.return_value = 42

        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", True), \
             patch("src.cfr.sog_trainer._run_db", mock_run_db):
            config = _fast_config()
            trainer = SoGTrainer(config=config, game_config=None)

            mock_run_db.get_db.assert_called_once()
            mock_run_db.upsert_run.assert_called_once()
            # Check status="running" is passed
            call_kwargs = mock_run_db.upsert_run.call_args
            assert call_kwargs[1]["status"] == "running"
            assert call_kwargs[1]["algorithm"] == "sog"
            assert trainer._db_run_id == 42
            assert trainer._db_conn is not None


class TestSoGTrainerCallsRegisterCheckpoint:
    """Verify save_checkpoint calls register_checkpoint when DB is available."""

    def test_register_checkpoint_called_on_save(self):
        mock_run_db = MagicMock()
        mock_run_db.get_db.return_value = MagicMock()
        mock_run_db.infer_algorithm.return_value = "sog"
        mock_run_db.upsert_run.return_value = 7
        mock_run_db.register_checkpoint.return_value = 1

        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", True), \
             patch("src.cfr.sog_trainer._run_db", mock_run_db):
            config = _fast_config()
            trainer = SoGTrainer(config=config, game_config=None)
            trainer.current_epoch = 5

            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = os.path.join(tmpdir, "test_sog.pt")
                trainer.save_checkpoint(ckpt_path)

                mock_run_db.register_checkpoint.assert_called_once()
                args = mock_run_db.register_checkpoint.call_args[0]
                # args: (db_conn, run_id, epoch, path)
                assert args[1] == 7  # run_id
                assert args[2] == 5  # epoch

                mock_run_db.compute_retention_flags.assert_called_once()


class TestSoGTrainerUpdateRunStatus:
    """Verify train() calls update_run_status on completion and interruption."""

    def test_completed_status_on_normal_finish(self):
        mock_run_db = MagicMock()
        mock_run_db.get_db.return_value = MagicMock()
        mock_run_db.infer_algorithm.return_value = "sog"
        mock_run_db.upsert_run.return_value = 10

        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", True), \
             patch("src.cfr.sog_trainer._run_db", mock_run_db):
            config = _fast_config(save_interval=0)
            trainer = SoGTrainer(config=config, game_config=None)
            fake = _fake_samples(3)

            with patch.object(trainer, "_generate_episodes", return_value=fake):
                trainer.train(num_epochs=1)

            # Should have called update_run_status with "completed"
            calls = mock_run_db.update_run_status.call_args_list
            statuses = [c[0][2] for c in calls]
            assert "completed" in statuses

    def test_interrupted_status_on_shutdown(self):
        mock_run_db = MagicMock()
        mock_run_db.get_db.return_value = MagicMock()
        mock_run_db.infer_algorithm.return_value = "sog"
        mock_run_db.upsert_run.return_value = 11

        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", True), \
             patch("src.cfr.sog_trainer._run_db", mock_run_db):
            config = _fast_config()
            trainer = SoGTrainer(config=config, game_config=None)

            def raise_keyboard(*a, **kw):
                raise KeyboardInterrupt()

            with patch.object(trainer, "_generate_episodes", side_effect=raise_keyboard):
                from src.cfr.exceptions import GracefulShutdownException
                with pytest.raises(GracefulShutdownException):
                    trainer.train(num_epochs=1)

            calls = mock_run_db.update_run_status.call_args_list
            statuses = [c[0][2] for c in calls]
            assert "interrupted" in statuses


class TestSoGTrainerNonFatalWhenDbUnavailable:
    """SoGTrainer initializes and operates fine when run_db is unavailable."""

    def test_init_without_db(self):
        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", False):
            config = _fast_config()
            trainer = SoGTrainer(config=config, game_config=None)

            assert trainer._db_conn is None
            assert trainer._db_run_id is None
            assert trainer.cvpn is not None
            assert trainer.current_epoch == 0

    def test_save_checkpoint_without_db(self):
        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", False):
            config = _fast_config()
            trainer = SoGTrainer(config=config, game_config=None)

            with tempfile.TemporaryDirectory() as tmpdir:
                ckpt_path = os.path.join(tmpdir, "test_sog.pt")
                trainer.save_checkpoint(ckpt_path)
                assert os.path.exists(ckpt_path)

    def test_train_without_db(self):
        with patch("src.cfr.sog_trainer._RUN_DB_AVAILABLE", False):
            config = _fast_config(save_interval=0)
            trainer = SoGTrainer(config=config, game_config=None)
            fake = _fake_samples(3)

            with patch.object(trainer, "_generate_episodes", return_value=fake):
                trainer.train(num_epochs=1)

            assert trainer.current_epoch == 1
