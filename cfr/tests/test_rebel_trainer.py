"""
tests/test_rebel_trainer.py

Tests for the ReBeL training loop orchestrator (rebel_trainer.py).

Tests are designed to run without the Go FFI library (libcambia.so) by
mocking self-play episode generation. All network and buffer operations
are tested with synthetic data.
"""

import io
import os
import sys
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.config import DeepCfrConfig
from src.cfr.rebel_trainer import (
    ReBeLTrainer,
    PBS_INPUT_DIM,
    VALUE_OUTPUT_DIM,
    POLICY_OUTPUT_DIM,
    _rebel_batch_worker,
)
from src.cfr.rebel_worker import EpisodeSample

if __name__ == "__main__":
    # __main__ guard: prevents ProcessPoolExecutor spawn workers from re-running tests.
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fast_config(**overrides) -> DeepCfrConfig:
    """Minimal DeepCfrConfig suitable for fast unit tests.

    All rebel_* fields are set explicitly as instance attributes so that this
    function works with both the real config.py DeepCfrConfig and the stub
    injected by conftest.py (which predates rebel_* fields).
    """
    cfg = DeepCfrConfig()
    # Base CFR settings
    cfg.batch_size = 8
    cfg.train_steps_per_iteration = 2
    cfg.save_interval = 0
    cfg.max_tasks_per_child = None
    cfg.worker_memory_budget_pct = 0.10
    cfg.device = "cpu"
    cfg.alpha = 1.5
    # ReBeL settings — must be set explicitly for conftest stub compatibility
    cfg.rebel_value_buffer_capacity = 200
    cfg.rebel_policy_buffer_capacity = 200
    cfg.rebel_value_hidden_dim = 32    # tiny network for speed
    cfg.rebel_policy_hidden_dim = 32
    cfg.rebel_value_learning_rate = 1e-3
    cfg.rebel_policy_learning_rate = 1e-3
    cfg.rebel_games_per_epoch = 2
    cfg.rebel_epochs = 1
    cfg.rebel_subgame_depth = 1
    cfg.rebel_cfr_iterations = 5
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_fake_episode_samples(n: int = 5) -> List[EpisodeSample]:
    """Generate n synthetic EpisodeSamples with correct shapes and dtypes."""
    samples = []
    for _ in range(n):
        features = np.random.rand(PBS_INPUT_DIM).astype(np.float32)
        value_target = np.random.rand(VALUE_OUTPUT_DIM).astype(np.float32)

        mask = np.zeros(POLICY_OUTPUT_DIM, dtype=bool)
        legal = np.random.choice(POLICY_OUTPUT_DIM, size=5, replace=False)
        mask[legal] = True

        policy_target = np.zeros(POLICY_OUTPUT_DIM, dtype=np.float32)
        policy_target[legal] = 1.0 / len(legal)

        samples.append(
            EpisodeSample(
                features=features,
                value_target=value_target,
                policy_target=policy_target,
                action_mask=mask,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Test 1: Trainer initializes with default rebel config
# ---------------------------------------------------------------------------


def test_trainer_initializes():
    """ReBeLTrainer must initialize with networks, buffers, and correct dims."""
    config = make_fast_config()
    trainer = ReBeLTrainer(config=config)

    # Networks exist and are on CPU
    assert isinstance(trainer.value_net, torch.nn.Module)
    assert isinstance(trainer.policy_net, torch.nn.Module)
    assert next(trainer.value_net.parameters()).device == torch.device("cpu")

    # Buffers exist with correct capacities and dims
    assert len(trainer.value_buffer) == 0
    assert len(trainer.policy_buffer) == 0
    assert trainer.value_buffer.capacity == config.rebel_value_buffer_capacity
    assert trainer.policy_buffer.capacity == config.rebel_policy_buffer_capacity
    assert trainer.value_buffer._input_dim == PBS_INPUT_DIM
    assert trainer.value_buffer._target_dim == VALUE_OUTPUT_DIM
    assert trainer.value_buffer._has_mask is False
    assert trainer.policy_buffer._has_mask is True

    # Initial state
    assert trainer.current_iteration == 0
    assert trainer.value_loss_history == []
    assert trainer.policy_loss_history == []


# ---------------------------------------------------------------------------
# Test 2: Single iteration produces valid losses
# ---------------------------------------------------------------------------


def test_single_iteration_produces_valid_losses():
    """train(1) must complete and record finite v_loss and p_loss."""
    config = make_fast_config()
    trainer = ReBeLTrainer(config=config)

    fake_samples = make_fake_episode_samples(10)

    # Patch ProcessPoolExecutor.submit to return fake samples
    mock_future = MagicMock()
    mock_future.result.return_value = fake_samples

    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future
    mock_executor.__enter__ = MagicMock(return_value=mock_executor)
    mock_executor.__exit__ = MagicMock(return_value=False)

    with patch(
        "src.cfr.rebel_trainer.concurrent.futures.ProcessPoolExecutor",
        return_value=mock_executor,
    ):
        trainer.train(num_iterations=1)

    # Must have recorded losses after 1 iteration
    assert len(trainer.value_loss_history) == 1
    assert len(trainer.policy_loss_history) == 1

    iter_v, v_loss = trainer.value_loss_history[0]
    iter_p, p_loss = trainer.policy_loss_history[0]
    assert iter_v == 1
    assert iter_p == 1
    assert np.isfinite(v_loss), f"v_loss is not finite: {v_loss}"
    assert np.isfinite(p_loss), f"p_loss is not finite: {p_loss}"


# ---------------------------------------------------------------------------
# Test 3: Checkpoint save/load roundtrip preserves weights and metadata
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip():
    """save_checkpoint + load_checkpoint must restore identical network weights."""
    config = make_fast_config()
    trainer = ReBeLTrainer(config=config)

    # Insert some fake samples so buffers are non-empty
    fake_samples = make_fake_episode_samples(8)
    trainer._insert_samples(fake_samples, iteration=3)
    trainer.current_iteration = 3
    trainer.value_loss_history = [(1, 0.5), (2, 0.4), (3, 0.3)]
    trainer.policy_loss_history = [(1, 0.6), (2, 0.5), (3, 0.4)]

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "rebel_test.pt")
        trainer.save_checkpoint(ckpt_path)

        assert os.path.exists(ckpt_path), "Checkpoint file not created"
        assert os.path.exists(ckpt_path.replace(".pt", "_rebel_value_buffer.npz")), (
            "Value buffer file not created"
        )
        assert os.path.exists(ckpt_path.replace(".pt", "_rebel_policy_buffer.npz")), (
            "Policy buffer file not created"
        )

        # Capture original weights before loading into a fresh trainer
        orig_value_weights = {
            k: v.clone() for k, v in trainer.value_net.state_dict().items()
        }
        orig_policy_weights = {
            k: v.clone() for k, v in trainer.policy_net.state_dict().items()
        }

        # Load into a fresh trainer
        trainer2 = ReBeLTrainer(config=config)
        trainer2.load_checkpoint(ckpt_path)

        # Iteration restored
        assert trainer2.current_iteration == 3

        # Loss histories restored
        assert trainer2.value_loss_history == trainer.value_loss_history
        assert trainer2.policy_loss_history == trainer.policy_loss_history

        # Buffers loaded
        assert len(trainer2.value_buffer) == len(trainer.value_buffer)
        assert len(trainer2.policy_buffer) == len(trainer.policy_buffer)

        # Network weights are identical
        for key in orig_value_weights:
            torch.testing.assert_close(
                trainer2.value_net.state_dict()[key],
                orig_value_weights[key],
                msg=f"value_net weight mismatch: {key}",
            )
        for key in orig_policy_weights:
            torch.testing.assert_close(
                trainer2.policy_net.state_dict()[key],
                orig_policy_weights[key],
                msg=f"policy_net weight mismatch: {key}",
            )

        # Metadata in checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert ckpt["metadata"]["sampling_method"] == "rebel"
        assert ckpt["metadata"]["iteration"] == 3
        assert ckpt["metadata"]["buffer_sizes"]["value"] > 0
        assert ckpt["metadata"]["buffer_sizes"]["policy"] > 0


# ---------------------------------------------------------------------------
# Test 4: Buffer sizes grow after inserting episode samples
# ---------------------------------------------------------------------------


def test_buffer_sizes_grow():
    """_insert_samples must increase both value and policy buffer sizes."""
    config = make_fast_config()
    trainer = ReBeLTrainer(config=config)

    assert len(trainer.value_buffer) == 0
    assert len(trainer.policy_buffer) == 0

    n_samples = 15
    fake_samples = make_fake_episode_samples(n_samples)
    trainer._insert_samples(fake_samples, iteration=1)

    assert len(trainer.value_buffer) == n_samples
    assert len(trainer.policy_buffer) == n_samples

    # Insert more and verify further growth (up to capacity)
    more_samples = make_fake_episode_samples(10)
    trainer._insert_samples(more_samples, iteration=2)

    # Total might be capped at capacity but at least as large as before
    assert len(trainer.value_buffer) >= n_samples
    assert len(trainer.policy_buffer) >= n_samples


# ---------------------------------------------------------------------------
# Test 5: CLI routing — train rebel command exists
# ---------------------------------------------------------------------------


def test_cli_rebel_command_exists():
    """cli.py must expose a 'train rebel' command that imports ReBeLTrainer."""
    from src.cli import train_app

    # Verify the rebel subcommand is registered
    command_names = [cmd.name for cmd in train_app.registered_commands]
    assert "rebel" in command_names, (
        f"'rebel' command not found in train_app. Found: {command_names}"
    )


# ---------------------------------------------------------------------------
# Test 6: Headless output format contains expected fields
# ---------------------------------------------------------------------------


def test_headless_output_format(capsys):
    """train() must print a line containing [rebel], v_loss, p_loss, v_buf, p_buf."""
    config = make_fast_config()
    trainer = ReBeLTrainer(config=config)

    fake_samples = make_fake_episode_samples(10)

    mock_future = MagicMock()
    mock_future.result.return_value = fake_samples
    mock_executor = MagicMock()
    mock_executor.submit.return_value = mock_future

    with patch(
        "src.cfr.rebel_trainer.concurrent.futures.ProcessPoolExecutor",
        return_value=mock_executor,
    ):
        trainer.train(num_iterations=1)

    captured = capsys.readouterr()
    output = captured.out

    assert "[rebel]" in output, f"Missing '[rebel]' in output: {output!r}"
    assert "iter" in output, f"Missing 'iter' in output: {output!r}"
    assert "v_loss=" in output, f"Missing 'v_loss=' in output: {output!r}"
    assert "p_loss=" in output, f"Missing 'p_loss=' in output: {output!r}"
    assert "v_buf=" in output, f"Missing 'v_buf=' in output: {output!r}"
    assert "p_buf=" in output, f"Missing 'p_buf=' in output: {output!r}"
    assert "self_play=" in output, f"Missing 'self_play=' in output: {output!r}"
    assert "value_train=" in output, f"Missing 'value_train=' in output: {output!r}"
    assert "policy_train=" in output, f"Missing 'policy_train=' in output: {output!r}"
