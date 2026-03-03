"""
tests/test_slot_film_integration.py

End-to-end smoke test for the SlotFiLM training pipeline.

This test verifies that DeepCFRTrainer with network_type='slot_film' and
encoding_mode='ep_pbs' can:
  1. Initialize without errors
  2. Run 2 training iterations (traversal is mocked — no libcambia.so needed)
  3. Save a checkpoint
  4. Load the checkpoint and run a forward pass

The traversal step (_run_traversals_batch) is patched to return synthetic
ReservoirSample instances so the test works in CI without the Go engine.

This test is slow (~15-30s on CPU). Mark with CAMBIA_DETERMINISTIC=1 for
reproducible results.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(save_path: str) -> MagicMock:
    """Minimal Config mock sufficient for DeepCFRTrainer."""
    from src.config import CambiaRulesConfig

    cfg = MagicMock()
    cfg.persistence.agent_data_save_path = save_path
    cfg.cfr_training.num_iterations = 2
    cfg.cfr_training.num_workers = 1
    cfg.cambia_rules = CambiaRulesConfig()
    cfg.agent_params.memory_level = 1
    return cfg


def _make_synthetic_samples(n: int, input_dim: int, num_actions: int, iteration: int):
    """Generate n random ReservoirSamples for the given input/output dims."""
    from src.reservoir import ReservoirSample

    samples = []
    for _ in range(n):
        features = np.random.randn(input_dim).astype(np.float32)
        target = np.random.randn(num_actions).astype(np.float32)
        mask = np.zeros(num_actions, dtype=bool)
        # Mark a random subset of 5 actions as legal
        legal_idxs = np.random.choice(num_actions, size=5, replace=False)
        mask[legal_idxs] = True
        samples.append(
            ReservoirSample(
                features=features,
                target=target,
                action_mask=mask,
                iteration=iteration,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestSlotFiLMPipelineSmoke:
    """Integration smoke tests for the SlotFiLM end-to-end training pipeline."""

    def test_slot_film_trainer_init(self):
        """SlotFiLMAdvantageNetwork is constructed with EP-PBS input dim."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.networks import SlotFiLMAdvantageNetwork
        from src.constants import EP_PBS_INPUT_DIM

        dcfr = DeepCFRConfig(
            encoding_mode="ep_pbs",
            network_type="slot_film",
            use_sd_cfr=True,
            device="cpu",
            pipeline_training=False,
        )
        assert dcfr.input_dim == EP_PBS_INPUT_DIM

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "smoke.pt")
            trainer = DeepCFRTrainer(_make_config(save_path), deep_cfr_config=dcfr)

        assert isinstance(trainer.advantage_net, SlotFiLMAdvantageNetwork), (
            f"Expected SlotFiLMAdvantageNetwork, got {type(trainer.advantage_net).__name__}"
        )
        assert trainer.strategy_net is None  # SD-CFR mode

    def test_slot_film_two_iterations_mocked_traversal(self, tmp_path):
        """
        Full training loop: 2 iterations with mocked traversal.

        Patches _run_traversals_batch so no Go engine / libcambia.so is needed.
        Verifies:
          - No crash over 2 iterations
          - Checkpoint file written
          - Loaded checkpoint can do a forward pass with the advantage network
        """
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.constants import EP_PBS_INPUT_DIM
        from src.encoding import NUM_ACTIONS

        dcfr = DeepCFRConfig(
            encoding_mode="ep_pbs",
            network_type="slot_film",
            use_sd_cfr=True,
            device="cpu",
            pipeline_training=False,   # avoid subprocess spawn in tests
            traversals_per_step=50,
            train_steps_per_iteration=5,
            batch_size=32,
            advantage_buffer_capacity=1000,
            strategy_buffer_capacity=1000,
            save_interval=1,
            validate_inputs=False,  # skip NaN checks for speed
            use_ema=True,
        )

        save_path = str(tmp_path / "slot_film_smoke.pt")
        config = _make_config(save_path)

        trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr)

        # Build a fake _run_traversals_batch that returns synthetic samples
        # without touching the game engine.
        def _fake_traversal_batch(
            iteration_offset,
            total_traversals_offset,
            config,
            network_weights,
            network_config,
            traversals_per_step,
            num_workers,
            run_log_dir,
            run_timestamp,
            progress_queue=None,
            archive_queue=None,
        ):
            n_samples = 50
            adv_samples = _make_synthetic_samples(
                n_samples, EP_PBS_INPUT_DIM, NUM_ACTIONS, iteration=iteration_offset
            )
            strat_samples = []  # SD-CFR: no strategy samples
            value_samples = []
            timing = {"min_s": 0.0, "max_s": 0.0, "mean_s": 0.0, "total_s": 0.0, "count": n_samples}
            return adv_samples, strat_samples, value_samples, n_samples, n_samples, timing

        patch_target = "src.cfr.deep_trainer._run_traversals_batch"
        with patch(patch_target, side_effect=_fake_traversal_batch):
            trainer.train(num_training_steps=2)

        # Checkpoint must be written (save_interval=1 → save after every step)
        assert os.path.exists(save_path), (
            f"Expected checkpoint at {save_path} but it was not created."
        )

        # Load and verify forward pass
        from src.cfr.deep_trainer import DeepCFRTrainer as _Trainer2

        trainer2 = _Trainer2(_make_config(save_path), deep_cfr_config=dcfr)
        trainer2.load_checkpoint(save_path)

        # Run a forward pass through the loaded network
        trainer2.advantage_net.eval()
        batch_size = 4
        features = torch.randn(batch_size, EP_PBS_INPUT_DIM)
        mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)
        mask[:, :5] = True

        with torch.inference_mode():
            out = trainer2.advantage_net(features, mask)

        assert out.shape == (batch_size, NUM_ACTIONS), (
            f"Expected output shape ({batch_size}, {NUM_ACTIONS}), got {out.shape}"
        )
        # Legal actions should be finite
        assert torch.isfinite(out[:, :5]).all(), "Legal action outputs contain non-finite values"
        # Illegal actions should be -inf
        assert (out[:, 5:] == float("-inf")).all(), "Illegal actions should be masked to -inf"

    def test_slot_film_checkpoint_encodes_network_type(self, tmp_path):
        """Checkpoint metadata records network_type='slot_film'."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(
            encoding_mode="ep_pbs",
            network_type="slot_film",
            use_sd_cfr=True,
            device="cpu",
            pipeline_training=False,
        )
        save_path = str(tmp_path / "slot_film_meta.pt")
        trainer = DeepCFRTrainer(_make_config(save_path), deep_cfr_config=dcfr)
        trainer.save_checkpoint(save_path)

        ckpt = torch.load(save_path, map_location="cpu", weights_only=True)
        assert ckpt["dcfr_config"]["network_type"] == "slot_film"
        assert ckpt["dcfr_config"]["encoding_mode"] == "ep_pbs"
