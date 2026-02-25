"""Tests for SD-CFR: ResidualAdvantageNetwork, snapshot averaging, trainer gating, eval wrapper."""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

def _make_mock_config(agent_data_save_path: str = "") -> MagicMock:
    """Minimal Config mock sufficient for DeepCFRTrainer and agent wrappers."""
    from src.config import CambiaRulesConfig
    cfg = MagicMock()
    cfg.persistence.agent_data_save_path = agent_data_save_path
    cfg.cfr_training.num_iterations = 1
    cfg.cfr_training.num_workers = 1
    cfg.cambia_rules = CambiaRulesConfig()
    cfg.agent_params.memory_level = 1
    return cfg


from src.networks import (
    AdvantageNetwork,
    ResidualAdvantageNetwork,
    build_advantage_network,
    get_strategy_from_advantages,
)
from src.encoding import INPUT_DIM, NUM_ACTIONS


class TestResidualNetwork:
    def test_residual_network_forward_shape(self):
        """Output shape, masking, ~500K param count."""
        net = ResidualAdvantageNetwork(
            input_dim=INPUT_DIM, hidden_dim=256, num_hidden_layers=3,
            output_dim=NUM_ACTIONS, dropout=0.1,
        )
        params = sum(p.numel() for p in net.parameters())
        assert 400_000 < params < 600_000, f"Expected ~500K params, got {params}"

        batch = 4
        features = torch.randn(batch, INPUT_DIM)
        mask = torch.zeros(batch, NUM_ACTIONS, dtype=torch.bool)
        mask[:, :10] = True  # Only first 10 actions legal

        out = net(features, mask)
        assert out.shape == (batch, NUM_ACTIONS)
        # Illegal actions should be -inf
        assert (out[:, 10:] == float("-inf")).all()
        # Legal actions should be finite
        assert torch.isfinite(out[:, :10]).all()

    def test_residual_network_gradient_flow(self):
        """Gradient reaches first layer through skip connections."""
        net = ResidualAdvantageNetwork(
            input_dim=INPUT_DIM, hidden_dim=256, num_hidden_layers=3,
            output_dim=NUM_ACTIONS,
        )
        features = torch.randn(2, INPUT_DIM, requires_grad=True)
        mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)

        out = net(features, mask)
        loss = out.sum()
        loss.backward()

        # Check gradient flows to input_proj (first layer)
        first_layer = net.input_proj[0]  # Linear layer
        assert first_layer.weight.grad is not None
        assert first_layer.weight.grad.abs().sum() > 0

    def test_build_advantage_network_factory(self):
        """Factory returns correct class based on flags."""
        net_std = build_advantage_network(use_residual=False)
        assert isinstance(net_std, AdvantageNetwork)

        net_res = build_advantage_network(use_residual=True, num_hidden_layers=3)
        assert isinstance(net_res, ResidualAdvantageNetwork)


class TestSnapshotMechanics:
    def test_snapshot_reservoir_sampling(self):
        """300 inserts into max=200 store → len stays 200."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.config import Config

        config = Config()
        dcfr_config = DeepCFRConfig(
            use_sd_cfr=True, sd_cfr_max_snapshots=200, device="cpu",
            use_residual=False,  # smaller net for speed
        )
        trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr_config)

        for i in range(300):
            trainer.training_step = i
            trainer._take_advantage_snapshot()

        assert len(trainer._sd_snapshots) == 200
        assert len(trainer._sd_snapshot_iterations) == 200

    def test_snapshot_weighting_linear_vs_uniform(self):
        """Different weighting produces different averaged strategies."""
        # Create two fake snapshots with different weights
        net1 = build_advantage_network(use_residual=False)
        net2 = build_advantage_network(use_residual=False)

        # Force different outputs
        with torch.no_grad():
            for p in net2.parameters():
                p.mul_(2.0)

        features = torch.randn(1, INPUT_DIM)
        mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)

        with torch.inference_mode():
            adv1 = net1(features, mask)
            adv2 = net2(features, mask)
            strat1 = get_strategy_from_advantages(adv1, mask)
            strat2 = get_strategy_from_advantages(adv2, mask)

        # Linear weighting (iter 1 vs iter 100): heavily weights iter 100
        linear_avg = (2.0 * strat1 + 101.0 * strat2) / 103.0
        # Uniform weighting
        uniform_avg = (strat1 + strat2) / 2.0

        assert not torch.allclose(linear_avg, uniform_avg, atol=1e-6)


class TestTrainerSDCFR:
    def test_trainer_sd_cfr_skips_strategy(self):
        """strategy_net/buffer/optimizer are None when use_sd_cfr=True."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.config import Config

        config = Config()
        trainer = DeepCFRTrainer(
            config, deep_cfr_config=DeepCFRConfig(use_sd_cfr=True, device="cpu")
        )
        assert trainer.strategy_net is None
        assert trainer.strategy_buffer is None
        assert trainer.strategy_optimizer is None
        assert trainer.advantage_net is not None
        assert trainer.advantage_optimizer is not None

    def test_trainer_sd_cfr_takes_snapshots(self):
        """Snapshot count grows each iteration."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.config import Config

        config = Config()
        trainer = DeepCFRTrainer(
            config, deep_cfr_config=DeepCFRConfig(use_sd_cfr=True, device="cpu", use_residual=False)
        )
        assert len(trainer._sd_snapshots) == 0

        for i in range(5):
            trainer.training_step = i + 1
            trainer._take_advantage_snapshot()

        assert len(trainer._sd_snapshots) == 5
        assert trainer._sd_snapshot_iterations == [1, 2, 3, 4, 5]

    def test_checkpoint_roundtrip_sd_cfr(self):
        """Save+load preserves snapshots, advantage_net, training state."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(use_sd_cfr=True, device="cpu", use_residual=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.pt")
            config = _make_mock_config(path)
            trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr)

            # Add some snapshots
            for i in range(5):
                trainer.training_step = i + 1
                trainer._take_advantage_snapshot()
            trainer.total_traversals = 5000

            trainer.save_checkpoint(path)

            # Verify files exist
            base = os.path.splitext(path)[0]
            assert os.path.exists(path)
            assert os.path.exists(f"{base}_sd_snapshots.pt")
            assert os.path.exists(f"{base}_advantage_buffer.npz")

            # Load into fresh trainer
            config2 = _make_mock_config(path)
            trainer2 = DeepCFRTrainer(config2, deep_cfr_config=dcfr)
            trainer2.load_checkpoint(path)

            assert len(trainer2._sd_snapshots) == 5
            assert trainer2._sd_snapshot_iterations == [1, 2, 3, 4, 5]
            assert trainer2.total_traversals == 5000
            assert trainer2.training_step == 5

    def test_checkpoint_backward_compat_old_to_new(self):
        """SD-CFR trainer loads old checkpoint (with strategy_net key)."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        old_dcfr = DeepCFRConfig(use_sd_cfr=False, use_residual=False, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old_checkpoint.pt")
            config = _make_mock_config(path)

            # Create old-style checkpoint (non-SD-CFR)
            old_trainer = DeepCFRTrainer(config, deep_cfr_config=old_dcfr)
            old_trainer.training_step = 100
            old_trainer.total_traversals = 10000
            old_trainer.save_checkpoint(path)

            # Load into SD-CFR trainer
            new_dcfr = DeepCFRConfig(use_sd_cfr=True, use_residual=False, device="cpu")
            new_trainer = DeepCFRTrainer(_make_mock_config(path), deep_cfr_config=new_dcfr)
            new_trainer.load_checkpoint(path)

            assert new_trainer.training_step == 100
            assert new_trainer.strategy_net is None
            assert len(new_trainer._sd_snapshots) == 0

    def test_checkpoint_backward_compat_new_to_old(self):
        """Non-SD-CFR trainer loads SD-CFR checkpoint (no strategy_net key)."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        sd_dcfr = DeepCFRConfig(use_sd_cfr=True, use_residual=False, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sd_checkpoint.pt")
            config = _make_mock_config(path)

            # Create SD-CFR checkpoint
            sd_trainer = DeepCFRTrainer(config, deep_cfr_config=sd_dcfr)
            sd_trainer.training_step = 50
            sd_trainer.save_checkpoint(path)

            # Load into non-SD-CFR trainer (has strategy_net)
            old_dcfr = DeepCFRConfig(use_sd_cfr=False, use_residual=False, device="cpu")
            old_trainer = DeepCFRTrainer(_make_mock_config(path), deep_cfr_config=old_dcfr)
            old_trainer.load_checkpoint(path)

            assert old_trainer.training_step == 50
            assert old_trainer.strategy_net is not None


class TestSDCFRAgentWrapper:
    def test_sdcfr_agent_wrapper_inference(self):
        """SDCFRAgentWrapper produces valid action distributions."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(use_sd_cfr=True, device="cpu", use_residual=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sd_test.pt")
            mock_config = _make_mock_config(path)
            trainer = DeepCFRTrainer(mock_config, deep_cfr_config=dcfr)

            # Create snapshots
            for i in range(3):
                trainer.training_step = i + 1
                trainer._take_advantage_snapshot()

            trainer.save_checkpoint(path)

            from src.evaluate_agents import SDCFRAgentWrapper
            # Disable EMA to test snapshot path (EMA was not updated separately)
            wrapper = SDCFRAgentWrapper(
                player_id=0, config=mock_config, checkpoint_path=path, device="cpu",
                use_ema=False,
            )
            assert len(wrapper._snapshot_nets) == 3

            # Test inference with fake game state (use default rules)
            from src.game.engine import CambiaGameState
            game = CambiaGameState()
            wrapper.initialize_state(game)

            legal_actions = game.get_legal_actions()
            action = wrapper.choose_action(game, legal_actions)
            assert action in legal_actions


class TestEMAWeights:
    def test_ema_math_correctness(self):
        """EMA formula matches manual weighted average with alpha=1.5 over 10 steps."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig
        from src.config import Config

        config = Config()
        dcfr_config = DeepCFRConfig(
            use_sd_cfr=True, use_ema=True, device="cpu", use_residual=False, alpha=1.5,
        )
        trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr_config)

        # Manually compute the expected weighted average
        param_key = list(trainer.advantage_net.state_dict().keys())[0]
        snapshots = []
        weights = []

        for step in range(10):
            trainer.training_step = step
            # Perturb weights so each snapshot is distinct
            with torch.no_grad():
                for p in trainer.advantage_net.parameters():
                    p.add_(torch.ones_like(p) * 0.1)
            trainer._take_advantage_snapshot()
            trainer._update_ema()
            w = float((step + 1) ** 1.5)
            weights.append(w)
            snapshots.append(trainer.advantage_net.state_dict()[param_key].cpu().numpy().copy())

        # Compute manual weighted average
        total_w = sum(weights)
        manual_avg = sum(w * s for w, s in zip(weights, snapshots)) / total_w

        assert trainer._ema_state_dict is not None
        np.testing.assert_allclose(
            trainer._ema_state_dict[param_key], manual_avg, atol=1e-5,
        )

    def test_ema_checkpoint_roundtrip(self):
        """Save and reload EMA state yields exact match."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(use_sd_cfr=True, use_ema=True, device="cpu", use_residual=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ema_ckpt.pt")
            config = _make_mock_config(path)
            trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr)

            for i in range(5):
                trainer.training_step = i
                trainer._take_advantage_snapshot()
                trainer._update_ema()

            trainer.save_checkpoint(path)

            base = os.path.splitext(path)[0]
            assert os.path.exists(f"{base}_ema.pt"), "EMA file should be saved"

            # Load into fresh trainer
            config2 = _make_mock_config(path)
            trainer2 = DeepCFRTrainer(config2, deep_cfr_config=dcfr)
            trainer2.load_checkpoint(path)

            assert trainer2._ema_state_dict is not None
            assert abs(trainer2._ema_weight_sum - trainer._ema_weight_sum) < 1e-6
            for key in trainer._ema_state_dict:
                np.testing.assert_array_equal(
                    trainer2._ema_state_dict[key], trainer._ema_state_dict[key]
                )

    def test_ema_inference_parity(self):
        """EMA action distribution is close to full snapshot average (within noise)."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(use_sd_cfr=True, use_ema=True, device="cpu", use_residual=False, alpha=1.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ema_parity.pt")
            config = _make_mock_config(path)
            trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr)

            for i in range(5):
                trainer.training_step = i
                with torch.no_grad():
                    for p in trainer.advantage_net.parameters():
                        p.add_(torch.ones_like(p) * 0.05)
                trainer._take_advantage_snapshot()
                trainer._update_ema()

            trainer.save_checkpoint(path)

            from src.evaluate_agents import SDCFRAgentWrapper
            from src.networks import get_strategy_from_advantages
            from src.encoding import INPUT_DIM, NUM_ACTIONS

            # Load with EMA
            ema_wrapper = SDCFRAgentWrapper(
                player_id=0, config=config, checkpoint_path=path, device="cpu", use_ema=True,
            )
            # Load without EMA (snapshot averaging)
            snap_wrapper = SDCFRAgentWrapper(
                player_id=0, config=config, checkpoint_path=path, device="cpu", use_ema=False,
            )

            assert ema_wrapper._ema_net is not None
            assert len(snap_wrapper._snapshot_nets) > 0

            # Compare strategies on a fixed input
            features = torch.randn(1, INPUT_DIM)
            mask = torch.ones(1, NUM_ACTIONS, dtype=torch.bool)

            with torch.inference_mode():
                ema_adv = ema_wrapper._ema_net(features, mask)
                ema_strat = get_strategy_from_advantages(ema_adv, mask).squeeze(0).numpy()

                snap_strat = torch.zeros(1, NUM_ACTIONS)
                total_w = 0.0
                for i, net in enumerate(snap_wrapper._snapshot_nets):
                    adv = net(features, mask)
                    s = get_strategy_from_advantages(adv, mask)
                    w = float(snap_wrapper._snapshot_iterations[i] + 1)
                    snap_strat += w * s
                    total_w += w
                snap_strat = (snap_strat / total_w).squeeze(0).numpy()

            # EMA averages weights (linear), snapshot averaging averages strategies (nonlinear).
            # These agree to first order; larger differences arise from network nonlinearity.
            np.testing.assert_allclose(ema_strat, snap_strat, atol=5e-3)

    def test_ema_disabled(self):
        """use_ema=False: no EMA file saved, _ema_state_dict stays None."""
        from src.cfr.deep_trainer import DeepCFRTrainer, DeepCFRConfig

        dcfr = DeepCFRConfig(use_sd_cfr=True, use_ema=False, device="cpu", use_residual=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "no_ema.pt")
            config = _make_mock_config(path)
            trainer = DeepCFRTrainer(config, deep_cfr_config=dcfr)

            for i in range(3):
                trainer.training_step = i
                trainer._take_advantage_snapshot()
                trainer._update_ema()

            assert trainer._ema_state_dict is None
            assert trainer._ema_weight_sum == 0.0

            trainer.save_checkpoint(path)

            base = os.path.splitext(path)[0]
            assert not os.path.exists(f"{base}_ema.pt"), "EMA file should NOT be saved when disabled"
