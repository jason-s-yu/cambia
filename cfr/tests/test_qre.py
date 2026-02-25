"""
tests/test_qre.py

Tests for QRE (Quantal Response Equilibrium) regularization and N-player
trainer configuration integration.
"""

import math
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adv_and_mask(batch, num_actions, legal_indices=None):
    """Return (advantages, legal_mask) tensors.
    If legal_indices is None, all actions are legal.
    """
    adv = torch.randn(batch, num_actions)
    mask = torch.zeros(batch, num_actions, dtype=torch.bool)
    if legal_indices is None:
        mask[:] = True
    else:
        for b in range(batch):
            mask[b, legal_indices] = True
    return adv, mask


# ---------------------------------------------------------------------------
# TestQREStrategy
# ---------------------------------------------------------------------------

class TestQREStrategy:
    """Unit tests for the qre_strategy() module-level function."""

    def setup_method(self):
        from src.cfr.deep_trainer import qre_strategy
        self.qre_strategy = qre_strategy

    def test_qre_softmax_basic(self):
        """QRE softmax produces valid probability distribution."""
        adv, mask = _make_adv_and_mask(1, 10)
        sigma = self.qre_strategy(adv, mask, lam=0.1)
        assert sigma.shape == (1, 10)
        prob_sum = sigma.sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(1), atol=1e-5), f"sum={prob_sum}"
        assert (sigma >= 0).all()

    def test_qre_softmax_masking(self):
        """Illegal actions get zero probability."""
        legal = [0, 2, 4]
        adv, mask = _make_adv_and_mask(1, 10, legal_indices=legal)
        sigma = self.qre_strategy(adv, mask, lam=0.5)
        illegal = [i for i in range(10) if i not in legal]
        for i in illegal:
            assert sigma[0, i].item() == pytest.approx(0.0, abs=1e-6), \
                f"illegal action {i} has nonzero prob {sigma[0, i].item()}"

    def test_qre_high_lambda_uniform(self):
        """High lambda → near-uniform over legal actions."""
        legal = [0, 1, 2, 3]
        adv, mask = _make_adv_and_mask(1, 10, legal_indices=legal)
        # Force large advantage differences to make the test meaningful
        adv[0, 0] = 100.0
        adv[0, 1] = -100.0
        adv[0, 2] = 50.0
        adv[0, 3] = -50.0
        sigma = self.qre_strategy(adv, mask, lam=1e6)
        # With huge lambda, strategy should be near-uniform over 4 legal actions
        expected = 1.0 / len(legal)
        for i in legal:
            assert sigma[0, i].item() == pytest.approx(expected, abs=0.01), \
                f"action {i}: expected ~{expected}, got {sigma[0, i].item()}"

    def test_qre_low_lambda_greedy(self):
        """Low lambda → concentrates on highest advantage."""
        adv = torch.zeros(1, 10)
        adv[0, 3] = 100.0  # action 3 has dominant advantage
        mask = torch.ones(1, 10, dtype=torch.bool)
        sigma = self.qre_strategy(adv, mask, lam=1e-6)
        # Action 3 should get essentially all the probability
        assert sigma[0, 3].item() > 0.99, f"expected dominant action 3, got {sigma[0, 3].item()}"

    def test_qre_no_nan(self):
        """No NaN even with extreme advantage values."""
        # Simulate extreme values (as arise from early-stage advantage nets)
        adv = torch.full((4, 146), float('-inf'))
        mask = torch.zeros(4, 146, dtype=torch.bool)
        # Give each row at least one legal action
        for b in range(4):
            mask[b, b * 10] = True
            adv[b, b * 10] = float(b * 100)

        sigma = self.qre_strategy(adv, mask, lam=0.1)
        assert not torch.isnan(sigma).any(), "NaN in QRE output with extreme advantages"
        assert not torch.isinf(sigma).any(), "Inf in QRE output with extreme advantages"

    def test_qre_batch_independence(self):
        """Each row's strategy depends only on its own advantages (per-row max, not global max).

        This is the CRITICAL test verifying _1c FATAL bug avoidance:
        - Wrong impl: advantages[legal_mask].max() → global max → cross-row underflow → NaN
        - Correct impl: max(dim=-1, keepdim=True) → per-row max
        """
        # Row 0: modest advantages in [0, 1]
        # Row 1: enormous advantages in [0, 1e6] — if global max is used,
        #         row 0 underflows to NaN
        adv = torch.zeros(2, 5)
        adv[0, 0] = 1.0
        adv[0, 1] = 0.5
        adv[1, 0] = 1e6
        adv[1, 1] = 5e5
        mask = torch.ones(2, 5, dtype=torch.bool)

        sigma = self.qre_strategy(adv, mask, lam=1.0)

        assert not torch.isnan(sigma).any(), "per-row max failed — global max caused NaN"
        # Row 0 should assign highest prob to action 0
        assert sigma[0, 0] > sigma[0, 1], "row 0: action 0 should dominate"
        # Both rows should sum to 1
        assert sigma[0].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert sigma[1].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# TestQREAnnealing
# ---------------------------------------------------------------------------

class TestQREAnnealing:
    """Tests for _get_qre_lambda() annealing schedule on DeepCFRTrainer."""

    def _make_trainer(self, training_step=0, num_iterations=1000):
        """Create a minimal DeepCFRTrainer-like object to test _get_qre_lambda."""
        from src.cfr.deep_trainer import DeepCFRConfig

        class _FakeCFRTrainingConfig:
            num_iterations = 1000

        class _FakeConfig:
            cfr_training = _FakeCFRTrainingConfig()

        cfg = _FakeConfig()
        cfg.cfr_training.num_iterations = num_iterations

        dcfr = DeepCFRConfig(
            qre_lambda_start=0.5,
            qre_lambda_end=0.05,
            qre_anneal_fraction=0.6,
        )

        class _FakeTrainer:
            pass

        t = _FakeTrainer()
        t.dcfr_config = dcfr
        t.config = cfg
        t.training_step = training_step
        # Bind the method
        from src.cfr.deep_trainer import DeepCFRTrainer
        t._get_qre_lambda = DeepCFRTrainer._get_qre_lambda.__get__(t, type(t))
        return t

    def test_lambda_start(self):
        """At iteration 0, lambda equals qre_lambda_start."""
        t = self._make_trainer(training_step=0, num_iterations=1000)
        lam = t._get_qre_lambda()
        assert lam == pytest.approx(0.5, abs=1e-6), f"expected 0.5, got {lam}"

    def test_lambda_end(self):
        """After anneal_fraction * total_iters, lambda equals qre_lambda_end."""
        # anneal_end = int(1000 * 0.6) = 600; at step 600+ should return end=0.05
        t = self._make_trainer(training_step=600, num_iterations=1000)
        lam = t._get_qre_lambda()
        assert lam == pytest.approx(0.05, abs=1e-6), f"expected 0.05, got {lam}"

    def test_lambda_monotonic(self):
        """Lambda decreases monotonically during annealing."""
        prev = None
        for step in range(0, 601, 100):
            t = self._make_trainer(training_step=step, num_iterations=1000)
            lam = t._get_qre_lambda()
            if prev is not None:
                assert lam <= prev + 1e-9, f"lambda increased at step {step}: {prev} -> {lam}"
            prev = lam

    def test_lambda_midpoint(self):
        """At 50% through the annealing period, lambda is midway between start and end."""
        # anneal_end = int(1000 * 0.6) = 600; midpoint = 300
        t = self._make_trainer(training_step=300, num_iterations=1000)
        lam = t._get_qre_lambda()
        expected = 0.5 + (0.05 - 0.5) * (300 / 600)
        assert lam == pytest.approx(expected, abs=1e-6), f"expected {expected}, got {lam}"

    def test_lambda_after_anneal_end_stays_constant(self):
        """Lambda stays at qre_lambda_end after the annealing period ends."""
        lam_at_600 = self._make_trainer(training_step=600, num_iterations=1000)._get_qre_lambda()
        lam_at_900 = self._make_trainer(training_step=900, num_iterations=1000)._get_qre_lambda()
        assert lam_at_600 == pytest.approx(lam_at_900, abs=1e-9)


# ---------------------------------------------------------------------------
# TestNPlayerTrainerConfig
# ---------------------------------------------------------------------------

class TestNPlayerTrainerConfig:
    """Tests for N-player dim overrides in DeepCFRConfig.__post_init__."""

    def test_nplayer_dims(self):
        """num_players > 2 sets input_dim=580, output_dim=452."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=4)
        assert cfg.input_dim == N_PLAYER_INPUT_DIM, \
            f"expected input_dim={N_PLAYER_INPUT_DIM}, got {cfg.input_dim}"
        assert cfg.output_dim == N_PLAYER_NUM_ACTIONS, \
            f"expected output_dim={N_PLAYER_NUM_ACTIONS}, got {cfg.output_dim}"

    def test_2p_dims_unchanged(self):
        """num_players=2 keeps legacy dims (222-dim input, 146 actions)."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=2)
        assert cfg.input_dim == INPUT_DIM, \
            f"expected input_dim={INPUT_DIM}, got {cfg.input_dim}"
        assert cfg.output_dim == NUM_ACTIONS, \
            f"expected output_dim={NUM_ACTIONS}, got {cfg.output_dim}"

    def test_3p_also_uses_nplayer_dims(self):
        """num_players=3 also triggers N-player dims."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=3)
        assert cfg.input_dim == N_PLAYER_INPUT_DIM
        assert cfg.output_dim == N_PLAYER_NUM_ACTIONS

    def test_6p_max_players(self):
        """num_players=6 (max) also uses N-player dims."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=6)
        assert cfg.input_dim == N_PLAYER_INPUT_DIM
        assert cfg.output_dim == N_PLAYER_NUM_ACTIONS

    def test_nplayer_network_config_propagates(self):
        """_get_network_config includes num_players for workers."""
        # We can only test this by inspecting the config object itself,
        # since creating a full trainer requires a real Config object.
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=4)
        assert cfg.num_players == 4
        assert cfg.input_dim == N_PLAYER_INPUT_DIM
        assert cfg.output_dim == N_PLAYER_NUM_ACTIONS


# ---------------------------------------------------------------------------
# TestNPlayerDispatch
# ---------------------------------------------------------------------------

class TestNPlayerDispatch:
    """Tests verifying N-player traversal dispatch configuration."""

    def test_dispatch_selects_nplayer_traversal(self):
        """When num_players > 2, DeepCFRConfig sets dims for N-player traversal."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=4, engine_backend="go", sampling_method="outcome")
        # The worker receives network_config with these dims, which triggers
        # N-player traversal in _deep_traverse_os_go_nplayer
        assert cfg.input_dim == N_PLAYER_INPUT_DIM
        assert cfg.output_dim == N_PLAYER_NUM_ACTIONS
        assert cfg.num_players == 4

    def test_2p_dispatch_uses_legacy_traversal(self):
        """When num_players=2, DeepCFRConfig keeps 2P dims."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.encoding import INPUT_DIM, NUM_ACTIONS

        cfg = DeepCFRConfig(num_players=2, engine_backend="go", sampling_method="outcome")
        assert cfg.input_dim == INPUT_DIM
        assert cfg.output_dim == NUM_ACTIONS
        assert cfg.num_players == 2

    def test_nplayer_config_from_yaml_config(self):
        """from_yaml_config correctly propagates num_players and QRE fields."""
        from src.cfr.deep_trainer import DeepCFRConfig
        from src.constants import N_PLAYER_INPUT_DIM, N_PLAYER_NUM_ACTIONS

        # Build a minimal stub config
        class _FakeDeepCfg:
            hidden_dim = 256
            dropout = 0.1
            learning_rate = 1e-3
            batch_size = 512
            train_steps_per_iteration = 100
            alpha = 1.5
            traversals_per_step = 10
            advantage_buffer_capacity = 1000
            strategy_buffer_capacity = 1000
            save_interval = 10
            device = "cpu"
            sampling_method = "outcome"
            exploration_epsilon = 0.6
            engine_backend = "go"
            es_validation_interval = 10
            es_validation_depth = 10
            es_validation_traversals = 100
            pipeline_training = False
            use_amp = False
            use_compile = False
            num_traversal_threads = 1
            validate_inputs = False
            traversal_depth_limit = 0
            max_tasks_per_child = None
            worker_memory_budget_pct = 0.10
            traversal_method = "outcome"
            value_hidden_dim = 512
            value_learning_rate = 1e-3
            value_buffer_capacity = 1000
            batch_counterfactuals = True
            use_sd_cfr = False
            sd_cfr_max_snapshots = 10
            sd_cfr_snapshot_weighting = "linear"
            num_hidden_layers = 3
            use_residual = True
            use_ema = False
            encoding_mode = "legacy"
            num_players = 4
            qre_lambda_start = 0.5
            qre_lambda_end = 0.05
            qre_anneal_fraction = 0.6

        class _FakeConfig:
            deep_cfr = _FakeDeepCfg()

        dcfr = DeepCFRConfig.from_yaml_config(_FakeConfig())
        assert dcfr.num_players == 4
        assert dcfr.input_dim == N_PLAYER_INPUT_DIM
        assert dcfr.output_dim == N_PLAYER_NUM_ACTIONS
        assert dcfr.qre_lambda_start == pytest.approx(0.5)
        assert dcfr.qre_lambda_end == pytest.approx(0.05)
        assert dcfr.qre_anneal_fraction == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Overflow regression tests
# ---------------------------------------------------------------------------

class TestQREOverflowRegression:
    def test_qre_tiny_lambda_no_overflow(self):
        """Tiny lambda should not cause exp() overflow — no NaN/Inf in output."""
        from src.cfr.deep_trainer import qre_strategy
        adv = torch.tensor([[10.0, 5.0, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        sigma = qre_strategy(adv, mask, 1e-8)
        assert not torch.isnan(sigma).any(), "NaN in sigma"
        assert not torch.isinf(sigma).any(), "Inf in sigma"
        assert sigma[0, 0].item() > 0.99, f"Expected sigma[0,0]>0.99, got {sigma[0,0].item()}"

    def test_qre_large_spread_no_overflow(self):
        """Large advantage spread should not cause overflow."""
        from src.cfr.deep_trainer import qre_strategy
        adv = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        sigma = qre_strategy(adv, mask, 0.01)
        assert not torch.isnan(sigma).any(), "NaN in sigma"
        assert not torch.isinf(sigma).any(), "Inf in sigma"
        assert abs(sigma.sum().item() - 1.0) < 1e-5, f"sigma does not sum to 1: {sigma.sum().item()}"
