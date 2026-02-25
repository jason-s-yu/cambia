"""
Tests for HistoryValueNetwork in src/networks.py

Covers:
- Correct input/output shapes (444 -> batch, 1)
- Forward pass with various batch dims
- Gradient flow
- validate_inputs flag (NaN detection on/off)
- Kaiming weight initialization stats
- Device movement (cpu)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.networks import HistoryValueNetwork
from src.cfr.exceptions import InvalidNetworkInputError

VALUE_INPUT_DIM = INPUT_DIM * 2  # 444


class TestHistoryValueNetworkShape:
    def test_output_shape_single(self):
        net = HistoryValueNetwork()
        x = torch.randn(1, VALUE_INPUT_DIM)
        out = net(x)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    def test_output_shape_batch(self):
        net = HistoryValueNetwork()
        x = torch.randn(32, VALUE_INPUT_DIM)
        out = net(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"

    def test_output_shape_large_batch(self):
        net = HistoryValueNetwork()
        x = torch.randn(256, VALUE_INPUT_DIM)
        out = net(x)
        assert out.shape == (256, 1)

    def test_default_input_dim(self):
        net = HistoryValueNetwork()
        assert net._input_dim == VALUE_INPUT_DIM

    def test_custom_input_dim(self):
        net = HistoryValueNetwork(input_dim=100)
        x = torch.randn(4, 100)
        out = net(x)
        assert out.shape == (4, 1)

    def test_wrong_input_dim_raises(self):
        net = HistoryValueNetwork()
        x = torch.randn(4, INPUT_DIM)  # wrong: should be 444, not 222
        with pytest.raises(InvalidNetworkInputError, match="Invalid features_both shape"):
            net(x)

    def test_1d_input_raises(self):
        net = HistoryValueNetwork()
        x = torch.randn(VALUE_INPUT_DIM)  # missing batch dim
        with pytest.raises(InvalidNetworkInputError):
            net(x)


class TestHistoryValueNetworkForward:
    def test_forward_no_error(self):
        net = HistoryValueNetwork(validate_inputs=False)
        x = torch.randn(8, VALUE_INPUT_DIM)
        out = net(x)
        assert not torch.isnan(out).any()

    def test_output_is_scalar_per_sample(self):
        net = HistoryValueNetwork()
        x = torch.randn(16, VALUE_INPUT_DIM)
        out = net(x)
        # Each sample gets exactly one scalar value
        assert out.shape[-1] == 1

    def test_output_can_be_negative(self):
        """Value estimates are unbounded scalars, not probabilities."""
        net = HistoryValueNetwork()
        results = []
        for _ in range(10):
            x = torch.randn(32, VALUE_INPUT_DIM)
            out = net(x)
            results.append(out.min().item())
        assert any(v < 0 for v in results), "Expected some negative utility values"


class TestHistoryValueNetworkGradients:
    def test_gradient_flow(self):
        net = HistoryValueNetwork()
        x = torch.randn(8, VALUE_INPUT_DIM, requires_grad=False)
        out = net(x)
        loss = out.mean()
        loss.backward()
        # Check all parameters have non-None, non-zero gradients
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestHistoryValueNetworkValidation:
    def test_nan_detection_enabled(self):
        net = HistoryValueNetwork(validate_inputs=True)
        x = torch.full((4, VALUE_INPUT_DIM), float("nan"))
        with pytest.raises(InvalidNetworkInputError, match="NaN"):
            net(x)

    def test_nan_detection_disabled(self):
        net = HistoryValueNetwork(validate_inputs=False)
        x = torch.randn(4, VALUE_INPUT_DIM)
        x[0, 0] = float("nan")
        # Should not raise; NaN propagates silently
        out = net(x)
        assert out is not None

    def test_validate_inputs_default_true(self):
        net = HistoryValueNetwork()
        assert net._validate_inputs is True


class TestHistoryValueNetworkWeightInit:
    def test_kaiming_init_variance(self):
        """Linear layers should have approximately Kaiming normal distribution."""
        net = HistoryValueNetwork()
        # Check that weight std is not trivially small or large
        for module in net.net:
            if isinstance(module, nn.Linear):
                std = module.weight.std().item()
                assert 0.01 < std < 1.0, f"Unexpected weight std={std} for {module}"
                # Bias should be zero
                assert module.bias.abs().max().item() < 1e-7

    def test_hidden_dim_configurable(self):
        net_small = HistoryValueNetwork(hidden_dim=128)
        net_large = HistoryValueNetwork(hidden_dim=1024)
        x = torch.randn(4, VALUE_INPUT_DIM)
        out_small = net_small(x)
        out_large = net_large(x)
        assert out_small.shape == (4, 1)
        assert out_large.shape == (4, 1)


class TestHistoryValueNetworkDevice:
    def test_cpu_forward(self):
        net = HistoryValueNetwork().to("cpu")
        x = torch.randn(4, VALUE_INPUT_DIM)
        out = net(x)
        assert out.device.type == "cpu"

    def test_parameters_on_cpu(self):
        net = HistoryValueNetwork()
        for p in net.parameters():
            assert p.device.type == "cpu"
