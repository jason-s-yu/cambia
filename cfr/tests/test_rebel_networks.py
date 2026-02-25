"""tests/test_rebel_networks.py

Tests for PBSValueNetwork, PBSPolicyNetwork, and ReBeL config fields.
"""

import io
import textwrap
import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.networks import PBSValueNetwork, PBSPolicyNetwork
from src.config import DeepCfrConfig, load_config
from src.cfr.exceptions import InvalidNetworkInputError


# ---------------------------------------------------------------------------
# PBSValueNetwork
# ---------------------------------------------------------------------------


def test_pbs_value_network_output_shape():
    net = PBSValueNetwork()
    x = torch.randn(4, 956)
    out = net(x)
    assert out.shape == (4, 936), f"Expected (4, 936), got {tuple(out.shape)}"


def test_pbs_value_network_gradient_flow():
    net = PBSValueNetwork()
    x = torch.randn(4, 956, requires_grad=True)
    out = net(x)
    loss = out.sum()
    loss.backward()
    # All Linear parameters should have gradients
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
    assert x.grad is not None, "Input has no gradient"


def test_pbs_value_network_no_nan():
    net = PBSValueNetwork()
    x = torch.randn(8, 956)
    out = net(x)
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_pbs_value_network_param_count():
    net = PBSValueNetwork()
    total = sum(p.numel() for p in net.parameters())
    assert 2_500_000 <= total <= 5_000_000, (
        f"Expected parameter count in [2.5M, 5M], got {total:,}"
    )


# ---------------------------------------------------------------------------
# PBSPolicyNetwork
# ---------------------------------------------------------------------------


def test_pbs_policy_network_output_shape():
    net = PBSPolicyNetwork()
    x = torch.randn(4, 956)
    mask = torch.ones(4, 146, dtype=torch.bool)
    out = net(x, mask)
    assert out.shape == (4, 146), f"Expected (4, 146), got {tuple(out.shape)}"


def test_pbs_policy_network_masked_softmax():
    net = PBSPolicyNetwork()
    batch = 4
    x = torch.randn(batch, 956)
    # Only first 10 actions legal
    mask = torch.zeros(batch, 146, dtype=torch.bool)
    mask[:, :10] = True
    out = net(x, mask)
    # Each row should sum to ~1
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(batch), atol=1e-5), (
        f"Row sums not close to 1: {row_sums}"
    )
    # Masked-out positions should be ~0
    masked_vals = out[:, 10:]
    assert (masked_vals < 1e-6).all(), (
        f"Masked positions are non-zero: max={masked_vals.max().item()}"
    )


def test_pbs_policy_network_gradient_flow():
    net = PBSPolicyNetwork()
    x = torch.randn(4, 956, requires_grad=True)
    mask = torch.ones(4, 146, dtype=torch.bool)
    out = net(x, mask)
    loss = out.sum()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
    assert x.grad is not None, "Input has no gradient"


def test_pbs_policy_network_param_count():
    net = PBSPolicyNetwork()
    total = sum(p.numel() for p in net.parameters())
    assert 500_000 <= total <= 1_000_000, (
        f"Expected parameter count in [0.5M, 1M], got {total:,}"
    )


# ---------------------------------------------------------------------------
# validate_inputs
# ---------------------------------------------------------------------------


def test_validate_inputs_catches_wrong_dims():
    value_net = PBSValueNetwork(validate_inputs=True)
    policy_net = PBSPolicyNetwork(validate_inputs=True)

    # Wrong input dim for value network
    with pytest.raises(InvalidNetworkInputError):
        value_net(torch.randn(2, 100))

    # Wrong input dim for policy network
    mask = torch.ones(2, 146, dtype=torch.bool)
    with pytest.raises(InvalidNetworkInputError):
        policy_net(torch.randn(2, 100), mask)

    # NaN input for value network
    nan_input = torch.full((2, 956), float("nan"))
    with pytest.raises(InvalidNetworkInputError):
        value_net(nan_input)

    # NaN input for policy network
    nan_input_p = torch.full((2, 956), float("nan"))
    with pytest.raises(InvalidNetworkInputError):
        policy_net(nan_input_p, mask)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_rebel_config_defaults():
    cfg = DeepCfrConfig()
    assert cfg.rebel_subgame_depth == 4
    assert cfg.rebel_cfr_iterations == 200
    assert cfg.rebel_value_hidden_dim == 1024
    assert cfg.rebel_policy_hidden_dim == 512
    assert cfg.rebel_value_learning_rate == 1e-3
    assert cfg.rebel_policy_learning_rate == 1e-3
    assert cfg.rebel_value_buffer_capacity == 500_000
    assert cfg.rebel_policy_buffer_capacity == 500_000
    assert cfg.rebel_games_per_epoch == 100
    assert cfg.rebel_epochs == 500


def test_rebel_config_from_yaml(tmp_path):
    yaml_content = textwrap.dedent("""\
        deep_cfr:
          rebel_subgame_depth: 6
          rebel_cfr_iterations: 100
          rebel_value_hidden_dim: 512
          rebel_policy_hidden_dim: 256
          rebel_value_learning_rate: 0.0005
          rebel_policy_learning_rate: 0.0002
          rebel_value_buffer_capacity: 200000
          rebel_policy_buffer_capacity: 300000
          rebel_games_per_epoch: 50
          rebel_epochs: 250
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(str(config_file))
    assert cfg is not None
    dc = cfg.deep_cfr
    assert dc.rebel_subgame_depth == 6
    assert dc.rebel_cfr_iterations == 100
    assert dc.rebel_value_hidden_dim == 512
    assert dc.rebel_policy_hidden_dim == 256
    assert abs(dc.rebel_value_learning_rate - 0.0005) < 1e-9
    assert abs(dc.rebel_policy_learning_rate - 0.0002) < 1e-9
    assert dc.rebel_value_buffer_capacity == 200_000
    assert dc.rebel_policy_buffer_capacity == 300_000
    assert dc.rebel_games_per_epoch == 50
    assert dc.rebel_epochs == 250
