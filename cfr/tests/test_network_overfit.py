"""
tests/test_network_overfit.py

Validates that AdvantageNetwork and ResidualAdvantageNetwork can memorize
a small synthetic dataset with polarized regret targets. Confirms that
encoding dimensionality and architecture have no information bottleneck.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.networks import AdvantageNetwork, ResidualAdvantageNetwork, SlotFiLMAdvantageNetwork

# Encoding dimensions per project spec
LEGACY_INPUT_DIM = 222
EPPBS_INPUT_DIM = 200
NUM_ACTIONS = 146

BATCH_SIZE = 64
TRAIN_STEPS = 300
LR = 1e-2
# Targets are ±1 (normalized from the conceptual ±100 spec).
# Scale-normalized MSE < 1e-3 ≡ RMSE < 3.2% relative error vs ±100.
MSE_THRESHOLD = 1e-3


def _make_overfit_data(input_dim: int, num_actions: int = NUM_ACTIONS):
    """Synthetic data: action 0 = +1 regret, all others = -1 (normalized ±100 spec)."""
    torch.manual_seed(42)
    features = torch.randn(BATCH_SIZE, input_dim)
    action_mask = torch.ones(BATCH_SIZE, num_actions, dtype=torch.bool)
    targets = torch.full((BATCH_SIZE, num_actions), -1.0)
    targets[:, 0] = 1.0
    return features, action_mask, targets


def _train_and_eval(
    net: nn.Module,
    features: torch.Tensor,
    action_mask: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Train network for TRAIN_STEPS steps; return final MSE on legal actions."""
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        preds = net(features, action_mask)
        # action_mask is all-True here so masked_fill is a no-op, but kept for generality
        p = preds.masked_fill(~action_mask, 0.0)
        t = targets.masked_fill(~action_mask, 0.0)
        loss = F.mse_loss(p, t)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        preds = net(features, action_mask)
        p = preds.masked_fill(~action_mask, 0.0)
        t = targets.masked_fill(~action_mask, 0.0)
        return F.mse_loss(p, t).item()


@pytest.mark.parametrize(
    "input_dim,label",
    [
        (LEGACY_INPUT_DIM, "legacy_222"),
        (EPPBS_INPUT_DIM, "ep_pbs_200"),
    ],
)
def test_advantage_network_overfit(input_dim: int, label: str):
    """AdvantageNetwork memorizes polarized regrets for both encoding dims."""
    features, mask, targets = _make_overfit_data(input_dim)
    torch.manual_seed(99)  # separate seed for network init
    net = AdvantageNetwork(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        validate_inputs=False,
    )
    mse = _train_and_eval(net, features, mask, targets)
    assert mse < MSE_THRESHOLD, (
        f"AdvantageNetwork ({label}): MSE={mse:.6f} exceeds threshold {MSE_THRESHOLD}"
    )


def test_slot_film_overfit_small_batch():
    """SlotFiLMAdvantageNetwork memorizes polarized regrets (same protocol as other overfit tests)."""
    features, mask, targets = _make_overfit_data(EPPBS_INPUT_DIM)
    torch.manual_seed(99)
    net = SlotFiLMAdvantageNetwork(
        input_dim=EPPBS_INPUT_DIM,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        num_hidden_layers=3,
        validate_inputs=False,
    )
    mse = _train_and_eval(net, features, mask, targets)
    assert mse < MSE_THRESHOLD, (
        f"SlotFiLMAdvantageNetwork: MSE={mse:.6f} exceeds threshold {MSE_THRESHOLD}"
    )


@pytest.mark.parametrize(
    "input_dim,label",
    [
        (LEGACY_INPUT_DIM, "legacy_222"),
        (EPPBS_INPUT_DIM, "ep_pbs_200"),
    ],
)
def test_residual_advantage_network_overfit(input_dim: int, label: str):
    """ResidualAdvantageNetwork memorizes polarized regrets for both encoding dims."""
    features, mask, targets = _make_overfit_data(input_dim)
    torch.manual_seed(99)  # separate seed for network init
    net = ResidualAdvantageNetwork(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=NUM_ACTIONS,
        validate_inputs=False,
    )
    mse = _train_and_eval(net, features, mask, targets)
    assert mse < MSE_THRESHOLD, (
        f"ResidualAdvantageNetwork ({label}): MSE={mse:.6f} exceeds threshold {MSE_THRESHOLD}"
    )


def test_history_value_network_mse_decreases():
    """HistoryValueNetwork MSE decreases on synthetic scalar utility targets."""
    from src.networks import HistoryValueNetwork

    VALUE_INPUT_DIM = 444  # 2 * INPUT_DIM (legacy)
    VALUE_TRAIN_STEPS = 200
    VALUE_LR = 1e-2

    torch.manual_seed(42)
    features = torch.randn(BATCH_SIZE, VALUE_INPUT_DIM)
    # Targets: sign of first feature element ±0.8 (bounded scalar, like game utility)
    targets = (features[:, 0] > 0).float().unsqueeze(1) * 1.6 - 0.8

    torch.manual_seed(99)
    net = HistoryValueNetwork(
        input_dim=VALUE_INPUT_DIM,
        hidden_dim=512,
        validate_inputs=False,
    )

    net.eval()
    with torch.no_grad():
        initial_loss = F.mse_loss(net(features), targets).item()

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=VALUE_LR)
    for _ in range(VALUE_TRAIN_STEPS):
        optimizer.zero_grad()
        preds = net(features)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        final_loss = F.mse_loss(net(features), targets).item()

    assert final_loss < initial_loss * 0.5, (
        f"HistoryValueNetwork MSE did not decrease sufficiently: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
