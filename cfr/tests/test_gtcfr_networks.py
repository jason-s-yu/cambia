"""
tests/test_gtcfr_networks.py

Unit tests for CVPN (Counterfactual Value-and-Policy Network) and related utilities.
"""

import pytest
import torch

from src.networks import CVPN, build_cvpn, warm_start_cvpn_from_rebel
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from src.encoding import NUM_ACTIONS

VALUE_DIM = 2 * NUM_HAND_TYPES  # 936
BATCH = 4


@pytest.fixture
def cvpn():
    return CVPN(validate_inputs=True)


@pytest.fixture
def pbs_batch():
    return torch.randn(BATCH, PBS_INPUT_DIM)


@pytest.fixture
def mask_batch():
    mask = torch.zeros(BATCH, NUM_ACTIONS, dtype=torch.bool)
    mask[:, :10] = True  # first 10 actions legal
    return mask


def test_cvpn_forward_shapes(cvpn, pbs_batch, mask_batch):
    values, logits = cvpn(pbs_batch, mask_batch)
    assert values.shape == (BATCH, VALUE_DIM), f"Expected ({BATCH}, {VALUE_DIM}), got {values.shape}"
    assert logits.shape == (BATCH, NUM_ACTIONS), f"Expected ({BATCH}, {NUM_ACTIONS}), got {logits.shape}"


def test_cvpn_policy_masking(cvpn, pbs_batch, mask_batch):
    _, logits = cvpn(pbs_batch, mask_batch)
    # Illegal actions (indices 10+) should be -inf
    assert torch.all(logits[:, 10:] == float("-inf")), "Illegal actions should be -inf"
    # Legal actions should be finite
    assert torch.all(torch.isfinite(logits[:, :10])), "Legal actions should be finite"


def test_cvpn_policy_probs_sum_to_one(cvpn, pbs_batch, mask_batch):
    probs = cvpn.policy_probs(pbs_batch, mask_batch)
    assert probs.shape == (BATCH, NUM_ACTIONS)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), f"Probs should sum to 1, got {sums}"


def test_cvpn_param_count(cvpn):
    total = sum(p.numel() for p in cvpn.parameters())
    # Default config (hidden_dim=512, 4 ResBlocks, value_dim=936, policy_dim=146)
    # yields ~3.1M params. Verify within 50% tolerance of expected.
    expected = 3_141_178
    assert int(expected * 0.5) < total < int(expected * 1.5), (
        f"Unexpected param count: {total} (expected ~{expected})"
    )


def test_build_cvpn_factory():
    net = build_cvpn()
    assert isinstance(net, CVPN)
    pbs = torch.randn(2, PBS_INPUT_DIM)
    mask = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
    values, logits = net(pbs, mask)
    assert values.shape == (2, VALUE_DIM)
    assert logits.shape == (2, NUM_ACTIONS)


def test_warm_start_from_rebel(cvpn):
    # Build a fake policy state dict with matching and non-matching keys
    fake_policy = {}
    for name, param in cvpn.named_parameters():
        if "policy_head" in name:
            fake_policy[name] = torch.zeros_like(param)

    # Add a key that won't match
    fake_policy["nonexistent.weight"] = torch.randn(3, 3)

    skipped = warm_start_cvpn_from_rebel(cvpn, fake_policy)

    # nonexistent.weight should be skipped
    assert "nonexistent.weight" in skipped

    # Matching policy head weights should be zero now
    for name, param in cvpn.named_parameters():
        if "policy_head" in name and name in fake_policy:
            assert torch.all(param == 0.0), f"Expected zeros in {name}"
