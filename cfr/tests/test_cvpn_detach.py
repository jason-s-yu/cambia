"""Tests for CVPN detach_policy_grad feature."""

import torch


def test_detach_policy_grad_stops_trunk_gradients():
    """With detach=True, policy loss backward should NOT produce gradients in trunk."""
    from src.networks import build_cvpn

    cvpn = build_cvpn(detach_policy_grad=True, validate_inputs=False)
    pbs = torch.randn(2, 956)
    mask = torch.ones(2, 146, dtype=torch.bool)

    values, logits = cvpn(pbs, mask)
    # Only backprop through policy
    policy_loss = logits.sum()
    policy_loss.backward()

    # Trunk params should have NO gradient (detached)
    for name, p in cvpn.trunk.named_parameters():
        assert p.grad is None or (p.grad == 0).all(), (
            f"trunk param {name} has gradient with detach=True"
        )

    # Policy head params SHOULD have gradient
    for name, p in cvpn.policy_head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"policy_head param {name} missing gradient"
        )


def test_no_detach_allows_trunk_gradients():
    """With detach=False, policy loss backward SHOULD produce gradients in trunk."""
    from src.networks import build_cvpn

    cvpn = build_cvpn(detach_policy_grad=False, validate_inputs=False)
    pbs = torch.randn(2, 956)
    mask = torch.ones(2, 146, dtype=torch.bool)

    values, logits = cvpn(pbs, mask)
    policy_loss = logits.sum()
    policy_loss.backward()

    trunk_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cvpn.trunk.parameters()
    )
    assert trunk_has_grad, "trunk should have gradients when detach=False"


def test_detach_value_head_still_trains_trunk():
    """With detach=True, value loss backward SHOULD still produce gradients in trunk."""
    from src.networks import build_cvpn

    cvpn = build_cvpn(detach_policy_grad=True, validate_inputs=False)
    pbs = torch.randn(2, 956)
    mask = torch.ones(2, 146, dtype=torch.bool)

    values, logits = cvpn(pbs, mask)
    # Only backprop through value head
    value_loss = values.sum()
    value_loss.backward()

    trunk_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in cvpn.trunk.parameters()
    )
    assert trunk_has_grad, "trunk should have gradients from value loss even with detach=True"
