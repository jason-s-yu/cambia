"""Tier 1-2 tests for the DESCA networks (v3.1 Phase 1).

Covers:
- Tier 1 shape: forward output shapes for batched inputs across all three nets.
- Tier 1 mask: AvgStrategyNetwork zero-probability illegal actions, sums to 1.
- Tier 1 round-trip: state_dict save/load yields identical forward outputs.
- Tier 1 determinism: V_omni regressed onto a fixed scalar payoff converges to
  match the target within 1e-3 on a deterministic mini-batch (proxy for the
  "deterministic-play game" Tier 1 acceptance criterion in the contract).
- Tier 2 information flow: V_omni vs V_fair produce different outputs on a
  batch with non-trivial hidden-card structure.
- Tier 2 param count: total params sit in the architecture-derived band
  (raised vs the contract's ~3M estimate; see test docstring for rationale).
- Security / gradient leakage: backward through V_omni does not produce
  gradients on AvgStrategyNetwork parameters.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.action_abstraction import NUM_ABSTRACT_ACTIONS_2P
from src.constants import EP_PBS_V2_INPUT_DIM
from src.desca_networks import (
    DEFAULT_OMNISCIENT_DIM_2P,
    AvgStrategyNetwork,
    HistoryValueNetwork,
    RegretNetwork,
)


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def _rand_input(batch: int, dim: int = EP_PBS_V2_INPUT_DIM) -> torch.Tensor:
    return torch.randn(batch, dim)


def _rand_mask(batch: int, num_actions: int = NUM_ABSTRACT_ACTIONS_2P) -> torch.Tensor:
    """Boolean mask with at least 2 legal actions per row."""
    mask = torch.rand(batch, num_actions) > 0.5
    for i in range(batch):
        legal = int(mask[i].sum().item())
        if legal < 2:
            idx = torch.randperm(num_actions)[: 2 - legal + int(legal == 0)]
            mask[i, idx] = True
    return mask


# ----- Tier 1: shape -----


def test_regret_forward_shape():
    net = RegretNetwork()
    out = net(_rand_input(8))
    assert out.shape == (8, NUM_ABSTRACT_ACTIONS_2P)
    assert out.dtype == torch.float32


def test_avg_strategy_forward_shape_and_normalization():
    net = AvgStrategyNetwork()
    batch = 16
    x = _rand_input(batch)
    mask = _rand_mask(batch)
    probs = net(x, mask)
    assert probs.shape == (batch, NUM_ABSTRACT_ACTIONS_2P)
    illegal = ~mask
    assert torch.all(probs[illegal] == 0.0), "illegal actions must be zero prob"
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_history_value_forward_shapes_omni_and_fair():
    net = HistoryValueNetwork()
    batch = 4
    x = _rand_input(batch)
    fair_out = net(x, hidden_cards=None)
    assert fair_out.shape == (batch, 1)
    omni = torch.zeros(batch, DEFAULT_OMNISCIENT_DIM_2P)
    omni[:, 0] = 1.0
    omni_out = net(x, hidden_cards=omni)
    assert omni_out.shape == (batch, 1)


def test_history_value_rejects_wrong_omni_shape():
    net = HistoryValueNetwork()
    x = _rand_input(2)
    with pytest.raises(ValueError):
        net(x, hidden_cards=torch.zeros(2, DEFAULT_OMNISCIENT_DIM_2P + 1))
    with pytest.raises(ValueError):
        net(x, hidden_cards=torch.zeros(3, DEFAULT_OMNISCIENT_DIM_2P))


def test_avg_strategy_rejects_wrong_mask():
    net = AvgStrategyNetwork()
    x = _rand_input(2)
    with pytest.raises(ValueError):
        net(x, torch.ones(2, NUM_ABSTRACT_ACTIONS_2P + 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        net(x, torch.ones(3, NUM_ABSTRACT_ACTIONS_2P, dtype=torch.bool))


def test_regret_rejects_wrong_input_dim():
    net = RegretNetwork()
    with pytest.raises(ValueError):
        net(torch.randn(4, EP_PBS_V2_INPUT_DIM - 1))


# ----- Tier 1: checkpoint round-trip -----


def _state_dict_round_trip(net: torch.nn.Module) -> torch.nn.Module:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    loaded_state = torch.load(buf)
    clone = type(net)()
    clone.load_state_dict(loaded_state)
    clone.eval()
    return clone


def test_regret_checkpoint_round_trip_identity():
    net = RegretNetwork().eval()
    clone = _state_dict_round_trip(net)
    x = _rand_input(5)
    with torch.no_grad():
        a = net(x)
        b = clone(x)
    assert torch.allclose(a, b, atol=0.0)


def test_avg_strategy_checkpoint_round_trip_identity():
    net = AvgStrategyNetwork().eval()
    clone = _state_dict_round_trip(net)
    x = _rand_input(5)
    mask = _rand_mask(5)
    with torch.no_grad():
        a = net(x, mask)
        b = clone(x, mask)
    assert torch.allclose(a, b, atol=0.0)


def test_history_value_checkpoint_round_trip_identity():
    net = HistoryValueNetwork().eval()
    clone = _state_dict_round_trip(net)
    x = _rand_input(5)
    omni = torch.randn(5, DEFAULT_OMNISCIENT_DIM_2P)
    with torch.no_grad():
        a_omni = net(x, omni)
        b_omni = clone(x, omni)
        a_fair = net(x, None)
        b_fair = clone(x, None)
    assert torch.allclose(a_omni, b_omni, atol=0.0)
    assert torch.allclose(a_fair, b_fair, atol=0.0)


# ----- Tier 1: deterministic regression of V_omni to a fixed payoff -----


@pytest.mark.slow
def test_history_value_regresses_to_fixed_terminal_payoff():
    """Train V_omni on a tiny fixed batch toward a known scalar payoff.

    Stand-in for the contract's "deterministic-play game" Tier 1 criterion: the
    net should drive its output to within 1e-3 of the target after enough SGD
    steps on a single fixed minibatch. This validates that the head wiring,
    loss path, and trunk are numerically capable of fitting a fixed regression
    target (i.e., the network can reproduce a known terminal payoff when given
    the same features deterministically).
    """
    torch.manual_seed(7)
    net = HistoryValueNetwork(dropout=0.0)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    batch = 8
    x = torch.randn(batch, EP_PBS_V2_INPUT_DIM)
    omni = torch.zeros(batch, DEFAULT_OMNISCIENT_DIM_2P)
    target_value = 0.5
    target = torch.full((batch, 1), target_value)

    for _ in range(3000):
        optim.zero_grad()
        pred = net(x, omni)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optim.step()

    net.eval()
    with torch.no_grad():
        final = net(x, omni)
    assert torch.allclose(
        final, target, atol=1e-3
    ), f"V_omni did not converge: max diff {(final - target).abs().max().item():.6f}"


# ----- Tier 2: information flow -----


def test_v_omni_differs_from_v_fair_on_nontrivial_hidden_cards():
    """V_omni and V_fair must respond differently to non-trivial omniscient input.

    This guards against silent regression where the trunk ignores the
    omniscient channel (e.g. dead-ReLU paths or unwired concat).
    """
    torch.manual_seed(3)
    net = HistoryValueNetwork()
    batch = 100
    x = torch.randn(batch, EP_PBS_V2_INPUT_DIM)
    omni = torch.zeros(batch, DEFAULT_OMNISCIENT_DIM_2P)
    rng = torch.Generator().manual_seed(11)
    for i in range(batch):
        for slot in range(12):
            bucket = int(torch.randint(0, 10, (1,), generator=rng).item())
            omni[i, slot * 10 + bucket] = 1.0

    net.eval()
    with torch.no_grad():
        v_omni = net(x, omni)
        v_fair = net(x, None)
    diff = (v_omni - v_fair).abs().mean().item()
    assert diff > 0.01, (
        f"V_omni and V_fair produced near-identical outputs on non-trivial "
        f"hidden cards (mean abs diff {diff:.6g}); omniscient channel may be "
        "wired but inert"
    )


# ----- Tier 2: parameter count band -----


def test_total_param_count_matches_architecture_spec():
    """Validate the per-net and total parameter counts.

    The contract / spec stated "~1M trunk per net, ~3M total across nets" as a
    rough estimate. The literal architecture (`Linear(input_dim, 512) ->
    LayerNorm -> SiLU -> 3 x _ResBlock(512)`) actually carries more params; the
    architecture is the instruction, the estimate was an under-count. Lead
    approved widening the band to match the architecture (option 1). The
    spec / contract docs will be updated separately to reflect the ~5.2M total.

    Computed breakdown (input_dim=257, hidden_dim=512, num_blocks=3, 2P):

        Per trunk (input 257):                              ~1,715,200
            input_proj   Linear(257, 512) + bias               132,096
            input_norm   LayerNorm(512)                          1,024
            3 x _ResBlock(512) (each ~527,360)               1,582,080

        RegretNetwork (input 257):                          ~1,731,616
            trunk                                            1,715,200
            head Linear(512, 32) + bias                         16,416

        AvgStrategyNetwork (input 257):                     ~1,731,616
            trunk                                            1,715,200
            head Linear(512, 32) + bias                         16,416

        HistoryValueNetwork (input 257 + omni 120 = 377):   ~1,777,153
            input_proj Linear(377, 512) + bias                 193,536
            input_norm LayerNorm(512)                            1,024
            3 x _ResBlock(512)                               1,582,080
            head Linear(512, 1) + bias                             513

        Grand total across the three networks:              ~5,240,385
    """
    r = RegretNetwork()
    a = AvgStrategyNetwork()
    v = HistoryValueNetwork()
    n_r = sum(p.numel() for p in r.parameters())
    n_a = sum(p.numel() for p in a.parameters())
    n_v = sum(p.numel() for p in v.parameters())
    total = n_r + n_a + n_v

    assert (
        1_500_000 <= n_r <= 1_900_000
    ), f"RegretNetwork param count out of band: {n_r:,}"
    assert (
        1_500_000 <= n_a <= 1_900_000
    ), f"AvgStrategyNetwork param count out of band: {n_a:,}"
    assert (
        1_500_000 <= n_v <= 1_950_000
    ), f"HistoryValueNetwork param count out of band: {n_v:,}"
    assert 4_500_000 <= total <= 6_000_000, f"Total param count out of band: {total:,}"


# ----- Security: gradient leakage -----


def test_history_value_backward_does_not_leak_to_avg_strategy_params():
    """Backward through V_omni must not populate gradients on AvgStrategyNetwork.

    Although the two networks are distinct `nn.Module` instances and Python
    cannot route gradients across them implicitly, this test makes the
    invariant explicit and guards future refactors that might (e.g.) merge
    optimizers, share trunks, or accidentally mix losses.
    """
    torch.manual_seed(5)
    avg = AvgStrategyNetwork()
    v = HistoryValueNetwork()

    for p in avg.parameters():
        p.grad = None
    for p in v.parameters():
        p.grad = None

    batch = 4
    x = _rand_input(batch)
    omni = torch.randn(batch, DEFAULT_OMNISCIENT_DIM_2P)
    val = v(x, omni)
    val.sum().backward()

    for name, p in avg.named_parameters():
        assert p.grad is None, (
            f"AvgStrategyNetwork.{name} received a gradient from V_omni backward; "
            "omniscient gradient leakage detected"
        )

    saw_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0 for p in v.parameters()
    )
    assert saw_grad, "HistoryValueNetwork did not receive any gradient (sanity check)"


def test_history_value_backward_does_not_leak_to_regret_params():
    torch.manual_seed(6)
    regret = RegretNetwork()
    v = HistoryValueNetwork()

    for p in regret.parameters():
        p.grad = None
    for p in v.parameters():
        p.grad = None

    x = _rand_input(3)
    omni = torch.randn(3, DEFAULT_OMNISCIENT_DIM_2P)
    val = v(x, omni)
    val.sum().backward()

    for name, p in regret.named_parameters():
        assert (
            p.grad is None
        ), f"RegretNetwork.{name} received a gradient from V_omni backward"


# ----- Smoke: gradient flow (forward/backward stability) -----


def test_all_networks_backward_produces_finite_gradients():
    torch.manual_seed(8)
    r = RegretNetwork()
    a = AvgStrategyNetwork()
    v = HistoryValueNetwork()

    batch = 4
    x = _rand_input(batch)
    mask = _rand_mask(batch)
    omni = torch.randn(batch, DEFAULT_OMNISCIENT_DIM_2P)

    r_out = r(x)
    a_out = a(x, mask)
    v_out = v(x, omni)

    (r_out.sum() + a_out.sum() + v_out.sum()).backward()

    for net in (r, a, v):
        for name, p in net.named_parameters():
            assert p.grad is not None, f"{type(net).__name__}.{name} missing grad"
            assert torch.isfinite(
                p.grad
            ).all(), f"{type(net).__name__}.{name} non-finite grad"
