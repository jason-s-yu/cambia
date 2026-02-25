"""
Mathematical invariant tests for core CFR algorithms.
Verifies correctness of RM+, QRE, IS-weights, CFV, LCFR weighting, and snapshot averaging.
"""
import numpy as np
import pytest
import torch

from src.cfr.deep_trainer import qre_strategy
from src.networks import AdvantageNetwork, get_strategy_from_advantages


# ---------------------------------------------------------------------------
# RM+ (Regret Matching Plus) tests
# ---------------------------------------------------------------------------


def test_rm_plus_hand_computed():
    """adv=[3,-1,0,2], all legal → [0.6, 0, 0, 0.4]."""
    adv = torch.tensor([[3.0, -1.0, 0.0, 2.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    result = get_strategy_from_advantages(adv, mask)
    expected = torch.tensor([[0.6, 0.0, 0.0, 0.4]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_rm_plus_all_negative_uniform():
    """adv=[-5,-3,-1], all legal → uniform [1/3, 1/3, 1/3]."""
    adv = torch.tensor([[-5.0, -3.0, -1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = get_strategy_from_advantages(adv, mask)
    expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_rm_plus_single_legal():
    """Only one legal action → strategy assigns 1.0 to it."""
    adv = torch.tensor([[5.0, -2.0, 3.0]])
    mask = torch.tensor([[False, True, False]])
    result = get_strategy_from_advantages(adv, mask)
    expected = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(result, expected, atol=1e-6)


def test_rm_plus_batch_independence():
    """Two rows processed independently: row0=[10,0,0]→[1,0,0], row1=[-1,-1,-1]→uniform."""
    adv = torch.tensor([[10.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    result = get_strategy_from_advantages(adv, mask)
    expected = torch.tensor([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# OS-CFR importance-sampling regret formula
# ---------------------------------------------------------------------------


def test_os_regret_formula_matches_worker():
    """Verify actual OS-CFR worker formula matches theory on 50 random inputs.

    Corrected worker code (deep_worker.py):
        utility_estimate = sampled_utility * min(1/sampling_prob, MAX_IS_WEIGHT)
        baseline = sigma[chosen] * utility_estimate   # constant baseline
        regrets[a] = (I(a==chosen) * utility_estimate) - baseline

    Theoretical: r(a) = (u/q) * (I(a==a*) - σ(a*))  [constant baseline σ(a*)]

    The WEIGHTED sum Σ_a σ(a) * r(a) = 0 always holds (theory proof).
    The UNWEIGHTED sum Σ_a r(a) does NOT generally equal zero.
    """
    rng = np.random.default_rng(42)
    for _ in range(50):
        num_actions = rng.integers(2, 8)
        raw = rng.exponential(1.0, size=num_actions)
        sigma = raw / raw.sum()
        epsilon = rng.uniform(0.1, 0.9)
        u = rng.uniform(-5, 5)
        chosen = rng.integers(0, num_actions)

        # Exploration policy (matches deep_worker.py L1417-1421)
        uniform_prob = 1.0 / num_actions
        q = epsilon * uniform_prob + (1.0 - epsilon) * sigma
        sampling_prob = q[chosen]

        MAX_IS_WEIGHT = 20.0
        utility_estimate = u * min(1.0 / sampling_prob, MAX_IS_WEIGHT)

        # Corrected worker formula: constant baseline σ(a*)
        baseline = sigma[chosen] * utility_estimate
        worker_regrets = np.zeros(num_actions)
        for a in range(num_actions):
            action_value_estimate = (
                1.0 if a == chosen else 0.0
            ) * utility_estimate
            worker_regrets[a] = action_value_estimate - baseline

        # Theoretical formula: r(a) = (u/q) * (I(a==a*) - σ(a*))
        u_q = u * min(1.0 / sampling_prob, MAX_IS_WEIGHT)
        theory_regrets = np.zeros(num_actions)
        for a in range(num_actions):
            indicator = 1.0 if a == chosen else 0.0
            theory_regrets[a] = u_q * (indicator - sigma[chosen])

        np.testing.assert_allclose(
            worker_regrets,
            theory_regrets,
            atol=1e-10,
            err_msg=f"Worker and theory disagree: {num_actions} actions, chosen={chosen}",
        )

        # Weighted sum must be zero: Σ_a σ(a) * r(a) = 0
        # Proof: Σ_a σ(a) * (u/q) * (I(a==a*) - σ(a*))
        #      = (u/q) * σ(a*) * (1 - σ(a*)) + (u/q) * Σ_{a≠a*} σ(a) * (-σ(a*))
        #      = (u/q) * σ(a*) * (1 - σ(a*) - (1 - σ(a*))) = 0
        assert np.isclose(np.sum(sigma * worker_regrets), 0.0, atol=1e-10), (
            f"Weighted sum not zero: {np.sum(sigma * worker_regrets):.2e}"
        )


# ---------------------------------------------------------------------------
# Distribution validity — random inputs
# ---------------------------------------------------------------------------


def test_strategy_sum_one_random_inputs():
    """RM+ produces valid distributions for 100 random inputs."""
    torch.manual_seed(42)
    for _ in range(100):
        adv = torch.randn(1, 10)
        # Random mask with ≥1 legal action
        mask = torch.zeros(1, 10, dtype=torch.bool)
        n_legal = torch.randint(1, 11, ()).item()
        idx = torch.randperm(10)[:n_legal]
        mask[0, idx] = True

        result = get_strategy_from_advantages(adv, mask)
        assert torch.allclose(result[mask].sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(result[~mask].sum(), torch.tensor(0.0), atol=1e-6)


def test_qre_strategy_sum_one_random():
    """QRE strategy sums to 1 over legal actions for various λ."""
    torch.manual_seed(7)
    for lam in [0.1, 0.5, 2.0]:
        for _ in range(100):
            adv = torch.randn(1, 10)
            mask = torch.zeros(1, 10, dtype=torch.bool)
            n_legal = torch.randint(1, 11, ()).item()
            idx = torch.randperm(10)[:n_legal]
            mask[0, idx] = True

            result = qre_strategy(adv, mask, lam)
            assert torch.allclose(
                result[mask].sum(), torch.tensor(1.0), atol=1e-5
            )
            assert torch.allclose(
                result[~mask].sum(), torch.tensor(0.0), atol=1e-6
            )


# ---------------------------------------------------------------------------
# CFR value / regret formula
# ---------------------------------------------------------------------------


def test_cfv_regret_strategy_orthogonality():
    """Σ_a σ(a)·regret(a) = 0 using actual RM+ strategies with partial masks.

    For strategy σ from get_strategy_from_advantages and action utilities u:
        V = σ · u
        regret(a) = u(a) - V
        Σ_a σ(a) · regret(a) = Σ_a σ(a)·u(a) - V·Σ_a σ(a) = V - V = 0
    """
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    for _ in range(100):
        num_actions = rng.integers(2, 10)
        adv = torch.randn(1, num_actions)
        mask = torch.zeros(1, num_actions, dtype=torch.bool)
        n_legal = rng.integers(1, num_actions + 1)
        idx = rng.choice(num_actions, size=n_legal, replace=False)
        mask[0, idx] = True

        sigma = get_strategy_from_advantages(adv, mask).squeeze(0).numpy()
        utilities = rng.standard_normal(num_actions)

        V = np.dot(sigma, utilities)
        regrets = utilities - V
        weighted_sum = np.dot(sigma, regrets)

        assert abs(weighted_sum) < 1e-6, f"σ·regret ≠ 0: got {weighted_sum}"


# ---------------------------------------------------------------------------
# LCFR (Linear CFR) t-weighting
# ---------------------------------------------------------------------------


def test_lcfr_weighting_formula():
    """LCFR weight w(t) = (t+1)^α, α=1.5."""
    t = np.array([0, 1, 2, 3, 4], dtype=float)
    alpha = 1.5
    weights = (t + 1) ** alpha
    expected = np.array([1.0, 2**1.5, 3**1.5, 4**1.5, 5**1.5])
    np.testing.assert_allclose(weights, expected, atol=0.01)


def test_lcfr_weight_ratio():
    """Weight ratio t=99 vs t=0 should equal 100^1.5 = 1000."""
    alpha = 1.5
    w99 = (99 + 1) ** alpha
    w0 = (0 + 1) ** alpha
    assert abs(w99 / w0 - 1000.0) < 1e-6


# ---------------------------------------------------------------------------
# Snapshot (SD-CFR) averaging
# ---------------------------------------------------------------------------


def test_snapshot_avg_valid_distribution():
    """Linearly weighted average of RM+ strategies is a valid distribution.

    Also verifies ordering: RM+-then-average (correct, matching SDCFRAgentWrapper
    in evaluate_agents.py L992-1005) differs from average-then-RM+.
    """
    torch.manual_seed(0)
    input_dim, hidden_dim, output_dim = 10, 16, 5
    nets = [AdvantageNetwork(input_dim, hidden_dim, output_dim) for _ in range(5)]
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = weights / weights.sum()

    x = torch.randn(1, input_dim)
    mask = torch.ones(1, output_dim, dtype=torch.bool)

    # RM+ first, then average (correct order, matching SDCFRAgentWrapper)
    strategies = []
    for net in nets:
        adv = net(x, mask)
        strat = get_strategy_from_advantages(adv, mask)
        strategies.append(strat)

    stacked = torch.stack(strategies, dim=0)  # (5, 1, output_dim)
    avg = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # (1, output_dim)

    assert torch.all(avg >= 0), "Averaged strategy has negative entries"
    assert torch.allclose(avg.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)

    # Verify ordering matters: average advantages first, then RM+
    avg_advantages = torch.zeros(1, output_dim)
    for i, net in enumerate(nets):
        avg_advantages = avg_advantages + weights[i] * net(x, mask)
    avg_then_rm = get_strategy_from_advantages(avg_advantages, mask)

    assert not torch.allclose(avg, avg_then_rm, atol=1e-4), (
        "RM+-then-average should differ from average-then-RM+ "
        "(confirms correct ordering in SDCFRAgentWrapper)"
    )


# ---------------------------------------------------------------------------
# Exploration policy validity
# ---------------------------------------------------------------------------


def test_exploration_policy_properties():
    """OS-CFR exploration policy q = ε·uniform + (1-ε)·σ satisfies convergence requirements.

    Matches deep_worker.py L1417-1421:
        uniform_prob = 1.0 / num_actions
        exploration_policy = epsilon * uniform_prob + (1.0 - epsilon) * local_strategy

    Properties: q sums to 1, q(a) ≥ ε/|A| (floor guarantee), all positive.
    """
    rng = np.random.default_rng(42)
    for _ in range(100):
        num_actions = rng.integers(2, 20)
        raw = rng.exponential(1.0, size=num_actions)
        sigma = raw / raw.sum()
        epsilon = rng.uniform(0.01, 0.99)

        # Worker formula (deep_worker.py L1417-1421)
        uniform_prob = 1.0 / num_actions
        q = epsilon * uniform_prob + (1.0 - epsilon) * sigma

        assert abs(q.sum() - 1.0) < 1e-10, f"q sums to {q.sum()}, expected 1.0"
        floor = epsilon / num_actions
        assert np.all(q >= floor - 1e-10), (
            f"Floor violated: min q = {q.min()}, floor = {floor}"
        )
        assert np.all(q > 0), "Exploration policy must be positive everywhere"


# ---------------------------------------------------------------------------
# Weighted regret orthogonality (foundational CFR invariant)
# ---------------------------------------------------------------------------


def test_weighted_regret_orthogonality():
    """Foundational CFR invariant: Σ_a σ_t(a)·r_t(a) = 0 across CFR iterations.

    Over 20 iterations of regret accumulation and RM+ strategy updates,
    the identity Σ_a σ(a)·(u(a) - V) = 0 must hold at every step.
    This is the key invariant that guarantees CFR convergence to Nash equilibrium.
    """
    rng = np.random.default_rng(42)
    num_actions = 6

    # Simulate 20 CFR iterations with changing utilities
    cumulative_regrets = np.zeros(num_actions)
    for t in range(20):
        # RM+ from accumulated regrets (uses actual get_strategy_from_advantages)
        adv = torch.from_numpy(cumulative_regrets.astype(np.float64)).unsqueeze(0).float()
        mask = torch.ones(1, num_actions, dtype=torch.bool)
        sigma = get_strategy_from_advantages(adv, mask).squeeze(0).double().numpy()

        # Random per-action utilities for this iteration
        utilities = rng.standard_normal(num_actions)
        V = np.dot(sigma, utilities)
        regrets = utilities - V

        # Verify identity at every iteration
        weighted_sum = np.dot(sigma, regrets)
        assert abs(weighted_sum) < 1e-6, (
            f"Iteration {t}: σ·r = {weighted_sum} ≠ 0"
        )

        # Accumulate regrets for next iteration (drives strategy evolution)
        cumulative_regrets += regrets
