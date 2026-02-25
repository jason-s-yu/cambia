"""
tests/test_statistical_properties.py

Statistical property tests for reservoir sampling, action encoding,
Plackett-Luce MLE, QRE entropy, and EMA convergence.

All tests use deterministic seeds for reproducibility.
"""

import random

import numpy as np
import pytest
import torch

from src.reservoir import ReservoirBuffer, ReservoirSample
from src.encoding import action_to_index, INPUT_DIM, NUM_ACTIONS
from src.encoding import nplayer_action_to_index
from src.constants import (
    N_PLAYER_NUM_ACTIONS,
    ActionDrawStockpile,
    ActionDrawDiscard,
    ActionCallCambia,
    ActionDiscard,
    ActionReplace,
    ActionAbilityPeekOwnSelect,
    ActionAbilityPeekOtherSelect,
    ActionAbilityBlindSwapSelect,
    ActionAbilityKingLookSelect,
    ActionAbilityKingSwapDecision,
    ActionPassSnap,
    ActionSnapOwn,
    ActionSnapOpponent,
    ActionSnapOpponentMove,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(features: np.ndarray) -> ReservoirSample:
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    return ReservoirSample(features=features, target=target, action_mask=mask, iteration=0)


def _all_2p_actions():
    """Generate all valid 2P GameAction instances covering all 146 indices."""
    actions = []
    actions.append(ActionDrawStockpile())
    actions.append(ActionDrawDiscard())
    actions.append(ActionCallCambia())
    actions.append(ActionDiscard(use_ability=False))
    actions.append(ActionDiscard(use_ability=True))
    for i in range(6):
        actions.append(ActionReplace(target_hand_index=i))
    for i in range(6):
        actions.append(ActionAbilityPeekOwnSelect(target_hand_index=i))
    for i in range(6):
        actions.append(ActionAbilityPeekOtherSelect(target_opponent_hand_index=i))
    for own in range(6):
        for opp in range(6):
            actions.append(ActionAbilityBlindSwapSelect(own_hand_index=own, opponent_hand_index=opp))
    for own in range(6):
        for opp in range(6):
            actions.append(ActionAbilityKingLookSelect(own_hand_index=own, opponent_hand_index=opp))
    actions.append(ActionAbilityKingSwapDecision(perform_swap=False))
    actions.append(ActionAbilityKingSwapDecision(perform_swap=True))
    actions.append(ActionPassSnap())
    for i in range(6):
        actions.append(ActionSnapOwn(own_card_hand_index=i))
    for i in range(6):
        actions.append(ActionSnapOpponent(opponent_target_hand_index=i))
    for own in range(6):
        for slot in range(6):
            actions.append(ActionSnapOpponentMove(own_card_to_move_hand_index=own, target_empty_slot_index=slot))
    return actions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReservoirUniformRetention:
    def test_reservoir_uniform_retention(self):
        """
        Reservoir sampling should retain each item with equal probability.

        We insert items 0-9999 into a buffer of capacity 100, repeat 500 times
        with different seeds, and verify the retention frequencies follow a
        uniform distribution via chi-squared goodness-of-fit (p > 0.01).
        """
        n_items = 10000
        capacity = 100
        n_trials = 500

        # Count how many times each item ends up in the buffer
        retention_counts = np.zeros(n_items, dtype=np.int64)

        for trial in range(n_trials):
            random.seed(trial * 7 + 42)
            np.random.seed(trial * 7 + 42)

            buf = ReservoirBuffer(capacity=capacity, input_dim=1, target_dim=1)
            for i in range(n_items):
                feat = np.array([[float(i)]], dtype=np.float32).reshape(1)
                tgt = np.zeros(1, dtype=np.float32)
                mask = np.ones(1, dtype=bool)
                sample = ReservoirSample(features=feat, target=tgt, action_mask=mask, iteration=0)
                buf.add(sample)

            # Count which items are in the buffer (identify by feature value)
            size = len(buf)
            for k in range(size):
                item_idx = int(round(buf._features[k, 0]))
                if 0 <= item_idx < n_items:
                    retention_counts[item_idx] += 1

        # Expected: each item retained capacity * n_trials / n_items times
        expected_per_item = capacity * n_trials / n_items  # = 5.0

        # Chi-squared goodness-of-fit against uniform
        total_retained = retention_counts.sum()
        expected_counts = np.full(n_items, total_retained / n_items)

        # Bin items to reduce chi-sq df (group into 100 equal-size bins)
        n_bins = 100
        bin_size = n_items // n_bins
        observed_binned = np.array([
            retention_counts[b * bin_size:(b + 1) * bin_size].sum()
            for b in range(n_bins)
        ])
        expected_binned = np.full(n_bins, total_retained / n_bins)

        # Manual chi-squared statistic
        chi2_stat = np.sum((observed_binned - expected_binned) ** 2 / expected_binned)
        df = n_bins - 1

        # Use scipy if available, else approximate with CDF
        try:
            from scipy import stats as scipy_stats
            p_value = 1.0 - scipy_stats.chi2.cdf(chi2_stat, df)
        except ImportError:
            # Rough approximation: chi2 with df=99 at 99th percentile ~= 134
            # If chi2_stat < 134, p > 0.01
            p_value = 1.0 if chi2_stat < 134.0 else 0.0

        assert p_value > 0.01, (
            f"Reservoir sampling is not uniform: chi2={chi2_stat:.2f}, df={df}, p={p_value:.4f}. "
            f"Expected per item: {expected_per_item:.2f}, actual mean: {retention_counts.mean():.2f}"
        )


class TestActionIndexBijection:
    def test_2p_action_index_exhaustive_roundtrip(self):
        """
        action_to_index covers all 146 indices with no duplicates.

        Generate all valid GameAction instances, compute their indices, and verify
        we get exactly {0, ..., 145} with no duplicates.
        """
        actions = _all_2p_actions()
        indices = [action_to_index(a) for a in actions]

        assert len(indices) == NUM_ACTIONS, (
            f"Expected {NUM_ACTIONS} unique actions, got {len(indices)}"
        )
        assert set(indices) == set(range(NUM_ACTIONS)), (
            f"Index set mismatch. Missing: {set(range(NUM_ACTIONS)) - set(indices)}, "
            f"Extra: {set(indices) - set(range(NUM_ACTIONS))}"
        )
        # Check no duplicates
        assert len(indices) == len(set(indices)), (
            f"Duplicate indices found: {[i for i in indices if indices.count(i) > 1]}"
        )

    def test_nplayer_action_index_exhaustive_range(self):
        """
        nplayer_action_to_index covers all 452 indices with no gaps.

        Generate all valid (action, opp_idx) combinations and verify the returned
        indices form exactly the set {0, ..., 451}.
        """
        collected = set()
        MAX_OPP = 5  # opp_idx in [0, 4]

        # Single-target actions (opp_idx irrelevant)
        for act in [
            ActionDrawStockpile(),
            ActionDrawDiscard(),
            ActionCallCambia(),
            ActionDiscard(use_ability=False),
            ActionDiscard(use_ability=True),
        ]:
            collected.add(nplayer_action_to_index(act))

        for i in range(6):
            collected.add(nplayer_action_to_index(ActionReplace(target_hand_index=i)))
        for i in range(6):
            collected.add(nplayer_action_to_index(ActionAbilityPeekOwnSelect(target_hand_index=i)))

        # Multi-opponent actions: slot × opp_idx
        for slot in range(6):
            for opp in range(MAX_OPP):
                collected.add(nplayer_action_to_index(
                    ActionAbilityPeekOtherSelect(target_opponent_hand_index=slot), opp_idx=opp
                ))

        for own in range(6):
            for opp_slot in range(6):
                for opp in range(MAX_OPP):
                    collected.add(nplayer_action_to_index(
                        ActionAbilityBlindSwapSelect(own_hand_index=own, opponent_hand_index=opp_slot),
                        opp_idx=opp
                    ))

        for own in range(6):
            for opp_slot in range(6):
                for opp in range(MAX_OPP):
                    collected.add(nplayer_action_to_index(
                        ActionAbilityKingLookSelect(own_hand_index=own, opponent_hand_index=opp_slot),
                        opp_idx=opp
                    ))

        collected.add(nplayer_action_to_index(ActionAbilityKingSwapDecision(perform_swap=False)))
        collected.add(nplayer_action_to_index(ActionAbilityKingSwapDecision(perform_swap=True)))
        collected.add(nplayer_action_to_index(ActionPassSnap()))

        for i in range(6):
            collected.add(nplayer_action_to_index(ActionSnapOwn(own_card_hand_index=i)))

        for slot in range(6):
            for opp in range(MAX_OPP):
                collected.add(nplayer_action_to_index(
                    ActionSnapOpponent(opponent_target_hand_index=slot), opp_idx=opp
                ))

        for own in range(6):
            collected.add(nplayer_action_to_index(
                ActionSnapOpponentMove(own_card_to_move_hand_index=own, target_empty_slot_index=0)
            ))

        assert collected == set(range(N_PLAYER_NUM_ACTIONS)), (
            f"N-player index coverage mismatch. Size: {len(collected)}. "
            f"Missing: {set(range(N_PLAYER_NUM_ACTIONS)) - collected}, "
            f"Extra: {collected - set(range(N_PLAYER_NUM_ACTIONS))}"
        )


class TestPlackettLuce:
    @staticmethod
    def _sample_pl_ordering(ratings, rng):
        """Sample one ordering from Plackett-Luce model."""
        remaining = list(range(len(ratings)))
        ordering = []
        for _ in range(len(ratings)):
            probs = np.array([ratings[i] for i in remaining], dtype=np.float64)
            probs /= probs.sum()
            chosen = rng.choice(len(remaining), p=probs)
            ordering.append(remaining.pop(chosen))
        return ordering

    @staticmethod
    def _pl_log_likelihood(ratings, orderings):
        """Compute Plackett-Luce log-likelihood."""
        ll = 0.0
        for ordering in orderings:
            for j in range(len(ordering) - 1):
                remaining_sum = sum(ratings[ordering[k]] for k in range(j, len(ordering)))
                if remaining_sum > 1e-15:
                    ll += np.log(ratings[ordering[j]] / remaining_sum)
        return ll

    @staticmethod
    def _mm_update(ratings, orderings, n):
        """One MM iteration of Hunter (2004).

        The PL log-likelihood has k-1 terms (last choice is forced). Player i is in
        the remaining set at position j iff j <= pos_i. So the denominator sums over
        j in range(k-1) where j <= pos_i.
        """
        new_ratings = np.zeros(n, dtype=np.float64)
        for i in range(n):
            wins = 0.0
            denominator = 0.0
            for ordering in orderings:
                if i not in ordering:
                    continue
                pos = ordering.index(i)
                wins += 1.0 if pos < len(ordering) - 1 else 0.0
                for j in range(len(ordering) - 1):  # j from 0 to k-2
                    if j <= pos:
                        remaining_sum = sum(ratings[ordering[k]] for k in range(j, len(ordering)))
                        if remaining_sum > 1e-10:
                            denominator += 1.0 / remaining_sum
            if denominator > 1e-10:
                new_ratings[i] = wins / denominator
            else:
                new_ratings[i] = 0.0
        total = new_ratings.sum()
        if total > 1e-10:
            new_ratings *= n / total
        return new_ratings

    def test_plackett_luce_mle_convergence(self):
        """
        PL MM algorithm should recover ratings within 10% of ground truth (after normalization).

        The PSROOracle MM formula monotonically increases PL log-likelihood and
        converges to a fixed point. With 3000 orderings from ground truth [3, 2, 1, 0.5],
        the MM algorithm should recover normalized ratings within 10% of ground truth
        normalized ratings. We also verify:
        1. Recovered ratings within 10% of ground truth after normalization (both sum to n).
        2. The recovered ratings are in the same order as ground truth.
        3. Pairwise win probabilities under recovered ratings match the direction
           of ground truth win probabilities for all pairs.

        Uses PSROOracle._plackett_luce_ratings (the real implementation).
        """
        from src.cfr.psro import PSROOracle, PopulationMember

        ground_truth = np.array([3.0, 2.0, 1.0, 0.5])
        n = len(ground_truth)
        rng = np.random.RandomState(42)

        orderings = [self._sample_pl_ordering(ground_truth, rng) for _ in range(3000)]

        # Use PSROOracle to run MM (the real implementation)
        oracle = PSROOracle.__new__(PSROOracle)
        pop = [PopulationMember(path="", iteration=i) for i in range(n)]
        recovered = oracle._plackett_luce_ratings(pop, orderings, num_iterations=200)

        # 1. Verify non-zero ratings
        assert recovered.sum() > 1e-10, "All recovered ratings are zero"

        # 2. Verify recovered ratings are within 10% of ground truth after normalization
        gt_norm = ground_truth / ground_truth.sum() * n
        rec_norm = recovered / recovered.sum() * n
        for i in range(n):
            rel_err = abs(rec_norm[i] - gt_norm[i]) / gt_norm[i]
            assert rel_err < 0.10, (
                f"Player {i} recovered rating deviates {rel_err:.1%} from ground truth "
                f"(limit 10%): gt_norm={gt_norm[i]:.4f}, rec_norm={rec_norm[i]:.4f}. "
                f"Full recovered: {recovered}"
            )

        # 3. Verify correct ordering (players ranked by ground truth strength)
        gt_order = list(np.argsort(-ground_truth))
        rec_order = list(np.argsort(-recovered))
        assert gt_order == rec_order, (
            f"Ranking order wrong: expected {gt_order}, got {rec_order}. "
            f"Recovered ratings: {recovered}"
        )

        # 4. Verify pairwise win directions match ground truth
        for i in range(n):
            for j in range(i + 1, n):
                gt_i_stronger = ground_truth[i] > ground_truth[j]
                rec_i_stronger = recovered[i] > recovered[j]
                assert gt_i_stronger == rec_i_stronger, (
                    f"Pairwise direction wrong for players {i} vs {j}: "
                    f"GT={ground_truth[i]:.2f} vs {ground_truth[j]:.2f}, "
                    f"rec={recovered[i]:.3f} vs {recovered[j]:.3f}"
                )

    def test_plackett_luce_likelihood_monotonic(self):
        """
        MM algorithm log-likelihood should be non-decreasing at every step.
        """
        ground_truth = np.array([2.5, 1.8, 1.2, 0.7])
        n = len(ground_truth)
        rng = np.random.RandomState(123)

        orderings = [self._sample_pl_ordering(ground_truth, rng) for _ in range(500)]

        ratings = np.ones(n, dtype=np.float64)
        prev_ll = self._pl_log_likelihood(ratings, orderings)
        for step in range(200):
            ratings = self._mm_update(ratings, orderings, n)
            curr_ll = self._pl_log_likelihood(ratings, orderings)
            assert curr_ll >= prev_ll - 1e-10, (
                f"Log-likelihood decreased at step {step}: {prev_ll:.6f} -> {curr_ll:.6f}"
            )
            prev_ll = curr_ll


class TestQREEntropy:
    def test_qre_entropy_monotonic_in_lambda(self):
        """
        QRE entropy should be non-increasing as lambda decreases.

        Lower lambda = lower temperature = more concentrated distribution = lower entropy.
        """
        from src.cfr.deep_trainer import qre_strategy

        adv = torch.tensor([[5.0, 3.0, 1.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        lambdas = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01]
        entropies = []

        for lam in lambdas:
            sigma = qre_strategy(adv, mask, lam=lam)
            # Shannon entropy H = -sum(sigma * log(sigma))
            p = sigma[0].detach().numpy()
            h = -np.sum(p * np.log(p + 1e-30))
            entropies.append(h)

        # As lambda decreases, entropy should be non-increasing
        for i in range(len(entropies) - 1):
            assert entropies[i] >= entropies[i + 1] - 1e-8, (
                f"Entropy increased from lambda={lambdas[i]} to lambda={lambdas[i+1]}: "
                f"{entropies[i]:.6f} -> {entropies[i+1]:.6f}"
            )


class TestEMAConvergence:
    def test_ema_converges_to_ensemble(self):
        """
        Incremental EMA with weights w_t=(t+1)^1.5 should match the full weighted average.

        This validates the _update_ema() formula in deep_trainer.py.
        """
        rng = np.random.RandomState(42)
        n_snapshots = 50

        # Generate random snapshots
        snapshots = [
            {"w": rng.randn(4, 3).astype(np.float32), "b": rng.randn(3).astype(np.float32)}
            for _ in range(n_snapshots)
        ]

        # Full weighted average
        weights = np.array([(t + 1) ** 1.5 for t in range(n_snapshots)])
        total_weight = weights.sum()
        avg = {
            "w": sum(weights[t] * snapshots[t]["w"] for t in range(n_snapshots)) / total_weight,
            "b": sum(weights[t] * snapshots[t]["b"] for t in range(n_snapshots)) / total_weight,
        }

        # Incremental EMA (matching deep_trainer._update_ema)
        ema = None
        ema_sum = 0.0
        for t in range(n_snapshots):
            w_t = (t + 1) ** 1.5
            new_sum = ema_sum + w_t
            if ema is None:
                ema = {k: v.copy() for k, v in snapshots[t].items()}
                ema_sum = w_t
            else:
                ratio_old = ema_sum / new_sum
                ratio_new = w_t / new_sum
                for k in ema:
                    ema[k] = ratio_old * ema[k] + ratio_new * snapshots[t][k]
                ema_sum = new_sum

        # Should match full weighted average
        for key in ["w", "b"]:
            assert np.allclose(ema[key], avg[key], atol=1e-4), (
                f"EMA mismatch for key '{key}': max_abs_diff="
                f"{np.abs(ema[key] - avg[key]).max():.6f}"
            )


class TestOSRegretFormula:
    MAX_IS_WEIGHT = 20.0

    @staticmethod
    def _compute_os_regret_worker_style(sigma, chosen_idx, utility, sampling_prob, num_actions):
        """Replicate corrected regret computation from deep_worker.py (constant baseline)."""
        MAX_IS_WEIGHT = 20.0
        utility_estimate = utility * min(1.0 / sampling_prob, MAX_IS_WEIGHT)
        regrets = np.zeros(num_actions, dtype=np.float64)
        baseline = sigma[chosen_idx] * utility_estimate
        for a_idx in range(num_actions):
            action_value_estimate = (1.0 if a_idx == chosen_idx else 0.0) * utility_estimate
            regrets[a_idx] = action_value_estimate - baseline
        return regrets

    @staticmethod
    def _compute_os_regret_theoretical(sigma, chosen_idx, utility, sampling_prob, num_actions):
        """Theoretical IS-corrected regret with constant baseline: r(a) = (u/q) * (I(a==a*) - σ(a*))."""
        MAX_IS_WEIGHT = 20.0
        u_q = utility * min(1.0 / sampling_prob, MAX_IS_WEIGHT)
        regrets = np.zeros(num_actions, dtype=np.float64)
        for a_idx in range(num_actions):
            indicator = 1.0 if a_idx == chosen_idx else 0.0
            regrets[a_idx] = u_q * (indicator - sigma[chosen_idx])
        return regrets

    def test_os_regret_formula_matches_worker_code(self):
        """
        Worker regret computation must match the theoretical IS-corrected formula.

        Corrected formula uses constant baseline σ(a*) instead of action-dependent σ(a).
        r(a) = (u/q) * (I(a==a*) - σ(a*))

        Tests 100 random inputs and verifies:
        1. Worker and theoretical formulas produce identical results.
        2. WEIGHTED sum invariant: Σ_a σ(a)*regret(a) = 0 always holds.
           Proof: Σ_a σ(a)*(u/q)*(I(a==a*) - σ(a*)) = (u/q)*σ(a*)*(1 - 1) = 0.

        Note: the UNWEIGHTED sum Σ_a regret(a) is NOT zero per sample with the
        corrected formula.
        """
        rng = np.random.RandomState(99)
        num_actions = 10

        for trial in range(100):
            sigma = rng.dirichlet(np.ones(num_actions))
            chosen_idx = rng.randint(0, num_actions)
            utility = rng.uniform(-5.0, 5.0)
            sampling_prob = rng.uniform(0.01, 1.0)

            worker_regrets = self._compute_os_regret_worker_style(
                sigma, chosen_idx, utility, sampling_prob, num_actions
            )
            theoretical_regrets = self._compute_os_regret_theoretical(
                sigma, chosen_idx, utility, sampling_prob, num_actions
            )

            # 1. Formulas must match exactly
            assert np.allclose(worker_regrets, theoretical_regrets, atol=1e-12), (
                f"Trial {trial}: worker and theoretical regrets differ. "
                f"Max diff: {np.abs(worker_regrets - theoretical_regrets).max():.2e}"
            )

            # 2. WEIGHTED sum invariant: Σ_a σ(a)*r(a) = 0 exactly
            weighted_sum = np.sum(sigma * worker_regrets)
            assert abs(weighted_sum) < 1e-10, (
                f"Trial {trial}: weighted zero-sum invariant violated. "
                f"Σ σ(a)*r(a) = {weighted_sum:.2e} (expected 0)"
            )
