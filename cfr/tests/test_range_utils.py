"""Tests for per-hand-type range update utilities."""

import numpy as np
import pytest
import torch


@pytest.fixture
def small_cvpn():
    """Build a small CVPN with random weights (no checkpoint needed)."""
    from src.networks import build_cvpn

    cvpn = build_cvpn(
        hidden_dim=64,
        num_blocks=1,
        validate_inputs=False,
        detach_policy_grad=False,
    )
    cvpn.eval()
    return cvpn


@pytest.fixture
def uniform_ranges():
    """Return uniform range distributions for both players."""
    from src.pbs import uniform_range

    return uniform_range(), uniform_range()


@pytest.fixture
def sample_pbs(uniform_ranges):
    """Build a sample PBS with uniform ranges and default public features."""
    from src.pbs import PBS, make_public_features, PHASE_DRAW

    r0, r1 = uniform_ranges
    pub = make_public_features(
        turn=3,
        max_turns=46,
        phase=PHASE_DRAW,
        discard_top_bucket=None,
        stockpile_remaining=40,
        stockpile_total=46,
    )
    return PBS(range_p0=r0, range_p1=r1, public_features=pub)


@pytest.fixture
def sample_legal_mask():
    """Return a legal mask with a few legal actions."""
    from src.encoding import NUM_ACTIONS

    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    # Mark a handful of actions as legal (draw stockpile, draw discard, some discards)
    mask[0] = True  # draw_stockpile
    mask[1] = True  # draw_discard
    mask[10] = True
    mask[20] = True
    mask[50] = True
    return mask


class TestComputePolicyMatrixFromPbs:
    """Tests for compute_policy_matrix_cvpn_from_pbs."""

    def test_shape(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Output shape is (NUM_HAND_TYPES, NUM_ACTIONS)."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs
        from src.pbs import NUM_HAND_TYPES
        from src.encoding import NUM_ACTIONS

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            result = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        assert result.shape == (NUM_HAND_TYPES, NUM_ACTIONS)
        assert result.dtype == np.float32

    def test_rows_differ(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Different hand types produce different policy rows.

        With random CVPN weights, different delta-range inputs should produce
        distinguishable outputs (not all identical like the old np.tile approach).
        """
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            result = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        # Check that not all rows are identical (the whole point of the fix)
        # Compare first few rows. With random weights this should always differ.
        row_0 = result[0]
        row_100 = result[100]
        row_300 = result[300]

        # At least some pair should differ
        assert not np.allclose(row_0, row_100) or not np.allclose(row_0, row_300), (
            "All rows are identical: delta-range PBS encoding is not working"
        )

    def test_rows_sum_to_one(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Each row sums to ~1.0 (valid probability distribution)."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            result = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_illegal_actions_zero(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Illegal actions should have zero probability (masked out)."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            result = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        illegal_mask = ~sample_legal_mask
        # All probabilities at illegal action indices should be 0
        assert np.allclose(result[:, illegal_mask], 0.0, atol=1e-7)

    def test_acting_player_1(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Works correctly for acting_player=1 (delta applied to P1 range)."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs
        from src.pbs import NUM_HAND_TYPES
        from src.encoding import NUM_ACTIONS

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            result = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=1, range_p0=r0, range_p1=r1,
            )

        assert result.shape == (NUM_HAND_TYPES, NUM_ACTIONS)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestRangeUpdateWithProperMatrix:
    """Tests that update_range with per-hand-type matrix changes the distribution."""

    def test_range_changes(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Range update with per-hand-type matrix produces a non-uniform result."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs
        from src.pbs import update_range, uniform_range

        r0, r1 = uniform_ranges
        with torch.inference_mode():
            policy_matrix = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        # Pick a legal action
        legal_idx = np.where(sample_legal_mask)[0][0]
        updated = update_range(r0, int(legal_idx), policy_matrix)

        # Updated range should still sum to 1
        assert abs(updated.sum() - 1.0) < 1e-5

        # Updated range should differ from uniform (the whole point)
        uniform = uniform_range()
        assert not np.allclose(updated, uniform, atol=1e-4), (
            "Range did not change after update with per-hand-type policy matrix"
        )

    def test_tiled_range_stays_uniform(self, uniform_ranges):
        """Verify the bug: np.tile(policy, (468,1)) keeps range uniform (identity update)."""
        from src.pbs import update_range, uniform_range, NUM_HAND_TYPES
        from src.encoding import NUM_ACTIONS

        r0, _ = uniform_ranges

        # Simulate old broken behavior: same policy for all hand types
        fake_policy = np.random.dirichlet(np.ones(5))
        # Spread across full action space
        full_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        full_policy[:5] = fake_policy.astype(np.float32)
        tiled_matrix = np.tile(full_policy, (NUM_HAND_TYPES, 1))

        updated = update_range(r0, 0, tiled_matrix)

        # With uniform range and tiled (identical) rows, the update is identity:
        # range_new[h] = range[h] * policy[action] / Z
        # Since range[h] = 1/N and policy[action] is same for all h,
        # range_new[h] = (1/N * p) / (N * 1/N * p) = 1/N
        uniform = uniform_range()
        np.testing.assert_allclose(updated, uniform, atol=1e-6)


class TestRangeEntropyDecreases:
    """Range entropy should decrease after successive updates with proper policy matrix."""

    def test_entropy_decreases(self, small_cvpn, sample_pbs, sample_legal_mask, uniform_ranges):
        """Multiple range updates should concentrate the distribution (lower entropy)."""
        from src.cfr.range_utils import compute_policy_matrix_cvpn_from_pbs
        from src.pbs import update_range

        r0, r1 = uniform_ranges

        def entropy(p):
            p_safe = p[p > 0]
            return -np.sum(p_safe * np.log(p_safe))

        initial_entropy = entropy(r0)

        with torch.inference_mode():
            policy_matrix = compute_policy_matrix_cvpn_from_pbs(
                small_cvpn, sample_pbs, sample_legal_mask,
                acting_player=0, range_p0=r0, range_p1=r1,
            )

        # Apply several updates with different legal actions
        legal_indices = np.where(sample_legal_mask)[0]
        current_range = r0.copy()
        for i, action_idx in enumerate(legal_indices[:3]):
            current_range = update_range(current_range, int(action_idx), policy_matrix)

        final_entropy = entropy(current_range)

        # Entropy should decrease (distribution concentrating)
        assert final_entropy < initial_entropy, (
            f"Entropy did not decrease: {initial_entropy:.4f} -> {final_entropy:.4f}"
        )
