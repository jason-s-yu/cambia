"""
tests/test_rebel_worker.py

Tests for the ReBeL self-play episode runner (rebel_worker.py).

All tests require libcambia.so and the Go subgame solver exports (DEV1 + DEV2
complete). Episodes are short (depth=1, iters=5) to keep test runtime low.
"""

import warnings

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.networks import PBSValueNetwork, PBSPolicyNetwork
from src.config import DeepCfrConfig
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from src.encoding import NUM_ACTIONS
from src.cfr.rebel_worker import (
    EpisodeSample,
    VALUE_DIM,
    rebel_self_play_episode,
    run_rebel_episodes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def value_net():
    net = PBSValueNetwork(validate_inputs=False)
    net.eval()
    return net


@pytest.fixture(scope="module")
def policy_net():
    net = PBSPolicyNetwork(validate_inputs=False)
    net.eval()
    return net


@pytest.fixture(scope="module")
def fast_config():
    """Minimal DeepCfrConfig for fast test episodes."""
    cfg = DeepCfrConfig()
    cfg.rebel_subgame_depth = 1
    cfg.rebel_cfr_iterations = 5
    return cfg


def _run_episode(value_net, policy_net, fast_config):
    """Run one episode with Go-defaults house rules (suppressing deprecation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        samples = rebel_self_play_episode(
            game_config=None,
            value_net=value_net,
            policy_net=policy_net,
            rebel_config=fast_config,
            exploration_epsilon=0.5,
        )
    return samples


# ---------------------------------------------------------------------------
# Test 1: Episode produces at least one sample and terminates
# ---------------------------------------------------------------------------


def test_episode_terminates_and_produces_samples(value_net, policy_net, fast_config):
    """Episode must finish (no infinite loop) and yield >= 1 sample."""
    samples = _run_episode(value_net, policy_net, fast_config)
    # Game must produce at least one decision point
    assert isinstance(samples, list)
    assert len(samples) >= 1


# ---------------------------------------------------------------------------
# Test 2: Training data shapes
# ---------------------------------------------------------------------------


def test_episode_sample_shapes(value_net, policy_net, fast_config):
    """Each EpisodeSample must have correct shapes and dtypes."""
    samples = _run_episode(value_net, policy_net, fast_config)
    assert len(samples) >= 1

    for i, s in enumerate(samples):
        assert isinstance(s, EpisodeSample), f"sample {i} is not EpisodeSample"

        # features: PBS encoding (956,)
        assert s.features.shape == (PBS_INPUT_DIM,), (
            f"sample {i} features shape {s.features.shape} != ({PBS_INPUT_DIM},)"
        )
        assert s.features.dtype == np.float32, f"sample {i} features dtype {s.features.dtype}"

        # value_target: (936,) tiled from per-player root values
        assert s.value_target.shape == (VALUE_DIM,), (
            f"sample {i} value_target shape {s.value_target.shape} != ({VALUE_DIM},)"
        )
        assert s.value_target.dtype == np.float32

        # policy_target: (146,)
        assert s.policy_target.shape == (NUM_ACTIONS,), (
            f"sample {i} policy_target shape {s.policy_target.shape} != ({NUM_ACTIONS},)"
        )
        assert s.policy_target.dtype == np.float32

        # action_mask: (146,) bool
        assert s.action_mask.shape == (NUM_ACTIONS,), (
            f"sample {i} action_mask shape {s.action_mask.shape} != ({NUM_ACTIONS},)"
        )
        assert s.action_mask.dtype == bool


# ---------------------------------------------------------------------------
# Test 3: Value semantics — finite values, policy sums to ~1
# ---------------------------------------------------------------------------


def test_value_targets_finite(value_net, policy_net, fast_config):
    """Value targets must be finite (no NaN / Inf)."""
    samples = _run_episode(value_net, policy_net, fast_config)
    for i, s in enumerate(samples):
        assert np.isfinite(s.value_target).all(), (
            f"sample {i} value_target contains non-finite values: {s.value_target}"
        )


def test_policy_targets_sum_to_one(value_net, policy_net, fast_config):
    """Policy targets must sum to approximately 1 (they are a valid strategy)."""
    samples = _run_episode(value_net, policy_net, fast_config)
    for i, s in enumerate(samples):
        total = float(s.policy_target.sum())
        assert abs(total - 1.0) < 1e-4, (
            f"sample {i} policy_target sum = {total:.6f} (expected ~1.0)"
        )


def test_features_finite(value_net, policy_net, fast_config):
    """PBS feature vectors must be finite."""
    samples = _run_episode(value_net, policy_net, fast_config)
    for i, s in enumerate(samples):
        assert np.isfinite(s.features).all(), (
            f"sample {i} features contains non-finite values"
        )


# ---------------------------------------------------------------------------
# Test 4: Action masks match legal actions
# ---------------------------------------------------------------------------


def test_action_masks_have_legal_actions(value_net, policy_net, fast_config):
    """Each action mask must have at least one legal action."""
    samples = _run_episode(value_net, policy_net, fast_config)
    for i, s in enumerate(samples):
        n_legal = int(s.action_mask.sum())
        assert n_legal >= 1, f"sample {i} has no legal actions in mask"


def test_policy_target_zero_on_illegal_actions(value_net, policy_net, fast_config):
    """Policy probability must be 0 on illegal actions."""
    samples = _run_episode(value_net, policy_net, fast_config)
    for i, s in enumerate(samples):
        illegal = ~s.action_mask
        if illegal.any():
            illegal_mass = float(s.policy_target[illegal].sum())
            assert abs(illegal_mass) < 1e-5, (
                f"sample {i} has {illegal_mass:.2e} probability on illegal actions"
            )


# ---------------------------------------------------------------------------
# Test 5: Multiple episodes produce different data (stochastic)
# ---------------------------------------------------------------------------


def test_multiple_episodes_differ(value_net, policy_net, fast_config):
    """Running many episodes should yield at least some variation in episode length or policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        episodes = [
            rebel_self_play_episode(
                None, value_net, policy_net, fast_config, exploration_epsilon=0.9
            )
            for _ in range(5)
        ]

    # All episodes must have produced samples
    for i, ep in enumerate(episodes):
        assert len(ep) >= 1, f"episode {i} has no samples"

    # PBS features at step 0 are deterministic (uniform ranges + fixed initial game state).
    # By step 1, games diverge due to different random card deals and action choices.
    # Check that any episode pair with >= 2 steps produces different second-step features,
    # or that episode lengths vary across the 5 runs (at least one pair differs).
    # Verify episodes are not all trivially identical — check that either
    # episode lengths differ or any features differ at any step
    all_lengths = [len(ep) for ep in episodes]
    lengths_vary = len(set(all_lengths)) > 1
    if not lengths_vary:
        # All same length — check if any features differ across episodes
        any_different = False
        for step_idx in range(min(all_lengths)):
            step_features = [ep[step_idx].features for ep in episodes]
            for j in range(1, len(step_features)):
                if not np.allclose(step_features[0], step_features[j]):
                    any_different = True
                    break
            if any_different:
                break
        assert any_different or len(episodes) < 2, (
            "Multiple episodes produced fully identical feature sequences — "
            "likely a seeding or game-divergence bug"
        )


# ---------------------------------------------------------------------------
# Test 6: run_rebel_episodes batch interface
# ---------------------------------------------------------------------------


def test_run_rebel_episodes_returns_correct_count(value_net, policy_net, fast_config):
    """run_rebel_episodes must return exactly num_episodes lists."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = run_rebel_episodes(
            num_episodes=3,
            value_net=value_net,
            policy_net=policy_net,
            rebel_config=fast_config,
            game_config=None,
        )

    assert len(results) == 3
    for ep in results:
        assert isinstance(ep, list)
        assert len(ep) >= 1


def test_run_rebel_episodes_valid_shapes(value_net, policy_net, fast_config):
    """All samples from run_rebel_episodes must have correct shapes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = run_rebel_episodes(
            num_episodes=2,
            value_net=value_net,
            policy_net=policy_net,
            rebel_config=fast_config,
            game_config=None,
        )

    for ep_idx, ep in enumerate(results):
        for s_idx, s in enumerate(ep):
            assert s.features.shape == (PBS_INPUT_DIM,), (
                f"ep {ep_idx} sample {s_idx}: features shape {s.features.shape}"
            )
            assert s.value_target.shape == (VALUE_DIM,), (
                f"ep {ep_idx} sample {s_idx}: value_target shape {s.value_target.shape}"
            )
            assert s.policy_target.shape == (NUM_ACTIONS,), (
                f"ep {ep_idx} sample {s_idx}: policy_target shape {s.policy_target.shape}"
            )
            assert s.action_mask.shape == (NUM_ACTIONS,), (
                f"ep {ep_idx} sample {s_idx}: action_mask shape {s.action_mask.shape}"
            )


# ---------------------------------------------------------------------------
# Test 7: EpisodeSample dimension constants
# ---------------------------------------------------------------------------


def test_value_dim_constant():
    """VALUE_DIM must equal 2 * NUM_HAND_TYPES = 936."""
    assert VALUE_DIM == 2 * NUM_HAND_TYPES
    assert VALUE_DIM == 936


def test_pbs_input_dim():
    """PBS_INPUT_DIM must be 956."""
    assert PBS_INPUT_DIM == 956
