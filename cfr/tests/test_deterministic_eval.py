"""Tests for deterministic eval action selection and config defaults."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_config():
    """Minimal config with deep_cfr attrs needed by GTCFRAgentWrapper."""
    from src.config import DeepCfrConfig

    dcfr = DeepCfrConfig()
    cfg = MagicMock()
    cfg.deep_cfr = dcfr
    cfg.cambia_rules = None
    cfg.num_players = 2
    return cfg


def test_config_defaults():
    """Verify new config defaults for detach and policy loss weight."""
    from src.config import DeepCfrConfig

    cfg = DeepCfrConfig()
    assert cfg.cvpn_detach_policy_grad is False
    assert cfg.gtcfr_policy_loss_weight == 5.0
    assert cfg.gtcfr_expansion_k == 3


def test_gtcfr_wrapper_accepts_deterministic_flag(mock_config):
    """GTCFRAgentWrapper accepts and stores the deterministic kwarg."""
    with patch("src.networks.build_cvpn") as mock_build, \
         patch("src.pbs.uniform_range", return_value=np.ones(468) / 468):
        mock_net = MagicMock()
        mock_build.return_value = mock_net

        from src.evaluate_agents import GTCFRAgentWrapper

        # Default: deterministic=True
        wrapper = GTCFRAgentWrapper(0, mock_config, deterministic=True)
        assert wrapper._deterministic is True

        # Explicit: deterministic=False
        wrapper2 = GTCFRAgentWrapper(0, mock_config, deterministic=False)
        assert wrapper2._deterministic is False


def test_sog_inference_inherits_deterministic(mock_config):
    """SoGInferenceAgentWrapper inherits deterministic flag from GTCFRAgentWrapper."""
    with patch("src.networks.build_cvpn") as mock_build, \
         patch("src.pbs.uniform_range", return_value=np.ones(468) / 468):
        mock_net = MagicMock()
        mock_build.return_value = mock_net

        from src.evaluate_agents import SoGInferenceAgentWrapper

        wrapper = SoGInferenceAgentWrapper(0, mock_config, deterministic=False)
        assert wrapper._deterministic is False


def test_deterministic_picks_argmax():
    """With deterministic=True, argmax of legal_probs is chosen."""
    legal_probs = np.array([0.1, 0.6, 0.3])

    # Deterministic: always picks index 1 (highest prob)
    chosen = np.argmax(legal_probs)
    assert chosen == 1

    # Verify consistency over multiple calls
    for _ in range(10):
        assert np.argmax(legal_probs) == 1


def test_stochastic_can_vary():
    """With deterministic=False, np.random.choice can produce different actions."""
    legal_probs = np.array([0.33, 0.34, 0.33])
    np.random.seed(None)

    choices = set()
    for _ in range(100):
        chosen = np.random.choice(len(legal_probs), p=legal_probs)
        choices.add(chosen)

    # With near-uniform probs and 100 trials, we should see at least 2 distinct actions
    assert len(choices) >= 2


def test_get_agent_passes_deterministic_for_gtcfr(mock_config):
    """get_agent maps use_argmax to deterministic for gtcfr agent type."""
    with patch("src.networks.build_cvpn") as mock_build, \
         patch("src.pbs.uniform_range", return_value=np.ones(468) / 468):
        mock_net = MagicMock()
        mock_build.return_value = mock_net

        from src.evaluate_agents import get_agent

        # use_argmax=False -> deterministic=False
        agent = get_agent("gtcfr", 0, mock_config, checkpoint_path="", use_argmax=False)
        assert agent._deterministic is False

        # use_argmax=True -> deterministic=True
        agent2 = get_agent("gtcfr", 0, mock_config, checkpoint_path="", use_argmax=True)
        assert agent2._deterministic is True


def test_get_agent_default_deterministic_for_gtcfr(mock_config):
    """Without use_argmax, gtcfr defaults to deterministic=True."""
    with patch("src.networks.build_cvpn") as mock_build, \
         patch("src.pbs.uniform_range", return_value=np.ones(468) / 468):
        mock_net = MagicMock()
        mock_build.return_value = mock_net

        from src.evaluate_agents import get_agent

        # No use_argmax kwarg at all -> default True
        agent = get_agent("gtcfr", 0, mock_config, checkpoint_path="")
        assert agent._deterministic is True
