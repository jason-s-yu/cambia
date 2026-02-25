"""
tests/test_rebel_eval_wrapper.py

Tests for ReBeLAgentWrapper: instantiation with mock checkpoint, registry
entry, and choose_action smoke test.
"""

import tempfile

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="ReBeL is deprecated: mathematically unsound for N-player FFA with continuous beliefs"
)

from src.networks import PBSValueNetwork, PBSPolicyNetwork
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config():
    config = type("Config", (), {})()

    rules = type("CambiaRulesConfig", (), {})()
    rules.allowDrawFromDiscardPile = False
    rules.allowReplaceAbilities = False
    rules.snapRace = False
    rules.penaltyDrawCount = 2
    rules.use_jokers = 0
    rules.cards_per_player = 4
    rules.initial_view_count = 2
    rules.cambia_allowed_round = 0
    rules.allowOpponentSnapping = False
    rules.max_game_turns = 100
    config.cambia_rules = rules

    agent_params = type("AgentParamsConfig", (), {})()
    agent_params.memory_level = 1
    agent_params.time_decay_turns = 10
    config.agent_params = agent_params

    agents_cfg = type("AgentsConfig", (), {})()
    agents_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_cambia_threshold = 5
    config.agents = agents_cfg

    return config


def _make_rebel_checkpoint(path: str, value_hidden_dim: int = 64, policy_hidden_dim: int = 32):
    """Save a freshly-initialized ReBeL checkpoint with small hidden dims for speed."""
    value_net = PBSValueNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=value_hidden_dim,
        output_dim=2 * NUM_HAND_TYPES,
        validate_inputs=False,
    )
    policy_net = PBSPolicyNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=policy_hidden_dim,
        validate_inputs=False,
    )
    checkpoint = {
        "rebel_value_net_state_dict": value_net.state_dict(),
        "rebel_policy_net_state_dict": policy_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {
            "rebel_value_hidden_dim": value_hidden_dim,
            "rebel_policy_hidden_dim": policy_hidden_dim,
            "rebel_subgame_depth": 4,
            "rebel_cfr_iterations": 10,
        },
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


def test_rebel_wrapper_instantiates():
    """ReBeLAgentWrapper loads correctly from a mock checkpoint."""
    from src.evaluate_agents import ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    assert wrapper.value_net is not None
    assert wrapper.policy_net is not None
    assert wrapper.rebel_depth == 4
    assert wrapper.rebel_cfr_iterations == 10


def test_rebel_wrapper_networks_eval_mode():
    """Both networks are in eval mode after construction."""
    from src.evaluate_agents import ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    assert not wrapper.value_net.training
    assert not wrapper.policy_net.training


def test_rebel_wrapper_defaults_from_checkpoint():
    """Wrapper reads rebel_* config fields from checkpoint dcfr_config."""
    from src.evaluate_agents import ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    # Checkpoint with no dcfr_config — should use class defaults
    value_net = PBSValueNetwork(
        input_dim=PBS_INPUT_DIM, hidden_dim=1024, output_dim=2 * NUM_HAND_TYPES, validate_inputs=False
    )
    policy_net = PBSPolicyNetwork(
        input_dim=PBS_INPUT_DIM, hidden_dim=512, validate_inputs=False
    )
    torch.save(
        {
            "rebel_value_net_state_dict": value_net.state_dict(),
            "rebel_policy_net_state_dict": policy_net.state_dict(),
        },
        path,
    )

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    # Default values from config.py DeepCfrConfig
    assert wrapper.rebel_depth == 4
    assert wrapper.rebel_cfr_iterations == 200


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_rebel_in_agent_registry():
    """'rebel' key exists in AGENT_REGISTRY."""
    from src.evaluate_agents import AGENT_REGISTRY

    assert "rebel" in AGENT_REGISTRY


def test_get_agent_rebel():
    """get_agent('rebel') returns ReBeLAgentWrapper."""
    from src.evaluate_agents import get_agent, ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    agent = get_agent("rebel", player_id=0, config=config, checkpoint_path=path, device="cpu")
    assert isinstance(agent, ReBeLAgentWrapper)


def test_get_agent_rebel_missing_checkpoint():
    """get_agent('rebel') without checkpoint_path raises ValueError."""
    from src.evaluate_agents import get_agent

    config = _make_config()
    with pytest.raises(ValueError, match="checkpoint_path"):
        get_agent("rebel", player_id=0, config=config)


# ---------------------------------------------------------------------------
# choose_action smoke test
# ---------------------------------------------------------------------------


def test_rebel_choose_action_without_state():
    """choose_action falls back to random when agent_state is None."""
    from src.evaluate_agents import ReBeLAgentWrapper
    from src.constants import ActionDrawStockpile, ActionCallCambia

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    # agent_state is None — should fall back to random without error
    legal_actions = {ActionDrawStockpile(), ActionCallCambia()}
    action = wrapper.choose_action(game_state=None, legal_actions=legal_actions)
    assert action in legal_actions


def test_rebel_choose_action_with_state():
    """choose_action runs PBSPolicyNetwork inference with initialized agent state."""
    from src.evaluate_agents import ReBeLAgentWrapper
    from src.game.engine import CambiaGameState

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    wrapper.initialize_state(game_state)
    assert wrapper.agent_state is not None

    legal_actions = game_state.get_legal_actions()
    assert legal_actions

    action = wrapper.choose_action(game_state, legal_actions)
    assert action in legal_actions


def test_rebel_build_pbs_shape():
    """_build_pbs returns a PBS with correct shapes."""
    from src.evaluate_agents import ReBeLAgentWrapper
    from src.game.engine import CambiaGameState
    from src.pbs import NUM_HAND_TYPES, NUM_PUBLIC_FEATURES

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    game_state = CambiaGameState(house_rules=config.cambia_rules)

    pbs = wrapper._build_pbs(game_state)
    assert pbs.range_p0.shape == (NUM_HAND_TYPES,)
    assert pbs.range_p1.shape == (NUM_HAND_TYPES,)
    assert pbs.public_features.shape == (NUM_PUBLIC_FEATURES,)
    # Uniform ranges sum to 1
    assert abs(pbs.range_p0.sum() - 1.0) < 1e-5
    assert abs(pbs.range_p1.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Bug fix tests (1a–1e)
# ---------------------------------------------------------------------------


def test_initialize_state_called_via_isinstance():
    """isinstance check now covers NeuralAgentWrapper, so ReBeLAgentWrapper.initialize_state IS called."""
    from src.evaluate_agents import ReBeLAgentWrapper, NeuralAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    _make_rebel_checkpoint(path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    # ReBeLAgentWrapper is a NeuralAgentWrapper, not DeepCFRAgentWrapper —
    # the fixed isinstance check must include NeuralAgentWrapper.
    assert isinstance(wrapper, NeuralAgentWrapper), (
        "ReBeLAgentWrapper must be an instance of NeuralAgentWrapper"
    )
    assert wrapper.agent_state is None  # not yet initialized

    from src.game.engine import CambiaGameState
    game_state = CambiaGameState(house_rules=config.cambia_rules)
    wrapper.initialize_state(game_state)
    assert wrapper.agent_state is not None


def test_checkpoint_rebel_config_key_loads():
    """ReBeLAgentWrapper reads 'rebel_config' key from checkpoint (trainer save format)."""
    from src.evaluate_agents import ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    value_net = PBSValueNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=64,
        output_dim=2 * NUM_HAND_TYPES,
        validate_inputs=False,
    )
    policy_net = PBSPolicyNetwork(
        input_dim=PBS_INPUT_DIM,
        hidden_dim=32,
        validate_inputs=False,
    )
    checkpoint = {
        "rebel_value_net_state_dict": value_net.state_dict(),
        "rebel_policy_net_state_dict": policy_net.state_dict(),
        # trainer saves as "rebel_config", NOT "dcfr_config"
        "rebel_config": {
            "rebel_value_hidden_dim": 64,
            "rebel_policy_hidden_dim": 32,
            "rebel_subgame_depth": 7,
            "rebel_cfr_iterations": 50,
        },
    }
    torch.save(checkpoint, path)

    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    # Values must come from rebel_config
    assert wrapper.rebel_depth == 7
    assert wrapper.rebel_cfr_iterations == 50


def test_checkpoint_dcfr_config_key_backward_compat():
    """ReBeLAgentWrapper falls back to 'dcfr_config' key for backward compatibility."""
    from src.evaluate_agents import ReBeLAgentWrapper

    config = _make_config()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    # _make_rebel_checkpoint saves "dcfr_config" (old format)
    _make_rebel_checkpoint(path, value_hidden_dim=64, policy_hidden_dim=32)

    # Should load without error and use dcfr_config values
    wrapper = ReBeLAgentWrapper(player_id=0, config=config, checkpoint_path=path, device="cpu")
    assert wrapper.rebel_depth == 4
    assert wrapper.rebel_cfr_iterations == 10


def test_imported_dims_match_expected_values():
    """Imported dimension constants must match known expected values."""
    from src.pbs import PBS_INPUT_DIM as actual_pbs_dim, NUM_HAND_TYPES as actual_hand_types
    from src.encoding import NUM_ACTIONS as actual_num_actions

    assert actual_pbs_dim == 956, f"PBS_INPUT_DIM expected 956, got {actual_pbs_dim}"
    assert actual_hand_types == 468, f"NUM_HAND_TYPES expected 468, got {actual_hand_types}"
    assert actual_num_actions == 146, f"NUM_ACTIONS expected 146, got {actual_num_actions}"
    assert 2 * actual_hand_types == 936, "VALUE_OUTPUT_DIM (2*NUM_HAND_TYPES) must be 936"
