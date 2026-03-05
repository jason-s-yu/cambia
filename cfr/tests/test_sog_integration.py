"""
tests/test_sog_integration.py

Integration tests for SoG eval harness:
  - Registry entries exist
  - SoGInferenceAgentWrapper choose_action (CVPN-only)
  - End-to-end game vs random (small budget, GoEngine optional)
  - Budget decoupling: SoGSearch uses eval_budget
"""

import numpy as np
import pytest

try:
    from src.ffi.bridge import GoEngine

    HAS_GO = True
except Exception:
    HAS_GO = False

skipgo = pytest.mark.skipif(not HAS_GO, reason="libcambia.so not available")


def _make_config(eval_budget: int = 3):
    cfg = type("Config", (), {})()
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
    rules.lockCallerHand = True
    rules.num_decks = 1
    rules.num_players = 2
    cfg.cambia_rules = rules

    agent_params = type("AgentParamsConfig", (), {})()
    agent_params.memory_level = 1
    agent_params.time_decay_turns = 10
    cfg.agent_params = agent_params

    agents_cfg = type("AgentsConfig", (), {})()
    agents_cfg.cambia_call_threshold = 10
    agents_cfg.greedy_cambia_threshold = 5
    cfg.agents = agents_cfg

    deep_cfr = type("DeepCfrConfig", (), {})()
    deep_cfr.gtcfr_expansion_budget = eval_budget
    deep_cfr.gtcfr_cvpn_hidden_dim = 64
    deep_cfr.gtcfr_cvpn_num_blocks = 1
    deep_cfr.gtcfr_c_puct = 2.0
    deep_cfr.gtcfr_cfr_iters_per_expansion = 2
    deep_cfr.gtcfr_buffer_capacity = 100
    deep_cfr.gtcfr_value_loss_weight = 1.0
    deep_cfr.gtcfr_policy_loss_weight = 1.0
    deep_cfr.gtcfr_cvpn_learning_rate = 3e-4
    deep_cfr.gtcfr_games_per_epoch = 1
    deep_cfr.gtcfr_epochs = 1
    deep_cfr.batch_size = 4
    cfg.deep_cfr = deep_cfr

    return cfg


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_sog_in_agent_registry():
    """'sog' must be registered in AGENT_REGISTRY."""
    from src.evaluate_agents import AGENT_REGISTRY, SoGAgentWrapper
    assert "sog" in AGENT_REGISTRY
    assert AGENT_REGISTRY["sog"] is SoGAgentWrapper


def test_sog_inference_in_agent_registry():
    """'sog_inference' must be registered in AGENT_REGISTRY."""
    from src.evaluate_agents import AGENT_REGISTRY, SoGInferenceAgentWrapper
    assert "sog_inference" in AGENT_REGISTRY
    assert AGENT_REGISTRY["sog_inference"] is SoGInferenceAgentWrapper


# ---------------------------------------------------------------------------
# SoGInferenceAgentWrapper (pure CVPN, no GoEngine needed)
# ---------------------------------------------------------------------------


def test_sog_inference_choose_action_with_state():
    """SoGInferenceAgentWrapper.choose_action returns a valid action."""
    from src.evaluate_agents import SoGInferenceAgentWrapper
    from src.game.engine import CambiaGameState

    config = _make_config()
    wrapper = SoGInferenceAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    wrapper.initialize_state(game_state)

    legal = game_state.get_legal_actions()
    action = wrapper.choose_action(game_state, legal)
    assert action in legal


def test_sog_inference_full_game_vs_random():
    """SoGInferenceAgentWrapper completes a game vs random without error."""
    from src.evaluate_agents import SoGInferenceAgentWrapper, NeuralAgentWrapper
    from src.agents.baseline_agents import RandomAgent
    from src.game.engine import CambiaGameState

    config = _make_config()
    agents = [
        SoGInferenceAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu"),
        RandomAgent(player_id=1, config=config),
    ]

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    for agent in agents:
        if isinstance(agent, NeuralAgentWrapper):
            agent.initialize_state(game_state)

    turns = 0
    while not game_state.is_terminal() and turns < 300:
        turns += 1
        pid = game_state.get_acting_player()
        if pid == -1:
            break
        legal = game_state.get_legal_actions()
        if not legal:
            break
        action = agents[pid].choose_action(game_state, legal)
        game_state.apply_action(action)

    assert game_state.is_terminal() or turns >= 300


# ---------------------------------------------------------------------------
# Budget decoupling
# ---------------------------------------------------------------------------


def test_sog_search_budget_toggle():
    """SoGSearch.use_eval_budget changes inner expansion_budget."""
    from src.cfr.sog_search import SoGSearch
    from src.networks import build_cvpn

    cvpn = build_cvpn(hidden_dim=64, num_blocks=1, validate_inputs=False)
    search = SoGSearch(cvpn, train_budget=10, eval_budget=50, c_puct=1.0)

    # Initially on train budget
    assert search._current_budget == 10

    search.use_eval_budget()
    assert search._current_budget == 50

    search.use_train_budget()
    assert search._current_budget == 10


# ---------------------------------------------------------------------------
# End-to-end game vs random with GoEngine (skipped if FFI unavailable)
# ---------------------------------------------------------------------------


@skipgo
def test_sog_agent_full_game_vs_random():
    """SoGAgentWrapper (with GoEngine) completes a game vs random without error."""
    from src.evaluate_agents import SoGAgentWrapper, NeuralAgentWrapper
    from src.agents.baseline_agents import RandomAgent
    from src.game.engine import CambiaGameState

    config = _make_config(eval_budget=3)
    agents = [
        SoGAgentWrapper(
            player_id=0, config=config, checkpoint_path="", device="cpu",
            eval_budget=3, cfr_iters=2,
        ),
        RandomAgent(player_id=1, config=config),
    ]

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    for agent in agents:
        if isinstance(agent, NeuralAgentWrapper):
            agent.initialize_state(game_state)

    turns = 0
    while not game_state.is_terminal() and turns < 300:
        turns += 1
        pid = game_state.get_acting_player()
        if pid == -1:
            break
        legal = game_state.get_legal_actions()
        if not legal:
            break
        action = agents[pid].choose_action(game_state, legal)
        game_state.apply_action(action)

    assert game_state.is_terminal() or turns >= 300
    # Cleanup
    for agent in agents:
        if hasattr(agent, "_cleanup_search"):
            agent._cleanup_search()
