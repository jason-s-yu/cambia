"""
tests/test_gtcfr_integration.py

Integration tests for GT-CFR:
  - GTCFRAgentWrapper instantiation and choose_action
  - AGENT_REGISTRY registration
  - end-to-end pipeline (buffer → train step → finite loss)
  - vs-random gameplay smoke test (no GoEngine required)
"""

import numpy as np
import pytest
import torch

from src.evaluate_agents import AGENT_REGISTRY, GTCFRAgentWrapper, get_agent
from src.networks import build_cvpn, CVPN
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    expansion_budget: int = 3,
    cvpn_hidden_dim: int = 64,
    cvpn_num_blocks: int = 1,
):
    """Minimal mock Config with all fields needed by GTCFRAgentWrapper."""
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
    deep_cfr.gtcfr_expansion_budget = expansion_budget
    deep_cfr.gtcfr_cvpn_hidden_dim = cvpn_hidden_dim
    deep_cfr.gtcfr_cvpn_num_blocks = cvpn_num_blocks
    deep_cfr.gtcfr_c_puct = 2.0
    deep_cfr.gtcfr_cfr_iters_per_expansion = 2
    deep_cfr.gtcfr_buffer_capacity = 1000
    deep_cfr.gtcfr_value_loss_weight = 1.0
    deep_cfr.gtcfr_policy_loss_weight = 1.0
    deep_cfr.gtcfr_cvpn_learning_rate = 3e-4
    deep_cfr.gtcfr_games_per_epoch = 1
    deep_cfr.gtcfr_epochs = 1
    deep_cfr.batch_size = 4
    cfg.deep_cfr = deep_cfr

    return cfg


def _run_simple_game(agents, max_turns: int = 200):
    """Run one game with given agents using Python CambiaGameState."""
    from src.game.engine import CambiaGameState
    from src.evaluate_agents import NeuralAgentWrapper, CFRAgentWrapper

    config = agents[0].config
    game_state = CambiaGameState(house_rules=config.cambia_rules)

    for agent in agents:
        if isinstance(agent, (NeuralAgentWrapper, CFRAgentWrapper)):
            agent.initialize_state(game_state)

    turn = 0
    while not game_state.is_terminal() and turn < max_turns:
        turn += 1
        pid = game_state.get_acting_player()
        if pid == -1:
            break
        legal = game_state.get_legal_actions()
        if not legal:
            break
        action = agents[pid].choose_action(game_state, legal)
        game_state.apply_action(action)

    return game_state


# ---------------------------------------------------------------------------
# Registry test
# ---------------------------------------------------------------------------


def test_gtcfr_in_agent_registry():
    """'gtcfr' must be registered in AGENT_REGISTRY."""
    assert "gtcfr" in AGENT_REGISTRY
    assert AGENT_REGISTRY["gtcfr"] is GTCFRAgentWrapper


# ---------------------------------------------------------------------------
# Wrapper instantiation and choose_action
# ---------------------------------------------------------------------------


def test_gtcfr_eval_wrapper_instantiates():
    """GTCFRAgentWrapper initializes without error (no checkpoint)."""
    config = _make_config()
    wrapper = GTCFRAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")
    assert wrapper._cvpn is not None
    assert not wrapper._cvpn.training  # eval mode
    assert wrapper._range_p0.shape == (NUM_HAND_TYPES,)
    assert wrapper._range_p1.shape == (NUM_HAND_TYPES,)


def test_gtcfr_eval_wrapper_choose_action_fallback():
    """choose_action falls back to random when agent_state is None."""
    from src.constants import ActionDrawStockpile, ActionCallCambia

    config = _make_config()
    wrapper = GTCFRAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")
    legal_actions = {ActionDrawStockpile(), ActionCallCambia()}
    action = wrapper.choose_action(game_state=None, legal_actions=legal_actions)
    assert action in legal_actions


def test_gtcfr_eval_wrapper_choose_action_with_state():
    """GTCFRAgentWrapper.choose_action returns a valid GameAction with initialized state."""
    from src.game.engine import CambiaGameState

    config = _make_config()
    wrapper = GTCFRAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")

    game_state = CambiaGameState(house_rules=config.cambia_rules)
    wrapper.initialize_state(game_state)
    assert wrapper.agent_state is not None

    legal_actions = game_state.get_legal_actions()
    assert legal_actions

    action = wrapper.choose_action(game_state, legal_actions)
    assert action in legal_actions


def test_gtcfr_eval_wrapper_reset():
    """reset() restores ranges to uniform."""
    config = _make_config()
    wrapper = GTCFRAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")

    # Dirty the ranges
    wrapper._range_p0 = np.zeros(NUM_HAND_TYPES, dtype=np.float32)
    wrapper._range_p1 = np.zeros(NUM_HAND_TYPES, dtype=np.float32)

    wrapper.reset()

    assert abs(wrapper._range_p0.sum() - 1.0) < 1e-5
    assert abs(wrapper._range_p1.sum() - 1.0) < 1e-5


def test_get_agent_gtcfr():
    """get_agent('gtcfr') returns GTCFRAgentWrapper with no checkpoint."""
    config = _make_config()
    agent = get_agent("gtcfr", player_id=0, config=config, device="cpu")
    assert isinstance(agent, GTCFRAgentWrapper)


# ---------------------------------------------------------------------------
# End-to-end pipeline: buffer → train step → finite loss
# ---------------------------------------------------------------------------


def test_gtcfr_end_to_end():
    """Synthetic samples → insert into GTCFRTrainer buffers → 1 train step → finite loss."""
    from src.config import DeepCfrConfig
    from src.cfr.gtcfr_trainer import GTCFRTrainer, VALUE_DIM, POLICY_DIM
    from src.reservoir import ReservoirSample

    cfg = DeepCfrConfig()
    cfg.batch_size = 4
    cfg.device = "cpu"
    cfg.gtcfr_cvpn_hidden_dim = 64
    cfg.gtcfr_cvpn_num_blocks = 1
    cfg.gtcfr_expansion_budget = 3
    cfg.gtcfr_cfr_iters_per_expansion = 2
    cfg.gtcfr_c_puct = 2.0
    cfg.gtcfr_buffer_capacity = 100
    cfg.gtcfr_cvpn_learning_rate = 3e-4
    cfg.gtcfr_value_loss_weight = 1.0
    cfg.gtcfr_policy_loss_weight = 1.0
    cfg.gtcfr_games_per_epoch = 1
    cfg.gtcfr_epochs = 1
    cfg.gtcfr_exploration_epsilon = 0.5
    cfg.gtcfr_warm_start_rebel_checkpoint = ""

    trainer = GTCFRTrainer(config=cfg)

    # Insert synthetic samples directly into both buffers
    n_samples = 8
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        features = rng.standard_normal(PBS_INPUT_DIM).astype(np.float32)
        value_target = rng.standard_normal(VALUE_DIM).astype(np.float32)
        policy_target = rng.dirichlet(np.ones(POLICY_DIM)).astype(np.float32)
        mask = np.ones(POLICY_DIM, dtype=bool)

        trainer.value_buffer.add(ReservoirSample(
            features=features,
            target=value_target,
            action_mask=np.empty(0, dtype=bool),
            iteration=0,
        ))
        trainer.policy_buffer.add(ReservoirSample(
            features=features,
            target=policy_target,
            action_mask=mask,
            iteration=0,
        ))

    assert len(trainer.value_buffer) == n_samples
    assert len(trainer.policy_buffer) == n_samples

    # Run one train step
    v_loss, p_loss = trainer._train_step(num_steps=1)

    assert np.isfinite(v_loss), f"Value loss is not finite: {v_loss}"
    assert np.isfinite(p_loss), f"Policy loss is not finite: {p_loss}"


# ---------------------------------------------------------------------------
# vs-random smoke test (no GoEngine required)
# ---------------------------------------------------------------------------


def test_gtcfr_vs_random():
    """Run 5 games of GTCFRAgentWrapper (random init) vs RandomAgent. No crashes."""
    from src.evaluate_agents import RandomAgent

    config = _make_config(expansion_budget=3, cvpn_hidden_dim=64, cvpn_num_blocks=1)

    for game_num in range(5):
        gtcfr_agent = GTCFRAgentWrapper(player_id=0, config=config, checkpoint_path="", device="cpu")
        random_agent = RandomAgent(player_id=1, config=config)
        agents = [gtcfr_agent, random_agent]

        final_state = _run_simple_game(agents, max_turns=200)
        # Just verify the game completed without crashing
        # (may or may not be terminal if max_turns hit)
        assert final_state is not None, f"Game {game_num} returned None state"
