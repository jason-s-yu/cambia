"""
tests/test_escher_agent.py

Tests for ESCHER Phase 0 implementation:
  - NeuralAgentWrapper base class (shared state management)
  - ESCHERAgentWrapper (PolicyNetwork inference)
  - DeepCFRAgentWrapper regression (still works after refactor)
  - Registry integration
  - run_head_to_head_typed() function
"""

import copy
import logging
import tempfile
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

# conftest.py handles config stub and sys.path injection
from src.networks import AdvantageNetwork, StrategyNetwork
from src.encoding import INPUT_DIM, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config():
    """Return a minimal config object for evaluation tests."""
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


def _make_deep_cfr_checkpoint(path: str):
    """Save a freshly-initialized Deep CFR checkpoint."""
    advantage_net = AdvantageNetwork(validate_inputs=False)
    strategy_net = StrategyNetwork(validate_inputs=False)
    checkpoint = {
        "advantage_net_state_dict": advantage_net.state_dict(),
        "strategy_net_state_dict": strategy_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {"hidden_dim": 256},
    }
    torch.save(checkpoint, path)


def _make_escher_checkpoint(path: str):
    """Save a freshly-initialized ESCHER checkpoint (strategy_net_state_dict key)."""
    strategy_net = StrategyNetwork(validate_inputs=False)
    checkpoint = {
        "strategy_net_state_dict": strategy_net.state_dict(),
        "training_step": 0,
        "total_traversals": 0,
        "dcfr_config": {"hidden_dim": 256},
    }
    torch.save(checkpoint, path)


@contextmanager
def _no_wrapper_fallback():
    """Fail if a wrapper logs an error while choosing inside this block.

    choose_action answers a random legal action when encoding or inference
    raises, and logs the reason at ERROR before it does. Without this, an
    assertion that the choice is legal passes whether or not the network was
    ever consulted, which is the failure mode the Go-engine move has to avoid
    reintroducing.
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.ERROR)
    logger = logging.getLogger("src.evaluate_agents")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert records == [], f"wrapper fell back instead of using its network: {records}"


# ---------------------------------------------------------------------------
# NeuralAgentWrapper base class tests
# ---------------------------------------------------------------------------


class TestNeuralAgentWrapperBase:
    def test_base_class_is_abstract(self):
        """NeuralAgentWrapper cannot be instantiated directly."""
        import abc
        from src.evaluate_agents import NeuralAgentWrapper

        assert hasattr(NeuralAgentWrapper, "__abstractmethods__")
        assert "choose_action" in NeuralAgentWrapper.__abstractmethods__

    def test_initialize_state_creates_agent_state(self):
        """initialize_state() should attach a Go belief to the wrapper."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.ffi.bridge import GoAgentState, GoEngine

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            assert agent.agent_state is None

            with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
                agent.initialize_state(game)

                assert agent.agent_state is not None
                assert isinstance(agent.agent_state, GoAgentState)
                assert agent.belief_handle() >= 0
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)

    def test_update_state_updates_agent_state(self):
        """update_state() should not raise and should keep the belief attached.

        Belief advances inside the engine now (cambia-1426), so update_state is
        a retained no-op for callers that still push an observation. What this
        holds is that such a caller neither raises nor detaches the belief. The
        observation is built here rather than through _create_observation, which
        belongs to the tabular CFRAgentWrapper and was never on this one.
        """
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentObservation
        from src.ffi.bridge import GoEngine

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")

            with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
                agent.initialize_state(game)
                handle_before = agent.belief_handle()

                obs = AgentObservation(
                    acting_player=game.acting_player(),
                    action=None,
                    discard_top_card=None,
                    player_hand_sizes=list(agent.agent_state.get_hand_lens()),
                    stockpile_size=game.stock_len(),
                    drawn_card=None,
                    peeked_cards=None,
                    snap_results=[],
                    did_cambia_get_called=False,
                    who_called_cambia=None,
                    is_game_over=False,
                    current_turn=agent.agent_state.get_current_turn(),
                )

                # update_state must not raise
                agent.update_state(obs)
                assert agent.agent_state is not None
                assert agent.belief_handle() == handle_before
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)

    def test_update_state_no_op_when_not_initialized(self):
        """update_state() is silent (no raise) when agent_state is None."""
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.agent_state import AgentObservation
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")
            game_state = CambiaGameState(house_rules=config.cambia_rules)

            obs = AgentObservation(
                acting_player=-1,
                action=None,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[4, 4],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=[],
                did_cambia_get_called=False,
                who_called_cambia=None,
                is_game_over=False,
                current_turn=0,
            )
            # Should not raise
            agent.update_state(obs)
            assert agent.agent_state is None
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# ESCHERAgentWrapper tests
# ---------------------------------------------------------------------------


class TestESCHERAgentWrapper:
    def test_loads_mock_escher_checkpoint(self):
        """ESCHERAgentWrapper loads StrategyNetwork from mock checkpoint."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
            assert hasattr(agent, "policy_net")
            assert isinstance(agent.policy_net, StrategyNetwork)
        finally:
            os.unlink(ckpt_path)

    def test_choose_action_returns_valid_legal_action(self):
        """choose_action() returns one of the legal actions after initialization."""
        from src.agents import action_codec
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.ffi.bridge import GoEngine

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

            with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
                agent.initialize_state(game)

                legal_actions = action_codec.actions_from_mask(game.legal_actions_mask())
                assert len(legal_actions) > 0

                with _no_wrapper_fallback():
                    chosen = agent.choose_action(game, legal_actions)
                assert chosen in legal_actions
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)

    def test_choose_action_fallback_when_no_agent_state(self):
        """choose_action() falls back to random when agent_state is None."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.game.engine import CambiaGameState

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
            # Do NOT call initialize_state: agent_state remains None

            game_state = CambiaGameState(house_rules=config.cambia_rules)
            legal_actions = game_state.get_legal_actions()
            assert len(legal_actions) > 0

            # Should not raise; falls back to random
            chosen = agent.choose_action(game_state, legal_actions)
            assert chosen in legal_actions
        finally:
            os.unlink(ckpt_path)

    def test_registered_in_agent_registry(self):
        """ESCHERAgentWrapper is registered as 'escher' in AGENT_REGISTRY."""
        from src.evaluate_agents import AGENT_REGISTRY, ESCHERAgentWrapper

        assert "escher" in AGENT_REGISTRY
        assert AGENT_REGISTRY["escher"] is ESCHERAgentWrapper

    def test_get_agent_escher_returns_wrapper(self):
        """get_agent('escher', ...) with checkpoint_path returns ESCHERAgentWrapper."""
        from src.evaluate_agents import get_agent, ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = get_agent(
                "escher", 0, config, checkpoint_path=ckpt_path, device="cpu"
            )
            assert isinstance(agent, ESCHERAgentWrapper)
        finally:
            os.unlink(ckpt_path)

    def test_get_agent_escher_raises_without_checkpoint(self):
        """get_agent('escher', ...) without checkpoint_path raises ValueError."""
        from src.evaluate_agents import get_agent

        config = _make_config()
        with pytest.raises(ValueError, match="checkpoint_path"):
            get_agent("escher", 0, config)

    def test_invalid_checkpoint_raises_error(self):
        """Loading from a non-existent path raises an appropriate error."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with pytest.raises(Exception):
            ESCHERAgentWrapper(0, config, "/nonexistent/path.pt", device="cpu")

    def test_invalid_checkpoint_missing_key_raises_error(self):
        """Checkpoint missing 'strategy_net_state_dict' raises KeyError."""
        from src.evaluate_agents import ESCHERAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            # Save checkpoint without the required key
            torch.save({"dcfr_config": {"hidden_dim": 256}}, ckpt_path)
            with pytest.raises(KeyError):
                ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# DeepCFRAgentWrapper regression tests
# ---------------------------------------------------------------------------


class TestDeepCFRAgentWrapperRegression:
    def test_registered_as_deep_cfr(self):
        """DeepCFRAgentWrapper is still registered as 'deep_cfr' in AGENT_REGISTRY."""
        from src.evaluate_agents import AGENT_REGISTRY, DeepCFRAgentWrapper

        assert "deep_cfr" in AGENT_REGISTRY
        assert AGENT_REGISTRY["deep_cfr"] is DeepCFRAgentWrapper

    def test_choose_action_returns_valid_action(self):
        """DeepCFRAgentWrapper.choose_action() returns a legal action (regression)."""
        from src.agents import action_codec
        from src.evaluate_agents import DeepCFRAgentWrapper
        from src.ffi.bridge import GoEngine

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = DeepCFRAgentWrapper(0, config, ckpt_path, device="cpu")

            with GoEngine(seed=7, house_rules=config.cambia_rules) as game:
                agent.initialize_state(game)

                legal_actions = action_codec.actions_from_mask(game.legal_actions_mask())
                assert len(legal_actions) > 0

                with _no_wrapper_fallback():
                    chosen = agent.choose_action(game, legal_actions)
                assert chosen in legal_actions
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)

    def test_inherits_from_neural_agent_wrapper(self):
        """DeepCFRAgentWrapper is a subclass of NeuralAgentWrapper (refactor check)."""
        from src.evaluate_agents import DeepCFRAgentWrapper, NeuralAgentWrapper

        assert issubclass(DeepCFRAgentWrapper, NeuralAgentWrapper)

    def test_get_agent_deep_cfr_returns_wrapper(self):
        """get_agent('deep_cfr', ...) still works correctly after refactor."""
        from src.evaluate_agents import get_agent, DeepCFRAgentWrapper

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_deep_cfr_checkpoint(ckpt_path)
            agent = get_agent(
                "deep_cfr", 0, config, checkpoint_path=ckpt_path, device="cpu"
            )
            assert isinstance(agent, DeepCFRAgentWrapper)
        finally:
            os.unlink(ckpt_path)


# ---------------------------------------------------------------------------
# Head-to-head typed tests
# ---------------------------------------------------------------------------


class TestRunHeadToHeadTyped:
    def test_completes_10_games(self):
        """run_head_to_head_typed() completes 10 games without error."""
        from src.evaluate_agents import run_head_to_head_typed

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name
        try:
            _make_escher_checkpoint(ckpt_a)
            _make_deep_cfr_checkpoint(ckpt_b)

            result = run_head_to_head_typed(
                agent_a_type="escher",
                checkpoint_a=ckpt_a,
                agent_b_type="deep_cfr",
                checkpoint_b=ckpt_b,
                num_games=10,
                config=config,
                device="cpu",
            )

            assert result["num_games"] == 10
            total = result["wins_a"] + result["wins_b"] + result["draws"]
            assert total + result.get("errors", 0) >= 10
            assert 0.0 <= result["win_rate_a"] <= 1.0
            assert 0.0 <= result["win_rate_b"] <= 1.0
            assert "avg_game_turns" in result
            assert "std_game_turns" in result
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)

    def test_seat_alternation(self):
        """Seat assignment alternates: odd games have A as P0, even games have B as P0.

        Read off which agent sits at seat 0 in each game, not off how many
        agents were built. Since cambia-1974 the four agents (each side at each
        seat) are constructed once for the whole match: building them inside
        the loop is what let a model load reseed the global random module
        between games, which left every game dealt from the same number.
        """
        from src.evaluate_agents import run_head_to_head_typed

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name

        try:
            _make_escher_checkpoint(ckpt_a)
            _make_escher_checkpoint(ckpt_b)

            instantiated: list = []
            seat0_per_game: list = []

            from src import evaluate_agents as ea

            real_get_agent_fn = ea.get_agent
            real_game = ea._GoEvalGame

            def mock_get_agent(agent_type, player_id, cfg, **kwargs):
                agent = real_get_agent_fn(agent_type, player_id, cfg, **kwargs)
                instantiated.append((player_id, kwargs.get("checkpoint_path", "")))
                return agent

            class _RecordingGame(real_game):
                def __init__(self, house_rules, seed, num_players, agents):
                    super().__init__(house_rules, seed, num_players, agents)
                    seat0_per_game.append(id(agents[0]))

            import unittest.mock as mock

            with mock.patch("src.evaluate_agents.get_agent", side_effect=mock_get_agent):
                with mock.patch.object(ea, "_GoEvalGame", _RecordingGame):
                    run_head_to_head_typed(
                        agent_a_type="escher",
                        checkpoint_a=ckpt_a,
                        agent_b_type="escher",
                        checkpoint_b=ckpt_b,
                        num_games=4,
                        config=config,
                        device="cpu",
                    )

            # One agent per (side, seat), built once for the match.
            assert len(instantiated) == 4, f"built {len(instantiated)} agents for 4 games"
            assert sorted(instantiated) == sorted(
                [(0, ckpt_a), (1, ckpt_a), (0, ckpt_b), (1, ckpt_b)]
            )

            # Odd games seat A at P0, even games seat B, so seat 0 alternates
            # between exactly two agents and neither holds it twice running.
            assert len(seat0_per_game) == 4
            assert seat0_per_game[0] == seat0_per_game[2], "A did not return to P0"
            assert seat0_per_game[1] == seat0_per_game[3], "B did not return to P0"
            assert seat0_per_game[0] != seat0_per_game[1], "seats did not alternate"
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)

    def test_returns_correct_keys(self):
        """run_head_to_head_typed() result dict has all required keys."""
        from src.evaluate_agents import run_head_to_head_typed

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fa:
            ckpt_a = fa.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fb:
            ckpt_b = fb.name
        try:
            _make_escher_checkpoint(ckpt_a)
            _make_escher_checkpoint(ckpt_b)

            result = run_head_to_head_typed(
                agent_a_type="escher",
                checkpoint_a=ckpt_a,
                agent_b_type="escher",
                checkpoint_b=ckpt_b,
                num_games=5,
                config=config,
                device="cpu",
            )

            required_keys = {
                "wins_a",
                "wins_b",
                "draws",
                "win_rate_a",
                "win_rate_b",
                "num_games",
                "errors",
                "avg_game_turns",
                "std_game_turns",
            }
            assert required_keys.issubset(result.keys())
        finally:
            os.unlink(ckpt_a)
            os.unlink(ckpt_b)


# ---------------------------------------------------------------------------
# ESCHER agent state reset tests
# ---------------------------------------------------------------------------


class TestESCHERAgentStateReset:
    """Verify ESCHER agent properly reinitializes state for new games."""

    def test_agent_state_fresh_per_game(self):
        """Agent state must be freshly initialized for each new game.

        After initialize_state() with a second game the wrapper must hold a
        brand-new belief that reflects the new game, not the old one. The
        belief is a GoAgentState now (cambia-1522), so game 1 is dirtied by
        playing it rather than by assigning to Python attributes, and the
        checks read the Go getters:
          1. the belief is attached and is a different object after re-init.
          2. the observation turn counter is back to zero, after game 1 moved
             it off zero.
          3. exactly initial_view_count slots carry a known bucket, the rest
             are Unknown, so no game-1 knowledge bled through.
        """
        from src.agents import action_codec
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.ffi.bridge import GoEngine, apply_games_batch

        #: engine/agent's BucketUnknown, the value the Go getter reports for a
        #: slot the seat knows nothing about. Python's CardBucket.UNKNOWN is 99
        #: and does not line up (see src/agents/go_belief_view.py).
        go_bucket_unknown = 9

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

            # --- Game 1: attach, then play it so belief accumulates ---
            with GoEngine(seed=11, house_rules=config.cambia_rules) as game1:
                agent.initialize_state(game1)
                assert agent.agent_state is not None
                state_after_game1 = agent.agent_state

                for _ in range(6):
                    if game1.is_terminal():
                        break
                    legal_mask = np.asarray(game1.legal_actions_mask())
                    idx = int(np.flatnonzero(legal_mask)[0])
                    apply_games_batch(
                        [game1.handle], [state_after_game1.handle], [-1], [idx]
                    )

                turn_after_game1 = state_after_game1.get_current_turn()
                assert turn_after_game1 > 0, (
                    "the first game did not advance the belief, so this test "
                    "would not detect a stale one"
                )

            # --- Game 2 ---
            with GoEngine(seed=12, house_rules=config.cambia_rules) as game2:
                agent.initialize_state(game2)

                assert (
                    agent.agent_state is not None
                ), "agent_state must not be None after re-init"

                # 1. Must be a fresh object, not the game-1 belief.
                assert agent.agent_state is not state_after_game1, (
                    "initialize_state() must attach a new GoAgentState, not "
                    "reuse the old one"
                )

                # 2. The observation counter is back to the start of a game.
                assert agent.agent_state.get_current_turn() == 0, (
                    "the re-attached belief carries game 1's turn counter "
                    f"({agent.agent_state.get_current_turn()} after "
                    f"{turn_after_game1} in game 1)"
                )

                # 3. Only the initial peek is known; nothing bled from game 1.
                own = agent.agent_state.get_own_hand_buckets_and_seen()
                own_len, _ = agent.agent_state.get_hand_lens()
                known = [
                    slot
                    for slot in range(own_len)
                    if int(own[slot, 0]) != go_bucket_unknown
                ]
                assert len(known) == config.cambia_rules.initial_view_count, (
                    f"slots {known} carry a known bucket after re-init; "
                    f"expected exactly {config.cambia_rules.initial_view_count} "
                    "from the initial peek"
                )
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)

    def test_agent_state_object_replaced_not_mutated(self):
        """initialize_state() must replace agent_state, not mutate it in place."""
        from src.evaluate_agents import ESCHERAgentWrapper
        from src.ffi.bridge import GoEngine

        config = _make_config()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            _make_escher_checkpoint(ckpt_path)
            agent = ESCHERAgentWrapper(0, config, ckpt_path, device="cpu")

            with GoEngine(seed=11, house_rules=config.cambia_rules) as game1:
                agent.initialize_state(game1)
                state1 = agent.agent_state

            with GoEngine(seed=12, house_rules=config.cambia_rules) as game2:
                agent.initialize_state(game2)
                state2 = agent.agent_state

                # Object identity, not the handle: attach_belief releases the
                # old handle first, so the pool is free to hand the same number
                # back to the new belief.
                assert state2 is not state1, (
                    "initialize_state() must assign a new GoAgentState instance, "
                    "not reuse the existing one"
                )
                agent.release_belief()
        finally:
            os.unlink(ckpt_path)
