"""Comprehensive tests for the PPO best-response diagnostic feature.

Covers CambiaEnv (gymnasium wrapper), make_env factory, PPOAgentWrapper,
action-mask consistency, encoding consistency, and edge-case regressions.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

sb3_contrib = pytest.importorskip(
    "sb3_contrib", reason="sb3-contrib required for PPO tests"
)
from sb3_contrib import MaskablePPO  # noqa: E402

from src.encoding import (  # noqa: E402
    EP_PBS_INPUT_DIM,
    NUM_ACTIONS,
    action_to_index,
    encode_action_mask,
    encode_infoset_eppbs_interleaved,
)
from src.evaluate_agents import AGENT_REGISTRY, get_agent  # noqa: E402
from src.ppo_env import CambiaEnv, make_env  # noqa: E402


# ------------------------------------------------------------------
# Config stubs — mirror conftest pattern but with agent_params
# ------------------------------------------------------------------

@dataclass
class _GreedyAgentConfig:
    cambia_call_threshold: int = 5


@dataclass
class _AgentsConfig:
    greedy_agent: _GreedyAgentConfig = field(
        default_factory=_GreedyAgentConfig
    )


@dataclass
class _AgentParamsConfig:
    memory_level: int = 1
    time_decay_turns: int = 3


@dataclass
class _CambiaRulesConfig:
    allowDrawFromDiscardPile: bool = False
    allowReplaceAbilities: bool = False
    snapRace: bool = False
    penaltyDrawCount: int = 2
    use_jokers: int = 2
    cards_per_player: int = 4
    initial_view_count: int = 2
    cambia_allowed_round: int = 0
    allowOpponentSnapping: bool = False
    max_game_turns: int = 100
    lockCallerHand: bool = True


@dataclass
class _DeepCfrConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 2048
    train_steps_per_iteration: int = 4000
    alpha: float = 1.5
    traversals_per_step: int = 1000
    advantage_buffer_capacity: int = 2_000_000
    strategy_buffer_capacity: int = 2_000_000
    save_interval: int = 10
    device: str = "cpu"
    sampling_method: str = "outcome"
    exploration_epsilon: float = 0.6
    engine_backend: str = "python"
    es_validation_interval: int = 10
    es_validation_depth: int = 10
    es_validation_traversals: int = 1000
    pipeline_training: bool = True
    use_amp: bool = False
    use_compile: bool = False
    num_traversal_threads: int = 1
    validate_inputs: bool = True
    traversal_depth_limit: int = 0
    traversal_method: str = "outcome"
    value_hidden_dim: int = 512
    value_learning_rate: float = 1e-3
    value_buffer_capacity: int = 2_000_000
    batch_counterfactuals: bool = True
    use_sd_cfr: bool = False
    sd_cfr_max_snapshots: int = 200
    sd_cfr_snapshot_weighting: str = "linear"
    num_hidden_layers: int = 3
    use_residual: bool = True
    network_type: str = "residual"
    use_pos_embed: bool = True
    use_ema: bool = True
    encoding_mode: str = "legacy"
    encoding_layout: str = "auto"
    memory_archetype: str = "perfect"
    memory_decay_lambda: float = 0.1
    memory_capacity: int = 3
    num_players: int = 2
    target_buffer_passes: float = 0.0
    value_target_buffer_passes: float = 2.0


@dataclass
class _TestConfig:
    agents: _AgentsConfig = field(default_factory=_AgentsConfig)
    cambia_rules: _CambiaRulesConfig = field(
        default_factory=_CambiaRulesConfig
    )
    agent_params: _AgentParamsConfig = field(
        default_factory=_AgentParamsConfig
    )
    deep_cfr: _DeepCfrConfig = field(default_factory=_DeepCfrConfig)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def test_config():
    return _TestConfig()


@pytest.fixture
def env(test_config):
    """CambiaEnv with config injected directly (bypass YAML load)."""
    e = CambiaEnv(opponent_type="random", seed=42, agent_seat=0)
    e._config = test_config
    return e


@pytest.fixture
def env_seat1(test_config):
    """CambiaEnv where the PPO agent sits in seat 1."""
    e = CambiaEnv(opponent_type="random", seed=42, agent_seat=1)
    e._config = test_config
    return e


@pytest.fixture
def reset_env(env):
    """Pre-reset env ready for step calls."""
    env.reset()
    return env


def _run_episode(env, max_steps=500):
    """Play a full episode, returning (total_reward, steps, terminated)."""
    obs, info = env.reset()
    total_reward = 0.0
    for step_i in range(max_steps):
        mask = env.action_masks()
        legal_indices = np.where(mask)[0]
        action = int(np.random.choice(legal_indices))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            return total_reward, step_i + 1, True
    return total_reward, max_steps, False


# ==================================================================
# Unit Tests — CambiaEnv observation/action spaces
# ==================================================================


class TestEnvSpaces:
    def test_env_observation_space(self, env):
        """Observation space is Box(-5, 5, (224,), float32)."""
        assert env.observation_space.shape == (EP_PBS_INPUT_DIM,)
        assert env.observation_space.dtype == np.float32
        np.testing.assert_allclose(env.observation_space.low, -5.0)
        np.testing.assert_allclose(env.observation_space.high, 5.0)

    def test_env_action_space(self, env):
        """Action space is Discrete(146)."""
        assert env.action_space.n == NUM_ACTIONS
        assert env.action_space.n == 146


# ==================================================================
# Unit Tests — CambiaEnv reset
# ==================================================================


class TestEnvReset:
    def test_env_reset_returns_valid_obs(self, env):
        """reset() produces observation with shape (224,) and float32."""
        obs, info = env.reset()
        assert obs.shape == (EP_PBS_INPUT_DIM,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))

    def test_env_reset_returns_valid_mask(self, env):
        """After reset, action_masks() is bool array with >= 1 True."""
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool
        assert mask.any(), "At least one action must be legal after reset"

    def test_env_multiple_resets(self, env):
        """Resetting 3 times produces valid obs each time."""
        for seed in [0, 1, 2]:
            obs, info = env.reset(seed=seed)
            assert obs.shape == (EP_PBS_INPUT_DIM,)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))


# ==================================================================
# Unit Tests — CambiaEnv step
# ==================================================================


class TestEnvStep:
    def test_env_step_valid_action(self, reset_env):
        """Stepping with a legal action returns correct tuple types."""
        mask = reset_env.action_masks()
        legal_idx = int(np.where(mask)[0][0])
        obs, reward, terminated, truncated, info = reset_env.step(legal_idx)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (EP_PBS_INPUT_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_env_step_illegal_action_fallback(self, reset_env):
        """Stepping with an illegal action doesn't crash (random fallback)."""
        mask = reset_env.action_masks()
        # Pick an index that is NOT legal
        illegal_indices = np.where(~mask)[0]
        if len(illegal_indices) == 0:
            pytest.skip("All actions are legal — cannot test fallback")
        action = int(illegal_indices[0])
        obs, reward, terminated, truncated, info = reset_env.step(action)
        assert obs.shape == (EP_PBS_INPUT_DIM,)
        assert isinstance(reward, float)

    def test_env_nonterminal_reward_zero(self, reset_env):
        """First step reward is 0.0 (games don't end in 1 action)."""
        mask = reset_env.action_masks()
        legal_idx = int(np.where(mask)[0][0])
        _, reward, terminated, _, _ = reset_env.step(legal_idx)
        if not terminated:
            assert reward == 0.0

    def test_env_terminal_reward_values(self, env):
        """Terminal reward is in {-1.0, 0.0, 1.0}."""
        total_reward, steps, terminated = _run_episode(env)
        assert terminated, "Episode should terminate within 500 steps"
        assert total_reward in {
            -1.0,
            0.0,
            1.0,
        }, f"Terminal reward {total_reward} not in {{-1, 0, 1}}"

    def test_env_full_episode_terminates(self, env):
        """Full episode completes within 500 steps."""
        _, steps, terminated = _run_episode(env, max_steps=500)
        assert terminated, f"Episode did not terminate within 500 steps"

    def test_env_truncated_always_false(self, env):
        """truncated is always False (no time limits in env)."""
        obs, _ = env.reset()
        for _ in range(200):
            mask = env.action_masks()
            legal_indices = np.where(mask)[0]
            action = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action)
            assert truncated is False
            if terminated:
                break


# ==================================================================
# Unit Tests — agent_seat variants
# ==================================================================


class TestEnvSeatVariants:
    def test_env_agent_seat_one(self, env_seat1):
        """CambiaEnv(agent_seat=1) works; reset+step produce valid results."""
        obs, info = env_seat1.reset()
        assert obs.shape == (EP_PBS_INPUT_DIM,)
        mask = env_seat1.action_masks()
        assert mask.any()
        legal_idx = int(np.where(mask)[0][0])
        obs2, reward, terminated, truncated, info = env_seat1.step(
            legal_idx
        )
        assert obs2.shape == (EP_PBS_INPUT_DIM,)

    def test_env_agent_seat_one_full_episode(self, env_seat1):
        """Full episode with agent_seat=1 terminates and has valid reward."""
        total_reward, steps, terminated = _run_episode(env_seat1)
        assert terminated
        assert total_reward in {-1.0, 0.0, 1.0}


# ==================================================================
# Unit Tests — different opponent types
# ==================================================================


class TestEnvOpponents:
    @pytest.mark.parametrize(
        "opp_type",
        ["random", "random_no_cambia", "imperfect_greedy"],
    )
    def test_env_different_opponents(self, test_config, opp_type):
        """Env works with different opponent types."""
        e = CambiaEnv(opponent_type=opp_type, seed=7)
        e._config = test_config
        total_reward, steps, terminated = _run_episode(e)
        assert terminated, f"Episode didn't terminate with opponent={opp_type}"
        assert total_reward in {-1.0, 0.0, 1.0}


# ==================================================================
# Unit Tests — action mask consistency
# ==================================================================


class TestActionMaskConsistency:
    def test_action_masks_match_legal_actions(self, reset_env):
        """Each True in mask maps to a valid GameAction via action_to_index."""
        gs = reset_env._game_state
        legal = list(gs.get_legal_actions())
        mask = reset_env.action_masks()

        legal_indices = {action_to_index(a) for a in legal}
        mask_indices = set(np.where(mask)[0].tolist())

        assert legal_indices == mask_indices, (
            f"Mask indices {mask_indices} != legal indices {legal_indices}"
        )

    def test_action_masks_all_false_never_happens(self, env):
        """Mask always has >= 1 True before terminal."""
        obs, _ = env.reset()
        for _ in range(300):
            mask = env.action_masks()
            gs = env._game_state
            if gs.is_terminal():
                break
            assert mask.any(), "Mask is all-False on non-terminal state"
            legal_indices = np.where(mask)[0]
            action = int(np.random.choice(legal_indices))
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    def test_env_mask_covers_all_legal_actions(self, reset_env):
        """Env mask True positions are exactly action_to_index(a) for all
        legal actions."""
        gs = reset_env._game_state
        legal = list(gs.get_legal_actions())
        mask = reset_env.action_masks()

        expected_mask = encode_action_mask(legal)
        np.testing.assert_array_equal(mask, expected_mask)


# ==================================================================
# Unit Tests — make_env factory
# ==================================================================


class TestMakeEnvFactory:
    def test_make_env_returns_callable(self):
        """make_env() returns a callable."""
        factory = make_env(opponent_type="random", seed=0)
        assert callable(factory)

    def test_make_env_produces_valid_env(self, test_config):
        """Calling the factory returns a CambiaEnv that can reset."""
        factory = make_env(opponent_type="random", seed=0)
        e = factory()
        assert isinstance(e, CambiaEnv)
        # Inject config to bypass YAML load
        e._config = test_config
        obs, info = e.reset()
        assert obs.shape == (EP_PBS_INPUT_DIM,)


# ==================================================================
# Integration Tests — PPOAgentWrapper and AGENT_REGISTRY
# ==================================================================


class TestPPOAgentRegistry:
    def test_ppo_agent_registry(self):
        """'ppo' is in AGENT_REGISTRY."""
        assert "ppo" in AGENT_REGISTRY

    def test_ppo_get_agent_requires_model_path(self, test_config):
        """get_agent('ppo') without path raises ValueError."""
        with pytest.raises(ValueError, match="model_path"):
            get_agent("ppo", player_id=0, config=test_config)


# ------------------------------------------------------------------
# Helper: train a tiny PPO model and save it
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ppo_model_path():
    """Train a 100-timestep MaskablePPO and return the saved path.

    Module-scoped so we only train once across all tests that need it.
    """
    from src.config import load_config

    config = load_config(
        os.path.join(
            os.path.dirname(__file__), "..", "config", "deep_train.yaml"
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        env = CambiaEnv(opponent_type="random", seed=42)
        env._config = config

        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",
            seed=42,
            policy_kwargs={"net_arch": [64, 64]},
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)
        save_path = os.path.join(tmpdir, "tiny_ppo")
        model.save(save_path)
        env.close()

        # Yield path (file is save_path.zip)
        yield save_path


class TestPPOAgentWrapperIntegration:
    def test_ppo_agent_wrapper_full_game(
        self, tiny_ppo_model_path, test_config
    ):
        """Load trained model and play a complete game via eval harness."""
        from src.game.engine import CambiaGameState
        from src.agent_state import AgentState, AgentObservation
        from src.constants import NUM_PLAYERS
        import copy

        agent = get_agent(
            "ppo",
            player_id=0,
            config=test_config,
            model_path=tiny_ppo_model_path,
        )

        gs = CambiaGameState(house_rules=test_config.cambia_rules)
        agent.initialize_state(gs)

        # Set up a simple opponent (random)
        opponent = get_agent("random", player_id=1, config=test_config)

        def _make_obs(game_state, acting_player, action):
            return AgentObservation(
                acting_player=acting_player,
                action=action,
                discard_top_card=game_state.get_discard_top(),
                player_hand_sizes=[
                    game_state.get_player_card_count(i)
                    for i in range(NUM_PLAYERS)
                ],
                stockpile_size=game_state.get_stockpile_size(),
                drawn_card=None,
                peeked_cards=None,
                snap_results=list(game_state.snap_results_log),
                did_cambia_get_called=(
                    game_state.cambia_caller_id is not None
                ),
                who_called_cambia=game_state.cambia_caller_id,
                is_game_over=game_state.is_terminal(),
                current_turn=game_state.get_turn_number(),
            )

        for step in range(500):
            if gs.is_terminal():
                break
            acting = gs.get_acting_player()
            legal = gs.get_legal_actions()

            if acting == 0:
                action = agent.choose_action(gs, legal)
            else:
                action = opponent.choose_action(gs, legal)

            gs.apply_action(action)
            obs = _make_obs(gs, acting, action)

            # Strip private info
            filtered = copy.copy(obs)
            filtered.drawn_card = None
            filtered.peeked_cards = None

            agent.update_state(filtered)

        assert gs.is_terminal(), "Game should terminate within 500 steps"
        utility = gs.get_utility(0)
        assert utility in {-1, 0, 1}


# ==================================================================
# Integration Tests — encoding consistency
# ==================================================================


class TestEncodingConsistency:
    def test_obs_encoding_matches_manual(self, reset_env):
        """Env's _get_obs() produces the same result as manually encoding
        the agent's infoset via encode_infoset_eppbs_interleaved."""
        env = reset_env
        st = env._agent_states[env._agent_seat]
        gs = env._game_state

        # Manually replicate what _get_obs does
        ctx = env._get_decision_context(gs)
        if st.cambia_caller is None:
            cambia_state = 2
        elif st.cambia_caller == env._agent_seat:
            cambia_state = 0
        else:
            cambia_state = 1

        manual_enc = encode_infoset_eppbs_interleaved(
            slot_tags=[
                t.value if hasattr(t, "value") else int(t)
                for t in st.slot_tags
            ],
            slot_buckets=[int(b) for b in st.slot_buckets],
            discard_top_bucket=(
                st.known_discard_top_bucket.value
                if hasattr(st.known_discard_top_bucket, "value")
                else int(st.known_discard_top_bucket)
            ),
            stock_estimate=(
                st.stockpile_estimate.value
                if hasattr(st.stockpile_estimate, "value")
                else int(st.stockpile_estimate)
            ),
            game_phase=(
                st.game_phase.value
                if hasattr(st.game_phase, "value")
                else int(st.game_phase)
            ),
            decision_context=(
                ctx.value if hasattr(ctx, "value") else int(ctx)
            ),
            cambia_state=cambia_state,
            own_hand_size=len(st.own_hand),
            opp_hand_size=st.opponent_card_count,
        ).astype(np.float32)

        env_obs = env._get_obs()
        np.testing.assert_array_equal(env_obs, manual_enc)


# ==================================================================
# Regression Tests
# ==================================================================


class TestRegressions:
    def test_env_step_after_terminal(self, env):
        """Calling step after game ends returns terminated=True, reward=0.0."""
        obs, _ = env.reset()
        # Play to completion
        for _ in range(500):
            mask = env.action_masks()
            legal_indices = np.where(mask)[0]
            action = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated, "Game should have ended"

        # Now step again — should return terminated=True, reward=0.0
        obs2, reward2, terminated2, truncated2, info2 = env.step(0)
        assert terminated2 is True
        assert reward2 == 0.0
        assert obs2.shape == (EP_PBS_INPUT_DIM,)

    def test_config_lazy_loading(self, test_config):
        """Config is not loaded until first reset."""
        e = CambiaEnv(opponent_type="random", seed=0)
        assert e._config is None
        # Inject config so reset works without YAML
        e._config = test_config
        e.reset()
        assert e._config is not None

    def test_env_reward_accumulation_single_nonzero(self, env):
        """Only the final step has nonzero reward; all others are 0.0."""
        obs, _ = env.reset()
        rewards = []
        for _ in range(500):
            mask = env.action_masks()
            legal_indices = np.where(mask)[0]
            action = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated:
                break

        nonzero_rewards = [r for r in rewards if r != 0.0]
        # At most 1 nonzero reward (the terminal one); ties give 0.0
        assert len(nonzero_rewards) <= 1, (
            f"Expected at most 1 nonzero reward, got {len(nonzero_rewards)}"
        )
        # All non-terminal steps should be 0.0
        for r in rewards[:-1]:
            assert r == 0.0

    def test_env_obs_within_bounds(self, env):
        """Observations stay within declared space bounds during episode."""
        obs, _ = env.reset()
        for _ in range(200):
            assert env.observation_space.contains(obs), (
                f"Obs out of bounds: min={obs.min()}, max={obs.max()}"
            )
            mask = env.action_masks()
            legal_indices = np.where(mask)[0]
            action = int(np.random.choice(legal_indices))
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

    def test_env_multiple_episodes_independent(self, env):
        """Playing multiple episodes gives potentially different outcomes."""
        results = []
        for seed in range(10):
            env.reset(seed=seed + 100)
            _, steps, terminated = _run_episode(env, max_steps=500)
            assert terminated
            results.append(steps)
        # With different seeds, we should get at least 2 distinct step counts
        assert len(set(results)) > 1, (
            "All 10 episodes had identical step counts — likely not random"
        )
