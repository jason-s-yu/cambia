"""
tests/test_sog_worker.py

Tests for sog_worker.py (SoG self-play episode runner).

Tests run without Go FFI by mocking GoEngine, GoAgentState, and SoGSearch.
All shapes and dtypes are verified with synthetic data.
"""

import random
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import DeepCfrConfig
from src.cfr.gtcfr_worker import EpisodeSample
from src.cfr.sog_worker import sog_self_play_episode, _sog_batch_worker
from src.networks import build_cvpn
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from src.encoding import NUM_ACTIONS
from src.cfr.gtcfr_search import SearchResult

VALUE_DIM = 2 * NUM_HAND_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_fast_sog_config(**overrides) -> DeepCfrConfig:
    cfg = DeepCfrConfig()
    cfg.gtcfr_cvpn_hidden_dim = 32
    cfg.gtcfr_cvpn_num_blocks = 1
    cfg.sog_train_budget = 2
    cfg.sog_eval_budget = 5
    cfg.sog_c_puct = 2.0
    cfg.sog_cfr_iters_per_expansion = 1
    cfg.sog_max_persist_depth = 4
    cfg.sog_max_persist_handles = 64
    cfg.sog_safety_margin = 0.01
    cfg.sog_games_per_epoch = 1
    cfg.sog_epochs = 1
    cfg.sog_exploration_epsilon = 0.5
    cfg.device = "cpu"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_mock_game(num_steps: int = 3, legal_count: int = 3):
    game = MagicMock()
    call_count = {"n": 0}

    def is_terminal_side_effect():
        return call_count["n"] >= num_steps

    def apply_action_side_effect(action):
        call_count["n"] += 1

    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    mask[:legal_count] = 1

    game.is_terminal.side_effect = is_terminal_side_effect
    game.legal_actions_mask.return_value = mask.copy()
    game.decision_ctx.return_value = 0
    game.turn_number.return_value = 3
    game.discard_top.return_value = 2
    game.stock_len.return_value = 20
    game.acting_player.return_value = 0
    game.apply_action.side_effect = apply_action_side_effect
    game.update_both = MagicMock()
    game.__enter__ = lambda s: s
    game.__exit__ = MagicMock(return_value=False)
    return game


def make_mock_search_result():
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[:3] = 1.0 / 3
    root_values = np.random.rand(VALUE_DIM).astype(np.float32)
    return SearchResult(
        policy=policy,
        root_values=root_values,
        tree_size=5,
        depth_stats={"min": 0, "max": 2, "mean": 1.0},
    )


# ---------------------------------------------------------------------------
# Tests: EpisodeSample import
# ---------------------------------------------------------------------------


class TestEpisodeSampleImport:
    def test_episode_sample_importable_from_sog_worker(self):
        """EpisodeSample should be importable via gtcfr_worker (not redefined in sog_worker)."""
        from src.cfr.sog_worker import EpisodeSample as SogEpisodeSample
        from src.cfr.gtcfr_worker import EpisodeSample as GtcfrEpisodeSample
        assert SogEpisodeSample is GtcfrEpisodeSample

    def test_episode_sample_dataclass(self):
        s = EpisodeSample(
            features=np.zeros(PBS_INPUT_DIM, dtype=np.float32),
            value_target=np.zeros(VALUE_DIM, dtype=np.float32),
            policy_target=np.zeros(NUM_ACTIONS, dtype=np.float32),
            action_mask=np.zeros(NUM_ACTIONS, dtype=bool),
        )
        assert s.features.shape == (PBS_INPUT_DIM,)
        assert s.value_target.shape == (VALUE_DIM,)


# ---------------------------------------------------------------------------
# Tests: sog_self_play_episode
# ---------------------------------------------------------------------------


class TestSogSelfPlayEpisode:
    def _run_episode_mocked(self, num_steps: int = 3):
        config = make_fast_sog_config()
        cvpn = build_cvpn(
            hidden_dim=config.gtcfr_cvpn_hidden_dim,
            num_blocks=config.gtcfr_cvpn_num_blocks,
            validate_inputs=False,
        )
        cvpn.eval()

        mock_game = make_mock_game(num_steps=num_steps)
        mock_agent = MagicMock()
        mock_agent.close = MagicMock()
        mock_result = make_mock_search_result()

        mock_engine_cm = MagicMock()
        mock_engine_cm.__enter__ = MagicMock(return_value=mock_game)
        mock_engine_cm.__exit__ = MagicMock(return_value=False)

        mock_sog = MagicMock()
        mock_sog.search.return_value = mock_result
        mock_sog.get_tree.return_value = None
        mock_sog.cleanup = MagicMock()
        mock_sog.use_train_budget = MagicMock()

        with (
            patch("src.ffi.bridge.GoEngine", return_value=mock_engine_cm),
            patch("src.ffi.bridge.GoAgentState", return_value=mock_agent),
            patch("src.cfr.sog_worker.SoGSearch", return_value=mock_sog),
        ):
            samples = sog_self_play_episode(
                game_config=None,
                cvpn=cvpn,
                config=config,
                exploration_epsilon=0.5,
            )
        return samples, mock_sog

    def test_episode_produces_samples(self):
        samples, _ = self._run_episode_mocked(num_steps=3)
        assert len(samples) > 0

    def test_episode_sample_shapes(self):
        samples, _ = self._run_episode_mocked(num_steps=4)
        for s in samples:
            assert s.features.shape == (PBS_INPUT_DIM,)
            assert s.value_target.shape == (VALUE_DIM,)
            assert s.policy_target.shape == (NUM_ACTIONS,)
            assert s.action_mask.shape == (NUM_ACTIONS,)
            assert s.action_mask.dtype == bool
            assert s.features.dtype == np.float32
            assert s.value_target.dtype == np.float32
            assert s.policy_target.dtype == np.float32

    def test_episode_uses_train_budget(self):
        _, mock_sog = self._run_episode_mocked(num_steps=2)
        mock_sog.use_train_budget.assert_called()

    def test_episode_cleanup_called(self):
        _, mock_sog = self._run_episode_mocked(num_steps=2)
        mock_sog.cleanup.assert_called()

    def test_episode_threads_prior_tree(self):
        """search() should be called with prior_tree from get_tree() on 2nd+ call."""
        config = make_fast_sog_config()
        cvpn = build_cvpn(hidden_dim=32, num_blocks=1, validate_inputs=False)
        cvpn.eval()

        mock_game = make_mock_game(num_steps=2)
        mock_agent = MagicMock()
        mock_agent.close = MagicMock()

        # Return a fake tree on first call, then None
        fake_tree = MagicMock()
        mock_result = make_mock_search_result()

        call_count_tree = {"n": 0}

        def get_tree_side():
            call_count_tree["n"] += 1
            if call_count_tree["n"] == 1:
                return fake_tree
            return None

        mock_sog = MagicMock()
        mock_sog.search.return_value = mock_result
        mock_sog.get_tree.side_effect = get_tree_side
        mock_sog.cleanup = MagicMock()
        mock_sog.use_train_budget = MagicMock()

        mock_engine_cm = MagicMock()
        mock_engine_cm.__enter__ = MagicMock(return_value=mock_game)
        mock_engine_cm.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.ffi.bridge.GoEngine", return_value=mock_engine_cm),
            patch("src.ffi.bridge.GoAgentState", return_value=mock_agent),
            patch("src.cfr.sog_worker.SoGSearch", return_value=mock_sog),
        ):
            sog_self_play_episode(None, cvpn, config, exploration_epsilon=0.5)

        calls = mock_sog.search.call_args_list
        if len(calls) >= 2:
            # Second call should have prior_tree=fake_tree
            second_call_kwargs = calls[1][1]
            assert second_call_kwargs.get("prior_tree") is fake_tree


# ---------------------------------------------------------------------------
# Tests: _sog_batch_worker
# ---------------------------------------------------------------------------


class TestSogBatchWorker:
    def test_batch_worker_returns_list(self):
        """_sog_batch_worker should return a list (may be empty if episodes fail)."""
        config = make_fast_sog_config()
        cvpn = build_cvpn(hidden_dim=32, num_blocks=1, validate_inputs=False)
        cvpn_state = {k: v.numpy() for k, v in cvpn.state_dict().items()}

        fake_samples = [
            EpisodeSample(
                features=np.zeros(PBS_INPUT_DIM, dtype=np.float32),
                value_target=np.zeros(VALUE_DIM, dtype=np.float32),
                policy_target=np.zeros(NUM_ACTIONS, dtype=np.float32),
                action_mask=np.zeros(NUM_ACTIONS, dtype=bool),
            )
        ]

        with patch("src.cfr.sog_worker.sog_self_play_episode", return_value=fake_samples):
            result = _sog_batch_worker((2, cvpn_state, config, None))

        assert isinstance(result, list)
        assert len(result) == 2  # 2 episodes × 1 sample each
