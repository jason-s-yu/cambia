"""Tests for src/cfr/deep_trainer.py's reservoir RNG seeding (cambia-1809).

ReservoirBuffer.sample_batch drew from the unseeded process-global numpy RNG,
so Deep CFR minibatch composition had no way to be reproducible under a
config seed. DeepCFRConfig gained a ``seed`` field; DeepCFRTrainer builds a
dedicated Generator from it (``self._fit_rng``) and threads it into every
reservoir buffer's sample_batch/load call. config.py's DeepCfrConfig has no
matching YAML field yet (out of this ticket's scope), so ``seed`` reaches
DeepCFRConfig only via ``from_yaml_config``'s override mechanism or direct
construction until that follow-up lands.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.reservoir import ReservoirBuffer, ReservoirSample
from src.cfr.deep_trainer import DeepCFRConfig, DeepCFRTrainer


def _make_config():
    """Minimal Config-shaped stand-in: DeepCFRTrainer.__init__ only reads
    config.cfr_training and config.persistence off the top-level Config
    (mirrors tests/test_device_config.py's TestTrainerDeviceInit pattern)."""
    config = SimpleNamespace()
    config.cfr_training = SimpleNamespace(num_iterations=1, num_workers=1)
    config.persistence = SimpleNamespace(agent_data_save_path="test_ckpt.pt")
    return config


def _stub_deep_cfr_config_without_seed():
    """A stand-in for Config.deep_cfr with no `seed` attribute -- mirrors the
    real (pydantic) DeepCfrConfig in src/config.py today, which has no seed
    field yet; from_yaml_config reads it via getattr(deep_cfg, "seed", None)
    precisely for this case."""
    fields = dict(vars(DeepCFRConfig()))
    fields.pop("seed", None)
    return SimpleNamespace(**fields)


def _tiny_dcfr_config(**overrides):
    base = dict(device="cpu", hidden_dim=8, num_hidden_layers=1, num_traversal_threads=1)
    base.update(overrides)
    return DeepCFRConfig(**base)


def _make_sample(iteration=0, value=0.0):
    return ReservoirSample(
        features=np.full(INPUT_DIM, value, dtype=np.float32),
        target=np.full(NUM_ACTIONS, value, dtype=np.float32),
        action_mask=np.ones(NUM_ACTIONS, dtype=bool),
        iteration=iteration,
    )


class TestDeepCFRConfigSeed:
    def test_seed_defaults_to_none(self):
        assert DeepCFRConfig().seed is None

    def test_seed_round_trips(self):
        assert DeepCFRConfig(seed=42).seed == 42

    def test_from_yaml_config_defaults_seed_none(self):
        config = SimpleNamespace(deep_cfr=_stub_deep_cfr_config_without_seed())
        dcfr = DeepCFRConfig.from_yaml_config(config)
        assert dcfr.seed is None

    def test_from_yaml_config_override_sets_seed(self):
        config = SimpleNamespace(deep_cfr=_stub_deep_cfr_config_without_seed())
        dcfr = DeepCFRConfig.from_yaml_config(config, seed=7)
        assert dcfr.seed == 7


class TestDeepCFRTrainerFitRng:
    def test_seeded_config_builds_generator(self):
        dcfr = _tiny_dcfr_config(seed=123)
        trainer = DeepCFRTrainer(config=_make_config(), deep_cfr_config=dcfr)
        assert isinstance(trainer._fit_rng, np.random.Generator)

    def test_unseeded_config_leaves_rng_none_and_warns(self, caplog):
        dcfr = _tiny_dcfr_config()
        assert dcfr.seed is None
        with caplog.at_level(logging.WARNING):
            trainer = DeepCFRTrainer(config=_make_config(), deep_cfr_config=dcfr)
        assert trainer._fit_rng is None
        assert "cambia-1809" in caplog.text

    def test_same_seed_gives_identical_fit_rng_draws(self):
        trainer_a = DeepCFRTrainer(
            config=_make_config(), deep_cfr_config=_tiny_dcfr_config(seed=5)
        )
        trainer_b = DeepCFRTrainer(
            config=_make_config(), deep_cfr_config=_tiny_dcfr_config(seed=5)
        )
        draw_a = trainer_a._fit_rng.choice(1000, 20, replace=False)
        draw_b = trainer_b._fit_rng.choice(1000, 20, replace=False)
        np.testing.assert_array_equal(draw_a, draw_b)

    def test_different_seeds_give_different_fit_rng_draws(self):
        trainer_a = DeepCFRTrainer(
            config=_make_config(), deep_cfr_config=_tiny_dcfr_config(seed=5)
        )
        trainer_b = DeepCFRTrainer(
            config=_make_config(), deep_cfr_config=_tiny_dcfr_config(seed=6)
        )
        draw_a = trainer_a._fit_rng.choice(1000, 20, replace=False)
        draw_b = trainer_b._fit_rng.choice(1000, 20, replace=False)
        assert not np.array_equal(draw_a, draw_b)


class TestAdvantageBufferSampleReproducibility:
    """Exercises the fixed mechanism (ReservoirBuffer.sample_batch's rng
    kwarg) the way DeepCFRTrainer now drives it, without the heavier
    traversal/multiprocessing machinery a full trainer.train() needs."""

    @staticmethod
    def _populated_buffer():
        buf = ReservoirBuffer(capacity=200, input_dim=INPUT_DIM, target_dim=NUM_ACTIONS)
        for i in range(200):
            buf.add(_make_sample(iteration=i, value=float(i)))
        return buf

    def test_same_seed_same_minibatch_sequence(self):
        buf_a = self._populated_buffer()
        buf_b = self._populated_buffer()
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        for _ in range(5):
            batch_a = buf_a.sample_batch(16, rng=rng_a)
            batch_b = buf_b.sample_batch(16, rng=rng_b)
            np.testing.assert_array_equal(batch_a.iterations, batch_b.iterations)
            np.testing.assert_array_equal(batch_a.features, batch_b.features)

    def test_different_seeds_diverge(self):
        buf_a = self._populated_buffer()
        buf_b = self._populated_buffer()
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(2)
        batch_a = buf_a.sample_batch(16, rng=rng_a)
        batch_b = buf_b.sample_batch(16, rng=rng_b)
        assert not np.array_equal(batch_a.iterations, batch_b.iterations)
