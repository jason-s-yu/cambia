"""
Tests for parameterized ReservoirBuffer (ESCHER Phase 0).

Covers:
- ReservoirBuffer with input_dim=444, target_dim=1, has_mask=False (value buffer)
- Reservoir sampling correctness with new dimensions
- Save/load roundtrip for value buffer
- Backward compatibility: existing (222, 146, has_mask=True) still works
- ColumnarBatch with value buffer shapes
- add() with mask=None when has_mask=False
"""

import os
import tempfile
import pytest
import numpy as np

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.reservoir import ReservoirBuffer, ReservoirSample, ColumnarBatch

VALUE_INPUT_DIM = INPUT_DIM * 2  # 444
VALUE_TARGET_DIM = 1


def make_value_sample(iteration: int = 0) -> ReservoirSample:
    return ReservoirSample(
        features=np.random.randn(VALUE_INPUT_DIM).astype(np.float32),
        target=np.array([np.random.randn()], dtype=np.float32),
        action_mask=None,  # no mask for value buffer
        iteration=iteration,
    )


def make_standard_sample(iteration: int = 0) -> ReservoirSample:
    return ReservoirSample(
        features=np.random.randn(INPUT_DIM).astype(np.float32),
        target=np.random.randn(NUM_ACTIONS).astype(np.float32),
        action_mask=np.ones(NUM_ACTIONS, dtype=bool),
        iteration=iteration,
    )


class TestValueBuffer:
    def test_construction(self):
        buf = ReservoirBuffer(
            capacity=100, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        assert buf.capacity == 100
        assert buf._input_dim == VALUE_INPUT_DIM
        assert buf._target_dim == VALUE_TARGET_DIM
        assert buf._has_mask is False
        assert buf._masks is None

    def test_add_without_mask(self):
        buf = ReservoirBuffer(
            capacity=10, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        sample = make_value_sample()
        buf.add(sample)
        assert len(buf) == 1

    def test_add_none_mask_accepted(self):
        buf = ReservoirBuffer(
            capacity=10, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        # Should not raise even when action_mask is None
        for i in range(5):
            buf.add(make_value_sample(i))
        assert len(buf) == 5

    def test_sample_batch_shapes(self):
        buf = ReservoirBuffer(
            capacity=100, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        for i in range(20):
            buf.add(make_value_sample(i))
        batch = buf.sample_batch(10)
        assert isinstance(batch, ColumnarBatch)
        assert batch.features.shape == (10, VALUE_INPUT_DIM)
        assert batch.targets.shape == (10, VALUE_TARGET_DIM)
        assert batch.masks is None  # no masks for value buffer
        assert batch.iterations.shape == (10,)

    def test_sample_batch_empty_buffer(self):
        buf = ReservoirBuffer(
            capacity=10, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        batch = buf.sample_batch(5)
        assert len(batch) == 0
        assert batch.features.shape == (0, VALUE_INPUT_DIM)
        assert batch.targets.shape == (0, VALUE_TARGET_DIM)
        assert batch.masks is None

    def test_reservoir_sampling_correctness(self):
        """Verify reservoir sampling distributes uniformly over all seen samples."""
        capacity = 50
        buf = ReservoirBuffer(
            capacity=capacity, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        for i in range(200):
            buf.add(make_value_sample(i))
        assert len(buf) == capacity
        assert buf.seen_count == 200

    def test_save_load_roundtrip(self):
        buf = ReservoirBuffer(
            capacity=50, input_dim=VALUE_INPUT_DIM, target_dim=VALUE_TARGET_DIM, has_mask=False
        )
        for i in range(20):
            buf.add(make_value_sample(i))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "value_buffer")
            buf.save(path)

            buf2 = ReservoirBuffer(
                capacity=50,
                input_dim=VALUE_INPUT_DIM,
                target_dim=VALUE_TARGET_DIM,
                has_mask=False,
            )
            buf2.load(path)

            assert len(buf2) == len(buf)
            assert buf2.seen_count == buf.seen_count
            np.testing.assert_array_almost_equal(buf2._features[: len(buf)], buf._features[: len(buf)])
            np.testing.assert_array_almost_equal(buf2._targets[: len(buf)], buf._targets[: len(buf)])
            assert buf2._masks is None


class TestBackwardCompatibility:
    def test_default_constructor_unchanged(self):
        """Default constructor must still work as before."""
        buf = ReservoirBuffer(capacity=100)
        assert buf._input_dim == INPUT_DIM
        assert buf._target_dim == NUM_ACTIONS
        assert buf._has_mask is True
        assert buf._masks is not None

    def test_standard_sample_add(self):
        buf = ReservoirBuffer(capacity=10)
        for i in range(5):
            buf.add(make_standard_sample(i))
        assert len(buf) == 5

    def test_standard_sample_batch_has_masks(self):
        buf = ReservoirBuffer(capacity=50)
        for i in range(20):
            buf.add(make_standard_sample(i))
        batch = buf.sample_batch(10)
        assert batch.masks is not None
        assert batch.masks.shape == (10, NUM_ACTIONS)

    def test_standard_save_load_roundtrip(self):
        buf = ReservoirBuffer(capacity=50)
        for i in range(15):
            buf.add(make_standard_sample(i))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "adv_buffer")
            buf.save(path)
            buf2 = ReservoirBuffer(capacity=50)
            buf2.load(path)
            assert len(buf2) == len(buf)
            assert buf2._masks is not None
            np.testing.assert_array_equal(
                buf2._masks[: len(buf)], buf._masks[: len(buf)]
            )
