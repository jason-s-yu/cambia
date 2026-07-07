"""
Tests for src/disk_reservoir.py

Covers:
- Contract parity vs ReservoirBuffer on small capacities: add/add_batch (ragged
  variable-length rows), sample_batch shape/dtype (padded to batch max, not
  seq_cap), __len__, statistical (not exact-sequence) Vitter Algorithm R
  uniformity.
- Round-trip save/load resume, including mid-stream resume (seen_count/RNG
  state/pool cursors restored so post-resume fill decisions are correct).
- dtype/int-cast path into torch (feature dtype int16/int32, .long() cast,
  pack_padded_sequence compatibility via the lengths field).
- Overflow behavior at capacity.
- Pool compaction: dead space reclaimed, data integrity preserved across it.
- Throughput smoke test at ~1M ragged rows (marked slow).
"""

import json
import time

import numpy as np
import pytest

from src.disk_reservoir import DiskReservoir, FEATURE_DTYPE
from src.cfr.exceptions import ReservoirIOError
from src.reservoir import ColumnarBatch, ReservoirSample
from src.sequence_encoding import PAD_ID, SEQ_CAP, VOCAB_SIZE

torch = pytest.importorskip("torch")


SEQ_CAP_TEST = 32
TARGET_DIM_TEST = 146


def _make_ragged_batch(
    n,
    iteration_start=0,
    seq_cap=SEQ_CAP_TEST,
    target_dim=TARGET_DIM_TEST,
    seed=None,
    min_len=1,
    max_len=None,
):
    """Variable-length synthetic samples: features is a LIST of n 1D int
    arrays of independent length in [min_len, max_len], NOT a padded 2D array
    -- this is the add_batch contract."""
    rng = np.random.default_rng(seed)
    if max_len is None:
        max_len = seq_cap
    lengths = rng.integers(min_len, max_len + 1, size=n)
    features = [
        rng.integers(0, VOCAB_SIZE, size=int(ln)).astype(FEATURE_DTYPE) for ln in lengths
    ]
    targets = rng.standard_normal((n, target_dim)).astype(np.float32)
    masks = rng.integers(0, 2, size=(n, target_dim)).astype(bool)
    iterations = np.arange(iteration_start, iteration_start + n, dtype=np.int64)
    return features, targets, masks, iterations


def _make_sample(iteration=0, length=8, target_dim=TARGET_DIM_TEST, value=0):
    return ReservoirSample(
        features=np.full(length, value, dtype=FEATURE_DTYPE),
        target=np.full(target_dim, float(value), dtype=np.float32),
        action_mask=np.ones(target_dim, dtype=bool),
        iteration=iteration,
    )


# ===== Basic operations =====


class TestDiskReservoirBasic:
    def test_empty_buffer(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r1"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        assert len(res) == 0
        assert res.seen_count == 0
        assert res.pool_used == 0
        assert res.pool_dead == 0

    def test_add_single_sample(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r2"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        res.add(_make_sample(iteration=5, value=3, length=6))
        assert len(res) == 1
        assert res.seen_count == 1
        batch = res.sample_batch(1)
        assert batch.iterations[0] == 5
        assert batch.lengths[0] == 6
        assert np.all(batch.features[0, :6] == 3)

    def test_add_batch_below_capacity(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r3"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(30, seed=1)
        res.add_batch(features, targets, masks, iterations)
        assert len(res) == 30
        assert res.seen_count == 30
        assert res.pool_used == sum(len(f) for f in features)

    def test_add_batch_exceeds_capacity(self, tmp_path):
        """Buffer size stays at capacity when more items are added than fit."""
        res = DiskReservoir(
            str(tmp_path / "r4"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        for i in range(10):
            features, targets, masks, iterations = _make_ragged_batch(
                5, iteration_start=i * 5, seed=i
            )
            res.add_batch(features, targets, masks, iterations)
        assert len(res) == 20
        assert res.seen_count == 50

    def test_add_batch_empty_noop(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r5"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(0, seed=1)
        res.add_batch(features, targets, masks, iterations)
        assert len(res) == 0
        assert res.seen_count == 0

    def test_add_batch_requires_masks_when_has_mask(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r6"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            has_mask=True,
        )
        features, targets, _, iterations = _make_ragged_batch(3, seed=1)
        with pytest.raises(ValueError):
            res.add_batch(features, targets, masks=None, iterations=iterations)

    def test_no_mask_buffer(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r7"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            has_mask=False,
        )
        features, targets, _, iterations = _make_ragged_batch(5, seed=1)
        res.add_batch(features, targets, masks=None, iterations=iterations)
        assert len(res) == 5
        batch = res.sample_batch(5)
        assert batch.masks is None

        with pytest.raises(ValueError):
            _, _, masks, _ = _make_ragged_batch(2, seed=2)
            res.add_batch(
                features[:2], targets[:2], masks=masks[:2], iterations=iterations[:2]
            )

    def test_row_length_exceeds_seq_cap_raises(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "r8"), capacity=10, seq_cap=8, target_dim=TARGET_DIM_TEST
        )
        features = [np.zeros(9, dtype=FEATURE_DTYPE)]  # exceeds seq_cap=8
        targets = np.zeros((1, TARGET_DIM_TEST), dtype=np.float32)
        masks = np.ones((1, TARGET_DIM_TEST), dtype=bool)
        iterations = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError):
            res.add_batch(features, targets, masks, iterations)

    def test_variable_length_rows_preserved_exactly(self, tmp_path):
        """Different rows in the same add_batch call keep their own distinct
        lengths and content (the core ragged-storage property)."""
        res = DiskReservoir(
            str(tmp_path / "r9"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features = [
            np.array([1, 2, 3], dtype=FEATURE_DTYPE),
            np.array([4, 5], dtype=FEATURE_DTYPE),
            np.array([6, 7, 8, 9, 10], dtype=FEATURE_DTYPE),
        ]
        targets = np.zeros((3, TARGET_DIM_TEST), dtype=np.float32)
        masks = np.ones((3, TARGET_DIM_TEST), dtype=bool)
        iterations = np.array([0, 1, 2], dtype=np.int64)
        res.add_batch(features, targets, masks, iterations)

        batch = res.sample_batch(3)
        by_iter = {int(it): i for i, it in enumerate(batch.iterations.tolist())}
        assert batch.lengths[by_iter[0]] == 3
        assert batch.lengths[by_iter[1]] == 2
        assert batch.lengths[by_iter[2]] == 5
        np.testing.assert_array_equal(batch.features[by_iter[0], :3], [1, 2, 3])
        np.testing.assert_array_equal(batch.features[by_iter[1], :2], [4, 5])
        np.testing.assert_array_equal(batch.features[by_iter[2], :5], [6, 7, 8, 9, 10])


# ===== Sampling =====


class TestDiskReservoirSampling:
    def test_sample_batch_size(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s1"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(50, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(10)
        assert len(batch) == 10
        assert isinstance(batch, ColumnarBatch)

    def test_sample_batch_exceeds_buffer(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s2"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(3, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(100)
        assert len(batch) == 3

    def test_sample_batch_empty(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s3"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        batch = res.sample_batch(10)
        assert len(batch) == 0
        assert batch.targets.shape == (0, TARGET_DIM_TEST)
        assert batch.masks.shape == (0, TARGET_DIM_TEST)
        assert batch.lengths.shape == (0,)

    def test_sample_batch_padded_to_batch_max_not_seq_cap(self, tmp_path):
        """Features width must equal the longest row in the SAMPLED batch, not
        the seq_cap allocation bound."""
        res = DiskReservoir(
            str(tmp_path / "s4"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(
            40, seed=1, min_len=1, max_len=5
        )
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(40)
        assert batch.features.shape[1] == max(len(f) for f in features)
        assert batch.features.shape[1] < SEQ_CAP_TEST

    def test_sample_batch_padding_is_pad_id(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s5"),
            capacity=10,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features = [
            np.array([1, 2, 3], dtype=FEATURE_DTYPE),
            np.array([9], dtype=FEATURE_DTYPE),
        ]
        targets = np.zeros((2, TARGET_DIM_TEST), dtype=np.float32)
        masks = np.ones((2, TARGET_DIM_TEST), dtype=bool)
        iterations = np.array([0, 1], dtype=np.int64)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(2)
        by_iter = {int(it): i for i, it in enumerate(batch.iterations.tolist())}
        row = batch.features[by_iter[1]]
        assert row[0] == 9
        assert np.all(row[1:] == PAD_ID)

    def test_sample_batch_lengths_field_present_and_matches_content(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s6"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(20, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(20)
        assert batch.lengths is not None
        assert batch.lengths.dtype == np.int64
        for i in range(len(batch)):
            ln = int(batch.lengths[i])
            # Beyond ln, everything must be PAD_ID.
            assert np.all(batch.features[i, ln:] == PAD_ID)

    def test_sample_batch_no_duplicates(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s7"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(20, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(15)
        assert len(set(batch.iterations.tolist())) == 15

    def test_sample_batch_returns_copies_not_memmap_views(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "s8"),
            capacity=100,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(
            10, seed=1, min_len=5, max_len=5
        )
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(10)
        batch.features[:] = 0
        batch2 = res.sample_batch(10)
        assert not np.all(batch2.features == 0)


# ===== Reservoir sampling (Vitter Algorithm R) statistical property =====


class TestDiskReservoirUniformity:
    def test_reservoir_property_statistical(self, tmp_path):
        """Each item retained with ~uniform probability across many trials.

        Mirrors test_reservoir.py's equivalent ReservoirBuffer test (same
        capacity/total_items/num_trials scale, chosen there for a safety
        margin of several standard deviations against per-item binomial noise
        across many independently-tested items). Exact sequence parity with
        ReservoirBuffer is not required (different RNG source and ragged
        storage); only the uniformity property is asserted.
        """
        capacity = 100
        total_items = 1000
        num_trials = 500

        counts = np.zeros(total_items)
        for trial in range(num_trials):
            res = DiskReservoir(
                str(tmp_path / f"stat_{trial}"),
                capacity=capacity,
                seq_cap=SEQ_CAP_TEST,
                target_dim=TARGET_DIM_TEST,
                seed=trial,
            )
            features, targets, masks, iterations = _make_ragged_batch(
                total_items, seed=1000 + trial
            )
            res.add_batch(features, targets, masks, iterations)
            batch = res.sample_batch(capacity)
            for it in batch.iterations.tolist():
                counts[it] += 1

        expected = num_trials * capacity / total_items
        assert np.all(counts > expected * 0.4), "some items never retained"
        assert np.all(counts < expected * 1.6), "some items retained too often"

    def test_early_items_not_overrepresented(self, tmp_path):
        capacity = 40
        total = 400
        num_trials = 150

        early_count = 0
        late_count = 0
        for trial in range(num_trials):
            res = DiskReservoir(
                str(tmp_path / f"early_{trial}"),
                capacity=capacity,
                seq_cap=SEQ_CAP_TEST,
                target_dim=TARGET_DIM_TEST,
                seed=trial,
            )
            features, targets, masks, iterations = _make_ragged_batch(
                total, seed=2000 + trial
            )
            res.add_batch(features, targets, masks, iterations)
            batch = res.sample_batch(capacity)
            for it in batch.iterations.tolist():
                if it < total // 2:
                    early_count += 1
                else:
                    late_count += 1

        total_retained = early_count + late_count
        early_ratio = early_count / total_retained
        assert 0.4 < early_ratio < 0.6, f"early/late ratio {early_ratio} too skewed"

    def test_incremental_add_batch_matches_single_batch_statistics(self, tmp_path):
        """Adding in several small add_batch calls (as the trainer will, one
        traversal at a time) preserves the same uniformity as one big call --
        the per-row occurrence index must thread correctly across calls."""
        # Same capacity/total/trials scale as the fixed-batch uniformity test
        # above (chosen there for a several-sigma safety margin against
        # per-item binomial noise across many independently-tested items).
        capacity = 100
        total = 1000
        num_trials = 500
        chunk = 50

        counts = np.zeros(total)
        for trial in range(num_trials):
            res = DiskReservoir(
                str(tmp_path / f"chunked_{trial}"),
                capacity=capacity,
                seq_cap=SEQ_CAP_TEST,
                target_dim=TARGET_DIM_TEST,
                seed=trial,
            )
            for start in range(0, total, chunk):
                features, targets, masks, iterations = _make_ragged_batch(
                    chunk, iteration_start=start, seed=3000 + trial * 100 + start
                )
                res.add_batch(features, targets, masks, iterations)
            assert res.seen_count == total
            batch = res.sample_batch(capacity)
            for it in batch.iterations.tolist():
                counts[it] += 1

        expected = num_trials * capacity / total
        assert np.all(
            counts > expected * 0.4
        ), "some items never retained across chunked adds"
        assert np.all(
            counts < expected * 1.6
        ), "some items over-retained across chunked adds"


# ===== Save / load round trip =====


class TestDiskReservoirSaveLoad:
    def test_save_writes_meta_json(self, tmp_path):
        path = tmp_path / "sl1"
        res = DiskReservoir(
            str(path), capacity=50, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        features, targets, masks, iterations = _make_ragged_batch(10, seed=1)
        res.add_batch(features, targets, masks, iterations)
        res.save()
        meta_path = path / "meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["count"] == 10
        assert meta["seen_count"] == 10
        assert meta["capacity"] == 50
        assert meta["feature_dtype"] == np.dtype(FEATURE_DTYPE).name
        assert meta["pool_used"] == sum(len(f) for f in features)

    def test_load_missing_meta_raises(self, tmp_path):
        path = tmp_path / "sl2"
        res = DiskReservoir(
            str(path), capacity=10, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        with pytest.raises(ReservoirIOError):
            res.load()

    def test_round_trip_same_directory_resume(self, tmp_path):
        """The primary resume flow: a fresh process constructs against the same
        directory (re-opening the same memmap files) and calls load() to
        restore count/seen_count/pool cursors/RNG state."""
        path = tmp_path / "sl3"
        capacity = 20

        res1 = DiskReservoir(
            str(path),
            capacity=capacity,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=7,
        )
        for i in range(5):
            features, targets, masks, iterations = _make_ragged_batch(
                10, iteration_start=i * 10, seed=i
            )
            res1.add_batch(features, targets, masks, iterations)
        assert len(res1) == capacity
        assert res1.seen_count == 50
        res1.save()

        res2 = DiskReservoir(
            str(path), capacity=capacity, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        assert len(res2) == 0  # fresh instance before load()
        res2.load()
        assert len(res2) == capacity
        assert res2.seen_count == 50
        assert res2.pool_used == res1.pool_used
        assert res2.pool_dead == res1.pool_dead

        batch1 = res1.sample_batch(capacity)
        batch2 = res2.sample_batch(capacity)
        assert set(batch1.iterations.tolist()) == set(batch2.iterations.tolist())

    def test_resume_continues_filling_correctly(self, tmp_path):
        """After a resume, further add_batch calls must continue the Vitter
        Algorithm R bookkeeping (seen_count, RNG, pool cursors) rather than
        restarting -- the core 'Load must resume filling correctly' requirement."""
        path = tmp_path / "sl4"
        capacity = 10

        res1 = DiskReservoir(
            str(path),
            capacity=capacity,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=1,
        )
        features, targets, masks, iterations = _make_ragged_batch(
            10, iteration_start=0, seed=1
        )
        res1.add_batch(features, targets, masks, iterations)
        res1.save()
        del res1

        res2 = DiskReservoir(
            str(path), capacity=capacity, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        res2.load()
        assert res2.seen_count == 10
        assert len(res2) == 10

        # Buffer is already full: subsequent adds must go through the
        # replace-phase branch (occurrence index continuing from 10, not 0).
        features2, targets2, masks2, iterations2 = _make_ragged_batch(
            50, iteration_start=100, seed=2
        )
        res2.add_batch(features2, targets2, masks2, iterations2)
        assert len(res2) == capacity  # still capped
        assert res2.seen_count == 60

        batch = res2.sample_batch(capacity)
        assert any(it >= 100 for it in batch.iterations.tolist())

    def test_save_load_different_directory_copies_data(self, tmp_path):
        src = tmp_path / "orig"
        dst = tmp_path / "snapshot"
        res = DiskReservoir(
            str(src),
            capacity=15,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=3,
        )
        features, targets, masks, iterations = _make_ragged_batch(15, seed=1)
        res.add_batch(features, targets, masks, iterations)
        res.save(str(dst))

        assert (dst / "meta.json").exists()
        assert (dst / "pool.mm").exists()

        res2 = DiskReservoir(
            str(dst), capacity=15, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        res2.load()
        assert len(res2) == 15
        assert res2.seen_count == 15

    def test_load_dimension_mismatch_raises(self, tmp_path):
        path = tmp_path / "sl5"
        res = DiskReservoir(
            str(path), capacity=10, seq_cap=SEQ_CAP_TEST, target_dim=TARGET_DIM_TEST
        )
        features, targets, masks, iterations = _make_ragged_batch(5, seed=1)
        res.add_batch(features, targets, masks, iterations)
        res.save()

        # Constructing directly against the same path but a different seq_cap;
        # the mismatch must surface at load() time (the same-directory branch
        # validates dims against meta.json -- the pool file itself carries no
        # seq_cap-derived shape, so construction alone would not catch this).
        res2 = DiskReservoir(
            str(path), capacity=10, seq_cap=SEQ_CAP_TEST + 1, target_dim=TARGET_DIM_TEST
        )
        with pytest.raises(ReservoirIOError):
            res2.load()


# ===== Pool compaction =====


class TestDiskReservoirCompaction:
    def test_compaction_reclaims_dead_space_and_preserves_data(self, tmp_path):
        """Force compaction with an aggressive threshold, verify pool_dead
        resets and every currently-live row's content still round-trips."""
        res = DiskReservoir(
            str(tmp_path / "c1"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=5,
            compaction_threshold=0.3,
            compaction_min_tokens=50,
            initial_pool_tokens=64,
        )
        # Fill, then overflow heavily so many replacements retire pool space.
        for i in range(60):
            features, targets, masks, iterations = _make_ragged_batch(
                5, iteration_start=i * 5, seed=100 + i, min_len=3, max_len=10
            )
            res.add_batch(features, targets, masks, iterations)

        assert len(res) == 20
        # At least one compaction should have fired given the aggressive settings.
        assert res.pool_used < res.pool_capacity or res.pool_dead == 0

        # Data integrity: every live slot's sampled content must match its
        # recorded length and be internally consistent (no cross-row bleed).
        batch = res.sample_batch(20)
        for i in range(len(batch)):
            ln = int(batch.lengths[i])
            assert 3 <= ln <= 10
            assert np.all(batch.features[i, ln:] == PAD_ID)

    def test_compaction_then_further_adds_still_correct(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "c2"),
            capacity=15,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=6,
            compaction_threshold=0.2,
            compaction_min_tokens=20,
            initial_pool_tokens=32,
        )
        for i in range(40):
            features, targets, masks, iterations = _make_ragged_batch(
                5, iteration_start=i * 5, seed=200 + i, min_len=2, max_len=8
            )
            res.add_batch(features, targets, masks, iterations)

        # More adds after compaction(s) have (likely) occurred.
        features, targets, masks, iterations = _make_ragged_batch(
            30, iteration_start=10_000, seed=999, min_len=1, max_len=8
        )
        res.add_batch(features, targets, masks, iterations)
        assert len(res) == 15
        batch = res.sample_batch(15)
        assert len(batch) == 15
        for i in range(len(batch)):
            ln = int(batch.lengths[i])
            assert np.all(batch.features[i, ln:] == PAD_ID)


# ===== dtype / torch cast path =====


class TestDiskReservoirTorchCast:
    def test_features_are_integer_dtype(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "t1"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(5, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(5)
        assert np.issubdtype(batch.features.dtype, np.integer)

    def test_torch_from_numpy_long_cast(self, tmp_path):
        """Mirrors prtcfr_trainer.py: torch.from_numpy(batch.features).long()."""
        res = DiskReservoir(
            str(tmp_path / "t2"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(8, seed=1)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(8)

        tk = torch.from_numpy(batch.features).long()
        assert tk.dtype == torch.int64
        assert tk.shape[0] == 8
        np.testing.assert_array_equal(tk.numpy(), batch.features.astype(np.int64))

        tgt = torch.from_numpy(batch.targets).float()
        assert tgt.dtype == torch.float32
        mk = torch.from_numpy(batch.masks)
        assert mk.dtype == torch.bool
        it = torch.from_numpy(batch.iterations.astype(np.float32))
        assert it.dtype == torch.float32

    def test_pack_padded_sequence_compatibility(self, tmp_path):
        """The lengths field must work directly with
        torch.nn.utils.rnn.pack_padded_sequence (the trainer's stated use)."""
        res = DiskReservoir(
            str(tmp_path / "t3"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features, targets, masks, iterations = _make_ragged_batch(
            12, seed=1, min_len=2, max_len=SEQ_CAP_TEST
        )
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(12)

        tokens = torch.from_numpy(batch.features).long()
        lengths = torch.from_numpy(batch.lengths)
        embed = torch.nn.Embedding(VOCAB_SIZE, 8)
        embedded = embed(tokens)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        assert packed.data.shape[0] == int(batch.lengths.sum())

    def test_token_ids_within_vocab_round_trip(self, tmp_path):
        """Token ids up to VOCAB_SIZE-1 survive the int16 storage cast intact."""
        res = DiskReservoir(
            str(tmp_path / "t4"),
            capacity=5,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        features = [np.array([0, 1, VOCAB_SIZE - 1, 5], dtype=np.int32)]
        targets = np.zeros((1, TARGET_DIM_TEST), dtype=np.float32)
        masks = np.ones((1, TARGET_DIM_TEST), dtype=bool)
        iterations = np.array([0], dtype=np.int64)
        res.add_batch(features, targets, masks, iterations)
        batch = res.sample_batch(1)
        assert batch.features[0, 2] == VOCAB_SIZE - 1


# ===== Overflow behavior =====


class TestDiskReservoirOverflow:
    def test_overflow_caps_at_capacity(self, tmp_path):
        res = DiskReservoir(
            str(tmp_path / "o1"),
            capacity=25,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
        )
        for i in range(20):
            features, targets, masks, iterations = _make_ragged_batch(
                10, iteration_start=i * 10, seed=i
            )
            res.add_batch(features, targets, masks, iterations)
        assert len(res) == 25
        assert res.seen_count == 200

    def test_overflow_all_slots_eventually_reassigned(self, tmp_path):
        """After heavy overflow, sampled iterations should span a wide range of
        the stream, not just the earliest fill-phase items (regression guard
        against a fill-only bug that never actually replaces)."""
        res = DiskReservoir(
            str(tmp_path / "o2"),
            capacity=20,
            seq_cap=SEQ_CAP_TEST,
            target_dim=TARGET_DIM_TEST,
            seed=42,
        )
        features, targets, masks, iterations = _make_ragged_batch(5000, seed=42)
        res.add_batch(features, targets, masks, iterations)
        assert len(res) == 20
        assert res.seen_count == 5000
        batch = res.sample_batch(20)
        assert batch.iterations.max() > 100


# ===== Throughput smoke test =====


class TestDiskReservoirThroughput:
    def test_throughput_1m_samples(self, tmp_path):
        """Rough add / sample_batch throughput at production-like dtypes,
        ragged lengths (mean ~400 tokens), and a 1M-row buffer. Not a hard
        pass/fail gate on speed -- reports numbers for the sprint record and
        the actual on-disk pool footprint against the ~16GB/20M-row target
        (linearly: ~800MB/1M-row); asserts both paths complete correctly and
        clear a generous sanity floor.
        """
        capacity = 1_000_000
        seq_cap = SEQ_CAP  # 256 today; production plans to raise this to ~2048
        target_dim = TARGET_DIM_TEST
        add_batch_size = 8192

        res = DiskReservoir(
            str(tmp_path / "throughput"),
            capacity=capacity,
            seq_cap=max(seq_cap, 2048),
            target_dim=target_dim,
            seed=0,
        )

        n_batches = capacity // add_batch_size + 1
        t0 = time.perf_counter()
        added = 0
        total_tokens = 0
        for b in range(n_batches):
            n = min(add_batch_size, capacity - added)
            if n <= 0:
                break
            # Mean ~400 tokens/row (uniform 1..799), the production estimate.
            features, targets, masks, iterations = _make_ragged_batch(
                n,
                iteration_start=added,
                seq_cap=res.seq_cap,
                target_dim=target_dim,
                seed=b,
                min_len=1,
                max_len=799,
            )
            total_tokens += sum(len(f) for f in features)
            res.add_batch(features, targets, masks, iterations)
            added += n
        add_elapsed = time.perf_counter() - t0
        add_rate = added / add_elapsed

        assert len(res) == capacity

        n_sample_calls = 50
        t1 = time.perf_counter()
        for _ in range(n_sample_calls):
            batch = res.sample_batch(add_batch_size)
            assert batch.features.shape[0] == add_batch_size
        sample_elapsed = time.perf_counter() - t1
        sample_rate = (n_sample_calls * add_batch_size) / sample_elapsed

        pool_bytes = res.pool_used * np.dtype(FEATURE_DTYPE).itemsize
        mean_len = total_tokens / added

        print(
            f"\n[DiskReservoir throughput] add: {added} samples in {add_elapsed:.2f}s "
            f"({add_rate:,.0f} samples/s); sample_batch: {n_sample_calls * add_batch_size} "
            f"samples in {sample_elapsed:.2f}s ({sample_rate:,.0f} samples/s); "
            f"mean_len={mean_len:.1f} tokens; pool footprint={pool_bytes / 1e6:.1f}MB "
            f"for {capacity:,} rows (target ~{800:.0f}MB at 1M rows / ~16GB at 20M rows)"
        )

        # Sanity floor, not a tight SLO: both paths should sustain at least
        # 50k samples/s on any reasonable dev machine.
        assert add_rate > 50_000
        assert sample_rate > 50_000
