"""
src/disk_reservoir.py

Disk-backed reservoir buffer for PRT-CFR production training (Phase 2, v0.4).

``ReservoirBuffer`` (src/reservoir.py) pre-allocates fixed-width columnar
arrays in RAM, which is fine at the ~2M-sample scale of earlier phases but
does not fit a 20M-sample-per-player reservoir of packed token sequences
(contract.md AC5), nor the fact that full-game token sequences vary widely in
length (mean ~400, cap ~2048) -- a fixed seq_cap-width row would waste most of
its allocation on padding at scale.

``DiskReservoir`` therefore uses RAGGED storage: each row's tokens are stored
at their natural length in an append-only int16 pool, with a fixed per-slot
(offset, length) indirection table pointing into it. Vitter's Algorithm R
semantics are preserved exactly: a uniform random sample over the full stream
regardless of how many items have been seen, independent of row length.

Public contract parity with ReservoirBuffer (see reservoir.py for the
authoritative in-RAM reference):
  - ``add(sample)``: single-sample insert via reservoir sampling.
  - ``add_batch(features, targets, masks, iterations)``: batched insert (the
    production entry point -- avoids per-sample Python overhead and per-sample
    fsync; a whole traversal's samples are appended in one call). ``features``
    is a sequence of n variable-length 1D int array-likes (NOT padded); each
    row's own length is inferred and stored alongside it.
  - ``sample_batch(batch_size) -> ColumnarBatch``: features are a dense int
    array padded to the LONGEST ROW IN THE SAMPLED BATCH (not to seq_cap), plus
    a ``lengths`` field (see reservoir.ColumnarBatch) for pack_padded_sequence.
    targets/masks/iterations are unchanged: float32[target_dim] / bool[target_dim]
    / int64.
  - ``__len__``: current sample count.
  - ``save()`` / ``load()``: persist/restore metadata (count, seen_count,
    pool bookkeeping, RNG state). The memmap files themselves ARE the storage
    -- there is no full-array serialization of 20M rows.

Ragged storage scheme (slot-indirection into an append-only pool + periodic
compaction; chosen per the sprint's window-semantics amendment, which named
this the acceptable simple-and-correct option over per-slot max-length
segments -- the latter would cost capacity * seq_cap regardless of actual
length, e.g. 20M * 2048 * 2 bytes = ~82GB, blowing the ~16GB footprint target
at the real mean length of ~400 tokens):
  - ``pool`` (features.mm... see below): a growable int16 memmap, append-only.
    New/replacing rows are always appended at the current write cursor
    (``pool_used``); in-place overwrite is not attempted (rows differ in
    length, and partial overwrites of variable-length spans are a correctness
    hazard not worth the complexity here).
  - ``offsets`` / ``lengths``: fixed-size (capacity,) memmaps, one entry per
    reservoir slot, pointing into ``pool``.
  - Dead space: every Algorithm R *replacement* (not fill) retires the
    replaced slot's previous pool span (its bytes remain in the file but are
    no longer referenced by any slot) -- tracked in ``pool_dead``. When
    ``pool_dead`` crosses ``compaction_threshold`` of ``pool_used`` (and
    ``pool_used`` exceeds ``compaction_min_tokens``), a compaction pass
    rewrites all live rows into a fresh, tightly-packed pool file and swaps it
    in atomically. This bounds long-run disk usage close to the live working
    set (capacity * mean_len) rather than growing without bound across many
    replacement cycles.

Offset/length dtype note (deviates from the literal "int32 offsets/lengths"
phrasing of the amendment): ``lengths`` is int32 (rows are bounded by seq_cap,
~2048, comfortably within int32 -- or even int16). ``offsets`` is int64: at
the ~16GB / 20M-row / mean-400-token footprint target, the pool holds on the
order of 8 billion tokens, which overflows int32's ~2.1 billion range. Using
int32 offsets here would silently wrap at production scale, so offsets are
int64 -- correctness over the literal spec wording (flagged to @chief).

Feature (token) pool dtype is chosen from sequence_encoding.VOCAB_SIZE at
import time: int16 if the vocabulary fits (it does today: VOCAB_SIZE=326),
else int32, same as the fixed-width design this replaces.

Directory layout (this instance owns ``self.path`` as its storage directory):
  <path>/pool.mm        -- growable memmap (pool_capacity,), FEATURE_DTYPE; the
                            packed token pool, append-only + periodically compacted.
  <path>/offsets.mm     -- memmap (capacity,) int64; per-slot pool offset.
  <path>/lengths.mm     -- memmap (capacity,) int32; per-slot valid token count.
  <path>/targets.mm     -- memmap (capacity, target_dim) float32.
  <path>/masks.mm       -- memmap (capacity, target_dim) bool (only if has_mask).
  <path>/iterations.mm  -- memmap (capacity,) int64.
  <path>/meta.json      -- sidecar metadata; written LAST via atomic rename so a
                            crash mid-save never leaves a meta.json inconsistent
                            with the (already-durable, memmap-backed) array data.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .reservoir import ColumnarBatch, ReservoirSample
from .sequence_encoding import VOCAB_SIZE
from .cfr.exceptions import ReservoirIOError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature (token) dtype selection -- asserted at import time.
# ---------------------------------------------------------------------------
# Token ids run [0, VOCAB_SIZE). int16 covers ids up to 32767. VOCAB_SIZE is
# currently 326 (see sequence_encoding.py: 4 special + 5 frame + 8 actor + 240
# action + 54 card + 9 slot + 5 outcome + 1 peek-result marker), comfortably
# under the int16 ceiling.
# If the vocabulary ever grows past it, fail loudly here rather than silently
# wrapping/truncating token ids into a too-narrow dtype.
if VOCAB_SIZE >= 2**31:
    raise AssertionError(
        f"sequence_encoding.VOCAB_SIZE={VOCAB_SIZE} exceeds the int32 range; "
        "DiskReservoir has no dtype wide enough for this vocabulary."
    )
FEATURE_DTYPE = np.int16 if VOCAB_SIZE < 32768 else np.int32

OFFSET_DTYPE = np.int64  # see module docstring: must span the full pool, not int32
LENGTH_DTYPE = np.int32  # rows are bounded by seq_cap (~2048); int32 is ample headroom


_META_FILENAME = "meta.json"
_POOL_FILENAME = "pool.mm"
_OFFSETS_FILENAME = "offsets.mm"
_LENGTHS_FILENAME = "lengths.mm"
_TARGETS_FILENAME = "targets.mm"
_MASKS_FILENAME = "masks.mm"
_ITERATIONS_FILENAME = "iterations.mm"


def _nbytes(shape, dtype) -> int:
    n = np.dtype(dtype).itemsize
    for d in shape:
        n *= int(d)
    return n


class DiskReservoir:
    """
    Fixed-capacity (row count), ragged-width (token length) reservoir sampling
    buffer, memmap-backed, using Vitter's Algorithm R.

    Args:
        path: Directory this reservoir owns for its memmap files + metadata.
            Created if it does not exist. Re-pointing an existing instance's
            construction at a directory that already contains matching memmap
            files re-opens (not re-creates) that storage; call ``.load()``
            afterward to restore the fill-state bookkeeping (count, seen_count,
            pool cursors, RNG state) from that directory's ``meta.json``.
        capacity: Maximum number of rows to store (20M for the Phase 2 production
            reservoir; small values are used freely in tests).
        seq_cap: Max-length allocation bound for a single row's token sequence
            (the pinned-interface name; NOT a row width -- rows are stored at
            their natural length, this is only a defensive ceiling asserted
            against on add). Production value ~2048; pass explicitly, do not
            rely on the default.
        target_dim: Target dimension per sample (146 = NUM_ACTIONS).
        has_mask: Whether to allocate and track action masks.
        seed: Optional seed for the instance-owned RNG (reproducibility in
            tests; production leaves this unset for OS entropy on a fresh
            reservoir, and relies on ``load()`` to restore RNG state on resume).
        initial_pool_tokens: Starting size of the token pool file, in tokens.
            Grows automatically (doubling) as needed; this is just the first
            allocation, not a hard bound.
        pool_growth_factor: Multiplicative growth factor when the pool needs
            to grow past its current capacity.
        compaction_threshold: Trigger compaction when pool_dead / pool_used
            exceeds this fraction.
        compaction_min_tokens: Skip compaction below this pool_used size (avoids
            constant compaction overhead while the reservoir is still small).
    """

    def __init__(
        self,
        path: str,
        capacity: int = 20_000_000,
        seq_cap: int = 2048,
        target_dim: int = 146,
        has_mask: bool = True,
        seed: Optional[int] = None,
        initial_pool_tokens: int = 65_536,
        pool_growth_factor: float = 2.0,
        compaction_threshold: float = 0.5,
        compaction_min_tokens: int = 1_000_000,
    ):
        self.path = Path(path)
        self.capacity = int(capacity)
        self.seq_cap = int(seq_cap)
        self.target_dim = int(target_dim)
        self.has_mask = bool(has_mask)
        self._size: int = 0
        self.seen_count: int = 0
        self._rng = np.random.default_rng(seed)

        self._initial_pool_tokens = int(initial_pool_tokens)
        self._pool_growth_factor = float(pool_growth_factor)
        self._compaction_threshold = float(compaction_threshold)
        self._compaction_min_tokens = int(compaction_min_tokens)
        self.pool_used = 0
        self.pool_dead = 0

        self.path.mkdir(parents=True, exist_ok=True)
        self._open_all_memmaps()

    # ------------------------------------------------------------------
    # Memmap plumbing
    # ------------------------------------------------------------------

    def _memmap_path(self, name: str) -> str:
        return str(self.path / name)

    def _open_memmap(self, name: str, shape, dtype) -> np.memmap:
        """Open a FIXED-shape memmap (everything except the token pool)."""
        filepath = self._memmap_path(name)
        expected_bytes = _nbytes(shape, dtype)
        if os.path.exists(filepath):
            actual_bytes = os.path.getsize(filepath)
            if actual_bytes != expected_bytes:
                raise ReservoirIOError(
                    f"DiskReservoir storage file {filepath} is {actual_bytes} bytes; "
                    f"expected {expected_bytes} bytes for shape {shape} dtype "
                    f"{np.dtype(dtype).name}. capacity/target_dim/has_mask must "
                    "match the directory's existing storage."
                )
            mode = "r+"
        else:
            mode = "w+"
        return np.memmap(
            filepath, dtype=dtype, mode=mode, shape=tuple(int(d) for d in shape)
        )

    def _open_or_create_pool(self):
        """Open the growable token pool, preserving its current size if it
        already exists (growth history persists across process restarts even
        before an explicit load() restores the used/dead cursors)."""
        filepath = self._memmap_path(_POOL_FILENAME)
        itemsize = np.dtype(FEATURE_DTYPE).itemsize
        if os.path.exists(filepath):
            size_bytes = os.path.getsize(filepath)
            capacity_tokens = max(1, size_bytes // itemsize)
        else:
            capacity_tokens = max(1, self._initial_pool_tokens)
            with open(filepath, "wb") as f:
                f.truncate(capacity_tokens * itemsize)
        self._pool = np.memmap(
            filepath, dtype=FEATURE_DTYPE, mode="r+", shape=(int(capacity_tokens),)
        )
        self.pool_capacity = int(capacity_tokens)

    def _open_all_memmaps(self):
        self._offsets = self._open_memmap(
            _OFFSETS_FILENAME, (self.capacity,), OFFSET_DTYPE
        )
        self._lengths = self._open_memmap(
            _LENGTHS_FILENAME, (self.capacity,), LENGTH_DTYPE
        )
        self._targets = self._open_memmap(
            _TARGETS_FILENAME, (self.capacity, self.target_dim), np.float32
        )
        self._masks = (
            self._open_memmap(_MASKS_FILENAME, (self.capacity, self.target_dim), bool)
            if self.has_mask
            else None
        )
        self._iterations = self._open_memmap(
            _ITERATIONS_FILENAME, (self.capacity,), np.int64
        )
        self._open_or_create_pool()

    def _close_memmaps(self):
        for attr in (
            "_pool",
            "_offsets",
            "_lengths",
            "_targets",
            "_masks",
            "_iterations",
        ):
            arr = getattr(self, attr, None)
            if arr is not None:
                arr.flush()
            setattr(self, attr, None)

    def _flush(self):
        self._pool.flush()
        self._offsets.flush()
        self._lengths.flush()
        self._targets.flush()
        if self.has_mask:
            self._masks.flush()
        self._iterations.flush()

    # ------------------------------------------------------------------
    # Pool growth + compaction
    # ------------------------------------------------------------------

    def _ensure_pool_capacity(self, extra_tokens: int):
        needed = self.pool_used + extra_tokens
        if needed <= self.pool_capacity:
            return
        new_capacity = max(needed, int(self.pool_capacity * self._pool_growth_factor))
        self._resize_pool_file(new_capacity)

    def _resize_pool_file(self, new_capacity: int):
        self._pool.flush()
        filepath = self._memmap_path(_POOL_FILENAME)
        itemsize = np.dtype(FEATURE_DTYPE).itemsize
        self._pool = None  # release the mapping before truncating the file
        with open(filepath, "r+b") as f:
            f.truncate(new_capacity * itemsize)
        self._pool = np.memmap(
            filepath, dtype=FEATURE_DTYPE, mode="r+", shape=(new_capacity,)
        )
        self.pool_capacity = new_capacity

    def _maybe_compact(self):
        if self.pool_used == 0 or self.pool_used < self._compaction_min_tokens:
            return
        if self.pool_dead / self.pool_used < self._compaction_threshold:
            return
        self._compact()

    def _compact(self):
        """Rewrite the pool keeping only rows referenced by slots [0, size),
        packed contiguously, and swap the new file in atomically."""
        live_n = self._size
        if live_n == 0:
            return
        lengths_live = np.asarray(self._lengths[:live_n]).astype(np.int64)
        offsets_live = np.asarray(self._offsets[:live_n]).astype(np.int64)
        total_live_tokens = int(lengths_live.sum())
        new_capacity = max(total_live_tokens, self._initial_pool_tokens, 1)

        tmp_path = self._memmap_path(_POOL_FILENAME + ".compact_tmp")
        itemsize = np.dtype(FEATURE_DTYPE).itemsize
        with open(tmp_path, "wb") as f:
            f.truncate(new_capacity * itemsize)
        new_pool = np.memmap(
            tmp_path, dtype=FEATURE_DTYPE, mode="r+", shape=(new_capacity,)
        )

        new_offsets = np.empty(live_n, dtype=np.int64)
        cursor = 0
        for i in range(live_n):
            ln = int(lengths_live[i])
            new_offsets[i] = cursor
            if ln == 0:
                continue
            off = int(offsets_live[i])
            new_pool[cursor : cursor + ln] = self._pool[off : off + ln]
            cursor += ln
        new_pool.flush()
        del new_pool

        self._offsets[:live_n] = new_offsets
        self._offsets.flush()

        self._pool.flush()
        self._pool = None  # release old mapping before the rename
        old_path = self._memmap_path(_POOL_FILENAME)
        os.replace(tmp_path, old_path)
        self._pool = np.memmap(
            old_path, dtype=FEATURE_DTYPE, mode="r+", shape=(new_capacity,)
        )
        self.pool_capacity = new_capacity
        self.pool_used = cursor
        self.pool_dead = 0
        logger.info(
            "DiskReservoir compacted pool: %d live tokens, capacity now %d tokens",
            cursor,
            new_capacity,
        )

    # ------------------------------------------------------------------
    # Core reservoir API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    def add(self, sample: ReservoirSample):
        """Add a single sample (see ``add_batch`` for the production entry point)."""
        self.add_batch(
            features=[np.asarray(sample.features)],
            targets=np.asarray(sample.target)[np.newaxis, :],
            masks=(
                np.asarray(sample.action_mask)[np.newaxis, :] if self.has_mask else None
            ),
            iterations=np.array([sample.iteration], dtype=np.int64),
        )

    def add_batch(
        self,
        features: Sequence[np.ndarray],
        targets: np.ndarray,
        masks: Optional[np.ndarray] = None,
        iterations: Optional[np.ndarray] = None,
    ):
        """
        Append a batch of ``n`` stream items via Vitter's Algorithm R, vectorized.

        Args:
            features: sequence of n variable-length 1D int array-likes (token
                id sequences), each at its own natural length <= seq_cap. NOT
                padded -- pass raw per-row token lists/arrays; padding is a
                sample_batch()-time concern only.
            targets: (n, target_dim) array-like.
            masks: (n, target_dim) bool array-like, required iff has_mask.
            iterations: (n,) int array-like of CFR iteration numbers.

        Each row is treated as the next item in the stream: row j's occurrence
        index is ``seen_count + j + 1`` (1-indexed), exactly as ReservoirBuffer's
        sequential ``add`` treats each call. The accept/replace decision per row
        depends only on its own occurrence index -- unaffected by raggedness --
        so the whole batch's random draws are vectorized in one call; rows that
        fill remaining capacity are kept unconditionally, and rows drawn after
        the buffer is full replace a uniformly random existing slot with
        probability capacity/occurrence_index, matching the in-RAM buffer.
        """
        n = len(features)
        if n == 0:
            return
        targets = np.asarray(targets)
        if iterations is None:
            raise ValueError("add_batch requires iterations")
        iterations = np.asarray(iterations, dtype=np.int64)
        if self.has_mask:
            if masks is None:
                raise ValueError("add_batch requires masks when has_mask=True")
            masks = np.asarray(masks)
        elif masks is not None:
            raise ValueError("add_batch got masks but this reservoir has has_mask=False")

        row_arrays = [np.asarray(f, dtype=FEATURE_DTYPE).reshape(-1) for f in features]
        row_lengths = np.array([len(r) for r in row_arrays], dtype=np.int64)
        if np.any(row_lengths > self.seq_cap):
            bad = int(np.argmax(row_lengths))
            raise ValueError(
                f"add_batch: row {bad} has length {int(row_lengths[bad])} exceeding "
                f"seq_cap (max-length bound) {self.seq_cap}"
            )

        dest = np.empty(n, dtype=np.int64)
        keep = np.ones(n, dtype=bool)

        remaining_fill = max(0, self.capacity - self._size)
        n_fill = min(remaining_fill, n)
        if n_fill > 0:
            dest[:n_fill] = self._size + np.arange(n_fill, dtype=np.int64)
            self._size += n_fill
        self.seen_count += n_fill

        n_replace = n - n_fill
        if n_replace > 0:
            # occurrence_index (1-indexed) for replace-phase row j is
            # seen_count + j + 1 (seen_count already reflects fill-phase rows).
            occurrence = self.seen_count + 1 + np.arange(n_replace, dtype=np.int64)
            draws = self._rng.integers(0, occurrence)  # uniform in [0, occurrence-1]
            self.seen_count += n_replace
            dest[n_fill:] = draws
            keep[n_fill:] = draws < self.capacity

        if not np.any(keep):
            return
        # Fancy-index assignment with (possibly) duplicate destination indices
        # resolves last-in-array-order-wins, which is exactly correct here: if
        # two rows in this batch land on the same slot, the later row in stream
        # order is the one that should occupy it (the earlier row's brief
        # occupancy is immediately superseded, same as sequential Algorithm R).
        dest_w = dest[keep]
        src_w = np.nonzero(keep)[0]

        # Dead-space accounting: replace-phase writes overwrite an
        # already-populated slot (fill-phase writes never do, by construction:
        # they only ever land on brand-new slots >= old self._size).
        is_replace = src_w >= n_fill
        if np.any(is_replace):
            replaced_slots = dest_w[is_replace]
            old_lengths = np.asarray(self._lengths[replaced_slots]).astype(np.int64)
            self.pool_dead += int(old_lengths.sum())

        kept_lengths = row_lengths[src_w]
        total_new_tokens = int(kept_lengths.sum())
        self._ensure_pool_capacity(total_new_tokens)

        cursor = self.pool_used
        if total_new_tokens > 0:
            concatenated = np.concatenate([row_arrays[i] for i in src_w])
            self._pool[cursor : cursor + total_new_tokens] = concatenated
        new_offsets = cursor + np.cumsum(kept_lengths) - kept_lengths
        self.pool_used += total_new_tokens

        self._offsets[dest_w] = new_offsets
        self._lengths[dest_w] = kept_lengths.astype(LENGTH_DTYPE)
        self._targets[dest_w] = targets[src_w].astype(np.float32, copy=False)
        if self.has_mask:
            self._masks[dest_w] = masks[src_w].astype(bool, copy=False)
        self._iterations[dest_w] = iterations[src_w]

        self._maybe_compact()

    def sample_batch(self, batch_size: int) -> ColumnarBatch:
        """
        Sample a random batch without replacement, as new arrays (not memmap
        views). Features are padded to the longest row IN THE SAMPLED BATCH
        (not to seq_cap) with PAD_ID=0 (sequence_encoding's PAD_ID), alongside
        a ``lengths`` field for pack_padded_sequence-style consumption.
        """
        actual_size = min(int(batch_size), self._size)
        if actual_size == 0:
            return ColumnarBatch(
                features=np.empty((0, 0), dtype=FEATURE_DTYPE),
                targets=np.empty((0, self.target_dim), dtype=np.float32),
                masks=(
                    np.empty((0, self.target_dim), dtype=bool) if self.has_mask else None
                ),
                iterations=np.empty(0, dtype=np.int64),
                lengths=np.empty(0, dtype=np.int64),
            )

        indices = self._rng.choice(self._size, actual_size, replace=False)
        indices.sort()  # locality: sequential-ish reads off the memmaps; sample
        # order doesn't matter downstream (the trainer consumes an unordered batch).

        lengths = np.asarray(self._lengths[indices]).astype(np.int64)
        offsets = np.asarray(self._offsets[indices]).astype(np.int64)
        max_len_in_batch = int(lengths.max())
        features = np.zeros((actual_size, max_len_in_batch), dtype=FEATURE_DTYPE)
        pool = self._pool
        for row_i in range(actual_size):
            ln = int(lengths[row_i])
            if ln > 0:
                off = int(offsets[row_i])
                features[row_i, :ln] = pool[off : off + ln]

        return ColumnarBatch(
            features=features,
            targets=np.asarray(self._targets[indices]),
            masks=np.asarray(self._masks[indices]) if self.has_mask else None,
            iterations=np.asarray(self._iterations[indices]),
            lengths=lengths,
        )

    # ------------------------------------------------------------------
    # Save / load: metadata sidecar only -- memmap files ARE the storage.
    # ------------------------------------------------------------------

    def _meta_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "seq_cap": self.seq_cap,
            "target_dim": self.target_dim,
            "has_mask": self.has_mask,
            "feature_dtype": np.dtype(FEATURE_DTYPE).name,
            "offset_dtype": np.dtype(OFFSET_DTYPE).name,
            "length_dtype": np.dtype(LENGTH_DTYPE).name,
            "count": self._size,
            "seen_count": self.seen_count,
            "pool_capacity": self.pool_capacity,
            "pool_used": self.pool_used,
            "pool_dead": self.pool_dead,
            "rng_state": self._rng.bit_generator.state,
        }

    def _write_meta_atomic(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        tmp_path = directory / (_META_FILENAME + ".tmp")
        final_path = directory / _META_FILENAME
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._meta_dict(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)  # atomic on POSIX
        except (OSError, IOError, PermissionError) as e:
            raise ReservoirIOError(
                f"Failed to write DiskReservoir metadata to {final_path}: {e}"
            ) from e

    def save(self, path: Optional[str] = None):
        """
        Flush pending memmap writes and persist metadata (count, seen_count,
        pool cursors, RNG state, dims, dtypes) as a sidecar ``meta.json``,
        written last via atomic rename so a crash mid-save cannot leave
        stale/inconsistent metadata next to already-durable array data.

        If ``path`` is None (the common case), persists in place (``self.path``)
        -- the memmap files already live there. If ``path`` names a different
        directory, the array files are copied there first (a full on-disk
        snapshot) before the metadata is written; this reservoir's own
        ``self.path`` and open memmaps are unaffected.
        """
        target_dir = Path(path) if path is not None else self.path
        self._flush()
        if target_dir != self.path:
            target_dir.mkdir(parents=True, exist_ok=True)
            for fname in (
                _POOL_FILENAME,
                _OFFSETS_FILENAME,
                _LENGTHS_FILENAME,
                _TARGETS_FILENAME,
                _ITERATIONS_FILENAME,
            ):
                shutil.copy2(self._memmap_path(fname), target_dir / fname)
            if self.has_mask:
                shutil.copy2(
                    self._memmap_path(_MASKS_FILENAME), target_dir / _MASKS_FILENAME
                )
        self._write_meta_atomic(target_dir)
        logger.info(
            "Saved DiskReservoir (%d samples, %d seen, %d pool tokens used) metadata to %s",
            self._size,
            self.seen_count,
            self.pool_used,
            target_dir / _META_FILENAME,
        )

    def load(self, path: Optional[str] = None):
        """
        Restore fill-state bookkeeping (count, seen_count, pool cursors, RNG
        state) from a ``meta.json`` sidecar.

        If ``path`` is None (the common resume case), restores from
        ``self.path``, assuming this instance's already-open memmap files are
        the same ones a prior process wrote. If ``path`` names a different
        directory, this instance re-points its storage there first: current
        memmaps are flushed and released, and new ones are opened against
        ``path``'s files before the metadata is applied.
        """
        source_dir = Path(path) if path is not None else self.path
        meta_path = source_dir / _META_FILENAME
        if not meta_path.exists():
            raise ReservoirIOError(f"DiskReservoir metadata not found: {meta_path}")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except (OSError, IOError, PermissionError) as e:
            raise ReservoirIOError(
                f"Failed to read DiskReservoir metadata {meta_path}: {e}"
            ) from e
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ReservoirIOError(
                f"Corrupted DiskReservoir metadata {meta_path}: {e}"
            ) from e

        try:
            capacity = int(meta["capacity"])
            seq_cap = int(meta["seq_cap"])
            target_dim = int(meta["target_dim"])
            has_mask = bool(meta["has_mask"])
            count = int(meta["count"])
            seen_count = int(meta["seen_count"])
            pool_used = int(meta["pool_used"])
            pool_dead = int(meta["pool_dead"])
            rng_state = meta["rng_state"]
        except (KeyError, ValueError, TypeError) as e:
            raise ReservoirIOError(
                f"Malformed DiskReservoir metadata {meta_path}: {e}"
            ) from e

        if source_dir == self.path:
            if (
                capacity != self.capacity
                or seq_cap != self.seq_cap
                or target_dim != self.target_dim
                or has_mask != self.has_mask
            ):
                raise ReservoirIOError(
                    f"DiskReservoir metadata at {meta_path} describes "
                    f"(capacity={capacity}, seq_cap={seq_cap}, target_dim={target_dim}, "
                    f"has_mask={has_mask}), but this instance was constructed with "
                    f"(capacity={self.capacity}, seq_cap={self.seq_cap}, "
                    f"target_dim={self.target_dim}, has_mask={self.has_mask}). "
                    "Construct with matching dimensions before loading."
                )
        else:
            self._close_memmaps()
            self.path = source_dir
            self.capacity = capacity
            self.seq_cap = seq_cap
            self.target_dim = target_dim
            self.has_mask = has_mask
            self._open_all_memmaps()

        if count > self.capacity:
            raise ReservoirIOError(
                f"DiskReservoir metadata at {meta_path} claims count={count} "
                f"exceeding capacity={self.capacity}"
            )
        if pool_used > self.pool_capacity:
            logger.warning(
                "DiskReservoir metadata pool_used=%d exceeds the live pool file's "
                "capacity=%d at %s; the pool file may be truncated/stale.",
                pool_used,
                self.pool_capacity,
                meta_path,
            )
        self._size = count
        self.seen_count = seen_count
        self.pool_used = pool_used
        self.pool_dead = pool_dead
        try:
            self._rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as e:
            logger.warning(
                "Could not restore DiskReservoir RNG state from %s (%s); "
                "continuing with a fresh RNG (uniformity preserved, resume "
                "sequence not bit-identical).",
                meta_path,
                e,
            )
        logger.info(
            "Loaded DiskReservoir: %d samples, %d seen, %d pool tokens used, capacity %d from %s",
            self._size,
            self.seen_count,
            self.pool_used,
            self.capacity,
            meta_path,
        )
