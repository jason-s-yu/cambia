"""
src/reservoir.py

Reservoir sampling buffers for Deep CFR training.

Uses Vitter's Algorithm R to maintain a fixed-capacity uniform random sample
of all training samples ever generated. Two separate buffers are used:
  - Mv: advantage/regret samples
  - Mpi: strategy samples

Samples store iteration number for linear CFR weighting during training.

Storage is columnar: four contiguous numpy arrays (features, targets, masks,
iterations) rather than a list of Python objects. This eliminates the
per-sample np.stack overhead in the training loop.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .encoding import INPUT_DIM, NUM_ACTIONS
from .cfr.exceptions import ReservoirIOError

logger = logging.getLogger(__name__)


@dataclass
class ReservoirSample:
    """A single training sample for the reservoir buffer."""

    features: np.ndarray  # (INPUT_DIM,) float32 -- encoded infoset
    target: np.ndarray  # (NUM_ACTIONS,) float32 -- regrets or strategy
    action_mask: np.ndarray  # (NUM_ACTIONS,) bool -- legal actions
    iteration: int  # CFR iteration number for weighting
    infoset_key_raw: Optional[Tuple] = None  # Optional debugging metadata


class ColumnarBatch:
    """
    A batch of samples returned by ReservoirBuffer.sample_batch().

    Provides pre-stacked numpy arrays for direct conversion to PyTorch tensors,
    and also supports len() and iteration over ReservoirSample objects for
    backward compatibility.
    """

    __slots__ = ("features", "targets", "masks", "iterations", "_size")

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        iterations: np.ndarray,
    ):
        self.features = features  # (N, INPUT_DIM) float32
        self.targets = targets  # (N, NUM_ACTIONS) float32
        self.masks = masks  # (N, NUM_ACTIONS) bool
        self.iterations = iterations  # (N,) int64
        self._size = len(features)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self):
        """Iterate as ReservoirSample objects (backward compatibility)."""
        for i in range(self._size):
            yield ReservoirSample(
                features=self.features[i],
                target=self.targets[i],
                action_mask=self.masks[i],
                iteration=int(self.iterations[i]),
            )

    def __getitem__(self, idx):
        """Index into the batch, returning a ReservoirSample."""
        return ReservoirSample(
            features=self.features[idx],
            target=self.targets[idx],
            action_mask=self.masks[idx],
            iteration=int(self.iterations[idx]),
        )


class _BufferView:
    """
    Provides list-like access to the columnar buffer for backward compatibility.

    Tests access buf.buffer[i].iteration and iterate over buf.buffer. This view
    creates ReservoirSample objects lazily on access.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: "ReservoirBuffer"):
        self._owner = owner

    def __len__(self) -> int:
        return self._owner._size

    def __bool__(self) -> bool:
        return self._owner._size > 0

    def __iter__(self):
        owner = self._owner
        for i in range(owner._size):
            yield ReservoirSample(
                features=owner._features[i].copy(),
                target=owner._targets[i].copy(),
                action_mask=owner._masks[i].copy(),
                iteration=int(owner._iterations[i]),
            )

    def __getitem__(self, idx):
        owner = self._owner
        if isinstance(idx, slice):
            indices = range(*idx.indices(owner._size))
            return [
                ReservoirSample(
                    features=owner._features[i].copy(),
                    target=owner._targets[i].copy(),
                    action_mask=owner._masks[i].copy(),
                    iteration=int(owner._iterations[i]),
                )
                for i in indices
            ]
        if idx < 0:
            idx += owner._size
        if idx < 0 or idx >= owner._size:
            raise IndexError(f"buffer index {idx} out of range")
        return ReservoirSample(
            features=owner._features[idx].copy(),
            target=owner._targets[idx].copy(),
            action_mask=owner._masks[idx].copy(),
            iteration=int(owner._iterations[idx]),
        )


class ReservoirBuffer:
    """
    Fixed-capacity reservoir sampling buffer using Vitter's Algorithm R.

    Guarantees a uniform random sample over all items ever added,
    regardless of how many items have been seen.

    Internal storage is columnar: four contiguous numpy arrays are
    pre-allocated to ``capacity`` rows. This avoids per-sample Python
    object overhead and enables O(1) batch sampling via fancy indexing.
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity = capacity
        self._size: int = 0
        self.seen_count: int = 0

        # Pre-allocate columnar storage
        self._features = np.zeros((capacity, INPUT_DIM), dtype=np.float32)
        self._targets = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self._masks = np.zeros((capacity, NUM_ACTIONS), dtype=bool)
        self._iterations = np.zeros(capacity, dtype=np.int64)

    # Backward-compatible property: tests access buf.buffer
    @property
    def buffer(self) -> _BufferView:
        return _BufferView(self)

    def __len__(self) -> int:
        return self._size

    def add(self, sample: ReservoirSample):
        """
        Add a sample to the buffer using reservoir sampling.

        If the buffer is not full, the sample is appended directly.
        Once full, each new sample has a (capacity / seen_count) probability
        of replacing a random existing sample.
        """
        self.seen_count += 1
        if self._size < self.capacity:
            idx = self._size
            self._size += 1
        else:
            idx = random.randint(0, self.seen_count - 1)
            if idx >= self.capacity:
                return

        self._features[idx] = sample.features
        self._targets[idx] = sample.target
        self._masks[idx] = sample.action_mask
        self._iterations[idx] = sample.iteration

    def sample_batch(self, batch_size: int) -> ColumnarBatch:
        """
        Sample a random batch from the buffer.

        Args:
            batch_size: Number of samples to draw (without replacement).

        Returns:
            ColumnarBatch with pre-stacked arrays. len() == min(batch_size, buffer size).
        """
        actual_size = min(batch_size, self._size)
        if actual_size == 0:
            return ColumnarBatch(
                features=np.empty((0, INPUT_DIM), dtype=np.float32),
                targets=np.empty((0, NUM_ACTIONS), dtype=np.float32),
                masks=np.empty((0, NUM_ACTIONS), dtype=bool),
                iterations=np.empty(0, dtype=np.int64),
            )

        indices = np.random.choice(self._size, actual_size, replace=False)
        return ColumnarBatch(
            features=self._features[indices],
            targets=self._targets[indices],
            masks=self._masks[indices],
            iterations=self._iterations[indices],
        )

    def save(self, path: str):
        """
        Save the buffer to disk as a compressed numpy archive.

        Stores features, targets, masks, and iterations as arrays,
        plus scalar metadata for seen_count and capacity.

        Raises:
            ReservoirIOError: If file I/O operations fail.
        """
        try:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            n = self._size
            np.savez_compressed(
                str(filepath),
                features=self._features[:n],
                targets=self._targets[:n],
                masks=self._masks[:n],
                iterations=self._iterations[:n],
                meta=np.array([self.seen_count, self.capacity], dtype=np.int64),
            )
            logger.info(
                "Saved reservoir buffer (%d samples, %d seen) to %s",
                n,
                self.seen_count,
                filepath,
            )
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to save reservoir buffer to %s: %s", path, e)
            raise ReservoirIOError(f"Failed to save reservoir buffer to {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error saving reservoir buffer to %s: %s", path, e)
            raise ReservoirIOError(f"Unexpected error saving reservoir buffer to {path}: {e}") from e

    def load(self, path: str):
        """
        Load the buffer from a numpy archive saved by save().

        Replaces the current buffer contents entirely.

        Raises:
            ReservoirIOError: If file I/O operations fail or file is corrupted.
        """
        try:
            filepath = Path(path)
            # Handle .npz extension
            if not filepath.suffix:
                filepath = filepath.with_suffix(".npz")
            if not str(path).endswith(".npz") and not filepath.exists():
                filepath = Path(str(path) + ".npz")

            data = np.load(str(filepath))

            meta = data["meta"]
            self.seen_count = int(meta[0])
            saved_capacity = int(meta[1])

            features = data["features"]
            targets = data["targets"]
            masks = data["masks"]
            iterations = data["iterations"]

            n = len(features)

            # If loaded capacity differs from current, log and keep current capacity
            if saved_capacity != self.capacity:
                logger.info(
                    "Loaded buffer had capacity %d, current capacity is %d. "
                    "Adjusting buffer if needed.",
                    saved_capacity,
                    self.capacity,
                )
                if n > self.capacity:
                    # Truncate to current capacity via random subsample
                    keep = np.random.choice(n, self.capacity, replace=False)
                    features = features[keep]
                    targets = targets[keep]
                    masks = masks[keep]
                    iterations = iterations[keep]
                    n = self.capacity

            # Re-allocate arrays if capacity changed since construction
            if self._features.shape[0] != self.capacity:
                self._features = np.zeros((self.capacity, INPUT_DIM), dtype=np.float32)
                self._targets = np.zeros((self.capacity, NUM_ACTIONS), dtype=np.float32)
                self._masks = np.zeros((self.capacity, NUM_ACTIONS), dtype=bool)
                self._iterations = np.zeros(self.capacity, dtype=np.int64)

            self._features[:n] = features
            self._targets[:n] = targets
            self._masks[:n] = masks
            self._iterations[:n] = iterations
            self._size = n

            logger.info(
                "Loaded reservoir buffer: %d samples, %d seen, capacity %d from %s",
                self._size,
                self.seen_count,
                self.capacity,
                filepath,
            )
        except FileNotFoundError as e:
            logger.error("Reservoir buffer file not found: %s", path)
            raise ReservoirIOError(f"Reservoir buffer file not found: {path}") from e
        except (OSError, IOError, PermissionError) as e:
            logger.error("Failed to load reservoir buffer from %s: %s", path, e)
            raise ReservoirIOError(f"Failed to load reservoir buffer from {path}: {e}") from e
        except (KeyError, ValueError, IndexError) as e:
            logger.error("Corrupted reservoir buffer file %s: %s", path, e)
            raise ReservoirIOError(f"Corrupted reservoir buffer file {path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error loading reservoir buffer from %s: %s", path, e)
            raise ReservoirIOError(f"Unexpected error loading reservoir buffer from {path}: {e}") from e

    def resize(self, new_capacity: int):
        """
        Resize the buffer capacity.

        If shrinking, randomly subsample the current buffer to the new capacity.
        If growing, just update the capacity (new samples will fill naturally).
        The reservoir sampling property is preserved.

        Args:
            new_capacity: The new maximum capacity.
        """
        old_capacity = self.capacity

        if self._size > new_capacity:
            # Subsample to new capacity
            keep = np.random.choice(self._size, new_capacity, replace=False)
            new_features = np.zeros((new_capacity, INPUT_DIM), dtype=np.float32)
            new_targets = np.zeros((new_capacity, NUM_ACTIONS), dtype=np.float32)
            new_masks = np.zeros((new_capacity, NUM_ACTIONS), dtype=bool)
            new_iterations = np.zeros(new_capacity, dtype=np.int64)

            new_features[:new_capacity] = self._features[keep]
            new_targets[:new_capacity] = self._targets[keep]
            new_masks[:new_capacity] = self._masks[keep]
            new_iterations[:new_capacity] = self._iterations[keep]

            self._features = new_features
            self._targets = new_targets
            self._masks = new_masks
            self._iterations = new_iterations
            self._size = new_capacity
            self.capacity = new_capacity

            logger.info(
                "Resized buffer from %d to %d (truncated %d samples)",
                old_capacity,
                new_capacity,
                old_capacity - new_capacity,
            )
        else:
            # Growing or same: re-allocate larger arrays, copy existing data
            new_features = np.zeros((new_capacity, INPUT_DIM), dtype=np.float32)
            new_targets = np.zeros((new_capacity, NUM_ACTIONS), dtype=np.float32)
            new_masks = np.zeros((new_capacity, NUM_ACTIONS), dtype=bool)
            new_iterations = np.zeros(new_capacity, dtype=np.int64)

            n = self._size
            if n > 0:
                new_features[:n] = self._features[:n]
                new_targets[:n] = self._targets[:n]
                new_masks[:n] = self._masks[:n]
                new_iterations[:n] = self._iterations[:n]

            self._features = new_features
            self._targets = new_targets
            self._masks = new_masks
            self._iterations = new_iterations
            self.capacity = new_capacity

            logger.info(
                "Resized buffer capacity from %d to %d (current size %d, no truncation)",
                old_capacity,
                new_capacity,
                self._size,
            )

    def clear(self):
        """Clear all samples and reset the counter."""
        self._size = 0
        self.seen_count = 0


def samples_to_tensors(
    samples: List[ReservoirSample],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of ReservoirSamples into batched numpy arrays suitable
    for conversion to PyTorch tensors.

    Args:
        samples: List of ReservoirSample instances.

    Returns:
        Tuple of (features, targets, masks, iterations):
          - features: (N, INPUT_DIM) float32
          - targets: (N, NUM_ACTIONS) float32
          - masks: (N, NUM_ACTIONS) bool
          - iterations: (N,) int64
    """
    if not samples:
        return (
            np.empty((0, INPUT_DIM), dtype=np.float32),
            np.empty((0, NUM_ACTIONS), dtype=np.float32),
            np.empty((0, NUM_ACTIONS), dtype=bool),
            np.empty(0, dtype=np.int64),
        )

    features = np.stack([s.features for s in samples])
    targets = np.stack([s.target for s in samples])
    masks = np.stack([s.action_mask for s in samples])
    iterations = np.array([s.iteration for s in samples], dtype=np.int64)

    return features, targets, masks, iterations
