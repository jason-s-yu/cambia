#!/usr/bin/env python3
"""
Micro-benchmark for _train_network data preparation.
Compares old np.stack pattern vs new columnar ColumnarBatch.

Fills a buffer with N samples, then measures time to sample + collate
batches for training, isolating the data prep overhead.
"""

import time
import numpy as np
import torch

from src.encoding import INPUT_DIM, NUM_ACTIONS
from src.reservoir import ReservoirBuffer, ReservoirSample, ColumnarBatch


def fill_buffer(buf, n_samples):
    """Fill buffer with random samples."""
    for i in range(n_samples):
        sample = ReservoirSample(
            features=np.random.randn(INPUT_DIM).astype(np.float32),
            target=np.random.randn(NUM_ACTIONS).astype(np.float32),
            action_mask=(np.random.rand(NUM_ACTIONS) > 0.5).astype(bool),
            iteration=i % 100,
        )
        buf.add(sample)


def bench_columnar(buf, batch_size, num_iters, device):
    """New columnar path: batch.features/targets/masks/iterations are pre-stacked."""
    times = []
    for _ in range(num_iters):
        batch = buf.sample_batch(batch_size)
        t0 = time.perf_counter()
        features_t = torch.from_numpy(batch.features).float().to(device)
        targets_t = torch.from_numpy(batch.targets).float().to(device)
        masks_t = torch.from_numpy(batch.masks.copy()).to(device)
        iterations_t = torch.from_numpy(batch.iterations.astype(np.float32)).to(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_old_npstack(buf, batch_size, num_iters, device):
    """Old path: sample_batch returns list, then np.stack per field."""
    times = []
    for _ in range(num_iters):
        batch = buf.sample_batch(batch_size)
        # Simulate old path: iterate to build list of samples, then np.stack
        samples = list(batch)  # ColumnarBatch.__iter__ yields ReservoirSample
        t0 = time.perf_counter()
        features_batch = np.stack([s.features for s in samples])
        targets_batch = np.stack([s.target for s in samples])
        masks_batch = np.stack([s.action_mask for s in samples])
        iterations_batch = np.array([s.iteration for s in samples], dtype=np.float32)
        features_t = torch.from_numpy(features_batch).float().to(device)
        targets_t = torch.from_numpy(targets_batch).float().to(device)
        masks_t = torch.from_numpy(masks_batch).to(device)
        iterations_t = torch.from_numpy(iterations_batch).float().to(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main():
    device = torch.device("cpu")
    batch_size = 2048
    num_iters = 100
    buffer_size = 100_000

    print(f"Filling buffer with {buffer_size} samples...")
    buf = ReservoirBuffer(capacity=buffer_size)
    fill_buffer(buf, buffer_size)
    print(f"Buffer filled: {len(buf)} samples")

    # Warmup
    for _ in range(5):
        bench_columnar(buf, batch_size, 1, device)
        bench_old_npstack(buf, batch_size, 1, device)

    print(f"\nBenchmarking data prep: batch_size={batch_size}, {num_iters} iterations")
    print("=" * 60)

    # Columnar path
    col_times = bench_columnar(buf, batch_size, num_iters, device)
    col_avg = np.mean(col_times) * 1000
    col_std = np.std(col_times) * 1000
    print(f"Columnar (new):  {col_avg:.3f} +/- {col_std:.3f} ms/batch")

    # Old np.stack path
    old_times = bench_old_npstack(buf, batch_size, num_iters, device)
    old_avg = np.mean(old_times) * 1000
    old_std = np.std(old_times) * 1000
    print(f"np.stack (old):  {old_avg:.3f} +/- {old_std:.3f} ms/batch")

    speedup = old_avg / col_avg
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Savings per batch: {old_avg - col_avg:.3f} ms")

    # At 4000 train_steps_per_iteration (production config), two networks:
    savings_per_iter = (old_avg - col_avg) * 4000 * 2 / 1000
    print(f"Estimated savings per training iteration (4000 steps x 2 networks): {savings_per_iter:.1f}s")

    # Also test with GPU if available
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        print(f"\n--- GPU ({torch.cuda.get_device_name(0)}) ---")

        # Warmup
        for _ in range(5):
            bench_columnar(buf, batch_size, 1, device_gpu)
            bench_old_npstack(buf, batch_size, 1, device_gpu)

        col_times_gpu = bench_columnar(buf, batch_size, num_iters, device_gpu)
        col_avg_gpu = np.mean(col_times_gpu) * 1000
        col_std_gpu = np.std(col_times_gpu) * 1000
        print(f"Columnar (new):  {col_avg_gpu:.3f} +/- {col_std_gpu:.3f} ms/batch")

        old_times_gpu = bench_old_npstack(buf, batch_size, num_iters, device_gpu)
        old_avg_gpu = np.mean(old_times_gpu) * 1000
        old_std_gpu = np.std(old_times_gpu) * 1000
        print(f"np.stack (old):  {old_avg_gpu:.3f} +/- {old_std_gpu:.3f} ms/batch")

        speedup_gpu = old_avg_gpu / col_avg_gpu
        print(f"\nSpeedup: {speedup_gpu:.2f}x")
        savings_gpu = (old_avg_gpu - col_avg_gpu) * 4000 * 2 / 1000
        print(f"Estimated savings per training iteration (4000 steps x 2 networks): {savings_gpu:.1f}s")


if __name__ == "__main__":
    main()
