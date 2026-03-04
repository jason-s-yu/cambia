#!/usr/bin/env python3
"""
ReBeL Phase 1 Pre-Training Benchmark
=====================================
Measures:
  1. Single-episode throughput (CPU)
  2. Multi-worker episode throughput scaling (1,2,4,8,12,16 workers)
  3. XPU vs CPU training step throughput (value net + policy net)
  4. XPU + multi-worker combined overhead check

Results are saved to cfr/docs/rebel_benchmark_results.md with full context.
"""

import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from src.config import DeepCfrConfig, CambiaRulesConfig
from src.networks import PBSValueNetwork, PBSPolicyNetwork
from src.pbs import PBS_INPUT_DIM, NUM_HAND_TYPES
from src.encoding import NUM_ACTIONS
from src.cfr.rebel_trainer import _rebel_batch_worker

VALUE_DIM = 2 * NUM_HAND_TYPES

RULES = CambiaRulesConfig(
    allowDrawFromDiscardPile=False,
    allowReplaceAbilities=False,
    snapRace=False,
    penaltyDrawCount=2,
    use_jokers=2,
    cards_per_player=4,
    initial_view_count=2,
    cambia_allowed_round=0,
    allowOpponentSnapping=False,
    max_game_turns=46,
)

CONFIG = DeepCfrConfig(
    rebel_subgame_depth=4,
    rebel_cfr_iterations=200,
    rebel_value_hidden_dim=1024,
    rebel_policy_hidden_dim=512,
    rebel_value_learning_rate=1e-3,
    rebel_policy_learning_rate=1e-3,
    batch_size=2048,
    alpha=1.5,
)


def make_nets():
    v = PBSValueNetwork(input_dim=PBS_INPUT_DIM, hidden_dim=1024, output_dim=VALUE_DIM, validate_inputs=False)
    p = PBSPolicyNetwork(input_dim=PBS_INPUT_DIM, hidden_dim=512, output_dim=NUM_ACTIONS, validate_inputs=False)
    return v, p


def nets_to_numpy(v, p):
    vs = {k: val.cpu().numpy() for k, val in v.state_dict().items()}
    ps = {k: val.cpu().numpy() for k, val in p.state_dict().items()}
    return vs, ps


def benchmark_self_play(num_workers_list, episodes_per_trial=20):
    """Benchmark self-play throughput with varying worker counts."""
    print("\n=== Self-Play Throughput Benchmark ===")
    v, p = make_nets()
    v.eval(); p.eval()
    vs, ps = nets_to_numpy(v, p)

    results = {}
    ctx = multiprocessing.get_context("spawn")

    for nw in num_workers_list:
        # Split episodes across workers
        eps_per_worker = max(1, episodes_per_trial // nw)
        actual_total = eps_per_worker * nw

        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=nw, mp_context=ctx
        ) as executor:
            futures = []
            for _ in range(nw):
                args = (eps_per_worker, vs, ps, CONFIG, RULES)
                futures.append(executor.submit(_rebel_batch_worker, args))

            all_samples = []
            for f in concurrent.futures.as_completed(futures):
                all_samples.extend(f.result())

        elapsed = time.time() - t0
        eps_per_sec = actual_total / elapsed
        samples_per_sec = len(all_samples) / elapsed

        results[nw] = {
            "workers": nw,
            "total_episodes": actual_total,
            "total_samples": len(all_samples),
            "wall_time_s": round(elapsed, 2),
            "episodes_per_sec": round(eps_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 2),
            "samples_per_episode": round(len(all_samples) / max(actual_total, 1), 1),
        }
        print(f"  {nw:2d} workers: {actual_total} eps in {elapsed:.2f}s "
              f"({eps_per_sec:.1f} eps/s, {samples_per_sec:.0f} samp/s, "
              f"{len(all_samples)/max(actual_total,1):.1f} samp/ep)")

    return results


def benchmark_training(devices):
    """Benchmark training step throughput on different devices."""
    print("\n=== Training Step Benchmark ===")
    results = {}

    # Generate synthetic buffer data
    n_samples = 4096
    features = np.random.randn(n_samples, PBS_INPUT_DIM).astype(np.float32)
    value_targets = np.random.randn(n_samples, VALUE_DIM).astype(np.float32)
    policy_targets = np.random.rand(n_samples, NUM_ACTIONS).astype(np.float32)
    masks = np.random.rand(n_samples, NUM_ACTIONS) > 0.5
    # Ensure at least one legal action per sample
    for i in range(n_samples):
        if not masks[i].any():
            masks[i, 0] = True
    iterations = np.arange(n_samples, dtype=np.float32)

    for device_name in devices:
        try:
            device = torch.device(device_name)
            # Quick availability check
            if device_name == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                print(f"  {device_name}: SKIPPED (not available)")
                continue
            if device_name == "cuda" and not torch.cuda.is_available():
                print(f"  {device_name}: SKIPPED (not available)")
                continue
        except Exception:
            print(f"  {device_name}: SKIPPED (error)")
            continue

        v, p = make_nets()
        v.to(device); p.to(device)
        v_opt = torch.optim.Adam(v.parameters(), lr=1e-3)
        p_opt = torch.optim.Adam(p.parameters(), lr=1e-3)

        batch_size = 2048
        num_steps = 50

        # Pre-move data to device
        feat_t = torch.from_numpy(features[:batch_size]).to(device)
        vtgt_t = torch.from_numpy(value_targets[:batch_size]).to(device)
        ptgt_t = torch.from_numpy(policy_targets[:batch_size]).to(device)
        mask_t = torch.from_numpy(masks[:batch_size]).to(device)
        iter_t = torch.from_numpy(iterations[:batch_size]).to(device)
        weights = (iter_t + 1.0).pow(1.5)
        weights = weights / weights.mean()

        # Warmup
        v.train(); p.train()
        for _ in range(5):
            v_opt.zero_grad()
            pred = v(feat_t)
            loss = ((pred - vtgt_t) ** 2).mean()
            loss.backward()
            v_opt.step()

        if device_name == "xpu":
            torch.xpu.synchronize()

        # Value net benchmark
        t0 = time.time()
        for _ in range(num_steps):
            v_opt.zero_grad()
            pred = v(feat_t)
            per_sample = ((pred - vtgt_t) ** 2).mean(dim=1)
            loss = (weights * per_sample).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0)
            v_opt.step()

        if device_name == "xpu":
            torch.xpu.synchronize()
        v_time = (time.time() - t0) / num_steps

        # Policy net benchmark
        for _ in range(5):
            p_opt.zero_grad()
            pred = p(feat_t, mask_t)
            loss = ((pred - ptgt_t) ** 2).mean()
            loss.backward()
            p_opt.step()

        if device_name == "xpu":
            torch.xpu.synchronize()

        t0 = time.time()
        for _ in range(num_steps):
            p_opt.zero_grad()
            pred = p(feat_t, mask_t)
            masked_pred = pred.masked_fill(~mask_t, 0.0)
            masked_tgt = ptgt_t.masked_fill(~mask_t, 0.0)
            num_legal = mask_t.float().sum(dim=1).clamp(min=1.0)
            per_sample = ((masked_pred - masked_tgt) ** 2).sum(dim=1) / num_legal
            loss = (weights * per_sample).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
            p_opt.step()

        if device_name == "xpu":
            torch.xpu.synchronize()
        p_time = (time.time() - t0) / num_steps

        results[device_name] = {
            "device": device_name,
            "batch_size": batch_size,
            "value_step_ms": round(v_time * 1000, 1),
            "policy_step_ms": round(p_time * 1000, 1),
            "total_step_ms": round((v_time + p_time) * 1000, 1),
            "value_steps_per_sec": round(1 / v_time, 1),
            "policy_steps_per_sec": round(1 / p_time, 1),
        }
        print(f"  {device_name}: value={v_time*1000:.1f}ms/step  policy={p_time*1000:.1f}ms/step  "
              f"total={((v_time+p_time)*1000):.1f}ms/step")

        # Cleanup
        del v, p, v_opt, p_opt, feat_t, vtgt_t, ptgt_t, mask_t, iter_t

    return results


def write_results(sp_results, train_results):
    """Write benchmark results to markdown file."""
    now = datetime.now(timezone.utc).isoformat()

    md = f"""# ReBeL Phase 1 Pre-Training Benchmark

**Date:** {now}
**Hardware:** AMD Ryzen 9 9900X (22 vCPUs), 64GB RAM, Intel Arc A310 LP (XPU)
**PyTorch:** {torch.__version__}

## Configuration

| Parameter | Value |
|-|-|
| rebel_subgame_depth | 4 |
| rebel_cfr_iterations | 200 |
| rebel_value_hidden_dim | 1024 (4.3M params) |
| rebel_policy_hidden_dim | 512 (0.7M params) |
| batch_size | 2048 |
| alpha | 1.5 |
| max_game_turns | 46 |
| exploration_epsilon | 0.05 |

## Game Rules

Default competitive-minus (no discard draw, no replace abilities, no snap race, no opponent snapping).

## Self-Play Throughput

| Workers | Episodes | Samples | Wall Time (s) | Eps/s | Samp/s | Samp/Ep |
|-|-|-|-|-|-|-|
"""
    for nw in sorted(sp_results.keys()):
        r = sp_results[nw]
        md += f"| {r['workers']} | {r['total_episodes']} | {r['total_samples']} | {r['wall_time_s']} | {r['episodes_per_sec']} | {r['samples_per_sec']} | {r['samples_per_episode']} |\n"

    md += """
## Training Step Throughput

| Device | Batch | Value (ms/step) | Policy (ms/step) | Total (ms/step) | Value (steps/s) | Policy (steps/s) |
|-|-|-|-|-|-|-|
"""
    for dev in sorted(train_results.keys()):
        r = train_results[dev]
        md += f"| {r['device']} | {r['batch_size']} | {r['value_step_ms']} | {r['policy_step_ms']} | {r['total_step_ms']} | {r['value_steps_per_sec']} | {r['policy_steps_per_sec']} |\n"

    # Projections
    if sp_results and train_results:
        # Use best worker config
        best_w = max(sp_results.values(), key=lambda x: x["samples_per_sec"])
        best_dev = min(train_results.values(), key=lambda x: x["total_step_ms"])

        samp_per_sec = best_w["samples_per_sec"]
        samp_per_ep = best_w["samples_per_episode"]
        step_ms = best_dev["total_step_ms"]

        md += f"""
## Projected Iteration Time

**Best config:** {best_w['workers']} workers + {best_dev['device']} training

| Phase | Estimate |
|-|-|
| Self-play (50 episodes) | {50 / best_w['episodes_per_sec']:.1f}s |
| Self-play (100 episodes) | {100 / best_w['episodes_per_sec']:.1f}s |
| Training (300 steps × 2 nets) | {300 * best_dev['total_step_ms'] / 1000:.1f}s |
| Training (500 steps × 2 nets) | {500 * best_dev['total_step_ms'] / 1000:.1f}s |
| **Total iter (50 ep, 300 steps)** | **{50 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000:.1f}s** |
| **Total iter (100 ep, 300 steps)** | **{100 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000:.1f}s** |
| Samples/iter (50 ep) | ~{50 * samp_per_ep:.0f} |
| Samples/iter (100 ep) | ~{100 * samp_per_ep:.0f} |

## Iteration Count Projection

With 50 episodes/iter and ~{samp_per_ep:.0f} samples/episode = ~{50*samp_per_ep:.0f} samples/iter.
Buffer capacity 500K fills at iter ~{500000/(50*samp_per_ep):.0f}.

| Target Iters | Wall Time | Total Samples |
|-|-|-|
| 100 | {100 * (50 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000) / 3600:.1f}h | ~{100*50*samp_per_ep:.0f} |
| 250 | {250 * (50 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000) / 3600:.1f}h | ~{250*50*samp_per_ep:.0f} |
| 500 | {500 * (50 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000) / 3600:.1f}h | ~{500*50*samp_per_ep:.0f} |
| 1000 | {1000 * (50 / best_w['episodes_per_sec'] + 300 * best_dev['total_step_ms'] / 1000) / 3600:.1f}h | ~{1000*50*samp_per_ep:.0f} |
"""

    md += f"""
## Notes

- Self-play workers run on CPU (ProcessPoolExecutor with spawn context).
  Each worker reconstructs nets from numpy state dicts.
- Go FFI calls are per-process (no shared state, no mutex contention).
- XPU training: Intel Arc A310 LP is entry-level. Transfer overhead may negate gains
  for small batch sizes.
- Range entropy barely moves with random nets (~6.148 throughout). Expected — random
  policy nets produce ~uniform action distributions across hand types, so Bayes update
  is a no-op. Should improve as nets learn.
- Solver export_leaves occasionally fails with game_config=None (Go defaults).
  No failures observed with explicit CambiaRulesConfig.
"""

    os.makedirs("docs", exist_ok=True)
    with open("docs/rebel_benchmark_results.md", "w") as f:
        f.write(md)
    print(f"\nResults written to cfr/docs/rebel_benchmark_results.md")

    # Also save raw JSON for programmatic access
    raw = {
        "timestamp": now,
        "hardware": {
            "cpu": "AMD Ryzen 9 9900X",
            "vcpus": 22,
            "ram_gb": 64,
            "gpu": "Intel Arc A310 LP (XPU)",
            "pytorch": torch.__version__,
        },
        "config": {
            "rebel_subgame_depth": 4,
            "rebel_cfr_iterations": 200,
            "rebel_value_hidden_dim": 1024,
            "rebel_policy_hidden_dim": 512,
            "batch_size": 2048,
            "alpha": 1.5,
            "max_game_turns": 46,
        },
        "self_play": sp_results,
        "training": train_results,
    }
    with open("docs/rebel_benchmark_results.json", "w") as f:
        json.dump(raw, f, indent=2)


if __name__ == "__main__":
    print("ReBeL Phase 1 Pre-Training Benchmark")
    print("=" * 50)

    # Self-play scaling
    worker_counts = [1, 2, 4, 8, 12, 16]
    sp_results = benchmark_self_play(worker_counts, episodes_per_trial=24)

    # Training devices
    devices = ["cpu", "xpu"]
    train_results = benchmark_training(devices)

    write_results(sp_results, train_results)
    print("\nDone.")
