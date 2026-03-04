# ReBeL Phase 1 Pre-Training Benchmark

**Date:** 2026-03-03T22:31:09+00:00
**Hardware:** AMD Ryzen 9 9900X (22 vCPUs, no SMT), 64GB RAM, Intel Arc A310 LP (XPU)
**PyTorch:** 2.10.0+xpu

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
| 1 | 24 | 228 | 18.52 | 1.3 | 12.31 | 9.5 |
| 2 | 24 | 177 | 20.04 | 1.2 | 8.83 | 7.4 |
| 4 | 24 | 194 | 27.1 | 0.89 | 7.16 | 8.1 |
| 8 | 24 | 238 | 40.93 | 0.59 | 5.81 | 9.9 |
| 12 | 24 | 241 | 38.59 | 0.62 | 6.24 | 10.0 |
| 16 | 16 | 210 | 36.24 | 0.44 | 5.8 | 13.1 |

**Conclusion:** Serial (1 worker) is fastest. ProcessPoolExecutor spawn overhead + CPU cache contention on Go solver's 468-wide CFR iterations makes parallelism counterproductive. Each episode is too short (~0.77s) for IPC to amortize.

## Training Step Throughput

| Device | Batch | Value (ms/step) | Policy (ms/step) | Total (ms/step) | Value (steps/s) | Policy (steps/s) |
|-|-|-|-|-|-|-|
| cpu | 2048 | 96.2 | 23.1 | 119.3 | 10.4 | 43.3 |
| xpu | 2048 | 26.2 | 6.6 | 32.8 | 38.1 | 151.1 |

**Conclusion:** XPU gives 3.6x training speedup. No multi-worker contention (self-play is serial on CPU, training is sequential after self-play).

## Per-Step Profiling (10 episodes, 114 steps)

| Component | ms/call | % total | Language |
|-|-|-|-|
| solve_ranged (Go CFR) | 151.77 | 65.5 | Go (compiled) |
| Policy matrix (468 fwd) | 40.95 | 22.4 | PyTorch (C++ backend) |
| Leaf value net | 26.93 | 11.6 | PyTorch (C++ backend) |
| Export leaves | 0.57 | 0.3 | Go FFI |
| Solver build | 0.14 | 0.1 | Go FFI |
| Apply action | 0.08 | 0.0 | Go FFI |
| PBS build + encode | 0.02 | 0.0 | Python/numpy |
| Range update | 0.01 | 0.0 | Python/numpy |

**Go FFI total: 66.0% — Python/PyTorch total: 34.0%**
Average: 182.9 ms/step.

### Language Switch Analysis

**Would rewriting in Go/C++ help?** No.

- The 66% bottleneck is already Go. C++ might gain single-digit % from compiler differences.
- The 34% is neural net matmul (PyTorch calls into C++ libTorch/MKL). In C++ you'd use libtorch or ONNX Runtime — same BLAS kernels.
- Python dispatch overhead is <0.1% of step time (PBS build + range update).
- The cost is algorithmic (200 CFR iters x depth-4 tree x 468 hand types), not linguistic.

### Viable Optimization Levers (No Rewrite)

| Optimization | Savings | Risk |
|-|-|-|
| Reduce CFR iters 200 → 100 | ~33% wall time | Weaker solver targets |
| ONNX Runtime for inference | ~2x on 34% PyTorch (~17% total) | Build complexity |
| Reduce subgame depth 4 → 3 | Fewer leaves, faster solve | Shallower lookahead |
| Skip policy matrix on low-info states | Variable | Stale range updates |

## Projected Iteration Time

**Best config:** 1 worker (serial) + XPU training

| Phase | Estimate |
|-|-|
| Self-play (50 episodes) | 38.5s |
| Self-play (100 episodes) | 76.9s |
| Training (200 steps x 2 nets) | 6.6s |
| Training (300 steps x 2 nets) | 9.8s |
| **Total iter (50 ep, 200 steps)** | **45.1s** |
| **Total iter (100 ep, 300 steps)** | **86.8s** |
| Samples/iter (50 ep) | ~475 |
| Samples/iter (100 ep) | ~950 |

## Iteration Count Projection

With 50 episodes/iter and ~9.5 samples/episode = ~475 samples/iter.
Buffer capacity 500K fills at iter ~1053.

| Target Iters | Wall Time | Total Samples |
|-|-|-|
| 100 | 1.3h | ~47,500 |
| 250 | 3.1h | ~118,750 |
| 500 | 6.3h | ~237,500 |
| 1000 | 12.5h | ~475,000 |

## Literature Calibration

| Paper | Game | Training Scale | Effective State Space |
|-|-|-|-|
| ReBeL (Liar's Dice) | 1x4f LD | 256M samples, 10K epochs, 1024 CFR/subgame, buf 2M, batch 512 | ~10^3 infosets |
| DeepStack | HUNL poker | ~10M solved subgames, 1000 CFR+ iters each | ~10^14 abstracted |
| Pluribus | 6P NLHE | 12,400 CPU core-hours MCCFR | >> HUNL |
| SoG | Poker/Chess/Go | 1.1M gradient steps (poker), 400 GT-CFR sims/move | ~10^160 raw |

**Our game:** ~10^9 raw infosets, PBS abstraction → ~10^5-10^6 effective PBS nodes.
Between Liar's Dice (~10^3) and HUNL (~10^14).

**Target:** >40% mean_imp (not convergence). PPO reaches 60% in 500K gradient steps.
ReBeL's subgame-solved targets are much richer than PPO's scalar rewards.

**Recommendation:** 1000 iterations (~475K samples, ~12.5h). Eval every 50 iters.
If mean_imp still climbing at 1000, extend to 2000.

## Notes

- Self-play workers run on CPU (ProcessPoolExecutor, spawn context).
  Each worker reconstructs nets from numpy state dicts.
- Go FFI calls are per-process (no shared state, no mutex contention).
- Range entropy barely moves with random nets (~6.148 throughout). Expected — random
  policy nets produce ~uniform action distributions across hand types, so Bayes update
  is a no-op. Should improve as nets learn.
- Solver export_leaves fails with game_config=None (Go defaults).
  No failures with explicit CambiaRulesConfig.
