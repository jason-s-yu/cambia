# Definitions

Project-specific metrics, terminology, and abbreviations for the Cambia CFR training pipeline. Standard literature terms (CFR, MCCFR, OS-MCCFR, ES-MCCFR, SD-CFR, ESCHER, EMA, PPO, etc.) are not defined here.

## Metrics

**mean_imp / mi**
The primary evaluation metric. Mean win rate of the CFR agent (P0) across 5 baseline opponents, with 5000 games per baseline per checkpoint. Baselines: `random_no_cambia`, `random_late_cambia`, `imperfect_greedy`, `memory_heuristic`, `aggressive_snap`. Identical to mean_imp(5). Defined as the constant `MEAN_IMP_BASELINES` in eval_watcher.py, reeval_checkpoints.py, collect_metrics.py, plot_metrics.py, and validation_runner.py.

**mean_imp(3) / mi(3)**
Legacy metric. Mean win rate across 3 baselines: `imperfect_greedy`, `memory_heuristic`, `aggressive_snap`. Used in all results prior to 2026-02-27. Approximately 6-7 percentage points lower than mean_imp(5) because it excludes the random-floor baselines against which the agent performs near 50%. Not directly comparable to mean_imp(5).

**mean_imp(5) / mi(5)**
Identical to mean_imp. Explicit notation used when distinguishing from mean_imp(3).

**T1 Cambia rate**
Fraction of evaluated games where the CFR agent (P0) calls Cambia on turn 1. Healthy: below 10%. Warning: above 40% (indicates a degenerate early-termination equilibrium). A rising T1 Cambia rate alongside falling mean_imp is the primary signal of strategy regression. Reported in metrics.jsonl per checkpoint.

**T1 Cambia pathology**
The persistent 22-35% T1 Cambia rate observed across all OS-dCFR and SD-CFR training runs. The agent calls Cambia on turn 1 regardless of hand quality, even though unconditional T1 Cambia achieves only 39.4% WR (harmful). T1 Cambia is only profitable when the bottom-2 known cards sum to 9 or less (approximately 30.6% of deals).

## Encoding

**EP-PBS (Expectation-Propagation Public Belief State)**
The 224-dimensional tensor encoding (`EP_PBS_INPUT_DIM=224`) that encodes agent state using card bucket slots. Contrast with the legacy 222-dimensional flat encoding (`INPUT_DIM=222`). EP-PBS supports multiple layout variants: interleaved, flat, and flat de-aliased.

**encoding_layout**
Configuration field controlling how EP-PBS tensor features are arranged. Values:

| Value | Behavior |
|-|-|
| `"auto"` | Infer from network_type via `_INTERLEAVED_NETWORK_TYPES` |
| `"interleaved"` | Own/opponent slot pairs adjacent; best performing layout |
| `"flat_dealiased"` | Separate own/opponent blocks with de-aliased card indices |

Only relevant when the encoding mode is `"ep_pbs"`. Threaded from config.py YAML through deep_trainer.py and into deep_worker.py at 8 call sites.

**interleaved encoding**
EP-PBS layout where own and opponent card slots are interleaved: own_slot_0, opp_slot_0, own_slot_1, opp_slot_1, and so on. Provides spatial locality for slot-aware networks. Empirically outperformed flat layouts by approximately 1pp mi(3).

## Training

**target_buffer_passes**
Configuration field for adaptive train step scaling. When greater than 0, the number of SGD steps per iteration is computed as:

```
min(train_steps_per_iteration, max(250, int(len(buffer) * target_buffer_passes / batch_size)))
```

Prevents overtraining on small early buffers and undertraining on large late buffers. A value of 0.0 disables adaptive scaling and uses the fixed `train_steps_per_iteration`. The floor of 250 causes approximately 559x overtraining at iteration 1 (buffer size ~1832, batch size 4096) and is a known issue.

**advantage overfitting**
When the advantage network memorizes the current reservoir buffer rather than learning a generalizable strategy. Manifests as: mean_imp peaks then decays over 100-300 iterations, T1 Cambia rate climbs. Mitigated by adaptive train_steps (`target_buffer_passes`) and lower `train_steps_per_iteration`.

**worker recycling**
The process of killing and respawning pipeline training subprocesses (via `max_tasks_per_child`) to combat glibc malloc fragmentation. Workers accumulate approximately 723 MB of RSS per training step due to glibc retaining freed heap pages. `gc.collect()` and `malloc_trim(0)` are ineffective. The auto formula calculates the recycling interval from available system RAM:

```
max_tasks = clamp(floor((system_ram_mb * budget_pct - 1600) / 723), 2, 100)
```

On 64 GB with the default 10% budget: max_tasks_per_child = 6.

**EMA nonlinearity**
The gap between averaging network parameters and averaging the strategies those parameters produce. Strategy extraction via regret matching is nonlinear: `RM(EMA(theta_1,...,theta_N)) != mean(RM(theta_1),...,RM(theta_N))` because ReLU and normalize are nonlinear operations. Estimated impact: 1-2 percentage points of mean_imp. Can be fixed by averaging strategies post-RM instead of averaging parameters.

**gradient starvation**
A training bias caused by dividing per-sample MSE loss by the number of legal actions. Turn-1 decisions (approximately 3 legal actions) receive roughly 12x larger gradients than mid-game blind-swap decisions (approximately 36 legal actions). This structurally over-optimizes early-game accuracy at the expense of mid-game discrimination, and may partially explain the T1 Cambia pathology.

## Bugs and Fixes

**H3 fix**
Correction to OS-MCCFR traversal (applied 2026-02-27). Before the fix, exploration_epsilon (0.6) was applied at both traverser and opponent nodes, but importance sampling correction was only applied at traverser nodes. This caused the agent to train a best response against a 60%-random opponent rather than the learned strategy. Fix: set epsilon=0 at opponent nodes (gate on `player == updating_player`). Applied to `_deep_traverse_os_go`, `_deep_traverse_os_go_nplayer`, and `_deep_traverse_os`. ES-MCCFR and ESCHER are not affected.

**eval encoding mismatch**
A critical evaluation bug (fixed 2026-02-27) where `NeuralAgentWrapper._encode_eppbs()` hardcoded the flat encoder regardless of the checkpoint's `encoding_layout`. All interleaved and de-aliased eval numbers prior to the fix were invalid; training was unaffected. Fix: dispatch encoding based on `self._encoding_layout` and `self._network_type` read from checkpoint metadata.

## Run Directory Convention

**run directory**
Each training experiment lives in `cfr/runs/<descriptive-name>/` containing:

| File/Dir | Purpose |
|-|-|
| `config.yaml` | Launch and runtime configuration (Pydantic v2 model) |
| `run_meta.json` | Machine-readable run metadata (auto-generated by run_db) |
| `eval_summary.jsonl` | Aggregated eval results with Wilson CI (auto-generated) |
| `NOTES.md` | Experiment description (auto-generated if absent) |
| `checkpoints/` | Saved model weights |
| `logs/` | Training logs |

**NOTES.md**
Experiment description file auto-generated by `deep_trainer.py:_write_run_notes()` at training start. Contains config summary, hardware info, and expected outcomes. Will not overwrite an existing file.

## Baselines

**MEAN_IMP_BASELINES**
A Python constant (list of strings) containing the 5 baseline names used in mean_imp computation. Defined identically in: eval_watcher.py, reeval_checkpoints.py, collect_metrics.py, plot_metrics.py, and validation_runner.py. Current value: `["random_no_cambia", "random_late_cambia", "imperfect_greedy", "memory_heuristic", "aggressive_snap"]`.

**context-only baselines**
The `random` and `greedy` baselines. Evaluated and logged per checkpoint but excluded from mean_imp.

- `random`: True uniform distribution over all legal actions including Cambia. Games end on turn 1-2. Win rate against this baseline is near 50% and does not reflect agent strength.
- `greedy`: Perfect-information oracle. Provides a theoretical ceiling reference. Consistently beats all trained agents.
