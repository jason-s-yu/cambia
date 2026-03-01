# Deep CFR Architecture

Architecture reference for the Deep CFR training pipeline in `cfr/`.

## 1. Overview

Deep CFR replaces tabular regret storage with neural networks and fixed-capacity reservoir
buffers. Instead of a dict mapping infoset keys to regret arrays, the advantage network
approximates per-action regret values, and the strategy network approximates the average
strategy. Samples are collected via MCCFR traversals running in worker subprocesses and
stored in buffers via Algorithm R (reservoir sampling).

Two main sampling algorithms are supported:

- **OS-MCCFR** (outcome sampling): samples one action at each node, applies IS-weight
  correction to recover counterfactual regrets. Default for production runs.
- **ESCHER**: samples one action at traverser nodes, uses a learned value network to
  estimate counterfactual values for unchosen actions without IS weights.

SD-CFR mode is a variant that drops the strategy network entirely and computes the average
strategy from an exponentially-weighted moving average of advantage network snapshots.

## 2. Pipeline

Each training iteration proceeds as follows:

1. The trainer serializes the current advantage network weights as numpy arrays and passes
   them to worker processes.
2. Workers receive the weights, reconstruct the network on CPU, and run MCCFR traversals
   (OS or ESCHER) against the Go engine backend via FFI. Each traversal produces
   advantage samples (one per traverser decision node) and, in non-SD-CFR mode, strategy
   samples (one per opponent decision node). ESCHER traversals also produce value samples.
3. The trainer adds all returned samples to the respective reservoir buffers using
   Algorithm R, which gives each incoming sample an equal probability of inclusion
   regardless of buffer fill level.
4. The advantage network is trained on mini-batches drawn from the advantage buffer.
   Loss is weighted MSE: `mean((t+1)^alpha * MSE(pred, target))` where `t` is the
   sample's iteration index and `alpha=1.5` by default. Loss is computed over legal
   actions only (masked fill with 0; illegal positions are excluded). Gradients are
   clipped to max norm 1.0.
5. In standard mode, the strategy network is trained identically on the strategy buffer.
   In SD-CFR mode, this step is skipped.
6. With pipeline training enabled, the strategy network training step overlaps with the
   next round of traversals via `ProcessPoolExecutor` (spawn context, to avoid CUDA fork
   issues).
7. Repeat from step 1 with the updated network weights.

The adaptive training step count (`target_buffer_passes`) computes steps as:

```
steps = min(train_steps_per_iteration, max(250, int(len(buffer) * passes / batch_size)))
```

When `target_buffer_passes=0.0`, the fixed `train_steps_per_iteration` is used.

## 3. Network Architectures

All networks are defined in `src/networks.py`. The factory function
`build_advantage_network()` dispatches to the correct class based on `network_type`.

### AdvantageNetwork (MLP)

Three hidden layers: `input_dim -> 256 -> 256 -> 128 -> output_dim`. ReLU activations,
dropout(0.1) after the first two layers. Kaiming normal weight init. Illegal actions
are masked to `-inf` in the forward pass.

`network_type: "mlp"` routes here regardless of `use_residual`.

### ResidualAdvantageNetwork

Current best-performing architecture (~507K params at `hidden_dim=256, num_hidden_layers=3`).

Architecture: `input_proj -> [ResBlock x N] -> output_head`

- `input_proj`: `Linear(input_dim, hidden_dim) -> LayerNorm -> ReLU`
- `ResBlock`: `Linear(dim, dim) -> LayerNorm -> ReLU -> Dropout -> Linear(dim, dim) -> LayerNorm`, with additive skip and final ReLU
- `output_head`: `Linear(hidden_dim, hidden_dim//2) -> ReLU -> Linear(hidden_dim//2, output_dim)`

`network_type: "residual"` with `use_residual: true` routes here.

Factory: `build_advantage_network(use_residual=True, network_type="residual")`

### StrategyNetwork

Same MLP architecture as `AdvantageNetwork`, softmax output. Illegal actions are set to
`-inf` before softmax, yielding zero probability. Omitted in SD-CFR mode.

### HistoryValueNetwork (ESCHER)

Predicts scalar utility `V(h)` for the traversing player at a game history `h`. Input is
the concatenation of both players' infoset encodings (2 x input_dim). Architecture:

```
2*input_dim -> 512 -> 512 -> 256 -> 1
```

Hidden dim defaults to 512 (`value_hidden_dim`). Trained separately with its own buffer
and learning rate (`value_learning_rate=1e-3`, `value_target_buffer_passes=2.0`).

### SlotFiLM / SlotMultiply

Slot-structured networks designed for the EP-PBS interleaved encoding. Process the
224-dim input as `[public(42)][12 x slot(13)][pad(2)]`, applying per-slot FiLM
conditioning or multiplicative gating, then aggregating slot representations.

These architectures collapsed to ~24.9% mi(3) in ablation and are not recommended.
`network_type: "slot_film"` or `"slot_multiply"`.

## 4. Encoding

Encoding functions are in `src/encoding.py` and `src/constants.py`. The Go-side encoders
are in `engine/agent/encoding.go`, exported via CGO.

### Legacy (222-dim)

`INPUT_DIM = 222`. Flat per-slot one-hot encoding:

| Region | Dims |
|-|-|
| Own hand (6 slots x 15-dim one-hot) | 90 |
| Opponent beliefs (6 slots x 15-dim one-hot) | 90 |
| Own card count (normalized) | 1 |
| Opponent card count (normalized) | 1 |
| Drawn card bucket (11-dim: 10 buckets + NONE) | 11 |
| Discard top bucket (10-dim) | 10 |
| Stockpile estimate (4-dim) | 4 |
| Game phase (6-dim) | 6 |
| Decision context (6-dim) | 6 |
| Cambia caller (3-dim: SELF/OPP/NONE) | 3 |

### EP-PBS (224-dim)

`EP_PBS_INPUT_DIM = 224`. Epistemic Partial Belief State encoding. Two layouts:

- **Interleaved**: own/opponent slot pairs are adjacent in the tensor. Best-performing
  layout in ablations. Required for SlotFiLM. Used when `encoding_layout="interleaved"`
  or when `encoding_layout="auto"` and `network_type` is in
  `_INTERLEAVED_NETWORK_TYPES = {"slot_film", "slot_multiply"}`.
- **Flat de-aliased**: own slots followed by opponent slots, with de-aliased card
  indices. Tested in ablation; ~1pp below interleaved.

Config fields:

- `encoding_mode`: `"legacy"` or `"ep_pbs"`
- `encoding_layout`: `"auto"` (infer from network_type) or `"interleaved"` (force)

### N-Player

`N_PLAYER_INPUT_DIM = 856`, `N_PLAYER_NUM_ACTIONS = 620`. Used when `num_players > 2` (MaxPlayers=8).

### Action Space

2-player: `NUM_ACTIONS = 146`. N-player: `N_PLAYER_NUM_ACTIONS = 620`. All action indices
are fixed at build time; the mapping is defined in `src/encoding.py`.

## 5. Sampling Methods

| Method | Traverser nodes | Opponent nodes | IS correction | Cost |
|-|-|-|-|-|
| OS-MCCFR | Sample 1 action, IS-correct regrets | Sample 1 action | Yes | O(1)/node |
| ES-MCCFR | Enumerate all actions, exact regrets | Sample 1 action | No | O(branching)/node |
| ESCHER | Sample 1 action, value net for counterfactuals | Sample 1 action | No | O(1)/node + value net |

**OS-MCCFR** (`traversal_method: "outcome"`): uses exploration policy
`q(a|h) = epsilon * uniform + (1-epsilon) * sigma(a|h)` at both traverser and opponent
nodes. IS weights correct for the mixture. `MAX_IS_WEIGHT = 20.0` clips outlier weights.
Epsilon is applied only at the traverser's own nodes; opponent nodes use epsilon=0 (fixed
2026-02-27 to eliminate the IS correction bug that trained a best-response to a 60%-random
opponent).

**ES-MCCFR** (`traversal_method: "external"`): enumerates all traverser actions for exact
counterfactual regrets. Computationally infeasible for full Cambia due to branching factor.
Not used in production.

**ESCHER** (`traversal_method: "escher"`): no exploration epsilon, no IS correction. At
each traverser node, samples one action, recurses, then queries the value network to
estimate what the counterfactual values would have been for the unchosen actions. Requires
a separate `HistoryValueNetwork` trained jointly. `batch_counterfactuals=True` batches
these value queries for efficiency.

Config field: `traversal_method`. The traversal function entry points are
`_deep_traverse_os_go()` (OS) and `_escher_traverse_go()` (ESCHER) in `deep_worker.py`.

## 6. SD-CFR Mode

SD-CFR (Self-Distributional CFR) drops the strategy network entirely. Instead:

1. At each training step, the current advantage network weights are saved as a snapshot.
2. Up to `sd_cfr_max_snapshots=200` snapshots are retained.
3. At inference time, the average strategy is approximated by the EMA of all snapshots,
   weighted by `(t+1)^1.5` (linear weighting by default).
4. `use_ema=True` maintains a running EMA parameter vector for O(1) inference without
   summing all snapshots at query time.

Enable with `use_sd_cfr: true` in config.

Known limitation: EMA of parameters is not equivalent to the average of the strategies
those parameters define, because ReLU and the normalize operation are nonlinear. This
introduces ~1-2pp bias in the average strategy estimate.

## 7. Go Engine Backend

The Go engine is the primary backend for production training. It exposes a C API via CGO,
compiled to `cfr/libcambia.so`.

The FFI bridge in `src/ffi/bridge.py` loads the library once at module import time using
cffi (ABI mode, not ctypes). It defines two classes:

- **GoEngine**: manages game lifecycle. Methods: `new`, `apply_action`, `legal_actions_mask`,
  `save`, `restore`, `is_terminal`, `get_utility`, `acting_player`, `decision_ctx`.
  Save/restore uses full-state snapshots (cheaper than undo-log replay for the ~200-byte
  flat game state).
- **GoAgentState**: wraps a C agent handle for belief tracking. Methods: `update`,
  `update_both` (batch update both agents in one call), `clone`, `encode`.

Hot-loop exports added for traversal performance:

- `cambia_game_decision_ctx`: returns decision context without a full agent update call.
- `cambia_agents_update_both`: updates both agents in a single CGO boundary crossing.
- Unsafe encode variants: `cambia_agent_encode`, `cambia_agent_encode_eppbs`,
  `cambia_agent_encode_eppbs_interleaved`, `cambia_agent_encode_eppbs_dealiased`.

The library path defaults to `cfr/libcambia.so` and can be overridden with
`LIBCAMBIA_PATH`. Build with `make libcambia` from the repo root.

Config field: `engine_backend: "go"` (or `"python"` for the pure-Python fallback).

## 8. Worker Process Management

Workers run in a `ProcessPoolExecutor` with spawn context (required to avoid CUDA fork
deadlock when the main process holds a CUDA context).

**Pipeline training** (`pipeline_training: true`): the strategy network training step
runs in a background thread while the next round of traversals is already underway in
the worker pool. This overlaps CPU-bound traversal with GPU-bound training.

**Worker recycling**: each worker process accumulates ~723 MB RSS per training step due to
glibc malloc fragmentation (arena growth). The fix is to recycle workers after a fixed
number of tasks via `max_tasks_per_child`.

- `max_tasks_per_child: "auto"`: computes the recycling interval from system RAM and
  `worker_memory_budget_pct` (default 0.10). Formula in `_resolve_max_tasks_per_child()`
  in `deep_trainer.py`. On a 64 GB system with default 10% budget: interval = 6.
- `max_tasks_per_child: N`: explicit interval.
- `max_tasks_per_child: null`: no recycling (RSS grows unbounded).

## 9. Checkpoint Format

Checkpoints are saved as `.pt` files via `atomic_torch_save()` (write to temp, rename).

Contents:

- `advantage_net`: advantage network state dict
- `strategy_net`: strategy network state dict (absent in SD-CFR mode)
- `value_net`: ESCHER value network state dict (absent in non-ESCHER runs)
- `optimizer_adv`, `optimizer_strat`, `optimizer_value`: Adam optimizer states
- `scaler_adv`, `scaler_value`: GradScaler states (AMP only)
- `training_step`: current iteration counter
- `dcfr_config`: full config dict (used for backward compat encoding dim lookup)
- `loss_history`: list of per-step loss values
- `advantage_buffer_path`, `strategy_buffer_path`, `value_buffer_path`: paths to `.npz`
  buffer files stored alongside the checkpoint

SD-CFR advantage snapshots are stored as separate `.pt` files in the checkpoint directory,
named by iteration. On load, the 3-tier path fallback resolves snapshots from stored
absolute path, sibling file, or stripped `_iter_NNN` suffix.

Buffer `.npz` files contain: `features`, `targets`, `action_masks`, `iterations`.
