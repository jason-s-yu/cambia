# Deep CFR Architecture

Overview of the Deep CFR training pipeline and how it differs from the original tabular CFR+ system.

## Tabular CFR+ (Original)

The original system stores per-infoset regret and strategy data in Python dictionaries (`defaultdict[InfosetKey, np.ndarray]`). Each training iteration:

1. A snapshot of the regret dict is pickled and sent to 23 worker processes.
2. Workers run outcome sampling traversals, accumulating local dict updates.
3. The main process merges updates into the global dicts (serial loop over all keys).
4. RM+ clamping is applied to all updated keys.

The primary bottleneck is the unbounded growth of the regret/strategy dicts as more information sets are visited. At scale, the merge step and snapshot pickling dominate wall-clock time.

## Deep CFR (New)

Deep CFR replaces the tabular storage with two neural networks and reservoir sampling buffers. The regret dict is replaced by the advantage network; the strategy dict is replaced by the strategy network. Training samples are collected via external sampling traversals and stored in fixed-capacity buffers.

### Pipeline

```
                        ┌───────────────────────────────────────────┐
                        │           DeepCFRTrainer                  │
                        │                                           │
                        │  AdvantageNetwork (Vθ) ─── 125K params    │
                        │  StrategyNetwork (Πφ) ─── 125K params     │
                        │  AdvantageBuf (Mv) ──── 2M samples        │
                        │  StrategyBuf (Mπ) ──── 2M samples         │
                        └────────┬──────────────────────┬───────────┘
                                 │                      │
                    ┌────────────┘                      └────────────┐
                    │ serialize weights                 │ collect    │
                    │ (numpy arrays)                    │ samples    │
                    ▼                                   │            │
        ┌───────────────────────┐                       │            │
        │    Worker Pool        │                       │            │
        │                       │                       │            │
        │  ┌─────────────────┐  │                       │            │
        │  │ deep_worker.py  │  │                       │            │
        │  │                 │  │                       │            │
        │  │  encode_infoset │──┼─ encoding.py          │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  AdvantageNet   │  │                       │            │
        │  │  (inference)    │  │                       │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  ReLU+normalize │──┼─ networks.py          │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  External       │  │                       │            │
        │  │  Sampling       │  │                       │            │
        │  │  Traversal      │──┼─ game/engine.py       │            │
        │  │       │         │  │                       │            │
        │  │       ▼         │  │                       │            │
        │  │  ReservoirSample│──┼───────────────────────┘            │
        │  │  (adv + strat)  │  │                                    │
        │  └─────────────────┘  │                                    │
        └───────────────────────┘                                    │
                                                                     │
        ┌────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Network Training (main process)    │
  │                                     │
  │  Sample batch from Mv               │
  │       │                             │
  │       ▼                             │
  │  Weighted MSE loss:                 │
  │  ((t+1)^α / mean) * MSE(Vθ, target) │
  │       │                             │
  │       ▼                             │
  │  Adam optimizer + grad clip         │
  │                                     │
  │  (repeat for Πφ with Mπ)            │
  └─────────────────────────────────────┘
```

### Data Flow Step by Step

1. The trainer serializes the current advantage network weights to numpy arrays and distributes them to workers.

2. Each worker initializes a game (`CambiaGameState`) and agent belief states (`AgentState`).

3. The worker runs a single external sampling traversal:
   - At the traverser's decision node: encodes the infoset via `encode_infoset()`, queries the advantage network for the current strategy (`ReLU + normalize`), then enumerates all legal actions. For each action, the worker applies it to the game state, recurses, and undoes the action. This produces exact per-action counterfactual values.
   - At opponent decision nodes: encodes the infoset, computes the strategy from the network, samples one action, recurses.
   - At terminal nodes: returns the utility vector.

4. After traversal, the worker has collected:
   - Advantage samples (one per traverser decision node): `(features, regret_target, action_mask, iteration)`.
   - Strategy samples (one per opponent decision node): `(features, strategy_target, action_mask, iteration)`.

5. The trainer adds all samples to the respective reservoir buffers via Algorithm R.

6. The trainer trains each network by sampling mini-batches from the buffers. The loss is weighted MSE where the weight for each sample is `(t+1)^alpha`, normalized by the batch mean. Loss is computed only over legal actions (masked). Gradients are clipped to max norm 1.0.

7. The loop repeats from step 1 with updated network weights.

## Go Engine Backend

The deep_worker now supports routing game simulation to either the Python engine or the Go engine, controlled by the `engine_backend` config field.

### Go Path

```
ctypes -> libcambia.so -> engine.GameState (Go)
```

Instead of `CambiaGameState.apply_action()` / `.undo()`, the Go path uses save/restore: the FFI bridge snapshots the full game state before recursing and restores it afterward. This matches the Go engine's `game_clone` / `game_restore` semantics, which are cheaper than a full undo-log replay for the game's flat memory layout (~200 bytes).

The FFI bridge lives in `src/ffi/bridge.py` and exposes two classes that are drop-in replacements for the Python engine:

- `GoEngine` handles game lifecycle (`new`, `apply_action`, `legal_actions_mask`, `save`, `restore`, `is_terminal`, `get_utility`).
- `GoAgentState` wraps the C agent handle (`update`, `clone`, `encode`).

The shared library is loaded once as a module-level singleton. The path defaults to `cfr/libcambia.so` and can be overridden with the `LIBCAMBIA_PATH` environment variable. Build it with `make libcambia` before using the Go backend.

### Updated Pipeline Diagram

The pipeline diagram above shows `game/engine.py` as the game simulation backend. With `engine_backend: go`, replace that node with `ffi/bridge.py -> libcambia.so`. Everything else (encoding, networks, reservoir buffers, training loop) is identical between the two paths.

```
  External                      External
  Sampling                      Sampling
  Traversal ─── game/engine.py  Traversal ─── ffi/bridge.py ─── libcambia.so
  (Python)                      (Go)
```

The two paths produce the same sample format (`ReservoirSample`) and are interchangeable from the trainer's perspective.

### External Sampling vs. Outcome Sampling

The traversal mode changed from outcome sampling to external sampling:

| Aspect                | Outcome Sampling (old)      | External Sampling (new)                |
| --------------------- | --------------------------- | -------------------------------------- |
| Traverser nodes       | Sample 1 action, IS-correct | Enumerate all actions                  |
| Opponent nodes        | Sample 1 action             | Sample 1 action                        |
| Regret quality        | Noisy (importance sampling) | Exact (no IS correction)               |
| Cost per traversal    | O(1) per node               | O(branching factor) at traverser nodes |
| Samples per traversal | 1 per visited node          | 1 per traverser node (higher quality)  |

External sampling is more expensive per traversal but produces exact regret estimates, requiring fewer total iterations for convergence. For Cambia's typical branching factor of 5-15 at decision nodes, this is a favorable tradeoff.

### What Changed vs. What Stayed

Changed:

- Storage: tabular dicts -> neural networks + reservoir buffers
- Traversal: outcome sampling -> external sampling
- Worker output: dict updates -> `ReservoirSample` lists
- Merge step: eliminated (samples append to buffers, no per-key merging)
- Strategy lookup: `regret_sum[key] -> RM+` -> `network(features) -> ReLU+normalize`
- Pickling: entire regret dict -> fixed-size network weights (~500KB)

Unchanged:

- Game engine (`CambiaGameState`) and its apply/undo mechanics
- Agent belief state (`AgentState`) and observation model
- Action types (`GameAction` NamedTuples)
- Card abstraction (`CardBucket`, `DecayCategory`)
- Memory decay model (event and time-based)
- Worker-level logging and progress reporting infrastructure

### ES Validation

The trainer runs periodic exploitability checks using `ESValidator` (in `src/cfr/es_validator.py`). It runs short-depth ES traversals against both players and reports metrics: mean regret, max regret, strategy entropy, node count, and elapsed time. This runs every `es_validation_interval` training steps (default 10) and logs to the standard training metrics. It supports both the Python and Go backends via the same `engine_backend` config flag. Set `es_validation_interval: 0` to disable.
