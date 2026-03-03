# Deep CFR Module Reference

Reference documentation for the core modules used in the Deep CFR training pipeline.

## `src/encoding.py`

Converts `AgentState` + legal actions into fixed-size numpy tensors for neural network input.

### Constants

- `MAX_HAND = 6` -- maximum hand slots encoded. Hands exceeding this are clamped.
- `SLOT_ENCODING_DIM = 15` -- one-hot dimension per hand slot (10 `CardBucket` + 3 `DecayCategory` + UNKNOWN + EMPTY).
- `INPUT_DIM = 222` -- legacy flat feature vector length.
- `EP_PBS_INPUT_DIM = 224` -- EP-PBS feature vector length (used by encoding_mode="ep_pbs").
- `NUM_ACTIONS = 146` -- total fixed action space size (2-player).

### Legacy Feature Vector Layout (222-dim)

| Offset | Feature | Dimensions | Encoding |
|-|-|-|-|
| 0-89 | Own hand (6 slots) | 6 x 15 = 90 | One-hot per slot |
| 90-179 | Opponent beliefs (6 slots) | 6 x 15 = 90 | One-hot per slot |
| 180 | Own card count | 1 | Normalized scalar /6 |
| 181 | Opponent card count | 1 | Normalized scalar /6 |
| 182-192 | Drawn card bucket | 11 | One-hot (10 buckets + NONE) |
| 193-202 | Discard top bucket | 10 | One-hot |
| 203-206 | Stockpile estimate | 4 | One-hot |
| 207-212 | Game phase | 6 | One-hot |
| 213-218 | Decision context | 6 | One-hot |
| 219-221 | Cambia caller | 3 | One-hot (SELF/OPP/NONE) |

Each hand/belief slot uses a unified 15-dim one-hot encoding:

- Indices 0-8: `CardBucket` values (ZERO through HIGH_KING)
- Index 9: unused (gap between bucket 8 and decay 10)
- Indices 10-12: `DecayCategory` values (LIKELY_LOW, LIKELY_MID, LIKELY_HIGH)
- Index 13: UNKNOWN (shared by both `CardBucket.UNKNOWN` and `DecayCategory.UNKNOWN`)
- Index 14: EMPTY (slot does not exist)

### EP-PBS Feature Vector Layout (224-dim)

The EP-PBS encoding separates epistemic tag (what you know about knowing) from identity (what the card is). Two layouts exist, both producing a 224-dim vector:

**Flat layout** (default for non-slot networks):

| Offset | Feature | Dimensions |
|-|-|-|
| 0-39 | Public features | 40 |
| 40-87 | Slot tags: 12 slots x 4-dim one-hot | 48 |
| 88-195 | Slot identities: 12 slots x 9-dim one-hot | 108 |
| 196-199 | Hand sizes + padding | 4 |
| 200-223 | History features (obs ages, dead-card histogram, turn progress) | 24 |

**Interleaved layout** (required for SlotFiLM and slot_multiply networks):

Slot layout pairs own and opponent slots: `[own0, opp0, own1, opp1, ...]`. Public features use a 42-dim encoding with hand sizes appended. Slot data is 12 x 13-dim per slot (4-dim tag + 9-dim identity), followed by 2 bytes of padding, then 24 history dims.

**De-aliased flat layout** (`encoding_layout: "flat_dealiased"`):

Same structure as flat, but empty slots beyond `hand_size` are all-zeros in both tag and identity regions. Hand sizes are encoded at dims [196] and [197]. History features at [200-223].

### Encoding Layout Routing

`_INTERLEAVED_NETWORK_TYPES = frozenset({"slot_film", "slot_multiply"})` -- when `encoding_layout="auto"` (the default), network types in this set force interleaved layout automatically. Set `encoding_layout="interleaved"` to force interleaved regardless of network type (needed for `network_type: "residual"` runs that use the interleaved encoding, e.g. the Phase 2 best-result config).

The routing function `_encode_ep_pbs(agent_state, decision_context, drawn_bucket, network_type, encoding_layout)` in `deep_worker.py` handles dispatch.

### Action Index Layout

| Range | Action Type | Parameters |
|-|-|-|
| 0 | `ActionDrawStockpile` | -- |
| 1 | `ActionDrawDiscard` | -- |
| 2 | `ActionCallCambia` | -- |
| 3 | `ActionDiscard(use_ability=False)` | -- |
| 4 | `ActionDiscard(use_ability=True)` | -- |
| 5-10 | `ActionReplace(idx)` | idx 0-5 |
| 11-16 | `ActionAbilityPeekOwnSelect(idx)` | idx 0-5 |
| 17-22 | `ActionAbilityPeekOtherSelect(idx)` | idx 0-5 |
| 23-58 | `ActionAbilityBlindSwapSelect(own, opp)` | own*6 + opp |
| 59-94 | `ActionAbilityKingLookSelect(own, opp)` | own*6 + opp |
| 95-96 | `ActionAbilityKingSwapDecision(bool)` | False=95, True=96 |
| 97 | `ActionPassSnap` | -- |
| 98-103 | `ActionSnapOwn(idx)` | idx 0-5 |
| 104-109 | `ActionSnapOpponent(idx)` | idx 0-5 |
| 110-145 | `ActionSnapOpponentMove(own, slot)` | own*6 + slot |

### Public Functions

```python
def encode_infoset(
    agent_state: AgentState,
    decision_context: DecisionContext,
    drawn_card_bucket: Optional[CardBucket] = None,
) -> np.ndarray:
```

Encodes an agent's information set into a `(222,)` float32 array using the legacy layout. The `drawn_card_bucket` parameter should be provided at `POST_DRAW` decision points.

```python
def encode_infoset_eppbs_interleaved(
    slot_tags: list,          # 12 EpistemicTag values
    slot_buckets: list,       # 12 bucket values (0 if unknown)
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    ...                       # additional public state args
) -> np.ndarray:              # (224,) float32
```

EP-PBS encoding with interleaved own/opponent slot pairs. Required for SlotFiLM and slot_multiply network architectures. In practice, the Go FFI path via `GoAgentState.encode_eppbs_interleaved()` is used during training; this Python function exists for testing and the Python-engine fallback path.

```python
def encode_infoset_eppbs_dealiased(
    slot_tags: list,
    slot_buckets: list,
    discard_top_bucket: int,
    stock_estimate: int,
    game_phase: int,
    ...
) -> np.ndarray:              # (224,) float32
```

De-aliased flat EP-PBS encoding. Fixes the aliasing problem in the standard flat encoder where empty slots (past hand_size) could have non-zero values inherited from prior game states. Empty slots are forced to all-zeros.

```python
def action_to_index(action: GameAction) -> int:
```

Maps a `GameAction` NamedTuple to its fixed index in `[0, 146)`. Raises `ValueError` for unrecognized types or out-of-range hand indices.

```python
def index_to_action(index: int, legal_actions: List[GameAction]) -> GameAction:
```

Reverse mapping: finds the legal action matching a given index by scanning the list. Raises `ValueError` if no match is found.

```python
def encode_action_mask(legal_actions: List[GameAction]) -> np.ndarray:
```

Creates a `(146,)` boolean mask with `True` for each legal action's index. Actions with hand indices beyond `MAX_HAND` are silently skipped.

### Design Decisions

- The drawn card bucket is encoded as a separate 11-dim one-hot rather than folding it into the hand encoding. This makes the feature vector layout static regardless of whether a card has been drawn.
- `index_to_action` uses linear scan over legal actions rather than building a reverse lookup table, since legal action lists are small (typically 5-15 actions).
- Hand slots beyond `MAX_HAND` are silently dropped in `encode_action_mask`. The `own_card_count` scalar retains the true count as a signal to the network.
- EP-PBS separates epistemic tag (what you know about knowing: PRIV_OWN, PRIV_OPP, MEMORY, UNKNOWN) from bucket identity. This factored structure is the motivation for slot-aware architectures.

## `src/networks.py`

PyTorch `nn.Module` definitions for the advantage, strategy, and value networks.

### Classes

#### `AdvantageNetwork` (MLP baseline)

```python
class AdvantageNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 222,
        hidden_dim: int = 256,
        output_dim: int = 146,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    )

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
```

Predicts per-action advantage (regret) values. Architecture:

```
Linear(input_dim, 256) -> ReLU -> Dropout(0.1)
-> Linear(256, 256) -> ReLU -> Dropout(0.1)
-> Linear(256, 128) -> ReLU -> Linear(128, output_dim)
```

Input shapes: `features (batch, input_dim)`, `action_mask (batch, output_dim)` bool. Output: `(batch, output_dim)` float, with illegal actions set to `-inf`. ~175K parameters with default dims.

#### `ResidualAdvantageNetwork` (current best architecture)

```python
class ResidualAdvantageNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 222,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        output_dim: int = 146,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    )

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
```

Advantage network with residual connections for improved gradient flow. Architecture:

```
input_proj: Linear(input_dim, hidden_dim) -> LayerNorm -> ReLU
-> [ResBlock x num_hidden_layers]
-> output_head: Linear(hidden_dim, hidden_dim//2) -> ReLU -> Linear(hidden_dim//2, output_dim)
```

Each `_ResBlock`: `Linear(dim, dim) -> LayerNorm -> ReLU -> Dropout -> Linear(dim, dim) -> LayerNorm + skip connection -> ReLU`.

`num_hidden_layers` controls the number of residual blocks. With `hidden_dim=256` and `num_hidden_layers=3`, approximately 507K parameters. FiLM gamma/beta init is zeros; all other linear layers use Kaiming normal init.

This is the architecture used by all production runs. Enabled via `network_type: "residual"` and `use_residual: true` in config.

#### `StrategyNetwork`

```python
class StrategyNetwork(nn.Module):
    def __init__(...)  # Same signature as AdvantageNetwork

    def forward(self, features: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
```

Predicts per-action strategy probabilities. Same MLP architecture as `AdvantageNetwork`. Masks illegal actions to `-inf` before softmax, producing a valid probability distribution. Includes NaN guard for edge cases where all actions are masked. Not used in SD-CFR mode (`use_sd_cfr: true`).

#### `HistoryValueNetwork` (ESCHER value network)

```python
class HistoryValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM * 2,  # 444 for legacy, 448 for EP-PBS
        hidden_dim: int = 512,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    )

    def forward(self, features_both: torch.Tensor) -> torch.Tensor:
        # Input: (batch, input_dim), Output: (batch, 1)
```

Predicts scalar utility `V(h)` for the traversing player at game history `h`. Input is the concatenation of both players' infoset encodings: `2 * input_dim` dimensions (444 for legacy 222-dim encoding, 448 for EP-PBS 224-dim encoding). Used in ESCHER traversal to estimate counterfactual regrets for unchosen actions without recursing.

Architecture: `input_dim -> 512 -> 512 -> 256 -> 1` with ReLU and Dropout.

Note: `input_dim` is computed dynamically in `deep_trainer.py` based on `dcfr_config.input_dim * 2` to avoid the hardcoding bug that invalidated the first ESCHER run.

#### `SlotFiLMAdvantageNetwork` (experimental, not recommended)

Slot-structured advantage network with Feature-wise Linear Modulation (FiLM) conditioning. Processes the EP-PBS interleaved encoding. Requires `network_type: "slot_film"`. Collapsed to 24.9% mean_imp(3) in ablation (Phase 1a). Not recommended for new runs.

#### Deprecated Networks

`PBSValueNetwork` and `PBSPolicyNetwork` emit `DeprecationWarning` on instantiation. ReBeL/PBS-based subgame solving is mathematically unsound for N-player free-for-all games with continuous beliefs.

### Factory Function

```python
def build_advantage_network(
    input_dim: int = INPUT_DIM,
    hidden_dim: int = 256,
    output_dim: int = NUM_ACTIONS,
    dropout: float = 0.1,
    validate_inputs: bool = True,
    num_hidden_layers: int = 2,
    use_residual: bool = False,
    network_type: str = "residual",
    **kwargs,
) -> nn.Module:
```

Factory that dispatches to the correct network class based on `network_type`:

| network_type | use_residual | Result |
|-|-|-|
| "mlp" | any | `AdvantageNetwork` |
| "residual" | False | `AdvantageNetwork` |
| "residual" | True | `ResidualAdvantageNetwork` |
| "slot_film" | any | `SlotFiLMAdvantageNetwork(use_film=True)` |
| "slot_multiply" | any | `SlotFiLMAdvantageNetwork(use_film=False)` |

`**kwargs` forwards extra arguments to `SlotFiLMAdvantageNetwork` (e.g. `use_pos_embed`, `num_players`). The factory is the sole construction path used by `deep_trainer.py` for creating the advantage network.

### Standalone Function

```python
def get_strategy_from_advantages(
    advantages: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
```

Converts advantage network output into a strategy distribution using regret matching: `ReLU(advantages) -> mask illegal -> normalize`. Falls back to uniform over legal actions when all advantages are non-positive.

Used during both traversal (to derive current strategy from advantage predictions) and evaluation (in `NeuralAgentWrapper` to produce action probabilities). Using sampled actions from this distribution is the primary eval mode -- argmax eval is 2.3pp worse in practice.

### Design Decisions

- The advantage network uses `-inf` masking rather than zeroing because downstream consumers need to distinguish "predicted zero regret" from "illegal action."
- The strategy network uses softmax (not ReLU+normalize) because it is trained to predict the average strategy directly, not instantaneous advantages.
- `get_strategy_from_advantages` uses ReLU+normalize (not softmax) to match the RM+ convergence guarantee: only actions with positive predicted advantage get probability mass.
- `validate_inputs` gates the NaN check in the forward pass. Disabling it eliminates the GPU-to-CPU sync on every forward pass (~85% GPU overhead at large batch sizes).

## `src/reservoir.py`

Fixed-capacity reservoir sampling buffers for Deep CFR training data.

### Classes

```python
@dataclass
class ReservoirSample:
    features: np.ndarray      # (input_dim,) float32
    target: np.ndarray        # (NUM_ACTIONS,) float32 -- regrets or strategy
    action_mask: np.ndarray   # (NUM_ACTIONS,) bool
    iteration: int            # CFR iteration number for t^alpha weighting
    infoset_key_raw: Optional[Tuple] = None  # Debugging metadata (not persisted)
```

A single training sample. The `iteration` field is used during training to compute `(t+1)^alpha` weights in the loss function. For ESCHER value samples, `target` is a scalar wrapped in a 1-dim array and `action_mask` is ignored.

```python
class ReservoirBuffer:
    def __init__(self, capacity: int = 2_000_000)
    def __len__(self) -> int
    def add(self, sample: ReservoirSample)
    def sample_batch(self, batch_size: int) -> List[ReservoirSample]
    def save(self, path: str)
    def load(self, path: str)
    def resize(self, new_capacity: int)
    def clear(self)
```

Implements Vitter's Algorithm R: each new sample has `capacity / seen_count` probability of entering the buffer. The `seen_count` is tracked separately from buffer length to maintain the guarantee after the buffer fills.

`save()` and `load()` use `np.savez_compressed`. On load, if the saved capacity differs from the current instance's capacity, the buffer is truncated via random subsampling. `resize()` adjusts capacity at runtime.

### Standalone Function

```python
def samples_to_tensors(
    samples: List[ReservoirSample]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Batch-converts a list of samples into stacked numpy arrays: `(N, input_dim)` features, `(N, NUM_ACTIONS)` targets, `(N, NUM_ACTIONS)` masks, `(N,)` iterations. Returns empty arrays with correct shapes when given an empty list.

### Memory Estimates

| Capacity | Approx. memory per buffer |
|-|-|
| 100K | ~60 MB |
| 500K | ~300 MB |
| 2M (default) | ~1.2 GB |
| 5M | ~3 GB |

## `src/cfr/deep_trainer.py`

Orchestrates the Deep CFR training loop: traversal scheduling, sample collection, and network training.

### `DeepCFRConfig`

```python
@dataclass
class DeepCFRConfig:
    # Network architecture
    input_dim: int = INPUT_DIM          # 222 (set from encoding_mode at construction)
    hidden_dim: int = 256
    output_dim: int = NUM_ACTIONS       # 146
    dropout: float = 0.1
    num_hidden_layers: int = 3
    use_residual: bool = True
    network_type: str = "residual"      # "mlp", "residual", "slot_film", "slot_multiply"
    use_pos_embed: bool = True          # position embeddings in SlotFiLM

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 2048
    train_steps_per_iteration: int = 4000
    alpha: float = 1.5                  # iteration weighting exponent (t^alpha)
    target_buffer_passes: float = 0.0   # adaptive train steps; 0 = fixed
    traversals_per_step: int = 1000

    # Device: "auto" = cuda -> xpu -> cpu
    device: str = "auto"

    # Sampling method and epsilon
    sampling_method: str = "outcome"
    exploration_epsilon: float = 0.6

    # Engine backend
    engine_backend: str = "python"      # "python" or "go"

    # Buffer capacities
    advantage_buffer_capacity: int = 2_000_000
    strategy_buffer_capacity: int = 2_000_000

    # SD-CFR mode
    use_sd_cfr: bool = False
    use_ema: bool = True
    sd_cfr_max_snapshots: int = 200

    # ESCHER value network
    value_hidden_dim: int = 512
    value_learning_rate: float = 1e-3
    value_buffer_capacity: int = 2_000_000
    value_target_buffer_passes: float = 2.0

    # Encoding
    encoding_mode: str = "legacy"       # "legacy" or "ep_pbs"
    encoding_layout: str = "auto"       # "auto", "interleaved", or "flat_dealiased"

    # Worker recycling (glibc malloc fragmentation mitigation)
    max_tasks_per_child: Optional[Union[int, str]] = "auto"
    worker_memory_budget_pct: float = 0.10

    # Pipeline training
    pipeline_training: bool = True
    use_amp: bool = False
    use_compile: bool = False
    validate_inputs: bool = True
    ...
```

This is the runtime config dataclass (in `deep_trainer.py`). It is separate from `DeepCfrConfig` (in `config.py`), which is the YAML-facing Pydantic model. `from_yaml_config()` bridges the two.

Full field list and defaults: see source. Reference `cfr/config/` YAML files for typical production values.

### Key Methods

```python
@classmethod
def DeepCFRConfig.from_yaml_config(cls, config: Config, **overrides) -> DeepCFRConfig:
```

Constructs a `DeepCFRConfig` from `Config.deep_cfr` (a `DeepCfrConfig` from `config.py`), applying any CLI overrides. Override keys with `None` values are ignored.

```python
def _resolve_device(device: str) -> str:
```

Module-level function (not a method). Resolves `"auto"` to `"cuda"` if `torch.cuda.is_available()`, else `"xpu"` if `torch.xpu.is_available()`, else `"cpu"`. Raises `RuntimeError` if an explicit device (`"xpu"`, `"cuda"`) is requested but not available.

```python
def _resolve_max_tasks_per_child(
    max_tasks_per_child: Optional[Union[int, str]],
    worker_memory_budget_pct: float,
) -> Optional[int]:
```

Module-level function. Resolves `"auto"` using empirical RSS growth constants (warmup=1600 MB, growth=723 MB/step) and system RAM. On a 64 GB system with `worker_memory_budget_pct=0.10`: auto resolves to `mtpc=6`. `None` disables recycling (RSS grows unbounded).

```python
def _update_ema(self):
```

Updates EMA serving weights after each advantage snapshot (SD-CFR mode). Uses dynamically weighted EMA to match the `t^alpha` weighted ensemble in O(1) space:

```
w_T = (T+1)^alpha
new_sum = old_sum + w_T
theta_EMA = (old_sum/new_sum) * theta_EMA + (w_T/new_sum) * theta_current
```

Only runs when `use_sd_cfr=True` and `use_ema=True`.

```python
def _compute_train_steps(self, buffer) -> int:
```

Computes adaptive training steps when `target_buffer_passes > 0`:

```
adaptive_steps = int(len(buffer) * target_buffer_passes / batch_size)
num_steps = min(train_steps_per_iteration, max(10, adaptive_steps))
```

Floor of 10 prevents running zero steps on small early buffers. Returns `train_steps_per_iteration` unchanged when `target_buffer_passes == 0.0`.

```python
def _train_value_network(self, num_steps: int) -> float:
```

Trains the ESCHER value network on the value buffer. Separate from `_train_network()` because the value net outputs a scalar `(batch, 1)` rather than a per-action vector, so action masking does not apply. Uses the same `(t+1)^alpha` weighting as advantage training. Returns mean MSE loss for the step.

```python
def _write_run_notes(self):
```

Auto-generates `NOTES.md` in the run directory at training start if one does not already exist. Writes the run config, start time, and key hyperparameters. Will not overwrite manually created notes.

### `DeepCFRTrainer`

```python
class DeepCFRTrainer:
    def __init__(
        self, config: Config,
        deep_cfr_config: Optional[DeepCFRConfig] = None,
        run_log_dir: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        shutdown_event: Optional[threading.Event] = None,
        progress_queue: Optional[ProgressQueue] = None,
        live_display_manager: Optional[LiveDisplayManager] = None,
        archive_queue: Optional[Union[queue.Queue, multiprocessing.Queue]] = None,
    )

    def train(self, num_training_steps: Optional[int] = None)
    def save_checkpoint(self, filepath: Optional[str] = None)
    def load_checkpoint(self, filepath: Optional[str] = None)
    def get_strategy_network(self) -> StrategyNetwork
    def get_advantage_network(self) -> AdvantageNetwork
```

### Training Loop

Each training step:

1. Serialize advantage network weights (and value net weights if ESCHER) as numpy arrays for pickle-friendly worker transfer.
2. Dispatch `traversals_per_step` traversals across the worker pool via `ProcessPoolExecutor`. Workers run `run_deep_cfr_worker()` and return `DeepCFRWorkerResult`.
3. Add collected samples to the advantage buffer (and strategy buffer, and value buffer for ESCHER).
4. Train the advantage network on the advantage buffer using weighted MSE: `weight = ((t+1)^alpha) / mean_weight`, applied per-sample. Loss computed only over legal actions.
5. Train the strategy network on the strategy buffer with the same formulation (skipped in SD-CFR mode).
6. For ESCHER: train the value network on the value buffer via `_train_value_network()`.
7. For SD-CFR: take a snapshot of the current advantage network and update EMA weights.
8. Save checkpoint at configured intervals.

Pipeline training (`pipeline_training=True`) overlaps traversal in one process with strategy network training in the main process. Worker recycling via `max_tasks_per_child` prevents RSS growth from glibc malloc fragmentation.

### Checkpoint Format

The main checkpoint (`.pt` file via `torch.save`) contains:

- Advantage and strategy network state dicts and optimizer state dicts
- Value net state dict and optimizer state dict (ESCHER only)
- GradScaler state dict (AMP only)
- Training step, total traversals, iteration count
- Full `dcfr_config` dict (used for backward-compat encoding dispatch on load)
- Loss history for all networks
- SD-CFR snapshot paths and EMA weight sum (SD-CFR only)
- Paths to reservoir buffer `.npz` files (saved alongside)

### Design Decisions

- Network weights are serialized to numpy arrays before passing to workers. This avoids workers needing to handle device placement and keeps pickled data device-agnostic.
- `_snapshot_count` (SD-CFR) tracks reservoir stream count independently of `training_step`. This prevents warm-start bias when loading a checkpoint mid-training.
- Gradient clipping (`max_norm=1.0`) is applied to both networks.
- NaN loss fix: use `predictions.masked_fill(~mask_bool, 0.0)` rather than multiplication, to avoid `0 * -inf = NaN`.
- Two separate Pydantic models (`DeepCfrConfig` in config.py, `DeepCFRConfig` in deep_trainer.py) exist for historical reasons. The YAML-facing one uses snake_case fields with YAML-compatible types; the runtime one has the full set of derived fields.

## `src/cfr/deep_worker.py`

Implements the Deep CFR worker process. Supports OS-MCCFR, ES-MCCFR, and ESCHER traversal algorithms against the Go engine backend.

### Constants

```python
MAX_IS_WEIGHT = 20.0
```

Module-level constant for importance sampling weight clipping in OS-MCCFR. Applied as `min(1.0 / sampling_prob, MAX_IS_WEIGHT)` to bound variance from rare actions.

### `DeepCFRWorkerResult`

```python
@dataclass
class DeepCFRWorkerResult:
    advantage_samples: List[ReservoirSample] = field(default_factory=list)
    strategy_samples: List[ReservoirSample] = field(default_factory=list)
    value_samples: List[ReservoirSample] = field(default_factory=list)   # ESCHER only
    stats: WorkerStats
    simulation_nodes: List[SimulationNodeData]
    final_utility: Optional[List[float]]
```

Returned by each worker invocation. `value_samples` is populated only during ESCHER traversal.

### Traversal Functions

#### `_deep_traverse_os_go` (primary production path)

```python
def _deep_traverse_os_go(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],
    updating_player: int,
    network: Optional[AdvantageNetwork],
    iteration: int,
    config: Config,
    advantage_samples: List[ReservoirSample],
    strategy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    ...,
    exploration_epsilon: float,
    depth_limit: Optional[int] = None,
    recursion_limit: Optional[int] = None,
) -> np.ndarray:
```

Recursive Outcome Sampling MCCFR using the Go engine backend. At every node, samples ONE action using the exploration policy `q(a|h) = epsilon * uniform + (1-epsilon) * sigma(a|h)`, then applies IS correction.

H3 fix (2026-02-27): epsilon is applied only at traverser nodes. At opponent nodes, the strategy is sampled directly without epsilon mixing. Prior to this fix, the agent was training a best response to a 60%-random opponent.

Returns utility vector `(2,)` float64.

#### `_deep_traverse_os_go_nplayer`

```python
def _deep_traverse_os_go_nplayer(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],  # length = num_players
    updating_player: int,
    ...
) -> np.ndarray:
```

N-player variant of `_deep_traverse_os_go`. Uses 620-action space and 856-dim encoding. Structure mirrors the 2-player version; returns utility vector `(num_players,)` float64.

#### `_escher_traverse_go`

```python
def _escher_traverse_go(
    engine: "GoEngine",
    agent_states: List["GoAgentState"],
    updating_player: int,
    regret_net: Optional[AdvantageNetwork],
    value_net: Optional[HistoryValueNetwork],
    iteration: int,
    config: Config,
    regret_samples: List[ReservoirSample],
    value_samples: List[ReservoirSample],
    policy_samples: List[ReservoirSample],
    depth: int,
    worker_stats: WorkerStats,
    ...,
    batch_counterfactuals: bool = True,
    depth_limit: Optional[int] = None,
    recursion_limit: Optional[int] = None,
) -> np.ndarray:
```

ESCHER traversal using the Go engine backend. Key differences from OS-MCCFR:

- Samples directly from the strategy (no epsilon mixing, no IS weights).
- Encodes BOTH players' infosets and concatenates to a `2 * input_dim` vector for value net input.
- Stores value samples at ALL non-terminal nodes (not just traverser nodes).
- Computes counterfactual regrets via the value network for unchosen actions rather than recursing into each.
- Stores regret samples only at traverser nodes.
- Stores policy samples at opponent nodes.

`batch_counterfactuals=True` groups counterfactual value queries into a single batched forward pass rather than one call per unchosen action.

#### `_deep_traverse_os` and `_deep_traverse_es` (Python-engine variants)

Python-engine traversal functions. Same logic as the Go variants but use `CambiaGameState.apply_action()` + `AgentState.update()` rather than FFI handles. Used as fallback when `engine_backend="python"`. The Go backend is the production path; these are retained for testing and reference.

### Encoding Layout Dispatch

```python
_INTERLEAVED_NETWORK_TYPES = frozenset({"slot_film", "slot_multiply"})

def _encode_ep_pbs(agent_state, decision_context, drawn_bucket,
                   network_type: str, encoding_layout: str = "auto") -> np.ndarray:
```

Routes EP-PBS encoding to one of three layouts:

- `encoding_layout="flat_dealiased"`: calls `agent_state.encode_eppbs_dealiased()`
- `encoding_layout="interleaved"` OR `network_type in _INTERLEAVED_NETWORK_TYPES`: calls `agent_state.encode_eppbs_interleaved()`
- Otherwise: calls `agent_state.encode_eppbs()` (standard flat layout)

The `encoding_layout` parameter is threaded through from `Config.deep_cfr.encoding_layout` at each of the 8 call sites in `deep_worker.py`.

### Worker Entry Point

```python
def run_deep_cfr_worker(
    worker_args: Tuple[
        int,        # iteration
        Config,
        Optional[Dict],   # network_weights_serialized
        Dict,             # network_config
        Optional[queue.Queue],
        Optional[Any],    # archive_queue
        int,        # worker_id
        str,        # run_log_dir
        str,        # run_timestamp
    ],
    file_handler_override: Optional[logging.Handler] = None,
) -> Optional[DeepCFRWorkerResult]:
```

Top-level worker entry point invoked by the trainer via `ProcessPoolExecutor`. Sets up per-worker logging, deserializes network weights, initializes Go engine and agent handles, dispatches to the appropriate traversal function based on `config.deep_cfr.traversal_method`, and returns `DeepCFRWorkerResult`.

Worker weight deserialization filters the `__value_net__` key from the advantage weights dict before calling `load_state_dict()` on the advantage network. This prevents the nested dict from poisoning the advantage net's weights when ESCHER distributes both network weight sets together.

### How It Connects

```
deep_worker.py
  +-- encoding.py: _encode_ep_pbs() dispatches to appropriate encoder
  +-- networks.py: AdvantageNetwork, HistoryValueNetwork, get_strategy_from_advantages()
  +-- reservoir.py: ReservoirSample
  +-- ffi/bridge.py: GoEngine, GoAgentState (when engine_backend="go")
  +-- game/engine.py: CambiaGameState (when engine_backend="python")
  +-- agent_state.py: AgentState (Python-engine path)
```

### Design Decisions

- Workers reconstruct network instances on every invocation (stateless). Network state is passed as serialized numpy arrays for pickle compatibility across process boundaries.
- `_snapshot_count` (SD-CFR) is tracked independently of `training_step` to avoid warm-start bias when resuming from checkpoint.
- The updating player alternates each iteration (`iteration % NUM_PLAYERS`), matching the standard Deep CFR protocol.
- `MAX_IS_WEIGHT = 20.0` is a module-level constant, not inline, to make it easy to tune and to keep the three call sites consistent.

## `src/ffi/bridge.py`

Python cffi ABI-mode wrapper around `libcambia.so`, providing `GoEngine` and `GoAgentState` as drop-in replacements for `CambiaGameState` and `AgentState`.

### Library Loading

The library is loaded lazily on first use via a module-level singleton `_LIB`. The path is resolved in this order:

1. The `LIBCAMBIA_PATH` environment variable.
2. Relative to `bridge.py`: `cfr/libcambia.so`.
3. Current working directory.

If the library is not found, `_get_lib()` raises `FileNotFoundError` with a message pointing to `make libcambia`. This means importing any class that calls `_get_lib()` in a worker process will fail fast if the build step was skipped.

Build dependency: `make libcambia` from the repo root. This compiles `engine/cgo/exports.go` via `go build -buildmode=c-shared` and copies the result to `cfr/libcambia.so`.

The full FFI surface is declared via `_ffi.cdef(...)` at module load time. The library is opened with `_ffi.dlopen()` (ABI mode, no compile step required).

### `GoEngine`

```python
class GoEngine:
    INPUT_DIM: int = 222
    NUM_ACTIONS: int = 146
    N_PLAYER_INPUT_DIM: int = 856
    N_PLAYER_NUM_ACTIONS: int = 620

    def __init__(
        self,
        seed: Optional[int] = None,
        house_rules=None,         # CambiaRulesConfig or compatible
        num_players: int = 2,
    ) -> None

    # Core game API
    def legal_actions_mask(self) -> np.ndarray      # (146,) uint8
    def apply_action(self, action_idx: int) -> None
    def is_terminal(self) -> bool
    def get_utility(self) -> np.ndarray              # (2,) float32
    def acting_player(self) -> int
    def turn_number(self) -> int
    def stock_len(self) -> int

    # Snapshot API (replaces apply/undo pattern)
    def save(self) -> int                            # returns snapshot handle
    def restore(self, snap_h: int) -> None
    def free_snapshot(self, snap_h: int) -> None

    # Context helpers
    def decision_ctx(self) -> int                    # 0=StartTurn..5=Terminal
    def get_drawn_card_bucket(self) -> int           # -1 if none

    # Batch update (one FFI call for both agents)
    def update_both(self, a0: "GoAgentState", a1: "GoAgentState") -> None

    # N-player API
    def nplayer_legal_actions_mask(self) -> np.ndarray  # (620,) uint8
    def apply_nplayer_action(self, action_idx: int) -> None
    def get_nplayer_utility(self) -> np.ndarray          # (num_players,) float32

    # Lifecycle
    def close(self) -> None
    def __enter__(self) / def __exit__(self, ...)        # context manager
```

When `house_rules` is provided, `GoEngine` calls `cambia_game_new_with_rules()` which forwards all `CambiaRulesConfig` fields to the Go engine. When `house_rules=None`, it calls `cambia_game_new()` with Go defaults and emits `DeprecationWarning` (Go defaults differ from Python defaults: `AllowDrawFromDiscard=true`, `AllowOpponentSnapping=true`).

The save/restore API replaces the apply/undo pattern. Before recursing into a child node, the worker calls `save()` to capture full game state. After returning, it calls `restore(handle)`. `free_snapshot()` releases the handle back to the pool. The Go-side snapshot pool has a fixed size; running out of slots causes `save()` to return a negative handle and raise `RuntimeError`.

`decision_ctx()` returns the current decision context as an integer, avoiding the cost of constructing a Python `DecisionContext` enum on every call.

`update_both()` calls `cambia_agents_update_both()` in a single FFI round-trip instead of two separate `cambia_agent_update()` calls. Used in the hot traversal loop.

### `GoAgentState`

```python
class GoAgentState:
    def __init__(
        self,
        engine: GoEngine,
        player_id: int,
        memory_level: int = 0,
        time_decay_turns: int = 0,
    ) -> None

    # Factory methods
    @classmethod
    def new_with_memory(
        cls, engine, player_id, memory_level, time_decay_turns,
        memory_archetype: int = 0,   # 0=Perfect, 1=Decaying, 2=HumanLike
        memory_decay_lambda: float = 0.1,
        memory_capacity: int = 3,
    ) -> "GoAgentState"

    @classmethod
    def new_nplayer(
        cls, engine, player_id, num_players=2, memory_level=0, time_decay_turns=0
    ) -> "GoAgentState"

    # Belief update
    def update(self, engine: GoEngine) -> None
    def update_nplayer(self, engine: GoEngine) -> None

    # Memory decay (for Decaying and HumanLike archetypes)
    def apply_decay(self, rng_seed: int = 0) -> None

    # Clone and lifecycle
    def clone(self) -> "GoAgentState"
    def close(self) -> None

    # Encoding (all return numpy arrays)
    def encode(self, decision_context: int, drawn_bucket: int = -1) -> np.ndarray        # (222,) float32
    def encode_eppbs(self, decision_context: int, drawn_bucket: int = -1) -> np.ndarray  # (224,) float32
    def encode_eppbs_interleaved(self, ...) -> np.ndarray                                 # (224,) float32
    def encode_eppbs_dealiased(self, ...) -> np.ndarray                                   # (224,) float32
    def encode_nplayer(self, ...) -> np.ndarray                                           # (856,) float32
    def nplayer_action_mask(self, engine: GoEngine) -> np.ndarray                         # (620,) uint8
```

Agent handles are allocated from a fixed-size pool in Go (512 slots). `clone()` allocates a new handle and copies belief state via `cambia_agent_clone()`. `close()` releases the handle back to the pool. The deep worker calls `close()` on cloned agent states after each traversal branch returns.

`new_with_memory()` creates an agent with a specific memory archetype via `cambia_agent_new_with_memory()`. `apply_decay()` triggers memory decay/eviction via `cambia_agent_apply_decay()`.

`encode_eppbs_interleaved()` is required for slot-aware networks. `encode_eppbs_dealiased()` uses the de-aliased flat layout. Both return `(224,)` float32 arrays matching the Go-side encoding functions.

`encode()` uses pre-allocated `float[222]` cffi buffers to avoid per-call allocation. EP-PBS encode methods allocate fresh buffers (`float[224]`) per call; this is acceptable because they are not on the inner hot loop in the standard 2-player Go-backend path.

### Handle Pool Diagnostics

```python
def cambia_handle_pool_stats(games_out, agents_out, snaps_out):
```

FFI export for inspecting current pool occupancy. Called in the trainer when `enable_traversal_profiling=True` to detect pool exhaustion before it causes failures.

### How It Connects

```
ffi/bridge.py
  +-- libcambia.so:
  |     cambia_game_new, cambia_game_new_with_rules, cambia_game_apply_action
  |     cambia_game_save, cambia_game_restore, cambia_snapshot_free
  |     cambia_game_legal_actions, cambia_game_is_terminal, cambia_game_get_utility
  |     cambia_game_decision_ctx, cambia_agents_update_both
  |     cambia_agent_new, cambia_agent_new_with_memory, cambia_agent_new_nplayer
  |     cambia_agent_clone, cambia_agent_free, cambia_agent_update
  |     cambia_agent_encode, cambia_agent_encode_eppbs
  |     cambia_agent_encode_eppbs_interleaved, cambia_agent_encode_eppbs_dealiased
  |     cambia_agent_encode_nplayer, cambia_agent_apply_decay
  |     cambia_handle_pool_stats
  +-- deep_worker.py: GoEngine and GoAgentState used when engine_backend="go"
  +-- evaluate_agents.py: NeuralAgentWrapper uses GoAgentState for eval encoding
```

## `src/cfr/es_validator.py`

Runs short-depth ES traversals to estimate exploitability and monitor convergence during training.

### Classes

```python
class ESValidator:
    def __init__(
        self,
        config: Config,
        deep_cfr_config: DeepCFRConfig,
        engine_backend: str = "python",
    )

    def compute_exploitability(
        self,
        advantage_weights: Optional[Dict],
        num_traversals: int,
    ) -> dict:
```

`compute_exploitability` runs `num_traversals` ES traversals capped at `es_validation_depth` game steps. It traverses as both player 0 and player 1 and collects regret statistics across all visited decision nodes.

### Returned Metrics

The returned dict contains:

| Key | Description |
|-|-|
| `mean_regret` | Average per-node regret across all traversals |
| `max_regret` | Maximum per-node regret observed |
| `strategy_entropy` | Mean Shannon entropy of the strategy distribution at visited nodes |
| `traversals` | Number of traversals completed |
| `depth` | Mean traversal depth reached |
| `elapsed_seconds` | Wall-clock time for the validation run |
| `total_nodes` | Total decision nodes visited across all traversals |

### Backend Support

The validator accepts the same `engine_backend` config flag as the main training loop. With `engine_backend="go"`, it imports `GoEngine` and `GoAgentState` from `ffi/bridge.py`. The import is deferred (inside `compute_exploitability`) so that the validator can be constructed without the library present. With `engine_backend="python"`, no FFI dependency is needed.

### Integration with DeepCFRTrainer

`DeepCFRTrainer.train()` calls `compute_exploitability` every `es_validation_interval` steps using the current advantage network weights. Results are logged to the metrics stream and displayed in the live training display. Setting `es_validation_interval: 0` disables ES validation entirely.
