# Monte Carlo CFR Sampling Methods

Sampling strategies used for Deep CFR training in Cambia, covering all three MCCFR methods plus SD-CFR.

## The Need for Sampling

Vanilla Counterfactual Regret Minimization (CFR) is guaranteed to converge to a Nash Equilibrium in two-player zero-sum games. However, it requires traversing the *entire* game tree on each iteration. For games with large state spaces or long potential game lengths---like Cambia---full traversal is computationally infeasible, even for a single iteration.

Monte Carlo CFR (MCCFR) methods address this by sampling only a portion of the game tree per iteration. The goal is to obtain unbiased (or low-bias) estimates of counterfactual regrets while reducing per-iteration cost enough that many iterations become tractable. The tradeoff: more iterations may be needed compared to vanilla CFR because sampling introduces variance.

The primary practical motivation here is parallelization. Multiple workers can independently run sampled traversals and send results back to the trainer for aggregation, allowing the training loop to scale with available CPU cores.

## Outcome Sampling (OS-MCCFR)

**Traversal:** At every decision node (traverser, opponent, chance), sample ONE action according to the current strategy profile or chance probabilities. This produces a single path from root to a terminal state per traversal.

**Sampling policy:** traverser nodes sample from `q(a|I) = ε · uniform + (1-ε) · σ(a|I)` with `exploration_epsilon` defaulting to 0.6; every other seat samples from `σ` itself.

**Regret update:** only the sampled path is visited, so the regret estimate needs two importance corrections, not one. The local one divides by `q(a*|I)` at the node being updated. The tail one corrects for the traverser continuing to explore off `σ` below that node: the utility that comes back from the recursion is the value of the ε-mixed continuation, not of `σ`'s. The tail correction is the ratio `π^σ(I·a*, z) / q(I·a* -> z)` accumulated on the way back up (Lanctot et al. 2009). Each node multiplies in `σ(a|h) / q(a|h)` for the action it sampled and returns the running product to its parent; nodes that sample from `σ` contribute exactly 1.

```
tail(z)      = 1
tail(h)      = σ(a|h)/q(a|h) · tail(h·a)          # a = the action sampled at h
w            = tail(I·a*) / q(a*|I)
utility_estimate = u(z) · w
regret(a)    = (1 if a == a* else 0) · utility_estimate - σ(a|I) · utility_estimate
```

**What is unbiased:** at the root of a traversal, `E[regret(a)]` is exactly the counterfactual regret `v^σ(I·a) - v^σ(I)` whenever the weight `w` stays under the clip. `tests/test_os_tail_importance.py` measures this on a three-node tree with an exactly computable counterfactual value: the corrected estimator averages to `(0.084, -0.126)` against an exact `(0.084, -0.126)`, and the same code with the tail ratio dropped averages to `(-0.050, 0.077)`, which not only misses the target but inverts which root action looks better.

**What is not unbiased:**

- **The clip.** `MAX_IS_WEIGHT = 20.0` (module-level constant in `deep_worker.py`) is applied to the full weight `w`, not to the local `1/q(a*|I)` alone. It is a truncated importance sampling estimator: weights above the bound are pulled down to it, so trajectories `σ` strongly prefers but the ε-mixed sampler rarely draws are under-counted, biasing those actions' regret targets low. The bound buys a finite second moment, which the unclipped estimator does not have: `w` is a product of per-node ratios over the traverser's whole remaining decision horizon, and its variance grows without bound in that horizon. On a game as long as Cambia at `ε = 0.6` the clip is load-bearing, so OS regret targets should be read as biased-but-finite-variance rather than unbiased.
- **Deeper infosets.** The estimator is written per node without the root-to-node factors `π^σ_{-i}(I) / q(root -> I)`. Those cancel at the root, where both are 1, but not below it, so a non-root advantage sample is not scaled to its counterfactual weight in the reservoir. All reservoir samples are then fit with equal weight.
- **Depth truncation.** `traversal_depth_limit > 0` returns a utility of 0 for the truncated subtree, which is a fabricated value, not an estimate.
- **Strategy samples.** The opponent-node target is `σ` at that infoset with no `π^σ_i(I) / q(root -> I)` weighting, so the strategy network fits the sampling-frequency-weighted average of `σ`, not the reach-weighted average strategy.

**H3 bug and fix (2026-02-27):** Prior to the fix, `exploration_epsilon` (default 0.6) was applied at opponent nodes as well as traverser nodes, but IS correction was only applied at traverser nodes. This caused the agent to train a best response to a 60%-random opponent rather than to the learned strategy. The fix: `epsilon=0` at opponent nodes (`if player == updating_player`). The affected functions were `_deep_traverse_os_go`, `_deep_traverse_os_go_nplayer`, and the since-retired Python `_deep_traverse_os`. ES-MCCFR and ESCHER were not affected by this bug.

**Tail correction added (cambia-715):** the H3 fix removed exploration from opponent nodes but left the traverser's own exploration uncorrected below the node being updated, so the advantage target still estimated the value of a 60% random continuation of the traverser's own play. The tail ratio above closes that gap, and the clip moved from the local factor to the full weight.

**Pros:**
- Lowest cost per traversal.
- Simple traversal logic.

**Cons:**
- High variance in regret estimates (both players' actions are sampled).
- May require more iterations to converge than ES or ESCHER.

## External Sampling (ES-MCCFR)

**Traversal:** At traverser nodes: enumerate ALL legal actions. At opponent and chance nodes: sample ONE action from the current strategy or chance distribution. This produces a subtree rooted at the traverser's decision nodes.

**Regret update:** Because all traverser actions are explored, regret estimates are closer to vanilla CFR:

```
regret(a) = v(I -> a) - v(I)
```

The counterfactual values `v(I -> a)` are obtained via the sampled opponent/chance subtrees. No IS correction is needed at the traverser's nodes, making ES immune to the H3 bug by construction.

**Feasibility for full Cambia:** ES is computationally infeasible for the full game. Enumerating all traverser actions at every node makes the branching factor too large: at full branching (146 actions), ES is approximately 10x slower than OS per traversal. At 333 traversals/step, a step would take roughly 550s vs 22s for OS. ES is only practical for reduced-action subgames or very short depth limits.

**Prior result:** A prior ES run on the legacy 222-dim encoding achieved approximately 34% mean_imp(3), matching OS-MCCFR on the same encoding. This confirmed the sampling algorithm itself is not the convergence bottleneck.

**Pros:**
- Lower variance than OS (traverser contribution is not sampled).
- Potentially faster convergence per iteration.
- No IS correction needed.

**Cons:**
- High per-traversal cost; infeasible at full Cambia branching factor.
- Traversal logic is more complex than OS.

## ESCHER

ESCHER (McAleer et al., 2023) is designed to get variance closer to ES without the cost of enumerating all traverser actions. It uses a learned value network `V(h)` to estimate counterfactual values for unchosen actions.

**Traversal:**

- At traverser nodes: sample ONE action `a*` (like OS). For `a*`, recurse to get the actual utility `u`. For all other actions `a`: compute counterfactual regret as `V(h·a) - V(h)` using the value network, with no additional traversal.
- At opponent nodes: sample ONE action from the pure learned strategy. No exploration epsilon is applied, so epsilon=0 by construction. This eliminates the need for IS correction at opponent nodes.
- At chance nodes: sample one outcome.

**Regret computation:**

```
r(a*) = u - V(h)           # actual utility minus value net baseline
r(a)  = V(h·a) - V(h)     # counterfactual value net estimate, for a != a*
```

**Value network:** `HistoryValueNetwork` sees both players' full encodings concatenated: input dimension is `2 × input_dim` (448 for EP-PBS encoding). It is trained via weighted MSE on terminal utilities observed during traversal. The value network state is checkpointed alongside the advantage network.

**IS-weight-free:** Because opponent nodes use the pure strategy (no epsilon mixing), there is no IS weight to correct for. This eliminates the primary variance source in OS-MCCFR.

**Known failure mode:** ESCHER's regret quality depends directly on the value network's accuracy. A poorly trained `V(h)` produces wrong counterfactual estimates for unchosen actions, which corrupts regret updates. This is exactly what happened during the first ESCHER run: three worker bugs caused the value network to never be trained, never be distributed to workers, and to use the wrong input dimension. Workers fell back to uniform random for 600 iterations, and the resulting run produced only random-quality play (~20% mean_imp(3)).

**Bugs fixed (2026-02-28):**
1. Value network was never trained: `_train_value_network()` was not called in the training loop.
2. Value network weights were never distributed to workers: the `__value_net__` key was missing from `_get_network_weights_for_workers()`.
3. Value net input dim was hardcoded to `INPUT_DIM * 2 = 444` in the worker, but the trainer used `EP_PBS_INPUT_DIM * 2 = 448`. Fixed to read from `network_config.get("input_dim", INPUT_DIM) * 2`.

After these fixes, ESCHER needs a full retrain before its convergence can be assessed.

**Pros:**
- Low variance without enumerating all traverser actions.
- IS-weight-free.
- Scales similarly to OS in cost per traversal.

**Cons:**
- Requires a separately trained value network; poor value estimates corrupt regrets.
- Additional network adds memory and inference overhead.
- Convergence quality is contingent on value network accuracy.

## SD-CFR (Sample-based Deep CFR)

SD-CFR modifies the training loop rather than the traversal method. It is orthogonal to the choice of OS-MCCFR or ESCHER.

**Key change:** The strategy network is dropped entirely. After each iteration, the current advantage network weights are stored as a snapshot. At evaluation time, the average strategy is approximated from those snapshots via exponential moving average (EMA).

**EMA update rule:**

```
w_T = (T + 1)^1.5
new_sum = old_sum + w_T
θ_EMA = (old_sum / new_sum) * θ_EMA + (w_T / new_sum) * θ_current
```

This weighting scheme gives more influence to later (more-converged) iterations.

**Known limitation:** EMA averages network parameters, not strategies. Because strategy extraction involves a nonlinear operation (ReLU on advantages followed by normalization), averaging parameters is not the same as averaging the resulting strategies. This introduces a Jensen's inequality error estimated at approximately 1-2pp of win rate.

**Prior result:** SD-CFR on the legacy 222-dim encoding achieved approximately 34.6% mean_imp(3), matching OS-dCFR (34.3%). This confirms the algorithm is not the bottleneck.

## Comparison Table

| Aspect | OS-MCCFR | ES-MCCFR | ESCHER |
|-|-|-|-|
| Traverser nodes | Sample 1 (IS-corrected) | Enumerate all (exact) | Sample 1 (value-net corrected) |
| Opponent nodes | Sample 1 (pure strategy) | Sample 1 | Sample 1 (pure strategy) |
| Regret quality | Noisy (importance sampling) | Exact | Low variance (value net dependent) |
| IS weights | Yes (clipped at 20.0) | None | None |
| Cost per traversal | Low | High (infeasible at full branching) | Medium (value net inference) |
| Additional network | None | None | HistoryValueNetwork |
| H3 bug exposure | Yes (fixed 2026-02-27) | Not affected | Not affected |
| Status | Production default | Not recommended for full game | Retrain needed after bug fixes |

## Current Implementation

All three methods are implemented in `cfr/src/cfr/deep_worker.py`. Selection is controlled by the `sampling_method` config field in `DeepCfrConfig`.

**OS-MCCFR:**
- `_deep_traverse_os_go()`: Go engine backend via FFI. Used for production runs.
- `_deep_traverse_os_go_nplayer()`: N-player variant (Go backend).
- Config: `sampling_method: outcome` (default).

**ES-MCCFR:**
- `_deep_traverse_go()`: Go engine backend via FFI.
- Config: `sampling_method: external`.
- Not recommended for full game runs due to cost.

**ESCHER:**
- `_escher_traverse_go()`: Go engine backend only. Requires value_net weights in the worker state (loaded from `__value_net__` key in the weight dict).
- Config: `sampling_method: escher`.
- Requires `network_type: "residual"` or another supported type, plus EP-PBS encoding.

**ES Validation** runs separately from training traversals. `ESValidator` (in `cfr/src/cfr/es_validator.py`) runs short-depth ES traversals at a configurable interval to estimate convergence diagnostics. See `exploitability.md` for details.
