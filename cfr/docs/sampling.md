# Monte Carlo CFR Sampling Methods

Sampling strategies considered for accelerating CFR training for the 2-player variant of Cambia, with a focus on parallel execution.

## The Need for Sampling

Vanilla Counterfactual Regret Minimization (CFR) is guaranteed to converge to a Nash Equilibrium in two-player zero-sum games. However, it requires traversing the *entire* game tree on each iteration. For games with large state spaces or long potential game lengths, like Cambia, this full traversal becomes computationally infeasible or prohibitively slow, even for a single iteration.

Monte Carlo CFR (MCCFR) methods address this by sampling only a portion of the game tree on each iteration. The goal is to obtain unbiased estimates of the regrets while significantly reducing the computational cost per iteration. This allows for faster training in terms of wall-clock time, even if more iterations might be needed compared to vanilla CFR due to the variance introduced by sampling.

The primary goal for using MCCFR here is to take advantage of multi-core processors via parallelization. Multiple workers can independently run sampled game simulations, and their results can be aggregated to update the agent's strategy.

## Sampling Methods Considered

Two primary MCCFR sampling methods were considered:

### Outcome Sampling (OS)

* **Traversal Method:** Samples ONE action at EVERY decision node (acting player, opponent, chance) according to the current strategy profile ($\sigma^t$) or chance probabilities. This results in traversing a single path from the root to a terminal state in each simulation.
* **Regret Update:** Since only one path is explored, regret updates require careful handling to remain unbiased. Importance sampling is typically used. The update at an information set `I` visited along the sampled path `z` depends on the utility `u_i(z)` obtained at the end of the path, the probability of sampling that path `q(z)`, the opponent's reach probability `π_{-i}(I)`, and the current strategy `σ(a|I)`.
  * **Our Planned Implementation:** Uses a specific OS-derived formula where the regret for action `a` at infoset `I` (if chosen action was `a*` with probability `p(a*)`) is estimated based on the returned utility `v(I->a*)` from the recursive call:
    * `utility_estimate = v(I->a*) / p(a*)`
    * `regret(a) ≈ (1 if a=a* else 0) * utility_estimate - σ(a|I) * utility_estimate`
    * The final update is weighted by `π_{-i}(I)` and the iteration weight (for CFR+).
* **Pros:**
  * Lowest computational cost per sampled path (minimal traversal).
  * Relatively simple traversal logic to implement.
* **Cons:**
  * Generally introduces higher variance into the regret estimates compared to ES, as both players' actions are sampled.
  * May require more iterations to converge due to higher variance.

### External Sampling (ES)

* **Traversal Method:** Samples actions only for the **opponent** and **chance** nodes. At nodes where the **acting player** (`i`) makes a decision, *all* legal actions are explored. Recursion proceeds down each of these branches, but opponent/chance moves within those branches are sampled.
* **Regret Update:** Since all of the acting player's actions are explored locally, the regret update is closer to Vanilla CFR (`regret(a) = v(I->a) - v(I)`). However, the counterfactual values `v(I->a)` and `v(I)` are *estimates* obtained via the sampled opponent/chance traversals. Updates are weighted by `π_{-i}(I)`.
* **Pros:**
  * Lower variance in regret estimates than OS because the acting player's contribution is not sampled.
  * Potentially faster convergence in terms of the number of iterations required.
  * Proven to require only a constant factor more iterations than Vanilla CFR while reducing per-iteration cost.
* **Cons:**
  * Higher computational cost per iteration compared to OS, as it explores all branches at the acting player's nodes.
  * Traversal logic is slightly more complex than OS.

| Feature                 | Outcome Sampling (OS)                             | External Sampling (ES)                                    | Vanilla CFR (Full Tree) |
| :---------------------- | :------------------------------------------------ | :-------------------------------------------------------- | :---------------------- |
| **Player Action**       | Sample 1                                          | Explore All                                               | Explore All             |
| **Opponent Action**     | Sample 1                                          | Sample 1                                                  | Explore All             |
| **Chance Action**       | Sample 1                                          | Sample 1                                                  | Explore All             |
| **Cost / Iteration**    | Low                                               | Medium                                                    | High                    |
| **Variance**            | High                                              | Low                                                       | Zero (Deterministic)    |
| **Convergence (Iters)** | Potentially Slower                                | Potentially Faster                                        | Baseline                |
| **Convergence (Time)**  | Potentially Fast (if low cost/iter >> more iters) | Potentially Fast (good balance of cost/iter and variance) | Slow                    |

## Current Implementation

Both methods are now implemented. OS is still available and remains the default `sampling_method` in config, but ES is what the Deep CFR training loop actually uses.

**External Sampling** is the active traversal method for Deep CFR training. It lives in `deep_worker.py` as `_deep_traverse_es` (Python backend) and `_deep_traverse_go` (Go engine backend via FFI). The reason for switching from OS to ES for Deep CFR is that advantage samples require exact counterfactual values at the traverser's nodes: you need to enumerate all actions to compute `v(I->a) - v(I)` without importance sampling noise. OS can't give you that without high variance corrections.

**Outcome Sampling** is still in the codebase (`_deep_traverse_os`) and is selected when `sampling_method: outcome` is set in config. It's useful for quick exploratory runs or situations where you want the lower per-traversal cost and can tolerate higher variance.

**ES Validation** runs separately from the main training traversals. `ESValidator` (in `src/cfr/es_validator.py`) runs short-depth ES traversals at a configurable interval to estimate convergence. It reports: `mean_regret`, `max_regret`, `strategy_entropy`, `traversals`, `depth`, `elapsed_seconds`, and `total_nodes`. Both the Python and Go backends are supported there too. The depth and traversal count are controlled separately from the main training config via `es_validation_depth` and `es_validation_traversals`.

The original reasons I chose OS over ES (lower per-traversal cost, simpler logic) still hold for tabular CFR on this game, but for Deep CFR the sample quality difference matters more than the per-traversal cost difference.
