# Deep CFR Convergence in Cambia

Mathematical analysis and empirical results for Deep Counterfactual Regret Minimization applied to Cambia. Covers game tree complexity, convergence theory, all training runs through Feb 2026, and the ceiling analysis.

## Table of Contents

1. [Overview](#1-overview)
2. [Game Tree Complexity](#2-game-tree-complexity)
3. [Theoretical Convergence Bounds](#3-theoretical-convergence-bounds)
4. [Empirical Results](#4-empirical-results)
5. [Ceiling Analysis](#5-ceiling-analysis)
6. [Identified Issues](#6-identified-issues)
7. [ESCHER Status](#7-escher-status)
8. [Argmax Evaluation](#8-argmax-evaluation)
9. [Open Questions](#9-open-questions)

## 1. Overview

Cambia's game tree is too large for tabular CFR methods: the full information-set space is estimated at ~2.53 x 10^13, placing it at poker scale. All training therefore uses Deep CFR, which replaces tabular regret accumulators with neural networks. This allows the algorithm to generalize across information sets it has never visited directly, but it also removes the formal convergence guarantee that tabular CFR enjoys.

The core problem identified as of Feb 2026 is that all functional training runs plateau at approximately 34-35% mean_imp(3) against imperfect-information baselines. The random_no_cambia baseline, which calls no Cambia and plays uniformly at random among non-Cambia actions, achieves approximately 50% win rate with zero learned strategy. A Nash equilibrium in this symmetric two-player zero-sum game guarantees at least 50% win rate against any opponent. The trained agent at 35% plays 15 percentage points below the random-play floor, which means the training pipeline has not converged to Nash. The ceiling is not game-theoretic: it is a convergence failure.

## 2. Game Tree Complexity

### 2.1 Deck Combinatorics

Cambia uses a 54-card deck (52 standard + 2 Jokers). Each player receives 4 cards. Initial deal combinations:

- Player 1 hand: C(54, 4) = 316,251
- Player 2 hand given P1: C(50, 4) = 230,300
- Total joint initial deals: ~7.28 x 10^10

Hand score range: minimum -2 (two Red Kings + two Jokers), maximum +50 (two Black Kings + two Queens).

### 2.2 Information Set Structure

An information set in Cambia encodes:
- Known own cards (observed via initial peek, peek abilities, or king-look)
- Belief state over unknown hand positions (9-bucket card abstraction)
- Opponent action history (draws, discards, snaps, ability usage)
- Game phase (one of 6 DecisionContexts: START_TURN, POST_DRAW, SNAP_DECISION, ABILITY_SELECT, SNAP_MOVE, TERMINAL)
- Discard pile top
- Memory decay state

Full upper bound: |I| <= 2.53 x 10^13.

With 9-bucket abstraction applied to all 4 hand positions:

    |I_abstract| ~ C(12, 4) * 230,300 * 6 * 6
                 = 495 * 230,300 * 36
                 ~ 4.1 * 10^9

The neural network's 224-dimensional EP-PBS input encoding compresses this further. The effective complexity depends on the function approximator's capacity.

### 2.3 Branching Factor and Depth

Action counts by decision context (production rules: `allowDrawFromDiscard=false`, `allowOpponentSnapping=false`):

| Context | Actions | Frequency weight |
|-|-|-|
| START_TURN | 2 (DrawStockpile, CallCambia) | ~0.35 |
| POST_DRAW | 5-6 (4 Replace + DiscardNoAbility + DiscardWithAbility) | ~0.30 |
| ABILITY_SELECT | 4-16 (Peek: 4, BlindSwap: 16, KingLook: 16 + 2 KingDecision) | ~0.15 |
| SNAP_DECISION | 2-3 (Pass, Snap candidates) | ~0.15 |
| SNAP_MOVE | 1-4 (target selection) | ~0.05 |

Weighted average branching factor: b = 5.0 (measured across 10,000 random games). Maximum action count at any node: 146 (the full action space).

Game depth:

| Metric | Decision nodes |
|-|-|
| Minimum game | ~6 (2 turns) |
| Average vs random opponents | ~23 (10 turns) |
| Average vs strategic opponents | ~57 (25 turns) |
| Maximum (config cap: 46 turns) | ~106 |

### 2.4 Comparison to Benchmark Games

| Game | \|I\| | \|A\| | Branching | Solved? |
|-|-|-|-|-|
| Kuhn Poker | 12 | 2 | 2 | Yes, tabular CFR |
| Leduc Hold'em | 936 | 3 | 3 | Yes, tabular CFR |
| Cambia (abstract) | ~4.1 x 10^9 | 146 | 5.0 | No (deep CFR, ongoing) |
| Cambia (full) | ~2.53 x 10^13 | 146 | 5.0 | No |
| Heads-up Limit Texas Hold'em | 3.16 x 10^14 | 3 | 2.5 | Yes, ~900 CPU-years |

Cambia's higher branching factor (5.0 vs 2-3 for poker) combined with deep strategic play (~57 nodes) makes outcome sampling particularly expensive. Tabular methods are computationally infeasible: plugging Cambia's numbers into the standard exploitability bound gives T* ~ 10^33 iterations.

## 3. Theoretical Convergence Bounds

### 3.1 Tabular CFR

The standard tabular CFR guarantee (Zinkevich et al. 2007): in a two-player zero-sum game, if both players use regret matching over T iterations, the average strategy profile forms an epsilon-Nash equilibrium where:

    epsilon(T) <= sum_i Delta_i * |I_i| * sqrt(|A_i|) / sqrt(T)

This gives an O(1/sqrt(T)) convergence rate in exploitability. The bound is vacuous for Cambia at any practical T.

### 3.2 Monte Carlo CFR (Outcome Sampling)

OS-MCCFR (Lanctot et al. 2009) samples a single trajectory per traversal, making per-iteration cost independent of tree size. The convergence guarantee becomes:

    epsilon = O( Delta * |I| * sqrt(|A|) / (delta * sqrt(T)) )

where delta is the minimum probability of sampling any terminal history. With depth D=57, max actions 146, and epsilon=0.6 exploration:

    delta >= (0.6 / 146)^57 ~ 10^{-136}

This worst-case bound is vacuous. In practice, convergence occurs because average-case trajectory probabilities are far higher than the worst case, and the neural function approximator generalizes across unvisited states.

### 3.3 Deep CFR: No Formal Guarantee

Deep CFR (Brown et al. 2019) replaces tabular accumulators with neural networks. The convergence bound becomes:

    Exploitability(T) <= 2 * sum_i Delta_i * |I_i| * sqrt(|A_i|) / sqrt(T) + 2 * T * epsilon_approx

The second term, 2 * T * epsilon_approx, grows with T. Convergence requires epsilon_approx to decrease faster than 1/T as training progresses, which holds only if the network has sufficient capacity and the training procedure converges to low approximation error at each iteration. There is no formal guarantee that this occurs: if the network fails to approximate the advantage function accurately, the exploitability bound can diverge. This is the theoretical explanation for why Deep CFR can plateau.

Key properties:
- Reservoir sampling is required. A sliding window buffer does not preserve convergence.
- Linear weighting (t^1.5 loss) mimics CFR+ and accelerates convergence in practice.
- Approximation error epsilon_approx does not vanish with iterations alone; it requires both network capacity and successful optimization at each step.

## 4. Empirical Results

### 4.1 Summary of All Runs

All percentages are win rate. mi(3) = mean_imp(3): mean WR across imperfect_greedy, memory_heuristic, aggressive_snap. mi(5) = mean_imp(5): mean WR across 5 baselines including random_no_cambia and random_late_cambia. Dashes indicate not measured.

| Run | Encoding | Network | Algorithm | Iters | mi(3) | mi(5) | Notes |
|-|-|-|-|-|-|-|-|
| prod-full-333 | Legacy 222 | MLP | OS-dCFR | 1075 | 34.3% | -- | Baseline run |
| sd-cfr-500k | Legacy 222 | MLP | SD-CFR | 500 | 34.6% | -- | Algorithm not bottleneck |
| eppbs-2p | EP-PBS flat | MLP | OS-dCFR | 1500 | 33.8% | 41.1% | MLP layout mismatch |
| ablation-interleaved-resnet | EP-PBS interleaved | ResNet | OS-dCFR | 200 | 34.4% | 41.4% | Climbing at endpoint |
| interleaved-resnet-adaptive | EP-PBS interleaved | ResNet | SD-CFR | 600 | 35.5% peak | 42.7% peak | Sustained, no decay |
| score-margin-224dim (archived) | EP-PBS + 224-dim | ResNet | SD-CFR | 914 | 31-33% | 38-39% | Score-margin NEGATIVE |
| ablation-dealiased-flat | EP-PBS dealiased | ResNet | SD-CFR | 200 | 33.8% | 40.9% | ~1pp below interleaved |
| ablation-slotfilm | EP-PBS interleaved | SlotFiLM | SD-CFR | 200 | 24.9% | -- | Collapsed |
| escher-full-333 (archived) | EP-PBS interleaved | ResNet | ESCHER | 600 | 20.0% | -- | Never functional (worker bugs) |

### 4.2 Phase 2 Detail (Best Configuration)

Interleaved EP-PBS encoding, ResNet (~507K params), SD-CFR mode, adaptive train_steps (floor=250, cap=1000), binary {-1, 0, +1} utility:

| Iter | mi(3) | mi(5) | greedy | T1 Cambia | Avg turns |
|-|-|-|-|-|-|
| 100 | 32.2% | 40.0% | 21.9% | 22.9% | 6.0 |
| 200 | 34.6% | 41.8% | 24.1% | 26.9% | 5.7 |
| 300 | 35.0% | 41.8% | 24.0% | 30.2% | 5.5 |
| 400 | 35.3% | 41.8% | 24.1% | 30.1% | 5.6 |
| 450 | 35.5% | 42.7% | 24.2% | 29.6% | 5.6 |
| 500 | 34.0% | 41.4% | 23.7% | 27.2% | 5.8 |
| 600 | 35.2% | 42.3% | 23.7% | 26.8% | 5.8 |

Config: `cfr/runs/interleaved-resnet-adaptive/config.yaml`.

### 4.3 Observations

All functional OS-dCFR and SD-CFR runs converge to the same 34-35% mi(3) plateau regardless of encoding scheme or algorithm variant. SD-CFR matches OS-dCFR on the legacy encoding (34.6% vs 34.3%), which confirms the algorithm is not the bottleneck. EP-PBS interleaved with ResNet and adaptive train_steps is the best configuration, sustaining 34.5-35.5% mi(3) over 600 iterations without decay.

Score-margin utility hurts by 2-3pp relative to binary {-1, 0, +1}: Phase 3 was flat at 31-33% mi(3) for 900 iterations and was killed early. Binary utility is correct. The additional 224-dim encoding features (observation ages, dead-card histogram, turn progress) also provided no benefit.

SlotFiLM collapsed to 24.9% mi(3). With approximately 1900 samples per iteration, the architecture is too expressive and overfits rather than generalizing. The interleaved slot structure requires more data than Deep CFR can produce at this traversal rate.

## 5. Ceiling Analysis

### 5.1 The Core Finding

35% mi(3) is not a game-theoretic limit. The argument:

1. Cambia is a symmetric two-player zero-sum game.
2. The minimax theorem guarantees that a Nash equilibrium strategy achieves E[utility] >= 0, equivalently WR >= 50%, against any opponent including the worst case.
3. The random_no_cambia baseline (uniform over non-Cambia actions, never calls Cambia) achieves approximately 50% WR with zero learned strategy.
4. The trained agent at 35% plays 15 percentage points below the zero-skill floor.

The regret formulas have been verified correct by a parallel code audit (Feb 2026): OS-MCCFR, ES-MCCFR, ESCHER, and N-player regret update formulas all match the published algorithms. The ceiling is not in the math. It is a convergence failure in the function approximation pipeline.

### 5.2 T1 Cambia Decomposition

The T1 Cambia pathology is present in every run: 22-35% of games end on turn 1 due to the agent calling Cambia on the first move. T1 Cambia is only profitable when the bottom-2 card sum is <= 9 (approximately 30.6% of deals). The conditional WR analysis:

- Bottom-2 sum <= 5: WR 68-87%
- Bottom-2 sum = 9 (breakeven): WR 51.6%
- Bottom-2 sum >= 10: WR < 50% (harmful)
- Unconditional T1 Cambia WR: 39.4%

With approximately 30% T1 Cambia at 39.4% WR, the non-Cambia WR can be derived:

    non_cambia_WR = (35% - 0.30 * 39.4%) / 0.70 = 33.1%

Both components are sub-random. The agent calls Cambia regardless of hand quality, and its non-Cambia play is 17pp below random. A luck-skill decomposition confirms the situation is impossible under positive skill: even assuming 80% of outcomes are luck-determined, achieving 35% WR would require negative skill (WR below 50% in skill-determined outcomes). The agent is actively hurting itself.

### 5.3 Baseline Characterization

The three mi(3) baselines are not strong:
- imperfect_greedy: incomplete information heuristic, knows only seen cards
- memory_heuristic: tracks card history with limited memory
- aggressive_snap: snaps on any opportunity, greedy draw policy

The greedy oracle (perfect information, always optimal choice) achieves approximately 75% WR. There is substantial room above 50%. These baselines should be beatable by any strategy better than random, which the agent is not.

## 6. Identified Issues

The following issues have been identified through code audit and experimental analysis. Combined impact estimates are 5-8pp of the 15pp gap; the remaining 7-10pp is attributed to general function approximation generalization failure.

### 6.1 EMA Nonlinearity (~1-2pp)

SD-CFR uses exponential moving average over network parameters (weights). At evaluation time, the EMA-averaged parameters are passed through the forward pass, then through ReLU + normalize to extract a strategy. This is nonlinear: RM(EMA(theta_1, theta_2)) is not equal to EMA(RM(theta_1), RM(theta_2)).

Example: if network_1 outputs [5, -3] and network_2 outputs [-2, 4], their correct average strategy is [0.5, 0.5]. The EMA of parameters might produce [1.5, 0.5], yielding strategy [0.75, 0.25], which is wrong.

The snapshot-averaging path in SDCFRAgentWrapper (without EMA) does it correctly by averaging strategies post-RM. The EMA fast path, which is the default and was used in all Phase 2 runs, violates Jensen's inequality. The fix is strategy-space EMA: average regret-matched strategies across K recent snapshots rather than averaging parameters.

### 6.2 Early Overtraining (~2-3pp)

At iteration 1, the advantage buffer contains approximately 1832 samples. The training floor is 250 SGD steps with batch size 4096. Since batch_size > buffer_size, each step sees the entire buffer. The result: each sample is seen approximately 559 times at iteration 1. The network memorizes random-play traversal patterns.

With EMA, these overfitted early weights contaminate the running average for the rest of training. The overtraining ratio at subsequent iterations:

| Iter | Buffer size | Steps | Repeat ratio |
|-|-|-|-|
| 1 | ~1,832 | 250 | ~559x |
| 10 | ~18,000 | 250 | ~54x |
| 100 | ~183,000 | 250 | ~5.6x |
| 600 | ~1,100,000 | ~1000 | ~3.7x |

The fix: lower the training floor from 250 to something like `max(10, min(250, ...))`, or skip training for the first N iterations until the buffer has enough data.

### 6.3 Gradient Starvation (impact unknown)

The advantage loss divides per-sample MSE by the number of legal actions (`/ num_legal`). This normalizes each sample to a per-action scale, but the effect is to give turn-1 decisions (|A| = 3) approximately 12x larger gradient than mid-game blind-swap decisions (|A| = 36). The training structurally over-optimizes early-game accuracy at the expense of mid-game play.

This finding was raised in the Feb 2026 review cycle (reviewer notation C1). The fix is to remove the `/num_legal` division at `deep_trainer.py:1064`. Expected impact: 2-3pp, though this has not been isolated experimentally.

### 6.4 IS Clipping (<1pp)

`MAX_IS_WEIGHT = 20.0` clips importance sampling weights, muting regret signals from deep-tree trajectories that were sampled with low probability. Worst-case distortion at sigma=0 is about 3x. Typical-case distortion under epsilon=0.6 exploration is less than 1.7x. Not a primary ceiling cause, but contributes to variance in deep-tree regret estimates.

### 6.5 T1 Cambia Pathology (persistent, all runs)

T1 Cambia rates of 22-35% across every run indicate the agent has not learned to condition Cambia calls on hand quality. No encoding change, utility formulation, or algorithm variant has fixed this. The pathology appears to be a stable equilibrium under the current training dynamics: T1 Cambia yields 39.4% WR on average, which is better than the agent's non-Cambia play (33.1%), so the agent is locally incentivized to maintain it.

Breaking this pathology likely requires either fixing the other training issues (so non-Cambia play improves above 50%) or adding hand-quality conditioning to the encoding in a way that makes the Cambia/no-Cambia action value distinction sharper.

## 7. ESCHER Status

ESCHER was never functional. The original `escher-full-333` run (archived) (600 iterations) ran all traversals with uniform random strategy due to two worker bugs:

**Bug 4:** The value net weights were nested inside the advantage weights dict under key `__value_net__`. When workers called `load_state_dict` on the advantage network, this extra key caused silent failure (PyTorch `load_state_dict` with `strict=True` rejects unexpected keys). On failure, the worker fell back to uniform random. This happened for every worker rotation: 150 failures per step x 600 steps = 90,000 consecutive failures.

**Bug 5:** The value network input dimension was hardcoded to `INPUT_DIM * 2 = 444` in worker initialization. The trainer used EP-PBS encoding with `EP_PBS_INPUT_DIM * 2 = 448`. The dimension mismatch caused network construction to fail.

Both bugs are fixed. The `escher-full-333` run's 20.0% mi(3) is not a valid ESCHER data point. It reflects residual OS-MCCFR signal from a warm-start checkpoint combined with inflated pre-bugfix eval numbers. ESCHER retrain is pending after Phase 4 completes (single GPU constraint).

Three earlier bugs were also fixed before the `escher-full-333` run but did not save it:
1. Value net was never trained: `_train_value_network()` was not called.
2. Value net weights were never distributed to workers: `__value_net__` key absent from `_get_network_weights_for_workers()`.
3. Value net input_dim hardcoded instead of dynamic.

A reviewer who validated all three bugs (Feb 2026) derived ESCHER hyperparameters for the retrain: value_hidden=512, value_lr=1e-3, value_buffer=1M, traversals_per_step=150, value_target_buffer_passes=2.0. Reviewer projection: 55-65% mi(3) post-fix.

Config for ESCHER retrain: `cfr/runs/escher-interleaved/config.yaml`.

## 8. Argmax Evaluation

Phase 2 iteration 450 was evaluated under two action-selection policies:
- Sampled: draw from the strategy distribution proportional to regret-matched advantages
- Argmax: select the action with highest advantage (deterministic greedy)

Results:

| Policy | mi(3) | T1 Cambia rate | Avg turns |
|-|-|-|-|
| Sampled | 35.5% | 29.6% | 5.6 |
| Argmax | 33.2% | ~0% | 11.5 |

Argmax is 2.3pp worse. The mixed strategy is load-bearing.

Under argmax, T1 Cambia disappears (the deterministic policy does not find Cambia advantageous in expectation) and games last twice as long, but WR drops. The agent exploits weak baselines via stochastic early-Cambia mixing: in approximately 30% of deals where T1 Cambia is optimal (bottom-2 sum <= 9), the stochastic policy captures that gain. Argmax loses this because it integrates over all deal states and Cambia has negative expected value overall.

This is expected behavior for a mixed-strategy solution: the Nash equilibrium is a mixed strategy, and evaluating the argmax of advantages is not the same as evaluating the average strategy. The correct evaluation procedure is sampled action selection.

## 9. Open Questions

**Why does non-Cambia play achieve only 33.1% WR?** Even if the T1 Cambia pathology is excluded, the agent plays sub-randomly in the majority of game states. The mid-game is where most WR is determined, and the agent is not learning to play it well. Whether this is function approximation failure, gradient starvation, or training data quality is not yet known.

**Would fixing the identified issues close the gap?** The quantified issues (EMA nonlinearity ~1-2pp, early overtraining ~2-3pp, gradient starvation ~2-3pp) account for up to 7-8pp. The random-play floor is 15pp above the current ceiling. Fixing all identified issues may not be sufficient.

**Is PPO a useful diagnostic?** A PPO best-response agent (using the Go engine as a Gymnasium environment, approximately 200 LOC with stable-baselines3) trained against the current baselines would establish whether the game is beatable above 50% at all. If PPO exceeds 45% mi(3), the ceiling is CFR-specific. This would be the definitive test.

**Does tabular CFR+ converge on a simplified variant?** Mini-Cambia (12-card deck, 2 cards per hand, no abilities, no snap) has an infoset count of approximately 10K-100K, which is feasible for tabular CFR+. A converged Nash equilibrium on mini-Cambia would provide ground truth for whether the encoding and training infrastructure are correctly set up.

**Phase 4 result (pending):** The OS-MCCFR Phase 4 run applies the H3 fix (epsilon=0 at opponent nodes, fixing incorrect IS correction) and sets `cambia_allowed_round=3` to disable T1 Cambia. If Phase 4 exceeds 38% mi(3) sustained, the H3 IS correction bug was a significant factor. If it matches Phase 2, the remaining gap is function approximation generalization.
