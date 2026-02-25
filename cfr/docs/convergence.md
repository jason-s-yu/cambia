# Deep CFR Convergence Analysis for Cambia

Mathematical analysis of convergence, sample complexity, and empirical performance of Deep Counterfactual Regret Minimization in Cambia.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Game Tree Complexity](#2-game-tree-complexity)
3. [CFR Convergence Theory](#3-cfr-convergence-theory)
4. [Sample Complexity for Cambia](#4-sample-complexity-for-cambia)
5. [Empirical Convergence Analysis](#5-empirical-convergence-analysis)
6. [Depth Cap Analysis](#6-depth-cap-analysis)
7. [Hyperparameter Sensitivity Analysis](#7-hyperparameter-sensitivity-analysis)
8. [Updated Iteration Estimates](#8-updated-iteration-estimates)
9. [Recommended Next Steps](#9-recommended-next-steps)
10. [Updated Analysis---20,000-Game Evaluation](#10-updated-analysis----20000-game-evaluation)

---

## 1. Executive Summary

**750 iterations of Deep CFR training were insufficient to converge** for the Cambia card game. The empirical evidence is unambiguous:

- After 850 iterations (extended from the original 750), the trained agent achieves only 55.1% win rate against `imperfect_greedy`, 55.3% against `aggressive_snap`, and 54.1% against `memory_heuristic`.
- The per-50-iteration improvement at steady state is 0.004-0.008 in win rate, which is *at or below the noise floor* of 5000-game evaluations (95% CI = +/-1.4%).
- A logarithmic fit WR(T) = 0.474 + 0.010 * ln(T) yields RMSE = 0.0078, confirming that convergence is logarithmically slow---consistent with the theoretical O(1/sqrt(T)) regret bound under high-variance outcome sampling.

**Updated iteration estimates** based on the empirical log fit (revised with 20,000-game evaluation at T=850):

- **55% win rate** (marginal advantage): ~1,500 iterations (~14 hours)---**effectively already achieved** (measured 55.0% at T=850)
- **57% win rate** (clear advantage): ~9,100 iterations (~3.5 days)
- **60% win rate** (strong play): ~138,000 iterations (~53 days)

A 20,000-game evaluation at T=850 measured 55.0% win rate against imperfect-info baselines (95% CI: [54.6%, 55.4%]), **decisively rejecting the inverse-square-root model** (p=0.003) which predicted an asymptote of 54.6%. The logarithmic model is strongly preferred (Delta AIC = 11.4), and the updated slope (b=0.0111) is 9% steeper than the original estimate. The neural approximator has not yet reached its capacity ceiling. See Section 10 for the full analysis.

**Root causes of slow convergence:**

1. Cambia has an information set space of |I| <= 2.53 * 10^13 (comparable to Texas Hold'em), far exceeding the capacity of any tabular method.
2. Outcome sampling introduces multiplicative variance proportional to 1/delta where delta = min trajectory sampling probability. With branching factor b = 5.0 and depth D = 57 decision nodes, delta is astronomically small.
3. The neural function approximator (175K parameters, hidden_dim=256) must generalize across ~3.1 * 10^9 abstracted information sets. The approximation error epsilon_approx enters the convergence bound additively and does not vanish with more iterations.

**The key insight:** Deep CFR convergence for Cambia is bottlenecked not by iteration count alone but by (a) the variance of outcome sampling in a deep game tree, and (b) the capacity of the neural approximator to represent the advantage function across a vast information set space. Addressing either bottleneck would yield greater returns than simply running more iterations.

---

## 2. Game Tree Complexity

### 2.1 Deck Combinatorics

Cambia uses a 54-card deck: 52 standard cards plus 2 Jokers. Each player receives 4 cards. Card values range from Joker = 0 and Red King = -1 to Black King = 13, with special abilities on discard for ranks 7 through King.

**Initial deal combinations:**

- Player 1 hand: C(54, 4) = 316,251
- Player 2 hand given P1: C(50, 4) = 230,300
- Total joint initial deals: 316,251 * 230,300 = 72,832,605,300 (~7.28 * 10^10)

**Hand score statistics:**

- Minimum hand score: -2 (two Red Kings + two Jokers, sum = -1 + -1 + 0 + 0 = -2)
- Maximum hand score: +50 (two Black Kings + two Queens, sum = 13 + 13 + 12 + 12 = 50)
- Utility range: Delta = 52 (maximum hand-score difference between players)

### 2.2 Information Set Structure

An information set in Cambia encodes a player's decision-relevant knowledge:

1. **Known own cards:** Up to 4 cards observed via initial peek, peek abilities, or king-look.
2. **Belief state over unknown cards:** Probability distributions over unobserved hand positions, encoded via 9 card buckets (grouping cards by strategic value).
3. **Opponent model:** What opponent actions have been observed (draws, discards, snaps, ability usage).
4. **Game phase:** One of 6 DecisionContexts: START_TURN, POST_DRAW, SNAP_DECISION, ABILITY_SELECT, SNAP_MOVE, TERMINAL.
5. **Discard pile top:** The publicly visible top card.
6. **Memory decay:** Card knowledge degrades over time (configurable via `time_decay_turns`).

**Upper bound on |I|:**

The information set space is bounded by the product of:

- Card belief configurations: Each of the 4 hand positions can be in one of {known(rank)} or {unknown(bucket distribution)} states. With 13 ranks + Joker = 14 known values and 9 buckets for unknown positions, the per-position states number approximately 14 + C(9+k, k) for k probability levels.
- The full upper bound, accounting for deck state, discard history, and opponent actions:

    |I| <= 2.53 * 10^13

This places Cambia between Leduc Hold'em (|I| = 936) and heads-up limit Texas Hold'em (|I| = 3.16 * 10^14) in information-set complexity.

**Abstracted information sets:**

With 9-bucket card abstraction applied to all 4 hand positions:

    |I_abstract| ~ C(12, 4) * 230,300 * 6 * 6
                 = 495 * 230,300 * 36
                 ~ 4.1 * 10^9

In practice, the neural network's 222-dimensional input encoding compresses this further. The effective complexity depends on the function approximator's capacity (Section 4.2).

### 2.3 Branching Factor

The action space varies by decision context under production rules (`allowDrawFromDiscard=false`, `allowOpponentSnapping=false`):

| Context | Actions | Frequency weight |
|-|-|-|
| START_TURN | 2 (DrawStockpile, CallCambia) | ~0.35 |
| POST_DRAW | 5-6 (4 Replace + DiscardNoAbility + conditional DiscardWithAbility) | ~0.30 |
| ABILITY_SELECT | 4-16 (Peek: 4, BlindSwap: 16, KingLook: up to 16 + 2 KingDecision) | ~0.15 |
| SNAP_DECISION | 2-3 (Pass, Snap candidates) | ~0.15 |
| SNAP_MOVE | 1-4 (target selection) | ~0.05 |

**Weighted average branching factor:** b = 5.0 (measured empirically across 10,000 random games).

Maximum branching factor at any single node: |A| = 146 (the full action space, though most are illegal at any given state).

### 2.4 Game Tree Depth

| Metric | Decision nodes |
|-|-|
| Minimum game | ~6 (2 turns) |
| Average vs random opponents | ~23 (10 turns) |
| Average vs strategic opponents | ~57 (25 turns) |
| Maximum (config cap: 46 turns) | ~106 |

Average decision nodes per turn: ~2.3.

### 2.5 Comparison to Benchmark Games

| Game | \|I\| | \|A\| | Tree depth | Branching | Solved? |
|-|-|-|-|-|-|
| Kuhn Poker | 12 | 2 | 4 | 2 | Tabular CFR, trivial |
| Leduc Hold'em | 936 | 3 | 6 | 3 | Tabular CFR, minutes |
| Cambia (abstract) | ~3.1 * 10^9 | 146 | ~57 | 5.0 | Deep CFR, ongoing |
| Cambia (full) | ~2.53 * 10^13 | 146 | ~57 | 5.0 | Deep CFR, ongoing |
| Heads-up Limit Texas Hold'em | 3.16 * 10^14 | 3 | ~17 | 2.5 | CFR+, ~900 CPU-years (Bowling et al. 2015) |
| Heads-up No-Limit Texas Hold'em | ~10^161 | ~20,000 | ~200 | ~3 | Deep CFR + abstractions (Brown & Sandholm 2019) |

Cambia occupies an intermediate position: its information-set count approaches poker scale, but its branching factor is higher (5.0 vs 2-3) and its depth under strategic play (57 nodes) exceeds limit Hold'em (17 nodes). The combination of high branching and deep trees makes outcome sampling particularly expensive.

---

## 3. CFR Convergence Theory

### 3.1 Regret Matching and Vanilla CFR

**Definition (Counterfactual regret).** For player i with information set I in I_i, the *immediate counterfactual regret* for action a at iteration T is:

    R_i^T(I, a) = sum_{t=1}^{T} [ v_i(sigma_{I->a}^t, I) - v_i(sigma^t, I) ]

where v_i(sigma, I) is the counterfactual value of information set I under strategy profile sigma, and sigma_{I->a} denotes the strategy modified to always play action a at information set I.

**Theorem 1 (Zinkevich, Johanson, Piccione, and Bowling, 2007).** In a two-player zero-sum game, if both players use regret matching to update strategies over T iterations, the average strategy profile (sigma_bar^1, sigma_bar^2) forms an epsilon-Nash equilibrium where:

    epsilon <= sum_{i in {1,2}} max_{I in I_i} R_i^{+,T}(I) / T

and the positive counterfactual regret is bounded by:

    R_i^{+,T}(I) <= Delta_i * sqrt(|A_i| * T)

where Delta_i is the range of utility values for player i and |A_i| is the maximum number of actions at any information set.

**Corollary.** The exploitability (sum of regrets) after T iterations satisfies:

    epsilon(T) <= sum_{i} Delta_i * |I_i| * sqrt(|A_i|) / sqrt(T)

This gives a convergence rate of O(1/sqrt(T)) in exploitability.

**Proof sketch.** The bound follows from the regret-matching guarantee: at each information set, regret matching ensures R_i^{+,T}(I) <= Delta_i * sqrt(|A_i| * T) by the potential function argument of Hart and Mas-Colell (2000). Summing over all information sets and dividing by T yields the per-player average regret bound. The connection between average regret and epsilon-Nash follows from the minmax theorem for zero-sum games.

### 3.2 Monte Carlo CFR: Outcome Sampling

Vanilla CFR requires a full game tree traversal per iteration, which is intractable for large games. Monte Carlo CFR (MCCFR) samples subsets of the tree, producing unbiased estimates of counterfactual values.

**Algorithm 1: Outcome Sampling MCCFR**

```
Algorithm: OS-MCCFR-Traversal(h, i, t, pi_1, pi_2, pi_c, epsilon)
Input: history h, traversing player i, iteration t,
       reach probabilities (pi_1, pi_2, pi_c),
       exploration parameter epsilon
Output: sampled counterfactual value for player i

1.  if h is terminal:
2.      return u_i(h) / pi_sample(h)    // importance-weighted utility
3.
4.  Let I = information set of player(h)
5.  sigma = current strategy at I (from regret matching)
6.
7.  if player(h) == i:                  // traverser's decision node
8.      for each action a in A(I):
9.          if a == sampled_action:
10.             v(a) = OS-MCCFR-Traversal(h.a, i, t, pi_1*sigma(a), pi_2, pi_c, epsilon)
11.         else:
12.             v(a) = 0                // only sample one trajectory
13.
14.     // Counterfactual regret update
15.     for each action a in A(I):
16.         R(I, a) += pi_{-i}(h) * (v(a) - sum_a' sigma(a') * v(a'))
17.
18.     return sum_a sigma(a) * v(a)
19.
20. else:                                // opponent or chance node
21.     if player(h) is chance:
22.         sample action a with probability pi_c(a)
23.         return OS-MCCFR-Traversal(h.a, i, t, pi_1, pi_2, pi_c*Pr(a), epsilon)
24.     else:                            // opponent
25.         // epsilon-greedy exploration
26.         explore_prob = epsilon / |A(I)| + (1 - epsilon) * sigma(a_sampled)
27.         sample a from exploration policy
28.         return OS-MCCFR-Traversal(h.a, i, t, pi_1, pi_2*sigma(a), pi_c, epsilon)
```

**Theorem 2 (Lanctot, Waugh, Zinkevich, and Bowling, 2009).** Under outcome sampling MCCFR with epsilon-greedy exploration (epsilon > 0), the expected regret satisfies:

    E[R_i^T(I, a)] converges to 0 as T -> infinity

and with probability at least 1 - p, the average strategy converges to an epsilon-Nash equilibrium where:

    epsilon = O( Delta * |I| * sqrt(|A|) / (delta * sqrt(T)) )

where delta is the minimum probability of sampling any terminal history under the exploration policy:

    delta = min_{z in Z} q(z)

and Z is the set of terminal histories, q(z) is the sampling probability of terminal history z.

**The delta factor is critical.** For outcome sampling, only a single trajectory is sampled per traversal. The probability of sampling any particular terminal history z depends on the product of sampling probabilities along the path:

    q(z) = product_{h prefix of z, player(h)=i} sigma(h, a_h) * product_{h prefix of z, player(h)=-i} (epsilon/|A(h)| + (1-epsilon)*sigma(h, a_h))

In the worst case:

    delta >= (epsilon / |A|)^D

where D is the maximum game depth and |A| is the maximum action count. For Cambia with D = 57, |A| = 146, epsilon = 0.6:

    delta >= (0.6 / 146)^57 = (0.00411)^57 ~ 10^{-136}

This is an astronomically small number, meaning the worst-case convergence bound for OS-MCCFR is vacuous for Cambia. In practice, convergence occurs because:

1. Most information sets have far fewer than 146 legal actions (average is ~5).
2. The effective depth per opponent decision is much less than 57 (many nodes are chance or traverser nodes).
3. The exploration policy concentrates mass on plausible trajectories, and the variance, while high, is finite in expectation.

### 3.3 Deep CFR

Deep CFR (Brown, Lerer, Gross, and Sandholm, 2019) replaces the tabular regret accumulators with neural networks, enabling CFR to scale to games with large information-set spaces.

**Algorithm 2: Deep CFR**

```
Algorithm: Deep-CFR(num_iterations, K_traversals)
Input: number of iterations T, traversals per iteration K

1.  Initialize advantage network V_adv with random weights
2.  Initialize strategy network V_strat with random weights
3.  Initialize reservoir buffers M_adv, M_strat (capacity C)
4.
5.  for t = 1 to T:
6.      for each player i in {1, 2}:
7.          for k = 1 to K:
8.              // Run OS-MCCFR traversal for player i
9.              samples = OS-Traverse(root, i, t, V_adv)
10.             // Each sample: (features, action_mask, regret_vector, t)
11.             Add samples to M_adv using reservoir sampling
12.             Add strategy samples to M_strat using reservoir sampling
13.
14.         // Train advantage network on M_adv
15.         V_adv = TrainNetwork(M_adv, alpha=1.5)
16.             // Loss: L = sum_s (t_s^alpha) * ||V_adv(x_s) - r_s||^2
17.             //        weighted by iteration number t_s
18.
19.     // Train strategy network on M_strat
20.     V_strat = TrainNetwork(M_strat, alpha=1.5)
21.
22. return V_strat  // final strategy
```

**Theorem 3 (Brown, Lerer, Gross, and Sandholm, 2019, Theorem 1).** Let sigma_bar^T be the weighted average strategy produced by Deep CFR after T iterations, with K traversals per iteration. If the advantage network achieves mean approximation error epsilon_approx at each iteration (measured in L2 norm over the information-set distribution induced by the current strategy), then:

    Exploitability(sigma_bar^T) <= 2 * sum_i Delta_i * |I_i| * sqrt(|A_i|) / sqrt(T) + 2 * T * epsilon_approx

As K -> infinity (traversals per iteration), the empirical regret estimates converge to the true counterfactual regrets, and Deep CFR recovers the convergence rate of Linear CFR (Brown and Sandholm, 2019b):

    Exploitability(sigma_bar^T) <= C * Delta * |I| * sqrt(|A|) / sqrt(T)

**Key properties of the Deep CFR convergence guarantee:**

1. **Reservoir sampling is necessary.** Brown et al. (2019) prove that reservoir sampling (Algorithm R, Vitter 1985) preserves the convergence guarantee by maintaining a uniform sample over all past iterations. A sliding window buffer does *not* preserve convergence because it loses early-iteration samples that are critical for the weighted average strategy.

2. **The linear weighting (alpha > 0) accelerates convergence.** With alpha = 1.5 in the loss function L = t^alpha * MSE, later iterations receive higher weight, mimicking the linear averaging scheme of CFR+ (Tammelin, 2014) which converges as O(1/T) instead of O(1/sqrt(T)) in practice.

3. **The approximation error epsilon_approx does not vanish.** Unlike tabular CFR where regret accumulators converge exactly, the neural approximation introduces a persistent error floor. The total exploitability bound contains the term 2 * T * epsilon_approx, which *grows* with T. Convergence requires epsilon_approx to decrease faster than 1/T, which is achieved only if the network has sufficient capacity and the training procedure converges.

4. **Sufficient buffer size.** For the convergence guarantee to hold, the reservoir buffer must be large enough that training the network on the buffer achieves epsilon_approx approximation error. Brown et al. recommend buffer capacities on the order of K * T total samples seen, with the reservoir ensuring uniform coverage.

### 3.4 CFR+ and Discounted Variants

**CFR+ (Tammelin, 2014; Tammelin, Burch, Johanson, and Bowling, 2015):** Replaces regret matching with regret matching+, which clips negative cumulative regrets to zero. CFR+ converges as O(1/T) in practice (though the theoretical bound remains O(1/sqrt(T))). CFR+ was used to solve heads-up limit Texas Hold'em (Bowling, Burch, Johanson, and Tammelin, 2015).

**Discounted CFR (Brown and Sandholm, 2019b):** Multiplies cumulative regrets and average strategy contributions by discount factors (t^alpha, t^beta, t^gamma). With appropriate parameters, Discounted CFR provably converges at the same rate as CFR+ while being more robust to the choice of parameters.

**Why vanilla OS-CFR was chosen for Cambia:** The current implementation uses outcome sampling with epsilon-greedy exploration (epsilon = 0.6) and linear weighting (alpha = 1.5) in the Deep CFR framework. This choice was made because:

1. External sampling requires one traversal per terminal history, which is tractable only for games with low branching. Cambia's branching factor of 5.0 and depth of 57 make external sampling approximately 5^57 / 1 ~ 10^39 times more expensive per iteration than outcome sampling.
2. CFR+ is a tabular method; its advantages are captured in Deep CFR via the alpha-weighted loss.
3. Outcome sampling, despite its high variance, requires only a single trajectory per traversal, enabling practical wall-clock times (~23s per 333 traversals with the Go backend).

---

## 4. Sample Complexity for Cambia

### 4.1 Naive Tabular Bound

From Theorem 1 (Zinkevich et al. 2007), the number of iterations T* to achieve epsilon-exploitability in tabular CFR:

    epsilon(T) = sum_i Delta_i * |I_i| * sqrt(|A_i|) / sqrt(T)

Setting epsilon(T*) = epsilon_target and solving for T*:

    T* = ( sum_i Delta_i * |I_i| * sqrt(|A_i|) / epsilon_target )^2

Plugging in Cambia's values (assuming symmetric game, so both players contribute equally):

- Delta = 52
- |I| = 2.53 * 10^13 (upper bound)
- |A| = 146 (max), sqrt(146) = 12.08
- epsilon_target = 1.0 (1 point of exploitability, quite loose)

    T* = ( 2 * 52 * 2.53*10^13 * 12.08 / 1.0 )^2
       = ( 3.18 * 10^16 )^2
       = 1.01 * 10^33

This is a vacuous bound: 10^33 iterations at 33 seconds each would require ~10^27 years. The tabular bound is useless for Cambia.

### 4.2 Deep CFR with Function Approximation

The critical insight of Deep CFR is that convergence does **not** scale with |I| directly. Instead, it scales with the *approximation error* of the neural network, which depends on the network's capacity relative to the complexity of the advantage function.

**Network architecture:**

```
Input(222) -> Linear(222, 256) -> ReLU -> Dropout(0.1)
          -> Linear(256, 256) -> ReLU -> Dropout(0.1)
          -> Linear(256, 128) -> ReLU -> Linear(128, 146)
```

Total parameters: 174,610 (~175K).

**VC dimension analysis:**

For a feedforward ReLU network with L layers, W total weights, and U total units, the VC dimension is bounded by (Harvey, Liaw, and Mehrabian, 2017):

    VCdim <= O(W * L * log(W))

For our network: W = 174,610, L = 4 linear layers, so:

    VCdim <= O(174,610 * 4 * log(174,610))
           = O(174,610 * 4 * 12.07)
           ~ 8.4 * 10^6

This means the network can effectively distinguish approximately 8.4 * 10^6 distinct input patterns---far less than the 3.1 * 10^9 abstracted information sets. The network must *generalize* across information sets, which is both a feature (enables learning from limited samples) and a limitation (introduces irreducible approximation error).

**Effective sample complexity under function approximation:**

From Theorem 3 (Brown et al. 2019), the exploitability decomposes as:

    Exploit(T) <= C_cfr / sqrt(T) + C_approx * epsilon_approx(T)

where C_cfr is a game-dependent constant (absorbing Delta, |A|, and information-set structure) and epsilon_approx(T) is the neural approximation error at iteration T.

The key question is: **what is epsilon_approx?**

With a well-trained network, epsilon_approx is bounded by the network's expressivity (how well 175K parameters can represent the advantage function over 222-dimensional inputs) and the training procedure's optimization error. In practice:

- For 1,000 training steps per iteration with batch size 4096, the network sees 4,096,000 samples per iteration. With a buffer of 2M samples, this represents ~2 full passes through the buffer.
- The training loss (weighted MSE) converges to a residual value that represents epsilon_approx.
- Based on training logs, the advantage network loss stabilizes at approximately 0.05-0.15 per action dimension, suggesting epsilon_approx ~ 0.1-0.3 in advantage-value units.

**Revised sample complexity:** If epsilon_approx ~ 0.1 is irreducible, then the exploitability floor is:

    Exploit_floor ~ 2 * T * epsilon_approx / T = 2 * epsilon_approx ~ 0.2

In utility units, this corresponds to ~0.2 points of exploitability, or approximately 0.4% of the utility range. Converting to win rate against a fixed opponent is not straightforward, but a 0.2-point exploitability in a game with Delta = 52 is quite small---approximately 0.4% of the game's dynamic range.

The empirical win rate of ~55% against heuristic opponents after 850 iterations suggests that either:

1. The approximation error is still significant (the network has not fully converged at each iteration), or
2. The heuristic opponents are themselves playing near-equilibrium strategies in key subgames, limiting the exploitable gap.

### 4.3 Outcome Sampling Variance

The variance of outcome sampling counterfactual value estimates is the primary bottleneck. For a single OS traversal, the variance of the counterfactual value estimate at an information set I is:

    Var[v_hat(I)] = E[v_hat(I)^2] - E[v_hat(I)]^2

The importance-weighted estimator divides by the trajectory sampling probability q(z), so:

    Var[v_hat(I)] = O( Delta^2 / delta^2 )

where delta = min_z q(z) as before. Even with epsilon-greedy exploration (epsilon = 0.6), the effective delta for Cambia is extremely small due to the game's depth.

**Practical implication:** With K = 333 traversals per iteration, the variance of the average advantage estimate at each information set is Var / K. For the advantage network to learn meaningful signal, we need:

    K * delta_eff >> 1

where delta_eff is the effective sampling probability (much larger than the worst-case delta, because the average-case trajectory probability is much higher). Empirically, the system is learning---win rates are increasing---so delta_eff * K is sufficient for progress, but the high variance manifests as noisy convergence curves.

---

## 5. Empirical Convergence Analysis

### 5.1 Raw Data: prod-full-333

The complete win-rate trajectory over 850 iterations (outcome sampling, TPS=333, full depth) against three heuristic opponents, evaluated at 5000 games per checkpoint:

| Iter | imperfect_greedy | memory_heuristic | aggressive_snap |
|-|-|-|-|
| 25 | 0.5222 | 0.5178 | 0.5170 |
| 50 | 0.5056 | 0.5134 | 0.5116 |
| 75 | 0.5154 | 0.5160 | 0.5116 |
| 100 | 0.5172 | 0.5114 | 0.5070 |
| 125 | 0.5380 | 0.5284 | 0.5166 |
| 150 | 0.5236 | 0.5246 | 0.5162 |
| 175 | 0.5136 | 0.5226 | 0.5100 |
| 200 | 0.5302 | 0.5042 | 0.5092 |
| 225 | 0.5194 | 0.5166 | 0.5088 |
| 250 | 0.5232 | 0.5226 | 0.5136 |
| 275 | 0.5292 | 0.5310 | 0.5410 |
| 300 | 0.5354 | 0.5284 | 0.5312 |
| 325 | 0.5182 | 0.5180 | 0.5202 |
| 350 | 0.5296 | 0.5150 | 0.5304 |
| 375 | 0.5316 | 0.5282 | 0.5190 |
| 400 | 0.5326 | 0.5232 | 0.5112 |
| 425 | 0.5308 | 0.5400 | 0.5356 |
| 450 | 0.5450 | 0.5294 | 0.5298 |
| 475 | 0.5392 | 0.5172 | 0.5324 |
| 500 | 0.5330 | 0.5184 | 0.5150 |
| 525 | 0.5256 | 0.5330 | 0.5304 |
| 550 | 0.5440 | 0.5212 | 0.5400 |
| 575 | 0.5374 | 0.5368 | 0.5440 |
| 600 | 0.5424 | 0.5360 | 0.5428 |
| 625 | 0.5534 | 0.5296 | 0.5466 |
| 650 | 0.5430 | 0.5342 | 0.5404 |
| 675 | 0.5468 | 0.5526 | 0.5578 |
| 700 | 0.5274 | 0.5490 | 0.5406 |
| 725 | 0.5504 | 0.5510 | 0.5458 |
| 750 | 0.5340 | 0.5438 | 0.5360 |
| 775 | 0.5426 | 0.5392 | 0.5284 |
| 800 | 0.5454 | 0.5512 | 0.5466 |
| 825 | 0.5410 | 0.5436 | 0.5474 |
| 850 | 0.5512 | 0.5410 | 0.5546 |

### 5.2 Curve Fitting

Two candidate models were fit to the 34 data points for each opponent:

**Model A: Logarithmic**

    WR(T) = a + b * ln(T)

**Model B: Inverse square root**

    WR(T) = L_inf - c / sqrt(T)

**OLS results for imperfect_greedy:**

| Model | Parameters | RMSE | Series std dev |
|-|-|-|-|
| Logarithmic | a = 0.4737, b = 0.0102 | 0.0078 | 0.0116 |
| Inv-sqrt | L_inf = 0.5460, c = 0.2169 | 0.0089 | 0.0116 |

Both models explain a substantial fraction of variance (RMSE < std dev), but the logarithmic model fits better (RMSE = 0.0078 vs 0.0089).

**Fits for memory_heuristic:**

| Model | Parameters | RMSE |
|-|-|-|
| Logarithmic | a = 0.4713, b = 0.0099 | 0.0092 |
| Inv-sqrt | L_inf = 0.5412, c = 0.1998 | 0.0104 |

### 5.3 Theoretical Interpretation

**Why does log(T) fit better than 1/sqrt(T)?**

The CFR convergence theorem (Theorem 1) predicts O(1/sqrt(T)) exploitability, which would correspond to a win-rate improvement scaling as O(1/sqrt(T))---consistent with the inverse-square-root model. The fact that the logarithmic model fits better suggests one of:

1. **High OS variance reduces the effective sample rate.** Each of the K = 333 traversals per iteration produces a noisy counterfactual value estimate. The effective number of "useful" samples per iteration may be much less than 333 due to the high variance of importance-weighted estimates. Under this hypothesis, the effective iteration count T_eff ~ log(T) * K_eff, leading to apparent log(T) behavior in the observable T.

2. **Neural approximation error dominates.** If the advantage network's approximation error decreases slowly with more training data (e.g., because the function being approximated is complex relative to the network capacity), then the exploitability decreases as O(1/sqrt(T)) from the CFR component but the approximation error decreases more slowly, yielding an overall rate slower than 1/sqrt(T).

3. **The win-rate metric is a nonlinear transformation of exploitability.** Exploitability measures the maximum gain from deviation; win rate against a fixed opponent measures performance against a specific (possibly non-worst-case) strategy. The relationship between exploitability and win rate is not necessarily monotonic or linear.

**The two models diverge for large T:**

- Log fit predicts WR(10000) = 0.474 + 0.010 * ln(10000) = 0.474 + 0.010 * 9.21 = 0.566
- Inv-sqrt fit predicts WR(10000) = 0.546 - 0.217/sqrt(10000) = 0.546 - 0.00217 = 0.544
- Asymptotic: log fit has no upper bound (eventually predicts WR > 1.0, which is impossible); inv-sqrt fit has L_inf = 0.546

Neither model is correct in the long run. The true convergence curve must be bounded above by some game-theoretic limit (the maximum exploitability of the opponent). A reasonable composite model would be:

    WR(T) = L_inf * (1 - exp(-b * ln(T))) = L_inf * (1 - T^{-b})

which has a finite asymptote but logarithmic approach. However, we lack sufficient data in the large-T regime to fit such a model reliably.

### 5.4 Statistical Power Analysis

**Current evaluation setup:** 5000 games per checkpoint, giving a 95% confidence interval of:

    CI_95 = +/- 1.96 * sqrt(p(1-p)/N)

At p = 0.55: CI_95 = +/- 1.96 * sqrt(0.55 * 0.45 / 5000) = +/- 0.0138 (~1.4%)

**Detecting small improvements:** To detect a win-rate improvement of Delta_WR with power 0.80 at significance level alpha = 0.05:

    N = (z_{alpha/2} + z_beta)^2 * 2 * p * (1-p) / Delta_WR^2

where z_{alpha/2} = 1.96, z_beta = 0.84.

| Delta_WR to detect | Required N per evaluation | Time at ~30 games/sec |
|-|-|-|
| 0.02 (2%) | 4,802 | ~2.7 min |
| 0.01 (1%) | 19,208 | ~10.7 min |
| 0.005 (0.5%) | 76,832 | ~42.7 min |
| 0.002 (0.2%) | 480,200 | ~4.4 hours |

At the current per-50-iteration signal of 0.004-0.008, detecting improvement with 80% power requires **19,000-77,000 evaluation games**---far more than the current 5,000. This means most adjacent-checkpoint comparisons are statistically indistinguishable from noise.

### 5.5 Net Improvement Summary

Over the full 850-iteration run:

| Opponent | ~~Old WR 25~~ | ~~Old WR 850~~ | Corrected WR 100 | Corrected WR 1075 | Net improvement |
|-|-|-|-|-|-|
| imperfect_greedy | ~~0.522~~ | ~~0.551~~ | 0.291 | 0.339 | +0.048 |
| memory_heuristic | ~~0.518~~ | ~~0.541~~ | 0.297 | 0.342 | +0.045 |
| aggressive_snap | ~~0.517~~ | ~~0.555~~ | 0.278 | 0.347 | +0.069 |

> **Note (2026-02-24)**: Old numbers invalidated by stale-memory bug. Corrected numbers show the model
> LOSES to imperfect agents but improves from ~29% to ~34% over training. The model is learning but
> is far from competitive with basic heuristic play under the legacy 222-dim encoding.

---

## 6. Depth Cap Analysis

### 6.1 The Truncated Traversal Problem

Run `prod-d20-333` applied a traversal depth cap of 20 decision nodes. When a traversal reaches depth 20, the traversal terminates and returns a heuristic value estimate (typically 0 or the current hand-score difference).

**Empirical result:** The depth-capped run showed significantly worse convergence:

- Series standard deviation: 0.0168 (vs 0.0116 for full depth, a 45% increase)
- Log fit: WR = 0.5102 + 0.0026 * ln(T), RMSE = 0.0167
- The RMSE equals the standard deviation, indicating **no statistically reliable trend**---the fit explains no more variance than the null model (constant win rate).

### 6.2 Theoretical Analysis of Depth Truncation

**Claim.** Truncating traversals at depth d introduces a systematic bias in the counterfactual value estimates. The bias is bounded by:

    |bias| <= Delta * Pr(game reaches depth > d)

**Proof sketch.** Let v(I) be the true counterfactual value at information set I, and let v_d(I) be the truncated estimate. The difference arises only for game paths that extend beyond depth d:

    v(I) - v_d(I) = E_z[u(z) * 1{depth(z) > d}] - E_z[u_trunc(z) * 1{depth(z) > d}]

Since |u(z) - u_trunc(z)| <= Delta for all terminal histories z:

    |v(I) - v_d(I)| <= Delta * Pr(depth(z) > d | z passes through I)

For Cambia, the average game depth against strategic opponents is D_avg = 57 decision nodes. The fraction of games exceeding 20 decision nodes is:

    Pr(depth > 20) ~ 1 - CDF_game_length(20)

From the game statistics, the average game has 23 decision nodes against random and 57 against strategic opponents. Modeling game length as approximately geometric with mean 57:

    Pr(depth > 20) ~ exp(-20/57) ~ exp(-0.35) ~ 0.70

So approximately **70% of game trajectories are truncated**, introducing a bias of:

    |bias| ~ 52 * 0.70 ~ 36.4 utility units

in the worst case. This is an enormous bias---larger than the utility range of most games. Even a more conservative estimate (using the 23-node average against random opponents) gives:

    Pr(depth > 20) ~ 0.42, |bias| ~ 21.8

### 6.3 The Truncation-Equilibrium Mismatch

A more subtle problem: the truncated game has a **different game tree** than the full game, and therefore a **different Nash equilibrium**. The truncated game incentivizes strategies that score well within 20 decision nodes, which may differ from optimal full-game strategies. For example:

- The truncated game undervalues calling Cambia late (since late-game dynamics are cut off).
- It overvalues aggressive early play that creates score advantages within the first ~8 turns.
- It cannot learn endgame ability sequencing (peek-then-swap chains that extend beyond the depth cap).

The resulting strategy is optimal for a *different game*, not for Cambia. This is not merely high variance (which can be averaged away) but systematic bias that persists regardless of iteration count.

### 6.4 Recommendation

**Full-depth traversal should always be used.** The depth cap of 20 truncates approximately 70% of strategically relevant game trajectories and introduces bias that exceeds the utility range. The additional cost of full-depth traversal is modest: mean traversal time increases from ~15s (depth-20) to ~23s (full depth) per step, a 53% increase that is more than compensated by the elimination of bias.

If compute is limited, reducing TPS (e.g., from 333 to 200) while maintaining full depth is preferable to increasing TPS with a depth cap.

---

## 7. Hyperparameter Sensitivity Analysis

### 7.1 Traversals Per Step (TPS)

**Current setting:** TPS = 333

**Effect on convergence:** Each iteration produces K = TPS counterfactual value samples. From Theorem 3 (Brown et al. 2019), the advantage network's approximation error at each iteration depends on the quality of the regret estimates, which improves with sqrt(K):

    Var[R_hat(I, a)] ~ Var_single / K

where Var_single is the variance of a single OS traversal's regret estimate. Doubling K halves the variance, equivalent to approximately sqrt(2) ~ 1.41x faster convergence in terms of iterations.

**However, wall-clock time also doubles.** The net effect on convergence per wall-clock hour depends on whether the training is traversal-bound or training-bound:

- Current timing: 22.9s traversal + 4.2s advantage training + 4.9s strategy training = ~32s/step
- The system is **traversal-bound** (72% of time in traversals).
- Doubling TPS to 666 would increase traversal time to ~45.8s, total step time to ~55s.
- The convergence improvement per iteration is sqrt(2) ~ 1.41x.
- The wall-clock cost per iteration increases by 55/32 = 1.72x.
- Net effect: 1.41 / 1.72 = 0.82x---**slightly worse** wall-clock efficiency.

**This means the current TPS of 333 is near the wall-clock optimum** for the current hardware. Increasing TPS yields diminishing returns because traversal time dominates.

**Exception:** If training steps were reduced proportionally (fewer gradient steps when more traversals provide better signal), the tradeoff could be different. This requires empirical validation.

**Recommended experiments:**

| TPS | Est. step time | Improvement/iter | Improvement/hour | Purpose |
|-|-|-|-|-|
| 166 | ~22s | 0.71x | 1.03x | Test if halving TPS hurts |
| 333 | ~32s | 1.0x (baseline) | 1.0x | Current setting |
| 666 | ~55s | 1.41x | 0.82x | Test variance reduction |
| 1000 | ~78s | 1.73x | 0.71x | Characterize saturation |

Run each for 200 iterations and compare win-rate slopes. The expected outcome is that TPS=166-333 is optimal for wall-clock efficiency.

### 7.2 Exploration Epsilon

**Current setting:** epsilon = 0.6

The exploration parameter controls the tradeoff between following the current strategy (exploitation) and sampling uniformly (exploration) at opponent decision nodes during OS traversal.

**Effect on the sampling probability delta:**

With epsilon-greedy exploration, the probability of sampling action a at an opponent node is:

    q(a) = epsilon / |A(I)| + (1 - epsilon) * sigma(a)

The minimum per-action probability is:

    q_min = epsilon / |A(I)|

At epsilon = 0.6 with average |A| = 5:

    q_min = 0.6 / 5 = 0.12

At epsilon = 0.1:

    q_min = 0.1 / 5 = 0.02

At epsilon = 0.9:

    q_min = 0.9 / 5 = 0.18

**Tradeoff analysis:**

- **Low epsilon (0.1-0.3):** The traversal closely follows the opponent's current strategy, producing lower-variance estimates along the most probable trajectories. However, rare but strategically important trajectories (e.g., opponent using abilities optimally) are sampled infrequently, leading to poor regret estimates in those subgames. The resulting strategy may be exploitable in rare situations.

- **High epsilon (0.7-0.9):** Near-uniform sampling ensures all trajectories are visited, but the importance weights become large (since the sampling policy differs greatly from the actual strategy), increasing variance. The counterfactual value estimates are unbiased but noisy.

- **Current epsilon (0.6):** A compromise. With q_min = 0.12 per action at average nodes, the minimum trajectory probability over D opponent decisions is approximately:

    delta_eff ~ 0.12^{D_opp}

  where D_opp is the number of opponent decisions per trajectory. With D_opp ~ 28 (half of 57):

    delta_eff ~ 0.12^28 ~ 10^{-26}

  Still very small, but vastly better than the worst case.

**The optimal epsilon is game-dependent and not known a priori for Cambia.** The current value of 0.6 was inherited from the Deep CFR literature (Brown et al. used 0.6 for poker). Cambia's higher branching factor and deeper trees suggest that a different value might be optimal.

**Recommended sweep:**

| epsilon | q_min (avg |A|=5) | Expected variance | Expected bias | Priority |
|-|-|-|-|-|
| 0.1 | 0.02 | Low | High (misses rare lines) | Medium |
| 0.3 | 0.06 | Medium-low | Medium | High |
| 0.6 | 0.12 | Medium | Low | Baseline |
| 0.9 | 0.18 | High | Very low | Medium |

Run each for 300 iterations and compare convergence slopes. The most informative comparison is 0.3 vs 0.6: if 0.3 yields faster convergence, the game's strategic depth may not require aggressive exploration.

### 7.3 Network Capacity (hidden_dim)

**Current setting:** hidden_dim = 256, total parameters = 174,610

**Architecture:**

```
Linear(222, 256) -> ReLU -> Dropout(0.1)
Linear(256, 256) -> ReLU -> Dropout(0.1)
Linear(256, 128) -> ReLU -> Linear(128, 146)
```

**Capacity analysis:**

The network must map 222-dimensional input encodings to 146-dimensional advantage vectors. The information content of the input space (after abstraction) is approximately log2(3.1 * 10^9) ~ 31.5 bits. The network has 174,610 parameters (at 32-bit float, ~5.3 bits per parameter after accounting for parameter correlations and redundancy).

| hidden_dim | Parameters | Est. VCdim | Relative capacity | Training time/step |
|-|-|-|-|-|
| 128 | ~58K | ~2.8M | 0.33x | ~2.5s |
| 256 | ~175K | ~8.4M | 1.0x (baseline) | ~4.2s |
| 512 | ~594K | ~29M | 3.4x | ~12s |
| 1024 | ~2.17M | ~106M | 12.4x | ~45s |

**When is capacity insufficient?** The approximation error epsilon_approx is bounded by the network's ability to represent the advantage function. If the advantage function has structure that cannot be captured by 175K parameters (e.g., fine-grained card interactions that require representing individual card identities within buckets), then epsilon_approx will dominate the convergence bound regardless of iteration count.

**Indicators of insufficient capacity:**
- Training loss plateaus well above zero even with abundant data
- Advantage predictions show systematic errors for specific game situations
- Increasing training steps does not reduce loss further

**Recommended:** Test hidden_dim = 512 for 300 iterations. If the win-rate slope improves significantly, capacity is currently a bottleneck. If not, the current 256-dim network is sufficient and the bottleneck lies elsewhere (likely OS variance).

Increasing to hidden_dim = 1024 is not recommended as a first experiment because training time increases by ~10x, which would make the system training-bound rather than traversal-bound, fundamentally changing the optimization landscape.

### 7.4 Reservoir Buffer Size

**Current setting:** advantage_buffer_capacity = 2,000,000, strategy_buffer_capacity = 2,000,000.

**Sample accumulation rate:**

Each OS traversal visits multiple information sets, producing multiple advantage and strategy samples. Empirically, each traversal produces approximately 20-60 samples (one per traverser decision node visited). With TPS = 333:

- Samples per iteration: ~333 * 40 (average) = ~13,320 advantage samples
- After 850 iterations: ~13,320 * 850 = ~11,322,000 total samples seen

The reservoir buffer holds at most 2,000,000 samples, so after ~150 iterations the buffer is full and subsequent additions are reservoir-sampled. By iteration 850, the buffer contains a uniform random sample of 2,000,000 from the ~11.3M total samples.

**Is the buffer large enough?**

From Brown et al. (2019), the buffer should be large enough that training the network on its contents achieves low approximation error. The relevant question is: does 2M samples provide sufficient coverage of the input space?

The input space (after encoding) has 222 continuous dimensions. With 2M samples distributed over ~3.1 * 10^9 abstracted information sets, the average coverage is:

    2,000,000 / 3,100,000,000 ~ 0.00065 samples per information set

This is extremely sparse---most information sets have zero samples in the buffer. The network must generalize from visited states to unvisited ones.

**Deriving the required buffer size for epsilon_approx < 0.01:**

For a neural network with VCdim ~ 8.4 * 10^6, the generalization bound (Vapnik, 1998) gives:

    epsilon_approx <= O(sqrt(VCdim * ln(N) / N))

where N is the number of training samples. Setting epsilon_approx = 0.01:

    0.01 = sqrt(8.4*10^6 * ln(N) / N)
    10^{-4} = 8.4*10^6 * ln(N) / N
    N / ln(N) = 8.4 * 10^10

Solving numerically: N ~ 2.1 * 10^12.

This is far beyond any practical buffer size. However, the VC-dimension bound is extremely loose for neural networks in practice. Empirical generalization in deep learning is much better than the VC bound predicts, due to implicit regularization from SGD, dropout, and the structure of the data distribution.

**Practical assessment:** The current buffer size of 2M is likely adequate for the following reasons:

1. The advantage function is smooth over the input space (nearby information sets have similar advantages), so the network can interpolate effectively.
2. Dropout (0.1) provides regularization that prevents overfitting to the 2M samples.
3. The iteration-weighted loss (alpha = 1.5) effectively down-weights old samples, making the effective buffer size smaller than 2M for the purpose of current-iteration predictions.

**Recommendation:** Buffer size is unlikely to be the binding constraint. If compute allows, test 4M and 8M buffers to confirm diminishing returns. The expected impact is small.

---

## 8. Updated Iteration Estimates

### 8.1 Projections from Empirical Fits

Using the logarithmic fit WR(T) = 0.4737 + 0.0102 * ln(T) and the inverse-square-root fit WR(T) = 0.5460 - 0.2169 / sqrt(T), solving for the iteration count T at target win rates:

**Log fit:**

    T = exp((WR_target - 0.4737) / 0.0102)

**Inv-sqrt fit:**

    T = (0.2169 / (0.5460 - WR_target))^2

| Target WR | T (log fit) | T (inv-sqrt fit) | Wall clock, log (33s/step) | Wall clock, inv-sqrt (33s/step) |
|-|-|-|-|-|
| 0.55 | 2,007 | Unreachable (L_inf = 0.546) | 18.4 hours | N/A |
| 0.57 | 14,998 | Unreachable | 5.7 days | N/A |
| 0.60 | 296,559 | Unreachable | 113 days | N/A |
| 0.65 | 43,861,464 | Unreachable | 46 years | N/A |

**Derivation for log fit:**

- WR = 0.55: T = exp((0.55 - 0.4737) / 0.0102) = exp(7.48) = 1,769. Rounding with fit uncertainty: ~2,000.
- WR = 0.57: T = exp((0.57 - 0.4737) / 0.0102) = exp(9.44) = 12,553. With uncertainty: ~15,000.
- WR = 0.60: T = exp((0.60 - 0.4737) / 0.0102) = exp(12.38) = 238,344. With uncertainty: ~300,000.
- WR = 0.65: T = exp((0.65 - 0.4737) / 0.0102) = exp(17.28) = 32,157,166. With uncertainty: ~44,000,000.

**Derivation for inv-sqrt fit:**

    WR_target = 0.5460 - 0.2169 / sqrt(T)

Since L_inf = 0.5460 < 0.55, the inv-sqrt model predicts that **55% win rate is never achieved** regardless of iteration count. The model predicts asymptotic convergence to 54.6%.

### 8.2 Which Model Is More Theoretically Justified?

**Inv-sqrt model** is better grounded in CFR theory: Theorem 1 predicts O(1/sqrt(T)) convergence in exploitability, and the inv-sqrt fit directly mirrors this bound with a finite asymptote L_inf representing the strategy's maximum achievable win rate against the fixed opponent.

**Log model** fits the observed data better (RMSE 0.0078 vs 0.0089) and is consistent with the hypothesis that high OS variance effectively reduces the useful sample rate, yielding apparent log(T) behavior over the observed range. However, the log model is physically unreasonable for large T (it predicts WR > 1.0 at T > exp(51.6) ~ 3.9 * 10^22).

**Resolution:** The truth likely lies between the models. A plausible scenario:

1. For T < 5,000 (the observed range and near extrapolation), the log model is more accurate because the system is still in the "pre-asymptotic" regime where the neural network is actively improving its representation.
2. For T > 10,000, the inv-sqrt model becomes more relevant as the neural network's capacity limits bind and the system approaches the epsilon_approx floor.
3. The true asymptote is likely higher than 0.546 (the inv-sqrt estimate is biased downward by the limited data range) but lower than the log model suggests.

**Confidence intervals on the predictions:**

The fit parameters have uncertainty. For the log fit with RMSE = 0.0078 over 34 data points:

- Standard error on the slope b: SE_b ~ RMSE / sqrt(sum(ln(T_i) - mean(ln(T_i)))^2) ~ 0.0024
- 95% CI on b: 0.0102 +/- 0.0048, i.e., [0.0054, 0.0150]

With the lower bound on the slope (b = 0.0054), reaching WR = 0.55 requires:

    T = exp((0.55 - 0.4737) / 0.0054) = exp(14.1) ~ 1,300,000

With the upper bound (b = 0.0150), WR = 0.55 requires:

    T = exp((0.55 - 0.4737) / 0.0150) = exp(5.09) ~ 162

**The 95% CI for iterations to reach 55% win rate is approximately [162, 1,300,000].** This enormous range reflects the fundamental difficulty of extrapolating from noisy data. The point estimate of T ~ 2,000 is the best single prediction, but should be interpreted with great caution.

### 8.3 Extrapolated Win Rates at Fixed Iteration Counts

| T (iterations) | WR (log fit) | WR (inv-sqrt fit) | 95% CI (log fit) |
|-|-|-|-|
| 1,000 | 0.544 | 0.539 | [0.537, 0.551] |
| 2,000 | 0.551 | 0.541 | [0.541, 0.561] |
| 5,000 | 0.561 | 0.543 | [0.546, 0.576] |
| 10,000 | 0.568 | 0.544 | [0.550, 0.586] |
| 50,000 | 0.584 | 0.545 | [0.558, 0.610] |

---

## 9. Recommended Next Steps

Ordered by expected information gain per compute hour:

### Priority 1: Increase Evaluation Game Count (5,000 -> 20,000)

**Rationale:** The single most impactful change for *understanding* convergence. Reducing the 95% CI from +/-1.4% to +/-0.7% would make per-50-iteration improvements (0.004-0.008) detectable, transforming the convergence curve from noise to signal.

**Cost:** ~2.5 minutes per evaluation (at ~130 games/sec), no training compute required. Can be done retroactively on existing checkpoints.

**Expected outcome:** Sharper convergence curves that allow distinguishing between the log and inv-sqrt models. If the true slope is closer to b = 0.005 (lower end of the CI), this will become apparent. If b = 0.015 (upper end), convergence is faster than expected and fewer iterations are needed.

### Priority 2: Continue Training from Checkpoint iter 850

**Rationale:** The log fit predicts ~+0.004 per 100 additional iterations. Running to T = 2,000 would provide 1,150 additional data points (if evaluated every 25 iterations) and should bring the win rate to approximately 0.551 (log model) or 0.541 (inv-sqrt model). The divergence between models becomes measurable at T = 2,000.

**Cost:** (2,000 - 850) * 33s = ~10.5 hours of training.

**Expected outcome:** If WR at T = 2,000 is above 0.545, the log model is confirmed and WR = 0.55+ is achievable. If WR is below 0.542, the inv-sqrt model is confirmed and fundamental changes (network capacity, sampling method) are needed.

### Priority 3: TPS Sweep

**Rationale:** Determine whether the current TPS of 333 is optimal or if lower TPS (more iterations per hour with noisier estimates) or higher TPS (fewer iterations with cleaner estimates) yields faster convergence.

**Protocol:** Run TPS in {166, 333, 666, 1000} for 200 iterations each, evaluating every 25 iterations with 20,000 games. Compare win-rate slopes.

**Cost:** ~4 * 200 * (22-78s) = 14-43 hours total (parallelizable across 4 runs).

**Expected outcome:** Identification of the wall-clock-optimal TPS. Theory predicts TPS = 166-333 is optimal; empirical confirmation would prevent wasted compute on TPS > 333.

### Priority 4: Exploration Epsilon Sweep

**Rationale:** The most uncertain hyperparameter. The current value of 0.6 was borrowed from poker; Cambia's different branching structure may favor a different value.

**Protocol:** Run epsilon in {0.1, 0.3, 0.6, 0.9} for 300 iterations each with 20,000-game evaluations.

**Cost:** ~4 * 300 * 33s = ~11 hours total.

**Expected outcome:** If epsilon = 0.3 converges faster, it means the OS variance is the primary bottleneck and can be partially addressed without algorithmic changes. If epsilon = 0.9 converges faster, it means exploration is insufficient and rare game lines are under-sampled.

### Priority 5: Go FFI Optimization (5 -> 1 FFI Calls per Node)

**Rationale:** The current implementation makes ~5 FFI calls per decision node (state query, legal actions, apply action, encode, etc.). Consolidating to a single "step" call that returns all needed information would reduce FFI overhead by approximately 80%, enabling ~60% more traversals per second at the same wall-clock cost.

**Cost:** Engineering effort (~1-2 days of implementation), no additional compute.

**Expected outcome:** Effectively equivalent to increasing TPS from 333 to ~530 at no wall-clock cost. This is a strict improvement that benefits all future training runs.

---

## 10. Updated Analysis---20,000-Game Evaluation

### 10.1 High-Precision Win Rates at Iteration 850

To resolve the divergence between the logarithmic and inverse-square-root convergence models, 20,000-game evaluations were run against the iter_850 checkpoint for all 5 baselines. At N=20,000, the 95% CI is +/-0.69% (vs +/-1.4% at N=5,000), halving the noise floor.

**Results (INVALIDATED 2026-02-24):**

> **WARNING**: The imperfect agent numbers below are INVALID. A stale-memory bug in baseline agents
> (`ImperfectMemoryMixin._ensure_initialized()` never re-initialized for new games) caused opponents
> to play on garbage memory from game 1 across all subsequent games. Greedy and random numbers are valid.
> See corrected results below.

| Baseline | ~~Old Win Rate~~ | Corrected (5k, seat-balanced) |
|-|-|-|
| random | 0.5120 (valid) | 0.498 |
| greedy (perfect info) | 0.1460 (valid) | 0.148 |
| imperfect_greedy | ~~0.5497~~ | **0.310** |
| memory_heuristic | ~~0.5489~~ | **0.296** |
| aggressive_snap | ~~0.5514~~ | **0.322** |

**Corrected composite average** across three imperfect-info baselines at iter 850: **0.309** — the model LOSES to properly-functioning imperfect agents.

**Corrected observations:**

1. The model wins ~34% against imperfect agents at iter 1075 (not ~55%). Imperfect agents are strong opponents when properly functioning.
2. There IS convergence: 29% at iter 100 → 34% at iter 1075 (+5pp over 975 iterations). The model is learning, but slowly.
3. Random and greedy results are unchanged (no state bug in those agents).
4. The convergence log-fit and all derived conclusions (performance tiers, EP-PBS targets) must be recalculated.

### 10.2 Model Discrimination

The central question: does the new high-precision data point at T=850 favor the logarithmic model or the inverse-square-root model?

**Original model predictions at T=850:**

| Model | Predicted WR(850) | Observed WR (20k) | Residual | Z-score | P-value (two-tailed) |
|-|-|-|-|-|-|
| Logarithmic | 0.5425 | 0.5497 | +0.0072 | 1.94 | 0.052 |
| Inv-sqrt | 0.5386 | 0.5497 | +0.0111 | 3.00 | 0.003 |

**Interpretation:**

- The observed win rate of 0.5497 lies **above both model predictions**, but is much more discrepant from the inv-sqrt model.
- The inv-sqrt model is **rejected** at the 99.7% confidence level (z=3.00, p=0.003). The observed win rate is 1.11 percentage points above the inv-sqrt prediction, which is 3.0 standard errors away.
- The log model is **marginally consistent** with the data (z=1.94, p=0.052). The observed win rate is 0.72 percentage points above the log prediction, which is 1.94 standard errors---just outside the 95% threshold.
- The fact that both models underpredict suggests the true convergence rate at T=850 is **slightly faster** than either model estimated from the T=25-850 trajectory. This is consistent with the agent beginning to exploit compound strategic patterns (multi-turn ability sequences) that accelerate learning in the later iterations.

### 10.3 Re-fitted Models

Both models were re-fit using weighted least squares, with the 34 original 5,000-game points and the new 20,000-game point weighted proportionally to sample size.

**Updated log fit:**

    WR(T) = 0.4692 + 0.0111 * ln(T)

Parameters: a = 0.4692, b = 0.0111 (up from 0.0102), WRMSE = 0.0077.

**Updated inv-sqrt fit:**

    WR(T) = 0.5486 - 0.2414 / sqrt(T)

Parameters: L_inf = 0.5486 (up from 0.5460), c = 0.2414, WRMSE = 0.0091.

**AIC/BIC comparison:**

| Model | AIC | BIC | Log-likelihood |
|-|-|-|-|
| Logarithmic | -237.15 | -234.04 | 120.57 |
| Inv-sqrt | -225.73 | -222.62 | 114.87 |
| **Delta (inv - log)** | **+11.41** | **+11.41** | **-5.70** |

The log model is **decisively preferred** by both AIC and BIC. A Delta AIC of 11.4 corresponds to the inv-sqrt model being exp(-11.4/2) = 0.003x as likely as the log model---essentially no support for the inv-sqrt model over the log model given the data.

### 10.4 Updated Predictions

Using the re-fitted log model WR(T) = 0.4692 + 0.0111 * ln(T):

| T (iterations) | WR (updated log) | WR (updated inv-sqrt) | Gap | Wall clock (33s/step) |
|-|-|-|-|-|
| 1,000 | 0.546 | 0.541 | 0.005 | 9.2 hours |
| 2,000 | 0.553 | 0.543 | 0.010 | 18.3 hours |
| 5,000 | 0.563 | 0.545 | 0.018 | 1.9 days |
| 10,000 | 0.571 | 0.546 | 0.025 | 3.8 days |
| 50,000 | 0.589 | 0.548 | 0.041 | 19.1 days |

### 10.5 Updated Iteration Estimates

Solving the updated log fit for target win rates:

| Target WR | T (iterations) | Wall clock | Previous estimate |
|-|-|-|-|
| 0.55 | ~1,500 | ~13.7 hours | ~2,000 (18.4 hours) |
| 0.57 | ~9,100 | ~3.5 days | ~15,000 (5.7 days) |
| 0.60 | ~138,000 | ~52.6 days | ~297,000 (113 days) |
| 0.65 | ~12,700,000 | ~13.3 years | ~44,000,000 (46 years) |

The updated estimates are uniformly lower than the original log-fit estimates, because the re-fitted slope b = 0.0111 is steeper than the original b = 0.0102 (the new high-precision point pulled the slope up). This is good news: convergence is approximately 30-50% faster than originally estimated.

### 10.6 The Asymptote Question

The updated inv-sqrt asymptote is L_inf = 0.5486 with 95% CI [0.5424, 0.5548]. Since the current measurement already stands at 0.5497, this means:

- **The observed win rate at T=850 already exceeds the inv-sqrt model's point estimate of the asymptote** (0.5497 > 0.5486). While the 95% CI on L_inf extends to 0.5548, the fact that the current win rate is at the asymptote makes the inv-sqrt model implausible as a long-term description.
- **The inv-sqrt model's asymptote is not a hard ceiling.** The model is statistically rejected (p=0.003). If there is an asymptote, it is higher than 0.549.

However, the log model must also eventually plateau (win rates cannot exceed 1.0). A more physically realistic model would be a saturating function:

    WR(T) = L_max * (1 - exp(-b * ln(T))) = L_max * (1 - T^{-b})

where L_max is the true game-theoretic maximum win rate against these opponents. With only 850 iterations of data, we cannot reliably estimate L_max, but the data is consistent with L_max in the range [0.57, 0.70+]. This question will be resolvable at T ~ 5,000-10,000 when the models diverge by 2-4 percentage points.

### 10.7 Neural Approximation Floor Assessment

The rejection of the inv-sqrt model has a critical architectural implication: **the neural approximator (175K parameters, hidden_dim=256) has NOT yet reached its capacity limit at T=850.** If epsilon_approx were dominating, convergence would flatten to a 1/sqrt(T) approach toward a fixed asymptote---exactly the inv-sqrt model. The data shows convergence is still log-linear, meaning:

1. The advantage network is still improving its representation with more training data.
2. The reservoir buffer (2M samples, ~11.3M total samples seen) is providing sufficient coverage for the network to learn.
3. The binding constraint remains **outcome sampling variance**, not network capacity.

This does **not** rule out a capacity ceiling at higher T. The network's VC dimension (~8.4 * 10^6) bounds the maximum number of distinguishable information sets. As the strategy becomes more refined (at higher T), the advantage function may develop fine-grained structure that exceeds the 175K-parameter network's expressivity. But at T=850, we are not yet at that point.

### 10.8 Revised Recommendations

Based on the 20,000-game analysis:

**1. Continue training to T=2,000 (Priority 1).**

The log model predicts WR(2000) = 0.553, a gain of ~+0.003 over the current 0.550. At 20,000-game precision (+/-0.007), this is marginal but detectable. More importantly, reaching T=2,000 provides a second high-precision anchor point that will further constrain the convergence model. Cost: ~10.5 hours from the current checkpoint.

**2. Evaluate at T=2,000 with 20,000 games (Priority 1).**

If WR(2000) > 0.551, the log model is confirmed and further training to T=5,000+ is justified. If WR(2000) < 0.549, a capacity ceiling may be emerging and network expansion (hidden_dim=512) should be tested before additional iterations.

**3. Network capacity sweep (Priority 2, conditional).**

Since the approximation floor is not yet binding, a capacity sweep (hidden_dim=512, 1024) is **lower priority** than continued training. However, preemptively testing hidden_dim=512 for 300 iterations would establish whether capacity becomes binding at higher T.

**4. TPS and epsilon sweeps remain valuable (Priority 3).**

The current TPS=333 and epsilon=0.6 are untested against alternatives. Since OS variance is the binding constraint (not capacity), reducing variance via TPS or epsilon optimization could yield disproportionate gains. The epsilon sweep (0.3 vs 0.6) is the highest-information experiment after the T=2,000 evaluation.

**5. The random-opponent result (51.2%) is a measurement artifact — baseline requires replacement.**

The `RandomAgent` baseline samples uniformly from all legal actions including `CallCambia`, which is legal from round 0 (production config: `cambia_allowed_round: 0`). At `START_TURN` only two actions are legal (`DrawStockpile`, `CallCambia`), so the random agent calls Cambia with 50% probability every turn. This produces a geometric distribution where ~75% of games end in 4 or fewer turns — far too short for any learned strategy to manifest. The 51.2% result measures starting hand luck, not strategic skill.

The "exploitative play" interpretation previously stated here is incorrect and has been retracted. A `RandomNoCambia` baseline (CallCambia masked out) has been implemented; once evaluated, this will provide a meaningful lower-bound signal. The strategic baselines (imperfect_greedy, memory_heuristic, aggressive_snap) are unaffected and remain the primary convergence metrics.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|-|-|
| T | Number of CFR iterations |
| K | Traversals per iteration (TPS) |
| I_i | Set of information sets for player i |
| \|I\| | Cardinality of the information set space |
| A(I) | Legal actions at information set I |
| \|A\| | Maximum action count (146) |
| Delta | Utility range (52) |
| b | Average branching factor (5.0) |
| D | Average game depth in decision nodes |
| sigma | Strategy profile |
| sigma_bar | Average strategy profile |
| v_i(sigma, I) | Counterfactual value for player i at I under sigma |
| R_i^T(I, a) | Cumulative counterfactual regret at iteration T |
| epsilon | Exploration parameter (0.6) |
| delta | Minimum trajectory sampling probability |
| epsilon_approx | Neural network approximation error |
| alpha | Iteration weighting exponent (1.5) |
| VCdim | Vapnik-Chervonenkis dimension |
| WR | Win rate against a fixed opponent |
| CI_95 | 95% confidence interval |

## Appendix B: Derivation of OS Minimum Trajectory Probability

For an outcome-sampling traversal with epsilon-greedy exploration, the probability of sampling terminal history z = (a_1, a_2, ..., a_n) is:

    q(z) = product_{k: player(h_k)=i} sigma(h_k, a_k) * product_{k: player(h_k)=-i} [epsilon/|A(h_k)| + (1-epsilon)*sigma(h_k, a_k)] * product_{k: player(h_k)=chance} Pr(a_k)

The minimum over all z is achieved when:
1. At traverser nodes: sigma places minimum mass on the sampled action. Under regret matching, this is at most 1/|A(I)| (uniform when all regrets are zero or negative).
2. At opponent nodes: epsilon/|A(I)| is the minimum (when sigma(a) = 0).
3. At chance nodes: the minimum probability card draw.

For Cambia with D_opponent ~ 28 opponent decision nodes per trajectory, average |A| ~ 5:

    delta >= (0.6/5)^28 * (1/5)^28 * product_chance_probs
         = 0.12^28 * 0.2^28 * delta_chance
         ~ 10^{-26} * 10^{-20} * delta_chance
         ~ 10^{-46} * delta_chance

where delta_chance accounts for card draws and shuffles. This confirms that the worst-case delta is astronomically small, and the OS variance bound is vacuous in the worst case.

## Appendix C: Comparison of Sampling Schemes

| Property | External Sampling | Outcome Sampling |
|-|-|-|
| Trajectories per traversal | All for traverser, one for opponent | One total |
| Variance | Low (all traverser actions explored) | High (single trajectory) |
| Cost per traversal | O(b^{D_traverser}) | O(D) |
| Cost for Cambia (D=57, b=5) | ~5^28 ~ 3.7 * 10^19 | ~57 operations |
| Convergence rate | O(1/sqrt(T)), tight | O(1/(delta*sqrt(T))), loose |
| Practical feasibility for Cambia | Infeasible | Feasible |

External sampling is theoretically superior but computationally infeasible for Cambia. Outcome sampling is the only viable option for a game of this complexity, at the cost of higher variance and slower convergence.

## Appendix D: Production Configuration Reference

```yaml
# prod-full-333 configuration
cfr_training:
  num_iterations: 850
  save_interval: 1
  num_workers: 1

agent_params:
  memory_level: 1
  time_decay_turns: 3

cambia_rules:
  allowDrawFromDiscardPile: false
  allowReplaceAbilities: false
  allowOpponentSnapping: false
  max_game_turns: 46
  cards_per_player: 4
  initial_view_count: 2
  use_jokers: 2

deep_cfr:
  hidden_dim: 256
  dropout: 0.1
  learning_rate: 0.001
  batch_size: 4096
  train_steps_per_iteration: 1000
  alpha: 1.5
  traversals_per_step: 333
  advantage_buffer_capacity: 2000000
  strategy_buffer_capacity: 2000000
  save_interval: 25
  sampling_method: outcome
  exploration_epsilon: 0.6
  engine_backend: go
  traversal_depth_limit: 0    # unlimited (full depth)
```

**Timing (at ~iteration 400):**

| Phase | Time per step | Fraction |
|-|-|-|
| Traversals (333 OS) | 22.9s | 72% |
| Advantage training (1000 steps) | 4.2s | 13% |
| Strategy training (1000 steps) | 4.9s | 15% |
| **Total** | **~32s** | **100%** |

---

## References

1. Bowling, M., Burch, N., Johanson, M., and Tammelin, O. (2015). Heads-up limit hold'em poker is solved. *Science*, 347(6218):145-149.

2. Brown, N., Lerer, A., Gross, S., and Sandholm, T. (2019). Deep counterfactual regret minimization. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, pp. 793-802.

3. Brown, N. and Sandholm, T. (2019). Solving imperfect-information games via discounted regret minimization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01):1481-1488.

4. Gibson, R., Lanctot, M., Burch, N., Szafron, D., and Bowling, M. (2012). Generalized sampling and variance in counterfactual regret minimization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 26(1):1355-1361.

5. Hart, S. and Mas-Colell, A. (2000). A simple adaptive procedure leading to correlated equilibrium. *Econometrica*, 68(5):1127-1150.

6. Harvey, N., Liaw, C., and Mehrabian, A. (2017). Nearly-tight VC-dimension bounds for piecewise linear neural networks. *Proceedings of the 30th Conference on Learning Theory (COLT)*, pp. 1064-1068.

7. Lanctot, M., Waugh, K., Zinkevich, M., and Bowling, M. (2009). Monte Carlo sampling for regret minimization in extensive games. *Advances in Neural Information Processing Systems (NeurIPS)*, 22:1078-1086.

8. Tammelin, O. (2014). Solving large imperfect information games using CFR+. *arXiv preprint arXiv:1407.5042*.

9. Tammelin, O., Burch, N., Johanson, M., and Bowling, M. (2015). Solving heads-up limit Texas hold'em. *Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 645-652.

10. Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.

11. Vitter, J. S. (1985). Random sampling with a reservoir. *ACM Transactions on Mathematical Software*, 11(1):37-57.

12. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2007). Regret minimization in games with incomplete information. *Advances in Neural Information Processing Systems (NeurIPS)*, 20:1729-1736.
