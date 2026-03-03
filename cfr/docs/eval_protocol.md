# Evaluation Protocol: Cambia Deep CFR

## 1. Primary Metric: mean_imp

**mean_imp** is the mean win rate of the CFR agent (P0) across 5 baseline opponents, 5000 games per baseline per checkpoint.

The canonical baseline set is defined as `MEAN_IMP_BASELINES` in `cfr/src/evaluate_agents.py` and imported from there by all evaluation scripts.

### Baselines in mean_imp (5)

| Baseline | Description |
|-|-|
| random_no_cambia | Uniform random actions; never calls Cambia |
| random_late_cambia | Uniform random; calls Cambia only when hand sum ≤ threshold |
| imperfect_greedy | Greedy policy using belief state (imperfect information) |
| memory_heuristic | Tracks observed cards; uses memory for decisions |
| aggressive_snap | Actively snaps matching cards when possible |

### Context-Only Baselines (excluded from mean_imp)

| Baseline | Reason Excluded |
|-|-|
| random | True uniform including random Cambia calls; games end turn 1-2; near coin-flip, uninformative |
| greedy | Perfect-information oracle; theoretical ceiling reference only |

Both context-only baselines are evaluated and logged but do not contribute to mean_imp.

### Historical Note

Prior to 2026-02-27, mean_imp used only 3 baselines (imperfect_greedy, memory_heuristic, aggressive_snap), referred to as **mean_imp(3)**. Current mean_imp is **mean_imp(5)**. Values are not directly comparable: mean_imp(5) runs approximately 6-7 percentage points higher than mean_imp(3) due to inclusion of random floor baselines.

## 2. Behavioral Metrics

Behavioral metrics are collected per-checkpoint alongside win rates and reported in `metrics.jsonl`.

### T1 Cambia Rate

Fraction of games where the CFR agent (P0) calls Cambia on turn 1.

- **Target**: < 10%
- **Warning threshold**: > 40% (indicates degenerate early-termination equilibrium where the agent has learned to end games before observing any information)
- A rising T1 Cambia rate alongside falling mean_imp is the primary signal of regression to a degenerate strategy

### Average Game Length

Mean number of turns per game across all evaluated games.

- **Healthy range**: 15-40 turns
- **Degenerate signal**: 3-5 turns (indicates the agent, or a combination of agent + baseline, is terminating games abnormally early)
- Average game length is reported per-baseline and as an aggregate across all baseline evaluations

## 3. Head-to-Head Cross-Iteration Evaluation

Baseline win rates can be gamed by RPS-cycling against fixed opponents. Head-to-head evaluation provides a complementary signal of genuine improvement.

### Matchups

| Label | Description |
|-|-|
| previous | iter T win rate vs iter T-1 (adjacent checkpoint; local step signal) |
| earliest | iter T win rate vs earliest checkpoint (absolute improvement anchor) |
| t_minus_500 | iter T win rate vs closest checkpoint to iter T-500 (local improvement at scale) |

### Protocol

- 2000 games per matchup (default; configurable via `--h2h-games`)
- Position alternation: agent A is P0 on odd-numbered games, P1 on even-numbered games
- Win rate reported from agent A's perspective (iter T)

### Interpretation Threshold

- **> 55% win rate**: genuine improvement over comparison checkpoint
- **≤ 55% win rate** despite rising mean_imp: suspect RPS cycling against baselines, not genuine Nash convergence

### Implementation Status

Head-to-head evaluation is fully implemented in `cfr/scripts/eval_watcher.py` (`evaluate_head_to_head` function). It runs automatically after each checkpoint evaluation unless disabled with `--h2h-games 0`. Results are written to `head_to_head.jsonl` and dual-written to the SQLite run database.

## 4. Future: RL Best-Response Exploitability

**Status: DEFERRED**

Freeze the EMA strategy network and train a PPO or DQN agent to maximally exploit it. The RL agent's win rate against the frozen CFR strategy constitutes an exploitability upper bound.

### Deferral Reasons

- No existing RL infrastructure in the codebase
- Estimated 4-12 hours compute per checkpoint
- Cannot run concurrently with CFR training on Intel Arc A310 (4 GB VRAM)
- Mean_imp + H2H provides sufficient signal for current training phases

## 5. Baseline Agent Descriptions

### Included in mean_imp

**random_no_cambia**: Samples uniformly from all legal actions on each turn except Cambia calls. Provides the random-play lower floor with games of realistic length.

**random_late_cambia**: Samples uniformly from all legal actions. Calls Cambia when the estimated hand sum falls below a configured threshold, simulating a player who terminates late in a game. Provides a slightly higher floor than random_no_cambia.

**imperfect_greedy**: Greedy policy operating on belief state rather than ground truth. Chooses the action that minimizes expected hand value under the current belief distribution. Represents a competent heuristic player without memory exploitation.

**memory_heuristic**: Tracks all observed cards and uses this memory to inform swap and snap decisions. Represents a methodical player who leverages information accumulation. Stronger than imperfect_greedy on games that reach mid-late stages.

**aggressive_snap**: Prioritizes snapping cards whenever a legal snap is available. Represents an opponent who actively contests the discard pile. Tests the CFR agent's ability to handle snap-race dynamics.

### Context-Only

**random**: True uniform over all legal actions including Cambia. Games frequently end on turn 1 or 2 because the agent randomly calls Cambia before accumulating information. Win rate against this baseline is not a strength metric and is excluded from mean_imp.

**greedy**: Perfect-information oracle that acts optimally given full knowledge of all card positions. Provides a theoretical ceiling. The CFR agent is not expected to exceed 50% win rate against this baseline in the near term.

## 6. Output File Schemas

### `metrics.jsonl`

One record per (checkpoint, baseline) pair. Written by the eval watcher and `collect_metrics.py` as checkpoints are evaluated.

```json
{
  "run": "string",
  "iter": 0,
  "baseline": "string",
  "win_rate": 0.0,
  "ci_low": 0.0,
  "ci_high": 0.0,
  "games_played": 5000,
  "p0_wins": 0,
  "p1_wins": 0,
  "ties": 0,
  "adv_loss": 0.0,
  "strat_loss": 0.0,
  "avg_game_turns": 0.0,
  "t1_cambia_rate": 0.0,
  "avg_score_margin": 0.0,
  "timestamp": "ISO8601"
}
```

**Notes:**
- `ci_low` and `ci_high` are Wilson score interval bounds at 95% confidence (z=1.96), computed as `wilson_ci(p0_wins, games_played)`. Both `eval_watcher.py` and `collect_metrics.py` use this calculation.
- `adv_loss` and `strat_loss` are copied from the checkpoint's training metadata, not recomputed during eval
- `avg_score_margin` is the mean score difference (CFR agent score minus opponent score) across all games; negative is better (lower hand = winning)
- `t1_cambia_rate` is computed only for the CFR agent (P0), not the baseline

### `head_to_head.jsonl`

One record per head-to-head matchup evaluation. Written automatically by the eval watcher after each checkpoint evaluation.

```json
{
  "run": "string",
  "iter_a": 0,
  "iter_b": 0,
  "label": "previous|earliest|t_minus_500",
  "a_wins": 0,
  "b_wins": 0,
  "ties": 0,
  "a_win_rate": 0.0,
  "avg_game_turns": 0.0,
  "timestamp": "ISO8601"
}
```

**Notes:**
- `iter_a` is always the later (newer) checkpoint
- `iter_b` is the earlier comparison checkpoint
- `a_win_rate` = `a_wins / (a_wins + b_wins + ties)`
- `label` is one of: `previous` (adjacent iter), `earliest` (first checkpoint in run), `t_minus_500` (closest checkpoint ~500 iters back)
- Records are also dual-written to the SQLite run database via `run_db.insert_head_to_head`

### `evaluations/iter_N/{baseline}.jsonl`

Per-game records for detailed analysis. Written only when per-game logging is enabled (controlled by `log_per_game: true` in eval config).

```json
{
  "game_id": "uuid",
  "winner": 0,
  "turns": 0,
  "duration_ms": 0,
  "actions": [
    {
      "turn": 0,
      "player": 0,
      "action": 0,
      "legal_count": 0
    }
  ]
}
```

**Notes:**
- `winner` is 0 (CFR agent wins), 1 (baseline wins), or -1 (tie)
- `legal_count` records the size of the legal action set at the time of the action for action diversity analysis
- Per-game logs are large; enable only for targeted diagnostic evaluations

## 7. Interpretation Guide

### Convergence Patterns

| mean_imp | T1 Cambia Rate | Game Length | H2H vs T-500 | Diagnosis |
|-|-|-|-|-|
| Rising | Dropping | Increasing | > 55% | Healthy convergence |
| Flat | Stable | Stable | ~50% | Plateau: architectural or algorithmic change needed |
| Rising | Flat/Rising | Short | ~50% | RPS cycling: baseline exploitation, not genuine improvement |
| Dropping | Rising | Dropping | < 50% | Regression to degenerate equilibrium |
| Rising then dropping | Rising | Dropping | < 50% | Advantage overfitting: reduce train_steps_per_iteration |

### Advantage Overfitting Diagnosis

Advantage overfitting manifests as:
1. mean_imp peaks then decays over 100-300 iterations
2. T1 Cambia rate climbs as the agent learns degenerate early-termination
3. Reducing `train_steps_per_iteration` (e.g., 1000 → 250) eliminates the decay

Root cause: the advantage network memorizes the current buffer rather than learning a generalizable strategy. Adaptive train steps (`target_buffer_passes`) mitigate this by scaling training to buffer size.

### Early Stopping Criteria

An experiment should be terminated early if:
- T1 Cambia rate exceeds 40% for 3+ consecutive checkpoints
- mean_imp drops below 25% after previously exceeding 30%
- H2H win rate vs anchor drops below 40% (agent is actively regressing)

### Healthy Baseline Ordering

Expected agent hierarchy from weakest to strongest:
```
random < random_no_cambia ≈ random_late_cambia < aggressive_snap < memory_heuristic < imperfect_greedy < CFR agent < greedy
```

A well-trained CFR agent should exceed 50% win rate against all mean_imp baselines and approach (but not exceed) 50% against greedy.
