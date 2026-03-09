# Evaluation Protocol: Cambia CFR

## 1. Primary Metric: mean_imp

mean_imp is the mean win rate of the trained agent (P0) across 5 baseline opponents, 5000 games per baseline per checkpoint.

The canonical baseline set is defined as `MEAN_IMP_BASELINES` in `cfr/src/evaluate_agents.py` and imported from there by all evaluation scripts.

### Baselines in mean_imp (5)

| Baseline | Description |
|-|-|
| random_no_cambia | Uniform random actions; never calls Cambia |
| random_late_cambia | Uniform random; calls Cambia only when hand sum is at or below threshold |
| imperfect_greedy | Greedy policy using belief state (imperfect information) |
| memory_heuristic | Tracks observed cards; uses memory for decisions |
| aggressive_snap | Actively snaps matching cards when possible |

### Context-only baselines (excluded from mean_imp)

| Baseline | Reason excluded |
|-|-|
| random | True uniform including random Cambia calls; games end turn 1-2; near coin-flip, uninformative |
| greedy | Perfect-information oracle; theoretical ceiling reference only |

Both context-only baselines are evaluated and logged but do not contribute to mean_imp.

### Historical note

Prior to 2026-02-27, mean_imp used only 3 baselines (imperfect_greedy, memory_heuristic, aggressive_snap), referred to as mean_imp(3). Current mean_imp is mean_imp(5). Values are not directly comparable: mean_imp(5) runs approximately 6-7 percentage points higher than mean_imp(3) due to inclusion of random floor baselines.

## 2. Behavioral Metrics

Behavioral metrics are collected per-checkpoint alongside win rates and reported in `metrics.jsonl`.

### T1 Cambia Rate

Fraction of games where the agent (P0) calls Cambia on turn 1.

- Target: < 10%
- Warning threshold: > 40% (indicates degenerate early-termination equilibrium where the agent has learned to end games before observing any information)
- A rising T1 Cambia rate alongside falling mean_imp is the primary signal of regression to a degenerate strategy

### Average Game Length

Mean number of turns per game across all evaluated games.

- Healthy range: 15-40 turns
- Degenerate signal: 3-5 turns (indicates the agent, or a combination of agent + baseline, is terminating games abnormally early)
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

### Interpretation threshold

- \> 55% win rate: genuine improvement over comparison checkpoint
- At or below 55% despite rising mean_imp: suspect RPS cycling against baselines, not genuine Nash convergence

### Implementation

Head-to-head evaluation is implemented in `cfr/scripts/eval_watcher.py` (`evaluate_head_to_head` function). It runs automatically after each checkpoint evaluation unless disabled with `--h2h-games 0`. Results are written to `head_to_head.jsonl` and dual-written to the SQLite run database.

## 4. Running Evaluations

### Run-dir mode (preferred)

The `cambia evaluate` command accepts a run directory as its positional argument. This is the recommended invocation for all evaluation work because it auto-detects configuration, agent type, and checkpoint prefix from the run directory contents, and persists results to both `metrics.jsonl` and the SQLite run database.

```bash
# Evaluate the latest checkpoint in a run directory (5000 games per baseline)
cambia evaluate runs/sog-phase3-v3/ --latest

# Evaluate a specific epoch
cambia evaluate runs/gtcfr-phase2/ --epoch 200

# Override game count for a quick spot-check
cambia evaluate runs/rebel-phase1/ --latest -n 100
```

Auto-detection works as follows:

- Config: reads `config.yaml` from the run directory
- Algorithm: `run_db.infer_algorithm()` inspects config fields and checkpoint keys
- Agent type: `run_db.algo_to_agent_type()` maps the detected algorithm to the eval wrapper
- Checkpoint prefix: `run_db.algo_to_checkpoint_prefix()` maps to the filename glob pattern
- Game count defaults to 5000 in run-dir mode (100 in file mode)

All flags can be overridden explicitly. For example, `--agent-type sog` forces search-at-eval instead of the default `sog_inference` (CVPN-only).

### File mode (backward compatible)

Passing a .pt file directly still works:

```bash
cambia evaluate runs/rebel-phase1/checkpoints/rebel_checkpoint_iter_500.pt \
  -c config/rebel_train.yaml --agent-type rebel -n 5000
```

When the checkpoint sits inside a `runs/*/checkpoints/` directory with a `config.yaml` in the parent, results are automatically persisted to that run directory. Otherwise, results print to stdout only.

### Eval watcher (continuous polling)

For long training runs, the eval watcher polls for new checkpoints and evaluates them automatically:

```bash
cambia eval-watch runs/sog-phase3-v3/ --agent-type sog_inference --games 5000
```

The watcher writes `metrics.jsonl`, `head_to_head.jsonl`, per-game JSONL in `evaluations/iter_N/`, and SQLite records. It tracks state in `eval_watcher_state.json` to avoid re-evaluating checkpoints.

## 5. Multi-Agent-Type Support

### Algorithm detection and mapping

`run_db.infer_algorithm()` detects the training algorithm from config fields and checkpoint keys. Two mapping tables translate the detected algorithm to eval parameters:

| Algorithm | Agent type | Checkpoint prefix |
|-|-|-|
| os-mccfr | deep_cfr | deep_cfr_checkpoint |
| es-mccfr | deep_cfr | deep_cfr_checkpoint |
| escher | escher | deep_cfr_checkpoint |
| sd-cfr | sd_cfr | deep_cfr_checkpoint |
| rebel | rebel | rebel_checkpoint |
| gtcfr | gtcfr | gtcfr_checkpoint |
| sog | sog_inference | sog_checkpoint |
| psro | deep_cfr | deep_cfr_checkpoint |

Detection priority: sog (requires `sog_metadata` in checkpoint keys or `sog_epochs` in config) comes before gtcfr (requires `cvpn_state_dict` in keys or `gtcfr_epochs` in config), since sog checkpoints also contain `cvpn_state_dict`.

These mappings live in `cfr/src/run_db.py` as `ALGO_TO_AGENT_TYPE` and `ALGO_TO_CHECKPOINT_PREFIX`.

### Agent wrappers

Each agent type has a corresponding wrapper class in `evaluate_agents.py`:

| Agent type | Wrapper class | Notes |
|-|-|-|
| deep_cfr | DeepCFRAgentWrapper | OS-MCCFR, Deep CFR |
| rebel | ReBeLAgentWrapper | PBS subgame solving |
| gtcfr | GTCFRAgentWrapper | CVPN direct inference. `deterministic=True` (argmax), `per_hand_ranges=False` (fast tiled range updates). |
| sog_inference | SoGInferenceAgentWrapper | CVPN-only (inherits GTCFRAgentWrapper). Overrides choose_action for fast tiled range updates. |
| sog | SoGAgentWrapper | Full search at eval via GoEngine FFI. `deterministic=True`. Per-hand-type range updates via `range_utils`. |
| sd_cfr | SDCFRAgentWrapper | Stochastic Discount CFR with snapshot averaging |
| escher | ESCHERAgentWrapper | ESCHER or SD-CFR checkpoint |

## 6. Result Persistence

### Dual-write architecture

All evaluation results are written to two stores:

- `metrics.jsonl` (append-only JSONL in the run directory): human-inspectable, consumed by `plot_metrics.py`
- SQLite `eval_results` table in `cfr/runs/cambia_runs.db`: queryable, consumed by the training dashboard

The shared function `persist_eval_results()` in `evaluate_agents.py` handles both writes. It is called by `cambia evaluate` (run-dir mode) and `eval_watcher.py`. SQLite writes are non-fatal; if the database is unavailable, JSONL still works.

### What gets written where

| Output | Written by | Location |
|-|-|-|
| metrics.jsonl | persist_eval_results | `runs/{name}/metrics.jsonl` |
| eval_results table | persist_eval_results | `runs/cambia_runs.db` |
| head_to_head.jsonl | eval_watcher | `runs/{name}/head_to_head.jsonl` |
| head_to_head table | eval_watcher | `runs/cambia_runs.db` |
| per-game JSONL | eval_watcher (via --output-dir) | `runs/{name}/evaluations/iter_N/{baseline}.jsonl` |
| eval_summary.jsonl | run_db export | `runs/{name}/eval_summary.jsonl` |
| run_meta.json | run_db export | `runs/{name}/run_meta.json` |

### Run database integration

Both `deep_trainer.py` and `sog_trainer.py` register with the SQLite run database at init via `run_db.upsert_run()` and track checkpoints via `run_db.register_checkpoint()`. This enables the training dashboard and `cambia runs list` to report on active and completed runs.

## 7. Parallel Training + Eval (Operational Guide)

Training and evaluation should run in separate processes to avoid CPU contention (which causes 7-10x slowdown in self-play).

### Recommended setup

Terminal 1, training:
```bash
cd cfr
cambia train sog -c config/sog_train.yaml
```

Terminal 2, eval watcher (start after first checkpoint appears):
```bash
cd cfr
cambia eval-watch runs/sog-phase3-v3/ --agent-type sog_inference --games 5000
```

Alternatively, for one-off evaluation of a specific checkpoint:
```bash
cd cfr
cambia evaluate runs/sog-phase3-v3/ --epoch 100
```

### Resource partitioning

| Process | CPU | GPU/XPU | Notes |
|-|-|-|-|
| Training (self-play) | All cores (Go CFR solver) | XPU (training steps only) | Self-play is CPU-bound |
| Eval watcher | All cores (game simulation) | None (CPU inference) | Runs between checkpoints |

Do not run eval concurrently with active self-play iterations. Both saturate CPU; overlapping them degrades both. The eval watcher polls every 30s (configurable via `--poll-interval`) and only evaluates new checkpoints. It naturally fills gaps between self-play rounds.

## 8. Future: RL Best-Response Exploitability

Status: deferred.

Freeze the EMA strategy network and train a PPO or DQN agent to maximally exploit it. The RL agent's win rate against the frozen CFR strategy constitutes an exploitability upper bound.

Deferred because:
- No existing RL infrastructure in the codebase
- Estimated 4-12 hours compute per checkpoint
- Cannot run concurrently with CFR training on Intel Arc A310 (4 GB VRAM)
- mean_imp + H2H provides sufficient signal for current training phases

## 9. Baseline Agent Descriptions

### Included in mean_imp

random_no_cambia: Samples uniformly from all legal actions on each turn except Cambia calls. Provides the random-play lower floor with games of realistic length.

random_late_cambia: Samples uniformly from all legal actions. Calls Cambia when the estimated hand sum falls below a configured threshold, simulating a player who terminates late in a game. Provides a slightly higher floor than random_no_cambia.

imperfect_greedy: Greedy policy operating on belief state rather than ground truth. Chooses the action that minimizes expected hand value under the current belief distribution. Represents a competent heuristic player without memory exploitation.

memory_heuristic: Tracks all observed cards and uses this memory to inform swap and snap decisions. Represents a methodical player who takes advantage of information accumulation. Stronger than imperfect_greedy on games that reach mid-late stages.

aggressive_snap: Prioritizes snapping cards whenever a legal snap is available. Represents an opponent who actively contests the discard pile. Tests the agent's ability to handle snap-race dynamics.

### Context-only

random: True uniform over all legal actions including Cambia. Games frequently end on turn 1 or 2 because the agent randomly calls Cambia before accumulating information. Win rate against this baseline is not a strength metric and is excluded from mean_imp.

greedy: Perfect-information oracle that acts optimally given full knowledge of all card positions. Provides a theoretical ceiling. The trained agent is not expected to exceed 50% win rate against this baseline in the near term.

## 10. Output File Schemas

### metrics.jsonl

One record per (checkpoint, baseline) pair. Written by `persist_eval_results()` as checkpoints are evaluated, whether via `cambia evaluate` or `eval_watcher.py`.

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

Notes:
- ci_low and ci_high are Wilson score interval bounds at 95% confidence (z=1.96)
- adv_loss and strat_loss come from training metadata when available (eval_watcher passes them through); null when running `cambia evaluate` directly
- avg_score_margin is the mean score difference (agent score minus opponent score); negative is better (lower hand = winning)
- t1_cambia_rate is computed for the agent (P0) only, not the baseline

### head_to_head.jsonl

One record per head-to-head matchup evaluation. Written by the eval watcher after each checkpoint evaluation.

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

Notes:
- iter_a is always the later (newer) checkpoint
- iter_b is the earlier comparison checkpoint
- a_win_rate = a_wins / (a_wins + b_wins + ties)
- label is one of: previous (adjacent iter), earliest (first checkpoint in run), t_minus_500 (closest checkpoint ~500 iters back)
- Records are also dual-written to the SQLite run database

### evaluations/iter_N/{baseline}.jsonl

Per-game records for detailed analysis. Written when per-game logging is enabled (via --output-dir or eval_watcher).

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

Notes:
- winner is 0 (agent wins), 1 (baseline wins), or -1 (tie)
- legal_count records the size of the legal action set at the time of the action
- Per-game logs are large; enable only for targeted diagnostic evaluations

## 11. Interpretation Guide

### Convergence patterns

| mean_imp | T1 Cambia Rate | Game Length | H2H vs T-500 | Diagnosis |
|-|-|-|-|-|
| Rising | Dropping | Increasing | > 55% | Healthy convergence |
| Flat | Stable | Stable | ~50% | Plateau: architectural or algorithmic change needed |
| Rising | Flat/Rising | Short | ~50% | RPS cycling: baseline exploitation, not genuine improvement |
| Dropping | Rising | Dropping | < 50% | Regression to degenerate equilibrium |
| Rising then dropping | Rising | Dropping | < 50% | Advantage overfitting: reduce train_steps_per_iteration |

### Advantage overfitting diagnosis

Advantage overfitting manifests as:
1. mean_imp peaks then decays over 100-300 iterations
2. T1 Cambia rate climbs as the agent learns degenerate early-termination
3. Reducing `train_steps_per_iteration` (e.g., 1000 -> 250) eliminates the decay

Root cause: the advantage network memorizes the current buffer rather than learning a generalizable strategy. Adaptive train steps (`target_buffer_passes`) mitigate this by scaling training to buffer size.

### Early stopping criteria

An experiment should be terminated early if:
- T1 Cambia rate exceeds 40% for 3+ consecutive checkpoints
- mean_imp drops below 25% after previously exceeding 30%
- H2H win rate vs anchor drops below 40% (agent is actively regressing)

### Healthy baseline ordering

Expected agent hierarchy from weakest to strongest:
```
random < random_no_cambia ~ random_late_cambia < aggressive_snap < memory_heuristic < imperfect_greedy < trained agent < greedy
```

A well-trained agent should exceed 50% win rate against all mean_imp baselines and approach (but not exceed) 50% against greedy.
