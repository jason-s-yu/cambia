# ISMCTS-BR confirming-column budget memo (cambia-413)

Prep evidence for the pass-2 frozen baseline exploitability table (the pass that
adds the ISMCTS-BR confirming column) and for P3W8 mixture-exploitability parity.
The confirming column is measured by `src.cfr.ismcts_br.ismcts_br` and is bounded
by a single budget knob (`--ismcts-budget`) that sets the estimator's
`num_infosets`, `ismcts_iterations`, and `eval_games` together. Both scripts
(`scripts/prtcfr_baseline_exploitability.py` and
`scripts/prtcfr_mixture_exploitability.py`) now thread this knob identically and
share `DEFAULT_ISMCTS_BUDGET`.

The final budget ruling is the chief's. This memo is the measured evidence and a
recommendation, nothing more. It does not change the pinned constant
(`DEFAULT_ISMCTS_BUDGET = 200`); applying the recommendation means passing
`--ismcts-budget <N>` at pass-2 generation time (and the matching P3W8 run) or
bumping the shared constant at verdict time.

## Method

Host: this machine, CPU only. Env: `/home/jasonyu/.pyenv/versions/cfr/bin/python`,
`PYTHONPATH` pinned to the worktree (import path asserted under the worktree to
avoid the editable-install `.pth` trap). Config: `config/prtcfr_production.yaml`
(the production rule profile the baseline script defaults to). Seed 42.

The probe calls the exact measurement path the baseline table uses:
`measure_ismcts_br_for_agent(agent, config, num_infosets=B, ismcts_iterations=B,
eval_games=B, seed=42)` with the strong-opponent convention
(`DEFAULT_TRAJECTORY_OPPONENT` = ImperfectGreedy at the fixed seat). Baselines are
built the same way the table builds them (`get_agent(name, 0, config)`).

Two baselines were probed: `random_no_cambia` (a random leg) and
`imperfect_greedy` (a strategic leg, and the default binding scope of the G3
ordering verdict). Budgets 50 / 100 / 200 (the requested grid) were extended to
6000 once the low grid showed a degenerate regime for the strategic leg.

System load: 1-min load average sat between 1.4 and 3.2 across the run (32 cores,
co-tenant noise). Values are seed-deterministic and load-independent; only wall
time is load-sensitive. The load-driven timing noise did not corrupt the cost fit
(R^2 = 0.998, below).

## Probe table

Exploitability is the responder-seat gap `BR_value - game_value`, clamped to
`>= 0`. `br` = extracted best-response value, `gv` = target self-play value,
`tree` = distinct responder information sets searched (`len(tree)`), std_err on
the +/-1 utility scale.

`random_no_cambia`:

| budget | wall_s | expl | std_err | br | gv | tree |
|-|-|-|-|-|-|-|
| 50 | 0.221 | 0.040 | 0.126 | -0.760 | -0.800 | 47 |
| 100 | 0.495 | 0.180 | 0.093 | -0.660 | -0.840 | 92 |
| 200 | 1.047 | 0.225 | 0.056 | -0.710 | -0.935 | 178 |
| 500 | 2.273 | 0.234 | 0.038 | -0.678 | -0.912 | 392 |
| 1000 | 4.525 | 0.329 | 0.027 | -0.609 | -0.938 | 717 |
| 2000 | 9.045 | 0.308 | 0.021 | -0.576 | -0.883 | 1391 |
| 4000 | 19.821 | 0.355 | 0.015 | -0.551 | -0.906 | 2735 |
| 6000 | 28.844 | 0.363 | 0.012 | -0.536 | -0.899 | 3998 |

`imperfect_greedy`:

| budget | wall_s | expl | std_err | br | gv | tree |
|-|-|-|-|-|-|-|
| 50 | 0.204 | 0.000 | 0.169 | -0.760 | 0.160 | 47 |
| 100 | 0.378 | 0.000 | 0.125 | -0.660 | 0.110 | 92 |
| 200 | 0.763 | 0.000 | 0.086 | -0.710 | 0.075 | 178 |
| 500 | 2.504 | 0.000 | 0.056 | -0.678 | -0.010 | 392 |
| 1000 | 4.462 | 0.000 | 0.040 | -0.609 | -0.012 | 717 |
| 2000 | 9.021 | 0.000 | 0.029 | -0.576 | -0.024 | 1391 |
| 4000 | 16.616 | 0.000 | 0.021 | -0.551 | -0.039 | 2735 |
| 6000 | 27.327 | 0.000 | 0.017 | -0.536 | -0.011 | 3998 |

Note that `br` and `tree` are identical across the two baselines at each budget.
This is structurally correct: the search phase builds the tree against the fixed
opponent and the responder's own UCB actions, and does not use the target policy
at all (the target enters only in the `game_value` self-play estimate). So the
extracted best-response value is target-independent; the whole exploitability
difference between baselines rides in `game_value`.

## Cost model and extrapolation

Log-log least-squares fit `wall = a * budget^p` over budgets 50 to 6000:

- `random_no_cambia`: `wall = 0.00469 * budget^1.0015` (R^2 = 0.9988)
- `imperfect_greedy`: `wall = 0.00356 * budget^1.0287` (R^2 = 0.9979)

The scaling is linear (p between 1.00 and 1.03), not superlinear. The reason:
ISMCTS-BR builds the search tree once (`iters` = budget simulations), then the
eval phase plays `2 * budget` games greedily off the frozen tree. There are no
per-move simulations in the eval phase, so total work is `O(budget)` in both
phases. Per-baseline cost is roughly 4 to 5 ms per budget unit; the ~30 percent
spread between the two baselines comes from the `game_value` playouts (random play
runs longer, differently-terminating games than the heuristic).

Extrapolated wall time for the pass-2 table (per baseline, seconds). The three
unmeasured baselines map onto the two measured cost curves: `random_late_cambia`
onto the random curve, `memory_heuristic` and `aggressive_snap` onto the
imperfect_greedy curve (all cheap heuristics; the search cost is shared and
target-independent, so this mapping is tight).

| baseline | b=200 | b=500 | b=1000 | b=2000 |
|-|-|-|-|-|
| random_no_cambia | 0.9 | 2.4 | 4.7 | 9.5 |
| random_late_cambia | 0.9 | 2.4 | 4.7 | 9.5 |
| imperfect_greedy | 0.8 | 2.1 | 4.3 | 8.8 |
| memory_heuristic | 0.8 | 2.1 | 4.3 | 8.8 |
| aggressive_snap | 0.8 | 2.1 | 4.3 | 8.8 |
| 5-baseline total (s) | 4.4 | 11.1 | 22.5 | 45.5 |
| 5-baseline total (min) | 0.07 | 0.19 | 0.37 | 0.76 |

Cost is a non-constraint. Even at budget 2000 the confirming column adds under one
minute to the ~73-minute Tier-B LBR table (pass 1, 2000 infosets). Budgets 4000
and 6000 would add roughly 1.5 and 2.4 minutes: still negligible against the
binding Tier-B cost.

## Convergence assessment

The two legs behave very differently, and both are under-converged on the full
game in the strict sense (the extracted `br` value rises monotonically with budget
and never plateaus: -0.760 at 50 sims up to -0.536 at 6000 sims, still climbing).
Tree coverage is the cause: `tree` grows near-linearly with budget (3998 distinct
information sets at 6000 sims, roughly `0.67 * budget`, no saturation), so the
greedy best-response policy is uniform-random on almost every unseen set and the
search never approaches the true best response.

Contrast with the P3W4 tiny-game calibration ({A,6}, `tests/test_ismcts_br.py`):
there the exact perfect-recall BR gap is ~0.76, and at 6000 sims / 4000 eval games
the estimate lands within `CALIB_TOL = 0.08` (worst |error| 0.050); at 300 sims
the |error| is ~0.4. The tiny game converges by a few thousand sims because its
whole reachable information-set space is small (8 deals). The full 54-card game
does not converge at the same sim count because its space is astronomically
larger. Same sim budget, opposite outcome.

The consequence differs by leg:

- Random legs are usable despite non-convergence. `random_no_cambia` climbs from
  0.040 (b50) to a rough 0.31 to 0.36 plateau by b1000 and above, with std_err
  falling from 0.126 to 0.021 (b2000) to 0.012 (b6000). The value sits well above
  the leg's Tier-B LBR reference (0.0805), which is the expected direction
  (ISMCTS-BR is a tighter, higher lower-bound than one-ply LBR). The ordering is
  stable and CI-tight by b1000 to b2000. The current default of 200 is too low
  for this leg: at b200 the value is still climbing steeply (0.225) with a wide
  std_err (0.056).

- Strategic legs are a hard floor artifact at every budget. `imperfect_greedy`
  reports exploitability 0.000 at all budgets from 50 to 6000, because the
  sparse-tree best response (`br` ~ -0.54 even at 6000 sims) plays worse than the
  heuristic itself (`game_value` ~ 0.0 against the strong opponent), so
  `BR_value - game_value` is negative and clamps to zero. This is not the
  0.4-error regime; it is a budget-independent zero floor. The confirming column
  cannot reveal the strategic legs' exploitability on the full game (Tier-B LBR
  puts imperfect_greedy at 0.43; ISMCTS-BR reports 0.0). More budget does not fix
  this: `br` gains only ~0.22 utility over a 120x budget increase and would need
  to cross roughly +0.05 to make the gap positive.

This strategic-leg degeneracy is the load-bearing finding. It is a property of
the full-game information-set space versus a tractable search budget, not a knob
the pass-2 table can turn.

## Recommended budget

Recommend `--ismcts-budget 2000` for the pass-2 frozen table and the matching
P3W8 mixture measurement, so both scripts share one convention at verdict time.

Reasoning:

- Cost is a non-constraint (linear; ~0.76 min added to the ~73-min table at
  b2000), so the budget is chosen for convergence quality, not runtime.
- For the legs where the confirming column is informative (the random legs), the
  estimate is CI-tight and clearly ordered above the Tier-B LBR reference by
  b1000 to b2000. b2000 gives std_err ~0.021 on the random legs.
- 2000 matches the Tier-B `num_infosets = 2000` for a clean shared mental model
  (both estimators at 2000), which is worth having when two scripts must agree at
  verdict time.
- The current pinned default (200) is too low even for the random legs; the
  recommendation raises it.

Two hard caveats the chief should carry into the ruling:

1. Keep ISMCTS-BR non-binding. Do not pass `--ismcts-binding` on the strategic
   legs (the default G3 binding scope). The strategic-leg confirming value is a
   0.0 floor artifact at every budget, so a binding strategic confirming
   comparison would fail closed or mislead. The verdict script already treats
   ISMCTS-BR as confirming-only by default; that default should stand. The
   confirming column's real jobs are (a) the random legs and (b) an independent
   sanity read on the mixture's own value, not confirming the strategic ordering.

2. Budget does not rescue the strategic legs. If the chief wants tighter random-leg
   CIs, b4000 to b6000 costs only 1.5 to 2.4 extra minutes and drops std_err to
   ~0.015 to 0.012, but the point value is still drifting upward (not converged)
   and the strategic legs stay at the 0.0 floor. If the chief wants to trim,
   b1000 halves the (already trivial) cost with random-leg std_err ~0.027. None
   of these change the strategic-leg conclusion. b2000 is the sane default among
   them.

## Reproduction

```
cd cfr
env PYTHONPATH=$PWD /home/jasonyu/.pyenv/versions/cfr/bin/python - <<'PY'
from src.config import load_config
from src.evaluate_agents import get_agent
from scripts.prtcfr_baseline_exploitability import (
    _try_import_ismcts_br, measure_ismcts_br_for_agent)
cfg = load_config("config/prtcfr_production.yaml")
fn = _try_import_ismcts_br()
for base in ("random_no_cambia", "imperfect_greedy"):
    for b in (50, 100, 200, 500, 1000, 2000):
        r = measure_ismcts_br_for_agent(
            fn, get_agent(base, 0, cfg), cfg,
            num_infosets=b, seed=42, ismcts_iterations=b, eval_games=b)
        print(base, b, round(r["exploitability"], 4), round(r["std_err"], 4))
PY
```
