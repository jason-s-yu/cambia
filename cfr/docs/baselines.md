# Baseline Agents

Reference for all baseline agents used in evaluation. Each agent is implemented in `src/agents/baseline_agents.py` and registered in `AGENT_REGISTRY` (`src/evaluate_agents.py`).

## Overview

Baselines fall into three categories:

1. **Random baselines.** Uniform random action selection with different Cambia-calling behavior. Provide the random-play floor.
2. **Imperfect-information heuristics.** Rule-based agents that track observed cards and make greedy decisions from belief state. Represent competent but non-optimal human-like play.
3. **Perfect-information oracle.** Has full access to all card positions. Theoretical ceiling reference only.

All imperfect-information agents share the `ImperfectMemoryMixin`, which maintains a memory model: a dict mapping hand slot indices to observed card values (or `None` for unknown slots). Unknown cards are estimated at `UNKNOWN_CARD_EXPECTED_VALUE = 6.5` (the expected value of a random card in the standard 54-card deck). Memory is initialized from the initial peek (bottom 2 cards by default) and updated through peek abilities, replacements, and swaps.

## Random Baselines

### `random`

True uniform random over all legal actions, including Cambia. Because Cambia is a legal action on most turns, the agent frequently calls Cambia on turn 1 or 2, ending games before any meaningful play occurs. Win rate against this baseline is essentially a coin flip and is not informative as a strength metric.

Registry key: `random`. Class: `RandomAgent`.

**Excluded from mean_imp.** Evaluated and logged as a context-only baseline.

### `random_no_cambia`

Uniform random over all legal actions except Cambia. The agent never calls Cambia, so games run their full length. This provides the clean random-play floor: an agent with zero skill that does not artificially shorten games.

Against this baseline, a well-trained agent should exceed 50% WR. The fact that current training runs achieve only ~50-51% against random_no_cambia is itself a diagnostic signal (the trained agent is barely better than random at non-Cambia play).

Registry key: `random_no_cambia`. Class: `RandomNoCambiaAgent` (extends `RandomAgent`, filters out `ActionCallCambia` from legal actions).

**Included in mean_imp.**

### `random_late_cambia`

Uniform random over all legal actions, with Cambia calls suppressed until turn 8 (configurable via `n_turns`). After turn 8, the agent reverts to full uniform random (including Cambia). This simulates a player who plays randomly but only terminates in the mid-to-late game.

Produces slightly higher WR for the CFR agent than `random_no_cambia` because the late random Cambia calls sometimes terminate games at disadvantageous times for the baseline.

Registry key: `random_late_cambia`. Class: `RandomLateCambiaAgent` (extends `RandomAgent`, suppresses Cambia before turn threshold).

**Included in mean_imp.**

### Why three random baselines?

The `random` baseline is uninformative because games end on turn 1-2 (random Cambia calls). `random_no_cambia` provides the true random-play floor with realistic game length. `random_late_cambia` bridges the gap: random play with realistic game termination behavior. Together, the two non-degenerate random baselines (no_cambia and late_cambia) establish the floor that any trained agent should comfortably exceed.

## Imperfect-Information Heuristics

All three agents below use `ImperfectMemoryMixin` for card tracking and share several behaviors:

- Initial knowledge comes from the peek phase (bottom 2 cards).
- Unknown own cards are valued at 6.5 (deck expected value).
- Cambia is called when: 3+ own cards are known AND estimated hand total is at or below `cambia_threshold + 4`. Fallback: always call Cambia after turn 20.
- Snap: only snap own cards that are known to match the current discard. Never snap blind.

The `cambia_threshold` is shared across all three agents and defaults to 5 (configurable via `agents.greedy_agent.cambia_call_threshold` in config).

### `imperfect_greedy`

Greedy policy operating on belief state. Chooses the action that minimizes expected hand value under the current belief distribution.

Decision priorities (in order):
1. Handle ability phases (peek own unknowns, peek opponent, swap).
2. Handle snap-move phases.
3. Snap own cards if known to match discard; otherwise pass.
4. Call Cambia if conditions met (see above).
5. Post-draw: replace the highest-known own card if the drawn card is lower. If the drawn card is low (at or below threshold), replace an unknown slot instead. Otherwise, discard (using ability if the card has one).
6. Draw from stockpile.

This is the strongest imperfect-information heuristic. It draws only from stockpile (never discard pile), prioritizes information gain through abilities, and makes conservative Cambia calls.

Registry key: `imperfect_greedy`. Class: `ImperfectGreedyAgent`.

**Included in mean_imp.**

### `memory_heuristic`

Human-like player that tracks observed cards and uses memory to inform decisions. Very similar to `imperfect_greedy` with minor behavioral differences:

- Always draws from stockpile (never the discard pile).
- Replaces highest known card if drawn card is lower. If drawn card value is 3 or less, replaces any unknown slot.
- Uses abilities for information gathering: peeks own unknowns first, then opponent cards for snap setup. Blind-swaps own highest known with opponent unknown.
- Calls Cambia under the same conditions as `imperfect_greedy`.

The practical difference from `imperfect_greedy` is in the replacement threshold for unknown slots (value <= 3 vs value <= threshold) and some ability-phase ordering details. In evaluation, `memory_heuristic` and `imperfect_greedy` perform similarly, with `imperfect_greedy` slightly stronger.

Registry key: `memory_heuristic`. Class: `MemoryHeuristicAgent`.

**Included in mean_imp.**

### `aggressive_snap`

High-risk card elimination agent. Unlike the other heuristics, this agent actively seeks snap opportunities and uses abilities offensively.

Decision priorities:
1. Aggressive ability handling: peeks opponent cards first (for snap setup), then own unknowns. Uses King look-and-swap to offload high cards.
2. Snaps both own AND opponent cards when confident of a match. For opponent snaps, gives away the lowest-value own card.
3. Calls Cambia aggressively: when hand size drops to 2 or fewer cards, or when estimated total is 4 or less.
4. Post-draw: same replacement logic as `memory_heuristic`.
5. Always draws from stockpile.

This agent tests the CFR agent's ability to handle snap-race dynamics and aggressive discard-pile contention. It has the most volatile game outcomes of any baseline due to the high-risk snap strategy.

Registry key: `aggressive_snap`. Class: `AggressiveSnapAgent`.

**Included in mean_imp.**

## Perfect-Information Oracle

### `greedy`

Perfect-information greedy agent with direct access to `game_state` (sees all cards, including opponent hand and the stockpile). Makes decisions using true card values rather than belief estimates.

Decision priorities:
1. Call Cambia if hand value is at or below `cambia_threshold` (default 5).
2. Snap when the agent knows a match exists (which it always does, having perfect information).
3. Post-draw: replace the highest-value card in hand if the drawn card is lower.
4. Use abilities for further optimization.
5. Draw from stockpile or discard pile (whichever offers a lower-value card).

This agent is the theoretical upper bound. A trained CFR agent operating under imperfect information is not expected to exceed 50% WR against `greedy`. Current best results are ~24% WR against `greedy`.

Registry key: `greedy`. Class: `GreedyAgent`.

**Excluded from mean_imp.** Evaluated and logged as a context-only baseline.

## Baseline Strength Hierarchy

Observed ordering from weakest to strongest, based on head-to-head evaluation:

```
random < random_no_cambia ≈ random_late_cambia < aggressive_snap < memory_heuristic < imperfect_greedy < greedy
```

All five mean_imp baselines are mutually close in strength. Head-to-head between the three heuristic agents typically falls in the 48-52% range (near indistinguishable). The random-floor baselines are clearly weaker but not by a large margin.

## mean_imp Composition

The primary evaluation metric, mean_imp, is the mean win rate across the 5 baselines marked "Included" above:

| Baseline | Category | Role in mean_imp |
|-|-|-|
| `random_no_cambia` | Random floor | Included |
| `random_late_cambia` | Random floor | Included |
| `imperfect_greedy` | Heuristic | Included |
| `memory_heuristic` | Heuristic | Included |
| `aggressive_snap` | Heuristic | Included |
| `random` | Degenerate random | Excluded (games too short) |
| `greedy` | Perfect-info oracle | Excluded (theoretical ceiling) |

See `eval_protocol.md` for the full evaluation protocol and `definitions.md` for metric definitions.
