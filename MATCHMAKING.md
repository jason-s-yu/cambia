# CAMBIA MATCHMAKING & RATING SPECIFICATION

## 1. Scope

Ranked Cambia supports two visible formats plus a hidden matchmaking pool:

| Format | Players | Deck | Rating System | Visibility |
| :----- | :------ | :--- | :------------ | :--------- |
| **H2H Quick Play** | 2 | 1 × 54 cards | Glicko-2 | **Hidden** (SBMM only) |
| **H2H Ranked** | 2 | 1 × 54 cards | Glicko-2 | Visible |
| **FFA-4 Ranked** | 4 | 1 × 54 cards | OpenSkill (Plackett-Luce) | Visible |

All three pools use score-differential linear utility for rating updates. Each player maintains three independent ratings: H2H Quick Play (hidden), H2H Ranked, and FFA-4.

**H2H Quick Play** is a single-game (Bo1) queue with hidden MMR used exclusively for skill-based matchmaking. No rank is displayed. At $R = 0.11$ per game (~89% noise), single-game outcomes are too volatile for meaningful visible ratings. This pool may be surfaced as a visible rank in a future season if population health supports it (target: median player has 50+ games/season).

**Unranked/casual modes** may support 3, 5, or 6 players (single deck up to 4; double deck for 5–8) but carry no rating implications. Lobbies can be created and open for public matchmaking as desired for unranked modes.

### 1.1 Deck Scaling (Reference)

| Players | Decks | Total Cards | Residual Pool | Mode |
| :------ | :---- | :---------- | :------------ | :--- |
| 2       | 1     | 54          | 46            | Ranked |
| 3       | 1     | 54          | 42            | Casual only |
| 4       | 1     | 54          | 38            | Ranked |
| 5–8     | 2     | 108         | 88–68         | Casual only |
| 9–12    | 3     | 162         | 126–114       | Party mode |

## 2. All circuit rounds take place in multiples of 4.

All ranked multi-round match lengths must be strict multiples of 4 for the following reasons:

### 2.1 Positional Equity (Combinatorial Argument)

In 4-player Cambia with strict clockwise seat rotation, the position assignment has period 4: after exactly 4 rounds, each player has occupied each seat (1st-actor, 2nd-actor, 3rd-actor, dealer) precisely once. Any round count $n \not\equiv 0 \pmod{4}$ creates an asymmetry where some players hold advantaged positions more often.

In H2H, the seat period is 2 (first-actor / dealer). The least common multiple of the two format periods is $\text{lcm}(2, 4) = 4$.

Multiples of 4 guarantee positional equity for both formats simultaneously, enabling a unified round-count system.

### 2.2 Statistical Replication (Variance Argument)

We model each round's score as a linear combination:

$$S_{i,r} = \mu_i + \pi_{p(i,r)} + \varepsilon_{i,r}$$

where $\mu_i$ is player $i$'s true skill, $\pi_{p(i,r)}$ is the position effect for the seat player $i$ occupies in round $r$, and $\varepsilon_{i,r} \sim \mathcal{N}(0, \sigma^2_\varepsilon)$ is round noise.

With $n = 4k$ total rounds, each player occupies each position exactly $k$ times. The skill estimator (cumulative score, position-balanced) has variance:

$$\text{Var}(\hat{\mu}_i) = \frac{\sigma^2_\varepsilon}{4k}$$

| Cycles $k$ | Rounds | Properties |
| :---------- | :----- | :--------- |
| $k = 1$ | 4 | Positional equity achieved, but **zero within-position replication**. Position effects are perfectly confounded with round-specific noise — you cannot distinguish "position 1 is advantaged" from "round 1 had favorable draws." |
| $k = 2$ | 8 | **Minimum replication.** Each position is observed twice per player. One degree of freedom per position enables partial separation of position effects from round noise. Estimation variance halved vs. $k=1$. |
| $k = 3$ | 12 | Further variance reduction ($1/3$ of $k=1$). Sufficient for robust position-effect estimation. |
| $k = 4$ | 16 | Diminishing marginal returns begin. Each additional cycle buys less than the previous. |

The jump from $k=1$ to $k=2$ (4 → 8 rounds) is the single largest marginal improvement. This makes 8 rounds the minimum statistically robust match length for both formats.

### 2.3 Efficiency Knee Alignment

From the Spearman-Brown reliability analysis, the marginal reliability gain per round:

$$\frac{dR}{dn} = \frac{r(1-r)}{(1+(n-1)r)^2}$$

drops below 0.03 (the practical "not worth another 5–7 minutes" threshold) at:

| Format | $r$ (score-diff) | Knee (rounds) | Nearest $4k$ |
| :----- | :---------------- | :------------ | :------------ |
| H2H    | 0.110             | ~8            | **8**         |
| FFA-4  | 0.080             | ~10–12        | **8 or 12**   |

The efficiency knee naturally aligns with multiples of 4.

### 2.4 Conclusion

Multiples of 4 simultaneously satisfy:

1. Positional equity for both H2H (period 2) and FFA-4 (period 4)
2. Statistical replication for separating position effects from noise
3. Alignment with the efficiency knee of reliability curves

Bo1 (Quick Play) operates in a separate hidden pool because it cannot satisfy these constraints.

## 3. Statistical Framework

### 3.1 Per-Round Reliability

The single-round reliability coefficient $r$ quantifies what fraction of score variance is skill versus luck. Anchored on empirically validated 4-player tournament data ($R(8) = 0.28$, $R(12) = 0.37$, $R(20) = 0.50$), with the Spearman-Brown inversion yielding $r_4^{\text{binary}} = 0.046$.

| Format | $r$ (Binary W/L) | $r$ (Score-Differential) |
| :----- | :---------------- | :----------------------- |
| H2H    | 0.060             | 0.110                    |
| FFA-4  | 0.046             | 0.080                    |

Score-differential updates approximately double effective information per round by encoding margin of victory rather than binary outcome.

### 3.2 Multi-Round Reliability

$$R(n) = \frac{n \cdot r}{1 + (n-1) \cdot r}$$

#### H2H ($r = 0.11$)

| Rounds | $R(n)$ | $\Delta R$ per round | Time (~5 min/rd) |
| :----- | :----- | :------------------- | :---------------- |
| 1      | 0.110  | —                    | ~5 min            |
| 4      | 0.331  | 0.055                | ~20 min           |
| **8**  | **0.497** | **0.042**         | **~40 min**       |
| 12     | 0.597  | 0.025                | ~60 min           |
| 16     | 0.664  | 0.017                | ~80 min           |

#### FFA-4 ($r = 0.08$)

| Rounds | $R(n)$ | $\Delta R$ per round | Time (~7 min/rd) |
| :----- | :----- | :------------------- | :---------------- |
| 4      | 0.250  | 0.063                | ~28 min           |
| **8**  | **0.410** | **0.040**         | **~55 min**       |
| **12** | **0.511** | **0.025**         | **~85 min**       |
| 16     | 0.580  | 0.017                | ~112 min          |
| 20     | 0.632  | 0.013                | ~140 min          |

### 3.3 Win Probability by Skill Gap

For two players with true skill difference $\Delta\mu$, the probability the better player produces the lower score in a single round:

$$P(\text{win}) = \Phi\!\left(\frac{\Delta\mu}{\sqrt{2 \cdot \text{Var}(\varepsilon)}}\right) = \Phi\!\left(\frac{\Delta\mu}{7.87}\right)$$

| Skill Gap | $\Delta\mu$ (pts) | $P$(win single round) | $P$(win Bo8 cumulative) |
| :-------- | :----------------- | :-------------------- | :---------------------- |
| 0.5 SD    | 0.71               | 53.6%                 | 57.8%                   |
| 1.0 SD    | 1.41               | 57.1%                 | 65.2%                   |
| 1.5 SD    | 2.12               | 60.6%                 | 72.0%                   |
| 2.0 SD    | 2.83               | 64.1%                 | 78.1%                   |

### 3.4 Games to Convergence (Score-Differential, 95% Confidence)

| Skill Gap | Quick Play (Bo1) | H2H Blitz (Bo4) | H2H Rapid (Bo8) | FFA-4 Standard (Bo8) |
| :-------- | :---------------- | :--------------- | :--------------- | :-------------------- |
| 0.5 SD    | ~200              | ~140             | ~100             | ~160                  |
| 1.0 SD    | ~50               | ~35              | ~25              | ~40                   |
| 1.5 SD    | ~22               | ~15              | ~11              | ~18                   |
| 2.0 SD    | ~13               | ~9               | ~7               | ~10                   |

## 4. Time Controls

Modeled on chess time controls, each tier trades match duration for measurement reliability.

### 4.1 H2H Quick Play (Hidden Pool)

| Rounds | $R(n)$ | Est. Time | Description |
| :----- | :----- | :-------- | :---------- |
| 1      | 0.11   | ~5 min    | Single game. No positional equity. First-actor random. Hidden MMR for SBMM only. |

No aggression subsidy. No multi-round structure. Pure single-game Cambia.

### 4.2 H2H Ranked (Visible Pool)

| Tier          | Rounds | Cycles ($k$) | $R(n)$ | Est. Time | Description |
| :------------ | :----- | :----------- | :----- | :-------- | :---------- |
| **Blitz**     | 4      | 2            | 0.33   | ~20 min   | Quick competitive match. Minimum positional equity. |
| **Rapid**     | 8      | 4            | 0.50   | ~40 min   | **Primary queue.** Statistical sweet spot at efficiency knee. |
| **Classical** | 16     | 8            | 0.66   | ~80 min   | Highest confidence. Full replication depth. |

### 4.3 FFA-4 (4-Player)

| Tier          | Rounds | Cycles ($k$) | $R(n)$ | Est. Time | Description |
| :------------ | :----- | :----------- | :----- | :-------- | :---------- |
| **Standard**  | 8      | 2            | 0.41   | ~55 min   | **Primary queue.** Minimum robust replication at efficiency knee. |
| **Classical** | 12     | 3            | 0.51   | ~85 min   | Enhanced confidence. Past the knee but justifiable for serious play. |

FFA-4 does not offer Sprint or Blitz: the 4-player positional equity constraint makes Bo1 and Bo4 (only 1 cycle) statistically inadequate for meaningful rating updates in a 4-player context.

### 4.4 Recommended Queues

| Queue ID         | Players | Tier      | Rounds | Rating Pool       | Rank Display |
| :--------------- | :------ | :-------- | :----- | :---------------- | :----------- |
| `h2h_quickplay`  | 2       | Quick Play| 1      | H2H Quick Play    | **Hidden**   |
| `h2h_blitz`      | 2       | Blitz     | 4      | H2H Ranked        | Visible      |
| `h2h_rapid`      | 2       | Rapid     | 8      | H2H Ranked        | Visible      |
| `h2h_classical`  | 2       | Classical | 16     | H2H Ranked        | Visible      |
| `ffa4_standard`  | 4       | Standard  | 8      | FFA-4 Ranked      | Visible      |
| `ffa4_classical` | 4       | Classical | 12     | FFA-4 Ranked      | Visible      |

`h2h_rapid` and `ffa4_standard` are the **primary** queues (default selection in UI). `h2h_blitz` is always available. `h2h_classical` and `ffa4_classical` are on-demand (custom lobbies with ranked tracking).

## 5. House Rules

All ranked queues enforce a fixed rule configuration to address the T1C (Turn-1 Cambia) problem.

### 5.1 The T1C Problem

In 2-player games with `lockCallerHand = true`, calling Cambia on turn 1 with hand sum $\leq$ 4 yields a strictly positive expected win rate against any possible discard state. The Nash-optimal T1C rate under these conditions is **23.95%**. Nearly one in four games is "solved" before meaningful play begins.

### 5.2 Rule Configuration (All Ranked Queues)

| Parameter               | Value     | Reason  |
| :---------------------- | :-------- | :------ |
| `allowDrawFromDiscard`  | **true**  | Enables strategic use of public information. |
| `allowReplaceAbilities` | **true**  | Enables utility storage and complex middle-game decisions. |
| `lockCallerHand`        | **false** | Eliminates T1C degeneracy. Opponents can sabotage caller's hand during final turn(s). |
| `snapRace`              | **true**  | First-to-snap rewards reaction speed. |
| `useJokers`             | **true**  | Full 54-card deck. |

**Lock hand = false** transforms the Cambia call from a declaration of victory into a strategic gamble:

- Opponents' final turns become dual-objective: minimize own score AND maximize the caller's score via targeted swaps
- Calling reveals information about hand strength, which opponents exploit
- Memory skill gains value (knowing opponents' high card locations enables sabotage)
- The T1C threshold shifts from sum ≤ 4 (near-certain win) to a complex function of opponent information state

### 5.3 Aggression Subsidy (Multi-Round Formats Only)

Applied to cumulative totals after each round. Not applicable to Quick Play (Bo1).

| Placement | H2H (2p) | FFA-4 (4p) |
| :-------- | :------- | :--------- |
| 1st       | −3       | −5         |
| 2nd       | 0        | −2         |
| 3rd       | —        | 0          |
| 4th       | —        | 0          |

**H2H subsidy is −3/0** (vs. −5/−2 in FFA-4) because the Cambia caller in H2H faces only one opponent's final turn, making the call inherently less risky. The smaller subsidy maintains incentive without over-rewarding.

Tiebreaker: The Cambia caller wins ties for bonus distribution. If neither tied player called, both receive the higher placement bonus.

## 6. Rating Systems

### 6.1 H2H Quick Play: Glicko-2 (Hidden)

Identical algorithm to H2H Ranked but with no displayed rating. Used solely for skill-based matchmaking in the Bo1 queue.

#### Parameters

Same as H2H Ranked (Section 6.2), with one modification:

| Parameter              | Value | Rationale |
| :--------------------- | :---- | :-------- |
| Rating period          | 24 hours OR 10 games, whichever comes first. | Higher game threshold (vs. 5 for Ranked) because each Bo1 game carries less information. |

#### Surfacing Criteria

The Quick Play pool may be surfaced as a visible rank in a future season if all of the following hold:
- Median active player has 50+ games in the current season
- Queue population sustains < 60 sec average wait times at peak hours
- Internal analysis confirms rating convergence quality (median $\phi < 120$ among 50+ game players)

### 6.2 H2H Ranked Uses Glicko-2

Glicko-2's three-parameter model ($\mu$, $\phi$, $\sigma$) is ideal for Cambia's high-variance environment. The volatility parameter $\sigma$ distinguishes consistent grinders from volatile players — a distinction invisible to two-parameter systems.

#### Parameters

| Parameter              | Value | Rationale |
| :--------------------- | :---- | :-------- |
| Initial $\mu$          | 1500  | Standard baseline. |
| Initial $\phi$ (RD)    | 350   | High initial uncertainty; reflects Cambia's variance. |
| Initial $\sigma$ (vol) | 0.06  | Default Glicko-2 volatility. |
| System constant $\tau$  | 0.5   | Moderate volatility change rate. Lower than chess (0.6) to dampen Cambia's inherent noise. |
| Rating period          | 24 hours OR 5 matches, whichever comes first. |

#### Score-Differential Outcome Mapping

Standard Glicko-2 uses outcomes $s \in \{0, 0.5, 1\}$. We extend to a continuous outcome via logistic mapping of the cumulative score margin:

$$s = \frac{1}{1 + e^{-k \cdot (S_{\text{opp}} - S_{\text{self}})}}$$

where $S$ is the cumulative match score (sum of all rounds in the match, including aggression subsidies).

| Parameter | Value | Effect |
| :-------- | :---- | :----- |
| $k$       | 0.15  | A 10-point cumulative win → $s \approx 0.82$. A 20-point blowout → $s \approx 0.95$. |
| Tie band  | 3 pts | If $\|S_1 - S_2\| \leq 3$, force $s = 0.50$ to filter final-turn deck variance. |

#### Match Aggregation

For multi-round matches (Blitz/Rapid/Classical), the cumulative score across **all** rounds determines a single outcome fed to Glicko-2. One match = one rating update, regardless of round count.

#### Placement & Display

| Condition                  | Display |
| :------------------------- | :------ |
| < 15 matches               | "Placement: X/15 games remaining" |
| 15–30 matches              | Rating with wide confidence band ($\mu \pm 2\phi$) |
| 30+ matches, $\phi < 100$  | Settled rating with narrow band |

Confidence intervals are always visible (e.g., "1520 ± 140").

### 6.3 FFA-4 Uses OpenSkill (Plackett-Luce)

OpenSkill PL handles multiplayer rankings natively, computing updates from a full placement vector rather than pairwise comparisons.

#### Parameters

| Parameter              | Value | Rationale |
| :--------------------- | :---- | :-------- |
| Initial $\mu$          | 25.0  | OpenSkill default scale. |
| Initial $\sigma$       | 8.333 | Default ($\mu / 3$). |
| $\beta$ (scale)        | 8.0   | High variance; reflects Cambia's noise profile. |
| $\tau$ (dynamics)      | 0.0   | **Locked to 0 within a match.** Prevents intra-match rating drift. |
| Score margin threshold | 3 pts | Cumulative scores within 3 points treated as tied. |

#### Update Frequency

One update upon match conclusion, using **final cumulative scores** (including aggression subsidies) to determine the placement ordering. Pairs within 3 cumulative points are recorded as tied; OpenSkill PL handles ties natively, dampening update magnitude.

#### Placement & Display

| Condition                     | Display |
| :---------------------------- | :------ |
| < 10 matches                  | "Placement: X/10 games remaining" |
| 10–25 matches                 | Rating with confidence band ($\mu \pm 2\sigma$) |
| 25+ matches, $\sigma < 3.0$  | Settled rating |

### 6.4 Rating Pool Structure

Three fully independent rating pools. Cross-pool skill transfer is not assumed.

| Pool                | Queues | Display |
| :------------------ | :----- | :------ |
| **H2H Quick Play**  | h2h_quickplay | Hidden (SBMM only) |
| **H2H Ranked**      | h2h_blitz, h2h_rapid, h2h_classical | Visible rank + tiers |
| **FFA-4 Ranked**    | ffa4_standard, ffa4_classical | Visible rank + tiers |

Within the H2H Ranked pool, all time controls feed the same rating. A Blitz result and a Rapid result both update H2H Ranked rating; the Rapid result naturally carries more statistical weight due to its tighter cumulative score distribution.

H2H Quick Play and H2H Ranked are separate pools despite both being 2-player. Bo1 incentivizes meaningfully different play patterns (aggressive snaps, early Cambia calls, high-variance strategies) compared to multi-round play where consistency is rewarded and volatility is punished across rounds. Keeping them separate prevents Bo1 variance from contaminating the Ranked signal, and vice versa.

## 7. Matchmaking

### 7.1 Match Quality Function

**H2H (Quick Play and Ranked):**

$$Q = \exp\!\left(-\frac{(\mu_1 - \mu_2)^2}{2(c^2 + \phi_1^2 + \phi_2^2)}\right)$$

where $c = \sqrt{2} \cdot \beta_{\text{Glicko}}$.

**FFA-4:** OpenSkill's built-in `predict_draw` function across all 4 players.

### 7.2 Queue Degradation

Target match quality degrades over time to prevent indefinite waits:

| Time in Queue | Minimum $Q$ |
| :------------ | :---------- |
| 0–30 sec      | 0.80        |
| 30–60 sec     | 0.70        |
| 60–120 sec    | 0.55        |
| 120+ sec      | 0.40        |

### 7.3 FFA-4 Lobby Formation

1. Sort waiting players by $\mu$.
2. Form candidate lobbies of 4 consecutive players.
3. Select the lobby with highest $Q$ meeting the time-degraded minimum.
4. If no lobby forms within 180 seconds, offer AI backfill (max 1 AI seat, rated $\mu = 15.0$, $\sigma = 4.0$).

### 7.4 Anti-Abuse

| Rule                 | Implementation |
| :------------------- | :------------- |
| Dodge penalty        | 2-min timeout, escalating 2× per dodge within 1 hour. |
| Rating manipulation  | Flag accounts matched against same opponent > 3× in 24 hours with anomalous outcomes. |
| Smurf detection      | Performance above $\mu + 3\phi$ over 10+ matches triggers expedited placement (2× RD reduction). |

## 8. Disconnection & Forfeiture

| Event | H2H | FFA-4 |
| :---- | :--- | :---- |
| Disconnect < 60 sec | AI plays defensively. Score counts normally upon reconnection. | Same. |
| Disconnect > 60 sec | Round forfeited. Player receives **41 points** ($+2\sigma$ blind hand maximum). | Same. |
| Miss 2+ consecutive rounds | — | Tournament abandonment. Remaining rounds scored as 41. 15-min queue lockout. |
| Full match abandonment | Remaining rounds scored as 41. Rating updated normally (massive loss). | Same as above. |

## 9. Turn Order

### 9.1 Quick Play (Bo1)

- First-actor assigned randomly
- No positional equity guarantee

### 9.2 Multi-Round Matches

- Round 1: Dealer chosen randomly; play proceeds clockwise.
- Rounds 2+: Strict clockwise rotation. The dealer and first-actor positions rotate exactly one seat to the left unconditionally. Outcomes do not alter turn order.

This guarantees that after every 4 rounds (H2H: every 2 rounds for the 2-player subset), each player has occupied every position equally.

## 10. Rank Tiers

Percentile-based, recalibrated monthly via rolling 30-day snapshots. H2H quick play will launch with no visible rank: its hidden MMR is used only for matchmaking.

### 10.1 H2H Ranked (Glicko-2)

Initial estimates:

| Tier        | Percentile | Approx. $\mu$ |
| :---------- | :--------- | :------------- |
| Bronze      | 0–25%      | < 1400         |
| Silver      | 25–50%     | 1400–1500      |
| Gold        | 50–75%     | 1500–1600      |
| Platinum    | 75–90%     | 1600–1720      |
| Diamond     | 90–97%     | 1720–1850      |
| Master      | 97–99.5%   | 1850–2000      |
| Grandmaster | 99.5%+     | 2000+          |

### 10.2 FFA-4 (OpenSkill)

Initial estimates:

| Tier        | Percentile | Approx. $\mu$ |
| :---------- | :--------- | :------------- |
| Bronze      | 0–25%      | < 22           |
| Silver      | 25–50%     | 22–25          |
| Gold        | 50–75%     | 25–28          |
| Platinum    | 75–90%     | 28–32          |
| Diamond     | 90–97%     | 32–37          |
| Master      | 97–99.5%   | 37–42          |
| Grandmaster | 99.5%+     | 42+            |

## 11. Ranked Seasons

We plan for seasons to last 3 months.

At reset:

- Soft reset: $\mu_{\text{new}} = 0.75 \cdot \mu_{\text{old}} + 0.25 \cdot \mu_{\text{default}}$
- Uncertainty inflation: $\phi$ increased by 50% (Glicko-2); $\sigma$ increased by 40% (OpenSkill)
- Previous season peak tier preserved as badge
- Leaderboards archived
