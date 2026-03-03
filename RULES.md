# CAMBIA (a.k.a. Cabo, Cambio) RULES

**Objective:** End the game with the lowest total score.
**Players:** 2–6
**Deck:** Standard 52-card deck + 2 Jokers.

## 1. Card Values

Minimize your score. **Red Kings** are the best; **Black Kings** are the worst.

| Rank           | Value               |
| :------------- | :------------------ |
| **Red King**   | **-1** (Best Card)  |
| **Joker**      | 0                   |
| **Ace**        | 1                   |
| **2 - 10**     | Face Value          |
| **Jack**       | 11                  |
| **Queen**      | 12                  |
| **Black King** | **13** (Worst Card) |

## 2. The Setup

1. Deal **4 cards** face-down to each player arranged in a square (2x2).
2. Place the remaining deck (Stockpile) in the center. Flip the top card to start the **Discard Pile**.
3. **Memorize:** At the start *only*, players may peek at their **bottom two cards**. After this, you must rely on memory.

## 3. The Turn

Starting with the player to the dealer's left, take **one** of three actions:

### A. Draw from Stockpile

You look at the card and decide to:

1. **Swap:** Exchange it with one of your face-down cards (discard the old card face-up).
2. **Discard immediately:** If you discard it immediately, you may use its **Special Ability** (see Section 4).

### B. Draw from Discard Pile

Take the top visible card and **Swap** it with one of your face-down cards. (You *cannot* use Special Abilities when drawing from the discard pile).

### C. Call "Cambia"

If you believe you have the lowest score, call "Cambia" instead of drawing.

* Your turn ends immediately.
* Every other player gets **one final turn**.
* Your hand is **locked** and cannot be altered by any player, including yourself (snaps, swaps, etc.)
* All cards are revealed.

## 4. Special Abilities

Abilities only trigger when you draw from the **Stockpile** and **discard immediately**.

* **7 or 8 (Peek Own):** Look at one of your own hidden cards.
* **9 or 10 (Peek Other):** Look at one of an opponent's hidden cards.
* **Jack or Queen (Blind Swap):** Swap one of your cards with an opponent's card (without looking at either).
* **King (Look & Swap):** Look at one of your cards and one of an opponent's; you may choose to swap them.

## 5. Snapping (The Speed Rule)

*Any time* a card is discarded (by any player), if you know exactly where a matching card is (in your hand OR an opponent's hand), you may race to "Snap" it onto the discard pile.

* **Snap Own:** If you snap your own card, you now have one fewer card (this is good!).
* **Snap Opponent:** If you snap an opponent's card, you must move one of your cards to fill their empty spot.
* **Penalty:** If you snap incorrectly (the cards don't match), you receive a **2-card penalty** from the deck.

## 6. Winning & Scoring

After a "Cambia" call and the final round:

1. Everyone sums their cards.
2. **Lowest Score Wins.**
3. If there is a tie among multiple players, the winner is the player who called "cambia," assuming they are among the tied winners.

## 7. "House" Rules

Cambia allows for game hosts to customize certain key parameters of the game.

* `allowDrawFromDiscardPile` (true/false): are players allowed to draw the top card from the discard pile (which is face-up and thus public information)?
* `allowReplaceAbilities` (true/false): can players invoke card abilities when they are played from the hand? If true, players can intentionally keep expensive ability cards for information payoffs later in the game.
* `lockCallerHand` (true/false): upon calling "Cambia," is that player's hand "locked" and protected from any action?
* `snapRace` (true/false): is a successful snap given only to the first person who snaps the card?
* `useJokers` (true/false): are jokers included in the deck?

---

# TOURNAMENT CIRCUIT MODE (Ranked Cambia)

A multi-round competitive format where cumulative score determines the overall champion.

## T1. Format

* A fixed lobby of players (4 players strictly recommended for Ranked equity) plays a set number of rounds.
* After each round, every player’s final score is added to their cumulative tournament total.
* The champion is the player with the lowest cumulative score.
* Tie-Breakers:
  1. Lowest Pre-Bonus Raw Score (rewards fundamental play over bonus-farming), then
  2. Head-to-head round placement record, then
  3. Lowest single-round score achieved.
* The following house rules are enforced:
  1. `allowDrawFromDiscardPile = true`
  2. `allowReplaceAbilities = true`
  3. `lockCallerHand = false`
  * This combination of rules allows for more complex middle games, utility storage (players may intentionally keep expensive cards for payoffs later), and bluffing (since your hand is not locked post-"Cambia," players need to alter their strategy)

## T2. Ranked Match Lengths

* To guarantee positional equity, round counts are strict multiples of 4. See MATCHMAKING.md §4 for the full time-control schedule and statistical justification.
  * H2H Blitz (4 Rounds): ~20 min (Reliability: 0.33)
  * H2H Rapid / FFA-4 Standard (8 Rounds): ~40–55 min (Reliability: 0.41–0.50) — **primary queues**
  * H2H Classical (16 Rounds): ~80 min (Reliability: 0.66)
  * FFA-4 Classical (12 Rounds): ~85 min (Reliability: 0.51)

## T3. The Aggression Subsidy (Round Win Bonus)

To mathematically offset the statistical risk of calling Cambia, fixed score reductions are applied immediately after each round to the cumulative totals of the top finishers. Subsidies differ by format (see MATCHMAKING.md §5.3 for rationale):

| Placement | H2H (2p) | FFA-4 (3–4p) | 5+ players (casual) |
| :-------- | :------- | :----------- | :------------------ |
| 1st       | −3       | −5           | −5                  |
| 2nd       | 0        | −2           | −2                  |
| 3rd       | —        | 0            | −1                  |
| 4th+      | —        | 0            | 0                   |

**H2H subsidy is −3/0** (vs. −5/−2 in FFA-4) because the caller faces only one opponent's final turn in H2H, making the call inherently less risky.

**Tie Rule:** The Cambia caller wins ties for bonus distribution. If neither tied player called Cambia, both receive the bonus of the higher placement.

## T4. Turn Order Equity

* Round 1: Dealer chosen randomly; play proceeds clockwise.
* Subsequent rounds: Strict Clockwise Rotation. The First-Actor and Dealer positions rotate exactly one seat to the left unconditionally. Win/loss outcomes do not alter turn order.

## T5. Disconnection & Forfeiture

* 60 second grace period for reconnections. AI takes over defensively if absent.
* If a player misses an entire round, they receive a score of 41 points (the $+2\sigma$ statistical maximum for a blind hand) to punish the player without corrupting lobby MMR.
* Abandonment: Missing 2+ consecutive rounds abandons the tournament. All remaining rounds are scored as 41.

## T6. Rating System (Ranked MMR)

* Rating points are tracked using OpenSkill (Plackett-Luce) with a high-variance scale parameter (β=8.0).
* Update Frequency: Ratings are updated strictly once per tournament upon conclusion, using the Final Cumulative Scores.
* Score Margins: OpenSkill uses a 3-point margin threshold. If a player finishes within 3 cumulative points of another, the algorithm treats the interaction statistically as a Tie to filter out final-turn deck variance. Intra-tournament uncertainty drift (τ) is locked to 0.
