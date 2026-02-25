package engine

import (
	"math/rand/v2"
)

// ---------------------------------------------------------------------------
// Helper: remainingDeck
// ---------------------------------------------------------------------------

// remainingDeck returns a copy of the cards currently in the stockpile.
// These are the cards not yet assigned to any player's hand or on the discard
// pile — the pool from which unknown card slots are sampled.
func (g *GameState) remainingDeck() []Card {
	n := int(g.StockLen)
	out := make([]Card, n)
	copy(out, g.Stockpile[:n])
	return out
}

// ---------------------------------------------------------------------------
// Helper: gameStateHash
// ---------------------------------------------------------------------------

// GameStateHash returns a fast 64-bit hash of the game state for seeding
// Monte Carlo PRNGs deterministically. The same game state always produces
// the same hash value.
func (g *GameState) GameStateHash() uint64 {
	return g.gameStateHash()
}

// gameStateHash is the internal implementation used by both GameStateHash and
// TerminalEvalMC.
func (g *GameState) gameStateHash() uint64 {
	h := uint64(14695981039346656037) // FNV-1a offset basis
	const prime = uint64(1099511628211)

	np := g.Rules.numPlayers()
	for p := uint8(0); p < np; p++ {
		for i := uint8(0); i < g.Players[p].HandLen; i++ {
			h ^= uint64(g.Players[p].Hand[i])
			h *= prime
		}
		h ^= uint64(g.Players[p].HandLen) << 8
		h *= prime
	}
	for i := uint8(0); i < g.StockLen; i++ {
		h ^= uint64(g.Stockpile[i])
		h *= prime
	}
	h ^= uint64(g.DiscardLen) << 16
	h *= prime
	h ^= uint64(g.TurnNumber) << 32
	h *= prime
	h ^= uint64(g.CurrentPlayer) << 48
	h *= prime
	if g.CambiaCaller >= 0 {
		h ^= uint64(g.CambiaCaller+1) << 56
		h *= prime
	}
	return h
}

// ---------------------------------------------------------------------------
// TerminalEvalLinear — O(1) linear evaluation via linearity of expectation
// ---------------------------------------------------------------------------

// TerminalEvalLinear computes expected utility using linearity of expectation.
//
// From the evaluating player's perspective:
//   - Their own hand cards are fully known.
//   - Opponent hand cards are unknown; each slot is expected to equal μ_deck,
//     the mean card value of the remaining stockpile.
//
// expected_score[self] = sum(own hand)
// expected_score[opp]  = HandLen[opp] * μ_deck
//
// For 2P: returns a continuous utility in [-1, 1] proportional to relative
// expected score difference.
// For N>2: u_i = (Σ expected_j - N * expected_i) / (N-1).
//
// Edge case: if the stockpile is empty, μ_deck = 0.
func (g *GameState) TerminalEvalLinear(evaluatingPlayer uint8) float32 {
	n := g.Rules.numPlayers()

	// Compute μ_deck from remaining stockpile.
	var deckSum float32
	for i := uint8(0); i < g.StockLen; i++ {
		deckSum += float32(g.Stockpile[i].Value())
	}
	var muDeck float32
	if g.StockLen > 0 {
		muDeck = deckSum / float32(g.StockLen)
	}

	// Expected score per player.
	var expected [MaxPlayers]float32
	for p := uint8(0); p < n; p++ {
		if p == evaluatingPlayer {
			for i := uint8(0); i < g.Players[p].HandLen; i++ {
				expected[p] += float32(g.Players[p].Hand[i].Value())
			}
		} else {
			expected[p] = float32(g.Players[p].HandLen) * muDeck
		}
	}

	if n <= 2 {
		opp := uint8(1 - evaluatingPlayer)
		// Scores range −6 to 78; normalize difference to [-1, 1].
		diff := expected[opp] - expected[evaluatingPlayer]
		const maxDiff = float32(78)
		if diff > maxDiff {
			diff = maxDiff
		} else if diff < -maxDiff {
			diff = -maxDiff
		}
		return diff / maxDiff
	}

	// N-player pairwise utility.
	var totalExpected float32
	for p := uint8(0); p < n; p++ {
		totalExpected += expected[p]
	}
	raw := totalExpected - float32(n)*expected[evaluatingPlayer]
	return raw / float32(n-1)
}

// ---------------------------------------------------------------------------
// TerminalEvalDP — exact 1D DP convolution for 2P games
// ---------------------------------------------------------------------------

// TerminalEvalDP computes the exact expected utility for 2-player games using
// 1D dynamic programming. It builds a probability mass function over all
// possible opponent score totals (sampling unknown cards from the stockpile
// uniformly with replacement over deck composition), then computes the
// expected win/tie/loss utility.
//
// Self's hand is fully known; opponent's hand is unknown (sampled from deck).
// Only valid when NumPlayers == 2; panics otherwise.
func (g *GameState) TerminalEvalDP(evaluatingPlayer uint8) float32 {
	n := g.Rules.numPlayers()
	if n != 2 {
		panic("TerminalEvalDP: only valid for 2-player games")
	}

	// Self's known score.
	var scoreSelf int
	for i := uint8(0); i < g.Players[evaluatingPlayer].HandLen; i++ {
		scoreSelf += int(g.Players[evaluatingPlayer].Hand[i].Value())
	}

	// Opponent's unknown hand.
	oppIdx := uint8(1 - evaluatingPlayer)
	oppUnknown := int(g.Players[oppIdx].HandLen)

	// Deck card value frequencies.
	deck := g.remainingDeck()
	deckSize := len(deck)

	// If opponent has no unknown cards, evaluate directly.
	if oppUnknown == 0 {
		oppScore := 0 // opponent known score = 0 (no cards tracked)
		if scoreSelf < oppScore {
			return 1.0
		} else if scoreSelf > oppScore {
			return -1.0
		}
		if g.CambiaCaller >= 0 && uint8(g.CambiaCaller) == evaluatingPlayer {
			return 1.0
		}
		if g.CambiaCaller >= 0 {
			return -1.0
		}
		return 0.0
	}

	// Build card value histogram over deck.
	// Card values: -1 to 13.
	const minVal = -1
	const maxVal = 13
	const valRange = maxVal - minVal + 1 // 15

	var hist [valRange]int
	for _, c := range deck {
		v := int(c.Value()) - minVal
		if v >= 0 && v < valRange {
			hist[v]++
		}
	}

	// DP over opponent's unknown slots.
	// pmf[offset] = probability that opp extra sum = offset + (minVal * oppUnknown)
	oppMinSum := minVal * oppUnknown
	oppMaxSum := maxVal * oppUnknown
	pmfSize := oppMaxSum - oppMinSum + 1
	pmf := make([]float64, pmfSize)
	pmf[-oppMinSum] = 1.0 // start: 0 extra

	for draw := 0; draw < oppUnknown; draw++ {
		if deckSize == 0 {
			break
		}
		newPmf := make([]float64, pmfSize)
		for offset, prob := range pmf {
			if prob == 0 {
				continue
			}
			curSum := offset + oppMinSum
			for vi, cnt := range hist {
				if cnt == 0 {
					continue
				}
				cardVal := vi + minVal
				p := float64(cnt) / float64(deckSize)
				newSum := curSum + cardVal
				newOffset := newSum - oppMinSum
				if newOffset >= 0 && newOffset < pmfSize {
					newPmf[newOffset] += prob * p
				}
			}
		}
		pmf = newPmf
	}

	// Compute expected utility.
	var eUtil float64
	for offset, prob := range pmf {
		if prob == 0 {
			continue
		}
		oppScore := offset + oppMinSum
		var u float64
		if scoreSelf < oppScore {
			u = 1.0
		} else if scoreSelf > oppScore {
			u = -1.0
		} else {
			if g.CambiaCaller >= 0 && uint8(g.CambiaCaller) == evaluatingPlayer {
				u = 1.0
			} else if g.CambiaCaller >= 0 {
				u = -1.0
			}
		}
		eUtil += prob * u
	}
	return float32(eUtil)
}

// ---------------------------------------------------------------------------
// TerminalEvalMC — Monte Carlo evaluation for N≥3 games
// ---------------------------------------------------------------------------

// TerminalEvalMC estimates expected utility via Monte Carlo sampling.
//
// For each sample, opponent unknown card slots are filled by sampling from
// the remaining stockpile without replacement using partial Fisher-Yates.
// Exact scores are computed and pairwise utility is accumulated, then averaged.
//
// The caller must supply a *rand.Rand (math/rand/v2) for thread safety and
// determinism. Recommended seeding:
//
//	seed := g.gameStateHash()
//	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeefcafe1234))
//
// numSamples of 50–100 provides good accuracy for N≥3 games.
func (g *GameState) TerminalEvalMC(evaluatingPlayer uint8, numSamples int, rng *rand.Rand) float32 {
	if numSamples <= 0 {
		return 0
	}

	n := g.Rules.numPlayers()

	// Self's known score.
	var knownSelf int
	for i := uint8(0); i < g.Players[evaluatingPlayer].HandLen; i++ {
		knownSelf += int(g.Players[evaluatingPlayer].Hand[i].Value())
	}

	// Count unknown slots per opponent.
	var unknownSlots [MaxPlayers]int
	totalUnknown := 0
	for p := uint8(0); p < n; p++ {
		if p == evaluatingPlayer {
			continue
		}
		unknownSlots[p] = int(g.Players[p].HandLen)
		totalUnknown += unknownSlots[p]
	}

	deck := g.remainingDeck()
	deckSize := len(deck)

	// Working buffer for sampling (reused across samples).
	drawBuf := make([]Card, deckSize)

	var totalUtil float32

	for s := 0; s < numSamples; s++ {
		// Reset draw buffer.
		copy(drawBuf, deck)

		// Partial Fisher-Yates: shuffle only the first totalUnknown positions.
		limit := totalUnknown
		if limit > deckSize {
			limit = deckSize
		}
		for i := 0; i < limit; i++ {
			j := i + int(rng.Uint32N(uint32(deckSize-i)))
			drawBuf[i], drawBuf[j] = drawBuf[j], drawBuf[i]
		}

		// Assign drawn cards to opponent slots.
		var scores [MaxPlayers]int
		scores[evaluatingPlayer] = knownSelf

		drawIdx := 0
		for p := uint8(0); p < n; p++ {
			if p == evaluatingPlayer {
				continue
			}
			for slot := 0; slot < unknownSlots[p] && drawIdx < limit; slot++ {
				scores[p] += int(drawBuf[drawIdx].Value())
				drawIdx++
			}
		}

		// Compute utility for evaluating player.
		var u float32
		if n <= 2 {
			opp := uint8(1 - evaluatingPlayer)
			selfScore := scores[evaluatingPlayer]
			oppScore := scores[opp]
			if selfScore < oppScore {
				u = 1.0
			} else if selfScore > oppScore {
				u = -1.0
			} else {
				if g.CambiaCaller >= 0 && uint8(g.CambiaCaller) == evaluatingPlayer {
					u = 1.0
				} else if g.CambiaCaller >= 0 {
					u = -1.0
				}
			}
		} else {
			// N-player pairwise: u_i = (totalScore - N * score_i) / (N-1).
			var totalScore int
			for p := uint8(0); p < n; p++ {
				totalScore += scores[p]
			}
			raw := float32(totalScore) - float32(n)*float32(scores[evaluatingPlayer])
			u = raw / float32(n-1)
		}
		totalUtil += u
	}

	return totalUtil / float32(numSamples)
}
