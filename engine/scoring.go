package engine

import "math"

// ComputeAggressionSubsidy returns per-player score subsidies (negative = bonus)
// based on placement and format. placements[i] is 0-indexed placement for player i.
// cambiaCallerIdx is the player who called Cambia (-1 if none).
// Tie rule: Cambia caller wins ties for bonus distribution; if neither called, both get the higher bonus.
func ComputeAggressionSubsidy(numPlayers int, placements []int, cambiaCallerIdx int) []int {
	subsidies := make([]int, len(placements))

	// Determine subsidy schedule by format.
	var schedule []int
	switch {
	case numPlayers <= 2:
		schedule = []int{-3, 0}
	case numPlayers <= 4:
		schedule = []int{-5, -2, 0, 0}
	default:
		schedule = []int{-5, -2, -1, 0}
	}
	scheduleBonus := func(place int) int {
		if place < len(schedule) {
			return schedule[place]
		}
		return 0
	}

	// Group players by placement to apply tie rule.
	// For each player, find if any other player shares the same placement.
	for i := 0; i < len(placements); i++ {
		place := placements[i]
		myBonus := scheduleBonus(place)

		// Check for ties at this placement.
		tiedWith := []int{}
		for j := 0; j < len(placements); j++ {
			if j != i && placements[j] == place {
				tiedWith = append(tiedWith, j)
			}
		}
		if len(tiedWith) > 0 {
			// Tie rule: Cambia caller wins (gets better/higher-ranked bonus).
			// If current player is Cambia caller, they keep myBonus.
			// If a tied player is Cambia caller, current player may get worse bonus.
			// If neither called Cambia, both get the higher (more negative) bonus.
			callerInTie := false
			for _, j := range tiedWith {
				if j == cambiaCallerIdx {
					callerInTie = true
					break
				}
			}
			if callerInTie && i != cambiaCallerIdx {
				// The other tied player called Cambia; they take the better placement.
				// This player gets the next worse placement bonus.
				myBonus = scheduleBonus(place + 1)
			}
			// If neither called Cambia, both tied players get the higher placement bonus (myBonus unchanged).
		}
		subsidies[i] = myBonus
	}
	return subsidies
}

// ComputeScoreDiffOutcome maps a score-differential to a [0,1] outcome for Glicko-2.
// Returns 1/(1+exp(-k*(score2-score1))); positive diff means player1 wins (lower score is better).
// If abs(score1-score2) <= tieBand, returns 0.5. Diff is clamped to [-30, 30].
func ComputeScoreDiffOutcome(score1, score2 int, k float64, tieBand int) float64 {
	diff := score2 - score1
	if diff < 0 {
		diff = -diff
	}
	if diff <= tieBand {
		return 0.5
	}
	diff = score2 - score1
	const cap = 30
	if diff > cap {
		diff = cap
	} else if diff < -cap {
		diff = -cap
	}
	return 1.0 / (1.0 + math.Exp(-k*float64(diff)))
}

// computeScores returns the sum of card values for each player.
func (g *GameState) computeScores() [MaxPlayers]int16 {
	var scores [MaxPlayers]int16
	n := g.Rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		for i := uint8(0); i < g.Players[p].HandLen; i++ {
			scores[p] += int16(g.Players[p].Hand[i].Value())
		}
	}
	return scores
}

// computeUtilities returns the game outcome as utilities.
//
// For 2P: winner gets +1, loser gets -1. Ties with no Cambia caller → [0,0].
//
// For N-player (N>2): pairwise utility u_i = (Σ scores_j - N*score_i) / (N-1).
// This is zero-sum and reflects how much better/worse each player does vs opponents.
//
// If the game is not over, returns all zeros.
func (g *GameState) computeUtilities() [MaxPlayers]float32 {
	if !g.IsTerminal() {
		return [MaxPlayers]float32{}
	}

	scores := g.computeScores()
	n := g.Rules.numPlayers()
	var utils [MaxPlayers]float32

	if n <= 2 {
		// Preserve exact 2P behavior.
		if scores[0] < scores[1] {
			return [MaxPlayers]float32{1.0, -1.0}
		} else if scores[1] < scores[0] {
			return [MaxPlayers]float32{-1.0, 1.0}
		}
		// Tied scores.
		if g.CambiaCaller >= 0 {
			caller := uint8(g.CambiaCaller)
			utils[0] = -1.0
			utils[1] = -1.0
			utils[caller] = 1.0
			return utils
		}
		// True tie with no Cambia caller.
		return [MaxPlayers]float32{}
	}

	// N-player pairwise utility: u_i = (Σ score_j - N*score_i) / (N-1)
	// Equivalent to (totalScore - N*score_i) / (N-1).
	var totalScore int16
	for p := uint8(0); p < n; p++ {
		totalScore += scores[p]
	}
	for p := uint8(0); p < n; p++ {
		raw := float32(totalScore) - float32(n)*float32(scores[p])
		utils[p] = raw / float32(n-1)
	}
	return utils
}

// GetUtility returns the utility (outcome) for all players.
// Only meaningful when the game is terminal; returns zeros otherwise.
func (g *GameState) GetUtility() [MaxPlayers]float32 {
	return g.computeUtilities()
}
