package engine

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
