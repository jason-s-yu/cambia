package engine

// computeScores returns the sum of card values for each player.
func (g *GameState) computeScores() [2]int16 {
	var scores [2]int16
	for p := uint8(0); p < 2; p++ {
		for i := uint8(0); i < g.Players[p].HandLen; i++ {
			scores[p] += int16(g.Players[p].Hand[i].Value())
		}
	}
	return scores
}

// computeUtilities returns the game outcome as utilities in [-1, +1].
// Winner gets +1, loser gets -1. Ties with no Cambia caller result in [0, 0].
//
// Scoring rules:
//   - If a player has a strictly lower score → that player wins (+1, other -1).
//   - If scores are tied:
//     - If Cambia caller is one of the tied players → caller wins (+1, opponent -1).
//     - If no Cambia caller (or caller isn't tied) → true tie, both 0.
//
// If the game is not over, returns [0, 0].
func (g *GameState) computeUtilities() [2]float32 {
	if !g.IsTerminal() {
		return [2]float32{0, 0}
	}

	scores := g.computeScores()

	if scores[0] < scores[1] {
		// Player 0 has lower score → player 0 wins
		return [2]float32{1.0, -1.0}
	} else if scores[1] < scores[0] {
		// Player 1 has lower score → player 1 wins
		return [2]float32{-1.0, 1.0}
	}

	// Scores are tied
	if g.CambiaCaller >= 0 {
		// Cambia caller wins the tie
		caller := uint8(g.CambiaCaller)
		var u [2]float32
		u[0] = -1.0
		u[1] = -1.0
		u[caller] = 1.0
		return u
	}

	// True tie with no Cambia caller
	return [2]float32{0, 0}
}

// GetUtility returns the utility (outcome) for all players.
// Only meaningful when the game is terminal; returns [0, 0] otherwise.
func (g *GameState) GetUtility() [MaxPlayers]float32 {
	return g.computeUtilities()
}
