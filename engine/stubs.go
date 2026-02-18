package engine

// GetActingPlayer returns the index of the acting player.
// Delegates to ActingPlayer (alias for fuzz test compatibility).
func (g *GameState) GetActingPlayer() uint8 {
	return g.ActingPlayer()
}

// HandLen returns the number of cards in the given player's hand.
func (g *GameState) HandLen(player uint8) uint8 {
	return g.Players[player].HandLen
}
