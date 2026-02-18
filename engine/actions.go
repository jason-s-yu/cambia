package engine

import "fmt"

// ApplyAction applies an action by index. Returns an error if the action is illegal.
func (g *GameState) ApplyAction(actionIdx uint16) error {
	if g.IsGameOver() {
		return fmt.Errorf("game is already over")
	}

	// Snap move resolution takes priority over snap phase.
	if g.Pending.Type == PendingSnapMove {
		if ownIdx, slotIdx, ok := ActionIsSnapOpponentMove(actionIdx); ok {
			return g.snapOpponentMove(ownIdx, slotIdx)
		}
		return fmt.Errorf("pending snap move: expected SnapOpponentMove action, got %d", actionIdx)
	}

	// Snap phase actions.
	if g.Snap.Active {
		if actionIdx == ActionPassSnap {
			return g.passSnap()
		}
		if targetIdx, ok := ActionIsSnapOwn(actionIdx); ok {
			return g.snapOwn(targetIdx)
		}
		if oppIdx, ok := ActionIsSnapOpponent(actionIdx); ok {
			return g.snapOpponent(oppIdx)
		}
		return fmt.Errorf("snap phase active: unhandled action index %d", actionIdx)
	}

	switch actionIdx {
	case ActionDrawStockpile:
		return g.drawStockpile()
	case ActionDrawDiscard:
		return g.drawDiscard()
	case ActionCallCambia:
		return g.callCambia()
	case ActionDiscardNoAbility:
		return g.discardDrawn()
	case ActionDiscardWithAbility:
		return g.discardWithAbility()
	case ActionKingSwapNo:
		return g.kingSwapDecision(false)
	case ActionKingSwapYes:
		return g.kingSwapDecision(true)
	default:
		if targetIdx, ok := ActionIsReplace(actionIdx); ok {
			return g.replace(targetIdx)
		}
		if targetIdx, ok := ActionIsPeekOwn(actionIdx); ok {
			return g.peekOwn(targetIdx)
		}
		if targetIdx, ok := ActionIsPeekOther(actionIdx); ok {
			return g.peekOther(targetIdx)
		}
		if ownIdx, oppIdx, ok := ActionIsBlindSwap(actionIdx); ok {
			return g.blindSwap(ownIdx, oppIdx)
		}
		if ownIdx, oppIdx, ok := ActionIsKingLook(actionIdx); ok {
			return g.kingLook(ownIdx, oppIdx)
		}
		return fmt.Errorf("unhandled action index %d", actionIdx)
	}
}

// drawStockpile pops the top card from the stockpile and sets it as the pending draw.
func (g *GameState) drawStockpile() error {
	if g.Pending.Type != PendingNone {
		return fmt.Errorf("already have a pending action (type %d)", g.Pending.Type)
	}
	// If stockpile is empty, attempt a reshuffle first.
	if g.StockLen == 0 {
		g.attemptReshuffle()
	}
	if g.StockLen == 0 {
		return fmt.Errorf("stockpile is empty and cannot be reshuffled")
	}

	// Pop top card (last element in stockpile array).
	g.StockLen--
	drawn := g.Stockpile[g.StockLen]

	// Set pending state.
	g.Pending.Type = PendingDiscard
	g.Pending.PlayerID = g.CurrentPlayer
	g.Pending.Data[0] = uint8(drawn)
	g.Pending.Data[1] = DrawnFromStockpile

	// Record in LastAction.
	g.LastAction.ActionIdx = ActionDrawStockpile
	g.LastAction.ActingPlayer = g.CurrentPlayer
	g.LastAction.DrawnFrom = DrawnFromStockpile

	return nil
}

// drawDiscard pops the top card from the discard pile and sets it as the pending draw.
func (g *GameState) drawDiscard() error {
	if g.Pending.Type != PendingNone {
		return fmt.Errorf("already have a pending action (type %d)", g.Pending.Type)
	}
	if !g.Rules.AllowDrawFromDiscard {
		return fmt.Errorf("drawing from discard pile is not allowed by house rules")
	}
	if g.DiscardLen == 0 {
		return fmt.Errorf("discard pile is empty")
	}

	// Pop top card (last element in discard array).
	g.DiscardLen--
	drawn := g.DiscardPile[g.DiscardLen]

	// Set pending state.
	g.Pending.Type = PendingDiscard
	g.Pending.PlayerID = g.CurrentPlayer
	g.Pending.Data[0] = uint8(drawn)
	g.Pending.Data[1] = DrawnFromDiscard

	// Record in LastAction.
	g.LastAction.ActionIdx = ActionDrawDiscard
	g.LastAction.ActingPlayer = g.CurrentPlayer
	g.LastAction.DrawnFrom = DrawnFromDiscard

	return nil
}

// callCambia handles a player calling "Cambia", signalling the final round.
func (g *GameState) callCambia() error {
	if g.Pending.Type != PendingNone {
		return fmt.Errorf("cannot call Cambia with a pending action (type %d)", g.Pending.Type)
	}
	if g.IsCambiaCalled() {
		return fmt.Errorf("Cambia has already been called")
	}
	currentRound := g.TurnNumber / uint16(MaxPlayers)
	if currentRound < uint16(g.Rules.CambiaAllowedRound) {
		return fmt.Errorf("cannot call Cambia before round %d (current round %d)", g.Rules.CambiaAllowedRound, currentRound)
	}

	g.CambiaCaller = int8(g.CurrentPlayer)
	g.Flags |= FlagCambiaCalled

	g.LastAction.ActionIdx = ActionCallCambia
	g.LastAction.ActingPlayer = g.CurrentPlayer

	// Calling Cambia counts as a turn — advance so other player gets final turn.
	g.advanceTurn()
	return nil
}

// discardDrawn discards the drawn card to the discard pile without using an ability.
func (g *GameState) discardDrawn() error {
	if g.Pending.Type == PendingNone {
		return fmt.Errorf("no pending drawn card to discard")
	}
	if g.Pending.Type != PendingDiscard {
		return fmt.Errorf("pending action is not a discard (type %d)", g.Pending.Type)
	}

	drawn := Card(g.Pending.Data[0])

	// Place drawn card on top of discard pile.
	g.DiscardPile[g.DiscardLen] = drawn
	g.DiscardLen++

	// Record in LastAction.
	g.LastAction.ActionIdx = ActionDiscardNoAbility
	g.LastAction.ActingPlayer = g.Pending.PlayerID
	g.LastAction.RevealedCard = drawn

	// Clear pending.
	g.Pending = PendingAction{}

	g.initiateSnapPhase(drawn)
	return nil
}

// replace swaps the drawn card into hand[targetIdx], discarding the old card.
func (g *GameState) replace(targetIdx uint8) error {
	if g.Pending.Type == PendingNone {
		return fmt.Errorf("no pending drawn card to replace with")
	}
	if g.Pending.Type != PendingDiscard {
		return fmt.Errorf("pending action is not a discard (type %d)", g.Pending.Type)
	}

	acting := g.Pending.PlayerID
	handLen := g.Players[acting].HandLen
	if targetIdx >= handLen {
		return fmt.Errorf("replace target index %d out of range (hand size %d)", targetIdx, handLen)
	}

	drawn := Card(g.Pending.Data[0])
	old := g.Players[acting].Hand[targetIdx]

	// Swap drawn card into hand.
	g.Players[acting].Hand[targetIdx] = drawn

	// Discard old card.
	g.DiscardPile[g.DiscardLen] = old
	g.DiscardLen++

	// Record in LastAction.
	g.LastAction.ActionIdx = EncodeReplace(targetIdx)
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = old
	g.LastAction.RevealedIdx = targetIdx
	g.LastAction.RevealedOwner = acting

	// Clear pending.
	g.Pending = PendingAction{}

	g.initiateSnapPhase(old)
	return nil
}

// advanceTurn rotates to the next player, increments turn number, and checks for game end.
func (g *GameState) advanceTurn() {
	if g.IsGameOver() {
		return
	}

	g.TurnNumber++
	g.CurrentPlayer = g.OpponentOf(g.CurrentPlayer)

	if g.IsCambiaCalled() {
		g.TurnsAfterC++
	}

	g.checkGameEnd()
}

// checkGameEnd checks end conditions and sets the GameOver flag if met.
func (g *GameState) checkGameEnd() {
	if g.IsGameOver() {
		return
	}

	// 1. Max turns exceeded.
	if g.Rules.MaxGameTurns > 0 && g.TurnNumber >= uint16(g.Rules.MaxGameTurns) {
		g.Flags |= FlagGameOver
		return
	}

	// 2. Cambia final round completed (all players have had their last turn).
	if g.IsCambiaCalled() && g.TurnsAfterC >= MaxPlayers {
		g.Flags |= FlagGameOver
		return
	}

	// 3. Stalemate: stockpile and discard both empty (no way to draw).
	if g.StockLen == 0 && g.DiscardLen == 0 {
		g.Flags |= FlagGameOver
		return
	}
}

// attemptReshuffle moves all discard cards (except the top) back into the stockpile and shuffles.
func (g *GameState) attemptReshuffle() {
	// Need at least 2 cards in discard (one stays, rest go to stockpile).
	if g.DiscardLen <= 1 {
		return
	}

	// Keep the top discard card in place.
	topCard := g.DiscardPile[g.DiscardLen-1]

	// Move all other discard cards into the stockpile.
	count := g.DiscardLen - 1
	for i := uint8(0); i < count; i++ {
		g.Stockpile[i] = g.DiscardPile[i]
	}
	g.StockLen = count

	// Reset discard pile to just the top card.
	g.DiscardPile[0] = topCard
	g.DiscardLen = 1

	// Shuffle the new stockpile using Fisher-Yates.
	for i := int(g.StockLen) - 1; i > 0; i-- {
		j := int(g.randN(uint64(i + 1)))
		g.Stockpile[i], g.Stockpile[j] = g.Stockpile[j], g.Stockpile[i]
	}
}
