package engine

import "fmt"

// initiateSnapPhase starts the snap phase after a card is discarded.
// It builds the snapper list, sets SnapState fields, and either begins
// the phase or falls through to advanceTurn if no eligible snappers exist.
func (g *GameState) initiateSnapPhase(discardedCard Card) {
	g.Snap.DiscardedRank = discardedCard.Rank()

	// The acting player just discarded — they are the "discarder".
	// Non-acting player gets first snap opportunity, then the acting player.
	discarder := g.CurrentPlayer
	nonDiscarder := g.OpponentOf(discarder)

	count := uint8(0)
	var snappers [2]uint8

	// Discarder first, then non-discarder — skip anyone who called Cambia,
	// and only include players who actually have a matching-rank card (or whose
	// opponent does, when AllowOpponentSnapping is on).
	// NOTE: Per RULES.md the non-discarder should go first, but Python's
	// _initiate_snap_phase computes discarder_player incorrectly (off by one),
	// resulting in the actual discarder going first. We mirror that behavior.
	for _, p := range [2]uint8{discarder, nonDiscarder} {
		if int8(p) == g.CambiaCaller {
			continue
		}
		// Check if this player can snap own card.
		canSnapOwn := false
		hand := &g.Players[p]
		for i := uint8(0); i < hand.HandLen; i++ {
			if hand.Hand[i].Rank() == discardedCard.Rank() {
				canSnapOwn = true
				break
			}
		}
		// Check if this player can snap opponent's card.
		canSnapOpp := false
		if g.Rules.AllowOpponentSnapping && hand.HandLen > 0 {
			opp := g.OpponentOf(p)
			if int8(opp) != g.CambiaCaller {
				oppHand := &g.Players[opp]
				for i := uint8(0); i < oppHand.HandLen; i++ {
					if oppHand.Hand[i].Rank() == discardedCard.Rank() {
						canSnapOpp = true
						break
					}
				}
			}
		}
		if canSnapOwn || canSnapOpp {
			snappers[count] = p
			count++
		}
	}

	if count == 0 {
		// No eligible snappers — skip snap phase entirely.
		g.advanceTurn()
		return
	}

	g.Snap.Active = true
	g.Snap.Snappers = snappers
	g.Snap.NumSnappers = count
	g.Snap.CurrentSnapperIdx = 0
}

// endSnapPhase clears snap state and advances the main game turn.
func (g *GameState) endSnapPhase() {
	g.Snap = SnapState{}
	g.advanceTurn()
}

// passSnap passes the current snapper's opportunity and advances to the next.
func (g *GameState) passSnap() error {
	if !g.Snap.Active {
		return fmt.Errorf("snap phase is not active")
	}
	g.LastAction.ActionIdx = ActionPassSnap
	g.LastAction.ActingPlayer = g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	g.advanceSnapper()
	return nil
}

// snapOwn attempts to snap the snapper's own card at the given hand index.
// On success, the card is removed from hand and placed on the discard pile.
// On failure, the snapper draws PenaltyDrawCount cards.
func (g *GameState) snapOwn(idx uint8) error {
	if !g.Snap.Active {
		return fmt.Errorf("snap phase is not active")
	}

	snapperIdx := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	handLen := g.Players[snapperIdx].HandLen

	g.LastAction.ActionIdx = EncodeSnapOwn(idx)
	g.LastAction.ActingPlayer = snapperIdx

	if idx >= handLen {
		// Invalid index — treat as failed snap.
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
		return nil
	}

	card := g.Players[snapperIdx].Hand[idx]
	if card.Rank() == g.Snap.DiscardedRank {
		// SUCCESS: remove card from hand, add to discard pile.
		g.removeCardFromHand(snapperIdx, idx)
		g.DiscardPile[g.DiscardLen] = card
		g.DiscardLen++

		g.LastAction.SnapSuccess = true
		g.LastAction.RevealedCard = card
		g.LastAction.RevealedIdx = idx
		g.LastAction.RevealedOwner = snapperIdx

		// SnapRace: end snap phase immediately on first success.
		if g.Rules.SnapRace {
			g.endSnapPhase()
			return nil
		}
	} else {
		// FAIL: draw penalty cards.
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
	}

	g.advanceSnapper()
	return nil
}

// snapOpponent attempts to snap the opponent's card at oppIdx.
// Requires AllowOpponentSnapping house rule to be enabled.
// On success, sets PendingSnapMove so the snapper must move one of their cards.
// On failure, the snapper draws penalty cards.
func (g *GameState) snapOpponent(oppIdx uint8) error {
	if !g.Snap.Active {
		return fmt.Errorf("snap phase is not active")
	}
	if !g.Rules.AllowOpponentSnapping {
		return fmt.Errorf("opponent snapping is not allowed by house rules")
	}

	snapperIdx := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	opponent := g.OpponentOf(snapperIdx)
	oppHandLen := g.Players[opponent].HandLen

	g.LastAction.ActionIdx = EncodeSnapOpponent(oppIdx)
	g.LastAction.ActingPlayer = snapperIdx

	// Snapper must have at least one card to move to the opponent's slot.
	if g.Players[snapperIdx].HandLen == 0 {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
		return nil
	}

	if oppIdx >= oppHandLen {
		// Invalid index — treat as failed snap.
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
		return nil
	}

	card := g.Players[opponent].Hand[oppIdx]
	if card.Rank() == g.Snap.DiscardedRank {
		// SUCCESS: remove card from opponent's hand, put on discard pile.
		g.removeCardFromHand(opponent, oppIdx)
		g.DiscardPile[g.DiscardLen] = card
		g.DiscardLen++

		g.LastAction.SnapSuccess = true
		g.LastAction.RevealedCard = card
		g.LastAction.RevealedIdx = oppIdx
		g.LastAction.RevealedOwner = opponent

		// Set pending move: snapper must now move one of their cards to fill the vacated slot.
		g.Pending.Type = PendingSnapMove
		g.Pending.PlayerID = snapperIdx
		g.Pending.Data[0] = opponent      // which opponent's hand to place card in
		g.Pending.Data[1] = oppIdx        // the vacated slot index (cards shifted, this is now end)

		// Pause the snap phase (Pending takes priority, snap phase remains active but paused).
		// The snap phase Active flag stays true; advanceSnapper is called after the move.
	} else {
		// FAIL: draw penalty cards.
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
	}

	return nil
}

// snapOpponentMove moves the snapper's card at ownIdx into the opponent's hand at slotIdx,
// completing a successful snapOpponent action.
func (g *GameState) snapOpponentMove(ownIdx, slotIdx uint8) error {
	if g.Pending.Type != PendingSnapMove {
		return fmt.Errorf("no pending snap move action")
	}

	snapperIdx := g.Pending.PlayerID
	opponent := g.Pending.Data[0]
	oppHandLen := g.Players[opponent].HandLen

	g.LastAction.ActionIdx = EncodeSnapOpponentMove(ownIdx, slotIdx)
	g.LastAction.ActingPlayer = snapperIdx

	snapperHandLen := g.Players[snapperIdx].HandLen
	if ownIdx >= snapperHandLen {
		return fmt.Errorf("snapOpponentMove: own index %d out of range (hand size %d)", ownIdx, snapperHandLen)
	}
	// slotIdx can be 0..oppHandLen (append to end if == oppHandLen).
	if slotIdx > oppHandLen {
		return fmt.Errorf("snapOpponentMove: slot index %d out of range (opp hand size %d)", slotIdx, oppHandLen)
	}
	if oppHandLen >= MaxHandSize {
		return fmt.Errorf("snapOpponentMove: opponent hand is full (%d)", oppHandLen)
	}

	// Remove card from snapper's hand.
	card := g.removeCardFromHand(snapperIdx, ownIdx)

	// Insert card into opponent's hand at slotIdx, shifting cards right.
	// Shift cards from slotIdx to oppHandLen one position right.
	for i := oppHandLen; i > slotIdx; i-- {
		g.Players[opponent].Hand[i] = g.Players[opponent].Hand[i-1]
	}
	g.Players[opponent].Hand[slotIdx] = card
	g.Players[opponent].HandLen++

	// Clear pending.
	g.Pending = PendingAction{}

	// SnapRace: end snap phase immediately after successful opponent move.
	if g.Rules.SnapRace {
		g.endSnapPhase()
		return nil
	}

	g.advanceSnapper()
	return nil
}

// removeCardFromHand removes the card at cardIdx from a player's hand,
// shifting remaining cards left to fill the gap, and decrements HandLen.
// Returns the removed card.
func (g *GameState) removeCardFromHand(playerIdx, cardIdx uint8) Card {
	handLen := g.Players[playerIdx].HandLen
	card := g.Players[playerIdx].Hand[cardIdx]

	// Shift cards left.
	for i := cardIdx; i < handLen-1; i++ {
		g.Players[playerIdx].Hand[i] = g.Players[playerIdx].Hand[i+1]
	}
	// Clear the last slot.
	g.Players[playerIdx].Hand[handLen-1] = EmptyCard
	g.Players[playerIdx].HandLen--

	return card
}

// drawPenalty draws PenaltyDrawCount cards from the stockpile into the player's hand.
// If the stockpile is empty, it attempts a reshuffle first.
// Cards are capped at MaxHandSize.
func (g *GameState) drawPenalty(playerIdx uint8) {
	count := g.Rules.PenaltyDrawCount
	for i := uint8(0); i < count; i++ {
		if g.Players[playerIdx].HandLen >= MaxHandSize {
			break
		}
		if g.StockLen == 0 {
			g.attemptReshuffle()
		}
		if g.StockLen == 0 {
			break // No cards left at all.
		}
		g.StockLen--
		card := g.Stockpile[g.StockLen]
		handLen := g.Players[playerIdx].HandLen
		g.Players[playerIdx].Hand[handLen] = card
		g.Players[playerIdx].HandLen++
	}
}

// advanceSnapper moves to the next snapper or ends the snap phase if all have acted.
func (g *GameState) advanceSnapper() {
	g.Snap.CurrentSnapperIdx++
	if g.Snap.CurrentSnapperIdx >= g.Snap.NumSnappers {
		g.endSnapPhase()
	}
}
