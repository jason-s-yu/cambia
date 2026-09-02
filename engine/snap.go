package engine

import "fmt"

// initiateSnapPhase starts the snap phase after a card is discarded.
// It builds the snapper list, sets SnapState fields, and either begins
// the phase or falls through to advanceTurn if no eligible snappers exist.
func (g *GameState) initiateSnapPhase(discardedCard Card) {
	g.Snap.DiscardedRank = discardedCard.Rank()

	// The acting player just discarded - they are the "discarder".
	// Build the snapper list: discarder first, then all others in turn order.
	// Skip a Cambia caller whose hand the lock has frozen: RULES.md 3C bars them from both
	// directions of a snap, since snapping their own card empties a slot in the locked hand and
	// snapping an opponent's obliges them to pay a card out of it. The exclusion follows the house
	// rule rather than the call, because with lockCallerHand off the caller keeps playing the snap
	// window like anyone else - which is the configuration ranked play uses (MATCHMAKING.md 5.2),
	// where an unconditional exclusion silently dropped a whole player's snaps (cambia-1118).
	// NOTE: race-OFF is the sequential discarder-first model, the ratified and
	// frozen default recorded at engine/rules.go:14 (cambia-564); RULES.md 5
	// describes the race-ON model instead, a simultaneous imperfect-info
	// commit with no ordering. See also cambia-543.
	discarder := g.CurrentPlayer
	n := g.Rules.numPlayers()

	// Build ordered candidate list: discarder first, then remaining players in
	// turn order starting from discarder+1.
	count := uint8(0)
	var snappers [MaxPlayers]uint8

	// Collect players in discarder-first order.
	for step := uint8(0); step < n; step++ {
		p := (discarder + step) % n
		if g.handLocked(p) {
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
		// Check if this player can snap any opponent's card.
		canSnapOpp := false
		if g.Rules.AllowOpponentSnapping && hand.HandLen > 0 {
			for opp := uint8(0); opp < n; opp++ {
				if opp == p || g.handLocked(opp) {
					continue
				}
				oppHand := &g.Players[opp]
				for i := uint8(0); i < oppHand.HandLen; i++ {
					if oppHand.Hand[i].Rank() == discardedCard.Rank() {
						canSnapOpp = true
						break
					}
				}
				if canSnapOpp {
					break
				}
			}
		}
		if canSnapOwn || canSnapOpp {
			snappers[count] = p
			count++
		}
	}

	if count == 0 {
		// No eligible snappers - skip snap phase entirely.
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
		// Invalid index - treat as failed snap.
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
	opponent := g.seatOpponent(snapperIdx)
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
		// Invalid index - treat as failed snap.
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
		g.Pending.Data[0] = opponent // which opponent's hand to place card in
		g.Pending.Data[1] = oppIdx   // the vacated slot index (cards shifted, this is now end)

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

// SnapMoveCard performs the RULES.md 5 fill: the card at fromIdx in fromPlayer's hand moves into
// toPlayer's hand at slotIdx, shifting that hand right so the card lands in the slot the snapped
// card vacated. slotIdx may equal the destination hand's length, which appends. It reports whether
// the move happened: false when either index is out of range or the destination hand is already at
// MaxHandSize, in which case no hand is touched.
//
// It mutates hands only: no snap-phase, pending-action or turn state moves, so a caller owns
// whatever sequencing its own model needs. Exported for the same reason as DrawPenaltyCard: the
// service answers snaps asynchronously, outside the engine's sequential snap phase, so it cannot
// reach snapOpponentMove and would otherwise hand-roll the shift-and-insert this rule is (the
// service skipped the fill entirely until cambia-936). Routing both callers through here keeps the
// mechanics of the fill defined once.
func (g *GameState) SnapMoveCard(fromPlayer, fromIdx, toPlayer, slotIdx uint8) bool {
	if int(fromPlayer) >= MaxPlayers || int(toPlayer) >= MaxPlayers {
		return false
	}
	if fromIdx >= g.Players[fromPlayer].HandLen {
		return false
	}
	toHandLen := g.Players[toPlayer].HandLen
	if slotIdx > toHandLen || toHandLen >= MaxHandSize {
		return false
	}

	card := g.removeCardFromHand(fromPlayer, fromIdx)

	// The removal above shifts the source hand left; when both hands are the same player's the
	// destination length has to be re-read, since it just changed.
	toHandLen = g.Players[toPlayer].HandLen
	if slotIdx > toHandLen {
		slotIdx = toHandLen
	}
	for i := toHandLen; i > slotIdx; i-- {
		g.Players[toPlayer].Hand[i] = g.Players[toPlayer].Hand[i-1]
	}
	g.Players[toPlayer].Hand[slotIdx] = card
	g.Players[toPlayer].HandLen++
	return true
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

	// Remove the snapper's card and insert it into the opponent's hand at slotIdx.
	if !g.SnapMoveCard(snapperIdx, ownIdx, opponent, slotIdx) {
		return fmt.Errorf("snapOpponentMove: cannot move card %d from player %d into player %d slot %d", ownIdx, snapperIdx, opponent, slotIdx)
	}

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
		if !g.DrawPenaltyCard(playerIdx) {
			break
		}
	}
}

// DrawPenaltyCard draws a single snap-penalty card from the stockpile into the player's hand,
// reshuffling the discard pile back into the stockpile first when the stockpile has run dry.
// It reports whether a card was drawn: false when the hand already holds MaxHandSize cards, or
// when the deck is exhausted (empty stockpile and a discard pile too thin to reshuffle), which
// is how a penalty is paid short.
//
// Exported for adapters that cannot drive the engine's sequential snap phase and would otherwise
// hand-roll the penalty draw: the service answers snaps asynchronously, so its snap window has
// already closed by the time a client's snap arrives and snapOwn/snapOpponent (which draw the
// penalty themselves) are no longer applicable. Routing that path through this primitive keeps
// the penalty draw rules, the reshuffle included, defined here only.
func (g *GameState) DrawPenaltyCard(playerIdx uint8) bool {
	if int(playerIdx) >= MaxPlayers {
		return false
	}
	if g.Players[playerIdx].HandLen >= MaxHandSize {
		return false
	}
	if g.StockLen == 0 {
		g.attemptReshuffle()
	}
	if g.StockLen == 0 {
		return false // No cards left at all.
	}
	g.StockLen--
	card := g.Stockpile[g.StockLen]
	handLen := g.Players[playerIdx].HandLen
	g.Players[playerIdx].Hand[handLen] = card
	g.Players[playerIdx].HandLen++
	return true
}

// advanceSnapper moves to the next snapper or ends the snap phase if all have acted.
// Under race-ON (SnapRace) the final advance triggers resolveSnapRace, which draws
// the winner among willing committers and settles the window; otherwise the phase
// ends immediately once every sequential snapper has acted (race-OFF).
func (g *GameState) advanceSnapper() {
	g.Snap.CurrentSnapperIdx++
	if g.Snap.CurrentSnapperIdx >= g.Snap.NumSnappers {
		if g.Rules.SnapRace {
			g.resolveSnapRace()
			return
		}
		g.endSnapPhase()
	}
}
