package engine

import "fmt"

// handLocked reports whether seat's hand is frozen by the LockCallerHand house rule. RULES.md 3C
// makes the Cambia caller's hand untouchable for the rest of the round ("cannot be altered by any
// player, including yourself (snaps, swaps, etc.)"), and the house rule decides whether that clause
// applies at all: ranked play turns it off (MATCHMAKING.md 5.2), which leaves the caller's hand as
// reachable as anyone else's.
//
// legal.go spells the same three-term condition inline at each mask site. This is the copy the
// apply paths use, because a mask is not a guard for them: the service adapter hands an action
// straight to ApplyAction or ApplyNPlayerAction without consulting one, so a hand-rolled or stale
// client's swap reached the caller's hand unopposed (cambia-1118).
func (g *GameState) handLocked(seat uint8) bool {
	return g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(seat) == g.CambiaCaller
}

// discardWithAbility discards the drawn card to the discard pile.
// If the card was drawn from the stockpile and has an ability, the corresponding
// pending ability state is set. Otherwise, the turn advances normally.
func (g *GameState) discardWithAbility() error {
	if g.Pending.Type == PendingNone {
		return fmt.Errorf("no pending drawn card to discard")
	}
	if g.Pending.Type != PendingDiscard {
		return fmt.Errorf("pending action is not a discard (type %d)", g.Pending.Type)
	}

	drawn := Card(g.Pending.Data[0])
	drawnFrom := g.Pending.Data[1]
	acting := g.Pending.PlayerID

	// Place drawn card on top of discard pile.
	g.DiscardPile[g.DiscardLen] = drawn
	g.DiscardLen++

	// Record in LastAction.
	g.LastAction.ActionIdx = ActionDiscardWithAbility
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = drawn
	g.LastAction.DrawnFrom = drawnFrom

	// Clear pending discard state before potentially setting ability pending.
	g.Pending = PendingAction{}

	// Ability only triggers if drawn from stockpile AND card has an ability.
	if drawnFrom != DrawnFromStockpile {
		g.initiateSnapPhase(drawn)
		return nil
	}

	// The ability arms only if the 2-player action space has an action that can resolve it.
	// abilityHasTarget2P answers that as legalAbilitySelect would, which adds the LockCallerHand
	// condition the hand-count checks here used to miss: a Jack, Queen or King discarded while the
	// lock puts the only opponent's hand out of reach armed a pending ability whose legal set was
	// empty, and the engine refuses every other action while it holds one, so the table stopped
	// with no action able to move it. Python already fizzles that case at trigger time
	// (_trigger_discard_ability, cambia-650), so this closes an engine divergence as well.
	if pending := pendingForAbility(drawn.Ability()); g.abilityHasTarget2P(pending, acting) {
		g.Pending.Type = pending
		g.Pending.PlayerID = acting
		return nil
	}

	// No ability or ability fizzles - snap phase for the discarded card.
	g.initiateSnapPhase(drawn)
	return nil
}

// peekOwn resolves a PendingPeekOwn ability: the acting player observes one of
// their own cards. This is observation-only - no cards are moved.
func (g *GameState) peekOwn(targetIdx uint8) error {
	if g.Pending.Type != PendingPeekOwn {
		return fmt.Errorf("pending type is not PendingPeekOwn (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	if targetIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("peekOwn target %d out of range (hand size %d)", targetIdx, g.Players[acting].HandLen)
	}

	revealed := g.Players[acting].Hand[targetIdx]

	// Record observation.
	g.LastAction.ActionIdx = EncodePeekOwn(targetIdx)
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = revealed
	g.LastAction.RevealedIdx = targetIdx
	g.LastAction.RevealedOwner = acting

	// Clear pending and initiate snap phase for the discarded ability card.
	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// peekOther resolves a PendingPeekOther ability: the acting player observes one of
// the opponent's cards. This is observation-only - no cards are moved.
func (g *GameState) peekOther(targetIdx uint8) error {
	if g.Pending.Type != PendingPeekOther {
		return fmt.Errorf("pending type is not PendingPeekOther (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	opp := g.seatOpponent(acting)
	if targetIdx >= g.Players[opp].HandLen {
		return fmt.Errorf("peekOther target %d out of range (opponent hand size %d)", targetIdx, g.Players[opp].HandLen)
	}

	revealed := g.Players[opp].Hand[targetIdx]

	// Record observation.
	g.recordLegacyAction(EncodePeekOther(targetIdx))
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = revealed
	g.LastAction.RevealedIdx = targetIdx
	g.LastAction.RevealedOwner = opp

	// Clear pending and initiate snap phase for the discarded ability card.
	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// blindSwap resolves a PendingBlindSwap ability: the acting player swaps one of
// their own cards with one of the opponent's cards, without seeing either.
func (g *GameState) blindSwap(ownIdx, oppIdx uint8) error {
	if g.Pending.Type != PendingBlindSwap {
		return fmt.Errorf("pending type is not PendingBlindSwap (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	opp := g.seatOpponent(acting)
	if g.handLocked(opp) {
		return fmt.Errorf("blindSwap: opponent %d has called Cambia and their hand is locked", opp)
	}
	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("blindSwap own index %d out of range (hand size %d)", ownIdx, g.Players[acting].HandLen)
	}
	if oppIdx >= g.Players[opp].HandLen {
		return fmt.Errorf("blindSwap opp index %d out of range (opponent hand size %d)", oppIdx, g.Players[opp].HandLen)
	}

	// Perform the swap.
	g.Players[acting].Hand[ownIdx], g.Players[opp].Hand[oppIdx] =
		g.Players[opp].Hand[oppIdx], g.Players[acting].Hand[ownIdx]

	// Record swap info.
	g.recordLegacyAction(EncodeBlindSwap(ownIdx, oppIdx))
	g.LastAction.ActingPlayer = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppIdx

	// Clear pending and initiate snap phase for the discarded ability card.
	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// kingLook is the first step of the King ability: the acting player looks at one
// of their own cards and one of the opponent's cards. The engine then waits for a
// KingSwapDecision action.
func (g *GameState) kingLook(ownIdx, oppIdx uint8) error {
	if g.Pending.Type != PendingKingLook {
		return fmt.Errorf("pending type is not PendingKingLook (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	opp := g.seatOpponent(acting)
	if g.handLocked(opp) {
		return fmt.Errorf("kingLook: opponent %d has called Cambia and their hand is locked", opp)
	}
	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("kingLook own index %d out of range (hand size %d)", ownIdx, g.Players[acting].HandLen)
	}
	if oppIdx >= g.Players[opp].HandLen {
		return fmt.Errorf("kingLook opp index %d out of range (opponent hand size %d)", oppIdx, g.Players[opp].HandLen)
	}

	ownCard := g.Players[acting].Hand[ownIdx]
	oppCard := g.Players[opp].Hand[oppIdx]

	// Record both revealed cards in LastAction (own card in primary slot, opp card can be
	// inferred from context, but we store opp card as the secondary via SwapOppIdx).
	g.recordLegacyAction(EncodeKingLook(ownIdx, oppIdx))
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = ownCard // own card revealed
	g.LastAction.RevealedIdx = ownIdx
	g.LastAction.RevealedOwner = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppIdx

	// Store the looked-at cards and indices in Pending.Data for the decision step.
	// Data[0] = ownIdx, Data[1] = oppIdx, Data[2] = ownCard, Data[3] = oppCard
	g.Pending.Type = PendingKingDecision
	g.Pending.Data[0] = ownIdx
	g.Pending.Data[1] = oppIdx
	g.Pending.Data[2] = uint8(ownCard)
	g.Pending.Data[3] = uint8(oppCard)
	// PlayerID stays the same (acting player).

	return nil
}

// kingSwapDecision is the second step of the King ability: the acting player
// decides whether to swap the two cards they looked at.
func (g *GameState) kingSwapDecision(performSwap bool) error {
	if g.Pending.Type != PendingKingDecision {
		return fmt.Errorf("pending type is not PendingKingDecision (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	opp := g.seatOpponent(acting)

	ownIdx := g.Pending.Data[0]
	oppIdx := g.Pending.Data[1]

	// Validate indices are still in range (hand sizes shouldn't change between look and decide).
	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("kingSwap own index %d out of range (hand size %d)", ownIdx, g.Players[acting].HandLen)
	}
	if oppIdx >= g.Players[opp].HandLen {
		return fmt.Errorf("kingSwap opp index %d out of range (opponent hand size %d)", oppIdx, g.Players[opp].HandLen)
	}

	actionIdx := ActionKingSwapNo
	if performSwap {
		actionIdx = ActionKingSwapYes
		// Perform the swap.
		g.Players[acting].Hand[ownIdx], g.Players[opp].Hand[oppIdx] =
			g.Players[opp].Hand[oppIdx], g.Players[acting].Hand[ownIdx]
	}

	// Record in LastAction.
	g.recordLegacyAction(actionIdx)
	g.LastAction.ActingPlayer = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppIdx
	g.recordSwapTargetPlayer(opp)

	// Clear pending and initiate snap phase for the discarded ability card.
	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// ===========================================================================
// N-Player ability handlers - accept explicit target player
// ===========================================================================

// peekOtherNPlayer resolves PendingPeekOther targeting a specific player.
func (g *GameState) peekOtherNPlayer(slot uint8, targetPlayer uint8) error {
	if g.Pending.Type != PendingPeekOther {
		return fmt.Errorf("pending type is not PendingPeekOther (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	if targetPlayer >= g.Rules.numPlayers() || targetPlayer == acting {
		return fmt.Errorf("peekOtherNPlayer: invalid target player %d", targetPlayer)
	}
	if slot >= g.Players[targetPlayer].HandLen {
		return fmt.Errorf("peekOtherNPlayer: slot %d out of range (hand size %d)", slot, g.Players[targetPlayer].HandLen)
	}

	revealed := g.Players[targetPlayer].Hand[slot]

	oppRel, ok := g.relIdxOfOpponent(acting, targetPlayer)
	if !ok {
		return fmt.Errorf("peekOtherNPlayer: player %d is not an opponent of %d", targetPlayer, acting)
	}

	g.recordNPlayerAction(NPlayerEncodePeekOther(slot, oppRel))
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = revealed
	g.LastAction.RevealedIdx = slot
	g.LastAction.RevealedOwner = targetPlayer

	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// blindSwapNPlayer resolves PendingBlindSwap targeting a specific opponent.
func (g *GameState) blindSwapNPlayer(ownIdx, oppSlot uint8, targetPlayer uint8) error {
	if g.Pending.Type != PendingBlindSwap {
		return fmt.Errorf("pending type is not PendingBlindSwap (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	if targetPlayer >= g.Rules.numPlayers() || targetPlayer == acting {
		return fmt.Errorf("blindSwapNPlayer: invalid target player %d", targetPlayer)
	}
	// A swap moves a card out of the seat it names, so the lock refuses it (RULES.md 3C). This
	// matches the mask that omits the same targets (nplayerLegalAbilitySelect, legal.go).
	if g.handLocked(targetPlayer) {
		return fmt.Errorf("blindSwapNPlayer: target player %d has called Cambia and their hand is locked", targetPlayer)
	}
	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("blindSwapNPlayer: own index %d out of range (hand size %d)", ownIdx, g.Players[acting].HandLen)
	}
	if oppSlot >= g.Players[targetPlayer].HandLen {
		return fmt.Errorf("blindSwapNPlayer: opp slot %d out of range (hand size %d)", oppSlot, g.Players[targetPlayer].HandLen)
	}

	oppRel, ok := g.relIdxOfOpponent(acting, targetPlayer)
	if !ok {
		return fmt.Errorf("blindSwapNPlayer: player %d is not an opponent of %d", targetPlayer, acting)
	}

	g.Players[acting].Hand[ownIdx], g.Players[targetPlayer].Hand[oppSlot] =
		g.Players[targetPlayer].Hand[oppSlot], g.Players[acting].Hand[ownIdx]

	g.recordNPlayerAction(NPlayerEncodeBlindSwap(ownIdx, oppSlot, oppRel))
	g.LastAction.ActingPlayer = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppSlot
	g.recordSwapTargetPlayer(targetPlayer)

	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// kingLookNPlayer resolves the look phase of PendingKingLook targeting a specific opponent.
// Stores targetPlayer in Pending.Data[3] for the subsequent swap decision.
func (g *GameState) kingLookNPlayer(ownIdx, oppSlot uint8, targetPlayer uint8) error {
	if g.Pending.Type != PendingKingLook {
		return fmt.Errorf("pending type is not PendingKingLook (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	if targetPlayer >= g.Rules.numPlayers() || targetPlayer == acting {
		return fmt.Errorf("kingLookNPlayer: invalid target player %d", targetPlayer)
	}
	// The look binds the pair its decision may then swap, so it is refused for the same reason
	// blindSwapNPlayer is: the decision would move a card out of a locked hand.
	if g.handLocked(targetPlayer) {
		return fmt.Errorf("kingLookNPlayer: target player %d has called Cambia and their hand is locked", targetPlayer)
	}
	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("kingLookNPlayer: own index %d out of range (hand size %d)", ownIdx, g.Players[acting].HandLen)
	}
	if oppSlot >= g.Players[targetPlayer].HandLen {
		return fmt.Errorf("kingLookNPlayer: opp slot %d out of range (hand size %d)", oppSlot, g.Players[targetPlayer].HandLen)
	}

	ownCard := g.Players[acting].Hand[ownIdx]

	oppRel, ok := g.relIdxOfOpponent(acting, targetPlayer)
	if !ok {
		return fmt.Errorf("kingLookNPlayer: player %d is not an opponent of %d", targetPlayer, acting)
	}

	g.recordNPlayerAction(NPlayerEncodeKingLook(ownIdx, oppSlot, oppRel))
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = ownCard
	g.LastAction.RevealedIdx = ownIdx
	g.LastAction.RevealedOwner = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppSlot
	g.recordSwapTargetPlayer(targetPlayer)

	// Transition to PendingKingDecision.
	// Data[0]=ownIdx, Data[1]=oppSlot, Data[2]=ownCard, Data[3]=targetPlayer.
	g.Pending.Type = PendingKingDecision
	g.Pending.Data[0] = ownIdx
	g.Pending.Data[1] = oppSlot
	g.Pending.Data[2] = uint8(ownCard)
	g.Pending.Data[3] = targetPlayer
	return nil
}

// kingSwapDecisionNPlayer resolves the swap decision for N-player KingLook.
// Reads targetPlayer from Pending.Data[3].
func (g *GameState) kingSwapDecisionNPlayer(performSwap bool) error {
	if g.Pending.Type != PendingKingDecision {
		return fmt.Errorf("pending type is not PendingKingDecision (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	ownIdx := g.Pending.Data[0]
	oppSlot := g.Pending.Data[1]
	targetPlayer := g.Pending.Data[3]

	if ownIdx >= g.Players[acting].HandLen {
		return fmt.Errorf("kingSwapDecisionNPlayer: own index %d out of range", ownIdx)
	}
	if oppSlot >= g.Players[targetPlayer].HandLen {
		return fmt.Errorf("kingSwapDecisionNPlayer: opp slot %d out of range", oppSlot)
	}

	actionIdx := NPlayerActionKingSwapNo
	if performSwap {
		actionIdx = NPlayerActionKingSwapYes
		g.Players[acting].Hand[ownIdx], g.Players[targetPlayer].Hand[oppSlot] =
			g.Players[targetPlayer].Hand[oppSlot], g.Players[acting].Hand[ownIdx]
	}

	g.recordNPlayerAction(actionIdx)
	g.LastAction.ActingPlayer = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppSlot
	// The decision is the last chance to record the seat the look bound: Pending.Data[3]
	// holds it only until the clear below, and every observer runs after that (cambia-1548).
	g.recordSwapTargetPlayer(targetPlayer)

	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}

// discardWithAbilityNPlayer handles DiscardWithAbility for N-player games (>2 players).
// Checks ALL opponents to determine if an ability can fire.
func (g *GameState) discardWithAbilityNPlayer() error {
	if g.Pending.Type != PendingDiscard {
		return fmt.Errorf("pending action is not a discard (type %d)", g.Pending.Type)
	}

	drawn := Card(g.Pending.Data[0])
	drawnFrom := g.Pending.Data[1]
	acting := g.Pending.PlayerID

	g.DiscardPile[g.DiscardLen] = drawn
	g.DiscardLen++

	g.LastAction.ActionIdx = ActionDiscardWithAbility
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = drawn
	g.LastAction.DrawnFrom = drawnFrom

	g.Pending = PendingAction{}

	if drawnFrom != DrawnFromStockpile {
		g.initiateSnapPhase(drawn)
		return nil
	}

	// Same arm-or-fizzle rule as the 2-player path, asked of every opponent instead of one seat:
	// abilityHasTargetNP mirrors nplayerLegalAbilitySelect, and walks the seats in place where this
	// used to allocate a slice per opponent-facing ability through Opponents().
	if pending := pendingForAbility(drawn.Ability()); g.abilityHasTargetNP(pending, acting) {
		g.Pending.Type = pending
		g.Pending.PlayerID = acting
		return nil
	}

	g.initiateSnapPhase(drawn)
	return nil
}

// ResolveUntargetableArmedAbility discharges an armed ability that no action in the caller's action
// space can resolve: it clears the pending ability and runs the snap phase for the card that armed
// it, which is what resolving any ability does and which advances the turn when no one can snap.
// It reports whether it resolved anything, and is a no-op whenever the ability still has a legal
// target or nothing is armed.
//
// The engine refuses every action while it holds a pending ability, so an armed ability with an
// empty legal set stops the table: no action exists that could change the condition, and the
// service's timeout path had nothing to do but re-arm the same player's clock forever
// (cambia-1171). The arm sites above no longer create that state, so this is a guard for the paths
// that can still reach it - replace() arms an ability through canUseAbility, which asks about any
// opponent rather than the one seat the 2-player space encodes - and for callers holding a state
// built before this fix. The loop it guards against was never reproduced.
//
// nPlayerSpace names the action space the caller drives, because "no legal target" is a different
// question in each: the N-player space reaches every opponent, the 2-player space only the seat
// seatOpponent names. The caller passes the space whose mask refused it - the service passes
// isNPlayerTable(), the FFI 2-player surface passes false - so the answer here matches the mask
// that stranded the ability rather than the other one.
func (g *GameState) ResolveUntargetableArmedAbility(nPlayerSpace bool) bool {
	switch g.Pending.Type {
	case PendingPeekOwn, PendingPeekOther, PendingBlindSwap, PendingKingLook:
	default:
		// PendingKingDecision always has both answers legal, and no other pending state is an
		// armed ability.
		return false
	}
	if g.DiscardLen == 0 {
		// The card that armed the ability is the one the snap phase runs for; without it there is
		// nothing to resolve into.
		return false
	}
	acting := g.Pending.PlayerID
	if nPlayerSpace {
		if g.abilityHasTargetNP(g.Pending.Type, acting) {
			return false
		}
	} else if g.abilityHasTarget2P(g.Pending.Type, acting) {
		return false
	}

	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return true
}
