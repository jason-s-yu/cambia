package engine

import "fmt"

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

	ability := drawn.Ability()
	opp := g.OpponentOf(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	switch ability {
	case AbilityPeekOwn:
		if ownHandLen > 0 {
			g.Pending.Type = PendingPeekOwn
			g.Pending.PlayerID = acting
			return nil
		}
	case AbilityPeekOther:
		if oppHandLen > 0 {
			g.Pending.Type = PendingPeekOther
			g.Pending.PlayerID = acting
			return nil
		}
	case AbilityBlindSwap:
		if ownHandLen > 0 && oppHandLen > 0 {
			g.Pending.Type = PendingBlindSwap
			g.Pending.PlayerID = acting
			return nil
		}
	case AbilityKingLook:
		if ownHandLen > 0 && oppHandLen > 0 {
			g.Pending.Type = PendingKingLook
			g.Pending.PlayerID = acting
			return nil
		}
	}

	// No ability or ability fizzles — snap phase for the discarded card.
	g.initiateSnapPhase(drawn)
	return nil
}

// peekOwn resolves a PendingPeekOwn ability: the acting player observes one of
// their own cards. This is observation-only — no cards are moved.
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
// the opponent's cards. This is observation-only — no cards are moved.
func (g *GameState) peekOther(targetIdx uint8) error {
	if g.Pending.Type != PendingPeekOther {
		return fmt.Errorf("pending type is not PendingPeekOther (got %d)", g.Pending.Type)
	}
	acting := g.Pending.PlayerID
	opp := g.OpponentOf(acting)
	if targetIdx >= g.Players[opp].HandLen {
		return fmt.Errorf("peekOther target %d out of range (opponent hand size %d)", targetIdx, g.Players[opp].HandLen)
	}

	revealed := g.Players[opp].Hand[targetIdx]

	// Record observation.
	g.LastAction.ActionIdx = EncodePeekOther(targetIdx)
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
	opp := g.OpponentOf(acting)
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
	g.LastAction.ActionIdx = EncodeBlindSwap(ownIdx, oppIdx)
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
	opp := g.OpponentOf(acting)
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
	g.LastAction.ActionIdx = EncodeKingLook(ownIdx, oppIdx)
	g.LastAction.ActingPlayer = acting
	g.LastAction.RevealedCard = ownCard  // own card revealed
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
	opp := g.OpponentOf(acting)

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
	g.LastAction.ActionIdx = actionIdx
	g.LastAction.ActingPlayer = acting
	g.LastAction.SwapOwnIdx = ownIdx
	g.LastAction.SwapOppIdx = oppIdx

	// Clear pending and initiate snap phase for the discarded ability card.
	g.Pending = PendingAction{}
	g.initiateSnapPhase(g.DiscardPile[g.DiscardLen-1])
	return nil
}
