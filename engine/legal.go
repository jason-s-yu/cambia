package engine

// DecisionCtx returns the current decision context for the acting player.
func (g *GameState) DecisionCtx() DecisionContext {
	if g.IsTerminal() {
		return CtxTerminal
	}
	if g.Snap.Active {
		if g.Pending.Type == PendingSnapMove {
			return CtxSnapMove
		}
		return CtxSnapDecision
	}
	switch g.Pending.Type {
	case PendingDiscard:
		return CtxPostDraw
	case PendingPeekOwn, PendingPeekOther, PendingBlindSwap, PendingKingLook, PendingKingDecision:
		return CtxAbilitySelect
	}
	return CtxStartTurn
}

// setBit sets bit idx in the bitmask.
func setBit(mask *[3]uint64, idx uint16) {
	mask[idx/64] |= 1 << (idx % 64)
}

// LegalActions returns a bitmask of legal action indices.
// Bit i of result[i/64] is set if action i is legal.
// Zero heap allocation.
func (g *GameState) LegalActions() [3]uint64 {
	var mask [3]uint64

	switch g.DecisionCtx() {
	case CtxTerminal:
		// No legal actions.

	case CtxStartTurn:
		g.legalStartTurn(&mask)

	case CtxPostDraw:
		g.legalPostDraw(&mask)

	case CtxAbilitySelect:
		g.legalAbilitySelect(&mask)

	case CtxSnapDecision:
		g.legalSnapDecision(&mask)

	case CtxSnapMove:
		g.legalSnapMove(&mask)
	}

	return mask
}

// LegalActionsList returns legal actions as a slice (for testing; allocates).
func (g *GameState) LegalActionsList() []uint16 {
	mask := g.LegalActions()
	var actions []uint16
	for i := uint16(0); i < NumActions; i++ {
		if mask[i/64]>>(i%64)&1 == 1 {
			actions = append(actions, i)
		}
	}
	return actions
}

// legalStartTurn populates legal actions for CtxStartTurn.
func (g *GameState) legalStartTurn(mask *[3]uint64) {
	// DrawStockpile: always legal (stockpile non-empty, or can reshuffle).
	// Reshuffle possible if discard has >1 card.
	if g.StockLen > 0 || g.DiscardLen > 1 {
		setBit(mask, ActionDrawStockpile)
	}

	// DrawDiscard: legal if rule allows AND discard pile non-empty.
	if g.Rules.AllowDrawFromDiscard && g.DiscardLen > 0 {
		setBit(mask, ActionDrawDiscard)
	}

	// CallCambia: legal if current round >= CambiaAllowedRound AND no one has
	// called Cambia yet. Round = TurnNumber / NumPlayers.
	currentRound := g.TurnNumber / uint16(g.Rules.numPlayers())
	if g.CambiaCaller == -1 && currentRound >= uint16(g.Rules.CambiaAllowedRound) {
		setBit(mask, ActionCallCambia)
	}
}

// legalPostDraw populates legal actions for CtxPostDraw.
func (g *GameState) legalPostDraw(mask *[3]uint64) {
	acting := g.Pending.PlayerID
	drawnCard := Card(g.Pending.Data[0])
	drawnFrom := g.Pending.Data[1]

	// DiscardNoAbility: always legal.
	setBit(mask, ActionDiscardNoAbility)

	// DiscardWithAbility: legal only if card has ability AND was drawn from stockpile.
	if drawnCard.HasAbility() && drawnFrom == DrawnFromStockpile {
		// Also check ability can actually be used.
		if g.canUseAbility(acting, drawnCard) {
			setBit(mask, ActionDiscardWithAbility)
		}
	}

	// Replace(i): legal for each i < acting player's hand length.
	// When LockCallerHand is true and Cambia has been called, the caller
	// cannot replace cards — they may only discard.
	if !(g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(acting) == g.CambiaCaller) {
		handLen := g.Players[acting].HandLen
		for i := uint8(0); i < handLen; i++ {
			setBit(mask, EncodeReplace(i))
		}
	}
}

// canUseAbility returns true if the ability of the given card can be used.
// Mirrors Python's ability fizzle conditions.
func (g *GameState) canUseAbility(acting uint8, card Card) bool {
	opp := g.OpponentOf(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	switch card.Ability() {
	case AbilityPeekOwn:
		return ownHandLen > 0
	case AbilityPeekOther:
		return oppHandLen > 0
	case AbilityBlindSwap, AbilityKingLook:
		// When LockCallerHand is true and the opponent is the Cambia caller,
		// swap abilities cannot target them — fizzle at ability-select stage.
		if g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller {
			return false
		}
		return ownHandLen > 0 && oppHandLen > 0
	default:
		return false
	}
}

// legalAbilitySelect populates legal actions for CtxAbilitySelect.
func (g *GameState) legalAbilitySelect(mask *[3]uint64) {
	acting := g.Pending.PlayerID
	opp := g.OpponentOf(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	switch g.Pending.Type {
	case PendingPeekOwn:
		// PeekOwn(i) for i = 0..ownHandLen-1.
		for i := uint8(0); i < ownHandLen; i++ {
			setBit(mask, EncodePeekOwn(i))
		}

	case PendingPeekOther:
		// PeekOther(i) for i = 0..oppHandLen-1.
		for i := uint8(0); i < oppHandLen; i++ {
			setBit(mask, EncodePeekOther(i))
		}

	case PendingBlindSwap:
		// BlindSwap(own_i, opp_j) for all valid combinations.
		// When LockCallerHand is true and opponent is the Cambia caller,
		// targeting their hand is forbidden — ability produces no actions.
		if !(g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller) {
			for i := uint8(0); i < ownHandLen; i++ {
				for j := uint8(0); j < oppHandLen; j++ {
					setBit(mask, EncodeBlindSwap(i, j))
				}
			}
		}

	case PendingKingLook:
		// KingLook(own_i, opp_j) for all valid combinations.
		// When LockCallerHand is true and opponent is the Cambia caller,
		// targeting their hand is forbidden — ability produces no actions.
		if !(g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller) {
			for i := uint8(0); i < ownHandLen; i++ {
				for j := uint8(0); j < oppHandLen; j++ {
					setBit(mask, EncodeKingLook(i, j))
				}
			}
		}

	case PendingKingDecision:
		// Both KingSwapNo and KingSwapYes are always legal.
		setBit(mask, ActionKingSwapNo)
		setBit(mask, ActionKingSwapYes)
	}
}

// legalSnapDecision populates legal actions for CtxSnapDecision.
func (g *GameState) legalSnapDecision(mask *[3]uint64) {
	acting := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	opp := g.OpponentOf(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	// PassSnap: always legal.
	setBit(mask, ActionPassSnap)

	// SnapOwn(i): legal for each i < acting player's hand length.
	for i := uint8(0); i < ownHandLen; i++ {
		setBit(mask, EncodeSnapOwn(i))
	}

	// SnapOpponent(i): legal for each i < opponent's hand length, only if rule allows.
	if g.Rules.AllowOpponentSnapping && ownHandLen > 0 {
		for i := uint8(0); i < oppHandLen; i++ {
			setBit(mask, EncodeSnapOpponent(i))
		}
	}
}

// legalSnapMove populates legal actions for CtxSnapMove.
// The snapper must move one of their cards to the vacated slot in the opponent's hand.
func (g *GameState) legalSnapMove(mask *[3]uint64) {
	snapperIdx := g.Pending.PlayerID
	slotIdx := g.Pending.Data[1] // The vacated slot index in the opponent's hand.
	ownHandLen := g.Players[snapperIdx].HandLen

	// SnapOpponentMove(own_i, slot_j): for each own_i < snapper's hand length.
	for i := uint8(0); i < ownHandLen; i++ {
		setBit(mask, EncodeSnapOpponentMove(i, slotIdx))
	}
}

// ===========================================================================
// N-Player legal action generation (452-action space)
// ===========================================================================

// nplayerSetBit sets bit idx in an [8]uint64 bitmask (512 bits, covers 452).
func nplayerSetBit(mask *[8]uint64, idx uint16) {
	mask[idx/64] |= 1 << (idx % 64)
}

// NPlayerLegalActions returns a bitmask of legal N-player action indices (452 actions).
// Uses [8]uint64 (512 bits). Zero heap allocation.
func (g *GameState) NPlayerLegalActions() [8]uint64 {
	var mask [8]uint64

	switch g.DecisionCtx() {
	case CtxTerminal:
		// No legal actions.

	case CtxStartTurn:
		g.nplayerLegalStartTurn(&mask)

	case CtxPostDraw:
		g.nplayerLegalPostDraw(&mask)

	case CtxAbilitySelect:
		g.nplayerLegalAbilitySelect(&mask)

	case CtxSnapDecision:
		g.nplayerLegalSnapDecision(&mask)

	case CtxSnapMove:
		g.nplayerLegalSnapMove(&mask)
	}

	return mask
}

// NPlayerLegalActionsList returns N-player legal actions as a slice (allocates; for testing).
func (g *GameState) NPlayerLegalActionsList() []uint16 {
	mask := g.NPlayerLegalActions()
	var actions []uint16
	for i := uint16(0); i < NPlayerNumActions; i++ {
		if mask[i/64]>>(i%64)&1 == 1 {
			actions = append(actions, i)
		}
	}
	return actions
}

func (g *GameState) nplayerLegalStartTurn(mask *[8]uint64) {
	if g.StockLen > 0 || g.DiscardLen > 1 {
		nplayerSetBit(mask, NPlayerActionDrawStockpile)
	}
	if g.Rules.AllowDrawFromDiscard && g.DiscardLen > 0 {
		nplayerSetBit(mask, NPlayerActionDrawDiscard)
	}
	currentRound := g.TurnNumber / uint16(g.Rules.numPlayers())
	if g.CambiaCaller == -1 && currentRound >= uint16(g.Rules.CambiaAllowedRound) {
		nplayerSetBit(mask, NPlayerActionCallCambia)
	}
}

func (g *GameState) nplayerLegalPostDraw(mask *[8]uint64) {
	acting := g.Pending.PlayerID
	drawnCard := Card(g.Pending.Data[0])
	drawnFrom := g.Pending.Data[1]

	nplayerSetBit(mask, NPlayerActionDiscardNoAbility)

	if drawnCard.HasAbility() && drawnFrom == DrawnFromStockpile {
		if g.nplayerCanUseAbility(acting, drawnCard) {
			nplayerSetBit(mask, NPlayerActionDiscardWithAbility)
		}
	}

	if !(g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(acting) == g.CambiaCaller) {
		handLen := g.Players[acting].HandLen
		for i := uint8(0); i < handLen; i++ {
			nplayerSetBit(mask, NPlayerEncodeReplace(i))
		}
	}
}

// nplayerCanUseAbility checks whether the ability card can actually be used in N-player context.
func (g *GameState) nplayerCanUseAbility(acting uint8, card Card) bool {
	ownHandLen := g.Players[acting].HandLen
	opps := g.Opponents(acting)

	switch card.Ability() {
	case AbilityPeekOwn:
		return ownHandLen > 0
	case AbilityPeekOther:
		for _, opp := range opps {
			if g.Players[opp].HandLen > 0 {
				return true
			}
		}
		return false
	case AbilityBlindSwap, AbilityKingLook:
		if ownHandLen == 0 {
			return false
		}
		// At least one non-caller opponent must have cards.
		for _, opp := range opps {
			if g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller {
				continue
			}
			if g.Players[opp].HandLen > 0 {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func (g *GameState) nplayerLegalAbilitySelect(mask *[8]uint64) {
	acting := g.Pending.PlayerID
	opps := g.Opponents(acting)
	ownHandLen := g.Players[acting].HandLen

	switch g.Pending.Type {
	case PendingPeekOwn:
		for i := uint8(0); i < ownHandLen; i++ {
			nplayerSetBit(mask, NPlayerEncodePeekOwn(i))
		}

	case PendingPeekOther:
		for oppRelIdx, opp := range opps {
			oppHandLen := g.Players[opp].HandLen
			for i := uint8(0); i < oppHandLen; i++ {
				nplayerSetBit(mask, NPlayerEncodePeekOther(i, uint8(oppRelIdx)))
			}
		}

	case PendingBlindSwap:
		for oppRelIdx, opp := range opps {
			if g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller {
				continue
			}
			oppHandLen := g.Players[opp].HandLen
			for i := uint8(0); i < ownHandLen; i++ {
				for j := uint8(0); j < oppHandLen; j++ {
					nplayerSetBit(mask, NPlayerEncodeBlindSwap(i, j, uint8(oppRelIdx)))
				}
			}
		}

	case PendingKingLook:
		for oppRelIdx, opp := range opps {
			if g.Rules.LockCallerHand && g.IsCambiaCalled() && int8(opp) == g.CambiaCaller {
				continue
			}
			oppHandLen := g.Players[opp].HandLen
			for i := uint8(0); i < ownHandLen; i++ {
				for j := uint8(0); j < oppHandLen; j++ {
					nplayerSetBit(mask, NPlayerEncodeKingLook(i, j, uint8(oppRelIdx)))
				}
			}
		}

	case PendingKingDecision:
		nplayerSetBit(mask, NPlayerActionKingSwapNo)
		nplayerSetBit(mask, NPlayerActionKingSwapYes)
	}
}

func (g *GameState) nplayerLegalSnapDecision(mask *[8]uint64) {
	acting := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	opps := g.Opponents(acting)
	ownHandLen := g.Players[acting].HandLen

	nplayerSetBit(mask, NPlayerActionPassSnap)

	for i := uint8(0); i < ownHandLen; i++ {
		nplayerSetBit(mask, NPlayerEncodeSnapOwn(i))
	}

	if g.Rules.AllowOpponentSnapping && ownHandLen > 0 {
		for oppRelIdx, opp := range opps {
			oppHandLen := g.Players[opp].HandLen
			for i := uint8(0); i < oppHandLen; i++ {
				nplayerSetBit(mask, NPlayerEncodeSnapOpponent(i, uint8(oppRelIdx)))
			}
		}
	}
}

func (g *GameState) nplayerLegalSnapMove(mask *[8]uint64) {
	snapperIdx := g.Pending.PlayerID
	ownHandLen := g.Players[snapperIdx].HandLen
	for i := uint8(0); i < ownHandLen; i++ {
		nplayerSetBit(mask, NPlayerEncodeSnapOpponentMove(i))
	}
}
