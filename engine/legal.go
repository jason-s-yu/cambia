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
	// cannot replace cards - they may only discard.
	if !g.HandLocked(acting) {
		handLen := g.Players[acting].HandLen
		for i := uint8(0); i < handLen; i++ {
			setBit(mask, EncodeReplace(i))
		}
	}
}

// canUseAbility returns true if the ability of the given card can be used.
// Mirrors Python's ability fizzle conditions.
//
// An opponent-facing ability needs SOME opponent it can still reach, which is the same question
// the N-player ability-select mask asks, so both go through abilityHasTargetNP. It used to ask
// about the single seat OpponentOf(acting) names, which is 1-acting: correct at two seats, and
// from seat 2 an underflow to 255 that indexed off the end of the player array. replace() calls
// this on every replace when AllowReplaceAbilities is on, and it is shared by both action spaces,
// so at a ranked FFA table a replace from seat 2 or 3 panicked the game (cambia-1125). At two
// seats the two readings are the same question, so nothing about the 2-player action space or its
// legal masks changes.
func (g *GameState) canUseAbility(acting uint8, card Card) bool {
	return g.abilityHasTargetNP(pendingForAbility(card.Ability()), acting)
}

// seatOpponent names the seat the 146-action space treats as "the opponent". That space encodes
// exactly one, so a table with more seats is the N-player space's business (NPlayerLegalActions and
// the *NPlayer apply handlers); this only has to stay in bounds when the 2-player surface is used
// at such a table, which the FFI surface (cambia_game_legal_actions / cambia_game_apply_action)
// permits.
//
// OpponentOf is 1-acting, which is that answer at two seats and an underflow to 255 from seat 2,
// indexing off the end of Players exactly as the replace path did before cambia-1125. The next
// seat round the table is the same seat at two players, so no bit of the 2-player mask moves and
// no 2-player apply path resolves against a different seat.
//
// Both halves of the 2-player surface read the opponent here, mask and apply alike: if they
// disagreed, the mask would enumerate one seat's slots and the apply path would index another's.
func (g *GameState) seatOpponent(acting uint8) uint8 {
	if n := g.Rules.numPlayers(); n > 2 {
		return (acting + 1) % n
	}
	return g.OpponentOf(acting)
}

// pendingForAbility maps a card's ability to the pending state that ability arms, or PendingNone
// for a card with no ability. Both discard-with-ability arms and canUseAbility key off it, so the
// ability -> pending mapping is stated once.
func pendingForAbility(ability AbilityType) PendingType {
	switch ability {
	case AbilityPeekOwn:
		return PendingPeekOwn
	case AbilityPeekOther:
		return PendingPeekOther
	case AbilityBlindSwap:
		return PendingBlindSwap
	case AbilityKingLook:
		return PendingKingLook
	default:
		return PendingNone
	}
}

// abilityHasTarget2P reports whether legalAbilitySelect would set at least one bit for an armed
// ability of type pending held by acting. An ability the 2-player space can produce no action for
// cannot be resolved by any caller that respects the mask, and the engine refuses every other
// action while it holds a pending ability, so arming one stops the table: the arm sites resolve it
// instead (cambia-1171).
//
// Kept in step with legalAbilitySelect below by construction - same seat via seatOpponent, and
// both ask HandLocked rather than restating the rule - and by
// TestAbilityHasTargetMatchesAbilitySelectMask.
func (g *GameState) abilityHasTarget2P(pending PendingType, acting uint8) bool {
	opp := g.seatOpponent(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	switch pending {
	case PendingPeekOwn:
		return ownHandLen > 0
	case PendingPeekOther:
		// Not gated on the lock: peeking reads a hand rather than moving a card out of it, and
		// legalAbilitySelect does not gate it either.
		return oppHandLen > 0
	case PendingBlindSwap, PendingKingLook:
		if g.HandLocked(opp) {
			return false
		}
		return ownHandLen > 0 && oppHandLen > 0
	case PendingKingDecision:
		// Both swap answers are always legal.
		return true
	default:
		return false
	}
}

// abilityHasTargetNP is abilityHasTarget2P's N-player counterpart: it reports whether
// nplayerLegalAbilitySelect would set at least one bit, which asks the same question of every
// opponent rather than of one seat. Seats are walked in place rather than through Opponents(),
// which allocates: this sits on the legal-mask path.
func (g *GameState) abilityHasTargetNP(pending PendingType, acting uint8) bool {
	ownHandLen := g.Players[acting].HandLen

	reachableOpponent := func(skipLockedCaller bool) bool {
		n := g.Rules.numPlayers()
		for opp := uint8(0); opp < n; opp++ {
			if opp == acting || g.Players[opp].HandLen == 0 {
				continue
			}
			if skipLockedCaller && g.HandLocked(opp) {
				continue
			}
			return true
		}
		return false
	}

	switch pending {
	case PendingPeekOwn:
		return ownHandLen > 0
	case PendingPeekOther:
		return reachableOpponent(false)
	case PendingBlindSwap, PendingKingLook:
		// When LockCallerHand is true and every opponent left is the Cambia caller, swap
		// abilities cannot target anyone - fizzle at ability-select stage.
		return ownHandLen > 0 && reachableOpponent(true)
	case PendingKingDecision:
		return true
	default:
		return false
	}
}

// legalAbilitySelect populates legal actions for CtxAbilitySelect.
func (g *GameState) legalAbilitySelect(mask *[3]uint64) {
	acting := g.Pending.PlayerID
	opp := g.seatOpponent(acting)
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
		// targeting their hand is forbidden - ability produces no actions.
		if !g.HandLocked(opp) {
			for i := uint8(0); i < ownHandLen; i++ {
				for j := uint8(0); j < oppHandLen; j++ {
					setBit(mask, EncodeBlindSwap(i, j))
				}
			}
		}

	case PendingKingLook:
		// KingLook(own_i, opp_j) for all valid combinations.
		// When LockCallerHand is true and opponent is the Cambia caller,
		// targeting their hand is forbidden - ability produces no actions.
		if !g.HandLocked(opp) {
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
	opp := g.seatOpponent(acting)
	ownHandLen := g.Players[acting].HandLen
	oppHandLen := g.Players[opp].HandLen

	// PassSnap: always legal.
	setBit(mask, ActionPassSnap)

	// SnapOwn(i): legal for each i < acting player's hand length.
	for i := uint8(0); i < ownHandLen; i++ {
		setBit(mask, EncodeSnapOwn(i))
	}

	// SnapOpponent(i): legal for each i < opponent's hand length, only if rule allows.
	// A locked caller's hand is not a snap target either (RULES.md 3C names snaps first), and a
	// seat reaches this mask on canSnapOwn alone, so the snapper being unlocked says nothing
	// about the target. snapOpponent refuses the same position on the apply side.
	if g.Rules.AllowOpponentSnapping && ownHandLen > 0 && !g.HandLocked(opp) {
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
// N-Player legal action generation (620-action space)
// ===========================================================================

// nplayerSetBit sets bit idx in an [10]uint64 bitmask (640 bits, covers 620).
func nplayerSetBit(mask *[10]uint64, idx uint16) {
	mask[idx/64] |= 1 << (idx % 64)
}

// NPlayerLegalActions returns a bitmask of legal N-player action indices (620 actions).
// Uses [10]uint64 (640 bits). Zero heap allocation.
func (g *GameState) NPlayerLegalActions() [10]uint64 {
	var mask [10]uint64

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

func (g *GameState) nplayerLegalStartTurn(mask *[10]uint64) {
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

func (g *GameState) nplayerLegalPostDraw(mask *[10]uint64) {
	acting := g.Pending.PlayerID
	drawnCard := Card(g.Pending.Data[0])
	drawnFrom := g.Pending.Data[1]

	nplayerSetBit(mask, NPlayerActionDiscardNoAbility)

	// One reachability rule, one implementation: canUseAbility already asks whether any seat
	// other than acting is still reachable, which is the same question at any table size, and it
	// walks the seats in place rather than through Opponents(), which allocates on this path.
	// The N-player arm used to hold a second copy of the rule, so a fix to either one (cambia-1125
	// corrected the 2-player copy) left the other saying something else.
	if drawnCard.HasAbility() && drawnFrom == DrawnFromStockpile {
		if g.canUseAbility(acting, drawnCard) {
			nplayerSetBit(mask, NPlayerActionDiscardWithAbility)
		}
	}

	if !g.HandLocked(acting) {
		handLen := g.Players[acting].HandLen
		for i := uint8(0); i < handLen; i++ {
			nplayerSetBit(mask, NPlayerEncodeReplace(i))
		}
	}
}

func (g *GameState) nplayerLegalAbilitySelect(mask *[10]uint64) {
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
			if g.HandLocked(opp) {
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
			if g.HandLocked(opp) {
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

func (g *GameState) nplayerLegalSnapDecision(mask *[10]uint64) {
	acting := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	opps := g.Opponents(acting)
	ownHandLen := g.Players[acting].HandLen

	nplayerSetBit(mask, NPlayerActionPassSnap)

	for i := uint8(0); i < ownHandLen; i++ {
		nplayerSetBit(mask, NPlayerEncodeSnapOwn(i))
	}

	if g.Rules.AllowOpponentSnapping && ownHandLen > 0 {
		for oppRelIdx, opp := range opps {
			// The locked caller is an opponent like any other here, and at three or more seats
			// the snapper reaching this mask is usually not the caller, so the target needs its
			// own test. nplayerSnapOpponent refuses the same position on the apply side.
			if g.HandLocked(opp) {
				continue
			}
			oppHandLen := g.Players[opp].HandLen
			for i := uint8(0); i < oppHandLen; i++ {
				nplayerSetBit(mask, NPlayerEncodeSnapOpponent(i, uint8(oppRelIdx)))
			}
		}
	}
}

func (g *GameState) nplayerLegalSnapMove(mask *[10]uint64) {
	snapperIdx := g.Pending.PlayerID
	ownHandLen := g.Players[snapperIdx].HandLen
	for i := uint8(0); i < ownHandLen; i++ {
		nplayerSetBit(mask, NPlayerEncodeSnapOpponentMove(i))
	}
}
