package engine

import "fmt"

// ApplyNPlayerAction applies an action by N-player index (452-action space).
// Returns an error if the action is illegal or the state is invalid.
//
// For 2-player games, ApplyAction (146-action space) remains the preferred path.
// ApplyNPlayerAction works for any number of players.
func (g *GameState) ApplyNPlayerAction(actionIdx uint16) error {
	if g.IsGameOver() {
		return fmt.Errorf("game is already over")
	}

	// Snap move resolution takes priority.
	if g.Pending.Type == PendingSnapMove {
		if ownIdx, ok := NPlayerDecodeSnapOpponentMove(actionIdx); ok {
			return g.nplayerSnapOpponentMove(ownIdx)
		}
		return fmt.Errorf("pending snap move: expected NPlayerSnapOpponentMove action, got %d", actionIdx)
	}

	// Snap phase actions.
	if g.Snap.Active {
		if actionIdx == NPlayerActionPassSnap {
			return g.passSnap()
		}
		if slot, ok := NPlayerDecodeSnapOwn(actionIdx); ok {
			return g.snapOwn(slot)
		}
		if slot, oppRelIdx, ok := NPlayerDecodeSnapOpponent(actionIdx); ok {
			return g.nplayerSnapOpponent(slot, oppRelIdx)
		}
		return fmt.Errorf("snap phase active: unhandled N-player action index %d", actionIdx)
	}

	switch actionIdx {
	case NPlayerActionDrawStockpile:
		return g.drawStockpile()
	case NPlayerActionDrawDiscard:
		return g.drawDiscard()
	case NPlayerActionCallCambia:
		return g.callCambiaNPlayer()
	case NPlayerActionDiscardNoAbility:
		return g.discardDrawn()
	case NPlayerActionDiscardWithAbility:
		if g.Rules.numPlayers() > 2 {
			return g.discardWithAbilityNPlayer()
		}
		return g.discardWithAbility()
	case NPlayerActionKingSwapNo:
		if g.Rules.numPlayers() > 2 {
			return g.kingSwapDecisionNPlayer(false)
		}
		return g.kingSwapDecision(false)
	case NPlayerActionKingSwapYes:
		if g.Rules.numPlayers() > 2 {
			return g.kingSwapDecisionNPlayer(true)
		}
		return g.kingSwapDecision(true)
	default:
		if slot, ok := NPlayerDecodeReplace(actionIdx); ok {
			return g.replace(slot)
		}
		if slot, ok := NPlayerDecodePeekOwn(actionIdx); ok {
			return g.peekOwn(slot)
		}
		if slot, oppRelIdx, ok := NPlayerDecodePeekOther(actionIdx); ok {
			target, err := g.oppRelIdxToAbsolute(oppRelIdx)
			if err != nil {
				return err
			}
			if g.Rules.numPlayers() > 2 {
				return g.peekOtherNPlayer(slot, target)
			}
			return g.peekOther(slot)
		}
		if ownSlot, oppSlot, oppRelIdx, ok := NPlayerDecodeBlindSwap(actionIdx); ok {
			target, err := g.oppRelIdxToAbsolute(oppRelIdx)
			if err != nil {
				return err
			}
			if g.Rules.numPlayers() > 2 {
				return g.blindSwapNPlayer(ownSlot, oppSlot, target)
			}
			return g.blindSwap(ownSlot, oppSlot)
		}
		if ownSlot, oppSlot, oppRelIdx, ok := NPlayerDecodeKingLook(actionIdx); ok {
			target, err := g.oppRelIdxToAbsolute(oppRelIdx)
			if err != nil {
				return err
			}
			if g.Rules.numPlayers() > 2 {
				return g.kingLookNPlayer(ownSlot, oppSlot, target)
			}
			return g.kingLook(ownSlot, oppSlot)
		}
		return fmt.Errorf("unhandled N-player action index %d", actionIdx)
	}
}

// oppRelIdxToAbsolute converts a relative opponent index (0-based among Opponents(acting))
// to an absolute player index.
func (g *GameState) oppRelIdxToAbsolute(oppRelIdx uint8) (uint8, error) {
	acting := g.Pending.PlayerID
	if g.Snap.Active {
		acting = g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	}
	opps := g.Opponents(acting)
	if int(oppRelIdx) >= len(opps) {
		return 0, fmt.Errorf("oppRelIdx %d out of range (numOpponents=%d)", oppRelIdx, len(opps))
	}
	return opps[oppRelIdx], nil
}

// callCambiaNPlayer handles CallCambia using the N-player round calculation.
func (g *GameState) callCambiaNPlayer() error {
	if g.Pending.Type != PendingNone {
		return fmt.Errorf("cannot call Cambia with a pending action (type %d)", g.Pending.Type)
	}
	if g.IsCambiaCalled() {
		return fmt.Errorf("Cambia has already been called")
	}
	currentRound := g.TurnNumber / uint16(g.Rules.numPlayers())
	if currentRound < uint16(g.Rules.CambiaAllowedRound) {
		return fmt.Errorf("cannot call Cambia before round %d (current round %d)", g.Rules.CambiaAllowedRound, currentRound)
	}

	g.CambiaCaller = int8(g.CurrentPlayer)
	g.Flags |= FlagCambiaCalled

	g.LastAction.ActionIdx = NPlayerActionCallCambia
	g.LastAction.ActingPlayer = g.CurrentPlayer

	g.advanceTurn()
	return nil
}

// nplayerSnapOpponent handles snapping an opponent's card in N-player context.
// oppRelIdx is the relative opponent index (0-based among Opponents(snapper)).
func (g *GameState) nplayerSnapOpponent(slot, oppRelIdx uint8) error {
	if !g.Snap.Active {
		return fmt.Errorf("snap phase is not active")
	}
	if !g.Rules.AllowOpponentSnapping {
		return fmt.Errorf("opponent snapping is not allowed by house rules")
	}

	snapperIdx := g.Snap.Snappers[g.Snap.CurrentSnapperIdx]
	opps := g.Opponents(snapperIdx)
	if int(oppRelIdx) >= len(opps) {
		return fmt.Errorf("nplayerSnapOpponent: oppRelIdx %d out of range (numOpponents=%d)", oppRelIdx, len(opps))
	}
	opponent := opps[oppRelIdx]
	oppHandLen := g.Players[opponent].HandLen

	g.LastAction.ActionIdx = NPlayerEncodeSnapOpponent(slot, oppRelIdx)
	g.LastAction.ActingPlayer = snapperIdx

	if g.Players[snapperIdx].HandLen == 0 {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
		return nil
	}

	if slot >= oppHandLen {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
		return nil
	}

	card := g.Players[opponent].Hand[slot]
	if card.Rank() == g.Snap.DiscardedRank {
		g.removeCardFromHand(opponent, slot)
		g.DiscardPile[g.DiscardLen] = card
		g.DiscardLen++

		g.LastAction.SnapSuccess = true
		g.LastAction.RevealedCard = card
		g.LastAction.RevealedIdx = slot
		g.LastAction.RevealedOwner = opponent

		g.Pending.Type = PendingSnapMove
		g.Pending.PlayerID = snapperIdx
		g.Pending.Data[0] = opponent
		g.Pending.Data[1] = slot

		if g.Rules.SnapRace {
			g.endSnapPhase()
			return nil
		}
	} else {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(snapperIdx)
		g.advanceSnapper()
	}

	return nil
}

// nplayerSnapOpponentMove moves snapper's card to fill the vacated opponent slot.
// In N-player, SnapOpponentMove only encodes ownIdx (not the slot — slot is in Pending.Data[1]).
func (g *GameState) nplayerSnapOpponentMove(ownIdx uint8) error {
	if g.Pending.Type != PendingSnapMove {
		return fmt.Errorf("no pending snap move action")
	}

	snapperIdx := g.Pending.PlayerID
	opponent := g.Pending.Data[0]
	slotIdx := g.Pending.Data[1]
	oppHandLen := g.Players[opponent].HandLen

	g.LastAction.ActionIdx = NPlayerEncodeSnapOpponentMove(ownIdx)
	g.LastAction.ActingPlayer = snapperIdx

	snapperHandLen := g.Players[snapperIdx].HandLen
	if ownIdx >= snapperHandLen {
		return fmt.Errorf("nplayerSnapOpponentMove: own index %d out of range (hand size %d)", ownIdx, snapperHandLen)
	}
	if slotIdx > oppHandLen {
		return fmt.Errorf("nplayerSnapOpponentMove: slot index %d out of range (opp hand size %d)", slotIdx, oppHandLen)
	}
	if oppHandLen >= MaxHandSize {
		return fmt.Errorf("nplayerSnapOpponentMove: opponent hand is full (%d)", oppHandLen)
	}

	card := g.removeCardFromHand(snapperIdx, ownIdx)

	for i := oppHandLen; i > slotIdx; i-- {
		g.Players[opponent].Hand[i] = g.Players[opponent].Hand[i-1]
	}
	g.Players[opponent].Hand[slotIdx] = card
	g.Players[opponent].HandLen++

	g.Pending = PendingAction{}

	if g.Rules.SnapRace {
		g.endSnapPhase()
		return nil
	}

	g.advanceSnapper()
	return nil
}
