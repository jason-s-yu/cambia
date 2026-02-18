// engine_adapter.go — Bridge between engine.GameState and CambiaGame.
package game

import (
	"log"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// CardUUIDTracker mirrors engine card positions with UUIDs for client communication.
// Updated in lockstep with every engine action.
type CardUUIDTracker struct {
	Players    [engine.MaxPlayers]PlayerUUIDState
	StockUUIDs [engine.DeckSize]uuid.UUID
	StockLen   uint8
	DiscardUUIDs [engine.DeckSize]uuid.UUID
	DiscardLen  uint8

	// Registry maps UUID -> full card details for event payloads.
	Registry map[uuid.UUID]*models.Card
}

// PlayerUUIDState holds UUID tracking for a single player's cards.
type PlayerUUIDState struct {
	HandUUIDs     [engine.MaxHandSize]uuid.UUID
	DrawnCardUUID uuid.UUID // UUID of drawn card in PendingDiscard, if any.
}

// engineRankToString converts an engine rank uint8 to service rank string.
func engineRankToString(rank uint8) string {
	switch rank {
	case engine.RankAce:   return "A"
	case engine.RankTwo:   return "2"
	case engine.RankThree: return "3"
	case engine.RankFour:  return "4"
	case engine.RankFive:  return "5"
	case engine.RankSix:   return "6"
	case engine.RankSeven: return "7"
	case engine.RankEight: return "8"
	case engine.RankNine:  return "9"
	case engine.RankTen:   return "T"
	case engine.RankJack:  return "J"
	case engine.RankQueen: return "Q"
	case engine.RankKing:  return "K"
	case engine.RankJoker: return "O"
	default:               return "?"
	}
}

// engineSuitToString converts an engine suit uint8 to service suit string.
func engineSuitToString(suit uint8) string {
	switch suit {
	case engine.SuitHearts:      return "H"
	case engine.SuitDiamonds:    return "D"
	case engine.SuitClubs:       return "C"
	case engine.SuitSpades:      return "S"
	case engine.SuitRedJoker:    return "R"
	case engine.SuitBlackJoker:  return "B"
	default:                     return "?"
	}
}

// engineCardToDetails converts an engine.Card to a service *models.Card with the given UUID.
func engineCardToDetails(c engine.Card, id uuid.UUID) *models.Card {
	return &models.Card{
		ID:    id,
		Rank:  engineRankToString(c.Rank()),
		Suit:  engineSuitToString(c.Suit()),
		Value: int(c.Value()),
	}
}

// mapHouseRulesToEngine maps service HouseRules to engine.HouseRules.
func (g *CambiaGame) mapHouseRulesToEngine() engine.HouseRules {
	penaltyCount := uint8(g.HouseRules.PenaltyDrawCount)
	if penaltyCount == 0 {
		penaltyCount = 2
	}
	return engine.HouseRules{
		MaxGameTurns:          46,
		CardsPerPlayer:        4,
		CambiaAllowedRound:    0,
		PenaltyDrawCount:      penaltyCount,
		AllowDrawFromDiscard:  g.HouseRules.AllowDrawFromDiscardPile,
		AllowReplaceAbilities: g.HouseRules.AllowReplaceAbilities,
		AllowOpponentSnapping: true,
		SnapRace:              g.HouseRules.SnapRace,
	}
}

// initCardTracker assigns UUIDs to all cards currently in the engine after Deal().
func (g *CambiaGame) initCardTracker() {
	tracker := &g.CardTracker
	tracker.Registry = make(map[uuid.UUID]*models.Card)

	// Assign UUIDs to player hands.
	for p := uint8(0); p < engine.MaxPlayers; p++ {
		for i := uint8(0); i < g.Engine.Players[p].HandLen; i++ {
			id, _ := uuid.NewRandom()
			card := g.Engine.Players[p].Hand[i]
			tracker.Players[p].HandUUIDs[i] = id
			tracker.Registry[id] = engineCardToDetails(card, id)
		}
	}

	// Assign UUIDs to stockpile.
	tracker.StockLen = g.Engine.StockLen
	for i := uint8(0); i < g.Engine.StockLen; i++ {
		id, _ := uuid.NewRandom()
		card := g.Engine.Stockpile[i]
		tracker.StockUUIDs[i] = id
		tracker.Registry[id] = engineCardToDetails(card, id)
	}

	// Assign UUIDs to discard pile.
	tracker.DiscardLen = g.Engine.DiscardLen
	for i := uint8(0); i < g.Engine.DiscardLen; i++ {
		id, _ := uuid.NewRandom()
		card := g.Engine.DiscardPile[i]
		tracker.DiscardUUIDs[i] = id
		tracker.Registry[id] = engineCardToDetails(card, id)
	}
}

// updateCardTracker updates the UUID tracker after an engine action is applied.
// preStock/preDiscard are the stockpile/discard lengths before the action.
func (g *CambiaGame) updateCardTracker(actionIdx uint16, actorEngineIdx uint8, preStockLen uint8, preDiscardLen uint8) {
	tracker := &g.CardTracker

	switch {
	case actionIdx == engine.ActionDrawStockpile:
		// Stock top -> player's DrawnCard (pending).
		if preStockLen > 0 {
			drawnUUID := tracker.StockUUIDs[preStockLen-1]
			tracker.Players[actorEngineIdx].DrawnCardUUID = drawnUUID
			tracker.StockLen = g.Engine.StockLen
		}

	case actionIdx == engine.ActionDrawDiscard:
		// Discard top -> player's DrawnCard (pending).
		if preDiscardLen > 0 {
			drawnUUID := tracker.DiscardUUIDs[preDiscardLen-1]
			tracker.Players[actorEngineIdx].DrawnCardUUID = drawnUUID
			tracker.DiscardLen = g.Engine.DiscardLen
		}

	case actionIdx == engine.ActionDiscardNoAbility || actionIdx == engine.ActionDiscardWithAbility:
		// Pending -> discard top.
		drawnUUID := tracker.Players[actorEngineIdx].DrawnCardUUID
		newDiscardLen := g.Engine.DiscardLen
		if newDiscardLen > 0 {
			tracker.DiscardUUIDs[newDiscardLen-1] = drawnUUID
		}
		tracker.DiscardLen = newDiscardLen
		tracker.Players[actorEngineIdx].DrawnCardUUID = uuid.Nil

	case actionIdx == engine.ActionCallCambia:
		// No card movement.

	default:
		if targetIdx, ok := engine.ActionIsReplace(actionIdx); ok {
			// Replace: Pending -> Hand[targetIdx], Hand[targetIdx] -> Discard.
			drawnUUID := tracker.Players[actorEngineIdx].DrawnCardUUID
			oldHandUUID := tracker.Players[actorEngineIdx].HandUUIDs[targetIdx]
			// Old hand card goes to discard.
			newDiscardLen := g.Engine.DiscardLen
			if newDiscardLen > 0 {
				tracker.DiscardUUIDs[newDiscardLen-1] = oldHandUUID
			}
			tracker.DiscardLen = newDiscardLen
			// Drawn card goes to hand slot.
			tracker.Players[actorEngineIdx].HandUUIDs[targetIdx] = drawnUUID
			tracker.Players[actorEngineIdx].DrawnCardUUID = uuid.Nil

		} else if _, ok := engine.ActionIsPeekOwn(actionIdx); ok {
			// No card movement.

		} else if _, ok := engine.ActionIsPeekOther(actionIdx); ok {
			// No card movement.

		} else if ownIdx, oppIdx, ok := engine.ActionIsBlindSwap(actionIdx); ok {
			// Determine the opponent engine index.
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			// Swap UUIDs.
			tracker.Players[actorEngineIdx].HandUUIDs[ownIdx], tracker.Players[oppEngineIdx].HandUUIDs[oppIdx] =
				tracker.Players[oppEngineIdx].HandUUIDs[oppIdx], tracker.Players[actorEngineIdx].HandUUIDs[ownIdx]

		} else if _, _, ok := engine.ActionIsKingLook(actionIdx); ok {
			// No card movement (peek only).

		} else if actionIdx == engine.ActionKingSwapNo {
			// No card movement.

		} else if actionIdx == engine.ActionKingSwapYes {
			// Swap based on Pending.Data from BEFORE the action (stored in LastAction).
			ownIdx := g.Engine.LastAction.SwapOwnIdx
			oppIdx := g.Engine.LastAction.SwapOppIdx
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			tracker.Players[actorEngineIdx].HandUUIDs[ownIdx], tracker.Players[oppEngineIdx].HandUUIDs[oppIdx] =
				tracker.Players[oppEngineIdx].HandUUIDs[oppIdx], tracker.Players[actorEngineIdx].HandUUIDs[ownIdx]

		} else if actionIdx == engine.ActionPassSnap {
			// No card movement.

		} else if targetIdx, ok := engine.ActionIsSnapOwn(actionIdx); ok {
			g.updateTrackerForSnap(actorEngineIdx, targetIdx, true, preStockLen)

		} else if targetIdx, ok := engine.ActionIsSnapOpponent(actionIdx); ok {
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			g.updateTrackerForSnapOpponent(actorEngineIdx, oppEngineIdx, targetIdx, preStockLen)

		} else if ownIdx, slotIdx, ok := engine.ActionIsSnapOpponentMove(actionIdx); ok {
			// Move own hand card to opponent's hand.
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			movedUUID := tracker.Players[actorEngineIdx].HandUUIDs[ownIdx]
			// Remove from own hand (shift).
			for i := int(ownIdx); i < int(g.Engine.Players[actorEngineIdx].HandLen); i++ {
				tracker.Players[actorEngineIdx].HandUUIDs[i] = tracker.Players[actorEngineIdx].HandUUIDs[i+1]
			}
			tracker.Players[actorEngineIdx].HandUUIDs[g.Engine.Players[actorEngineIdx].HandLen] = uuid.Nil
			// Insert into opponent's hand at slotIdx.
			oppHandLen := g.Engine.Players[oppEngineIdx].HandLen
			for i := int(oppHandLen) - 1; i >= int(slotIdx); i-- {
				tracker.Players[oppEngineIdx].HandUUIDs[i+1] = tracker.Players[oppEngineIdx].HandUUIDs[i]
			}
			tracker.Players[oppEngineIdx].HandUUIDs[slotIdx] = movedUUID
		}
	}
}

// updateTrackerForSnap updates UUID positions for a snap of own card.
func (g *CambiaGame) updateTrackerForSnap(snapperIdx uint8, targetIdx uint8, isOwnSnap bool, preStockLen uint8) {
	tracker := &g.CardTracker
	snapSuccess := g.Engine.LastAction.SnapSuccess
	penaltyCount := g.Engine.LastAction.SnapPenalty

	if snapSuccess {
		// Card at targetIdx moves to discard.
		snappedUUID := tracker.Players[snapperIdx].HandUUIDs[targetIdx]
		handLen := g.Engine.Players[snapperIdx].HandLen
		// Shift UUIDs left.
		for i := int(targetIdx); i < int(handLen); i++ {
			tracker.Players[snapperIdx].HandUUIDs[i] = tracker.Players[snapperIdx].HandUUIDs[i+1]
		}
		tracker.Players[snapperIdx].HandUUIDs[handLen] = uuid.Nil
		// Add to discard.
		newDiscardLen := g.Engine.DiscardLen
		if newDiscardLen > 0 {
			tracker.DiscardUUIDs[newDiscardLen-1] = snappedUUID
		}
		tracker.DiscardLen = newDiscardLen
	} else {
		// Failed snap: penalty cards drawn from stockpile.
		handLen := g.Engine.Players[snapperIdx].HandLen
		oldHandLen := handLen - penaltyCount
		for i := uint8(0); i < penaltyCount; i++ {
			stockIdx := preStockLen - 1 - i
			if stockIdx < preStockLen { // bounds check
				penaltyUUID := tracker.StockUUIDs[stockIdx]
				tracker.Players[snapperIdx].HandUUIDs[oldHandLen+i] = penaltyUUID
				// Register the card in Registry if not already there.
			}
		}
		tracker.StockLen = g.Engine.StockLen
	}
}

// updateTrackerForSnapOpponent updates UUID positions for a snap of opponent's card.
func (g *CambiaGame) updateTrackerForSnapOpponent(snapperIdx uint8, oppIdx uint8, targetIdx uint8, preStockLen uint8) {
	tracker := &g.CardTracker
	snapSuccess := g.Engine.LastAction.SnapSuccess
	penaltyCount := g.Engine.LastAction.SnapPenalty

	if snapSuccess {
		// Opponent's card at targetIdx moves to discard.
		snappedUUID := tracker.Players[oppIdx].HandUUIDs[targetIdx]
		oppHandLen := g.Engine.Players[oppIdx].HandLen
		// Shift opponent's UUIDs left.
		for i := int(targetIdx); i < int(oppHandLen); i++ {
			tracker.Players[oppIdx].HandUUIDs[i] = tracker.Players[oppIdx].HandUUIDs[i+1]
		}
		tracker.Players[oppIdx].HandUUIDs[oppHandLen] = uuid.Nil
		// Add to discard.
		newDiscardLen := g.Engine.DiscardLen
		if newDiscardLen > 0 {
			tracker.DiscardUUIDs[newDiscardLen-1] = snappedUUID
		}
		tracker.DiscardLen = newDiscardLen
	} else {
		// Failed snap penalty: cards go to snapper's hand.
		handLen := g.Engine.Players[snapperIdx].HandLen
		oldHandLen := handLen - penaltyCount
		for i := uint8(0); i < penaltyCount; i++ {
			stockIdx := preStockLen - 1 - i
			if stockIdx < preStockLen {
				penaltyUUID := tracker.StockUUIDs[stockIdx]
				tracker.Players[snapperIdx].HandUUIDs[oldHandLen+i] = penaltyUUID
			}
		}
		tracker.StockLen = g.Engine.StockLen
	}
}

// syncPlayerHandsFromEngine updates service Player.Hand from engine state.
// Called after Deal() and after each action to keep Player model in sync.
func (g *CambiaGame) syncPlayerHandsFromEngine() {
	for i, p := range g.Players {
		engineIdx, ok := g.PlayerToEngine[p.ID]
		if !ok {
			continue
		}
		handLen := int(g.Engine.Players[engineIdx].HandLen)
		p.Hand = make([]*models.Card, handLen)
		for j := 0; j < handLen; j++ {
			cardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[j]
			if c, exists := g.CardTracker.Registry[cardUUID]; exists {
				p.Hand[j] = c
			}
		}
		g.Players[i] = p
	}
}

// currentPlayerID returns the UUID of the current acting player.
func (g *CambiaGame) currentPlayerID() uuid.UUID {
	actingIdx := g.Engine.ActingPlayer()
	return g.EngineToPlayer[actingIdx]
}

// isCambiaCalled returns true if Cambia has been called.
func (g *CambiaGame) isCambiaCalled() bool {
	return g.Engine.IsCambiaCalled()
}

// cambiaCallerID returns the UUID of the player who called Cambia, or uuid.Nil.
func (g *CambiaGame) cambiaCallerID() uuid.UUID {
	caller := g.Engine.CambiaCaller
	if caller < 0 {
		return uuid.Nil
	}
	return g.EngineToPlayer[uint8(caller)]
}

// stockpileSize returns current stockpile size from engine.
func (g *CambiaGame) stockpileSize() int {
	return int(g.Engine.StockLen)
}

// discardSize returns current discard pile size from engine.
func (g *CambiaGame) discardSize() int {
	return int(g.Engine.DiscardLen)
}

// discardTopCard returns the top discard card and its UUID, or nil if empty.
func (g *CambiaGame) discardTopCard() (*models.Card, uuid.UUID) {
	if g.Engine.DiscardLen == 0 {
		return nil, uuid.Nil
	}
	topIdx := g.Engine.DiscardLen - 1
	topCard := g.Engine.DiscardPile[topIdx]
	topUUID := g.CardTracker.DiscardUUIDs[topIdx]
	if topUUID == uuid.Nil {
		return nil, uuid.Nil
	}
	return engineCardToDetails(topCard, topUUID), topUUID
}

// applyEngineAction applies an engine action, updates UUID tracker, emits events.
// Returns any error from the engine.
func (g *CambiaGame) applyEngineAction(actionIdx uint16, actorID uuid.UUID) error {
	engineIdx, ok := g.PlayerToEngine[actorID]
	if !ok {
		engineIdx = g.Engine.ActingPlayer()
	}

	// Snapshot pre-action state for diffing.
	preStockLen := g.Engine.StockLen
	preDiscardLen := g.Engine.DiscardLen

	// Apply to engine.
	if err := g.Engine.ApplyAction(actionIdx); err != nil {
		log.Printf("Game %s: Engine error for action %d: %v", g.ID, actionIdx, err)
		g.fireEventToPlayer(actorID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": err.Error()},
		})
		return err
	}

	// Update UUID tracker.
	g.updateCardTracker(actionIdx, engineIdx, preStockLen, preDiscardLen)

	// Sync Player model hands (keeps service-level code working).
	g.syncPlayerHandsFromEngine()

	// Emit WebSocket events.
	g.emitEventsForAction(actionIdx, actorID, engineIdx, preStockLen, preDiscardLen)

	// Check for game end.
	if g.Engine.IsTerminal() {
		g.EndGame()
		return nil
	}

	// Handle snap phase.
	if g.Engine.Snap.Active {
		g.autoProcessSnapPhase()
		// Re-check terminal after snap phase resolution (advanceTurn may trigger game end).
		if g.Engine.IsTerminal() {
			g.EndGame()
			return nil
		}
	} else if !g.Engine.Snap.Active && g.Engine.Pending.Type == engine.PendingNone {
		// Check if this action ended a turn.
		switch actionIdx {
		case engine.ActionDrawStockpile, engine.ActionDrawDiscard:
			// Turn not over; player still needs to discard/replace.
		case engine.ActionCallCambia:
			g.onTurnAdvanced()
		default:
			if _, ok := engine.ActionIsReplace(actionIdx); ok {
				g.onTurnAdvanced()
			} else if actionIdx == engine.ActionDiscardNoAbility {
				g.onTurnAdvanced()
			} else if actionIdx == engine.ActionKingSwapNo || actionIdx == engine.ActionKingSwapYes {
				g.onTurnAdvanced()
			}
			// PeekOwn, PeekOther, BlindSwap: turn advances after ability.
			if _, ok := engine.ActionIsPeekOwn(actionIdx); ok {
				g.onTurnAdvanced()
			} else if _, ok := engine.ActionIsPeekOther(actionIdx); ok {
				g.onTurnAdvanced()
			} else if _, _, ok := engine.ActionIsBlindSwap(actionIdx); ok {
				g.onTurnAdvanced()
			}
		}
	}

	return nil
}

// emitEventsForAction sends the appropriate WebSocket events for a completed engine action.
func (g *CambiaGame) emitEventsForAction(actionIdx uint16, actorID uuid.UUID, actorEngineIdx uint8, preStockLen uint8, preDiscardLen uint8) {
	switch {
	case actionIdx == engine.ActionDrawStockpile:
		// Public draw event (card ID only).
		drawnUUID := g.CardTracker.Players[actorEngineIdx].DrawnCardUUID
		g.fireEvent(GameEvent{
			Type: EventPlayerDrawStockpile,
			User: &EventUser{ID: actorID},
			Card: &EventCard{ID: drawnUUID},
			Payload: map[string]interface{}{
				"stockpileSize": g.Engine.StockLen,
				"source":        "stockpile",
			},
		})
		// Private draw event (full details).
		drawnCard := g.CardTracker.Registry[drawnUUID]
		if drawnCard != nil {
			g.fireEventToPlayer(actorID, GameEvent{
				Type: EventPrivateDrawStockpile,
				Card: &EventCard{ID: drawnUUID, Rank: drawnCard.Rank, Suit: drawnCard.Suit, Value: drawnCard.Value},
				Payload: map[string]interface{}{"source": "stockpile"},
			})
		}
		g.logAction(actorID, string(EventPlayerDrawStockpile), map[string]interface{}{
			"cardId": drawnUUID, "newSize": g.Engine.StockLen,
		})

	case actionIdx == engine.ActionDrawDiscard:
		// Public draw event (full details since came from discard).
		drawnUUID := g.CardTracker.Players[actorEngineIdx].DrawnCardUUID
		drawnCard := g.CardTracker.Registry[drawnUUID]
		if drawnCard != nil {
			g.fireEvent(GameEvent{
				Type: EventPlayerDrawStockpile,
				User: &EventUser{ID: actorID},
				Card: &EventCard{ID: drawnUUID, Rank: drawnCard.Rank, Suit: drawnCard.Suit, Value: drawnCard.Value},
				Payload: map[string]interface{}{
					"source":      "discardpile",
					"discardSize": g.Engine.DiscardLen,
				},
			})
			g.fireEventToPlayer(actorID, GameEvent{
				Type: EventPrivateDrawStockpile,
				Card: &EventCard{ID: drawnUUID, Rank: drawnCard.Rank, Suit: drawnCard.Suit, Value: drawnCard.Value},
				Payload: map[string]interface{}{"source": "discardpile"},
			})
		}
		g.logAction(actorID, "action_draw_discardpile", map[string]interface{}{
			"cardId": drawnUUID, "newSize": g.Engine.DiscardLen,
		})

	case actionIdx == engine.ActionDiscardNoAbility || actionIdx == engine.ActionDiscardWithAbility:
		// Discard events are handled in handleDiscardViaEngine before applying.
		// The discard event was already fired in buffered flow for ability cards.
		// For non-ability cards, emit here.
		if actionIdx == engine.ActionDiscardNoAbility {
			discardLen := g.Engine.DiscardLen
			if discardLen > 0 {
				discardedUUID := g.CardTracker.DiscardUUIDs[discardLen-1]
				discardedCard := g.CardTracker.Registry[discardedUUID]
				if discardedCard != nil {
					g.fireEvent(GameEvent{
						Type: EventPlayerDiscard,
						User: &EventUser{ID: actorID},
						Card: &EventCard{ID: discardedUUID, Rank: discardedCard.Rank, Suit: discardedCard.Suit, Value: discardedCard.Value},
					})
				}
				g.logAction(actorID, string(EventPlayerDiscard), map[string]interface{}{"cardId": discardedUUID})
			}
		}

	case actionIdx == engine.ActionCallCambia:
		g.fireEvent(GameEvent{
			Type: EventPlayerCambia,
			User: &EventUser{ID: actorID},
		})
		g.logAction(actorID, string(EventPlayerCambia), nil)

	default:
		if targetIdx, ok := engine.ActionIsReplace(actionIdx); ok {
			// Replace: old hand card goes to discard.
			discardLen := g.Engine.DiscardLen
			if discardLen > 0 {
				discardedUUID := g.CardTracker.DiscardUUIDs[discardLen-1]
				discardedCard := g.CardTracker.Registry[discardedUUID]
				if discardedCard != nil {
					idx := int(targetIdx)
					g.fireEvent(GameEvent{
						Type: EventPlayerDiscard,
						User: &EventUser{ID: actorID},
						Card: &EventCard{ID: discardedUUID, Rank: discardedCard.Rank, Suit: discardedCard.Suit, Value: discardedCard.Value, Idx: &idx},
					})
					g.logAction(actorID, string(EventPlayerDiscard), map[string]interface{}{"cardId": discardedUUID, "index": targetIdx})
				}
			}

		} else if targetIdx, ok := engine.ActionIsPeekOwn(actionIdx); ok {
			cardUUID := g.CardTracker.Players[actorEngineIdx].HandUUIDs[targetIdx]
			card := g.CardTracker.Registry[cardUUID]
			if card != nil {
				idx := int(targetIdx)
				// Private: reveal details.
				g.fireEventToPlayer(actorID, GameEvent{
					Type:    EventPrivateSpecialSuccess,
					Special: "peek_self",
					Card1:   &EventCard{ID: cardUUID, Rank: card.Rank, Suit: card.Suit, Value: card.Value, Idx: &idx, User: &EventUser{ID: actorID}},
				})
				// Public: show index only.
				g.fireEvent(GameEvent{
					Type:    EventPlayerSpecialAction,
					User:    &EventUser{ID: actorID},
					Special: "peek_self",
					Card1:   &EventCard{ID: cardUUID, Idx: &idx, User: &EventUser{ID: actorID}},
				})
				g.logAction(actorID, "action_special_peek_self", map[string]interface{}{"cardId": cardUUID, "idx": targetIdx})
			}

		} else if targetIdx, ok := engine.ActionIsPeekOther(actionIdx); ok {
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			oppID := g.EngineToPlayer[oppEngineIdx]
			cardUUID := g.CardTracker.Players[oppEngineIdx].HandUUIDs[targetIdx]
			card := g.CardTracker.Registry[cardUUID]
			if card != nil {
				idx := int(targetIdx)
				// Private: reveal to peeker.
				g.fireEventToPlayer(actorID, GameEvent{
					Type:    EventPrivateSpecialSuccess,
					Special: "peek_other",
					Card1:   &EventCard{ID: cardUUID, Rank: card.Rank, Suit: card.Suit, Value: card.Value, Idx: &idx, User: &EventUser{ID: oppID}},
				})
				// Public: obfuscated.
				g.fireEvent(GameEvent{
					Type:    EventPlayerSpecialAction,
					User:    &EventUser{ID: actorID},
					Special: "peek_other",
					Card1:   &EventCard{ID: cardUUID, Idx: &idx, User: &EventUser{ID: oppID}},
				})
				g.logAction(actorID, "action_special_peek_other", map[string]interface{}{"cardId": cardUUID, "idx": targetIdx, "targetPlayer": oppID})
			}

		} else if ownIdx, oppIdx, ok := engine.ActionIsBlindSwap(actionIdx); ok {
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			oppID := g.EngineToPlayer[oppEngineIdx]
			// After swap, UUIDs are already swapped in tracker.
			ownCardUUID := g.CardTracker.Players[actorEngineIdx].HandUUIDs[ownIdx]
			oppCardUUID := g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppIdx]
			ownIdxInt := int(ownIdx)
			oppIdxInt := int(oppIdx)
			g.fireEvent(GameEvent{
				Type:    EventPlayerSpecialAction,
				User:    &EventUser{ID: actorID},
				Special: "swap_blind",
				Card1:   &EventCard{ID: ownCardUUID, Idx: &ownIdxInt, User: &EventUser{ID: actorID}},
				Card2:   &EventCard{ID: oppCardUUID, Idx: &oppIdxInt, User: &EventUser{ID: oppID}},
			})
			g.logAction(actorID, "action_special_swap_blind", map[string]interface{}{
				"ownIdx": ownIdx, "oppIdx": oppIdx,
			})

		} else if ownIdx, oppIdx, ok := engine.ActionIsKingLook(actionIdx); ok {
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			oppID := g.EngineToPlayer[oppEngineIdx]
			ownCardUUID := g.CardTracker.Players[actorEngineIdx].HandUUIDs[ownIdx]
			oppCardUUID := g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppIdx]
			ownCard := g.CardTracker.Registry[ownCardUUID]
			oppCard := g.CardTracker.Registry[oppCardUUID]
			if ownCard != nil && oppCard != nil {
				ownIdxInt := int(ownIdx)
				oppIdxInt := int(oppIdx)
				// Private: reveal both cards.
				g.fireEventToPlayer(actorID, GameEvent{
					Type:    EventPrivateSpecialSuccess,
					Special: "swap_peek_reveal",
					Card1:   &EventCard{ID: ownCardUUID, Rank: ownCard.Rank, Suit: ownCard.Suit, Value: ownCard.Value, Idx: &ownIdxInt, User: &EventUser{ID: actorID}},
					Card2:   &EventCard{ID: oppCardUUID, Rank: oppCard.Rank, Suit: oppCard.Suit, Value: oppCard.Value, Idx: &oppIdxInt, User: &EventUser{ID: oppID}},
				})
				// Public: obfuscated.
				g.fireEvent(GameEvent{
					Type:    EventPlayerSpecialAction,
					User:    &EventUser{ID: actorID},
					Special: "swap_peek_reveal",
					Card1:   &EventCard{ID: ownCardUUID, Idx: &ownIdxInt, User: &EventUser{ID: actorID}},
					Card2:   &EventCard{ID: oppCardUUID, Idx: &oppIdxInt, User: &EventUser{ID: oppID}},
				})
				g.logAction(actorID, "action_special_swap_peek_reveal", map[string]interface{}{
					"ownIdx": ownIdx, "oppIdx": oppIdx,
				})
			}

		} else if actionIdx == engine.ActionKingSwapYes {
			ownIdx := g.Engine.LastAction.SwapOwnIdx
			oppIdx := g.Engine.LastAction.SwapOppIdx
			oppEngineIdx := uint8(1 - int(actorEngineIdx))
			oppID := g.EngineToPlayer[oppEngineIdx]
			// After swap, UUIDs already updated.
			ownCardUUID := g.CardTracker.Players[actorEngineIdx].HandUUIDs[ownIdx]
			oppCardUUID := g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppIdx]
			ownIdxInt := int(ownIdx)
			oppIdxInt := int(oppIdx)
			g.fireEvent(GameEvent{
				Type:    EventPlayerSpecialAction,
				User:    &EventUser{ID: actorID},
				Special: "swap_peek_swap",
				Card1:   &EventCard{ID: ownCardUUID, Idx: &ownIdxInt, User: &EventUser{ID: actorID}},
				Card2:   &EventCard{ID: oppCardUUID, Idx: &oppIdxInt, User: &EventUser{ID: oppID}},
			})
			g.logAction(actorID, "action_special_swap_peek_swap", nil)

		} else if actionIdx == engine.ActionKingSwapNo {
			// No public event for skip.
			g.logAction(actorID, "action_special_king_no_swap", nil)
		}
	}
}

// handleDiscardViaEngine processes a discard action with the buffered ability-choice flow.
func (g *CambiaGame) handleDiscardViaEngine(playerID uuid.UUID, engineIdx uint8, payload map[string]interface{}) {
	// Validate engine state: player must have a drawn card pending.
	if g.Engine.Pending.Type != engine.PendingDiscard || g.Engine.Pending.PlayerID != engineIdx {
		log.Printf("Game %s: Player %s discard ignored — no pending drawn card in engine.", g.ID, playerID)
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "You must draw a card first."},
		})
		return
	}

	// Validate card ID from payload.
	cardIDStr, _ := payload["id"].(string)
	cardID, err := uuid.Parse(cardIDStr)
	if err != nil {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Invalid card ID for discard."},
		})
		return
	}

	// Verify card ID matches the drawn card.
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	if drawnUUID != cardID {
		log.Printf("Game %s: Player %s discard card ID mismatch. Expected %s, got %s.", g.ID, playerID, drawnUUID, cardID)
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Card ID mismatch for discard."},
		})
		return
	}

	drawnCard := engine.Card(g.Engine.Pending.Data[0])
	drawnFrom := g.Engine.Pending.Data[1]

	// Check if card has ability and was drawn from stockpile.
	hasAbility := drawnCard.HasAbility() && drawnFrom == engine.DrawnFromStockpile

	if hasAbility {
		// Buffer the discard — fire special choice event, wait for ability decision.
		g.pendingDiscardAbilityChoice = true
		g.pendingDiscardCardID = cardID

		rankStr := engineRankToString(drawnCard.Rank())
		specialType := rankToSpecial(rankStr)

		// Fire discard event (card goes to discard pile visually).
		g.fireEvent(GameEvent{
			Type: EventPlayerDiscard,
			User: &EventUser{ID: playerID},
			Card: &EventCard{ID: cardID, Rank: rankStr, Suit: engineSuitToString(drawnCard.Suit()), Value: int(drawnCard.Value())},
		})
		g.logAction(playerID, string(EventPlayerDiscard), map[string]interface{}{"cardId": cardID, "source": "drawn"})

		// Fire special choice event.
		g.fireEvent(GameEvent{
			Type:    EventPlayerSpecialChoice,
			User:    &EventUser{ID: playerID},
			Card:    &EventCard{ID: cardID, Rank: rankStr},
			Special: specialType,
		})
		g.logAction(playerID, string(EventPlayerSpecialChoice), map[string]interface{}{"cardId": cardID, "rank": rankStr, "special": specialType})

		// Activate special action state for backward compat with ProcessSpecialAction routing.
		g.SpecialAction = SpecialActionState{
			Active:   true,
			PlayerID: playerID,
			CardRank: rankStr,
		}

		g.ResetTurnTimer()
	} else {
		// No ability — apply directly.
		g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
	}
}

// handleReplaceViaEngine processes a replace action.
func (g *CambiaGame) handleReplaceViaEngine(playerID uuid.UUID, engineIdx uint8, payload map[string]interface{}) {
	// Validate engine state.
	if g.Engine.Pending.Type != engine.PendingDiscard || g.Engine.Pending.PlayerID != engineIdx {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "You must draw a card first."},
		})
		return
	}

	// Extract target index from payload.
	idxFloat, idxOK := payload["idx"].(float64)
	if !idxOK {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Missing index for replacement."},
		})
		return
	}
	targetIdx := uint8(int(idxFloat))

	// Validate index is within hand bounds.
	handLen := g.Engine.Players[engineIdx].HandLen
	if targetIdx >= handLen {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Invalid index for replacement."},
		})
		return
	}

	// Check if replace ability should trigger.
	if g.HouseRules.AllowReplaceAbilities {
		oldCard := g.Engine.Players[engineIdx].Hand[targetIdx]
		if oldCard.HasAbility() {
			// Buffer replace with ability — for simplicity, apply replace then trigger special.
			// Actually the engine handles this through ActionDiscardWithAbility flow for replace.
			// For now, apply replace (which puts old card to discard) and then trigger special.
			if err := g.applyEngineAction(engine.EncodeReplace(targetIdx), playerID); err != nil {
				return
			}
			// The replaced card (old one) is now on discard. Check if it triggers ability.
			discardLen := g.Engine.DiscardLen
			if discardLen > 0 {
				discardedUUID := g.CardTracker.DiscardUUIDs[discardLen-1]
				if discardedCard := g.CardTracker.Registry[discardedUUID]; discardedCard != nil {
					specialType := rankToSpecial(discardedCard.Rank)
					if specialType != "" {
						g.SpecialAction = SpecialActionState{
							Active:   true,
							PlayerID: playerID,
							CardRank: discardedCard.Rank,
						}
						g.fireEvent(GameEvent{
							Type:    EventPlayerSpecialChoice,
							User:    &EventUser{ID: playerID},
							Card:    &EventCard{ID: discardedUUID, Rank: discardedCard.Rank},
							Special: specialType,
						})
						g.ResetTurnTimer()
						return
					}
				}
			}
			return
		}
	}

	// Normal replace: no ability.
	g.applyEngineAction(engine.EncodeReplace(targetIdx), playerID)
}

// handleSnapViaEngine processes a snap action.
func (g *CambiaGame) handleSnapViaEngine(playerID uuid.UUID, engineIdx uint8, payload map[string]interface{}) {
	// Validate payload.
	cardIDStr, _ := payload["id"].(string)
	cardID, err := uuid.Parse(cardIDStr)
	if err != nil {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Invalid card ID format for snap."},
		})
		return
	}
	g.logAction(playerID, "action_snap_attempt", map[string]interface{}{"cardId": cardID})

	// Check discard pile.
	if g.Engine.DiscardLen == 0 {
		g.handleSnapFailure(playerID, engineIdx, nil)
		return
	}

	// Check SnapRace rule.
	if g.HouseRules.SnapRace && g.snapUsedForThisDiscard {
		g.handleSnapFailure(playerID, engineIdx, nil)
		return
	}

	discardTopRank := g.Engine.DiscardPile[g.Engine.DiscardLen-1].Rank()

	// Find card in own hand first.
	for i := uint8(0); i < g.Engine.Players[engineIdx].HandLen; i++ {
		if g.CardTracker.Players[engineIdx].HandUUIDs[i] == cardID {
			cardRank := g.Engine.Players[engineIdx].Hand[i].Rank()
			if cardRank == discardTopRank {
				// Successful snap from own hand.
				if g.HouseRules.SnapRace {
					g.snapUsedForThisDiscard = true
				}

				// Directly remove card from hand and add to discard pile.
				// Cannot use engine.ApplyAction since snap actions require snap phase.
				snapCard := g.Engine.Players[engineIdx].Hand[i]
				handLen := g.Engine.Players[engineIdx].HandLen
				// Shift remaining hand cards left.
				for k := i; k < handLen-1; k++ {
					g.Engine.Players[engineIdx].Hand[k] = g.Engine.Players[engineIdx].Hand[k+1]
					g.CardTracker.Players[engineIdx].HandUUIDs[k] = g.CardTracker.Players[engineIdx].HandUUIDs[k+1]
				}
				g.Engine.Players[engineIdx].Hand[handLen-1] = engine.EmptyCard
				g.CardTracker.Players[engineIdx].HandUUIDs[handLen-1] = uuid.Nil
				g.Engine.Players[engineIdx].HandLen--

				// Add card to discard pile.
				discardPos := g.Engine.DiscardLen
				g.Engine.DiscardPile[discardPos] = snapCard
				g.Engine.DiscardLen++
				g.CardTracker.DiscardUUIDs[discardPos] = cardID
				g.CardTracker.DiscardLen = g.Engine.DiscardLen
				g.snapUsedForThisDiscard = true

				g.syncPlayerHandsFromEngine()
				g.emitSnapSuccessEvents(playerID, cardID, cardRank, int(i))
				_ = snapCard
				return
			}
			// Found card but wrong rank — fail.
			g.handleSnapFailure(playerID, engineIdx, &cardID)
			return
		}
	}

	// Check opponent's hand.
	oppEngineIdx := uint8(1 - int(engineIdx))
	for i := uint8(0); i < g.Engine.Players[oppEngineIdx].HandLen; i++ {
		if g.CardTracker.Players[oppEngineIdx].HandUUIDs[i] == cardID {
			cardRank := g.Engine.Players[oppEngineIdx].Hand[i].Rank()
			if cardRank == discardTopRank {
				// Successful snap from opponent's hand.
				if g.HouseRules.SnapRace {
					g.snapUsedForThisDiscard = true
				}

				// Directly remove card from opponent's hand and add to discard pile.
				// Cannot use engine.ApplyAction since snap actions require snap phase.
				snapCard := g.Engine.Players[oppEngineIdx].Hand[i]
				oppHandLen := g.Engine.Players[oppEngineIdx].HandLen
				// Shift remaining hand cards left.
				for k := i; k < oppHandLen-1; k++ {
					g.Engine.Players[oppEngineIdx].Hand[k] = g.Engine.Players[oppEngineIdx].Hand[k+1]
					g.CardTracker.Players[oppEngineIdx].HandUUIDs[k] = g.CardTracker.Players[oppEngineIdx].HandUUIDs[k+1]
				}
				g.Engine.Players[oppEngineIdx].Hand[oppHandLen-1] = engine.EmptyCard
				g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppHandLen-1] = uuid.Nil
				g.Engine.Players[oppEngineIdx].HandLen--

				// Add card to discard pile.
				discardPos := g.Engine.DiscardLen
				g.Engine.DiscardPile[discardPos] = snapCard
				g.Engine.DiscardLen++
				g.CardTracker.DiscardUUIDs[discardPos] = cardID
				g.CardTracker.DiscardLen = g.Engine.DiscardLen
				g.snapUsedForThisDiscard = true

				g.syncPlayerHandsFromEngine()
				g.emitSnapSuccessEvents(playerID, cardID, cardRank, int(i))
				_ = snapCard
				return
			}
			g.handleSnapFailure(playerID, engineIdx, &cardID)
			return
		}
	}

	// Card not found in any hand.
	g.handleSnapFailure(playerID, engineIdx, nil)
}

// emitSnapSuccessEvents fires public snap success event.
func (g *CambiaGame) emitSnapSuccessEvents(playerID uuid.UUID, cardID uuid.UUID, rank uint8, idx int) {
	card := g.CardTracker.Registry[cardID]
	rankStr := engineRankToString(rank)
	ev := GameEvent{
		Type: EventPlayerSnapSuccess,
		User: &EventUser{ID: playerID},
		Card: &EventCard{ID: cardID, Rank: rankStr, Idx: &idx},
	}
	if card != nil {
		ev.Card.Suit = card.Suit
		ev.Card.Value = card.Value
	}
	g.fireEvent(ev)
	g.logAction(playerID, string(EventPlayerSnapSuccess), map[string]interface{}{"cardId": cardID, "rank": rankStr})
}

// handleSnapFailure processes a failed snap and applies penalties.
func (g *CambiaGame) handleSnapFailure(playerID uuid.UUID, engineIdx uint8, attemptedCardID *uuid.UUID) {
	log.Printf("Game %s: Player %s snap failed. Penalizing.", g.ID, playerID)
	if attemptedCardID != nil {
		g.logAction(playerID, string(EventPlayerSnapFail), map[string]interface{}{"attemptedCardId": *attemptedCardID})
	} else {
		g.logAction(playerID, string(EventPlayerSnapFail), nil)
	}

	// Broadcast public failure event.
	failEvent := GameEvent{
		Type: EventPlayerSnapFail,
		User: &EventUser{ID: playerID},
	}
	if attemptedCardID != nil && *attemptedCardID != uuid.Nil {
		card := g.CardTracker.Registry[*attemptedCardID]
		if card != nil {
			failEvent.Card = &EventCard{
				ID:    *attemptedCardID,
				Rank:  card.Rank,
				Suit:  card.Suit,
				Value: card.Value,
			}
		}
	}
	g.fireEvent(failEvent)

	// Apply penalty draws.
	penaltyCount := g.HouseRules.PenaltyDrawCount
	if penaltyCount <= 0 {
		return
	}
	log.Printf("Game %s: Applying %d penalty cards to player %s.", g.ID, penaltyCount, playerID)

	for i := 0; i < penaltyCount; i++ {
		if g.Engine.StockLen == 0 {
			break
		}
		if g.Engine.Players[engineIdx].HandLen >= engine.MaxHandSize {
			break // Hand is full — cannot add more penalty cards.
		}
		preStock := g.Engine.StockLen
		// Draw from stockpile into player's hand by modifying engine directly.
		// We use the engine's internal state since there's no ApplyAction for penalty in isolation.
		stockCard := g.Engine.Stockpile[preStock-1]
		g.Engine.StockLen--
		handLen := g.Engine.Players[engineIdx].HandLen
		g.Engine.Players[engineIdx].Hand[handLen] = stockCard
		g.Engine.Players[engineIdx].HandLen++

		// Update tracker.
		penaltyUUID := g.CardTracker.StockUUIDs[preStock-1]
		g.CardTracker.Players[engineIdx].HandUUIDs[handLen] = penaltyUUID
		g.CardTracker.StockLen = g.Engine.StockLen

		// Broadcast public penalty event.
		g.fireEvent(GameEvent{
			Type: EventPlayerSnapPenalty,
			User: &EventUser{ID: playerID},
			Card: &EventCard{ID: penaltyUUID},
			Payload: map[string]interface{}{
				"count": i + 1,
				"total": penaltyCount,
			},
		})

		// Private penalty event with card details.
		if penaltyCard := g.CardTracker.Registry[penaltyUUID]; penaltyCard == nil {
			// Register the card.
			g.CardTracker.Registry[penaltyUUID] = engineCardToDetails(stockCard, penaltyUUID)
		}
		penaltyCard := g.CardTracker.Registry[penaltyUUID]
		privateIdx := int(handLen)
		if penaltyCard != nil {
			g.fireEventToPlayer(playerID, GameEvent{
				Type: EventPrivateSnapPenalty,
				Card: &EventCard{ID: penaltyUUID, Idx: &privateIdx, Rank: penaltyCard.Rank, Suit: penaltyCard.Suit, Value: penaltyCard.Value},
				Payload: map[string]interface{}{
					"count": i + 1,
					"total": penaltyCount,
				},
			})
		}
	}

	// Sync player models.
	g.syncPlayerHandsFromEngine()
	g.logAction(playerID, "player_snap_penalty_applied", map[string]interface{}{"count": penaltyCount})
}

// autoProcessSnapPhase immediately passes all snappers through the engine's snap phase.
// This preserves the service's async snap model while satisfying the engine's sequential snap phase.
func (g *CambiaGame) autoProcessSnapPhase() {
	for g.Engine.Snap.Active {
		snapperEngineIdx := g.Engine.Snap.Snappers[g.Engine.Snap.CurrentSnapperIdx]
		snapperUUID := g.EngineToPlayer[snapperEngineIdx]

		preStock := g.Engine.StockLen
		preDiscard := g.Engine.DiscardLen

		if err := g.Engine.ApplyAction(engine.ActionPassSnap); err != nil {
			log.Printf("Game %s: Engine PassSnap error: %v", g.ID, err)
			break
		}
		g.updateCardTracker(engine.ActionPassSnap, snapperEngineIdx, preStock, preDiscard)
		_ = snapperUUID // Snap phase silently passed.
	}
}

// onTurnAdvanced is called after the engine's turn advances.
// Broadcasts the new turn event and schedules the turn timer.
func (g *CambiaGame) onTurnAdvanced() {
	g.TurnID++
	g.snapUsedForThisDiscard = false

	if g.Engine.IsTerminal() || g.GameOver {
		return
	}

	// Sync player hands.
	g.syncPlayerHandsFromEngine()

	// Schedule timer and broadcast turn.
	g.scheduleNextTurnTimerEngine()
	g.broadcastPlayerTurnEngine()
}

// scheduleNextTurnTimerEngine schedules a turn timer using engine state.
func (g *CambiaGame) scheduleNextTurnTimerEngine() {
	if g.turnTimer != nil {
		g.turnTimer.Stop()
		g.turnTimer = nil
	}
	if g.TurnDuration <= 0 || g.GameOver || !g.Started {
		return
	}
	if g.Engine.IsTerminal() {
		return
	}

	actingEngineIdx := g.Engine.ActingPlayer()
	currentPlayerUUID := g.EngineToPlayer[actingEngineIdx]

	// Find player by UUID.
	currentPlayer := g.getPlayerByID(currentPlayerUUID)
	if currentPlayer == nil {
		log.Printf("Game %s: Cannot schedule timer — acting player %s not found.", g.ID, currentPlayerUUID)
		return
	}

	if !currentPlayer.Connected {
		log.Printf("Game %s: Current player %s is disconnected. Advancing turn.", g.ID, currentPlayerUUID)
		// Apply a no-op to advance (or end game).
		// For simplicity: just skip their turn by applying ActionDrawStockpile + ActionDiscardNoAbility.
		return
	}

	curTurnID := g.TurnID
	capturedPlayerUUID := currentPlayerUUID

	g.turnTimer = time.AfterFunc(g.TurnDuration, func() {
		go func(expectedTurnID int) {
			g.Mu.Lock()
			defer g.Mu.Unlock()

			isValid := !g.GameOver && g.Started && g.TurnID == expectedTurnID
			if isValid {
				log.Printf("Game %s, Turn %d: Timer fired for player %s.", g.ID, g.TurnID, capturedPlayerUUID)
				g.handleTimeoutEngine(capturedPlayerUUID)
			}
		}(curTurnID)
	})
}

// broadcastPlayerTurnEngine notifies all players of the current player's turn using engine state.
func (g *CambiaGame) broadcastPlayerTurnEngine() {
	if g.GameOver || !g.Started || g.Engine.IsTerminal() {
		return
	}
	actingEngineIdx := g.Engine.ActingPlayer()
	currentPlayerUUID := g.EngineToPlayer[actingEngineIdx]
	log.Printf("Game %s: Turn %d starting for player %s (engine idx %d).", g.ID, g.TurnID, currentPlayerUUID, actingEngineIdx)
	g.fireEvent(GameEvent{
		Type: EventGamePlayerTurn,
		User: &EventUser{ID: currentPlayerUUID},
		Payload: map[string]interface{}{
			"turn": g.TurnID,
		},
	})
	g.logAction(currentPlayerUUID, string(EventGamePlayerTurn), map[string]interface{}{"turn": g.TurnID})
}

// handleTimeoutEngine processes a turn timeout using engine state.
func (g *CambiaGame) handleTimeoutEngine(playerID uuid.UUID) {
	log.Printf("Game %s: Player %s timed out on turn %d.", g.ID, playerID, g.TurnID)
	g.logAction(playerID, "player_timeout", map[string]interface{}{"turn": g.TurnID})

	engineIdx, ok := g.PlayerToEngine[playerID]
	if !ok {
		log.Printf("Game %s: Timed out player %s not in engine mapping.", g.ID, playerID)
		return
	}

	// If special action pending, skip it.
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID {
		g.processSkipSpecialAction(playerID)
		return
	}

	// If pending ability choice, resolve as no-ability.
	if g.pendingDiscardAbilityChoice && g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		g.SpecialAction = SpecialActionState{}
		g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
		return
	}

	// If player has a drawn card pending in engine, discard it.
	if g.Engine.Pending.Type == engine.PendingDiscard && g.Engine.Pending.PlayerID == engineIdx {
		g.logAction(playerID, "player_timeout_discard", nil)
		g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
		return
	}

	// Player timed out without drawing — draw and immediately discard.
	log.Printf("Game %s: Player %s timed out without drawing. Drawing and discarding.", g.ID, playerID)
	if err := g.applyEngineAction(engine.ActionDrawStockpile, playerID); err != nil {
		return
	}
	g.logAction(playerID, "player_timeout_discard", nil)
	g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
}

