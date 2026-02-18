// internal/game/special_actions.go
package game

import (
	"fmt"
	"log"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// ProcessSpecialAction handles player requests to use special card abilities (peek, swap)
// or to skip the ability. Routes based on the card rank that triggered the special action.
// This function assumes the game lock is HELD by the caller.
func (g *CambiaGame) ProcessSpecialAction(
	userID uuid.UUID,
	special string,
	card1Data map[string]interface{},
	card2Data map[string]interface{},
) {
	// Verify special action state is active for this player.
	if !g.SpecialAction.Active || g.SpecialAction.PlayerID != userID {
		log.Printf("Game %s: ProcessSpecialAction called by player %s, but no matching special action is active. Ignoring.", g.ID, userID)
		g.FireEventPrivateSpecialActionFail(userID, "No special action in progress for you.", special, nil, nil)
		return
	}

	rank := g.SpecialAction.CardRank
	g.logAction(userID, "action_special_received", map[string]interface{}{"special": special, "rank": rank, "card1": card1Data, "card2": card2Data})

	// Handle "skip" universally.
	if special == "skip" {
		g.processSkipSpecialAction(userID)
		return
	}

	engineIdx, ok := g.PlayerToEngine[userID]
	if !ok {
		g.FireEventPrivateSpecialActionFail(userID, "Player not in engine mapping.", special, nil, nil)
		return
	}

	// Route based on triggering rank.
	switch rank {
	case "7", "8":
		if special != "peek_self" {
			g.FailSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for 7/8 special action.", special))
			return
		}
		g.doPeekSelfEngine(userID, engineIdx, card1Data)

	case "9", "T":
		if special != "peek_other" {
			g.FailSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for 9/T special action.", special))
			return
		}
		g.doPeekOtherEngine(userID, engineIdx, card1Data)

	case "J", "Q":
		if special != "swap_blind" {
			g.FailSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for J/Q special action.", special))
			return
		}
		g.doSwapBlindEngine(userID, engineIdx, card1Data, card2Data)

	case "K":
		if special == "swap_peek" {
			if g.SpecialAction.FirstStepDone {
				g.FailSpecialAction(userID, "Invalid step 'swap_peek' for King action - reveal already done.")
				return
			}
			g.doKingLookEngine(userID, engineIdx, card1Data, card2Data)
		} else if special == "swap_peek_swap" {
			if !g.SpecialAction.FirstStepDone {
				g.FailSpecialAction(userID, "Invalid step 'swap_peek_swap' for King action - must peek first.")
				return
			}
			g.doKingSwapYesEngine(userID)
		} else {
			g.FailSpecialAction(userID, fmt.Sprintf("Invalid 'special' value '%s' for King action.", special))
		}

	default:
		g.FailSpecialAction(userID, fmt.Sprintf("Unsupported card rank '%s' for special action.", rank))
	}
}

// doPeekSelfEngine handles 7/8 peek self using engine action.
func (g *CambiaGame) doPeekSelfEngine(playerID uuid.UUID, engineIdx uint8, card1Data map[string]interface{}) {
	// Need to first apply ActionDiscardWithAbility (pending discard), then PeekOwn.
	// The pending discard was buffered in handleDiscardViaEngine.
	_, _, idx, ok := parseCardTarget(card1Data)
	if !ok || idx < 0 {
		g.FailSpecialAction(playerID, "Invalid card specified for peek_self.")
		return
	}

	// Apply buffered discard with ability.
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		if err := g.applyEngineActionRaw(engine.ActionDiscardWithAbility, playerID, engineIdx); err != nil {
			return
		}
	}

	// Apply PeekOwn.
	g.SpecialAction = SpecialActionState{} // Clear before applying.
	if err := g.applyEngineAction(engine.EncodePeekOwn(uint8(idx)), playerID); err != nil {
		return
	}
}

// doPeekOtherEngine handles 9/T peek other using engine action.
func (g *CambiaGame) doPeekOtherEngine(playerID uuid.UUID, engineIdx uint8, card1Data map[string]interface{}) {
	_, _, idx, ok := parseCardTarget(card1Data)
	if !ok || idx < 0 {
		g.FailSpecialAction(playerID, "Invalid card specified for peek_other.")
		return
	}

	// Apply buffered discard with ability.
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		if err := g.applyEngineActionRaw(engine.ActionDiscardWithAbility, playerID, engineIdx); err != nil {
			return
		}
	}

	// Apply PeekOther.
	g.SpecialAction = SpecialActionState{}
	if err := g.applyEngineAction(engine.EncodePeekOther(uint8(idx)), playerID); err != nil {
		return
	}
}

// doSwapBlindEngine handles J/Q blind swap using engine action.
func (g *CambiaGame) doSwapBlindEngine(playerID uuid.UUID, engineIdx uint8, card1Data, card2Data map[string]interface{}) {
	_, owner1ID, idx1, ok1 := parseCardTarget(card1Data)
	_, owner2ID, idx2, ok2 := parseCardTarget(card2Data)

	if !ok1 || !ok2 || idx1 < 0 || idx2 < 0 {
		g.FailSpecialAction(playerID, "Invalid card specification for swap_blind.")
		return
	}

	// Determine own/opp indices.
	var ownIdx, oppIdx uint8
	if owner1ID == playerID {
		ownIdx = uint8(idx1)
		oppIdx = uint8(idx2)
	} else {
		ownIdx = uint8(idx2)
		oppIdx = uint8(idx1)
	}

	// Check Cambia lock.
	opp1 := g.getPlayerByID(owner1ID)
	opp2 := g.getPlayerByID(owner2ID)
	if opp1 != nil && opp1.HasCalledCambia || opp2 != nil && opp2.HasCalledCambia {
		idx1v := idx1
		idx2v := idx2
		g.FireEventPrivateSpecialActionFail(playerID, "Cannot swap cards with a player who has called Cambia.", "swap_blind",
			buildEventCard(&models.Card{ID: g.CardTracker.Players[engineIdx].HandUUIDs[ownIdx]}, &idx1v, owner1ID, false),
			buildEventCard(&models.Card{ID: g.CardTracker.Players[1-engineIdx].HandUUIDs[oppIdx]}, &idx2v, owner2ID, false))
		g.ResetTurnTimer()
		return
	}

	// Apply buffered discard with ability.
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		if err := g.applyEngineActionRaw(engine.ActionDiscardWithAbility, playerID, engineIdx); err != nil {
			return
		}
	}

	// Apply BlindSwap.
	g.SpecialAction = SpecialActionState{}
	if err := g.applyEngineAction(engine.EncodeBlindSwap(ownIdx, oppIdx), playerID); err != nil {
		return
	}
}

// doKingLookEngine handles King's first step (look) using engine action.
func (g *CambiaGame) doKingLookEngine(playerID uuid.UUID, engineIdx uint8, card1Data, card2Data map[string]interface{}) {
	_, owner1ID, idx1, ok1 := parseCardTarget(card1Data)
	_, _, idx2, ok2 := parseCardTarget(card2Data)

	if !ok1 || !ok2 || idx1 < 0 || idx2 < 0 {
		g.FailSpecialAction(playerID, "Invalid card specification for King peek.")
		return
	}

	// Determine own/opp indices.
	var ownIdx, oppIdx uint8
	if owner1ID == playerID {
		ownIdx = uint8(idx1)
		oppIdx = uint8(idx2)
	} else {
		ownIdx = uint8(idx2)
		oppIdx = uint8(idx1)
	}

	// Store context for second step.
	g.SpecialAction.FirstStepDone = true
	g.SpecialAction.Card1 = &models.Card{ID: g.CardTracker.Players[engineIdx].HandUUIDs[ownIdx]}
	g.SpecialAction.Card1Owner = playerID
	oppEngineIdx := uint8(1 - int(engineIdx))
	g.SpecialAction.Card2 = &models.Card{ID: g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppIdx]}
	g.SpecialAction.Card2Owner = g.EngineToPlayer[oppEngineIdx]

	// Apply buffered discard with ability.
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		if err := g.applyEngineActionRaw(engine.ActionDiscardWithAbility, playerID, engineIdx); err != nil {
			g.SpecialAction = SpecialActionState{}
			return
		}
	}

	// Apply KingLook.
	if err := g.applyEngineAction(engine.EncodeKingLook(ownIdx, oppIdx), playerID); err != nil {
		g.SpecialAction = SpecialActionState{}
		return
	}

	// After KingLook, don't advance turn — wait for swap decision.
	// The SpecialAction state (with FirstStepDone=true) signals the second step.
	g.ResetTurnTimer()
}

// doKingSwapYesEngine applies the king swap yes decision.
func (g *CambiaGame) doKingSwapYesEngine(playerID uuid.UUID) {
	g.SpecialAction = SpecialActionState{}
	g.applyEngineAction(engine.ActionKingSwapYes, playerID)
}

// applyEngineActionRaw applies an engine action without full event emission (for buffered discard).
func (g *CambiaGame) applyEngineActionRaw(actionIdx uint16, actorID uuid.UUID, actorEngineIdx uint8) error {
	preStockLen := g.Engine.StockLen
	preDiscardLen := g.Engine.DiscardLen

	if err := g.Engine.ApplyAction(actionIdx); err != nil {
		log.Printf("Game %s: Engine error for raw action %d: %v", g.ID, actionIdx, err)
		return err
	}

	g.updateCardTracker(actionIdx, actorEngineIdx, preStockLen, preDiscardLen)
	g.syncPlayerHandsFromEngine()
	return nil
}

// processSkipSpecialAction handles the "skip" sub-action for any pending special ability.
// Assumes lock is held by caller.
func (g *CambiaGame) processSkipSpecialAction(userID uuid.UUID) {
	rank := g.SpecialAction.CardRank
	log.Printf("Game %s: Player %s chose to skip special action for rank %s.", g.ID, userID, rank)
	g.logAction(userID, "action_special_skip", map[string]interface{}{"rank": rank})

	// If there's a buffered discard waiting, apply it as no-ability.
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardAbilityChoice = false
		g.pendingDiscardCardID = uuid.Nil
		g.SpecialAction = SpecialActionState{}
		g.applyEngineAction(engine.ActionDiscardNoAbility, userID)
		return
	}

	// For King second step skip: apply KingSwapNo.
	if rank == "K" && g.SpecialAction.FirstStepDone {
		g.SpecialAction = SpecialActionState{}
		g.applyEngineAction(engine.ActionKingSwapNo, userID)
		return
	}

	g.SpecialAction = SpecialActionState{}
	g.onTurnAdvanced()
}

// parseCardTarget extracts card ID, owner ID, and index from a client payload map.
// Returns cardID, ownerID, index (-1 if not provided/invalid), ok (bool for basic success).
// Assumes lock is held by caller.
func parseCardTarget(data map[string]interface{}) (cardID uuid.UUID, ownerID uuid.UUID, idx int, ok bool) {
	idx = -1
	cardID = uuid.Nil
	ownerID = uuid.Nil

	if data == nil {
		return
	}

	// Card ID (required).
	cardIDStr, idOk := data["id"].(string)
	if !idOk || cardIDStr == "" {
		return
	}
	var err error
	cardID, err = uuid.Parse(cardIDStr)
	if err != nil {
		cardID = uuid.Nil
		return
	}

	// Index (optional).
	idxFloat, idxProvided := data["idx"].(float64)
	if idxProvided {
		if idxFloat != float64(int(idxFloat)) || idxFloat < 0 {
			// Keep idx = -1.
		} else {
			idx = int(idxFloat)
		}
	}

	// Owner User ID.
	userMap, userProvided := data["user"].(map[string]interface{})
	if userProvided && userMap != nil {
		userIDStr, uidOk := userMap["id"].(string)
		if uidOk && userIDStr != "" {
			ownerID, err = uuid.Parse(userIDStr)
			if err != nil {
				ownerID = uuid.Nil
			}
		}
	}

	ok = (cardID != uuid.Nil)
	return
}

// findCardByID locates a card in a specific player's hand by ID.
// Uses CardTracker for UUID lookup.
// Assumes lock is held by caller.
func (g *CambiaGame) findCardByID(playerID uuid.UUID, cardID uuid.UUID) (*models.Card, int) {
	engineIdx, ok := g.PlayerToEngine[playerID]
	if !ok {
		return nil, -1
	}
	handLen := g.Engine.Players[engineIdx].HandLen
	for i := uint8(0); i < handLen; i++ {
		if g.CardTracker.Players[engineIdx].HandUUIDs[i] == cardID {
			uuid_ := g.CardTracker.Players[engineIdx].HandUUIDs[i]
			if card := g.CardTracker.Registry[uuid_]; card != nil {
				return card, int(i)
			}
		}
	}
	return nil, -1
}

// buildEventCard creates an EventCard struct used in event payloads.
func buildEventCard(card *models.Card, idx *int, ownerID uuid.UUID, includePrivate bool) *EventCard {
	if card == nil {
		return nil
	}
	ec := &EventCard{
		ID:  card.ID,
		Idx: idx,
	}
	if ownerID != uuid.Nil {
		ec.User = &EventUser{ID: ownerID}
	}
	if includePrivate {
		ec.Rank = card.Rank
		ec.Suit = card.Suit
		ec.Value = card.Value
	}
	return ec
}
