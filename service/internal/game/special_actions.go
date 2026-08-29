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
// Public entry point: acquires mu.
func (g *CambiaGame) ProcessSpecialAction(
	userID uuid.UUID,
	special string,
	card1Data map[string]interface{},
	card2Data map[string]interface{},
) {
	g.mu.Lock()
	defer g.mu.Unlock()
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
			g.RejectSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for 7/8 special action.", special))
			return
		}
		g.doPeekSelfEngine(userID, engineIdx, card1Data)

	case "9", "T":
		if special != "peek_other" {
			g.RejectSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for 9/T special action.", special))
			return
		}
		g.doPeekOtherEngine(userID, engineIdx, card1Data)

	case "J", "Q":
		if special != "swap_blind" {
			g.RejectSpecialAction(userID, fmt.Sprintf("Invalid step '%s' for J/Q special action.", special))
			return
		}
		g.doSwapBlindEngine(userID, engineIdx, card1Data, card2Data)

	case "K":
		if special == "swap_peek" {
			if g.SpecialAction.FirstStepDone {
				g.RejectSpecialAction(userID, "Invalid step 'swap_peek' for King action - reveal already done.")
				return
			}
			g.doKingLookEngine(userID, engineIdx, card1Data, card2Data)
		} else if special == "swap_peek_swap" {
			if !g.SpecialAction.FirstStepDone {
				g.RejectSpecialAction(userID, "Invalid step 'swap_peek_swap' for King action - must peek first.")
				return
			}
			g.doKingSwapYesEngine(userID)
		} else {
			g.RejectSpecialAction(userID, fmt.Sprintf("Invalid 'special' value '%s' for King action.", special))
		}

	default:
		g.RejectSpecialAction(userID, fmt.Sprintf("Unsupported card rank '%s' for special action.", rank))
	}
}

// doPeekSelfEngine handles 7/8 peek self using engine action.
func (g *CambiaGame) doPeekSelfEngine(playerID uuid.UUID, engineIdx uint8, card1Data map[string]interface{}) {
	// Need to first apply ActionDiscardWithAbility (pending discard), then PeekOwn.
	// The pending discard was buffered in handleDiscardViaEngine.
	_, _, idx, ok := parseCardTarget(card1Data)
	if !ok || idx < 0 {
		g.RejectSpecialAction(playerID, "Invalid card specified for peek_self.")
		return
	}
	// Bounds-check before touching engine state. An out-of-range slot must reject-and-wait, never
	// mutate the engine: validating after the buffered discard is applied would leave the engine's
	// ability-pending state divergent from a cleared service SpecialAction (the cambia-509 wedge).
	if idx >= int(g.Engine.Players[engineIdx].HandLen) {
		g.RejectSpecialAction(playerID, "Card index out of range for peek_self.")
		return
	}

	// Apply buffered discard with ability.
	if err := g.applyBufferedDiscard(engine.ActionDiscardWithAbility, playerID); err != nil {
		return
	}

	// Apply PeekOwn.
	g.SpecialAction = SpecialActionState{} // Clear before applying.
	if err := g.applyEngineAction(engine.EncodePeekOwn(uint8(idx)), playerID); err != nil {
		return
	}
}

// doPeekOtherEngine handles 9/T peek other using engine action.
func (g *CambiaGame) doPeekOtherEngine(playerID uuid.UUID, engineIdx uint8, card1Data map[string]interface{}) {
	cardID, ownerID, idx, ok := parseCardTarget(card1Data)
	if !ok || idx < 0 {
		g.RejectSpecialAction(playerID, "Invalid card specified for peek_other.")
		return
	}
	// Resolve the seat the client named from the target card's owner, and bounds-check its slot,
	// before touching engine state (reject-and-wait; see cambia-509). The seat used to be derived
	// as `1 - engineIdx`, which wraps past seat 1 and panicked at a 4-seat table (cambia-946).
	oppEngineIdx, oppSlot, reason, ok := g.resolveOpponentTarget(engineIdx, ownerID, cardID, idx)
	if !ok {
		g.RejectSpecialAction(playerID, reason)
		return
	}

	// Apply buffered discard with ability.
	if err := g.applyBufferedDiscard(engine.ActionDiscardWithAbility, playerID); err != nil {
		return
	}

	// Apply PeekOther against the resolved seat.
	g.SpecialAction = SpecialActionState{}
	if err := g.applyEngineActionSeat(engine.EncodePeekOther(oppSlot), playerID, oppEngineIdx); err != nil {
		return
	}
}

// doSwapBlindEngine handles J/Q blind swap using engine action.
func (g *CambiaGame) doSwapBlindEngine(playerID uuid.UUID, engineIdx uint8, card1Data, card2Data map[string]interface{}) {
	pair, ok := g.resolveSwapPair(playerID, engineIdx, card1Data, card2Data, "swap_blind")
	if !ok {
		return
	}

	// Check Cambia lock.
	opp1 := g.getPlayerByID(pair.ownOwnerID)
	opp2 := g.getPlayerByID(pair.oppOwnerID)
	if opp1 != nil && opp1.HasCalledCambia || opp2 != nil && opp2.HasCalledCambia {
		ownIdxV := int(pair.ownSlot)
		oppIdxV := int(pair.oppSlot)
		g.FireEventPrivateSpecialActionFail(playerID, "Cannot swap cards with a player who has called Cambia.", "swap_blind",
			buildEventCard(&models.Card{ID: g.CardTracker.Players[engineIdx].HandUUIDs[pair.ownSlot]}, &ownIdxV, pair.ownOwnerID, false),
			buildEventCard(&models.Card{ID: g.CardTracker.Players[pair.oppSeat].HandUUIDs[pair.oppSlot]}, &oppIdxV, pair.oppOwnerID, false))
		g.scheduleNextTurnTimer()
		return
	}

	// Apply buffered discard with ability.
	if err := g.applyBufferedDiscard(engine.ActionDiscardWithAbility, playerID); err != nil {
		return
	}

	// Apply BlindSwap against the resolved seat.
	g.SpecialAction = SpecialActionState{}
	if err := g.applyEngineActionSeat(engine.EncodeBlindSwap(pair.ownSlot, pair.oppSlot), playerID, pair.oppSeat); err != nil {
		return
	}
}

// swapTarget is a resolved two-card ability target: the actor's own slot and the opponent seat,
// slot and owner the client named.
type swapTarget struct {
	ownSlot    uint8
	ownOwnerID uuid.UUID
	oppSeat    uint8
	oppSlot    uint8
	oppOwnerID uuid.UUID
}

// resolveSwapPair resolves the own/opponent pair a two-card ability (J/Q blind swap, King look)
// names, rejecting the action and firing the private fail when either half does not resolve.
// Nothing here touches engine state, so a rejected target reject-and-waits with the buffered
// discard intact (cambia-509), and the opponent seat comes from the payload's owner id rather than
// `1 - engineIdx`, which only holds at a 2-seat table (cambia-946).
func (g *CambiaGame) resolveSwapPair(playerID uuid.UUID, engineIdx uint8, card1Data, card2Data map[string]interface{}, special string) (swapTarget, bool) {
	card1ID, owner1ID, idx1, ok1 := parseCardTarget(card1Data)
	card2ID, owner2ID, idx2, ok2 := parseCardTarget(card2Data)

	if !ok1 || !ok2 || idx1 < 0 || idx2 < 0 {
		g.RejectSpecialAction(playerID, fmt.Sprintf("Invalid card specification for %s.", special))
		return swapTarget{}, false
	}

	// The actor owns exactly one of the two cards. The owner ids the client sends decide which;
	// when neither payload carries one, the card that is actually in the actor's hand does.
	ownFirst := owner1ID == playerID
	if owner1ID == uuid.Nil && owner2ID == uuid.Nil {
		_, ownFirst = g.slotOfCard(engineIdx, card1ID)
	}

	ownCardID, ownIdx, ownOwnerID := card2ID, idx2, owner2ID
	oppCardID, oppIdx, oppOwnerID := card1ID, idx1, owner1ID
	if ownFirst {
		ownCardID, ownIdx, ownOwnerID = card1ID, idx1, owner1ID
		oppCardID, oppIdx, oppOwnerID = card2ID, idx2, owner2ID
	}
	if ownOwnerID == uuid.Nil {
		ownOwnerID = playerID
	}
	if ownOwnerID != playerID {
		g.RejectSpecialAction(playerID, fmt.Sprintf("One of the two cards must be your own for %s.", special))
		return swapTarget{}, false
	}

	ownSlot, ok := g.resolveOwnSlot(engineIdx, ownCardID, ownIdx)
	if !ok {
		g.RejectSpecialAction(playerID, fmt.Sprintf("Card index out of range for %s.", special))
		return swapTarget{}, false
	}

	oppSeat, oppSlot, reason, ok := g.resolveOpponentTarget(engineIdx, oppOwnerID, oppCardID, oppIdx)
	if !ok {
		g.RejectSpecialAction(playerID, reason)
		return swapTarget{}, false
	}
	if oppOwnerID == uuid.Nil {
		oppOwnerID = g.EngineToPlayer[oppSeat]
	}

	return swapTarget{
		ownSlot:    ownSlot,
		ownOwnerID: ownOwnerID,
		oppSeat:    oppSeat,
		oppSlot:    oppSlot,
		oppOwnerID: oppOwnerID,
	}, true
}

// doKingLookEngine handles King's first step (look) using engine action.
func (g *CambiaGame) doKingLookEngine(playerID uuid.UUID, engineIdx uint8, card1Data, card2Data map[string]interface{}) {
	// Resolve both halves before any engine mutation or HandUUIDs read (the fixed array would panic
	// on an out-of-range index, and the seat used to be derived as `1 - engineIdx`, which wraps past
	// seat 1: cambia-946). An unresolved target must reject-and-wait so the buffered discard stays
	// unapplied and the turn never advances against an unmoved engine (the cambia-509 King wedge):
	// resolving here, before FirstStepDone is set and before the discard is applied, keeps engine
	// and service state in lockstep.
	pair, ok := g.resolveSwapPair(playerID, engineIdx, card1Data, card2Data, "King peek")
	if !ok {
		return
	}
	ownIdx, oppIdx, oppEngineIdx := pair.ownSlot, pair.oppSlot, pair.oppSeat

	// Store context for second step.
	g.SpecialAction.FirstStepDone = true
	g.SpecialAction.Card1 = &models.Card{ID: g.CardTracker.Players[engineIdx].HandUUIDs[ownIdx]}
	g.SpecialAction.Card1Owner = playerID
	g.SpecialAction.Card2 = &models.Card{ID: g.CardTracker.Players[oppEngineIdx].HandUUIDs[oppIdx]}
	g.SpecialAction.Card2Owner = g.EngineToPlayer[oppEngineIdx]

	// Apply buffered discard with ability.
	if err := g.applyBufferedDiscard(engine.ActionDiscardWithAbility, playerID); err != nil {
		g.SpecialAction = SpecialActionState{}
		return
	}

	// Apply KingLook against the resolved seat.
	if err := g.applyEngineActionSeat(engine.EncodeKingLook(ownIdx, oppIdx), playerID, oppEngineIdx); err != nil {
		g.SpecialAction = SpecialActionState{}
		return
	}

	// After KingLook, don't advance turn - wait for swap decision.
	// The SpecialAction state (with FirstStepDone=true) signals the second step.
	g.scheduleNextTurnTimer()
}

// doKingSwapYesEngine applies the king swap yes decision.
func (g *CambiaGame) doKingSwapYesEngine(playerID uuid.UUID) {
	g.SpecialAction = SpecialActionState{}
	g.applyEngineAction(engine.ActionKingSwapYes, playerID)
}

// applyEngineActionRaw applies an engine action without full event emission (for buffered discard).
// At a 3+ seat table the action routes through the engine's N-player entry point, so
// DiscardWithAbility arms the ability against every opponent instead of only seat `1 - acting`
// (engine/abilities.go discardWithAbilityNPlayer, cambia-946).
func (g *CambiaGame) applyEngineActionRaw(actionIdx uint16, actorID uuid.UUID, actorEngineIdx uint8) error {
	preStockLen := g.Engine.StockLen
	preDiscardLen := g.Engine.DiscardLen

	engineActionIdx, useNPlayer, err := g.engineActionForSeat(actionIdx, actorEngineIdx, engineSeatNone)
	if err != nil {
		log.Printf("Game %s: cannot encode raw action %d for seat %d: %v", g.ID, actionIdx, actorEngineIdx, err)
		return err
	}
	if err := g.applyToEngine(engineActionIdx, useNPlayer); err != nil {
		log.Printf("Game %s: Engine error for raw action %d: %v", g.ID, actionIdx, err)
		return err
	}

	g.updateCardTracker(actionIdx, actorEngineIdx, engineSeatNone, preStockLen, preDiscardLen)

	// The engine pushed this discard on top of everything snapped while it was buffered; the clients
	// put it underneath them. Rotate it back down to the order they were shown (cambia-1033).
	if g.applyingAnnouncedDiscard {
		g.sinkAnnouncedDiscardBeneathWindowSnaps()
	}

	g.syncPlayerHandsFromEngine()
	return nil
}

// processSkipSpecialAction handles the "skip" sub-action for any pending special ability.
// Assumes lock is held by caller.
func (g *CambiaGame) processSkipSpecialAction(userID uuid.UUID) {
	rank := g.SpecialAction.CardRank
	log.Printf("Game %s: Player %s chose to skip special action for rank %s.", g.ID, userID, rank)
	g.logAction(userID, "action_special_skip", map[string]interface{}{"rank": rank})

	// If there's a buffered discard waiting, apply it as no-ability. The card was announced when it
	// was played, so this settles the buffer without a second player_discard (buffered_discard.go).
	if g.pendingDiscardAbilityChoice {
		g.SpecialAction = SpecialActionState{}
		g.applyBufferedDiscard(engine.ActionDiscardNoAbility, userID)
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
