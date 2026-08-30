// engine_adapter.go - Bridge between engine.GameState and CambiaGame.
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
	Players      [engine.MaxPlayers]PlayerUUIDState
	StockUUIDs   [engine.MaxDeckSize]uuid.UUID
	StockLen     uint8
	DiscardUUIDs [engine.MaxDeckSize]uuid.UUID
	DiscardLen   uint8

	// Registry maps UUID -> full card details for event payloads.
	Registry map[uuid.UUID]*models.Card

	// SeenByPlayer[p] is the set of card UUIDs that player p has legitimately observed the
	// identity of: the pregame peeks, a card the player personally drew, a peek-own (7/8)
	// target, and the own card viewed during a King look. Knowledge is keyed by UUID, not slot,
	// so it travels with the card across moves: a card swapped INTO a hand by a blind swap (J/Q)
	// or a King swap is absent from the new holder's set, while a card that leaves a hand takes
	// its entry nowhere. Only own-card observations are recorded, so an opponent card the actor
	// merely peeked (9/T) or looked at during a King does not enter the set if it later swaps in.
	//
	// Written, and read by nothing on the server today. It is not a rendering gate: since
	// cambia-1094 the self-view in getCurrentObfuscatedGameState hides EVERY own card
	// unconditionally, because the physical game turns the pregame peek face-down at the start and
	// plays the round on memory, and each reveal travels in its own event that the client shows
	// for that event's window only. No card face is ever derived from this set, on the wire or
	// anywhere else. It is recorded rather than dropped because it is the knowledge the snap-fill
	// and swap rules are stated in - a card moved by a snap fill changes hands unseen, so the
	// entry that travels with it is what leaves the snapper knowing a face its new holder does not
	// (snap_fill.go) - and because a server-side agent or a replay consumer would need it.
	SeenByPlayer [engine.MaxPlayers]map[uuid.UUID]bool
}

// markCardSeen records that the player at engineIdx has legitimately observed cardUUID's
// identity. No-op for the nil UUID or an out-of-range index. Assumes the game lock is held.
func (g *CambiaGame) markCardSeen(engineIdx uint8, cardUUID uuid.UUID) {
	if cardUUID == uuid.Nil || int(engineIdx) >= engine.MaxPlayers {
		return
	}
	if g.CardTracker.SeenByPlayer[engineIdx] == nil {
		g.CardTracker.SeenByPlayer[engineIdx] = make(map[uuid.UUID]bool)
	}
	g.CardTracker.SeenByPlayer[engineIdx][cardUUID] = true
}

// PlayerUUIDState holds UUID tracking for a single player's cards.
type PlayerUUIDState struct {
	HandUUIDs     [engine.MaxHandSize]uuid.UUID
	DrawnCardUUID uuid.UUID // UUID of drawn card in PendingDiscard, if any.
}

// engineRankToString converts an engine rank uint8 to service rank string.
func engineRankToString(rank uint8) string {
	switch rank {
	case engine.RankAce:
		return "A"
	case engine.RankTwo:
		return "2"
	case engine.RankThree:
		return "3"
	case engine.RankFour:
		return "4"
	case engine.RankFive:
		return "5"
	case engine.RankSix:
		return "6"
	case engine.RankSeven:
		return "7"
	case engine.RankEight:
		return "8"
	case engine.RankNine:
		return "9"
	case engine.RankTen:
		return "T"
	case engine.RankJack:
		return "J"
	case engine.RankQueen:
		return "Q"
	case engine.RankKing:
		return "K"
	case engine.RankJoker:
		return "O"
	default:
		return "?"
	}
}

// engineSuitToString converts an engine suit uint8 to service suit string.
func engineSuitToString(suit uint8) string {
	switch suit {
	case engine.SuitHearts:
		return "H"
	case engine.SuitDiamonds:
		return "D"
	case engine.SuitClubs:
		return "C"
	case engine.SuitSpades:
		return "S"
	case engine.SuitRedJoker:
		return "R"
	case engine.SuitBlackJoker:
		return "B"
	default:
		return "?"
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

// mapHouseRulesToEngine maps service HouseRules to engine.HouseRules. Every field except
// NumPlayers comes straight from the lobby's house rules, whose ranges HouseRules.Update
// already bounds to what the engine can represent (internal/game/rules.go).
//
// The two zero-value fallbacks below cover a CambiaGame whose HouseRules were assigned
// wholesale from a struct built without DefaultHouseRules (an older persisted payload, or a
// caller that only set the field it cared about): 0 penalty cards and 0 dealt cards are not
// configurations any lobby can produce, so they are read as "unset" rather than obeyed.
// Every other count is taken literally, since 0 is a meaningful setting for it (no jokers,
// no pregame peek, unlimited turns) and NumDecks==0 already means one deck in NewGame.
func (g *CambiaGame) mapHouseRulesToEngine() engine.HouseRules {
	penaltyCount := uint8(g.HouseRules.PenaltyDrawCount)
	if penaltyCount == 0 {
		penaltyCount = 2
	}
	cardsPerPlayer := uint8(g.HouseRules.CardsPerPlayer)
	if cardsPerPlayer == 0 {
		cardsPerPlayer = 4
	}
	// In circuit mode, use tournament-enforced rules.
	if g.Circuit.Enabled {
		hr := engine.TournamentHouseRules()
		hr.PenaltyDrawCount = penaltyCount
		hr.NumPlayers = uint8(len(g.Players))
		return hr
	}
	return engine.HouseRules{
		MaxGameTurns:          uint16(g.HouseRules.MaxGameTurns),
		CardsPerPlayer:        cardsPerPlayer,
		CambiaAllowedRound:    uint8(g.HouseRules.CambiaAllowedRound),
		PenaltyDrawCount:      penaltyCount,
		AllowDrawFromDiscard:  g.HouseRules.AllowDrawFromDiscardPile,
		AllowReplaceAbilities: g.HouseRules.AllowReplaceAbilities,
		AllowOpponentSnapping: g.HouseRules.AllowOpponentSnapping,
		SnapRace:              g.HouseRules.SnapRace,
		NumJokers:             uint8(g.HouseRules.NumJokers),
		LockCallerHand:        g.HouseRules.LockCallerHand,
		// Deal() sizes hands off Rules.numPlayers(), which treats 0 as 2 (engine/rules.go). Left
		// unset, any non-circuit lobby with 3+ players would only get 2 hands dealt. Not a house
		// rule: the player count comes from the lobby roster, never from the settings panel.
		NumPlayers:       uint8(len(g.Players)),
		InitialViewCount: uint8(g.HouseRules.InitialViewCount),
		NumDecks:         uint8(g.HouseRules.NumDecks),
	}
}

// initCardTracker assigns UUIDs to all cards currently in the engine after Deal().
func (g *CambiaGame) initCardTracker() {
	tracker := &g.CardTracker
	tracker.Registry = make(map[uuid.UUID]*models.Card)

	// Reset per-player seen-own-card knowledge for this deal.
	for p := uint8(0); p < engine.MaxPlayers; p++ {
		tracker.SeenByPlayer[p] = make(map[uuid.UUID]bool)
	}

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
//
// oppEngineIdx is the seat the action targeted, resolved from the client payload before the action
// was applied (engineSeatNone when the action targets no opponent). Every opponent-facing branch
// below reads it instead of deriving `1 - actorEngineIdx`, which only holds at a 2-seat table
// (cambia-946).
func (g *CambiaGame) updateCardTracker(actionIdx uint16, actorEngineIdx uint8, oppEngineIdx uint8, preStockLen uint8, preDiscardLen uint8) {
	tracker := &g.CardTracker

	switch {
	case actionIdx == engine.ActionDrawStockpile:
		// Stock top -> player's DrawnCard (pending).
		if preStockLen > 0 {
			drawnUUID := tracker.StockUUIDs[preStockLen-1]
			tracker.Players[actorEngineIdx].DrawnCardUUID = drawnUUID
			tracker.StockLen = g.Engine.StockLen
			// The drawer sees the drawn card's face; it stays seen if placed into the hand.
			g.markCardSeen(actorEngineIdx, drawnUUID)
		}

	case actionIdx == engine.ActionDrawDiscard:
		// Discard top -> player's DrawnCard (pending).
		if preDiscardLen > 0 {
			drawnUUID := tracker.DiscardUUIDs[preDiscardLen-1]
			tracker.Players[actorEngineIdx].DrawnCardUUID = drawnUUID
			tracker.DiscardLen = g.Engine.DiscardLen
			// The drawer sees the drawn card's face; it stays seen if placed into the hand.
			g.markCardSeen(actorEngineIdx, drawnUUID)
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

		} else if targetIdx, ok := engine.ActionIsPeekOwn(actionIdx); ok {
			// No card movement. The actor sees their own card at targetIdx.
			g.markCardSeen(actorEngineIdx, tracker.Players[actorEngineIdx].HandUUIDs[targetIdx])

		} else if _, ok := engine.ActionIsPeekOther(actionIdx); ok {
			// No card movement, and no own-hand knowledge change: peeking an opponent's card does
			// not reveal any of the actor's own cards.

		} else if ownIdx, oppIdx, ok := engine.ActionIsBlindSwap(actionIdx); ok {
			if !g.trackerSeatOK(oppEngineIdx, "blind swap") {
				return
			}
			// Swap UUIDs.
			tracker.Players[actorEngineIdx].HandUUIDs[ownIdx], tracker.Players[oppEngineIdx].HandUUIDs[oppIdx] =
				tracker.Players[oppEngineIdx].HandUUIDs[oppIdx], tracker.Players[actorEngineIdx].HandUUIDs[ownIdx]

		} else if ownIdx, _, ok := engine.ActionIsKingLook(actionIdx); ok {
			// No card movement (peek only). The actor views their own card at ownIdx (seen) and an
			// opponent card. Only the own card is recorded: if the actor later swaps for the peeked
			// opponent card, that incoming card is treated as unseen (see the King swap branch).
			g.markCardSeen(actorEngineIdx, tracker.Players[actorEngineIdx].HandUUIDs[ownIdx])

		} else if actionIdx == engine.ActionKingSwapNo {
			// No card movement.

		} else if actionIdx == engine.ActionKingSwapYes {
			// Swap based on Pending.Data from BEFORE the action (stored in LastAction).
			ownIdx := g.Engine.LastAction.SwapOwnIdx
			oppIdx := g.Engine.LastAction.SwapOppIdx
			if !g.trackerSeatOK(oppEngineIdx, "king swap") {
				return
			}
			tracker.Players[actorEngineIdx].HandUUIDs[ownIdx], tracker.Players[oppEngineIdx].HandUUIDs[oppIdx] =
				tracker.Players[oppEngineIdx].HandUUIDs[oppIdx], tracker.Players[actorEngineIdx].HandUUIDs[ownIdx]

		} else if actionIdx == engine.ActionPassSnap {
			// No card movement.

		} else if targetIdx, ok := engine.ActionIsSnapOwn(actionIdx); ok {
			g.updateTrackerForSnap(actorEngineIdx, targetIdx, true, preStockLen)

		} else if targetIdx, ok := engine.ActionIsSnapOpponent(actionIdx); ok {
			if !g.trackerSeatOK(oppEngineIdx, "snap opponent") {
				return
			}
			g.updateTrackerForSnapOpponent(actorEngineIdx, oppEngineIdx, targetIdx, preStockLen)

		} else if ownIdx, slotIdx, ok := engine.ActionIsSnapOpponentMove(actionIdx); ok {
			// Move own hand card to opponent's hand.
			if !g.trackerSeatOK(oppEngineIdx, "snap opponent move") {
				return
			}
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

// discardSize returns the discard pile's size as the table sees it: the engine's own pile, plus the
// announced-but-unapplied card while an ability discard is buffered (see effectiveDiscardTop and
// buffered_discard.go). Reporting the engine's count inside that window left every client's pile a
// card short of the one it had already rendered (cambia-1033).
func (g *CambiaGame) discardSize() int {
	if g.pendingDiscardAbilityChoice && g.Engine.Pending.Type == engine.PendingDiscard {
		return int(g.Engine.DiscardLen) + 1
	}
	return int(g.Engine.DiscardLen)
}

// effectiveDiscardTop returns the card the table sees on top of the discard pile, its id, and
// whether there is one.
//
// A drawn card that carries an ability is announced as discarded the moment it is played, but its
// engine action is buffered until the discarder resolves or skips the ability: the engine models
// "use the ability" as part of the discard action itself (ActionDiscardWithAbility vs
// ActionDiscardNoAbility, engine/abilities.go and engine/actions.go) and offers no way to decline
// an ability it has already begun, so handleDiscardViaEngine cannot pick the action until the
// player chooses. Throughout that window the engine's own pile top is still the card the played
// one covered, while every client has already moved the played card onto the pile (the
// player_discard event, mirrored by the web client's gameStore). Anything judged against the
// engine's top inside that window therefore reads a card no player can see (cambia-956).
//
// Once a snap lands inside the window the announced card is covered in turn, and the engine's own
// top is the card the table sees again: a successful snap is applied to the engine pile
// immediately, and it matched the announced card's rank to get there, so the rank a later snap is
// judged against is the same either way and only the identity differs (cambia-1033).
func (g *CambiaGame) effectiveDiscardTop() (engine.Card, uuid.UUID, bool) {
	if g.pendingDiscardAbilityChoice && g.Engine.Pending.Type == engine.PendingDiscard &&
		g.pendingDiscardWindowSnaps == 0 {
		return engine.Card(g.Engine.Pending.Data[0]), g.pendingDiscardCardID, true
	}
	if g.Engine.DiscardLen == 0 {
		return engine.EmptyCard, uuid.Nil, false
	}
	topIdx := g.Engine.DiscardLen - 1
	return g.Engine.DiscardPile[topIdx], g.CardTracker.DiscardUUIDs[topIdx], true
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

// applyToEngine hands an already-translated action index to the engine entry point that decodes
// that action space: the N-player (620) space at a 3+ seat table, the legacy 146 space at a
// 2-seat one.
func (g *CambiaGame) applyToEngine(engineActionIdx uint16, useNPlayer bool) error {
	if useNPlayer {
		return g.Engine.ApplyNPlayerAction(engineActionIdx)
	}
	return g.Engine.ApplyAction(engineActionIdx)
}

// trackerSeatOK reports whether an opponent-facing tracker branch has a real seat to work with.
// An unresolved seat means the caller lost the target between resolution and application; the
// branch is skipped and logged rather than indexing a wrapped seat (cambia-946).
func (g *CambiaGame) trackerSeatOK(oppEngineIdx uint8, what string) bool {
	if oppEngineIdx == engineSeatNone || int(oppEngineIdx) >= engine.MaxPlayers {
		log.Printf("Game %s: %s has no resolved target seat; card tracker left untouched.", g.ID, what)
		return false
	}
	return true
}

// applyEngineAction applies an engine action that targets no opponent seat.
func (g *CambiaGame) applyEngineAction(actionIdx uint16, actorID uuid.UUID) error {
	return g.applyEngineActionSeat(actionIdx, actorID, engineSeatNone)
}

// applyEngineActionSeat applies an engine action, updates the UUID tracker and emits events.
// Returns any error from the engine.
//
// actionIdx is always in the engine's 2-player (146) action space, the adapter's internal
// representation. targetSeat is the opponent seat the client named for an opponent-facing ability
// or snap, resolved before this call (engineSeatNone for everything else); at a 3+ seat table the
// action is translated into the engine's N-player space so the engine mutates that seat instead of
// `1 - actingSeat` (cambia-946).
func (g *CambiaGame) applyEngineActionSeat(actionIdx uint16, actorID uuid.UUID, targetSeat uint8) error {
	engineIdx, ok := g.PlayerToEngine[actorID]
	if !ok {
		engineIdx = g.Engine.ActingPlayer()
	}

	// The King's swap decision carries no target of its own: it settles the pair the look bound,
	// so the seat is read back out of the engine's pending decision state before it is cleared.
	if targetSeat == engineSeatNone &&
		(actionIdx == engine.ActionKingSwapYes || actionIdx == engine.ActionKingSwapNo) {
		targetSeat = g.kingDecisionSeat(engineIdx)
	}

	oppEngineIdx, _ := g.opponentSeatForAction(engineIdx, targetSeat)

	engineActionIdx, useNPlayer, err := g.engineActionForSeat(actionIdx, engineIdx, targetSeat)
	if err != nil {
		log.Printf("Game %s: cannot target action %d from seat %d: %v", g.ID, actionIdx, engineIdx, err)
		g.fireEventToPlayer(actorID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Invalid target for that action."},
		})
		return err
	}

	// Snapshot pre-action state for diffing.
	preStockLen := g.Engine.StockLen
	preDiscardLen := g.Engine.DiscardLen
	stockpileWasEmpty := actionIdx == engine.ActionDrawStockpile && preStockLen == 0

	// A stockpile draw off an empty stockpile forces the engine to reshuffle the discard pile
	// back into the stockpile before drawing (engine/actions.go drawStockpile -> attemptReshuffle).
	// Drive that reshuffle here, ahead of the draw, the same way cambia-799's handleSnapFailure
	// drives it for penalty draws: snapshot the discard mirror before the engine scrambles it and
	// rebuild the tracker against it, then let preStockLen carry the just-reshuffled count into the
	// draw below. That keeps updateCardTracker's existing `tracker.StockUUIDs[preStockLen-1]`
	// lookup resolving the drawn card out of the freshly rebuilt mirror instead of skipping the
	// update outright (cambia-819: the `if preStockLen > 0` guard used to skip this branch
	// whenever the draw reshuffled, leaving DrawnCardUUID stale and StockUUIDs never rebuilt).
	if stockpileWasEmpty {
		prevDiscardUUIDs := append([]uuid.UUID(nil), g.CardTracker.DiscardUUIDs[:preDiscardLen]...)
		if g.Engine.AttemptReshuffle() {
			g.mirrorReshuffleIntoTracker(prevDiscardUUIDs)
			preStockLen = g.Engine.StockLen
		}
	}

	// Apply to engine.
	if err := g.applyToEngine(engineActionIdx, useNPlayer); err != nil {
		log.Printf("Game %s: Engine error for action %d: %v", g.ID, actionIdx, err)
		g.fireEventToPlayer(actorID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": err.Error()},
		})
		return err
	}

	// Update UUID tracker.
	g.updateCardTracker(actionIdx, engineIdx, oppEngineIdx, preStockLen, preDiscardLen)

	// A discard the table was already shown lands under whatever was snapped on top of it while it
	// waited on the ability choice, before any event reads the pile (cambia-1033).
	if g.applyingAnnouncedDiscard {
		g.sinkAnnouncedDiscardBeneathWindowSnaps()
	}

	// Sync Player model hands (keeps service-level code working).
	g.syncPlayerHandsFromEngine()

	// A stockpile draw that started with an empty stockpile always reshuffled the discard pile
	// back into the stockpile first (engine drawStockpile -> attemptReshuffle, engine/actions.go:
	// StockLen==0 triggers the reshuffle unconditionally before the draw). Broadcast it so clients
	// correct their locally-tracked discard/stockpile counts immediately instead of drifting until
	// the next full sync_state (cambia-763 F3). Checked against stockpileWasEmpty (the original
	// pre-reshuffle preStockLen==0 signal) rather than preStockLen itself, since the block above
	// may have already advanced preStockLen to the post-reshuffle count by this point.
	if stockpileWasEmpty {
		g.fireEvent(GameEvent{
			Type: EventGameReshuffleStockpile,
			Payload: map[string]interface{}{
				"stockpileSize": int(g.Engine.StockLen),
				"discardSize":   int(g.Engine.DiscardLen),
			},
		})
	}

	// Emit WebSocket events.
	g.emitEventsForAction(actionIdx, actorID, engineIdx, oppEngineIdx, preStockLen, preDiscardLen)

	// Check for game end.
	if g.Engine.IsTerminal() {
		g.endGame()
		return nil
	}

	// Handle snap phase.
	if g.Engine.Snap.Active {
		g.autoProcessSnapPhase()
		// Re-check terminal after snap phase resolution (advanceTurn may trigger game end).
		if g.Engine.IsTerminal() {
			g.endGame()
			return nil
		}
		// A snap phase only opens on a turn-ending discard (plain discard, replace, or the card
		// discarded by a resolved ability). autoProcessSnapPhase passes every snapper, which drives
		// the engine through endSnapPhase -> advanceTurn, so the engine turn has now advanced. The
		// onTurnAdvanced() calls below live in the non-snap branch, so without notifying the service
		// lifecycle here the turn silently advances in the engine while the previous player's timer
		// stays armed and no game_player_turn is emitted (cambia-506 wedge). Fire it once, guarding
		// on the snap phase having fully resolved.
		if !g.Engine.Snap.Active && g.Engine.Pending.Type == engine.PendingNone {
			g.onTurnAdvanced()
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
// oppEngineIdx is the seat the action targeted (engineSeatNone when it targeted no opponent), so
// every event names the player the client actually targeted at a 3+ seat table (cambia-946).
func (g *CambiaGame) emitEventsForAction(actionIdx uint16, actorID uuid.UUID, actorEngineIdx uint8, oppEngineIdx uint8, preStockLen uint8, preDiscardLen uint8) {
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
				Type:    EventPrivateDrawStockpile,
				Card:    &EventCard{ID: drawnUUID, Rank: drawnCard.Rank, Suit: drawnCard.Suit, Value: drawnCard.Value},
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
				Type:    EventPrivateDrawStockpile,
				Card:    &EventCard{ID: drawnUUID, Rank: drawnCard.Rank, Suit: drawnCard.Suit, Value: drawnCard.Value},
				Payload: map[string]interface{}{"source": "discardpile"},
			})
		}
		g.logAction(actorID, "action_draw_discardpile", map[string]interface{}{
			"cardId": drawnUUID, "newSize": g.Engine.DiscardLen,
		})

	case actionIdx == engine.ActionDiscardNoAbility || actionIdx == engine.ActionDiscardWithAbility:
		// A card played for its ability is announced by handleDiscardViaEngine the moment it is
		// played, long before the buffered action reaches the engine, so the skip and timeout paths
		// that settle that buffer as a no-ability discard must not announce it again: a client reads
		// the second player_discard as a second card, counting the pile up by one and rendering the
		// ability card back on top of anything snapped over it since (cambia-1033).
		if actionIdx == engine.ActionDiscardNoAbility && !g.applyingAnnouncedDiscard {
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
			if !g.trackerSeatOK(oppEngineIdx, "peek other event") {
				return
			}
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
			if !g.trackerSeatOK(oppEngineIdx, "blind swap event") {
				return
			}
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
			if !g.trackerSeatOK(oppEngineIdx, "king look event") {
				return
			}
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
			if !g.trackerSeatOK(oppEngineIdx, "king swap event") {
				return
			}
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
		log.Printf("Game %s: Player %s discard ignored - no pending drawn card in engine.", g.ID, playerID)
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
		// Buffer the discard - fire special choice event, wait for ability decision. Nothing has been
		// snapped on top of this card yet; the count runs until applyBufferedDiscard closes the
		// window (buffered_discard.go).
		g.pendingDiscardAbilityChoice = true
		g.pendingDiscardCardID = cardID
		g.pendingDiscardWindowSnaps = 0

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

		g.scheduleNextTurnTimer()
	} else {
		// No ability - apply directly.
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

	// Apply the replace, then read back what the engine did with the card it put on the discard
	// pile. Whether that card's ability fires is the engine's decision and not one the service can
	// predict from the rank: replace() arms an ability only when the house rule is on, the drawn
	// card came off the STOCKPILE (RULES.md 3B: a discard-pile draw carries no ability) and the
	// ability has a legal target (engine/actions.go, canUseAbility). Prompting off the rank alone
	// announced abilities the engine had never armed, leaving the previous player holding a prompt
	// that refused their every action on their next turn (cambia-1125).
	if err := g.applyEngineAction(engine.EncodeReplace(targetIdx), playerID); err != nil {
		return
	}
	g.promptEngineArmedAbility(playerID, engineIdx)
}

// pendingAbilityCard names the card whose ability the engine is currently holding pending: its
// rank read off the engine's own discard top, its id off the UUID mirror of the same slot. Reading
// the rank from the engine rather than the mirror's registry keeps it in step with Pending.Type by
// construction - replace() arms the pending type from exactly this card.
// Assumes the lock is held by the caller.
func (g *CambiaGame) pendingAbilityCard() (uuid.UUID, string, bool) {
	switch g.Engine.Pending.Type {
	case engine.PendingPeekOwn, engine.PendingPeekOther, engine.PendingBlindSwap, engine.PendingKingLook:
	default:
		return uuid.Nil, "", false
	}
	discardLen := g.Engine.DiscardLen
	if discardLen == 0 {
		return uuid.Nil, "", false
	}
	return g.CardTracker.DiscardUUIDs[discardLen-1], engineRankToString(g.Engine.DiscardPile[discardLen-1].Rank()), true
}

// promptEngineArmedAbility prompts for an ability the engine armed on its own, which today is only
// the one replace() arms under AllowReplaceAbilities. It is a no-op when the engine armed nothing,
// which is how the service stays in step with replace()'s own conditions rather than re-deriving
// them (cambia-1125).
//
// The prompt is marked Mandatory: there is no action in the engine's space that declines an ability
// already armed, so the player's only way out of the pending state is to resolve it. See
// SpecialActionState.Mandatory.
// Assumes the lock is held by the caller.
func (g *CambiaGame) promptEngineArmedAbility(playerID uuid.UUID, engineIdx uint8) {
	if g.Engine.Pending.PlayerID != engineIdx {
		return
	}
	cardUUID, rank, ok := g.pendingAbilityCard()
	if !ok {
		return
	}
	specialType := rankToSpecial(rank)
	if specialType == "" {
		return
	}

	g.SpecialAction = SpecialActionState{
		Active:    true,
		PlayerID:  playerID,
		CardRank:  rank,
		Mandatory: true,
	}
	g.fireEvent(GameEvent{
		Type:    EventPlayerSpecialChoice,
		User:    &EventUser{ID: playerID},
		Card:    &EventCard{ID: cardUUID, Rank: rank},
		Special: specialType,
		Payload: map[string]interface{}{"mandatory": true},
	})
	g.logAction(playerID, string(EventPlayerSpecialChoice), map[string]interface{}{
		"cardId": cardUUID, "rank": rank, "special": specialType, "mandatory": true,
	})
	g.scheduleNextTurnTimer()
}

// firstOpponentWithCards returns the lowest-numbered seat other than actorSeat that still holds a
// card, which is the target an auto-resolved opponent-facing ability plays against. skipLocked
// drops the Cambia caller, whose hand LockCallerHand freezes against anything that moves a card
// (RULES.md 3C): that is the same skip the engine's canUseAbility applies through
// reachableOpponent(true) and both legal-mask builders apply to PendingBlindSwap and PendingKingLook
// (engine/legal.go). A look that moves nothing keeps the caller in reach, so peek-other does not
// pass it.
// Assumes the lock is held by the caller.
func (g *CambiaGame) firstOpponentWithCards(actorSeat uint8, skipLocked bool) (uint8, bool) {
	for seat := uint8(0); seat < g.seatCount(); seat++ {
		if seat == actorSeat || g.Engine.Players[seat].HandLen == 0 {
			continue
		}
		if skipLocked && g.handLocked(seat) {
			continue
		}
		return seat, true
	}
	return engineSeatNone, false
}

// engineActionLegal reports whether the engine's own legal mask accepts an action the adapter is
// about to apply on a player's behalf, translated into whichever action space this table runs in.
// The auto-resolution picks its target out of engine state rather than from a client, so this is
// what makes that choice legal by construction instead of by restating the engine's rules a second
// time in the adapter (cambia-1125).
// Assumes the lock is held by the caller.
func (g *CambiaGame) engineActionLegal(actionIdx uint16, actorSeat, targetSeat uint8) bool {
	engineActionIdx, useNPlayer, err := g.engineActionForSeat(actionIdx, actorSeat, targetSeat)
	if err != nil {
		return false
	}
	if useNPlayer {
		if engineActionIdx >= engine.NPlayerNumActions {
			return false
		}
		mask := g.Engine.NPlayerLegalActions()
		return mask[engineActionIdx/64]>>(engineActionIdx%64)&1 == 1
	}
	if engineActionIdx >= engine.NumActions {
		return false
	}
	mask := g.Engine.LegalActions()
	return mask[engineActionIdx/64]>>(engineActionIdx%64)&1 == 1
}

// autoResolveArmedAbility plays out an engine-armed ability whose owner let their turn timer run
// out. Neither of the other two outcomes is available: the ability cannot be declined
// (SpecialActionState.Mandatory), and leaving it pending stops the table, because the engine
// refuses every action while it holds one - including the fallback draw the rest of this timeout
// path would take, which leaves the turn timer unrescheduled on a dead clock (cambia-1125).
//
// The target is the first legal one, and the King settles on no swap, which is the same defensive
// line the rest of the timeout path takes: the obligation is discharged without moving a card the
// player did not ask to move. A blind swap has no such line - the engine models no no-op blind
// swap - and the player chose to play that Jack or Queen out of their own hand.
//
// "Legal" is the engine's own answer, not the adapter's: the seat is picked out of engine state and
// then put to the legal mask, so a target the mask refuses falls through to the !resolvable branch
// rather than being applied. That is what keeps the LockCallerHand skip below in step with
// engine/legal.go instead of restating it (cambia-1125).
// Assumes the lock is held by the caller.
func (g *CambiaGame) autoResolveArmedAbility(playerID uuid.UUID) {
	engineIdx, ok := g.PlayerToEngine[playerID]
	if !ok {
		engineIdx = g.Engine.ActingPlayer()
	}
	oppSeat, haveOpp := g.firstOpponentWithCards(engineIdx, false)
	swapSeat, haveSwapSeat := g.firstOpponentWithCards(engineIdx, true)
	ownHandLen := g.Engine.Players[engineIdx].HandLen

	var actionIdx uint16
	targetSeat := uint8(engineSeatNone)
	var resolvable bool
	switch g.Engine.Pending.Type {
	case engine.PendingPeekOwn:
		actionIdx, resolvable = engine.EncodePeekOwn(0), ownHandLen > 0
	case engine.PendingPeekOther:
		actionIdx, targetSeat, resolvable = engine.EncodePeekOther(0), oppSeat, haveOpp
	case engine.PendingBlindSwap:
		actionIdx, targetSeat, resolvable = engine.EncodeBlindSwap(0, 0), swapSeat, haveSwapSeat && ownHandLen > 0
	case engine.PendingKingLook:
		actionIdx, targetSeat, resolvable = engine.EncodeKingLook(0, 0), swapSeat, haveSwapSeat && ownHandLen > 0
	default:
		// Nothing armed after all: fall back to the ordinary skip so the prompt does not outlive
		// the state that justified it.
		g.processSkipSpecialAction(playerID)
		return
	}
	resolvable = resolvable && g.engineActionLegal(actionIdx, engineIdx, targetSeat)
	if !resolvable {
		// A snap taken during the ability window emptied the only hand the ability could target, so
		// the engine has no legal action left for it. Re-arming this player's clock was the old
		// answer to that, and nothing could change the condition: the engine refuses every action
		// while it holds a pending ability, including the fallback draw, so the same timeout fired
		// against the same state forever. cambia-1173 bars a decline action and any other
		// action-space change, so the escape is the engine resolving the ability itself: an empty
		// legal-target set discharges it into the snap phase for the card that armed it, which is
		// what resolving any ability does and which advances the turn when nobody can snap
		// (cambia-1171). The loop this replaces was never reproduced; this is a guard.
		pending := g.Engine.Pending.Type
		rank := g.SpecialAction.CardRank
		if g.Engine.ResolveUntargetableArmedAbility(g.isNPlayerTable()) {
			log.Printf("Game %s: pending ability %d for player %s had no legal target; resolved it and advanced.", g.ID, pending, playerID)
			g.logAction(playerID, "action_special_timeout_fizzle", map[string]interface{}{
				"rank": rank, "pending": pending,
			})
			g.SpecialAction = SpecialActionState{}
			g.FireEventPrivateSpecialActionFail(playerID, "That ability had no legal target and was discharged.", rank, nil, nil)
			g.settleEngineResolution()
			return
		}
		// The engine still has a legal target for the ability even though the action this path
		// picked was refused, so the two are out of step rather than the ability being stranded.
		// Leave the prompt and re-arm the clock rather than clearing state the engine still holds.
		log.Printf("Game %s: cannot auto-resolve pending ability %d for player %s: no legal target.", g.ID, pending, playerID)
		g.scheduleNextTurnTimer()
		return
	}

	g.logAction(playerID, "action_special_timeout_resolve", map[string]interface{}{
		"rank": g.SpecialAction.CardRank, "pending": g.Engine.Pending.Type,
	})
	prompt := g.SpecialAction
	g.SpecialAction = SpecialActionState{}
	if err := g.applyEngineActionSeat(actionIdx, playerID, targetSeat); err != nil {
		// The engine refused a target read out of its own state under this lock, so something is
		// out of step. Put the prompt back and re-arm the clock rather than leaving a cleared
		// prompt over an ability the engine still holds, which is the wedge shape this whole path
		// exists to avoid.
		g.SpecialAction = prompt
		g.scheduleNextTurnTimer()
		return
	}
	// A King's look leaves the swap decision pending; take the no-swap side of it. A refused
	// decline would otherwise leave the engine holding PendingKingDecision with the prompt already
	// cleared and no timer armed, which is the dead-clock wedge in a second shape, so the prompt is
	// rebuilt as the King's second step and the clock re-armed. From there both a live client's
	// skip and the next timeout route to ActionKingSwapNo again.
	if g.Engine.Pending.Type == engine.PendingKingDecision && g.Engine.Pending.PlayerID == engineIdx {
		if err := g.applyEngineAction(engine.ActionKingSwapNo, playerID); err != nil {
			g.restoreKingDecisionPrompt(playerID, engineIdx, targetSeat, 0, 0)
		}
	}
}

// settleEngineResolution mirrors the tail of applyEngineActionSeat for a state change the engine
// made without an action index: hands resync, a game end ends the game, a snap phase the resolution
// opened is played out, and the turn advance is announced exactly once. There is no
// emitEventsForAction call because no action was applied - clients learn the new state from the turn
// broadcast and the hand sync it carries.
// Assumes the lock is held by the caller.
func (g *CambiaGame) settleEngineResolution() {
	g.syncPlayerHandsFromEngine()
	if g.Engine.IsTerminal() {
		g.endGame()
		return
	}
	if g.Engine.Snap.Active {
		g.autoProcessSnapPhase()
		if g.Engine.IsTerminal() {
			g.endGame()
			return
		}
	}
	if !g.Engine.Snap.Active && g.Engine.Pending.Type == engine.PendingNone {
		g.onTurnAdvanced()
	}
}

// restoreKingDecisionPrompt rebuilds the prompt for a King whose look has landed but whose swap
// decision is still open, and re-arms the turn clock. It reproduces the state doKingLookEngine
// leaves behind on the interactive path: Mandatory with FirstStepDone set, which is the one
// combination processSkipSpecialAction reads as a decline it may apply.
// Assumes the lock is held by the caller.
func (g *CambiaGame) restoreKingDecisionPrompt(playerID uuid.UUID, actorSeat, targetSeat, ownSlot, oppSlot uint8) {
	log.Printf("Game %s: King swap decline was refused for player %s; restoring the decision prompt.", g.ID, playerID)
	g.SpecialAction = SpecialActionState{
		Active:        true,
		PlayerID:      playerID,
		CardRank:      "K",
		Mandatory:     true,
		FirstStepDone: true,
	}
	if int(actorSeat) < engine.MaxPlayers && ownSlot < g.Engine.Players[actorSeat].HandLen {
		g.SpecialAction.Card1 = &models.Card{ID: g.CardTracker.Players[actorSeat].HandUUIDs[ownSlot]}
		g.SpecialAction.Card1Owner = playerID
	}
	if targetSeat != engineSeatNone && int(targetSeat) < engine.MaxPlayers && oppSlot < g.Engine.Players[targetSeat].HandLen {
		g.SpecialAction.Card2 = &models.Card{ID: g.CardTracker.Players[targetSeat].HandUUIDs[oppSlot]}
		g.SpecialAction.Card2Owner = g.EngineToPlayer[targetSeat]
	}
	g.scheduleNextTurnTimer()
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

	// LockCallerHand freezes the caller's whole hand, not just other players' access to it
	// (RULES.md 3C: "Your hand is locked and cannot be altered by any player, including yourself
	// (snaps, swaps, etc.)"). This is checked before either hand search below: the engine's own
	// snap phase never gives the caller a Snap.Snappers slot at all once they have called
	// (engine/snap.go initiateSnapPhase), so the caller draws no penalty for attempting to snap
	// while locked, any more than they would for an action the engine never made legal.
	if g.handLocked(engineIdx) {
		g.refuseSnapAttempt(playerID, cardID, "snapper's hand is locked by LockCallerHand")
		return
	}

	// Check discard pile. The top is read through effectiveDiscardTop so a snap that lands while an
	// ability discard is still buffered is judged against the card the table was shown, not the one
	// it covered: reading the engine's pile directly rejected clean rank matches with the
	// invalid-snap penalty for the length of the ability window, and accepted snaps of the covered
	// card's rank in the same window (cambia-956).
	discardTop, _, hasDiscardTop := g.effectiveDiscardTop()
	if !hasDiscardTop {
		g.handleSnapFailure(playerID, engineIdx, nil)
		return
	}

	// Check SnapRace rule.
	if g.HouseRules.SnapRace && g.snapUsedForThisDiscard {
		g.handleSnapFailure(playerID, engineIdx, nil)
		return
	}

	discardTopRank := discardTop.Rank()

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
				g.recordWindowSnap()

				g.syncPlayerHandsFromEngine()
				g.emitSnapSuccessEvents(playerID, playerID, cardID, cardRank, int(i))
				_ = snapCard
				return
			}
			// Found card but wrong rank - fail.
			g.handleSnapFailure(playerID, engineIdx, &cardID)
			return
		}
	}

	// Check every other seat's hand. The snap payload carries only the card id (the client sends
	// the card that was clicked, cambia-913), so the owner is whichever seat currently holds that
	// id: searching seat `1 - engineIdx` only ever found one opponent, so at a 3+ seat table a
	// third player's card fell through to the failed-snap penalty, and from seat 2 the subtraction
	// wrapped to seat 255 and panicked the process (cambia-946).
	oppEngineIdx, i, foundOpp := g.seatHoldingCard(cardID)
	if foundOpp && oppEngineIdx != engineIdx {
		// AllowOpponentSnapping gates the whole category of action, not just this card's rank: a
		// crafted frame naming an opponent's card with the rule off is refused outright, mirroring
		// engine/snap.go snapOpponent and nplayer_actions.go nplayerSnapOpponent, which return a
		// plain error before touching any hand when the rule is off, rather than the drawPenalty
		// path a legal-but-mismatched snap takes. Before cambia-1043 only the client checked this
		// flag (DsGameTable.tsx houseRules.allowOpponentSnapping), so a hand-crafted action_snap
		// frame could snap an opponent's card with the rule off.
		if !g.HouseRules.AllowOpponentSnapping {
			g.refuseSnapAttempt(playerID, cardID, "opponent snapping is disabled by house rules")
			return
		}
		// LockCallerHand protects the caller's hand from everyone else too (RULES.md 3C). The
		// engine's initiateSnapPhase never offers the caller's cards as a snap target once they
		// have called, so this is refused the same way as the AllowOpponentSnapping case above:
		// no penalty, because the target was never legal to name.
		if g.handLocked(oppEngineIdx) {
			g.refuseSnapAttempt(playerID, cardID, "target's hand is locked by LockCallerHand")
			return
		}
		cardRank := g.Engine.Players[oppEngineIdx].Hand[i].Rank()
		if cardRank != discardTopRank {
			g.handleSnapFailure(playerID, engineIdx, &cardID)
			return
		}
		// A snapper with nothing left to give cannot pay the card the snap owes (RULES.md 5), so
		// the attempt fails and draws the penalty instead. This is the engine's own answer to the
		// same position (engine/snap.go snapOpponent, the HandLen == 0 branch), mirrored here
		// because the service resolves snaps outside the engine's snap phase.
		if g.Engine.Players[engineIdx].HandLen == 0 {
			g.handleSnapFailure(playerID, engineIdx, &cardID)
			return
		}

		// Successful snap from that seat's hand.
		if g.HouseRules.SnapRace {
			g.snapUsedForThisDiscard = true
		}

		// Directly remove card from the owner's hand and add to discard pile.
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
		g.recordWindowSnap()

		g.syncPlayerHandsFromEngine()
		g.emitSnapSuccessEvents(playerID, g.EngineToPlayer[oppEngineIdx], cardID, cardRank, int(i))
		// The snap took a card out of another hand, so it owes one back into the slot it emptied
		// (RULES.md 5). The snapper picks which; beginSnapFill prompts them and arms the deadline
		// that settles it if they never answer (cambia-936).
		g.beginSnapFill(playerID, engineIdx, g.EngineToPlayer[oppEngineIdx], oppEngineIdx, i)
		return
	}

	// Card not found in any hand.
	g.handleSnapFailure(playerID, engineIdx, nil)
}

// emitSnapSuccessEvents fires the public snap success event. playerID is the snapper; ownerID owns
// the hand the card left, which is the snapper only for an own-hand snap. Both ride the event: the
// snapper names who acted, the owner tells every client which hand shrank and at which slot, since
// idx indexes the OWNER's hand. Without the owner a client can only assume the snapper, and an
// opponent snap left the victim's hand a card too long on every screen until the next
// private_sync_state (cambia-913).
func (g *CambiaGame) emitSnapSuccessEvents(playerID uuid.UUID, ownerID uuid.UUID, cardID uuid.UUID, rank uint8, idx int) {
	card := g.CardTracker.Registry[cardID]
	rankStr := engineRankToString(rank)
	ev := GameEvent{
		Type: EventPlayerSnapSuccess,
		User: &EventUser{ID: playerID},
		Card: &EventCard{ID: cardID, Rank: rankStr, Idx: &idx, User: &EventUser{ID: ownerID}},
	}
	if card != nil {
		ev.Card.Suit = card.Suit
		ev.Card.Value = card.Value
	}
	g.fireEvent(ev)
	g.logAction(playerID, string(EventPlayerSnapSuccess), map[string]interface{}{"cardId": cardID, "rank": rankStr, "ownerId": ownerID})
}

// handLocked reports whether engineIdx's hand is frozen by the LockCallerHand house rule.
// RULES.md 3C: once Cambia is called, "your hand is locked and cannot be altered by any player,
// including yourself (snaps, swaps, etc.)"; the LockCallerHand field comment (rules.go) names
// exactly that scope - snaps, swaps and replacements - for this flag. Swaps and replacements
// reach that protection for free because they route through engine.ApplyAction, which already
// gates on Rules.LockCallerHand (engine/legal.go); snap does not (see handleSnapViaEngine), which
// is why this helper exists and every snap-path caller must consult it explicitly.
func (g *CambiaGame) handLocked(engineIdx uint8) bool {
	return g.HouseRules.LockCallerHand && g.isCambiaCalled() && g.Engine.CambiaCaller >= 0 && uint8(g.Engine.CambiaCaller) == engineIdx
}

// fireSnapFailEvent logs and broadcasts the public failure notice a rejected snap fires, whether
// or not it draws a penalty. Shared by handleSnapFailure (RULES.md 5: wrong card, or nothing left
// to pay with) and refuseSnapAttempt (a target the house rules never made legal to name).
func (g *CambiaGame) fireSnapFailEvent(playerID uuid.UUID, attemptedCardID *uuid.UUID) {
	if attemptedCardID != nil {
		g.logAction(playerID, string(EventPlayerSnapFail), map[string]interface{}{"attemptedCardId": *attemptedCardID})
	} else {
		g.logAction(playerID, string(EventPlayerSnapFail), nil)
	}

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
}

// refuseSnapAttempt rejects a snap attempt the house rules forbid outright: an opponent target
// with AllowOpponentSnapping off, or any target once LockCallerHand has frozen the acting or
// targeted hand. Unlike handleSnapFailure this draws no penalty - the request was never a legal
// action to attempt in the first place, not a legal one that turned out wrong, mirroring the
// engine's own answer to the same request: snapOpponent/nplayerSnapOpponent return a plain error
// before touching any hand when AllowOpponentSnapping is off, and initiateSnapPhase never offers
// the caller as a snapper or as a target once LockCallerHand applies, so neither ever reaches the
// engine's drawPenalty path either. Reuses the public player_snap_fail event (the client already
// renders it as a no-op, cambia-1043) rather than private_special_action_fail, which the client
// reads as an unresolved special-ability choice and would leave pendingAction stuck.
func (g *CambiaGame) refuseSnapAttempt(playerID uuid.UUID, attemptedCardID uuid.UUID, reason string) {
	log.Printf("Game %s: Player %s snap refused (%s).", g.ID, playerID, reason)
	g.fireSnapFailEvent(playerID, &attemptedCardID)
}

// handleSnapFailure processes a failed snap and applies penalties.
func (g *CambiaGame) handleSnapFailure(playerID uuid.UUID, engineIdx uint8, attemptedCardID *uuid.UUID) {
	log.Printf("Game %s: Player %s snap failed. Penalizing.", g.ID, playerID)
	g.fireSnapFailEvent(playerID, attemptedCardID)

	// Apply penalty draws. This path cannot go through engine.ApplyAction: the engine draws snap
	// penalties inside its snap actions (engine/snap.go snapOwn/snapOpponent -> drawPenalty), but
	// the service answers snaps asynchronously and applyEngineAction already drained the engine's
	// sequential snap phase (autoProcessSnapPhase) when the discard opened it, so Snap.Active is
	// false here and a snap action would be rejected. Each penalty card is therefore drawn through
	// the engine's exported per-card primitive, which keeps the engine the single authority on the
	// penalty rules: the hand-size cap, the empty-stockpile reshuffle, and paying a short penalty
	// when the deck is genuinely exhausted. The hand-rolled draw this replaced never reshuffled, so
	// a penalty owed on an empty stockpile was silently skipped instead (cambia-799).
	penaltyCount := g.HouseRules.PenaltyDrawCount
	if penaltyCount <= 0 {
		return
	}
	log.Printf("Game %s: Applying %d penalty cards to player %s.", g.ID, penaltyCount, playerID)

	reshuffled := false
	for i := 0; i < penaltyCount; i++ {
		// The draw below would reshuffle on its own, but the UUID mirror has to be rebuilt against
		// the discard pile as it stood before the reshuffle, so drive it here instead. The
		// conditions match the engine's own order (hand cap first, then an empty stockpile), which
		// keeps the service from reshuffling on a draw the engine would refuse. Once this has run,
		// the draw finds a stocked pile and reshuffles nothing.
		if g.Engine.Players[engineIdx].HandLen < engine.MaxHandSize && g.Engine.StockLen == 0 {
			prevDiscardUUIDs := append([]uuid.UUID(nil), g.CardTracker.DiscardUUIDs[:g.Engine.DiscardLen]...)
			if g.Engine.AttemptReshuffle() {
				g.mirrorReshuffleIntoTracker(prevDiscardUUIDs)
				reshuffled = true
			}
		}

		handLen := g.Engine.Players[engineIdx].HandLen
		if !g.Engine.DrawPenaltyCard(engineIdx) {
			// Hand is full, or the deck is exhausted (empty stockpile and a discard pile too thin to
			// reshuffle). Either way the penalty is paid short, exactly as the engine pays it.
			break
		}

		// The engine popped the stockpile's top card into hand slot handLen.
		stockIdx := g.Engine.StockLen
		drawnCard := g.Engine.Players[engineIdx].Hand[handLen]

		// Update tracker. The mirror should already name this card; when it does not, mint a fresh
		// identity instead of handing the client an ID that belongs to a card sitting elsewhere, and
		// log it, since a mismatch means some other path moved cards without mirroring them.
		penaltyUUID := g.CardTracker.StockUUIDs[stockIdx]
		penaltyCard := g.CardTracker.Registry[penaltyUUID]
		if penaltyUUID == uuid.Nil || penaltyCard == nil || !registryCardMatches(penaltyCard, drawnCard) {
			log.Printf("Game %s: stockpile UUID mirror drift at slot %d; minting an ID for the penalty card.", g.ID, stockIdx)
			penaltyUUID, _ = uuid.NewRandom()
			penaltyCard = engineCardToDetails(drawnCard, penaltyUUID)
			g.CardTracker.Registry[penaltyUUID] = penaltyCard
		}
		g.CardTracker.Players[engineIdx].HandUUIDs[handLen] = penaltyUUID
		g.CardTracker.StockLen = g.Engine.StockLen

		// Broadcast public penalty event. It reports both pile sizes as they stand after this card
		// was drawn, so a client sets the counts it displays from the server's own numbers instead
		// of reconstructing them by subtraction (cambia-821). Both piles are public knowledge:
		// sync_state broadcasts stockpileSize and discardSize to every player already. The counts
		// go on the public event only, which the penalized player receives alongside the private
		// one, rather than being repeated on both.
		// The discard count is the one the table sees (discardSize), not the engine's: a snap that
		// fails while an ability discard is still buffered would otherwise hand every client a count
		// one short of the pile it has already rendered, and clients set their count from this
		// payload outright (cambia-1033).
		g.fireEvent(GameEvent{
			Type: EventPlayerSnapPenalty,
			User: &EventUser{ID: playerID},
			Card: &EventCard{ID: penaltyUUID},
			Payload: map[string]interface{}{
				"count":         i + 1,
				"total":         penaltyCount,
				"stockpileSize": int(g.Engine.StockLen),
				"discardSize":   g.discardSize(),
			},
		})

		// Private penalty event. It names the new card and where it landed, never its face: a
		// penalty card is drawn unseen, so its rank stays hidden from the penalized player exactly
		// as it is from everyone else. doc/game_actions.md has said so since this event was
		// specified ("Note that no card details are to be revealed, just the new cards"), and the
		// client, the engine's agent state and the CFR reference all already model it that way.
		// Only this emitter disagreed, leaking a face nothing consumed (cambia-820).
		privateIdx := int(handLen)
		g.fireEventToPlayer(playerID, GameEvent{
			Type: EventPrivateSnapPenalty,
			Card: &EventCard{ID: penaltyUUID, Idx: &privateIdx},
			Payload: map[string]interface{}{
				"count": i + 1,
				"total": penaltyCount,
			},
		})
	}

	// Sync player models.
	g.syncPlayerHandsFromEngine()

	// Tell clients about the reshuffle once the penalty has settled, so the counts carried here are
	// the post-reshuffle-and-draw ones. Since cambia-821 the penalty events carry the same counts,
	// so this is no longer the only thing keeping a client's piles in step; it stays because the
	// reshuffle itself is a public fact a client may act on, and because a penalty paid entirely
	// short after a reshuffle emits no penalty event to carry the new counts.
	if reshuffled {
		g.fireEvent(GameEvent{
			Type: EventGameReshuffleStockpile,
			Payload: map[string]interface{}{
				"stockpileSize": int(g.Engine.StockLen),
				"discardSize":   g.discardSize(),
			},
		})
	}

	g.logAction(playerID, "player_snap_penalty_applied", map[string]interface{}{"count": penaltyCount})
}

// mirrorReshuffleIntoTracker rebuilds the CardUUIDTracker's stockpile and discard mirrors after the
// engine reshuffled the discard pile back into the stockpile. prevDiscardUUIDs holds the discard
// pile's UUIDs as they stood immediately before the reshuffle, index-aligned with the engine's
// pre-reshuffle discard pile.
//
// The engine shuffles the cards it moves, so the mirror cannot follow them positionally: each new
// stockpile slot is matched back to the UUID of the identical card in the pre-reshuffle discard
// pile. Identity travels with the UUID, so seen-sets and registry lookups stay valid across the
// reshuffle. Without this the tracker keeps the stale UUIDs of cards drawn out of the stockpile
// long ago, and the next card off the stockpile is handed to a client under an ID that already
// belongs to a card sitting in someone's hand.
func (g *CambiaGame) mirrorReshuffleIntoTracker(prevDiscardUUIDs []uuid.UUID) {
	if len(prevDiscardUUIDs) == 0 {
		return
	}
	tracker := &g.CardTracker

	// The engine leaves the top discard card in place and moves every card below it.
	pool := prevDiscardUUIDs[:len(prevDiscardUUIDs)-1]
	used := make([]bool, len(pool))

	for i := uint8(0); i < g.Engine.StockLen; i++ {
		card := g.Engine.Stockpile[i]
		assigned := uuid.Nil
		for j, id := range pool {
			if used[j] {
				continue
			}
			if details := tracker.Registry[id]; details != nil && registryCardMatches(details, card) {
				used[j] = true
				assigned = id
				break
			}
		}
		if assigned == uuid.Nil {
			// The tracker never held an identity for this card. Mint one rather than leave a nil or
			// recycled ID in the mirror.
			assigned, _ = uuid.NewRandom()
			tracker.Registry[assigned] = engineCardToDetails(card, assigned)
		}
		tracker.StockUUIDs[i] = assigned
	}
	tracker.StockLen = g.Engine.StockLen

	// The discard pile is left holding only the card that was on top.
	top := prevDiscardUUIDs[len(prevDiscardUUIDs)-1]
	for i := 1; i < len(prevDiscardUUIDs); i++ {
		tracker.DiscardUUIDs[i] = uuid.Nil
	}
	tracker.DiscardUUIDs[0] = top
	tracker.DiscardLen = g.Engine.DiscardLen
}

// registryCardMatches reports whether a registry entry describes the given engine card. Rank and
// suit identify a card within a deck; with several decks in play the identical cards are
// interchangeable, so matching any one of them preserves identity.
func registryCardMatches(details *models.Card, c engine.Card) bool {
	return details.Rank == engineRankToString(c.Rank()) && details.Suit == engineSuitToString(c.Suit())
}

// autoProcessSnapPhase immediately passes all snappers through the engine's snap phase.
// This preserves the service's async snap model while satisfying the engine's sequential snap phase.
func (g *CambiaGame) autoProcessSnapPhase() {
	for g.Engine.Snap.Active {
		snapperEngineIdx := g.Engine.Snap.Snappers[g.Engine.Snap.CurrentSnapperIdx]
		snapperUUID := g.EngineToPlayer[snapperEngineIdx]

		preStock := g.Engine.StockLen
		preDiscard := g.Engine.DiscardLen

		passIdx, useNPlayer, err := g.engineActionForSeat(engine.ActionPassSnap, snapperEngineIdx, engineSeatNone)
		if err != nil {
			log.Printf("Game %s: cannot encode PassSnap for seat %d: %v", g.ID, snapperEngineIdx, err)
			break
		}
		if err := g.applyToEngine(passIdx, useNPlayer); err != nil {
			log.Printf("Game %s: Engine PassSnap error: %v", g.ID, err)
			break
		}
		g.updateCardTracker(engine.ActionPassSnap, snapperEngineIdx, engineSeatNone, preStock, preDiscard)
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
	// Clear any previously advertised deadline; only re-armed below if a timer actually starts.
	g.TurnDeadline = time.Time{}
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
		log.Printf("Game %s: Cannot schedule timer - acting player %s not found.", g.ID, currentPlayerUUID)
		return
	}

	// A disconnected player is clocked like everyone else: while their reconnect window is open the
	// table has to keep playing (RULES.md T5, MATCHMAKING.md 8), and once it closes their turns
	// still have to resolve for the remaining players to finish the game. The timeout path draws
	// and discards without touching their hand, which is the defensive play those rules describe.
	// Without an armed timer the turn simply never ends (cambia-955).
	//
	// This used to be declined whenever ForfeitOnDisconnect was off, on the reading that a game
	// nobody can be forfeited from can wait for its player. Nothing else ends that turn, so the
	// wait had no end either, and every circuit round is created with the forfeit rule off
	// (handlers.CreateGameInstance): a circuit whose actor dropped stalled with neither a clock nor
	// a forfeit (cambia-1117 D4). TurnTimerSec 0 remains the one configuration that leaves a turn
	// unclocked, and it returns at the TurnDuration check above.
	if !currentPlayer.Connected {
		log.Printf("Game %s: Current player %s is disconnected; arming their turn timer so the turn still resolves.", g.ID, currentPlayerUUID)
	}

	curTurnID := g.TurnID
	capturedPlayerUUID := currentPlayerUUID
	g.TurnDeadline = time.Now().Add(g.TurnDuration)

	// The AfterFunc runs in its own goroutine, so it acquires mu before reading lifecycle
	// state and mutating via handleTimeoutEngine. The TurnID guard drops a stale fire whose
	// turn already advanced (Stop() does not block the in-flight callback, so a reschedule
	// that increments TurnID makes this callback a no-op once it acquires the lock).
	g.turnTimer = time.AfterFunc(g.TurnDuration, func() {
		g.mu.Lock()
		defer g.mu.Unlock()
		if g.GameOver || !g.Started || g.TurnID != curTurnID {
			return
		}
		log.Printf("Game %s, Turn %d: Timer fired for player %s.", g.ID, g.TurnID, capturedPlayerUUID)
		g.handleTimeoutEngine(capturedPlayerUUID)
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

	// serverNow lets clients compute their clock skew against this event's send time; turnDeadline
	// (absolute server-clock epoch ms) is only included when a turn timer is actually armed, so
	// games with TurnTimerSec=0 emit no deadline and the UI falls back to an informational render.
	payload := map[string]interface{}{
		"turn":      g.TurnID,
		"serverNow": time.Now().UnixMilli(),
	}
	if !g.TurnDeadline.IsZero() {
		payload["turnDeadline"] = g.TurnDeadline.UnixMilli()
	}
	g.fireEvent(GameEvent{
		Type:    EventGamePlayerTurn,
		User:    &EventUser{ID: currentPlayerUUID},
		Payload: payload,
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

	// An unpaid snap fill is settled before the turn is played out: the fill has its own deadline,
	// but a snapper whose turn came round first would otherwise have every action below refused by
	// HandlePlayerAction's fill gate (cambia-936). Settling it here leaves the turn free to resolve
	// normally in the same pass.
	if fill, owed := g.snapFills[playerID]; owed {
		g.autoSnapFill(fill)
	}

	// If special action pending, skip it - unless the engine armed it, in which case skipping is
	// not on offer and the ability has to be played out for the turn to end at all (cambia-1125).
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID {
		if g.SpecialAction.MustResolve() {
			g.autoResolveArmedAbility(playerID)
			return
		}
		g.processSkipSpecialAction(playerID)
		return
	}

	// If pending ability choice, resolve as no-ability. The card is already on the table's pile, so
	// this settles the buffer without announcing it a second time (buffered_discard.go).
	if g.pendingDiscardAbilityChoice && g.SpecialAction.Active && g.SpecialAction.PlayerID == playerID {
		g.SpecialAction = SpecialActionState{}
		g.applyBufferedDiscard(engine.ActionDiscardNoAbility, playerID)
		return
	}

	// If player has a drawn card pending in engine, discard it.
	if g.Engine.Pending.Type == engine.PendingDiscard && g.Engine.Pending.PlayerID == engineIdx {
		g.logAction(playerID, "player_timeout_discard", nil)
		g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
		return
	}

	// Player timed out without drawing - draw and immediately discard.
	log.Printf("Game %s: Player %s timed out without drawing. Drawing and discarding.", g.ID, playerID)
	if err := g.applyEngineAction(engine.ActionDrawStockpile, playerID); err != nil {
		return
	}
	g.logAction(playerID, "player_timeout_discard", nil)
	g.applyEngineAction(engine.ActionDiscardNoAbility, playerID)
}
