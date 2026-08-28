// internal/game/nplayer_targeting.go - opponent seat resolution for 3+ seat games.
//
// The adapter speaks the engine's 2-player (146) action space internally: every encode and decode
// in engine_adapter.go is written against it. That space carries no target seat, so the engine
// resolves the opponent as `1 - actingSeat` (engine/game.go OpponentOf), which is only meaningful
// at a 2-seat table. The service used to mirror that subtraction, so from seat 2 the uint8 wrapped
// to 255 and indexed the engine's fixed 8-slot Players array, panicking the process, and from
// seats 0 and 1 it silently hit the wrong player at a 4-seat table (cambia-946).
//
// The fix keeps the adapter's internal encoding and adds the missing half: the seat the client
// named, resolved from the target card's owner id, carried alongside the action index. At a table
// with more than two seats the action is translated into the engine's N-player (620) space, whose
// PeekOther / BlindSwap / KingLook / SnapOpponent encodings carry a relative opponent index, so
// the engine mutates the seat the client asked for. A 2-seat table keeps the 146-space path
// untouched, byte for byte.
package game

import (
	"fmt"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
)

// engineSeatNone marks an action that names no opponent seat (a draw, a discard, an own-card
// ability). It is deliberately out of range for engine.MaxPlayers so a missed resolution indexes
// nothing.
const engineSeatNone uint8 = 0xFF

// seatCount returns the number of seats the engine was dealt for.
func (g *CambiaGame) seatCount() uint8 {
	return g.Engine.NumActivePlayers()
}

// isNPlayerTable reports whether this game needs the engine's N-player action space. A 2-seat
// table stays on the legacy 146-action path.
func (g *CambiaGame) isNPlayerTable() bool {
	return g.seatCount() > 2
}

// slotOfCard returns the hand slot holding cardID in the given seat's hand.
func (g *CambiaGame) slotOfCard(seat uint8, cardID uuid.UUID) (uint8, bool) {
	if int(seat) >= engine.MaxPlayers || cardID == uuid.Nil {
		return 0, false
	}
	for i := uint8(0); i < g.Engine.Players[seat].HandLen; i++ {
		if g.CardTracker.Players[seat].HandUUIDs[i] == cardID {
			return i, true
		}
	}
	return 0, false
}

// seatHoldingCard finds the seat and slot holding cardID, searching every seat at the table.
func (g *CambiaGame) seatHoldingCard(cardID uuid.UUID) (seat uint8, slot uint8, ok bool) {
	if cardID == uuid.Nil {
		return 0, 0, false
	}
	for s := uint8(0); s < g.seatCount(); s++ {
		if slot, found := g.slotOfCard(s, cardID); found {
			return s, slot, true
		}
	}
	return 0, 0, false
}

// resolveOwnSlot resolves the actor's own hand slot from a client card payload. The named index is
// authoritative when it is in range and names the card the client sent; otherwise the card id is
// looked up in the actor's hand, which self-heals a client whose slot numbering lags a swap.
func (g *CambiaGame) resolveOwnSlot(actorSeat uint8, cardID uuid.UUID, idx int) (uint8, bool) {
	if int(actorSeat) >= engine.MaxPlayers {
		return 0, false
	}
	handLen := g.Engine.Players[actorSeat].HandLen
	if idx >= 0 && idx < int(handLen) {
		if cardID == uuid.Nil || g.CardTracker.Players[actorSeat].HandUUIDs[idx] == cardID {
			return uint8(idx), true
		}
	}
	return g.slotOfCard(actorSeat, cardID)
}

// resolveOpponentTarget resolves the opponent seat and hand slot a client named for an
// opponent-facing ability. The owner id from the payload picks the seat, so a 4-seat table targets
// the player the user clicked rather than a hardcoded other seat; the card id picks the slot when
// the named index does not match, and stands in for a missing owner id (the snap payload carries
// no owner, cambia-913).
//
// Returns a reason string for RejectSpecialAction on every failure, so an unknown owner, the actor
// naming their own hand as the opponent, or an out-of-range slot all reject-and-wait rather than
// mutating the engine first (cambia-509).
func (g *CambiaGame) resolveOpponentTarget(actorSeat uint8, ownerID, cardID uuid.UUID, idx int) (seat, slot uint8, reason string, ok bool) {
	seat = engineSeatNone

	if ownerID != uuid.Nil {
		mapped, known := g.PlayerToEngine[ownerID]
		if !known {
			return engineSeatNone, 0, "Target player is not in this game.", false
		}
		if mapped == actorSeat {
			return engineSeatNone, 0, "That ability must target another player's card.", false
		}
		seat = mapped
	} else {
		found, foundSlot, ok := g.seatHoldingCard(cardID)
		if !ok {
			return engineSeatNone, 0, "Target card is not in any player's hand.", false
		}
		if found == actorSeat {
			return engineSeatNone, 0, "That ability must target another player's card.", false
		}
		seat, slot = found, foundSlot
		return seat, slot, "", true
	}

	if seat >= g.seatCount() {
		return engineSeatNone, 0, "Target player is not seated in this game.", false
	}

	handLen := g.Engine.Players[seat].HandLen
	if idx >= 0 && idx < int(handLen) {
		if cardID == uuid.Nil || g.CardTracker.Players[seat].HandUUIDs[idx] == cardID {
			return seat, uint8(idx), "", true
		}
	}
	if found, ok := g.slotOfCard(seat, cardID); ok {
		return seat, found, "", true
	}
	return engineSeatNone, 0, "Card index out of range for the target player.", false
}

// opponentRelIdx converts an absolute seat into the relative opponent index the engine's N-player
// encodings use: the position of targetSeat within Opponents(actorSeat), which lists every other
// seat in ascending order.
func (g *CambiaGame) opponentRelIdx(actorSeat, targetSeat uint8) (uint8, error) {
	if targetSeat == actorSeat || targetSeat >= g.seatCount() {
		return 0, fmt.Errorf("seat %d is not an opponent of seat %d", targetSeat, actorSeat)
	}
	if targetSeat > actorSeat {
		return targetSeat - 1, nil
	}
	return targetSeat, nil
}

// engineActionForSeat translates an action index in the 2-player space into the space the engine
// must interpret it in for this table, binding targetSeat where the encoding carries one.
// useNPlayer tells the caller which Apply entry point to use.
//
// The first seventeen indices (draw, discard, call Cambia, Replace, PeekOwn) are identical in both
// spaces and their handlers are player-count agnostic, but they still route through
// ApplyNPlayerAction at a 3+ seat table so DiscardWithAbility and the King decision reach the
// N-player handlers, which consider every opponent instead of seat `1 - acting`.
func (g *CambiaGame) engineActionForSeat(actionIdx uint16, actorSeat, targetSeat uint8) (idx uint16, useNPlayer bool, err error) {
	if !g.isNPlayerTable() {
		return actionIdx, false, nil
	}

	rel := func() (uint8, error) {
		if targetSeat == engineSeatNone {
			return 0, fmt.Errorf("action %d needs a target seat at a %d-seat table", actionIdx, g.seatCount())
		}
		return g.opponentRelIdx(actorSeat, targetSeat)
	}

	switch {
	case actionIdx <= engine.ActionDiscardWithAbility:
		// 0-4 share their encoding across both spaces.
		return actionIdx, true, nil

	case actionIdx == engine.ActionKingSwapNo:
		return engine.NPlayerActionKingSwapNo, true, nil
	case actionIdx == engine.ActionKingSwapYes:
		return engine.NPlayerActionKingSwapYes, true, nil
	case actionIdx == engine.ActionPassSnap:
		return engine.NPlayerActionPassSnap, true, nil
	}

	if slot, ok := engine.ActionIsReplace(actionIdx); ok {
		return engine.NPlayerEncodeReplace(slot), true, nil
	}
	if slot, ok := engine.ActionIsPeekOwn(actionIdx); ok {
		return engine.NPlayerEncodePeekOwn(slot), true, nil
	}
	if slot, ok := engine.ActionIsPeekOther(actionIdx); ok {
		oppRel, err := rel()
		if err != nil {
			return 0, true, err
		}
		return engine.NPlayerEncodePeekOther(slot, oppRel), true, nil
	}
	if ownSlot, oppSlot, ok := engine.ActionIsBlindSwap(actionIdx); ok {
		oppRel, err := rel()
		if err != nil {
			return 0, true, err
		}
		return engine.NPlayerEncodeBlindSwap(ownSlot, oppSlot, oppRel), true, nil
	}
	if ownSlot, oppSlot, ok := engine.ActionIsKingLook(actionIdx); ok {
		oppRel, err := rel()
		if err != nil {
			return 0, true, err
		}
		return engine.NPlayerEncodeKingLook(ownSlot, oppSlot, oppRel), true, nil
	}
	if slot, ok := engine.ActionIsSnapOwn(actionIdx); ok {
		return engine.NPlayerEncodeSnapOwn(slot), true, nil
	}
	if slot, ok := engine.ActionIsSnapOpponent(actionIdx); ok {
		oppRel, err := rel()
		if err != nil {
			return 0, true, err
		}
		return engine.NPlayerEncodeSnapOpponent(slot, oppRel), true, nil
	}
	if ownIdx, _, ok := engine.ActionIsSnapOpponentMove(actionIdx); ok {
		// The N-player move encoding carries only the snapper's own slot: the vacated opponent
		// slot rides in the engine's pending state.
		return engine.NPlayerEncodeSnapOpponentMove(ownIdx), true, nil
	}

	return 0, true, fmt.Errorf("no N-player translation for action %d", actionIdx)
}

// opponentSeatForAction resolves the opponent seat the tracker and event emitters need for an
// action, given the seat the caller resolved from the client payload. A 2-seat table keeps the
// engine's own convention (the other seat); at a larger table an unnamed seat is a bug in the
// caller, reported as not-resolved rather than guessed.
func (g *CambiaGame) opponentSeatForAction(actorSeat, targetSeat uint8) (uint8, bool) {
	if targetSeat != engineSeatNone {
		if targetSeat == actorSeat || targetSeat >= g.seatCount() {
			return engineSeatNone, false
		}
		return targetSeat, true
	}
	if g.isNPlayerTable() {
		return engineSeatNone, false
	}
	if actorSeat > 1 {
		return engineSeatNone, false
	}
	return 1 - actorSeat, true
}

// kingDecisionSeat reads back the seat a King look bound as its target. The engine parks it in the
// pending decision state (Pending.Data[3] at a 3+ seat table), so the swap step targets the same
// seat the look did without the client having to name it again.
func (g *CambiaGame) kingDecisionSeat(actorSeat uint8) uint8 {
	if g.Engine.Pending.Type != engine.PendingKingDecision {
		return engineSeatNone
	}
	if !g.isNPlayerTable() {
		if actorSeat > 1 {
			return engineSeatNone
		}
		return 1 - actorSeat
	}
	seat := g.Engine.Pending.Data[3]
	if seat == actorSeat || seat >= g.seatCount() {
		return engineSeatNone
	}
	return seat
}
