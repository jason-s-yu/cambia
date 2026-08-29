// internal/game/snap_fill.go
// The fill a successful opponent snap owes (RULES.md 5, cambia-936).
//
// "Snap Opponent: If you snap an opponent's card, you must move one of your cards to fill their
// empty spot." The engine models this as a pending decision (engine/snap.go snapOpponent sets
// PendingSnapMove, snapOpponentMove settles it), but the service cannot reach that path: snaps
// arrive asynchronously, long after applyEngineAction drained the engine's sequential snap phase
// (autoProcessSnapPhase), so handleSnapViaEngine mutates the hands itself. Until this file it
// mutated only the victim's: the snapped card left their hand and nothing ever went back, so every
// live game played opponent snaps as a free card off the victim, which is the strongest move in the
// game rather than the even trade the rulebook describes.
//
// The obligation is therefore service-owned state: one entry per snapper (two players can each owe
// a fill off the same discard when snapRace is off), the snapper's other actions are refused until
// they settle it, and a timer settles it for them so a snapper who never answers cannot leave the
// victim permanently a card short. The card itself moves through engine.SnapMoveCard, so the
// shift-and-insert stays defined in the engine alongside the path the CFR pipeline uses.
package game

import (
	"log"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
)

// snapFillState is one outstanding fill: who owes it, whose hand it goes into, and where.
type snapFillState struct {
	SnapperID  uuid.UUID
	SnapperIdx uint8
	VictimID   uuid.UUID
	VictimIdx  uint8
	// Slot is the slot the snapped card vacated. The victim's hand can move before the fill lands
	// (they may draw and replace, be snapped again, or take a penalty), so it is a target, not a
	// promise: applySnapFill clamps it to the hand's current length, which appends.
	Slot uint8
	// Deadline is when autoSnapFill takes the choice away, zero when no timer is armed (the game
	// runs without a turn timer).
	Deadline time.Time
	timer    *time.Timer
	// gen distinguishes this obligation from a later one owed by the same player: Timer.Stop cannot
	// un-fire a callback already waiting on g.mu, so the callback re-reads the entry and compares.
	gen uint64
}

// owesSnapFill reports whether this player still owes a fill.
// Assumes the lock is held by the caller.
func (g *CambiaGame) owesSnapFill(playerID uuid.UUID) bool {
	_, owed := g.snapFills[playerID]
	return owed
}

// dropUnpayableSnapFill lapses an obligation its owner can no longer pay, and reports whether it
// did. A snapper's hand can empty after the snap without them acting at all - another player can
// snap their cards away - and a debt with nothing left to settle it would otherwise refuse that
// player's every action for the rest of the game on a table with no turn timer, where no deadline
// is armed to clear it.
// Assumes the lock is held by the caller.
func (g *CambiaGame) dropUnpayableSnapFill(playerID uuid.UUID, engineIdx uint8) bool {
	fill, owed := g.snapFills[playerID]
	if !owed || int(engineIdx) >= engine.MaxPlayers || g.Engine.Players[engineIdx].HandLen > 0 {
		return false
	}
	log.Printf("Game %s: player %s has no card left to pay the snap fill they owe %s; dropping it.", g.ID, playerID, fill.VictimID)
	g.clearSnapFill(fill)
	return true
}

// snapFillDuration is how long a snapper has to choose the card they give up. It tracks the turn
// timer rather than adding a house rule of its own: a table that plays on a clock clocks this too,
// and a table with the timer off (TurnTimerSec 0) leaves the obligation open the same way it leaves
// a turn open, since nothing on such a table is ever taken out of a player's hands.
func (g *CambiaGame) snapFillDuration() time.Duration {
	return g.TurnDuration
}

// beginSnapFill records the fill a successful opponent snap owes and prompts the snapper.
// Assumes the lock is held by the caller.
func (g *CambiaGame) beginSnapFill(snapperID uuid.UUID, snapperIdx uint8, victimID uuid.UUID, victimIdx, slot uint8) {
	if g.snapFills == nil {
		g.snapFills = make(map[uuid.UUID]*snapFillState)
	}
	// A snapper who already owes a fill cannot snap again (HandlePlayerAction refuses their
	// actions), so an entry here would mean two obligations for one hand: settle the older one
	// first rather than lose it.
	if prev, owed := g.snapFills[snapperID]; owed {
		log.Printf("Game %s: player %s owes a second snap fill; settling the first automatically.", g.ID, snapperID)
		g.autoSnapFill(prev)
	}

	fill := &snapFillState{
		SnapperID:  snapperID,
		SnapperIdx: snapperIdx,
		VictimID:   victimID,
		VictimIdx:  victimIdx,
		Slot:       slot,
		gen:        g.snapFillGen + 1,
	}
	g.snapFillGen = fill.gen
	g.snapFills[snapperID] = fill

	// The slot rides the card (Card.Idx), the way every other event names a hand position; the
	// payload carries only the clock the client needs to render the deadline.
	payload := map[string]interface{}{"serverNow": time.Now().UnixMilli()}
	if d := g.snapFillDuration(); d > 0 {
		fill.Deadline = time.Now().Add(d)
		payload["deadline"] = fill.Deadline.UnixMilli()
		gen := fill.gen
		// The callback runs in its own goroutine, so it takes mu and re-checks the obligation it
		// was armed for before touching any hand.
		fill.timer = time.AfterFunc(d, func() {
			g.mu.Lock()
			defer g.mu.Unlock()
			cur, owed := g.snapFills[snapperID]
			if !owed || cur.gen != gen || g.GameOver {
				return
			}
			log.Printf("Game %s: snap fill timer fired for player %s.", g.ID, snapperID)
			g.autoSnapFill(cur)
		})
	}

	slotInt := int(slot)
	g.fireEvent(GameEvent{
		Type:    EventPlayerSnapMoveRequired,
		User:    &EventUser{ID: snapperID},
		Card:    &EventCard{Idx: &slotInt, User: &EventUser{ID: victimID}},
		Payload: payload,
	})
	g.logAction(snapperID, string(EventPlayerSnapMoveRequired), map[string]interface{}{
		"victimId": victimID, "slot": slotInt,
	})
}

// handleSnapMoveViaEngine settles the fill the sender owes with the card they named.
// Assumes the lock is held by the caller.
func (g *CambiaGame) handleSnapMoveViaEngine(playerID uuid.UUID, engineIdx uint8, payload map[string]interface{}) {
	fill, owed := g.snapFills[playerID]
	if !owed {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "You have no card to move into an opponent's hand."},
		})
		return
	}

	// The card id is the primary key and the index the fallback, the same order every other
	// own-hand target resolves in (resolveOwnSlot).
	cardID := uuid.Nil
	if idStr, ok := payload["id"].(string); ok {
		if parsed, err := uuid.Parse(idStr); err == nil {
			cardID = parsed
		}
	}
	idx := -1
	if idxFloat, ok := payload["idx"].(float64); ok {
		idx = int(idxFloat)
	}
	if cardID == uuid.Nil && idx < 0 {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "Name one of your cards to move into the empty slot."},
		})
		return
	}

	ownSlot, ok := g.resolveOwnSlot(engineIdx, cardID, idx)
	if !ok {
		g.fireEventToPlayer(playerID, GameEvent{
			Type:    EventPrivateSpecialFail,
			Payload: map[string]interface{}{"message": "That card is not in your hand."},
		})
		return
	}

	g.applySnapFill(fill, ownSlot, false)
}

// autoSnapFill settles a fill the snapper did not answer for. It gives up the last slot in their
// hand: the rule leaves them no way out of giving a card, so the timeout only has to be a fixed
// choice rather than a good one, and the last slot is the one no other timeout path competes for
// (a penalty draw appends there, a turn timeout never touches the hand).
// Assumes the lock is held by the caller.
func (g *CambiaGame) autoSnapFill(fill *snapFillState) {
	handLen := g.Engine.Players[fill.SnapperIdx].HandLen
	if handLen == 0 {
		// Nothing left to give: the snapper emptied their hand between the snap and the deadline.
		log.Printf("Game %s: player %s owes a snap fill with an empty hand; dropping it.", g.ID, fill.SnapperID)
		g.clearSnapFill(fill)
		return
	}
	g.applySnapFill(fill, handLen-1, true)
}

// applySnapFill moves the snapper's card at ownSlot into the victim's vacated slot, mirrors the
// move in the UUID tracker and announces it. auto marks a fill the deadline chose.
// Assumes the lock is held by the caller.
func (g *CambiaGame) applySnapFill(fill *snapFillState, ownSlot uint8, auto bool) {
	snapperIdx, victimIdx := fill.SnapperIdx, fill.VictimIdx

	if ownSlot >= g.Engine.Players[snapperIdx].HandLen {
		log.Printf("Game %s: snap fill slot %d out of range for player %s.", g.ID, ownSlot, fill.SnapperID)
		return
	}
	// The victim's hand can lock between the snap that opened this obligation and the fill that
	// pays it: the snap itself lands mid-turn, but nothing stops the victim from calling Cambia on
	// their own later turn while the fill is still outstanding (the snapper who owes it, not the
	// victim, is the one HandlePlayerAction blocks from acting in the meantime). RULES.md is silent
	// on a fill landing after the hand it targets has locked, so this takes the conservative
	// reading: LockCallerHand forbids altering that hand at all (RULES.md 3C), so the fill lapses
	// the same way it does when the hand is already full, rather than writing into a locked hand.
	// The snapper keeps the card they would have given up and their obligation is cleared either
	// way (cambia-1043).
	if g.handLocked(victimIdx) {
		log.Printf("Game %s: player %s's hand is now locked by LockCallerHand; dropping the snap fill owed by %s.", g.ID, fill.VictimID, fill.SnapperID)
		g.clearSnapFill(fill)
		return
	}
	victimHandLen := g.Engine.Players[victimIdx].HandLen
	if victimHandLen >= engine.MaxHandSize {
		// The victim filled back up on their own (penalty draws) while the fill was outstanding.
		// There is no slot left to fill, so the obligation lapses rather than overflowing the hand.
		log.Printf("Game %s: player %s's hand is full; dropping the snap fill owed by %s.", g.ID, fill.VictimID, fill.SnapperID)
		g.clearSnapFill(fill)
		return
	}
	// The vacated slot is a target, not a promise: the victim's hand may have shrunk since.
	destSlot := fill.Slot
	if destSlot > victimHandLen {
		destSlot = victimHandLen
	}

	cardUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[ownSlot]
	if !g.Engine.SnapMoveCard(snapperIdx, ownSlot, victimIdx, destSlot) {
		log.Printf("Game %s: engine refused the snap fill from %s slot %d into %s slot %d.", g.ID, fill.SnapperID, ownSlot, fill.VictimID, destSlot)
		return
	}
	g.moveTrackerCard(snapperIdx, ownSlot, victimIdx, destSlot)
	g.clearSnapFill(fill)
	g.syncPlayerHandsFromEngine()

	// The moved card stays face down: it changes hands unseen, so only the id crosses the wire.
	// Knowledge of its face is keyed by UUID (CardTracker.SeenByPlayer) and travels with the card,
	// which is what leaves the snapper knowing a card in the victim's hand and the victim knowing
	// nothing new - exactly the information the rule creates.
	destInt := int(destSlot)
	g.fireEvent(GameEvent{
		Type: EventPlayerSnapMove,
		User: &EventUser{ID: fill.SnapperID},
		Card: &EventCard{ID: cardUUID, Idx: &destInt, User: &EventUser{ID: fill.VictimID}},
		Payload: map[string]interface{}{
			"fromIdx": int(ownSlot),
			"auto":    auto,
		},
	})
	g.logAction(fill.SnapperID, string(EventPlayerSnapMove), map[string]interface{}{
		"cardId": cardUUID, "victimId": fill.VictimID, "fromIdx": int(ownSlot), "toIdx": destInt, "auto": auto,
	})
}

// clearSnapFill drops an obligation and stops its timer.
// Assumes the lock is held by the caller.
func (g *CambiaGame) clearSnapFill(fill *snapFillState) {
	if fill.timer != nil {
		fill.timer.Stop()
		fill.timer = nil
	}
	delete(g.snapFills, fill.SnapperID)
}

// cancelSnapFills drops every outstanding obligation. Called when the game ends: a fill cannot be
// paid into a finished game's scoring, and leaving the timers armed serves nothing.
// Assumes the lock is held by the caller.
func (g *CambiaGame) cancelSnapFills() {
	for id, fill := range g.snapFills {
		if fill.timer != nil {
			fill.timer.Stop()
			fill.timer = nil
		}
		delete(g.snapFills, id)
	}
}

// moveTrackerCard mirrors engine.SnapMoveCard in the UUID tracker: the id leaves the source hand
// (the rest shifting left) and lands at toSlot in the destination hand (the rest shifting right).
// It must be called AFTER the engine move, and reads the engine's post-move hand lengths, so the
// two mirrors cannot disagree about how far each shift runs. The two seats are always different
// here: a snap fill pays another player.
// Assumes the lock is held by the caller.
func (g *CambiaGame) moveTrackerCard(fromSeat, fromSlot, toSeat, toSlot uint8) {
	if int(fromSeat) >= engine.MaxPlayers || int(toSeat) >= engine.MaxPlayers || fromSeat == toSeat {
		log.Printf("Game %s: refusing to mirror a card move from seat %d to seat %d.", g.ID, fromSeat, toSeat)
		return
	}
	from := &g.CardTracker.Players[fromSeat]
	cardUUID := from.HandUUIDs[fromSlot]

	// Post-move, the source hand is one shorter than the number of ids the mirror still holds.
	fromLen := g.Engine.Players[fromSeat].HandLen
	for i := fromSlot; i < fromLen; i++ {
		from.HandUUIDs[i] = from.HandUUIDs[i+1]
	}
	from.HandUUIDs[fromLen] = uuid.Nil

	// And the destination hand is one longer, so the shift starts at the slot the insert created.
	to := &g.CardTracker.Players[toSeat]
	toLen := g.Engine.Players[toSeat].HandLen
	for i := toLen - 1; i > toSlot; i-- {
		to.HandUUIDs[i] = to.HandUUIDs[i-1]
	}
	to.HandUUIDs[toSlot] = cardUUID
}
