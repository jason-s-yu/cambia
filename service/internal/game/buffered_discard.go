// internal/game/buffered_discard.go
// The window between announcing an ability discard and applying it (cambia-1033).
//
// A drawn card that carries an ability is announced discarded the moment it is played
// (handleDiscardViaEngine fires player_discard, every client moves it onto the pile) but its engine
// action is held back until the player resolves or skips the ability: the engine folds ability use
// into the discard action itself (ActionDiscardWithAbility vs ActionDiscardNoAbility) and offers no
// way to decline an ability it has already begun, so the service cannot pick the action until the
// player chooses. For the length of that window the engine's discard pile is one card short of the
// pile the table was shown, and everything the service answers out of the engine pile is answered
// about a card no player can see.
//
// Two consequences follow from the pile, and this file owns both:
//
//   - Snapping is legal "any time a card is discarded" (RULES.md section 5) and snaps are judged
//     against the announced top since cambia-956, so successful snaps land on the engine pile
//     during the window. The buffered card is then pushed on top of them when it finally applies,
//     inverting the order every client rendered. sinkAnnouncedDiscardBeneathWindowSnaps puts it
//     back underneath them.
//   - The card was already announced, so applying the buffered action must not announce it again.
//     applyBufferedDiscard marks the apply, and emitEventsForAction skips the second
//     player_discard that would otherwise pull every client back onto the ability card.
//
// The view the window presents (effectiveDiscardTop, discardSize) lives with the rest of the
// engine-reading helpers in engine_adapter.go.
package game

import (
	"log"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
)

// applyBufferedDiscard applies the discard action handleDiscardViaEngine buffered behind the
// ability choice and closes the announcement window. It is the only path that resolves that buffer:
// ActionDiscardWithAbility when the player uses the ability, ActionDiscardNoAbility when they skip
// it or their turn timer answers for them.
//
// The ability discard applies raw (no event emission) because the ability action that follows it
// carries the events for the pair; the no-ability discard applies through the normal path, which
// ends the turn. Either way the announced card sinks beneath any snaps taken during the window, and
// the caller's error handling is unchanged: nothing buffered is a no-op, and a refused action
// leaves the window closed exactly as the hand-rolled blocks this replaced did.
// Assumes the lock is held by the caller.
func (g *CambiaGame) applyBufferedDiscard(actionIdx uint16, playerID uuid.UUID) error {
	if !g.pendingDiscardAbilityChoice {
		return nil
	}
	engineIdx, ok := g.PlayerToEngine[playerID]
	if !ok {
		engineIdx = g.Engine.ActingPlayer()
	}

	g.pendingDiscardAbilityChoice = false
	g.pendingDiscardCardID = uuid.Nil
	g.applyingAnnouncedDiscard = true
	defer func() {
		g.applyingAnnouncedDiscard = false
		g.pendingDiscardWindowSnaps = 0
	}()

	if actionIdx == engine.ActionDiscardWithAbility {
		return g.applyEngineActionRaw(actionIdx, playerID, engineIdx)
	}
	return g.applyEngineAction(actionIdx, playerID)
}

// recordWindowSnap notes that a snapped card just landed on the discard pile above a card the table
// has already been shown on it. Only a snap can reach the pile inside the window: the discarder owes
// the ability choice and every other action of theirs is refused, and the other seats may only snap
// or pay a snap fill (HandlePlayerAction).
// Assumes the lock is held by the caller.
func (g *CambiaGame) recordWindowSnap() {
	if g.pendingDiscardAbilityChoice {
		g.pendingDiscardWindowSnaps++
	}
}

// sinkAnnouncedDiscardBeneathWindowSnaps rotates the just-applied ability discard down under the
// cards snapped onto the pile while it was still buffered, in the engine pile and its UUID mirror
// together.
//
// The engine pushes the buffered card on top, since from its side nothing was ever discarded until
// now; the clients put it under those snaps, because they were told about it first. Left alone the
// two disagree about which card is on top, so action_draw_discardpile hands out the wrong card and
// sync_state names the wrong top - both by identity, which for a King is also the wrong value (a
// red King scores -1 against a black King's 13).
// Assumes the lock is held by the caller.
func (g *CambiaGame) sinkAnnouncedDiscardBeneathWindowSnaps() {
	snaps := g.pendingDiscardWindowSnaps
	if snaps <= 0 {
		return
	}
	top := int(g.Engine.DiscardLen) - 1
	if top < 1 {
		return
	}
	if snaps > top {
		// A failed snap's penalty draw emptied the stockpile mid-window and reshuffled the pile back
		// into it, which leaves only the card that was on top (handleSnapFailure). Fewer of the
		// snapped cards are still on the pile than were counted, so only they are rotated over.
		log.Printf("Game %s: %d cards snapped during the ability window but only %d are still on the pile; sinking the announced discard under those.", g.ID, snaps, top)
		snaps = top
	}
	base := top - snaps

	// One rotation settles the window, whichever apply path reached it: the count is spent here so a
	// second call cannot sink the same card twice.
	g.pendingDiscardWindowSnaps = 0

	card := g.Engine.DiscardPile[top]
	cardUUID := g.CardTracker.DiscardUUIDs[top]
	copy(g.Engine.DiscardPile[base+1:top+1], g.Engine.DiscardPile[base:top])
	copy(g.CardTracker.DiscardUUIDs[base+1:top+1], g.CardTracker.DiscardUUIDs[base:top])
	g.Engine.DiscardPile[base] = card
	g.CardTracker.DiscardUUIDs[base] = cardUUID
}
