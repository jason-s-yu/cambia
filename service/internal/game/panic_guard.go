// internal/game/panic_guard.go
// The recover boundary every game-owned goroutine and timer callback runs under (cambia-1243).
package game

import (
	"log"
	"runtime/debug"
)

// runGuarded runs fn under a recover boundary.
//
// A game owns goroutines nothing else is standing under: the turn, pre-game, disconnect-grace and
// snap-fill timers all fire on their own goroutine out of time.AfterFunc, and the persistence and
// action-log publishers run detached. A panic in any of them unwound straight past the runtime and
// killed the process, so one table's bug ended every other table's game with it, which is the
// blast radius the hub's own boundary (hub.runStep) was added to close on the message path.
//
// what names the caller for the log line. The panic is logged with the game id and its stack, then
// the game is ended: whatever state the panic left behind is not state to keep playing on, and
// ending through endGame is what stops the remaining timers and tells the hub and the players the
// game is over. Every callback below takes g.mu with a defer, so the unwind has already released
// it by the time this recovers and the abort can take it itself.
func (g *CambiaGame) runGuarded(what string, fn func()) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		log.Printf("Game %s: panic in %s: %v\n%s", g.ID, what, r, debug.Stack())
		g.abortAfterPanic(what)
	}()
	fn()
}

// guarded wraps fn in the same boundary and hands back the wrapper, for the time.AfterFunc call
// sites that need a func() rather than a call.
func (g *CambiaGame) guarded(what string, fn func()) func() {
	return func() { g.runGuarded(what, fn) }
}

// abortAfterPanic ends the game on the way out of a recovered panic. Best-effort by construction:
// it runs over state a panic already left behind, so it takes its own boundary rather than risking
// a second unwind out of the same goroutine. endGame marks the game over and stops its timers
// before it computes or broadcasts anything, so even a failed attempt leaves nothing armed.
//
// It ends the game as EndReasonInternalError rather than through the plain EndGame, which is the
// difference between a table being told its game ended and being told why (cambia-1831). Scores
// are still computed and recorded, off hands the panic may have left mid-move; the reason is what
// keeps them from being read as a result, and what withholds the rating update they would
// otherwise feed (ratePerGame).
func (g *CambiaGame) abortAfterPanic(what string) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Game %s: panic while ending the game after a panic in %s: %v", g.ID, what, r)
		}
	}()
	g.endWithReason(EndReasonInternalError)
}
