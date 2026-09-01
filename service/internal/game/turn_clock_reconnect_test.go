// internal/game/turn_clock_reconnect_test.go
// A reconnect does not refill the turn clock (cambia-1545).
//
// The drop side has always refused to restart a live clock: pausing or extending it would let a
// player buy thinking time by pulling their network out (memory cambia-991, circuit_disconnect_test
// .go). The return side gave it away instead, because hub.notePlayerReconnected fires on every
// socket join and the reschedule was gated on the returning player being the actor rather than on
// their having been away.
package game

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// turnDeadlineOf reads the armed deadline under the lock the timer goroutine writes it beneath.
func turnDeadlineOf(g *CambiaGame) time.Time {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.TurnDeadline
}

// turnIDOf reads the turn counter under the lock, so a turn the clock ends is observed safely.
func turnIDOf(g *CambiaGame) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.TurnID
}

// turnClockArmed reports whether a turn timer is currently armed.
func turnClockArmed(g *CambiaGame) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.turnTimer != nil
}

// TestReconnectDoesNotExtendARunningTurnClock is the measurement from the ticket: the acting player
// drops halfway through their turn and comes straight back. The clock they left running is the one
// they return to, and it still ends the turn at the deadline the table was already counting down
// to. Before the fix the return added a whole fresh TurnDuration.
func TestReconnectDoesNotExtendARunningTurnClock(t *testing.T) {
	g, _, mb := buildDropTestGame(t, 2, false, false, 0, 400*time.Millisecond, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	armed := turnDeadlineOf(g)
	require.False(t, armed.IsZero(), "precondition: the turn should be clocked")
	turnID := turnIDOf(g)

	time.Sleep(200 * time.Millisecond)
	mb.clear()
	g.HandleDisconnect(acting.ID)
	g.HandleReconnect(acting.ID, nil)

	assert.WithinDuration(t, armed, turnDeadlineOf(g), 5*time.Millisecond,
		"the return must leave the deadline the drop preserved alone")

	// The sync the returning client is handed quotes that same deadline, not one the re-arm moved.
	sync := mb.getLastPlayerEvent(acting.ID)
	require.NotNil(t, sync, "the returning player is sent a state snapshot")
	require.Equal(t, EventPrivateSyncState, sync.Type)
	require.NotNil(t, sync.State)
	require.NotNil(t, sync.State.TurnDeadline, "the snapshot carries the turn clock")
	assert.Equal(t, armed.UnixMilli(), *sync.State.TurnDeadline,
		"the returning client must be told the deadline actually in force")

	// And the turn ends where it was always going to end.
	require.Eventually(t, func() bool { return turnIDOf(g) != turnID },
		time.Until(armed)+300*time.Millisecond, 5*time.Millisecond,
		"the turn must still time out at its original deadline")
}

// TestRepeatedReconnectCyclesDoNotExtendTheTurn covers the compounding case: six drop-and-return
// cycles inside one turn held a 300ms turn open for 1.2s, because every cycle refilled the clock.
// The turn's wall time must stay the one TurnDuration it was given.
func TestRepeatedReconnectCyclesDoNotExtendTheTurn(t *testing.T) {
	g, _, _ := buildDropTestGame(t, 2, false, false, 0, 400*time.Millisecond, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	armed := turnDeadlineOf(g)
	require.False(t, armed.IsZero(), "precondition: the turn should be clocked")
	turnID := turnIDOf(g)

	for i := 0; i < 6; i++ {
		time.Sleep(30 * time.Millisecond)
		g.HandleDisconnect(acting.ID)
		g.HandleReconnect(acting.ID, nil)
	}

	assert.WithinDuration(t, armed, turnDeadlineOf(g), 5*time.Millisecond,
		"six cycles must leave the deadline where the turn started, not six windows further out")
	require.Eventually(t, func() bool { return turnIDOf(g) != turnID },
		time.Until(armed)+200*time.Millisecond, 5*time.Millisecond,
		"the turn's wall clock must stay the one TurnDuration it was given")
}

// TestBareReconnectDoesNotExtendTheTurn is the same rule for the join with no drop behind it.
// hub.notePlayerReconnected runs on every socket join, and HandleReconnect falls through its
// "already marked connected" branch, so a second tab or a repaired socket refilled the clock for
// the acting player without anyone having disconnected at all.
func TestBareReconnectDoesNotExtendTheTurn(t *testing.T) {
	g, _, _ := buildDropTestGame(t, 2, false, false, 0, 400*time.Millisecond, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	armed := turnDeadlineOf(g)
	require.False(t, armed.IsZero(), "precondition: the turn should be clocked")

	time.Sleep(150 * time.Millisecond)
	g.HandleReconnect(acting.ID, nil)

	assert.WithinDuration(t, armed, turnDeadlineOf(g), 5*time.Millisecond,
		"a join by an already-connected acting player must not touch the clock")
}

// TestReconnectArmsAClockForAnUnclockedTurn is the other half of the gate: the reschedule still has
// work to do where nothing is clocking the turn, or the returning player's seat would sit on a turn
// nothing can end. The unclocked turn is produced here directly, standing in for whatever left the
// turn without a timer; the guard the fix adds is g.turnTimer == nil, so that is the state to test.
func TestReconnectArmsAClockForAnUnclockedTurn(t *testing.T) {
	g, _, _ := buildDropTestGame(t, 2, false, false, 0, 5*time.Second, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)

	g.mu.Lock()
	g.turnTimer.Stop()
	g.turnTimer = nil
	g.TurnDeadline = time.Time{}
	g.mu.Unlock()
	require.False(t, turnClockArmed(g), "precondition: the turn is unclocked")

	g.HandleReconnect(acting.ID, nil)

	assert.True(t, turnClockArmed(g), "a returning actor on an unclocked turn must be given a clock")
	deadline := turnDeadlineOf(g)
	require.False(t, deadline.IsZero(), "and the deadline must be advertised")
	assert.WithinDuration(t, time.Now().Add(5*time.Second), deadline, 500*time.Millisecond,
		"the new clock runs a full TurnDuration")
}
