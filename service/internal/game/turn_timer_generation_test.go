// internal/game/turn_timer_generation_test.go
// The turn clock's generation guard (cambia-1546).
//
// Every test here builds the same shape: a short clock is armed, g.mu is held past the point it
// fires so its callback is awake and blocked on the lock, the clock is re-armed inside that same
// lock hold on the same TurnID, and the lock is released. Timer.Stop cannot recall the parked
// callback, so what stops it acting on the turn it no longer owns is turnTimerGen alone.
package game

import (
	"testing"
	"time"

	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	// dyingClock is short enough that the test can wait it out under the lock.
	dyingClock = 40 * time.Millisecond
	// parkWait is how long the test holds the lock past dyingClock, which is what leaves the
	// callback awake and queued on g.mu.
	parkWait = 90 * time.Millisecond
	// freshClock is the window the re-arm hands out. Long enough that a turn timed out during the
	// test is unambiguously the parked callback's doing.
	freshClock = 10 * time.Second
	// settleWait gives the parked callback its turn at the lock before anything is asserted.
	settleWait = 120 * time.Millisecond
)

// armDyingClock re-arms the turn clock at dyingClock and returns with g.mu still held, the
// callback already fired and queued behind the caller. The caller re-arms and unlocks.
func armDyingClock(t *testing.T, g *CambiaGame) {
	t.Helper()
	g.mu.Lock()
	g.TurnDuration = dyingClock
	g.scheduleNextTurnTimerEngine()
	time.Sleep(parkWait)
}

// buildClockTestGame returns a started 2-player game whose turn clock is parked well out of the
// way, so only the arming a test does itself can fire.
func buildClockTestGame(t *testing.T) (*CambiaGame, []*models.Player, *mockBroadcaster) {
	t.Helper()
	g, players, mb := setupTestGame(t, 2, testHouseRules(15, 2))
	g.mu.Lock()
	g.TurnDuration = freshClock
	g.scheduleNextTurnTimerEngine()
	g.mu.Unlock()
	t.Cleanup(func() { g.EndGame() })
	mb.clear()
	return g, players, mb
}

// TestStaleTurnTimerCallbackDoesNotActOnARearmedTurn is the reproduction from the ticket: a 40ms
// timer fires while the lock is held, the clock is re-armed at 10s on the same TurnID inside that
// hold, and the parked callback then runs. Before the generation stamp it passed the TurnID guard
// and played the timeout out - 'timed out without drawing', TurnID 0 to 1, one card off the stock.
func TestStaleTurnTimerCallbackDoesNotActOnARearmedTurn(t *testing.T) {
	g, _, _ := buildClockTestGame(t)

	armDyingClock(t, g)
	turnID := g.TurnID
	stockLen := g.Engine.StockLen
	acting := g.Engine.ActingPlayer()
	// The re-arm the seven mid-turn sites all make: same turn, new window.
	g.TurnDuration = freshClock
	g.scheduleNextTurnTimerEngine()
	deadline := g.TurnDeadline
	g.mu.Unlock()

	time.Sleep(settleWait)

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, turnID, g.TurnID, "the stale callback must not advance the turn it no longer owns")
	assert.Equal(t, stockLen, g.Engine.StockLen, "the stale callback must not draw the timeout's fallback card")
	assert.Equal(t, acting, g.Engine.ActingPlayer(), "the turn must still belong to the same seat")
	assert.Equal(t, deadline, g.TurnDeadline, "the window handed out by the re-arm must still stand")
}

// TestAbilityPromptArmedOnADyingClockIsNotAutoSkipped covers the case the ticket names first: a
// discard arms an ability prompt in the last few milliseconds of the clock, so the prompt's own
// re-arm and the old callback race. The prompt must survive - the timeout path auto-skips or
// auto-resolves a pending ability, which took the ability away from a player who had just been
// offered it.
func TestAbilityPromptArmedOnADyingClockIsNotAutoSkipped(t *testing.T) {
	g, _, _ := buildClockTestGame(t)

	actor := currentTurnPlayer(g)
	require.NotNil(t, actor)
	seat := g.PlayerToEngine[actor.ID]

	// A 7 discarded off the draw arms peek-own, which is the prompt at engine_adapter.go's
	// post-discard site.
	forceStockTop(g, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[seat].DrawnCardUUID
	require.NotEqual(t, drawnUUID.String(), "", "the draw should have produced a card")

	armDyingClock(t, g)
	turnID := g.TurnID
	// The discard runs on the lock-held internal the public entry routes to, so the parked
	// callback cannot slip in between the discard and the prompt's re-arm.
	g.TurnDuration = freshClock
	g.handleDiscardViaEngine(actor.ID, seat, map[string]interface{}{"id": drawnUUID.String()})
	require.True(t, g.SpecialAction.Active, "precondition: the discard should have armed the prompt")
	g.mu.Unlock()

	time.Sleep(settleWait)

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.True(t, g.SpecialAction.Active, "the prompt must survive the clock it was armed against")
	assert.Equal(t, actor.ID, g.SpecialAction.PlayerID, "and must still belong to the player who earned it")
	assert.Equal(t, turnID, g.TurnID, "the turn must not have been played out from under the prompt")
	assert.False(t, g.TurnDeadline.IsZero(), "the prompt keeps a window to answer in")
}

// TestReconnectPathRearmSurvivesTheParkedCallback covers the reconnect side: the returning acting
// player is handed a window and the parked callback must not spend it. The re-arm is made through
// scheduleNextTurnTimer, the call HandleReconnect makes for the acting player, because the two have
// to be exercised in one lock hold for the ordering to be the one the bug needs. cambia-1545 then
// narrows when HandleReconnect makes that call at all; the guard here covers whichever of the
// mid-turn sites re-arms.
func TestReconnectPathRearmSurvivesTheParkedCallback(t *testing.T) {
	g, _, _ := buildClockTestGame(t)

	actor := currentTurnPlayer(g)
	require.NotNil(t, actor)

	armDyingClock(t, g)
	turnID := g.TurnID
	stockLen := g.Engine.StockLen
	g.TurnDuration = freshClock
	g.scheduleNextTurnTimer()
	g.mu.Unlock()

	time.Sleep(settleWait)

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, actor.ID, g.currentPlayerID(), "the returning player must still hold the turn")
	assert.Equal(t, turnID, g.TurnID, "the reconnect's window must not be spent by the old callback")
	assert.Equal(t, stockLen, g.Engine.StockLen, "and no fallback draw may be played for them")
}
