// internal/game/special_action_preconditions_test.go
//
// ProcessSpecialAction's whole gate used to be that a SpecialAction was active for the sender: it
// re-checked neither GameOver, Started, the sender's Connected flag, nor the outstanding-snap-fill
// gate HandlePlayerAction applies to every other action. That let the prompt holder snap during
// their own ability window, then resolve the ability before paying the fill the snap opened, and
// let a stale prompt from the race window around endGame apply to a dead game (cambia-1566).
package game

import (
	"testing"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestProcessSpecialActionRequiresSnapFillPaidFirst drives the scenario cambia-1566 named: the
// ability holder snaps an opponent's card during their own ability window (RULES.md 5 allows a
// snap anytime, and HandlePlayerAction exempts action_snap from the special-action-pending gate),
// which opens a fill they owe. Resolving the ability before paying it must now be refused, and
// paying it must unblock the ability.
func TestProcessSpecialActionRequiresSnapFillPaidFirst(t *testing.T) {
	g, ids, mb := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	curID, curIdx := driveDrawDiscardSeven(t, g)
	oppID := opponentOf(ids, curID)
	oppIdx := g.PlayerToEngine[oppID]

	// Give the opponent a card matching the just-discarded 7 so the ability holder's snap lands.
	giveMatchingCard(g, oppIdx, 0, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	oppCardUUID := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	mb.clear()
	g.HandlePlayerAction(curID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": oppCardUUID.String()},
	})
	require.True(t, g.owesSnapFill(curID), "snapping the opponent's card should open a fill")
	require.True(t, g.SpecialAction.Active, "the ability should still be pending")

	// Attempting to resolve the ability while the fill is unpaid must be refused.
	mb.clear()
	ownSlot0 := g.CardTracker.Players[curIdx].HandUUIDs[0]
	g.ProcessSpecialAction(curID, "peek_self", cardTarget(ownSlot0, curID, 0), nil)

	assert.True(t, g.SpecialAction.Active, "the ability must stay pending until the fill is paid")
	assert.True(t, g.owesSnapFill(curID), "the fill is still owed")
	ev := mb.getLastPlayerEvent(curID)
	require.NotNil(t, ev, "the refusal should be reported to the snapper")
	assert.Equal(t, EventPrivateSpecialFail, ev.Type)

	// Paying the fill (index 1, so slot 0's card used for the peek target below is untouched)
	// unblocks the ability.
	mb.clear()
	givenUUID := g.CardTracker.Players[curIdx].HandUUIDs[1]
	g.HandlePlayerAction(curID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": givenUUID.String(), "idx": float64(1)},
	})
	require.False(t, g.owesSnapFill(curID), "the fill should be settled")

	mb.clear()
	g.ProcessSpecialAction(curID, "peek_self", cardTarget(ownSlot0, curID, 0), nil)
	assert.False(t, g.SpecialAction.Active, "the ability resolves once the fill is paid")
}

// TestEndGameClearsSpecialActionAndBufferedDiscardWindow pins cambia-1566 AC2: a pending ability
// and its buffered discard window must not survive the game they belonged to.
func TestEndGameClearsSpecialActionAndBufferedDiscardWindow(t *testing.T) {
	g, _, _ := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	driveDrawDiscardSeven(t, g)
	require.True(t, g.SpecialAction.Active, "setup should leave the ability pending")
	require.True(t, g.pendingDiscardAbilityChoice, "setup should leave the discard buffered")

	// Stands in for the race window cambia-1566 describes: something else (a forfeit, a turn
	// cap) ends the game while the ability is still pending.
	g.EndGame()

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, SpecialActionState{}, g.SpecialAction, "endGame must clear the pending ability")
	assert.False(t, g.pendingDiscardAbilityChoice, "endGame must close the buffered-discard window")
	assert.Equal(t, uuid.Nil, g.pendingDiscardCardID, "endGame must clear the buffered card id")
	assert.Equal(t, 0, g.pendingDiscardWindowSnaps, "endGame must clear the buffered window snap count")
}

// TestProcessSpecialActionAfterEndGameEmitsNothing pins cambia-1566 AC3's second half across both
// ways a game ends: a normal round completion, where the engine itself reaches its own terminal
// condition, and a forfeit-driven end, which is service-owned and never touches it. Before this
// fix, ProcessSpecialAction's only gate was SpecialAction.Active; with none pending here it would
// still have fired a "no special action in progress" failure event under the old code. The new
// GameOver check refuses it before that point, so nothing is emitted at all.
func TestProcessSpecialActionAfterEndGameEmitsNothing(t *testing.T) {
	t.Run("engine terminal", func(t *testing.T) {
		g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
		driveToCambiaEnd(t, g, players)
		require.True(t, g.GameOver)
		require.True(t, g.Engine.IsTerminal(), "a completed Cambia round is a real engine-terminal state")

		actor := players[0]
		preStock := g.Engine.StockLen
		mb.clear()

		g.ProcessSpecialAction(actor.ID, "skip", nil, nil)

		assert.Nil(t, mb.getLastEvent(), "nothing should broadcast")
		assert.Nil(t, mb.getLastPlayerEvent(actor.ID), "nothing should reach the player")
		assert.Equal(t, preStock, g.Engine.StockLen, "the engine must not mutate")
		assert.Equal(t, SpecialActionState{}, g.SpecialAction)
	})

	t.Run("forfeit-driven", func(t *testing.T) {
		g, players, mb := buildDropTestGame(t, 2, false, true, 0, 30*time.Second, 0)
		stayer, quitter := players[0], players[1]

		g.HandleDisconnect(quitter.ID)
		require.True(t, g.GameOver)
		require.False(t, g.Engine.IsTerminal(), "a forfeit-driven end never reaches the engine's own terminal condition")

		preStock := g.Engine.StockLen
		mb.clear()

		g.ProcessSpecialAction(stayer.ID, "skip", nil, nil)

		assert.Nil(t, mb.getLastEvent(), "nothing should broadcast")
		assert.Nil(t, mb.getLastPlayerEvent(stayer.ID), "nothing should reach the player")
		assert.Equal(t, preStock, g.Engine.StockLen, "the engine must not mutate")
		assert.Equal(t, SpecialActionState{}, g.SpecialAction)
	})
}
