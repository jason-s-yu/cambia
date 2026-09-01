// internal/game/turn_deadline_publish_test.go
// The turn clock is published wherever it moves (cambia-1556).
//
// g.TurnDeadline reached the wire only from the turn broadcast and the sync_state snapshot, and
// neither runs on a mid-turn re-arm. The client sets its countdown from those two frames alone, so
// a re-armed clock left the Turn bar running down to a deadline the server had already replaced.
package game

import (
	"testing"
	"time"

	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAbilityPromptRearmPublishesTheDeadline covers the site the ticket leads with: a discard that
// arms an ability prompt re-arms the clock, and no game_player_turn may be announced over that
// prompt, so the deadline has to travel on its own frame.
func TestAbilityPromptRearmPublishesTheDeadline(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(15, 2))
	g.mu.Lock()
	g.TurnDuration = 10 * time.Second
	g.scheduleNextTurnTimerEngine()
	g.mu.Unlock()
	t.Cleanup(func() { g.EndGame() })

	actor := currentTurnPlayer(g)
	require.NotNil(t, actor)
	seat := g.PlayerToEngine[actor.ID]

	// A 7 discarded off the draw arms peek-own.
	forceStockTop(g, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[seat].DrawnCardUUID

	mb.clear()
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "precondition: the discard should have armed the prompt")

	frame := lastEventOfType(mb, EventGameTurnDeadline)
	require.NotNil(t, frame, "the re-arm must publish the deadline it moved")
	require.NotNil(t, frame.Payload)

	deadlineMs, ok := frame.Payload["turnDeadline"].(int64)
	require.True(t, ok, "the frame carries an int64 epoch-ms deadline")
	assert.Equal(t, g.TurnDeadline.UnixMilli(), deadlineMs, "and it is the deadline actually in force")

	serverNowMs, ok := frame.Payload["serverNow"].(int64)
	require.True(t, ok, "the frame carries serverNow so the client can correct its clock skew")
	assert.Greater(t, deadlineMs, serverNowMs, "the published deadline is ahead of the send time")
	assert.Equal(t, g.TurnID, frame.Payload["turn"], "and names the turn it belongs to")

	assert.Nil(t, lastEventOfType(mb, EventGamePlayerTurn),
		"the deadline must not ride game_player_turn: no turn may be announced over a pending ability")
}

// TestTurnStartPublishesTheDeadline pins that the frame also covers the ordinary turn boundary, so
// a client has one rule for the turn clock rather than one per carrier.
func TestTurnStartPublishesTheDeadline(t *testing.T) {
	g, _, mb := buildTimedTestGame(t, 5*time.Second)
	t.Cleanup(func() { g.EndGame() })

	frame := lastEventOfType(mb, EventGameTurnDeadline)
	require.NotNil(t, frame, "the first turn's clock is published like any other")
	deadlineMs, ok := frame.Payload["turnDeadline"].(int64)
	require.True(t, ok)
	assert.Equal(t, g.TurnDeadline.UnixMilli(), deadlineMs)
}

// TestNoDeadlineFrameWhenTheTableHasNoClock pins that a table played without a turn timer emits
// nothing: TurnDeadline never moves off its zero value, so there is no change to publish.
func TestNoDeadlineFrameWhenTheTableHasNoClock(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	require.True(t, g.TurnDeadline.IsZero(), "precondition: no clock is configured")

	actor := currentTurnPlayer(g)
	require.NotNil(t, actor)
	playSimpleTurn(t, g, actor)
	_ = players

	assert.Nil(t, lastEventOfType(mb, EventGameTurnDeadline),
		"a table with no turn timer has no deadline to advertise")
}

// TestReconnectSyncCarriesThePostRearmDeadline is the reconnect half: where the return does arm a
// clock, the snapshot the returning client is handed quotes the deadline that arming produced, not
// the one it replaced. The snapshots used to be sent ahead of the re-arm.
func TestReconnectSyncCarriesThePostRearmDeadline(t *testing.T) {
	g, _, mb := buildDropTestGame(t, 2, false, false, 0, 5*time.Second, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)

	// Strip the clock, standing in for a turn that reaches the reconnect unclocked.
	g.mu.Lock()
	g.turnTimer.Stop()
	g.turnTimer = nil
	g.TurnDeadline = time.Time{}
	g.mu.Unlock()

	mb.clear()
	g.HandleReconnect(acting.ID, nil)

	inForce := turnDeadlineOf(g)
	require.False(t, inForce.IsZero(), "precondition: the reconnect armed a clock")

	sync := mb.getLastPlayerEvent(acting.ID)
	require.NotNil(t, sync, "the returning player is sent a state snapshot")
	require.Equal(t, EventPrivateSyncState, sync.Type)
	require.NotNil(t, sync.State)
	require.NotNil(t, sync.State.TurnDeadline, "the snapshot carries the turn clock")
	assert.Equal(t, inForce.UnixMilli(), *sync.State.TurnDeadline,
		"the snapshot must quote the deadline the re-arm produced")

	frame := lastEventOfType(mb, EventGameTurnDeadline)
	require.NotNil(t, frame, "and the table is told the clock moved")
	assert.Equal(t, inForce.UnixMilli(), frame.Payload["turnDeadline"])
}
