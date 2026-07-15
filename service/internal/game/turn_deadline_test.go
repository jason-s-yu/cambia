// internal/game/turn_deadline_test.go
package game

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildTimedTestGame creates a 2-player game with a turn timer configured, shrinking the
// scheduled duration below the house-rule value before StartGame arms the first timer (mirrors
// the pattern in concurrency_test.go: BeginPreGame reads HouseRules.TurnTimerSec into
// TurnDuration, then the test overrides TurnDuration before StartGame reads it to schedule).
func buildTimedTestGame(t *testing.T, turnDuration time.Duration) (*CambiaGame, []uuid.UUID, *mockBroadcaster) {
	t.Helper()

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	g.HouseRules = HouseRules{TurnTimerSec: 5, PenaltyDrawCount: 2, ForfeitOnDisconnect: false}

	ids := make([]uuid.UUID, 2)
	for i := range ids {
		ids[i] = uuid.New()
		g.AddPlayer(&models.Player{
			ID:        ids[i],
			Connected: true,
			User:      &models.User{ID: ids[i], Username: "P"},
		})
	}

	g.BeginPreGame()
	g.TurnDuration = turnDuration
	g.StartGame()

	return g, ids, mb
}

// TestTurnDeadlineEmittedOnTurnStart verifies that broadcastPlayerTurnEngine includes an
// absolute turnDeadline (epoch ms) and serverNow in the game_player_turn event payload when a
// turn timer is armed, and that the deadline lands roughly TurnDuration after send time.
func TestTurnDeadlineEmittedOnTurnStart(t *testing.T) {
	turnDuration := 5 * time.Second
	before := time.Now()
	g, _, mb := buildTimedTestGame(t, turnDuration)
	after := time.Now()

	ev := mb.findEventByType(EventGamePlayerTurn)
	require.NotNil(t, ev, "expected a game_player_turn event to be broadcast on game start")
	require.NotNil(t, ev.Payload, "game_player_turn event should carry a payload")

	rawDeadline, ok := ev.Payload["turnDeadline"]
	require.True(t, ok, "game_player_turn payload should include turnDeadline when a turn timer is armed")
	deadlineMs, ok := rawDeadline.(int64)
	require.True(t, ok, "turnDeadline should be an int64 epoch-ms value")

	rawServerNow, ok := ev.Payload["serverNow"]
	require.True(t, ok, "game_player_turn payload should include serverNow")
	serverNowMs, ok := rawServerNow.(int64)
	require.True(t, ok, "serverNow should be an int64 epoch-ms value")

	deadline := time.UnixMilli(deadlineMs)
	serverNow := time.UnixMilli(serverNowMs)

	// serverNow must fall within the window the event was actually built in. UnixMilli truncates
	// (not rounds), so allow a 1ms slack against the pre-call timestamp.
	assert.True(t, !serverNow.Before(before.Add(-time.Millisecond)) && !serverNow.After(after.Add(50*time.Millisecond)),
		"serverNow %v should fall within test execution window [%v, %v]", serverNow, before, after)

	// The deadline should be ~turnDuration after serverNow (within scheduling slack).
	delta := deadline.Sub(serverNow)
	assert.InDelta(t, turnDuration.Milliseconds(), delta.Milliseconds(), 200,
		"turnDeadline should be ~%v after serverNow, got delta %v", turnDuration, delta)

	// CambiaGame.TurnDeadline should match what was broadcast.
	assert.WithinDuration(t, deadline, g.TurnDeadline, time.Millisecond)
}

// TestTurnDeadlineOmittedWhenTimerDisabled verifies that games with TurnTimerSec=0 emit no
// turnDeadline field (the UI falls back to an informational render).
func TestTurnDeadlineOmittedWhenTimerDisabled(t *testing.T) {
	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	g.HouseRules = HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2, ForfeitOnDisconnect: false}

	ids := make([]uuid.UUID, 2)
	for i := range ids {
		ids[i] = uuid.New()
		g.AddPlayer(&models.Player{
			ID:        ids[i],
			Connected: true,
			User:      &models.User{ID: ids[i], Username: "P"},
		})
	}
	g.BeginPreGame()
	g.StartGame()

	ev := mb.findEventByType(EventGamePlayerTurn)
	require.NotNil(t, ev, "expected a game_player_turn event to be broadcast on game start")
	require.NotNil(t, ev.Payload)

	_, hasDeadline := ev.Payload["turnDeadline"]
	assert.False(t, hasDeadline, "turnDeadline should be absent when no turn timer is configured")
	_, hasServerNow := ev.Payload["serverNow"]
	assert.True(t, hasServerNow, "serverNow should still be present even without a timer")

	assert.True(t, g.TurnDeadline.IsZero(), "CambiaGame.TurnDeadline should be zero when no timer is armed")
}

// TestSyncStateIncludesTurnDeadline verifies getCurrentObfuscatedGameState surfaces the same
// deadline (and a fresh serverNow) for reconnect/desync-recovery sync_state snapshots.
func TestSyncStateIncludesTurnDeadline(t *testing.T) {
	turnDuration := 3 * time.Second
	g, ids, _ := buildTimedTestGame(t, turnDuration)

	obf := g.GetCurrentObfuscatedGameState(ids[0])

	require.NotNil(t, obf.TurnDeadline, "sync_state should include turnDeadline when a turn timer is armed")
	assert.Equal(t, g.TurnDeadline.UnixMilli(), *obf.TurnDeadline)
	assert.Greater(t, obf.ServerNow, int64(0), "sync_state should include a nonzero serverNow")

	remaining := time.UnixMilli(*obf.TurnDeadline).Sub(time.UnixMilli(obf.ServerNow))
	assert.Greater(t, remaining, time.Duration(0), "turnDeadline should be in the future relative to serverNow")
	assert.LessOrEqual(t, remaining, turnDuration+time.Second, "turnDeadline should not be far beyond the configured turn duration")
}

// TestSyncStateOmitsTurnDeadlineWhenTimerDisabled verifies sync_state emits a nil turnDeadline
// (and thus a JSON-omitted field) for games with no configured turn timer.
func TestSyncStateOmitsTurnDeadlineWhenTimerDisabled(t *testing.T) {
	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	g.HouseRules = HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2, ForfeitOnDisconnect: false}

	ids := make([]uuid.UUID, 2)
	for i := range ids {
		ids[i] = uuid.New()
		g.AddPlayer(&models.Player{
			ID:        ids[i],
			Connected: true,
			User:      &models.User{ID: ids[i], Username: "P"},
		})
	}
	g.BeginPreGame()
	g.StartGame()

	obf := g.GetCurrentObfuscatedGameState(ids[0])
	assert.Nil(t, obf.TurnDeadline, "sync_state should omit turnDeadline when no turn timer is configured")
	assert.Greater(t, obf.ServerNow, int64(0), "sync_state should still include serverNow")
}
