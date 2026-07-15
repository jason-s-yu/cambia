// internal/game/start_sync_test.go
package game

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// lastPlayerSyncState returns the most recent private_sync_state event captured for playerID, or
// nil if none was emitted. Reads mb.playerEvents directly (single-goroutine test), matching the
// established pattern in game_test.go.
func lastPlayerSyncState(mb *mockBroadcaster, playerID uuid.UUID) *GameEvent {
	events := mb.playerEvents[playerID]
	var last *GameEvent
	for i := range events {
		if events[i].Type == EventPrivateSyncState {
			last = &events[i]
		}
	}
	return last
}

// TestGameStartEmitsSyncStateToEachPlayer verifies that the start path emits a private_sync_state
// to every player at both the pre-game reveal and the transition to live play. The web client
// discards every game event until it receives a private_sync_state (gameStore.ts), so without
// these the table never renders and the client never learns the game started (cambia-501).
func TestGameStartEmitsSyncStateToEachPlayer(t *testing.T) {
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

	// Each player must have received a sync during the pre-game reveal so that the subsequent
	// private_initial_cards event (and the table render) lands on populated client state.
	for _, id := range ids {
		ev := lastPlayerSyncState(mb, id)
		require.NotNilf(t, ev, "player %s should receive a private_sync_state on pre-game start", id)
		require.NotNil(t, ev.State, "pre-game sync should carry a state snapshot")
		assert.True(t, ev.State.PreGameActive, "pre-game sync should mark the pre-game phase active")
		assert.False(t, ev.State.Started, "pre-game sync should not yet mark the game started")
	}

	// TurnDuration is read by StartGame to arm the first timer; shrink it so the test does not
	// depend on the house-rule value.
	g.TurnDuration = 5 * time.Second
	g.StartGame()

	// After the transition to live play each player must have received a fresh sync marking the
	// game started; game_player_turn alone does not carry the started transition.
	for _, id := range ids {
		ev := lastPlayerSyncState(mb, id)
		require.NotNilf(t, ev, "player %s should receive a private_sync_state on game start", id)
		require.NotNil(t, ev.State, "post-start sync should carry a state snapshot")
		assert.True(t, ev.State.Started, "post-start sync should mark the game started")
		assert.False(t, ev.State.PreGameActive, "post-start sync should mark the pre-game phase over")
	}
}
