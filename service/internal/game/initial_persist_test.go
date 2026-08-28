// internal/game/initial_persist_test.go
//
// Regression for the cambia-908 L1 edit that put persistInitialGameState's background write on
// CambiaGame.PersistWG (cambia-942 F2). Nothing covered it: the suite stayed green with the
// Add/Done pair removed, because the only other test that reads the initial-state row polls
// games.status instead of waiting on the group, and the handlers-side cleanups that now depend
// on the wait (cambia-942 F4) leak silently rather than fail when it returns early.
package game

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/require"
)

// TestPersistInitialGameStateIsWaitable holds a row lock on the games row the initial-state
// upsert has to update, so the upsert is provably still in flight, and then asserts that
// PersistWG.Wait() refuses to return - which it can only do if persistInitialGameState registered
// its goroutine before launching it. Without the Add, Wait sees a zero counter and returns
// immediately, which is the failure mode: a caller that waits and then reads (or deletes) the row
// races a write it believes has landed.
func TestPersistInitialGameStateIsWaitable(t *testing.T) {
	setupGameDBTest(t)

	host := createGameDBTestUser(t, "initpersist-a-"+uuid.NewString())
	guest := createGameDBTestUser(t, "initpersist-b-"+uuid.NewString())

	g := NewCambiaGame()
	g.Emitter = newMockBroadcaster()
	g.LobbyID = uuid.New()
	g.HostUserID = host.ID
	g.LobbyType = "private"
	g.HouseRules = *testHouseRules(0, 2)
	g.TurnDuration = 0
	g.PersistWG = &sync.WaitGroup{}
	g.AddPlayer(&models.Player{ID: host.ID, Connected: true, User: &models.User{ID: host.ID}})
	g.AddPlayer(&models.Player{ID: guest.ID, Connected: true, User: &models.User{ID: guest.ID}})

	ctx := context.Background()

	// Seed exactly the rows persistInitialGameState's transaction will conflict with, so the
	// games INSERT ... ON CONFLICT DO UPDATE has an existing row to lock rather than a fresh
	// insert nothing can block. The sentinel payload doubles as the "the write never landed"
	// marker for the final assertion. Both rows are cleaned up by createGameDBTestUser's
	// registered cleanup: it deletes lobbies by host_user_id, which cascades games.
	_, err := database.DB.Exec(ctx,
		`INSERT INTO lobbies (id, host_user_id, type, mode, ranked) VALUES ($1, $2, 'private', 'casual', false)`,
		g.LobbyID, host.ID)
	require.NoError(t, err, "seed lobbies row")
	_, err = database.DB.Exec(ctx,
		`INSERT INTO games (id, lobby_id, status, initial_game_state, start_time) VALUES ($1, $2, 'in_progress', $3, NOW())`,
		g.ID, g.LobbyID, []byte(`{"sentinel":true}`))
	require.NoError(t, err, "seed games row")

	// Hold a row lock on that games row on a dedicated connection. The upsert blocks on it until
	// this transaction ends, so the window in which the write is unfinished is under the test's
	// control instead of being a race against the goroutine's own speed.
	conn, err := database.DB.Acquire(ctx)
	require.NoError(t, err, "acquire lock connection")
	tx, err := conn.Begin(ctx)
	require.NoError(t, err, "begin lock transaction")
	var lockedID uuid.UUID
	require.NoError(t,
		tx.QueryRow(ctx, `SELECT id FROM games WHERE id = $1 FOR UPDATE`, g.ID).Scan(&lockedID),
		"lock games row")
	var releaseOnce sync.Once
	release := func() {
		releaseOnce.Do(func() {
			_ = tx.Rollback(ctx)
			conn.Release()
		})
	}
	defer release()

	// BeginPreGame deals and calls persistInitialGameState under the game lock.
	g.BeginPreGame()
	require.True(t, g.PreGameActive, "PreGame should be active after BeginPreGame")

	waited := make(chan struct{})
	go func() {
		g.PersistWG.Wait()
		close(waited)
	}()
	select {
	case <-waited:
		t.Fatalf("PersistWG.Wait() returned while the initial-state upsert was still blocked on a row lock: persistInitialGameState is not registering its background write on PersistWG (cambia-908 L1), so every caller that drains the group before reading or deleting the row races it")
	case <-time.After(500 * time.Millisecond):
	}

	release()
	awaitPersistence(t, g.PersistWG, 10*time.Second)

	var raw []byte
	require.NoError(t,
		database.DB.QueryRow(ctx, `SELECT initial_game_state FROM games WHERE id = $1`, g.ID).Scan(&raw),
		"read back initial_game_state")
	var decoded struct {
		Sentinel      bool                      `json:"sentinel"`
		StockpileSize *int                      `json:"stockpileSize"`
		Players       map[string][]*models.Card `json:"players"`
	}
	require.NoError(t, json.Unmarshal(raw, &decoded), "decode initial_game_state")
	require.False(t, decoded.Sentinel, "the seeded sentinel payload survived: the upsert never landed")
	require.NotNil(t, decoded.StockpileSize, "initial_game_state carries no stockpileSize: %s", raw)
	require.Len(t, decoded.Players, 2, "initial_game_state should hold both dealt hands: %s", raw)

	// Leave the game in a state whose pre-game timer cannot outlive the test.
	g.StartGame()
	require.True(t, g.Started, "game should be marked as started")
}
