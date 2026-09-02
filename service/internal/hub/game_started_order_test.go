// internal/hub/game_started_order_test.go
//
// The server half of the game_started-before-sync exemption (cambia-1244).
//
// The web client discards every game frame while its gameState is null and only ever populates
// that field from private_sync_state, so gameStore.processGameWebSocketMessage gates on it. Two
// types are exempt from that gate: private_sync_state itself, and game_started (gameStore.ts,
// cambia-958 D6). The second exemption exists only because of the emission order this file pins:
// createAndStartGame emits game_started and then calls BeginPreGame, which is what sends the
// round's first private_sync_state. On the first round of a session there is no earlier snapshot,
// so game_started always lands on a null gameState and would be dropped without the exemption.
//
// The order is the load-bearing fact, and nothing else in this package asserts it. game_started
// carries the game id and is the frame that resets the previous round's terminal table, so a
// client that drops it renders the finished round's game-over UI over the new deal.
package hub

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// indexOfType returns the position of the first envelope of the given type, or -1.
func indexOfType(envs []Envelope, typ string) int {
	for i := range envs {
		if envs[i].Type == typ {
			return i
		}
	}
	return -1
}

// TestGameStartedPrecedesFirstSyncState drives the real createAndStartGame -> BeginPreGame path
// with a real CambiaGame emitting through the hub, and holds every participant's frame stream to
// the order the client's exemption assumes: game_started first, the round's first
// private_sync_state after it.
func TestGameStartedPrecedesFirstSyncState(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	defer h.Shutdown()

	h.CreateGame = func(_ *lobby.Lobby, playerIDs []uuid.UUID, _ map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
		g := game.NewCambiaGame()
		g.LobbyID = lob.ID
		g.Emitter = emitter
		// BeginPreGame arms a timer that flips the game to live play when this elapses. Parking it
		// an hour out keeps StartGame's own sync out of the frames under test without the test
		// waiting on, or racing, any clock: every assertion below runs after createAndStartGame has
		// returned, and the timer cannot fire inside the test binary's lifetime. Hub.Shutdown does
		// not stop it; it only closes the hub's shutdown channel.
		g.PreGameDuration = time.Hour
		for _, pid := range playerIDs {
			g.AddPlayer(&models.Player{
				ID:        pid,
				Connected: true,
				User:      &models.User{ID: pid},
			})
		}
		return g
	}

	connA := newFakeConn(idA, "A")
	connB := newFakeConn(idB, "B")
	h.conns[idA] = connA
	h.conns[idB] = connB

	require.True(t, h.createAndStartGame([]uuid.UUID{idA, idB}), "the game must be created")
	require.NotNil(t, h.Game, "the hub must route the created game")
	require.True(t, h.Game.PreGameActive, "BeginPreGame must have run, since it is what sends the first sync")

	for name, conn := range map[string]*Connection{"A": connA, "B": connB} {
		frames := drainEnvelopes(t, conn)

		started := indexOfType(frames, "game_started")
		require.GreaterOrEqual(t, started, 0, "client %s must receive game_started", name)

		firstSync := indexOfType(frames, string(game.EventPrivateSyncState))
		require.GreaterOrEqual(t, firstSync, 0, "client %s must receive a private_sync_state for the new round", name)

		// The exemption's premise: game_started is what arrives while the client still holds a null
		// gameState. Reverse this and the client-side exemption is dead code covering nothing, and
		// a client that ever tightens the gate loses the frame with no symptom until the second
		// round of a match renders over the first.
		assert.Less(t, started, firstSync,
			"client %s must receive game_started before the round's first private_sync_state", name)
	}
}

// TestGameStartedIsFirstFrameOfARound pins the same ordering across a round boundary, the case
// the client exemption's comment calls out by name: startNextRound re-invokes createAndStartGame
// while the store still holds the previous round's terminal state, and game_started is the frame
// that clears it. Asserting only the session's first round would leave the multi-round path,
// where a dropped game_started is visible as the old table surviving the new deal, uncovered.
func TestGameStartedIsFirstFrameOfARound(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	defer h.Shutdown()

	h.CreateGame = func(_ *lobby.Lobby, playerIDs []uuid.UUID, _ map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
		g := game.NewCambiaGame()
		g.LobbyID = lob.ID
		g.Emitter = emitter
		g.PreGameDuration = time.Hour
		for _, pid := range playerIDs {
			g.AddPlayer(&models.Player{ID: pid, Connected: true, User: &models.User{ID: pid}})
		}
		return g
	}

	connA := newFakeConn(idA, "A")
	h.conns[idA] = connA
	h.conns[idB] = newFakeConn(idB, "B")

	// Round one, drained away: what this test asserts is the shape of the NEXT round's stream.
	require.True(t, h.createAndStartGame([]uuid.UUID{idA, idB}))
	drainEnvelopes(t, connA)

	// A second round starts on the same connection. createAndStartGame refuses while h.Game is
	// set, which is the state startNextRound clears before re-invoking it.
	h.Game = nil
	require.True(t, h.createAndStartGame([]uuid.UUID{idA, idB}), "a second round must be creatable")

	frames := drainEnvelopes(t, connA)
	require.NotEmpty(t, frames, "the second round must emit to the still-connected client")
	assert.Equal(t, "game_started", frames[0].Type,
		"game_started must be the first frame of a round, ahead of every frame the client would drop against a stale gameState")
}
