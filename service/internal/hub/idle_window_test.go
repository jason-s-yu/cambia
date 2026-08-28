// internal/hub/idle_window_test.go
//
// Two idle windows (cambia-884). The 45 minute window from cambia-836 exists to carry a live
// game across a disconnect, but it was also what an abandoned pre-game or post-game lobby sat
// through, leaving dead lobbies on the dashboard for the best part of an hour. A hub with no
// game in progress now reaps on the much shorter EmptyIdleTTL, and a running game is exempt from
// the reap decision itself for as long as it runs.
package hub

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// newWindowHub starts a hub with both idle windows set explicitly, optionally already in game.
func newWindowHub(t *testing.T, idleTTL, emptyTTL time.Duration, inGame bool) (*Hub, uuid.UUID, chan uuid.UUID) {
	t.Helper()

	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.InGame = inGame

	h := NewHub(lob)
	h.IdleTTL = idleTTL
	h.EmptyIdleTTL = emptyTTL
	reaped := make(chan uuid.UUID, 4)
	h.OnIdle = func(id uuid.UUID) { reaped <- id }

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	waitAlive(t, h, true)
	t.Cleanup(h.Shutdown)

	return h, host, reaped
}

// TestEmptyLobbyReapsOnTheShortWindow is the headline: with no game in progress the hub does not
// wait out the long window.
func TestEmptyLobbyReapsOnTheShortWindow(t *testing.T) {
	h, _, reaped := newWindowHub(t, 30*time.Second, 40*time.Millisecond, false)

	id, ok := awaitReap(reaped, 2*time.Second)
	require.True(t, ok, "a lobby with no game in progress must reap on the short window")
	assert.Equal(t, h.ID, id)
}

// TestInGameLobbyIsNeverReapedWhileItsGameRuns is the guard the short window needs: a table whose
// players all dropped mid-game is not reclaimed under them, so the turn timers and the forfeit
// rule (cambia-837) still get to finish the game. What protects it is the reap-time exemption and
// not the length of the window: many short windows elapse here and each fire declines and re-arms.
func TestInGameLobbyIsNeverReapedWhileItsGameRuns(t *testing.T) {
	_, _, reaped := newWindowHub(t, 30*time.Second, 40*time.Millisecond, true)

	if id, ok := awaitReap(reaped, 500*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped while its game was in progress", id)
	}
}

// TestShortWindowAppliesOnceTheGameEnds covers the case the dashboard actually showed: the game
// finished, everyone closed their tab, and the lobby lingered. The window that opens on that
// departure is the short one, not the game's grace.
func TestShortWindowAppliesOnceTheGameEnds(t *testing.T) {
	h, host, reaped := newWindowHub(t, 30*time.Second, 50*time.Millisecond, true)

	conn := newFakeConn(host, "host")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))

	// The game ends; the results screen is still up, then the last tab closes.
	h.Lobby.Mu.Lock()
	h.Lobby.InGame = false
	h.Lobby.Mu.Unlock()
	h.Leave(host)

	id, ok := awaitReap(reaped, 2*time.Second)
	require.True(t, ok, "an abandoned post-game lobby must reap on the short window")
	assert.Equal(t, h.ID, id)
}

// TestShortWindowAppliesWhenTheGameEndsAfterTheDrop is the ordering an arm-time choice of window
// used to miss: the table drops first, so the window opens while the game is still in progress,
// and the game ends later with nobody connected. Nothing re-arms on that transition, so a hub that
// had armed the long TTL would hold the finished game's lobby for the whole of it.
func TestShortWindowAppliesWhenTheGameEndsAfterTheDrop(t *testing.T) {
	h, host, reaped := newWindowHub(t, 30*time.Second, 50*time.Millisecond, true)

	conn := newFakeConn(host, "host")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))

	h.Leave(host) // the window opens here, with the game still running

	if id, ok := awaitReap(reaped, 300*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped while its game was still in progress", id)
	}

	// The game ends with nobody connected and nothing else to notice it.
	h.Lobby.Mu.Lock()
	h.Lobby.InGame = false
	h.Lobby.Mu.Unlock()

	id, ok := awaitReap(reaped, 2*time.Second)
	require.True(t, ok, "a game ending after its table dropped must not hold the lobby for the long window")
	assert.Equal(t, h.ID, id)
}

// TestShortWindowNeverExceedsTheIdleTTL keeps the two settings ordered: a deployment (or a test)
// that lowers the overall idle TTL below the empty-lobby one gets the lower of the two rather
// than a lobby that outlives its own idle window.
func TestShortWindowNeverExceedsTheIdleTTL(t *testing.T) {
	_, _, reaped := newWindowHub(t, 40*time.Millisecond, 30*time.Second, false)

	_, ok := awaitReap(reaped, 2*time.Second)
	assert.True(t, ok, "the effective window must be the shorter of the two")
}

// TestIdleWindowsDisabledByZeroIdleTTL keeps the opt-out from cambia-836 intact: IdleTTL zero
// switches reaping off however the empty window is set.
func TestIdleWindowsDisabledByZeroIdleTTL(t *testing.T) {
	_, _, reaped := newWindowHub(t, 0, 40*time.Millisecond, false)

	if id, ok := awaitReap(reaped, 300*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped with the idle window disabled", id)
	}
}
