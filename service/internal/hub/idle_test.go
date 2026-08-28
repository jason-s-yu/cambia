// internal/hub/idle_test.go
//
// Idle reaping (cambia-836). Closing a tab is not a leave, so a lobby whose members all walk away
// keeps its membership and never reaches OnEmpty; since cambia-808 gave the hub the lobby's
// lifetime, its goroutine then parks for the life of the process. A hub with nothing connected and
// no game in progress now tears its own lobby down after IdleTTL.
package hub

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// newIdleHub starts a running hub with a short idle window and a recording OnIdle.
func newIdleHub(t *testing.T, ttl time.Duration) (*Hub, uuid.UUID, chan uuid.UUID) {
	t.Helper()

	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)
	h.IdleTTL = ttl
	reaped := make(chan uuid.UUID, 4)
	h.OnIdle = func(id uuid.UUID) { reaped <- id }

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	waitAlive(t, h, true)
	t.Cleanup(h.Shutdown)

	return h, host, reaped
}

// awaitReap waits for a reap, returning false on timeout.
func awaitReap(reaped chan uuid.UUID, timeout time.Duration) (uuid.UUID, bool) {
	select {
	case id := <-reaped:
		return id, true
	case <-time.After(timeout):
		return uuid.Nil, false
	}
}

// TestIdleLobbyIsReaped is the headline: a hub that nobody ever connected to reclaims its lobby
// rather than parking forever.
func TestIdleLobbyIsReaped(t *testing.T) {
	h, _, reaped := newIdleHub(t, 40*time.Millisecond)

	id, ok := awaitReap(reaped, 2*time.Second)
	require.True(t, ok, "an idle lobby must be reaped")
	assert.Equal(t, h.ID, id, "the reap must name this lobby")
}

// TestIdleReapFiresAfterEveryoneDisconnects is the case the ticket describes: members with live
// membership all close their tabs, so nothing releases the lobby the deliberate way.
func TestIdleReapFiresAfterEveryoneDisconnects(t *testing.T) {
	h, host, reaped := newIdleHub(t, 60*time.Millisecond)

	conn := newFakeConn(host, "host")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))

	if _, ok := awaitReap(reaped, 150*time.Millisecond); ok {
		t.Fatal("a hub with a live connection must not be reaped")
	}

	h.Leave(host)

	id, ok := awaitReap(reaped, 2*time.Second)
	require.True(t, ok, "the last connection dropping must open an idle window")
	assert.Equal(t, h.ID, id)

	// Membership is untouched by the drop, which is exactly why the reaper has to exist.
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	joined, present := h.Lobby.Users[host]
	assert.True(t, present && joined, "the reap must not depend on membership having been released")
}

// TestIdleReapDoesNotFireWithALiveConnection guards the obvious false positive.
func TestIdleReapDoesNotFireWithALiveConnection(t *testing.T) {
	h, host, reaped := newIdleHub(t, 40*time.Millisecond)

	conn := newFakeConn(host, "host")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))

	if id, ok := awaitReap(reaped, 400*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped while a client was connected", id)
	}
	assert.True(t, h.Alive(), "the hub must still be serving")
}

// TestReconnectResetsTheIdleClock is what makes a generation counter necessary: a timer armed by
// an earlier departure must not fire on the lobby's new occupants. The window here is short
// enough that the first timer's deadline passes while the reconnected client is sitting there.
func TestReconnectResetsTheIdleClock(t *testing.T) {
	h, host, reaped := newIdleHub(t, 80*time.Millisecond)

	first := newFakeConn(host, "host")
	h.Join(first)
	require.NotNil(t, waitEnvelope(t, first, "lobby_state", time.Second))

	h.Leave(host) // arms the window
	time.Sleep(40 * time.Millisecond)

	second := newFakeConn(host, "host")
	h.Join(second)
	require.NotNil(t, waitEnvelope(t, second, "lobby_state", time.Second), "the reconnect must be served")

	// Well past the first window's deadline, and past a second full window too.
	if id, ok := awaitReap(reaped, 300*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped by a timer the reconnect should have cancelled", id)
	}

	// Dropping again opens a fresh window, so the clock was reset rather than disarmed.
	h.Leave(host)
	_, ok := awaitReap(reaped, 2*time.Second)
	assert.True(t, ok, "a later departure must arm a new idle window")
}

// TestIdleReapExemptsAnInProgressGame keeps the reaper away from a running table: every socket
// can be gone while the turn timers and the forfeit rule still have to reach their own end.
func TestIdleReapExemptsAnInProgressGame(t *testing.T) {
	h, _, reaped := newIdleHub(t, 40*time.Millisecond)

	h.Lobby.Mu.Lock()
	h.Lobby.InGame = true
	h.Lobby.Mu.Unlock()

	if id, ok := awaitReap(reaped, 400*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped mid-game", id)
	}
	assert.True(t, h.Alive(), "an in-game hub must keep running")

	// The exemption re-arms rather than disarming: once the game is over the lobby is reclaimed
	// without needing another departure to notice it.
	h.Lobby.Mu.Lock()
	h.Lobby.InGame = false
	h.Lobby.Mu.Unlock()

	_, ok := awaitReap(reaped, 2*time.Second)
	assert.True(t, ok, "the lobby must be reconsidered once the game ends")
}

// TestIdleReapDisabledByZeroTTL keeps the reaper opt-out honest for callers that build hubs
// directly.
func TestIdleReapDisabledByZeroTTL(t *testing.T) {
	_, _, reaped := newIdleHub(t, 0)

	if id, ok := awaitReap(reaped, 300*time.Millisecond); ok {
		t.Fatalf("lobby %s was reaped with the idle window disabled", id)
	}
}

// TestStaleIdleFireIsDiscarded pins the generation check the reconnect case leans on. Stop()
// cannot un-fire a timer that already ran, so a fire that was in flight when the window closed
// arrives at the Run loop regardless and must be ruled out by the generation it carries. Driven
// directly because the window it guards is a race that no sleep can schedule reliably.
func TestStaleIdleFireIsDiscarded(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)
	h.IdleTTL = time.Hour // nothing fires on its own; the test supplies the fires
	reaped := 0
	h.OnIdle = func(uuid.UUID) { reaped++ }
	defer h.Shutdown()

	h.armIdleReap()
	stale := h.idleGen
	h.cancelIdleReap() // a client reconnects, closing that window

	h.handleIdleReap(stale)
	assert.Equal(t, 0, reaped, "a fire from a superseded window must not reap the lobby")

	h.armIdleReap()
	h.handleIdleReap(h.idleGen)
	assert.Equal(t, 1, reaped, "the current window's fire must reap")
}

// TestHandleIdleReapLogsTheArmedWindow pins F2 (cambia-887): the reap log must name the window
// the fire's timer was actually armed with, not h.idleWindow() recomputed at fire time. The two
// diverge whenever EmptyIdleTTL/IdleTTL change between arm and fire - exactly the kind of drift a
// reconfiguration or a race could cause - and the log should describe what happened, not what the
// hub's current fields say now.
func TestHandleIdleReapLogsTheArmedWindow(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)

	h := NewHub(lob)
	h.IdleTTL = time.Hour
	h.EmptyIdleTTL = 50 * time.Millisecond // armIdleReap arms this, the shorter of the two
	reaped := 0
	h.OnIdle = func(uuid.UUID) { reaped++ }
	defer h.Shutdown()

	h.armIdleReap()
	require.NotNil(t, h.idleTimer, "arming with a positive IdleTTL and a non-nil OnIdle must start a timer")
	h.idleTimer.Stop() // the test fires handleIdleReap directly; the real timer must not race it

	// Mutate the TTLs after arming, before the fire: idleWindow() recomputed now would report a
	// different window than the one actually armed.
	h.EmptyIdleTTL = 9 * time.Second
	h.IdleTTL = 10 * time.Second

	var buf bytes.Buffer
	prevOutput := log.Writer()
	prevFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(prevOutput)
		log.SetFlags(prevFlags)
	}()

	h.handleIdleReap(h.idleGen)

	assert.Equal(t, 1, reaped, "the reap decision must still fire OnIdle")
	logged := buf.String()
	assert.Contains(t, logged, "50ms", "the reap log must name the window the timer was armed with")
	assert.False(t, strings.Contains(logged, "9s") || strings.Contains(logged, "10s"),
		"the reap log must not name a window recomputed from fields mutated after arming, got: %s", logged)
}

// TestIdleReapDroppedByShutdown guards the timer contract the postgame reset already follows: a
// hub torn down inside its idle window must not leave a fire behind.
func TestIdleReapDroppedByShutdown(t *testing.T) {
	h, _, reaped := newIdleHub(t, 60*time.Millisecond)

	h.Shutdown()
	waitAlive(t, h, false)

	if id, ok := awaitReap(reaped, 400*time.Millisecond); ok {
		t.Fatalf("a shut-down hub reaped lobby %s", id)
	}
}
