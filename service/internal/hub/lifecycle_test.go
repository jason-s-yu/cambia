// internal/hub/lifecycle_test.go
//
// Hub lifetime (cambia-808). A hub used to return from Run() as soon as its last connection
// left, while staying registered with its owner: the next WebSocket to that lobby was accepted
// and then never answered, and the phase, the routing to a live game and a ranked match's
// cumulative scores went with it. The hub now outlives its connections and stops only when its
// lobby is torn down, deregistering itself on the way out.
package hub

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// waitAlive blocks until the hub's liveness matches want, or fails the test.
func waitAlive(t *testing.T, h *Hub, want bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.Alive() == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("hub %s: expected Alive()==%v within timeout", h.ID, want)
}

// waitEnvelope waits for a frame of the given type on a fake connection.
func waitEnvelope(t *testing.T, conn *Connection, typ string, timeout time.Duration) *Envelope {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		select {
		case data := <-conn.outChan:
			var env Envelope
			require.NoError(t, json.Unmarshal(data, &env))
			if env.Type == typ {
				return &env
			}
		case <-time.After(10 * time.Millisecond):
		}
	}
	return nil
}

// TestHubSurvivesLastConnectionLeaving is the cambia-808 regression: the hub must still be
// running, and must still answer a fresh connection, after every client has disconnected. This
// is the everyone-blinked case, where the lobby and its members are all still there.
func TestHubSurvivesLastConnectionLeaving(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	waitAlive(t, h, true)
	defer h.Shutdown()

	first := newFakeConn(host, "host")
	h.Join(first)
	require.NotNil(t, waitEnvelope(t, first, "lobby_state", time.Second), "the first connection must be served")

	h.Leave(host)

	// The hub keeps running: nothing dissolved it, because the lobby still exists.
	time.Sleep(50 * time.Millisecond)
	assert.True(t, h.Alive(), "a hub whose lobby still exists must keep running with no connections")

	// And the reconnect that used to hang is served.
	second := newFakeConn(host, "host")
	h.Join(second)
	assert.NotNil(t, waitEnvelope(t, second, "lobby_state", time.Second),
		"a reconnect after every client dropped must be answered, not swallowed")
}

// TestHubKeepsMatchStateAcrossFullDisconnect covers what dissolving used to throw away: state
// that lives only on the hub and cannot be rebuilt from the lobby store.
func TestHubKeepsMatchStateAcrossFullDisconnect(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)
	h.IsRanked = true
	h.TotalRounds = 3
	h.RoundsPlayed = 1
	h.CumulativeScores[host] = 7

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	waitAlive(t, h, true)
	defer h.Shutdown()

	conn := newFakeConn(host, "host")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))
	h.Leave(host)
	time.Sleep(50 * time.Millisecond)

	rejoin := newFakeConn(host, "host")
	h.Join(rejoin)
	env := waitEnvelope(t, rejoin, "lobby_state", time.Second)
	require.NotNil(t, env, "the reconnect must be served")

	match, ok := payloadOf(t, *env)["match_state"].(map[string]interface{})
	require.True(t, ok, "a ranked hub must still report its match state after a full disconnect")
	assert.Equal(t, float64(1), match["current_round"], "rounds played must survive the disconnect")
	assert.Equal(t, float64(7), match["cumulative_scores"].(map[string]interface{})[host.String()],
		"cumulative scores must survive the disconnect")
}

// TestHubDeregistersItselfWhenItStops checks the invariant that makes the swallow unreachable:
// a hub that stops serving stops being discoverable, whatever stopped it.
func TestHubDeregistersItselfWhenItStops(t *testing.T) {
	for _, tc := range []struct {
		name string
		stop func(h *Hub, cancel context.CancelFunc)
	}{
		{"shutdown", func(h *Hub, _ context.CancelFunc) { h.Shutdown() }},
		{"context cancel", func(_ *Hub, cancel context.CancelFunc) { cancel() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			host := uuid.New()
			lob := lobby.NewLobbyWithDefaults(host)
			lob.JoinUser(host)

			h := NewHub(lob)
			dissolved := make(chan uuid.UUID, 1)
			h.OnDissolve = func(id uuid.UUID) { dissolved <- id }

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go h.Run(ctx)
			waitAlive(t, h, true)

			tc.stop(h, cancel)

			select {
			case id := <-dissolved:
				assert.Equal(t, h.ID, id, "the hub must deregister itself under its own id")
			case <-time.After(2 * time.Second):
				t.Fatal("a stopped hub never deregistered itself")
			}
			waitAlive(t, h, false)
		})
	}
}

// TestJoinOnStoppedHubClosesConnection covers the last way a socket could be accepted and never
// served: handing a connection to a hub that has already stopped. It must be closed, not parked.
func TestJoinOnStoppedHubClosesConnection(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	waitAlive(t, h, true)

	h.Shutdown()
	waitAlive(t, h, false)

	closed := make(chan struct{})
	conn := newFakeConn(host, "host")
	conn.cancel = func() { close(closed) }

	h.Join(conn)

	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("a connection handed to a stopped hub was neither served nor closed")
	}
}

// TestCleanupClosesQueuedConnections covers the narrow window on the other side of the same
// hand-off: a connection queued for a hub that stops before it registers must be closed by the
// hub's exit path rather than left sitting in the channel.
func TestCleanupClosesQueuedConnections(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	h := NewHub(lob)

	closed := make(chan struct{})
	queued := newFakeConn(host, "host")
	queued.cancel = func() { close(closed) }
	h.join <- queued // Run() is not started: the connection is queued and nothing will pick it up

	h.exit()

	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("a connection left queued on the join channel was never closed")
	}
}

// TestShutdownIsIdempotent guards the teardown path: the lobby owner stops the hub and the hub's
// own exit closes the same channel, and a second close would panic the process.
func TestShutdownIsIdempotent(t *testing.T) {
	lob := lobby.NewLobbyWithDefaults(uuid.New())
	h := NewHub(lob)

	assert.NotPanics(t, func() {
		h.Shutdown()
		h.Shutdown()
		h.exit()
	}, "closing the shutdown channel twice must not panic")
}

// TestLeaveDoesNotTouchMembership is the other half of cambia-807: connection-level departures
// (a dropped socket, a closed tab) must leave the lobby membership alone, or reconnecting and
// the resume banner would both break.
func TestLeaveDoesNotTouchMembership(t *testing.T) {
	host := uuid.New()
	other := uuid.New()

	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(other)

	h := NewHub(lob)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	waitAlive(t, h, true)
	defer h.Shutdown()

	conn := newFakeConn(other, "other")
	h.Join(conn)
	require.NotNil(t, waitEnvelope(t, conn, "lobby_state", time.Second))

	h.Leave(other)
	time.Sleep(50 * time.Millisecond)

	lob.Mu.Lock()
	defer lob.Mu.Unlock()
	joined, present := lob.Users[other]
	assert.True(t, present && joined, "a dropped connection must not release lobby membership")
}
