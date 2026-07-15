// internal/hub/seq_test.go
package hub

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newFakeConn builds a Connection backed by a buffered outChan (no real WebSocket). Send and
// SendEnvelope only touch outChan and cancel, so tests can read the frames the hub emitted.
func newFakeConn(userID uuid.UUID, username string, isHost bool) *Connection {
	return &Connection{
		ID:       uuid.New(),
		UserID:   userID,
		Username: username,
		IsHost:   isHost,
		outChan:  make(chan []byte, 32),
		cancel:   func() {},
	}
}

// drainEnvelopes reads every frame currently queued on conn without blocking.
func drainEnvelopes(t *testing.T, conn *Connection) []Envelope {
	t.Helper()
	var out []Envelope
	for {
		select {
		case data := <-conn.outChan:
			var env Envelope
			require.NoError(t, json.Unmarshal(data, &env))
			out = append(out, env)
		default:
			return out
		}
	}
}

func maxSeq(envs []Envelope) uint64 {
	var m uint64
	for _, e := range envs {
		if e.Seq > m {
			m = e.Seq
		}
	}
	return m
}

func containsType(envs []Envelope, typ string) bool {
	for _, e := range envs {
		if e.Type == typ {
			return true
		}
	}
	return false
}

// TestBroadcastLobbyUpdateStampsSharedSeq verifies the one-seq-per-broadcast invariant: a single
// broadcastLobbyUpdate leaves every recipient caught up to the same (current) seq. Pre-fix, EmitTo
// bumped seq per recipient, so with two connections one client always trailed h.seq (cambia-502).
func TestBroadcastLobbyUpdateStampsSharedSeq(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A", true)
	connB := newFakeConn(idB, "B", false)
	h.conns[idA] = connA
	h.conns[idB] = connB

	h.broadcastLobbyUpdate()

	seqA := maxSeq(drainEnvelopes(t, connA))
	seqB := maxSeq(drainEnvelopes(t, connB))

	assert.Equal(t, seqB, seqA, "both clients must observe the same seq after one broadcast")
	assert.Equal(t, h.seq, seqA, "both clients must be caught up to the hub's current seq")
}

// TestReadyRaceAfterBroadcastAcceptsEachClientEcho reproduces the 2-client ready race: after a
// lobby broadcast, each client echoing its own last-seen seq on a ready must be accepted, not gated
// as stale (which would emit a sync_state and silently drop the ready). Pre-fix, the first client's
// ready was dropped because it never held the broadcast's final seq (cambia-502).
func TestReadyRaceAfterBroadcastAcceptsEachClientEcho(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	// Isolate the seq gate from countdown machinery: a full ready set with AutoStart would trigger
	// beginCountdown (timers, phase change), which is unrelated to what this test guards.
	lob.LobbySettings.AutoStart = false
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A", true)
	connB := newFakeConn(idB, "B", false)
	h.conns[idA] = connA
	h.conns[idB] = connB

	// One logical lobby broadcast, as would fire when the second player joins.
	h.broadcastLobbyUpdate()
	seqA := maxSeq(drainEnvelopes(t, connA))
	seqB := maxSeq(drainEnvelopes(t, connB))
	require.Equal(t, seqA, seqB, "precondition: both clients caught up to the same seq")

	// Client A echoes its last-seen seq with a ready. It must be handled, not rejected.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: seqA, Type: "ready"})
	aFrames := drainEnvelopes(t, connA)
	assert.False(t, containsType(aFrames, "sync_state"), "client A's ready must not be rejected as stale")
	assert.True(t, containsType(aFrames, "lobby_state"), "accepted ready should broadcast an updated lobby_state")
	assert.True(t, lob.ReadyStates[idA], "client A should be marked ready")

	// A's ready triggered a fresh broadcast; client B, like the real client, now echoes the newest
	// seq it has seen. It too must be accepted.
	seqB = maxSeq(drainEnvelopes(t, connB))
	require.Equal(t, h.seq, seqB, "client B should have received the post-ready broadcast at the current seq")
	h.dispatch(ClientMsg{UserID: idB, LastSeq: seqB, Type: "ready"})
	bFrames := drainEnvelopes(t, connB)
	assert.False(t, containsType(bFrames, "sync_state"), "client B's ready must not be rejected as stale")
	assert.True(t, containsType(bFrames, "lobby_state"), "accepted ready should broadcast an updated lobby_state")
	assert.True(t, lob.ReadyStates[idB], "client B should be marked ready")
}
