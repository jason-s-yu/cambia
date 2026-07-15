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

// payloadOf decodes an envelope's payload into a generic map for field assertions.
func payloadOf(t *testing.T, env Envelope) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	require.NoError(t, json.Unmarshal(env.Payload, &m))
	return m
}

// findByType returns the first envelope of the given type, or nil.
func findByType(envs []Envelope, typ string) *Envelope {
	for i := range envs {
		if envs[i].Type == typ {
			return &envs[i]
		}
	}
	return nil
}

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

// TestGameEndedDrivesPostGameWithSharedSeq verifies that a casual game's end (the "_game_ended"
// synthetic message NotifyGameEnded queues) transitions the hub to PhasePostGame and emits
// phase_change to every connection at one shared seq (cambia-510). Pre-fix, attachOnGameEnd never
// drove this transition at all, so casual clients stayed on PhaseInGame after the game ended.
func TestGameEndedDrivesPostGameWithSharedSeq(t *testing.T) {
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
	h.Phase = PhaseInGame

	h.dispatch(ClientMsg{Type: "_game_ended"})

	assert.Equal(t, PhasePostGame, h.Phase, "hub must transition to PhasePostGame when the game ends")

	aFrames := drainEnvelopes(t, connA)
	bFrames := drainEnvelopes(t, connB)

	aChange := findByType(aFrames, "phase_change")
	bChange := findByType(bFrames, "phase_change")
	require.NotNil(t, aChange, "client A must receive phase_change")
	require.NotNil(t, bChange, "client B must receive phase_change")

	assert.Equal(t, "post_game", payloadOf(t, *aChange)["phase"])
	assert.Equal(t, "post_game", payloadOf(t, *bChange)["phase"])

	// One-seq-per-broadcast invariant (cambia-502): both recipients' copies of this one logical
	// phase_change broadcast must carry the same seq, matching h.seq after the emit.
	assert.Equal(t, aChange.Seq, bChange.Seq, "both clients must observe the same seq for phase_change")
	assert.Equal(t, h.seq, aChange.Seq, "phase_change seq must match the hub's current seq")
}

// TestGameEndedIgnoredOutsideInGame guards the phase check in the "_game_ended" dispatch case:
// a stray or duplicate notification must not clobber a hub that has already moved on.
func TestGameEndedIgnoredOutsideInGame(t *testing.T) {
	idA := uuid.New()
	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A", true)
	h.conns[idA] = connA
	h.Phase = PhaseOpen

	h.dispatch(ClientMsg{Type: "_game_ended"})

	assert.Equal(t, PhaseOpen, h.Phase, "_game_ended outside PhaseInGame must be a no-op")
	assert.False(t, containsType(drainEnvelopes(t, connA), "phase_change"), "no phase_change should fire when the guard rejects the transition")
}

// TestNotifyGameEndedQueuesSyntheticMessage verifies NotifyGameEnded is the safe cross-goroutine
// entry point attachOnGameEnd uses: it must enqueue onto h.incoming rather than mutating h.Phase
// directly, since OnGameEnd can run on a goroutine other than the hub's Run() loop.
func TestNotifyGameEndedQueuesSyntheticMessage(t *testing.T) {
	idA := uuid.New()
	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)

	h := NewHub(lob)
	h.Phase = PhaseInGame

	h.NotifyGameEnded()

	select {
	case msg := <-h.incoming:
		assert.Equal(t, "_game_ended", msg.Type)
	default:
		t.Fatal("NotifyGameEnded did not queue a message onto h.incoming")
	}
	// h.Phase is untouched until dispatch() (the Run() loop) processes the queued message.
	assert.Equal(t, PhaseInGame, h.Phase)
}
