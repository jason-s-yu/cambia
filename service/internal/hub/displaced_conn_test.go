// internal/hub/displaced_conn_test.go
//
// Two tabs on one hub (cambia-1543). h.conns is keyed by user, so a second socket overwrote the
// first without closing it, and the leave that eventually followed the first socket's own read
// failure deleted and closed whatever was registered under that user id: the live tab. The
// client read that clean StatusGoingAway close as deliberate and stopped retrying, and the
// disconnect grace the same leave armed expired into a forfeit. Closing one tab forfeited the
// seat the other tab was playing. The displaced socket was meanwhile registered nowhere and
// still had its actions accepted.
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

// newTrackedConn is newFakeConn with an observable close. A real Connection's cancel func is its
// socket context's cancel, and Close calls it, so a closed connection is one whose context is
// done: that is what tells a socket the hub evicted from one it left alone.
func newTrackedConn(userID uuid.UUID, username string) (*Connection, context.Context) {
	ctx, cancel := context.WithCancel(context.Background())
	return &Connection{
		ID:       uuid.New(),
		UserID:   userID,
		Username: username,
		outChan:  make(chan []byte, 32),
		cancel:   cancel,
	}, ctx
}

// waitDone reports whether ctx was cancelled inside the timeout.
func waitDone(ctx context.Context, timeout time.Duration) bool {
	select {
	case <-ctx.Done():
		return true
	case <-time.After(timeout):
		return false
	}
}

// runHub starts a stopped hub's Run loop and waits for it to be serving.
func runHub(t *testing.T, h *Hub) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	waitAlive(t, h, true)
}

// TestJoinClosesTheSocketItDisplaces pins the invariant the rest of this rests on: one user, one
// socket. A second tab overwrote the map entry and left the first socket live but registered
// nowhere, still able to submit actions on a seat every reply went to its replacement for.
func TestJoinClosesTheSocketItDisplaces(t *testing.T) {
	user := uuid.New()
	lob := lobby.NewLobbyWithDefaults(user)
	lob.JoinUser(user)

	h := NewHub(lob)
	h.IdleTTL = 0
	runHub(t, h)
	t.Cleanup(h.Shutdown)

	first, firstCtx := newTrackedConn(user, "tab-one")
	h.Join(first)
	require.NotNil(t, waitEnvelope(t, first, "lobby_state", 2*time.Second), "the first tab must be served")

	second, secondCtx := newTrackedConn(user, "tab-two")
	h.Join(second)
	require.NotNil(t, waitEnvelope(t, second, "lobby_state", 2*time.Second), "the second tab must be served")

	assert.True(t, waitDone(firstCtx, 2*time.Second), "the join must close the socket it displaced")
	assert.NoError(t, secondCtx.Err(), "the socket that displaced it must stay open")
	assert.Same(t, second, h.getConn(user), "the newer socket must be the registered one")
	assert.Equal(t, 1, h.connCount(), "a user may hold one connection on a hub")
}

// TestLeaveFromADisplacedTabLeavesTheLiveSocketAlone is the reported bug, on the two-player
// in-game fixture it was reproduced on: the first tab's teardown must not evict the second tab,
// must not tell the game anybody dropped, and must not arm the grace window that forfeits the
// seat. The frames the displaced socket keeps sending must be dropped too.
func TestLeaveFromADisplacedTabLeavesTheLiveSocketAlone(t *testing.T) {
	h, ids, g, ended := newInGameHubStopped(t, 2, true, 0, 0, 3*time.Second)
	runHub(t, h)

	survivor, _ := newTrackedConn(ids[0], "P0")
	h.Join(survivor)
	require.NotNil(t, waitEnvelope(t, survivor, "lobby_state", 2*time.Second))

	first, firstCtx := newTrackedConn(ids[1], "P1-tab-one")
	h.Join(first)
	require.NotNil(t, waitEnvelope(t, first, "lobby_state", 2*time.Second))

	second, secondCtx := newTrackedConn(ids[1], "P1-tab-two")
	h.Join(second)
	require.NotNil(t, waitEnvelope(t, second, "lobby_state", 2*time.Second))
	require.True(t, waitDone(firstCtx, 2*time.Second), "the second tab must displace the first")
	require.Same(t, second, h.getConn(ids[1]))

	drainEnvelopes(t, survivor)
	drainEnvelopes(t, second)

	// The first tab's ReadPump finally returns, up to a ping interval late.
	h.LeaveConn(first)

	// The negative that carries the ticket: nothing about the live seat may change. This wait is
	// also what gives the hub time to process the leave before the assertions below read state.
	assert.Nil(t, waitEnvelope(t, survivor, "player_reconnecting", 500*time.Millisecond),
		"a displaced tab closing must not open a grace window on the live seat")

	assert.Same(t, second, h.getConn(ids[1]), "the live socket must still be registered")
	assert.NoError(t, secondCtx.Err(), "the live socket must still be open")
	assert.True(t, connectedIn(t, g, ids[0], ids[1]), "the game must not see the seat as dropped")
	for _, p := range g.GetCurrentObfuscatedGameState(ids[0]).Players {
		if p.PlayerID == ids[1] {
			assert.Nil(t, p.ReconnectDeadline, "no grace timer may be armed")
			assert.False(t, p.Forfeited, "the seat must not forfeit")
		}
	}
	select {
	case res := <-ended:
		t.Fatalf("closing a displaced tab ended the game (winner %s, scores %v)", res.winner, res.scores)
	default:
	}

	// A frame that arrives on the displaced socket is dropped before dispatch does anything with
	// it. The deliberately stale last_seq is the tell: a dispatched frame answers the registered
	// connection with a sync_state repair, and the control below shows that answer does arrive
	// when the frame comes from the socket that is actually registered.
	h.Incoming() <- ClientMsg{ConnID: first.ID, UserID: ids[1], LastSeq: 0, Type: "action_draw_stockpile"}
	assert.Nil(t, waitEnvelope(t, second, "sync_state", 500*time.Millisecond),
		"a frame from a displaced socket must not be dispatched")

	h.Incoming() <- ClientMsg{ConnID: second.ID, UserID: ids[1], LastSeq: 0, Type: "action_draw_stockpile"}
	assert.NotNil(t, waitEnvelope(t, second, "sync_state", 2*time.Second),
		"the control: a frame from the registered socket is dispatched")
}

// TestLeaveByUserStillDropsWhicheverSocketIsCurrent keeps the user-addressed entry point honest.
// It is what the deliberate HTTP leave calls, which holds no socket of its own, so it has to
// drop whatever is registered now rather than a connection it was handed earlier.
func TestLeaveByUserStillDropsWhicheverSocketIsCurrent(t *testing.T) {
	user := uuid.New()
	lob := lobby.NewLobbyWithDefaults(user)
	lob.JoinUser(user)

	h := NewHub(lob)
	h.IdleTTL = 0
	runHub(t, h)
	t.Cleanup(h.Shutdown)

	first, _ := newTrackedConn(user, "tab-one")
	h.Join(first)
	require.NotNil(t, waitEnvelope(t, first, "lobby_state", 2*time.Second))

	second, secondCtx := newTrackedConn(user, "tab-two")
	h.Join(second)
	require.NotNil(t, waitEnvelope(t, second, "lobby_state", 2*time.Second))

	h.Leave(user)

	assert.True(t, waitDone(secondCtx, 2*time.Second), "a user-addressed leave must drop the current socket")
	assert.Nil(t, h.getConn(user), "the user must hold no connection afterwards")
}
