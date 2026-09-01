// internal/hub/postgame_test.go
package hub

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// newPostGameHub builds a two-player hub sitting in PhasePostGame with a finished game still
// attached and the lobby still flagged in-game, i.e. the state a casual lobby is left in after
// its first game ends.
func newPostGameHub(t *testing.T) (*Hub, uuid.UUID, uuid.UUID, *Connection, *Connection) {
	t.Helper()

	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A")
	connB := newFakeConn(idB, "B")
	h.conns[idA] = connA
	h.conns[idB] = connB

	finished := game.NewCambiaGame()
	h.Game = finished
	h.Phase = PhasePostGame

	lob.InGame = true
	lob.GameID = finished.ID
	lob.GameInstanceCreated = true
	lob.ReadyStates[idA] = true
	lob.ReadyStates[idB] = true

	return h, idA, idB, connA, connB
}

// TestReturnToLobbyReopensLobbyForNextGame is the cambia-793 regression: PhasePostGame was
// terminal and h.Game was never cleared, so beginGame/createAndStartGame refused every later
// start and a lobby could never play a second game. The reset must drop the finished game, clear
// the lobby's in-game flags, unready everyone, and return the hub to PhaseOpen.
func TestReturnToLobbyReopensLobbyForNextGame(t *testing.T) {
	h, idA, idB, connA, connB := newPostGameHub(t)

	h.dispatch(ClientMsg{Type: "_return_to_lobby"})

	assert.Equal(t, PhaseOpen, h.Phase, "hub must return to PhaseOpen after post-game")
	assert.Nil(t, h.Game, "the finished game must be cleared so the next one can be created")
	assert.False(t, h.Lobby.InGame, "lobby must no longer be flagged in-game")
	assert.Equal(t, uuid.Nil, h.Lobby.GameID, "lobby must no longer point at the finished game")
	assert.False(t, h.Lobby.GameInstanceCreated, "lobby must no longer claim a live game instance")
	assert.False(t, h.Lobby.ReadyStates[idA], "player A must be unready")
	assert.False(t, h.Lobby.ReadyStates[idB], "player B must be unready")

	aFrames := drainEnvelopes(t, connA)
	bFrames := drainEnvelopes(t, connB)

	aChange := findByType(aFrames, "phase_change")
	bChange := findByType(bFrames, "phase_change")
	require.NotNil(t, aChange, "client A must receive phase_change")
	require.NotNil(t, bChange, "client B must receive phase_change")
	assert.Equal(t, "open", payloadOf(t, *aChange)["phase"])
	assert.Equal(t, "open", payloadOf(t, *bChange)["phase"])

	// One-seq-per-broadcast invariant (cambia-502): both copies of the one logical phase_change
	// carry the same seq.
	assert.Equal(t, aChange.Seq, bChange.Seq, "both clients must observe the same phase_change seq")

	// The refreshed lobby snapshot is what lets a client re-render the ready controls.
	assert.True(t, containsType(aFrames, "lobby_state"), "client A must receive a refreshed lobby_state")
	assert.True(t, containsType(bFrames, "lobby_state"), "client B must receive a refreshed lobby_state")
}

// TestSecondGameStartsAfterPostGameReset drives the actual user-visible bug end to end at the hub
// level: after game one ends and the post-game reset lands, both players readying again must run
// the existing round-one creation path a second time.
func TestSecondGameStartsAfterPostGameReset(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	// Both timers run for real and the test dispatches what they queue. The countdown used to be
	// set to an hour so it could never land a stray _begin_game on h.incoming (nothing drains it
	// here; Run() is not started), which is exactly the stray fire cambia-1557 made harmless: a
	// _begin_game now carries the generation of the countdown that armed it.
	h.CountdownDuration = 20 * time.Millisecond
	h.PostGameDuration = 20 * time.Millisecond

	defer h.Shutdown() // releases any timer still parked

	created := 0
	h.CreateGame = func(_ *lobby.Lobby, playerIDs []uuid.UUID, _ map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
		created++
		g := game.NewCambiaGame()
		g.LobbyID = lob.ID
		g.Emitter = emitter
		return g
	}

	connA := newFakeConn(idA, "A")
	connB := newFakeConn(idB, "B")
	h.conns[idA] = connA
	h.conns[idB] = connB

	// Game one: both ready -> countdown -> creation.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseCountdown, h.Phase, "both ready should start the countdown")
	begin := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_begin_game", begin.Type, "the countdown must schedule a game start")
	h.dispatch(begin)
	require.Equal(t, PhaseInGame, h.Phase)
	require.Equal(t, 1, created, "game one must be created")
	require.NotNil(t, h.Game)

	// Game one ends: the hub shows results, then schedules its own return to the lobby.
	h.dispatch(ClientMsg{Type: "_game_ended"})
	require.Equal(t, PhasePostGame, h.Phase)

	msg := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_return_to_lobby", msg.Type, "post-game must schedule a return to the lobby")
	h.dispatch(msg)
	require.Equal(t, PhaseOpen, h.Phase)

	// Game two: the same ready -> countdown -> creation path must run again.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseCountdown, h.Phase, "a re-readied lobby must count down again")
	begin = waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_begin_game", begin.Type, "the second countdown must schedule its own start")
	h.dispatch(begin)

	assert.Equal(t, PhaseInGame, h.Phase, "the second game must start")
	assert.Equal(t, 2, created, "the game factory must run a second time")
	require.NotNil(t, h.Game, "the hub must route the second game")
}

// TestGameEndedSchedulesPostGameReset verifies the trigger: entering PhasePostGame arms the timer
// that queues "_return_to_lobby" back onto the hub goroutine, mirroring scheduleGameStart so the
// transition itself never runs on the timer goroutine.
func TestGameEndedSchedulesPostGameReset(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	h.PostGameDuration = 20 * time.Millisecond
	h.conns[idA] = newFakeConn(idA, "A")
	h.conns[idB] = newFakeConn(idB, "B")
	h.Phase = PhaseInGame

	h.dispatch(ClientMsg{Type: "_game_ended"})
	require.Equal(t, PhasePostGame, h.Phase)

	msg := waitForIncoming(t, h, 2*time.Second)
	assert.Equal(t, "_return_to_lobby", msg.Type)
	// The phase only moves when the hub goroutine dispatches the queued message.
	assert.Equal(t, PhasePostGame, h.Phase, "the timer goroutine must not mutate the phase itself")
}

// TestPostGameResetCancelledByShutdown guards the goroutine contract: a hub that shuts down while
// showing results must not leave a pending reset behind.
func TestPostGameResetCancelledByShutdown(t *testing.T) {
	idA := uuid.New()
	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)

	h := NewHub(lob)
	h.PostGameDuration = time.Hour // never fires on its own
	h.Phase = PhaseInGame
	h.conns[idA] = newFakeConn(idA, "A")

	h.dispatch(ClientMsg{Type: "_game_ended"})
	h.Shutdown()

	select {
	case msg := <-h.incoming:
		t.Fatalf("shutdown must drop the pending reset, got %q", msg.Type)
	case <-time.After(100 * time.Millisecond):
	}
}

// TestReturnToLobbyIgnoredOutsidePostGame guards the phase check: a stale reset must not clobber a
// hub that has already moved on (e.g. a second game already under way).
func TestReturnToLobbyIgnoredOutsidePostGame(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A")
	h.conns[idA] = connA
	h.Phase = PhaseInGame
	live := game.NewCambiaGame()
	h.Game = live

	h.dispatch(ClientMsg{Type: "_return_to_lobby"})

	assert.Equal(t, PhaseInGame, h.Phase, "_return_to_lobby outside PhasePostGame must be a no-op")
	assert.Same(t, live, h.Game, "a live game must not be dropped by a stale reset")
	assert.False(t, containsType(drainEnvelopes(t, connA), "phase_change"), "no phase_change should fire when the guard rejects the reset")
}

// waitForIncoming pops the next synthetic message the hub queued for its Run() loop.
func waitForIncoming(t *testing.T, h *Hub, timeout time.Duration) ClientMsg {
	t.Helper()
	select {
	case msg := <-h.incoming:
		return msg
	case <-time.After(timeout):
		t.Fatal("no message was queued onto h.incoming")
		return ClientMsg{}
	}
}
