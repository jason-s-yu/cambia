// internal/hub/countdown_gen_test.go
//
// Countdown generations (cambia-1557). scheduleGameStart fired a bare _begin_game with nothing
// naming the countdown it belonged to. An unready during a countdown dropped the hub to
// PhaseOpen without cancelling that pending fire, readying again armed a second timer while the
// first was still in flight, and dispatch admitted any _begin_game in PhaseCountdown: the first
// countdown's fire landed inside the second countdown and started the game early. Reachable
// straight from the UI, where auto-start is on and the ready toggle stays live mid-countdown.
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

// newCountdownHub builds a two-seat lobby hub with a counting game factory and both players
// connected, with Run() left stopped so the test dispatches every synthetic message itself and
// controls exactly when each countdown's fire lands.
func newCountdownHub(t *testing.T, countdown time.Duration) (*Hub, uuid.UUID, uuid.UUID, *int) {
	t.Helper()

	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	h.CountdownDuration = countdown
	t.Cleanup(h.Shutdown) // releases any timer still parked

	created := 0
	h.CreateGame = func(_ *lobby.Lobby, _ []uuid.UUID, _ map[uuid.UUID]string, emitter game.Emitter) *game.CambiaGame {
		created++
		g := game.NewCambiaGame()
		g.LobbyID = lob.ID
		g.Emitter = emitter
		return g
	}

	h.conns[idA] = newFakeConn(idA, "A")
	h.conns[idB] = newFakeConn(idB, "B")
	return h, idA, idB, &created
}

// TestAbortedCountdownsFireCannotStartTheSecondCountdownsGame is the reported bug: ready,
// unready, ready again, then deliver the first countdown's _begin_game. It must be dropped, and
// the second countdown's own fire must still start the game at its own deadline.
func TestAbortedCountdownsFireCannotStartTheSecondCountdownsGame(t *testing.T) {
	h, idA, idB, created := newCountdownHub(t, 30*time.Millisecond)

	// First countdown.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseCountdown, h.Phase, "both ready must start the countdown")
	first := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_begin_game", first.Type)

	// One seat unreadies, which aborts the start, then both ready into a fresh countdown.
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "unready"})
	require.Equal(t, PhaseOpen, h.Phase, "an unready must abort the pending start")
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseCountdown, h.Phase, "readying again must count down again")

	// The first countdown's fire arrives partway through the second countdown.
	h.dispatch(first)
	assert.Equal(t, PhaseCountdown, h.Phase, "a superseded countdown's fire must not start the game")
	assert.Equal(t, 0, *created, "no game may be created by a superseded fire")
	assert.Nil(t, h.Game, "the hub must still hold no game")

	// The second countdown's own fire is the one that starts it, at its own deadline.
	second := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_begin_game", second.Type, "the second countdown must schedule its own start")
	h.dispatch(second)
	assert.Equal(t, PhaseInGame, h.Phase, "the second countdown's own fire must start the game")
	assert.Equal(t, 1, *created, "exactly one game may be created")
	require.NotNil(t, h.Game)
}

// TestCountdownFireFromAnEarlierGameIsDropped covers the same device across a whole game: a fire
// left over from the countdown that started game one must not start game two, which is the shape
// the postgame test used to dodge with an hour-long countdown.
func TestCountdownFireFromAnEarlierGameIsDropped(t *testing.T) {
	h, idA, idB, created := newCountdownHub(t, 20*time.Millisecond)
	h.PostGameDuration = 20 * time.Millisecond

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	firstFire := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_begin_game", firstFire.Type)
	h.dispatch(firstFire)
	require.Equal(t, PhaseInGame, h.Phase)
	require.Equal(t, 1, *created)

	// Game one ends and the lobby reopens.
	h.dispatch(ClientMsg{Type: "_game_ended"})
	require.Equal(t, PhasePostGame, h.Phase)
	h.dispatch(waitForIncoming(t, h, 2*time.Second))
	require.Equal(t, PhaseOpen, h.Phase)

	// The table readies for game two, and game one's spent fire is redelivered mid-countdown.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseCountdown, h.Phase)
	h.dispatch(firstFire)
	assert.Equal(t, PhaseCountdown, h.Phase, "an earlier game's countdown fire must not start this one")
	assert.Equal(t, 1, *created, "no second game may be created by a spent fire")
}

// TestEveryExitFromCountdownDisownsItsPendingFire keeps the generation tied to the phase rather
// than to the one abort path that was reported. Any exit from PhaseCountdown must invalidate the
// fire that countdown armed, whichever route the hub left by.
func TestEveryExitFromCountdownDisownsItsPendingFire(t *testing.T) {
	exits := map[string]func(h *Hub){
		"unready":    func(h *Hub) { h.setPhase(PhaseOpen) },
		"abort":      func(h *Hub) { h.abortToOpen("test") },
		"search":     func(h *Hub) { h.applySearchState(SearchState{Searching: true, QueueID: "q"}) },
		"game start": func(h *Hub) { h.setPhase(PhaseInGame) },
	}

	for name, exit := range exits {
		t.Run(name, func(t *testing.T) {
			h, idA, idB, created := newCountdownHub(t, time.Hour) // this test never waits on a timer

			h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "ready"})
			h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "ready"})
			require.Equal(t, PhaseCountdown, h.Phase)
			armed := h.countdownGen

			exit(h)
			require.NotEqual(t, PhaseCountdown, h.Phase, "the exit must leave the countdown")

			// Back in a countdown, with the earlier countdown's fire arriving late.
			h.setPhase(PhaseCountdown)
			h.dispatch(ClientMsg{Type: "_begin_game", gen: armed})
			assert.Equal(t, PhaseCountdown, h.Phase, "the disowned fire must not start the game")
			assert.Equal(t, 0, *created, "no game may be created by a disowned fire")
		})
	}
}
