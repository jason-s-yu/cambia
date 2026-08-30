// internal/hub/postgame_exit_test.go
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

// countByType counts the envelopes of the given type.
func countByType(envs []Envelope, typ string) int {
	n := 0
	for _, e := range envs {
		if e.Type == typ {
			n++
		}
	}
	return n
}

// TestHostExitsResultsBeforeTimer is the cambia-1238 fix: the results screen used to be leavable
// only by waiting out PostGameDuration, because dispatch admitted nothing but chat in
// PhasePostGame. The host's control now runs the same reset the timer runs.
func TestHostExitsResultsBeforeTimer(t *testing.T) {
	h, idA, idB, connA, connB := newPostGameHub(t)
	drainEnvelopes(t, connA)
	drainEnvelopes(t, connB)

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhaseOpen, h.Phase, "the host's exit must reopen the lobby")
	assert.Nil(t, h.Game, "the finished game must be cleared so the next one can be created")
	assert.False(t, h.Lobby.InGame, "lobby must no longer be flagged in-game")
	assert.Equal(t, uuid.Nil, h.Lobby.GameID)
	assert.False(t, h.Lobby.GameInstanceCreated)
	assert.False(t, h.Lobby.ReadyStates[idA], "player A must be unready")
	assert.False(t, h.Lobby.ReadyStates[idB], "player B must be unready")

	aFrames := drainEnvelopes(t, connA)
	bFrames := drainEnvelopes(t, connB)

	// One transition, announced to everyone: the seat that clicked is not the only one leaving
	// the results screen.
	require.Equal(t, 1, countByType(aFrames, "phase_change"), "exactly one phase_change for the host")
	require.Equal(t, 1, countByType(bFrames, "phase_change"), "exactly one phase_change for the other seat")
	aChange := findByType(aFrames, "phase_change")
	bChange := findByType(bFrames, "phase_change")
	assert.Equal(t, "open", payloadOf(t, *aChange)["phase"])
	assert.Equal(t, "open", payloadOf(t, *bChange)["phase"])
	assert.Equal(t, aChange.Seq, bChange.Seq, "both clients must observe the same phase_change seq")

	assert.True(t, containsType(aFrames, "lobby_state"), "the refreshed lobby snapshot must follow")
	assert.False(t, containsType(aFrames, "error"), "the host must not be refused")
}

// TestPostGameExitRefusedForNonHost pins the admission rule: the exit closes the results screen
// for the whole table, so a seat that does not hold the host role cannot take it away from
// everyone else.
func TestPostGameExitRefusedForNonHost(t *testing.T) {
	h, idA, idB, connA, connB := newPostGameHub(t)
	require.Equal(t, idA, h.Lobby.HostUserID, "precondition: A hosts")
	drainEnvelopes(t, connA)
	drainEnvelopes(t, connB)

	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhasePostGame, h.Phase, "a non-host must not move the phase")
	assert.NotNil(t, h.Game, "the finished game must still be routed")

	bFrames := drainEnvelopes(t, connB)
	assert.True(t, containsType(bFrames, "error"), "the sender must be told why nothing happened")
	assert.False(t, containsType(bFrames, "phase_change"))
	assert.False(t, containsType(drainEnvelopes(t, connA), "phase_change"), "no seat sees a transition")
}

// TestPostGameExitAllowedForAnySeatOfSystemHostedLobby covers the fallback the host gate needs to
// have: a matchmade lobby has no player host at all (cambia-1087), so gating on the role would
// leave every seat of a finished match stuck on the results screen. Any seat that played may
// close it there.
func TestPostGameExitAllowedForAnySeatOfSystemHostedLobby(t *testing.T) {
	h, _, idB, _, connB := newPostGameHub(t)
	h.Lobby.Mu.Lock()
	h.Lobby.AdoptSystemHostUnsafe()
	h.Lobby.Mu.Unlock()
	require.True(t, h.Lobby.SystemHosted(), "precondition: the queue hosts this lobby")
	drainEnvelopes(t, connB)

	h.dispatch(ClientMsg{UserID: idB, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhaseOpen, h.Phase, "a seat of a matchmade lobby must be able to leave the results")
	assert.Nil(t, h.Game)
	bFrames := drainEnvelopes(t, connB)
	assert.True(t, containsType(bFrames, "phase_change"))
	assert.False(t, containsType(bFrames, "error"))
}

// TestPostGameExitRefusedForNonMemberOfSystemHostedLobby keeps the fallback from being no gate at
// all: it widens the role to the lobby's own seats, not to whoever holds a socket.
func TestPostGameExitRefusedForNonMemberOfSystemHostedLobby(t *testing.T) {
	h, _, _, _, _ := newPostGameHub(t)
	h.Lobby.Mu.Lock()
	h.Lobby.AdoptSystemHostUnsafe()
	h.Lobby.Mu.Unlock()

	stranger := uuid.New()
	connS := newFakeConn(stranger, "S")
	h.conns[stranger] = connS

	h.dispatch(ClientMsg{UserID: stranger, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhasePostGame, h.Phase, "a non-member must not close the table's results")
	assert.True(t, containsType(drainEnvelopes(t, connS), "error"))
}

// TestMatchEndExitReturnsToLobby covers the other results phase. PhaseMatchEnd arms no timer at
// all (HandleRoundEnd only sets the phase), so the client control is the only way out of it.
func TestMatchEndExitReturnsToLobby(t *testing.T) {
	h, idA, _, connA, _ := newPostGameHub(t)
	h.Phase = PhaseMatchEnd
	drainEnvelopes(t, connA)

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhaseOpen, h.Phase, "a finished match must not strand its players on the results")
	assert.Nil(t, h.Game)
	aFrames := drainEnvelopes(t, connA)
	require.Equal(t, 1, countByType(aFrames, "phase_change"))
	assert.Equal(t, "open", payloadOf(t, *findByType(aFrames, "phase_change"))["phase"])
}

// TestPendingResetInertAfterEarlyExit is the idempotency requirement. Stopping a timer does not
// un-fire one that already ran, and the early exit leaves the armed reset queued behind it: the
// phase guard alone catches it only while the hub is still open, so a hub that has since ended a
// SECOND game would have had that game's results cut short by the first game's timer.
func TestPendingResetInertAfterEarlyExit(t *testing.T) {
	h, idA, _, connA, connB := newPostGameHub(t)
	h.PostGameDuration = 20 * time.Millisecond
	h.Phase = PhaseInGame
	defer h.Shutdown()

	// Game one ends, arming its reset.
	h.dispatch(ClientMsg{Type: "_game_ended"})
	require.Equal(t, PhasePostGame, h.Phase)
	stale := waitForIncoming(t, h, 2*time.Second)
	require.Equal(t, "_return_to_lobby", stale.Type)

	// The host leaves the results early.
	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})
	require.Equal(t, PhaseOpen, h.Phase)
	drainEnvelopes(t, connA)
	drainEnvelopes(t, connB)

	// Game two runs and ends: the hub is showing its results when the first game's reset lands.
	h.Game = game.NewCambiaGame()
	h.Phase = PhaseInGame
	h.dispatch(ClientMsg{Type: "_game_ended"})
	require.Equal(t, PhasePostGame, h.Phase)
	drainEnvelopes(t, connA)

	h.dispatch(stale)

	assert.Equal(t, PhasePostGame, h.Phase, "a superseded reset must not close a later results screen")
	assert.NotNil(t, h.Game, "a superseded reset must not drop the later game")
	assert.False(t, containsType(drainEnvelopes(t, connA), "phase_change"), "no second transition")

	// The reset game two armed for itself is still the one that works.
	current := waitForIncoming(t, h, 2*time.Second)
	h.dispatch(current)
	assert.Equal(t, PhaseOpen, h.Phase, "the current results screen's own reset must still fire")
}

// TestReturnToLobbyRepeatedIsNoOp is the plainest reading of the same rule: the reset runs once.
func TestReturnToLobbyRepeatedIsNoOp(t *testing.T) {
	h, idA, _, connA, _ := newPostGameHub(t)

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})
	require.Equal(t, PhaseOpen, h.Phase)
	drainEnvelopes(t, connA)

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})
	h.dispatch(ClientMsg{Type: "_return_to_lobby"})

	assert.Equal(t, PhaseOpen, h.Phase)
	assert.False(t, containsType(drainEnvelopes(t, connA), "phase_change"), "only the first exit transitions")
}

// TestInternalMessageFromSocketRefused guards the gate itself. ReadPump builds a ClientMsg out of
// whatever type string the frame carried, so a client can name an internal one; each internal
// type runs a phase transition with no check on the sender, and _return_to_lobby is precisely the
// transition the host gate exists to control.
func TestInternalMessageFromSocketRefused(t *testing.T) {
	h, _, idB, _, connB := newPostGameHub(t)

	h.dispatch(ClientMsg{ConnID: uuid.New(), UserID: idB, Type: "_return_to_lobby"})

	assert.Equal(t, PhasePostGame, h.Phase, "a spoofed internal message must not walk past the host gate")
	assert.NotNil(t, h.Game)
	assert.False(t, containsType(drainEnvelopes(t, connB), "phase_change"))
}

// TestPostGameExitIgnoredOutsideResults keeps the new message to the two phases it belongs to: a
// live game is not left by asking the lobby to reopen.
func TestPostGameExitIgnoredOutsideResults(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()

	lob := lobby.NewLobbyWithDefaults(idA)
	lob.JoinUser(idA)
	lob.JoinUser(idB)

	h := NewHub(lob)
	connA := newFakeConn(idA, "A")
	h.conns[idA] = connA
	h.conns[idB] = newFakeConn(idB, "B")
	h.Phase = PhaseInGame
	live := game.NewCambiaGame()
	h.Game = live

	h.dispatch(ClientMsg{UserID: idA, LastSeq: h.seq, Type: "return_to_lobby"})

	assert.Equal(t, PhaseInGame, h.Phase)
	assert.Same(t, live, h.Game, "a live game must not be dropped by the results control")
}
