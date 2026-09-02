// internal/hub/terminal_resend_test.go
//
// Delivery of the results frame to a socket that was not there for the broadcast (cambia-1241).
// The hub holds the last game_results/match_end it sent (rememberTerminal) so a reconnect can be
// answered with it, and these pin the two edges of that hold: the moment before the frame exists,
// and the moment after the results screen has closed. Kept apart from grace_test.go, which owns
// the reconnect-window half of the same reload story (cambia-955).
package hub

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
)

// resultsPayload is the shape these tests read back off the wire.
type resultsPayload struct {
	Winner string         `json:"winner"`
	Scores map[string]int `json:"scores"`
}

// TestJoinBetweenGameOverAndTheResultsEmitStillGetsTheResults covers the narrowest window a
// reload can land in: the game is over, so the seat is past every "is this game live" gate, but
// the results frame does not exist yet, so there is nothing stored to replay either. The join is
// staged from inside OnGameEnd, which is where the service runs the game end from and which holds
// the game's own lock throughout, so the hub's Run goroutine is parked on that lock for as long
// as the window is open.
func TestJoinBetweenGameOverAndTheResultsEmitStillGetsTheResults(t *testing.T) {
	// A grace long enough that the drop only holds the seat: the game has to end from EndGame
	// below, with the socket already gone, rather than from the forfeit the drop would otherwise
	// trigger.
	h, ids, g, _ := newInGameHub(t, 2, true, 0, 0, 3*time.Second)

	// The player's tab reloads: the old socket is gone before the game ends.
	h.Leave(ids[1])
	rejoin := newFakeConn(ids[1], "P1")

	// OnGameEnd runs on this goroutine, inside endGame's lock, exactly as the service's own
	// callback does (handlers.attachOnGameEnd). The join lands there, and the results follow it.
	// Which of the two carries them is the hub's business and deliberately not asserted: the
	// socket either registers in time for the broadcast or is answered by the stored frame when
	// the hub reaches its reconnect. Nothing here waits on the hub, since the Run goroutine is
	// parked on the game lock this callback holds.
	g.OnGameEnd = func(_ uuid.UUID, _ uuid.UUID, _ map[uuid.UUID]int, _ map[uuid.UUID]string, _ map[uuid.UUID]int, _ uuid.UUID, _ []game.FinalHand) {
		h.Join(rejoin)
		h.Emit("game_results", map[string]interface{}{
			"type":   "game_results",
			"winner": ids[0].String(),
			"scores": map[string]int{ids[0].String(): 9, ids[1].String(): 24},
		})
	}
	g.EndGame()
	h.NotifyGameEnded()

	env := waitEnvelope(t, rejoin, "game_results", 2*time.Second)
	require.NotNil(t, env, "a join in the window between game over and the results emit got no results")
	var payload resultsPayload
	require.NoError(t, json.Unmarshal(env.Payload, &payload))
	assert.Equal(t, ids[0].String(), payload.Winner)
	assert.Equal(t, 24, payload.Scores[ids[1].String()], "the results must carry the scores")
}

// TestResultsOutliveTheResultsScreen is the other edge: the hold used to end when the results
// screen did. returnToLobby drops h.Game, and the resend was gated on there being one, so a
// reload after PostGameDuration landed on a lobby with no record of the game just played even
// though the hub still had the frame. The stored results now last until a new game replaces them.
func TestResultsOutliveTheResultsScreen(t *testing.T) {
	h, ids, g, _ := newInGameHubStopped(t, 2, true, 0, 0, 0)
	// The results screen is closed by its own timer; a short one keeps the test to its subject.
	h.PostGameDuration = 50 * time.Millisecond
	runAndConnect(t, h, ids)

	survivor := h.getConn(ids[0])
	require.NotNil(t, survivor)

	g.EndGame()
	h.Emit("game_results", map[string]interface{}{
		"type":   "game_results",
		"winner": ids[0].String(),
		"scores": map[string]int{ids[0].String(): 7, ids[1].String(): 33},
	})
	h.NotifyGameEnded()

	// The phase belongs to the Run goroutine, so the transitions are read off the frames it
	// broadcasts: post_game when the game ends, open again once the screen's timer has run.
	requirePhase(t, survivor, "post_game")
	requirePhase(t, survivor, "open")

	h.Leave(ids[1])
	rejoin := newFakeConn(ids[1], "P1")
	h.Join(rejoin)

	env := waitEnvelope(t, rejoin, "game_results", 2*time.Second)
	require.NotNil(t, env, "a reload after the results screen closed got no results")
	var payload resultsPayload
	require.NoError(t, json.Unmarshal(env.Payload, &payload))
	assert.Equal(t, 33, payload.Scores[ids[1].String()], "the resent results must still carry the scores")

	// And the one transition that does invalidate them: the next game starting.
	h.Emit("game_started", map[string]interface{}{"game_id": uuid.New().String()})
	h.Leave(ids[1])
	second := newFakeConn(ids[1], "P1")
	h.Join(second)
	require.NotNil(t, waitEnvelope(t, second, "lobby_state", 2*time.Second), "the socket was never served")
	assert.Nil(t, waitEnvelope(t, second, "game_results", 300*time.Millisecond),
		"a new game must clear the previous one's results")
}

// requirePhase waits for the next phase_change frame and asserts which phase it announced.
func requirePhase(t *testing.T, conn *Connection, want string) {
	t.Helper()
	env := waitEnvelope(t, conn, "phase_change", 2*time.Second)
	require.NotNil(t, env, "the hub never announced the %s phase", want)
	var phase struct {
		Phase string `json:"phase"`
	}
	require.NoError(t, json.Unmarshal(env.Payload, &phase))
	require.Equal(t, want, phase.Phase)
}
