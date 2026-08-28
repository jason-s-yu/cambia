// internal/hub/grace_test.go
//
// Disconnect grace window and post-game reconnect (cambia-955). A reload dropped the socket, the
// forfeit rule took the seat inside a few hundred milliseconds, and in a two-player game that
// ended the match before the returning client could resume; the reloaded tab then landed on a
// "Game over" view with no winner and no scores, because game_results had already been broadcast
// to a socket that no longer existed. These tests pin both halves: the seat survives a drop for
// the length of the grace, and a hub that still holds a finished game answers a reconnect with
// the results again.
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

// handSignature reads a player's own hand as a card-id list, which is what a restored table has
// to match: the count alone would not catch a reshuffled or re-dealt hand.
func handSignature(t *testing.T, g *game.CambiaGame, playerID uuid.UUID) []string {
	t.Helper()
	state := g.GetCurrentObfuscatedGameState(playerID)
	for _, p := range state.Players {
		if p.PlayerID != playerID {
			continue
		}
		ids := make([]string, 0, len(p.RevealedHand))
		for _, c := range p.RevealedHand {
			ids = append(ids, c.ID.String())
		}
		return ids
	}
	t.Fatalf("player %s is not in the game state", playerID)
	return nil
}

// waitForfeited polls until the game reports subject forfeited, or the timeout expires.
func waitForfeited(t *testing.T, g *game.CambiaGame, subject uuid.UUID, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if g.IsForfeited(subject) {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

// TestDropInsideTheGraceWindowKeepsTheSeat is the reported bug: with the grace open, a two-player
// game must not end on a drop, the table must be told the player is reconnecting, and the player
// who comes back gets the same hand and the same stockpile they left.
func TestDropInsideTheGraceWindowKeepsTheSeat(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 2, true, 0, 0, 3*time.Second)

	before := handSignature(t, g, ids[1])
	stockBefore := g.GetCurrentObfuscatedGameState(ids[0]).StockpileSize
	survivor := h.getConn(ids[0])
	require.NotNil(t, survivor)

	h.Leave(ids[1])
	require.True(t, waitDisconnected(t, g, ids[0], ids[1], 2*time.Second), "the drop must reach the game")

	// The public event the remaining player needs to render the away seat.
	env := waitEnvelope(t, survivor, "player_reconnecting", 2*time.Second)
	require.NotNil(t, env, "the table was never told the seat is being held")
	var reconnecting struct {
		User    struct{ ID string } `json:"user"`
		Payload struct {
			GraceSeconds int   `json:"graceSeconds"`
			Deadline     int64 `json:"deadline"`
		} `json:"payload"`
	}
	require.NoError(t, json.Unmarshal(env.Payload, &reconnecting))
	assert.Equal(t, ids[1].String(), reconnecting.User.ID)
	assert.Equal(t, 3, reconnecting.Payload.GraceSeconds)
	assert.Greater(t, reconnecting.Payload.Deadline, time.Now().UnixMilli(), "the deadline must still be ahead")

	select {
	case res := <-ended:
		t.Fatalf("a drop inside the grace window ended the game (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(300 * time.Millisecond):
	}
	assert.False(t, g.IsForfeited(ids[1]), "nobody forfeits while their window is open")

	// The window is also visible to anyone who resyncs rather than watching for the event.
	away := g.GetCurrentObfuscatedGameState(ids[0])
	for _, p := range away.Players {
		if p.PlayerID == ids[1] {
			require.NotNil(t, p.ReconnectDeadline, "an open window must be published in sync_state")
			assert.False(t, p.Forfeited)
		}
	}

	rejoin := newFakeConn(ids[1], "P1")
	h.Join(rejoin)
	require.NotNil(t, waitEnvelope(t, rejoin, "private_sync_state", 2*time.Second),
		"the returning player must be sent the table again")
	require.NotNil(t, waitEnvelope(t, survivor, "player_reconnected", 2*time.Second),
		"the table was never told the player came back")

	assert.True(t, connectedIn(t, g, ids[0], ids[1]), "the reconnect must restore the seat")
	assert.Equal(t, before, handSignature(t, g, ids[1]), "the restored hand must be the hand they left")
	assert.Equal(t, stockBefore, g.GetCurrentObfuscatedGameState(ids[0]).StockpileSize, "the stockpile must be untouched")
	back := g.GetCurrentObfuscatedGameState(ids[0])
	for _, p := range back.Players {
		if p.PlayerID == ids[1] {
			assert.Nil(t, p.ReconnectDeadline, "the window must be closed once they are back")
		}
	}

	// And the forfeit that was pending on the window never lands afterwards.
	select {
	case res := <-ended:
		t.Fatalf("the forfeit still landed after the reconnect (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(600 * time.Millisecond):
	}
}

// TestGraceExpiryForfeitsAndEndsTheGame is the other end of the window: nobody comes back, so the
// forfeit lands and the existing only-one-left rule ends the two-player game.
func TestGraceExpiryForfeitsAndEndsTheGame(t *testing.T) {
	_, ids, g, ended := newInGameHubGraceStopped(t, 2, 150*time.Millisecond)

	g.HandleDisconnect(ids[1])

	select {
	case res := <-ended:
		assert.Equal(t, ids[0], res.winner, "the player still connected must win the forfeit")
		assert.Contains(t, res.scores, ids[0], "the remaining player must be scored")
		assert.NotContains(t, res.scores, ids[1], "a forfeited player must not be scored")
	case <-time.After(3 * time.Second):
		t.Fatal("the game never ended after the grace window expired")
	}
	assert.True(t, g.IsForfeited(ids[1]))
	assert.True(t, g.GetCurrentObfuscatedGameState(ids[0]).GameOver)
}

// TestGraceExpiryForfeitsWithoutEndingAMultiplayerGame keeps the two consequences apart: the
// forfeit is what the window produces, and ending the game is only what follows when nobody is
// left to play it.
func TestGraceExpiryForfeitsWithoutEndingAMultiplayerGame(t *testing.T) {
	_, ids, g, ended := newInGameHubGraceStopped(t, 3, 150*time.Millisecond)

	g.HandleDisconnect(ids[2])
	require.True(t, waitForfeited(t, g, ids[2], 2*time.Second), "the window must close on a forfeit")

	select {
	case res := <-ended:
		t.Fatalf("one forfeit of three ended the game (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(200 * time.Millisecond):
	}

	g.EndGame()
	select {
	case res := <-ended:
		assert.NotContains(t, res.scores, ids[2], "the forfeited player must be out of the scoring")
		assert.Len(t, res.scores, 2)
	case <-time.After(3 * time.Second):
		t.Fatal("the game never ended")
	}
}

// TestGameEndingInsideTheWindowStillScoresTheAbsentPlayer is the scoring half of the window: a
// player who is away but has not forfeited is still in the game, so a table that finishes without
// them scores their hand (MATCHMAKING.md 8, "score counts normally").
func TestGameEndingInsideTheWindowStillScoresTheAbsentPlayer(t *testing.T) {
	_, ids, g, ended := newInGameHubGraceStopped(t, 3, 10*time.Second)

	g.HandleDisconnect(ids[2])
	require.False(t, g.IsForfeited(ids[2]), "the window must still be open")

	g.EndGame()
	select {
	case res := <-ended:
		assert.Contains(t, res.scores, ids[2], "an absent player inside their window is still scored")
		assert.Len(t, res.scores, 3)
	case <-time.After(3 * time.Second):
		t.Fatal("the game never ended")
	}
}

// TestTurnTimerKeepsPlayingDuringTheGraceWindow pins the timer decision (RULES.md T5,
// MATCHMAKING.md 8): the table plays on while a seat is held, with the turn timeout auto-playing
// the absent player's turn. Pausing it instead would let anyone freeze a game for the length of
// the grace by pulling their network out.
func TestTurnTimerKeepsPlayingDuringTheGraceWindow(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 2, true, 1, 150*time.Millisecond, 5*time.Second)

	state := g.GetCurrentObfuscatedGameState(ids[0])
	victim := state.CurrentPlayerID // drop the player the game is actually waiting on
	observer := ids[0]
	if observer == victim {
		observer = ids[1]
	}
	before := state.TurnID

	h.Leave(victim)
	require.True(t, waitDisconnected(t, g, observer, victim, 2*time.Second), "the drop must reach the game")

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if g.GetCurrentObfuscatedGameState(observer).TurnID > before {
			select {
			case res := <-ended:
				t.Fatalf("the game ended inside the grace window (winner %s, scores %v)", res.winner, res.scores)
			default:
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("the turn never advanced: a held seat stalled the table for the length of the grace")
}

// TestReconnectToAFinishedGameResendsTheResults is the second half of the report: the reloaded
// client reached a "Game over" screen with no scores, because the results were broadcast while it
// had no socket and nothing re-sends them. A hub still holding its finished game now answers the
// reconnect with the same frame.
func TestReconnectToAFinishedGameResendsTheResults(t *testing.T) {
	h, ids, g, _ := newInGameHub(t, 2, true, 0, 0, 0)

	// End the game the way the production path does: the game ends, the owner broadcasts the
	// results, and the hub moves itself to the post-game phase.
	g.EndGame()
	h.Emit("game_results", map[string]interface{}{
		"type":   "game_results",
		"winner": ids[0].String(),
		"scores": map[string]int{ids[0].String(): 12, ids[1].String(): 30},
	})
	h.NotifyGameEnded()
	// The phase belongs to the Run goroutine, so wait on the frame it broadcasts rather than
	// reading the field.
	survivor := h.getConn(ids[0])
	require.NotNil(t, survivor)
	phaseEnv := waitEnvelope(t, survivor, "phase_change", 2*time.Second)
	require.NotNil(t, phaseEnv, "the hub never announced the post-game phase")
	var phase struct {
		Phase string `json:"phase"`
	}
	require.NoError(t, json.Unmarshal(phaseEnv.Payload, &phase))
	require.Equal(t, "post_game", phase.Phase)

	// The player reloads: the socket drops and a new one arrives on the same finished hub.
	h.Leave(ids[1])
	rejoin := newFakeConn(ids[1], "P1")
	h.Join(rejoin)

	env := waitEnvelope(t, rejoin, "game_results", 2*time.Second)
	require.NotNil(t, env, "a reconnect into a finished game got no results")
	var payload struct {
		Winner string         `json:"winner"`
		Scores map[string]int `json:"scores"`
	}
	require.NoError(t, json.Unmarshal(env.Payload, &payload))
	assert.Equal(t, ids[0].String(), payload.Winner, "the resent results must name the winner")
	assert.Equal(t, 30, payload.Scores[ids[1].String()], "the resent results must carry the scores")
}

// TestResultsAreNotResentIntoALiveGame keeps the replay to the case that needs it: a player who
// drops and returns mid-game must get the table, not a stale results frame from an earlier round.
func TestResultsAreNotResentIntoALiveGame(t *testing.T) {
	h, ids, _, _ := newInGameHub(t, 3, true, 0, 0, 3*time.Second)

	h.Emit("game_results", map[string]interface{}{"winner": ids[0].String(), "scores": map[string]int{}})
	// A new game clears the previous one's results (Emit's game_started case).
	h.Emit("game_started", map[string]interface{}{"game_id": uuid.New().String()})

	h.Leave(ids[2])
	rejoin := newFakeConn(ids[2], "P2")
	h.Join(rejoin)
	require.NotNil(t, waitEnvelope(t, rejoin, "private_sync_state", 2*time.Second))
	assert.Nil(t, waitEnvelope(t, rejoin, "game_results", 300*time.Millisecond),
		"a live game must not replay an earlier results frame")
}

// newInGameHubGraceStopped is newInGameHubStopped for the grace tests: two or three players, the
// forfeit rule on, no turn timer, and the given reconnect window. The Run loop is not started, so
// the game is driven directly and the hub is only there to receive its events.
func newInGameHubGraceStopped(t *testing.T, playerCount int, grace time.Duration) (*Hub, []uuid.UUID, *game.CambiaGame, chan endedGame) {
	t.Helper()
	return newInGameHubStopped(t, playerCount, true, 0, 0, grace)
}
