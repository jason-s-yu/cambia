// internal/hub/disconnect_test.go
//
// Mid-game disconnect handling (cambia-837). game.HandleDisconnect had no callers at all: a
// WebSocket that dropped during a game never marked the player disconnected, so the
// ForfeitOnDisconnect house rule was dead, the circuit grace timer never armed, scoring never
// omitted anyone, and a turn timer running out was the only thing that ever noticed. The hub now
// tells its running game when a player's socket goes and when it comes back.
package hub

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// endedGame captures what OnGameEnd reported, which is where a forfeit is observable: the
// forfeited player is omitted from scoring entirely (see computeScoresFromEngine).
type endedGame struct {
	winner uuid.UUID
	scores map[uuid.UUID]int
}

// newInGameHubStopped builds a hub in PhaseInGame with a real started CambiaGame behind it, but
// does not start Run(). Tests that drive the hub's own helpers directly use this, so hub fields
// the Run goroutine would otherwise own can be set from the test. turnDuration overrides the
// duration BeginPreGame derives from TurnTimerSec, so a timer-backed test does not have to wait
// whole seconds; grace does the same for the reconnect window derived from DisconnectGraceSec,
// and grace 0 is the immediate forfeit the rule performed before cambia-955.
func newInGameHubStopped(t *testing.T, playerCount int, forfeit bool, turnTimerSec int, turnDuration, grace time.Duration) (*Hub, []uuid.UUID, *game.CambiaGame, chan endedGame) {
	t.Helper()

	h, ids, g, ended := newPreGameHubStopped(t, playerCount, forfeit, turnTimerSec, turnDuration, grace, 0)
	g.StartGame()
	return h, ids, g, ended
}

// newPreGameHubStopped builds the same table but stops at the initial card reveal, before
// StartGame flips Started true. preGame overrides the reveal's length when non-zero, which is what
// lets a test decide whether a reconnect window opened during the reveal expires before or after
// the game proper begins (cambia-955 F1).
func newPreGameHubStopped(t *testing.T, playerCount int, forfeit bool, turnTimerSec int, turnDuration, grace, preGame time.Duration) (*Hub, []uuid.UUID, *game.CambiaGame, chan endedGame) {
	t.Helper()

	ids := make([]uuid.UUID, playerCount)
	for i := range ids {
		ids[i] = uuid.New()
	}

	lob := lobby.NewLobbyWithDefaults(ids[0])
	for _, id := range ids {
		lob.JoinUser(id)
	}

	h := NewHub(lob)
	h.IdleTTL = 0 // the idle reaper is not what these tests are about

	g := game.NewCambiaGame()
	g.LobbyID = lob.ID
	g.Emitter = h
	rules := game.DefaultHouseRules()
	rules.ForfeitOnDisconnect = forfeit
	rules.TurnTimerSec = turnTimerSec
	// The rule is carried in whole seconds and the event payload quotes it, so a sub-second test
	// window still advertises at least one second; g.DisconnectGrace below is what actually times.
	rules.DisconnectGraceSec = 0
	if grace > 0 {
		rules.DisconnectGraceSec = int(grace / time.Second)
		if rules.DisconnectGraceSec == 0 {
			rules.DisconnectGraceSec = 1
		}
	}
	g.HouseRules = rules

	ended := make(chan endedGame, 4)
	g.OnGameEnd = func(_ uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int, _ map[uuid.UUID]string) {
		ended <- endedGame{winner: winner, scores: scores}
	}

	for i, id := range ids {
		g.AddPlayer(&models.Player{
			ID:        id,
			Connected: true,
			Hand:      []*models.Card{},
			User:      &models.User{ID: id, Username: "P" + strconv.Itoa(i)},
		})
	}
	if preGame > 0 {
		g.PreGameDuration = preGame
	}
	g.BeginPreGame()
	if turnDuration > 0 {
		g.TurnDuration = turnDuration
	}
	g.DisconnectGrace = grace

	h.Game = g
	h.Phase = PhaseInGame
	lob.Mu.Lock()
	lob.InGame = true
	lob.GameID = g.ID
	lob.Mu.Unlock()

	t.Cleanup(func() {
		h.Shutdown()
		g.EndGame() // stops any timer still armed
	})

	return h, ids, g, ended
}

// newInGameHub is newInGameHubStopped with the Run loop started and every player connected: the
// state a mid-game drop has to be tested against.
func newInGameHub(t *testing.T, playerCount int, forfeit bool, turnTimerSec int, turnDuration, grace time.Duration) (*Hub, []uuid.UUID, *game.CambiaGame, chan endedGame) {
	t.Helper()

	h, ids, g, ended := newInGameHubStopped(t, playerCount, forfeit, turnTimerSec, turnDuration, grace)
	runAndConnect(t, h, ids)
	return h, ids, g, ended
}

// newPreGameHub is newInGameHub stopped at the initial card reveal: the Run loop is up and every
// player is connected, but StartGame has not run yet.
func newPreGameHub(t *testing.T, playerCount int, grace, preGame time.Duration) (*Hub, []uuid.UUID, *game.CambiaGame, chan endedGame) {
	t.Helper()

	h, ids, g, ended := newPreGameHubStopped(t, playerCount, true, 0, 0, grace, preGame)
	runAndConnect(t, h, ids)
	return h, ids, g, ended
}

// runAndConnect starts the hub's Run loop and joins every player, which is the state any drop
// test has to begin from.
func runAndConnect(t *testing.T, h *Hub, ids []uuid.UUID) {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	waitAlive(t, h, true)

	for i, id := range ids {
		conn := newFakeConn(id, "P"+strconv.Itoa(i))
		h.Join(conn)
		require.NotNil(t, waitEnvelope(t, conn, "lobby_state", 2*time.Second), "player %d must be served", i)
	}
}

// connectedIn reports the game's view of whether a player holds a live connection.
func connectedIn(t *testing.T, g *game.CambiaGame, observer, subject uuid.UUID) bool {
	t.Helper()
	for _, p := range g.GetCurrentObfuscatedGameState(observer).Players {
		if p.PlayerID == subject {
			return p.Connected
		}
	}
	t.Fatalf("player %s is not in the game state", subject)
	return false
}

// waitDisconnected polls until the game marks subject disconnected, or the timeout expires.
func waitDisconnected(t *testing.T, g *game.CambiaGame, observer, subject uuid.UUID, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !connectedIn(t, g, observer, subject) {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

// TestMidGameDropForfeitsWhenTheRuleIsOn is the dead rule brought back: with
// ForfeitOnDisconnect on, a two-player game whose second player drops ends there, and the
// forfeiting player is left out of the scoring.
func TestMidGameDropForfeitsWhenTheRuleIsOn(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 2, true, 0, 0, 0)

	h.Leave(ids[1]) // the socket drops; nothing releases lobby membership

	select {
	case res := <-ended:
		assert.Equal(t, ids[0], res.winner, "the player still connected must win the forfeit")
		assert.Contains(t, res.scores, ids[0], "the remaining player must be scored")
		assert.NotContains(t, res.scores, ids[1], "a forfeited player must not be scored")
	case <-time.After(3 * time.Second):
		t.Fatal("a drop under ForfeitOnDisconnect never ended the game")
	}

	assert.False(t, connectedIn(t, g, ids[0], ids[1]), "the dropped player must be marked disconnected")
	assert.True(t, g.GetCurrentObfuscatedGameState(ids[0]).GameOver, "the game must be over")
}

// TestMidGameDropKeepsLobbyMembership holds the cambia-807 line: a lost socket is transient, so
// the player keeps their seat and their resume entry even as the game marks them gone.
func TestMidGameDropKeepsLobbyMembership(t *testing.T) {
	h, ids, g, _ := newInGameHub(t, 3, true, 0, 0, 0)

	h.Leave(ids[2])
	require.True(t, waitDisconnected(t, g, ids[0], ids[2], 2*time.Second),
		"the drop must reach the game")

	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	joined, present := h.Lobby.Users[ids[2]]
	assert.True(t, present && joined, "a mid-game drop must not release lobby membership")
}

// TestMidGameDropWithoutForfeitKeepsTheGameRunning is the rule-off half: the drop is recorded,
// nothing forfeits, and the turn timer is still the backstop it always was.
//
// The player dropped is the one not on turn, so the game is left waiting on a live player: the
// engine's scheduler declines to arm a timer for a disconnected acting player, so dropping the
// player on turn would prove nothing about the timer still running.
func TestMidGameDropWithoutForfeitKeepsTheGameRunning(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 2, false, 1, 300*time.Millisecond, 0)

	state := g.GetCurrentObfuscatedGameState(ids[0])
	victim, observer := ids[1], ids[0]
	if state.CurrentPlayerID == victim {
		victim, observer = ids[0], ids[1]
	}
	before := state.TurnID

	h.Leave(victim)
	require.True(t, waitDisconnected(t, g, observer, victim, 2*time.Second),
		"the drop must still reach the game with the forfeit rule off")

	select {
	case res := <-ended:
		t.Fatalf("the game ended on a drop with ForfeitOnDisconnect off (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(200 * time.Millisecond):
	}
	assert.False(t, g.GetCurrentObfuscatedGameState(observer).GameOver, "the game must still be running")

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if g.GetCurrentObfuscatedGameState(observer).TurnID > before {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("the turn timer stopped advancing the game after the drop")
}

// TestReconnectBeforeTheForfeitLandsCancelsIt covers the window the forfeit rule leaves open: in a
// three-player game one drop is not yet fatal, and a player who comes back before the game ends is
// scored like everyone else.
func TestReconnectBeforeTheForfeitLandsCancelsIt(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 3, true, 0, 0, 0)

	h.Leave(ids[2])
	require.True(t, waitDisconnected(t, g, ids[0], ids[2], 2*time.Second), "the drop must reach the game")

	select {
	case res := <-ended:
		t.Fatalf("one drop of three must not end the game (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(200 * time.Millisecond):
	}

	rejoin := newFakeConn(ids[2], "P2")
	h.Join(rejoin)
	require.NotNil(t, waitEnvelope(t, rejoin, "private_sync_state", 2*time.Second),
		"a reconnecting player must be sent the game state they need to draw the table again")

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if connectedIn(t, g, ids[0], ids[2]) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	require.True(t, connectedIn(t, g, ids[0], ids[2]), "the reconnect must restore the seat")

	g.EndGame()
	select {
	case res := <-ended:
		assert.Contains(t, res.scores, ids[2], "a player who reconnected before the end must be scored")
		assert.Len(t, res.scores, 3, "all three players must be scored")
	case <-time.After(3 * time.Second):
		t.Fatal("the game never ended")
	}
}

// TestPlayerWhoNeverReconnectsIsForfeited is the control for the test above: without the
// reconnect, the same game leaves the dropped player out of the scoring.
func TestPlayerWhoNeverReconnectsIsForfeited(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 3, true, 0, 0, 0)

	h.Leave(ids[2])
	require.True(t, waitDisconnected(t, g, ids[0], ids[2], 2*time.Second), "the drop must reach the game")

	g.EndGame()
	select {
	case res := <-ended:
		assert.NotContains(t, res.scores, ids[2], "a player who never came back must be forfeited")
		assert.Len(t, res.scores, 2, "only the connected players are scored")
	case <-time.After(3 * time.Second):
		t.Fatal("the game never ended")
	}
}

// TestNotifyGameEndedNeverBlocksItsCaller guards the deadlock this wiring opens. A forfeit ends
// the game inside the leave case, so OnGameEnd - and the NotifyGameEnded it calls - now runs on
// the Run goroutine, which is the only goroutine that drains h.incoming. A blocking send onto a
// full queue would park the hub for good.
func TestNotifyGameEndedNeverBlocksItsCaller(t *testing.T) {
	lob := lobby.NewLobbyWithDefaults(uuid.New())
	h := NewHub(lob)
	defer h.Shutdown()

	// Fill the queue with nothing draining it, as a hub busy inside HandleDisconnect would be.
	for {
		select {
		case h.incoming <- ClientMsg{Type: "chat"}:
			continue
		default:
		}
		break
	}

	done := make(chan struct{})
	go func() {
		h.NotifyGameEnded()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("NotifyGameEnded blocked on a full queue; a forfeit would wedge the hub")
	}

	// The transition is deferred, not dropped: draining one slot lets it land.
	<-h.incoming
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		select {
		case msg := <-h.incoming:
			if msg.Type == "_game_ended" {
				return
			}
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
	t.Fatal("the deferred post-game transition never arrived")
}

// TestNonParticipantDropLeavesTheGameAlone keeps the wiring to the people in the game: a lobby
// member who arrived after the deal holds no seat, and their socket dropping must not disturb the
// table or write a player_disconnect into a game they were never in.
func TestNonParticipantDropLeavesTheGameAlone(t *testing.T) {
	h, ids, g, ended := newInGameHub(t, 2, true, 0, 0, 0)

	stranger := uuid.New()
	h.Lobby.JoinUser(stranger)
	strangerConn := newFakeConn(stranger, "stranger")
	h.Join(strangerConn)
	require.NotNil(t, waitEnvelope(t, strangerConn, "lobby_state", 2*time.Second))

	h.Leave(stranger)

	select {
	case res := <-ended:
		t.Fatalf("a non-participant's drop ended the game (winner %s, scores %v)", res.winner, res.scores)
	case <-time.After(300 * time.Millisecond):
	}
	assert.False(t, g.GetCurrentObfuscatedGameState(ids[0]).GameOver, "the game must be untouched")
	assert.True(t, connectedIn(t, g, ids[0], ids[1]), "no seated player may be marked disconnected")
}

// TestDropOutsideTheInGamePhaseIsIgnored pins the phase gate. A hub holds its finished game until
// the results interval returns it to the lobby (cambia-793), so without the gate a player closing
// the results screen would be marked disconnected on a game that is already over and would drag a
// stale sync state out to everyone else. Driven directly because h.Phase belongs to the Run
// goroutine.
func TestDropOutsideTheInGamePhaseIsIgnored(t *testing.T) {
	h, ids, g, _ := newInGameHubStopped(t, 2, true, 0, 0, 0)
	h.Phase = PhasePostGame

	h.notePlayerDisconnected(ids[1])

	assert.True(t, connectedIn(t, g, ids[0], ids[1]),
		"a drop outside the in-game phase must not reach the finished game")
}
