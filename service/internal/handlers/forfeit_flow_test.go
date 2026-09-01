// internal/handlers/forfeit_flow_test.go
//
// Mid-game disconnect forfeit, end to end (cambia-855). cambia-837 wired game.HandleDisconnect
// into the hub's Leave case, but BeginPreGame's 10 second pre-game timer had no injection point,
// so no test could drive a real WS game past the pre-game reveal and into Started without
// literally waiting out the timer. With PreGameDuration configurable on GameServer (same pattern
// as CountdownDuration/PostGameDuration), these tests drive the full lobby -> ready -> countdown
// -> pre-game -> started -> disconnect path over real WebSocket connections in milliseconds.
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// gameResultsPayload mirrors the game_results envelope emitted by attachOnGameEnd.
type gameResultsPayload struct {
	Winner string         `json:"winner"`
	Scores map[string]int `json:"scores"`
}

// newForfeitTestServer builds a GameServer wired for a fast end-to-end run: countdown and
// pre-game are both shortened to millisecond scale so the game reaches Started without the test
// waiting out real production timers.
func newForfeitTestServer(t *testing.T) (*GameServer, *httptest.Server) {
	t.Helper()
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 30 * time.Millisecond
	gs.PreGameDuration = 30 * time.Millisecond
	// Every game this server creates registers its background DB-write goroutines here (the
	// initial-state upsert and both game-end writes), so awaitGameEndPersistence and
	// cleanupLobbyDBRows can drain them deterministically instead of leaving them to outlive the
	// test and race a later test's database.ConnectDB() reassigning the shared pool (cambia-908)
	// or a cleanup DELETE against rows a write is still landing (cambia-942 F4).
	gs.PersistWG = &sync.WaitGroup{}

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)

	return gs, ts
}

// setDisconnectGrace overrides a lobby's reconnect grace before its game is built. The default
// is 90 seconds (cambia-955, raised from 60 by cambia-1609), so a test that wants the forfeit to
// land on the drop itself - the behaviour these end-to-end tests were written against - sets it
// to 0 rather than waiting the window out. The grace window itself is covered by the hub tests
// and by TestE2EReconnectInsideTheGraceWindowKeepsTheSeat below.
func setDisconnectGrace(t *testing.T, gs *GameServer, lobbyID uuid.UUID, seconds int) {
	t.Helper()
	lob, ok := gs.LobbyStore.GetLobby(lobbyID)
	if !ok {
		t.Fatalf("lobby %s not found", lobbyID)
	}
	lob.Mu.Lock()
	lob.HouseRules.DisconnectGraceSec = seconds
	lob.Mu.Unlock()
}

// awaitGameEndPersistence blocks, bounded by a 5s timeout, until every background DB-write
// goroutine launched by a game created by gs completes: persistFinalGameState's two (final
// state, then game_results + ratings) and persistInitialGameState's one.
//
// sync.WaitGroup requires that an Add taking the counter up from zero happen before a
// concurrent Wait, so this helper is safe only when both of these hold (cambia-942 F3 restates
// the argument cambia-908 L1 invalidated by putting the initial-state write on this same
// WaitGroup, where the original wording accounted for persistFinalGameState's Add alone):
//
//  1. The caller already knows the relevant game has ended - by having received its
//     game_results broadcast, by having polled GameOver, or by having called EndGame itself.
//     persistFinalGameState's Add and all of those signals happen inside the same endGame call
//     under the game's lock, so observing any one of them proves that Add already ran.
//  2. No other game on the same GameServer can still be starting. persistInitialGameState's Add
//     runs under the same lock inside BeginPreGame; for the game the caller observed that is
//     strictly earlier than its end, but for a second game on the same server it is not ordered
//     against this Wait at all. Every caller drives exactly one game per GameServer, which is
//     what makes this hold today; a multi-game test would need to observe every game's end
//     first, or its own WaitGroup per game.
//
// The two writes stay on one WaitGroup rather than splitting the initial-state write onto its
// own: cleanupLobbyDBRows has to drain both before deleting the lobbies row (cambia-942 F4), so
// a split would only make every waiter wait on two groups, and it would not remove the ordering
// constraint above - it would just move it onto whichever group the second game touched first.
func awaitGameEndPersistence(t *testing.T, gs *GameServer) {
	t.Helper()
	if gs.PersistWG == nil {
		return
	}
	done := make(chan struct{})
	go func() {
		gs.PersistWG.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		// Fatal, not Errorf: a caller past this point assumes persistence already landed (e.g.
		// re-reading the row a background goroutine is still writing). Errorf would let that
		// caller run its assertions against a half-written row instead of stopping here.
		t.Fatalf("cambia-908: game-end persistence goroutines did not finish within 5s")
	}
}

// TestE2EMidGameDropForfeitsAndOmitsScores drives a two-player game past the (shortened)
// pre-game reveal into Started, drops one player's socket, and asserts the cambia-837 semantics
// land end to end: with ForfeitOnDisconnect on (the default house rule) the game ends once one
// player remains connected, and the dropped player is omitted entirely from the final scores.
func TestE2EMidGameDropForfeitsAndOmitsScores(t *testing.T) {
	gs, ts := newForfeitTestServer(t)

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()
	setDisconnectGrace(t, gs, lobUUID, 0) // the forfeit under test is the immediate one

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	host.settle()
	p2.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("host never received game_started")
	}
	if host.waitForType("private_initial_cards", 5*time.Second) == nil {
		t.Fatalf("host never received private_initial_cards (pre-game reveal never ran)")
	}
	// game_player_turn only fires from StartGame, once the pre-game window elapses: proof
	// Started flipped true, so the drop below lands mid-game rather than mid-reveal (where
	// HandleDisconnect records the drop but never forfeits, since g.Started is still false).
	if host.waitForType("game_player_turn", 5*time.Second) == nil {
		t.Fatalf("host never received game_player_turn (game never left pre-game)")
	}

	// The pre-game reveal has to actually have run its (shortened) course rather than getting
	// skipped: the game must have been PreGameActive at some point before Started.
	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobbyID)
	}
	if !g.GetCurrentObfuscatedGameState(hostID).Started {
		t.Fatalf("game must be Started before the drop under test")
	}

	// Player 2's socket drops mid-game, the way a closed tab does.
	p2.close()

	results := host.waitForType("game_results", 5*time.Second)
	if results == nil {
		t.Fatalf("host never received game_results after the mid-game drop")
	}
	awaitGameEndPersistence(t, gs)
	var payload gameResultsPayload
	if err := json.Unmarshal(results.Payload, &payload); err != nil {
		t.Fatalf("decode game_results payload: %v", err)
	}
	if payload.Winner != hostID.String() {
		t.Fatalf("winner = %s, want the connected player %s", payload.Winner, hostID)
	}
	if _, ok := payload.Scores[hostID.String()]; !ok {
		t.Fatalf("scores missing the surviving player %s: %v", hostID, payload.Scores)
	}
	if _, ok := payload.Scores[p2ID.String()]; ok {
		t.Fatalf("scores must omit the forfeited player %s: %v", p2ID, payload.Scores)
	}
	if !g.GetCurrentObfuscatedGameState(hostID).GameOver {
		t.Fatalf("game must be over after the forfeit")
	}
}

// TestE2EReconnectBeforeForfeitKeepsGameRunning is the reconnect-before-end half of cambia-837,
// driven end to end: with three players connected, one drop is not immediately fatal (two
// players remain), and the dropped player reconnecting before the game ends restores their seat
// and their place in the final scores.
func TestE2EReconnectBeforeForfeitKeepsGameRunning(t *testing.T) {
	gs, ts := newForfeitTestServer(t)

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())
	p3ID := uuid.New()
	p3Token, _ := auth.CreateJWT(p3ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	defer p2.close()
	p3 := dialWSClient(t, ctx, ts.URL, lobbyID, p3Token)
	host.settle()
	p2.settle()
	p3.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p3.settle()
	p2.sendReliable("ready")
	host.settle()
	p3.settle()
	p3.sendReliable("ready")

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("host never received game_started")
	}
	if host.waitForType("game_player_turn", 5*time.Second) == nil {
		t.Fatalf("host never received game_player_turn (game never left pre-game)")
	}

	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobbyID)
	}

	// Player 3's socket drops mid-game. Two of three players remain connected, so
	// ForfeitOnDisconnect must not end the game yet.
	p3.close()

	if got := host.waitForType("game_results", 300*time.Millisecond); got != nil {
		t.Fatalf("game ended after one drop out of three connected players")
	}
	if g.GetCurrentObfuscatedGameState(hostID).GameOver {
		t.Fatalf("game must still be running after one drop of three")
	}

	// Player 3 reconnects before any forfeit lands.
	p3Rejoin := dialWSClient(t, ctx, ts.URL, lobbyID, p3Token)
	defer p3Rejoin.close()
	if p3Rejoin.waitForType("private_sync_state", 5*time.Second) == nil {
		t.Fatalf("reconnecting player never received private_sync_state")
	}

	deadline := time.Now().Add(2 * time.Second)
	reconnected := false
	for time.Now().Before(deadline) {
		for _, p := range g.GetCurrentObfuscatedGameState(hostID).Players {
			if p.PlayerID == p3ID && p.Connected {
				reconnected = true
			}
		}
		if reconnected {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !reconnected {
		t.Fatalf("player 3's seat was never restored after reconnecting")
	}
	if g.GetCurrentObfuscatedGameState(hostID).GameOver {
		t.Fatalf("game must still be running after the reconnect")
	}

	// Ending the game now must score the reconnected player like everyone else, proving the
	// forfeit window truly closed rather than merely being delayed.
	g.EndGame()
	results := host.waitForType("game_results", 5*time.Second)
	if results == nil {
		t.Fatalf("host never received game_results after EndGame")
	}
	awaitGameEndPersistence(t, gs)
	var payload gameResultsPayload
	if err := json.Unmarshal(results.Payload, &payload); err != nil {
		t.Fatalf("decode game_results payload: %v", err)
	}
	if len(payload.Scores) != 3 {
		t.Fatalf("expected all 3 players scored, got %d: %v", len(payload.Scores), payload.Scores)
	}
	if _, ok := payload.Scores[p3ID.String()]; !ok {
		t.Fatalf("reconnected player %s missing from final scores: %v", p3ID, payload.Scores)
	}
}
