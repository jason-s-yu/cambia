// internal/handlers/grace_flow_test.go
//
// Disconnect grace and post-game reconnect over the real WebSocket stack (cambia-955). The
// reported failure was a page reload mid-game: the socket closed, ForfeitOnDisconnect took the
// seat within about 700ms, the two-player game ended on "only 1 player(s) left connected", and
// the reloaded client landed on a results screen with no winner and no scores because game_results
// had already gone out to a socket that no longer existed.
package handlers

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// startGraceGame drives a fresh public lobby to a started game over real WebSockets and returns
// both clients, the lobby and player ids, the server URL to re-dial and the dial context.
// graceSec is the lobby's reconnect window, applied before the game is built.
func startGraceGame(t *testing.T, graceSec int) (*GameServer, *wsTestClient, *wsTestClient, uuid.UUID, uuid.UUID, uuid.UUID, string, context.Context) {
	t.Helper()
	gs, ts := newForfeitTestServer(t)

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()
	setDisconnectGrace(t, gs, lobUUID, graceSec)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	t.Cleanup(cancel)

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	t.Cleanup(host.close)
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
	if host.waitForType("game_player_turn", 5*time.Second) == nil {
		t.Fatalf("host never received game_player_turn (game never left pre-game)")
	}
	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobbyID)
	}
	// Registered last, so it runs first (t.Cleanup is LIFO): the game is ended while its sockets
	// are still open, which stops any armed grace timer. Without this a test whose game is still
	// running would leave a window to fire seconds later, ending the game - and launching its
	// persistence goroutines - after the test that owned them had finished (cambia-908).
	t.Cleanup(func() {
		g.EndGame()
		awaitGameEndPersistence(t, gs)
	})
	return gs, host, p2, lobUUID, hostID, p2ID, ts.URL, ctx
}

// TestE2EReconnectInsideTheGraceWindowKeepsTheSeat is the reported reload, driven end to end: the
// socket drops, the game must not end, the remaining player is told the seat is being held, and
// the client that comes back is dealt the same table it left.
func TestE2EReconnectInsideTheGraceWindowKeepsTheSeat(t *testing.T) {
	gs, host, p2, lobUUID, hostID, p2ID, tsURL, ctx := startGraceGame(t, 30)

	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	handBefore := map[string]bool{}
	for _, p := range g.GetCurrentObfuscatedGameState(p2ID).Players {
		if p.PlayerID == p2ID {
			for _, c := range p.RevealedHand {
				handBefore[c.ID.String()] = true
			}
		}
	}
	stockBefore := g.GetCurrentObfuscatedGameState(hostID).StockpileSize

	// The reload: the socket closes.
	p2.close()

	if env := host.waitForType("player_reconnecting", 5*time.Second); env == nil {
		t.Fatalf("the remaining player was never told the seat is being held")
	}
	if env := host.waitForType("game_results", 700*time.Millisecond); env != nil {
		t.Fatalf("the game ended inside the grace window: %s", string(env.Payload))
	}
	if g.GetCurrentObfuscatedGameState(hostID).GameOver {
		t.Fatalf("the game must still be running inside the grace window")
	}
	if g.IsForfeited(p2ID) {
		t.Fatalf("a player inside their window must not be forfeited")
	}

	// The reloaded page comes back on a new socket.
	rejoin := dialWSClient(t, ctx, tsURL, lobUUID.String(), tokenFor(t, p2ID))
	defer rejoin.close()
	sync := rejoin.waitForType("private_sync_state", 5*time.Second)
	if sync == nil {
		t.Fatalf("the returning player never received private_sync_state")
	}
	if env := host.waitForType("player_reconnected", 5*time.Second); env == nil {
		t.Fatalf("the table was never told the player came back")
	}

	var syncPayload struct {
		State struct {
			StockpileSize int `json:"stockpileSize"`
			Players       []struct {
				PlayerID     string `json:"playerId"`
				Connected    bool   `json:"connected"`
				Forfeited    bool   `json:"forfeited"`
				RevealedHand []struct {
					ID string `json:"id"`
				} `json:"revealedHand"`
			} `json:"players"`
		} `json:"state"`
	}
	if err := json.Unmarshal(sync.Payload, &syncPayload); err != nil {
		t.Fatalf("decode private_sync_state: %v", err)
	}
	if syncPayload.State.StockpileSize != stockBefore {
		t.Fatalf("stockpile = %d, want the %d it was before the drop", syncPayload.State.StockpileSize, stockBefore)
	}
	seen := 0
	for _, p := range syncPayload.State.Players {
		if p.PlayerID != p2ID.String() {
			continue
		}
		if !p.Connected || p.Forfeited {
			t.Fatalf("the restored seat reads connected=%v forfeited=%v", p.Connected, p.Forfeited)
		}
		for _, c := range p.RevealedHand {
			if !handBefore[c.ID] {
				t.Fatalf("the restored hand carries card %s, which was not in the hand before the drop", c.ID)
			}
			seen++
		}
	}
	if seen == 0 || seen != len(handBefore) {
		t.Fatalf("restored hand has %d cards, want the %d held before the drop", seen, len(handBefore))
	}
}

// TestE2EReloadAfterGameEndGetsTheResults is the second half: a client that reconnects to a hub
// whose game is already over is sent the results again, instead of a phase with no scores in it.
func TestE2EReloadAfterGameEndGetsTheResults(t *testing.T) {
	gs, host, p2, lobUUID, hostID, p2ID, tsURL, ctx := startGraceGame(t, 0)

	// Grace 0, so the drop forfeits and ends the two-player game straight away.
	p2.close()
	results := host.waitForType("game_results", 5*time.Second)
	if results == nil {
		t.Fatalf("host never received game_results after the drop")
	}
	awaitGameEndPersistence(t, gs)

	// The forfeited player reloads into the finished game.
	rejoin := dialWSClient(t, ctx, tsURL, lobUUID.String(), tokenFor(t, p2ID))
	defer rejoin.close()

	resent := rejoin.waitForType("game_results", 5*time.Second)
	if resent == nil {
		t.Fatalf("a reload into the finished game got no results")
	}
	var payload gameResultsPayload
	if err := json.Unmarshal(resent.Payload, &payload); err != nil {
		t.Fatalf("decode resent game_results: %v", err)
	}
	if payload.Winner != hostID.String() {
		t.Fatalf("resent results name winner %q, want %s", payload.Winner, hostID)
	}
	if _, ok := payload.Scores[hostID.String()]; !ok {
		t.Fatalf("resent results carry no score for the winner: %v", payload.Scores)
	}
	if g := gs.GameStore.GetGameByLobbyID(lobUUID); g != nil && !g.GetCurrentObfuscatedGameState(hostID).GameOver {
		t.Fatalf("the game must be over")
	}
}

// tokenFor mints a JWT for an id already in play, so a reconnect authenticates as the same user.
func tokenFor(t *testing.T, id uuid.UUID) string {
	t.Helper()
	tok, err := auth.CreateJWT(id.String())
	if err != nil {
		t.Fatalf("create JWT: %v", err)
	}
	return tok
}
