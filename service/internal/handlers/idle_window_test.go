// internal/handlers/idle_window_test.go
//
// The short idle window end to end (cambia-884). GameServer.LobbyEmptyIdleTTL is copied onto
// every hub it creates alongside LobbyIdleTTL, so an abandoned pre-game or post-game lobby is
// released in minutes while a live game keeps the long grace from cambia-836.
package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// TestAbandonedLobbyReapsOnTheShortWindow drives the real create and WebSocket handlers: the
// only thing that can reclaim this lobby inside the test's patience is the empty-lobby window.
func TestAbandonedLobbyReapsOnTheShortWindow(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 30 * time.Second
	gs.LobbyEmptyIdleTTL = 150 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobbyUUID := createPublicLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}
	client.close()

	if !waitLobbyGone(t, gs, lobbyUUID, 3*time.Second) {
		t.Fatalf("an abandoned lobby outlived the short idle window")
	}
	if _, stillRegistered := gs.HubStore.GetHub(lobbyUUID); stillRegistered {
		t.Fatalf("the reap must deregister the lobby's hub")
	}
}

// TestInGameLobbyIgnoresTheShortWindow is the other half, driven over the real handlers: the reap
// must not reach a table whose game is still running, however long its players have been gone,
// and it must reach the lobby that game leaves behind without waiting out the long TTL. The whole
// table drops before the game ends here, which is the ordering that decides the second half: the
// idle window opens mid-game and nothing re-arms it when the game finishes.
func TestInGameLobbyIgnoresTheShortWindow(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 10 * time.Second
	gs.LobbyEmptyIdleTTL = 200 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()

	lobbyUUID := createPublicLobby(t, gs, hostToken)
	lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", lobbyUUID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// Connect first: a live socket holds the window closed, so the setup is not racing the
	// empty-lobby timer for the lobby it is about to put in game.
	client := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}

	joinLobbyAs(t, gs, lobbyUUID, hostToken)
	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	// The table drops mid-game.
	client.close()

	// Past several empty-lobby windows, well short of one long window.
	time.Sleep(time.Second)

	if _, stillThere := gs.LobbyStore.GetLobby(lobbyUUID); !stillThere {
		t.Fatalf("an in-game lobby was reaped on the empty-lobby window")
	}

	// The game ends with nobody connected. Only the short window can reclaim the lobby inside
	// this deadline; the long TTL would hold it for another nine seconds (cambia-884 F1).
	lob.Mu.Lock()
	lob.InGame = false
	lob.GameID = uuid.Nil
	lob.Mu.Unlock()

	if !waitLobbyGone(t, gs, lobbyUUID, 3*time.Second) {
		t.Fatalf("a lobby whose game ended after its table dropped waited out the long idle window")
	}
}
