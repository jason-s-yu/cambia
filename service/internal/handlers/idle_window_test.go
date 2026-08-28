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

// TestInGameLobbyIgnoresTheShortWindow is the other half: the short window must not reach a
// table whose game is still running, however long its players have been gone. The lobby is
// reconsidered a long window later, and reaped then because the game has ended by that point.
func TestInGameLobbyIgnoresTheShortWindow(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 2 * time.Second
	gs.LobbyEmptyIdleTTL = 200 * time.Millisecond

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	// Past several empty-lobby windows, well short of one long window.
	time.Sleep(800 * time.Millisecond)

	if _, exists := gs.LobbyStore.GetLobby(lob.ID); !exists {
		t.Fatalf("an in-game lobby was reaped on the empty-lobby window")
	}

	// Once the game ends the lobby is reclaimed without needing another departure.
	lob.Mu.Lock()
	lob.InGame = false
	lob.GameID = uuid.Nil
	lob.Mu.Unlock()

	if !waitLobbyGone(t, gs, lob.ID, 4*time.Second) {
		t.Fatalf("a lobby idle since its game ended was never reaped")
	}
}
