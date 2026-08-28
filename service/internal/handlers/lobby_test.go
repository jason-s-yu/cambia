// internal/handlers/lobby_test.go
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// TestLobbyCreate checks that POST /lobby/create successfully creates an ephemeral lobby
// in the GameServer's LobbyStore using an authenticated user token.
func TestLobbyCreate(t *testing.T) {
	auth.Init() // Use ephemeral keys for JWT generation, no DB needed for this part.
	gs := NewGameServer()

	// Generate an ephemeral user ID and corresponding JWT token.
	uHost := uuid.New()
	token, _ := auth.CreateJWT(uHost.String())

	// Prepare request body and HTTP request.
	body := `{"type":"private","gameMode":"head_to_head"}`
	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBuffer([]byte(body)))
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()

	// Execute the handler.
	h := CreateLobbyHandler(gs)
	h.ServeHTTP(w, req)

	// Assert response status and decode the created lobby.
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}
	var newLobby lobby.Lobby
	if err := json.Unmarshal(w.Body.Bytes(), &newLobby); err != nil {
		t.Fatalf("failed to decode lobby: %v", err)
	}
	if newLobby.ID == uuid.Nil {
		t.Fatalf("lobby has no ID")
	}
	if newLobby.HostUserID != uHost {
		t.Fatalf("lobby host mismatch, expected %v got %v", uHost, newLobby.HostUserID)
	}

	// Verify the lobby was added to the store.
	_, exists := gs.LobbyStore.GetLobby(newLobby.ID)
	if !exists {
		t.Fatalf("lobby %s was not found in the game server's lobby store", newLobby.ID)
	}
}

// TestLobbyCreateWithName checks that an optional "name" in the create request is
// stored on the lobby and carried through to GET /lobby/list, and that omitting it
// leaves the name as an empty string rather than absent.
//
// The listing half needs a live connection: since cambia-884 the list carries only lobbies
// somebody is actually connected to, so the host joins over the WebSocket the way the web
// client does after creating a lobby.
func TestLobbyCreateWithName(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	uHost := uuid.New()
	token, _ := auth.CreateJWT(uHost.String())

	body := `{"type":"public","gameMode":"head_to_head","name":"Friday Night Cambia"}`
	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBuffer([]byte(body)))
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()

	CreateLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}
	var newLobby lobby.Lobby
	if err := json.Unmarshal(w.Body.Bytes(), &newLobby); err != nil {
		t.Fatalf("failed to decode lobby: %v", err)
	}
	if newLobby.Name != "Friday Night Cambia" {
		t.Fatalf("expected lobby name %q, got %q", "Friday Night Cambia", newLobby.Name)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	client := dialWSClient(t, ctx, ts.URL, newLobby.ID.String(), token)
	defer client.close()
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}
	if !waitListed(t, gs, token, newLobby.ID, true, 2*time.Second) {
		t.Fatalf("lobby %s missing from /lobby/list response", newLobby.ID)
	}

	// The name must also appear in GET /lobby/list.
	entry := listLobbies(t, gs, token)[newLobby.ID.String()]
	if entry.Lobby.Name != "Friday Night Cambia" {
		t.Fatalf("expected listed lobby name %q, got %q", "Friday Night Cambia", entry.Lobby.Name)
	}

	// A lobby created without "name" carries an empty string, not an absent field.
	body2 := `{"type":"public","gameMode":"head_to_head"}`
	req2 := httptest.NewRequest("POST", "/lobby/create", bytes.NewBuffer([]byte(body2)))
	req2.Header.Set("Cookie", "auth_token="+token)
	w2 := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w2.Code, w2.Body.String())
	}
	var newLobby2 lobby.Lobby
	if err := json.Unmarshal(w2.Body.Bytes(), &newLobby2); err != nil {
		t.Fatalf("failed to decode lobby: %v", err)
	}
	if newLobby2.Name != "" {
		t.Fatalf("expected empty lobby name when unset, got %q", newLobby2.Name)
	}
}
