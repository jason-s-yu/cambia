// internal/handlers/lobby_test.go
package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"
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
