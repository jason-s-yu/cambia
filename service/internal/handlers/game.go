// internal/handlers/game.go
package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// ServeHTTP handles legacy/debug HTTP routes for the game service.
// The primary game interaction now occurs via the WebSocket handler in game_ws.go.
// Routes handled:
// POST /game/create: Creates a game instance directly (for debugging).
// GET /game/reconnect/{uuid}: Acknowledges HTTP reconnect attempt (deprecated).
func (s *GameServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/game/create" && r.Method == http.MethodPost {
		s.handleCreateGame(w, r)
		return
	}

	// The HTTP reconnect handler is largely deprecated in favor of the WebSocket handler's reconnect logic.
	// It's kept here perhaps for testing or legacy reasons but doesn't handle full state synchronization.
	if strings.HasPrefix(r.URL.Path, "/game/reconnect/") {
		s.handleReconnect(w, r) // Note: This HTTP handler cannot fully reconnect the WS state.
		return
	}

	http.Error(w, "Unsupported route. Use /game/ws/{id} for game WebSockets.", http.StatusNotFound)
}

// handleCreateGame creates a new in-memory CambiaGame instance and adds it to the GameStore.
// This endpoint is primarily for debugging and does not handle player joining or authentication.
func (s *GameServer) handleCreateGame(w http.ResponseWriter, r *http.Request) {
	cg := game.NewCambiaGame()
	s.GameStore.AddGame(cg)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"game_id": cg.ID,
	})
}

// handleReconnect acknowledges an HTTP-based reconnect attempt.
// Deprecated: The WebSocket handler GameWSHandler provides a more robust reconnection flow.
// This HTTP handler only validates the game ID and instructs the client to use WebSockets.
// It does not authenticate the user or modify the game state.
func (s *GameServer) handleReconnect(w http.ResponseWriter, r *http.Request) {
	gameIDStr := strings.TrimPrefix(r.URL.Path, "/game/reconnect/")
	gameID, err := uuid.Parse(gameIDStr)
	if err != nil {
		http.Error(w, "invalid game id", http.StatusBadRequest)
		return
	}
	_, ok := s.GameStore.GetGame(gameID)
	if !ok {
		http.Error(w, "game not found", http.StatusNotFound)
		return
	}

	// Authentication logic would normally go here to verify the user belongs to the game.
	// Skipping auth for this deprecated handler.

	// Instruct the user to use WebSocket for actual reconnection.
	w.Write([]byte("Reconnect acknowledged via HTTP. Please establish a WebSocket connection to /game/ws/" + gameIDStr + " to fully rejoin."))
}
