// internal/handlers/lobby.go
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// Define valid enum-like values for lobby type and game mode.
var validGameTypes = map[string]bool{
	"private":     true,
	"public":      true,
	"matchmaking": true, // Although matchmaking logic isn't implemented yet.
}
var validGameModes = map[string]bool{
	"head_to_head": true,
	"group_of_4":   true,
	"circuit_4p":   true,
	"circuit_7p8p": true,
	"custom":       true, // Allow custom mode if needed.
}

// CreateLobbyHandler handles requests to create a new ephemeral lobby.
// It authenticates the user, creates a lobby with default or provided settings,
// configures it for automatic cleanup via OnEmpty, adds it to the store,
// and returns the created lobby's state.
func CreateLobbyHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return // Error response handled by authenticateAndGetUser.
		}

		// Create a new lobby instance with default settings, hosted by the authenticated user.
		lob := lobby.NewLobbyWithDefaults(userID)

		// Decode optional request body to override defaults.
		var reqBody map[string]interface{}
		// Allow empty body gracefully.
		err := json.NewDecoder(r.Body).Decode(&reqBody)
		if err != nil && !errors.Is(err, context.Canceled) && err.Error() != "EOF" {
			http.Error(w, "Invalid lobby creation payload: "+err.Error(), http.StatusBadRequest)
			return
		}

		// Apply overrides from request body if present.
		// lob.Update handles nested structures like houseRules, circuit, lobbySettings.
		if reqBody != nil {
			if reqType, ok := reqBody["type"].(string); ok {
				lob.Type = reqType // Explicitly set type if provided directly.
			}
			if reqMode, ok := reqBody["gameMode"].(string); ok {
				lob.GameMode = reqMode // Explicitly set gameMode if provided directly.
			}
			lob.Update(reqBody) // Apply overrides for rules/settings.
		}

		// Validate final lobby type and game mode.
		if !validGameTypes[lob.Type] {
			http.Error(w, "Invalid lobby type specified", http.StatusBadRequest)
			return
		}
		if !validGameModes[lob.GameMode] {
			http.Error(w, "Invalid game mode specified", http.StatusBadRequest)
			return
		}

		// Configure the OnEmpty callback to remove the lobby from the store when it becomes empty.
		lob.OnEmpty = func(lobbyID uuid.UUID) {
			gs.LobbyStore.DeleteLobby(lobbyID)
		}

		// Add the configured lobby to the central store.
		gs.LobbyStore.AddLobby(lob)

		// Respond with the state of the newly created lobby.
		w.Header().Set("Content-Type", "application/json")
		// Encode the lobby struct directly; sensitive fields are marked `json:"-"`.
		json.NewEncoder(w).Encode(lob)
	}
}

// ListLobbiesResponse defines the structure for each entry in the lobby list response.
// It includes the core lobby details plus player count information.
type ListLobbiesResponse struct {
	Lobby       *lobby.Lobby `json:"lobby"` // Core lobby state.
	PlayerCount int          `json:"playerCount"`
	MaxPlayers  int          `json:"maxPlayers"`
}

// ListLobbiesHandler returns a map of currently active ephemeral lobbies from the store.
// For each lobby, it includes player count and calculated max player count based on game mode.
func ListLobbiesHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Authentication is optional for listing lobbies, but included for consistency.
		_, ok := authenticateAndGetUser(w, r)
		if !ok {
			// If auth is required for listing, return here.
			// Currently, let it proceed even if auth fails.
		}

		lobbiesMap := gs.LobbyStore.GetLobbies() // Retrieve all active lobbies.
		responseMap := make(map[string]ListLobbiesResponse)

		for id, lob := range lobbiesMap {
			lob.Mu.Lock() // Lock lobby to safely read its current state.
			count := len(lob.Connections)
			gameMode := lob.GameMode
			// Create a safe copy of the lobby data for the response.
			// This avoids holding the lock during JSON marshaling.
			lobbyCopy := *lob
			lobbyCopy.Connections = nil // Exclude sensitive/internal data.
			lobbyCopy.Users = nil
			lobbyCopy.ReadyStates = nil
			lob.Mu.Unlock() // Unlock after reading.

			// Determine max players based on game mode.
			maxPlayers := 4 // Default max players.
			switch gameMode {
			case "head_to_head":
				maxPlayers = 2
			case "group_of_4", "circuit_4p":
				maxPlayers = 4
			case "circuit_7p8p":
				maxPlayers = 8
			}

			// Add lobby details to the response map.
			responseMap[id.String()] = ListLobbiesResponse{
				Lobby:       &lobbyCopy, // Use the safe copy.
				PlayerCount: count,
				MaxPlayers:  maxPlayers,
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(responseMap)
	}
}
