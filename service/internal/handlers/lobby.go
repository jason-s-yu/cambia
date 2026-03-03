// internal/handlers/lobby.go
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
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
			if vis, ok := reqBody["visibility"].(string); ok {
				lob.Visibility = vis
			}
			if mode, ok := reqBody["mode"].(string); ok {
				lob.Mode = mode
			}
			if qID, ok := reqBody["queueID"].(string); ok {
				lob.QueueID = qID
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

		// Create a hub for this lobby so WS connections can join immediately.
		h := hub.NewHub(lob)
		gs.HubStore.CreateHub(h)
		go h.Run(context.Background())

		// Respond with the state of the newly created lobby.
		w.Header().Set("Content-Type", "application/json")
		// Encode the lobby struct directly; sensitive fields are marked `json:"-"`.
		json.NewEncoder(w).Encode(lob)
	}
}

// JoinLobbyHandler handles POST /lobby/{id}/join requests.
// It validates the lobby exists and the user is permitted (public lobby or invited),
// marks the user as joined, and returns the lobby_id.
func JoinLobbyHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return
		}

		// Path: /lobby/{id}/join
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 3 || parts[2] != "join" {
			http.NotFound(w, r)
			return
		}
		lobbyID, err := uuid.Parse(parts[1])
		if err != nil {
			http.Error(w, "Invalid lobby ID", http.StatusBadRequest)
			return
		}

		lob, exists := gs.LobbyStore.GetLobby(lobbyID)
		if !exists {
			http.Error(w, "Lobby not found", http.StatusNotFound)
			return
		}

		lob.Mu.Lock()
		isPublic := lob.Type == "public"
		_, isInvited := lob.Users[userID]
		if !isPublic && !isInvited {
			lob.Mu.Unlock()
			http.Error(w, "Not invited to this private lobby", http.StatusForbidden)
			return
		}
		lob.Users[userID] = true
		lob.Mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"lobby_id": lobbyID.String()})
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
			count := lob.JoinedCount()
			gameMode := lob.GameMode
			// Create a safe copy of the lobby data for the response.
			lobbyCopy := *lob
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

// SearchLobbyHandler handles POST /lobby/{id}/search.
// Enqueues the lobby into the matchmaking queue.
func SearchLobbyHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return
		}

		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 3 || parts[2] != "search" {
			http.NotFound(w, r)
			return
		}
		lobbyID, err := uuid.Parse(parts[1])
		if err != nil {
			http.Error(w, "Invalid lobby ID", http.StatusBadRequest)
			return
		}

		lob, exists := gs.LobbyStore.GetLobby(lobbyID)
		if !exists {
			http.Error(w, "Lobby not found", http.StatusNotFound)
			return
		}

		lob.Mu.Lock()
		if userID != lob.HostUserID {
			lob.Mu.Unlock()
			http.Error(w, "Only the host can start search", http.StatusForbidden)
			return
		}
		if lob.Searching {
			lob.Mu.Unlock()
			http.Error(w, "Already searching", http.StatusConflict)
			return
		}
		queueID := lob.QueueID
		playerCount := lob.JoinedCount()
		lob.Mu.Unlock()

		if queueID == "" {
			http.Error(w, "No queue selected for this lobby", http.StatusBadRequest)
			return
		}

		queueCfg, ok := matchmaking.GetQueueConfig(queueID)
		if !ok {
			http.Error(w, "Unknown queue ID", http.StatusBadRequest)
			return
		}

		entry := &matchmaking.QueuedLobby{
			LobbyID:     lobbyID,
			PlayerCount: playerCount,
			QueueID:     queueID,
			TargetCount: queueCfg.Players,
			IsRanked:    queueCfg.Ranked,
			QueuedAt:    time.Now(),
		}
		if err := gs.Matchmaker.Enqueue(entry); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		lob.Mu.Lock()
		lob.Searching = true
		lob.Mu.Unlock()

		h, hasHub := gs.HubStore.GetHub(lobbyID)
		if hasHub {
			h.Phase = hub.PhaseSearching
			h.QueueID = queueID
			h.IsRanked = queueCfg.Ranked
			h.TotalRounds = queueCfg.Rounds
			h.Emit("phase_change", map[string]interface{}{"phase": "searching"})
			h.Emit("search_status", map[string]interface{}{"searching": true, "queue_id": queueID})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "searching", "queue_id": queueID})
	}
}

// CancelSearchHandler handles DELETE /lobby/{id}/search.
// Removes the lobby from the matchmaking queue.
func CancelSearchHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return
		}

		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 3 || parts[2] != "search" {
			http.NotFound(w, r)
			return
		}
		lobbyID, err := uuid.Parse(parts[1])
		if err != nil {
			http.Error(w, "Invalid lobby ID", http.StatusBadRequest)
			return
		}

		lob, exists := gs.LobbyStore.GetLobby(lobbyID)
		if !exists {
			http.Error(w, "Lobby not found", http.StatusNotFound)
			return
		}

		lob.Mu.Lock()
		if userID != lob.HostUserID {
			lob.Mu.Unlock()
			http.Error(w, "Only the host can cancel search", http.StatusForbidden)
			return
		}
		lob.Searching = false
		lob.Mu.Unlock()

		gs.Matchmaker.Dequeue(lobbyID)

		h, hasHub := gs.HubStore.GetHub(lobbyID)
		if hasHub {
			h.Phase = hub.PhaseOpen
			h.Emit("phase_change", map[string]interface{}{"phase": "open"})
			h.Emit("search_status", map[string]interface{}{"searching": false})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "cancelled"})
	}
}

// ListQueuesHandler handles GET /matchmaking/queues.
// Returns all configured queues with live stats.
func ListQueuesHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		stats := gs.Matchmaker.QueueStats()

		type queueResponse struct {
			QueueID      string  `json:"queueId"`
			Name         string  `json:"name"`
			Players      int     `json:"players"`
			Rounds       int     `json:"rounds"`
			RatingPool   string  `json:"ratingPool"`
			Ranked       bool    `json:"ranked"`
			HiddenRating bool    `json:"hiddenRating"`
			PlayerCount  int     `json:"playerCount"`
			AvgWaitSec   float64 `json:"avgWaitSec"`
		}

		names := map[string]string{
			"h2h_quickplay":  "Quick Play",
			"h2h_blitz":      "H2H Blitz",
			"h2h_rapid":      "H2H Rapid",
			"h2h_classical":  "H2H Classical",
			"ffa4_standard":  "FFA-4 Standard",
			"ffa4_classical": "FFA-4 Classical",
		}

		var queues []queueResponse
		for id, cfg := range matchmaking.QueueConfigs {
			stat := stats[id]
			queues = append(queues, queueResponse{
				QueueID:      id,
				Name:         names[id],
				Players:      cfg.Players,
				Rounds:       cfg.Rounds,
				RatingPool:   cfg.RatingPool,
				Ranked:       cfg.Ranked,
				HiddenRating: cfg.HiddenRating,
				PlayerCount:  stat.PlayerCount,
				AvgWaitSec:   stat.AvgWaitSec,
			})
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(queues)
	}
}
