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
			if name, ok := reqBody["name"].(string); ok {
				lob.Name = name
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

		// Auto-invite the host to their own lobby. lob.Users tracks who is allowed to join
		// (map presence, joined=true or invited=false) and HubWSHandler's private-lobby gate
		// checks that same membership before allowing the WS upgrade. Without this, a private
		// lobby's own creator has no entry in lob.Users and no self-invite path, so the gate
		// refuses their own connection (cambia-771). This is safe to call unlocked: the lobby
		// has not yet been added to the store or hub, so nothing else can observe it.
		lob.InviteUser(userID)

		// Configure the OnEmpty callback to release the lobby and its hub once the last joined
		// member leaves. Reachable only from lobby.RemoveUser, i.e. from a deliberate leave: a
		// dropped WebSocket keeps its membership (cambia-807).
		lob.OnEmpty = gs.tearDownLobby

		// Add the configured lobby to the central store.
		gs.LobbyStore.AddLobby(lob)

		// Create a hub for this lobby so WS connections can join immediately. Inject the game
		// factory and countdown length so the hub can create the engine-backed game itself when
		// the lobby countdown elapses (cambia-458).
		h := hub.NewHub(lob)
		h.CreateGame = gs.hubGameFactory()
		// A hub that has stopped serving must not stay discoverable, or the next WebSocket to
		// this lobby is accepted and never answered (cambia-808).
		h.OnDissolve = gs.HubStore.DeleteHub
		// Reaping an abandoned lobby runs the same teardown as the last member leaving. Nothing
		// else reclaims one: closing a tab is not a leave, so membership survives and OnEmpty is
		// never reached, and the hub now runs for the lobby's lifetime (cambia-836).
		h.OnIdle = gs.tearDownLobby
		if gs.CountdownDuration > 0 {
			h.CountdownDuration = gs.CountdownDuration
		}
		if gs.PostGameDuration > 0 {
			h.PostGameDuration = gs.PostGameDuration
		}
		if gs.LobbyIdleTTL > 0 {
			h.IdleTTL = gs.LobbyIdleTTL
		}
		if gs.LobbyEmptyIdleTTL > 0 {
			h.EmptyIdleTTL = gs.LobbyEmptyIdleTTL
		}
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
		lob.MarkJoinedUnsafe(userID)
		lob.Mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"lobby_id": lobbyID.String()})
	}
}

// LeaveLobbyHandler handles POST /lobby/{id}/leave: the deliberate counterpart to
// /lobby/{id}/join. Membership is released here, on the surface that granted it, and not over
// the WebSocket, for two reasons. A lost socket must never release membership, because
// reconnecting and the resume banner both key off it (cambia-783). And a client that sends a
// leave frame and closes its socket in the same breath races its own disconnect through the
// hub's select, so the release would land only sometimes.
//
// Leaving mid-game is refused: a seat in a running game is not something a lobby-level leave
// can release, and abandoning a game is what a disconnect already means. The lobby's InGame
// flag is the gate rather than the hub phase, which only the hub's Run goroutine may read.
//
// The response is 200 for a caller who holds no membership: leaving twice, or leaving a lobby
// somebody else already emptied, is not an error the client should surface.
func LeaveLobbyHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return
		}

		// Path: /lobby/{id}/leave
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) < 3 || parts[2] != "leave" {
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
		_, isMember := lob.Users[userID]
		inGame := lob.InGame
		lob.Mu.Unlock()

		if inGame {
			http.Error(w, "Cannot leave a lobby while its game is in progress", http.StatusConflict)
			return
		}
		if !isMember {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "left", "removed": false})
			return
		}

		// RemoveUser fires OnEmpty (gs.tearDownLobby) when the last joined member goes, which
		// deletes the lobby and stops its hub.
		removed := lob.RemoveUser(userID)

		// Drop the leaver's live connection and refresh the roster everyone else sees. Routed
		// through the hub's leave channel because hub state belongs to its Run goroutine; a
		// no-op once the hub has stopped, which is the case that just tore the lobby down.
		if h, hasHub := gs.HubStore.GetHub(lobbyID); hasHub {
			h.Leave(userID)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "left", "removed": removed})
	}
}

// ListLobbiesResponse defines the structure for each entry in the lobby list response.
// It includes the core lobby details plus player count information.
type ListLobbiesResponse struct {
	Lobby       *lobby.Lobby `json:"lobby"` // Core lobby state.
	PlayerCount int          `json:"playerCount"`
	MaxPlayers  int          `json:"maxPlayers"`
}

// ListLobbiesHandler returns a map of currently joinable ephemeral lobbies from the store.
// For each lobby, it includes player count and calculated max player count based on game mode.
//
// A lobby with no game in progress and no live WebSocket connection is left out entirely rather
// than listed as inactive. playerCount reports membership, and only a deliberate leave releases
// membership (cambia-807), so a table that finished a game and closed its tabs kept listing
// itself at 2/2 - rendered "Full", unjoinable, nobody there - for the whole idle window
// (cambia-884). Excluding is the honest answer because the list answers one question, "what can
// I join right now", and a lobby with nobody in it answers it no better with a label on it. The
// people who hold membership lose nothing: GET /lobby/active still offers them the lobby back
// for as long as the idle reaper leaves it standing.
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
			inGame := lob.InGame
			// Copy only the fields the response needs. Users, ReadyStates,
			// GameInstanceCreated, CountdownTimer, OnEmpty and Mu are left at
			// their zero value: the JSON encoding already ignores them via
			// `json:"-"`, and skipping them avoids copying lob.Mu (assigning
			// a sync.Mutex trips go vet's copylocks check and would leave
			// lobbyCopy holding a copy of the source lock).
			lobbyCopy := lobby.Lobby{
				ID:            lob.ID,
				HostUserID:    lob.HostUserID,
				Type:          lob.Type,
				GameMode:      lob.GameMode,
				Name:          lob.Name,
				GameID:        lob.GameID,
				InGame:        lob.InGame,
				HouseRules:    lob.HouseRules,
				Circuit:       lob.Circuit,
				LobbySettings: lob.LobbySettings,
				Visibility:    lob.Visibility,
				Mode:          lob.Mode,
				QueueID:       lob.QueueID,
				Searching:     lob.Searching,
			}
			lob.Mu.Unlock() // Unlock after reading.

			// Skip lobbies nobody is connected to. A running game keeps its listing whatever the
			// socket count: its players are mid-table and are expected back, the same exemption
			// the idle reaper makes.
			if live, serving := gs.HubStore.LiveConnections(id); !inGame && (!serving || live == 0) {
				continue
			}

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
