// internal/handlers/lobby.go
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
	"github.com/jason-s-yu/cambia/service/internal/rating"
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

// matchmakingQueueID resolves the queue a matchmaking lobby is being created for.
//
// The current shape carries it in queueID. The transitional one carries it in gameMode: the web
// bundle shipped before cambia-933 sent {type:"matchmaking", gameMode:"<queue id>"} and no
// queueID at all, and a tab holding that cached bundle keeps sending it after the deploy. That
// shape is accepted and logged as deprecated so those tabs keep working; remove it, and this
// fallback, after 2026-10-01.
//
// A game mode in gameMode is not a queue id and never becomes one: with no queueID the caller
// gets the missing-queue error rather than a silent default.
func matchmakingQueueID(lob *lobby.Lobby, userID uuid.UUID) (string, error) {
	if lob.QueueID != "" {
		return lob.QueueID, nil
	}
	if _, known := matchmaking.GetQueueConfig(lob.GameMode); known {
		log.Printf("lobby create: deprecated matchmaking shape from user %s: queue id %q sent as gameMode; send queueID instead", userID, lob.GameMode)
		return lob.GameMode, nil
	}
	return "", errors.New("Matchmaking lobby requires queueID")
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

		// Validate the lobby type before anything derived from it.
		if !validGameTypes[lob.Type] {
			http.Error(w, "Invalid lobby type specified", http.StatusBadRequest)
			return
		}

		// A matchmaking lobby is defined by its queue, not by a client-supplied game mode: the
		// queue config carries the player count, the round count and whether the queue is
		// ranked, and SearchLobbyHandler reads that same config again from lob.QueueID when the
		// lobby enters the queue. Deriving the mode-shaped fields here is what keeps the two
		// halves from disagreeing; before cambia-933 the client sent the queue id as gameMode
		// and every Play click 400'd on "Invalid game mode specified".
		if lob.Type == "matchmaking" {
			queueID, derr := matchmakingQueueID(lob, userID)
			if derr != nil {
				http.Error(w, derr.Error(), http.StatusBadRequest)
				return
			}
			cfg, known := matchmaking.GetQueueConfig(queueID)
			if !known {
				http.Error(w, "Unknown matchmaking queue: "+queueID, http.StatusBadRequest)
				return
			}
			// The queue's player count picks the game mode. Rounds and ranked-ness are
			// deliberately NOT copied onto a second home on the lobby: the hub reads them from
			// the queue config at search time (SearchLobbyHandler sets h.TotalRounds and
			// h.IsRanked from it), so the config stays the single source of truth. A
			// multi-round queue therefore keeps the player-count mode rather than a circuit_*
			// one until the round lifecycle lands (cambia-466).
			switch cfg.Players {
			case 2:
				lob.GameMode = "head_to_head"
			case 4:
				lob.GameMode = "group_of_4"
			default:
				http.Error(w, fmt.Sprintf("Matchmaking queue %s has an unsupported player count: %d", queueID, cfg.Players), http.StatusBadRequest)
				return
			}
			lob.QueueID = queueID
			if cfg.Ranked {
				lob.Mode = "ranked"
			} else {
				lob.Mode = "casual"
			}
		} else if lob.QueueID != "" {
			// A public or private lobby may legitimately carry a queue id: SearchLobbyHandler
			// gates on host, Searching and QueueID alone, so a standing lobby can queue its
			// party without being typed "matchmaking". The id is kept, and validated here so a
			// bogus one fails at create time rather than at search time.
			cfg, known := matchmaking.GetQueueConfig(lob.QueueID)
			if !known {
				http.Error(w, "Unknown matchmaking queue: "+lob.QueueID, http.StatusBadRequest)
				return
			}
			// Mirror the matchmaking-type branch above: a lobby's Mode is what
			// NewCambiaGameFromLobby reads to decide whether the game it produces is rated, and
			// what the ranked-lock on update_rules (hub.go) checks. Without this a party that
			// queued a standing public/private lobby into a ranked queue by carrying its id here
			// played a rated queue's match as an unrated, rule-editable casual game: the id was
			// kept and validated, but the one thing carrying a queue id is supposed to mean never
			// derived from it (cambia-966).
			if cfg.Ranked {
				lob.Mode = "ranked"
			} else {
				lob.Mode = "casual"
			}
		}

		// Validate the final game mode. For a matchmaking lobby this now checks a value this
		// handler derived, not one the client sent.
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

		// A matchmaking lobby's host is its party of one, so they join it outright rather than
		// only being invited. SearchLobbyHandler sizes the party from JoinedCount and the
		// matchmaker refuses a party of zero, while the client queues straight from the
		// dashboard: with an invite alone every search 400'd on "PlayerCount must be > 0", since
		// the WebSocket upgrade is the only other thing that promotes an invite to a join
		// (cambia-933).
		if lob.Type == "matchmaking" {
			lob.JoinUser(userID)
		}

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

// lobbyListingCreationGrace keeps a just-created lobby listed for a short window regardless of
// live-connection presence (cambia-887 F3). A lobby's host has to POST /lobby/create and then
// open their own WebSocket as two separate round trips; without a grace, the lobby is invisible
// to everyone else's list for that gap, and a brief blip in the host's own socket right after
// connecting flickers it back out. The window is generous enough to absorb both without masking
// a genuinely abandoned lobby for long: ListQueuesHandler is the sibling that answers a similar
// "what can I join" question with a bare cross-section and no such grace, because a queue has no
// per-entry creation moment to protect.
const lobbyListingCreationGrace = 30 * time.Second

// ListLobbiesHandler returns a map of currently joinable ephemeral lobbies from the store.
// For each lobby, it includes player count and calculated max player count based on game mode.
//
// Only lobbies of type "public" are listed. A private lobby's id, host id, host-typed name and
// house rules are not meant for a caller who was never invited, and the list carries no identity
// requirement below to tell an invited caller from an uninvited one anyway; a member reaches
// their own private lobby through GET /lobby/active instead (cambia-900 L1).
//
// A lobby with no game in progress and no live WebSocket connection is left out entirely rather
// than listed as inactive. playerCount reports membership, and only a deliberate leave releases
// membership (cambia-807), so a table that finished a game and closed its tabs kept listing
// itself at 2/2 - rendered "Full", unjoinable, nobody there - for the whole idle window
// (cambia-884). Excluding is the honest answer because the list answers one question, "what can
// I join right now", and a lobby with nobody in it answers it no better with a label on it. The
// people who hold membership lose nothing: GET /lobby/active still offers them the lobby back
// for as long as the idle reaper leaves it standing. A lobby inside lobbyListingCreationGrace of
// its own creation is exempt from this filter regardless of presence (cambia-887 F3).
//
// The list carries no identity requirement: nothing below reads the caller's user ID, so the
// list is public the same way ListQueuesHandler is, and the handler makes no auth call at all
// rather than authenticating and discarding the result. A discarded-but-attempted auth call
// used to leave an unauthenticated GET with a malformed body - authenticateAndGetUser's 401
// text, followed by this handler's own JSON encoding, both written to the same response body
// with no early return between them (cambia-887 F4).
func ListLobbiesHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		lobbiesMap := gs.LobbyStore.GetLobbies() // Retrieve all active lobbies.
		responseMap := make(map[string]ListLobbiesResponse)

		for id, lob := range lobbiesMap {
			lob.Mu.Lock() // Lock lobby to safely read its current state.
			count := lob.JoinedCount()
			gameMode := lob.GameMode
			inGame := lob.InGame
			createdAt := lob.CreatedAt
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
				CreatedAt:     createdAt,
				HouseRules:    lob.HouseRules,
				Circuit:       lob.Circuit,
				LobbySettings: lob.LobbySettings,
				Mode:          lob.Mode,
				QueueID:       lob.QueueID,
				Searching:     lob.Searching,
			}
			lob.Mu.Unlock() // Unlock after reading.

			// Skip non-public lobbies. This list carries no identity requirement (see doc
			// above) and answers "what can anyone join right now"; a private lobby's id, host
			// id, host-typed name and house rules are not meant for that audience, and its
			// members already have their own way back through GET /lobby/active (cambia-900
			// L1). Read under the same lock as inGame/createdAt above rather than re-reading
			// lob.Type here, unlocked.
			if lobbyCopy.Type != "public" {
				continue
			}

			// Skip lobbies nobody is connected to, unless a running game keeps its listing
			// (players are mid-table and expected back, the same exemption the idle reaper
			// makes) or the lobby is still inside its creation grace (its host may simply not
			// have opened their WebSocket yet, or just blipped it).
			if live, serving := gs.HubStore.LiveConnections(id); !inGame && (!serving || live == 0) {
				if time.Since(createdAt) >= lobbyListingCreationGrace {
					continue
				}
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
		// Snapshot the joined member ids while the lock is held so the rating lookup below (a DB
		// round trip) can run unlocked, the same trade the rest of this handler already makes for
		// playerCount.
		partyUserIDs := make([]uuid.UUID, 0, playerCount)
		for uid, joined := range lob.Users {
			if joined {
				partyUserIDs = append(partyUserIDs, uid)
			}
		}
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

		// Live ratings for every party member, pool-aware (rating.ModeForPlayerCount picks the
		// pool from the queue's target player count, the same gate applyRatingUpdate uses at game
		// end). A player with no rating row - or no DB reachable at all - reads back pool defaults
		// (database.LoadPartyRatings), never a zero value: AvgRating/MaxRD used to be left at their
		// zero defaults entirely (no writer ever touched them), which made glicko2Quality's spread
		// term zero for every pairing and the ranked quality gate pass everyone (cambia-1041).
		partyRatings := database.LoadPartyRatings(r.Context(), partyUserIDs, rating.ModeForPlayerCount(queueCfg.Players))

		// A party's own size has to obey the queue family's rules (solo queue only for H2H,
		// parties up to 2 for FFA-4) before it ever reaches the matchmaker: Enqueue only rejects
		// a party bigger than the queue's whole target size, which a full-size H2H party of two
		// friends still passes, pairing them with each other every time and skipping matchmaking
		// entirely. ValidateParty carried this rule already but had no caller (cambia-966). Its
		// FFA-4 spread check reads OpenSkill mu (ratingSpread's 15-unit limit is a mu-scale limit,
		// not an Elo one); before cambia-1041 it was always called with ratings=nil, so the check
		// never fired.
		if err := matchmaking.ValidateParty(queueID, playerCount, database.PartyOpenSkillMu(partyRatings)); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		avgRating, maxRD := database.AggregatePartyGlicko(partyRatings)

		entry := &matchmaking.QueuedLobby{
			LobbyID:     lobbyID,
			PlayerCount: playerCount,
			QueueID:     queueID,
			TargetCount: queueCfg.Players,
			AvgRating:   avgRating,
			MaxRD:       maxRD,
			IsRanked:    queueCfg.Ranked,
			QueuedAt:    time.Now(),
		}
		if err := gs.Matchmaker.Enqueue(entry); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		lob.Mu.Lock()
		lob.Searching = true
		// The hub's phase and match parameters belong to its Run goroutine, so the change is
		// handed to it rather than written from this one (cambia-933). The notice is sent inside
		// the same critical section as the Searching flag write: SetSearchState only pushes onto
		// a buffered channel and never blocks, so holding lob.Mu here is cheap, and it is what
		// keeps a concurrent cancel on this lobby from landing its hub notice out of the order
		// the two requests actually ran in (cambia-966). Sending it unlocked let a goroutine that
		// lost the race to acquire the lock still win the race to the hub's channel, so a
		// cancel-then-search from two overlapping requests could reach the hub as
		// search-then-cancel, leaving it phased as Searching while lob.Searching read false.
		if h, hasHub := gs.HubStore.GetHub(lobbyID); hasHub {
			h.SetSearchState(hub.SearchState{
				Searching:   true,
				QueueID:     queueID,
				IsRanked:    queueCfg.Ranked,
				TotalRounds: queueCfg.Rounds,
			})
		}
		lob.Mu.Unlock()

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
		// See the matching comment in SearchLobbyHandler: the hub notice is sent inside the same
		// critical section as the Searching flag write so the two requests' hub notices land in
		// the order the requests actually acquired lob.Mu, not whichever goroutine happens to
		// reach the hub's channel first (cambia-966).
		if h, hasHub := gs.HubStore.GetHub(lobbyID); hasHub {
			h.SetSearchState(hub.SearchState{Searching: false})
		}
		lob.Mu.Unlock()

		gs.Matchmaker.Dequeue(lobbyID)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "cancelled"})
	}
}

// ListQueuesHandler handles GET /matchmaking/queues.
// Returns all configured queues with live stats.
//
// The response is sorted by each queue's matchmaking.QueueConfig.Order (ties broken by
// QueueID), not by ranging over matchmaking.QueueConfigs directly: Go re-randomizes map
// iteration order on every range statement, so two calls in the same process - not just
// across restarts - could return different sequences, which previously left the six queue
// cards reordering themselves between dashboard loads with nothing wrong to look at
// (cambia-957).
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
			// Family-prefixed like the other five so the card label does not read as
			// the dashboard heading (cambia-922 F1: hero/queue name collision).
			"h2h_quickplay":  "H2H Quick",
			"h2h_blitz":      "H2H Blitz",
			"h2h_rapid":      "H2H Rapid",
			"h2h_classical":  "H2H Classical",
			"ffa4_standard":  "FFA-4 Standard",
			"ffa4_classical": "FFA-4 Classical",
		}

		// Range over the map to collect ids, then sort before building the response: the
		// map itself carries no order, and building queueResponse entries in iteration
		// order (the previous bug, cambia-957) would still leave a JSON array whose
		// element order Go never promises to repeat.
		ids := make([]string, 0, len(matchmaking.QueueConfigs))
		for id := range matchmaking.QueueConfigs {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(i, j int) bool {
			oi, oj := matchmaking.QueueConfigs[ids[i]].Order, matchmaking.QueueConfigs[ids[j]].Order
			if oi != oj {
				return oi < oj
			}
			return ids[i] < ids[j]
		})

		queues := make([]queueResponse, 0, len(ids))
		for _, id := range ids {
			cfg := matchmaking.QueueConfigs[id]
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
