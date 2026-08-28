// internal/handlers/active_session.go
package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
)

// ActiveSession describes the lobby a caller can return to after losing their tab, plus the
// in-progress game running in it, if any. LobbyID is the only routing target the client needs:
// the web client's /lobby/:lobbyId page renders the lobby or the live table off the hub phase.
type ActiveSession struct {
	LobbyID   string `json:"lobbyId"`
	LobbyType string `json:"lobbyType"`
	GameMode  string `json:"gameMode"`
	Name      string `json:"name,omitempty"`

	// Phase is "open", "searching" or "in_game", derived from lobby state rather than read
	// off the hub: the hub's own Phase is mutated only inside its Run goroutine and reading
	// it here would race. The three values are what a resume affordance needs to label
	// itself; the client gets the hub's exact phase from lobby_state once it reconnects.
	Phase string `json:"phase"`

	// GameID is set only while a live (not yet finished) game is registered for this lobby.
	GameID string `json:"gameId,omitempty"`

	// Seated distinguishes "your game is running" from "the lobby you belong to is playing a
	// game you have no seat in" (a member who joined after the deal).
	Seated bool `json:"seated"`

	// PlayerCount is the seated player count while in game, otherwise the count of joined
	// lobby members.
	PlayerCount int `json:"playerCount"`
}

// ActiveSessionResponse wraps the single session so the empty case is an explicit null rather
// than an absent body.
type ActiveSessionResponse struct {
	Active *ActiveSession `json:"active"`
}

// sessionRank orders candidate sessions when a caller is a member of several live lobbies: a
// game in progress outranks a matchmaking search, which outranks an idle lobby.
func sessionRank(phase string) int {
	switch phase {
	case "in_game":
		return 2
	case "searching":
		return 1
	default:
		return 0
	}
}

// ActiveSessionHandler handles GET /lobby/active. It scans the in-memory lobby store for
// lobbies the authenticated caller has joined and returns the one worth resuming, or a null
// session when there is none.
//
// Only lobbies whose hub is still alive are reported. A hub dissolves its Run loop when its
// last connection leaves but stays registered in the HubStore, so a dissolved hub would accept
// the client's reconnect and then never answer it: offering that as a resume target is worse
// than offering nothing.
//
// Lock discipline: the store copy is taken first (releasing the store lock), each lobby is read
// under its own lock, and the game lock is taken only after the lobby lock is released. endGame
// holds the game lock while reaching for the lobby and game stores through OnGameEnd, so no
// path here may hold either while acquiring a game lock.
func ActiveSessionHandler(gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		userID, ok := authenticateAndGetUser(w, r)
		if !ok {
			return
		}

		var best *ActiveSession
		for lobbyID, lob := range gs.LobbyStore.GetLobbies() {
			h, hasHub := gs.HubStore.GetHub(lobbyID)
			if !hasHub || !h.Alive() {
				continue
			}

			lob.Mu.Lock()
			joined := lob.Users[userID]
			candidate := ActiveSession{
				LobbyID:     lobbyID.String(),
				LobbyType:   lob.Type,
				GameMode:    lob.GameMode,
				Name:        lob.Name,
				Phase:       "open",
				PlayerCount: lob.JoinedCount(),
			}
			searching := lob.Searching
			inGame := lob.InGame
			gameID := lob.GameID
			lob.Mu.Unlock()

			if !joined {
				continue // invited but never joined, or another user's lobby
			}

			switch {
			case inGame && gameID != uuid.Nil:
				// GetGame takes the game store lock only; SessionSummaryFor takes the game
				// lock, both after the lobby lock above was released.
				if g, exists := gs.GameStore.GetGame(gameID); exists {
					if summary := g.SessionSummaryFor(userID); !summary.GameOver {
						candidate.Phase = "in_game"
						candidate.GameID = summary.GameID.String()
						candidate.Seated = summary.Seated
						candidate.PlayerCount = summary.PlayerCount
					}
				}
			case searching:
				candidate.Phase = "searching"
			}

			if best == nil || sessionRank(candidate.Phase) > sessionRank(best.Phase) ||
				(sessionRank(candidate.Phase) == sessionRank(best.Phase) && candidate.LobbyID < best.LobbyID) {
				// The lobby-id tie-break keeps the answer stable across calls: the store copy
				// is a map and its iteration order is not.
				chosen := candidate
				best = &chosen
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ActiveSessionResponse{Active: best})
	}
}
