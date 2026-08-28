// internal/handlers/history.go
package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

const (
	historyDefaultLimit = 20
	historyMaxLimit     = 100
)

// HistoryOpponentResponse is one other seat at the table in a historical game.
type HistoryOpponentResponse struct {
	UserID   uuid.UUID `json:"userId"`
	Username string    `json:"username"`
	Score    *int      `json:"score"`
	DidWin   *bool     `json:"didWin"`
	Ranking  *int16    `json:"ranking"`
}

// HistoryRatingResponse is the rating change a rated game produced for the caller.
// Delta is New-Old, precomputed so the client never has to reconstruct it.
type HistoryRatingResponse struct {
	Pool  string `json:"pool"`
	Old   int    `json:"old"`
	New   int    `json:"new"`
	Delta int    `json:"delta"`
}

// HistoryGameResponse is one row in a GET /user/history response.
type HistoryGameResponse struct {
	GameID      uuid.UUID                 `json:"gameId"`
	PlayedAt    time.Time                 `json:"playedAt"`
	Status      string                    `json:"status"`
	RoundIndex  int16                     `json:"roundIndex"`
	LobbyType   string                    `json:"lobbyType"`
	Mode        string                    `json:"mode"`
	Rated       bool                      `json:"rated"`
	PlayerCount int                       `json:"playerCount"`
	Score       *int                      `json:"score"`
	DidWin      *bool                     `json:"didWin"`
	Ranking     *int16                    `json:"ranking"`
	Rating      *HistoryRatingResponse    `json:"rating"`
	Opponents   []HistoryOpponentResponse `json:"opponents"`
}

// HistoryResponse is the JSON body for GET /user/history.
type HistoryResponse struct {
	Games  []HistoryGameResponse `json:"games"`
	Limit  int                   `json:"limit"`
	Offset int                   `json:"offset"`
	Total  int                   `json:"total"`
}

// PoolRatingResponse is the caller's standing in one Glicko-2 rating pool.
type PoolRatingResponse struct {
	Pool       string  `json:"pool"`
	Rating     int     `json:"rating"`
	RD         float64 `json:"rd"`
	Volatility float64 `json:"volatility"`
	Games      int     `json:"games"`
	Wins       int     `json:"wins"`
	Peak       int     `json:"peak"`
}

// RatingSummaryResponse is the JSON body for GET /user/ratings.
type RatingSummaryResponse struct {
	Pools     []PoolRatingResponse `json:"pools"`
	OpenSkill struct {
		Mu    float64 `json:"mu"`
		Sigma float64 `json:"sigma"`
	} `json:"openSkill"`
	Record struct {
		Games int `json:"games"`
		Wins  int `json:"wins"`
	} `json:"record"`
}

// GameHistoryHandler handles GET /user/history?limit=<n>&offset=<m>. Returns the
// authenticated caller's own recent games, newest first, with each game's opponents, the
// caller's score/outcome, the lobby context it was played under, and the rating change it
// produced when the game was rated. Never exposes another user's history: the page is
// keyed on the caller's id from their token, not on any request parameter.
//
// limit defaults to 20 and is capped at 100; offset defaults to 0. total reports how many
// games the caller has in all, so a client can page without a second request. A caller
// with no games gets 200 with an empty games array rather than a 404.
func GameHistoryHandler(w http.ResponseWriter, r *http.Request) {
	userID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return
	}

	limit := historyDefaultLimit
	if raw := r.URL.Query().Get("limit"); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n <= 0 {
			http.Error(w, "Invalid limit (expected a positive integer)", http.StatusBadRequest)
			return
		}
		limit = n
	}
	if limit > historyMaxLimit {
		limit = historyMaxLimit
	}

	offset := 0
	if raw := r.URL.Query().Get("offset"); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 0 {
			http.Error(w, "Invalid offset (expected a non-negative integer)", http.StatusBadRequest)
			return
		}
		offset = n
	}

	entries, total, err := database.GetUserGameHistory(r.Context(), userID, limit, offset)
	if err != nil {
		log.Printf("Failed to fetch game history for user %s: %v", userID, err)
		http.Error(w, "Failed to retrieve game history", http.StatusInternalServerError)
		return
	}

	resp := HistoryResponse{
		Games:  make([]HistoryGameResponse, 0, len(entries)),
		Limit:  limit,
		Offset: offset,
		Total:  total,
	}
	for _, e := range entries {
		g := HistoryGameResponse{
			GameID:      e.GameID,
			PlayedAt:    e.PlayedAt,
			Status:      e.Status,
			RoundIndex:  e.RoundIndex,
			LobbyType:   e.LobbyType,
			Mode:        e.LobbyMode,
			Rated:       e.Rated,
			PlayerCount: e.PlayerCount,
			Score:       e.Score,
			DidWin:      e.DidWin,
			Ranking:     e.Ranking,
			Opponents:   make([]HistoryOpponentResponse, 0, len(e.Opponents)),
		}
		if e.RatingMode != nil && e.OldRating != nil && e.NewRating != nil {
			g.Rating = &HistoryRatingResponse{
				Pool:  *e.RatingMode,
				Old:   *e.OldRating,
				New:   *e.NewRating,
				Delta: *e.NewRating - *e.OldRating,
			}
		}
		for _, o := range e.Opponents {
			g.Opponents = append(g.Opponents, HistoryOpponentResponse{
				UserID:   o.UserID,
				Username: o.Username,
				Score:    o.Score,
				DidWin:   o.DidWin,
				Ranking:  o.Ranking,
			})
		}
		resp.Games = append(resp.Games, g)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("Failed to write /user/history response for user %s: %v", userID, err)
	}
}

// RatingSummaryHandler handles GET /user/ratings. Returns the authenticated caller's
// current rating, deviation and volatility in every Glicko-2 pool, with the rated-game
// count, win count and peak rating each pool has behind it, plus the OpenSkill pair used
// for circuit play and the caller's lifetime recorded win/loss totals.
//
// Every pool is always present, at its baseline values for a player who has never played
// it, so the client renders a stable set of rows instead of a shifting one.
func RatingSummaryHandler(w http.ResponseWriter, r *http.Request) {
	userID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return
	}

	summary, err := database.GetUserRatingSummary(r.Context(), userID)
	if err != nil {
		log.Printf("Failed to fetch rating summary for user %s: %v", userID, err)
		http.Error(w, "Failed to retrieve rating summary", http.StatusInternalServerError)
		return
	}

	resp := RatingSummaryResponse{Pools: make([]PoolRatingResponse, 0, len(summary.Pools))}
	for _, p := range summary.Pools {
		resp.Pools = append(resp.Pools, PoolRatingResponse{
			Pool:       p.Pool,
			Rating:     p.Rating,
			RD:         p.RD,
			Volatility: p.Volatility,
			Games:      p.Games,
			Wins:       p.Wins,
			Peak:       p.Peak,
		})
	}
	resp.OpenSkill.Mu = summary.OpenSkillMu
	resp.OpenSkill.Sigma = summary.OpenSkillSigma
	resp.Record.Games = summary.TotalGames
	resp.Record.Wins = summary.TotalWins

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("Failed to write /user/ratings response for user %s: %v", userID, err)
	}
}
