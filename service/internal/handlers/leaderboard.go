// internal/handlers/leaderboard.go
package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

const (
	leaderboardDefaultLimit = 50
	leaderboardMaxLimit     = 100
)

// validLeaderboardPools mirrors the rating pools tracked per user (rating.Mode1v1,
// rating.Mode4p, rating.Mode7p8p): the elo_*/phi_* column pairs on the users table.
var validLeaderboardPools = map[string]bool{
	"1v1":  true,
	"4p":   true,
	"7p8p": true,
}

// LeaderboardRowResponse is a single ranked row in a /leaderboard response, and also
// the shape of the "you" field for the authenticated caller.
type LeaderboardRowResponse struct {
	Rank     int       `json:"rank"`
	UserID   uuid.UUID `json:"userId"`
	Username string    `json:"username"`
	Rating   int       `json:"rating"`
	RD       float64   `json:"rd"`
	Games    int       `json:"games"`
}

// LeaderboardResponse is the JSON body for GET /leaderboard.
type LeaderboardResponse struct {
	Pool string                   `json:"pool"`
	Rows []LeaderboardRowResponse `json:"rows"`
	You  *LeaderboardRowResponse  `json:"you"`
}

// LeaderboardHandler handles GET /leaderboard?pool=<pool>&limit=<n>. Returns the top
// `limit` users by rating in the given pool (default 50, capped at 100) plus the
// authenticated caller's own row: present with its true global rank even when
// outside the top N, null when the caller has no rated games in this pool.
func LeaderboardHandler(w http.ResponseWriter, r *http.Request) {
	userID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return
	}

	pool := r.URL.Query().Get("pool")
	if !validLeaderboardPools[pool] {
		http.Error(w, "Invalid or missing pool (expected one of: 1v1, 4p, 7p8p)", http.StatusBadRequest)
		return
	}

	limit := leaderboardDefaultLimit
	if raw := r.URL.Query().Get("limit"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			limit = n
		}
	}
	if limit > leaderboardMaxLimit {
		limit = leaderboardMaxLimit
	}

	rows, you, err := database.GetLeaderboard(r.Context(), pool, limit, userID)
	if err != nil {
		log.Printf("Failed to fetch leaderboard for pool %s: %v", pool, err)
		http.Error(w, "Failed to retrieve leaderboard", http.StatusInternalServerError)
		return
	}

	resp := LeaderboardResponse{Pool: pool, Rows: make([]LeaderboardRowResponse, 0, len(rows))}
	for _, row := range rows {
		resp.Rows = append(resp.Rows, LeaderboardRowResponse{
			Rank:     row.Rank,
			UserID:   row.UserID,
			Username: row.Username,
			Rating:   row.Rating,
			RD:       row.RD,
			Games:    row.Games,
		})
	}
	if you != nil {
		resp.You = &LeaderboardRowResponse{
			Rank:     you.Rank,
			UserID:   you.UserID,
			Username: you.Username,
			Rating:   you.Rating,
			RD:       you.RD,
			Games:    you.Games,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
