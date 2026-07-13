// internal/handlers/leaderboard_test.go
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
	_ "github.com/joho/godotenv/autoload" // Load .env for database connection.
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// insertRatingRow inserts a ratings row with a NULL game_id, for tests that need a
// user to have a rated-games count in a pool without standing up a games row.
func insertRatingRow(t *testing.T, ctx context.Context, userID uuid.UUID, oldRating, newRating int, mode string) {
	t.Helper()
	_, err := database.DB.Exec(ctx, `
		INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode)
		VALUES ($1, NULL, $2, $3, $4)
	`, userID, oldRating, newRating, mode)
	require.NoError(t, err)
}

// TestLeaderboardInvalidPool checks that an unknown pool query param is rejected
// with 400 before any database access, so it needs no live Postgres instance.
func TestLeaderboardInvalidPool(t *testing.T) {
	auth.Init()

	uID := uuid.New()
	token, err := auth.CreateJWT(uID.String())
	require.NoError(t, err)

	req := httptest.NewRequest("GET", "/leaderboard?pool=not-a-pool", nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()

	LeaderboardHandler(w, req)
	require.Equal(t, http.StatusBadRequest, w.Code, "expected 400 for unknown pool: %s", w.Body.String())
}

// TestLeaderboardRankingAndYou is an integration test covering pool-scoped ranking,
// the "you" row (present with true global rank even outside the top N), and the
// unrated (no rated games recorded) -> "you": null case.
func TestLeaderboardRankingAndYou(t *testing.T) {
	setupFriendTest(t) // dbAvailable gate + auth.Init() + database.ConnectDB(), shared with friend tests.

	ctx := context.Background()

	// Three rated (1v1) users with descending elo, one unrated user with no rows
	// in the ratings table for this pool.
	top := createTestUser(t, "lb-top@example.com", "pw", "lb-top")
	mid := createTestUser(t, "lb-mid@example.com", "pw", "lb-mid")
	low := createTestUser(t, "lb-low@example.com", "pw", "lb-low")
	unrated := createTestUser(t, "lb-unrated@example.com", "pw", "lb-unrated")

	require.NoError(t, database.UpdateUser1v1Rating(ctx, top.ID, 2000))
	require.NoError(t, database.UpdateUser1v1Rating(ctx, mid.ID, 1700))
	require.NoError(t, database.UpdateUser1v1Rating(ctx, low.ID, 1400))

	// ratings.game_id is nullable and FK-constrained to games(id); insert directly
	// with a NULL game_id rather than fabricating a games row this test doesn't need.
	insertRatingRow(t, ctx, top.ID, 1500, 2000, "1v1")
	insertRatingRow(t, ctx, mid.ID, 1500, 1700, "1v1")
	insertRatingRow(t, ctx, low.ID, 1500, 1400, "1v1")

	// low, limit=1: low is outside the top 1 (top has the highest rating) but "you"
	// must still report low's true rank and a nonzero games count.
	lowToken, err := auth.CreateJWT(low.ID.String())
	require.NoError(t, err)
	req := httptest.NewRequest("GET", "/leaderboard?pool=1v1&limit=1", nil)
	req.Header.Set("Cookie", "auth_token="+lowToken)
	w := httptest.NewRecorder()
	LeaderboardHandler(w, req)
	require.Equal(t, http.StatusOK, w.Code, "leaderboard request failed: %s", w.Body.String())

	var resp LeaderboardResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, "1v1", resp.Pool)
	require.Len(t, resp.Rows, 1, "limit=1 should return exactly one row")
	assert.Equal(t, 1, resp.Rows[0].Rank)
	assert.Equal(t, top.ID, resp.Rows[0].UserID, "top-rated user should rank first")

	require.NotNil(t, resp.You, "low's 'you' row should be present even outside the top N")
	assert.Equal(t, low.ID, resp.You.UserID)
	assert.GreaterOrEqual(t, resp.You.Rank, 2, "low should rank below the single top-1 row")
	assert.GreaterOrEqual(t, resp.You.Games, 1)

	// Unrated caller (no rows in ratings for this pool) -> "you" is null.
	unratedToken, err := auth.CreateJWT(unrated.ID.String())
	require.NoError(t, err)
	req2 := httptest.NewRequest("GET", "/leaderboard?pool=1v1&limit=1", nil)
	req2.Header.Set("Cookie", "auth_token="+unratedToken)
	w2 := httptest.NewRecorder()
	LeaderboardHandler(w2, req2)
	require.Equal(t, http.StatusOK, w2.Code, "leaderboard request failed: %s", w2.Body.String())

	var resp2 LeaderboardResponse
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &resp2))
	assert.Nil(t, resp2.You, "unrated caller should get a null 'you' row")
}
