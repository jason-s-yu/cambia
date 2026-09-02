// internal/handlers/history_test.go
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
	"github.com/jason-s-yu/cambia/service/internal/models"
	_ "github.com/joho/godotenv/autoload" // Load .env for database connection.
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// getAs issues an authenticated GET against handler as userID and returns the recorder.
func getAs(t *testing.T, handler http.HandlerFunc, target string, userID uuid.UUID) *httptest.ResponseRecorder {
	t.Helper()
	token, err := auth.CreateJWT(userID.String())
	require.NoError(t, err)
	req := httptest.NewRequest("GET", target, nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	handler(w, req)
	return w
}

// seedHistoryGame inserts a lobby + games row and records a finished 1v1 game between the
// two users, mirroring what the production game-end path writes.
func seedHistoryGame(t *testing.T, winner, loser models.User, rated bool) uuid.UUID {
	t.Helper()
	ctx := context.Background()
	gameID := uuid.New()

	var lobbyID uuid.UUID
	require.NoError(t, database.DB.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type, mode, ranked) VALUES ($1, 'private', $2, $3) RETURNING id`,
		winner.ID, map[bool]string{true: "ranked", false: "casual"}[rated], rated,
	).Scan(&lobbyID))
	_, err := database.DB.Exec(ctx,
		`INSERT INTO games (id, lobby_id, status) VALUES ($1, $2, 'in_progress')`, gameID, lobbyID)
	require.NoError(t, err)

	require.NoError(t, database.RecordGameAndResults(ctx, gameID,
		[]*models.Player{{ID: winner.ID}, {ID: loser.ID}},
		map[uuid.UUID]int{winner.ID: 4, loser.ID: 15},
		[]uuid.UUID{winner.ID}, rated))
	return gameID
}

// seedHistoryGameWithRound is seedHistoryGame plus an explicit games.round_index, so a test
// can assert a circuit round survives the read path (cambia-1240) without disturbing every
// other seedHistoryGame caller's signature.
func seedHistoryGameWithRound(t *testing.T, winner, loser models.User, rated bool, roundIndex int16) uuid.UUID {
	t.Helper()
	ctx := context.Background()
	gameID := uuid.New()

	var lobbyID uuid.UUID
	require.NoError(t, database.DB.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type, mode, ranked) VALUES ($1, 'private', $2, $3) RETURNING id`,
		winner.ID, map[bool]string{true: "ranked", false: "casual"}[rated], rated,
	).Scan(&lobbyID))
	_, err := database.DB.Exec(ctx,
		`INSERT INTO games (id, lobby_id, round_index, status) VALUES ($1, $2, $3, 'in_progress')`,
		gameID, lobbyID, roundIndex)
	require.NoError(t, err)

	require.NoError(t, database.RecordGameAndResults(ctx, gameID,
		[]*models.Player{{ID: winner.ID}, {ID: loser.ID}},
		map[uuid.UUID]int{winner.ID: 4, loser.ID: 15},
		[]uuid.UUID{winner.ID}, rated))
	return gameID
}

// TestGameHistoryReportsRoundIndex asserts a persisted non-zero games.round_index (as a
// circuit round writer would leave it) round-trips through GET /user/history unchanged,
// and that a game with no round (the default 0) reports 0 rather than a leftover value from
// another game (cambia-1240).
func TestGameHistoryReportsRoundIndex(t *testing.T) {
	setupFriendTest(t)

	me := createTestUser(t, "hist-round-"+uuid.NewString()+"@example.com", "pw", "hist-round")
	them := createTestUser(t, "hist-round-opp-"+uuid.NewString()+"@example.com", "pw", "hist-round-opp")

	roundGame := seedHistoryGameWithRound(t, me, them, true, 3)
	noRoundGame := seedHistoryGame(t, me, them, false)

	w := getAs(t, GameHistoryHandler, "/user/history?limit=10", me.ID)
	require.Equal(t, http.StatusOK, w.Code, "history request failed: %s", w.Body.String())

	var resp HistoryResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Equal(t, 2, resp.Total)

	byID := map[uuid.UUID]HistoryGameResponse{}
	for _, g := range resp.Games {
		byID[g.GameID] = g
	}

	rg, ok := byID[roundGame]
	require.True(t, ok, "the circuit-round game should be in the caller's history")
	assert.Equal(t, int16(3), rg.RoundIndex, "a persisted circuit round must round-trip through the handler, not read back as 0")

	nrg, ok := byID[noRoundGame]
	require.True(t, ok, "the non-circuit game should be in the caller's history")
	assert.Equal(t, int16(0), nrg.RoundIndex, "a non-circuit game reports round_index 0")
}

// TestGameHistoryRequiresAuth checks the endpoint rejects an unauthenticated caller before
// touching the database, so it needs no live Postgres.
func TestGameHistoryRequiresAuth(t *testing.T) {
	auth.Init()

	req := httptest.NewRequest("GET", "/user/history", nil)
	w := httptest.NewRecorder()
	GameHistoryHandler(w, req)
	assert.NotEqual(t, http.StatusOK, w.Code, "an unauthenticated history request must not succeed")

	req2 := httptest.NewRequest("GET", "/user/ratings", nil)
	w2 := httptest.NewRecorder()
	RatingSummaryHandler(w2, req2)
	assert.NotEqual(t, http.StatusOK, w2.Code, "an unauthenticated ratings request must not succeed")
}

// TestGameHistoryRejectsBadPagination checks limit/offset validation happens before any
// database access.
func TestGameHistoryRejectsBadPagination(t *testing.T) {
	auth.Init()
	uID := uuid.New()

	for _, target := range []string{
		"/user/history?limit=0",
		"/user/history?limit=-3",
		"/user/history?limit=abc",
		"/user/history?offset=-1",
		"/user/history?offset=xyz",
	} {
		w := getAs(t, GameHistoryHandler, target, uID)
		assert.Equal(t, http.StatusBadRequest, w.Code, "expected 400 for %s: %s", target, w.Body.String())
	}
}

// TestGameHistoryEmptyForNewUser covers the empty state a brand-new account sees: 200 with
// an empty (non-null) games array, not a 404.
func TestGameHistoryEmptyForNewUser(t *testing.T) {
	setupFriendTest(t)

	fresh := createTestUser(t, "hist-new-"+uuid.NewString()+"@example.com", "pw", "hist-new")

	w := getAs(t, GameHistoryHandler, "/user/history", fresh.ID)
	require.Equal(t, http.StatusOK, w.Code, "history request failed: %s", w.Body.String())

	var resp HistoryResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, 0, resp.Total)
	assert.Equal(t, historyDefaultLimit, resp.Limit)
	assert.Equal(t, 0, resp.Offset)
	require.NotNil(t, resp.Games, "games must serialize as [] so the client never has to null-check")
	assert.Empty(t, resp.Games)
}

// TestGameHistoryReturnsOwnGames is the end-to-end read: a rated game and a casual game
// come back with the caller's own outcome, the opponent, the lobby mode, and rating
// context only on the rated one. Also asserts a third user's history stays their own.
func TestGameHistoryReturnsOwnGames(t *testing.T) {
	setupFriendTest(t)

	me := createTestUser(t, "hist-me-"+uuid.NewString()+"@example.com", "pw", "hist-me")
	them := createTestUser(t, "hist-them-"+uuid.NewString()+"@example.com", "pw", "hist-them")
	stranger := createTestUser(t, "hist-str-"+uuid.NewString()+"@example.com", "pw", "hist-str")

	seedHistoryGame(t, me, them, false)
	ratedGame := seedHistoryGame(t, me, them, true)

	w := getAs(t, GameHistoryHandler, "/user/history?limit=10", me.ID)
	require.Equal(t, http.StatusOK, w.Code, "history request failed: %s", w.Body.String())

	var resp HistoryResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Equal(t, 2, resp.Total)
	require.Len(t, resp.Games, 2)

	byID := map[uuid.UUID]HistoryGameResponse{}
	for _, g := range resp.Games {
		byID[g.GameID] = g
	}

	rated, ok := byID[ratedGame]
	require.True(t, ok, "the rated game should be in the caller's history")
	assert.True(t, rated.Rated, "lobbies.ranked should surface as rated")
	assert.Equal(t, "ranked", rated.Mode)
	assert.Equal(t, "private", rated.LobbyType)
	assert.Equal(t, "completed", rated.Status)
	assert.Equal(t, 2, rated.PlayerCount)
	require.NotNil(t, rated.Score)
	assert.Equal(t, 4, *rated.Score)
	require.NotNil(t, rated.DidWin)
	assert.True(t, *rated.DidWin)
	require.Len(t, rated.Opponents, 1)
	assert.Equal(t, them.ID, rated.Opponents[0].UserID)
	assert.Equal(t, them.Username, rated.Opponents[0].Username)
	require.NotNil(t, rated.Rating, "a rated game should report its rating change")
	assert.Equal(t, "1v1", rated.Rating.Pool)
	assert.Equal(t, rated.Rating.New-rated.Rating.Old, rated.Rating.Delta)
	assert.Positive(t, rated.Rating.Delta, "the winner's rating should have gone up")

	var casual HistoryGameResponse
	for id, g := range byID {
		if id != ratedGame {
			casual = g
		}
	}
	assert.False(t, casual.Rated)
	assert.Equal(t, "casual", casual.Mode)
	assert.Nil(t, casual.Rating, "an unrated game carries no rating change")

	// A user who played none of these games sees none of them.
	w2 := getAs(t, GameHistoryHandler, "/user/history", stranger.ID)
	require.Equal(t, http.StatusOK, w2.Code, "history request failed: %s", w2.Body.String())
	var resp2 HistoryResponse
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &resp2))
	assert.Equal(t, 0, resp2.Total, "history is scoped to the authenticated caller")
}

// TestGameHistoryLimitIsCapped checks an oversized limit clamps to the maximum rather than
// erroring or honoring an unbounded page size.
func TestGameHistoryLimitIsCapped(t *testing.T) {
	setupFriendTest(t)

	fresh := createTestUser(t, "hist-cap-"+uuid.NewString()+"@example.com", "pw", "hist-cap")

	w := getAs(t, GameHistoryHandler, "/user/history?limit=100000", fresh.ID)
	require.Equal(t, http.StatusOK, w.Code, "history request failed: %s", w.Body.String())

	var resp HistoryResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	assert.Equal(t, historyMaxLimit, resp.Limit)
}

// TestRatingSummaryReportsEveryPool covers both the fresh-account empty state and the
// after-play state through the HTTP surface.
func TestRatingSummaryReportsEveryPool(t *testing.T) {
	setupFriendTest(t)

	fresh := createTestUser(t, "rate-new-"+uuid.NewString()+"@example.com", "pw", "rate-new")

	w := getAs(t, RatingSummaryHandler, "/user/ratings", fresh.ID)
	require.Equal(t, http.StatusOK, w.Code, "ratings request failed: %s", w.Body.String())

	var resp RatingSummaryResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Len(t, resp.Pools, 3, "every pool is reported for a player who has never played")
	for _, p := range resp.Pools {
		assert.Equal(t, 1500, p.Rating, "pool %s should sit at the baseline", p.Pool)
		assert.Equal(t, 0, p.Games)
	}
	assert.Equal(t, 0, resp.Record.Games)
	assert.Equal(t, 0, resp.Record.Wins)
	assert.InDelta(t, 25.0, resp.OpenSkill.Mu, 1e-9)

	// After a rated win the 1v1 pool moves and the untouched pools do not.
	them := createTestUser(t, "rate-opp-"+uuid.NewString()+"@example.com", "pw", "rate-opp")
	seedHistoryGame(t, fresh, them, true)

	w2 := getAs(t, RatingSummaryHandler, "/user/ratings", fresh.ID)
	require.Equal(t, http.StatusOK, w2.Code, "ratings request failed: %s", w2.Body.String())

	var resp2 RatingSummaryResponse
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &resp2))
	pools := map[string]PoolRatingResponse{}
	for _, p := range resp2.Pools {
		pools[p.Pool] = p
	}
	assert.Equal(t, 1, pools["1v1"].Games)
	assert.Equal(t, 1, pools["1v1"].Wins)
	assert.Greater(t, pools["1v1"].Rating, 1500)
	assert.Less(t, pools["1v1"].RD, 350.0)
	assert.GreaterOrEqual(t, pools["1v1"].Peak, pools["1v1"].Rating)
	assert.Equal(t, 1500, pools["4p"].Rating, "a 1v1 game must not move the 4p pool")
	assert.Equal(t, 0, pools["7p8p"].Games)
	assert.Equal(t, 1, resp2.Record.Games)
	assert.Equal(t, 1, resp2.Record.Wins)
}
