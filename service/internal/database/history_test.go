// internal/database/history_test.go
package database

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	_ "github.com/joho/godotenv/autoload" // Load .env for database connection.
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGetUserGameHistoryEmpty covers the new-user case: a user with no recorded results
// gets an empty page and a zero total, not an error.
func TestGetUserGameHistoryEmpty(t *testing.T) {
	setupGameTest(t)

	fresh := createGameTestUser(t, "hist-empty-"+uuid.NewString())

	entries, total, err := GetUserGameHistory(context.Background(), fresh.ID, 20, 0)
	require.NoError(t, err)
	assert.Equal(t, 0, total, "a user who has never played should report zero games")
	assert.Empty(t, entries)
}

// TestGetUserGameHistoryOrderingAndOpponents plays three rated 1v1 games between the same
// two players and asserts the caller's page carries newest-first ordering, the caller's
// own score/outcome, the opponent (and only the opponent) in Opponents, the lobby context,
// and the per-game rating delta recorded in the ratings table.
func TestGetUserGameHistoryOrderingAndOpponents(t *testing.T) {
	setupGameTest(t)

	ctx := context.Background()
	winner := createGameTestUser(t, "hist-w-"+uuid.NewString())
	loser := createGameTestUser(t, "hist-l-"+uuid.NewString())

	players := []*models.Player{{ID: winner.ID}, {ID: loser.ID}}
	scores := map[uuid.UUID]int{winner.ID: 3, loser.ID: 17}
	winners := []uuid.UUID{winner.ID}

	var played []uuid.UUID
	for i := 0; i < 3; i++ {
		gameID := uuid.New()
		seedGameRow(t, gameID, winner.ID)
		require.NoError(t, RecordGameAndResults(ctx, gameID, players, scores, winners, true))
		played = append(played, gameID)
	}

	entries, total, err := GetUserGameHistory(ctx, winner.ID, 20, 0)
	require.NoError(t, err)
	require.Equal(t, 3, total)
	require.Len(t, entries, 3)

	// Newest first. The three games share a wall-clock second, so assert on the set and on
	// the ordering key being non-increasing rather than on a fixed permutation.
	seen := map[uuid.UUID]bool{}
	for i, e := range entries {
		seen[e.GameID] = true
		if i > 0 {
			assert.False(t, entries[i-1].PlayedAt.Before(e.PlayedAt), "history must be ordered newest first")
		}
	}
	for _, gameID := range played {
		assert.True(t, seen[gameID], "game %v missing from the caller's history", gameID)
	}

	e := entries[0]
	assert.Equal(t, "completed", e.Status)
	assert.Equal(t, 2, e.PlayerCount)
	require.NotNil(t, e.Score)
	assert.Equal(t, 3, *e.Score, "the caller's own score, not the opponent's")
	require.NotNil(t, e.DidWin)
	assert.True(t, *e.DidWin)
	assert.Equal(t, "private", e.LobbyType, "lobby context should come through the games -> lobbies join")

	require.Len(t, e.Opponents, 1, "a 1v1 game has exactly one opponent and never lists the caller")
	assert.Equal(t, loser.ID, e.Opponents[0].UserID)
	assert.Equal(t, loser.Username, e.Opponents[0].Username)
	require.NotNil(t, e.Opponents[0].Score)
	assert.Equal(t, 17, *e.Opponents[0].Score)
	require.NotNil(t, e.Opponents[0].DidWin)
	assert.False(t, *e.Opponents[0].DidWin)

	require.NotNil(t, e.RatingMode, "a rated game should carry its rating row")
	assert.Equal(t, "1v1", *e.RatingMode)
	require.NotNil(t, e.OldRating)
	require.NotNil(t, e.NewRating)
	assert.Greater(t, *e.NewRating, *e.OldRating, "the winner's rating should have gone up")

	// The loser sees the same games from the other side.
	loserEntries, loserTotal, err := GetUserGameHistory(ctx, loser.ID, 20, 0)
	require.NoError(t, err)
	assert.Equal(t, 3, loserTotal)
	require.Len(t, loserEntries, 3)
	require.NotNil(t, loserEntries[0].DidWin)
	assert.False(t, *loserEntries[0].DidWin)
	require.Len(t, loserEntries[0].Opponents, 1)
	assert.Equal(t, winner.ID, loserEntries[0].Opponents[0].UserID)
}

// TestGetUserGameHistoryPagination checks that limit/offset walk the same result set
// without repeating or dropping a game, and that total stays the full count.
func TestGetUserGameHistoryPagination(t *testing.T) {
	setupGameTest(t)

	ctx := context.Background()
	a := createGameTestUser(t, "hist-page-a-"+uuid.NewString())
	b := createGameTestUser(t, "hist-page-b-"+uuid.NewString())

	players := []*models.Player{{ID: a.ID}, {ID: b.ID}}
	scores := map[uuid.UUID]int{a.ID: 5, b.ID: 9}
	winners := []uuid.UUID{a.ID}

	const numGames = 5
	for i := 0; i < numGames; i++ {
		gameID := uuid.New()
		seedGameRow(t, gameID, a.ID)
		require.NoError(t, RecordGameAndResults(ctx, gameID, players, scores, winners, true))
	}

	collected := map[uuid.UUID]bool{}
	for offset := 0; offset < numGames; offset += 2 {
		page, total, err := GetUserGameHistory(ctx, a.ID, 2, offset)
		require.NoError(t, err)
		assert.Equal(t, numGames, total, "total is the full count, independent of the page")
		for _, e := range page {
			assert.False(t, collected[e.GameID], "game %v appeared on two pages", e.GameID)
			collected[e.GameID] = true
		}
	}
	assert.Len(t, collected, numGames, "paging through should visit every game exactly once")

	// An offset past the end is an empty page, not an error.
	page, total, err := GetUserGameHistory(ctx, a.ID, 2, numGames+10)
	require.NoError(t, err)
	assert.Equal(t, numGames, total)
	assert.Empty(t, page)
}

// TestGetUserGameHistoryUnratedGame covers the casual path: game_results rows are still
// written, but no ratings row exists, so the entry carries no rating context.
func TestGetUserGameHistoryUnratedGame(t *testing.T) {
	setupGameTest(t)

	ctx := context.Background()
	a := createGameTestUser(t, "hist-casual-a-"+uuid.NewString())
	b := createGameTestUser(t, "hist-casual-b-"+uuid.NewString())

	gameID := uuid.New()
	seedGameRow(t, gameID, a.ID)
	require.NoError(t, RecordGameAndResults(ctx, gameID,
		[]*models.Player{{ID: a.ID}, {ID: b.ID}},
		map[uuid.UUID]int{a.ID: 2, b.ID: 8},
		[]uuid.UUID{a.ID}, false))

	entries, total, err := GetUserGameHistory(ctx, a.ID, 20, 0)
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, entries, 1)
	assert.Nil(t, entries[0].RatingMode, "an unrated game has no ratings row to report")
	assert.Nil(t, entries[0].OldRating)
	assert.Nil(t, entries[0].NewRating)
	require.NotNil(t, entries[0].Score, "game_results is written for casual games too")
	assert.Equal(t, 2, *entries[0].Score)
}

// TestGetUserRatingSummaryFreshUser checks the zero-games case: every pool is present at
// its schema defaults, with zero history behind it.
func TestGetUserRatingSummaryFreshUser(t *testing.T) {
	setupGameTest(t)

	fresh := createGameTestUser(t, "rating-fresh-"+uuid.NewString())

	summary, err := GetUserRatingSummary(context.Background(), fresh.ID)
	require.NoError(t, err)
	require.Len(t, summary.Pools, 3, "every pool is reported, played or not")

	pools := map[string]PoolRatingSummary{}
	for _, p := range summary.Pools {
		pools[p.Pool] = p
	}
	for _, name := range []string{"1v1", "4p", "7p8p"} {
		p, ok := pools[name]
		require.True(t, ok, "pool %s missing from the summary", name)
		assert.Equal(t, 1500, p.Rating, "pool %s should sit at the baseline rating", name)
		assert.Equal(t, 350.0, p.RD)
		assert.Equal(t, 0, p.Games)
		assert.Equal(t, 0, p.Wins)
		assert.Equal(t, p.Rating, p.Peak, "peak is floored at the current rating")
	}
	assert.Equal(t, 0, summary.TotalGames)
	assert.Equal(t, 0, summary.TotalWins)
	assert.InDelta(t, 25.0, summary.OpenSkillMu, 1e-9)
}

// TestGetUserRatingSummaryAfterPlay asserts the summary tracks a played pool (current
// rating off the users row, counts and peak off the ratings history) and leaves the pools
// the user has not played at their baselines.
func TestGetUserRatingSummaryAfterPlay(t *testing.T) {
	setupGameTest(t)

	ctx := context.Background()
	winner := createGameTestUser(t, "rating-w-"+uuid.NewString())
	loser := createGameTestUser(t, "rating-l-"+uuid.NewString())

	players := []*models.Player{{ID: winner.ID}, {ID: loser.ID}}
	scores := map[uuid.UUID]int{winner.ID: 1, loser.ID: 20}
	winners := []uuid.UUID{winner.ID}

	for i := 0; i < 2; i++ {
		gameID := uuid.New()
		seedGameRow(t, gameID, winner.ID)
		require.NoError(t, RecordGameAndResults(ctx, gameID, players, scores, winners, true))
	}

	summary, err := GetUserRatingSummary(ctx, winner.ID)
	require.NoError(t, err)

	pools := map[string]PoolRatingSummary{}
	for _, p := range summary.Pools {
		pools[p.Pool] = p
	}

	h2h := pools["1v1"]
	assert.Equal(t, 2, h2h.Games, "both rated games should be counted in the 1v1 pool")
	assert.Equal(t, 2, h2h.Wins, "the winner won both")
	assert.Greater(t, h2h.Rating, 1500, "current rating comes off the users row and should have risen")
	assert.Less(t, h2h.RD, 350.0, "rating deviation should have shrunk with play")
	assert.GreaterOrEqual(t, h2h.Peak, h2h.Rating, "peak is never below the current rating")

	assert.Equal(t, 1500, pools["4p"].Rating, "a 1v1 game must not move the 4p pool")
	assert.Equal(t, 0, pools["4p"].Games)

	assert.Equal(t, 2, summary.TotalGames, "lifetime record counts every recorded game")
	assert.Equal(t, 2, summary.TotalWins)

	loserSummary, err := GetUserRatingSummary(ctx, loser.ID)
	require.NoError(t, err)
	loserPools := map[string]PoolRatingSummary{}
	for _, p := range loserSummary.Pools {
		loserPools[p.Pool] = p
	}
	assert.Equal(t, 2, loserPools["1v1"].Games)
	assert.Equal(t, 0, loserPools["1v1"].Wins)
	assert.Less(t, loserPools["1v1"].Rating, 1500)
	assert.GreaterOrEqual(t, loserPools["1v1"].Peak, 1500, "peak keeps the higher rating the loser started from")
}
