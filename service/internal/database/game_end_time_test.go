// internal/database/game_end_time_test.go
//
// games.end_time and the boot sweep that closes out a game whose process went away
// (cambia-1904). Before this, end_time was written only by the historian, keyed on an action
// name nothing emitted, so it was NULL for every game ever played.
package database

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/require"
)

// readGameTimes returns a game's status, start_time and end_time.
func readGameTimes(t *testing.T, gameID uuid.UUID) (status string, startTime, endTime *time.Time) {
	t.Helper()

	require.NoError(t, DB.QueryRow(context.Background(),
		`SELECT status, start_time, end_time FROM games WHERE id = $1`, gameID,
	).Scan(&status, &startTime, &endTime), "read the games row")
	return status, startTime, endTime
}

// A finished game carries when it finished. RecordGameAndResults stamps end_time in the same
// transaction that marks the game completed, so a row can never be completed without one.
func TestRecordGameAndResultsStampsEndTime(t *testing.T) {
	setupGameTest(t)

	winner := createGameTestUser(t, "endtime-winner-"+uuid.NewString())
	loser := createGameTestUser(t, "endtime-loser-"+uuid.NewString())

	gameID := uuid.New()
	seedGameRow(t, gameID, winner.ID)

	_, startBefore, endBefore := readGameTimes(t, gameID)
	require.NotNil(t, startBefore, "seeded game should carry a start_time")
	require.Nil(t, endBefore, "a game in progress has not ended")

	players := []*models.Player{{ID: winner.ID}, {ID: loser.ID}}
	scores := map[uuid.UUID]int{winner.ID: 0, loser.ID: 10}
	require.NoError(t, RecordGameAndResults(
		context.Background(), gameID, players, scores, []uuid.UUID{winner.ID}, false))

	status, startAfter, endAfter := readGameTimes(t, gameID)
	require.Equal(t, "completed", status)
	require.NotNil(t, endAfter, "a completed game must carry an end_time")
	require.False(t, endAfter.Before(*startAfter),
		"end_time %v precedes start_time %v", endAfter, startAfter)
}

// The completion UPDATE is conditioned on the game not already being completed, which is what
// makes a duplicate end-event a no-op. end_time inherits that guard: it records when the game
// ended, not when the last stray end-event arrived.
func TestRecordGameAndResultsKeepsTheFirstEndTime(t *testing.T) {
	setupGameTest(t)

	winner := createGameTestUser(t, "endtime-dup-winner-"+uuid.NewString())
	loser := createGameTestUser(t, "endtime-dup-loser-"+uuid.NewString())

	gameID := uuid.New()
	seedGameRow(t, gameID, winner.ID)

	players := []*models.Player{{ID: winner.ID}, {ID: loser.ID}}
	scores := map[uuid.UUID]int{winner.ID: 0, loser.ID: 10}
	winners := []uuid.UUID{winner.ID}

	require.NoError(t, RecordGameAndResults(context.Background(), gameID, players, scores, winners, false))
	_, _, first := readGameTimes(t, gameID)
	require.NotNil(t, first)

	// Far enough apart that a second stamp could not read as the same instant.
	time.Sleep(10 * time.Millisecond)

	require.NoError(t, RecordGameAndResults(context.Background(), gameID, players, scores, winners, false))
	_, _, second := readGameTimes(t, gameID)
	require.NotNil(t, second)
	require.True(t, first.Equal(*second),
		"a duplicate end-event moved end_time from %v to %v", first, second)
}

// The boot sweep closes out games a previous process left behind. It runs inside a transaction
// this test rolls back: the sweep rewrites every in-progress row in the table, and the dev
// Postgres is shared by every checkout on the machine, so committing it would reach into other
// packages' fixtures and other people's runs.
func TestAbandonStaleGamesClosesOutGamesLeftInProgress(t *testing.T) {
	setupGameTest(t)

	host := createGameTestUser(t, "stale-host-"+uuid.NewString())
	other := createGameTestUser(t, "stale-other-"+uuid.NewString())

	// One game still in progress, and one already finished that the sweep must not touch.
	staleID := uuid.New()
	seedGameRow(t, staleID, host.ID)

	finishedID := uuid.New()
	seedGameRow(t, finishedID, host.ID)
	players := []*models.Player{{ID: host.ID}, {ID: other.ID}}
	scores := map[uuid.UUID]int{host.ID: 0, other.ID: 10}
	require.NoError(t, RecordGameAndResults(
		context.Background(), finishedID, players, scores, []uuid.UUID{host.ID}, false))
	_, _, finishedEnd := readGameTimes(t, finishedID)
	require.NotNil(t, finishedEnd)

	ctx := context.Background()
	tx, err := DB.Begin(ctx)
	require.NoError(t, err, "open the sweep transaction")
	defer func() {
		require.NoError(t, tx.Rollback(ctx), "roll the sweep back")
	}()

	closed, err := AbandonStaleGames(ctx, tx)
	require.NoError(t, err)
	require.GreaterOrEqual(t, closed, int64(1), "the sweep should have closed at least this test's stale game")

	// Read through the transaction: outside it, the rollback means nothing happened.
	var staleStatus string
	var staleEnd *time.Time
	require.NoError(t, tx.QueryRow(ctx,
		`SELECT status, end_time FROM games WHERE id = $1`, staleID,
	).Scan(&staleStatus, &staleEnd))
	require.Equal(t, "abandoned", staleStatus, "a game left in progress should be abandoned")
	require.NotNil(t, staleEnd, "an abandoned game must carry an end_time")

	var finishedStatus string
	var finishedEndAfter *time.Time
	require.NoError(t, tx.QueryRow(ctx,
		`SELECT status, end_time FROM games WHERE id = $1`, finishedID,
	).Scan(&finishedStatus, &finishedEndAfter))
	require.Equal(t, "completed", finishedStatus, "the sweep must not touch a finished game")
	require.NotNil(t, finishedEndAfter)
	require.True(t, finishedEnd.Equal(*finishedEndAfter),
		"the sweep moved a finished game's end_time from %v to %v", finishedEnd, finishedEndAfter)
}

// The sweep takes a pool in production and a transaction in tests, so both must satisfy its
// parameter. A compile-time check rather than a runtime one: the point is the signature.
var (
	_ gamesExecer = (pgx.Tx)(nil)
	_ gamesExecer = DB
)
