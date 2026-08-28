// internal/database/circuit_rating_test.go
package database

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/require"
)

// ratingRowCount returns how many ratings rows exist for a game.
func ratingRowCount(t *testing.T, gameID uuid.UUID) int {
	t.Helper()
	var n int
	require.NoError(t, DB.QueryRow(context.Background(),
		`SELECT count(*) FROM ratings WHERE game_id = $1`, gameID).Scan(&n))
	return n
}

// TestRecordCircuitRatingsTiesFinishersWithinThreeCumulativePoints is the tie half of the
// circuit cadence (RULES.md T6, MATCHMAKING.md 6.2): the single end-of-circuit update reads
// cumulative totals, and two finishers inside the 3-point band are recorded as a tie. Two
// players who start level and tie exchange no rating, so a moved elo_1v1 here means the band was
// not applied; the shrunken deviation is what proves the update ran at all rather than being
// skipped.
func TestRecordCircuitRatingsTiesFinishersWithinThreeCumulativePoints(t *testing.T) {
	setupGameTest(t)

	userA := createGameTestUser(t, "circuit-tie-a-"+uuid.NewString())
	userB := createGameTestUser(t, "circuit-tie-b-"+uuid.NewString())

	gameID := uuid.New()
	seedGameRow(t, gameID, userA.ID)

	ctx := context.Background()
	roster := []uuid.UUID{userA.ID, userB.ID}
	cumulative := map[uuid.UUID]int{userA.ID: 40, userB.ID: 42}
	require.NoError(t, RecordCircuitRatings(ctx, gameID, roster, cumulative))

	afterA, err := GetUserByID(ctx, userA.ID)
	require.NoError(t, err)
	afterB, err := GetUserByID(ctx, userB.ID)
	require.NoError(t, err)

	require.Equal(t, 1500, afterA.Elo1v1, "a 2-point cumulative margin is a tie, so no rating changes hands")
	require.Equal(t, 1500, afterB.Elo1v1, "a 2-point cumulative margin is a tie, so no rating changes hands")
	require.Less(t, afterA.Phi1v1, 350.0, "the circuit's single update must still run: rating deviation shrinks on a tie")
	require.Less(t, afterB.Phi1v1, 350.0, "the circuit's single update must still run: rating deviation shrinks on a tie")
	require.Equal(t, 2, ratingRowCount(t, gameID), "one ratings row per player: the circuit rates exactly once")
}

// TestRecordCircuitRatingsRanksFinishersOutsideTheBand is the decisive half: a cumulative margin
// wider than the 3-point band is a real result, and the lower cumulative total (better in Cambia)
// gains rating.
func TestRecordCircuitRatingsRanksFinishersOutsideTheBand(t *testing.T) {
	setupGameTest(t)

	winner := createGameTestUser(t, "circuit-win-"+uuid.NewString())
	loser := createGameTestUser(t, "circuit-loss-"+uuid.NewString())

	gameID := uuid.New()
	seedGameRow(t, gameID, winner.ID)

	ctx := context.Background()
	roster := []uuid.UUID{winner.ID, loser.ID}
	cumulative := map[uuid.UUID]int{winner.ID: 40, loser.ID: 60}
	require.NoError(t, RecordCircuitRatings(ctx, gameID, roster, cumulative))

	afterWinner, err := GetUserByID(ctx, winner.ID)
	require.NoError(t, err)
	afterLoser, err := GetUserByID(ctx, loser.ID)
	require.NoError(t, err)

	require.Greater(t, afterWinner.Elo1v1, 1500, "the lower cumulative total wins the circuit and gains rating")
	require.Less(t, afterLoser.Elo1v1, 1500, "the higher cumulative total loses the circuit and loses rating")
	require.Equal(t, 2, ratingRowCount(t, gameID))
}

// TestRecordCircuitRatingsUsesThePoolForTheRosterSize covers pool selection and the tie band
// together on a 4-player circuit: the update lands on the 4p pool (never the 1v1 columns), and
// the two finishers inside the band come out level with each other despite differing totals.
func TestRecordCircuitRatingsUsesThePoolForTheRosterSize(t *testing.T) {
	setupGameTest(t)

	users := make([]uuid.UUID, 4)
	for i := range users {
		users[i] = createGameTestUser(t, fmt.Sprintf("circuit-4p-%d-%s", i, uuid.NewString())).ID
	}

	gameID := uuid.New()
	seedGameRow(t, gameID, users[0])

	ctx := context.Background()
	cumulative := map[uuid.UUID]int{
		users[0]: 40, // tied with users[1] (2 points apart)
		users[1]: 42,
		users[2]: 58,
		users[3]: 71,
	}
	require.NoError(t, RecordCircuitRatings(ctx, gameID, users, cumulative))

	after := make([]models.User, len(users))
	for i, id := range users {
		u, err := GetUserByID(ctx, id)
		require.NoError(t, err)
		after[i] = *u
	}

	require.Equal(t, after[0].Elo4p, after[1].Elo4p, "finishers 2 cumulative points apart are tied, so they move identically")
	require.Greater(t, after[0].Elo4p, after[2].Elo4p, "a tied pair at the top of the standings still outrates the field below the band")
	require.Greater(t, after[2].Elo4p, after[3].Elo4p, "the worst cumulative total places last")
	require.Less(t, after[0].Phi4p, 350.0, "the 4p pool's rating deviation converges with play")
	require.Equal(t, 1500, after[0].Elo1v1, "a 4-player circuit must not touch the 1v1 pool")
	require.Equal(t, 350.0, after[0].Phi1v1, "a 4-player circuit must not touch the 1v1 pool")
	require.Equal(t, 4, ratingRowCount(t, gameID), "one ratings row per player: the circuit rates exactly once")
}
