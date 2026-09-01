// internal/database/roster_score_test.go
//
// The roster/score contract (cambia-1541). Both write paths used to index the score map by player
// id and take whatever came back, so a roster entry the map had no key for was recorded and rated
// as a 0. Lower is better in Cambia, which made that 0 the best score at the table. Both now
// refuse the roster instead, before any database work, which is why these tests need no Postgres.
package database

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

// TestRecordGameAndResultsRefusesARosterEntryWithNoScore covers the game_results write: the whole
// call is refused rather than storing a 0 for the seat whose score is missing.
func TestRecordGameAndResultsRefusesARosterEntryWithNoScore(t *testing.T) {
	scored := uuid.New()
	unscored := uuid.New()
	players := []*models.Player{{ID: scored}, {ID: unscored}}

	err := RecordGameAndResults(context.Background(), uuid.New(), players,
		map[uuid.UUID]int{scored: 7}, []uuid.UUID{scored}, true)

	require.Error(t, err, "a roster entry with no score must not be written")
	require.Contains(t, err.Error(), unscored.String(), "the error must name the seat it refused")
}

// TestApplyRatingUpdateRefusesARosterEntryWithNoScore covers the rating write, the half that
// produced the reported 1662/1338 split: FinalizeRatings sorts ascending, so the zero value a
// missing key yields ranks the absent seat first and rates it as the winner.
func TestApplyRatingUpdateRefusesARosterEntryWithNoScore(t *testing.T) {
	rated := uuid.New()
	unscored := uuid.New()

	err := applyRatingUpdate(context.Background(), uuid.New(),
		[]uuid.UUID{rated, unscored}, map[uuid.UUID]int{rated: 7})

	require.Error(t, err, "a roster entry with no score must not be rated")
	require.Contains(t, err.Error(), unscored.String(), "the error must name the seat it refused")
}

// TestApplyRatingUpdateAcceptsAFullRoster is the negative control: the refusal is about a missing
// score and nothing else, so a complete roster gets past it and stops only on the pool gate an
// unsupported seat count applies.
func TestApplyRatingUpdateAcceptsAFullRoster(t *testing.T) {
	a, b, c := uuid.New(), uuid.New(), uuid.New()

	// Three seats is not a rated pool (2, 4, 7 and 8 are), so this returns without touching the
	// database, which is what makes the pass observable with no Postgres running.
	err := applyRatingUpdate(context.Background(), uuid.New(),
		[]uuid.UUID{a, b, c}, map[uuid.UUID]int{a: 4, b: 8, c: 12})

	require.NoError(t, err)
}
