// internal/rating/circuit_test.go
package rating

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// TestCircuitRankScoresAppliesTheThreePointTieBand covers the rule RULES.md T6 and
// MATCHMAKING.md 6.3 state for a circuit's single end-of-tournament update: finishers within 3
// cumulative points of each other are recorded as tied, everyone else is separated by placement.
// Ranks are what the pool-aware Glicko-2 path consumes in place of raw scores, so equal ranks
// are what make the tied pair share a rank fraction there.
func TestCircuitRankScoresAppliesTheThreePointTieBand(t *testing.T) {
	a, b, c, d := uuid.New(), uuid.New(), uuid.New(), uuid.New()
	ids := []uuid.UUID{a, b, c, d}
	cumulative := map[uuid.UUID]int{a: 40, b: 42, c: 55, d: 70}

	ranks := CircuitRankScores(ids, cumulative)

	require.Equal(t, ranks[a], ranks[b], "finishers 2 cumulative points apart must share a rank (3-point tie band)")
	require.Equal(t, 1, ranks[a], "the tie group holding the lowest cumulative score places first")
	require.Equal(t, 3, ranks[c], "a finisher outside the band of the leading group keeps its own placement")
	require.Equal(t, 4, ranks[d], "the last finisher keeps its own placement")
}

// TestCircuitRankScoresSeparatesFinishersOutsideTheBand is the negative half: a 4-point gap is
// outside the band, so the two finishers must not be recorded as tied.
func TestCircuitRankScoresSeparatesFinishersOutsideTheBand(t *testing.T) {
	a, b := uuid.New(), uuid.New()
	ranks := CircuitRankScores([]uuid.UUID{a, b}, map[uuid.UUID]int{a: 40, b: 44})

	require.Equal(t, 1, ranks[a])
	require.Equal(t, 2, ranks[b], "a 4-point cumulative gap is outside the 3-point tie band")
}

// TestCircuitRankScoresHandlesAnEmptyRoster guards the degenerate call: the engine's ranking
// helper indexes its first element unconditionally, so an empty roster has to be turned away
// before it reaches there.
func TestCircuitRankScoresHandlesAnEmptyRoster(t *testing.T) {
	require.Empty(t, CircuitRankScores(nil, map[uuid.UUID]int{}))
}
