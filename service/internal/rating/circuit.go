// internal/rating/circuit.go
package rating

import (
	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
)

// CircuitTieMargin is the cumulative-score band inside which two circuit finishers are recorded
// as tied (RULES.md T6, MATCHMAKING.md 6.2/6.3): it filters final-turn deck variance out of the
// one rating update a circuit produces. It applies to cumulative tournament scores only; a
// single game's scores are never collapsed this way.
const CircuitTieMargin = 3

// CircuitRankScores converts a circuit's final cumulative scores (lower is better, aggression
// subsidies already applied) into placement ranks under CircuitTieMargin, keyed by player.
//
// The ranks are what the pool-aware rating path consumes in place of raw scores: FinalizeRatings
// reads its input only as an ordering with exact-equality ties, so feeding it ranks preserves the
// finishing order while collapsing every banded pair into one shared rank fraction. Feeding it
// the cumulative scores directly would instead tie only exact matches, which is not the rule.
//
// playerIDs is the circuit roster; a player absent from cumulative is ranked on a score of 0,
// the same zero value the per-game path reads for a missing score.
func CircuitRankScores(playerIDs []uuid.UUID, cumulative map[uuid.UUID]int) map[uuid.UUID]int {
	ranks := make(map[uuid.UUID]int, len(playerIDs))
	if len(playerIDs) == 0 {
		return ranks
	}
	scores := make([]int, len(playerIDs))
	for i, id := range playerIDs {
		scores[i] = cumulative[id]
	}
	for i, r := range engine.RanksFromScores(scores, CircuitTieMargin) {
		ranks[playerIDs[i]] = r
	}
	return ranks
}
