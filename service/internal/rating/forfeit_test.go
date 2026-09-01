// internal/rating/forfeit_test.go
//
// How a forfeited seat rates (cambia-1541). The seat used to reach FinalizeRatings with no entry
// in the score map at all, and the zero value that produced is a first-place finish here: the sort
// is ascending and lower is better in Cambia. Two fresh 1500 ratings came out 1662 for the seat
// that quit and 1338 for the seat that stayed. The seat now arrives carrying
// engine.ForfeitRoundScore, which is what these tests rate.
package rating

import (
	"testing"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// eloByID reads the 1v1 pool rating of each updated user back into a map.
func eloByID(updated []models.User) map[uuid.UUID]int {
	out := make(map[uuid.UUID]int, len(updated))
	for _, u := range updated {
		out[u.ID] = u.Elo1v1
	}
	return out
}

// TestForfeitScoreRatesTheQuitterDown is the reported bug's arithmetic, rerun with the seat
// scored: the player who walked away must not gain rating and the player who stayed must not lose
// it. The 1662/1338 split the missing key produced was exactly backwards.
func TestForfeitScoreRatesTheQuitterDown(t *testing.T) {
	quitter := models.User{ID: uuid.New(), Elo1v1: 1500}
	stayer := models.User{ID: uuid.New(), Elo1v1: 1500}

	scores := map[uuid.UUID]int{
		quitter.ID: engine.ForfeitRoundScore,
		stayer.ID:  12,
	}

	elos := eloByID(FinalizeRatings([]models.User{quitter, stayer}, scores, Mode1v1))
	quitterDelta := elos[quitter.ID] - 1500
	stayerDelta := elos[stayer.ID] - 1500
	t.Logf("two fresh 1500s, one forfeit: quitter %d (%+d), stayer %d (%+d)",
		elos[quitter.ID], quitterDelta, elos[stayer.ID], stayerDelta)

	if quitterDelta > 0 {
		t.Errorf("the seat that forfeited gained rating: %+d", quitterDelta)
	}
	if stayerDelta < 0 {
		t.Errorf("the seat that stayed lost rating: %+d", stayerDelta)
	}
}

// TestForfeitScoreNeverOutranksAFinisherInAFourSeatRating is the rating half of the four-seat
// case: a forfeit that did not end the round still has to rate below every seat that played it
// out. The scores are the map a finished four-seat game hands the rating roster.
func TestForfeitScoreNeverOutranksAFinisherInAFourSeatRating(t *testing.T) {
	first := models.User{ID: uuid.New(), Elo4p: 1500}
	second := models.User{ID: uuid.New(), Elo4p: 1500}
	third := models.User{ID: uuid.New(), Elo4p: 1500}
	quitter := models.User{ID: uuid.New(), Elo4p: 1500}

	scores := map[uuid.UUID]int{
		first.ID:   4,
		second.ID:  8,
		third.ID:   12,
		quitter.ID: engine.ForfeitRoundScore,
	}

	updated := FinalizeRatings([]models.User{first, second, third, quitter}, scores, Mode4p)
	elos := make(map[uuid.UUID]int, len(updated))
	for _, u := range updated {
		elos[u.ID] = u.Elo4p
	}
	t.Logf("four fresh 1500s, one forfeit: %d / %d / %d finishers, quitter %d",
		elos[first.ID], elos[second.ID], elos[third.ID], elos[quitter.ID])

	for _, finisher := range []models.User{first, second, third} {
		if elos[quitter.ID] >= elos[finisher.ID] {
			t.Errorf("the forfeited seat rated %d, at or above a finisher's %d",
				elos[quitter.ID], elos[finisher.ID])
		}
	}
	if elos[quitter.ID]-1500 > 0 {
		t.Errorf("the seat that forfeited gained rating: %+d", elos[quitter.ID]-1500)
	}
}

// TestForfeitScoreMatchesTheMissedCircuitRound holds the two absences to one number: a seat that
// forfeits a quick-play game and a seat that misses a circuit round are both worth 41, so the
// service and the engine cannot drift apart on what an absence costs.
func TestForfeitScoreMatchesTheMissedCircuitRound(t *testing.T) {
	cs, err := engine.NewCircuit(engine.CircuitConfig{
		Format:     engine.CircuitQuick,
		NumPlayers: 2,
		PlayerIDs:  []int{0, 1},
	})
	if err != nil {
		t.Fatalf("NewCircuit: %v", err)
	}
	if cs.Config.MissedRoundScore != engine.ForfeitRoundScore {
		t.Errorf("a missed circuit round is %d, a forfeited seat %d: the two must be one constant",
			cs.Config.MissedRoundScore, engine.ForfeitRoundScore)
	}
}
