package rating

import (
	"math"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// TestFinalizeRatingsSingleWinCanonicalDelta asserts that FinalizeRatings applies exactly
// one Glicko-2 rating period per game (cambia-243): two fresh players (Elo 1500, default
// RD 350, default sigma 0.06) play a decisive 1v1 game and the winner's rating must move
// by the canonical single-period Glicko-2 delta, not a re-applied multiple of it.
//
// Derivation (Glickman "Example of the Glicko-2 System" steps 3-8, independently computed
// with the same g/E/volatility formulas used in glicko2.go, phi = 350/173.7178):
//
//	g(oppPhi) = 1/sqrt(1+3*phi^2/pi^2) = 0.7123317...
//	E(mu, oppMu, oppPhi) = 0.5                      (mu == oppMu, fresh players)
//	v = 1 / (g^2 * E * (1-E)) = 8.935475
//	delta = v * g * (score - E) = v * g * 0.5 = 2.989226
//	sigma' via the Illinois root-finder (tau=0.5, sigma=0.06) = 0.05999968
//	phi* = sqrt(phi^2 + sigma'^2); phi' = 1/sqrt(1/phi*^2 + 1/v)  -> phi'*173.7178 = 290.3190
//	mu' = mu + phi'^2 * g * (score - E)
//	newElo = mu'*173.7178 + 1500 = 1662.3109  =>  delta = +162.3109
//
// This is well under the ~367 Elo swing the old MultiIterationGlicko2(..., 10) path
// produced by re-applying the same single-period match result ten times in a row.
func TestFinalizeRatingsSingleWinCanonicalDelta(t *testing.T) {
	winner := models.User{ID: uuid.New(), Elo1v1: 1500}
	loser := models.User{ID: uuid.New(), Elo1v1: 1500}

	// Lower score wins in Cambia; FinalizeRatings converts this into rank fractions
	// (1.0 for the winner, 0.0 for the loser) before running the Glicko-2 update.
	scores := map[uuid.UUID]int{
		winner.ID: 0,
		loser.ID:  10,
	}

	updated := FinalizeRatings([]models.User{winner, loser}, scores)
	if len(updated) != 2 {
		t.Fatalf("expected 2 updated players, got %d", len(updated))
	}

	var newWinnerElo, newLoserElo int
	for _, u := range updated {
		switch u.ID {
		case winner.ID:
			newWinnerElo = u.Elo1v1
		case loser.ID:
			newLoserElo = u.Elo1v1
		}
	}

	const wantDelta = 162.3109
	gotWinnerDelta := float64(newWinnerElo - 1500)
	if math.Abs(gotWinnerDelta-wantDelta) > 0.5 {
		t.Errorf("winner delta: got %.4f, want %.4f (+/-0.5)", gotWinnerDelta, wantDelta)
	}

	gotLoserDelta := float64(newLoserElo - 1500)
	if math.Abs(gotLoserDelta+wantDelta) > 0.5 {
		t.Errorf("loser delta: got %.4f, want %.4f (+/-0.5)", gotLoserDelta, -wantDelta)
	}
}

// TestFinalizeRatingsNoTenXOvershoot confirms the rating-period overshoot bug (cambia-243)
// is gone. FinalizeRatings previously called MultiIterationGlicko2(players, scores, 10),
// re-applying the same match result ten times and moving an equal-rating win by roughly
// +367 Elo. A true single rating-period application must stay well under 200 Elo.
func TestFinalizeRatingsNoTenXOvershoot(t *testing.T) {
	winner := models.User{ID: uuid.New(), Elo1v1: 1500}
	loser := models.User{ID: uuid.New(), Elo1v1: 1500}

	scores := map[uuid.UUID]int{
		winner.ID: 0,
		loser.ID:  10,
	}

	updated := FinalizeRatings([]models.User{winner, loser}, scores)
	for _, u := range updated {
		if u.ID != winner.ID {
			continue
		}
		delta := u.Elo1v1 - 1500
		if delta <= 0 {
			t.Errorf("winner rating should have increased, got delta %d", delta)
		}
		if delta >= 200 {
			t.Errorf("winner delta %d Elo is >= 200; the 10x rating-period overshoot appears to still be present", delta)
		}
	}
}

// TestFinalizeRatingsThreePlayerRankFractions exercises the multiplayer path (average
// "opponent" approximation) to confirm a single Glicko-2 application still produces a
// sane ordering: 1st place gains rating, last place loses rating, and no single-game
// swing approaches the old 10x-reapplied magnitude.
func TestFinalizeRatingsThreePlayerRankFractions(t *testing.T) {
	first := models.User{ID: uuid.New(), Elo1v1: 1500}
	second := models.User{ID: uuid.New(), Elo1v1: 1500}
	third := models.User{ID: uuid.New(), Elo1v1: 1500}

	scores := map[uuid.UUID]int{
		first.ID:  0,
		second.ID: 5,
		third.ID:  10,
	}

	updated := FinalizeRatings([]models.User{first, second, third}, scores)

	byID := make(map[uuid.UUID]int, len(updated))
	for _, u := range updated {
		byID[u.ID] = u.Elo1v1
	}

	if byID[first.ID] <= 1500 {
		t.Errorf("1st place should gain rating, got %d", byID[first.ID])
	}
	if byID[third.ID] >= 1500 {
		t.Errorf("last place should lose rating, got %d", byID[third.ID])
	}
	for id, elo := range byID {
		delta := elo - 1500
		if delta < 0 {
			delta = -delta
		}
		if delta >= 200 {
			t.Errorf("player %s delta %d Elo is >= 200; overshoot suspected", id, delta)
		}
	}
}
