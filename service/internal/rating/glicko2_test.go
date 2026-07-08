package rating

import (
	"math"
	"testing"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

func TestUpdate1v1(t *testing.T) {
	winner := models.User{Elo1v1: 1500}
	loser := models.User{Elo1v1: 1500}

	newW, newL := Update1v1(winner, loser)
	if newW.Elo1v1 <= 1500 {
		t.Errorf("winner's rating should have gone up, got %d", newW.Elo1v1)
	}
	if newL.Elo1v1 >= 1500 {
		t.Errorf("loser's rating should have gone down, got %d", newL.Elo1v1)
	}
}

// TestGlickmanWorkedExample reproduces the canonical worked example from Glickman's
// "Example of the Glicko-2 system" (player rated 1500 with RD 200, sigma 0.06, tau 0.5,
// facing three opponents: 1400/30 win, 1550/100 loss, 1700/300 loss). The paper's
// published results are r' = 1464.06, RD' = 151.52, sigma' = 0.05999.
func TestGlickmanWorkedExample(t *testing.T) {
	player := NewGlicko2Rating(1500, 200, 0.06)
	opps := []Glicko2Rating{
		NewGlicko2Rating(1400, 30, 0.06),
		NewGlicko2Rating(1550, 100, 0.06),
		NewGlicko2Rating(1700, 300, 0.06),
	}
	scores := []float64{1.0, 0.0, 0.0}

	res := updateGlickoMulti(player, opps, scores)

	gotElo := res.ToElo()
	gotRD := res.Phi * GlickoScale
	gotSigma := res.Sigma

	if math.Abs(gotElo-1464.06) > 0.05 {
		t.Errorf("rating: got %.4f, want 1464.06", gotElo)
	}
	if math.Abs(gotRD-151.52) > 0.05 {
		t.Errorf("RD: got %.4f, want 151.52", gotRD)
	}
	if math.Abs(gotSigma-0.05999) > 5e-5 {
		t.Errorf("sigma: got %.6f, want 0.05999", gotSigma)
	}
}

// TestGlickoNoGames confirms step 6 with no opponents: rating and volatility hold while
// the deviation grows by the volatility (uncertainty increases over an idle rating period).
func TestGlickoNoGames(t *testing.T) {
	player := NewGlicko2Rating(1500, 200, 0.06)
	res := updateGlickoMulti(player, nil, nil)

	if res.Mu != player.Mu {
		t.Errorf("mu should be unchanged with no games: got %.6f, want %.6f", res.Mu, player.Mu)
	}
	if res.Sigma != player.Sigma {
		t.Errorf("sigma should be unchanged with no games: got %.6f, want %.6f", res.Sigma, player.Sigma)
	}
	wantPhi := math.Sqrt(player.Phi*player.Phi + player.Sigma*player.Sigma)
	if math.Abs(res.Phi-wantPhi) > 1e-12 {
		t.Errorf("phi: got %.9f, want %.9f", res.Phi, wantPhi)
	}
	if res.Phi <= player.Phi {
		t.Errorf("deviation should grow with no games: got %.6f, was %.6f", res.Phi, player.Phi)
	}
}
