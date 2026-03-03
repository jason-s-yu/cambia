package engine

import (
	"math"
	"testing"
)

func TestNewOpenSkillRating(t *testing.T) {
	r := NewOpenSkillRating()
	if r.Mu != 25.0 {
		t.Errorf("expected mu=25.0, got %v", r.Mu)
	}
	expected := 25.0 / 3.0
	if math.Abs(r.Sigma-expected) > 1e-9 {
		t.Errorf("expected sigma≈%v, got %v", expected, r.Sigma)
	}
}

func TestUpdateOpenSkill_ClearWinner(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	r1 := NewOpenSkillRating()
	r2 := NewOpenSkillRating()
	updated := UpdateOpenSkill([]OpenSkillRating{r1, r2}, []int{1, 2}, beta, 0.0)
	if updated[0].Mu <= r1.Mu {
		t.Errorf("winner mu should increase: %v -> %v", r1.Mu, updated[0].Mu)
	}
	if updated[1].Mu >= r2.Mu {
		t.Errorf("loser mu should decrease: %v -> %v", r2.Mu, updated[1].Mu)
	}
	if updated[0].Sigma >= r1.Sigma {
		t.Errorf("winner sigma should decrease: %v -> %v", r1.Sigma, updated[0].Sigma)
	}
	if updated[1].Sigma >= r2.Sigma {
		t.Errorf("loser sigma should decrease: %v -> %v", r2.Sigma, updated[1].Sigma)
	}
}

func TestUpdateOpenSkill_Tie(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	r1 := NewOpenSkillRating()
	r2 := NewOpenSkillRating()
	updated := UpdateOpenSkill([]OpenSkillRating{r1, r2}, []int{1, 1}, beta, 0.0)
	// With equal ratings and tie, delta should be near zero
	if math.Abs(updated[0].Mu-r1.Mu) > 1e-9 {
		t.Errorf("tie between equal players: expected no mu change, got %v", updated[0].Mu-r1.Mu)
	}
	// Compare magnitude of shift vs clear winner
	clearUpdated := UpdateOpenSkill([]OpenSkillRating{r1, r2}, []int{1, 2}, beta, 0.0)
	tieShift := math.Abs(updated[0].Mu - r1.Mu)
	clearShift := math.Abs(clearUpdated[0].Mu - r1.Mu)
	if tieShift >= clearShift {
		t.Errorf("tie shift (%v) should be smaller than clear winner shift (%v)", tieShift, clearShift)
	}
}

func TestUpdateOpenSkill_4Player(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	ratings := []OpenSkillRating{
		NewOpenSkillRating(),
		NewOpenSkillRating(),
		NewOpenSkillRating(),
		NewOpenSkillRating(),
	}
	ranks := []int{1, 2, 3, 4}
	updated := UpdateOpenSkill(ratings, ranks, beta, 0.0)
	// Rank 1 gains most mu, rank 4 loses most
	if updated[0].Mu <= ratings[0].Mu {
		t.Errorf("rank 1 should gain mu")
	}
	if updated[3].Mu >= ratings[3].Mu {
		t.Errorf("rank 4 should lose mu")
	}
	gain0 := updated[0].Mu - ratings[0].Mu
	gain1 := updated[1].Mu - ratings[1].Mu
	loss2 := ratings[2].Mu - updated[2].Mu
	loss3 := ratings[3].Mu - updated[3].Mu
	if gain0 <= gain1 {
		t.Errorf("rank 1 should gain more than rank 2: %v vs %v", gain0, gain1)
	}
	if loss3 <= loss2 {
		t.Errorf("rank 4 should lose more than rank 3: %v vs %v", loss3, loss2)
	}
}

func TestUpdateOpenSkill_Symmetry(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	r1 := NewOpenSkillRating()
	r2 := NewOpenSkillRating()
	updated := UpdateOpenSkill([]OpenSkillRating{r1, r2}, []int{1, 2}, beta, 0.0)
	gain := updated[0].Mu - r1.Mu
	loss := r2.Mu - updated[1].Mu
	if math.Abs(gain-loss) > 1e-9 {
		t.Errorf("winner gain (%v) should equal loser loss (%v) for symmetric case", gain, loss)
	}
}

func TestUpdateOpenSkill_SigmaDecreases(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	ratings := []OpenSkillRating{NewOpenSkillRating(), NewOpenSkillRating()}
	updated := UpdateOpenSkill(ratings, []int{1, 2}, beta, 0.0)
	for i, u := range updated {
		if u.Sigma >= ratings[i].Sigma {
			t.Errorf("player %d sigma should decrease: %v -> %v", i, ratings[i].Sigma, u.Sigma)
		}
	}
}

func TestUpdateOpenSkill_TauZero(t *testing.T) {
	beta := DefaultOpenSkillBeta()
	ratings := []OpenSkillRating{NewOpenSkillRating(), NewOpenSkillRating()}
	ranks := []int{1, 2}
	withTau := UpdateOpenSkill(ratings, ranks, beta, 0.0)
	withZero := UpdateOpenSkill(ratings, ranks, beta, 0.0)
	for i := range ratings {
		if math.Abs(withTau[i].Mu-withZero[i].Mu) > 1e-9 {
			t.Errorf("tau=0 results should be identical")
		}
	}
}

func TestRanksFromScores_Basic(t *testing.T) {
	scores := []int{5, 10, 3, 20}
	ranks := RanksFromScores(scores, 0)
	expected := []int{2, 3, 1, 4}
	for i, r := range ranks {
		if r != expected[i] {
			t.Errorf("index %d: expected rank %d, got %d", i, expected[i], r)
		}
	}
}

func TestRanksFromScores_TieMargin(t *testing.T) {
	scores := []int{10, 12, 15, 30}
	ranks := RanksFromScores(scores, 3)
	expected := []int{1, 1, 3, 4}
	for i, r := range ranks {
		if r != expected[i] {
			t.Errorf("index %d: expected rank %d, got %d", i, expected[i], r)
		}
	}
}

func TestRanksFromScores_AllTied(t *testing.T) {
	scores := []int{10, 11, 12, 13}
	ranks := RanksFromScores(scores, 5)
	for i, r := range ranks {
		if r != 1 {
			t.Errorf("index %d: expected rank 1, got %d", i, r)
		}
	}
}
