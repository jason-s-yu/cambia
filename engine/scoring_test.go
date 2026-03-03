package engine

import (
	"math"
	"testing"
)

// helper: build a minimal terminal GameState with specified hands.
func makeTerminalGame(p0Hand []Card, p1Hand []Card, cambiaCaller int8) GameState {
	rules := DefaultHouseRules()
	gs := NewGame(42, rules)

	// Set player 0 hand.
	gs.Players[0].HandLen = uint8(len(p0Hand))
	for i, c := range p0Hand {
		gs.Players[0].Hand[i] = c
	}

	// Set player 1 hand.
	gs.Players[1].HandLen = uint8(len(p1Hand))
	for i, c := range p1Hand {
		gs.Players[1].Hand[i] = c
	}

	gs.CambiaCaller = cambiaCaller
	gs.Flags |= FlagGameOver
	return gs
}

// TestComputeScoresBasic verifies Ace+Two+Three+Joker = 1+2+3+0 = 6.
func TestComputeScoresBasic(t *testing.T) {
	p0Hand := []Card{
		NewCard(SuitHearts, RankAce),
		NewCard(SuitHearts, RankTwo),
		NewCard(SuitHearts, RankThree),
		NewCard(SuitRedJoker, RankJoker),
	}
	p1Hand := []Card{}
	gs := makeTerminalGame(p0Hand, p1Hand, -1)

	scores := gs.computeScores()
	if scores[0] != 6 {
		t.Errorf("expected score 6, got %d", scores[0])
	}
	if scores[1] != 0 {
		t.Errorf("expected score 0 for empty hand, got %d", scores[1])
	}
}

// TestComputeScoresRedKing verifies Red King has value -1.
func TestComputeScoresRedKing(t *testing.T) {
	p0Hand := []Card{
		NewCard(SuitHearts, RankKing),  // Red King = -1
		NewCard(SuitHearts, RankTwo),   // 2
	}
	gs := makeTerminalGame(p0Hand, nil, -1)

	scores := gs.computeScores()
	// -1 + 2 = 1
	if scores[0] != 1 {
		t.Errorf("expected score 1 (RedKing + Two), got %d", scores[0])
	}
}

// TestComputeScoresBlackKing verifies Black King has value 13.
func TestComputeScoresBlackKing(t *testing.T) {
	p0Hand := []Card{
		NewCard(SuitSpades, RankKing),  // Black King = 13
		NewCard(SuitHearts, RankAce),   // 1
	}
	gs := makeTerminalGame(p0Hand, nil, -1)

	scores := gs.computeScores()
	// 13 + 1 = 14
	if scores[0] != 14 {
		t.Errorf("expected score 14 (BlackKing + Ace), got %d", scores[0])
	}
}

// TestUtilityWinnerLoser: P0 lower score → utilities [1, -1].
func TestUtilityWinnerLoser(t *testing.T) {
	// P0: Ace = 1; P1: Ten = 10
	p0Hand := []Card{NewCard(SuitHearts, RankAce)}
	p1Hand := []Card{NewCard(SuitHearts, RankTen)}
	gs := makeTerminalGame(p0Hand, p1Hand, -1)

	u := gs.GetUtility()
	if u[0] != 1.0 {
		t.Errorf("expected u[0]=1.0, got %f", u[0])
	}
	if u[1] != -1.0 {
		t.Errorf("expected u[1]=-1.0, got %f", u[1])
	}
}

// TestUtilityTieNoCambia: Same score, no Cambia called → [0, 0].
func TestUtilityTieNoCambia(t *testing.T) {
	p0Hand := []Card{NewCard(SuitHearts, RankFive)} // 5
	p1Hand := []Card{NewCard(SuitClubs, RankFive)}  // 5
	gs := makeTerminalGame(p0Hand, p1Hand, -1)

	u := gs.GetUtility()
	if u[0] != 0.0 {
		t.Errorf("expected u[0]=0.0, got %f", u[0])
	}
	if u[1] != 0.0 {
		t.Errorf("expected u[1]=0.0, got %f", u[1])
	}
}

// TestUtilityCambiaCallerWins: Caller (P0) has lower score → caller wins.
func TestUtilityCambiaCallerWins(t *testing.T) {
	p0Hand := []Card{NewCard(SuitHearts, RankAce)}  // 1
	p1Hand := []Card{NewCard(SuitHearts, RankQueen)} // 12
	gs := makeTerminalGame(p0Hand, p1Hand, 0) // P0 called Cambia

	u := gs.GetUtility()
	if u[0] != 1.0 {
		t.Errorf("expected u[0]=1.0 (Cambia caller wins), got %f", u[0])
	}
	if u[1] != -1.0 {
		t.Errorf("expected u[1]=-1.0, got %f", u[1])
	}
}

// TestUtilityCambiaCallerTies: Tied score, caller wins the tie.
func TestUtilityCambiaCallerTies(t *testing.T) {
	p0Hand := []Card{NewCard(SuitHearts, RankSeven)} // 7
	p1Hand := []Card{NewCard(SuitClubs, RankSeven)}  // 7
	gs := makeTerminalGame(p0Hand, p1Hand, 0) // P0 called Cambia, scores tied

	u := gs.GetUtility()
	if u[0] != 1.0 {
		t.Errorf("expected u[0]=1.0 (Cambia caller wins tie), got %f", u[0])
	}
	if u[1] != -1.0 {
		t.Errorf("expected u[1]=-1.0, got %f", u[1])
	}
}

// TestUtilityFalseCambia: Cambia caller (P0) has HIGHER score → caller loses.
func TestUtilityFalseCambia(t *testing.T) {
	p0Hand := []Card{NewCard(SuitSpades, RankKing)} // 13 (Black King)
	p1Hand := []Card{NewCard(SuitHearts, RankAce)}  // 1
	gs := makeTerminalGame(p0Hand, p1Hand, 0) // P0 called Cambia but has higher score

	u := gs.GetUtility()
	if u[0] != -1.0 {
		t.Errorf("expected u[0]=-1.0 (False Cambia, caller loses), got %f", u[0])
	}
	if u[1] != 1.0 {
		t.Errorf("expected u[1]=1.0, got %f", u[1])
	}
}

// TestUtilityNotTerminal: Game not over → [0, 0].
func TestUtilityNotTerminal(t *testing.T) {
	rules := DefaultHouseRules()
	gs := NewGame(1, rules)
	gs.Deal()
	// Do NOT set FlagGameOver

	u := gs.GetUtility()
	if u[0] != 0.0 || u[1] != 0.0 {
		t.Errorf("expected [0, 0] for non-terminal game, got [%f, %f]", u[0], u[1])
	}
}

// ---- ComputeAggressionSubsidy tests ----

// TestAggressionSubsidyH2H: 2-player, 1st=-3, 2nd=0.
func TestAggressionSubsidyH2H(t *testing.T) {
	subs := ComputeAggressionSubsidy(2, []int{0, 1}, -1)
	if subs[0] != -3 {
		t.Errorf("H2H 1st: expected -3, got %d", subs[0])
	}
	if subs[1] != 0 {
		t.Errorf("H2H 2nd: expected 0, got %d", subs[1])
	}
}

// TestAggressionSubsidyFFA4: 4-player, 1st=-5, 2nd=-2, 3rd/4th=0.
func TestAggressionSubsidyFFA4(t *testing.T) {
	subs := ComputeAggressionSubsidy(4, []int{0, 1, 2, 3}, -1)
	want := []int{-5, -2, 0, 0}
	for i, w := range want {
		if subs[i] != w {
			t.Errorf("FFA4 place %d: expected %d, got %d", i, w, subs[i])
		}
	}
}

// TestAggressionSubsidy5Plus: 6-player, 1st=-5, 2nd=-2, 3rd=-1, 4th+=0.
func TestAggressionSubsidy5Plus(t *testing.T) {
	subs := ComputeAggressionSubsidy(6, []int{0, 1, 2, 3, 4, 5}, -1)
	want := []int{-5, -2, -1, 0, 0, 0}
	for i, w := range want {
		if subs[i] != w {
			t.Errorf("5+P place %d: expected %d, got %d", i, w, subs[i])
		}
	}
}

// TestAggressionSubsidyCallerWinsTie: tied players at placement 0; Cambia caller (idx=1) gets -3, other gets 0.
func TestAggressionSubsidyCallerWinsTie(t *testing.T) {
	// H2H: both at placement 0 (tied), player 1 called Cambia → player 1 gets -3, player 0 gets 0.
	subs := ComputeAggressionSubsidy(2, []int{0, 0}, 1)
	if subs[1] != -3 {
		t.Errorf("Cambia caller (tied 1st): expected -3, got %d", subs[1])
	}
	if subs[0] != 0 {
		t.Errorf("Non-caller (tied 1st): expected 0 (bumped to 2nd bonus), got %d", subs[0])
	}
}

// TestAggressionSubsidyTieNoCaller: both tied at placement 0, no Cambia caller → both get 1st bonus.
func TestAggressionSubsidyTieNoCaller(t *testing.T) {
	subs := ComputeAggressionSubsidy(2, []int{0, 0}, -1)
	if subs[0] != -3 || subs[1] != -3 {
		t.Errorf("Tied no caller: expected both -3, got %d %d", subs[0], subs[1])
	}
}

// ---- ComputeScoreDiffOutcome tests ----

// TestScoreDiffOutcomeTieBand: within tie band → 0.5.
func TestScoreDiffOutcomeTieBand(t *testing.T) {
	out := ComputeScoreDiffOutcome(10, 12, 0.15, 3)
	if out != 0.5 {
		t.Errorf("tie band: expected 0.5, got %f", out)
	}
}

// TestScoreDiffOutcomeLogistic: score2 > score1 → player1 wins (s > 0.5).
func TestScoreDiffOutcomeLogistic(t *testing.T) {
	// diff=10, k=0.15 → 1/(1+exp(-1.5)) ≈ 0.8176
	out := ComputeScoreDiffOutcome(10, 20, 0.15, 0)
	expected := 1.0 / (1.0 + math.Exp(-0.15*10))
	if math.Abs(out-expected) > 1e-9 {
		t.Errorf("logistic: expected %f, got %f", expected, out)
	}
}

// TestScoreDiffOutcomeSymmetry: swap scores → 1 - result.
func TestScoreDiffOutcomeSymmetry(t *testing.T) {
	s1 := ComputeScoreDiffOutcome(5, 20, 0.15, 0)
	s2 := ComputeScoreDiffOutcome(20, 5, 0.15, 0)
	if math.Abs(s1+s2-1.0) > 1e-9 {
		t.Errorf("symmetry: expected s1+s2=1.0, got %f+%f=%f", s1, s2, s1+s2)
	}
}

// TestScoreDiffOutcomeBoundaryClamp: large diff clamped to 30.
func TestScoreDiffOutcomeBoundaryClamp(t *testing.T) {
	out100 := ComputeScoreDiffOutcome(0, 100, 0.15, 0)
	out30 := ComputeScoreDiffOutcome(0, 30, 0.15, 0)
	if math.Abs(out100-out30) > 1e-9 {
		t.Errorf("clamp: diff=100 should equal diff=30; got %f vs %f", out100, out30)
	}
}

// TestScoreDiffOutcomeEqualScores: same scores, tieBand=0 → 0.5.
func TestScoreDiffOutcomeEqualScores(t *testing.T) {
	out := ComputeScoreDiffOutcome(7, 7, 0.15, 0)
	if out != 0.5 {
		t.Errorf("equal scores: expected 0.5, got %f", out)
	}
}
