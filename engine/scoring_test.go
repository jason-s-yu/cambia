package engine

import (
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
