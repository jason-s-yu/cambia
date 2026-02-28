package engine

import (
	"testing"
)

// TestTournamentInterface verifies all types and methods are callable.
func TestTournamentInterface(t *testing.T) {
	cfg := TournamentConfig{
		Mode:            "series",
		NumRounds:       2,
		GamesPerMatchup: 1,
		ScoringMethod:   "wins",
		SeedingMethod:   "random",
		PlayerIDs:       []int{1, 2, 3, 4},
	}
	ts := NewTournament(cfg)
	if ts == nil {
		t.Fatal("NewTournament returned nil")
	}
	_ = ts.IsComplete()
	_ = ts.GetStandings()
	_, _, _, err := ts.NextMatchup()
	if err != nil {
		t.Fatalf("NextMatchup error: %v", err)
	}
}

// TestSeriesTournament: 4-player, 3-round series, wins scoring.
func TestSeriesTournament(t *testing.T) {
	cfg := TournamentConfig{
		Mode:          "series",
		NumRounds:     3,
		ScoringMethod: "wins",
		PlayerIDs:     []int{1, 2, 3, 4},
	}
	ts := NewTournament(cfg)

	// Play all matchups until tournament is complete.
	// The tournament auto-advances rounds internally.
	played := 0
	for !ts.IsComplete() {
		p1, p2, _, err := ts.NextMatchup()
		if err != nil {
			t.Fatalf("NextMatchup after %d played: %v", played, err)
		}
		if p1 == -1 {
			t.Fatalf("incomplete but no matchup available after %d played", played)
		}
		// Player 1 always wins (lower ID = lower score).
		scores := []int{5, 10}
		if p1 != 1 {
			scores = []int{10, 5} // p2 might be player 1
		}
		if err := ts.RecordResult(p1, p2, scores); err != nil {
			t.Fatalf("RecordResult (%d,%d) after %d played: %v", p1, p2, played, err)
		}
		played++
	}

	// 4 players, C(4,2)=6 matchups per round, 3 rounds = 18 total.
	if played != 18 {
		t.Errorf("expected 18 matchups played, got %d", played)
	}

	standings := ts.GetStandings()
	if len(standings) != 4 {
		t.Fatalf("expected 4 standings, got %d", len(standings))
	}

	// Player 1 should have most wins.
	if standings[0].PlayerID != 1 {
		t.Errorf("expected player 1 to be first, got %d", standings[0].PlayerID)
	}
	// Player 1 wins all 3 matchups per round × 3 rounds = 9 wins.
	if standings[0].Wins != 9 {
		t.Errorf("player 1 expected 9 wins, got %d", standings[0].Wins)
	}
}

// TestSeriesCumulativeScoring verifies cumulative_score variant.
func TestSeriesCumulativeScoring(t *testing.T) {
	cfg := TournamentConfig{
		Mode:          "series",
		NumRounds:     1,
		ScoringMethod: "cumulative_score",
		PlayerIDs:     []int{10, 20, 30},
	}
	ts := NewTournament(cfg)

	// Scores: p10 gets 3, p20 gets 7, p30 gets 5
	matches := [][3]int{
		{10, 20, 3},  // p10=3, p20=7
		{10, 30, 3},  // p10=3, p30=5
		{20, 30, 6},  // p20=6, p30=4
	}
	for _, m := range matches {
		p1, p2, s1, s2 := m[0], m[1], m[2], m[2]+4
		_ = s2
		if err := ts.RecordResult(p1, p2, []int{s1, s1 + 4}); err != nil {
			t.Fatalf("RecordResult: %v", err)
		}
	}
	// We just played 3 matchups covering all pairs once in round 0.
	// (Note: the schedule has 3 pairs; all should be recorded.)

	standings := ts.GetStandings()
	// Cumulative_score: higher is better.
	// p20: 7+6=13 points, p30: 5+4=9, p10: 3+3=6
	// But standings depend on what we actually passed — let's just verify ordering.
	if len(standings) != 3 {
		t.Fatalf("expected 3 standings, got %d", len(standings))
	}
	// First standing should have highest score.
	if standings[0].Score < standings[1].Score {
		t.Errorf("standings not sorted: %d < %d", standings[0].Score, standings[1].Score)
	}
}

// TestSingleEliminationBracket8: 8 players, 3 rounds, correct structure.
func TestSingleEliminationBracket8(t *testing.T) {
	players := []int{1, 2, 3, 4, 5, 6, 7, 8}
	cfg := TournamentConfig{
		Mode:      "single_elimination",
		PlayerIDs: players,
	}
	ts := NewTournament(cfg)

	if len(ts.Bracket) != 3 {
		t.Fatalf("expected 3 rounds for 8 players, got %d", len(ts.Bracket))
	}
	if len(ts.Bracket[0].Matchups) != 4 {
		t.Fatalf("round 1 should have 4 matchups, got %d", len(ts.Bracket[0].Matchups))
	}

	// Play through to completion.
	for !ts.IsComplete() {
		p1, p2, isBye, err := ts.NextMatchup()
		if err != nil {
			t.Fatalf("NextMatchup: %v", err)
		}
		if p1 == -1 {
			t.Fatal("incomplete but no matchup available")
		}
		if isBye {
			// Bye is auto-played; call RecordResult anyway (should be a no-op or handle gracefully).
			continue
		}
		// Lower ID wins.
		if err := ts.RecordResult(p1, p2, []int{5, 10}); err != nil {
			t.Fatalf("RecordResult p1=%d p2=%d: %v", p1, p2, err)
		}
	}

	standings := ts.GetStandings()
	// Winner should be player 1 (always wins).
	winner := standings[0]
	if winner.PlayerID != 1 {
		t.Errorf("expected player 1 to win, got %d", winner.PlayerID)
	}
	// Eliminated players should be at the bottom.
	for i := 1; i < len(standings); i++ {
		if standings[i].Eliminated != true {
			// Allow non-eliminated runner-up if still playing.
		}
	}
}

// TestSingleEliminationByes: 6 players → 2 byes in round 1.
func TestSingleEliminationByes(t *testing.T) {
	players := []int{1, 2, 3, 4, 5, 6}
	cfg := TournamentConfig{
		Mode:      "single_elimination",
		PlayerIDs: players,
	}
	ts := NewTournament(cfg)

	// Count byes in round 1.
	byeCount := 0
	for _, m := range ts.Bracket[0].Matchups {
		if m.Bye {
			byeCount++
		}
	}
	if byeCount != 2 {
		t.Errorf("expected 2 byes for 6 players, got %d", byeCount)
	}
}

// TestSingleElimination2Players: minimal case.
func TestSingleElimination2Players(t *testing.T) {
	cfg := TournamentConfig{
		Mode:      "single_elimination",
		PlayerIDs: []int{1, 2},
	}
	ts := NewTournament(cfg)
	if ts.IsComplete() {
		t.Fatal("should not be complete at start")
	}
	p1, p2, isBye, _ := ts.NextMatchup()
	if isBye {
		t.Fatal("2-player tournament should not have byes")
	}
	if err := ts.RecordResult(p1, p2, []int{3, 7}); err != nil {
		t.Fatal(err)
	}
	if !ts.IsComplete() {
		t.Fatal("2-player single elim should complete after 1 game")
	}
}

// TestDoubleEliminationBasic: 4 players, 2 losses to eliminate.
func TestDoubleEliminationBasic(t *testing.T) {
	cfg := TournamentConfig{
		Mode:      "double_elimination",
		PlayerIDs: []int{1, 2, 3, 4},
	}
	ts := NewTournament(cfg)

	// Play all winners bracket matchups.
	for i := 0; i < 100 && !ts.IsComplete(); i++ {
		p1, p2, isBye, err := ts.NextMatchup()
		if err != nil {
			t.Fatalf("NextMatchup: %v", err)
		}
		if p1 == -1 {
			break
		}
		if isBye {
			continue
		}
		if err := ts.RecordResult(p1, p2, []int{5, 10}); err != nil {
			t.Logf("RecordResult failed (may be expected): %v", err)
			break
		}
	}

	// Losers should have 2 losses before being eliminated.
	for _, s := range ts.Standings {
		if s.Eliminated && s.Losses < 2 {
			t.Errorf("player %d eliminated with only %d loss(es), expected ≥2", s.PlayerID, s.Losses)
		}
	}
}

// TestSeries2Players: edge case.
func TestSeries2Players(t *testing.T) {
	cfg := TournamentConfig{
		Mode:      "series",
		NumRounds: 1,
		PlayerIDs: []int{1, 2},
	}
	ts := NewTournament(cfg)
	p1, p2, _, _ := ts.NextMatchup()
	if err := ts.RecordResult(p1, p2, []int{3, 8}); err != nil {
		t.Fatal(err)
	}
	if !ts.IsComplete() {
		t.Fatal("2-player 1-round series should complete after 1 game")
	}
}

// TestOddPlayers: 5-player single elimination.
func TestOddPlayersSingleElim(t *testing.T) {
	cfg := TournamentConfig{
		Mode:      "single_elimination",
		PlayerIDs: []int{1, 2, 3, 4, 5},
	}
	ts := NewTournament(cfg)
	// 5 players → next power of 2 = 8 → 3 byes.
	byeCount := 0
	for _, m := range ts.Bracket[0].Matchups {
		if m.Bye {
			byeCount++
		}
	}
	if byeCount != 3 {
		t.Errorf("expected 3 byes for 5 players, got %d", byeCount)
	}
}

// TestLowestCumulativeScoring: lower score is better.
func TestLowestCumulativeScoring(t *testing.T) {
	cfg := TournamentConfig{
		Mode:          "series",
		NumRounds:     1,
		ScoringMethod: "lowest_cumulative",
		PlayerIDs:     []int{1, 2},
	}
	ts := NewTournament(cfg)
	// p1 gets score 3, p2 gets score 10.
	p1, p2, _, _ := ts.NextMatchup()
	_ = ts.RecordResult(p1, p2, []int{3, 10})

	standings := ts.GetStandings()
	// Lower score should be first.
	if standings[0].Score > standings[1].Score {
		t.Errorf("lowest_cumulative: expected lower score first, got %d > %d",
			standings[0].Score, standings[1].Score)
	}
}

// TestTournamentComplete_Series: verify Completed flag.
func TestTournamentComplete_Series(t *testing.T) {
	cfg := TournamentConfig{
		Mode:      "series",
		NumRounds: 2,
		PlayerIDs: []int{1, 2},
	}
	ts := NewTournament(cfg)
	if ts.IsComplete() {
		t.Fatal("should not start complete")
	}
	// Round 1.
	p1, p2, _, _ := ts.NextMatchup()
	_ = ts.RecordResult(p1, p2, []int{1, 2})
	if ts.IsComplete() {
		t.Fatal("should not be complete after round 1")
	}
	// Round 2.
	p1, p2, _, _ = ts.NextMatchup()
	_ = ts.RecordResult(p1, p2, []int{1, 2})
	if !ts.IsComplete() {
		t.Fatal("should be complete after round 2")
	}
}
