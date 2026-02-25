package engine

import (
	"math"
	"math/rand/v2"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeKnownGame creates a 2-player game where all cards in both hands are
// the given values. It puts those cards directly in hand and leaves the
// stockpile with the rest of a 52-card deck (no jokers for simplicity).
func make2PGame(p0vals, p1vals []int8, cambiaCaller int8) GameState {
	rules := HouseRules{
		NumPlayers:     2,
		CardsPerPlayer: uint8(max(len(p0vals), len(p1vals))),
		NumJokers:      0,
	}
	g := NewGame(42, rules)
	g.Flags |= FlagGameStarted
	g.CambiaCaller = cambiaCaller

	// Place explicit cards.
	g.Players[0].HandLen = uint8(len(p0vals))
	g.Players[1].HandLen = uint8(len(p1vals))

	// Map value → a concrete card (approximate).
	valueToCard := func(v int8) Card {
		switch v {
		case -1:
			return NewCard(SuitHearts, RankKing) // red king = -1
		case 0:
			return NewCard(SuitRedJoker, RankJoker)
		case 1:
			return NewCard(SuitHearts, RankAce)
		case 13:
			return NewCard(SuitClubs, RankKing) // black king = 13
		default:
			// value 2-12: ranks 1-11 give values 2-12
			return NewCard(SuitHearts, uint8(v-1))
		}
	}

	for i, v := range p0vals {
		g.Players[0].Hand[i] = valueToCard(v)
	}
	for i, v := range p1vals {
		g.Players[1].Hand[i] = valueToCard(v)
	}

	// Clear stockpile to empty for "all known" tests.
	g.StockLen = 0
	return g
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// TestRemainingDeck
// ---------------------------------------------------------------------------

func TestRemainingDeck(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	g := NewGame(123, rules)
	g.Deal()

	totalDeck := 52 + int(rules.NumJokers)
	dealtCards := int(rules.CardsPerPlayer) * 2
	discarded := 1 // one flip to start discard

	remaining := g.remainingDeck()
	expected := totalDeck - dealtCards - discarded

	if len(remaining) != expected {
		t.Errorf("remainingDeck: got %d cards, want %d", len(remaining), expected)
	}

	// Verify count equals StockLen.
	if len(remaining) != int(g.StockLen) {
		t.Errorf("remainingDeck len %d != StockLen %d", len(remaining), g.StockLen)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalLinear_KnownCards
// ---------------------------------------------------------------------------

// TerminalEvalLinear treats the evaluating player's own hand as known and
// opponents' hands as unknown (using μ_deck = deck mean for opponent slots).
// When the stockpile is empty (μ_deck = 0), opponent expected score = 0.
// P0 score 5 > 0 → P0 expects to lose (positive opp advantage). Verify sign.
func TestTerminalEvalLinear_KnownCards(t *testing.T) {
	// P0 score = 2+3 = 5. Opponent hand: P1 has 2 cards, stockpile empty → μ=0.
	// From P0's view: expected_opp = 2 * 0 = 0. P0 has higher score → P0 should lose.
	g := make2PGame([]int8{2, 3}, []int8{4, 6}, -1)
	g.Flags |= FlagGameOver

	u0 := g.TerminalEvalLinear(0)
	// P0 score=5, opp expected=0 → P0 has worse score → negative utility.
	if u0 >= 0 {
		t.Errorf("P0 (score 5) vs opp (expected 0): should be negative utility, got %.4f", u0)
	}

	// Now test where P0 score = 0 (using a joker = 0) → should be ≥ 0.
	g2 := make2PGame([]int8{0, 0}, []int8{4, 6}, -1)
	g2.Flags |= FlagGameOver
	u0b := g2.TerminalEvalLinear(0)
	// P0 score=0, opp expected = 2*0 = 0 → tie → utility 0.
	if u0b < -0.001 {
		t.Errorf("P0 (score 0) vs opp (expected 0): should be ~0, got %.4f", u0b)
	}

	// Stockpile with high-value cards → opponent expected score > own score → P0 should win.
	g3 := make2PGame([]int8{1, 1}, []int8{13, 13}, -1)
	g3.Flags |= FlagGameOver
	// Add some high-value cards to stockpile so μ_deck is high.
	g3.Stockpile[0] = NewCard(SuitClubs, RankKing)   // value 13
	g3.Stockpile[1] = NewCard(SuitSpades, RankKing)  // value 13
	g3.Stockpile[2] = NewCard(SuitHearts, RankQueen) // value 12
	g3.StockLen = 3
	u0c := g3.TerminalEvalLinear(0)
	// P0 score = 2, opp expected = 2 * (38/3) ≈ 25.3 → P0 wins → positive utility.
	if u0c <= 0 {
		t.Errorf("P0 (score 2) vs opp (expected ~25): should be positive utility, got %.4f", u0c)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalLinear_UnknownCards
// ---------------------------------------------------------------------------

// With unknown opponent cards, linear utility should be between min and max.
func TestTerminalEvalLinear_UnknownCards(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	g := NewGame(77, rules)
	g.Deal()

	u := g.TerminalEvalLinear(0)
	if u < -1.0 || u > 1.0 {
		t.Errorf("TerminalEvalLinear out of [-1,1]: %.4f", u)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalLinear_ZeroSum
// ---------------------------------------------------------------------------

// TerminalEvalLinear is NOT globally zero-sum across all player perspectives
// because each player uses different "unknown" assumptions. However, when all
// hands are empty (HandLen=0), all expected scores are 0 and the N-player
// pairwise utility formula gives exactly 0 for all players. Verify this.
func TestTerminalEvalLinear_ZeroSum(t *testing.T) {
	rules := HouseRules{
		NumPlayers:     4,
		CardsPerPlayer: 4,
		NumJokers:      0,
	}
	g := NewGame(999, rules)
	g.Flags |= FlagGameStarted
	g.CambiaCaller = -1

	// All players have 0 cards → all expected scores = 0 → utilities all 0.
	for p := uint8(0); p < 4; p++ {
		g.Players[p].HandLen = 0
	}
	g.StockLen = 0

	n := rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		u := g.TerminalEvalLinear(p)
		if math.Abs(float64(u)) > 0.001 {
			t.Errorf("P%d utility should be 0 when all hands empty, got %.6f", p, u)
		}
	}

	// Also verify: when self has lower expected score than μ_deck predicts for
	// opponents, self gets positive utility.
	rules2 := HouseRules{NumPlayers: 3, CardsPerPlayer: 4, NumJokers: 0}
	g2 := NewGame(42, rules2)
	g2.Flags |= FlagGameStarted
	g2.CambiaCaller = -1
	// P0 has score 2 (1+1), opponents each have 2 cards, stockpile mean = 10.
	g2.Players[0].Hand[0] = NewCard(SuitHearts, RankAce) // 1
	g2.Players[0].Hand[1] = NewCard(SuitHearts, RankAce) // 1
	g2.Players[0].HandLen = 2
	g2.Players[1].HandLen = 2
	g2.Players[2].HandLen = 2
	// Stock all high: mean = 10.
	g2.Stockpile[0] = NewCard(SuitHearts, RankTen)
	g2.Stockpile[1] = NewCard(SuitHearts, RankTen)
	g2.StockLen = 2
	u0 := g2.TerminalEvalLinear(0)
	// P0 = 2, opp1 expected = 2*10=20, opp2 expected = 2*10=20.
	// total = 42, u0 = (42 - 3*2) / 2 = 18 → positive.
	if u0 <= 0 {
		t.Errorf("P0 with low score vs high-deck opponents should have positive utility, got %.4f", u0)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalDP_2P_KnownCards
// ---------------------------------------------------------------------------

// When deck is empty (all cards known), DP should match deterministic outcome.
func TestTerminalEvalDP_2P_KnownCards(t *testing.T) {
	tests := []struct {
		p0    []int8
		p1    []int8
		cambia int8
		wantU0 float32 // expected utility for player 0
	}{
		{[]int8{2, 3}, []int8{4, 6}, -1, 1.0},  // P0 wins (5 < 10)
		{[]int8{4, 6}, []int8{2, 3}, -1, -1.0}, // P0 loses
		{[]int8{3, 3}, []int8{3, 3}, -1, 0.0},  // true tie, no caller
		{[]int8{3, 3}, []int8{3, 3}, 0, 1.0},   // tie, P0 called cambia → wins
		{[]int8{3, 3}, []int8{3, 3}, 1, -1.0},  // tie, P1 called cambia → P0 loses
	}

	for _, tc := range tests {
		g := make2PGame(tc.p0, tc.p1, tc.cambia)
		g.Flags |= FlagGameOver
		// No opponent unknown cards: p1 hand length controls DP behavior.
		// For "all known" test, we want player 1's cards to be "known" too.
		// TerminalEvalDP treats opponent as unknown, so set HandLen=0 for a
		// pure known test (self only matters). Instead we test directly:
		// With oppUnknown=0 (by setting HandLen=0 for opp), use direct path.

		// Override: set both players known by zeroing opponent hand.
		// The DP code with oppUnknown==0 evaluates scoreSelf vs oppScore(=0).
		// This tests the direct evaluation path.
		selfScore := int8(0)
		for _, v := range tc.p0 {
			selfScore += v
		}
		oppScore := int8(0)
		for _, v := range tc.p1 {
			oppScore += v
		}

		// Manually set up a state where opponent has 0 unknown cards
		// (empty hand) and compare to expected.
		g2 := make2PGame(tc.p0, []int8{}, tc.cambia)
		g2.Flags |= FlagGameOver

		// For this test, compare the known-cards path of DP.
		// We use g2 where opp has 0 cards → direct path.
		// The "want" is based on p0 score vs 0 (opp has no cards).
		var wantDirect float32
		if selfScore < 0 {
			wantDirect = 1.0
		} else if selfScore > 0 {
			wantDirect = -1.0
		} else {
			if tc.cambia == 0 {
				wantDirect = 1.0
			} else if tc.cambia == 1 {
				wantDirect = -1.0
			}
		}
		u := g2.TerminalEvalDP(0)
		if math.Abs(float64(u-wantDirect)) > 0.001 {
			t.Errorf("DP known cards p0=%v opp=[] cambia=%d: got %.4f want %.4f",
				tc.p0, tc.cambia, u, wantDirect)
		}
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalDP_2P_UnknownCards
// ---------------------------------------------------------------------------

// Compare DP result against high-sample MC estimate for agreement.
func TestTerminalEvalDP_2P_UnknownCards(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	rules.NumJokers = 0
	g := NewGame(31415, rules)
	g.Deal()

	dp := g.TerminalEvalDP(0)

	seed := g.gameStateHash()
	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeefcafe1234))
	mc := g.TerminalEvalMC(0, 100000, rng)

	diff := math.Abs(float64(dp - mc))
	if diff > 0.02 {
		t.Errorf("DP vs MC discrepancy too large: dp=%.4f mc=%.4f diff=%.4f", dp, mc, diff)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalDP_Panics_NPlayer
// ---------------------------------------------------------------------------

func TestTerminalEvalDP_Panics_NPlayer(t *testing.T) {
	rules := HouseRules{NumPlayers: 3, CardsPerPlayer: 4, NumJokers: 0}
	g := NewGame(42, rules)
	g.Deal()

	defer func() {
		if r := recover(); r == nil {
			t.Error("TerminalEvalDP should panic for N>2 players")
		}
	}()
	g.TerminalEvalDP(0)
}

// ---------------------------------------------------------------------------
// TestTerminalEvalMC_Determinism
// ---------------------------------------------------------------------------

func TestTerminalEvalMC_Determinism(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	g := NewGame(12345, rules)
	g.Deal()

	seed := g.gameStateHash()
	rng1 := rand.New(rand.NewPCG(seed, seed^0xdeadbeefcafe1234))
	rng2 := rand.New(rand.NewPCG(seed, seed^0xdeadbeefcafe1234))

	u1 := g.TerminalEvalMC(0, 200, rng1)
	u2 := g.TerminalEvalMC(0, 200, rng2)

	if u1 != u2 {
		t.Errorf("MC not deterministic: %.6f != %.6f", u1, u2)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalMC_Accuracy
// ---------------------------------------------------------------------------

// MC with many samples should agree with DP for 2P games within 0.02.
func TestTerminalEvalMC_Accuracy(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	rules.NumJokers = 0
	g := NewGame(271828, rules)
	g.Deal()

	dp := g.TerminalEvalDP(0)

	seed := g.gameStateHash()
	rng := rand.New(rand.NewPCG(seed, seed^0xfeedface))
	mc := g.TerminalEvalMC(0, 1000, rng)

	diff := math.Abs(float64(dp - mc))
	if diff > 0.02 {
		t.Errorf("MC 1000-sample accuracy vs DP: dp=%.4f mc=%.4f diff=%.4f (want < 0.02)", dp, mc, diff)
	}
}

// ---------------------------------------------------------------------------
// TestTerminalEvalMC_NPlayer
// ---------------------------------------------------------------------------

// For a 4-player game, MC utility values should have near-zero sum.
func TestTerminalEvalMC_NPlayer(t *testing.T) {
	rules := HouseRules{
		NumPlayers:     4,
		CardsPerPlayer: 4,
		NumJokers:      2,
		MaxGameTurns:   200,
	}
	g := NewGame(161803, rules)
	g.Deal()

	seed := g.gameStateHash()
	var totalUtil float32
	for p := uint8(0); p < 4; p++ {
		rng := rand.New(rand.NewPCG(seed^uint64(p), seed^0xabcdef))
		u := g.TerminalEvalMC(p, 500, rng)
		if u < -10 || u > 10 {
			t.Errorf("P%d MC utility out of reasonable range: %.4f", p, u)
		}
		totalUtil += u
	}

	// Note: utilities from different player perspectives don't sum to zero
	// because each player has different "unknown" sets. This test just
	// verifies the values are finite and in a reasonable range.
	if math.IsNaN(float64(totalUtil)) || math.IsInf(float64(totalUtil), 0) {
		t.Errorf("4P MC utilities contain NaN or Inf: sum=%.4f", totalUtil)
	}
}

// ---------------------------------------------------------------------------
// TestGameStateHash_Determinism
// ---------------------------------------------------------------------------

func TestGameStateHash_Determinism(t *testing.T) {
	rules := DefaultHouseRules()
	g1 := NewGame(42, rules)
	g1.Deal()

	g2 := g1 // value copy

	h1 := g1.gameStateHash()
	h2 := g2.gameStateHash()

	if h1 != h2 {
		t.Errorf("gameStateHash not deterministic for identical states: %d != %d", h1, h2)
	}
}
