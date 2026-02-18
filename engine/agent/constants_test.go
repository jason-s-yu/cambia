package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// TestEnumValues asserts exact integer values match Python enum values.
func TestEnumValues(t *testing.T) {
	// CardBucket values must match Python CardBucket (compacted: 0-9)
	if BucketZero != 0 {
		t.Errorf("BucketZero = %d, want 0", BucketZero)
	}
	if BucketNegKing != 1 {
		t.Errorf("BucketNegKing = %d, want 1", BucketNegKing)
	}
	if BucketAce != 2 {
		t.Errorf("BucketAce = %d, want 2", BucketAce)
	}
	if BucketLowNum != 3 {
		t.Errorf("BucketLowNum = %d, want 3", BucketLowNum)
	}
	if BucketMidNum != 4 {
		t.Errorf("BucketMidNum = %d, want 4", BucketMidNum)
	}
	if BucketPeekSelf != 5 {
		t.Errorf("BucketPeekSelf = %d, want 5", BucketPeekSelf)
	}
	if BucketPeekOther != 6 {
		t.Errorf("BucketPeekOther = %d, want 6", BucketPeekOther)
	}
	if BucketSwapBlind != 7 {
		t.Errorf("BucketSwapBlind = %d, want 7", BucketSwapBlind)
	}
	if BucketHighKing != 8 {
		t.Errorf("BucketHighKing = %d, want 8", BucketHighKing)
	}
	if BucketUnknown != 9 {
		t.Errorf("BucketUnknown = %d, want 9", BucketUnknown)
	}

	// DecayCategory values (0-3)
	if DecayLikelyLow != 0 {
		t.Errorf("DecayLikelyLow = %d, want 0", DecayLikelyLow)
	}
	if DecayLikelyMid != 1 {
		t.Errorf("DecayLikelyMid = %d, want 1", DecayLikelyMid)
	}
	if DecayLikelyHigh != 2 {
		t.Errorf("DecayLikelyHigh = %d, want 2", DecayLikelyHigh)
	}
	if DecayUnknown != 3 {
		t.Errorf("DecayUnknown = %d, want 3", DecayUnknown)
	}

	// StockpileEstimate values (0-3)
	if StockHigh != 0 {
		t.Errorf("StockHigh = %d, want 0", StockHigh)
	}
	if StockMedium != 1 {
		t.Errorf("StockMedium = %d, want 1", StockMedium)
	}
	if StockLow != 2 {
		t.Errorf("StockLow = %d, want 2", StockLow)
	}
	if StockEmpty != 3 {
		t.Errorf("StockEmpty = %d, want 3", StockEmpty)
	}

	// GamePhase values (0-5)
	if PhaseStart != 0 {
		t.Errorf("PhaseStart = %d, want 0", PhaseStart)
	}
	if PhaseEarly != 1 {
		t.Errorf("PhaseEarly = %d, want 1", PhaseEarly)
	}
	if PhaseMid != 2 {
		t.Errorf("PhaseMid = %d, want 2", PhaseMid)
	}
	if PhaseLate != 3 {
		t.Errorf("PhaseLate = %d, want 3", PhaseLate)
	}
	if PhaseCambiaCalled != 4 {
		t.Errorf("PhaseCambiaCalled = %d, want 4", PhaseCambiaCalled)
	}
	if PhaseTerminal != 5 {
		t.Errorf("PhaseTerminal = %d, want 5", PhaseTerminal)
	}
}

// TestCardToBucket tests that every card rank/suit maps to the correct bucket.
func TestCardToBucket(t *testing.T) {
	suits := []uint8{
		engine.SuitHearts, engine.SuitDiamonds, engine.SuitClubs, engine.SuitSpades,
	}

	// All Aces → BucketAce
	for _, s := range suits {
		c := engine.NewCard(s, engine.RankAce)
		if got := CardToBucket(c); got != BucketAce {
			t.Errorf("Ace(%d) → %d, want BucketAce(%d)", s, got, BucketAce)
		}
	}

	// 2-4 → BucketLowNum
	for _, r := range []uint8{engine.RankTwo, engine.RankThree, engine.RankFour} {
		for _, s := range suits {
			c := engine.NewCard(s, r)
			if got := CardToBucket(c); got != BucketLowNum {
				t.Errorf("rank%d(%d) → %d, want BucketLowNum(%d)", r, s, got, BucketLowNum)
			}
		}
	}

	// 5-6 → BucketMidNum
	for _, r := range []uint8{engine.RankFive, engine.RankSix} {
		for _, s := range suits {
			c := engine.NewCard(s, r)
			if got := CardToBucket(c); got != BucketMidNum {
				t.Errorf("rank%d(%d) → %d, want BucketMidNum(%d)", r, s, got, BucketMidNum)
			}
		}
	}

	// 7-8 → BucketPeekSelf
	for _, r := range []uint8{engine.RankSeven, engine.RankEight} {
		for _, s := range suits {
			c := engine.NewCard(s, r)
			if got := CardToBucket(c); got != BucketPeekSelf {
				t.Errorf("rank%d(%d) → %d, want BucketPeekSelf(%d)", r, s, got, BucketPeekSelf)
			}
		}
	}

	// 9-T → BucketPeekOther
	for _, r := range []uint8{engine.RankNine, engine.RankTen} {
		for _, s := range suits {
			c := engine.NewCard(s, r)
			if got := CardToBucket(c); got != BucketPeekOther {
				t.Errorf("rank%d(%d) → %d, want BucketPeekOther(%d)", r, s, got, BucketPeekOther)
			}
		}
	}

	// J-Q → BucketSwapBlind
	for _, r := range []uint8{engine.RankJack, engine.RankQueen} {
		for _, s := range suits {
			c := engine.NewCard(s, r)
			if got := CardToBucket(c); got != BucketSwapBlind {
				t.Errorf("rank%d(%d) → %d, want BucketSwapBlind(%d)", r, s, got, BucketSwapBlind)
			}
		}
	}

	// Red Kings (Hearts, Diamonds) → BucketNegKing
	for _, s := range []uint8{engine.SuitHearts, engine.SuitDiamonds} {
		c := engine.NewCard(s, engine.RankKing)
		if got := CardToBucket(c); got != BucketNegKing {
			t.Errorf("RedKing(%d) → %d, want BucketNegKing(%d)", s, got, BucketNegKing)
		}
	}

	// Black Kings (Clubs, Spades) → BucketHighKing
	for _, s := range []uint8{engine.SuitClubs, engine.SuitSpades} {
		c := engine.NewCard(s, engine.RankKing)
		if got := CardToBucket(c); got != BucketHighKing {
			t.Errorf("BlackKing(%d) → %d, want BucketHighKing(%d)", s, got, BucketHighKing)
		}
	}

	// Jokers → BucketZero
	for _, s := range []uint8{engine.SuitRedJoker, engine.SuitBlackJoker} {
		c := engine.NewCard(s, engine.RankJoker)
		if got := CardToBucket(c); got != BucketZero {
			t.Errorf("Joker(%d) → %d, want BucketZero(%d)", s, got, BucketZero)
		}
	}
}

// TestBucketToDecay tests all 10 buckets map to correct decay category.
func TestBucketToDecay(t *testing.T) {
	cases := []struct {
		bucket CardBucket
		want   DecayCategory
	}{
		{BucketZero, DecayLikelyLow},
		{BucketNegKing, DecayLikelyLow},
		{BucketAce, DecayLikelyLow},
		{BucketLowNum, DecayLikelyLow},
		{BucketMidNum, DecayLikelyMid},
		{BucketPeekSelf, DecayLikelyMid},
		{BucketPeekOther, DecayLikelyHigh},
		{BucketSwapBlind, DecayLikelyHigh},
		{BucketHighKing, DecayLikelyHigh},
		{BucketUnknown, DecayUnknown},
	}

	for _, tc := range cases {
		if got := BucketToDecay(tc.bucket); got != tc.want {
			t.Errorf("BucketToDecay(%d) = %d, want %d", tc.bucket, got, tc.want)
		}
	}
}

// TestStockEstimateFromSize tests boundary values for stockpile estimation.
// Thresholds match Python: >=27 → High, >=10 → Medium, >0 → Low, 0 → Empty.
func TestStockEstimateFromSize(t *testing.T) {
	cases := []struct {
		size uint8
		want StockpileEstimate
	}{
		{0, StockEmpty},
		{1, StockLow},
		{9, StockLow},
		{10, StockMedium},  // boundary: >=10
		{26, StockMedium},
		{27, StockHigh},    // boundary: >=27
		{40, StockHigh},
		{54, StockHigh},
	}

	for _, tc := range cases {
		if got := StockEstimateFromSize(tc.size); got != tc.want {
			t.Errorf("StockEstimateFromSize(%d) = %d, want %d", tc.size, got, tc.want)
		}
	}
}

// TestGamePhaseFromState tests all phases including priority of terminal and cambia.
// Now stockpile-based: >=27 → Early, >=10 → Mid, <10 → Late.
func TestGamePhaseFromState(t *testing.T) {
	cases := []struct {
		stockLen uint8
		cambia   bool
		gameOver bool
		want     GamePhase
	}{
		// Terminal takes priority over everything
		{0, false, true, PhaseTerminal},
		{30, true, true, PhaseTerminal},
		{50, false, true, PhaseTerminal},

		// Cambia called (not game over)
		{0, true, false, PhaseCambiaCalled},
		{40, true, false, PhaseCambiaCalled},

		// Stockpile-based phases (no cambia, not terminal)
		{40, false, false, PhaseEarly},  // >=27 → Early
		{27, false, false, PhaseEarly},  // ==27 → Early (boundary)
		{26, false, false, PhaseMid},    // <27 → Mid
		{15, false, false, PhaseMid},    // >=10 → Mid
		{10, false, false, PhaseMid},    // ==10 → Mid (boundary)
		{9, false, false, PhaseLate},    // <10 → Late
		{1, false, false, PhaseLate},    // >0 → Late
		{0, false, false, PhaseLate},    // ==0 → Late (empty)
	}

	for _, tc := range cases {
		got := GamePhaseFromState(tc.stockLen, tc.cambia, tc.gameOver)
		if got != tc.want {
			t.Errorf("GamePhaseFromState(stockLen=%d, cambia=%v, gameOver=%v) = %d, want %d",
				tc.stockLen, tc.cambia, tc.gameOver, got, tc.want)
		}
	}
}
