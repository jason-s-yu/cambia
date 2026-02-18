package engine

import "testing"

// TestCardValues verifies card point values for all 54 cards.
func TestCardValues(t *testing.T) {
	tests := []struct {
		suit uint8
		rank uint8
		want int8
	}{
		// Jokers
		{SuitRedJoker, RankJoker, 0},
		{SuitBlackJoker, RankJoker, 0},
		// Ace
		{SuitHearts, RankAce, 1},
		{SuitDiamonds, RankAce, 1},
		{SuitClubs, RankAce, 1},
		{SuitSpades, RankAce, 1},
		// Two through Nine (rank+1)
		{SuitHearts, RankTwo, 2},
		{SuitHearts, RankThree, 3},
		{SuitHearts, RankFour, 4},
		{SuitHearts, RankFive, 5},
		{SuitHearts, RankSix, 6},
		{SuitHearts, RankSeven, 7},
		{SuitHearts, RankEight, 8},
		{SuitHearts, RankNine, 9},
		// Ten
		{SuitHearts, RankTen, 10},
		{SuitClubs, RankTen, 10},
		// Jack
		{SuitHearts, RankJack, 11},
		{SuitSpades, RankJack, 11},
		// Queen
		{SuitDiamonds, RankQueen, 12},
		{SuitClubs, RankQueen, 12},
		// Red Kings (Hearts, Diamonds) → -1
		{SuitHearts, RankKing, -1},
		{SuitDiamonds, RankKing, -1},
		// Black Kings (Clubs, Spades) → 13
		{SuitClubs, RankKing, 13},
		{SuitSpades, RankKing, 13},
	}
	for _, tt := range tests {
		c := NewCard(tt.suit, tt.rank)
		got := c.Value()
		if got != tt.want {
			t.Errorf("NewCard(%d,%d).Value() = %d, want %d", tt.suit, tt.rank, got, tt.want)
		}
	}
}

// TestCardSuitRank verifies Suit/Rank roundtrip for every suit×rank combo.
func TestCardSuitRank(t *testing.T) {
	suits := []uint8{SuitHearts, SuitDiamonds, SuitClubs, SuitSpades, SuitRedJoker, SuitBlackJoker}
	ranks := []uint8{RankAce, RankTwo, RankThree, RankFour, RankFive, RankSix,
		RankSeven, RankEight, RankNine, RankTen, RankJack, RankQueen, RankKing, RankJoker}
	for _, s := range suits {
		for _, r := range ranks {
			c := NewCard(s, r)
			if c.Suit() != s {
				t.Errorf("NewCard(%d,%d).Suit() = %d, want %d", s, r, c.Suit(), s)
			}
			if c.Rank() != r {
				t.Errorf("NewCard(%d,%d).Rank() = %d, want %d", s, r, c.Rank(), r)
			}
		}
	}
}

// TestCardAbility verifies ability types for all 54 cards.
func TestCardAbility(t *testing.T) {
	type tc struct {
		suit uint8
		rank uint8
		want AbilityType
	}
	cases := []tc{
		// Jokers — no ability
		{SuitRedJoker, RankJoker, AbilityNone},
		{SuitBlackJoker, RankJoker, AbilityNone},
		// Ace through Six — no ability
		{SuitHearts, RankAce, AbilityNone},
		{SuitHearts, RankTwo, AbilityNone},
		{SuitHearts, RankThree, AbilityNone},
		{SuitHearts, RankFour, AbilityNone},
		{SuitHearts, RankFive, AbilityNone},
		{SuitHearts, RankSix, AbilityNone},
		// Seven, Eight → PeekOwn
		{SuitHearts, RankSeven, AbilityPeekOwn},
		{SuitSpades, RankEight, AbilityPeekOwn},
		// Nine, Ten → PeekOther
		{SuitDiamonds, RankNine, AbilityPeekOther},
		{SuitClubs, RankTen, AbilityPeekOther},
		// Jack, Queen → BlindSwap
		{SuitHearts, RankJack, AbilityBlindSwap},
		{SuitSpades, RankQueen, AbilityBlindSwap},
		// King → KingLook
		{SuitHearts, RankKing, AbilityKingLook},
		{SuitClubs, RankKing, AbilityKingLook},
	}
	for _, tt := range cases {
		c := NewCard(tt.suit, tt.rank)
		got := c.Ability()
		if got != tt.want {
			t.Errorf("NewCard(%d,%d).Ability() = %v, want %v", tt.suit, tt.rank, got, tt.want)
		}
		// HasAbility consistency
		hasAbility := tt.want != AbilityNone
		if c.HasAbility() != hasAbility {
			t.Errorf("NewCard(%d,%d).HasAbility() = %v, want %v", tt.suit, tt.rank, c.HasAbility(), hasAbility)
		}
	}
}

// TestEmptyCard verifies EmptyCard doesn't panic.
func TestEmptyCard(t *testing.T) {
	c := EmptyCard
	_ = c.Suit()
	_ = c.Rank()
	_ = c.Value()
	_ = c.HasAbility()
	_ = c.Ability()
}

// TestActionIndexRoundTrip verifies encode→decode for every action index 0–145.
func TestActionIndexRoundTrip(t *testing.T) {
	// For each action category, encode then decode and verify identity.

	// Replace: indices 5–10
	for i := uint8(0); i < 6; i++ {
		idx := EncodeReplace(i)
		got, ok := ActionIsReplace(idx)
		if !ok || got != i {
			t.Errorf("EncodeReplace(%d) roundtrip failed: got (%d,%v)", i, got, ok)
		}
	}

	// PeekOwn: indices 11–16
	for i := uint8(0); i < 6; i++ {
		idx := EncodePeekOwn(i)
		got, ok := ActionIsPeekOwn(idx)
		if !ok || got != i {
			t.Errorf("EncodePeekOwn(%d) roundtrip failed: got (%d,%v)", i, got, ok)
		}
	}

	// PeekOther: indices 17–22
	for i := uint8(0); i < 6; i++ {
		idx := EncodePeekOther(i)
		got, ok := ActionIsPeekOther(idx)
		if !ok || got != i {
			t.Errorf("EncodePeekOther(%d) roundtrip failed: got (%d,%v)", i, got, ok)
		}
	}

	// BlindSwap: indices 23–58
	for own := uint8(0); own < 6; own++ {
		for opp := uint8(0); opp < 6; opp++ {
			idx := EncodeBlindSwap(own, opp)
			gOwn, gOpp, ok := ActionIsBlindSwap(idx)
			if !ok || gOwn != own || gOpp != opp {
				t.Errorf("EncodeBlindSwap(%d,%d) roundtrip failed: got (%d,%d,%v)", own, opp, gOwn, gOpp, ok)
			}
		}
	}

	// KingLook: indices 59–94
	for own := uint8(0); own < 6; own++ {
		for opp := uint8(0); opp < 6; opp++ {
			idx := EncodeKingLook(own, opp)
			gOwn, gOpp, ok := ActionIsKingLook(idx)
			if !ok || gOwn != own || gOpp != opp {
				t.Errorf("EncodeKingLook(%d,%d) roundtrip failed: got (%d,%d,%v)", own, opp, gOwn, gOpp, ok)
			}
		}
	}

	// KingSwapNo / KingSwapYes
	if swap, ok := ActionIsKingSwap(ActionKingSwapNo); !ok || swap {
		t.Errorf("KingSwapNo roundtrip failed: swap=%v ok=%v", swap, ok)
	}
	if swap, ok := ActionIsKingSwap(ActionKingSwapYes); !ok || !swap {
		t.Errorf("KingSwapYes roundtrip failed: swap=%v ok=%v", swap, ok)
	}

	// SnapOwn: indices 98–103
	for i := uint8(0); i < 6; i++ {
		idx := EncodeSnapOwn(i)
		got, ok := ActionIsSnapOwn(idx)
		if !ok || got != i {
			t.Errorf("EncodeSnapOwn(%d) roundtrip failed: got (%d,%v)", i, got, ok)
		}
	}

	// SnapOpponent: indices 104–109
	for i := uint8(0); i < 6; i++ {
		idx := EncodeSnapOpponent(i)
		got, ok := ActionIsSnapOpponent(idx)
		if !ok || got != i {
			t.Errorf("EncodeSnapOpponent(%d) roundtrip failed: got (%d,%v)", i, got, ok)
		}
	}

	// SnapOpponentMove: indices 110–145
	for own := uint8(0); own < 6; own++ {
		for slot := uint8(0); slot < 6; slot++ {
			idx := EncodeSnapOpponentMove(own, slot)
			gOwn, gSlot, ok := ActionIsSnapOpponentMove(idx)
			if !ok || gOwn != own || gSlot != slot {
				t.Errorf("EncodeSnapOpponentMove(%d,%d) roundtrip failed: got (%d,%d,%v)", own, slot, gOwn, gSlot, ok)
			}
		}
	}
}

// TestActionBounds verifies all encode functions produce indices in [0, 145].
func TestActionBounds(t *testing.T) {
	check := func(label string, idx uint16) {
		if idx >= NumActions {
			t.Errorf("%s produced out-of-range index %d (NumActions=%d)", label, idx, NumActions)
		}
	}
	for i := uint8(0); i < 6; i++ {
		check("EncodeReplace", EncodeReplace(i))
		check("EncodePeekOwn", EncodePeekOwn(i))
		check("EncodePeekOther", EncodePeekOther(i))
		check("EncodeSnapOwn", EncodeSnapOwn(i))
		check("EncodeSnapOpponent", EncodeSnapOpponent(i))
	}
	for own := uint8(0); own < 6; own++ {
		for opp := uint8(0); opp < 6; opp++ {
			check("EncodeBlindSwap", EncodeBlindSwap(own, opp))
			check("EncodeKingLook", EncodeKingLook(own, opp))
			check("EncodeSnapOpponentMove", EncodeSnapOpponentMove(own, opp))
		}
	}
	check("KingSwapNo", ActionKingSwapNo)
	check("KingSwapYes", ActionKingSwapYes)
	check("PassSnap", ActionPassSnap)
}

// TestActionDecodeInvalid verifies that indices >= 146 return ok=false.
func TestActionDecodeInvalid(t *testing.T) {
	invalid := []uint16{146, 147, 200, 0xFFFF}
	for _, idx := range invalid {
		if _, ok := ActionIsReplace(idx); ok {
			t.Errorf("ActionIsReplace(%d) should return ok=false", idx)
		}
		if _, ok := ActionIsPeekOwn(idx); ok {
			t.Errorf("ActionIsPeekOwn(%d) should return ok=false", idx)
		}
		if _, ok := ActionIsPeekOther(idx); ok {
			t.Errorf("ActionIsPeekOther(%d) should return ok=false", idx)
		}
		if _, _, ok := ActionIsBlindSwap(idx); ok {
			t.Errorf("ActionIsBlindSwap(%d) should return ok=false", idx)
		}
		if _, _, ok := ActionIsKingLook(idx); ok {
			t.Errorf("ActionIsKingLook(%d) should return ok=false", idx)
		}
		if _, ok := ActionIsKingSwap(idx); ok {
			t.Errorf("ActionIsKingSwap(%d) should return ok=false", idx)
		}
		if _, ok := ActionIsSnapOwn(idx); ok {
			t.Errorf("ActionIsSnapOwn(%d) should return ok=false", idx)
		}
		if _, ok := ActionIsSnapOpponent(idx); ok {
			t.Errorf("ActionIsSnapOpponent(%d) should return ok=false", idx)
		}
		if _, _, ok := ActionIsSnapOpponentMove(idx); ok {
			t.Errorf("ActionIsSnapOpponentMove(%d) should return ok=false", idx)
		}
	}
}
