package engine

import "testing"

// The exported penalty-draw and reshuffle primitives are the contract adapters use when they
// cannot drive the sequential snap phase (see DrawPenaltyCard's doc comment), so their return
// values and their order of checks are pinned here.

// TestDrawPenaltyCardReshufflesEmptyStockpile verifies a penalty card is still drawn when the
// stockpile has run dry and the discard pile can be reshuffled back into it.
func TestDrawPenaltyCardReshufflesEmptyStockpile(t *testing.T) {
	gs := NewGame(7, DefaultHouseRules())
	gs.Deal()

	gs.StockLen = 0
	gs.DiscardLen = 0
	for i := uint8(0); i < 3; i++ {
		gs.DiscardPile[gs.DiscardLen] = NewCard(SuitHearts, RankAce+i)
		gs.DiscardLen++
	}
	gs.Players[0].HandLen = 2

	if !gs.DrawPenaltyCard(0) {
		t.Fatal("DrawPenaltyCard = false, want true (the discard pile can be reshuffled)")
	}
	if gs.Players[0].HandLen != 3 {
		t.Errorf("HandLen = %d, want 3", gs.Players[0].HandLen)
	}
	if gs.StockLen != 1 {
		t.Errorf("StockLen = %d, want 1 (2 cards reshuffled in, 1 drawn)", gs.StockLen)
	}
	if gs.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d, want 1 (the top card stays)", gs.DiscardLen)
	}
}

// TestDrawPenaltyCardExhaustedDeck verifies a penalty goes unpaid, rather than fabricating a card,
// when the stockpile is empty and the discard pile is too thin to reshuffle.
func TestDrawPenaltyCardExhaustedDeck(t *testing.T) {
	gs := NewGame(7, DefaultHouseRules())
	gs.Deal()

	gs.StockLen = 0
	gs.DiscardLen = 1
	gs.Players[0].HandLen = 2

	if gs.DrawPenaltyCard(0) {
		t.Fatal("DrawPenaltyCard = true, want false (no card left to draw)")
	}
	if gs.Players[0].HandLen != 2 {
		t.Errorf("HandLen = %d, want 2 (unchanged)", gs.Players[0].HandLen)
	}
	if gs.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d, want 1 (unchanged)", gs.DiscardLen)
	}
}

// TestDrawPenaltyCardFullHandDoesNotReshuffle verifies the hand-size cap is checked before the
// stockpile: a player who cannot hold another card must not disturb the piles.
func TestDrawPenaltyCardFullHandDoesNotReshuffle(t *testing.T) {
	gs := NewGame(7, DefaultHouseRules())
	gs.Deal()

	gs.StockLen = 0
	gs.DiscardLen = 0
	for i := uint8(0); i < 3; i++ {
		gs.DiscardPile[gs.DiscardLen] = NewCard(SuitClubs, RankAce+i)
		gs.DiscardLen++
	}
	gs.Players[0].HandLen = MaxHandSize

	if gs.DrawPenaltyCard(0) {
		t.Fatal("DrawPenaltyCard = true, want false (hand is full)")
	}
	if gs.StockLen != 0 || gs.DiscardLen != 3 {
		t.Errorf("StockLen = %d, DiscardLen = %d, want 0 and 3 (piles untouched)", gs.StockLen, gs.DiscardLen)
	}
}

// TestAttemptReshuffleReportsMovedCards verifies the exported reshuffle reports whether it moved
// anything, which is how an adapter knows to rebuild its own mirror of the piles.
func TestAttemptReshuffleReportsMovedCards(t *testing.T) {
	gs := NewGame(7, DefaultHouseRules())
	gs.Deal()

	gs.StockLen = 0
	gs.DiscardLen = 1
	if gs.AttemptReshuffle() {
		t.Error("AttemptReshuffle = true, want false (a lone top card cannot be reshuffled)")
	}

	gs.DiscardLen = 0
	for i := uint8(0); i < 4; i++ {
		gs.DiscardPile[gs.DiscardLen] = NewCard(SuitSpades, RankAce+i)
		gs.DiscardLen++
	}
	if !gs.AttemptReshuffle() {
		t.Fatal("AttemptReshuffle = false, want true")
	}
	if gs.StockLen != 3 {
		t.Errorf("StockLen = %d, want 3", gs.StockLen)
	}
	if gs.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d, want 1", gs.DiscardLen)
	}
}

// TestAttemptReshuffleClampBoundary pins attemptReshuffle's `DiscardLen <= 1` clamp at both sides
// of the boundary: at 0 and 1 cards there is no material to reshuffle (a lone top card has nowhere
// to go), and at 2 the clamp releases and moves exactly the one card below the top into the
// stockpile, leaving the top card as the discard pile's sole survivor.
func TestAttemptReshuffleClampBoundary(t *testing.T) {
	gs := NewGame(7, DefaultHouseRules())
	gs.Deal()

	gs.StockLen = 0
	gs.DiscardLen = 0
	if gs.AttemptReshuffle() {
		t.Error("AttemptReshuffle = true, want false (an empty discard pile has no top to keep and nothing to move)")
	}
	if gs.StockLen != 0 || gs.DiscardLen != 0 {
		t.Errorf("StockLen = %d, DiscardLen = %d, want 0 and 0 (piles untouched)", gs.StockLen, gs.DiscardLen)
	}

	ace := NewCard(SuitHearts, RankAce)
	gs.DiscardPile[0] = ace
	gs.DiscardLen = 1
	if gs.AttemptReshuffle() {
		t.Error("AttemptReshuffle = true, want false (a lone top card cannot be reshuffled)")
	}
	if gs.StockLen != 0 || gs.DiscardLen != 1 {
		t.Errorf("StockLen = %d, DiscardLen = %d, want 0 and 1 (piles untouched)", gs.StockLen, gs.DiscardLen)
	}

	// DiscardPile[DiscardLen-1] is the pile's top (effectiveDiscardTop and attemptReshuffle both
	// index it that way); the two joins as the new top, so it is the card the clamp must keep and
	// the ace is the one card there is material to move.
	two := NewCard(SuitHearts, RankTwo)
	gs.DiscardPile[1] = two
	gs.DiscardLen = 2
	if !gs.AttemptReshuffle() {
		t.Fatal("AttemptReshuffle = false, want true (two discards releases the clamp: one stays, one moves)")
	}
	if gs.StockLen != 1 {
		t.Errorf("StockLen = %d, want 1 (the one non-top card moved)", gs.StockLen)
	}
	if gs.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d, want 1 (the top card stays)", gs.DiscardLen)
	}
	if gs.DiscardPile[0] != two {
		t.Errorf("DiscardPile[0] = %v, want the two to stay on top", gs.DiscardPile[0])
	}
	if gs.Stockpile[0] != ace {
		t.Errorf("Stockpile[0] = %v, want the ace to have moved into the stockpile", gs.Stockpile[0])
	}
}
