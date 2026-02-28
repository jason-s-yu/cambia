package engine

import (
	"testing"
)

// TestNumDecksInterface verifies NumDecks is present in HouseRules.
func TestNumDecksInterface(t *testing.T) {
	r := DefaultHouseRules()
	if r.NumDecks != 1 {
		t.Errorf("DefaultHouseRules().NumDecks = %d, want 1", r.NumDecks)
	}
}

// TestNumDecksRegression verifies NumDecks=1 produces the same card count as before.
func TestNumDecksRegression(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 1
	g := NewGame(42, rules)
	if g.StockLen != 54 {
		t.Errorf("NumDecks=1: StockLen = %d, want 54", g.StockLen)
	}
}

// TestNumDecksRegressionNoJokers verifies NumDecks=1, NumJokers=0 → 52 cards.
func TestNumDecksRegressionNoJokers(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 1
	rules.NumJokers = 0
	g := NewGame(42, rules)
	if g.StockLen != 52 {
		t.Errorf("NumDecks=1, NumJokers=0: StockLen = %d, want 52", g.StockLen)
	}
}

// TestNumDecksTwo verifies NumDecks=2 → 108 cards with 2 jokers.
func TestNumDecksTwo(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 2
	g := NewGame(42, rules)
	if g.StockLen != 108 {
		t.Errorf("NumDecks=2: StockLen = %d, want 108", g.StockLen)
	}
}

// TestNumDecksTwoNoJokers verifies NumDecks=2, NumJokers=0 → 104 cards.
func TestNumDecksTwoNoJokers(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 2
	rules.NumJokers = 0
	g := NewGame(42, rules)
	if g.StockLen != 104 {
		t.Errorf("NumDecks=2, NumJokers=0: StockLen = %d, want 104", g.StockLen)
	}
}

// TestNumDecksFour verifies NumDecks=4 → 216 cards with 2 jokers.
func TestNumDecksFour(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 4
	g := NewGame(42, rules)
	if g.StockLen != 216 {
		t.Errorf("NumDecks=4: StockLen = %d, want 216", g.StockLen)
	}
}

// TestNumDecksZeroTreatedAsOne verifies NumDecks=0 is treated as 1.
func TestNumDecksZeroTreatedAsOne(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 0
	g := NewGame(42, rules)
	if g.StockLen != 54 {
		t.Errorf("NumDecks=0: StockLen = %d, want 54 (treated as 1)", g.StockLen)
	}
}

// TestNumDecksGamePlayable verifies a game with NumDecks=2 can be dealt and played.
func TestNumDecksGamePlayable(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 2
	rules.NumPlayers = 2
	g := NewGame(42, rules)
	g.Deal()
	if g.IsTerminal() {
		t.Error("game is terminal immediately after deal")
	}
	// Verify each player got cards
	for p := 0; p < 2; p++ {
		if g.Players[p].HandLen != rules.CardsPerPlayer {
			t.Errorf("player %d HandLen = %d, want %d", p, g.Players[p].HandLen, rules.CardsPerPlayer)
		}
	}
	// Verify discard pile has exactly 1 card
	if g.DiscardLen != 1 {
		t.Errorf("DiscardLen = %d after deal, want 1", g.DiscardLen)
	}
	// Verify stock still has cards: 108 - 2*4 - 1 = 99
	expectedStock := uint8(108 - int(rules.CardsPerPlayer)*2 - 1)
	if g.StockLen != expectedStock {
		t.Errorf("StockLen = %d after deal, want %d", g.StockLen, expectedStock)
	}
}

// TestNumDecksDuplicateCards verifies that NumDecks=2 produces duplicate cards (e.g., two Ace of Spades).
func TestNumDecksDuplicateCards(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumDecks = 2
	rules.NumJokers = 0
	g := NewGame(42, rules)

	// Count occurrences of each card
	counts := make(map[Card]int)
	for i := 0; i < int(g.StockLen); i++ {
		counts[g.Stockpile[i]]++
	}

	// Every card should appear exactly twice
	for card, count := range counts {
		if count != 2 {
			t.Errorf("card %v appears %d times, want 2", card, count)
		}
	}
}
