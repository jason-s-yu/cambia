package engine

import (
	"testing"
)

// TestInitialViewCountDefault verifies that the default (2) matches legacy behavior.
func TestInitialViewCountDefault(t *testing.T) {
	rules := DefaultHouseRules()
	if rules.InitialViewCount != 2 {
		t.Fatalf("DefaultHouseRules().InitialViewCount = %d, want 2", rules.InitialViewCount)
	}

	g := NewGame(42, rules)
	g.Deal()

	n := rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		ps := g.Players[p]
		if ps.InitialPeekCount != 2 {
			t.Errorf("player %d: InitialPeekCount = %d, want 2", p, ps.InitialPeekCount)
		}
		if ps.InitialPeek[0] != 0 {
			t.Errorf("player %d: InitialPeek[0] = %d, want 0", p, ps.InitialPeek[0])
		}
		if ps.InitialPeek[1] != 1 {
			t.Errorf("player %d: InitialPeek[1] = %d, want 1", p, ps.InitialPeek[1])
		}
	}
}

// TestInitialViewCountZero verifies that 0 means no initial peeks.
func TestInitialViewCountZero(t *testing.T) {
	rules := DefaultHouseRules()
	rules.InitialViewCount = 0

	g := NewGame(42, rules)
	g.Deal()

	n := rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		ps := g.Players[p]
		if ps.InitialPeekCount != 0 {
			t.Errorf("player %d: InitialPeekCount = %d, want 0", p, ps.InitialPeekCount)
		}
	}
}

// TestInitialViewCountFour verifies that 4 peeks all cards when CardsPerPlayer=4.
func TestInitialViewCountFour(t *testing.T) {
	rules := DefaultHouseRules()
	rules.InitialViewCount = 4
	rules.CardsPerPlayer = 4

	g := NewGame(42, rules)
	g.Deal()

	n := rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		ps := g.Players[p]
		if ps.InitialPeekCount != 4 {
			t.Errorf("player %d: InitialPeekCount = %d, want 4", p, ps.InitialPeekCount)
		}
		for i := uint8(0); i < 4; i++ {
			if ps.InitialPeek[i] != i {
				t.Errorf("player %d: InitialPeek[%d] = %d, want %d", p, i, ps.InitialPeek[i], i)
			}
		}
	}
}

// TestInitialViewCountClampedToCardsPerPlayer verifies clamping when InitialViewCount > CardsPerPlayer.
func TestInitialViewCountClampedToCardsPerPlayer(t *testing.T) {
	rules := DefaultHouseRules()
	rules.InitialViewCount = 6
	rules.CardsPerPlayer = 4

	g := NewGame(42, rules)
	g.Deal()

	n := rules.numPlayers()
	for p := uint8(0); p < n; p++ {
		ps := g.Players[p]
		if ps.InitialPeekCount != 4 {
			t.Errorf("player %d: InitialPeekCount = %d, want 4 (clamped)", p, ps.InitialPeekCount)
		}
	}
}
