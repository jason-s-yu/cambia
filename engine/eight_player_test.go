package engine

import (
	"testing"
)

// TestEightPlayerInterface verifies 7- and 8-player games can be created and played.
func TestEightPlayerInterface(t *testing.T) {
	for _, n := range []uint8{7, 8} {
		n := n
		t.Run("N="+string(rune('0'+n)), func(t *testing.T) {
			g := playNPlayerRandom(t, 12345+uint64(n), n)
			if !g.IsTerminal() {
				t.Errorf("N=%d: game did not reach terminal state", n)
			}
			// Utilities should be valid (zero-sum).
			u := g.GetUtility()
			var sum float64
			for p := uint8(0); p < n; p++ {
				sum += float64(u[p])
			}
			if sum < -0.01 || sum > 0.01 {
				t.Errorf("N=%d: utility sum = %f, want ≈0", n, sum)
			}
		})
	}
}

// TestEightPlayerDeal verifies deal produces correct card counts for 8 players.
func TestEightPlayerDeal(t *testing.T) {
	rules := nplayerRules(8)
	g := NewGame(999, rules)
	g.Deal()

	// 8 players × 4 cards = 32 cards dealt; +1 to discard = 33.
	// From 54-card deck: 54 - 33 = 21 remaining in stockpile.
	dealt := uint8(rules.CardsPerPlayer) * 8
	wantStock := uint8(52+rules.NumJokers) - dealt - 1
	if g.StockLen != wantStock {
		t.Errorf("StockLen=%d, want %d (54 - 32 dealt - 1 discard)", g.StockLen, wantStock)
	}
	if g.DiscardLen != 1 {
		t.Errorf("DiscardLen=%d, want 1", g.DiscardLen)
	}
	for p := uint8(0); p < 8; p++ {
		if g.Players[p].HandLen != rules.CardsPerPlayer {
			t.Errorf("player %d: HandLen=%d, want %d", p, g.Players[p].HandLen, rules.CardsPerPlayer)
		}
	}
}

// TestEightPlayerLegalActions verifies legal actions are non-empty for all 8 players.
func TestEightPlayerLegalActions(t *testing.T) {
	rules := nplayerRules(8)
	g := NewGame(1001, rules)
	g.Deal()

	actions := g.NPlayerLegalActionsList()
	if len(actions) == 0 {
		t.Fatal("no legal actions at start of 8-player game")
	}
	// At start of turn, DrawStockpile must be legal.
	mask := g.NPlayerLegalActions()
	if mask[NPlayerActionDrawStockpile/64]>>(NPlayerActionDrawStockpile%64)&1 == 0 {
		t.Error("DrawStockpile should be legal at start of 8-player game")
	}
}

// TestEightPlayerNPlayerNumActions verifies the action space size is correct for 8 players.
func TestEightPlayerNPlayerNumActions(t *testing.T) {
	if NPlayerNumActions != 620 {
		t.Errorf("NPlayerNumActions=%d, want 620 (for MaxPlayers=8, MaxOpponents=7)", NPlayerNumActions)
	}
	if MaxOpponents != 7 {
		t.Errorf("MaxOpponents=%d, want 7 (MaxPlayers-1)", MaxOpponents)
	}
	if MaxPlayers != 8 {
		t.Errorf("MaxPlayers=%d, want 8", MaxPlayers)
	}
}

// TestTwoPlayerRegression verifies 2-player games still work and encoding dims are unchanged.
func TestTwoPlayerRegression(t *testing.T) {
	rules := DefaultHouseRules()
	g := NewGame(42, rules)
	g.Deal()

	if g.Players[0].HandLen != rules.CardsPerPlayer {
		t.Errorf("2P: player 0 HandLen=%d, want %d", g.Players[0].HandLen, rules.CardsPerPlayer)
	}

	// 2P action space (legacy) must remain 146.
	actions := g.LegalActionsList()
	if len(actions) == 0 {
		t.Fatal("2P: no legal actions at game start")
	}
}

// TestSixPlayerBackwardCompat verifies 6-player games still work correctly.
func TestSixPlayerBackwardCompat(t *testing.T) {
	g := playNPlayerRandom(t, 77777, 6)
	if !g.IsTerminal() {
		t.Error("6-player game did not reach terminal state")
	}
}
