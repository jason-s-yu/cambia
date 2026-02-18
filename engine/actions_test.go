package engine

import (
	"testing"
)

// newDealtGame creates a standard game that has been dealt, ready to play.
func newDealtGame(t *testing.T) *GameState {
	t.Helper()
	g := NewGame(42, DefaultHouseRules())
	g.Deal()
	return &g
}

// passSnapPhase advances through the snap phase by having all snappers pass.
// Call this after an action that triggers the snap phase when you don't want to test snap logic.
func passSnapPhase(t *testing.T, g *GameState) {
	t.Helper()
	for g.Snap.Active {
		if err := g.ApplyAction(ActionPassSnap); err != nil {
			t.Fatalf("passSnapPhase: PassSnap failed: %v", err)
		}
	}
}

// TestDrawStockpile verifies that drawing from the stockpile sets up pending state correctly.
func TestDrawStockpile(t *testing.T) {
	g := newDealtGame(t)

	stockBefore := g.StockLen
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile failed: %v", err)
	}

	if g.StockLen != stockBefore-1 {
		t.Errorf("StockLen: want %d, got %d", stockBefore-1, g.StockLen)
	}
	if g.Pending.Type != PendingDiscard {
		t.Errorf("Pending.Type: want PendingDiscard (%d), got %d", PendingDiscard, g.Pending.Type)
	}
	// Pending.Data[0] should hold the drawn card.
	if g.Pending.Data[0] == uint8(EmptyCard) {
		t.Error("Pending.Data[0] should not be EmptyCard after draw")
	}
	if g.Pending.Data[1] != DrawnFromStockpile {
		t.Errorf("Pending.Data[1]: want DrawnFromStockpile (%d), got %d", DrawnFromStockpile, g.Pending.Data[1])
	}
	if g.Pending.PlayerID != g.CurrentPlayer {
		t.Errorf("Pending.PlayerID: want %d, got %d", g.CurrentPlayer, g.Pending.PlayerID)
	}
}

// TestDrawDiscard verifies that drawing from the discard pile sets up pending state correctly.
func TestDrawDiscard(t *testing.T) {
	g := newDealtGame(t)

	discardBefore := g.DiscardLen
	if err := g.ApplyAction(ActionDrawDiscard); err != nil {
		t.Fatalf("DrawDiscard failed: %v", err)
	}

	if g.DiscardLen != discardBefore-1 {
		t.Errorf("DiscardLen: want %d, got %d", discardBefore-1, g.DiscardLen)
	}
	if g.Pending.Type != PendingDiscard {
		t.Errorf("Pending.Type: want PendingDiscard (%d), got %d", PendingDiscard, g.Pending.Type)
	}
	if g.Pending.Data[1] != DrawnFromDiscard {
		t.Errorf("Pending.Data[1]: want DrawnFromDiscard (%d), got %d", DrawnFromDiscard, g.Pending.Data[1])
	}
}

// TestDrawDiscardRejectedWhenNotAllowed verifies that drawing from discard is rejected by house rules.
func TestDrawDiscardRejectedWhenNotAllowed(t *testing.T) {
	rules := DefaultHouseRules()
	rules.AllowDrawFromDiscard = false
	g := NewGame(42, rules)
	g.Deal()

	if err := g.ApplyAction(ActionDrawDiscard); err == nil {
		t.Error("expected error when AllowDrawFromDiscard=false, got nil")
	}
}

// TestDiscardDrawnCard verifies that discarding the drawn card places it on the discard pile.
func TestDiscardDrawnCard(t *testing.T) {
	g := newDealtGame(t)

	// Draw first.
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	drawnCard := Card(g.Pending.Data[0])
	discardBefore := g.DiscardLen
	playerBefore := g.CurrentPlayer

	// Discard without ability.
	if err := g.ApplyAction(ActionDiscardNoAbility); err != nil {
		t.Fatalf("DiscardNoAbility: %v", err)
	}

	// Drawn card should be on top of discard pile.
	if g.DiscardLen != discardBefore+1 {
		t.Errorf("DiscardLen: want %d, got %d", discardBefore+1, g.DiscardLen)
	}
	if top := g.DiscardTop(); top != drawnCard {
		t.Errorf("DiscardTop: want %v, got %v", drawnCard, top)
	}
	// Pending should be cleared.
	if g.Pending.Type != PendingNone {
		t.Errorf("Pending.Type: want PendingNone, got %d", g.Pending.Type)
	}
	// Pass through snap phase so turn advances.
	passSnapPhase(t, g)
	// Turn should have advanced.
	if g.CurrentPlayer == playerBefore {
		t.Error("CurrentPlayer should have changed after discarding")
	}
}

// TestReplaceCard verifies that replacing a hand card works correctly.
func TestReplaceCard(t *testing.T) {
	g := newDealtGame(t)
	acting := g.CurrentPlayer

	// Draw first.
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	drawnCard := Card(g.Pending.Data[0])
	oldCard := g.Players[acting].Hand[0]
	handLenBefore := g.Players[acting].HandLen

	// Replace hand[0] with drawn card.
	if err := g.ApplyAction(EncodeReplace(0)); err != nil {
		t.Fatalf("Replace(0): %v", err)
	}

	// Hand size should be unchanged.
	if g.Players[acting].HandLen != handLenBefore {
		t.Errorf("HandLen: want %d, got %d", handLenBefore, g.Players[acting].HandLen)
	}
	// New card in hand should be the drawn card.
	if g.Players[acting].Hand[0] != drawnCard {
		t.Errorf("Hand[0]: want %v, got %v", drawnCard, g.Players[acting].Hand[0])
	}
	// Old card should be on top of discard pile.
	if top := g.DiscardTop(); top != oldCard {
		t.Errorf("DiscardTop: want old card %v, got %v", oldCard, top)
	}
	// Pending should be cleared.
	if g.Pending.Type != PendingNone {
		t.Errorf("Pending.Type: want PendingNone, got %d", g.Pending.Type)
	}
}

// TestReplaceInvalidIndex verifies that replacing with an out-of-range index returns an error.
func TestReplaceInvalidIndex(t *testing.T) {
	g := newDealtGame(t)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}

	acting := g.Pending.PlayerID
	invalidIdx := g.Players[acting].HandLen // exactly one past the end

	if err := g.ApplyAction(EncodeReplace(invalidIdx)); err == nil {
		t.Error("expected error for out-of-range replace index, got nil")
	}
}

// TestDoubleDrawRejected verifies that drawing again without discarding returns an error.
func TestDoubleDrawRejected(t *testing.T) {
	g := newDealtGame(t)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("First DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDrawStockpile); err == nil {
		t.Error("expected error on second draw without discarding, got nil")
	}
}

// TestDiscardWithoutDraw verifies that discarding without drawing first returns an error.
func TestDiscardWithoutDraw(t *testing.T) {
	g := newDealtGame(t)

	if err := g.ApplyAction(ActionDiscardNoAbility); err == nil {
		t.Error("expected error discarding without drawing first, got nil")
	}
}

// TestTurnAdvance verifies that the current player alternates after each draw-discard cycle.
func TestTurnAdvance(t *testing.T) {
	g := newDealtGame(t)
	player0 := g.CurrentPlayer

	// Player 0 draws and discards.
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("P0 DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardNoAbility); err != nil {
		t.Fatalf("P0 Discard: %v", err)
	}
	passSnapPhase(t, g)

	player1 := g.CurrentPlayer
	if player1 == player0 {
		t.Fatalf("Turn should have advanced: player0=%d, player1=%d", player0, player1)
	}

	// Player 1 draws and discards.
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("P1 DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardNoAbility); err != nil {
		t.Fatalf("P1 Discard: %v", err)
	}
	passSnapPhase(t, g)

	// Should be back to player 0.
	if g.CurrentPlayer != player0 {
		t.Errorf("After 2 turns, expected CurrentPlayer=%d, got %d", player0, g.CurrentPlayer)
	}
	if g.TurnNumber != 2 {
		t.Errorf("TurnNumber: want 2, got %d", g.TurnNumber)
	}
}

// TestGameEndsAtMaxTurns verifies that the game ends when MaxGameTurns is reached.
func TestGameEndsAtMaxTurns(t *testing.T) {
	rules := DefaultHouseRules()
	rules.MaxGameTurns = 4 // small limit for fast test
	gs := NewGame(42, rules)
	gs.Deal()

	for i := 0; i < 4; i++ {
		if gs.IsTerminal() {
			t.Fatalf("game ended early at turn %d", i)
		}
		if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
			t.Fatalf("turn %d DrawStockpile: %v", i, err)
		}
		if err := gs.ApplyAction(ActionDiscardNoAbility); err != nil {
			t.Fatalf("turn %d Discard: %v", i, err)
		}
		passSnapPhase(t, &gs)
	}

	if !gs.IsTerminal() {
		t.Errorf("game should be terminal after %d turns (MaxGameTurns=%d)", gs.TurnNumber, rules.MaxGameTurns)
	}
}

// TestReshuffle verifies that when the stockpile is empty a draw triggers a reshuffle.
func TestReshuffle(t *testing.T) {
	g := newDealtGame(t)

	// Artificially drain the stockpile.
	g.StockLen = 0

	// Add several cards to discard so reshuffle has material.
	// Keep the current top discard card in place (index 0 already exists from Deal).
	g.DiscardPile[1] = NewCard(SuitHearts, RankTwo)
	g.DiscardPile[2] = NewCard(SuitClubs, RankThree)
	g.DiscardLen = 3

	discardTopBefore := g.DiscardPile[g.DiscardLen-1] // the card that stays

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile after artificial drain: %v", err)
	}

	// After reshuffle, discard pile should have only the preserved top card.
	if g.DiscardLen != 1 {
		t.Errorf("DiscardLen after reshuffle: want 1, got %d", g.DiscardLen)
	}
	if g.DiscardPile[0] != discardTopBefore {
		t.Errorf("DiscardPile[0] after reshuffle: want %v, got %v", discardTopBefore, g.DiscardPile[0])
	}

	// Pending should be set (draw succeeded).
	if g.Pending.Type != PendingDiscard {
		t.Errorf("Pending.Type after reshuffle draw: want PendingDiscard, got %d", g.Pending.Type)
	}
}
