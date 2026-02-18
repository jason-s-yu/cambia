package engine

import (
	"testing"
)

// setupAbilityGame creates a dealt game and forces the current player to be player 0.
// It also fills the stockpile top with the given card so that the next draw returns it.
func setupAbilityGame(t *testing.T, forcedDrawCard Card) *GameState {
	t.Helper()
	gs := NewGame(42, DefaultHouseRules())
	gs.Deal()
	gs.CurrentPlayer = 0
	// Place the desired card at the top of the stockpile (last element).
	gs.Stockpile[gs.StockLen] = forcedDrawCard
	gs.StockLen++
	return &gs
}

// drawForcedCard draws from stockpile (pops the card we placed at the top).
func drawForcedCard(t *testing.T, g *GameState) {
	t.Helper()
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile failed: %v", err)
	}
}

// TestPeekOwn verifies the full PeekOwn ability flow.
func TestPeekOwn(t *testing.T) {
	// Seven and Eight both trigger PeekOwn.
	for _, rank := range []uint8{RankSeven, RankEight} {
		t.Run(rankName(rank), func(t *testing.T) {
			sevenCard := NewCard(SuitHearts, rank)
			g := setupAbilityGame(t, sevenCard)

			origPlayer := g.CurrentPlayer
			drawForcedCard(t, g)

			// Verify we drew the correct card.
			if Card(g.Pending.Data[0]) != sevenCard {
				t.Fatalf("expected drawn card %v, got %v", sevenCard, Card(g.Pending.Data[0]))
			}

			// Discard with ability.
			if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
				t.Fatalf("DiscardWithAbility failed: %v", err)
			}

			// Should now be in PendingPeekOwn state.
			if g.Pending.Type != PendingPeekOwn {
				t.Fatalf("expected PendingPeekOwn, got %d", g.Pending.Type)
			}
			if g.Pending.PlayerID != origPlayer {
				t.Fatalf("expected PlayerID %d, got %d", origPlayer, g.Pending.PlayerID)
			}

			// Verify card is on the discard pile.
			if g.DiscardTop() != sevenCard {
				t.Fatalf("expected discard top %v, got %v", sevenCard, g.DiscardTop())
			}

			// Record hand card we'll peek.
			peekTarget := uint8(0)
			expectedCard := g.Players[origPlayer].Hand[peekTarget]

			turnBefore := g.TurnNumber

			// Perform PeekOwn on card at index 0.
			if err := g.ApplyAction(EncodePeekOwn(peekTarget)); err != nil {
				t.Fatalf("PeekOwn failed: %v", err)
			}

			// Pending should be cleared.
			if g.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone after PeekOwn, got %d", g.Pending.Type)
			}

			// Pass through snap phase (initiated for the discarded ability card).
			passSnapPhase(t, g)

			// Turn should have advanced.
			if g.TurnNumber != turnBefore+1 {
				t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
			}

			// LastAction should record the revealed card.
			if g.LastAction.RevealedCard != expectedCard {
				t.Errorf("RevealedCard: want %v, got %v", expectedCard, g.LastAction.RevealedCard)
			}
			if g.LastAction.RevealedIdx != peekTarget {
				t.Errorf("RevealedIdx: want %d, got %d", peekTarget, g.LastAction.RevealedIdx)
			}
			if g.LastAction.RevealedOwner != origPlayer {
				t.Errorf("RevealedOwner: want %d, got %d", origPlayer, g.LastAction.RevealedOwner)
			}
		})
	}
}

// TestPeekOther verifies the full PeekOther ability flow.
func TestPeekOther(t *testing.T) {
	for _, rank := range []uint8{RankNine, RankTen} {
		t.Run(rankName(rank), func(t *testing.T) {
			nineCard := NewCard(SuitHearts, rank)
			g := setupAbilityGame(t, nineCard)

			origPlayer := g.CurrentPlayer // 0
			opp := g.OpponentOf(origPlayer) // 1

			drawForcedCard(t, g)
			if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
				t.Fatalf("DiscardWithAbility failed: %v", err)
			}

			if g.Pending.Type != PendingPeekOther {
				t.Fatalf("expected PendingPeekOther, got %d", g.Pending.Type)
			}

			peekTarget := uint8(0)
			expectedCard := g.Players[opp].Hand[peekTarget]

			turnBefore := g.TurnNumber
			if err := g.ApplyAction(EncodePeekOther(peekTarget)); err != nil {
				t.Fatalf("PeekOther failed: %v", err)
			}

			if g.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone, got %d", g.Pending.Type)
			}
			passSnapPhase(t, g)
			if g.TurnNumber != turnBefore+1 {
				t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
			}
			if g.LastAction.RevealedCard != expectedCard {
				t.Errorf("RevealedCard: want %v, got %v", expectedCard, g.LastAction.RevealedCard)
			}
			if g.LastAction.RevealedOwner != opp {
				t.Errorf("RevealedOwner: want %d (opponent), got %d", opp, g.LastAction.RevealedOwner)
			}
		})
	}
}

// TestBlindSwap verifies the full BlindSwap ability flow.
func TestBlindSwap(t *testing.T) {
	for _, rank := range []uint8{RankJack, RankQueen} {
		t.Run(rankName(rank), func(t *testing.T) {
			jackCard := NewCard(SuitHearts, rank)
			g := setupAbilityGame(t, jackCard)

			acting := g.CurrentPlayer   // 0
			opp := g.OpponentOf(acting) // 1

			ownIdx := uint8(0)
			oppIdx := uint8(1)

			originalOwn := g.Players[acting].Hand[ownIdx]
			originalOpp := g.Players[opp].Hand[oppIdx]

			drawForcedCard(t, g)
			if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
				t.Fatalf("DiscardWithAbility failed: %v", err)
			}

			if g.Pending.Type != PendingBlindSwap {
				t.Fatalf("expected PendingBlindSwap, got %d", g.Pending.Type)
			}

			turnBefore := g.TurnNumber
			if err := g.ApplyAction(EncodeBlindSwap(ownIdx, oppIdx)); err != nil {
				t.Fatalf("BlindSwap failed: %v", err)
			}

			// Cards should be swapped.
			if g.Players[acting].Hand[ownIdx] != originalOpp {
				t.Errorf("acting hand[%d]: want %v, got %v", ownIdx, originalOpp, g.Players[acting].Hand[ownIdx])
			}
			if g.Players[opp].Hand[oppIdx] != originalOwn {
				t.Errorf("opp hand[%d]: want %v, got %v", oppIdx, originalOwn, g.Players[opp].Hand[oppIdx])
			}

			if g.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone, got %d", g.Pending.Type)
			}
			passSnapPhase(t, g)
			if g.TurnNumber != turnBefore+1 {
				t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
			}

			// Swap info recorded in LastAction.
			if g.LastAction.SwapOwnIdx != ownIdx {
				t.Errorf("SwapOwnIdx: want %d, got %d", ownIdx, g.LastAction.SwapOwnIdx)
			}
			if g.LastAction.SwapOppIdx != oppIdx {
				t.Errorf("SwapOppIdx: want %d, got %d", oppIdx, g.LastAction.SwapOppIdx)
			}
		})
	}
}

// TestKingLookThenSwap tests the full King ability when the player chooses to swap.
func TestKingLookThenSwap(t *testing.T) {
	kingCard := NewCard(SuitClubs, RankKing) // Black King
	g := setupAbilityGame(t, kingCard)

	acting := g.CurrentPlayer
	opp := g.OpponentOf(acting)

	ownIdx := uint8(0)
	oppIdx := uint8(1)

	originalOwn := g.Players[acting].Hand[ownIdx]
	originalOpp := g.Players[opp].Hand[oppIdx]

	drawForcedCard(t, g)
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility failed: %v", err)
	}

	if g.Pending.Type != PendingKingLook {
		t.Fatalf("expected PendingKingLook, got %d", g.Pending.Type)
	}

	// King Look step.
	if err := g.ApplyAction(EncodeKingLook(ownIdx, oppIdx)); err != nil {
		t.Fatalf("KingLook failed: %v", err)
	}

	// Should now be waiting for the swap decision.
	if g.Pending.Type != PendingKingDecision {
		t.Fatalf("expected PendingKingDecision after KingLook, got %d", g.Pending.Type)
	}
	if g.Pending.Data[0] != ownIdx {
		t.Errorf("Pending.Data[0] (ownIdx): want %d, got %d", ownIdx, g.Pending.Data[0])
	}
	if g.Pending.Data[1] != oppIdx {
		t.Errorf("Pending.Data[1] (oppIdx): want %d, got %d", oppIdx, g.Pending.Data[1])
	}
	if Card(g.Pending.Data[2]) != originalOwn {
		t.Errorf("Pending.Data[2] (ownCard): want %v, got %v", originalOwn, Card(g.Pending.Data[2]))
	}
	if Card(g.Pending.Data[3]) != originalOpp {
		t.Errorf("Pending.Data[3] (oppCard): want %v, got %v", originalOpp, Card(g.Pending.Data[3]))
	}

	// Verify cards are revealed in LastAction.
	if g.LastAction.RevealedCard != originalOwn {
		t.Errorf("RevealedCard (own): want %v, got %v", originalOwn, g.LastAction.RevealedCard)
	}

	turnBefore := g.TurnNumber

	// King Swap Decision — choose to swap.
	if err := g.ApplyAction(ActionKingSwapYes); err != nil {
		t.Fatalf("KingSwapYes failed: %v", err)
	}

	// Cards should be swapped.
	if g.Players[acting].Hand[ownIdx] != originalOpp {
		t.Errorf("acting hand[%d]: want %v (was opp's), got %v", ownIdx, originalOpp, g.Players[acting].Hand[ownIdx])
	}
	if g.Players[opp].Hand[oppIdx] != originalOwn {
		t.Errorf("opp hand[%d]: want %v (was own's), got %v", oppIdx, originalOwn, g.Players[opp].Hand[oppIdx])
	}

	if g.Pending.Type != PendingNone {
		t.Errorf("expected PendingNone, got %d", g.Pending.Type)
	}
	passSnapPhase(t, g)
	if g.TurnNumber != turnBefore+1 {
		t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
	}
}

// TestKingLookThenNoSwap tests the King ability when the player chooses NOT to swap.
func TestKingLookThenNoSwap(t *testing.T) {
	kingCard := NewCard(SuitClubs, RankKing)
	g := setupAbilityGame(t, kingCard)

	acting := g.CurrentPlayer
	opp := g.OpponentOf(acting)

	ownIdx := uint8(0)
	oppIdx := uint8(1)

	originalOwn := g.Players[acting].Hand[ownIdx]
	originalOpp := g.Players[opp].Hand[oppIdx]

	drawForcedCard(t, g)
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility failed: %v", err)
	}
	if err := g.ApplyAction(EncodeKingLook(ownIdx, oppIdx)); err != nil {
		t.Fatalf("KingLook failed: %v", err)
	}
	if g.Pending.Type != PendingKingDecision {
		t.Fatalf("expected PendingKingDecision, got %d", g.Pending.Type)
	}

	turnBefore := g.TurnNumber

	// King Swap Decision — choose NOT to swap.
	if err := g.ApplyAction(ActionKingSwapNo); err != nil {
		t.Fatalf("KingSwapNo failed: %v", err)
	}

	// Cards should be unchanged.
	if g.Players[acting].Hand[ownIdx] != originalOwn {
		t.Errorf("acting hand[%d]: want %v (unchanged), got %v", ownIdx, originalOwn, g.Players[acting].Hand[ownIdx])
	}
	if g.Players[opp].Hand[oppIdx] != originalOpp {
		t.Errorf("opp hand[%d]: want %v (unchanged), got %v", oppIdx, originalOpp, g.Players[opp].Hand[oppIdx])
	}

	if g.Pending.Type != PendingNone {
		t.Errorf("expected PendingNone, got %d", g.Pending.Type)
	}
	passSnapPhase(t, g)
	if g.TurnNumber != turnBefore+1 {
		t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
	}
}

// TestAbilityNotTriggeredFromDiscardDraw verifies that drawing from the discard pile
// does NOT trigger an ability even when discarding with ability=true.
func TestAbilityNotTriggeredFromDiscardDraw(t *testing.T) {
	// Set up a game where the discard pile top is an ability card (Seven).
	g := NewGame(42, DefaultHouseRules())
	g.Deal()
	g.CurrentPlayer = 0

	// Place a Seven on top of the discard pile so the player can draw it.
	sevenCard := NewCard(SuitHearts, RankSeven)
	g.DiscardPile[g.DiscardLen] = sevenCard
	g.DiscardLen++

	// Draw from the discard pile.
	if err := g.ApplyAction(ActionDrawDiscard); err != nil {
		t.Fatalf("DrawDiscard failed: %v", err)
	}

	if g.Pending.Data[1] != DrawnFromDiscard {
		t.Fatalf("expected DrawnFromDiscard, got %d", g.Pending.Data[1])
	}

	turnBefore := g.TurnNumber

	// Discard with ability=true — but it was drawn from discard, so no ability.
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility failed: %v", err)
	}

	// No ability pending (snap phase may be active for the discarded card).
	if g.Pending.Type != PendingNone {
		t.Errorf("expected PendingNone (ability not triggered from discard draw), got %d", g.Pending.Type)
	}
	passSnapPhase(t, &g)
	if g.TurnNumber != turnBefore+1 {
		t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
	}
}

// TestDiscardWithAbilityNoAbilityCard verifies that drawing a low card (A-6) and
// discarding with ability=true is treated as a regular discard.
func TestDiscardWithAbilityNoAbilityCard(t *testing.T) {
	for _, rank := range []uint8{RankAce, RankTwo, RankThree, RankFour, RankFive, RankSix} {
		t.Run(rankName(rank), func(t *testing.T) {
			lowCard := NewCard(SuitSpades, rank)
			g := setupAbilityGame(t, lowCard)

			drawForcedCard(t, g)
			if Card(g.Pending.Data[0]) != lowCard {
				t.Fatalf("expected drawn card %v, got %v", lowCard, Card(g.Pending.Data[0]))
			}

			turnBefore := g.TurnNumber

			if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
				t.Fatalf("DiscardWithAbility on low card failed: %v", err)
			}

			// No ability — snap phase for the discarded card, then turn advances.
			if g.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone for no-ability card, got %d", g.Pending.Type)
			}
			passSnapPhase(t, g)
			if g.TurnNumber != turnBefore+1 {
				t.Errorf("TurnNumber: want %d, got %d", turnBefore+1, g.TurnNumber)
			}
			if g.DiscardTop() != lowCard {
				t.Errorf("DiscardTop: want %v, got %v", lowCard, g.DiscardTop())
			}
		})
	}
}

// TestPeekOwnInvalidPendingType verifies that peekOwn returns an error when pending type is wrong.
func TestPeekOwnInvalidPendingType(t *testing.T) {
	g := newDealtGame(t)
	// No pending action, try PeekOwn anyway.
	if err := g.ApplyAction(EncodePeekOwn(0)); err == nil {
		t.Error("expected error for PeekOwn without PendingPeekOwn, got nil")
	}
}

// TestPeekOtherInvalidPendingType verifies error when pending type is wrong.
func TestPeekOtherInvalidPendingType(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(EncodePeekOther(0)); err == nil {
		t.Error("expected error for PeekOther without PendingPeekOther, got nil")
	}
}

// TestBlindSwapInvalidPendingType verifies error when pending type is wrong.
func TestBlindSwapInvalidPendingType(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(EncodeBlindSwap(0, 1)); err == nil {
		t.Error("expected error for BlindSwap without PendingBlindSwap, got nil")
	}
}

// TestKingLookInvalidPendingType verifies error when pending type is wrong.
func TestKingLookInvalidPendingType(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(EncodeKingLook(0, 1)); err == nil {
		t.Error("expected error for KingLook without PendingKingLook, got nil")
	}
}

// TestKingSwapDecisionInvalidPendingType verifies error when pending type is wrong.
func TestKingSwapDecisionInvalidPendingType(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(ActionKingSwapYes); err == nil {
		t.Error("expected error for KingSwapYes without PendingKingDecision, got nil")
	}
}

// rankName returns a human-readable string for a rank constant.
func rankName(rank uint8) string {
	names := map[uint8]string{
		RankAce:   "Ace",
		RankTwo:   "Two",
		RankThree: "Three",
		RankFour:  "Four",
		RankFive:  "Five",
		RankSix:   "Six",
		RankSeven: "Seven",
		RankEight: "Eight",
		RankNine:  "Nine",
		RankTen:   "Ten",
		RankJack:  "Jack",
		RankQueen: "Queen",
		RankKing:  "King",
		RankJoker: "Joker",
	}
	if n, ok := names[rank]; ok {
		return n
	}
	return "Unknown"
}
