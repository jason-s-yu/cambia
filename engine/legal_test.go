package engine

import (
	"sort"
	"testing"
)

// sortedActions returns a sorted copy of the action slice for deterministic comparison.
func sortedActions(actions []uint16) []uint16 {
	cp := make([]uint16, len(actions))
	copy(cp, actions)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp
}

// containsAction returns true if action is in the slice.
func containsAction(actions []uint16, action uint16) bool {
	for _, a := range actions {
		if a == action {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// DecisionCtx tests
// ---------------------------------------------------------------------------

func TestDecisionCtxStartTurn(t *testing.T) {
	g := newDealtGame(t)
	ctx := g.DecisionCtx()
	if ctx != CtxStartTurn {
		t.Errorf("expected CtxStartTurn (%d), got %d", CtxStartTurn, ctx)
	}
}

func TestDecisionCtxPostDraw(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	ctx := g.DecisionCtx()
	if ctx != CtxPostDraw {
		t.Errorf("expected CtxPostDraw (%d), got %d", CtxPostDraw, ctx)
	}
}

func TestDecisionCtxAbilitySelect(t *testing.T) {
	// Force a Seven onto the stockpile to trigger PeekOwn after discard.
	sevenCard := NewCard(SuitHearts, RankSeven)
	g := setupAbilityGame(t, sevenCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility: %v", err)
	}
	// Should be PendingPeekOwn → CtxAbilitySelect.
	ctx := g.DecisionCtx()
	if ctx != CtxAbilitySelect {
		t.Errorf("expected CtxAbilitySelect (%d), got %d", CtxAbilitySelect, ctx)
	}
}

func TestDecisionCtxSnapDecision(t *testing.T) {
	g := newDealtGame(t)

	// Draw and discard to trigger snap phase.
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardNoAbility); err != nil {
		t.Fatalf("DiscardNoAbility: %v", err)
	}
	// After a discard, if snap phase is active, ctx should be CtxSnapDecision.
	if g.Snap.Active {
		ctx := g.DecisionCtx()
		if ctx != CtxSnapDecision {
			t.Errorf("expected CtxSnapDecision (%d), got %d", CtxSnapDecision, ctx)
		}
	}
}

func TestDecisionCtxSnapMove(t *testing.T) {
	g := newDealtGame(t)

	// We need to get into PendingSnapMove state.
	// Manually set up: snap active + pending snap move.
	g.Snap.Active = true
	g.Snap.Snappers[0] = 0
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Pending.Type = PendingSnapMove
	g.Pending.PlayerID = 0
	g.Pending.Data[0] = 1 // opponent index
	g.Pending.Data[1] = 0 // vacated slot

	ctx := g.DecisionCtx()
	if ctx != CtxSnapMove {
		t.Errorf("expected CtxSnapMove (%d), got %d", CtxSnapMove, ctx)
	}
}

func TestDecisionCtxTerminal(t *testing.T) {
	g := newDealtGame(t)
	g.Flags |= FlagGameOver
	ctx := g.DecisionCtx()
	if ctx != CtxTerminal {
		t.Errorf("expected CtxTerminal (%d), got %d", CtxTerminal, ctx)
	}
}

// ---------------------------------------------------------------------------
// LegalActions tests
// ---------------------------------------------------------------------------

func TestLegalActionsStartTurn(t *testing.T) {
	g := newDealtGame(t)
	actions := g.LegalActionsList()

	// DrawStockpile is always legal.
	if !containsAction(actions, ActionDrawStockpile) {
		t.Error("expected DrawStockpile to be legal")
	}

	// DrawDiscard legal when AllowDrawFromDiscard = true (default) and discard non-empty.
	if !containsAction(actions, ActionDrawDiscard) {
		t.Error("expected DrawDiscard to be legal with default rules")
	}

	// CallCambia: with CambiaAllowedRound=0, it should be legal from round 0.
	if !containsAction(actions, ActionCallCambia) {
		t.Error("expected CallCambia to be legal from round 0")
	}
}

func TestLegalActionsStartTurnNoDrawDiscard(t *testing.T) {
	rules := DefaultHouseRules()
	rules.AllowDrawFromDiscard = false
	g := NewGame(42, rules)
	g.Deal()

	actions := g.LegalActionsList()
	if containsAction(actions, ActionDrawDiscard) {
		t.Error("expected DrawDiscard to be illegal when AllowDrawFromDiscard=false")
	}
}

func TestLegalActionsStartTurnCambiaRound(t *testing.T) {
	rules := DefaultHouseRules()
	rules.CambiaAllowedRound = 2 // Only allowed from round 2.
	g := NewGame(42, rules)
	g.Deal()

	// Round 0 — Cambia not yet allowed.
	actions := g.LegalActionsList()
	if containsAction(actions, ActionCallCambia) {
		t.Error("expected CallCambia illegal before allowed round")
	}

	// Advance to round 2 (4 turns with 2 players).
	g.TurnNumber = 4 // round = 4/2 = 2
	actions = g.LegalActionsList()
	if !containsAction(actions, ActionCallCambia) {
		t.Error("expected CallCambia legal at or after CambiaAllowedRound")
	}
}

func TestLegalActionsStartTurnCambiaAlreadyCalled(t *testing.T) {
	g := newDealtGame(t)
	g.CambiaCaller = 0 // Player 0 already called Cambia.
	g.Flags |= FlagCambiaCalled

	actions := g.LegalActionsList()
	if containsAction(actions, ActionCallCambia) {
		t.Error("expected CallCambia illegal when already called")
	}
}

func TestLegalActionsPostDraw(t *testing.T) {
	g := newDealtGame(t)
	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}

	drawnCard := Card(g.Pending.Data[0])
	actions := g.LegalActionsList()

	// DiscardNoAbility always legal post-draw.
	if !containsAction(actions, ActionDiscardNoAbility) {
		t.Error("expected DiscardNoAbility to be legal post-draw")
	}

	// DiscardWithAbility: depends on drawn card having ability AND from stockpile.
	if drawnCard.HasAbility() {
		// Also need ability to be usable — default game has cards, so it should be usable.
		if !containsAction(actions, ActionDiscardWithAbility) {
			t.Logf("drawn card %v has ability but DiscardWithAbility not legal (hands may be empty?)", drawnCard)
		}
	}

	// Replace(i) for i < acting player's hand count.
	acting := g.Pending.PlayerID
	handLen := g.Players[acting].HandLen
	for i := uint8(0); i < handLen; i++ {
		if !containsAction(actions, EncodeReplace(i)) {
			t.Errorf("expected Replace(%d) to be legal", i)
		}
	}
	// Replace indices beyond hand should NOT be legal.
	if containsAction(actions, EncodeReplace(handLen)) {
		t.Errorf("expected Replace(%d) to be illegal (out of hand range)", handLen)
	}
}

func TestLegalActionsPostDrawAbilityFromStockpile(t *testing.T) {
	// Force a Seven onto the stockpile.
	sevenCard := NewCard(SuitHearts, RankSeven)
	g := setupAbilityGame(t, sevenCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}

	actions := g.LegalActionsList()
	if !containsAction(actions, ActionDiscardWithAbility) {
		t.Error("expected DiscardWithAbility legal when 7 drawn from stockpile")
	}
}

func TestLegalActionsPostDrawAbilityFromDiscard(t *testing.T) {
	// Force a Seven into the discard pile.
	sevenCard := NewCard(SuitHearts, RankSeven)
	g := newDealtGame(t)
	g.DiscardPile[g.DiscardLen] = sevenCard
	g.DiscardLen++
	g.CurrentPlayer = 0

	if err := g.ApplyAction(ActionDrawDiscard); err != nil {
		t.Fatalf("DrawDiscard: %v", err)
	}

	actions := g.LegalActionsList()
	// DiscardWithAbility NOT legal when drawn from discard.
	if containsAction(actions, ActionDiscardWithAbility) {
		t.Error("expected DiscardWithAbility illegal when drawn from discard pile")
	}
}

func TestLegalActionsPeekOwn(t *testing.T) {
	sevenCard := NewCard(SuitHearts, RankSeven)
	g := setupAbilityGame(t, sevenCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility: %v", err)
	}

	if g.Pending.Type != PendingPeekOwn {
		t.Fatalf("expected PendingPeekOwn, got %d", g.Pending.Type)
	}

	acting := g.Pending.PlayerID
	handLen := g.Players[acting].HandLen
	actions := g.LegalActionsList()

	for i := uint8(0); i < handLen; i++ {
		if !containsAction(actions, EncodePeekOwn(i)) {
			t.Errorf("expected PeekOwn(%d) to be legal", i)
		}
	}
	if containsAction(actions, EncodePeekOwn(handLen)) {
		t.Errorf("expected PeekOwn(%d) to be illegal (out of range)", handLen)
	}

	// No start-turn actions.
	if containsAction(actions, ActionDrawStockpile) {
		t.Error("DrawStockpile should not be legal during PeekOwn")
	}
}

func TestLegalActionsPeekOther(t *testing.T) {
	nineCard := NewCard(SuitHearts, RankNine)
	g := setupAbilityGame(t, nineCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility: %v", err)
	}

	if g.Pending.Type != PendingPeekOther {
		t.Fatalf("expected PendingPeekOther, got %d", g.Pending.Type)
	}

	acting := g.Pending.PlayerID
	opp := g.OpponentOf(acting)
	oppHandLen := g.Players[opp].HandLen
	actions := g.LegalActionsList()

	for i := uint8(0); i < oppHandLen; i++ {
		if !containsAction(actions, EncodePeekOther(i)) {
			t.Errorf("expected PeekOther(%d) to be legal", i)
		}
	}
	if containsAction(actions, EncodePeekOther(oppHandLen)) {
		t.Errorf("expected PeekOther(%d) to be illegal (out of range)", oppHandLen)
	}
}

func TestLegalActionsBlindSwap(t *testing.T) {
	jackCard := NewCard(SuitHearts, RankJack)
	g := setupAbilityGame(t, jackCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility: %v", err)
	}

	if g.Pending.Type != PendingBlindSwap {
		t.Fatalf("expected PendingBlindSwap, got %d", g.Pending.Type)
	}

	acting := g.Pending.PlayerID
	opp := g.OpponentOf(acting)
	ownLen := g.Players[acting].HandLen
	oppLen := g.Players[opp].HandLen
	actions := g.LegalActionsList()

	for i := uint8(0); i < ownLen; i++ {
		for j := uint8(0); j < oppLen; j++ {
			if !containsAction(actions, EncodeBlindSwap(i, j)) {
				t.Errorf("expected BlindSwap(%d,%d) to be legal", i, j)
			}
		}
	}
	// Out-of-range combinations should not be legal.
	if containsAction(actions, EncodeBlindSwap(ownLen, 0)) {
		t.Errorf("expected BlindSwap(%d,0) to be illegal (own index out of range)", ownLen)
	}
}

func TestLegalActionsKingLook(t *testing.T) {
	kingCard := NewCard(SuitClubs, RankKing) // Black King
	g := setupAbilityGame(t, kingCard)

	if err := g.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
		t.Fatalf("DiscardWithAbility: %v", err)
	}

	if g.Pending.Type != PendingKingLook {
		t.Fatalf("expected PendingKingLook, got %d", g.Pending.Type)
	}

	acting := g.Pending.PlayerID
	opp := g.OpponentOf(acting)
	ownLen := g.Players[acting].HandLen
	oppLen := g.Players[opp].HandLen
	actions := g.LegalActionsList()

	for i := uint8(0); i < ownLen; i++ {
		for j := uint8(0); j < oppLen; j++ {
			if !containsAction(actions, EncodeKingLook(i, j)) {
				t.Errorf("expected KingLook(%d,%d) to be legal", i, j)
			}
		}
	}
	// Out-of-range combinations should not be legal.
	if containsAction(actions, EncodeKingLook(ownLen, 0)) {
		t.Errorf("expected KingLook(%d,0) to be illegal (own index out of range)", ownLen)
	}
}

func TestLegalActionsKingDecision(t *testing.T) {
	g := newDealtGame(t)
	// Manually set PendingKingDecision.
	g.Pending.Type = PendingKingDecision
	g.Pending.PlayerID = g.CurrentPlayer

	actions := g.LegalActionsList()
	if !containsAction(actions, ActionKingSwapNo) {
		t.Error("expected KingSwapNo to be legal during KingDecision")
	}
	if !containsAction(actions, ActionKingSwapYes) {
		t.Error("expected KingSwapYes to be legal during KingDecision")
	}
	if len(actions) != 2 {
		t.Errorf("expected exactly 2 legal actions during KingDecision, got %d: %v", len(actions), actions)
	}
}

func TestLegalActionsSnapDecision(t *testing.T) {
	g := newDealtGame(t)

	// Set up snap phase manually.
	g.Snap.Active = true
	g.Snap.Snappers[0] = 0
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Snap.DiscardedRank = RankFive

	actions := g.LegalActionsList()

	// PassSnap always legal.
	if !containsAction(actions, ActionPassSnap) {
		t.Error("expected PassSnap to be legal during snap")
	}

	// SnapOwn(i) for each i < player 0's hand length.
	acting := g.Snap.Snappers[0]
	handLen := g.Players[acting].HandLen
	for i := uint8(0); i < handLen; i++ {
		if !containsAction(actions, EncodeSnapOwn(i)) {
			t.Errorf("expected SnapOwn(%d) to be legal", i)
		}
	}

	// SnapOpponent(i) for each i < opponent's hand length (rule allows it by default).
	opp := g.OpponentOf(acting)
	oppLen := g.Players[opp].HandLen
	if g.Rules.AllowOpponentSnapping {
		for i := uint8(0); i < oppLen; i++ {
			if !containsAction(actions, EncodeSnapOpponent(i)) {
				t.Errorf("expected SnapOpponent(%d) to be legal", i)
			}
		}
	}
}

func TestLegalActionsSnapDecisionNoOpponentSnapping(t *testing.T) {
	rules := DefaultHouseRules()
	rules.AllowOpponentSnapping = false
	g := NewGame(42, rules)
	g.Deal()

	g.Snap.Active = true
	g.Snap.Snappers[0] = 0
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Snap.DiscardedRank = RankFive

	actions := g.LegalActionsList()

	opp := g.OpponentOf(0)
	oppLen := g.Players[opp].HandLen
	for i := uint8(0); i < oppLen; i++ {
		if containsAction(actions, EncodeSnapOpponent(i)) {
			t.Errorf("expected SnapOpponent(%d) illegal when AllowOpponentSnapping=false", i)
		}
	}
}

func TestLegalActionsSnapMove(t *testing.T) {
	g := newDealtGame(t)

	// Set up pending snap move.
	snapperIdx := uint8(0)
	vacatedSlot := uint8(2)
	g.Snap.Active = true
	g.Snap.Snappers[0] = snapperIdx
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Pending.Type = PendingSnapMove
	g.Pending.PlayerID = snapperIdx
	g.Pending.Data[0] = g.OpponentOf(snapperIdx)
	g.Pending.Data[1] = vacatedSlot

	ownLen := g.Players[snapperIdx].HandLen
	actions := g.LegalActionsList()

	for i := uint8(0); i < ownLen; i++ {
		expected := EncodeSnapOpponentMove(i, vacatedSlot)
		if !containsAction(actions, expected) {
			t.Errorf("expected SnapOpponentMove(%d,%d) to be legal", i, vacatedSlot)
		}
	}

	// Only snap move actions should be legal.
	if containsAction(actions, ActionPassSnap) {
		t.Error("PassSnap should not be legal during SnapMove")
	}
	if containsAction(actions, ActionDrawStockpile) {
		t.Error("DrawStockpile should not be legal during SnapMove")
	}
}

func TestLegalActionsTerminal(t *testing.T) {
	g := newDealtGame(t)
	g.Flags |= FlagGameOver

	actions := g.LegalActionsList()
	if len(actions) != 0 {
		t.Errorf("expected no legal actions in terminal state, got %d: %v", len(actions), actions)
	}

	mask := g.LegalActions()
	if mask != ([3]uint64{}) {
		t.Errorf("expected zero bitmask in terminal state, got %v", mask)
	}
}

func TestLegalActionsBitmaskRoundTrip(t *testing.T) {
	// Test multiple states: start turn, post-draw, ability.
	states := []struct {
		name  string
		setup func(*GameState)
	}{
		{"start_turn", func(g *GameState) {}},
		{"post_draw", func(g *GameState) {
			if err := g.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
		}},
		{"snap_decision", func(g *GameState) {
			g.Snap.Active = true
			g.Snap.Snappers[0] = 0
			g.Snap.NumSnappers = 1
			g.Snap.CurrentSnapperIdx = 0
		}},
	}

	for _, tc := range states {
		t.Run(tc.name, func(t *testing.T) {
			g2 := newDealtGame(t)
			tc.setup(g2)

			mask := g2.LegalActions()
			list := g2.LegalActionsList()

			// Verify every action in the list is set in the bitmask.
			for _, a := range list {
				if mask[a/64]>>(a%64)&1 == 0 {
					t.Errorf("action %d in list but not in bitmask", a)
				}
			}

			// Verify every bit in the bitmask is in the list.
			for i := uint16(0); i < NumActions; i++ {
				if mask[i/64]>>(i%64)&1 == 1 {
					if !containsAction(list, i) {
						t.Errorf("action %d in bitmask but not in list", i)
					}
				}
			}
		})
	}
}

// BenchmarkLegalActions benchmarks zero-allocation legal action generation.
// Target: <200ns, 0 allocs.
func BenchmarkLegalActions(b *testing.B) {
	g := NewGame(42, DefaultHouseRules())
	g.Deal()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = g.LegalActions()
	}
}
