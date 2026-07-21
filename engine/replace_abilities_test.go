package engine

import (
	"testing"
)

// setupReplaceAbilityGame creates a game with AllowReplaceAbilities=true and
// places an ability card at the top of the stockpile, ready to be drawn.
func setupReplaceAbilityGame(t *testing.T, abilityCard Card) *GameState {
	t.Helper()
	rules := DefaultHouseRules()
	rules.AllowReplaceAbilities = true
	gs := NewGame(42, rules)
	gs.Deal()
	gs.CurrentPlayer = 0
	// Place the ability card at the top of stockpile (to be drawn next).
	gs.Stockpile[gs.StockLen] = abilityCard
	gs.StockLen++
	return &gs
}

// TestAllowReplaceAbilitiesField verifies the field exists and defaults to false.
func TestAllowReplaceAbilitiesField(t *testing.T) {
	rules := DefaultHouseRules()
	if rules.AllowReplaceAbilities != false {
		t.Errorf("DefaultHouseRules().AllowReplaceAbilities: want false, got %v", rules.AllowReplaceAbilities)
	}
	rules.AllowReplaceAbilities = true
	if !rules.AllowReplaceAbilities {
		t.Error("AllowReplaceAbilities should be settable to true")
	}
}

// TestReplaceNoAbilityByDefault verifies that with AllowReplaceAbilities=false (default),
// replacing an ability card never triggers an ability.
func TestReplaceNoAbilityByDefault(t *testing.T) {
	abilityRanks := []uint8{RankSeven, RankEight, RankNine, RankTen, RankJack, RankQueen, RankKing}
	for _, rank := range abilityRanks {
		t.Run(rankName(rank), func(t *testing.T) {
			// Default rules: AllowReplaceAbilities = false
			gs := NewGame(42, DefaultHouseRules())
			gs.Deal()
			gs.CurrentPlayer = 0

			// We'll place the ability card in the player's hand at index 0,
			// and a non-ability card on top of the stockpile to draw.
			drawCard := NewCard(SuitHearts, RankAce) // no ability
			gs.Stockpile[gs.StockLen] = drawCard
			gs.StockLen++

			// Force hand[0] to be an ability card.
			abilityCard := NewCard(SuitHearts, rank)
			gs.Players[0].Hand[0] = abilityCard

			// Draw from stockpile.
			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}

			// Replace hand[0] (the ability card) with the drawn Ace.
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			// Should NOT be in any ability pending state — just snap phase.
			if gs.Pending.Type != PendingNone && !gs.Snap.Active {
				t.Errorf("rank %s: expected PendingNone or snap phase, got Pending.Type=%d", rankName(rank), gs.Pending.Type)
			}
			if gs.Pending.Type == PendingPeekOwn || gs.Pending.Type == PendingPeekOther ||
				gs.Pending.Type == PendingBlindSwap || gs.Pending.Type == PendingKingLook {
				t.Errorf("rank %s: ability should NOT have triggered (AllowReplaceAbilities=false)", rankName(rank))
			}
		})
	}
}

// TestReplaceFromDiscardNoAbility verifies that replacing with a card drawn from the
// discard pile does NOT trigger ability even when AllowReplaceAbilities=true.
func TestReplaceFromDiscardNoAbility(t *testing.T) {
	// AllowReplaceAbilities = true AND AllowDrawFromDiscard = true
	rules := DefaultHouseRules()
	rules.AllowReplaceAbilities = true
	rules.AllowDrawFromDiscard = true
	gs := NewGame(42, rules)
	gs.Deal()
	gs.CurrentPlayer = 0

	// Place an ability card as the top of the discard pile.
	abilityCard := NewCard(SuitHearts, RankSeven) // PeekOwn ability
	gs.DiscardPile[gs.DiscardLen] = abilityCard
	gs.DiscardLen++

	// Force hand[0] to a non-ability card.
	gs.Players[0].Hand[0] = NewCard(SuitClubs, RankTwo)

	// Draw from DISCARD (not stockpile).
	if err := gs.ApplyAction(ActionDrawDiscard); err != nil {
		t.Fatalf("DrawDiscard: %v", err)
	}

	// Replace hand[0] with the drawn 7.
	if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	// Should NOT trigger ability — drawn from discard, not stockpile.
	if gs.Pending.Type == PendingPeekOwn {
		t.Error("PeekOwn ability should NOT trigger when replacing with discard-drawn card")
	}
}

// TestReplaceNonAbilityCardNoTrigger verifies that replacing a non-ability card from stockpile
// with AllowReplaceAbilities=true does not trigger any ability.
func TestReplaceNonAbilityCardNoTrigger(t *testing.T) {
	rules := DefaultHouseRules()
	rules.AllowReplaceAbilities = true
	gs := NewGame(42, rules)
	gs.Deal()
	gs.CurrentPlayer = 0

	// Non-ability card on stockpile.
	drawCard := NewCard(SuitHearts, RankFive)
	gs.Stockpile[gs.StockLen] = drawCard
	gs.StockLen++

	// Non-ability card in hand[0].
	gs.Players[0].Hand[0] = NewCard(SuitClubs, RankThree)

	if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	// Neither the drawn card nor the old card has ability — no ability pending.
	if gs.Pending.Type == PendingPeekOwn || gs.Pending.Type == PendingPeekOther ||
		gs.Pending.Type == PendingBlindSwap || gs.Pending.Type == PendingKingLook {
		t.Error("ability should NOT trigger when replacing a non-ability card")
	}
}

// TestReplaceTriggersPeekOwn verifies replacing a 7 from stockpile triggers PeekOwn.
func TestReplaceTriggersPeekOwn(t *testing.T) {
	for _, rank := range []uint8{RankSeven, RankEight} {
		t.Run(rankName(rank), func(t *testing.T) {
			sevenCard := NewCard(SuitHearts, rank)
			// Place ability card in hand[0], draw non-ability from stockpile.
			rules := DefaultHouseRules()
			rules.AllowReplaceAbilities = true
			gs := NewGame(42, rules)
			gs.Deal()
			gs.CurrentPlayer = 0

			drawCard := NewCard(SuitDiamonds, RankAce)
			gs.Stockpile[gs.StockLen] = drawCard
			gs.StockLen++
			gs.Players[0].Hand[0] = sevenCard

			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			// Should be in PendingPeekOwn.
			if gs.Pending.Type != PendingPeekOwn {
				t.Fatalf("expected PendingPeekOwn, got %d", gs.Pending.Type)
			}
			if gs.Pending.PlayerID != 0 {
				t.Errorf("expected PlayerID 0, got %d", gs.Pending.PlayerID)
			}

			// Discard top should be the replaced old card (sevenCard).
			if gs.DiscardTop() != sevenCard {
				t.Errorf("expected discard top %v, got %v", sevenCard, gs.DiscardTop())
			}

			// Resolve the peek.
			peekTarget := uint8(0)
			if err := gs.ApplyAction(EncodePeekOwn(peekTarget)); err != nil {
				t.Fatalf("PeekOwn: %v", err)
			}

			if gs.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone after PeekOwn, got %d", gs.Pending.Type)
			}

			// Pass snap phase.
			passSnapPhase(t, &gs)

			// Turn should advance after ability+snap resolution.
			if gs.TurnNumber == 0 {
				t.Error("TurnNumber should advance after replace+ability resolution")
			}
		})
	}
}

// TestReplaceTriggersPeekOther verifies replacing a 9/T from stockpile triggers PeekOther.
func TestReplaceTriggersPeekOther(t *testing.T) {
	for _, rank := range []uint8{RankNine, RankTen} {
		t.Run(rankName(rank), func(t *testing.T) {
			nineCard := NewCard(SuitSpades, rank)
			rules := DefaultHouseRules()
			rules.AllowReplaceAbilities = true
			gs := NewGame(42, rules)
			gs.Deal()
			gs.CurrentPlayer = 0

			drawCard := NewCard(SuitDiamonds, RankAce)
			gs.Stockpile[gs.StockLen] = drawCard
			gs.StockLen++
			gs.Players[0].Hand[0] = nineCard

			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			if gs.Pending.Type != PendingPeekOther {
				t.Fatalf("expected PendingPeekOther, got %d", gs.Pending.Type)
			}

			// Resolve peek other.
			if err := gs.ApplyAction(EncodePeekOther(0)); err != nil {
				t.Fatalf("PeekOther: %v", err)
			}

			if gs.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone after PeekOther, got %d", gs.Pending.Type)
			}
			passSnapPhase(t, &gs)
		})
	}
}

// TestReplaceTriggersBlindSwap verifies replacing a J/Q from stockpile triggers BlindSwap.
func TestReplaceTriggersBlindSwap(t *testing.T) {
	for _, rank := range []uint8{RankJack, RankQueen} {
		t.Run(rankName(rank), func(t *testing.T) {
			swapCard := NewCard(SuitClubs, rank)
			rules := DefaultHouseRules()
			rules.AllowReplaceAbilities = true
			gs := NewGame(42, rules)
			gs.Deal()
			gs.CurrentPlayer = 0

			drawCard := NewCard(SuitDiamonds, RankAce)
			gs.Stockpile[gs.StockLen] = drawCard
			gs.StockLen++
			gs.Players[0].Hand[0] = swapCard

			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			if gs.Pending.Type != PendingBlindSwap {
				t.Fatalf("expected PendingBlindSwap, got %d", gs.Pending.Type)
			}

			// Resolve blind swap.
			if err := gs.ApplyAction(EncodeBlindSwap(0, 0)); err != nil {
				t.Fatalf("BlindSwap: %v", err)
			}

			if gs.Pending.Type != PendingNone {
				t.Errorf("expected PendingNone after BlindSwap, got %d", gs.Pending.Type)
			}
			passSnapPhase(t, &gs)
		})
	}
}

// TestReplaceTriggersKingLook verifies replacing a King from stockpile triggers KingLook.
func TestReplaceTriggersKingLook(t *testing.T) {
	// Use a non-red king (black king has positive value, cleaner for testing).
	kingCard := NewCard(SuitClubs, RankKing)
	rules := DefaultHouseRules()
	rules.AllowReplaceAbilities = true
	gs := NewGame(42, rules)
	gs.Deal()
	gs.CurrentPlayer = 0

	drawCard := NewCard(SuitDiamonds, RankAce)
	gs.Stockpile[gs.StockLen] = drawCard
	gs.StockLen++
	gs.Players[0].Hand[0] = kingCard

	if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
		t.Fatalf("DrawStockpile: %v", err)
	}
	if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	if gs.Pending.Type != PendingKingLook {
		t.Fatalf("expected PendingKingLook, got %d", gs.Pending.Type)
	}

	// KingLook: look at own[0] and opp[0].
	if err := gs.ApplyAction(EncodeKingLook(0, 0)); err != nil {
		t.Fatalf("KingLook: %v", err)
	}

	if gs.Pending.Type != PendingKingDecision {
		t.Fatalf("expected PendingKingDecision, got %d", gs.Pending.Type)
	}

	// Choose not to swap.
	if err := gs.ApplyAction(ActionKingSwapNo); err != nil {
		t.Fatalf("KingSwapNo: %v", err)
	}

	if gs.Pending.Type != PendingNone {
		t.Errorf("expected PendingNone after KingSwapNo, got %d", gs.Pending.Type)
	}
	passSnapPhase(t, &gs)
}

// ---------------------------------------------------------------------------
// cambia-653: replace()'s ability-trigger block must route through
// canUseAbility, not a standalone hand-count switch, so LockCallerHand is
// respected. Before the fix, a locked opponent-caller scenario set Pending to
// BlindSwap/KingLook while legalAbilitySelect produced zero legal actions —
// a non-terminal, deadlocked game.
// ---------------------------------------------------------------------------

// TestReplaceAbilityLockGating verifies the replace()-triggered ability path
// mirrors canUseAbility's LockCallerHand gating for BlindSwap/KingLook, leaves
// peeks unaffected by the lock, and never leaves the game in a deadlocked
// pending state with zero legal actions.
func TestReplaceAbilityLockGating(t *testing.T) {
	cases := []struct {
		name        string
		rank        uint8
		lockOn      bool
		callerIsOpp bool // true: player 1 (opponent) is Cambia caller; false: player 0 (acting) is
		wantPend    PendingType
	}{
		// Item 1: lock on, opponent is caller -> BlindSwap/KingLook fizzle.
		{"jack_locked_opponent_caller_fizzles", RankJack, true, true, PendingNone},
		{"queen_locked_opponent_caller_fizzles", RankQueen, true, true, PendingNone},
		{"king_locked_opponent_caller_fizzles", RankKing, true, true, PendingNone},

		// Item 2: lock off -> ability triggers normally despite a Cambia caller.
		{"jack_unlocked_enters_pending", RankJack, false, true, PendingBlindSwap},
		{"king_unlocked_enters_pending", RankKing, false, true, PendingKingLook},

		// Item 3: lock on but the acting player (not the opponent) is the
		// caller -> unaffected, ability enters pending normally.
		{"jack_locked_acting_is_caller_enters_pending", RankJack, true, false, PendingBlindSwap},
		{"king_locked_acting_is_caller_enters_pending", RankKing, true, false, PendingKingLook},

		// Item 4: peeks are never lock-gated, even with lock on and the
		// opponent as caller.
		{"seven_locked_opponent_caller_enters_pending", RankSeven, true, true, PendingPeekOwn},
		{"eight_locked_opponent_caller_enters_pending", RankEight, true, true, PendingPeekOwn},
		{"nine_locked_opponent_caller_enters_pending", RankNine, true, true, PendingPeekOther},
		{"ten_locked_opponent_caller_enters_pending", RankTen, true, true, PendingPeekOther},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rules := DefaultHouseRules()
			rules.AllowReplaceAbilities = true
			rules.LockCallerHand = tc.lockOn
			gs := NewGame(42, rules)
			gs.Deal()
			gs.CurrentPlayer = 0

			if tc.callerIsOpp {
				gs.CambiaCaller = 1
			} else {
				gs.CambiaCaller = 0
			}
			gs.Flags |= FlagCambiaCalled

			abilityCard := NewCard(SuitClubs, tc.rank)
			gs.Players[0].Hand[0] = abilityCard
			gs.Stockpile[gs.StockLen] = NewCard(SuitDiamonds, RankAce)
			gs.StockLen++

			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			if gs.Pending.Type != tc.wantPend {
				t.Fatalf("Pending.Type: want %d, got %d", tc.wantPend, gs.Pending.Type)
			}

			// Whichever branch was taken, the game must never deadlock: some
			// legal action must exist next (ability-select actions if
			// pending, or snap/start-turn actions if fizzled).
			actions := gs.LegalActionsList()
			if len(actions) == 0 {
				t.Fatal("deadlock: LegalActions is empty after Replace")
			}

			switch tc.wantPend {
			case PendingBlindSwap:
				if !containsAction(actions, EncodeBlindSwap(0, 0)) {
					t.Error("expected BlindSwap(0,0) to be a legal select action")
				}
			case PendingKingLook:
				if !containsAction(actions, EncodeKingLook(0, 0)) {
					t.Error("expected KingLook(0,0) to be a legal select action")
				}
			case PendingPeekOwn:
				if !containsAction(actions, EncodePeekOwn(0)) {
					t.Error("expected PeekOwn(0) to be a legal select action")
				}
			case PendingPeekOther:
				if !containsAction(actions, EncodePeekOther(0)) {
					t.Error("expected PeekOther(0) to be a legal select action")
				}
			}
		})
	}
}

// TestReplaceAbilityHandCountFizzleUnchanged is a regression test: routing
// replace()'s ability trigger through canUseAbility must preserve the
// pre-existing hand-count fizzle behavior (empty opponent hand) independent
// of LockCallerHand.
func TestReplaceAbilityHandCountFizzleUnchanged(t *testing.T) {
	cases := []struct {
		name string
		rank uint8
	}{
		{"nine_opponent_hand_empty_fizzles", RankNine},
		{"jack_opponent_hand_empty_fizzles", RankJack},
		{"king_opponent_hand_empty_fizzles", RankKing},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rules := DefaultHouseRules()
			rules.AllowReplaceAbilities = true
			gs := NewGame(42, rules)
			gs.Deal()
			gs.CurrentPlayer = 0
			gs.Players[1].HandLen = 0 // opponent has no cards to target

			abilityCard := NewCard(SuitClubs, tc.rank)
			gs.Players[0].Hand[0] = abilityCard
			gs.Stockpile[gs.StockLen] = NewCard(SuitDiamonds, RankAce)
			gs.StockLen++

			if err := gs.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if err := gs.ApplyAction(EncodeReplace(0)); err != nil {
				t.Fatalf("Replace: %v", err)
			}

			if gs.Pending.Type == PendingPeekOther || gs.Pending.Type == PendingBlindSwap || gs.Pending.Type == PendingKingLook {
				t.Errorf("rank %s: ability should have fizzled with an empty opponent hand, got Pending.Type=%d", rankName(tc.rank), gs.Pending.Type)
			}
		})
	}
}
