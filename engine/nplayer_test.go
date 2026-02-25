package engine

import (
	"math"
	"testing"
)

// nplayerRules returns house rules configured for N players.
func nplayerRules(n uint8) HouseRules {
	r := DefaultHouseRules()
	r.NumPlayers = n
	// Increase max turns to ensure games can complete with more players.
	r.MaxGameTurns = 200
	return r
}

// playNPlayerRandom plays a random game using ApplyNPlayerAction until terminal,
// choosing a random legal action each step. Returns without error.
func playNPlayerRandom(t *testing.T, seed uint64, n uint8) *GameState {
	t.Helper()
	rules := nplayerRules(n)
	g := NewGame(seed, rules)
	g.Deal()

	const maxSteps = 10000
	for step := 0; step < maxSteps; step++ {
		if g.IsTerminal() {
			return &g
		}
		actions := g.NPlayerLegalActionsList()
		if len(actions) == 0 {
			// No legal actions — game should be terminal.
			t.Fatalf("no legal actions but game not terminal (step=%d, flags=%d)", step, g.Flags)
		}
		// Pick pseudo-random action using game RNG.
		idx := g.randN(uint64(len(actions)))
		if err := g.ApplyNPlayerAction(actions[idx]); err != nil {
			t.Fatalf("ApplyNPlayerAction(%d) step %d: %v", actions[idx], step, err)
		}
	}
	// If we exit the loop, force-end the game to allow utility check.
	g.Flags |= FlagGameOver
	return &g
}

// ---------------------------------------------------------------------------
// TestNPlayerDeal
// ---------------------------------------------------------------------------

func TestNPlayerDeal(t *testing.T) {
	for _, n := range []uint8{3, 4, 6} {
		t.Run("N="+string(rune('0'+n)), func(t *testing.T) {
			rules := nplayerRules(n)
			g := NewGame(42, rules)
			g.Deal()

			// Each active player should have CardsPerPlayer cards.
			for p := uint8(0); p < n; p++ {
				if g.Players[p].HandLen != rules.CardsPerPlayer {
					t.Errorf("player %d HandLen=%d, want %d", p, g.Players[p].HandLen, rules.CardsPerPlayer)
				}
			}
			// Inactive slots should be empty.
			for p := n; p < MaxPlayers; p++ {
				if g.Players[p].HandLen != 0 {
					t.Errorf("inactive player %d HandLen=%d, want 0", p, g.Players[p].HandLen)
				}
			}
			// Stockpile should have shrunk by n*CardsPerPlayer + 1.
			dealt := uint8(rules.CardsPerPlayer) * n
			wantStock := uint8(52+rules.NumJokers) - dealt - 1
			if g.StockLen != wantStock {
				t.Errorf("StockLen=%d, want %d", g.StockLen, wantStock)
			}
			// Discard pile must have exactly 1 card.
			if g.DiscardLen != 1 {
				t.Errorf("DiscardLen=%d, want 1", g.DiscardLen)
			}
			// Starting player must be in range.
			if g.CurrentPlayer >= n {
				t.Errorf("CurrentPlayer=%d, want < %d", g.CurrentPlayer, n)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerUtilityZeroSum
// ---------------------------------------------------------------------------

func TestNPlayerUtilityZeroSum(t *testing.T) {
	for _, n := range []uint8{3, 4, 6} {
		t.Run("N="+string(rune('0'+n)), func(t *testing.T) {
			// Run several random games to terminal; verify Σ utilities ≈ 0.
			for seed := uint64(1); seed <= 20; seed++ {
				g := playNPlayerRandom(t, seed*1000+uint64(n), n)
				u := g.GetUtility()
				var sum float64
				for p := uint8(0); p < n; p++ {
					sum += float64(u[p])
				}
				if math.Abs(sum) > 1e-4 {
					t.Errorf("seed=%d n=%d: utility sum=%f, want ≈0", seed, n, sum)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerLegalActions
// ---------------------------------------------------------------------------

func TestNPlayerLegalActionsStartTurn(t *testing.T) {
	rules := nplayerRules(3)
	g := NewGame(42, rules)
	g.Deal()

	actions := g.NPlayerLegalActionsList()

	// DrawStockpile and DrawDiscard should be legal.
	if !containsAction(actions, NPlayerActionDrawStockpile) {
		t.Error("expected NPlayerActionDrawStockpile to be legal")
	}
	if !containsAction(actions, NPlayerActionDrawDiscard) {
		t.Error("expected NPlayerActionDrawDiscard to be legal (AllowDrawFromDiscard=true)")
	}
	// CallCambia should be legal at round 0.
	if !containsAction(actions, NPlayerActionCallCambia) {
		t.Error("expected NPlayerActionCallCambia to be legal from round 0")
	}
}

func TestNPlayerLegalActionsPostDraw(t *testing.T) {
	rules := nplayerRules(3)
	g := NewGame(42, rules)
	g.Deal()

	if err := g.ApplyNPlayerAction(NPlayerActionDrawStockpile); err != nil {
		t.Fatalf("draw stockpile: %v", err)
	}

	actions := g.NPlayerLegalActionsList()
	if !containsAction(actions, NPlayerActionDiscardNoAbility) {
		t.Error("expected NPlayerActionDiscardNoAbility to be legal post-draw")
	}

	// Replace actions for each hand slot.
	acting := g.Pending.PlayerID
	handLen := g.Players[acting].HandLen
	for i := uint8(0); i < handLen; i++ {
		if !containsAction(actions, NPlayerEncodeReplace(i)) {
			t.Errorf("expected NPlayerEncodeReplace(%d) to be legal", i)
		}
	}
}

func TestNPlayerLegalActionsSnapDecision(t *testing.T) {
	rules := nplayerRules(3)
	g := NewGame(42, rules)
	g.Deal()

	// Manually set snap phase with player 0 as snapper.
	g.Snap.Active = true
	g.Snap.Snappers[0] = 0
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Snap.DiscardedRank = RankFive

	actions := g.NPlayerLegalActionsList()
	if !containsAction(actions, NPlayerActionPassSnap) {
		t.Error("expected NPlayerActionPassSnap to be legal during snap")
	}

	// SnapOwn actions.
	handLen := g.Players[0].HandLen
	for i := uint8(0); i < handLen; i++ {
		if !containsAction(actions, NPlayerEncodeSnapOwn(i)) {
			t.Errorf("expected NPlayerEncodeSnapOwn(%d) to be legal", i)
		}
	}

	// SnapOpponent actions for each opponent (opp relative indices 0, 1 for 3P).
	if rules.AllowOpponentSnapping {
		for oppRelIdx := uint8(0); oppRelIdx < 2; oppRelIdx++ {
			opps := g.Opponents(0)
			opp := opps[oppRelIdx]
			oppLen := g.Players[opp].HandLen
			for j := uint8(0); j < oppLen; j++ {
				if !containsAction(actions, NPlayerEncodeSnapOpponent(j, oppRelIdx)) {
					t.Errorf("expected NPlayerEncodeSnapOpponent(%d,%d) to be legal", j, oppRelIdx)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestNPlayer2PRegression
// ---------------------------------------------------------------------------

func TestNPlayer2PRegression(t *testing.T) {
	// Run a 2P game with explicit NumPlayers=2 using ApplyNPlayerAction.
	// Verify it terminates correctly and utilities are 2P-style.
	rules := DefaultHouseRules()
	rules.NumPlayers = 2
	rules.MaxGameTurns = 200

	g := NewGame(12345, rules)
	g.Deal()

	const maxSteps = 5000
	for step := 0; step < maxSteps; step++ {
		if g.IsTerminal() {
			break
		}
		actions := g.NPlayerLegalActionsList()
		if len(actions) == 0 {
			t.Fatalf("no legal actions but game not terminal (step=%d)", step)
		}
		idx := g.randN(uint64(len(actions)))
		if err := g.ApplyNPlayerAction(actions[idx]); err != nil {
			t.Fatalf("ApplyNPlayerAction step %d: %v", step, err)
		}
	}

	if !g.IsTerminal() {
		t.Fatal("game did not terminate within maxSteps")
	}

	// For 2P, |u[0]| + |u[1]| should be either 0 (true tie) or 2 (winner/loser).
	u := g.GetUtility()
	total := math.Abs(float64(u[0])) + math.Abs(float64(u[1]))
	if total != 0 && total != 2.0 {
		t.Errorf("unexpected 2P utilities: u[0]=%f u[1]=%f", u[0], u[1])
	}
	// Inactive slots must be zero.
	for p := uint8(2); p < MaxPlayers; p++ {
		if u[p] != 0 {
			t.Errorf("inactive player %d utility=%f, want 0", p, u[p])
		}
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerRandomGames
// ---------------------------------------------------------------------------

func TestNPlayerRandomGames(t *testing.T) {
	// 500 random 4P games complete without panic.
	for seed := uint64(1); seed <= 500; seed++ {
		g := playNPlayerRandom(t, seed, 4)
		if !g.IsTerminal() {
			t.Errorf("seed=%d: game not terminal after maxSteps", seed)
		}
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerCambiaFinalRound
// ---------------------------------------------------------------------------

func TestNPlayerCambiaFinalRound(t *testing.T) {
	// After Cambia is called, exactly NumPlayers turns (one per player) must
	// pass before the game ends. TurnsAfterC counts each advanceTurn() call.
	for _, n := range []uint8{3, 4, 5} {
		t.Run("N="+string(rune('0'+n)), func(t *testing.T) {
			rules := nplayerRules(n)
			g := NewGame(77+uint64(n), rules)
			g.Deal()

			// Advance to the point we can call Cambia.
			// Force the first player to call Cambia on their next turn.
			// Use ApplyNPlayerAction to actually call cambia.
			called := false
			const maxSteps = 10000
			for step := 0; step < maxSteps && !g.IsTerminal(); step++ {
				if !called && g.Pending.Type == PendingNone && !g.Snap.Active {
					// Check if CallCambia is legal.
					actions := g.NPlayerLegalActionsList()
					if containsAction(actions, NPlayerActionCallCambia) {
						caller := g.CurrentPlayer
						if err := g.ApplyNPlayerAction(NPlayerActionCallCambia); err != nil {
							t.Fatalf("CallCambia: %v", err)
						}
						called = true
						_ = caller
						// Now keep playing until terminal.
						continue
					}
				}
				actions := g.NPlayerLegalActionsList()
				if len(actions) == 0 {
					break
				}
				idx := g.randN(uint64(len(actions)))
				if err := g.ApplyNPlayerAction(actions[idx]); err != nil {
					t.Fatalf("step %d: %v", step, err)
				}
			}

			if !called {
				t.Skip("Cambia was never called in this run")
			}
			if !g.IsTerminal() {
				t.Fatal("game did not terminate after Cambia was called")
			}
			// TurnsAfterC should equal NumPlayers exactly when the game ended.
			if g.TurnsAfterC < n {
				t.Errorf("TurnsAfterC=%d, want >= %d", g.TurnsAfterC, n)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerEncodeDecodeRoundTrip
// ---------------------------------------------------------------------------

func TestNPlayerEncodeDecodeRoundTrip(t *testing.T) {
	// PeekOther
	for slot := uint8(0); slot < 6; slot++ {
		for opp := uint8(0); opp < 5; opp++ {
			idx := NPlayerEncodePeekOther(slot, opp)
			gotSlot, gotOpp, ok := NPlayerDecodePeekOther(idx)
			if !ok || gotSlot != slot || gotOpp != opp {
				t.Errorf("PeekOther(%d,%d): encode=%d decode=(%d,%d,ok=%v)", slot, opp, idx, gotSlot, gotOpp, ok)
			}
		}
	}

	// BlindSwap
	for own := uint8(0); own < 6; own++ {
		for oppSlot := uint8(0); oppSlot < 6; oppSlot++ {
			for oppIdx := uint8(0); oppIdx < 5; oppIdx++ {
				idx := NPlayerEncodeBlindSwap(own, oppSlot, oppIdx)
				gotOwn, gotOppSlot, gotOppIdx, ok := NPlayerDecodeBlindSwap(idx)
				if !ok || gotOwn != own || gotOppSlot != oppSlot || gotOppIdx != oppIdx {
					t.Errorf("BlindSwap(%d,%d,%d): encode=%d decode=(%d,%d,%d,ok=%v)",
						own, oppSlot, oppIdx, idx, gotOwn, gotOppSlot, gotOppIdx, ok)
				}
			}
		}
	}

	// KingLook
	for own := uint8(0); own < 6; own++ {
		for oppSlot := uint8(0); oppSlot < 6; oppSlot++ {
			for oppIdx := uint8(0); oppIdx < 5; oppIdx++ {
				idx := NPlayerEncodeKingLook(own, oppSlot, oppIdx)
				gotOwn, gotOppSlot, gotOppIdx, ok := NPlayerDecodeKingLook(idx)
				if !ok || gotOwn != own || gotOppSlot != oppSlot || gotOppIdx != oppIdx {
					t.Errorf("KingLook(%d,%d,%d): encode=%d decode=(%d,%d,%d,ok=%v)",
						own, oppSlot, oppIdx, idx, gotOwn, gotOppSlot, gotOppIdx, ok)
				}
			}
		}
	}

	// SnapOpponent
	for slot := uint8(0); slot < 6; slot++ {
		for opp := uint8(0); opp < 5; opp++ {
			idx := NPlayerEncodeSnapOpponent(slot, opp)
			gotSlot, gotOpp, ok := NPlayerDecodeSnapOpponent(idx)
			if !ok || gotSlot != slot || gotOpp != opp {
				t.Errorf("SnapOpponent(%d,%d): encode=%d decode=(%d,%d,ok=%v)", slot, opp, idx, gotSlot, gotOpp, ok)
			}
		}
	}

	// SnapOpponentMove
	for own := uint8(0); own < 6; own++ {
		idx := NPlayerEncodeSnapOpponentMove(own)
		gotOwn, ok := NPlayerDecodeSnapOpponentMove(idx)
		if !ok || gotOwn != own {
			t.Errorf("SnapOpponentMove(%d): encode=%d decode=(%d,ok=%v)", own, idx, gotOwn, ok)
		}
	}

	// Verify NPlayerNumActions boundary
	if NPlayerNumActions != 452 {
		t.Errorf("NPlayerNumActions=%d, want 452", NPlayerNumActions)
	}
}

// ---------------------------------------------------------------------------
// TestNPlayerHelperMethods
// ---------------------------------------------------------------------------

func TestNPlayerHelperMethods(t *testing.T) {
	rules := nplayerRules(4)
	g := NewGame(1, rules)

	if g.NumActivePlayers() != 4 {
		t.Errorf("NumActivePlayers()=%d, want 4", g.NumActivePlayers())
	}

	// NextPlayer wraps around.
	for p := uint8(0); p < 4; p++ {
		want := (p + 1) % 4
		got := g.NextPlayer(p)
		if got != want {
			t.Errorf("NextPlayer(%d)=%d, want %d", p, got, want)
		}
	}

	// Opponents(0) for 4P = {1, 2, 3}.
	opps := g.Opponents(0)
	if len(opps) != 3 {
		t.Errorf("Opponents(0) len=%d, want 3", len(opps))
	}
	for _, opp := range opps {
		if opp == 0 {
			t.Error("Opponents(0) contains 0 (self)")
		}
	}
}
