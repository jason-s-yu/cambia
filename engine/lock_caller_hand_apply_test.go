// engine/lock_caller_hand_apply_test.go
// LockCallerHand at the apply paths, where the legal mask is not a guard: the service adapter hands
// an action straight to ApplyAction or ApplyNPlayerAction (service/internal/game/engine_adapter.go
// applyToEngine), so a hand-rolled or stale client's swap onto the Cambia caller's frozen hand is
// refused here or not at all (cambia-1118).
package engine

import "testing"

// lockedCallerTable deals an n-seat game, has seat 1 call Cambia, and sets the lock rule.
func lockedCallerTable(t *testing.T, n uint8, lock bool) *GameState {
	t.Helper()
	g := NewGame(17, nplayerRules(n))
	g.Deal()
	g.Rules.LockCallerHand = lock
	g.CambiaCaller = 1
	g.Flags |= FlagCambiaCalled
	return &g
}

// handsOf snapshots every seat's hand, so a refused action can be shown to have moved nothing.
func handsOf(g *GameState) [MaxPlayers][MaxHandSize]Card {
	var out [MaxPlayers][MaxHandSize]Card
	for seat := 0; seat < MaxPlayers; seat++ {
		out[seat] = g.Players[seat].Hand
	}
	return out
}

// TestNPlayerSwapAbilitiesRefuseLockedCaller: blindSwapNPlayer and kingLookNPlayer refuse the
// locked caller as a target, matching the N-player mask that omits those same actions
// (nplayerLegalAbilitySelect).
func TestNPlayerSwapAbilitiesRefuseLockedCaller(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pending PendingType
		action  uint16
	}{
		{"blind swap", PendingBlindSwap, NPlayerEncodeBlindSwap(0, 0, 0)},
		{"king look", PendingKingLook, NPlayerEncodeKingLook(0, 0, 0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := lockedCallerTable(t, 3, true)
			g.Pending = PendingAction{Type: tc.pending, PlayerID: 0}
			if opps := g.Opponents(0); opps[0] != 1 {
				t.Fatalf("relative opponent 0 of seat 0 is seat %d, want the caller's seat 1", opps[0])
			}
			if mask := g.NPlayerLegalActions(); mask[tc.action/64]>>(tc.action%64)&1 == 1 {
				t.Fatal("the N-player mask offers the action this test applies against it")
			}

			before := handsOf(g)
			if err := g.ApplyNPlayerAction(tc.action); err == nil {
				t.Fatal("applying the action against the locked caller succeeded; it must be refused")
			}
			if got := handsOf(g); got != before {
				t.Error("the refused action moved a card")
			}
			if g.Pending.Type != tc.pending {
				t.Errorf("pending type is %d after the refusal, want %d left armed for a legal target", g.Pending.Type, tc.pending)
			}
		})
	}
}

// TestNPlayerSwapAbilitiesReachCallerWhenLockOff is the same table with the house rule off: the
// caller's hand is ordinary, so the swap lands.
func TestNPlayerSwapAbilitiesReachCallerWhenLockOff(t *testing.T) {
	g := lockedCallerTable(t, 3, false)
	g.Pending = PendingAction{Type: PendingBlindSwap, PlayerID: 0}

	ownBefore := g.Players[0].Hand[0]
	callerBefore := g.Players[1].Hand[0]
	if err := g.ApplyNPlayerAction(NPlayerEncodeBlindSwap(0, 0, 0)); err != nil {
		t.Fatalf("blind swap onto an unlocked caller: %v", err)
	}
	if g.Players[0].Hand[0] != callerBefore || g.Players[1].Hand[0] != ownBefore {
		t.Error("the cards did not change hands with lockCallerHand off")
	}
}

// TestTwoPlayerSwapAbilitiesRefuseLockedCaller covers the 2-seat action space, the one the adapter
// applies at a 2-seat table (engine_adapter.go applyToEngine) and the one CFR traverses.
func TestTwoPlayerSwapAbilitiesRefuseLockedCaller(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pending PendingType
		action  uint16
	}{
		{"blind swap", PendingBlindSwap, EncodeBlindSwap(0, 0)},
		{"king look", PendingKingLook, EncodeKingLook(0, 0)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := lockedCallerTable(t, 2, true)
			g.Pending = PendingAction{Type: tc.pending, PlayerID: 0}
			if mask := g.LegalActions(); mask[tc.action/64]>>(tc.action%64)&1 == 1 {
				t.Fatal("the 2-player mask offers the action this test applies against it")
			}

			before := handsOf(g)
			if err := g.ApplyAction(tc.action); err == nil {
				t.Fatal("applying the action against the locked caller succeeded; it must be refused")
			}
			if got := handsOf(g); got != before {
				t.Error("the refused action moved a card")
			}
		})
	}
}
