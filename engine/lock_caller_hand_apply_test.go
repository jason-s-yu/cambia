// engine/lock_caller_hand_apply_test.go
// LockCallerHand at the apply paths, where the legal mask is not a guard: the service adapter hands
// an action straight to ApplyAction or ApplyNPlayerAction (service/internal/game/engine_adapter.go
// applyToEngine), so a hand-rolled or stale client's swap onto the Cambia caller's frozen hand is
// refused here or not at all. The other side of the same rule is that the freeze follows the house
// rule rather than the call: with lockCallerHand off, the configuration ranked play uses
// (MATCHMAKING.md 5.2), the caller keeps playing the snap window like everyone else (cambia-1118).
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

// snapWindowFor plants rank across the named seats' first slot, empties the rest of every hand, and
// opens the snap window for that rank. It returns the seats the window admitted.
func snapWindowFor(t *testing.T, g *GameState, rank uint8, holders ...uint8) []uint8 {
	t.Helper()
	n := g.Rules.numPlayers()
	holds := make(map[uint8]bool, len(holders))
	for _, h := range holders {
		holds[h] = true
	}
	for seat := uint8(0); seat < n; seat++ {
		g.Players[seat].HandLen = 1
		g.Players[seat].Hand[0] = NewCard(SuitClubs, RankTwo)
		if holds[seat] {
			g.Players[seat].Hand[0] = NewCard(SuitClubs, rank)
		}
	}
	g.CurrentPlayer = 0
	g.initiateSnapPhase(NewCard(SuitHearts, rank))
	if !g.Snap.Active {
		return nil
	}
	return append([]uint8(nil), g.Snap.Snappers[:g.Snap.NumSnappers]...)
}

// TestLockedCallerSitsOutSnapWindow: RULES.md 3C bars the locked caller from snapping in either
// direction, since snapping their own card empties a slot in the frozen hand and snapping an
// opponent's obliges them to pay a card out of it.
func TestLockedCallerSitsOutSnapWindow(t *testing.T) {
	g := lockedCallerTable(t, 3, true)
	got := snapWindowFor(t, g, RankFive, 0, 1, 2)
	for _, seat := range got {
		if seat == 1 {
			t.Fatalf("the locked caller is in the snap window %v", got)
		}
	}
	if len(got) != 2 {
		t.Errorf("snap window %v, want the two unlocked seats", got)
	}
}

// TestCallerSnapsWhenLockOff is the rule the unconditional exclusion broke: with lockCallerHand off
// nothing freezes the caller's hand, so they play the snap window like any other seat.
func TestCallerSnapsWhenLockOff(t *testing.T) {
	g := lockedCallerTable(t, 3, false)
	got := snapWindowFor(t, g, RankFive, 0, 1, 2)
	found := false
	for _, seat := range got {
		if seat == 1 {
			found = true
		}
	}
	if !found {
		t.Errorf("snap window %v leaves out the caller with lockCallerHand off", got)
	}
}

// TestCallerIsSnappableWhenLockOff covers the other direction of the same exclusion: the caller's
// cards are a legal target for everyone else's snap once the lock is off. Only the caller holds the
// discarded rank, so seats 0 and 2 can only be in the window through the caller's hand, and with
// the lock on nobody is: the caller cannot snap their own card and no one else may take it.
func TestCallerIsSnappableWhenLockOff(t *testing.T) {
	for _, tc := range []struct {
		name string
		lock bool
		want []uint8
	}{
		{"locked", true, nil},
		{"unlocked", false, []uint8{0, 1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := lockedCallerTable(t, 3, tc.lock)
			g.Rules.AllowOpponentSnapping = true
			got := snapWindowFor(t, g, RankFive, 1)
			if len(got) != len(tc.want) {
				t.Fatalf("snap window %v, want %v", got, tc.want)
			}
			for i, seat := range tc.want {
				if got[i] != seat {
					t.Fatalf("snap window %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// TestSnapOpponentRefusesLockedCaller is the target side of the same rule, and the side the mask
// used to leave open: a seat enters the snap window on its own matching card alone, so being an
// admitted snapper says nothing about whether the hand it names is frozen. With the lock on, the
// caller's cards are neither offered by the mask nor taken by the apply path, in either action
// space, and the refusal costs no penalty because the target was never legal to name (cambia-1239).
func TestSnapOpponentRefusesLockedCaller(t *testing.T) {
	// Both seats hold the discarded rank, so seat 0 is admitted on its own card while the caller
	// at seat 1 is the frozen hand seat 0 must not be able to name.
	t.Run("two-player space", func(t *testing.T) {
		g := lockedCallerTable(t, 2, true)
		g.Rules.AllowOpponentSnapping = true
		got := snapWindowFor(t, g, RankFive, 0, 1)
		if len(got) != 1 || got[0] != 0 {
			t.Fatalf("snap window %v, want seat 0 alone (the caller sits out)", got)
		}

		action := EncodeSnapOpponent(0)
		if mask := g.LegalActions(); mask[action/64]>>(action%64)&1 == 1 {
			t.Error("the 2-player mask offers a snap against the locked caller's hand")
		}

		before := handsOf(g)
		handLen := g.Players[0].HandLen
		discardLen := g.DiscardLen
		if err := g.ApplyAction(action); err == nil {
			t.Fatal("snapping the locked caller's card succeeded; it must be refused")
		}
		if got := handsOf(g); got != before {
			t.Error("the refused snap moved a card")
		}
		if g.Players[0].HandLen != handLen {
			t.Errorf("the snapper's hand went from %d to %d cards; the refusal must draw no penalty", handLen, g.Players[0].HandLen)
		}
		if g.DiscardLen != discardLen {
			t.Error("the refused snap put a card on the discard pile")
		}
	})

	t.Run("n-player space", func(t *testing.T) {
		g := lockedCallerTable(t, 3, true)
		g.Rules.AllowOpponentSnapping = true
		got := snapWindowFor(t, g, RankFive, 0, 1)
		if len(got) == 0 || got[0] != 0 {
			t.Fatalf("snap window %v, want seat 0 acting first", got)
		}
		if opps := g.Opponents(0); opps[0] != 1 {
			t.Fatalf("relative opponent 0 of seat 0 is seat %d, want the caller's seat 1", opps[0])
		}

		action := NPlayerEncodeSnapOpponent(0, 0)
		if mask := g.NPlayerLegalActions(); mask[action/64]>>(action%64)&1 == 1 {
			t.Error("the N-player mask offers a snap against the locked caller's hand")
		}
		// The unlocked opponent stays snappable, so the skip is per target rather than a blanket
		// close of the opponent-snap branch.
		if other := NPlayerEncodeSnapOpponent(0, 1); g.NPlayerLegalActions()[other/64]>>(other%64)&1 == 0 {
			t.Error("the N-player mask dropped the unlocked opponent along with the caller")
		}

		before := handsOf(g)
		handLen := g.Players[0].HandLen
		discardLen := g.DiscardLen
		if err := g.ApplyNPlayerAction(action); err == nil {
			t.Fatal("snapping the locked caller's card succeeded; it must be refused")
		}
		if got := handsOf(g); got != before {
			t.Error("the refused snap moved a card")
		}
		if g.Players[0].HandLen != handLen {
			t.Errorf("the snapper's hand went from %d to %d cards; the refusal must draw no penalty", handLen, g.Players[0].HandLen)
		}
		if g.DiscardLen != discardLen {
			t.Error("the refused snap put a card on the discard pile")
		}
	})
}
