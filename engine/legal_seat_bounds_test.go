package engine

import "testing"

// The 146-action space encodes one opponent, so a 3+ seat table belongs to NPlayerLegalActions.
// Nothing in the service asks for the 2-player mask at such a table today, which is the only
// reason this was not the cambia-1125 panic: legalAbilitySelect and legalSnapDecision named the
// opponent with OpponentOf (1-acting), and from seat 2 that underflows to 255 and indexes off the
// end of Players. These tests hold the mask path in bounds at three seats and hold it byte-for-
// byte where it is defined, at two (cambia-1099 K3).

// threeSeatGame deals a 3-player game, the table shape the 2-player mask has no encoding for.
func threeSeatGame(t *testing.T) *GameState {
	t.Helper()
	g := NewGame(4242, nplayerRules(3))
	g.Deal()
	return &g
}

// TestLegalActionsAtThreeSeatsStaysInBounds drives every 2-player mask context that names an
// opponent from the highest seat, where 1-acting is not a seat at all.
func TestLegalActionsAtThreeSeatsStaysInBounds(t *testing.T) {
	const seat = uint8(2)

	abilities := []struct {
		name    string
		pending PendingType
	}{
		{"PeekOwn", PendingPeekOwn},
		{"PeekOther", PendingPeekOther},
		{"BlindSwap", PendingBlindSwap},
		{"KingLook", PendingKingLook},
		{"KingDecision", PendingKingDecision},
	}
	for _, a := range abilities {
		t.Run(a.name, func(t *testing.T) {
			g := threeSeatGame(t)
			g.Pending.Type = a.pending
			g.Pending.PlayerID = seat
			if ctx := g.DecisionCtx(); ctx != CtxAbilitySelect {
				t.Fatalf("DecisionCtx=%d, want CtxAbilitySelect (%d)", ctx, CtxAbilitySelect)
			}
			mask := g.LegalActions()
			if mask == [3]uint64{} {
				t.Errorf("%s from seat %d produced an empty mask", a.name, seat)
			}
		})
	}

	// The swap abilities read the Cambia caller through the seat they named, so the locked-hand
	// branch is walked from the high seat too.
	t.Run("BlindSwapWithLockedCaller", func(t *testing.T) {
		g := threeSeatGame(t)
		g.Rules.LockCallerHand = true
		g.Flags |= FlagCambiaCalled
		g.CambiaCaller = 0
		g.Pending.Type = PendingBlindSwap
		g.Pending.PlayerID = seat
		g.LegalActions()
	})

	t.Run("SnapDecision", func(t *testing.T) {
		g := threeSeatGame(t)
		g.Snap.Active = true
		g.Snap.Snappers[0] = seat
		g.Snap.NumSnappers = 1
		g.Snap.CurrentSnapperIdx = 0
		g.Snap.DiscardedRank = RankFive
		if ctx := g.DecisionCtx(); ctx != CtxSnapDecision {
			t.Fatalf("DecisionCtx=%d, want CtxSnapDecision (%d)", ctx, CtxSnapDecision)
		}
		if mask := g.LegalActions(); mask == [3]uint64{} {
			t.Errorf("snap decision from seat %d produced an empty mask", seat)
		}
	})
}

// TestMaskOpponentIsOpponentOfAtTwoSeats is the no-change half. maskOpponent is the only input
// either mask arm gained, so agreeing with OpponentOf on every seat of a 2-player table is
// agreeing on every bit those arms set.
func TestMaskOpponentIsOpponentOfAtTwoSeats(t *testing.T) {
	g := newDealtGame(t)
	if n := g.Rules.numPlayers(); n != 2 {
		t.Fatalf("numPlayers=%d, want the 2-player default", n)
	}
	for seat := uint8(0); seat < 2; seat++ {
		if got, want := g.maskOpponent(seat), g.OpponentOf(seat); got != want {
			t.Errorf("maskOpponent(%d)=%d, want OpponentOf(%d)=%d", seat, got, seat, want)
		}
	}
}

// TestTwoSeatMasksSurviveAPlayout walks a whole 2-player game and holds the same equality at
// every state the mask is actually asked for, rather than at the dealt state alone.
func TestTwoSeatMasksSurviveAPlayout(t *testing.T) {
	g := NewGame(7, DefaultHouseRules())
	g.Deal()

	for steps := 0; steps < 2000 && !g.IsTerminal(); steps++ {
		switch g.DecisionCtx() {
		case CtxAbilitySelect:
			checkSeatAgreement(t, &g, g.Pending.PlayerID)
		case CtxSnapDecision:
			checkSeatAgreement(t, &g, g.Snap.Snappers[g.Snap.CurrentSnapperIdx])
		}
		actions := g.LegalActionsList()
		if len(actions) == 0 {
			t.Fatalf("step %d: no legal actions but the game is not terminal (flags=%d)", steps, g.Flags)
		}
		idx := g.randN(uint64(len(actions)))
		if err := g.ApplyAction(actions[idx]); err != nil {
			t.Fatalf("step %d: ApplyAction(%d): %v", steps, actions[idx], err)
		}
	}
}

func checkSeatAgreement(t *testing.T, g *GameState, seat uint8) {
	t.Helper()
	if got, want := g.maskOpponent(seat), g.OpponentOf(seat); got != want {
		t.Fatalf("maskOpponent(%d)=%d, want OpponentOf(%d)=%d", seat, got, seat, want)
	}
}
