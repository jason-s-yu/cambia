package engine

import "testing"

// The legal-mask half of the 2-player surface was held in bounds at 3+ seats by cambia-1099 K3
// (legal_seat_bounds_test.go). The apply half still named the opponent with OpponentOf, which is
// 1-acting: correct at two seats, an underflow to 255 from seat 2, indexing off the end of Players.
// It stayed latent for the service, which routes a 3+ seat table through ApplyNPlayerAction, but
// not for the FFI surface: cambia_game_apply_action is the 2-player path at any seat count, and a
// 4-player GoEngine stepped uniformly over cambia_game_legal_actions panicked inside three games
// with "index out of range [255] with length 8" at snapOpponent (cambia-1426). These tests drive
// each 2-player apply path at 3+ seats and check it resolves against a real seat, not just that it
// survives (cambia-1171).

// TestTwoPlayerAbilityApplyPathsAtThreeSeatsResolveAgainstARealSeat drives every 2-player ability
// apply path from the highest seat of a 3-seat table, where 1-acting is not a seat. seatOpponent(2)
// at three seats is seat 0, so each assertion names seat 0 explicitly rather than settling for the
// absence of a panic.
func TestTwoPlayerAbilityApplyPathsAtThreeSeatsResolveAgainstARealSeat(t *testing.T) {
	const seat = uint8(2)
	const wantOpp = uint8(0) // (2+1) % 3

	t.Run("PeekOther", func(t *testing.T) {
		g := threeSeatGame(t)
		g.CurrentPlayer = seat
		g.Pending.Type = PendingPeekOther
		g.Pending.PlayerID = seat
		want := g.Players[wantOpp].Hand[0]

		if err := g.ApplyAction(EncodePeekOther(0)); err != nil {
			t.Fatalf("PeekOther(0) from seat %d: %v", seat, err)
		}
		if g.LastAction.RevealedOwner != wantOpp {
			t.Errorf("RevealedOwner=%d, want %d", g.LastAction.RevealedOwner, wantOpp)
		}
		if g.LastAction.RevealedCard != want {
			t.Errorf("RevealedCard=%d, want seat %d slot 0 (%d)", g.LastAction.RevealedCard, wantOpp, want)
		}
	})

	t.Run("BlindSwap", func(t *testing.T) {
		g := threeSeatGame(t)
		g.CurrentPlayer = seat
		g.Pending.Type = PendingBlindSwap
		g.Pending.PlayerID = seat
		beforeOwn := g.Players[seat].Hand[0]
		beforeOpp := g.Players[wantOpp].Hand[0]

		if err := g.ApplyAction(EncodeBlindSwap(0, 0)); err != nil {
			t.Fatalf("BlindSwap(0,0) from seat %d: %v", seat, err)
		}
		if g.Players[seat].Hand[0] != beforeOpp || g.Players[wantOpp].Hand[0] != beforeOwn {
			t.Errorf("swap did not exchange seat %d slot 0 with seat %d slot 0: own=%d opp=%d",
				seat, wantOpp, g.Players[seat].Hand[0], g.Players[wantOpp].Hand[0])
		}
	})

	t.Run("KingLookThenSwap", func(t *testing.T) {
		g := threeSeatGame(t)
		g.CurrentPlayer = seat
		g.Pending.Type = PendingKingLook
		g.Pending.PlayerID = seat
		beforeOwn := g.Players[seat].Hand[0]
		beforeOpp := g.Players[wantOpp].Hand[0]

		if err := g.ApplyAction(EncodeKingLook(0, 0)); err != nil {
			t.Fatalf("KingLook(0,0) from seat %d: %v", seat, err)
		}
		if g.LastAction.RevealedOwner != seat || g.LastAction.RevealedCard != beforeOwn {
			t.Errorf("look revealed owner=%d card=%d, want own seat %d card %d",
				g.LastAction.RevealedOwner, g.LastAction.RevealedCard, seat, beforeOwn)
		}
		if g.Pending.Type != PendingKingDecision {
			t.Fatalf("Pending.Type=%d after the look, want PendingKingDecision", g.Pending.Type)
		}
		if err := g.ApplyAction(ActionKingSwapYes); err != nil {
			t.Fatalf("KingSwapYes from seat %d: %v", seat, err)
		}
		if g.Players[seat].Hand[0] != beforeOpp || g.Players[wantOpp].Hand[0] != beforeOwn {
			t.Errorf("king swap did not exchange seat %d slot 0 with seat %d slot 0: own=%d opp=%d",
				seat, wantOpp, g.Players[seat].Hand[0], g.Players[wantOpp].Hand[0])
		}
	})

	t.Run("DiscardWithAbility", func(t *testing.T) {
		g := threeSeatGame(t)
		g.CurrentPlayer = seat
		g.Stockpile[g.StockLen] = NewCard(SuitClubs, RankTen) // peek other
		g.StockLen++

		if err := g.ApplyAction(ActionDrawStockpile); err != nil {
			t.Fatalf("DrawStockpile from seat %d: %v", seat, err)
		}
		if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
			t.Fatalf("DiscardWithAbility from seat %d: %v", seat, err)
		}
		if g.Pending.Type != PendingPeekOther {
			t.Fatalf("Pending.Type=%d, want PendingPeekOther (seat %d still holds cards)", g.Pending.Type, wantOpp)
		}
		if err := g.ApplyAction(EncodePeekOther(0)); err != nil {
			t.Fatalf("PeekOther(0) from seat %d: %v", seat, err)
		}
		if g.LastAction.RevealedOwner != wantOpp {
			t.Errorf("RevealedOwner=%d, want %d", g.LastAction.RevealedOwner, wantOpp)
		}
	})
}

// TestSnapOpponentAtThreeSeatsResolvesAgainstARealSeat is the snap.go half of the same problem, and
// the exact frame the cambia-1426 panic came out of: snapOpponent read the opponent through
// OpponentOf and then indexed Players with it.
func TestSnapOpponentAtThreeSeatsResolvesAgainstARealSeat(t *testing.T) {
	const seat = uint8(2)
	const wantOpp = uint8(0)

	g := threeSeatGame(t)
	if !g.Rules.AllowOpponentSnapping {
		t.Fatal("AllowOpponentSnapping is off in the default rules; this test needs it on")
	}
	g.CurrentPlayer = 0
	g.Players[wantOpp].Hand[0] = NewCard(SuitClubs, RankFive)
	g.Snap.Active = true
	g.Snap.Snappers[0] = seat
	g.Snap.NumSnappers = 1
	g.Snap.CurrentSnapperIdx = 0
	g.Snap.DiscardedRank = RankFive
	oppLenBefore := g.Players[wantOpp].HandLen

	if err := g.ApplyAction(EncodeSnapOpponent(0)); err != nil {
		t.Fatalf("SnapOpponent(0) from seat %d: %v", seat, err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatalf("snap of seat %d slot 0 failed; it held the discarded rank", wantOpp)
	}
	if g.LastAction.RevealedOwner != wantOpp {
		t.Errorf("RevealedOwner=%d, want %d", g.LastAction.RevealedOwner, wantOpp)
	}
	if g.Players[wantOpp].HandLen != oppLenBefore-1 {
		t.Errorf("seat %d HandLen=%d, want %d", wantOpp, g.Players[wantOpp].HandLen, oppLenBefore-1)
	}
	if g.Pending.Type != PendingSnapMove || g.Pending.Data[0] != wantOpp {
		t.Errorf("pending move: type=%d target seat=%d, want PendingSnapMove targeting seat %d",
			g.Pending.Type, g.Pending.Data[0], wantOpp)
	}
}

// TestUniformStepOverTwoPlayerMaskAtFourSeats is the cambia-1426 repro shape: a 4-seat game driven
// entirely through the 2-player surface, stepping uniformly over its own legal mask. Before the
// cambia-1171 fix it panicked inside a few games; the run below covers many more than that.
//
// An empty mask at a non-terminal state is the other failure this surface can produce, so it is
// checked here rather than being stepped past. Before cambia-1489, replace() armed an ability
// through canUseAbility, which asks about every opponent, while the 2-player mask enumerates only
// the seat seatOpponent names, and ResolveUntargetableArmedAbility was the only recovery (178
// stranded arms over 2000 seeds with AllowReplaceAbilities on). replace() now asks the calling
// space's own predicate at arm time, so the guard should never have to fire on this path any
// more; any empty mask, or any resolution the guard performs, fails the test.
func TestUniformStepOverTwoPlayerMaskAtFourSeats(t *testing.T) {
	for _, replaceAbilities := range []bool{false, true} {
		name := "replace_abilities_off"
		if replaceAbilities {
			name = "replace_abilities_on"
		}
		t.Run(name, func(t *testing.T) {
			const seeds = 2000
			resolved, steps := 0, 0
			for seed := uint64(1); seed <= seeds; seed++ {
				rules := nplayerRules(4)
				rules.AllowReplaceAbilities = replaceAbilities
				g := NewGame(seed, rules)
				g.Deal()

				for step := 0; step < 5000 && !g.IsTerminal(); step++ {
					steps++
					actions := g.LegalActionsList()
					if len(actions) == 0 {
						if g.ResolveUntargetableArmedAbility(false) {
							resolved++
							continue
						}
						t.Fatalf("seed %d step %d: no legal action at a non-terminal state (ctx=%d pending=%d snap=%v)",
							seed, step, g.DecisionCtx(), g.Pending.Type, g.Snap.Active)
					}
					idx := actions[g.randN(uint64(len(actions)))]
					if err := g.ApplyAction(idx); err != nil {
						t.Fatalf("seed %d step %d: ApplyAction(%d): %v", seed, step, idx, err)
					}
				}
			}
			// The repro panicked within about three games, so a run over this many is the
			// evidence. Games are short under uniform stepping because CambiaAllowedRound is 0 and
			// a random walk calls Cambia early, so the floor below only asserts the games play out
			// past the dealt state rather than pinning a length.
			if steps < seeds*10 {
				t.Errorf("only %d actions applied across %d seeds; the games are not playing out", steps, seeds)
			}
			if resolved != 0 {
				t.Errorf("%d armed abilities the 2-player mask could not target were left for the guard to resolve; replace() should fizzle them at arm time instead (cambia-1489)", resolved)
			}
			t.Logf("%d seeds, %d actions stepped clean; %d armed abilities the 2-player mask could not target were resolved by the guard", seeds, steps, resolved)
		})
	}
}

// TestAbilityHasTargetMatchesAbilitySelectMask holds the two arm-time predicates in step with the
// masks they mirror, over every seat count, seat, pending ability, hand-emptiness pattern and
// LockCallerHand configuration. Drift between them is what strands an ability: the arm site would
// say a target exists and the mask would offer no action to reach it.
func TestAbilityHasTargetMatchesAbilitySelectMask(t *testing.T) {
	pendings := []struct {
		name    string
		pending PendingType
	}{
		{"PeekOwn", PendingPeekOwn},
		{"PeekOther", PendingPeekOther},
		{"BlindSwap", PendingBlindSwap},
		{"KingLook", PendingKingLook},
		{"KingDecision", PendingKingDecision},
	}

	for _, n := range []uint8{2, 3, 4} {
		base := NewGame(31, nplayerRules(n))
		base.Deal()

		for _, p := range pendings {
			for acting := uint8(0); acting < n; acting++ {
				// Every subset of seats emptied, so the "some opponent but not that one" shapes
				// are all covered.
				for empties := 0; empties < 1<<n; empties++ {
					for _, lock := range []bool{false, true} {
						for caller := int8(-1); caller < int8(n); caller++ {
							g := base
							g.Rules.LockCallerHand = lock
							g.CambiaCaller = caller
							if caller >= 0 {
								g.Flags |= FlagCambiaCalled
							} else {
								g.Flags &^= FlagCambiaCalled
							}
							for seat := uint8(0); seat < n; seat++ {
								if empties&(1<<seat) != 0 {
									g.Players[seat].HandLen = 0
								}
							}
							g.Pending.Type = p.pending
							g.Pending.PlayerID = acting
							// KingDecision reads slots out of Pending.Data; leave them at 0.

							if ctx := g.DecisionCtx(); ctx != CtxAbilitySelect {
								t.Fatalf("n=%d %s acting=%d: DecisionCtx=%d, want CtxAbilitySelect", n, p.name, acting, ctx)
							}

							got2P := g.abilityHasTarget2P(p.pending, acting)
							want2P := g.LegalActions() != [3]uint64{}
							if got2P != want2P {
								t.Errorf("n=%d %s acting=%d empties=%04b lock=%v caller=%d: abilityHasTarget2P=%v, 2-player mask non-empty=%v",
									n, p.name, acting, empties, lock, caller, got2P, want2P)
							}

							gotNP := g.abilityHasTargetNP(p.pending, acting)
							wantNP := g.NPlayerLegalActions() != [10]uint64{}
							if gotNP != wantNP {
								t.Errorf("n=%d %s acting=%d empties=%04b lock=%v caller=%d: abilityHasTargetNP=%v, N-player mask non-empty=%v",
									n, p.name, acting, empties, lock, caller, gotNP, wantNP)
							}
						}
					}
				}
			}
		}
	}
}

// TestLockedCallerAbilityFizzlesAtDiscardInsteadOfArming covers the one 2-seat state where the
// discard arm and the ability-select mask disagreed: LockCallerHand puts the only opponent's hand
// out of reach, so legalAbilitySelect offers no blind swap or king look, but the arm checked hand
// counts alone and armed anyway. Nothing could then be applied, and the engine refuses every other
// action while it holds a pending ability.
//
// The mask assertion below is why that was never reproduced: legalPostDraw already refuses
// DiscardWithAbility here, so only a caller applying an action its own mask forbids could reach it.
// The fix is a guard on that basis. Python has fizzled this at trigger time since cambia-650
// (_trigger_discard_ability), so this also closes an engine divergence the parity gate could not
// generate.
func TestLockedCallerAbilityFizzlesAtDiscardInsteadOfArming(t *testing.T) {
	for _, tc := range []struct {
		name string
		rank uint8
	}{
		{"jack", RankJack},
		{"queen", RankQueen},
		{"king", RankKing},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGame(9, DefaultHouseRules())
			g.Deal()
			if !g.Rules.LockCallerHand {
				t.Fatal("LockCallerHand is off in the default rules; this test needs it on")
			}
			g.CurrentPlayer = 0
			g.Flags |= FlagCambiaCalled
			g.CambiaCaller = 1

			// No hand holds the discarded rank, so a fizzle resolves straight into advanceTurn
			// instead of opening a snap phase.
			for seat := uint8(0); seat < 2; seat++ {
				for i := uint8(0); i < g.Players[seat].HandLen; i++ {
					g.Players[seat].Hand[i] = NewCard(SuitClubs, RankTwo)
				}
			}
			g.Stockpile[g.StockLen] = NewCard(SuitDiamonds, tc.rank)
			g.StockLen++

			if err := g.ApplyAction(ActionDrawStockpile); err != nil {
				t.Fatalf("DrawStockpile: %v", err)
			}
			if mask := g.LegalActions(); mask[ActionDiscardWithAbility/64]>>(ActionDiscardWithAbility%64)&1 == 1 {
				t.Fatalf("legalPostDraw offers DiscardWithAbility with the caller's hand locked")
			}

			turnBefore := g.TurnNumber
			if err := g.ApplyAction(ActionDiscardWithAbility); err != nil {
				t.Fatalf("DiscardWithAbility: %v", err)
			}
			if g.Pending.Type != PendingNone {
				t.Errorf("Pending.Type=%d: an ability armed with an empty legal set", g.Pending.Type)
			}
			if g.Snap.Active {
				t.Errorf("snap phase opened; no hand holds rank %d", tc.rank)
			}
			if g.TurnNumber != turnBefore+1 {
				t.Errorf("TurnNumber=%d, want %d: the fizzle did not advance the turn", g.TurnNumber, turnBefore+1)
			}
		})
	}
}

// TestReplaceRoutesArmingThroughTheCallingSpacesPredicate covers the arm site cambia-1171 could
// not reach from abilities.go: replace() used to gate on canUseAbility, which asks about every
// opponent regardless of which action space called it, so at a 4 seat table driven through the
// 2-player surface it could arm an ability whose 2-player legal set was empty and strand the
// table behind ResolveUntargetableArmedAbility. replace() now asks abilityHasTarget2P or
// abilityHasTargetNP to match the space that called it (cambia-1489), so the same state produces
// different outcomes depending on which surface drove the Replace: the 2-player surface fizzles
// straight into the snap phase, and the N-player surface - which can still reach seats 2 and 3 -
// arms it as before.
func TestReplaceRoutesArmingThroughTheCallingSpacesPredicate(t *testing.T) {
	// newGame deals seat 0 a peek-other card to replace out of hand at a four seat table, with
	// seat 1 - the seat the 2-player space encodes as the opponent - holding nothing, and seats 2
	// and 3 holding cards. The 2-player mask cannot target the ability (its only opponent is seat
	// 1); the N-player mask can (seats 2 and 3 are reachable).
	newGame := func(t *testing.T) *GameState {
		t.Helper()
		rules := nplayerRules(4)
		rules.AllowReplaceAbilities = true
		g := NewGame(11, rules)
		g.Deal()
		g.CurrentPlayer = 0
		g.Players[1].HandLen = 0
		for seat := uint8(0); seat < 4; seat++ {
			for i := uint8(0); i < g.Players[seat].HandLen; i++ {
				g.Players[seat].Hand[i] = NewCard(SuitClubs, RankTwo)
			}
		}
		g.Players[0].Hand[0] = NewCard(SuitClubs, RankTen) // peek other, replaced out of hand
		g.Stockpile[g.StockLen] = NewCard(SuitDiamonds, RankAce)
		g.StockLen++

		if err := g.ApplyAction(ActionDrawStockpile); err != nil {
			t.Fatalf("DrawStockpile: %v", err)
		}
		return &g
	}

	t.Run("two_player_space_fizzles_instead_of_stranding", func(t *testing.T) {
		g := newGame(t)
		turnBefore := g.TurnNumber
		if err := g.ApplyAction(EncodeReplace(0)); err != nil {
			t.Fatalf("Replace(0) via the 2-player surface: %v", err)
		}
		if g.Pending.Type != PendingNone {
			t.Fatalf("Pending.Type=%d, want PendingNone: the 2-player mask cannot target this ability, so it should fizzle rather than arm", g.Pending.Type)
		}
		if g.Snap.Active {
			t.Errorf("snap phase opened; no hand holds the discarded rank (Ten)")
		}
		if g.TurnNumber != turnBefore+1 {
			t.Errorf("TurnNumber=%d, want %d: the fizzle did not advance the turn", g.TurnNumber, turnBefore+1)
		}
		// Nothing left for the guard to do: the fix removed the strand at the source.
		if g.ResolveUntargetableArmedAbility(false) {
			t.Error("guard resolved something; nothing should have been armed")
		}
	})

	t.Run("n_player_space_still_arms_it", func(t *testing.T) {
		g := newGame(t)
		if err := g.ApplyNPlayerAction(NPlayerEncodeReplace(0)); err != nil {
			t.Fatalf("Replace(0) via the N-player surface: %v", err)
		}
		if g.Pending.Type != PendingPeekOther {
			t.Fatalf("Pending.Type=%d, want PendingPeekOther: the N-player mask can still reach seats 2 and 3", g.Pending.Type)
		}
		if mask := g.NPlayerLegalActions(); mask == [10]uint64{} {
			t.Error("N-player mask is empty; the premise of this case does not hold")
		}
	})
}

// TestResolveUntargetableArmedAbilityGuardsAPreExistingStrandedState keeps
// ResolveUntargetableArmedAbility's own contract covered independently of any arm site: it is
// documented (abilities.go) to also guard callers holding a state built before cambia-1489's fix,
// e.g. a snapshot restored from before the fix landed. This constructs that state directly rather
// than through replace(), which no longer produces it.
func TestResolveUntargetableArmedAbilityGuardsAPreExistingStrandedState(t *testing.T) {
	rules := nplayerRules(4)
	g := NewGame(13, rules)
	g.Deal()
	g.Players[1].HandLen = 0
	g.DiscardPile[g.DiscardLen] = NewCard(SuitClubs, RankTen)
	g.DiscardLen++
	g.Pending.Type = PendingPeekOther
	g.Pending.PlayerID = 0

	turnBefore := g.TurnNumber
	if !g.ResolveUntargetableArmedAbility(false) {
		t.Fatal("guard did not resolve an ability the 2-player mask cannot target")
	}
	if g.Pending.Type != PendingNone {
		t.Errorf("Pending.Type=%d after resolution, want PendingNone", g.Pending.Type)
	}
	if g.TurnNumber != turnBefore+1 {
		t.Errorf("TurnNumber=%d, want %d: resolution did not advance the turn", g.TurnNumber, turnBefore+1)
	}
}

// TestActionSpaceSizesUnchanged pins both action-space sizes. The cambia-1173 ruling bars a decline
// action or any other action-space change as the escape from an unresolvable ability, so the fix
// had to be engine-internal (cambia-1171).
func TestActionSpaceSizesUnchanged(t *testing.T) {
	if NumActions != 146 {
		t.Errorf("NumActions=%d, want 146", NumActions)
	}
	if NPlayerNumActions != 620 {
		t.Errorf("NPlayerNumActions=%d, want 620", NPlayerNumActions)
	}
}
