package agent

import (
	"math/rand/v2"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// nplayer_hand_shift_test.go covers what the N-player belief handlers do when a hand
// changes length. A snap removes a card and shifts the rest of that hand left; the fill
// shifts the destination hand right; a failed snap draws a penalty onto the end. The
// handlers used to clear one slot and leave every slot above it naming the card that used
// to sit below, and the penalty draw was not modelled at all (cambia-1550).

// handShiftRules returns rules with opponent snapping on, which is what makes the fill
// reachable.
func handShiftRules(seats uint8, race bool) engine.HouseRules {
	r := engine.DefaultHouseRules()
	r.NumPlayers = seats
	r.AllowOpponentSnapping = true
	r.SnapRace = race
	r.MaxGameTurns = 200
	return r
}

// seatAgent returns an agent for one seat of a dealt game that knows every card of every
// hand, so any slot the handlers fail to shift shows up as a wrong bucket rather than as
// an absent one.
func seatAgent(g *engine.GameState, seat, seats uint8) AgentState {
	a := NewNPlayerAgentState(seat, seats, 1, 5)
	a.InitializeNPlayer(g)
	for s := uint8(0); s < seats; s++ {
		for c := uint8(0); c < g.Players[s].HandLen; c++ {
			a.nplayerSetKnown(nplayerSlot(s, c), CardToBucket(g.Players[s].Hand[c]))
		}
	}
	return a
}

// wantSlots asserts the agent's slot knowledge of one seat matches that seat's hand.
func wantSlots(t *testing.T, a *AgentState, g *engine.GameState, seat uint8, known ...bool) {
	t.Helper()
	if int(a.NPlayerHandLen[seat]) != len(known) {
		t.Fatalf("seat %d: tracked hand length %d, want %d", seat, a.NPlayerHandLen[seat], len(known))
	}
	if g.Players[seat].HandLen != a.NPlayerHandLen[seat] {
		t.Fatalf("seat %d: tracked hand length %d, engine holds %d",
			seat, a.NPlayerHandLen[seat], g.Players[seat].HandLen)
	}
	for c, wantKnown := range known {
		slot := nplayerSlot(seat, uint8(c))
		if a.NPlayerSlotKnown[slot] != wantKnown {
			t.Fatalf("seat %d slot %d: known = %v, want %v", seat, c, a.NPlayerSlotKnown[slot], wantKnown)
		}
		if !wantKnown {
			continue
		}
		if got, w := a.NPlayerSlotBuckets[slot], CardToBucket(g.Players[seat].Hand[c]); got != w {
			t.Fatalf("seat %d slot %d: bucket %d, want %d (the card the engine holds there)",
				seat, c, got, w)
		}
	}
	for c := len(known); c < engine.MaxHandSize; c++ {
		if a.NPlayerSlotKnown[nplayerSlot(seat, uint8(c))] {
			t.Fatalf("seat %d slot %d: known past the hand's end", seat, c)
		}
	}
}

// TestNPlayerSnapOwnShiftsTheSlotsAbove snaps out of the middle of a hand and reads back
// every remaining slot: each one must name the card the engine now holds there.
func TestNPlayerSnapOwnShiftsTheSlotsAbove(t *testing.T) {
	g := engine.NewGame(20261550, handShiftRules(4, false))
	g.Deal()
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitSpades, engine.RankSeven),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	a := seatAgent(&g, 0, 4)

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOwn(1)); err != nil {
		t.Fatalf("snap own: %v", err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatal("snap own on a matching rank did not succeed")
	}
	a.UpdateNPlayer(&g)

	wantSlots(t, &a, &g, 0, true, true, true)
}

// TestNPlayerFailedSnapClearsThePenaltySlots fails a snap and reads the slots the penalty
// draw occupies: the cards are dealt face down, so nobody knows them. The slots are seeded
// with a record first, since the guarantee the penalty has to make is that whatever a slot
// past the hand's end held, the card drawn into it is not it.
func TestNPlayerFailedSnapClearsThePenaltySlots(t *testing.T) {
	g := engine.NewGame(20261551, handShiftRules(4, false))
	g.Deal()
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour))
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitHearts, engine.RankKing))
	a := seatAgent(&g, 0, 4)
	a.nplayerSetKnown(nplayerSlot(0, 3), BucketHighKing)
	a.nplayerSetKnown(nplayerSlot(0, 4), BucketAce)

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOwn(0)); err != nil {
		t.Fatalf("snap own: %v", err)
	}
	if g.LastAction.SnapSuccess {
		t.Fatal("snap own on a mismatched rank succeeded")
	}
	if g.Players[0].HandLen != 5 {
		t.Fatalf("hand length after the penalty = %d, want 5", g.Players[0].HandLen)
	}
	a.UpdateNPlayer(&g)

	wantSlots(t, &a, &g, 0, true, true, true, false, false)
}

// TestNPlayerSnapFillShiftsBothHands snaps an opponent's card and pays the fill, then reads
// both hands: the mover's shifts left over the card it paid, the target's shifts right
// around the slot the card landed in, and the moved card keeps what was known about it.
func TestNPlayerSnapFillShiftsBothHands(t *testing.T) {
	g := engine.NewGame(20261552, handShiftRules(4, false))
	g.Deal()
	seedHand(&g, 0,
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitSpades, engine.RankEight),
		engine.NewCard(engine.SuitClubs, engine.RankThree))
	seedHand(&g, 2,
		engine.NewCard(engine.SuitClubs, engine.RankNine),
		engine.NewCard(engine.SuitDiamonds, engine.RankSeven),
		engine.NewCard(engine.SuitHearts, engine.RankFive))
	openSnapWindow(&g, 0, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	a := seatAgent(&g, 0, 4)
	paid := CardToBucket(g.Players[0].Hand[1])

	// Seat 2 is seat 0's second opponent (relative index 1).
	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOpponent(1, 1)); err != nil {
		t.Fatalf("snap opponent: %v", err)
	}
	if !g.LastAction.SnapSuccess {
		t.Fatal("snap opponent on a matching rank did not succeed")
	}
	a.UpdateNPlayer(&g)
	wantSlots(t, &a, &g, 2, true, true)

	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeSnapOpponentMove(1)); err != nil {
		t.Fatalf("snap fill: %v", err)
	}
	a.UpdateNPlayer(&g)

	wantSlots(t, &a, &g, 0, true, true)
	wantSlots(t, &a, &g, 2, true, true, true)
	if got := a.NPlayerSlotBuckets[nplayerSlot(2, 1)]; got != paid {
		t.Errorf("the moved card landed as bucket %d, want %d (what the mover knew it to be)", got, paid)
	}
}

// TestNPlayerBeliefMatchesEngineHandsUnderFuzz replays random four-seat games under both
// snap models and reads every seat's belief after every action: a slot the agent claims to
// know must name the card the engine holds there, no slot past a hand's end may be claimed
// at all, and the tracked hand lengths must match the engine's.
func TestNPlayerBeliefMatchesEngineHandsUnderFuzz(t *testing.T) {
	const seats = 4
	for _, race := range []bool{false, true} {
		for seed := uint64(1); seed <= 60; seed++ {
			g := engine.NewGame(seed, handShiftRules(seats, race))
			g.Deal()
			agents := make([]AgentState, seats)
			for i := range agents {
				agents[i] = NewNPlayerAgentState(uint8(i), seats, 1, 5)
				agents[i].InitializeNPlayer(&g)
			}
			rng := rand.New(rand.NewPCG(seed, 0x1550))

			for step := 0; step < 4000 && !g.IsTerminal(); step++ {
				actions := g.NPlayerLegalActionsList()
				if len(actions) == 0 {
					t.Fatalf("race=%v seed=%d step=%d: no legal action at a live state", race, seed, step)
				}
				act := actions[rng.IntN(len(actions))]
				if err := g.ApplyNPlayerAction(act); err != nil {
					t.Fatalf("race=%v seed=%d step=%d: ApplyNPlayerAction(%d): %v", race, seed, step, act, err)
				}
				for i := range agents {
					agents[i].UpdateNPlayer(&g)
					checkBeliefAgreesWithHands(t, &g, &agents[i], seats, race, seed, step, act)
				}
			}
		}
	}
}

// checkBeliefAgreesWithHands is the fuzz invariant: the agent may know less than the truth,
// but never something other than the truth.
func checkBeliefAgreesWithHands(t *testing.T, g *engine.GameState, a *AgentState,
	seats uint8, race bool, seed uint64, step int, act uint16) {
	t.Helper()
	for seat := uint8(0); seat < seats; seat++ {
		handLen := g.Players[seat].HandLen
		if a.NPlayerHandLen[seat] != handLen {
			t.Fatalf("race=%v seed=%d step=%d act=%d: agent %d tracks seat %d at %d cards, engine holds %d",
				race, seed, step, act, a.PlayerID, seat, a.NPlayerHandLen[seat], handLen)
		}
		for c := uint8(0); c < engine.MaxHandSize; c++ {
			slot := nplayerSlot(seat, c)
			if c >= handLen {
				if a.NPlayerSlotKnown[slot] {
					t.Fatalf("race=%v seed=%d step=%d act=%d: agent %d claims seat %d slot %d, which holds no card",
						race, seed, step, act, a.PlayerID, seat, c)
				}
				continue
			}
			if !a.NPlayerSlotKnown[slot] {
				continue
			}
			if got, want := a.NPlayerSlotBuckets[slot], CardToBucket(g.Players[seat].Hand[c]); got != want {
				t.Fatalf("race=%v seed=%d step=%d act=%d: agent %d reads seat %d slot %d as bucket %d, engine holds bucket %d",
					race, seed, step, act, a.PlayerID, seat, c, got, want)
			}
		}
	}
}
