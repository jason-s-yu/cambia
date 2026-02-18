//go:build agentfuzz

package agent

import (
	"fmt"
	"math/bits"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

const numFuzzGames = 100
const maxStepsPerGame = 10000

// pickAction selects a random legal action from the bitmask using a deterministic
// xorshift64 RNG. Returns 0 if no legal actions exist.
func pickAction(mask [3]uint64, rngState *uint64) uint16 {
	// Collect all legal action indices.
	var actions [146]uint16
	n := 0
	for word := 0; word < 3; word++ {
		w := mask[word]
		for w != 0 {
			bit := w & (-w)
			actions[n] = uint16(word*64 + bits.TrailingZeros64(bit))
			n++
			w &^= bit
		}
	}
	if n == 0 {
		return 0
	}
	// Advance xorshift RNG.
	x := *rngState
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*rngState = x
	return actions[x%uint64(n)]
}

// checkInvariants verifies all agent state invariants against the true game state.
// Returns the number of failures found.
func checkInvariants(t *testing.T, g *engine.GameState, a *AgentState, seed, step int, phase string) int {
	t.Helper()
	failures := 0
	label := fmt.Sprintf("seed=%d step=%d P%d", seed, step, a.PlayerID)
	if phase != "" {
		label += " (" + phase + ")"
	}

	// INV1: Hand Length Match
	if a.OwnHandLen != g.Players[a.PlayerID].HandLen {
		t.Errorf("%s INV1: OwnHandLen=%d, actual=%d",
			label, a.OwnHandLen, g.Players[a.PlayerID].HandLen)
		failures++
	}
	if a.OppHandLen != g.Players[a.OpponentID].HandLen {
		t.Errorf("%s INV1: OppHandLen=%d, actual=%d",
			label, a.OppHandLen, g.Players[a.OpponentID].HandLen)
		failures++
	}

	// INV2: Known Card Consistency — if agent claims to know a card, it must match actual.
	for i := uint8(0); i < a.OwnHandLen; i++ {
		if a.OwnHand[i].Card != engine.EmptyCard {
			actual := g.Players[a.PlayerID].Hand[i]
			if a.OwnHand[i].Card != actual {
				t.Errorf("%s INV2: OwnHand[%d].Card=%v, actual=%v",
					label, i, a.OwnHand[i].Card, actual)
				failures++
			}
			// INV3: Bucket self-consistency — bucket must match CardToBucket(card).
			if a.OwnHand[i].Bucket != CardToBucket(actual) {
				t.Errorf("%s INV3: OwnHand[%d].Bucket=%d, expected=%d (card=%v)",
					label, i, a.OwnHand[i].Bucket, CardToBucket(actual), actual)
				failures++
			}
		}
	}

	// INV4: Discard Top Bucket
	discardTop := g.DiscardTop()
	var expectedBucket CardBucket
	if discardTop == engine.EmptyCard {
		expectedBucket = BucketUnknown
	} else {
		expectedBucket = CardToBucket(discardTop)
	}
	if a.DiscardTopBucket != expectedBucket {
		t.Errorf("%s INV4: DiscardTopBucket=%d, expected=%d",
			label, a.DiscardTopBucket, expectedBucket)
		failures++
	}

	// INV5: Stock Estimate
	expectedStock := StockEstimateFromSize(g.StockLen)
	if a.StockEstimate != expectedStock {
		t.Errorf("%s INV5: StockEstimate=%d, expected=%d",
			label, a.StockEstimate, expectedStock)
		failures++
	}

	// INV6: Phase
	expectedPhase := GamePhaseFromState(g.StockLen, g.IsCambiaCalled(), g.IsTerminal())
	if a.Phase != expectedPhase {
		t.Errorf("%s INV6: Phase=%d, expected=%d",
			label, a.Phase, expectedPhase)
		failures++
	}

	// INV7: Cambia State
	if g.IsCambiaCalled() {
		if g.CambiaCaller == int8(a.PlayerID) && a.CambiaState != CambiaSelf {
			t.Errorf("%s INV7: expected CambiaSelf, got %d", label, a.CambiaState)
			failures++
		}
		if g.CambiaCaller != int8(a.PlayerID) && a.CambiaState != CambiaOpponent {
			t.Errorf("%s INV7: expected CambiaOpponent, got %d", label, a.CambiaState)
			failures++
		}
	} else if a.CambiaState != CambiaNone {
		t.Errorf("%s INV7: expected CambiaNone, got %d", label, a.CambiaState)
		failures++
	}

	// INV8: InfosetKey Determinism
	key1 := a.InfosetKey()
	key2 := a.InfosetKey()
	if key1 != key2 {
		t.Errorf("%s INV8: InfosetKey not deterministic: %v vs %v", label, key1, key2)
		failures++
	}

	// INV10: Opponent Belief Range — each OppBelief value must be in [0,13].
	for i := uint8(0); i < a.OppHandLen; i++ {
		v := uint8(a.OppBelief[i])
		if v > 13 {
			t.Errorf("%s INV10: OppBelief[%d]=%d out of range [0,13]", label, i, v)
			failures++
		}
	}

	return failures
}

// checkClone verifies that a clone is independent of the original.
// Returns the number of failures found.
func checkClone(t *testing.T, a *AgentState, seed, step int) int {
	t.Helper()
	failures := 0
	label := fmt.Sprintf("seed=%d step=%d P%d", seed, step, a.PlayerID)

	// INV9: Clone Independence
	origKey := a.InfosetKey()
	origBucket := a.OwnHand[0].Bucket

	clone := a.Clone()

	// Clone should have the same infoset key.
	if clone.InfosetKey() != origKey {
		t.Errorf("%s INV9: clone InfosetKey differs from original", label)
		failures++
	}

	// Mutate the clone and verify the original is unchanged.
	clone.OwnHand[0].Bucket = BucketHighKing
	if a.OwnHand[0].Bucket != origBucket {
		t.Errorf("%s INV9: clone mutation affected original OwnHand[0].Bucket", label)
		failures++
	}

	return failures
}

// TestAgentStateFuzz plays 100 random games and checks all agent invariants at every step.
func TestAgentStateFuzz(t *testing.T) {
	totalSteps := 0
	totalFailures := 0

	for seed := 0; seed < numFuzzGames; seed++ {
		g := engine.NewGame(uint64(seed+1), engine.DefaultHouseRules()) // seed+1 to avoid 0
		g.Deal()

		a0 := NewAgentState(0, 1, 0, 0) // memory level 0 (perfect memory)
		a1 := NewAgentState(1, 0, 0, 0)
		a0.Initialize(&g)
		a1.Initialize(&g)

		rngState := uint64(seed*1000 + 7)

		// Verify invariants after initialize.
		totalFailures += checkInvariants(t, &g, &a0, seed, 0, "init")
		totalFailures += checkInvariants(t, &g, &a1, seed, 0, "init")

		for step := 0; step < maxStepsPerGame; step++ {
			if g.IsTerminal() {
				break
			}

			mask := g.LegalActions()
			action := pickAction(mask, &rngState)

			if err := g.ApplyAction(action); err != nil {
				t.Fatalf("seed=%d step=%d: ApplyAction(%d) error: %v", seed, step, action, err)
			}

			a0.Update(&g)
			a1.Update(&g)

			totalFailures += checkInvariants(t, &g, &a0, seed, step+1, "")
			totalFailures += checkInvariants(t, &g, &a1, seed, step+1, "")

			// Clone check every 10 steps.
			if step%10 == 0 {
				totalFailures += checkClone(t, &a0, seed, step)
				totalFailures += checkClone(t, &a1, seed, step)
			}

			totalSteps++

			if totalFailures > 50 {
				t.Fatalf("too many failures (%d), aborting at seed=%d step=%d", totalFailures, seed, step)
			}
		}

		// Final invariant check at terminal.
		totalFailures += checkInvariants(t, &g, &a0, seed, -1, "terminal")
		totalFailures += checkInvariants(t, &g, &a1, seed, -1, "terminal")
	}

	t.Logf("Fuzz complete: %d games, %d total steps, %d failures", numFuzzGames, totalSteps, totalFailures)
}

// TestAgentStateFuzzMemoryLevels runs fewer games at memory levels 1 and 2
// to verify that decay logic doesn't corrupt the invariants.
func TestAgentStateFuzzMemoryLevels(t *testing.T) {
	for _, memLevel := range []uint8{1, 2} {
		memLevel := memLevel // capture for closure
		t.Run(fmt.Sprintf("memory%d", memLevel), func(t *testing.T) {
			for seed := 0; seed < 20; seed++ {
				g := engine.NewGame(uint64(seed+1), engine.DefaultHouseRules())
				g.Deal()

				a0 := NewAgentState(0, 1, memLevel, 3) // time decay at 3 turns
				a1 := NewAgentState(1, 0, memLevel, 3)
				a0.Initialize(&g)
				a1.Initialize(&g)

				rngState := uint64(seed*1000 + 7)

				for step := 0; step < maxStepsPerGame; step++ {
					if g.IsTerminal() {
						break
					}

					mask := g.LegalActions()
					action := pickAction(mask, &rngState)

					if err := g.ApplyAction(action); err != nil {
						t.Fatalf("seed=%d step=%d: ApplyAction(%d) error: %v", seed, step, action, err)
					}

					a0.Update(&g)
					a1.Update(&g)

					checkInvariants(t, &g, &a0, seed, step+1, "")
					checkInvariants(t, &g, &a1, seed, step+1, "")
				}
			}
		})
	}
}
