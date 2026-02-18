//go:build encodingfuzz

package agent

import (
	"fmt"
	"math"
	"math/bits"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// pickActionEncFuzz selects a random legal action from the bitmask using a
// deterministic xorshift64 RNG. Returns 0 if no legal actions exist.
// (Copied from fuzz_test.go since build tags differ.)
func pickActionEncFuzz(mask [3]uint64, rngState *uint64) uint16 {
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
	x := *rngState
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*rngState = x
	return actions[x%uint64(n)]
}

// decisionContextFromGame derives the DecisionContext from the current game state.
func decisionContextFromGame(g *engine.GameState) engine.DecisionContext {
	if g.IsTerminal() {
		return engine.CtxTerminal
	}
	if g.Snap.Active {
		if g.Pending.Type == engine.PendingSnapMove {
			return engine.CtxSnapMove
		}
		return engine.CtxSnapDecision
	}
	switch g.Pending.Type {
	case engine.PendingDiscard:
		return engine.CtxPostDraw
	case engine.PendingPeekOwn, engine.PendingPeekOther, engine.PendingBlindSwap,
		engine.PendingKingLook, engine.PendingKingDecision:
		return engine.CtxAbilitySelect
	case engine.PendingSnapMove:
		return engine.CtxSnapMove
	default:
		return engine.CtxStartTurn
	}
}

// checkOneHotGroup verifies that exactly one value in out[offset:offset+size] is 1.0
// and all others are 0.0. Returns the number of invariant violations found.
func checkOneHotGroup(t *testing.T, out []float32, offset, size int, prefix, label string) int {
	t.Helper()
	ones := 0
	for j := 0; j < size; j++ {
		v := out[offset+j]
		if v == 1.0 {
			ones++
		} else if v != 0.0 {
			t.Errorf("%s %s: pos %d = %v (not 0 or 1)", prefix, label, j, v)
			return 1
		}
	}
	if ones != 1 {
		t.Errorf("%s %s: %d ones, want 1", prefix, label, ones)
		return 1
	}
	return 0
}

// checkActionMask verifies INV-E13: the number of true entries in the ActionMask
// output matches the popcount of the legal actions bitmask.
func checkActionMask(t *testing.T, mask [3]uint64, seed, step int) int {
	t.Helper()
	var out [NumActions]bool
	ActionMask(mask, &out)

	trueCount := 0
	for _, v := range out {
		if v {
			trueCount++
		}
	}

	// Count set bits across all 146 action indices.
	bitCount := 0
	for word := 0; word < 3; word++ {
		w := mask[word]
		if word == 2 {
			// Only bits 0-17 are valid (146 - 128 = 18 bits in the third word).
			w &= (1 << 18) - 1
		}
		bitCount += bits.OnesCount64(w)
	}

	if trueCount != bitCount {
		t.Errorf("seed=%d step=%d INV-E13: ActionMask trueCount=%d, bitCount=%d",
			seed, step, trueCount, bitCount)
		return 1
	}
	return 0
}

// checkEncoding runs all encoding invariants on a single agent state snapshot.
// Returns the number of invariant violations found.
func checkEncoding(t *testing.T, a *AgentState, ctx engine.DecisionContext, drawnBucket int8, seed, step int, label string) int {
	t.Helper()
	failures := 0
	prefix := fmt.Sprintf("seed=%d step=%d %s", seed, step, label)

	var out [InputDim]float32
	a.Encode(ctx, drawnBucket, &out)

	// INV-E11: No NaN or Inf.
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("%s INV-E11: out[%d] = %v (NaN/Inf)", prefix, i, v)
			failures++
		}
	}

	// INV-E12: Value range [0, 1].
	for i, v := range out {
		if v < 0 || v > 1.0 {
			t.Errorf("%s INV-E12: out[%d] = %v out of [0,1]", prefix, i, v)
			failures++
		}
	}

	// INV-E2/E3: One-hot for each of the 12 hand slots (6 own + 6 opp), each 15-dim.
	for slot := 0; slot < 12; slot++ {
		base := slot * SlotDim
		ones := 0
		for j := 0; j < SlotDim; j++ {
			v := out[base+j]
			if v == 1.0 {
				ones++
			} else if v != 0.0 {
				t.Errorf("%s INV-E2/3: slot %d pos %d = %v (not 0 or 1)", prefix, slot, j, v)
				failures++
			}
		}
		if ones != 1 {
			t.Errorf("%s INV-E2/3: slot %d has %d ones, want 1", prefix, slot, ones)
			failures++
		}
	}

	// INV-E5: Drawn card one-hot (11 values at offset 182).
	failures += checkOneHotGroup(t, out[:], 182, 11, prefix, "INV-E5-drawn")

	// INV-E6: Discard top one-hot (10 values at offset 193).
	failures += checkOneHotGroup(t, out[:], 193, 10, prefix, "INV-E6-discard")

	// INV-E7: Stockpile estimate one-hot (4 values at offset 203).
	failures += checkOneHotGroup(t, out[:], 203, 4, prefix, "INV-E7-stock")

	// INV-E8: Game phase one-hot (6 values at offset 207).
	failures += checkOneHotGroup(t, out[:], 207, 6, prefix, "INV-E8-phase")

	// INV-E9: Decision context one-hot (6 values at offset 213).
	failures += checkOneHotGroup(t, out[:], 213, 6, prefix, "INV-E9-ctx")

	// INV-E10: Cambia state one-hot (3 values at offset 219).
	failures += checkOneHotGroup(t, out[:], 219, 3, prefix, "INV-E10-cambia")

	// INV-E14: Encoding is deterministic — encode again and compare.
	var out2 [InputDim]float32
	a.Encode(ctx, drawnBucket, &out2)
	if out != out2 {
		t.Errorf("%s INV-E14: encoding not deterministic", prefix)
		failures++
	}

	return failures
}

// TestEncodingFuzz plays 100 random games and checks all encoding invariants at every step.
func TestEncodingFuzz(t *testing.T) {
	const numGames = 100
	const maxSteps = 10000
	totalSteps := 0
	totalFailures := 0

	for seed := 0; seed < numGames; seed++ {
		g := engine.NewGame(uint64(seed+1), engine.DefaultHouseRules())
		g.Deal()

		a0 := NewAgentState(0, 1, 0, 0)
		a1 := NewAgentState(1, 0, 0, 0)
		a0.Initialize(&g)
		a1.Initialize(&g)

		rngState := uint64(seed*1000 + 7)

		// Check encoding after initialization.
		ctx := decisionContextFromGame(&g)
		totalFailures += checkEncoding(t, &a0, ctx, -1, seed, 0, "init-P0")
		totalFailures += checkEncoding(t, &a1, ctx, -1, seed, 0, "init-P1")

		for step := 0; step < maxSteps; step++ {
			if g.IsTerminal() {
				break
			}

			mask := g.LegalActions()

			// INV-E13: ActionMask consistency.
			totalFailures += checkActionMask(t, mask, seed, step)

			action := pickActionEncFuzz(mask, &rngState)

			if err := g.ApplyAction(action); err != nil {
				t.Fatalf("seed=%d step=%d: ApplyAction(%d) error: %v", seed, step, action, err)
			}

			a0.Update(&g)
			a1.Update(&g)

			ctx = decisionContextFromGame(&g)
			totalFailures += checkEncoding(t, &a0, ctx, -1, seed, step+1, "P0")
			totalFailures += checkEncoding(t, &a1, ctx, -1, seed, step+1, "P1")

			totalSteps++

			if totalFailures > 50 {
				t.Fatalf("too many failures (%d), aborting at seed=%d step=%d",
					totalFailures, seed, step)
			}
		}
	}

	t.Logf("Encoding fuzz complete: %d games, %d total steps, %d failures",
		numGames, totalSteps, totalFailures)
}
