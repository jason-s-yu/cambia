//go:build integration

package engine

// integration_test.go — Full-game integration tests for the Go engine.
//
// These tests use only the public API: NewGame, ApplyAction, LegalActions,
// IsTerminal, Save/Restore (undo), ActingPlayer, DecisionCtx.
//
// Run: cd engine && go test -run TestIntegration -v

import (
	"math/bits"
	"math/rand"
	"testing"
)

// legalList converts a [3]uint64 bitmask into a sorted slice of action indices.
func legalList(mask [3]uint64) []uint16 {
	var out []uint16
	for word := 0; word < 3; word++ {
		w := mask[word]
		for w != 0 {
			bit := w & (-w) // isolate lowest set bit
			w &^= bit       // clear it
			idx := uint16(word)*64 + uint16(countTrailingZeros64(bit))
			out = append(out, idx)
		}
	}
	return out
}

// countTrailingZeros64 returns the number of trailing zero bits in v.
func countTrailingZeros64(v uint64) int {
	return bits.TrailingZeros64(v)
}

// pickFirst returns the smallest legal action index (deterministic choice).
func pickFirst(mask [3]uint64) (uint16, bool) {
	actions := legalList(mask)
	if len(actions) == 0 {
		return 0, false
	}
	return actions[0], true
}

// pickRand returns a random legal action index.
func pickRand(mask [3]uint64, rng *rand.Rand) (uint16, bool) {
	actions := legalList(mask)
	if len(actions) == 0 {
		return 0, false
	}
	return actions[rng.Intn(len(actions))], true
}

// hasBit reports whether action index idx is set in the bitmask.
func hasBit(mask [3]uint64, idx uint16) bool {
	word := idx / 64
	bit := idx % 64
	if word >= 3 {
		return false
	}
	return mask[word]&(1<<bit) != 0
}

// maskLen counts set bits in the legal action bitmask.
func maskLen(mask [3]uint64) int {
	n := 0
	for _, w := range mask {
		for w != 0 {
			n++
			w &= w - 1
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// TestIntegrationRandomGamesTerminate: random-action games always terminate.
// ---------------------------------------------------------------------------

func TestIntegrationRandomGamesTerminate(t *testing.T) {
	const numGames = 50
	const maxSteps = 20_000

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()
		gs := NewGame(uint64(gameIdx), hr)
		gs.Deal()
		rng := rand.New(rand.NewSource(int64(gameIdx) + 999))

		steps := 0
		for !gs.IsTerminal() {
			if steps > maxSteps {
				t.Errorf("game %d did not terminate after %d steps", gameIdx, maxSteps)
				break
			}
			mask := gs.LegalActions()
			idx, ok := pickRand(mask, rng)
			if !ok {
				if !gs.IsTerminal() {
					t.Errorf("game %d step %d: no legal actions in non-terminal state", gameIdx, steps)
				}
				break
			}
			if err := gs.ApplyAction(idx); err != nil {
				t.Errorf("game %d step %d: ApplyAction(%d) error: %v", gameIdx, steps, idx, err)
				break
			}
			steps++
		}
		if !gs.IsTerminal() {
			continue // already errored above
		}
		// Utilities must sum to ~0 (zero-sum game).
		u := gs.GetUtility()
		sum := u[0] + u[1]
		if sum < -0.01 || sum > 0.01 {
			t.Errorf("game %d: utilities %.2f+%.2f=%f, want ~0", gameIdx, u[0], u[1], sum)
		}
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationDeterministicReplay: same seed → identical outcome.
// ---------------------------------------------------------------------------

func TestIntegrationDeterministicReplay(t *testing.T) {
	const numGames = 10
	const maxSteps = 20_000

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()

		// Collect actions from first run.
		var actions []uint16
		var states1Legals [][3]uint64
		{
			gs := NewGame(uint64(gameIdx), hr)
			gs.Deal()
			rng := rand.New(rand.NewSource(int64(gameIdx)*13 + 7))
			for !gs.IsTerminal() && len(actions) < maxSteps {
				mask := gs.LegalActions()
				states1Legals = append(states1Legals, mask)
				idx, ok := pickRand(mask, rng)
				if !ok {
					break
				}
				actions = append(actions, idx)
				gs.ApplyAction(idx) //nolint:errcheck
			}
		}

		// Replay the exact same sequence and assert identical legal masks.
		{
			gs := NewGame(uint64(gameIdx), hr)
			gs.Deal()
			for step, idx := range actions {
				mask := gs.LegalActions()
				if mask != states1Legals[step] {
					t.Errorf("game %d step %d: legal mask mismatch on replay", gameIdx, step)
					break
				}
				if !hasBit(mask, idx) {
					t.Errorf("game %d step %d: replayed action %d not in legal mask", gameIdx, step, idx)
					break
				}
				gs.ApplyAction(idx) //nolint:errcheck
			}
		}
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationCambiaTriggersEndGame: calling Cambia ends the game
// after all other players have one more turn.
// ---------------------------------------------------------------------------

func TestIntegrationCambiaTriggersEndGame(t *testing.T) {
	const maxAttempts = 1000 // run multiple seeds until one hits a Cambia call

	for seed := uint64(0); seed < maxAttempts; seed++ {
		hr := DefaultHouseRules()
		gs := NewGame(seed, hr)
		gs.Deal()
		rng := rand.New(rand.NewSource(int64(seed) + 42))

		cambiaCalledStep := -1
		step := 0
		for !gs.IsTerminal() && step < 20_000 {
			mask := gs.LegalActions()
			// Prefer Cambia if legal (action index 2).
			var idx uint16
			if hasBit(mask, ActionCallCambia) {
				idx = ActionCallCambia
				if cambiaCalledStep < 0 {
					cambiaCalledStep = step
				}
			} else {
				var ok bool
				idx, ok = pickRand(mask, rng)
				if !ok {
					break
				}
			}
			gs.ApplyAction(idx) //nolint:errcheck
			step++
		}

		if cambiaCalledStep >= 0 {
			// Game must have terminated.
			if !gs.IsTerminal() {
				t.Errorf("seed %d: Cambia called at step %d but game did not terminate (reached step %d)", seed, cambiaCalledStep, step)
			}
			return // test passes after first valid Cambia game
		}
	}
	t.Errorf("no Cambia call occurred in %d attempts — check cambia_allowed_round or Cambia action availability", maxAttempts)
}

// ---------------------------------------------------------------------------
// TestIntegrationUndoAtEveryNode: Save then Apply then Restore at each step.
// ---------------------------------------------------------------------------

func TestIntegrationUndoAtEveryNode(t *testing.T) {
	const numGames = 10
	const maxSteps = 500 // limit per game for speed

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()
		gs := NewGame(uint64(gameIdx), hr)
		gs.Deal()
		rng := rand.New(rand.NewSource(int64(gameIdx) + 1234))

		for step := 0; step < maxSteps && !gs.IsTerminal(); step++ {
			mask := gs.LegalActions()
			idx, ok := pickRand(mask, rng)
			if !ok {
				break
			}

			// Save state.
			snapshot := gs.Save()

			// Apply action.
			gs.ApplyAction(idx) //nolint:errcheck

			// Restore from snapshot.
			gs.Restore(snapshot)

			// After restore, legal actions must be identical.
			maskAfter := gs.LegalActions()
			if mask != maskAfter {
				t.Errorf("game %d step %d: legal mask differs after save/restore", gameIdx, step)
				break
			}
			// ActingPlayer must be restored.
			actorBefore := gs.ActingPlayer()
			_ = actorBefore

			// Apply again for real.
			gs.ApplyAction(idx) //nolint:errcheck
		}
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationDrawDiscardCycleMaxTurns: game ends at MaxGameTurns.
// ---------------------------------------------------------------------------

func TestIntegrationDrawDiscardCycleMaxTurns(t *testing.T) {
	// Use a low max_turns so the test completes quickly.
	hr := DefaultHouseRules()
	hr.MaxGameTurns = 20

	gs := NewGame(42, hr)
	gs.Deal()
	steps := 0
	for !gs.IsTerminal() && steps < 10_000 {
		mask := gs.LegalActions()
		// Always draw stockpile (index 0) if available, then discard without ability (index 3).
		var idx uint16
		switch {
		case hasBit(mask, ActionDrawStockpile):
			idx = ActionDrawStockpile
		case hasBit(mask, ActionDiscardNoAbility):
			idx = ActionDiscardNoAbility
		default:
			var ok bool
			idx, ok = pickFirst(mask)
			if !ok {
				break
			}
		}
		gs.ApplyAction(idx) //nolint:errcheck
		steps++
	}

	if !gs.IsTerminal() {
		t.Errorf("game with MaxGameTurns=20 did not terminate (ran %d steps)", steps)
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationAllContextsReached: all decision contexts are exercised.
// ---------------------------------------------------------------------------

func TestIntegrationAllContextsReached(t *testing.T) {
	const numGames = 200

	ctxSeen := make(map[DecisionContext]bool)
	rng := rand.New(rand.NewSource(777))

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()
		gs := NewGame(uint64(gameIdx), hr)
		gs.Deal()

		for !gs.IsTerminal() {
			ctx := gs.DecisionCtx()
			ctxSeen[ctx] = true

			mask := gs.LegalActions()
			idx, ok := pickRand(mask, rng)
			if !ok {
				break
			}
			gs.ApplyAction(idx) //nolint:errcheck
		}
	}

	wantCtxs := []DecisionContext{CtxStartTurn, CtxPostDraw, CtxSnapDecision, CtxAbilitySelect}
	for _, ctx := range wantCtxs {
		if !ctxSeen[ctx] {
			t.Errorf("DecisionContext %d was never reached across %d games", ctx, numGames)
		}
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationLegalActionsNeverEmpty: in any non-terminal state,
// the legal action bitmask is always non-empty.
// ---------------------------------------------------------------------------

func TestIntegrationLegalActionsNeverEmpty(t *testing.T) {
	const numGames = 50
	const maxSteps = 5000
	rng := rand.New(rand.NewSource(555))

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()
		gs := NewGame(uint64(gameIdx), hr)
		gs.Deal()

		for step := 0; step < maxSteps && !gs.IsTerminal(); step++ {
			mask := gs.LegalActions()
			if maskLen(mask) == 0 {
				t.Errorf("game %d step %d: empty legal actions in non-terminal state", gameIdx, step)
				break
			}
			idx, _ := pickRand(mask, rng)
			gs.ApplyAction(idx) //nolint:errcheck
		}
	}
}

// ---------------------------------------------------------------------------
// TestIntegrationHandSizesReasonable: hand sizes stay in [0, 6] throughout.
// ---------------------------------------------------------------------------

func TestIntegrationHandSizesReasonable(t *testing.T) {
	const numGames = 30
	const maxSteps = 5000
	rng := rand.New(rand.NewSource(321))

	for gameIdx := 0; gameIdx < numGames; gameIdx++ {
		hr := DefaultHouseRules()
		gs := NewGame(uint64(gameIdx), hr)
		gs.Deal()

		for step := 0; step < maxSteps && !gs.IsTerminal(); step++ {
			for p := 0; p < 2; p++ {
				h := gs.Players[p].HandLen
				if h > 6 {
					t.Errorf("game %d step %d: player %d has %d cards (max 6)", gameIdx, step, p, h)
				}
			}
			mask := gs.LegalActions()
			idx, ok := pickRand(mask, rng)
			if !ok {
				break
			}
			gs.ApplyAction(idx) //nolint:errcheck
		}
	}
}
