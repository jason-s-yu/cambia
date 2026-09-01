package main

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// Tests for cambia_game_resolve_untargetable_armed_ability (cambia-1489): the FFI export of
// engine.GameState.ResolveUntargetableArmedAbility, which was previously unreachable from FFI
// users (eval, PPO env, best-response search) despite being the only recovery for an armed
// ability no action in the caller's action space can resolve.

func TestCambiaGameResolveUntargetableArmedAbilityInvalidHandle(t *testing.T) {
	if got := testGameResolveUntargetableArmedAbility(-1, false); got != -1 {
		t.Errorf("resolve on handle -1 = %d, want -1", got)
	}
	if got := testGameResolveUntargetableArmedAbility(99999, true); got != -1 {
		t.Errorf("resolve on an out-of-range handle = %d, want -1", got)
	}
}

func TestCambiaGameResolveUntargetableArmedAbilityNoopWhenNothingArmed(t *testing.T) {
	h := testGameNewWithRules(1, 4)
	if h < 0 {
		t.Fatalf("testGameNewWithRules = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	turnBefore := testGameTurnNumber(h)
	for _, nPlayerSpace := range []bool{false, true} {
		if got := testGameResolveUntargetableArmedAbility(h, nPlayerSpace); got != 0 {
			t.Errorf("resolve(nPlayerSpace=%v) on a freshly dealt game = %d, want 0 (nothing armed)", nPlayerSpace, got)
		}
	}
	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields {
		t.Fatalf("get_pending = %d, want %d", rc, evalPendingFields)
	}
	if rec[0] != uint8(0) { // PendingNone
		t.Errorf("Pending.Type = %d after two no-op resolves, want PendingNone (0)", rec[0])
	}
	if got := testGameTurnNumber(h); got != turnBefore {
		t.Errorf("TurnNumber = %d after two no-op resolves, want unchanged %d", got, turnBefore)
	}
}

// TestCambiaGameResolveUntargetableArmedAbilityNoopWhenStillTargetable arms PendingPeekOther
// through a real discard-with-ability at two seats, where the 2-player mask can plainly still
// target the opponent. The export must leave a genuinely resolvable ability alone.
func TestCambiaGameResolveUntargetableArmedAbilityNoopWhenStillTargetable(t *testing.T) {
	const cardHT = 35 // H-T: peek-other ability
	h := newScriptedGame(t, cardHT)

	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	if rc := testGameApplyAction(h, engine.ActionDiscardWithAbility); rc < 0 {
		t.Fatalf("discard with ability failed: %d", rc)
	}
	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields || rec[0] != uint8(3) { // PendingPeekOther
		t.Fatalf("Pending.Type = %d (rc=%d), want PendingPeekOther (3)", rec[0], rc)
	}

	if got := testGameResolveUntargetableArmedAbility(h, false); got != 0 {
		t.Errorf("resolve(nPlayerSpace=false) on a still-targetable ability = %d, want 0 (no-op)", got)
	}
	_, rec = testGameGetPending(h)
	if rec[0] != uint8(3) {
		t.Errorf("Pending.Type after the no-op resolve = %d, want PendingPeekOther (3) unchanged", rec[0])
	}
}

// TestCambiaGameResolveUntargetableArmedAbilityResolvesStrandedState constructs, by direct poke,
// a state no live arm site produces any more since the cambia-1489 fix: an ability armed for
// seat 0 that the 2-player mask cannot target (its only opponent, seat 1, holds nothing) at a
// four seat table, while the N-player mask can still reach seats 2 and 3. This is exactly the
// "caller holding a state built before the fix" case ResolveUntargetableArmedAbility's own
// contract still promises to guard (see abilities.go).
func TestCambiaGameResolveUntargetableArmedAbilityResolvesStrandedState(t *testing.T) {
	const cardC2 = 1  // C-2, used to pin every hand away from the discarded rank
	const cardDT = 22 // D-T, the discarded card (rank Ten)

	newStrandedGame := func(t *testing.T) int32 {
		t.Helper()
		h := testGameNewWithRules(7, 4)
		if h < 0 {
			t.Fatalf("testGameNewWithRules = %d, want a valid handle", h)
		}
		t.Cleanup(func() { testGameFree(h) })
		for seat := uint8(0); seat < 4; seat++ {
			for slot := uint8(0); slot < 4; slot++ {
				testGameSetHandCard(h, seat, slot, cardC2)
			}
		}
		testGameSetHandLen(h, 1, 0) // seat 1: the 2-player space's opponent for seat 0
		testGamePushDiscard(h, cardDT)
		testGameSetPending(h, 3, 0) // PendingPeekOther, acting seat 0
		return h
	}

	t.Run("two_player_space_resolves", func(t *testing.T) {
		h := newStrandedGame(t)
		turnBefore := testGameTurnNumber(h)
		if got := testGameResolveUntargetableArmedAbility(h, false); got != 1 {
			t.Fatalf("resolve(nPlayerSpace=false) on a stranded ability = %d, want 1", got)
		}
		_, rec := testGameGetPending(h)
		if rec[0] != 0 {
			t.Errorf("Pending.Type after resolution = %d, want PendingNone (0)", rec[0])
		}
		if got := testGameTurnNumber(h); got != turnBefore+1 {
			t.Errorf("TurnNumber = %d, want %d: resolution should advance the turn (no hand holds rank Ten)", got, turnBefore+1)
		}
	})

	t.Run("n_player_space_leaves_it_armed", func(t *testing.T) {
		h := newStrandedGame(t)
		if got := testGameResolveUntargetableArmedAbility(h, true); got != 0 {
			t.Errorf("resolve(nPlayerSpace=true) = %d, want 0: seats 2 and 3 still hold cards", got)
		}
		_, rec := testGameGetPending(h)
		if rec[0] != 3 {
			t.Errorf("Pending.Type after the no-op resolve = %d, want PendingPeekOther (3) unchanged", rec[0])
		}
	})
}
