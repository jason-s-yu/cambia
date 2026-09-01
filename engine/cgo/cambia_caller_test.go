package main

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// Tests for cambia_game_cambia_caller (cambia-1488 part 1): the read-only FFI export of
// GameState.CambiaCaller, the only infoset-key input the tabular CFRAgentWrapper needed that the
// FFI did not already expose (own-hand buckets, opponent belief, hand lengths, discard-top bucket
// and stockpile estimate were all reachable through GoEngine and GoAgentState).

func TestCambiaGameCambiaCallerInvalidHandle(t *testing.T) {
	if got := testGameCambiaCaller(-1); got != -1 {
		t.Errorf("cambia_caller on handle -1 = %d, want -1", got)
	}
	if got := testGameCambiaCaller(99999); got != -1 {
		t.Errorf("cambia_caller on an out-of-range handle = %d, want -1", got)
	}
}

func TestCambiaGameCambiaCallerNoneUntilCalled(t *testing.T) {
	h := testGameNewWithRules(3, 2)
	if h < 0 {
		t.Fatalf("testGameNewWithRules = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	if got := testGameCambiaCaller(h); got != -1 {
		t.Errorf("cambia_caller on a freshly dealt game = %d, want -1 (no one has called)", got)
	}
}

func TestCambiaGameCambiaCallerReportsTheCallingSeat(t *testing.T) {
	// testGameNewWithRules sets cambiaAllowedRound=0, so ActionCallCambia is legal
	// immediately for the starting seat.
	h := testGameNewWithRules(3, 2)
	if h < 0 {
		t.Fatalf("testGameNewWithRules = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	acting := testGameActingPlayer(h)
	if rc := testGameApplyAction(h, engine.ActionCallCambia); rc < 0 {
		t.Fatalf("ActionCallCambia by seat %d failed: %d", acting, rc)
	}
	if got := testGameCambiaCaller(h); got != int32(acting) {
		t.Errorf("cambia_caller = %d, want %d (the seat that called it)", got, acting)
	}
}

func TestCambiaGameCambiaCallerAtFourSeatsViaNPlayerSpace(t *testing.T) {
	h := testGameNewWithRules(5, 4)
	if h < 0 {
		t.Fatalf("testGameNewWithRules = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	acting := testGameActingPlayer(h)
	if rc := testGameApplyNPlayerAction(h, engine.NPlayerActionCallCambia); rc < 0 {
		t.Fatalf("NPlayerActionCallCambia by seat %d failed: %d", acting, rc)
	}
	if got := testGameCambiaCaller(h); got != int32(acting) {
		t.Errorf("cambia_caller = %d, want %d (the seat that called it)", got, acting)
	}
}
