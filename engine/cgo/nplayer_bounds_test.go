package main

import "testing"

// cambia-542 F3: prior to HouseRules.Validate(), NumPlayers had no upper
// bound anywhere. cambia_game_new_with_rules only clamped the lower bound
// (np < 2 -> 2); numPlayers=9 sailed through, and Deal()'s round-robin loop
// then indexed g.Players[8] into a fixed [8]PlayerState array -- an
// unrecoverable panic inside libcambia.so, not a Go error the Python side
// could catch. These tests assert the FFI boundary now rejects it (-1)
// instead; they must never reach the panic path itself.

func TestCambiaGameNewWithRulesRejectsOverMaxPlayers(t *testing.T) {
	h := testGameNewWithRules(1, 9)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_rules(numPlayers=9) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithRulesAcceptsMaxPlayers(t *testing.T) {
	h := testGameNewWithRules(1, 8)
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(numPlayers=8) = %d, want a valid handle", h)
	}
	testGameFree(h)
}

func TestCambiaGameNewWithRulesRejectsOnePlayer(t *testing.T) {
	h := testGameNewWithRules(1, 1)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_rules(numPlayers=1) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithRulesDefaultsZeroToTwo(t *testing.T) {
	h := testGameNewWithRules(1, 0)
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(numPlayers=0) = %d, want a valid handle (0 defaults to 2)", h)
	}
	testGameFree(h)
}

func TestCambiaGameNewWithRulesAcceptsTwoPlayers(t *testing.T) {
	h := testGameNewWithRules(1, 2)
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(numPlayers=2) = %d, want a valid handle", h)
	}
	testGameFree(h)
}

// TestCambiaNPlayerDimsMatchConstants pins the live Go dim exports against
// their known-correct values (agent.NPlayerInputDim=856,
// engine.NPlayerNumActions=620 per engine/agent/constants.go and
// engine/types.go). The Python-side cross-check test
// (test_nplayer_encoding.py::TestNPlayerDimCrossCheck) asserts these same
// exports equal cfr/src/constants.py's N_PLAYER_INPUT_DIM/N_PLAYER_NUM_ACTIONS;
// this test guards the Go side of that contract independently.
func TestCambiaNPlayerDimsMatchConstants(t *testing.T) {
	if got := testNPlayerInputDim(); got != 856 {
		t.Errorf("cambia_nplayer_input_dim() = %d, want 856 (agent.NPlayerInputDim)", got)
	}
	if got := testNPlayerNumActions(); got != 620 {
		t.Errorf("cambia_nplayer_num_actions() = %d, want 620 (engine.NPlayerNumActions)", got)
	}
}
