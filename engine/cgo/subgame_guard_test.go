package main

import "testing"

// cambia-1554: BuildSubgameTree and both CFR iteration methods
// (SubgameNode.CFRIteration, CFRIterationRanged) hard-code a 2-player tree -
// buildNode copies only Utility[0]/Utility[1] of the terminal utility
// vector, and CFRIteration computes opp := uint8(1 - player), indexing
// [2]float32 arrays with it. At a 3+ seat table ActingPlayer() ranges 2..8,
// so 1-player underflows to 255 and the index panics inside libcambia.so -
// unrecoverable from Python (see subgame_solver.go's SubgameNode /
// CFRIteration doc comments for why this refuses outright instead of
// generalizing, in contrast to cambia-1171's seatOpponent posture for the
// 2-player action-space mask).
//
// These tests assert the FFI boundary rejects a non-2-player table before
// any of that code runs, at every entry point: build, solve, solve_ranged.

// TestCambiaSubgameBuildRejectsNonTwoPlayer asserts cambia_subgame_build
// returns -1 (rejected, the same sentinel this file uses for every other
// "call produced nothing usable" case) for every seat count from 3 through
// 8, and never hands back a usable solver handle.
func TestCambiaSubgameBuildRejectsNonTwoPlayer(t *testing.T) {
	for seats := uint8(3); seats <= 8; seats++ {
		gh := testGameNewWithRules(1, seats)
		if gh < 0 {
			t.Fatalf("testGameNewWithRules(numPlayers=%d) = %d, want a valid game handle", seats, gh)
		}

		sh := testSubgameBuild(gh, 2)
		if sh != -1 {
			if sh >= 0 {
				testSubgameFree(sh)
			}
			t.Errorf("cambia_subgame_build at %d seats = %d, want -1 (rejected: not a 2-player table)", seats, sh)
		}

		testGameFree(gh)
	}
}

// TestCambiaSubgameBuildAcceptsTwoPlayers is the control: the guard must not
// reject the 2-player table the solver actually supports.
func TestCambiaSubgameBuildAcceptsTwoPlayers(t *testing.T) {
	gh := testGameNewWithRules(1, 2)
	if gh < 0 {
		t.Fatalf("testGameNewWithRules(numPlayers=2) = %d, want a valid game handle", gh)
	}
	defer testGameFree(gh)

	sh := testSubgameBuild(gh, 2)
	if sh < 0 {
		t.Fatalf("cambia_subgame_build at 2 seats = %d, want a valid solver handle", sh)
	}
	testSubgameFree(sh)
}

// TestCambiaSubgameSolveRejectsNonTwoPlayer and its ranged sibling assert the
// same guard independently inside cambia_subgame_solve /
// cambia_subgame_solve_ranged (AC1: "cambia_subgame_solve and
// cambia_subgame_solve_ranged refuse likewise"). cambia_subgame_build never
// issues a handle for a non-2-player table, so the only way to reach this
// code path is a solverEntry installed some other way; these tests simulate
// that directly (same package as exports.go) by poking solverPool /
// solverInUse, which is exactly the defense-in-depth solverEntry.numPlayers
// exists to catch.
func TestCambiaSubgameSolveRejectsNonTwoPlayer(t *testing.T) {
	for seats := uint8(3); seats <= 8; seats++ {
		sh := allocSolver()
		if sh < 0 {
			t.Fatalf("allocSolver() = %d at numPlayers=%d, want a free slot", sh, seats)
		}
		root, leafCount := BuildSubgameTree(newTestGame(), 1)
		solverPool[sh].root = root
		solverPool[sh].leafCount = leafCount
		solverPool[sh].numPlayers = seats

		if rc := testSubgameSolve(sh, 1); rc != -1 {
			t.Errorf("cambia_subgame_solve at numPlayers=%d = %d, want -1 (rejected: not a 2-player table)", seats, rc)
		}
		if rc := testSubgameSolveRanged(sh, 1); rc != -1 {
			t.Errorf("cambia_subgame_solve_ranged at numPlayers=%d = %d, want -1 (rejected: not a 2-player table)", seats, rc)
		}

		freeSolver(sh)
	}
}

// TestCambiaSubgameSolveAcceptsTwoPlayers is the control for both solve
// entry points: the guard must not reject a tree actually built from a
// 2-player table.
func TestCambiaSubgameSolveAcceptsTwoPlayers(t *testing.T) {
	gh := testGameNewWithRules(1, 2)
	if gh < 0 {
		t.Fatalf("testGameNewWithRules(numPlayers=2) = %d, want a valid game handle", gh)
	}
	defer testGameFree(gh)

	sh := testSubgameBuild(gh, 2)
	if sh < 0 {
		t.Fatalf("cambia_subgame_build at 2 seats = %d, want a valid solver handle", sh)
	}
	defer testSubgameFree(sh)

	if rc := testSubgameSolve(sh, 1); rc != 0 {
		t.Errorf("cambia_subgame_solve at 2 seats = %d, want 0", rc)
	}
	if rc := testSubgameSolveRanged(sh, 1); rc != 0 {
		t.Errorf("cambia_subgame_solve_ranged at 2 seats = %d, want 0", rc)
	}
}
