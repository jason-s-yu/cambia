package main

import "testing"

// nplayer_agent_seats_test.go covers the seat range cambia_agent_new_nplayer accepts.
// AgentState.OpponentIDs held five entries while the engine deals up to eight seats, so a
// seven or eight seat agent wrote past the array and panicked inside libcambia.so, which
// no Python caller can catch; the export validated the game handle and nothing else
// (cambia-1551).

// TestCambiaAgentNewNPlayerConstructsAtEightSeats builds an agent for every seat of a full
// table and encodes from it, which is the path that used to take the process down.
func TestCambiaAgentNewNPlayerConstructsAtEightSeats(t *testing.T) {
	for _, seats := range []uint8{7, 8} {
		gh := testGameNewWithRules(7, seats)
		if gh < 0 {
			t.Fatalf("cambia_game_new_with_rules(numPlayers=%d) = %d, want a valid handle", seats, gh)
		}
		for seat := uint8(0); seat < seats; seat++ {
			ah := testAgentNewNPlayer(gh, seat, seats, 1, 5)
			if ah < 0 {
				t.Fatalf("cambia_agent_new_nplayer(seat=%d, numPlayers=%d) = %d, want a valid handle",
					seat, seats, ah)
			}
			out := make([]float32, testNPlayerInputDim())
			if rc := testAgentEncodeNPlayer(ah, 0, -1, out); rc != 0 {
				t.Fatalf("cambia_agent_encode_nplayer(seat=%d, numPlayers=%d) = %d, want 0",
					seat, seats, rc)
			}
			testAgentFree(ah)
		}
		testGameFree(gh)
	}
}

// TestCambiaAgentNewNPlayerRejectsAnImpossibleTable asserts the export refuses a seat count
// or a seat index the engine cannot deal, rather than returning a belief state that
// describes no game.
func TestCambiaAgentNewNPlayerRejectsAnImpossibleTable(t *testing.T) {
	gh := testGameNewWithRules(7, 4)
	if gh < 0 {
		t.Fatalf("cambia_game_new_with_rules(numPlayers=4) = %d, want a valid handle", gh)
	}
	defer testGameFree(gh)

	cases := []struct {
		name       string
		playerID   uint8
		numPlayers uint8
	}{
		{"nine seats", 0, 9},
		{"one seat", 0, 1},
		{"seat past the table", 4, 4},
	}
	for _, c := range cases {
		ah := testAgentNewNPlayer(gh, c.playerID, c.numPlayers, 1, 5)
		if ah >= 0 {
			testAgentFree(ah)
			t.Errorf("cambia_agent_new_nplayer(%s) = %d, want -1 (rejected)", c.name, ah)
		}
	}
}
