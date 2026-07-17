package main

/*
#include <stdint.h>
*/
import "C"

// Test-only plain-Go-typed wrappers around the cgo-exported functions. Go
// does not allow `import "C"` in _test.go files, so state_clone_test.go
// (and any other test needing the C ABI surface) calls through these instead.
// Unexported and carry no //export directive, so they contribute no symbols
// to libcambia.so's C ABI; harmless dead weight in the production build.

func testGameNew(seed uint64) int32 {
	return int32(cambia_game_new(C.uint64_t(seed)))
}

func testAgentNew(gameH int32, playerID, memoryLevel, timeDecay uint8) int32 {
	return int32(cambia_agent_new(
		C.int32_t(gameH), C.uint8_t(playerID), C.uint8_t(memoryLevel), C.uint8_t(timeDecay),
	))
}

func testGameFree(h int32)  { cambia_game_free(C.int32_t(h)) }
func testAgentFree(h int32) { cambia_agent_free(C.int32_t(h)) }

func testGameLegalActionsMask(h int32) [3]uint64 {
	var mask [3]C.uint64_t
	cambia_game_legal_actions(C.int32_t(h), &mask[0])
	return [3]uint64{uint64(mask[0]), uint64(mask[1]), uint64(mask[2])}
}

func testGameIsTerminal(h int32) bool {
	return cambia_game_is_terminal(C.int32_t(h)) == 1
}

func testGameTurnNumber(h int32) uint16 {
	return uint16(cambia_game_turn_number(C.int32_t(h)))
}

func testApplyOne(gameH, a0H, a1H int32, action uint16) int32 {
	gh := C.int32_t(gameH)
	a0 := C.int32_t(a0H)
	a1 := C.int32_t(a1H)
	act := C.uint16_t(action)
	return int32(cambia_games_apply_batch(&gh, &a0, &a1, &act, 1))
}

func testAgentTokenLen(h int32) int32 {
	return int32(cambia_agent_token_len(C.int32_t(h)))
}

// testAgentTokens returns a copy of the full token stream (nil on error or
// empty stream).
func testAgentTokens(h int32) []int32 {
	n := testAgentTokenLen(h)
	if n <= 0 {
		return nil
	}
	buf := make([]int32, n)
	got := cambia_agent_tokens(C.int32_t(h), (*C.int32_t)(&buf[0]), C.int32_t(n))
	if int32(got) != n {
		return nil
	}
	return buf
}

// testStateClone returns (ok, newGame, newA0, newA1). newGame/newA0/newA1 are
// -1 and must not be used when ok is false.
func testStateClone(gameH, a0H, a1H int32) (ok bool, newGame, newA0, newA1 int32) {
	var cg, ca0, ca1 C.int32_t
	ret := cambia_state_clone(C.int32_t(gameH), C.int32_t(a0H), C.int32_t(a1H), &cg, &ca0, &ca1)
	if ret != 0 {
		return false, -1, -1, -1
	}
	return true, int32(cg), int32(ca0), int32(ca1)
}

// testGameNewWithRules is a minimal test-only wrapper over
// cambia_game_new_with_rules covering only the fields nplayer_bounds_test.go
// needs (seed, numPlayers); every other field takes a DefaultHouseRules()-
// shaped value. Returns the raw handle (or the FFI's -1 failure sentinel).
func testGameNewWithRules(seed uint64, numPlayers uint8) int32 {
	return int32(cambia_game_new_with_rules(
		C.uint64_t(seed),
		C.uint16_t(46), // maxGameTurns
		C.uint8_t(4),   // cardsPerPlayer
		C.uint8_t(0),   // cambiaAllowedRound
		C.uint8_t(2),   // penaltyDrawCount
		C.uint8_t(1),   // allowDrawFromDiscard
		C.uint8_t(0),   // allowReplaceAbilities
		C.uint8_t(1),   // allowOpponentSnapping
		C.uint8_t(0),   // snapRace
		C.uint8_t(2),   // numJokers
		C.uint8_t(1),   // lockCallerHand
		C.uint8_t(numPlayers),
		C.uint8_t(2), // initialViewCount
		C.uint8_t(1), // numDecks
	))
}

func testNPlayerInputDim() int32   { return int32(cambia_nplayer_input_dim()) }
func testNPlayerNumActions() int32 { return int32(cambia_nplayer_num_actions()) }
