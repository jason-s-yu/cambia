package main

import (
	"math/bits"
	"reflect"
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
	agent "github.com/jason-s-yu/cambia/engine/agent"
)

// batch_workers_test.go covers cambia_set_batch_workers + the parallel fan-out
// path of cambia_games_apply_batch (cambia-656): a byte-parity check that
// workers=1 and workers=8 produce identical game state, token streams, and agent
// encodes over full randomized playouts, plus an error-index determinism check
// that both modes report the lowest failing game index.

// xorshiftNext advances a per-game xorshift64 chooser (same generator the engine
// uses). Seeds are kept nonzero by the caller.
func xorshiftNext(s *uint64) uint64 {
	x := *s
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*s = x
	return x
}

// legalActionList expands a 3-word legal-action mask into the list of set action
// indices, low index first.
func legalActionList(mask [3]uint64) []uint16 {
	var out []uint16
	for w := 0; w < 3; w++ {
		word := mask[w]
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			out = append(out, uint16(w*64+bit))
			word &= word - 1
		}
	}
	return out
}

// batchSimResult is the terminal snapshot of one full playout run: per-game
// engine state, both agents' belief state, token streams, and legacy encodes.
type batchSimResult struct {
	games    []engine.GameState
	a0       []agent.AgentState
	a1       []agent.AgentState
	tok0     [][]int32
	tok1     [][]int32
	enc0     [][]float32
	enc1     [][]float32
	terminal []bool
}

// runBatchSim plays nGames independent games to completion (or maxTicks),
// applying one deterministically chosen legal action per live game per tick via
// cambia_games_apply_batch at the given worker count, then snapshots and frees
// all handles. Game i is seeded seedBase+i and its action chooser is seeded from
// i alone, so two runs at different worker counts produce byte-identical
// trajectories: the only difference under test is the serial vs parallel apply.
func runBatchSim(t *testing.T, nGames int, workers int32, seedBase uint64) batchSimResult {
	t.Helper()
	testSetBatchWorkers(workers)
	defer testSetBatchWorkers(1)

	gh := make([]int32, nGames)
	a0 := make([]int32, nGames)
	a1 := make([]int32, nGames)
	choice := make([]uint64, nGames)
	for i := 0; i < nGames; i++ {
		g := testGameNew(seedBase + uint64(i))
		if g < 0 {
			t.Fatalf("cambia_game_new failed at game %d: %d", i, g)
		}
		p0 := testAgentNew(g, 0, 0, 0)
		p1 := testAgentNew(g, 1, 0, 0)
		if p0 < 0 || p1 < 0 {
			t.Fatalf("cambia_agent_new failed at game %d: %d/%d", i, p0, p1)
		}
		gh[i], a0[i], a1[i] = g, p0, p1
		// Chooser seed depends on i only (independent of seedBase/workers), kept
		// nonzero for xorshift.
		choice[i] = (uint64(i)*0x9e3779b97f4a7c15 ^ 0x2545f4914f6cdd1d) | 1
	}

	const maxTicks = 4000
	for tick := 0; tick < maxTicks; tick++ {
		var bg, ba0, ba1 []int32
		var bacts []uint16
		for i := 0; i < nGames; i++ {
			if testGameIsTerminal(gh[i]) {
				continue
			}
			acts := legalActionList(testGameLegalActionsMask(gh[i]))
			if len(acts) == 0 {
				continue
			}
			pick := acts[xorshiftNext(&choice[i])%uint64(len(acts))]
			bg = append(bg, gh[i])
			ba0 = append(ba0, a0[i])
			ba1 = append(ba1, a1[i])
			bacts = append(bacts, pick)
		}
		if len(bg) == 0 {
			break
		}
		if ret := testApplyBatch(bg, ba0, ba1, bacts); ret != 0 {
			t.Fatalf("apply batch failed (workers=%d tick=%d): %d", workers, tick, ret)
		}
	}

	res := batchSimResult{
		games:    make([]engine.GameState, nGames),
		a0:       make([]agent.AgentState, nGames),
		a1:       make([]agent.AgentState, nGames),
		tok0:     make([][]int32, nGames),
		tok1:     make([][]int32, nGames),
		enc0:     make([][]float32, nGames),
		enc1:     make([][]float32, nGames),
		terminal: make([]bool, nGames),
	}
	for i := 0; i < nGames; i++ {
		// Direct pool reads (in-package) capture full byte state before free.
		res.games[i] = gamePool[gh[i]]
		res.a0[i] = agentPool[a0[i]]
		res.a1[i] = agentPool[a1[i]]
		res.tok0[i] = testAgentTokens(a0[i])
		res.tok1[i] = testAgentTokens(a1[i])
		res.terminal[i] = testGameIsTerminal(gh[i])
		e0 := make([]float32, agent.InputDim)
		e1 := make([]float32, agent.InputDim)
		if r := testAgentEncode(a0[i], uint8(engine.CtxStartTurn), -1, e0); r != 0 {
			t.Fatalf("encode a0 failed at game %d: %d", i, r)
		}
		if r := testAgentEncode(a1[i], uint8(engine.CtxStartTurn), -1, e1); r != 0 {
			t.Fatalf("encode a1 failed at game %d: %d", i, r)
		}
		res.enc0[i], res.enc1[i] = e0, e1
	}
	for i := 0; i < nGames; i++ {
		freeTriple(gh[i], a0[i], a1[i])
	}
	return res
}

func equalFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestBatchWorkersByteParity asserts the parallel fan-out (workers=8) is
// byte-identical to the serial loop (workers=1) over full randomized playouts:
// identical final game state, agent belief state, token streams, and encodes for
// every game.
func TestBatchWorkersByteParity(t *testing.T) {
	// 512 games with minBatchChunk=64 yield an 8-way split on the early, fully
	// live ticks, genuinely exercising the parallel path.
	const nGames = 512
	const seedBase = 0xC0FFEE

	serial := runBatchSim(t, nGames, 1, seedBase)
	parallel := runBatchSim(t, nGames, 8, seedBase)

	// Sanity: the sim must have done real work, else parity is vacuous.
	termCount := 0
	totalTokens := 0
	for i := 0; i < nGames; i++ {
		if serial.terminal[i] {
			termCount++
		}
		totalTokens += len(serial.tok0[i])
	}
	if termCount == 0 {
		t.Fatalf("no games terminated; playout did no meaningful work")
	}
	if totalTokens < nGames {
		t.Fatalf("token streams implausibly short (total=%d over %d games)", totalTokens, nGames)
	}

	for i := 0; i < nGames; i++ {
		if !reflect.DeepEqual(serial.games[i], parallel.games[i]) {
			t.Fatalf("game %d: state differs between workers=1 and workers=8", i)
		}
		if !reflect.DeepEqual(serial.a0[i], parallel.a0[i]) {
			t.Fatalf("game %d: agent0 belief state differs between modes", i)
		}
		if !reflect.DeepEqual(serial.a1[i], parallel.a1[i]) {
			t.Fatalf("game %d: agent1 belief state differs between modes", i)
		}
		if !equalTokens(serial.tok0[i], parallel.tok0[i]) {
			t.Fatalf("game %d: agent0 token stream differs between modes", i)
		}
		if !equalTokens(serial.tok1[i], parallel.tok1[i]) {
			t.Fatalf("game %d: agent1 token stream differs between modes", i)
		}
		if !equalFloats(serial.enc0[i], parallel.enc0[i]) {
			t.Fatalf("game %d: agent0 encode differs between modes", i)
		}
		if !equalFloats(serial.enc1[i], parallel.enc1[i]) {
			t.Fatalf("game %d: agent1 encode differs between modes", i)
		}
	}
}

// setupErrBatch builds a fresh valid batch of nGames games + agents and returns
// parallel handle/action slices with a legal action chosen for each game. The
// caller mutates chosen entries to inject failures.
func setupErrBatch(t *testing.T, nGames int, seedBase uint64) (gh, a0, a1 []int32, acts []uint16) {
	t.Helper()
	gh = make([]int32, nGames)
	a0 = make([]int32, nGames)
	a1 = make([]int32, nGames)
	acts = make([]uint16, nGames)
	for i := 0; i < nGames; i++ {
		g := testGameNew(seedBase + uint64(i))
		if g < 0 {
			t.Fatalf("game_new failed at %d", i)
		}
		gh[i] = g
		a0[i] = testAgentNew(g, 0, 0, 0)
		a1[i] = testAgentNew(g, 1, 0, 0)
		acts[i] = firstLegalAction(t, g)
	}
	return gh, a0, a1, acts
}

func freeErrBatch(gh, a0, a1 []int32) {
	for i := range gh {
		freeTriple(gh[i], a0[i], a1[i])
	}
}

// firstIllegalAction returns an action index that is NOT legal for the game, so
// ApplyAction rejects it (apply-path -1 error).
func firstIllegalAction(t *testing.T, gh int32) uint16 {
	t.Helper()
	mask := testGameLegalActionsMask(gh)
	for a := 0; a < int(agent.NumActions); a++ {
		w, bit := a/64, uint(a%64)
		if mask[w]&(1<<bit) == 0 {
			return uint16(a)
		}
	}
	t.Fatalf("every action index is legal on handle %d; cannot pick an illegal one", gh)
	return 0
}

// TestBatchWorkersErrorIndexDeterminism asserts that both serial and parallel
// modes report the SAME error code for the lowest failing game index, across the
// handle-error and apply-path error classes.
func TestBatchWorkersErrorIndexDeterminism(t *testing.T) {
	const nGames = 256 // >= 2*minBatchChunk, so workers=8 splits into >=2 chunks
	const badGame = int32(1 << 30)

	// Case 1: a single invalid game handle mid-batch -> -1 in both modes.
	t.Run("InvalidHandleMidBatch", func(t *testing.T) {
		run := func(workers int32) int32 {
			gh, a0, a1, acts := setupErrBatch(t, nGames, 1)
			defer freeErrBatch(gh, a0, a1)
			gh[100] = badGame
			testSetBatchWorkers(workers)
			defer testSetBatchWorkers(1)
			return testApplyBatch(gh, a0, a1, acts)
		}
		if s, p := run(1), run(8); s != -1 || p != -1 {
			t.Fatalf("invalid-handle: serial=%d parallel=%d, want -1/-1", s, p)
		}
	})

	// Case 2: two invalid handles (lo<hi) -> lowest index still -1 in both modes.
	t.Run("TwoInvalidHandlesLowestWins", func(t *testing.T) {
		run := func(workers int32) int32 {
			gh, a0, a1, acts := setupErrBatch(t, nGames, 2)
			defer freeErrBatch(gh, a0, a1)
			gh[40] = badGame
			gh[200] = badGame
			testSetBatchWorkers(workers)
			defer testSetBatchWorkers(1)
			return testApplyBatch(gh, a0, a1, acts)
		}
		if s, p := run(1), run(8); s != -1 || p != -1 {
			t.Fatalf("two-invalid-handle: serial=%d parallel=%d, want -1/-1", s, p)
		}
	})

	// Case 3: apply-path errors at two indices with DISTINGUISHABLE codes, all
	// handles valid, proving the lowest failing index decides the return.
	// Inject a token overflow (-2) and an illegal action (-1); whichever sits at
	// the lower index must be the reported code in both modes.
	t.Run("LowestApplyPathIndexWins", func(t *testing.T) {
		// Sub-case A: overflow at 30 (-2), illegal action at 150 (-1) -> -2.
		runA := func(workers int32) int32 {
			gh, a0, a1, acts := setupErrBatch(t, nGames, 3)
			defer freeErrBatch(gh, a0, a1)
			// Force agent a0[30]'s next Observe to overflow; its action stays legal
			// so ApplyAction succeeds and control reaches Observe.
			tokenPool[a0[30]].Length = agent.MaxTokenStream
			acts[150] = firstIllegalAction(t, gh[150])
			testSetBatchWorkers(workers)
			defer testSetBatchWorkers(1)
			return testApplyBatch(gh, a0, a1, acts)
		}
		if s, p := runA(1), runA(8); s != -2 || p != -2 {
			t.Fatalf("overflow-low: serial=%d parallel=%d, want -2/-2", s, p)
		}

		// Sub-case B: illegal action at 30 (-1), overflow at 150 (-2) -> -1.
		runB := func(workers int32) int32 {
			gh, a0, a1, acts := setupErrBatch(t, nGames, 4)
			defer freeErrBatch(gh, a0, a1)
			acts[30] = firstIllegalAction(t, gh[30])
			tokenPool[a0[150]].Length = agent.MaxTokenStream
			testSetBatchWorkers(workers)
			defer testSetBatchWorkers(1)
			return testApplyBatch(gh, a0, a1, acts)
		}
		if s, p := runB(1), runB(8); s != -1 || p != -1 {
			t.Fatalf("illegal-low: serial=%d parallel=%d, want -1/-1", s, p)
		}
	})
}
