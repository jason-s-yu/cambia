package main

/*
#include <stdint.h>
*/
import "C"

import (
	"sync"
	"unsafe"

	engine "github.com/jason-s-yu/cambia/engine"
	baselines "github.com/jason-s-yu/cambia/engine/baselines"
)

// Engine-side evaluation baselines (cambia-1487).
//
// The mean_imp battery scores an agent against five pure-Python heuristics.
// Their per-decision cost sat above the ported engine's, so the battery stopped
// tracking engine throughput. These exports move each policy behind the same
// FFI the game itself is driven through: one crossing per decision, decided
// from the engine state and the legal-action bitmask, with no decoded action
// list in Python at all.
//
// A random policy's uniform draw stays in Python. mean_imp is a historical
// metric and its recorded numbers depend on the exact CPython RNG stream, so
// cambia_baseline_choose hands back the candidate sequence instead of choosing,
// and the Python wrapper draws against it with the same rng.choice call the
// reference makes.

// maxBaselines bounds the baseline handle pool. Baselines are built once per
// evaluation leg, not per game, so the pool only ever holds a handful.
const maxBaselines = 1024

var (
	baselineMu    sync.Mutex
	baselinePool  [maxBaselines]*baselines.Agent
	baselineInUse [maxBaselines]bool
)

// baselineChooseHeader is the fixed prefix cambia_baseline_choose writes:
// [0] the answer kind (0 decided, 1 uniform), [1] the action index or the
// candidate count.
const baselineChooseHeader = 2

// cambia_baseline_new allocates a baseline policy for one seat and returns its
// handle, or -1 when the kind is unknown or the pool is full.
//
// kind is baselines.Kind: 0 random, 1 random-no-cambia, 2 random-late-cambia,
// 3 imperfect-greedy, 4 memory-heuristic, 5 aggressive-snap.
// cambia_threshold is the config's agents.greedy_agent.cambia_call_threshold;
// late_cambia_turns is RandomLateCambiaAgent's n_turns (0 takes its default).
//
//export cambia_baseline_new
func cambia_baseline_new(kind C.uint8_t, seat C.uint8_t, cambiaThreshold C.int32_t, lateCambiaTurns C.int32_t) C.int32_t {
	k := baselines.Kind(kind)
	if !baselines.ValidKind(k) {
		return -1
	}
	if seat >= C.uint8_t(engine.MaxPlayers) {
		return -1
	}
	agent := baselines.New(k, uint8(seat), int(cambiaThreshold), int(lateCambiaTurns))

	baselineMu.Lock()
	defer baselineMu.Unlock()
	for i := 0; i < maxBaselines; i++ {
		if !baselineInUse[i] {
			baselineInUse[i] = true
			baselinePool[i] = agent
			return C.int32_t(i)
		}
	}
	return -1
}

// cambia_baseline_free releases a baseline handle. Idempotent.
//
//export cambia_baseline_free
func cambia_baseline_free(h C.int32_t) {
	baselineMu.Lock()
	defer baselineMu.Unlock()
	if h >= 0 && h < maxBaselines {
		baselineInUse[h] = false
		baselinePool[h] = nil
	}
}

// cambia_baseline_reset marks the baseline as being in a new game, discarding
// its memory. Returns 0, or -1 on a bad handle.
//
//export cambia_baseline_reset
func cambia_baseline_reset(h C.int32_t) C.int32_t {
	agent := baselineAt(h)
	if agent == nil {
		return -1
	}
	agent.Reset()
	return 0
}

// cambia_baseline_choose writes the baseline's answer for the current state of
// game_h into out and returns the number of int32 entries written.
//
// Layout:
//
//	out[0] == 0  a decided action: out[1] is its index. Written length 2.
//	out[0] == 1  a uniform draw: out[1] is the candidate count n and
//	             out[2..2+n) are the candidate action indices, ascending.
//	             Written length 2+n.
//
// Returns -1 on a bad handle or a buffer shorter than 2+engine.NumActions, -2
// when the state has no legal action (which the Python reference reports by
// raising), and -3 when the game has more than two seats, whose action space
// these policies do not speak: the caller falls back to the Python body there.
//
//export cambia_baseline_choose
func cambia_baseline_choose(h C.int32_t, game_h C.int32_t, out *C.int32_t, out_len C.int32_t) C.int32_t {
	agent := baselineAt(h)
	if agent == nil {
		return -1
	}
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(out_len) < baselineChooseHeader+int(engine.NumActions) {
		return -1
	}
	if gamePool[game_h].NumActivePlayers() != 2 {
		return -3
	}
	buf := unsafe.Slice((*int32)(unsafe.Pointer(out)), int(out_len))

	decision, ok := agent.Choose(&gamePool[game_h])
	if !ok {
		return -2
	}
	if !decision.Uniform {
		buf[0] = 0
		buf[1] = int32(decision.Action)
		return baselineChooseHeader
	}
	buf[0] = 1
	buf[1] = int32(len(decision.Candidates))
	for i, idx := range decision.Candidates {
		buf[baselineChooseHeader+i] = int32(idx)
	}
	return C.int32_t(baselineChooseHeader + len(decision.Candidates))
}

// cambia_game_legal_indices writes the legal action indices of the 2-player
// action space into out, ascending, and returns how many it wrote. Returns -1
// on a bad handle or a buffer shorter than engine.NumActions.
//
// The bitmask export (cambia_game_legal_actions) is the encoder-facing surface;
// this one is for a caller that wants the indices themselves and would
// otherwise pay a numpy unpack per decision.
//
//export cambia_game_legal_indices
func cambia_game_legal_indices(game_h C.int32_t, out *C.int32_t, out_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(out_len) < int(engine.NumActions) {
		return -1
	}
	buf := unsafe.Slice((*int32)(unsafe.Pointer(out)), int(out_len))
	mask := gamePool[game_h].LegalActions()
	n := 0
	for idx := uint16(0); idx < engine.NumActions; idx++ {
		if mask[idx/64]>>(idx%64)&1 == 1 {
			buf[n] = int32(idx)
			n++
		}
	}
	return C.int32_t(n)
}

// baselineAt resolves a baseline handle, or nil when it names no live agent.
func baselineAt(h C.int32_t) *baselines.Agent {
	baselineMu.Lock()
	defer baselineMu.Unlock()
	if h < 0 || h >= maxBaselines || !baselineInUse[h] {
		return nil
	}
	return baselinePool[h]
}
