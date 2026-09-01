// Package cgo provides C-exported functions for building libcambia.so.
//
// Build with: go build -buildmode=c-shared -o cfr/libcambia.so ./engine/cgo/
//
// PRT-CFR event-stream token surface (S1W2, additive): each agent carries an
// append-only int32 token stream mirroring cfr/src/sequence_encoding.py.
//
//	cambia_agent_token_len / cambia_agent_tokens / cambia_agent_tokens_since
//	  - read the full stream / an incremental tail.
//	cambia_games_apply_batch - vectorized apply + per-game agent belief+token
//	  update, O(n) with no per-game Python roundtrip.
//	cambia_set_batch_workers - opt-in parallel fan-out width for
//	  cambia_games_apply_batch (cambia-656); default 1 keeps the serial path.
//	cambia_state_save / cambia_state_restore / cambia_state_snapshot_free -
//	  token-inclusive (game, both-agents) checkpoint pair (additive to the
//	  game-only cambia_game_save/restore, which are unchanged).
//	cambia_state_clone - independent (game, both-agents) clone onto FRESH
//	  handles, for rollout fan-out (S1W12; distinct from state_save/restore's
//	  rewind-on-same-handles checkpointing).
//	cambia_token_vocab / cambia_token_encode_card / cambia_token_encode_action
//	  - expose the Go vocabulary layout + mappings for the parity cross-check.
//	cambia_token_stream_cap - the live MaxTokenStream value (S1W12), paired
//	  with cfr/src/cfr/prtcfr_worker.py::PRODUCTION_SEQ_CAP.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"unsafe"

	engine "github.com/jason-s-yu/cambia/engine"
	agent "github.com/jason-s-yu/cambia/engine/agent"
)

// ---------------------------------------------------------------------------
// Handle pools
// ---------------------------------------------------------------------------

const (
	// maxGames/maxAgents bound the flat handle pools. Raised from 2048/4096 to
	// lift the gen_chunk_games ceiling (cambia-534, X3 inference-dominance step
	// (a)): the old maxGames=2048 capped concurrent games per chunk near 64,
	// forcing ~128 sequential chunks and mean inference batch 114 (1.4% of
	// capacity). At 32768 the chunk ceiling rises ~16x.
	//
	// Cost: the game/agent pools stay package-level BSS arrays; tokenPool is
	// heap-allocated in init (see its declaration for why), reserving
	// maxAgents * sizeof(TokenStream) = 65536 * 49416B = 3.02 GiB VIRTUAL. It is demand-zero paged and TokenStream/AgentState are
	// pointer-free flat value types, so the GC never scans the region: resident
	// RAM stays ~0 at rest and tracks concurrently-live agents (a 1024-game /
	// ~2048-agent chunk peaks near 96 MiB). No VRAM cost (host-side only).
	maxGames     = 32768
	maxAgents    = 65536
	maxSnapshots = 256
	maxSolvers   = 32
)

// solverEntry holds a built subgame tree and associated metadata.
type solverEntry struct {
	root      *SubgameNode
	leafCount int
}

// stateSnapshot is a complete (game, both agents' belief + token state)
// checkpoint used by cambia_state_save/restore for per-node rollout fan-out.
type stateSnapshot struct {
	game engine.GameState
	a0   agent.AgentState
	a1   agent.AgentState
	t0   agent.TokenStream
	t1   agent.TokenStream
}

var (
	poolMu sync.Mutex

	gamePool  [maxGames]engine.GameState
	gameInUse [maxGames]bool

	agentPool  [maxAgents]agent.AgentState
	agentInUse [maxAgents]bool
	agentGameH [maxAgents]int32 // which game handle each agent is associated with

	// tokenPool holds the per-agent PRT-CFR observation token stream, parallel
	// to agentPool by handle. Kept outside AgentState so that Go-internal CFR
	// clone-by-value stays cheap; only FFI clone / state-snapshot copy it.
	//
	// Heap-backed (allocated once in init below) rather than a BSS array: at
	// maxAgents=65536 the array is 65536 * sizeof(TokenStream) = 3.02 GiB, which
	// exceeds the Go linker's hard 2 GB single-symbol limit and fails the
	// c-shared link. A slice sidesteps that limit; indexing is source-identical
	// to an array so every call site is unchanged. The backing is demand-zero
	// paged and TokenStream is a pointer-free flat value type, so the GC marks
	// the span noscan and never touches it: resident RAM stays ~0 at rest and
	// tracks concurrently-live agents. TokenStream keeps its inline array, so
	// stateSnapshot value-copy (state_save/restore/clone) semantics are intact.
	tokenPool []agent.TokenStream

	snapPool  [maxSnapshots]engine.GameState // Snapshot = GameState copy
	snapInUse [maxSnapshots]bool

	// stateSnapPool backs the token-inclusive cambia_state_save/restore pair.
	stateSnapPool  [maxSnapshots]stateSnapshot
	stateSnapInUse [maxSnapshots]bool

	solverPool  [maxSolvers]solverEntry
	solverInUse [maxSolvers]bool
)

// batchWorkers is the cambia_games_apply_batch fan-out knob (cambia-656). The
// default 1 keeps the historical serial loop as the exact, byte-identical code
// path; a value >1 enables chunked per-game parallel apply. It is read on every
// batch and set (rarely, at sampler init) via cambia_set_batch_workers, so the
// access is atomic to stay race-detector clean if the setter and a batch overlap.
var batchWorkers atomic.Int32

// minBatchChunk is the smallest per-worker game count for the parallel path. A
// batch below 2*minBatchChunk, or one that would yield a single effective
// worker, stays on the serial path, so n=1 applies and tiny batches never pay
// goroutine setup and keep today's behavior exactly.
const minBatchChunk = 64

// init allocates the heap-backed tokenPool once at library load. See the
// tokenPool declaration for why it is not a BSS array.
func init() {
	tokenPool = make([]agent.TokenStream, maxAgents)
	batchWorkers.Store(1)
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

func allocGame() int32 {
	poolMu.Lock()
	defer poolMu.Unlock()
	for i := 0; i < maxGames; i++ {
		if !gameInUse[i] {
			gameInUse[i] = true
			return int32(i)
		}
	}
	return -1
}

func freeGame(h int32) {
	poolMu.Lock()
	defer poolMu.Unlock()
	if h >= 0 && h < maxGames {
		gameInUse[h] = false
		gamePool[h] = engine.GameState{}
	}
}

func allocAgent() int32 {
	poolMu.Lock()
	defer poolMu.Unlock()
	for i := 0; i < maxAgents; i++ {
		if !agentInUse[i] {
			agentInUse[i] = true
			return int32(i)
		}
	}
	return -1
}

func freeAgent(h int32) {
	poolMu.Lock()
	defer poolMu.Unlock()
	if h >= 0 && h < maxAgents {
		agentInUse[h] = false
		agentPool[h] = agent.AgentState{}
		agentGameH[h] = -1
		tokenPool[h].Reset()
	}
}

func allocSnapshot() int32 {
	poolMu.Lock()
	defer poolMu.Unlock()
	for i := 0; i < maxSnapshots; i++ {
		if !snapInUse[i] {
			snapInUse[i] = true
			return int32(i)
		}
	}
	return -1
}

func freeSnapshot(h int32) {
	poolMu.Lock()
	defer poolMu.Unlock()
	if h >= 0 && h < maxSnapshots {
		snapInUse[h] = false
		snapPool[h] = engine.GameState{}
	}
}

func allocStateSnapshot() int32 {
	poolMu.Lock()
	defer poolMu.Unlock()
	for i := 0; i < maxSnapshots; i++ {
		if !stateSnapInUse[i] {
			stateSnapInUse[i] = true
			return int32(i)
		}
	}
	return -1
}

func freeStateSnapshot(h int32) {
	poolMu.Lock()
	defer poolMu.Unlock()
	if h >= 0 && h < maxSnapshots {
		stateSnapInUse[h] = false
		stateSnapPool[h] = stateSnapshot{}
	}
}

func allocSolver() int32 {
	poolMu.Lock()
	defer poolMu.Unlock()
	for i := 0; i < maxSolvers; i++ {
		if !solverInUse[i] {
			solverInUse[i] = true
			return int32(i)
		}
	}
	return -1
}

func freeSolver(h int32) {
	poolMu.Lock()
	defer poolMu.Unlock()
	if h >= 0 && h < maxSolvers {
		solverInUse[h] = false
		solverPool[h] = solverEntry{}
	}
}

// ---------------------------------------------------------------------------
// Card index conversion helpers
// ---------------------------------------------------------------------------

// indexToCard converts a canonical Python-side card index to a Go Card.
// Index encoding: suit*13+rank where suit C=0,D=1,H=2,S=3; rank A=0..K=12.
// Jokers: 52 = RedJoker, 53 = BlackJoker.
func indexToCard(idx uint8) engine.Card {
	if idx == 52 {
		return engine.NewCard(engine.SuitRedJoker, engine.RankJoker)
	}
	if idx == 53 {
		return engine.NewCard(engine.SuitBlackJoker, engine.RankJoker)
	}
	suit := idx / 13
	rank := idx % 13
	var goSuit uint8
	switch suit {
	case 0:
		goSuit = engine.SuitClubs
	case 1:
		goSuit = engine.SuitDiamonds
	case 2:
		goSuit = engine.SuitHearts
	default:
		goSuit = engine.SuitSpades
	}
	return engine.NewCard(goSuit, rank)
}

// ---------------------------------------------------------------------------
// Game lifecycle
// ---------------------------------------------------------------------------

//export cambia_game_new
func cambia_game_new(seed C.uint64_t) C.int32_t {
	h := allocGame()
	if h < 0 {
		return -1
	}
	gamePool[h] = engine.NewGame(uint64(seed), engine.DefaultHouseRules())
	gamePool[h].Deal()
	return C.int32_t(h)
}

//export cambia_game_new_with_rules
func cambia_game_new_with_rules(
	seed C.uint64_t,
	maxGameTurns C.uint16_t,
	cardsPerPlayer C.uint8_t,
	cambiaAllowedRound C.uint8_t,
	penaltyDrawCount C.uint8_t,
	allowDrawFromDiscard C.uint8_t,
	allowReplaceAbilities C.uint8_t,
	allowOpponentSnapping C.uint8_t,
	snapRace C.uint8_t,
	numJokers C.uint8_t,
	lockCallerHand C.uint8_t,
	numPlayers C.uint8_t,
	initialViewCount C.uint8_t,
	numDecks C.uint8_t,
) C.int32_t {
	h := allocGame()
	if h < 0 {
		return -1
	}
	rules := engine.HouseRules{
		MaxGameTurns:          uint16(maxGameTurns),
		CardsPerPlayer:        uint8(cardsPerPlayer),
		CambiaAllowedRound:    uint8(cambiaAllowedRound),
		PenaltyDrawCount:      uint8(penaltyDrawCount),
		AllowDrawFromDiscard:  allowDrawFromDiscard != 0,
		AllowReplaceAbilities: allowReplaceAbilities != 0,
		AllowOpponentSnapping: allowOpponentSnapping != 0,
		SnapRace:              snapRace != 0,
		NumJokers:             uint8(numJokers),
		LockCallerHand:        lockCallerHand != 0,
		NumPlayers:            uint8(numPlayers),
		InitialViewCount:      uint8(initialViewCount),
		NumDecks:              uint8(numDecks),
	}
	// cambia-542 F3: numPlayers arrives here as a raw, externally-supplied
	// uint8 from Python via the FFI. Reject out-of-range values instead of
	// silently clamping and succeeding - Deal() below indexes g.Players[p]
	// for p in [0, NumPlayers), and that array is fixed at [MaxPlayers]; an
	// unrejected NumPlayers=9 previously panicked inside libcambia.so.
	if err := rules.Validate(); err != nil {
		freeGame(h)
		return -1
	}
	gamePool[h] = engine.NewGame(uint64(seed), rules)
	gamePool[h].Deal()
	return C.int32_t(h)
}

//export cambia_game_new_with_deck
func cambia_game_new_with_deck(
	deckPtr *C.uint8_t, deckLen C.int32_t,
	numPlayers C.uint8_t, cardsPerPlayer C.uint8_t,
	startingPlayer C.uint8_t,
	maxGameTurns C.uint16_t, cambiaAllowedRound C.uint8_t,
	penaltyDrawCount C.uint8_t, allowDrawFromDiscard C.uint8_t,
	allowReplaceAbilities C.uint8_t, allowOpponentSnapping C.uint8_t,
	snapRace C.uint8_t, numJokers C.uint8_t, lockCallerHand C.uint8_t,
	initialViewCount C.uint8_t, numDecks C.uint8_t,
) C.int32_t {
	h := allocGame()
	if h < 0 {
		return -1
	}
	rules := engine.HouseRules{
		MaxGameTurns:          uint16(maxGameTurns),
		CardsPerPlayer:        uint8(cardsPerPlayer),
		CambiaAllowedRound:    uint8(cambiaAllowedRound),
		PenaltyDrawCount:      uint8(penaltyDrawCount),
		AllowDrawFromDiscard:  allowDrawFromDiscard != 0,
		AllowReplaceAbilities: allowReplaceAbilities != 0,
		AllowOpponentSnapping: allowOpponentSnapping != 0,
		SnapRace:              snapRace != 0,
		NumJokers:             uint8(numJokers),
		LockCallerHand:        lockCallerHand != 0,
		NumPlayers:            uint8(numPlayers),
		InitialViewCount:      uint8(initialViewCount),
		NumDecks:              uint8(numDecks),
	}
	// cambia-542 F3: reject an out-of-range NumPlayers instead of clamping
	// and continuing - the round-robin deal loop below indexes g.Players[p]
	// for p in [0, np), and that array is fixed at [MaxPlayers].
	if err := rules.Validate(); err != nil {
		freeGame(h)
		return -1
	}
	// Create a base game state with the right rules (deck contents will be overwritten).
	gamePool[h] = engine.NewGame(1, rules)
	g := &gamePool[h]
	np := g.NumActivePlayers() // applies the documented 0-defaults-to-2 sentinel

	// Load provided deck into stockpile in reverse order so that deck[0] is
	// the first card popped (i.e., placed at Stockpile[deckLen-1]).
	n := int(deckLen)
	if n > engine.MaxDeckSize {
		n = engine.MaxDeckSize
	}
	deck := (*[256]C.uint8_t)(unsafe.Pointer(deckPtr))
	for i := 0; i < n; i++ {
		g.Stockpile[n-1-i] = indexToCard(uint8(deck[i]))
	}
	g.StockLen = uint8(n)

	// Round-robin deal (same as Deal() but without Fisher-Yates shuffle).
	cpp := rules.CardsPerPlayer
	for c := uint8(0); c < cpp; c++ {
		for p := uint8(0); p < np; p++ {
			g.StockLen--
			card := g.Stockpile[g.StockLen]
			g.Players[p].Hand[c] = card
			g.Players[p].HandLen++
		}
	}

	// Set initial peek indices.
	for p := uint8(0); p < np; p++ {
		count := rules.InitialViewCount
		if count > cpp {
			count = cpp
		}
		for i := uint8(0); i < count; i++ {
			g.Players[p].InitialPeek[i] = i
		}
		g.Players[p].InitialPeekCount = count
	}

	// Flip top stockpile card to start the discard pile.
	g.StockLen--
	g.DiscardPile[0] = g.Stockpile[g.StockLen]
	g.DiscardLen = 1

	// Set starting player and mark game as started.
	g.CurrentPlayer = uint8(startingPlayer)
	g.Flags |= engine.FlagGameStarted

	return C.int32_t(h)
}

//export cambia_game_free
func cambia_game_free(h C.int32_t) {
	freeGame(int32(h))
}

//export cambia_game_apply_action
func cambia_game_apply_action(h C.int32_t, action_idx C.uint16_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	err := gamePool[h].ApplyAction(uint16(action_idx))
	if err != nil {
		return -1
	}
	return 0
}

//export cambia_game_legal_actions
func cambia_game_legal_actions(h C.int32_t, out *C.uint64_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	mask := gamePool[h].LegalActions()
	outSlice := (*[3]C.uint64_t)(unsafe.Pointer(out))
	outSlice[0] = C.uint64_t(mask[0])
	outSlice[1] = C.uint64_t(mask[1])
	outSlice[2] = C.uint64_t(mask[2])
	return 0
}

//export cambia_game_is_terminal
func cambia_game_is_terminal(h C.int32_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	if gamePool[h].IsTerminal() {
		return 1
	}
	return 0
}

//export cambia_game_get_utility
func cambia_game_get_utility(h C.int32_t, out *C.float) {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return
	}
	u := gamePool[h].GetUtility()
	outSlice := (*[2]C.float)(unsafe.Pointer(out))
	outSlice[0] = C.float(u[0])
	outSlice[1] = C.float(u[1])
}

//export cambia_game_acting_player
func cambia_game_acting_player(h C.int32_t) C.uint8_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 255 // error sentinel
	}
	return C.uint8_t(gamePool[h].ActingPlayer())
}

//export cambia_game_save
func cambia_game_save(game_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	sh := allocSnapshot()
	if sh < 0 {
		return -1
	}
	snapPool[sh] = gamePool[game_h] // value copy = snapshot
	return C.int32_t(sh)
}

//export cambia_game_restore
func cambia_game_restore(game_h C.int32_t, snap_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if snap_h < 0 || snap_h >= maxSnapshots || !snapInUse[snap_h] {
		return -1
	}
	gamePool[game_h] = snapPool[snap_h] // restore from snapshot
	return 0
}

//export cambia_snapshot_free
func cambia_snapshot_free(h C.int32_t) {
	freeSnapshot(int32(h))
}

// ---------------------------------------------------------------------------
// Agent lifecycle
// ---------------------------------------------------------------------------

//export cambia_agent_new
func cambia_agent_new(game_h C.int32_t, player_id C.uint8_t, memory_level C.uint8_t, time_decay_turns C.uint8_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	ah := allocAgent()
	if ah < 0 {
		return -1
	}
	pid := uint8(player_id)
	var oppID uint8
	if pid == 0 {
		oppID = 1
	} else {
		oppID = 0
	}
	agentPool[ah] = agent.NewAgentState(pid, oppID, uint8(memory_level), uint8(time_decay_turns))
	agentPool[ah].Initialize(&gamePool[game_h])
	agentGameH[ah] = int32(game_h)
	// Seed the PRT-CFR token stream with the observer's private init-peek frames.
	tokenPool[ah].Init(&gamePool[game_h], pid)
	return C.int32_t(ah)
}

//export cambia_agent_new_with_memory
func cambia_agent_new_with_memory(game_h C.int32_t, player_id C.uint8_t, memory_level C.uint8_t, time_decay_turns C.uint8_t, memory_archetype C.uint8_t, memory_decay_lambda C.double, memory_capacity C.uint8_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	ah := allocAgent()
	if ah < 0 {
		return -1
	}
	pid := uint8(player_id)
	var oppID uint8
	if pid == 0 {
		oppID = 1
	} else {
		oppID = 0
	}
	agentPool[ah] = agent.NewAgentState(pid, oppID, uint8(memory_level), uint8(time_decay_turns))
	agentPool[ah].MemoryArchetype = agent.MemoryArchetype(memory_archetype)
	agentPool[ah].MemoryDecayLambda = float32(memory_decay_lambda)
	agentPool[ah].MemoryCapacity = uint8(memory_capacity)
	agentPool[ah].Initialize(&gamePool[game_h])
	agentGameH[ah] = int32(game_h)
	tokenPool[ah].Init(&gamePool[game_h], pid)
	return C.int32_t(ah)
}

//export cambia_agent_apply_decay
func cambia_agent_apply_decay(agent_h C.int32_t, rng_seed C.int64_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	rng := rand.New(rand.NewPCG(uint64(rng_seed), 0))
	agentPool[agent_h].ApplyMemoryDecay(rng)
	return 0
}

//export cambia_agent_free
func cambia_agent_free(h C.int32_t) {
	freeAgent(int32(h))
}

//export cambia_agent_clone
func cambia_agent_clone(h C.int32_t) C.int32_t {
	if h < 0 || h >= maxAgents || !agentInUse[h] {
		return -1
	}
	newH := allocAgent()
	if newH < 0 {
		return -1
	}
	agentPool[newH] = agentPool[h].Clone()
	agentGameH[newH] = agentGameH[h]
	// Carry the token stream (and its snap-phase accumulator) into the clone so a
	// cloned trajectory retains the observer's full perfect-recall history.
	tokenPool[newH] = tokenPool[h]
	return C.int32_t(newH)
}

//export cambia_agent_update
func cambia_agent_update(agent_h C.int32_t, game_h C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	agentPool[agent_h].Update(&gamePool[game_h])
	return 0
}

//export cambia_agent_encode
func cambia_agent_encode(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.InputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].Encode(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

//export cambia_agent_encode_eppbs
func cambia_agent_encode_eppbs(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.EPPBSInputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].EncodeEPPBS(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

//export cambia_agent_encode_eppbs_interleaved
func cambia_agent_encode_eppbs_interleaved(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.EPPBSInputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].EncodeEPPBSInterleaved(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

//export cambia_agent_encode_eppbs_dealiased
func cambia_agent_encode_eppbs_dealiased(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.EPPBSInputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].EncodeEPPBSDealiased(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

//export cambia_agent_encode_eppbs_interleaved_v2
func cambia_agent_encode_eppbs_interleaved_v2(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.EPPBSV2InputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].EncodeEPPBSInterleavedV2(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

// cambia_game_get_all_cards writes a packed byte array of card bucket indices
// for every slot in every player's hand. Layout: player p's slot s at offset
// p*engine.MaxHandSize + s. Empty slots (s >= HandLen) and unrecognized cards
// write the sentinel value 0xFF. Returns the number of bytes written, or -1 on
// error (invalid handle or buf_len too small).
//
// Intended use: training-only omniscient feature extraction (see the Python-side
// compute_omniscient_features helper). Must NOT be called by eval-time code.
//
//export cambia_game_get_all_cards
func cambia_game_get_all_cards(game_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	g := &gamePool[game_h]
	np := int(g.Rules.NumPlayers)
	if np < 2 {
		np = 2
	}
	if np > engine.MaxPlayers {
		np = engine.MaxPlayers
	}
	required := np * int(engine.MaxHandSize)
	if int(buf_len) < required {
		return -1
	}

	outSlice := (*[engine.MaxPlayers * engine.MaxHandSize]C.uint8_t)(unsafe.Pointer(out_buf))
	// Fill the written region with the sentinel first so absent slots are explicit.
	for i := 0; i < required; i++ {
		outSlice[i] = 0xFF
	}
	for p := 0; p < np; p++ {
		ps := &g.Players[p]
		for s := 0; s < int(ps.HandLen) && s < int(engine.MaxHandSize); s++ {
			card := ps.Hand[s]
			if card == engine.EmptyCard {
				continue
			}
			b := agent.CardToBucket(card)
			if b >= agent.BucketUnknown {
				continue
			}
			outSlice[p*int(engine.MaxHandSize)+s] = C.uint8_t(uint8(b))
		}
	}
	return C.int32_t(required)
}

//export cambia_game_num_players
func cambia_game_num_players(h C.int32_t) C.uint8_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	np := gamePool[h].Rules.NumPlayers
	if np < 2 {
		np = 2
	}
	return C.uint8_t(np)
}

//export cambia_game_decision_ctx
func cambia_game_decision_ctx(h C.int32_t) C.uint8_t {
	poolMu.Lock()
	defer poolMu.Unlock()
	ah := int(h)
	if ah < 0 || ah >= maxGames || !gameInUse[ah] {
		return 0
	}
	return C.uint8_t(gamePool[ah].DecisionCtx())
}

//export cambia_agents_update_both
func cambia_agents_update_both(a0h C.int32_t, a1h C.int32_t, gh C.int32_t) C.int32_t {
	poolMu.Lock()
	defer poolMu.Unlock()
	if int(a0h) < 0 || int(a0h) >= maxAgents || !agentInUse[a0h] {
		return -1
	}
	if int(a1h) < 0 || int(a1h) >= maxAgents || !agentInUse[a1h] {
		return -1
	}
	if int(gh) < 0 || int(gh) >= maxGames || !gameInUse[gh] {
		return -1
	}
	agentPool[int(a0h)].Update(&gamePool[int(gh)])
	agentPool[int(a1h)].Update(&gamePool[int(gh)])
	return 0
}

// ---------------------------------------------------------------------------
// Utility exports
// ---------------------------------------------------------------------------

//export cambia_game_turn_number
func cambia_game_turn_number(h C.int32_t) C.uint16_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	return C.uint16_t(gamePool[h].TurnNumber)
}

//export cambia_game_stock_len
func cambia_game_stock_len(h C.int32_t) C.uint8_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	return C.uint8_t(gamePool[h].StockLen)
}

//export cambia_game_get_drawn_card_bucket
func cambia_game_get_drawn_card_bucket(h C.int32_t) C.int8_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	g := &gamePool[h]
	if g.Pending.Type != engine.PendingDiscard {
		return -1 // no drawn card pending
	}
	card := engine.Card(g.Pending.Data[0])
	return C.int8_t(agent.CardToBucket(card))
}

//export cambia_agent_action_mask
func cambia_agent_action_mask(game_h C.int32_t, out *C.uint8_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	mask := gamePool[game_h].LegalActions()
	var boolMask [agent.NumActions]bool
	agent.ActionMask(mask, &boolMask)
	outSlice := (*[agent.NumActions]C.uint8_t)(unsafe.Pointer(out))
	for i := 0; i < agent.NumActions; i++ {
		if boolMask[i] {
			outSlice[i] = 1
		} else {
			outSlice[i] = 0
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// Subgame solver
// ---------------------------------------------------------------------------

//export cambia_subgame_build
func cambia_subgame_build(game_h C.int32_t, max_depth C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	sh := allocSolver()
	if sh < 0 {
		return -1
	}
	root, leafCount := BuildSubgameTree(gamePool[game_h], int(max_depth))
	solverPool[sh].root = root
	solverPool[sh].leafCount = leafCount
	return C.int32_t(sh)
}

//export cambia_subgame_leaf_count
func cambia_subgame_leaf_count(solver_h C.int32_t) C.int32_t {
	if solver_h < 0 || solver_h >= maxSolvers || !solverInUse[solver_h] {
		return -1
	}
	return C.int32_t(solverPool[solver_h].leafCount)
}

//export cambia_subgame_export_leaves
func cambia_subgame_export_leaves(solver_h C.int32_t, game_handles_out *C.int32_t) C.int32_t {
	if solver_h < 0 || solver_h >= maxSolvers || !solverInUse[solver_h] {
		return -1
	}
	entry := &solverPool[solver_h]
	leafStates := CollectLeafStates(entry.root)
	handles := (*[maxGames]C.int32_t)(unsafe.Pointer(game_handles_out))
	for i, state := range leafStates {
		gh := allocGame()
		if gh < 0 {
			// Free previously allocated handles on error.
			for j := 0; j < i; j++ {
				freeGame(int32(handles[j]))
			}
			return -1
		}
		gamePool[gh] = state
		handles[i] = C.int32_t(gh)
	}
	return 0
}

//export cambia_subgame_solve
func cambia_subgame_solve(solver_h C.int32_t, num_iterations C.int32_t, leaf_values *C.float, strategy_out *C.float, root_values_out *C.float) C.int32_t {
	if solver_h < 0 || solver_h >= maxSolvers || !solverInUse[solver_h] {
		return -1
	}
	entry := &solverPool[solver_h]
	leafCount := entry.leafCount

	// Convert leaf_values C array to Go slice (caller provides leafCount*2 floats).
	leafValGo := make([]float32, leafCount*2)
	if leafCount > 0 && leaf_values != nil {
		src := (*[1 << 20]C.float)(unsafe.Pointer(leaf_values))
		for i := 0; i < leafCount*2; i++ {
			leafValGo[i] = float32(src[i])
		}
	}

	// Run CFR iterations.
	var rootValues [2]float32
	for iter := 1; iter <= int(num_iterations); iter++ {
		rootValues = entry.root.CFRIteration([2]float32{1.0, 1.0}, leafValGo, iter)
	}

	// Map root average strategy to full action space (NUM_ACTIONS slots).
	stratOut := (*[agent.NumActions]C.float)(unsafe.Pointer(strategy_out))
	for i := range stratOut {
		stratOut[i] = 0
	}
	rootAvg := entry.root.AverageStrategy()
	for i, child := range entry.root.Children {
		if int(child.ActionIdx) < agent.NumActions {
			stratOut[child.ActionIdx] = C.float(rootAvg[i])
		}
	}

	// Write root values.
	rootOut := (*[2]C.float)(unsafe.Pointer(root_values_out))
	rootOut[0] = C.float(rootValues[0])
	rootOut[1] = C.float(rootValues[1])
	return 0
}

// ---------------------------------------------------------------------------
// N-Player game exports
// ---------------------------------------------------------------------------

//export cambia_game_get_utility_n
func cambia_game_get_utility_n(h C.int32_t, out *C.float, n C.uint8_t) {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return
	}
	u := gamePool[h].GetUtility()
	count := int(n)
	if count > engine.MaxPlayers {
		count = engine.MaxPlayers
	}
	outSlice := (*[engine.MaxPlayers]C.float)(unsafe.Pointer(out))
	for i := 0; i < count; i++ {
		outSlice[i] = C.float(u[i])
	}
}

//export cambia_game_nplayer_legal_actions
func cambia_game_nplayer_legal_actions(h C.int32_t, out *C.uint64_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	mask := gamePool[h].NPlayerLegalActions()
	outSlice := (*[10]C.uint64_t)(unsafe.Pointer(out))
	for i := 0; i < 10; i++ {
		outSlice[i] = C.uint64_t(mask[i])
	}
	return 0
}

//export cambia_game_apply_nplayer_action
func cambia_game_apply_nplayer_action(h C.int32_t, action_idx C.uint16_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	err := gamePool[h].ApplyNPlayerAction(uint16(action_idx))
	if err != nil {
		return -1
	}
	return 0
}

// cambia_game_resolve_untargetable_armed_ability discharges an armed ability that no action in
// the caller's action space can resolve (engine.GameState.ResolveUntargetableArmedAbility). The
// engine refuses every other action while it holds a pending ability, so a caller stuck behind an
// empty legal mask needs this to make progress; n_player_space names which mask the caller was
// driving with when it found the mask empty (0 = the 146-action surface, nonzero = the 452-action
// surface), matching the space whose mask actually stranded it. Was unexported until cambia-1489:
// FFI callers (eval, PPO env, best-response search) had no way to discharge a stranded ability.
// Returns 1 if it resolved something, 0 if the ability still has a legal target or nothing is
// armed, -1 on an invalid handle.
//
//export cambia_game_resolve_untargetable_armed_ability
func cambia_game_resolve_untargetable_armed_ability(h C.int32_t, n_player_space C.uint8_t) C.int32_t {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return -1
	}
	if gamePool[h].ResolveUntargetableArmedAbility(n_player_space != 0) {
		return 1
	}
	return 0
}

// ---------------------------------------------------------------------------
// N-Player agent exports
// ---------------------------------------------------------------------------

//export cambia_agent_new_nplayer
func cambia_agent_new_nplayer(game_h C.int32_t, player_id C.uint8_t, num_players C.uint8_t, memory_level C.uint8_t, time_decay_turns C.uint8_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	ah := allocAgent()
	if ah < 0 {
		return -1
	}
	agentPool[ah] = agent.NewNPlayerAgentState(uint8(player_id), uint8(num_players), uint8(memory_level), uint8(time_decay_turns))
	agentPool[ah].InitializeNPlayer(&gamePool[game_h])
	agentGameH[ah] = int32(game_h)
	return C.int32_t(ah)
}

//export cambia_agent_update_nplayer
func cambia_agent_update_nplayer(agent_h C.int32_t, game_h C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	agentPool[agent_h].UpdateNPlayer(&gamePool[game_h])
	return 0
}

//export cambia_agent_encode_nplayer
func cambia_agent_encode_nplayer(agent_h C.int32_t, decision_ctx C.uint8_t, drawn_bucket C.int8_t, out *C.float) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	outBuf := (*[agent.NPlayerInputDim]float32)(unsafe.Pointer(out))
	agentPool[agent_h].EncodeNPlayer(
		engine.DecisionContext(decision_ctx),
		int8(drawn_bucket),
		outBuf,
	)
	return 0
}

//export cambia_agent_nplayer_action_mask
func cambia_agent_nplayer_action_mask(agent_h C.int32_t, game_h C.int32_t, out *C.uint8_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	mask := gamePool[game_h].NPlayerLegalActions()
	var boolMask [agent.NPlayerNumActions]bool
	agent.NPlayerActionMask(mask, &boolMask)
	outSlice := (*[agent.NPlayerNumActions]C.uint8_t)(unsafe.Pointer(out))
	for i := 0; i < agent.NPlayerNumActions; i++ {
		if boolMask[i] {
			outSlice[i] = 1
		} else {
			outSlice[i] = 0
		}
	}
	return 0
}

// cambia_nplayer_input_dim returns the live Go N-player encoding input
// dimension (agent.NPlayerInputDim). Paired with cfr/src/constants.py's
// N_PLAYER_INPUT_DIM; the FFI dim cross-check test asserts these stay equal
// so bridge.py can single-source its buffer sizes from the Python constant
// instead of a second hand-maintained literal (cambia-542 F8: bridge.py
// previously hardcoded a stale 580 here after a MaxPlayers bump).
//
//export cambia_nplayer_input_dim
func cambia_nplayer_input_dim() C.int32_t {
	return C.int32_t(agent.NPlayerInputDim)
}

// cambia_nplayer_num_actions returns the live Go N-player action-space size
// (engine.NPlayerNumActions). Paired with cfr/src/constants.py's
// N_PLAYER_NUM_ACTIONS (cambia-542 F8).
//
//export cambia_nplayer_num_actions
func cambia_nplayer_num_actions() C.int32_t {
	return C.int32_t(engine.NPlayerNumActions)
}

//export cambia_subgame_solve_ranged
func cambia_subgame_solve_ranged(
	solver_h C.int32_t,
	num_iterations C.int32_t,
	num_hand_types C.int32_t,
	leaf_values *C.float,
	range_p0 *C.float,
	range_p1 *C.float,
	strategy_out *C.float,
	root_cfvs_out *C.float,
) C.int32_t {
	if solver_h < 0 || solver_h >= maxSolvers || !solverInUse[solver_h] {
		return -1
	}
	entry := &solverPool[solver_h]
	leafCount := entry.leafCount
	nht := int(num_hand_types)

	// Convert leaf_values to Go slice: [leafCount * 2 * nht] floats.
	leafValGo := make([]float32, leafCount*2*nht)
	if leafCount > 0 && leaf_values != nil {
		src := (*[1 << 24]C.float)(unsafe.Pointer(leaf_values))
		for i := 0; i < leafCount*2*nht; i++ {
			leafValGo[i] = float32(src[i])
		}
	}

	// Convert range arrays.
	ranges := [2][]float32{
		make([]float32, nht),
		make([]float32, nht),
	}
	if range_p0 != nil {
		src := (*[1 << 16]C.float)(unsafe.Pointer(range_p0))
		for h := 0; h < nht; h++ {
			ranges[0][h] = float32(src[h])
		}
	}
	if range_p1 != nil {
		src := (*[1 << 16]C.float)(unsafe.Pointer(range_p1))
		for h := 0; h < nht; h++ {
			ranges[1][h] = float32(src[h])
		}
	}

	strategy, rootCFVs := SolveSubgameRanged(entry.root, int(num_iterations), leafValGo, ranges, nht)

	// Write strategy to full action space.
	stratOut := (*[agent.NumActions]C.float)(unsafe.Pointer(strategy_out))
	for i := range stratOut {
		stratOut[i] = 0
	}
	for i, child := range entry.root.Children {
		if int(child.ActionIdx) < agent.NumActions && i < len(strategy) {
			stratOut[child.ActionIdx] = C.float(strategy[i])
		}
	}

	// Write root CFVs: [2 * nht] floats - p0 values then p1 values.
	cfvOut := (*[1 << 16]C.float)(unsafe.Pointer(root_cfvs_out))
	for h := 0; h < nht; h++ {
		cfvOut[h] = C.float(rootCFVs[0][h])
		cfvOut[nht+h] = C.float(rootCFVs[1][h])
	}
	return 0
}

//export cambia_game_discard_top
func cambia_game_discard_top(game_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	top := gamePool[game_h].DiscardTop()
	if top == engine.EmptyCard {
		return -1
	}
	return C.int32_t(agent.CardToBucket(top))
}

//export cambia_subgame_free
func cambia_subgame_free(solver_h C.int32_t) {
	freeSolver(int32(solver_h))
}

// ---------------------------------------------------------------------------
// Terminal evaluation exports
// ---------------------------------------------------------------------------

//export cambia_terminal_eval_linear
func cambia_terminal_eval_linear(h C.int32_t, player C.uint8_t) C.float {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	return C.float(gamePool[h].TerminalEvalLinear(uint8(player)))
}

//export cambia_terminal_eval_dp
func cambia_terminal_eval_dp(h C.int32_t, player C.uint8_t) C.float {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	return C.float(gamePool[h].TerminalEvalDP(uint8(player)))
}

//export cambia_terminal_eval_mc
func cambia_terminal_eval_mc(h C.int32_t, player C.uint8_t, num_samples C.int32_t) C.float {
	if h < 0 || h >= maxGames || !gameInUse[h] {
		return 0
	}
	g := &gamePool[h]
	seed := g.GameStateHash()
	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeefcafe1234))
	return C.float(g.TerminalEvalMC(uint8(player), int(num_samples), rng))
}

// ---------------------------------------------------------------------------
// Handle pool diagnostics
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Agent attribute getters (training-only; for action_abstraction in Python)
// ---------------------------------------------------------------------------
//
// These helpers expose minimal AgentState fields required by
// cfr/src/action_abstraction.py when running the DESCA Go FFI env_factory.
// Each getter copies into a caller-owned buffer; the agent state is not
// mutated. Output layout matches the Python AgentState semantics so the
// adapter in cli.py can construct equivalent dict views.

// cambia_agent_get_own_hand fills out_buf with engine.MaxHandSize triplets:
//
//	(bucket, last_seen_turn_lo, last_seen_turn_hi, valid)
//
// where valid is 1 if the slot index is < OwnHandLen, else 0. Each triplet
// is 4 bytes wide, total = engine.MaxHandSize * 4 bytes. Returns 0 on
// success, -1 on bad agent handle.
//
// The Python adapter consumes this to populate `own_hand` as
//
//	{slot: KnownCardInfoLite(bucket, last_seen_turn) for slot in 0..OwnHandLen}
//
// matching action_abstraction.py's expectations.
//
//export cambia_agent_get_own_hand
func cambia_agent_get_own_hand(agent_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	required := engine.MaxHandSize * 4
	if int(buf_len) < required {
		return -1
	}
	a := &agentPool[agent_h]
	outSlice := (*[engine.MaxHandSize * 4]C.uint8_t)(unsafe.Pointer(out_buf))
	for s := 0; s < int(engine.MaxHandSize); s++ {
		base := s * 4
		if uint8(s) < a.OwnHandLen {
			outSlice[base+0] = C.uint8_t(a.OwnHand[s].Bucket)
			outSlice[base+1] = C.uint8_t(a.OwnHand[s].LastSeenTurn & 0xFF)
			outSlice[base+2] = C.uint8_t((a.OwnHand[s].LastSeenTurn >> 8) & 0xFF)
			outSlice[base+3] = 1
		} else {
			outSlice[base+0] = 0
			outSlice[base+1] = 0
			outSlice[base+2] = 0
			outSlice[base+3] = 0
		}
	}
	return 0
}

// cambia_agent_get_opp_belief fills out_buf with engine.MaxHandSize bucket
// values (one byte each) for the opponent's hand slots [0..OppHandLen). Slots
// beyond OppHandLen receive sentinel 0xFF. Decay-encoded beliefs collapse to
// BucketUnknown (sentinel 9) since action_abstraction only checks for
// known/unknown. Returns 0 on success, -1 on bad agent handle.
//
//export cambia_agent_get_opp_belief
func cambia_agent_get_opp_belief(agent_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	required := int(engine.MaxHandSize)
	if int(buf_len) < required {
		return -1
	}
	a := &agentPool[agent_h]
	outSlice := (*[engine.MaxHandSize]C.uint8_t)(unsafe.Pointer(out_buf))
	for s := 0; s < int(engine.MaxHandSize); s++ {
		if uint8(s) < a.OppHandLen {
			bv := a.OppBelief[s]
			if bv.IsBucket() {
				outSlice[s] = C.uint8_t(uint8(bv.Bucket()))
			} else {
				// Decay belief: action_abstraction treats anything non-bucket
				// as unknown. Map to BucketUnknown=9.
				outSlice[s] = C.uint8_t(9)
			}
		} else {
			outSlice[s] = C.uint8_t(0xFF)
		}
	}
	return 0
}

// cambia_agent_get_current_turn returns the agent's CurrentTurn observation
// counter as a uint16. Returns 0xFFFF on bad agent handle (callers must
// validate).
//
//export cambia_agent_get_current_turn
func cambia_agent_get_current_turn(agent_h C.int32_t) C.uint16_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return 0xFFFF
	}
	return C.uint16_t(agentPool[agent_h].CurrentTurn)
}

// cambia_agent_get_hand_lens fills a 2-byte buffer with [OwnHandLen,
// OppHandLen]. Returns 0 on success, -1 on bad agent handle.
//
//export cambia_agent_get_hand_lens
func cambia_agent_get_hand_lens(agent_h C.int32_t, out_buf *C.uint8_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	a := &agentPool[agent_h]
	outSlice := (*[2]C.uint8_t)(unsafe.Pointer(out_buf))
	outSlice[0] = C.uint8_t(a.OwnHandLen)
	outSlice[1] = C.uint8_t(a.OppHandLen)
	return 0
}

// cambia_handle_pool_stats writes the number of in-use slots for games,
// agents, and snapshots into the three output pointers.
// It is thread-safe and uses the existing poolMu mutex.
//
//export cambia_handle_pool_stats
func cambia_handle_pool_stats(games_out *C.int32_t, agents_out *C.int32_t, snaps_out *C.int32_t) {
	poolMu.Lock()
	defer poolMu.Unlock()
	var gCount, aCount, sCount int32
	for i := 0; i < maxGames; i++ {
		if gameInUse[i] {
			gCount++
		}
	}
	for i := 0; i < maxAgents; i++ {
		if agentInUse[i] {
			aCount++
		}
	}
	for i := 0; i < maxSnapshots; i++ {
		if snapInUse[i] {
			sCount++
		}
	}
	*games_out = C.int32_t(gCount)
	*agents_out = C.int32_t(aCount)
	*snaps_out = C.int32_t(sCount)
}

// ---------------------------------------------------------------------------
// PRT-CFR event-stream token FFI (S1W2)
// ---------------------------------------------------------------------------
//
// Each agent carries an append-only int32 token stream that mirrors
// cfr/src/sequence_encoding.py byte-for-byte (the FULL frame body: init-peek
// frames + one frame group per observed action, NO BOS/EOS, NO truncation).
// The stream is seeded at cambia_agent_new* and grown by cambia_games_apply_batch.
// Truncation to a window (SEQ_CAP) and BOS/EOS wrapping are the consumer's job
// at encode time; the Go side never truncates. Overflow past the 4096-token hard
// cap is an explicit error, never silent truncation.

//export cambia_agent_token_len
func cambia_agent_token_len(agent_h C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	return C.int32_t(tokenPool[agent_h].Len())
}

// cambia_agent_tokens copies up to max tokens of the agent's FULL token stream
// into out and returns the number written, or -1 on bad handle.
//
//export cambia_agent_tokens
func cambia_agent_tokens(agent_h C.int32_t, out *C.int32_t, max C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	m := int(max)
	if m <= 0 {
		return 0
	}
	dst := (*[1 << 22]int32)(unsafe.Pointer(out))[:m:m]
	return C.int32_t(tokenPool[agent_h].CopyTo(dst))
}

// cambia_agent_tokens_since copies the incremental tail tokens[since:] (up to
// max) into out and returns the number written, or -1 on bad handle. Used for
// incremental hidden-state carry in the batched GRU inference service.
//
//export cambia_agent_tokens_since
func cambia_agent_tokens_since(agent_h C.int32_t, since C.int32_t, out *C.int32_t, max C.int32_t) C.int32_t {
	if agent_h < 0 || agent_h >= maxAgents || !agentInUse[agent_h] {
		return -1
	}
	m := int(max)
	if m <= 0 {
		return 0
	}
	dst := (*[1 << 22]int32)(unsafe.Pointer(out))[:m:m]
	return C.int32_t(tokenPool[agent_h].CopySince(int32(since), dst))
}

// cambia_set_batch_workers sets the cambia_games_apply_batch fan-out width
// (cambia-656). n<=1 (the default) keeps the serial loop, which is the exact,
// byte-identical code path and error contract described on
// cambia_games_apply_batch. n>1 lets a sufficiently large batch split into up to
// n contiguous chunks applied on separate goroutines (each chunk stays >=
// minBatchChunk games; tiny batches and n=1 applies stay serial regardless).
//
// Per-game state is disjoint by design: each game carries its own RNG (a
// GameState field, not a package global), the token-vocab tables are init-only
// and read-only thereafter, and Update/Observe mutate only the receiver
// AgentState/TokenStream at that game's own handles. The one caller precondition
// is that game and agent handles are disjoint across the batch (the production
// sampler holds this); aliased handles across games would race under n>1.
//
// The knob is global and process-wide; set it once at sampler init. Parallel
// mode is opt-in only: nothing sets it above 1 by default.
//
//export cambia_set_batch_workers
func cambia_set_batch_workers(n C.int32_t) {
	w := int32(n)
	if w < 1 {
		w = 1
	}
	batchWorkers.Store(w)
}

// cambia_games_apply_batch applies one action to each of n games and updates
// that game's two agents (belief state + token stream) in a single FFI call,
// keeping per-call overhead O(n) with no per-game Python roundtrip. game_hs,
// a0s, a1s, actions are length-n arrays. An agent handle of -1 skips that agent.
//
// Returns 0 on success and, on failure:
//
//	-1  invalid game/agent handle or apply error
//	-2  token stream overflow (hard cap exceeded)
//
// Serial mode (batch-workers <= 1, the default): games are applied strictly in
// index order; the first failing game returns immediately and games after it are
// NOT applied (games before it already are).
//
// Parallel mode (batch-workers > 1, set via cambia_set_batch_workers, and only
// when the batch is large enough to split into >=2 chunks of >=minBatchChunk):
// all game/agent handles are validated serially up front, so a bad handle still
// returns -1 at the lowest offending index and applies nothing. After validation
// the batch fans out over contiguous chunks; every valid game is attempted (a
// per-game apply/overflow error in one chunk does NOT stop the others), and the
// return is the error code of the lowest failing game index. This diverges from
// serial prefix semantics - on error, parallel mode may have applied games both
// before and after the reported index - but both error classes are fatal to the
// sampler in practice, which never proceeds past a nonzero return.
//
//export cambia_games_apply_batch
func cambia_games_apply_batch(game_hs *C.int32_t, a0s *C.int32_t, a1s *C.int32_t, actions *C.uint16_t, n C.int32_t) C.int32_t {
	count := int(n)
	if count <= 0 {
		return 0
	}
	ghs := (*[1 << 20]C.int32_t)(unsafe.Pointer(game_hs))[:count:count]
	as0 := (*[1 << 20]C.int32_t)(unsafe.Pointer(a0s))[:count:count]
	as1 := (*[1 << 20]C.int32_t)(unsafe.Pointer(a1s))[:count:count]
	acts := (*[1 << 20]C.uint16_t)(unsafe.Pointer(actions))[:count:count]

	if workers := int(batchWorkers.Load()); workers > 1 && count >= 2*minBatchChunk {
		if maxByChunk := count / minBatchChunk; workers > maxByChunk {
			workers = maxByChunk
		}
		// workers is now >= 2 (count >= 2*minBatchChunk => maxByChunk >= 2).
		return applyBatchParallel(ghs, as0, as1, acts, count, workers)
	}

	// Serial path - byte-identical to the original loop and error contract.
	for i := 0; i < count; i++ {
		gh := int32(ghs[i])
		if gh < 0 || gh >= maxGames || !gameInUse[gh] {
			return -1
		}
		if gamePool[gh].ApplyAction(uint16(acts[i])) != nil {
			return -1
		}
		g := &gamePool[gh]
		for _, ah := range [2]int32{int32(as0[i]), int32(as1[i])} {
			if ah < 0 {
				continue
			}
			if ah >= maxAgents || !agentInUse[ah] {
				return -1
			}
			agentPool[ah].Update(g)
			if tokenPool[ah].Observe(g, agentPool[ah].PlayerID) != nil {
				return -2
			}
		}
	}
	return 0
}

// chunkResult carries a worker chunk's lowest failing game index (idx, -1 if the
// chunk had no failure) and that failure's error code.
type chunkResult struct {
	idx  int
	code int32
}

// applyBatchParallel is the n>1 fan-out path for cambia_games_apply_batch. It
// validates every handle up front, then applies contiguous index chunks on
// separate goroutines. Because per-game state (game RNG, agent belief, token
// stream) lives at disjoint handles, chunks touch disjoint memory and need no
// locking. See cambia_games_apply_batch for the full error contract.
func applyBatchParallel(ghs, as0, as1 []C.int32_t, acts []C.uint16_t, count, workers int) C.int32_t {
	// Validate all handles serially and in index order, so a bad handle returns
	// -1 at the lowest offending index (matching serial's lowest-index report for
	// the handle-error class) before any game is touched.
	for i := 0; i < count; i++ {
		gh := int32(ghs[i])
		if gh < 0 || gh >= maxGames || !gameInUse[gh] {
			return -1
		}
		if a := int32(as0[i]); a >= 0 && (a >= maxAgents || !agentInUse[a]) {
			return -1
		}
		if a := int32(as1[i]); a >= 0 && (a >= maxAgents || !agentInUse[a]) {
			return -1
		}
	}

	chunkSize := (count + workers - 1) / workers
	results := make([]chunkResult, workers)
	var wg sync.WaitGroup
	for c := 0; c < workers; c++ {
		lo := c * chunkSize
		if lo >= count {
			results[c].idx = -1
			continue
		}
		hi := lo + chunkSize
		if hi > count {
			hi = count
		}
		wg.Add(1)
		go func(c, lo, hi int) {
			defer wg.Done()
			res := &results[c]
			res.idx = -1
			for i := lo; i < hi; i++ {
				gh := int32(ghs[i])
				if gamePool[gh].ApplyAction(uint16(acts[i])) != nil {
					if res.idx < 0 {
						res.idx, res.code = i, -1
					}
					continue
				}
				g := &gamePool[gh]
				for _, ah := range [2]int32{int32(as0[i]), int32(as1[i])} {
					if ah < 0 {
						continue
					}
					agentPool[ah].Update(g)
					if tokenPool[ah].Observe(g, agentPool[ah].PlayerID) != nil {
						if res.idx < 0 {
							res.idx, res.code = i, -2
						}
						break
					}
				}
			}
		}(c, lo, hi)
	}
	wg.Wait()

	// Chunks cover ordered, contiguous index ranges, so the first chunk (in chunk
	// order) that recorded a failure holds the globally lowest failing index.
	for c := 0; c < workers; c++ {
		if results[c].idx >= 0 {
			return C.int32_t(results[c].code)
		}
	}
	return 0
}

// cambia_games_observe_batch reads, for each of n live games in ONE FFI
// crossing, the four per-tick quantities a generation sampler needs: terminal
// flag, acting player, legal-action mask, and the acting player's full
// token-stream body. It fuses what were four separate per-game FFI calls
// (cambia_game_is_terminal + cambia_game_acting_player + cambia_agent_action_mask
// + cambia_agent_tokens) into one crossing (X3 ladder step (d1), cambia-607).
// Additive: the four per-game exports stay unchanged.
//
// ADDITIVE PRIMITIVE, not currently wired into the generation hot path (see the
// cambia-607 report): a cross-stream batched observe forces the batched
// scheduler into an observe/query/apply phasing that fragments its single-drain
// inference batch (measured ~5x slower gen), and a per-stream n=1 consumer also
// regressed for an undiagnosed reason on the contended CPU host. This export is
// byte-identity proven (tests/test_prtcfr_ffi_observe.py) and kept for a future,
// correctly-scheduled consumer.
//
// Inputs: game_hs / a0s / a1s are length-n handle arrays (a0s[i]/a1s[i] are
// game i's two agent handles). tok_cap is the TOTAL capacity (int32 elements)
// of the packed token output buffer out_tok.
//
// Outputs (all caller-allocated):
//
//	out_terminal[i]  1 if game i is terminal, else 0
//	out_actor[i]     acting player (0/1) for a live game, 255 if terminal
//	out_masks[i*NumActions + k]  1 if action k is legal for game i, else 0
//	                 (all zero for a terminal game)
//	out_tok[...]     acting-player token bodies PACKED contiguously (game i's
//	                 body at out_tok[out_tok_offsets[i] : +out_tok_lens[i]]);
//	                 total volume is sum(out_tok_lens), not n*cap, so a batch of
//	                 short streams needs only a small buffer
//	out_tok_offsets[i]  start index of game i's body in out_tok
//	out_tok_lens[i]  game i's token-body length (0 for a terminal game)
//
// Returns 0 on success, -1 on an invalid game/agent handle, -2 if the packed
// bodies would exceed tok_cap (the caller grows out_tok and retries; the reads
// are idempotent so a retry recomputes cleanly).
//
//export cambia_games_observe_batch
func cambia_games_observe_batch(
	game_hs *C.int32_t, a0s *C.int32_t, a1s *C.int32_t, n C.int32_t,
	tok_cap C.int32_t,
	out_terminal *C.int8_t,
	out_actor *C.uint8_t,
	out_masks *C.uint8_t,
	out_tok *C.int32_t,
	out_tok_offsets *C.int32_t,
	out_tok_lens *C.int32_t,
) C.int32_t {
	count := int(n)
	if count <= 0 {
		return 0
	}
	capTok := int(tok_cap)
	na := agent.NumActions
	ghs := (*[1 << 20]C.int32_t)(unsafe.Pointer(game_hs))[:count:count]
	as0 := (*[1 << 20]C.int32_t)(unsafe.Pointer(a0s))[:count:count]
	as1 := (*[1 << 20]C.int32_t)(unsafe.Pointer(a1s))[:count:count]
	term := (*[1 << 20]C.int8_t)(unsafe.Pointer(out_terminal))[:count:count]
	actor := (*[1 << 20]C.uint8_t)(unsafe.Pointer(out_actor))[:count:count]
	masks := (*[1 << 27]C.uint8_t)(unsafe.Pointer(out_masks))[: count*na : count*na]
	offsets := (*[1 << 20]C.int32_t)(unsafe.Pointer(out_tok_offsets))[:count:count]
	toklens := (*[1 << 20]C.int32_t)(unsafe.Pointer(out_tok_lens))[:count:count]
	tokBytes := unsafe.Sizeof(C.int32_t(0))

	packOff := 0
	for i := 0; i < count; i++ {
		gh := int32(ghs[i])
		if gh < 0 || gh >= maxGames || !gameInUse[gh] {
			return -1
		}
		g := &gamePool[gh]
		base := i * na
		offsets[i] = C.int32_t(packOff)
		if g.IsTerminal() {
			term[i] = 1
			actor[i] = 255
			toklens[i] = 0
			for k := 0; k < na; k++ {
				masks[base+k] = 0
			}
			continue
		}
		term[i] = 0
		ap := g.ActingPlayer()
		actor[i] = C.uint8_t(ap)
		mask := g.LegalActions()
		var boolMask [agent.NumActions]bool
		agent.ActionMask(mask, &boolMask)
		for k := 0; k < na; k++ {
			if boolMask[k] {
				masks[base+k] = 1
			} else {
				masks[base+k] = 0
			}
		}
		var ah int32
		if ap == 0 {
			ah = int32(as0[i])
		} else {
			ah = int32(as1[i])
		}
		if ah < 0 || ah >= maxAgents || !agentInUse[ah] {
			return -1
		}
		tl := int(tokenPool[ah].Len())
		toklens[i] = C.int32_t(tl)
		if packOff+tl > capTok {
			return -2 // caller grows out_tok and retries
		}
		if tl > 0 {
			dstPtr := unsafe.Pointer(uintptr(unsafe.Pointer(out_tok)) + uintptr(packOff)*tokBytes)
			dst := (*[1 << 22]int32)(dstPtr)[:tl:tl]
			tokenPool[ah].CopyTo(dst)
			packOff += tl
		}
	}
	return 0
}

// cambia_state_save snapshots a complete (game, both agents' belief + token
// state) checkpoint into a new state-snapshot slot and returns its handle, or
// -1 on error. Additive to cambia_game_save (which is game-only and unchanged).
//
//export cambia_state_save
func cambia_state_save(game_h C.int32_t, a0_h C.int32_t, a1_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if a0_h < 0 || a0_h >= maxAgents || !agentInUse[a0_h] {
		return -1
	}
	if a1_h < 0 || a1_h >= maxAgents || !agentInUse[a1_h] {
		return -1
	}
	sh := allocStateSnapshot()
	if sh < 0 {
		return -1
	}
	stateSnapPool[sh].game = gamePool[game_h]
	stateSnapPool[sh].a0 = agentPool[a0_h]
	stateSnapPool[sh].a1 = agentPool[a1_h]
	stateSnapPool[sh].t0 = tokenPool[a0_h]
	stateSnapPool[sh].t1 = tokenPool[a1_h]
	return C.int32_t(sh)
}

// cambia_state_restore restores a (game, both agents' belief + token state)
// checkpoint. The a0_h/a1_h handles must match the ones passed to the paired
// cambia_state_save. Returns 0 on success, -1 on error.
//
//export cambia_state_restore
func cambia_state_restore(game_h C.int32_t, snap_h C.int32_t, a0_h C.int32_t, a1_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if snap_h < 0 || snap_h >= maxSnapshots || !stateSnapInUse[snap_h] {
		return -1
	}
	if a0_h < 0 || a0_h >= maxAgents || !agentInUse[a0_h] {
		return -1
	}
	if a1_h < 0 || a1_h >= maxAgents || !agentInUse[a1_h] {
		return -1
	}
	gamePool[game_h] = stateSnapPool[snap_h].game
	agentPool[a0_h] = stateSnapPool[snap_h].a0
	agentPool[a1_h] = stateSnapPool[snap_h].a1
	tokenPool[a0_h] = stateSnapPool[snap_h].t0
	tokenPool[a1_h] = stateSnapPool[snap_h].t1
	return 0
}

//export cambia_state_snapshot_free
func cambia_state_snapshot_free(h C.int32_t) {
	freeStateSnapshot(int32(h))
}

// cambia_state_clone allocates FRESH game+agent handles and value-copies the
// game state, both agents' belief state, and both agents' token streams into
// them: an INDEPENDENT clone, not a rewind-on-same-handles checkpoint like
// cambia_state_save/restore. This is the primitive the rollout fan-out
// sampler needs (p2-redesign.md: "clone the engine state" at a decision node,
// then apply DIVERGENT playouts on each clone without perturbing the source
// or sibling clones). Aliasing state_save/restore for this would be wrong:
// restore always rewinds the SAME handles, so N fan-out branches would fight
// over one game/agent triple instead of evolving independently.
//
// Writes the new handles to *out_game_h/*out_a0_h/*out_a1_h on success (0).
// On allocation failure (pool exhaustion) returns -1 and frees any handles
// already allocated for this call - never leaks a partial clone; the out
// pointers are left unwritten and must not be read by the caller on error.
//
//export cambia_state_clone
func cambia_state_clone(
	game_h C.int32_t, a0_h C.int32_t, a1_h C.int32_t,
	out_game_h *C.int32_t, out_a0_h *C.int32_t, out_a1_h *C.int32_t,
) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if a0_h < 0 || a0_h >= maxAgents || !agentInUse[a0_h] {
		return -1
	}
	if a1_h < 0 || a1_h >= maxAgents || !agentInUse[a1_h] {
		return -1
	}

	newGH := allocGame()
	if newGH < 0 {
		return -1
	}
	newA0 := allocAgent()
	if newA0 < 0 {
		freeGame(newGH)
		return -1
	}
	newA1 := allocAgent()
	if newA1 < 0 {
		freeGame(newGH)
		freeAgent(newA0)
		return -1
	}

	gamePool[newGH] = gamePool[game_h]
	agentPool[newA0] = agentPool[a0_h]
	agentPool[newA1] = agentPool[a1_h]
	tokenPool[newA0] = tokenPool[a0_h]
	tokenPool[newA1] = tokenPool[a1_h]
	agentGameH[newA0] = newGH
	agentGameH[newA1] = newGH

	*out_game_h = C.int32_t(newGH)
	*out_a0_h = C.int32_t(newA0)
	*out_a1_h = C.int32_t(newA1)
	return 0
}

// cambia_token_vocab writes the token vocabulary layout constants (fixed order,
// agent.TokenVocabFields entries) into out for the Python constants cross-check.
// Returns the count written, or -1 if max is too small.
//
//export cambia_token_vocab
func cambia_token_vocab(out *C.int32_t, max C.int32_t) C.int32_t {
	if int(max) < agent.TokenVocabFields {
		return -1
	}
	dst := (*[agent.TokenVocabFields]int32)(unsafe.Pointer(out))
	return C.int32_t(agent.TokenVocab(dst[:]))
}

// cambia_tokenizer_version returns the tokenizer stream version (agent.
// TokenizerVersion), exported alongside cambia_token_vocab so Python can assert
// Go == sequence_encoding.TOKENIZER_VERSION live (cross-check test) and so the
// tiny-game scorer can refuse a checkpoint recorded under a different version
// (cambia-612). Bumped on every change to the produced token stream.
//
//export cambia_tokenizer_version
func cambia_tokenizer_version() C.int32_t {
	return C.int32_t(agent.TokenizerVersion)
}

// cambia_token_encode_card returns the CARD-block token for a canonical Go card
// index (suit*13+rank; jokers 52/53). For the constants cross-check test.
//
//export cambia_token_encode_card
func cambia_token_encode_card(go_card_index C.uint8_t) C.int32_t {
	return C.int32_t(agent.EncodeCardToken(uint8(go_card_index)))
}

// cambia_token_encode_action returns the ACTION-block token for a 2-player
// action index, or -1 if it does not encode a known action. Cross-check test.
//
//export cambia_token_encode_action
func cambia_token_encode_action(action_idx C.uint16_t) C.int32_t {
	return C.int32_t(agent.EncodeActionToken(uint16(action_idx)))
}

// cambia_token_stream_cap returns the Go per-agent hard token-stream cap
// (agent.MaxTokenStream), so callers can assert live against the paired
// Python constant (cfr/src/cfr/prtcfr_worker.py::PRODUCTION_SEQ_CAP) instead
// of hardcoding either value. Also available as the GO_TOKEN_STREAM_CAP field
// of cambia_token_vocab; this is a direct single-value read for callers that
// only need the cap.
//
//export cambia_token_stream_cap
func cambia_token_stream_cap() C.int32_t {
	return C.int32_t(agent.MaxTokenStream)
}

// ---------------------------------------------------------------------------
// Evaluation surface: read-only game inspection (cambia-1425)
// ---------------------------------------------------------------------------
//
// These exports carry the state an eval-time agent used to read straight off
// the Python reference engine as attributes: get_player_hand, discard_pile /
// get_discard_top, pending_action_data, snap_phase_active /
// snap_discarded_card, and house_rules. Together with the pre-existing
// turn/stock/legal-action exports and the AgentState belief getters they make
// GoEngine + GoAgentState a sufficient surface for an agent, with no Python
// CambiaGameState anywhere in the loop.
//
// Every accessor here is read-only and allocation-free: it copies into a
// caller-owned buffer, never mutates GameState, and holds no Go heap memory
// across the boundary. None of them takes poolMu, matching the other
// single-handle read exports (cambia_game_turn_number and friends).
//
// Cards cross the boundary as canonical card indices, the same encoding
// cambia_game_new_with_deck consumes (suit*13 + rank; suits C=0, D=1, H=2,
// S=3; ranks A=0..K=12; 52 = red joker, 53 = black joker). One card currency
// in both directions. cardIndexNone (0xFF) marks the absence of a card: an
// empty hand slot, or a field that does not apply to the current state.

const (
	// cardIndexNone is the "no card here" marker in every eval-surface buffer.
	// It doubles as the "field does not apply" marker for the non-card slots of
	// the pending and snap records, so a reader has a single sentinel to test.
	cardIndexNone uint8 = 0xFF

	// Fixed byte widths of the packed eval-surface records. Mirrored on the
	// Python side by bridge.py's PENDING_FIELDS / SNAP_FIELDS /
	// HOUSE_RULE_FIELDS; the field-by-field layouts are documented on each
	// export below.
	evalPendingFields   = 10
	evalSnapFields      = 6
	evalHouseRuleFields = 14
)

// cardToIndex is the inverse of indexToCard: it converts a Go Card to the
// canonical card index Python uses. EmptyCard and any malformed card map to
// cardIndexNone.
func cardToIndex(c engine.Card) uint8 {
	if c == engine.EmptyCard {
		return cardIndexNone
	}
	r := c.Rank()
	s := c.Suit()
	if r == engine.RankJoker {
		if s == engine.SuitBlackJoker {
			return 53
		}
		return 52
	}
	if r > engine.RankKing {
		return cardIndexNone
	}
	switch s {
	case engine.SuitClubs:
		return r
	case engine.SuitDiamonds:
		return 13 + r
	case engine.SuitHearts:
		return 26 + r
	case engine.SuitSpades:
		return 39 + r
	}
	return cardIndexNone
}

// boolByte maps a rule flag to its wire byte.
func boolByte(b bool) C.uint8_t {
	if b {
		return 1
	}
	return 0
}

// cambia_game_get_hand fills out_buf with engine.MaxHandSize canonical card
// indices for the given seat's hand, slot 0 first. Slots at or past HandLen
// receive cardIndexNone (0xFF). Returns the seat's hand length, or -1 on
// error (bad handle, seat out of range, or buf_len < engine.MaxHandSize).
//
// This is the seat's true hand: the Go equivalent of the Python reference
// engine's CambiaGameState.get_player_hand, which the perfect-info baselines
// and any best-response search read directly. What a seat *believes* about
// its own or another seat's slots is a different surface, carried by
// AgentState (cambia_agent_get_own_hand / cambia_agent_get_opp_belief);
// imperfect-information agents read that one instead and consult this
// accessor only at the moments the rules reveal a card to them.
//
//export cambia_game_get_hand
func cambia_game_get_hand(game_h C.int32_t, seat C.uint8_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(buf_len) < engine.MaxHandSize {
		return -1
	}
	g := &gamePool[game_h]
	if uint8(seat) >= g.NumActivePlayers() {
		return -1
	}
	ps := &g.Players[uint8(seat)]
	out := (*[engine.MaxHandSize]C.uint8_t)(unsafe.Pointer(out_buf))
	for s := 0; s < engine.MaxHandSize; s++ {
		if s < int(ps.HandLen) {
			out[s] = C.uint8_t(cardToIndex(ps.Hand[s]))
		} else {
			out[s] = C.uint8_t(cardIndexNone)
		}
	}
	return C.int32_t(ps.HandLen)
}

// cambia_game_discard_len returns the number of cards in the discard pile, or
// -1 on a bad handle. Callers that only track pile growth (the discard-memory
// baselines diff this between turns) can poll it without copying the pile.
//
//export cambia_game_discard_len
func cambia_game_discard_len(game_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	return C.int32_t(gamePool[game_h].DiscardLen)
}

// cambia_game_get_discard_pile fills out_buf with the whole discard pile as
// canonical card indices, bottom card first, so the last written byte is the
// top of the pile. Returns the number of bytes written (the pile length), or
// -1 on a bad handle or a buffer shorter than the pile. Size the buffer with
// cambia_game_discard_len, or at engine.MaxDeckSize for a fixed allocation.
//
//export cambia_game_get_discard_pile
func cambia_game_get_discard_pile(game_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	g := &gamePool[game_h]
	n := int(g.DiscardLen)
	if int(buf_len) < n {
		return -1
	}
	if n == 0 {
		return 0
	}
	out := (*[engine.MaxDeckSize]C.uint8_t)(unsafe.Pointer(out_buf))
	for i := 0; i < n; i++ {
		out[i] = C.uint8_t(cardToIndex(g.DiscardPile[i]))
	}
	return C.int32_t(n)
}

// cambia_game_discard_top_card returns the canonical card index of the top
// discard, or -1 if the pile is empty or the handle is bad. Distinct from
// cambia_game_discard_top, which returns the lossy CardBucket: snap matching
// and card-counting need the rank and suit, which the bucket drops.
//
//export cambia_game_discard_top_card
func cambia_game_discard_top_card(game_h C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	top := gamePool[game_h].DiscardTop()
	idx := cardToIndex(top)
	if idx == cardIndexNone {
		return -1
	}
	return C.int32_t(idx)
}

// cambia_game_get_pending fills out_buf with the evalPendingFields-byte
// pending-action record, the Go counterpart of the Python engine's
// pending_action / pending_action_player / pending_action_data trio. Returns
// evalPendingFields on success, or -1 on a bad handle or short buffer.
//
// Layout (every field is cardIndexNone when it does not apply):
//
//	[0] pending type      engine.PendingType (0 = PendingNone)
//	[1] acting seat       the seat that owes the pending decision
//	[2] drawn card        canonical index; PendingDiscard only
//	[3] drawn from        engine.DrawnFromStockpile / DrawnFromDiscard;
//	                      PendingDiscard only
//	[4] own slot          own hand slot under decision; PendingKingDecision
//	[5] target slot       target seat's hand slot; PendingKingDecision and
//	                      PendingSnapMove (the vacated slot to refill)
//	[6] target seat       PendingKingDecision and PendingSnapMove
//	[7] own card          canonical index of the King-looked own card;
//	                      PendingKingDecision only
//	[8] target card       canonical index of the King-looked target card;
//	                      PendingKingDecision at 2 seats only, since the
//	                      N-player King path reuses that Data byte for the
//	                      target seat and never records the card
//	[9] reserved          always 0
//
// The peek and blind-swap pendings (PendingPeekOwn, PendingPeekOther,
// PendingBlindSwap, PendingKingLook) carry no data of their own: the engine
// records only the type and the acting seat, so fields [2..8] stay at the
// sentinel for them.
//
//export cambia_game_get_pending
func cambia_game_get_pending(game_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(buf_len) < evalPendingFields {
		return -1
	}
	g := &gamePool[game_h]
	out := (*[evalPendingFields]C.uint8_t)(unsafe.Pointer(out_buf))
	for i := 0; i < evalPendingFields; i++ {
		out[i] = C.uint8_t(cardIndexNone)
	}
	out[0] = C.uint8_t(uint8(g.Pending.Type))
	out[9] = 0
	if g.Pending.Type == engine.PendingNone {
		return C.int32_t(evalPendingFields)
	}
	out[1] = C.uint8_t(g.Pending.PlayerID)
	switch g.Pending.Type {
	case engine.PendingDiscard:
		out[2] = C.uint8_t(cardToIndex(engine.Card(g.Pending.Data[0])))
		out[3] = C.uint8_t(g.Pending.Data[1])
	case engine.PendingKingDecision:
		out[4] = C.uint8_t(g.Pending.Data[0])
		out[5] = C.uint8_t(g.Pending.Data[1])
		out[7] = C.uint8_t(cardToIndex(engine.Card(g.Pending.Data[2])))
		if g.NumActivePlayers() == 2 {
			out[6] = C.uint8_t(g.OpponentOf(g.Pending.PlayerID))
			out[8] = C.uint8_t(cardToIndex(engine.Card(g.Pending.Data[3])))
		} else {
			out[6] = C.uint8_t(g.Pending.Data[3])
		}
	case engine.PendingSnapMove:
		out[5] = C.uint8_t(g.Pending.Data[1])
		out[6] = C.uint8_t(g.Pending.Data[0])
	}
	return C.int32_t(evalPendingFields)
}

// cambia_game_get_snap_state fills out_buf with the evalSnapFields-byte snap
// window record. Returns evalSnapFields on success, or -1 on a bad handle or
// short buffer.
//
// Layout:
//
//	[0] active            1 while a snap window is open, else 0
//	[1] snapped rank      engine rank of the card that opened the window
//	                      (A=0..K=12, joker=13); cardIndexNone when inactive
//	[2] snapped card      canonical index of the card on top of the discard
//	                      pile; cardIndexNone when inactive or the pile is
//	                      empty. Its rank always equals field [1]. Under a
//	                      multi-snapper window a successful snap pushes its own
//	                      matching card on top, so the suit here is the most
//	                      recently discarded card of the snap rank, not
//	                      necessarily the card that opened the window. Rank is
//	                      what snap legality turns on; treat the suit as
//	                      advisory (it separates a red King from a black one).
//	[3] snapper count     number of eligible snappers in this window, 0 when
//	                      inactive
//	[4] snapper cursor    index into the snapper list of whoever acts next
//	[5] snapper seat      seat at the cursor, cardIndexNone when the window is
//	                      inactive or the cursor has run past the list
//
//export cambia_game_get_snap_state
func cambia_game_get_snap_state(game_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(buf_len) < evalSnapFields {
		return -1
	}
	g := &gamePool[game_h]
	out := (*[evalSnapFields]C.uint8_t)(unsafe.Pointer(out_buf))
	if !g.Snap.Active {
		out[0] = 0
		out[1] = C.uint8_t(cardIndexNone)
		out[2] = C.uint8_t(cardIndexNone)
		out[3] = 0
		out[4] = 0
		out[5] = C.uint8_t(cardIndexNone)
		return C.int32_t(evalSnapFields)
	}
	out[0] = 1
	out[1] = C.uint8_t(g.Snap.DiscardedRank)
	out[2] = C.uint8_t(cardToIndex(g.DiscardTop()))
	out[3] = C.uint8_t(g.Snap.NumSnappers)
	out[4] = C.uint8_t(g.Snap.CurrentSnapperIdx)
	if g.Snap.CurrentSnapperIdx < g.Snap.NumSnappers {
		out[5] = C.uint8_t(g.Snap.Snappers[g.Snap.CurrentSnapperIdx])
	} else {
		out[5] = C.uint8_t(cardIndexNone)
	}
	return C.int32_t(evalSnapFields)
}

// cambia_game_get_house_rules fills out_buf with the evalHouseRuleFields-byte
// rules record. Returns evalHouseRuleFields on success, or -1 on a bad handle
// or short buffer.
//
// Fields follow cambia_game_new_with_rules' parameter order, so the record
// reads back what that constructor was handed:
//
//	[0]  max game turns low byte    (0 = unlimited)
//	[1]  max game turns high byte
//	[2]  cards per player
//	[3]  cambia allowed round
//	[4]  penalty draw count
//	[5]  allow draw from discard    0/1
//	[6]  allow replace abilities    0/1
//	[7]  allow opponent snapping    0/1
//	[8]  snap race                  0/1
//	[9]  number of jokers
//	[10] lock caller hand           0/1
//	[11] number of players          effective count, so a rules struct built
//	                                with the 0 sentinel reads back as 2,
//	                                agreeing with cambia_game_num_players
//	[12] initial view count
//	[13] number of decks
//
//export cambia_game_get_house_rules
func cambia_game_get_house_rules(game_h C.int32_t, out_buf *C.uint8_t, buf_len C.int32_t) C.int32_t {
	if game_h < 0 || game_h >= maxGames || !gameInUse[game_h] {
		return -1
	}
	if int(buf_len) < evalHouseRuleFields {
		return -1
	}
	g := &gamePool[game_h]
	r := &g.Rules
	out := (*[evalHouseRuleFields]C.uint8_t)(unsafe.Pointer(out_buf))
	out[0] = C.uint8_t(uint8(r.MaxGameTurns & 0xFF))
	out[1] = C.uint8_t(uint8(r.MaxGameTurns >> 8))
	out[2] = C.uint8_t(r.CardsPerPlayer)
	out[3] = C.uint8_t(r.CambiaAllowedRound)
	out[4] = C.uint8_t(r.PenaltyDrawCount)
	out[5] = boolByte(r.AllowDrawFromDiscard)
	out[6] = boolByte(r.AllowReplaceAbilities)
	out[7] = boolByte(r.AllowOpponentSnapping)
	out[8] = boolByte(r.SnapRace)
	out[9] = C.uint8_t(r.NumJokers)
	out[10] = boolByte(r.LockCallerHand)
	out[11] = C.uint8_t(g.NumActivePlayers())
	out[12] = C.uint8_t(r.InitialViewCount)
	out[13] = C.uint8_t(r.NumDecks)
	return C.int32_t(evalHouseRuleFields)
}

// ---------------------------------------------------------------------------
// Test-only wrappers for the eval surface
// ---------------------------------------------------------------------------
//
// Same rationale as clone_test_helpers.go: Go forbids `import "C"` in a
// _test.go file, so eval_accessors_test.go cannot name the C types these
// exports take and calls through these plain-Go shims instead. They are
// unexported and carry no //export directive, so they add no symbol to
// libcambia.so's C ABI.

func testGameGetHand(gameH int32, seat uint8) (handLen int32, slots [engine.MaxHandSize]uint8) {
	var buf [engine.MaxHandSize]C.uint8_t
	handLen = int32(cambia_game_get_hand(C.int32_t(gameH), C.uint8_t(seat), &buf[0], engine.MaxHandSize))
	for i := range buf {
		slots[i] = uint8(buf[i])
	}
	return handLen, slots
}

func testGameGetHandShortBuf(gameH int32, seat uint8) int32 {
	var buf [engine.MaxHandSize]C.uint8_t
	return int32(cambia_game_get_hand(C.int32_t(gameH), C.uint8_t(seat), &buf[0], engine.MaxHandSize-1))
}

func testGameDiscardLen(gameH int32) int32 {
	return int32(cambia_game_discard_len(C.int32_t(gameH)))
}

func testGameGetDiscardPile(gameH int32) (n int32, pile []uint8) {
	var buf [engine.MaxDeckSize]C.uint8_t
	n = int32(cambia_game_get_discard_pile(C.int32_t(gameH), &buf[0], engine.MaxDeckSize))
	if n < 0 {
		return n, nil
	}
	pile = make([]uint8, n)
	for i := int32(0); i < n; i++ {
		pile[i] = uint8(buf[i])
	}
	return n, pile
}

func testGameGetDiscardPileShortBuf(gameH int32, bufLen int32) int32 {
	var buf [engine.MaxDeckSize]C.uint8_t
	return int32(cambia_game_get_discard_pile(C.int32_t(gameH), &buf[0], C.int32_t(bufLen)))
}

func testGameDiscardTopCard(gameH int32) int32 {
	return int32(cambia_game_discard_top_card(C.int32_t(gameH)))
}

func testGameGetPending(gameH int32) (rc int32, rec [evalPendingFields]uint8) {
	var buf [evalPendingFields]C.uint8_t
	rc = int32(cambia_game_get_pending(C.int32_t(gameH), &buf[0], evalPendingFields))
	for i := range buf {
		rec[i] = uint8(buf[i])
	}
	return rc, rec
}

func testGameGetSnapState(gameH int32) (rc int32, rec [evalSnapFields]uint8) {
	var buf [evalSnapFields]C.uint8_t
	rc = int32(cambia_game_get_snap_state(C.int32_t(gameH), &buf[0], evalSnapFields))
	for i := range buf {
		rec[i] = uint8(buf[i])
	}
	return rc, rec
}

func testGameGetHouseRules(gameH int32) (rc int32, rec [evalHouseRuleFields]uint8) {
	var buf [evalHouseRuleFields]C.uint8_t
	rc = int32(cambia_game_get_house_rules(C.int32_t(gameH), &buf[0], evalHouseRuleFields))
	for i := range buf {
		rec[i] = uint8(buf[i])
	}
	return rc, rec
}

// testGameNewWithDeck drives cambia_game_new_with_deck from Go tests so a
// scripted game can be dealt from a known deck order.
func testGameNewWithDeck(deck []uint8, numPlayers, cardsPerPlayer, startingPlayer, numJokers, initialViewCount uint8) int32 {
	cdeck := make([]C.uint8_t, len(deck))
	for i, v := range deck {
		cdeck[i] = C.uint8_t(v)
	}
	return int32(cambia_game_new_with_deck(
		&cdeck[0], C.int32_t(len(cdeck)),
		C.uint8_t(numPlayers), C.uint8_t(cardsPerPlayer), C.uint8_t(startingPlayer),
		C.uint16_t(0), C.uint8_t(0), C.uint8_t(2),
		C.uint8_t(1), C.uint8_t(0), C.uint8_t(1),
		C.uint8_t(0), C.uint8_t(numJokers), C.uint8_t(1),
		C.uint8_t(initialViewCount), C.uint8_t(1),
	))
}

// testGameApplyAction drives cambia_game_apply_action from Go tests.
func testGameApplyAction(gameH int32, action uint16) int32 {
	return int32(cambia_game_apply_action(C.int32_t(gameH), C.uint16_t(action)))
}

// testGameNumPlayers drives cambia_game_num_players from Go tests.
func testGameNumPlayers(gameH int32) uint8 {
	return uint8(cambia_game_num_players(C.int32_t(gameH)))
}

// testCardIndexRoundTrip converts an index to a Card and back, for the
// cardToIndex/indexToCard inverse check.
func testCardIndexRoundTrip(idx uint8) uint8 {
	return cardToIndex(indexToCard(idx))
}

// testGameApplyNPlayerAction drives cambia_game_apply_nplayer_action from Go tests.
func testGameApplyNPlayerAction(gameH int32, action uint16) int32 {
	return int32(cambia_game_apply_nplayer_action(C.int32_t(gameH), C.uint16_t(action)))
}

// testGameResolveUntargetableArmedAbility drives
// cambia_game_resolve_untargetable_armed_ability from Go tests.
func testGameResolveUntargetableArmedAbility(gameH int32, nPlayerSpace bool) int32 {
	var np C.uint8_t
	if nPlayerSpace {
		np = 1
	}
	return int32(cambia_game_resolve_untargetable_armed_ability(C.int32_t(gameH), np))
}

// testGameSetPending pokes Pending.Type/PlayerID directly, for constructing a state no live arm
// site produces any more (e.g. an ability armed with no reachable target, cambia-1489) so
// ResolveUntargetableArmedAbility's guard contract for such states stays covered.
func testGameSetPending(gameH int32, pendingType uint8, playerID uint8) {
	if gameH < 0 || gameH >= maxGames || !gameInUse[gameH] {
		return
	}
	gamePool[gameH].Pending.Type = engine.PendingType(pendingType)
	gamePool[gameH].Pending.PlayerID = playerID
}

// testGameSetHandLen pokes Players[seat].HandLen directly, e.g. to empty a seat's hand so a
// targeting predicate has nowhere to reach.
func testGameSetHandLen(gameH int32, seat uint8, handLen uint8) {
	if gameH < 0 || gameH >= maxGames || !gameInUse[gameH] || seat >= engine.MaxPlayers {
		return
	}
	gamePool[gameH].Players[seat].HandLen = handLen
}

// testGameSetHandCard pokes one hand slot directly, so a test can pin every dealt hand away from
// a chosen discard rank and make the resulting snap phase (or lack of one) deterministic instead
// of depending on what a random deal happened to hold.
func testGameSetHandCard(gameH int32, seat uint8, slot uint8, cardIdx uint8) {
	if gameH < 0 || gameH >= maxGames || !gameInUse[gameH] || seat >= engine.MaxPlayers || slot >= engine.MaxHandSize {
		return
	}
	gamePool[gameH].Players[seat].Hand[slot] = indexToCard(cardIdx)
}

// testGamePushDiscard pushes one card onto the discard pile, for setting up the card
// ResolveUntargetableArmedAbility's snap phase resolves against.
func testGamePushDiscard(gameH int32, cardIdx uint8) {
	if gameH < 0 || gameH >= maxGames || !gameInUse[gameH] {
		return
	}
	g := &gamePool[gameH]
	g.DiscardPile[g.DiscardLen] = indexToCard(cardIdx)
	g.DiscardLen++
}

func main() {}
