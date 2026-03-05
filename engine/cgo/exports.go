// Package cgo provides C-exported functions for building libcambia.so.
//
// Build with: go build -buildmode=c-shared -o cfr/libcambia.so ./engine/cgo/
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math/rand/v2"
	"sync"
	"unsafe"

	engine "github.com/jason-s-yu/cambia/engine"
	agent "github.com/jason-s-yu/cambia/engine/agent"
)

// ---------------------------------------------------------------------------
// Handle pools
// ---------------------------------------------------------------------------

const (
	maxGames     = 2048
	maxAgents    = 512
	maxSnapshots = 256
	maxSolvers   = 32
)

// solverEntry holds a built subgame tree and associated metadata.
type solverEntry struct {
	root      *SubgameNode
	leafCount int
}

var (
	poolMu sync.Mutex

	gamePool  [maxGames]engine.GameState
	gameInUse [maxGames]bool

	agentPool  [maxAgents]agent.AgentState
	agentInUse [maxAgents]bool
	agentGameH [maxAgents]int32 // which game handle each agent is associated with

	snapPool  [maxSnapshots]engine.GameState // Snapshot = GameState copy
	snapInUse [maxSnapshots]bool

	solverPool  [maxSolvers]solverEntry
	solverInUse [maxSolvers]bool
)

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
	np := uint8(numPlayers)
	if np < 2 {
		np = 2
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
		NumPlayers:            np,
		InitialViewCount:      uint8(initialViewCount),
		NumDecks:              uint8(numDecks),
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
	np := uint8(numPlayers)
	if np < 2 {
		np = 2
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
		NumPlayers:            np,
		InitialViewCount:      uint8(initialViewCount),
		NumDecks:              uint8(numDecks),
	}
	// Create a base game state with the right rules (deck contents will be overwritten).
	gamePool[h] = engine.NewGame(1, rules)
	g := &gamePool[h]

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

	// Write root CFVs: [2 * nht] floats — p0 values then p1 values.
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

//export cambia_handle_pool_stats
// cambia_handle_pool_stats writes the number of in-use slots for games,
// agents, and snapshots into the three output pointers.
// It is thread-safe and uses the existing poolMu mutex.
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

func main() {}
