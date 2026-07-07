package main

import (
	"testing"
)

// newCloneTestTriple creates a fresh game + two agents via the real exported
// entry points (through the plain-Go testGameNew/testAgentNew wrappers in
// clone_test_helpers.go -- Go does not permit `import "C"` in _test.go files,
// so the C ABI calls live there instead) and returns their handles.
func newCloneTestTriple(t *testing.T, seed uint64) (gh, a0h, a1h int32) {
	t.Helper()
	g := testGameNew(seed)
	if g < 0 {
		t.Fatalf("cambia_game_new failed: %d", g)
	}
	a0 := testAgentNew(g, 0, 0, 0)
	if a0 < 0 {
		t.Fatalf("cambia_agent_new(player 0) failed: %d", a0)
	}
	a1 := testAgentNew(g, 1, 0, 0)
	if a1 < 0 {
		t.Fatalf("cambia_agent_new(player 1) failed: %d", a1)
	}
	return g, a0, a1
}

func firstLegalAction(t *testing.T, gh int32) uint16 {
	t.Helper()
	mask := testGameLegalActionsMask(gh)
	for w := 0; w < 3; w++ {
		word := mask[w]
		if word == 0 {
			continue
		}
		for bit := 0; bit < 64; bit++ {
			if word&(1<<uint(bit)) != 0 {
				return uint16(w*64 + bit)
			}
		}
	}
	t.Fatalf("no legal actions on handle %d", gh)
	return 0
}

func applyOne(t *testing.T, gh, a0h, a1h int32) uint16 {
	t.Helper()
	action := firstLegalAction(t, gh)
	if ret := testApplyOne(gh, a0h, a1h, action); ret != 0 {
		t.Fatalf("cambia_games_apply_batch failed: %d", ret)
	}
	return action
}

func tokensOf(t *testing.T, ah int32) []int32 {
	t.Helper()
	return testAgentTokens(ah)
}

func equalTokens(a, b []int32) bool {
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

func freeTriple(gh, a0h, a1h int32) {
	testAgentFree(a0h)
	testAgentFree(a1h)
	testGameFree(gh)
}

// ---------------------------------------------------------------------------
// cambia_state_clone
// ---------------------------------------------------------------------------

// TestCambiaStateCloneIndependence asserts that applying divergent actions to
// a clone never mutates the source (token streams AND game state), and vice
// versa: the clone must be on genuinely fresh handles, not aliases.
func TestCambiaStateCloneIndependence(t *testing.T) {
	gh, a0h, a1h := newCloneTestTriple(t, 42)
	defer freeTriple(gh, a0h, a1h)

	// Advance the source a few steps before cloning.
	for i := 0; i < 5; i++ {
		applyOne(t, gh, a0h, a1h)
	}
	preCloneSrcTurn := testGameTurnNumber(gh)
	preCloneSrcTokens0 := tokensOf(t, a0h)
	preCloneSrcTokens1 := tokensOf(t, a1h)

	ok, cloneGh, cloneA0, cloneA1 := testStateClone(gh, a0h, a1h)
	if !ok {
		t.Fatalf("cambia_state_clone failed")
	}
	defer freeTriple(cloneGh, cloneA0, cloneA1)

	if cloneGh == gh || cloneA0 == a0h || cloneA1 == a1h {
		t.Fatalf("clone reused a source handle: game %d/%d agents %d/%d %d/%d",
			gh, cloneGh, a0h, cloneA0, a1h, cloneA1)
	}

	// Immediately after clone, both sides must agree byte-for-byte.
	if testGameTurnNumber(cloneGh) != preCloneSrcTurn {
		t.Fatalf("clone turn mismatch at clone time")
	}
	if !equalTokens(tokensOf(t, cloneA0), preCloneSrcTokens0) {
		t.Fatalf("clone a0 tokens mismatch at clone time")
	}
	if !equalTokens(tokensOf(t, cloneA1), preCloneSrcTokens1) {
		t.Fatalf("clone a1 tokens mismatch at clone time")
	}

	// Diverge: apply on the CLONE only. Source must stay byte-stable.
	for i := 0; i < 5; i++ {
		if testGameIsTerminal(cloneGh) {
			break
		}
		applyOne(t, cloneGh, cloneA0, cloneA1)
	}
	if testGameTurnNumber(gh) != preCloneSrcTurn {
		t.Fatalf("source turn mutated by clone-side apply")
	}
	if !equalTokens(tokensOf(t, a0h), preCloneSrcTokens0) {
		t.Fatalf("source a0 tokens mutated by clone-side apply")
	}
	if !equalTokens(tokensOf(t, a1h), preCloneSrcTokens1) {
		t.Fatalf("source a1 tokens mutated by clone-side apply")
	}

	// Now diverge the SOURCE further and verify the clone (already advanced
	// differently above) is unaffected by source-side apply.
	cloneTurnBefore := testGameTurnNumber(cloneGh)
	cloneTokensBefore0 := tokensOf(t, cloneA0)
	for i := 0; i < 3; i++ {
		if testGameIsTerminal(gh) {
			break
		}
		applyOne(t, gh, a0h, a1h)
	}
	if testGameTurnNumber(cloneGh) != cloneTurnBefore {
		t.Fatalf("clone turn mutated by source-side apply")
	}
	if !equalTokens(tokensOf(t, cloneA0), cloneTokensBefore0) {
		t.Fatalf("clone a0 tokens mutated by source-side apply")
	}
}

// TestCambiaStateCloneOfClone asserts a clone-of-a-clone is itself an
// independent third instance (chained fan-out).
func TestCambiaStateCloneOfClone(t *testing.T) {
	gh, a0h, a1h := newCloneTestTriple(t, 7)
	defer freeTriple(gh, a0h, a1h)
	for i := 0; i < 3; i++ {
		applyOne(t, gh, a0h, a1h)
	}

	ok1, clone1G, clone1A0, clone1A1 := testStateClone(gh, a0h, a1h)
	if !ok1 {
		t.Fatalf("first clone failed")
	}
	defer freeTriple(clone1G, clone1A0, clone1A1)

	for i := 0; i < 2; i++ {
		applyOne(t, clone1G, clone1A0, clone1A1)
	}

	ok2, clone2G, clone2A0, clone2A1 := testStateClone(clone1G, clone1A0, clone1A1)
	if !ok2 {
		t.Fatalf("clone-of-clone failed")
	}
	defer freeTriple(clone2G, clone2A0, clone2A1)

	if clone2G == clone1G || clone2G == gh || clone2A0 == clone1A0 || clone2A0 == a0h {
		t.Fatalf("clone-of-clone reused an ancestor handle")
	}
	if testGameTurnNumber(clone2G) != testGameTurnNumber(clone1G) {
		t.Fatalf("clone-of-clone turn mismatch at clone time")
	}
	if !equalTokens(tokensOf(t, clone2A0), tokensOf(t, clone1A0)) {
		t.Fatalf("clone-of-clone a0 tokens mismatch at clone time")
	}

	// Diverge the grandchild; parent and grandparent (clone1, source) unaffected.
	clone1Turn := testGameTurnNumber(clone1G)
	srcTurn := testGameTurnNumber(gh)
	for i := 0; i < 3; i++ {
		if testGameIsTerminal(clone2G) {
			break
		}
		applyOne(t, clone2G, clone2A0, clone2A1)
	}
	if testGameTurnNumber(clone1G) != clone1Turn {
		t.Fatalf("parent clone mutated by grandchild apply")
	}
	if testGameTurnNumber(gh) != srcTurn {
		t.Fatalf("source mutated by grandchild apply")
	}
}

// TestCambiaStateClonePoolExhaustion asserts pool exhaustion returns an error
// and leaks no partially-allocated handles (game pool usage is restored to
// its pre-call level after a failed clone).
func TestCambiaStateClonePoolExhaustion(t *testing.T) {
	gh, a0h, a1h := newCloneTestTriple(t, 99)
	defer freeTriple(gh, a0h, a1h)

	// Exhaust the GAME pool via the internal allocator (same package), leaving
	// zero games free.
	var filled []int32
	for {
		h := allocGame()
		if h < 0 {
			break
		}
		filled = append(filled, h)
	}
	defer func() {
		for _, h := range filled {
			freeGame(h)
		}
	}()

	if ok, _, _, _ := testStateClone(gh, a0h, a1h); ok {
		t.Fatalf("expected clone to fail on game-pool exhaustion, but it succeeded")
	}

	// Release exactly one game slot, then exhaust the AGENT pool instead and
	// confirm the game allocated during that attempt is freed on failure (no
	// leak): pool usage before/after must be identical.
	freeGame(filled[len(filled)-1])
	filled = filled[:len(filled)-1]

	var agentsFilled []int32
	for {
		h := allocAgent()
		if h < 0 {
			break
		}
		agentsFilled = append(agentsFilled, h)
	}
	defer func() {
		for _, h := range agentsFilled {
			freeAgent(h)
		}
	}()

	gamesBefore := countInUseGames()
	if ok, _, _, _ := testStateClone(gh, a0h, a1h); ok {
		t.Fatalf("expected clone to fail on agent-pool exhaustion, but it succeeded")
	}
	gamesAfter := countInUseGames()
	if gamesAfter != gamesBefore {
		t.Fatalf("agent-pool-exhaustion failure leaked a game handle: before=%d after=%d", gamesBefore, gamesAfter)
	}
}

func countInUseGames() int {
	poolMu.Lock()
	defer poolMu.Unlock()
	n := 0
	for i := 0; i < maxGames; i++ {
		if gameInUse[i] {
			n++
		}
	}
	return n
}

// TestCambiaStateCloneRolloutFanOutShaped mirrors the sampler's actual usage
// shape: one source state, many independent clones fanning out from it with
// divergent playouts, source byte-stable throughout.
func TestCambiaStateCloneRolloutFanOutShaped(t *testing.T) {
	gh, a0h, a1h := newCloneTestTriple(t, 2024)
	defer freeTriple(gh, a0h, a1h)
	for i := 0; i < 4; i++ {
		applyOne(t, gh, a0h, a1h)
	}
	srcTurn := testGameTurnNumber(gh)
	srcTokens0 := tokensOf(t, a0h)
	srcTokens1 := tokensOf(t, a1h)

	const nClones = 20
	type triple struct{ g, a0, a1 int32 }
	clones := make([]triple, 0, nClones)
	defer func() {
		for _, c := range clones {
			freeTriple(c.g, c.a0, c.a1)
		}
	}()

	for i := 0; i < nClones; i++ {
		ok, cg, ca0, ca1 := testStateClone(gh, a0h, a1h)
		if !ok {
			t.Fatalf("clone %d failed", i)
		}
		clones = append(clones, triple{cg, ca0, ca1})
	}

	// Divergent playouts: each clone applies a different number of steps.
	for i, c := range clones {
		steps := (i % 5) + 1
		for s := 0; s < steps; s++ {
			if testGameIsTerminal(c.g) {
				break
			}
			applyOne(t, c.g, c.a0, c.a1)
		}
	}

	// Source must be byte-stable after all 20 clones diverged.
	if testGameTurnNumber(gh) != srcTurn {
		t.Fatalf("source turn mutated by fan-out playouts")
	}
	if !equalTokens(tokensOf(t, a0h), srcTokens0) {
		t.Fatalf("source a0 tokens mutated by fan-out playouts")
	}
	if !equalTokens(tokensOf(t, a1h), srcTokens1) {
		t.Fatalf("source a1 tokens mutated by fan-out playouts")
	}

	// Clones must have genuinely diverged from each other (not all identical),
	// confirming independence rather than accidental aliasing to one buffer.
	distinctCount := 0
	seen := make([][]int32, 0, nClones)
	for _, c := range clones {
		toks := tokensOf(t, c.a0)
		isNew := true
		for _, prev := range seen {
			if equalTokens(prev, toks) {
				isNew = false
				break
			}
		}
		if isNew {
			distinctCount++
			seen = append(seen, toks)
		}
	}
	if distinctCount < 2 {
		t.Fatalf("expected clones to diverge from each other, got %d distinct token streams", distinctCount)
	}
}
