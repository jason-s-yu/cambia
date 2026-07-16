package main

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

// residentBytes reads RSS from /proc/self/statm (field 2, in pages).
func residentBytes(t *testing.T) uint64 {
	t.Helper()
	b, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		t.Skipf("cannot read /proc/self/statm: %v", err)
	}
	fields := strings.Fields(string(b))
	if len(fields) < 2 {
		t.Fatalf("unexpected statm format: %q", string(b))
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		t.Fatalf("parse resident pages: %v", err)
	}
	return pages * uint64(os.Getpagesize())
}

// TestTokenPoolDemandPaged proves the heap-backed tokenPool (3.02 GiB virtual
// reservation at maxAgents=65536) is demand-zero paged, not eagerly resident.
// init() has already run make([]TokenStream, 65536); if that were physically
// resident, RSS would exceed 3 GiB. Assert the whole process stays far under.
func TestTokenPoolDemandPaged(t *testing.T) {
	const virtualReservation = uint64(maxAgents) * 49416 // ~3.02 GiB
	rss := residentBytes(t)
	// Generous bound: real RSS is tens of MiB; anything under 1 GiB proves the
	// 3.02 GiB reservation is not physically backed.
	const bound = uint64(1) << 30
	if rss >= bound {
		t.Fatalf("RSS %d B >= %d B: tokenPool reservation (%d B) appears eagerly resident",
			rss, bound, virtualReservation)
	}
	t.Logf("RSS %d MiB with %d MiB tokenPool reserved (demand-paged)",
		rss>>20, virtualReservation>>20)
}

// countInUseAgents mirrors countInUseGames for the agent pool. Used by the
// ceiling test to confirm pool-stats accounting tracks slots past the old cap.
func countInUseAgents() int {
	poolMu.Lock()
	defer poolMu.Unlock()
	n := 0
	for i := 0; i < maxAgents; i++ {
		if agentInUse[i] {
			n++
		}
	}
	return n
}

// TestHandlePoolCeilingRaised proves the cambia-534 handle-pool ceiling raise:
// the old caps (maxGames=2048, maxAgents=4096) are gone and the pools hand out
// handles well past them. It overshoots the OLD caps by a modest margin (3000
// games, 6000 agents) rather than filling the new 32768/65536 pools; filling
// the agent pool would page in the ~3 GiB tokenPool reservation on purpose.
func TestHandlePoolCeilingRaised(t *testing.T) {
	if maxGames != 32768 {
		t.Fatalf("maxGames = %d, want 32768", maxGames)
	}
	if maxAgents != 65536 {
		t.Fatalf("maxAgents = %d, want 65536", maxAgents)
	}

	const (
		oldMaxGames  = 2048
		oldMaxAgents = 4096
		nGames       = 3000 // > old cap, << new cap
		nAgents      = 6000 // > old cap, << new cap
	)

	gamesBefore := countInUseGames()
	agentsBefore := countInUseAgents()

	games := make([]int32, 0, nGames)
	defer func() {
		for _, h := range games {
			freeGame(h)
		}
	}()
	maxGH := int32(-1)
	for i := 0; i < nGames; i++ {
		h := allocGame()
		if h < 0 {
			t.Fatalf("allocGame returned -1 at i=%d; pool exhausted before %d games", i, nGames)
		}
		games = append(games, h)
		if h > maxGH {
			maxGH = h
		}
	}
	if maxGH < oldMaxGames {
		t.Fatalf("max game handle %d never exceeded old cap %d", maxGH, oldMaxGames)
	}

	agents := make([]int32, 0, nAgents)
	defer func() {
		for _, h := range agents {
			freeAgent(h)
		}
	}()
	maxAH := int32(-1)
	for i := 0; i < nAgents; i++ {
		h := allocAgent()
		if h < 0 {
			t.Fatalf("allocAgent returned -1 at i=%d; pool exhausted before %d agents", i, nAgents)
		}
		agents = append(agents, h)
		if h > maxAH {
			maxAH = h
		}
	}
	if maxAH < oldMaxAgents {
		t.Fatalf("max agent handle %d never exceeded old cap %d", maxAH, oldMaxAgents)
	}

	// Pool-stats accounting (same bool-scan logic as cambia_handle_pool_stats)
	// must count slots past the old range.
	if got, want := countInUseGames()-gamesBefore, nGames; got != want {
		t.Fatalf("in-use game delta = %d, want %d", got, want)
	}
	if got, want := countInUseAgents()-agentsBefore, nAgents; got != want {
		t.Fatalf("in-use agent delta = %d, want %d", got, want)
	}
}
