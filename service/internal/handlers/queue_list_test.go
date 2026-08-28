// internal/handlers/queue_list_test.go
//
// GET /matchmaking/queues ordering (cambia-957). ListQueuesHandler used to range over
// matchmaking.QueueConfigs directly, and Go randomizes map iteration order per process run, so
// the six queue cards on the dashboard reordered themselves between loads with nothing actually
// changed. The handler now sorts by matchmaking.QueueConfig.Order (ties broken by QueueID)
// before encoding the response; these tests pin that ordering rather than trusting it stays
// implicit.
package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"testing"

	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// queueListEntry mirrors the subset of ListQueuesHandler's response this test cares about.
type queueListEntry struct {
	QueueID string `json:"queueId"`
}

// listQueues drives ListQueuesHandler and returns the decoded queue id sequence, in response order.
func listQueues(t *testing.T, gs *GameServer) []string {
	t.Helper()
	req := httptest.NewRequest("GET", "/matchmaking/queues", nil)
	w := httptest.NewRecorder()
	ListQueuesHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /matchmaking/queues, got %d: %s", w.Code, w.Body.String())
	}
	var entries []queueListEntry
	if err := json.Unmarshal(w.Body.Bytes(), &entries); err != nil {
		t.Fatalf("failed to decode queue list: %v", err)
	}
	ids := make([]string, len(entries))
	for i, e := range entries {
		ids[i] = e.QueueID
	}
	return ids
}

// expectedQueueOrder sorts matchmaking.QueueConfigs by Order (ties on QueueID), independent of
// the handler's own sort, so the test fails if the two diverge rather than trivially agreeing
// with itself.
func expectedQueueOrder(t *testing.T) []string {
	t.Helper()
	ids := make([]string, 0, len(matchmaking.QueueConfigs))
	for id := range matchmaking.QueueConfigs {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		oi, oj := matchmaking.QueueConfigs[ids[i]].Order, matchmaking.QueueConfigs[ids[j]].Order
		if oi != oj {
			return oi < oj
		}
		return ids[i] < ids[j]
	})
	return ids
}

// TestListQueuesOrderIsDeterministicAcrossCalls asserts two consecutive calls return the
// identical sequence. Before cambia-957 this was flaky by construction: map iteration order is
// randomized per Go process, so a run could pass or fail depending on the runtime's internal
// hash seed rather than on any code defect, which is exactly the bug report (the dashboard
// visibly reordering between loads with no state change).
func TestListQueuesOrderIsDeterministicAcrossCalls(t *testing.T) {
	gs := NewGameServer()

	first := listQueues(t, gs)
	for i := 0; i < 5; i++ {
		next := listQueues(t, gs)
		if len(next) != len(first) {
			t.Fatalf("call %d: got %d queues, first call returned %d", i, len(next), len(first))
		}
		for j := range first {
			if next[j] != first[j] {
				t.Fatalf("call %d diverged from call 0 at index %d: got %q, want %q (full: %v vs %v)",
					i, j, next[j], first[j], next, first)
			}
		}
	}
}

// TestListQueuesOrderMatchesConfiguredOrder asserts the response sequence matches the Order
// values on matchmaking.QueueConfigs, both generically (sorted independently in this test) and
// against the fixed cambia-957 assignment (h2h_quickplay=10, h2h_blitz=20, h2h_rapid=30,
// h2h_classical=40, ffa4_standard=50, ffa4_classical=60), so a future edit that reassigns an
// Order value without updating the ticket's intended ordering is caught here too.
func TestListQueuesOrderMatchesConfiguredOrder(t *testing.T) {
	gs := NewGameServer()

	got := listQueues(t, gs)
	want := expectedQueueOrder(t)
	if len(got) != len(want) {
		t.Fatalf("got %d queues, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %q, want %q (full: got %v, want %v)", i, got[i], want[i], got, want)
		}
	}

	fixedOrder := []string{
		"h2h_quickplay",
		"h2h_blitz",
		"h2h_rapid",
		"h2h_classical",
		"ffa4_standard",
		"ffa4_classical",
	}
	if len(got) != len(fixedOrder) {
		t.Fatalf("got %d queues, want %d fixed entries: %v", len(got), len(fixedOrder), got)
	}
	for i := range fixedOrder {
		if got[i] != fixedOrder[i] {
			t.Fatalf("index %d: got %q, want %q per cambia-957 Order assignment (full: %v)", i, got[i], fixedOrder[i], got)
		}
	}
}
