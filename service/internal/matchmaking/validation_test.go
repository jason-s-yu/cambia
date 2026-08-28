// internal/matchmaking/validation_test.go
package matchmaking

import "testing"

// TestQueueConfigsOrderIsFixed pins the Order assignment GET /matchmaking/queues sorts on
// (cambia-957). A caller ranging over QueueConfigs directly still gets Go's randomized map
// iteration order; Order is what makes a rendered queue list stable across process runs, so a
// silent edit here would silently reopen the flicker the ticket fixed.
func TestQueueConfigsOrderIsFixed(t *testing.T) {
	want := map[string]int{
		"h2h_quickplay":  10,
		"h2h_blitz":      20,
		"h2h_rapid":      30,
		"h2h_classical":  40,
		"ffa4_standard":  50,
		"ffa4_classical": 60,
	}

	if len(QueueConfigs) != len(want) {
		t.Fatalf("QueueConfigs has %d entries, want %d: %v", len(QueueConfigs), len(want), QueueConfigs)
	}

	for id, wantOrder := range want {
		cfg, ok := QueueConfigs[id]
		if !ok {
			t.Fatalf("QueueConfigs is missing queue %q", id)
		}
		if cfg.Order != wantOrder {
			t.Errorf("QueueConfigs[%q].Order = %d, want %d", id, cfg.Order, wantOrder)
		}
	}
}

// TestQueueConfigsOrderIsUnique guards the handler's Order-then-QueueID sort: a duplicate Order
// value still sorts deterministically (QueueID breaks the tie), but a duplicate was never part
// of the cambia-957 design and would mean two queue cards claim the same rank.
func TestQueueConfigsOrderIsUnique(t *testing.T) {
	seen := make(map[int]string, len(QueueConfigs))
	for id, cfg := range QueueConfigs {
		if other, dup := seen[cfg.Order]; dup {
			t.Errorf("queues %q and %q share Order %d", id, other, cfg.Order)
		}
		seen[cfg.Order] = id
	}
}
