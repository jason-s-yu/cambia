// internal/matchmaking/validation_test.go
package matchmaking

import (
	"strings"
	"testing"
)

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

// TestValidatePartyFFA4SpreadRejectsWideGap is the cambia-1041 regression for ValidateParty's
// own spread rule: cambia-966 wired ValidateParty into SearchLobbyHandler but always called it
// with ratings=nil, so len(ratings) >= 2 never held and this branch never ran. ratings is in
// OpenSkill mu units for the ffa4 pool (see ratingSpread's 15.0 limit), not Elo.
func TestValidatePartyFFA4SpreadRejectsWideGap(t *testing.T) {
	err := ValidateParty("ffa4_standard", 2, []float64{10.0, 40.0})
	if err == nil {
		t.Fatalf("expected a 30-mu spread to be rejected for an ffa4 party of 2")
	}
	if !strings.Contains(err.Error(), "spread") {
		t.Fatalf("expected the spread rule named in the error, got %q", err.Error())
	}
}

// TestValidatePartyFFA4SpreadAcceptsCloseGap is the positive pin: a party within the 15-mu limit
// is accepted.
func TestValidatePartyFFA4SpreadAcceptsCloseGap(t *testing.T) {
	if err := ValidateParty("ffa4_standard", 2, []float64{25.0, 30.0}); err != nil {
		t.Fatalf("expected a 5-mu spread to be accepted for an ffa4 party of 2, got %v", err)
	}
}

// TestValidatePartyFFA4SpreadSkippedWithoutRatings pins the pre-cambia-1041 nil-ratings shape:
// with fewer than 2 ratings supplied (nil included), the spread branch does not run at all and
// party size is the only check applied - the same behavior a caller with no rating source
// (matchmaking unavailable, DB unreachable) falls back to.
func TestValidatePartyFFA4SpreadSkippedWithoutRatings(t *testing.T) {
	if err := ValidateParty("ffa4_standard", 2, nil); err != nil {
		t.Fatalf("expected no ratings to skip the spread check rather than reject, got %v", err)
	}
}
