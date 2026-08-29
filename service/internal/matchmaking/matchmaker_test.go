// internal/matchmaking/matchmaker_test.go
package matchmaking

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

// queuedSolo builds a solo party entry for the two-player quickplay queue.
func queuedSolo(lobbyID uuid.UUID, queuedAt time.Time) *QueuedLobby {
	return &QueuedLobby{
		LobbyID:     lobbyID,
		PlayerCount: 1,
		QueueID:     "h2h_quickplay",
		TargetCount: 2,
		IsRanked:    true,
		QueuedAt:    queuedAt,
	}
}

// TestDormantPartyIsNotMatched is the cambia-933 F1 regression: a party whose members all closed
// their tabs must not be paired with a live one. Pairing it hands the live player a ready check
// the absent side can never answer, and nothing times that check out.
func TestDormantPartyIsNotMatched(t *testing.T) {
	m := NewMatchmaker()
	dormant := uuid.New()
	live := uuid.New()

	var results []MatchResult
	m.OnMatchFormed = func(r MatchResult) { results = append(results, r) }
	m.PartyLive = func(id uuid.UUID) bool { return id != dormant }

	now := time.Now()
	if err := m.Enqueue(queuedSolo(dormant, now.Add(-time.Minute))); err != nil {
		t.Fatalf("enqueue dormant: %v", err)
	}
	if err := m.Enqueue(queuedSolo(live, now.Add(-30*time.Second))); err != nil {
		t.Fatalf("enqueue live: %v", err)
	}

	m.processQueues()

	if len(results) != 0 {
		t.Fatalf("expected no match against a dormant party, got %d", len(results))
	}
	// Both keep their place in line: a dropped socket is not a cancelled search, and the hub's
	// idle window is what releases a lobby nobody comes back to.
	if got := m.QueueStats()["h2h_quickplay"].PlayerCount; got != 2 {
		t.Fatalf("expected both parties still queued, got playerCount %d", got)
	}
}

// TestDormantPartyMatchesOnceItReconnects pins the other half of the gate: holding a party back
// is temporary, so a player who refreshed the page is matched on the next tick.
func TestDormantPartyMatchesOnceItReconnects(t *testing.T) {
	m := NewMatchmaker()
	reconnecting := uuid.New()
	other := uuid.New()
	connected := false

	var results []MatchResult
	m.OnMatchFormed = func(r MatchResult) { results = append(results, r) }
	m.PartyLive = func(id uuid.UUID) bool { return id != reconnecting || connected }

	now := time.Now()
	if err := m.Enqueue(queuedSolo(reconnecting, now.Add(-time.Minute))); err != nil {
		t.Fatalf("enqueue reconnecting: %v", err)
	}
	if err := m.Enqueue(queuedSolo(other, now.Add(-30*time.Second))); err != nil {
		t.Fatalf("enqueue other: %v", err)
	}

	m.processQueues()
	if len(results) != 0 {
		t.Fatalf("expected no match while the party was dormant, got %d", len(results))
	}

	connected = true
	m.processQueues()

	if len(results) != 1 {
		t.Fatalf("expected one match after the party reconnected, got %d", len(results))
	}
	got := results[0]
	if got.HostLobbyID != reconnecting {
		t.Fatalf("expected the oldest party to host, got %s", got.HostLobbyID)
	}
	if len(got.Parties) != 2 {
		t.Fatalf("expected 2 parties in the result, got %d", len(got.Parties))
	}
	if got.TargetCount != 2 {
		t.Fatalf("expected TargetCount 2, got %d", got.TargetCount)
	}
	// The result carries whole queue entries so an abandoned match can be requeued as it stood.
	for _, p := range got.Parties {
		if p.QueuedAt.IsZero() || p.PlayerCount != 1 || p.QueueID != "h2h_quickplay" {
			t.Fatalf("party entry lost its queue fields: %+v", p)
		}
	}
	if got := m.QueueStats()["h2h_quickplay"].PlayerCount; got != 0 {
		t.Fatalf("expected the queue drained after the match, got playerCount %d", got)
	}
}

// TestNilPartyLiveMatchesEverything keeps the predicate optional: a Matchmaker with no liveness
// source behaves exactly as before.
func TestNilPartyLiveMatchesEverything(t *testing.T) {
	m := NewMatchmaker()
	var results []MatchResult
	m.OnMatchFormed = func(r MatchResult) { results = append(results, r) }

	now := time.Now()
	if err := m.Enqueue(queuedSolo(uuid.New(), now.Add(-time.Minute))); err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	if err := m.Enqueue(queuedSolo(uuid.New(), now.Add(-30*time.Second))); err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	m.processQueues()

	if len(results) != 1 {
		t.Fatalf("expected one match with no liveness predicate, got %d", len(results))
	}
}

// queuedRanked builds a ranked H2H solo entry carrying AvgRating/MaxRD, the fields cambia-1041
// found with no writer anywhere: every QueuedLobby read AvgRating=0, MaxRD=0 regardless of the
// party's real rating, which made glicko2Quality's spread term zero for every pairing and the
// ranked quality gate pass everyone.
func queuedRanked(lobbyID uuid.UUID, queuedAt time.Time, avgRating, maxRD float64) *QueuedLobby {
	e := queuedSolo(lobbyID, queuedAt)
	e.AvgRating = avgRating
	e.MaxRD = maxRD
	return e
}

// TestGlicko2QualityWideSpreadIsLow pins the quality-gate math directly: a 1200-point rating gap
// between two well-established players (low RD, so the c-term dominates less) must fall well
// below the 0.80 threshold an under-30-second wait requires (minQuality).
func TestGlicko2QualityWideSpreadIsLow(t *testing.T) {
	a := &QueuedLobby{AvgRating: 1000, MaxRD: 60}
	b := &QueuedLobby{AvgRating: 2200, MaxRD: 60}
	if q := glicko2Quality(a, b); q >= 0.80 {
		t.Fatalf("expected quality below 0.80 for a 1200-point spread, got %v", q)
	}
}

// TestGlicko2QualityCloseSpreadIsHigh is the positive pin: a 30-point gap between the same two
// well-established players clears the 0.80 immediate-match threshold easily.
func TestGlicko2QualityCloseSpreadIsHigh(t *testing.T) {
	a := &QueuedLobby{AvgRating: 1500, MaxRD: 60}
	b := &QueuedLobby{AvgRating: 1530, MaxRD: 60}
	if q := glicko2Quality(a, b); q < 0.80 {
		t.Fatalf("expected quality at or above 0.80 for a 30-point spread, got %v", q)
	}
}

// TestRankedH2HQualityGateRefusesWideSpread is the cambia-1041 end-to-end regression: with
// AvgRating/MaxRD actually populated, a wide-spread ranked H2H pairing must not be formed on the
// first tick (wait < 30s, minQuality 0.80). Before the fix both entries read AvgRating=0 and
// this pairing matched immediately regardless of spread.
func TestRankedH2HQualityGateRefusesWideSpread(t *testing.T) {
	m := NewMatchmaker()
	var results []MatchResult
	m.OnMatchFormed = func(r MatchResult) { results = append(results, r) }

	now := time.Now()
	if err := m.Enqueue(queuedRanked(uuid.New(), now, 1000, 60)); err != nil {
		t.Fatalf("enqueue low: %v", err)
	}
	if err := m.Enqueue(queuedRanked(uuid.New(), now, 2200, 60)); err != nil {
		t.Fatalf("enqueue high: %v", err)
	}

	m.processQueues()

	if len(results) != 0 {
		t.Fatalf("expected the wide rating spread to be refused, got %d matches: %+v", len(results), results)
	}
	if got := m.QueueStats()["h2h_quickplay"].PlayerCount; got != 2 {
		t.Fatalf("expected both parties to remain queued, got playerCount %d", got)
	}
}

// TestRankedH2HQualityGateAcceptsCloseSpread is the positive pin: a close-rating ranked H2H pair
// matches on the first tick.
func TestRankedH2HQualityGateAcceptsCloseSpread(t *testing.T) {
	m := NewMatchmaker()
	var results []MatchResult
	m.OnMatchFormed = func(r MatchResult) { results = append(results, r) }

	now := time.Now()
	if err := m.Enqueue(queuedRanked(uuid.New(), now, 1500, 60)); err != nil {
		t.Fatalf("enqueue a: %v", err)
	}
	if err := m.Enqueue(queuedRanked(uuid.New(), now, 1530, 60)); err != nil {
		t.Fatalf("enqueue b: %v", err)
	}

	m.processQueues()

	if len(results) != 1 {
		t.Fatalf("expected the close rating spread to match immediately, got %d matches", len(results))
	}
}
