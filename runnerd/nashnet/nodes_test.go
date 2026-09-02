package nashnet

import (
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func newRegistry(t *testing.T, clk *fakeClock) *NodeRegistry {
	t.Helper()
	return NewNodeRegistry(RegistryConfig{Now: clk.Now})
}

func TestRegisterUpsertsTheRecordAndBumpsTheEpoch(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := newRegistry(t, clk)

	first, err := r.Register("node-a", RegisterRequest{
		AgentVersion: "1.1.0",
		PlatformTag:  "linux-x86_64",
		Slots:        2,
		Kinds:        []string{"train", "evaluate"},
		Capabilities: json.RawMessage(`{"schema":1,"slots":2}`),
		GateReport:   json.RawMessage(`{"admit":true,"slots_offered":2}`),
		HaveCommits:  []string{"0123456789abcdef0123456789abcdef01234567"},
	})
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	if first.NodeEpoch != 1 || first.NodeID != "node-a" {
		t.Fatalf("first registration: %+v, want epoch 1", first)
	}
	if !first.RegisteredAt.Equal(clk.Now()) || !first.LastSeen.Equal(clk.Now()) {
		t.Fatalf("registration timestamps are not on the injected clock: %+v", first)
	}

	clk.Advance(time.Minute)
	second, err := r.Register("node-a", RegisterRequest{AgentVersion: "1.1.1", Slots: 4})
	if err != nil {
		t.Fatalf("re-register: %v", err)
	}
	if second.NodeEpoch != 2 {
		t.Fatalf("re-registration epoch = %d, want 2", second.NodeEpoch)
	}
	if second.AgentVersion != "1.1.1" || second.Slots != 4 {
		t.Fatalf("the declaration was not replaced: %+v", second)
	}
	if !second.RegisteredAt.Equal(first.RegisteredAt) {
		t.Fatalf("registered_at must name the first registration")
	}
}

func TestRegisterStoresTheDeclarationVerbatim(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := newRegistry(t, clk)
	decl := json.RawMessage(`{"schema":1,"slots":1000,"labels":["trusted"]}`)

	rec, err := r.Register("node-a", RegisterRequest{Capabilities: decl})
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	if string(rec.Capabilities) != string(decl) {
		t.Fatalf("the declaration must be stored verbatim, got %s", rec.Capabilities)
	}
	// The clone must not share the buffer, so a caller cannot rewrite the record
	// it was handed.
	rec.Capabilities[0] = 'X'
	again, _ := r.Get("node-a")
	if string(again.Capabilities) != string(decl) {
		t.Fatalf("a returned record aliases registry state: %s", again.Capabilities)
	}
}

func TestHeartbeatRefreshesFactsWithoutBumpingTheEpoch(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := newRegistry(t, clk)
	if _, err := r.Heartbeat("node-a", HeartbeatRequest{}); !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("heartbeat from an unregistered node = %v, want ErrUnknownNode", err)
	}
	if _, err := r.Register("node-a", RegisterRequest{Slots: 2}); err != nil {
		t.Fatalf("register: %v", err)
	}

	clk.Advance(30 * time.Second)
	got, err := r.Heartbeat("node-a", HeartbeatRequest{
		SlotsFree:  1,
		GateReport: json.RawMessage(`{"admit":false}`),
	})
	if err != nil {
		t.Fatalf("heartbeat: %v", err)
	}
	if got.NodeEpoch != 1 {
		t.Fatalf("a heartbeat must not bump the epoch, got %d", got.NodeEpoch)
	}
	if got.SlotsFree != 1 || string(got.GateReport) != `{"admit":false}` {
		t.Fatalf("the heartbeat did not refresh the volatile facts: %+v", got)
	}
	if !got.LastSeen.Equal(clk.Now()) {
		t.Fatalf("last_seen = %s, want the heartbeat time", got.LastSeen)
	}

	claimed, err := r.ObserveClaim("node-a", ClaimRequest{SlotsFree: 2, GateReport: json.RawMessage(`{"admit":true}`)})
	if err != nil {
		t.Fatalf("claim observation: %v", err)
	}
	if claimed.SlotsFree != 2 || string(claimed.GateReport) != `{"admit":true}` {
		t.Fatalf("a claim must refresh the same facts: %+v", claimed)
	}
}

func TestPresenceRendersStalenessAndSessionLoss(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := NewNodeRegistry(RegistryConfig{Now: clk.Now})
	if _, err := r.Register("node-a", RegisterRequest{}); err != nil {
		t.Fatalf("register: %v", err)
	}
	if _, err := r.HoldSession("node-a", clk.Now().Add(30*time.Second)); err != nil {
		t.Fatalf("hold session: %v", err)
	}

	if got, _ := r.Presence("node-a"); got != PresenceOnline {
		t.Fatalf("presence = %q, want online", got)
	}
	// The held request lapses and the session grace passes: disconnected, and the
	// sweeper is unaffected.
	clk.Advance(30*time.Second + DefaultSessionGraceSeconds*time.Second + time.Second)
	if got, _ := r.Presence("node-a"); got != PresenceDisconnected {
		t.Fatalf("presence after the session grace = %q, want disconnected", got)
	}
	// Nothing at all arrives for the node TTL: stale, which implies the above.
	clk.Advance(DefaultNodeTTLSeconds * time.Second)
	if got, _ := r.Presence("node-a"); got != PresenceStale {
		t.Fatalf("presence after the node TTL = %q, want stale", got)
	}
	if _, err := r.Presence("node-b"); !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("presence of an unknown node = %v, want ErrUnknownNode", err)
	}
}

func TestRevokeTombstonesTheRecordAndRefusesEveryLaterCall(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := newRegistry(t, clk)
	if _, err := r.Register("node-a", RegisterRequest{}); err != nil {
		t.Fatalf("register: %v", err)
	}

	rev, err := r.Revoke("node-a")
	if err != nil {
		t.Fatalf("revoke: %v", err)
	}
	if !rev.Revoked || rev.NodeEpoch != 2 || rev.RevokedAt.IsZero() {
		t.Fatalf("revocation must tombstone the record and bump the epoch: %+v", rev)
	}
	for name, call := range map[string]func() error{
		"register":  func() error { _, err := r.Register("node-a", RegisterRequest{}); return err },
		"heartbeat": func() error { _, err := r.Heartbeat("node-a", HeartbeatRequest{}); return err },
		"events":    func() error { _, err := r.HoldSession("node-a", clk.Now()); return err },
		"drain":     func() error { _, err := r.SetDrained("node-a", true); return err },
	} {
		if err := call(); !errors.Is(err, ErrNodeRevoked) {
			t.Errorf("%s after revocation = %v, want ErrNodeRevoked", name, err)
		}
	}
	if got, _ := r.Presence("node-a"); got != PresenceRevoked {
		t.Fatalf("presence of a revoked node = %q, want revoked", got)
	}
	if len(r.List()) != 1 {
		t.Fatalf("the record must be kept for audit")
	}
}

func TestSeedEpochKeepsARestoredLeaseFenceValid(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, _, err := s.Grant(GrantRequest{JobID: "job-a", NodeID: "node-a", NodeEpoch: 3})
	if err != nil {
		t.Fatalf("grant: %v", err)
	}

	// A restarted coordinator has the lease back from disk but an empty
	// registry; seeding from the restored lease keeps the next epoch above the
	// one the lease is fenced on.
	r := newRegistry(t, clk)
	r.SeedEpoch("node-a", l.NodeEpoch)
	rec, regErr := r.Register("node-a", RegisterRequest{})
	if regErr != nil {
		t.Fatalf("register: %v", regErr)
	}
	if rec.NodeEpoch <= l.NodeEpoch {
		t.Fatalf("post-restart epoch = %d, want it above the restored lease's %d", rec.NodeEpoch, l.NodeEpoch)
	}
}

func TestSetDrainedIsTwoWay(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	r := newRegistry(t, clk)
	if _, err := r.Register("node-a", RegisterRequest{}); err != nil {
		t.Fatalf("register: %v", err)
	}
	held, err := r.SetDrained("node-a", true)
	if err != nil || !held.Drained {
		t.Fatalf("drain: %+v %v", held, err)
	}
	lifted, err := r.SetDrained("node-a", false)
	if err != nil || lifted.Drained {
		t.Fatalf("lifting the hold must cost no second route: %+v %v", lifted, err)
	}
}
