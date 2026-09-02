package nashnet

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// newStore returns a store over a temp runs dir on the given clock.
func newStore(t *testing.T, clk *fakeClock) *LeaseStore {
	t.Helper()
	s, err := NewLeaseStore(StoreConfig{
		RunsDir: t.TempDir(),
		Policy:  DefaultPolicy(),
		Now:     clk.Now,
		Entropy: testEntropy(),
	})
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	return s
}

// grant places one job on one node and fails the test if it cannot.
func grant(t *testing.T, s *LeaseStore, jobID, nodeID string) (Lease, string) {
	t.Helper()
	l, token, err := s.Grant(GrantRequest{JobID: jobID, NodeID: nodeID, NodeEpoch: 1})
	if err != nil {
		t.Fatalf("grant %s on %s: %v", jobID, nodeID, err)
	}
	return l, token
}

// progressFor builds a fenced progress post for a lease: both epochs come from
// the record, as the coordinator supplies them at the route.
func progressFor(l Lease, token, phase string, pid bool) ProgressUpdate {
	return ProgressUpdate{
		LeaseID:      l.LeaseID,
		LeaseEpoch:   l.LeaseEpoch,
		NodeEpoch:    l.NodeEpoch,
		Token:        token,
		Phase:        phase,
		PIDProjected: pid,
	}
}

// leaseJSON reads the persisted record as a raw map, so a test asserts on the
// bytes on disk rather than on the struct that wrote them.
func leaseJSON(t *testing.T, s *LeaseStore, jobID string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(s.cfg.RunsDir, jobID, LeaseFileName))
	if err != nil {
		t.Fatalf("read lease.json for %s: %v", jobID, err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("decode lease.json for %s: %v", jobID, err)
	}
	return m
}

// advanceToExpiry moves the clock just past a lease deadline.
func advanceToExpiry(clk *fakeClock, l Lease) {
	clk.mu.Lock()
	defer clk.mu.Unlock()
	clk.t = l.Deadline.Add(time.Second)
}
