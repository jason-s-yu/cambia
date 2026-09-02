package harness

import (
	"testing"
	"time"
)

// TestRegisterAndClaim is the first D42 scenario: a node agent registers over
// the pinned HTTPS transport and is handed the one ready job. Everything after
// it in this file assumes this much works.
func TestRegisterAndClaim(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true, verbose: true})
	r.queueFixtureJob(t, "claim-me", 2, "quick")

	r.runCycle(t, 60*time.Second)

	rec, ok := r.pool.nodes.Get(r.node.id)
	if !ok {
		t.Fatal("the coordinator holds no record for the node that registered")
	}
	if rec.NodeEpoch == 0 {
		t.Fatal("register handed out no node epoch")
	}
	if got := len(r.requests.forPath("/nashnet/nodes/register")); got != 1 {
		t.Fatalf("register requests = %d, want exactly 1", got)
	}
	if got := len(r.requests.forPath("/nashnet/claim")); got != 1 {
		t.Fatalf("claim requests = %d, want exactly 1", got)
	}
	view, _ := r.disp.resolveView("claim-me")
	if !isTerminal(view.State) {
		t.Fatalf("state after one cycle = %q, want a terminal", view.State)
	}
}
