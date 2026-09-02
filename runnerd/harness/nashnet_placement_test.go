package harness

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// TestTwoConcurrentClaimsNeverShareAJob is AC(2): the scan reserves a job before
// it leaves the lock and the lease store refuses a second grant, so exactly one
// of two racing claims is handed the only queued job.
func TestTwoConcurrentClaimsNeverShareAJob(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)
	r.queueJob(t, JobSpec{Name: "contested-job"})

	var wg sync.WaitGroup
	results := make([]*nashnet.ClaimResponse, 2)
	statuses := make([]int, 2)
	nodes := []fixtureNode{r.nodeA, r.nodeB}
	start := make(chan struct{})
	for i := range nodes {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			resp, claimed := r.claim(t, nodes[i], nashnet.ClaimRequest{})
			statuses[i] = resp.StatusCode
			results[i] = claimed
			if claimed == nil {
				resp.Body.Close()
			}
		}(i)
	}
	close(start)
	wg.Wait()

	granted := 0
	for i, res := range results {
		if res != nil {
			granted++
			if res.JobID != "contested-job" {
				t.Fatalf("claim %d got job %q", i, res.JobID)
			}
		} else if statuses[i] != http.StatusNoContent {
			t.Fatalf("claim %d: got %d, want 204 for the loser", i, statuses[i])
		}
	}
	if granted != 1 {
		t.Fatalf("%d claims were handed the same job, want exactly 1", granted)
	}
	if live := len(r.pool.leases.LiveForNode("")); live != 1 {
		t.Fatalf("%d live leases for one job, want 1", live)
	}
}

// TestExclusiveBarrierHoldsTheSecondHandout is AC(3): a node holding an
// exclusive lease is handed nothing else, and a later ready job the node
// matches waits behind an older exclusive ready job it also matches.
func TestExclusiveBarrierHoldsTheSecondHandout(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 4)
	r.queueJob(t, JobSpec{Name: "measure-alone", Exclusive: true})
	r.queueJob(t, JobSpec{Name: "ordinary-job"})

	_, first := r.claim(t, r.nodeA, nashnet.ClaimRequest{SlotsFree: 4})
	if first == nil || first.JobID != "measure-alone" {
		t.Fatalf("first claim = %+v, want the exclusive job", first)
	}
	resp, second := r.claim(t, r.nodeA, nashnet.ClaimRequest{SlotsFree: 4})
	if second != nil {
		t.Fatalf("a node holding an exclusive lease was handed %q", second.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldExclusivePending {
		t.Fatalf("second claim hold = %q, want exclusive_pending", hold)
	}

	// The barrier also applies before the exclusive job is placed: a node
	// already holding an ordinary lease is not handed the later job over the
	// exclusive one waiting ahead of it.
	r2 := newPoolRig(t, poolRigConfig{})
	r2.register(t, r2.nodeA, 4)
	r2.queueJob(t, JobSpec{Name: "warmup-job"})
	r2.queueJob(t, JobSpec{Name: "barrier-job", Exclusive: true})
	r2.queueJob(t, JobSpec{Name: "later-job"})

	if _, got := r2.claim(t, r2.nodeA, nashnet.ClaimRequest{SlotsFree: 4}); got == nil || got.JobID != "warmup-job" {
		t.Fatalf("first claim = %+v, want warmup-job", got)
	}
	resp, blocked := r2.claim(t, r2.nodeA, nashnet.ClaimRequest{SlotsFree: 4})
	if blocked != nil {
		t.Fatalf("a job past the exclusive barrier was handed out: %q", blocked.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldExclusivePending {
		t.Fatalf("barriered claim hold = %q, want exclusive_pending", hold)
	}
}

// TestUnplaceableSurfacesAfterTheGrace is AC(4): a job no node matches, and a
// job every capable node's gate refuses, both render their hold after the grace
// and neither is failed.
func TestUnplaceableSurfacesAfterTheGrace(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{grace: 10 * time.Minute})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "cuda-job", Device: "cuda"})

	resp, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatalf("a cpu-only node was handed a cuda job: %q", claimed.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNoMatch {
		t.Fatalf("claim hold = %q, want no_match", hold)
	}
	// Inside the grace the view stays quiet: a momentary mismatch is not news.
	view, _ := r.disp.resolveView("cuda-job")
	if view.Placement != "" {
		t.Fatalf("placement rendered inside the grace: %q", view.Placement)
	}
	r.clock.advance(11 * time.Minute)
	view, _ = r.disp.resolveView("cuda-job")
	if view.Placement != PlacementWaitingForNode {
		t.Fatalf("placement = %q, want waiting_for_node", view.Placement)
	}
	if len(view.PlacementDetail) == 0 {
		t.Fatal("the union of match rejection reasons is empty")
	}
	if isTerminal(view.State) {
		t.Fatalf("an unplaceable job was failed: state = %q", view.State)
	}

	// A gate that will reopen renders as a window rather than a capability gap.
	reopen := r.clock.now().Add(2 * time.Hour)
	r.queueJob(t, JobSpec{Name: "gated-job"})
	resp = r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/claim", nashnet.ClaimRequest{
		SlotsFree: 1, Capabilities: declaration(1), GateReport: denyReport(reopen),
	})
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("gated claim: got %d, want 204", resp.StatusCode)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNodeGated {
		t.Fatalf("gated claim hold = %q, want node_gated", hold)
	}
	r.clock.advance(11 * time.Minute)
	view, _ = r.disp.resolveView("gated-job")
	if view.Placement != PlacementWaitingForNodeGate {
		t.Fatalf("placement = %q, want waiting_for_node_gate", view.Placement)
	}
	if isTerminal(view.State) {
		t.Fatalf("a gated job was failed: state = %q", view.State)
	}
	var sawWindow bool
	for _, d := range view.PlacementDetail {
		if d == "gate:windows" {
			sawWindow = true
		}
	}
	if !sawWindow {
		t.Fatalf("placement detail %v names no failing gate check", view.PlacementDetail)
	}
}

// TestNoGitRunsUnderThePlacementLock is AC(11): the bundle build is a git fork,
// and one node's cold fetch must not stall every other node's claim, so it runs
// after the scan has released the dispatcher lock.
func TestNoGitRunsUnderThePlacementLock(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.bundles.onCall = func(basis []string) error {
		if !r.disp.mu.TryLock() {
			return errors.New("bundle build ran while the placement lock was held")
		}
		r.disp.mu.Unlock()
		return nil
	}
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "bundle-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("the claim failed, so the bundle builder reported the lock was held")
	}
	if r.bundles.count() == 0 {
		t.Fatal("no bundle was built, so the assertion never ran")
	}
	if claimed.Snapshot.SHA256 == "" || claimed.Snapshot.Size == 0 {
		t.Fatalf("claim snapshot = %+v, want the resolved bundle descriptor", claimed.Snapshot)
	}
	if claimed.Snapshot.Commit != "0123456789abcdef0123456789abcdef01234567" {
		t.Fatalf("claim snapshot commit = %q, want the pinned commit", claimed.Snapshot.Commit)
	}
}

// TestMissingSeedNeverAnswersTheClaim is AC(15): an unresolvable seed is a
// job-level defect answered at the job, so the scan continues and the claim gets
// the next match, never 409 seed_missing.
func TestMissingSeedNeverAnswersTheClaim(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{grace: time.Minute})
	r.register(t, r.nodeA, 2)

	// The referenced run dir is gone entirely: spec-fatal.
	r.queueJob(t, JobSpec{Name: "eval-gone", Kind: KindEvaluate, Target: "vanished-run/snapshots/x.pt"})
	// The run dir exists but the named entry does not: unplaceable, not fatal.
	partial := filepath.Join(r.runsDir, "partial-run")
	if err := os.MkdirAll(partial, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := procmgr.WriteProcessState(partial, &procmgr.ProcessState{
		Name: "partial-run", Status: procmgr.StatusStopped,
	}); err != nil {
		t.Fatal(err)
	}
	r.queueJob(t, JobSpec{Name: "eval-partial", Kind: KindEvaluate, Target: "partial-run/snapshots/gone.pt"})
	r.queueJob(t, JobSpec{Name: "good-job"})

	resp, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if resp.StatusCode == http.StatusConflict {
		t.Fatal("a claim was answered with a seed conflict rather than the next match")
	}
	if claimed == nil || claimed.JobID != "good-job" {
		t.Fatalf("claim = %+v, want the next placeable job", claimed)
	}

	fatal, _ := r.disp.resolveView("eval-gone")
	if !isTerminal(fatal.State) {
		t.Fatalf("a job whose seed run dir is gone stayed at %q, want a terminal", fatal.State)
	}
	partialView, _ := r.disp.resolveView("eval-partial")
	if isTerminal(partialView.State) {
		t.Fatalf("a job whose seed run dir still exists was failed: %q", partialView.State)
	}
	if partialView.Placement != PlacementSeedMissing {
		t.Fatalf("placement = %q, want seed_missing", partialView.Placement)
	}
}

// TestPinAndLeaseCeilingBoundTheScan covers the two placement inputs the D11
// order names after Match: requires.node is a hard pin, and the coordinator's
// own per-node lease ceiling stops handing work to a node at its cap.
func TestPinAndLeaseCeilingBoundTheScan(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{maxLeases: 1})
	r.register(t, r.nodeA, 4)
	r.register(t, r.nodeB, 4)
	r.queueJob(t, JobSpec{Name: "pinned-job", Requires: &capability.Requires{Node: r.nodeB.id}})

	resp, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{SlotsFree: 4})
	if claimed != nil {
		t.Fatalf("a pin to another node was ignored: %q went to node-a", claimed.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNoMatch {
		t.Fatalf("pinned-away claim hold = %q, want no_match", hold)
	}
	if _, got := r.claim(t, r.nodeB, nashnet.ClaimRequest{SlotsFree: 4}); got == nil {
		t.Fatal("the pinned node was not handed its job")
	}

	// node-b now holds its one permitted lease.
	r.queueJob(t, JobSpec{Name: "second-job"})
	resp, second := r.claim(t, r.nodeB, nashnet.ClaimRequest{SlotsFree: 4})
	if second != nil {
		t.Fatalf("the lease ceiling was exceeded: %q", second.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNoMatch {
		t.Fatalf("ceilinged claim hold = %q, want no_match", hold)
	}
}

// TestNackReturnsTheJobWithoutAnAttempt is D8: a returned claim is a scheduling
// event, so the job keeps its queue position and its attempt, and the node that
// returned it is excluded for the cooldown while another node takes it.
func TestNackReturnsTheJobWithoutAnAttempt(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.register(t, r.nodeB, 1)
	r.queueJob(t, JobSpec{Name: "nacked-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/nack", nashnet.NackRequest{
			LeaseEpoch: claimed.LeaseEpoch, Reason: nashnet.NackGateBreach, CooldownSeconds: 300,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("nack: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again != nil {
		t.Fatal("the node that nacked was re-matched inside its cooldown")
	}
	_, other := r.claim(t, r.nodeB, nashnet.ClaimRequest{})
	if other == nil || other.JobID != "nacked-job" {
		t.Fatalf("another node claim = %+v, want the returned job", other)
	}
	if other.Attempt != claimed.Attempt {
		t.Fatalf("attempt = %d after a nack, want %d unchanged", other.Attempt, claimed.Attempt)
	}
	view, _ := r.disp.resolveView("nacked-job")
	if isTerminal(view.State) {
		t.Fatalf("a nack failed the job: state = %q", view.State)
	}
}
