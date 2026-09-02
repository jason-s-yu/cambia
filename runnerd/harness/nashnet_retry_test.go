package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// restartCoordinator rebuilds every nashnet store from disk and re-attaches the
// pool, which is what a coordinator restart is: lease.json, .nashnet/current.json,
// and the parts under the quarantine tree are the only state that crosses it
// (D34). The listener and the dispatcher stay up, so a node's in-flight calls
// keep landing on the same URL and the test asserts on what the restore
// rebuilt rather than on a second process.
func (r *poolRig) restartCoordinator(t *testing.T) {
	t.Helper()
	leases, err := nashnet.NewLeaseStore(nashnet.StoreConfig{
		RunsDir: r.runsDir, Policy: r.rigPolicy(), Now: r.clock.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	restored, skipped, err := leases.Restore()
	if err != nil {
		t.Fatalf("restore leases: %v", err)
	}
	if skipped != 0 {
		t.Fatalf("restore skipped %d records", skipped)
	}
	registry := nashnet.NewNodeRegistry(nashnet.RegistryConfig{Now: r.clock.now})
	for _, l := range leases.LiveForNode("") {
		registry.SeedEpoch(l.NodeID, l.NodeEpoch)
	}
	limits := quarantine.DefaultLimits()
	limits.MinFreeDiskGB = 0.0001
	limits.WatermarkMarginGB = 0.0001
	quar, err := quarantine.New(quarantine.Config{
		QuarantineDir: r.quarDir, RunsDir: r.runsDir, Limits: limits, Now: r.clock.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(PoolConfig{
		Dispatcher:       r.disp,
		Grants:           r.pool.grants,
		Leases:           leases,
		Registry:         registry,
		Quarantine:       quar,
		Bundles:          r.pool.bundles,
		RunsDir:          r.runsDir,
		NodesDir:         r.grantDir,
		OriginHost:       "coordinator.test",
		Policy:           r.rigPolicy(),
		Ceilings:         Ceilings{MaxClaimWaiters: r.cfg.maxClaimWaiters},
		MaxLeasesPerNode: r.cfg.maxLeases,
		UnplaceableGrace: r.cfg.grace,
		Now:              r.clock.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.StartupSweep(); err != nil {
		t.Fatalf("startup sweep: %v", err)
	}
	r.pool = pool
	r.srv.AttachPool(pool)
	_ = restored
}

// progress posts one progress tick on a lease and fails the test on anything
// but 200.
func (r *poolRig) progress(t *testing.T, c *nashnet.ClaimResponse, req nashnet.ProgressRequest) nashnet.ProgressResponse {
	t.Helper()
	req.LeaseEpoch = c.LeaseEpoch
	resp := r.doLease(t, c.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+c.LeaseID+"/progress", req)
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("progress on %s: got %d, want 200", c.LeaseID, resp.StatusCode)
	}
	var out nashnet.ProgressResponse
	decodeInto(t, resp, &out)
	return out
}

// nack returns a claim with the given reason and fails the test on anything but
// 200.
func (r *poolRig) nack(t *testing.T, c *nashnet.ClaimResponse, reason string) {
	t.Helper()
	resp := r.doLease(t, c.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+c.LeaseID+"/nack",
		nashnet.NackRequest{LeaseEpoch: c.LeaseEpoch, Reason: reason, CooldownSeconds: 1})
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("nack %s: got %d, want 200", c.LeaseID, resp.StatusCode)
	}
	resp.Body.Close()
}

// TestNackKeepsTheAttemptAndTheJobStaysReady is the nack row of D32 at the
// route: the job is handed out again at the same attempt, and the node that
// returned it is held off that job for its cooldown.
func TestNackKeepsTheAttemptAndTheJobStaysReady(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)
	r.queueJob(t, JobSpec{Name: "nacked-job"})

	_, first := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if first == nil || first.Attempt != 1 {
		t.Fatalf("first claim = %+v, want attempt 1", first)
	}
	r.nack(t, first, nashnet.NackSnapshotFailed)

	view, _ := r.disp.resolveView("nacked-job")
	if isTerminal(view.State) {
		t.Fatalf("a nack failed the job: %q", view.State)
	}
	// The nacking node is on cooldown for that job; another capable node takes
	// it at the same attempt.
	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again != nil {
		t.Fatal("the nacking node was handed the job it just returned")
	}
	_, second := r.claim(t, r.nodeB, nashnet.ClaimRequest{})
	if second == nil || second.Attempt != 1 {
		t.Fatalf("re-claim after a nack = %+v, want attempt 1", second)
	}
}

// TestAttemptsExhaustFailTheJob is the last row of D32: once the infrastructure
// budget is spent the job is failed with a last_error naming the last reason,
// rather than cycling the pool forever.
func TestAttemptsExhaustFailTheJob(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 4)
	r.queueJob(t, JobSpec{Name: "doomed-job", MaxAttempts: 2})

	for attempt := 1; attempt <= 2; attempt++ {
		_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
		if claimed == nil {
			t.Fatalf("attempt %d was not placed", attempt)
		}
		if claimed.Attempt != attempt {
			t.Fatalf("claim ran at attempt %d, want %d", claimed.Attempt, attempt)
		}
		r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhasePreparing})
		r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
		sweeper.Tick()
		if attempt == 1 {
			// A requeue un-projects the row the node's ticks left behind, so a
			// restart re-enqueues it in submit_seq order rather than stranding a
			// row that names a node holding nothing (D34).
			if st := readProcessState(t, r.runsDir, "doomed-job"); st.Status != procmgr.StatusCreated || st.Host != "" {
				t.Fatalf("requeued job projects %q on host %q, want created with no host",
					st.Status, st.Host)
			}
		}
	}
	view, _ := r.disp.resolveView("doomed-job")
	if view.State != StateFailed {
		t.Fatalf("state after the budget ran out = %q, want failed", view.State)
	}
	st := readProcessState(t, r.runsDir, "doomed-job")
	if st.LastError == "" {
		t.Fatal("an exhausted job must name the last reason in last_error")
	}
	if _, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); claimed != nil {
		t.Fatal("a failed job was handed out again")
	}
}

// TestPreemptedWithoutACheckpointReturnsToReady is the first gate row of D62: a
// gate stop that promoted no checkpoint is a gate release, not a terminal, so
// the job waits at its original position with no attempt charged.
func TestPreemptedWithoutACheckpointReturnsToReady(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)
	r.queueJob(t, JobSpec{Name: "gated-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	// The gate stops a job that was running and commits its final manifest
	// before posting, which is the shape D62 describes; the manifest names no
	// checkpoint, and that is the whole difference from the row below it.
	r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 2200})
	r.promote(t, claimed, artifact{path: "logs/train.log", body: []byte("stopped at the window edge\n")})
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/result",
		nashnet.ResultRequest{LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultPreempted,
			LastError: "gate:windows", FinalManifestDigest: r.headDigest(t, "gated-job")})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("preempted result: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	view, _ := r.disp.resolveView("gated-job")
	if isTerminal(view.State) {
		t.Fatalf("a gate release wrote a terminal: %q", view.State)
	}
	// The projection goes back to created with no host. A restart re-enqueues a
	// created row in submit_seq order and would strand one still naming the node
	// it left, since a row with a host is another machine's process (D34).
	st := readProcessState(t, r.runsDir, "gated-job")
	if st.Status != procmgr.StatusCreated || st.Host != "" {
		t.Fatalf("released job projects %q on host %q, want created with no host", st.Status, st.Host)
	}
	_, again := r.claim(t, r.nodeB, nashnet.ClaimRequest{})
	if again == nil {
		t.Fatal("the released job was not handed to another node")
	}
	if again.Attempt != 1 {
		t.Fatalf("a gate release charged an attempt: ran at %d, want 1", again.Attempt)
	}
}

// TestPreemptedWithACheckpointIsTerminal is the second gate row of D62 and
// AC5's central case: once a checkpoint is promoted nothing auto-resumes the
// job, and it waits for an operator.
func TestPreemptedWithACheckpointIsTerminal(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "checkpointed-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	r.promoteCheckpoint(t, claimed)

	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/result",
		nashnet.ResultRequest{LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultPreempted,
			LastError: "gate:power", FinalManifestDigest: r.headDigest(t, "checkpointed-job")})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("preempted result: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	st := readProcessState(t, r.runsDir, "checkpointed-job")
	if st.Status != nashnet.ResultPreempted {
		t.Fatalf("status = %q, want preempted", st.Status)
	}
	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again != nil {
		t.Fatal("a job with a promoted checkpoint was auto-resumed")
	}
}

// TestARequeueNeverResumesAPromotedCheckpoint is AC5's other half: even a
// requeue verdict, the one path that does return a job to ready, refuses a job
// whose checkpoint the coordinator already promoted (D33).
func TestARequeueNeverResumesAPromotedCheckpoint(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "promoted-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	r.promoteCheckpoint(t, claimed)
	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	out := sweeper.Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictRequeue {
		t.Fatalf("sweep = %+v, want the requeue verdict this test guards", out)
	}
	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again != nil {
		t.Fatal("a requeue re-placed a job whose checkpoint was promoted")
	}
	st := readProcessState(t, r.runsDir, "promoted-job")
	if st.Status != nashnet.ResultPreempted {
		t.Fatalf("status = %q, want preempted awaiting an operator", st.Status)
	}
}

// TestCircuitBreakerHoldsTheNodeAndTheQueueFlowsOn is AC4: three consecutive
// prepare_node_failed nacks hold the node, a re-register neither lifts the hold
// nor resets the counter, the cooldown doubles per trip, an operator lifts it,
// and the ready queue keeps flowing to the other node throughout.
func TestCircuitBreakerHoldsTheNodeAndTheQueueFlowsOn(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 4)
	r.register(t, r.nodeB, 4)
	for i := 0; i < 4; i++ {
		r.queueJob(t, JobSpec{Name: fmt.Sprintf("breaker-job-%d", i)})
	}

	// Two nacks, then a re-register: the hold and the counter are
	// coordinator-held, so the third nack still trips.
	for i := 0; i < 2; i++ {
		_, c := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
		if c == nil {
			t.Fatalf("nack %d: no claim", i)
		}
		r.nack(t, c, nashnet.NackPrepareNodeFailed)
	}
	r.register(t, r.nodeA, 4)
	if r.pool.breakerHeld(r.nodeA.id) {
		t.Fatal("two nacks tripped the breaker")
	}
	_, third := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if third == nil {
		t.Fatal("third nack: no claim")
	}
	r.nack(t, third, nashnet.NackPrepareNodeFailed)
	if !r.pool.breakerHeld(r.nodeA.id) {
		t.Fatal("three consecutive prepare_node_failed nacks did not trip the breaker")
	}

	// A re-register is a node route, so it clears nothing.
	r.register(t, r.nodeA, 4)
	if !r.pool.breakerHeld(r.nodeA.id) {
		t.Fatal("a re-register lifted the hold")
	}
	resp, _ := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if hold := holdReason(resp); hold != nashnet.HoldNodeGated {
		t.Fatalf("held claim = %q, want node_gated", hold)
	}
	// The queue is untouched: the other node keeps taking work.
	if _, other := r.claim(t, r.nodeB, nashnet.ClaimRequest{}); other == nil {
		t.Fatal("one node's breaker stopped the queue")
	}

	// The first hold is a minute; it lifts on its own timer, and the next trip
	// is held twice as long.
	r.clock.advance(BreakerCooldown + time.Second)
	if r.pool.breakerHeld(r.nodeA.id) {
		t.Fatal("the first hold outlasted its cooldown")
	}
	for i := 0; i < BreakerThreshold; i++ {
		_, c := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
		if c == nil {
			t.Fatalf("second trip, nack %d: no claim", i)
		}
		r.nack(t, c, nashnet.NackPrepareNodeFailed)
	}
	trips, held := r.pool.breakerReport(r.nodeA.id, r.clock.now())
	if trips != 2 {
		t.Fatalf("trip count = %d, want 2", trips)
	}
	if want := int64(2 * BreakerCooldown / time.Second); held != want {
		t.Fatalf("second hold = %ds, want %d", held, want)
	}

	// An operator lifts it and the counter goes with it.
	resp = r.do(http.MethodPost, "/nashnet/nodes/"+r.nodeA.id+"/drain",
		nashnet.DrainRequest{Drain: false, ClearBreaker: true})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("clear_breaker: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if r.pool.breakerHeld(r.nodeA.id) {
		t.Fatal("clear_breaker did not lift the hold")
	}
	if _, c := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); c == nil {
		t.Fatal("a cleared node was not handed work")
	}
}

// TestABreakerCooldownDoublesToItsCap pins the ladder itself, which the route
// test only walks two rungs of.
func TestABreakerCooldownDoublesToItsCap(t *testing.T) {
	for _, tc := range []struct {
		trips int
		want  time.Duration
	}{
		{1, BreakerCooldown},
		{2, 2 * BreakerCooldown},
		{3, 4 * BreakerCooldown},
		{6, 32 * BreakerCooldown},
		{7, BreakerCooldownCap},
		{20, BreakerCooldownCap},
	} {
		if got := cooldownFor(tc.trips); got != tc.want {
			t.Fatalf("cooldown after %d trips = %s, want %s", tc.trips, got, tc.want)
		}
	}
}

// TestRestartWithALiveLeaseIsInvisibleToTheNode is AC2: a coordinator restart
// mid-upload leaves the node's lease, its epoch fence, its manifest head, and
// its in-flight part exactly where they were, so its next calls all land.
func TestRestartWithALiveLeaseIsInvisibleToTheNode(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "surviving-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 4242})

	// Half of a two-chunk file is in flight when the coordinator goes down.
	body := []byte("first half of the checkpoint|second half of the checkpoint")
	digest := sha256Hex(body)
	r.patchChunk(t, claimed, digest, body[:20], 0, len(body))

	r.restartCoordinator(t)

	// The lease came back with its token, its epoch, and its phase, so the
	// node's next progress post is not superseded and the runtime cap still runs
	// from the original granted_at.
	out := r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseUploading, PID: 4242})
	if out.Revoke {
		t.Fatal("a restored lease answered revoke")
	}
	restored, ok := r.pool.leases.Get(claimed.LeaseID)
	if !ok || !restored.PIDProjected || restored.Phase != nashnet.PhaseUploading {
		t.Fatalf("restored lease lost its verdict facts: %+v", restored)
	}

	// The part survived as a plain file whose size is the resume offset.
	head := r.raw(t, http.MethodHead, "/nashnet/leases/"+claimed.LeaseID+"/blobs/"+digest, nil,
		map[string]string{nashnet.HeaderLeaseToken: claimed.LeaseToken})
	defer head.Body.Close()
	if head.StatusCode != http.StatusOK {
		t.Fatalf("resume probe after the restart: got %d, want 200", head.StatusCode)
	}
	if got := head.Header.Get(HeaderOffset); got != "20" {
		t.Fatalf("resume offset after the restart = %q, want 20", got)
	}
	r.patchChunk(t, claimed, digest, body[20:], 20, len(body))
}

// TestRestartWithAnExpiredLeaseFinalizes is AC3: a lease whose deadline passed
// while the coordinator was down is restored so the sweeper can settle it, and
// a job that had launched is finalized rather than run a second time.
func TestRestartWithAnExpiredLeaseFinalizes(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "lapsed-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 7788})

	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	r.restartCoordinator(t)

	out := r.pool.Sweeper().Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictFinalize {
		t.Fatalf("sweep after the restart = %+v, want one finalize", out)
	}
	view, _ := r.disp.resolveView("lapsed-job")
	if !isTerminal(view.State) {
		t.Fatalf("a launched job's lapsed lease left it at %q, want a terminal", view.State)
	}
	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again != nil {
		t.Fatal("a job that ran was handed out again after the restart")
	}
}

// TestTheLeaseLifetimeCapEndsALeaseAtTheRoute is the lease-lifetime row of D32
// at the route, with the node's own job_policy gate lowering the cap: the lease
// moves to revoking, the node is told to stop, and the phase reached decides
// what happens next.
func TestTheLeaseLifetimeCapEndsALeaseAtTheRoute(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "long-job"})

	hours := 0.5
	report := gates.Report{
		Admit: true, SlotsOffered: 2, EvaluatedAt: r.clock.now(),
		Checks: []gates.Check{{
			Gate: "job_policy.max_runtime_hours", OK: true, Required: &hours,
		}},
	}
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{GateReport: mustJSON(t, report)})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	lease, _ := r.pool.leases.Get(claimed.LeaseID)
	if lease.MaxRuntime != 30*time.Minute {
		t.Fatalf("lease runtime cap = %s, want the node's own 30m policy", lease.MaxRuntime)
	}
	r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 5150})

	// Renewals do not buy time past the cap: the node holds the lease open with
	// a progress tick a minute for the whole half hour, well inside the 120s
	// TTL, and the cap ends it anyway.
	var out []nashnet.Outcome
	for elapsed := time.Minute; elapsed <= 31*time.Minute; elapsed += time.Minute {
		r.clock.advance(time.Minute)
		if out = sweeper.Tick(); len(out) > 0 {
			if elapsed < 30*time.Minute {
				t.Fatalf("the lease ended at %s, inside its 30m cap: %+v", elapsed, out)
			}
			break
		}
		r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 5150})
	}
	if len(out) != 1 || out[0].State != nashnet.LeaseRevoking || out[0].Reason != nashnet.ReasonRuntimeCap {
		t.Fatalf("at the cap = %+v, want one revoking outcome on the runtime cap", out)
	}
	tick := r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 5150})
	if !tick.Revoke {
		t.Fatal("a revoking lease must answer revoke on the next progress post")
	}
	// The grace runs out with no final commit: the job ran, so it is finalized.
	r.clock.advance(2 * time.Duration(nashnet.DefaultLeaseTTLSeconds) * time.Second)
	out = sweeper.Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictFinalize {
		t.Fatalf("after the grace = %+v, want one finalize", out)
	}
	view, _ := r.disp.resolveView("long-job")
	if !isTerminal(view.State) {
		t.Fatalf("state after the cap = %q, want a terminal", view.State)
	}
}

// TestNodeByteRateBucketAdmitsAtItsRate covers the RUNNERD_NASHNET_NODE_MBPS
// bucket: a node spends its second of transfer, is refused, and is admitted
// again once the bucket refills.
func TestNodeByteRateBucketAdmitsAtItsRate(t *testing.T) {
	clk := &testClock{t: time.Date(2026, time.September, 1, 12, 0, 0, 0, time.UTC)}
	table := newByteTable(1, clk.now) // 1 Mbit/s = 125000 bytes/s
	if !table.spend("node-a", 125000) {
		t.Fatal("the first full second of transfer was refused")
	}
	if table.spend("node-a", 1000) {
		t.Fatal("a node over its byte rate was admitted")
	}
	if !table.spend("node-b", 1000) {
		t.Fatal("the bucket is per node, not global")
	}
	clk.advance(time.Second)
	if !table.spend("node-a", 125000) {
		t.Fatal("the bucket did not refill")
	}
	// A zero rate is the documented unlimited default.
	if !newByteTable(0, clk.now).spend("node-a", 1<<40) {
		t.Fatal("an unmetered pool refused a transfer")
	}
}

// TestResultRecordsAreBounded pins the retention that keeps the replay records
// of D6 from being a per-lease leak for the daemon's life.
func TestResultRecordsAreBounded(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	for i := 0; i < 3; i++ {
		id := "lease-" + strconv.Itoa(i)
		r.pool.recordResult(id, nashnet.ResultResponse{LeaseID: id}, "hash")
		r.clock.advance(resultRetention/2 + time.Second)
	}
	r.pool.mu.Lock()
	kept := len(r.pool.results)
	_, oldestKept := r.pool.results["lease-0"]
	auth, at := len(r.pool.resultAuth), len(r.pool.resultAt)
	r.pool.mu.Unlock()
	if oldestKept {
		t.Fatal("a record past the retention window was kept")
	}
	if kept != 2 || auth != 2 || at != 2 {
		t.Fatalf("retained %d results, %d auth hashes, %d stamps, want 2 of each", kept, auth, at)
	}
}

// sha256Hex is the digest a blob route keys on.
func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// mustJSON marshals a fixture into the raw-message shape the wire types carry.
func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// headDigest reads a job's promoted manifest head digest, which a result must
// carry to be accepted (D6).
func (r *poolRig) headDigest(t *testing.T, job string) string {
	t.Helper()
	head, err := r.pool.quar.ReadHead(job)
	if err != nil {
		t.Fatalf("read manifest head for %s: %v", job, err)
	}
	return head.Digest
}

// artifact is one file a run promotes: the path it lands on under the run dir
// and the bytes behind it.
type artifact struct {
	path string
	body []byte
}

// promote uploads each artifact and commits them as one final manifest, which
// is what a run leaves behind at its end. Everything the coordinator later
// reads as promoted state goes through here, so a test never writes into a run
// dir past the promotion gate.
func (r *poolRig) promote(t *testing.T, c *nashnet.ClaimResponse, files ...artifact) {
	t.Helper()
	entries := make([]quarantine.Entry, 0, len(files))
	for _, f := range files {
		entries = append(entries, quarantine.Entry{
			Path:   f.path,
			Digest: r.uploadBlob(t, c.LeaseToken, c.LeaseID, f.body),
			Size:   int64(len(f.body)), MTime: r.clock.now().UnixNano(),
		})
	}
	head, err := r.pool.quar.ReadHead(c.JobID)
	if err != nil {
		t.Fatalf("read manifest head for %s: %v", c.JobID, err)
	}
	resp := r.doLease(t, c.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+c.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: c.LeaseEpoch, Seq: head.Folded.Seq + 1,
			Parent: head.Digest, Final: true, Entries: entries,
		})
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("promote %d entries for %s: got %d, want 200 (%s)",
			len(entries), c.JobID, resp.StatusCode, body)
	}
	resp.Body.Close()
}

// promoteCheckpoint promotes the two files the PRT-CFR resume contract names,
// which is the state the no-auto-resume rule of D33 keys on.
func (r *poolRig) promoteCheckpoint(t *testing.T, c *nashnet.ClaimResponse) {
	t.Helper()
	r.promote(t, c,
		artifact{path: "snapshots/prtcfr_checkpoint.pt", body: []byte("prtcfr checkpoint bytes\n")},
		artifact{path: "resume_state.json", body: []byte(`{"iteration": 7}` + "\n")})
}

// promoteJournal uploads and commits a run_db.sqlite fixture as a final
// manifest, which is what a run leaves behind at its end. The corpus under
// testdata/rundb is the shared one the D55 validator is pinned against, so the
// promoted copy is a journal the coordinator accepted rather than bytes a test
// wrote past the gate.
func (r *poolRig) promoteJournal(t *testing.T, c *nashnet.ClaimResponse, fixture string) {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "rundb", fixture))
	if err != nil {
		t.Fatal(err)
	}
	r.promote(t, c, artifact{path: quarantine.RunDBPath, body: body})
}

// TestAResultLessTerminalIsSettledByTheSecondWitness is D35: a node that
// finished and then vanished is settled from the artifacts it committed. One
// sweep settles both jobs, and the only thing telling them apart is whether the
// promoted journal says the run completed; the job with no journal is recorded
// crashed with no exit code, the conservative verdict under every on_failure
// policy. The fixture's own runs.name is the job name, because the D55 identity
// check is what makes the promoted copy a witness rather than a file a node
// named run_db.sqlite.
func TestAResultLessTerminalIsSettledByTheSecondWitness(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "job-train-0001"})
	r.queueJob(t, JobSpec{Name: "silent-no-journal"})

	for _, job := range []string{"job-train-0001", "silent-no-journal"} {
		_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
		if claimed == nil || claimed.JobID != job {
			t.Fatalf("claim = %+v, want a lease on %s", claimed, job)
		}
		r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhaseRunning, PID: 3300})
		if job == "job-train-0001" {
			r.promoteJournal(t, claimed, "valid_train.sqlite")
		}
	}

	// The node stops answering: no result is ever posted for either job.
	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	out := sweeper.Tick()
	if len(out) != 2 {
		t.Fatalf("sweep = %+v, want both leases settled", out)
	}
	for _, o := range out {
		if o.Verdict != nashnet.VerdictFinalize {
			t.Fatalf("%s settled as %s, want finalize: a job that ran is never retried", o.JobID, o.Verdict)
		}
		if o.NextAttempt != 1 {
			t.Fatalf("%s advanced to attempt %d, want 1", o.JobID, o.NextAttempt)
		}
	}

	completed := readProcessState(t, r.runsDir, "job-train-0001")
	if completed.Status != procmgr.StatusStopped {
		t.Fatalf("a promoted journal reading completed settled as %q, want stopped", completed.Status)
	}
	if completed.ExitCode == nil || *completed.ExitCode != 0 {
		t.Fatalf("exit code = %v, want 0 inferred from the journal", completed.ExitCode)
	}
	absent := readProcessState(t, r.runsDir, "silent-no-journal")
	if absent.Status != procmgr.StatusCrashed {
		t.Fatalf("a job with no promoted journal settled as %q, want crashed", absent.Status)
	}
	if absent.ExitCode != nil {
		t.Fatalf("exit code = %d, want none: this daemon never waited on the process", *absent.ExitCode)
	}
}

// TestThePreemptedStateSpellsTheWireValue keeps the runnerd state enum and the
// node's terminal vocabulary from drifting: jobspec.go spells preempted rather
// than importing the wire package, so nothing but this compares the two.
func TestThePreemptedStateSpellsTheWireValue(t *testing.T) {
	if StatePreempted != nashnet.ResultPreempted {
		t.Fatalf("state %q and wire %q disagree on preempted", StatePreempted, nashnet.ResultPreempted)
	}
	if !isTerminal(StatePreempted) {
		t.Fatal("preempted must be terminal: a dependent gates on it as a non-success parent (D62)")
	}
}

// TestANackOfAResumedJobStillReturnsIt separates the two checkpoint questions.
// An operator resume runs because an earlier run promoted a checkpoint, so
// reading the job's whole promoted state on a nack would fail every resume the
// first time a node returned it. What is never re-placed is state promoted
// under the lease being settled, and a nack is pre-launch by construction.
func TestANackOfAResumedJobStillReturnsIt(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)
	r.finishPoolRun(t, "resumed-job")

	if _, err := r.disp.Resume("resumed-job"); err != nil {
		t.Fatalf("resume: %v", err)
	}
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil || claimed.JobID != "resumed-job" {
		t.Fatalf("claim = %+v, want the resumed job", claimed)
	}
	r.nack(t, claimed, nashnet.NackSnapshotFailed)

	view, _ := r.disp.resolveView("resumed-job")
	if isTerminal(view.State) {
		t.Fatalf("a nack of a resumed job settled it as %q", view.State)
	}
	// The pin holds it to its own node, so the re-placement is that node's, once
	// the nack cooldown on that pair has run out.
	r.clock.advance(2 * time.Second)
	if _, again := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); again == nil {
		t.Fatal("the resumed job was not handed out again after its nack")
	}
}

// TestARequeuedResumeIsNotUnProjected is the other half of the resume case: the
// row of a job holding a promoted checkpoint is left as it stands, because the
// resume intent lives in the queue handle and a created row would come back
// from a restart as a fresh launch over the checkpoint's own run dir.
func TestARequeuedResumeIsNotUnProjected(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 2)
	r.finishPoolRun(t, "resumed-expiry-job")

	if _, err := r.disp.Resume("resumed-expiry-job"); err != nil {
		t.Fatalf("resume: %v", err)
	}
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim on the resumed job")
	}
	r.progress(t, claimed, nashnet.ProgressRequest{Phase: nashnet.PhasePreparing})
	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	if out := sweeper.Tick(); len(out) != 1 || out[0].Verdict != nashnet.VerdictRequeue {
		t.Fatalf("sweep = %+v, want one requeue: the lease never launched", out)
	}
	if st := readProcessState(t, r.runsDir, "resumed-expiry-job"); st.Status == procmgr.StatusCreated {
		t.Fatal("a resume with a promoted checkpoint was un-projected to created")
	}
}
