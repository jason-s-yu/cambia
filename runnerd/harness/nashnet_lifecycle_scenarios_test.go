package harness

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
)

// The lifecycle half of D42: expiry, death, cancel, revocation, gate breaches,
// and a coordinator restart, each driven against a live node agent rather than
// a hand-built lease record.
//
// Both runnerd roles are goroutines in one test binary, so a signal aimed at
// the node would take the coordinator with it. Where D42 names a signal, the
// scenario models what the coordinator can actually observe: SIGSTOP is a node
// that stops answering, SIGKILL is a node whose supervision dies while its
// forked job keeps running. The signal-level cases are the live battery's
// (D43, AC4).

// longLeasePolicy is the policy the gate scenarios run under. Advancing the
// clock past a window also advances it past a short lease TTL, and the node's
// own loss-of-contact rule (D36) fires at twice the TTL, so the TTL is raised
// well past the clock jump to keep the gate the only thing under test.
func longLeasePolicy() nashnet.Policy {
	p := nashnet.DefaultPolicy()
	p.LeaseTTLSeconds = 7200
	p.ProgressIntervalSeconds = 1
	return p
}

// TestLeaseExpiryWithASilentNodeBeforeLaunch is the pre-launch half of D42's
// expiry scenario: a node that goes silent before it launches loses its lease
// to the sweep, and the job returns to the ready set rather than being consumed
// (D7).
func TestLeaseExpiryWithASilentNodeBeforeLaunch(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{slots: 1, claimWaitSeconds: 1})
	held := make(chan struct{})
	r.env.onPrepare = func(string) error {
		// The node stops answering in the middle of staging, which is what a
		// SIGSTOPped agent looks like from the coordinator.
		<-held
		return nil
	}
	r.queueFixtureJob(t, "expire-prelaunch", 2, "quick")

	stop := r.startAgent(t)
	defer func() { close(held); stop() }()

	lease := r.waitForNodePhase(t, "expire-prelaunch", nashnet.PhasePreparing, 30*time.Second)
	if lease.PIDProjected {
		t.Fatal("the lease already projected a pid: the scenario needs the pre-launch half")
	}

	r.clock.advance(time.Duration(r.rigPolicy().LeaseTTLSeconds+1) * time.Second)
	out := r.pool.Sweeper().Tick()
	if len(out) != 1 {
		t.Fatalf("sweep = %+v, want the one expired lease", out)
	}
	if out[0].Verdict != nashnet.VerdictRequeue {
		t.Fatalf("verdict = %q, want a requeue: nothing ran, so nothing is consumed", out[0].Verdict)
	}
	view, _ := r.disp.resolveView("expire-prelaunch")
	if isTerminal(view.State) {
		t.Fatalf("state = %q after a pre-launch expiry, want the job back in the queue", view.State)
	}
}

// TestLeaseExpiryWithASilentNodeAfterLaunch is the post-launch half: a node
// that launched and then went silent has its lease finalized rather than
// requeued, because a job that ran has already consumed its attempt (D7, D32).
func TestLeaseExpiryWithASilentNodeAfterLaunch(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{slots: 1, claimWaitSeconds: 1})
	r.queueFixtureJob(t, "expire-postlaunch", 2, "sleep")

	stop := r.startAgent(t)
	r.waitForNodePhase(t, "expire-postlaunch", nashnet.PhaseRunning, 30*time.Second)
	pid := r.nodePID(t, "expire-postlaunch", 10*time.Second)

	// The node's supervision dies; the process it forked does not. The
	// coordinator hears nothing more about either.
	stop()
	if !processAlive(pid) {
		t.Fatal("the forked job died with its agent: the scenario needs an orphaned process")
	}

	r.clock.advance(time.Duration(r.rigPolicy().LeaseTTLSeconds+1) * time.Second)
	out := r.pool.Sweeper().Tick()
	if len(out) != 1 {
		t.Fatalf("sweep = %+v, want the one expired lease", out)
	}
	if out[0].Verdict != nashnet.VerdictFinalize {
		t.Fatalf("verdict = %q, want a finalize: the job ran", out[0].Verdict)
	}
	view, _ := r.disp.resolveView("expire-postlaunch")
	if !isTerminal(view.State) {
		t.Fatalf("state = %q after a post-launch expiry, want a terminal", view.State)
	}
}

// TestNodeDeathMidJobIsRecoveredByAReattach is D42's node SIGKILL: the agent
// dies mid-job, its forked process survives, and a restarted agent reattaches
// to the lease it left on disk rather than orphaning the run (D37, D3). The
// coordinator re-binds exactly the leases the register names in live_leases.
//
// Re-run after cambia-1887, which moved resume persistence into jobspec.json.
func TestNodeDeathMidJobIsRecoveredByAReattach(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{
		slots: 1, claimWaitSeconds: 1, pool: poolRigConfig{policy: longLeasePolicy()},
	})
	r.queueFixtureJob(t, "node-death", 2, "sleep")

	stop := r.startAgent(t)
	before := r.waitForNodePhase(t, "node-death", nashnet.PhaseRunning, 30*time.Second)
	pid := r.nodePID(t, "node-death", 10*time.Second)
	stop()
	if !processAlive(pid) {
		t.Fatal("the forked job died with its agent: the scenario needs a survivor")
	}

	r.requests.reset()
	restarted := r.restartNodeAgent(t, true)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = restarted.Run(ctx)
	}()

	// The restarted agent adopts the lease rather than claiming a new one: the
	// lease id does not change and no second claim is handed out.
	waitUntil(t, "the restarted agent to renew the lease", 30*time.Second, func() bool {
		return len(r.requests.forPath("/progress")) > 0
	})
	after := r.pool.leases.LiveForNode(r.node.id)
	if len(after) != 1 || after[0].LeaseID != before.LeaseID {
		t.Fatalf("leases after the restart = %+v, want the original %s re-bound",
			after, before.LeaseID)
	}
	for _, row := range r.requests.forPath("/nashnet/claim") {
		if row.Status == http.StatusOK {
			t.Fatal("the restarted agent was handed a second lease for a job it already held")
		}
	}
	// The reattached agent supervises a job that sleeps for two minutes, so
	// the test ends the run rather than waiting it out.
	cancel()
	<-done
}

// TestCancelReachesTheNodeOnTheEventsPoll is D42's cancel: an operator cancel
// of a leased job routes through the lease rather than a local signal, and the
// held events poll delivers it within one round trip rather than at the next
// poll (D45, D62). The poll window is ten seconds and the stop lands well
// inside it, which is the assertion: the event woke the poll rather than the
// poll timing out.
//
// The second half is what the grace is for (D4, D62, cambia-2019): the stopped
// node commits its final manifest and posts canceled under the token the
// revoking state preserves, so the terminal comes from the node inside the
// grace rather than from the sweep at the end of it, and the output the run
// produced is promoted rather than orphaned. The pool clock never advances
// here, so nothing but the node can settle this job.
func TestCancelReachesTheNodeOnTheEventsPoll(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{
		slots: 1, claimWaitSeconds: 10, pool: poolRigConfig{policy: longLeasePolicy()},
	})
	r.queueFixtureJob(t, "cancel-me", 2, "sleep")

	stop := r.startAgent(t)
	defer stop()
	lease := r.waitForNodePhase(t, "cancel-me", nashnet.PhaseRunning, 30*time.Second)
	pid := r.nodePID(t, "cancel-me", 10*time.Second)

	started := time.Now()
	r.cancelJob(t, "cancel-me", false)

	// The cancel routed through the lease: the coordinator asked, it did not
	// signal (cambia-1724). The handler moves the lease before it answers, and
	// the node needs a round trip to hear about it, so the revoking state is
	// read here rather than after the stop the node posts a terminal for.
	got, ok := r.pool.leases.Get(lease.LeaseID)
	if !ok {
		t.Fatal("the lease is gone: a cancel revokes it, it does not drop it")
	}
	if got.State != nashnet.LeaseRevoking {
		t.Fatalf("lease state = %q after a cancel, want revoking", got.State)
	}

	waitUntil(t, "the node to stop the canceled job", 8*time.Second, func() bool {
		return !processAlive(pid)
	})
	if elapsed := time.Since(started); elapsed > 8*time.Second {
		t.Fatalf("the cancel took %s to reach the node, longer than one held poll", elapsed)
	}
	if n := r.cleanupCount("cancel-me"); n != 0 {
		t.Fatalf("the local cleanup path ran %d times for a leased job", n)
	}

	// The grace admits the wind-down: the node commits a final manifest and
	// posts canceled under the surviving token. No clock advance and no sweep
	// stand behind this terminal.
	waitUntil(t, "the node to post a canceled result", 20*time.Second, func() bool {
		return r.resultState(lease.LeaseID) == nashnet.ResultCanceled
	})
	head, err := r.pool.quar.ReadHead("cancel-me")
	if err != nil {
		t.Fatalf("read the manifest head of the canceled job: %v", err)
	}
	if !head.Folded.Final || head.Digest == "" {
		t.Fatalf("manifest head = %+v, want a final commit: a canceled run still commits (D62)", head.Folded)
	}
	if body := r.promotedBody(t, "cancel-me", "metrics.jsonl"); string(body) != "{\"iter\":0}\n" {
		t.Fatalf("promoted metrics.jsonl = %q: the canceled run's output was orphaned", body)
	}

	// The result finalizes the lease and settles the job.
	done, ok := r.pool.leases.Get(lease.LeaseID)
	if !ok {
		t.Fatal("the lease record is gone: a posted result releases it, it does not delete it")
	}
	if done.State != nashnet.LeaseReleased {
		t.Fatalf("lease state = %q after the node posted its terminal, want released", done.State)
	}
	view, _ := r.disp.resolveView("cancel-me")
	if view.State != nashnet.ResultCanceled {
		t.Fatalf("state = %q after the node's terminal, want canceled", view.State)
	}
}

// TestRevocationMidRunStopsTheProcessGroup is D42's revocation: an operator
// revoking a node ends every lease it holds, the events poll carries the
// revoke, and the node signals the job's whole process group rather than the
// one pid it forked (D36, D60).
func TestRevocationMidRunStopsTheProcessGroup(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{
		slots: 1, claimWaitSeconds: 10, pool: poolRigConfig{policy: longLeasePolicy()},
	})
	r.queueFixtureJob(t, "revoke-me", 2, "sleep")

	stop := r.startAgent(t)
	defer stop()
	r.waitForNodePhase(t, "revoke-me", nashnet.PhaseRunning, 30*time.Second)
	pid := r.nodePID(t, "revoke-me", 10*time.Second)

	resp := r.do(http.MethodPost, "/nashnet/nodes/"+r.node.id+"/revoke", nil)
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		t.Fatalf("revoke: got %d, want 200 or 202", resp.StatusCode)
	}
	resp.Body.Close()

	waitUntil(t, "the revoked node to stop its job group", 15*time.Second, func() bool {
		return !processAlive(pid)
	})
	// The revocation is the coordinator's, so the tombstone outlives the node's
	// own view of itself: a re-register is refused rather than re-admitted.
	again := r.doNode(t, r.node, http.MethodPost, "/nashnet/nodes/register",
		nashnet.RegisterRequest{AgentVersion: "1.1.0", Slots: 1, Capabilities: declaration(1)})
	defer again.Body.Close()
	if again.StatusCode != http.StatusUnauthorized && again.StatusCode != http.StatusForbidden {
		t.Fatalf("a revoked node re-registered: got %d, want 401 or 403", again.StatusCode)
	}
}

// TestGateWindowClosingAppliesEachOnBreach is D42's gate scenario: a usage
// window closes under a running job and the node applies that gate's own
// on_breach (D46, D62). Only stop preempts; drain and finish let the job it is
// already running finish, which is the property that tells the three apart.
func TestGateWindowClosingAppliesEachOnBreach(t *testing.T) {
	cases := []struct {
		name     string
		onBreach gates.OnBreach
		stopped  bool
	}{
		{name: "stop", onBreach: gates.OnBreachStop, stopped: true},
		{name: "drain", onBreach: gates.OnBreachDrain},
		{name: "finish", onBreach: gates.OnBreachFinish},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newNodeRig(t, nodeRigConfig{
				slots:            1,
				claimWaitSeconds: 1,
				pool:             poolRigConfig{policy: longLeasePolicy()},
				gates: gates.Config{Windows: []gates.Window{{
					Days:     []string{"mon", "tue", "wed", "thu", "fri", "sat", "sun"},
					From:     "00:00",
					To:       "12:30",
					TZ:       "UTC",
					OnBreach: tc.onBreach,
				}}},
			})
			job := "gate-" + tc.name
			r.queueFixtureJob(t, job, 2, "sleep")

			stop := r.startAgent(t)
			defer stop()
			lease := r.waitForNodePhase(t, job, nashnet.PhaseRunning, 30*time.Second)
			pid := r.nodePID(t, job, 10*time.Second)

			// The rig clock starts at 12:00 UTC, so forty minutes closes the
			// window under the running job.
			r.clock.advance(40 * time.Minute)

			if tc.stopped {
				waitUntil(t, "the breaching gate to stop the job", 15*time.Second, func() bool {
					return !processAlive(pid)
				})
				waitUntil(t, "the node to post a preempted result", 15*time.Second, func() bool {
					return r.resultState(lease.LeaseID) == nashnet.ResultPreempted
				})
				return
			}

			// drain and finish both leave the running job alone. Several tick
			// intervals pass before the assertion, so a stop that was going to
			// happen has had time to.
			time.Sleep(3 * time.Second)
			if !processAlive(pid) {
				t.Fatalf("on_breach %q stopped the running job: only stop preempts", tc.onBreach)
			}
			if state := r.resultState(lease.LeaseID); state != "" {
				t.Fatalf("on_breach %q posted a %q result for a job still running", tc.onBreach, state)
			}
		})
	}
}

// TestCoordinatorRestartMidUploadResumesFromThePartSize is D42's restart: the
// in-flight part under the quarantine tree is the only upload state that
// crosses a coordinator restart, and the node's HEAD probe finds it and appends
// from there rather than re-sending what already landed (D34, D50).
func TestCoordinatorRestartMidUploadResumesFromThePartSize(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{
		slots: 1, claimWaitSeconds: 1, pool: poolRigConfig{policy: longLeasePolicy()},
	})
	r.queueFixtureJob(t, "restart-upload", 2, "steps", "6")

	stop := r.startAgent(t)
	defer stop()
	lease := r.waitForNodePhase(t, "restart-upload", nashnet.PhaseRunning, 30*time.Second)

	// A step the job has not reached yet, so the part below is written before
	// the node has anything to upload under that digest.
	late := []byte("weights-6")
	digest := sha256Of(late)
	const resumeAt = 5
	part := r.partPath("restart-upload", lease.LeaseID, digest)
	if err := os.MkdirAll(filepath.Dir(part), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(part, late[:resumeAt], 0o644); err != nil {
		t.Fatal(err)
	}

	restartedAt := time.Now()
	r.requests.quiesce(t, func() { r.restartCoordinator(t) })

	waitUntil(t, "the job to promote its last step", 60*time.Second, func() bool {
		body, err := os.ReadFile(filepath.Join(r.coordRunDir("restart-upload"),
			"snapshots", "prtcfr_checkpoint.pt"))
		return err == nil && string(body) == string(late)
	})

	var starts []int64
	for _, row := range r.requests.forPath("/blobs/" + digest) {
		if row.Method == http.MethodPatch && row.At.After(restartedAt) {
			starts = append(starts, row.chunkStart())
		}
	}
	if len(starts) == 0 {
		t.Fatal("no chunk for the late digest crossed the restart")
	}
	if starts[0] != resumeAt {
		t.Fatalf("the first chunk after the restart started at %d, want the part size %d",
			starts[0], resumeAt)
	}
}
