package nashnet

import (
	"path/filepath"
	"testing"
	"time"
)

// TestNackReturnsTheJobWithNoAttemptIncrement is the first row of the D32
// table: a returned claim is a scheduling event, so the next lease for that job
// runs at the same attempt the returned one did.
func TestNackReturnsTheJobWithNoAttemptIncrement(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	first, _ := grant(t, s, "returned-job", "node-a")
	out, err := s.Return(first.LeaseID, NackPrepareNodeFailed)
	if err != nil {
		t.Fatalf("return: %v", err)
	}
	if out.Verdict != VerdictRequeue || out.NextAttempt != first.Attempt {
		t.Fatalf("nack outcome = %+v, want a requeue at attempt %d", out, first.Attempt)
	}
	second, _ := grant(t, s, "returned-job", "node-b")
	if second.Attempt != 1 {
		t.Fatalf("re-claim after a nack ran at attempt %d, want 1", second.Attempt)
	}
}

// TestExpiryBeforeLaunchIncrementsTheAttempt is the second row: a lease that
// never projected a pid requeues at attempt+1, and the number reaches the next
// grant.
func TestExpiryBeforeLaunchIncrementsTheAttempt(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	first, _ := grant(t, s, "expiring-job", "node-a")
	clk.Advance(s.TTL() + time.Second)
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].Verdict != VerdictRequeue || out[0].NextAttempt != 2 {
		t.Fatalf("sweep = %+v, want one requeue at attempt 2", out)
	}
	if rec := leaseJSON(t, s, "expiring-job"); rec["next_attempt"].(float64) != 2 {
		t.Fatalf("persisted next_attempt = %v, want 2", rec["next_attempt"])
	}
	second, _ := grant(t, s, "expiring-job", "node-b")
	if second.Attempt != 2 {
		t.Fatalf("the requeued job ran at attempt %d, want 2", second.Attempt)
	}
	_ = first
}

// TestTheAttemptCountSurvivesACoordinatorRestart pins the reason the count is
// on disk: a fresh store that restores the released record grants the attempt
// the last verdict decided, so max_attempts is not reset by a restart.
func TestTheAttemptCountSurvivesACoordinatorRestart(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	grant(t, s, "restarted-job", "node-a")
	clk.Advance(s.TTL() + time.Second)
	if _, err := s.Sweep(); err != nil {
		t.Fatalf("sweep: %v", err)
	}

	fresh, err := NewLeaseStore(StoreConfig{
		RunsDir: s.cfg.RunsDir, Policy: DefaultPolicy(), Now: clk.Now, Entropy: testEntropy(),
	})
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if restored, _, rerr := fresh.Restore(); rerr != nil || restored != 1 {
		t.Fatalf("restore = %d, %v, want 1 lease and no error", restored, rerr)
	}
	next, _ := grant(t, fresh, "restarted-job", "node-b")
	if next.Attempt != 2 {
		t.Fatalf("the restored job ran at attempt %d, want 2", next.Attempt)
	}
}

// TestANackAfterLaunchFinalizesRatherThanReturning covers the one way the nack
// row and D33 collide: a node that returns a claim after starting the process
// is settled by the phase verdict, because a job that ran is never re-run.
func TestANackAfterLaunchFinalizesRatherThanReturning(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	l, token := grant(t, s, "launched-job", "node-a")
	if _, err := s.Renew(progressFor(l, token, PhaseRunning, true)); err != nil {
		t.Fatalf("progress: %v", err)
	}
	out, err := s.Return(l.LeaseID, NackPrepareNodeFailed)
	if err != nil {
		t.Fatalf("return: %v", err)
	}
	if out.Verdict != VerdictFinalize {
		t.Fatalf("nack after launch = %s, want finalize", out.Verdict)
	}
}

// TestReRegisterWithoutTheLeaseSplitsOnThePhaseReached covers the two
// live_leases rows of D32: the same registration requeues a lease that never
// launched and finalizes one that did, with the record retained either way so
// the verdict table still applies to it.
func TestReRegisterWithoutTheLeaseSplitsOnThePhaseReached(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	pre, _ := grant(t, s, "pre-launch-job", "node-a")
	post, postToken := grant(t, s, "post-launch-job", "node-a")
	if _, err := s.Renew(progressFor(post, postToken, PhaseRunning, true)); err != nil {
		t.Fatalf("progress: %v", err)
	}

	res, err := s.ReBind("node-a", 2, nil)
	if err != nil {
		t.Fatalf("rebind: %v", err)
	}
	if len(res.Revoked) != 2 {
		t.Fatalf("revoked %d leases, want both", len(res.Revoked))
	}
	verdicts := map[string]Outcome{}
	for _, o := range res.Revoked {
		verdicts[o.JobID] = o
	}
	if got := verdicts["pre-launch-job"]; got.Verdict != VerdictRequeue || got.NextAttempt != 2 {
		t.Fatalf("pre-launch re-register = %+v, want a requeue at attempt 2", got)
	}
	if got := verdicts["post-launch-job"]; got.Verdict != VerdictFinalize || got.NextAttempt != 1 {
		t.Fatalf("post-launch re-register = %+v, want a finalize with no increment", got)
	}
	// The records are retained: the finalizer and a replayed result both still
	// resolve the lease they name.
	for _, id := range []string{pre.LeaseID, post.LeaseID} {
		if _, ok := s.Get(id); !ok {
			t.Fatalf("lease %s was dropped rather than retained", id)
		}
	}
	if _, err := ReadLease(filepath.Join(s.cfg.RunsDir, "post-launch-job")); err != nil {
		t.Fatalf("the retained record is not on disk: %v", err)
	}
}

// TestTheSpecRuntimeCapEndsALeaseUnderRenewal is the lease-lifetime row of D32:
// the cap is measured from granted_at regardless of renewals, moves the lease
// to revoking, and then applies the verdict for the phase reached.
func TestTheSpecRuntimeCapEndsALeaseUnderRenewal(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	l, token, err := s.Grant(GrantRequest{
		JobID: "capped-job", NodeID: "node-a", NodeEpoch: 1, MaxRuntime: 30 * time.Minute,
	})
	if err != nil {
		t.Fatalf("grant: %v", err)
	}
	// Renew past the cap: a node posting progress forever must not hold a lease
	// forever.
	for clk.Now().Before(l.GrantedAt.Add(31 * time.Minute)) {
		clk.Advance(s.TTL() / 2)
		if _, rerr := s.Renew(progressFor(l, token, PhaseRunning, true)); rerr != nil {
			t.Fatalf("progress: %v", rerr)
		}
		if out, serr := s.Sweep(); serr != nil {
			t.Fatalf("sweep: %v", serr)
		} else if len(out) > 0 {
			if out[0].State != LeaseRevoking || out[0].Reason != ReasonRuntimeCap {
				t.Fatalf("cap outcome = %+v, want revoking on the runtime cap", out[0])
			}
			break
		}
	}
	held, _ := s.Get(l.LeaseID)
	if held.State != LeaseRevoking {
		t.Fatalf("lease state at the cap = %s, want revoking", held.State)
	}
	// The grace runs out with no final commit: the phase reached decides, and
	// this one ran.
	clk.Advance(2 * s.TTL())
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].Verdict != VerdictFinalize || out[0].Reason != ReasonGraceElapsed {
		t.Fatalf("after the grace = %+v, want a finalize on the elapsed grace", out)
	}
}
