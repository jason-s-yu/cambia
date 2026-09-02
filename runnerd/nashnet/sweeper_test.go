package nashnet

import (
	"testing"
	"time"
)

// TestD7VerdictTable walks the expiry table of D7 row by row: the verdict is a
// function of the furthest phase reached and whether a pid was ever projected,
// with the operator-stop row keyed on the lease record rather than on any
// status string a node can write.
func TestD7VerdictTable(t *testing.T) {
	stop := mustTime(t, "2026-09-01T12:00:00Z")
	cases := []struct {
		name    string
		lease   Lease
		want    Verdict
		comment string
	}{
		{"claimed, no pid", Lease{Phase: PhaseClaimed}, VerdictRequeue, "row 1"},
		{"fetching, no pid", Lease{Phase: PhaseFetching}, VerdictRequeue, "row 1"},
		{"preparing, no pid", Lease{Phase: PhasePreparing}, VerdictRequeue, "row 1"},
		{"running, pid", Lease{Phase: PhaseRunning, PIDProjected: true}, VerdictFinalize, "row 2"},
		{"stopping, pid", Lease{Phase: PhaseStopping, PIDProjected: true}, VerdictFinalize, "row 2"},
		{"uploading, pid", Lease{Phase: PhaseUploading, PIDProjected: true}, VerdictFinalize, "row 2"},
		{"committing, pid", Lease{Phase: PhaseCommitting, PIDProjected: true}, VerdictFinalize, "row 2"},
		{"operator stop, running, pid", Lease{Phase: PhaseRunning, PIDProjected: true, StopRequestedAt: stop}, VerdictCanceled, "row 3"},
		{"operator stop, committing, pid", Lease{Phase: PhaseCommitting, PIDProjected: true, StopRequestedAt: stop}, VerdictCanceled, "row 3"},
		// Two readings the table leaves implicit and the code fixes: an operator
		// stop of a lease that never launched is still a cancel, never a requeue
		// of the job the operator just stopped; and a post-launch phase with no
		// pid recorded yet takes the conservative finalize rather than a second
		// execution.
		{"operator stop, fetching, no pid", Lease{Phase: PhaseFetching, StopRequestedAt: stop}, VerdictCanceled, "row 3 precedence"},
		{"running, no pid recorded", Lease{Phase: PhaseRunning}, VerdictFinalize, "conservative row 2"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.lease.Verdict(); got != c.want {
				t.Fatalf("%s: verdict = %q, want %q", c.comment, got, c.want)
			}
		})
	}
}

func TestSweepRequeuesALeaseThatNeverLaunched(t *testing.T) {
	for _, phase := range []string{PhaseClaimed, PhaseFetching, PhasePreparing} {
		t.Run(phase, func(t *testing.T) {
			clk := newClock(t, "2026-09-01T12:00:00Z")
			s := newStore(t, clk)
			l, token := grant(t, s, "job-a", "node-a")
			if _, err := s.Renew(progressFor(l, token, phase, false)); err != nil {
				t.Fatalf("renew: %v", err)
			}

			if out, err := s.Sweep(); err != nil || len(out) != 0 {
				t.Fatalf("a lease inside its deadline must not be swept: %v %v", out, err)
			}

			cur, _ := s.Get(l.LeaseID)
			advanceToExpiry(clk, cur)
			out, err := s.Sweep()
			if err != nil {
				t.Fatalf("sweep: %v", err)
			}
			if len(out) != 1 {
				t.Fatalf("swept %d leases, want 1", len(out))
			}
			got := out[0]
			if got.Verdict != VerdictRequeue || got.NextAttempt != 2 || got.Reason != ReasonExpired {
				t.Fatalf("expired pre-launch lease: %+v, want a requeue at attempt 2", got)
			}
			if got.JobID != "job-a" || got.NodeID != "node-a" || got.LeaseEpoch != l.LeaseEpoch+1 {
				t.Fatalf("outcome does not name the placement or bump the epoch: %+v", got)
			}
			if after, _ := s.Get(l.LeaseID); after.TokenHash != "" || after.State != LeaseReleased {
				t.Fatalf("an expired lease must lose its token: %+v", after)
			}
			if second, err := s.Sweep(); err != nil || len(second) != 0 {
				t.Fatalf("a released lease must not be swept twice: %v %v", second, err)
			}
		})
	}
}

func TestSweepFinalizesALeaseThatRan(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")
	if _, err := s.Renew(progressFor(l, token, PhaseRunning, true)); err != nil {
		t.Fatalf("renew: %v", err)
	}

	cur, _ := s.Get(l.LeaseID)
	advanceToExpiry(clk, cur)
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].Verdict != VerdictFinalize {
		t.Fatalf("a lease that ran must finalize by two witnesses: %+v", out)
	}
	if out[0].NextAttempt != 1 {
		t.Fatalf("a job that ran is never retried, got attempt %d", out[0].NextAttempt)
	}
}

func TestSweepRecordsAnOperatorStopAsCanceled(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")
	if _, err := s.Renew(progressFor(l, token, PhaseRunning, true)); err != nil {
		t.Fatalf("renew: %v", err)
	}
	if _, err := s.BeginRevoking(l.LeaseID, true); err != nil {
		t.Fatalf("begin revoking: %v", err)
	}

	// Inside the grace the lease is untouched: the node still has a credential
	// with which to commit a final manifest and post its terminal.
	clk.Advance(s.cfg.RevokeGrace - time.Second)
	if out, err := s.Sweep(); err != nil || len(out) != 0 {
		t.Fatalf("the grace period must not be swept: %v %v", out, err)
	}
	clk.Advance(2 * time.Second)
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].Verdict != VerdictCanceled || out[0].Reason != ReasonGraceElapsed {
		t.Fatalf("an unanswered operator stop must settle as canceled: %+v", out)
	}
}

// TestSweepAppliesTheRuntimeCapUnderContinuousRenewals is the coordinator-side
// runtime bound of D4: a node that posts progress forever must not hold its
// lease forever.
func TestSweepAppliesTheRuntimeCapUnderContinuousRenewals(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s, err := NewLeaseStore(StoreConfig{
		RunsDir: t.TempDir(),
		Policy:  Policy{LeaseTTLSeconds: 60, MaxLeaseSeconds: 300},
		Now:     clk.Now,
		Entropy: testEntropy(),
	})
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	l, token := grant(t, s, "job-a", "node-a")

	var capped Outcome
	for i := 0; i < 20; i++ {
		clk.Advance(20 * time.Second)
		_, renewErr := s.Renew(progressFor(l, token, PhaseRunning, true))
		out, err := s.Sweep()
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if len(out) > 0 {
			capped = out[0]
			if renewErr != nil {
				t.Fatalf("the renewal before the cap should have succeeded: %v", renewErr)
			}
			break
		}
		if renewErr != nil {
			t.Fatalf("renew: %v", renewErr)
		}
	}
	if capped.State != LeaseRevoking || capped.Reason != ReasonRuntimeCap {
		t.Fatalf("the runtime cap must move the lease to revoking: %+v", capped)
	}
	elapsed := clk.Now().Sub(l.GrantedAt)
	if elapsed < 300*time.Second || elapsed > 320*time.Second {
		t.Fatalf("the cap fired after %s, want it at the 300s mark", elapsed)
	}

	// The grace then runs out and the D7 verdict for the phase reached applies.
	clk.Advance(s.cfg.RevokeGrace + time.Second)
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].Verdict != VerdictFinalize || out[0].Reason != ReasonGraceElapsed {
		t.Fatalf("after the cap grace the verdict must be the D7 one: %+v", out)
	}
}

func TestSpecRuntimeCapLowersThePoolCap(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, _, err := s.Grant(GrantRequest{
		JobID: "job-a", NodeID: "node-a", NodeEpoch: 1, MaxRuntime: time.Hour,
	})
	if err != nil {
		t.Fatalf("grant: %v", err)
	}
	if l.RuntimeCapReached(clk.Now().Add(59*time.Minute), s.poolRuntimeCap()) {
		t.Fatalf("the spec cap fired early")
	}
	if !l.RuntimeCapReached(clk.Now().Add(61*time.Minute), s.poolRuntimeCap()) {
		t.Fatalf("the smaller of the two caps must govern")
	}
	// With no spec cap the pool's 72h ceiling governs.
	bare := Lease{GrantedAt: clk.Now()}
	if bare.RuntimeCapReached(clk.Now().Add(71*time.Hour), s.poolRuntimeCap()) {
		t.Fatalf("the pool cap fired before 72h")
	}
	if !bare.RuntimeCapReached(clk.Now().Add(73*time.Hour), s.poolRuntimeCap()) {
		t.Fatalf("the pool cap did not fire at 72h")
	}
}

// TestSweepGivesNoGraceToALeaseNobodyIsRenewing pins the ordering inside the
// pass: a lease past both its deadline and the runtime cap is revoked outright,
// since the wind-down window exists for a node that is still answering.
func TestSweepGivesNoGraceToALeaseNobodyIsRenewing(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s, err := NewLeaseStore(StoreConfig{
		RunsDir: t.TempDir(),
		Policy:  Policy{LeaseTTLSeconds: 60, MaxLeaseSeconds: 120},
		Now:     clk.Now,
		Entropy: testEntropy(),
	})
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	l, token := grant(t, s, "job-a", "node-a")
	if _, err := s.Renew(progressFor(l, token, PhaseRunning, true)); err != nil {
		t.Fatalf("renew: %v", err)
	}

	clk.Advance(200 * time.Second)
	out, err := s.Sweep()
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(out) != 1 || out[0].State != LeaseReleased || out[0].Reason != ReasonExpired {
		t.Fatalf("a silent lease past both bounds must be revoked outright: %+v", out)
	}
	if out[0].Verdict != VerdictFinalize {
		t.Fatalf("verdict = %q, want finalize", out[0].Verdict)
	}
}

func TestSweeperTickDispatchesOutcomesAtQuarterTTL(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, _ := grant(t, s, "job-a", "node-a")

	var seen []Outcome
	sw := NewSweeper(s, func(o Outcome) { seen = append(seen, o) }, nil)
	if sw.Interval() != s.TTL()/4 {
		t.Fatalf("sweep interval = %s, want a quarter of the TTL", sw.Interval())
	}
	if got := sw.Tick(); len(got) != 0 || len(seen) != 0 {
		t.Fatalf("a live lease must not be handled: %v", got)
	}

	advanceToExpiry(clk, l)
	got := sw.Tick()
	if len(got) != 1 || len(seen) != 1 || seen[0].JobID != "job-a" {
		t.Fatalf("the sweeper did not hand the outcome to its handler: %v %v", got, seen)
	}
}
