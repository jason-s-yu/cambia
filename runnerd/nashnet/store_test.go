package nashnet

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGrantWritesLeaseFileAndIncrementsTheEpoch(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	l, token := grant(t, s, "v0.4-prtcfr-r13", "node-a")
	if l.LeaseEpoch != 1 {
		t.Fatalf("first grant epoch = %d, want 1", l.LeaseEpoch)
	}
	if l.State != LeaseActive || l.Phase != PhaseClaimed || l.Attempt != 1 {
		t.Fatalf("unexpected fresh lease: %+v", l)
	}
	if !l.Deadline.Equal(clk.Now().Add(s.TTL())) {
		t.Fatalf("deadline = %s, want now plus the TTL", l.Deadline)
	}

	rec := leaseJSON(t, s, "v0.4-prtcfr-r13")
	if rec["lease_epoch"].(float64) != 1 {
		t.Fatalf("persisted lease_epoch = %v, want 1", rec["lease_epoch"])
	}
	if rec["state"] != LeaseActive || rec["node_id"] != "node-a" || rec["job_id"] != "v0.4-prtcfr-r13" {
		t.Fatalf("persisted record does not name the placement: %v", rec)
	}
	if rec["token_hash"] != HashLeaseToken(token) {
		t.Fatalf("the record must store the token hash, not the token")
	}
	if rec["token_hash"] == token {
		t.Fatalf("the record stored the token itself")
	}
	if rec["stop_requested_at"] != "" {
		t.Fatalf("a fresh lease carries no operator stop witness, got %v", rec["stop_requested_at"])
	}

	// A second grant for the same job is refused while the first is live, and
	// the epoch advances only once the lease has ended.
	if _, _, err := s.Grant(GrantRequest{JobID: "v0.4-prtcfr-r13", NodeID: "node-b", NodeEpoch: 1}); !errors.Is(err, ErrJobLeased) {
		t.Fatalf("grant over a live lease = %v, want ErrJobLeased", err)
	}
	if _, err := s.Release(l.LeaseID); err != nil {
		t.Fatalf("release: %v", err)
	}
	second, _ := grant(t, s, "v0.4-prtcfr-r13", "node-b")
	if second.LeaseEpoch != 2 {
		t.Fatalf("second grant epoch = %d, want 2", second.LeaseEpoch)
	}
	if rec := leaseJSON(t, s, "v0.4-prtcfr-r13"); rec["lease_epoch"].(float64) != 2 {
		t.Fatalf("persisted lease_epoch after the re-grant = %v, want 2", rec["lease_epoch"])
	}
}

func TestGrantRefusesAJobIDThatEscapesTheRunsDir(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	for _, bad := range []string{"", "../escape", "sub/dir", `back\slash`} {
		if _, _, err := s.Grant(GrantRequest{JobID: bad, NodeID: "node-a"}); !errors.Is(err, ErrInvalidJobID) {
			t.Errorf("grant for job id %q = %v, want ErrInvalidJobID", bad, err)
		}
	}
}

func TestRenewRefusesAStaleEpochOrAWrongToken(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")

	clk.Advance(40 * time.Second)
	got, err := s.Renew(ProgressUpdate{
		LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch, NodeEpoch: l.NodeEpoch,
		Token: token, Phase: PhaseRunning, PIDProjected: true,
	})
	if err != nil {
		t.Fatalf("renew: %v", err)
	}
	if !got.Deadline.Equal(clk.Now().Add(s.TTL())) {
		t.Fatalf("renew did not extend the deadline: %s", got.Deadline)
	}
	if got.Phase != PhaseRunning || !got.PIDProjected {
		t.Fatalf("renew did not record the phase and the pid witness: %+v", got)
	}

	cases := map[string]ProgressUpdate{
		"stale lease epoch":  {LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch - 1, NodeEpoch: l.NodeEpoch, Token: token, Phase: PhaseRunning},
		"ahead lease epoch":  {LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch + 1, NodeEpoch: l.NodeEpoch, Token: token, Phase: PhaseRunning},
		"absent lease epoch": {LeaseID: l.LeaseID, NodeEpoch: l.NodeEpoch, Token: token, Phase: PhaseRunning},
		"stale node epoch":   {LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch, NodeEpoch: l.NodeEpoch + 1, Token: token, Phase: PhaseRunning},
		"absent node epoch":  {LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch, Token: token, Phase: PhaseRunning},
		"wrong token":        {LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch, NodeEpoch: l.NodeEpoch, Token: "not-the-token", Phase: PhaseRunning},
	}
	for name, u := range cases {
		if _, err := s.Renew(u); !errors.Is(err, ErrLeaseSuperseded) {
			t.Errorf("renew with a %s = %v, want ErrLeaseSuperseded", name, err)
		}
	}

	if _, err := s.Renew(ProgressUpdate{LeaseID: "01NOSUCHLEASE0000000000000", Token: token}); !errors.Is(err, ErrUnknownLease) {
		t.Errorf("renew of an unknown lease = %v, want ErrUnknownLease", err)
	}
	if _, err := s.Renew(progressFor(l, token, PhaseStopping, false)); !errors.Is(err, ErrInvalidPhase) {
		t.Errorf("a node reporting stopping = %v, want ErrInvalidPhase", err)
	}
}

func TestRenewOnADroppedTokenIsNotASupersededLease(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")

	if _, err := s.Revoke(l.LeaseID, ReasonNodeRevoked); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	_, err := s.Renew(progressFor(l, token, PhaseFetching, false))
	if !errors.Is(err, ErrLeaseTokenDropped) {
		t.Fatalf("renew after revocation = %v, want ErrLeaseTokenDropped, which the server answers 401 (D36)", err)
	}
}

func TestRevokingKeepsTheTokenAndAdmitsOnlyTheWindDownRoutes(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")
	if _, err := s.Renew(progressFor(l, token, PhaseRunning, true)); err != nil {
		t.Fatalf("renew: %v", err)
	}

	rev, err := s.BeginRevoking(l.LeaseID, true)
	if err != nil {
		t.Fatalf("begin revoking: %v", err)
	}
	if rev.State != LeaseRevoking || rev.LeaseEpoch != l.LeaseEpoch || rev.TokenHash == "" {
		t.Fatalf("revoking must hold the epoch and the token: %+v", rev)
	}
	if rev.StopRequestedAt.IsZero() {
		t.Fatalf("an operator stop must record stop_requested_at")
	}
	if !rev.Deadline.Equal(clk.Now().Add(s.cfg.RevokeGrace)) {
		t.Fatalf("the grace window is not on the deadline: %s", rev.Deadline)
	}

	// A progress post is admitted, records its facts, and renews nothing.
	clk.Advance(30 * time.Second)
	got, err := s.Renew(progressFor(l, token, PhaseUploading, false))
	if err != nil {
		t.Fatalf("progress while revoking: %v", err)
	}
	if !got.Deadline.Equal(rev.Deadline) {
		t.Fatalf("progress renewed a revoking lease: %s", got.Deadline)
	}

	for _, route := range []Route{RouteLogs, RouteBlobs, RouteManifestFinal, RouteResult, RouteProgress} {
		if _, err := s.Fence(l.LeaseID, l.LeaseEpoch, l.NodeEpoch, token, route); err != nil {
			t.Errorf("route %s must be admitted while revoking: %v", route, err)
		}
	}
	for _, route := range []Route{RouteManifest, RouteNack, RouteSnapshot, RouteSeeds} {
		if _, err := s.Fence(l.LeaseID, l.LeaseEpoch, l.NodeEpoch, token, route); !errors.Is(err, ErrLeaseSuperseded) {
			t.Errorf("route %s while revoking = %v, want ErrLeaseSuperseded", route, err)
		}
	}

	if !got.PermitsResult(ResultCanceled) || !got.PermitsResult(ResultPreempted) {
		t.Errorf("a revoking lease must accept canceled and preempted")
	}
	if got.PermitsResult(ResultStopped) || got.PermitsResult(ResultCrashed) {
		t.Errorf("a revoking lease must not accept a clean stop or a crash")
	}

	// The job cannot be re-claimed while the lease is revoking.
	if _, _, err := s.Grant(GrantRequest{JobID: "job-a", NodeID: "node-b"}); !errors.Is(err, ErrJobLeased) {
		t.Fatalf("grant over a revoking lease = %v, want ErrJobLeased", err)
	}
}

func TestReleaseDropsTheTokenAndHoldsTheEpoch(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")

	rel, err := s.Release(l.LeaseID)
	if err != nil {
		t.Fatalf("release: %v", err)
	}
	if rel.State != LeaseReleased || rel.TokenHash != "" || rel.LeaseEpoch != l.LeaseEpoch {
		t.Fatalf("release must drop the token and hold the epoch: %+v", rel)
	}
	if _, err := s.Fence(l.LeaseID, l.LeaseEpoch, l.NodeEpoch, token, RouteResult); !errors.Is(err, ErrLeaseTokenDropped) {
		t.Fatalf("a released lease answers %v, want ErrLeaseTokenDropped", err)
	}
	if rec := leaseJSON(t, s, "job-a"); rec["state"] != LeaseReleased || rec["token_hash"] != "" {
		t.Fatalf("the released record is not persisted: %v", rec)
	}
}

func TestReBindKeepsTheNamedLeasesAndRevokesTheRest(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)

	kept, keptToken := grant(t, s, "job-kept", "node-a")
	dropped, _ := grant(t, s, "job-dropped", "node-a")
	ran, ranToken := grant(t, s, "job-ran", "node-a")
	other, otherToken := grant(t, s, "job-other", "node-b")

	// job-ran reached a pid, so its revocation goes to the finalizer rather than
	// back to the queue.
	if _, err := s.Renew(progressFor(ran, ranToken, PhaseRunning, true)); err != nil {
		t.Fatalf("renew job-ran: %v", err)
	}

	res, err := s.ReBind("node-a", 4, []LiveLease{
		{LeaseID: kept.LeaseID, TokenHash: HashLeaseToken(keptToken)},
		{LeaseID: other.LeaseID, TokenHash: HashLeaseToken(otherToken)},
		{LeaseID: "01NOSUCHLEASE0000000000000", TokenHash: "deadbeef"},
	})
	if err != nil {
		t.Fatalf("rebind: %v", err)
	}

	if len(res.Rebound) != 1 || res.Rebound[0] != kept.LeaseID {
		t.Fatalf("rebound = %v, want only the named lease of node-a", res.Rebound)
	}
	if len(res.Refused) != 2 {
		t.Fatalf("refused = %v, want the other node's lease and the unknown one", res.Refused)
	}
	if l, _ := s.Get(kept.LeaseID); l.NodeEpoch != 4 || l.State != LeaseActive || l.TokenHash == "" {
		t.Fatalf("the re-bound lease did not take the new node epoch: %+v", l)
	}
	if l, _ := s.Get(other.LeaseID); l.State != LeaseActive || l.NodeEpoch != 1 {
		t.Fatalf("node-b's lease must be untouched by node-a's registration: %+v", l)
	}

	verdicts := map[string]Outcome{}
	for _, o := range res.Revoked {
		verdicts[o.JobID] = o
	}
	if len(verdicts) != 2 {
		t.Fatalf("revoked = %v, want job-dropped and job-ran", res.Revoked)
	}
	if got := verdicts["job-dropped"]; got.Verdict != VerdictRequeue || got.NextAttempt != 2 || got.Reason != ReasonNodeReRegister {
		t.Fatalf("a lease that never launched must requeue with attempt++: %+v", got)
	}
	if got := verdicts["job-ran"]; got.Verdict != VerdictFinalize || got.NextAttempt != 1 {
		t.Fatalf("a lease whose pid was projected must go to the finalizer: %+v", got)
	}
	if l, _ := s.Get(dropped.LeaseID); l.State != LeaseReleased || l.TokenHash != "" || l.LeaseEpoch != dropped.LeaseEpoch+1 {
		t.Fatalf("a revoked lease must bump its epoch and drop its token: %+v", l)
	}
	if l, ok := s.Get(ran.LeaseID); !ok || l.JobID != "job-ran" {
		t.Fatalf("the record of an executed lease must be retained for the finalizer")
	}
}

func TestReBindRefusesAWrongTokenHash(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, _ := grant(t, s, "job-a", "node-a")

	res, err := s.ReBind("node-a", 2, []LiveLease{{LeaseID: l.LeaseID, TokenHash: HashLeaseToken("guessed")}})
	if err != nil {
		t.Fatalf("rebind: %v", err)
	}
	if len(res.Rebound) != 0 || len(res.Refused) != 1 || len(res.Revoked) != 1 {
		t.Fatalf("a wrong hash must refuse the entry and revoke the lease: %+v", res)
	}
}

func TestRevokeNodeEndsEveryLeaseWithNoGrace(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	a, _ := grant(t, s, "job-a", "node-a")
	b, _ := grant(t, s, "job-b", "node-a")
	keep, _ := grant(t, s, "job-c", "node-b")

	out, err := s.RevokeNode("node-a")
	if err != nil {
		t.Fatalf("revoke node: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("revoked %d leases, want 2", len(out))
	}
	for _, o := range out {
		if o.Reason != ReasonNodeRevoked || o.State != LeaseReleased {
			t.Errorf("unexpected outcome: %+v", o)
		}
	}
	for _, id := range []string{a.LeaseID, b.LeaseID} {
		if l, _ := s.Get(id); l.TokenHash != "" || l.State != LeaseReleased {
			t.Errorf("lease %s survived the node revocation: %+v", id, l)
		}
	}
	if l, _ := s.Get(keep.LeaseID); l.State != LeaseActive {
		t.Errorf("node-b's lease must survive node-a's revocation")
	}
}

func TestRestoreRebuildsPlacementAcrossACoordinatorRestart(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	l, token := grant(t, s, "job-a", "node-a")
	full := progressFor(l, token, PhaseRunning, true)
	full.ManifestSeq, full.ManifestDigest = 4, "abc123"
	if _, err := s.Renew(full); err != nil {
		t.Fatalf("renew: %v", err)
	}
	// A directory with no lease.json is skipped rather than counted.
	if err := os.MkdirAll(filepath.Join(s.cfg.RunsDir, "job-local"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	restarted, err := NewLeaseStore(StoreConfig{RunsDir: s.cfg.RunsDir, Policy: DefaultPolicy(), Now: clk.Now})
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	n, skipped, err := restarted.Restore()
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if n != 1 || skipped != 0 {
		t.Fatalf("restore = (%d, %d), want (1, 0)", n, skipped)
	}

	got, ok := restarted.Get(l.LeaseID)
	if !ok {
		t.Fatalf("the restored store does not know the lease")
	}
	if got.JobID != "job-a" || got.NodeID != "node-a" || got.LeaseEpoch != l.LeaseEpoch {
		t.Fatalf("restored lease does not match: %+v", got)
	}
	if !got.PIDProjected || got.Phase != PhaseRunning || got.ManifestSeq != 4 {
		t.Fatalf("the restored record lost the facts the D7 verdict reads: %+v", got)
	}
	if got.TokenHash != HashLeaseToken(token) {
		t.Fatalf("the restored record lost the token hash, so the node would be fenced out")
	}
	// The node notices nothing: it keeps posting under the same lease epoch and
	// token, and the coordinator holds no registry epoch to compare until the
	// node registers again, which is what SkipEpoch says.
	if _, err := restarted.Renew(ProgressUpdate{
		LeaseID: l.LeaseID, LeaseEpoch: l.LeaseEpoch, NodeEpoch: SkipEpoch,
		Token: token, Phase: PhaseRunning,
	}); err != nil {
		t.Fatalf("progress after the restart: %v", err)
	}
	if floor := restarted.NodeEpochFloor("node-a"); floor != l.NodeEpoch {
		t.Fatalf("node epoch floor = %d, want %d", floor, l.NodeEpoch)
	}
}

func TestLiveCountsAreScopedToTheNode(t *testing.T) {
	clk := newClock(t, "2026-09-01T12:00:00Z")
	s := newStore(t, clk)
	grant(t, s, "job-a", "node-a")
	b, _ := grant(t, s, "job-b", "node-a")
	grant(t, s, "job-c", "node-b")

	if got := s.LiveCountForNode("node-a"); got != 2 {
		t.Fatalf("node-a holds %d live leases, want 2", got)
	}
	if _, err := s.Release(b.LeaseID); err != nil {
		t.Fatalf("release: %v", err)
	}
	if got := s.LiveCountForNode("node-a"); got != 1 {
		t.Fatalf("node-a holds %d live leases after a release, want 1", got)
	}
	if got := len(s.LiveForNode("node-b")); got != 1 {
		t.Fatalf("node-b holds %d live leases, want 1", got)
	}
}

func TestGrantSetFencesReads(t *testing.T) {
	gs := GrantSet{
		Snapshot: "abc123",
		Seeds:    map[string][]GrantEntry{"resume": {{Path: "snapshots/ck.pt", SHA256: "def456"}}},
	}
	if digest, ok := gs.Allows("resume", "snapshots/ck.pt"); !ok || digest != "def456" {
		t.Fatalf("an in-set path must resolve to its digest, got (%q, %v)", digest, ok)
	}
	for _, c := range [][2]string{
		{"resume", "snapshots/other.pt"},
		{"other", "snapshots/ck.pt"},
		{"resume", "../escape"},
	} {
		if _, ok := gs.Allows(c[0], c[1]); ok {
			t.Errorf("seed %q path %q is outside the grant set and must not resolve", c[0], c[1])
		}
	}
}
