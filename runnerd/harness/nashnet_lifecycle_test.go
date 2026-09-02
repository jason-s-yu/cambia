package harness

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// leasedJob queues a job, claims it on node A, and reports it running with a
// pid, which is the state every lease-aware cancel test starts from: a lease in
// a launched phase whose process this daemon never forked.
func (r *poolRig) leasedJob(t *testing.T, name string) *nashnet.ClaimResponse {
	t.Helper()
	r.queueJob(t, JobSpec{Name: name})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatalf("expected a claim for %s", name)
	}
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhaseRunning, PID: 4242})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("progress for %s: got %d, want 200", name, resp.StatusCode)
	}
	resp.Body.Close()
	return claimed
}

// cancelJob issues the operator DELETE that a lease-aware cancel routes.
func (r *poolRig) cancelJob(t *testing.T, name string, force bool) {
	t.Helper()
	path := "/harness/jobs/" + name
	if force {
		path += "?force=true"
	}
	resp := r.do(http.MethodDelete, path, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("cancel %s: got %d, want 200", name, resp.StatusCode)
	}
}

// cleanupCount reports how many times the local terminal path called Cleanup
// for a job. The revoke path calls it never, so it is the witness that no local
// stop ran.
func (r *poolRig) cleanupCount(name string) int {
	r.env.mu.Lock()
	defer r.env.mu.Unlock()
	n := 0
	for _, id := range r.env.cleanups {
		if id == name {
			n++
		}
	}
	return n
}

// TestCancelOfALeasedJobRevokesRatherThanSignalling is AC(1) and AC(5): the
// cancel sends nothing locally, moves the lease to revoking with
// stop_requested_at recorded, writes the stopping projection, and the node
// reads {revoke, force} off its held events request within one round trip.
func TestCancelOfALeasedJobRevokesRatherThanSignalling(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	claimed := r.leasedJob(t, "revoke-job")

	events := make(chan *http.Response, 1)
	go func() {
		events <- r.doNode(t, r.nodeA, http.MethodGet,
			"/nashnet/nodes/"+r.nodeA.id+"/events?wait_seconds=5", nil)
	}()
	waitFor(t, func() bool { return r.pool.sessionCount() == 1 })

	r.cancelJob(t, "revoke-job", true)

	var out nashnet.EventsResponse
	decodeInto(t, <-events, &out)
	if len(out.Events) != 1 {
		t.Fatalf("events = %+v, want exactly the revoke", out.Events)
	}
	ev := out.Events[0]
	if ev.Type != nashnet.EventRevoke || ev.LeaseID != claimed.LeaseID {
		t.Fatalf("event = %+v, want a revoke for %s", ev, claimed.LeaseID)
	}
	if !ev.Force {
		t.Fatal("?force=true did not reach the node as force: true")
	}

	lease, ok := r.pool.leases.Get(claimed.LeaseID)
	if !ok || lease.State != nashnet.LeaseRevoking {
		t.Fatalf("lease state = %q, want revoking", lease.State)
	}
	if lease.StopRequestedAt.IsZero() {
		t.Fatal("stop_requested_at was not recorded, so the D7 verdict cannot read the operator stop")
	}
	if lease.LeaseEpoch != claimed.LeaseEpoch {
		t.Fatalf("lease epoch advanced to %d during the grace, want %d",
			lease.LeaseEpoch, claimed.LeaseEpoch)
	}

	st := readProcessState(t, r.runsDir, "revoke-job")
	if st.Status != procmgr.StatusStopping {
		t.Fatalf("projection = %q, want stopping (canceled means the local queued path ran)", st.Status)
	}
	if st.Host != r.nodeA.id || st.PGID != 0 {
		t.Fatalf("projection host=%q pgid=%d, want the node id and pgid 0", st.Host, st.PGID)
	}
	if n := r.cleanupCount("revoke-job"); n != 0 {
		t.Fatalf("Cleanup ran %d times, so a local terminal path was consulted", n)
	}

	// The second delivery path of D31 carries the same pair.
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhaseRunning, PID: 4242})
	var prog nashnet.ProgressResponse
	decodeInto(t, resp, &prog)
	if !prog.Revoke || !prog.Force {
		t.Fatalf("progress response = %+v, want revoke and force", prog)
	}
}

// TestRevokingLeaseBlocksAReClaim is AC(3): while the lease winds down the job
// is not handed to anyone, which is what keeps a superseded writer and a fresh
// lease off one run dir.
func TestRevokingLeaseBlocksAReClaim(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.register(t, r.nodeB, 1)
	r.leasedJob(t, "reclaim-job")
	r.cancelJob(t, "reclaim-job", false)

	resp, claimed := r.claim(t, r.nodeB, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatalf("node-b claimed %s while its lease was revoking", claimed.JobID)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("claim during the grace: got %d, want 204", resp.StatusCode)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNoMatch {
		t.Fatalf("hold = %q, want no_match", hold)
	}
}

// TestGraceAcceptsTheFinalCommitAndRefusesItAfterwards is AC(2): a final
// manifest and a canceled result posted under the surviving token during the
// grace are accepted, and the same calls after the grace ends are refused
// because the token died with the lease.
func TestGraceAcceptsTheFinalCommitAndRefusesItAfterwards(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	claimed := r.leasedJob(t, "grace-job")
	base := "/nashnet/leases/" + claimed.LeaseID
	r.cancelJob(t, "grace-job", false)

	content := []byte("metrics row\n")
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, content)
	entries := []quarantine.Entry{{
		Path: "metrics.jsonl", Digest: digest, Size: int64(len(content)),
		MTime: r.clock.now().UnixNano(),
	}}

	// The grace grants a final commit and nothing else: an ordinary rolling
	// commit on a revoking lease is still superseded (D4).
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest",
		quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1, Entries: entries,
		})
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("non-final manifest during the grace: got %d, want 409", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeLeaseSuperseded {
		t.Fatalf("non-final manifest code = %q, want lease_superseded", code)
	}

	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest",
		quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1, Final: true,
			Entries: entries,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final manifest during the grace: got %d, want 200", resp.StatusCode)
	}
	var committed quarantine.CommitResponse
	decodeInto(t, resp, &committed)

	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/result",
		nashnet.ResultRequest{
			LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultCanceled,
			FinalManifestDigest: committed.Digest,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("canceled result during the grace: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if st := readProcessState(t, r.runsDir, "grace-job"); st.Status != nashnet.ResultCanceled {
		t.Fatalf("terminal = %q, want canceled", st.Status)
	}

	// A second job whose node stays silent through the grace loses its token,
	// so the same two calls are refused after the sweep ends the lease.
	lapsed := r.leasedJob(t, "lapsed-job")
	lapsedBase := "/nashnet/leases/" + lapsed.LeaseID
	r.cancelJob(t, "lapsed-job", false)
	r.clock.advance(2 * r.pool.leases.TTL())
	r.pool.Sweeper().Tick()

	resp = r.doLease(t, lapsed.LeaseToken, http.MethodPost, lapsedBase+"/manifest",
		quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: lapsed.LeaseEpoch, Seq: 1, Final: true,
		})
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("final manifest after the grace: got %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()
	resp = r.doLease(t, lapsed.LeaseToken, http.MethodPost, lapsedBase+"/result",
		nashnet.ResultRequest{LeaseEpoch: lapsed.LeaseEpoch, State: nashnet.ResultCanceled})
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("result after the grace: got %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestSilentNodeIsFinalizedByTheSweep is AC(4): a node that never answers the
// revoke is settled by the expiry sweep against the operator-stop witness the
// coordinator wrote, as canceled with no exit code and no retry.
func TestSilentNodeIsFinalizedByTheSweep(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	claimed := r.leasedJob(t, "silent-job")
	r.cancelJob(t, "silent-job", false)

	r.clock.advance(2 * r.pool.leases.TTL())
	out := r.pool.Sweeper().Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictCanceled {
		t.Fatalf("sweep outcomes = %+v, want one canceled verdict", out)
	}
	if out[0].NextAttempt != claimed.Attempt {
		t.Fatalf("attempt advanced to %d, want %d: an operator stop is not a retry",
			out[0].NextAttempt, claimed.Attempt)
	}
	lease, _ := r.pool.leases.Get(claimed.LeaseID)
	if lease.State != nashnet.LeaseReleased || lease.LeaseEpoch != claimed.LeaseEpoch+1 {
		t.Fatalf("lease = %s epoch %d, want released at epoch %d",
			lease.State, lease.LeaseEpoch, claimed.LeaseEpoch+1)
	}

	st := readProcessState(t, r.runsDir, "silent-job")
	if st.Status != StateCanceled {
		t.Fatalf("terminal = %q, want canceled", st.Status)
	}
	if st.ExitCode != nil {
		t.Fatalf("exit code = %d, want none: this daemon never waited on the process", *st.ExitCode)
	}
	if view, _ := r.disp.resolveView("silent-job"); view.QueuePos != 0 {
		t.Fatalf("job is back at queue position %d, want it settled", view.QueuePos)
	}
}

// TestPurgeRemovesEveryLeaseTreeForTheJob is AC(6): purge deletes the trees of
// every lease the job ever had, not only its last one, and leaves the other
// job's tree alone.
func TestPurgeRemovesEveryLeaseTreeForTheJob(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	claimed := r.leasedJob(t, "purge-job")
	r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, []byte("uploaded bytes\n"))

	// An earlier lease of the same job, on the other node: exactly what a
	// re-claim after an expiry leaves behind.
	stale := filepath.Join(r.quarDir, r.nodeB.id, "purge-job", "lease-earlier")
	if err := os.MkdirAll(filepath.Join(stale, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	// A neighbouring job's tree, which the purge must not touch.
	neighbour := filepath.Join(r.quarDir, r.nodeA.id, "other-job", "lease-other")
	if err := os.MkdirAll(neighbour, 0o700); err != nil {
		t.Fatal(err)
	}

	r.cancelJob(t, "purge-job", false)
	r.clock.advance(2 * r.pool.leases.TTL())
	r.pool.Sweeper().Tick()

	resp := r.do(http.MethodDelete, "/harness/jobs/purge-job?purge=true", nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("purge: got %d, want 200", resp.StatusCode)
	}

	for _, dir := range []string{
		filepath.Join(r.quarDir, r.nodeA.id, "purge-job"),
		filepath.Join(r.quarDir, r.nodeB.id, "purge-job"),
	} {
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("quarantine tree %s survived the purge (%v)", dir, err)
		}
	}
	if _, err := os.Stat(neighbour); err != nil {
		t.Fatalf("purge removed another job's tree: %v", err)
	}
}

// TestPurgeCountsALeasedDependentAsLive is the dependent half of D31: a
// dependent a node holds a lease for is not started here, but it is live, so
// the parent's purge is refused without cascade.
func TestPurgeCountsALeasedDependentAsLive(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	parent := r.leasedJob(t, "parent-job")
	r.queueJob(t, JobSpec{Name: "child-job", After: []string{"parent-job"}})

	content := []byte("parent output\n")
	digest := r.uploadBlob(t, parent.LeaseToken, parent.LeaseID, content)
	resp := r.doLease(t, parent.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+parent.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: parent.LeaseEpoch, Seq: 1, Final: true,
			Entries: []quarantine.Entry{{
				Path: "metrics.jsonl", Digest: digest, Size: int64(len(content)),
				MTime: r.clock.now().UnixNano(),
			}},
		})
	var committed quarantine.CommitResponse
	decodeInto(t, resp, &committed)
	exit := 0
	resp = r.doLease(t, parent.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+parent.LeaseID+"/result", nashnet.ResultRequest{
			LeaseEpoch: parent.LeaseEpoch, State: nashnet.ResultStopped, ExitCode: &exit,
			FinalManifestDigest: committed.Digest,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("parent result: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	// The child is now placeable, and its lease is the only thing holding it.
	_, child := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if child == nil || child.JobID != "child-job" {
		t.Fatalf("expected node-a to claim child-job, got %+v", child)
	}
	resp = r.do(http.MethodDelete, "/harness/jobs/parent-job?purge=true", nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("purge with a leased dependent: got %d, want 409", resp.StatusCode)
	}
}

// finishPoolRun runs one job on node A to a clean terminal, promoting the named
// paths as its final manifest. With no paths it promotes the two files the
// resume contract needs, so a later resume has both promoted state and a prior
// executor to pin to.
func (r *poolRig) finishPoolRun(t *testing.T, name string, paths ...string) {
	t.Helper()
	claimed := r.leasedJob(t, name)
	if len(paths) == 0 {
		paths = []string{resumeCheckpointPath, resumeStatePath}
	}
	entries := make([]quarantine.Entry, 0, len(paths))
	for i, p := range paths {
		entries = append(entries, r.entryFor(t, claimed, p, []byte(name+" body "+itoa(i)+"\n")))
	}
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1, Final: true,
			Entries: entries,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("final manifest for %s: got %d, want 200", name, resp.StatusCode)
	}
	var committed quarantine.CommitResponse
	decodeInto(t, resp, &committed)
	if len(committed.Rejected) > 0 {
		t.Fatalf("manifest rejections for %s: %+v", name, committed.Rejected)
	}
	exit := 0
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/result", nashnet.ResultRequest{
			LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultStopped, ExitCode: &exit,
			FinalManifestDigest: committed.Digest,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("result for %s: got %d, want 200", name, resp.StatusCode)
	}
	resp.Body.Close()
}

// entryFor uploads one file and returns the manifest entry naming it.
func (r *poolRig) entryFor(t *testing.T, claimed *nashnet.ClaimResponse, path string, body []byte) quarantine.Entry {
	t.Helper()
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, body)
	return quarantine.Entry{
		Path: path, Digest: digest, Size: int64(len(body)), MTime: r.clock.now().UnixNano(),
	}
}

// TestResumePinsToThePriorNodeAndHoldsWhenItIsGone is AC(7): a resume is pinned
// to the node whose runs dir holds its reservoir, no other node may take it,
// and a resume whose node is gone holds as reservoir_unavailable naming that
// node rather than placing elsewhere.
func TestResumePinsToThePriorNodeAndHoldsWhenItIsGone(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)
	r.finishPoolRun(t, "resume-job")

	if _, err := r.disp.Resume("resume-job"); err != nil {
		t.Fatalf("resume: %v", err)
	}
	view, _ := r.disp.resolveView("resume-job")
	if view.Placement == PlacementReservoirGone {
		t.Fatal("resume held as reservoir_unavailable while its node is present")
	}
	if _, other := r.claim(t, r.nodeB, nashnet.ClaimRequest{}); other != nil {
		t.Fatalf("node-b took the pinned resume %s", other.JobID)
	}
	_, mine := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if mine == nil || mine.JobID != "resume-job" {
		t.Fatalf("the pinned node did not get its own resume, got %+v", mine)
	}
	if !mine.Resume {
		t.Fatal("the claim did not carry resume")
	}

	// Nack it back to the ready set, then take the pinned node out of the pool:
	// the job holds for it by name rather than moving to the other node.
	resp := r.doLease(t, mine.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+mine.LeaseID+"/nack",
		nashnet.NackRequest{LeaseEpoch: mine.LeaseEpoch, Reason: nashnet.NackGateBreach})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("nack: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.do(http.MethodPost, "/nashnet/nodes/"+r.nodeA.id+"/drain",
		map[string]any{"drain": true})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("drain: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	r.disp.refreshPinHolds()

	view, _ = r.disp.resolveView("resume-job")
	if view.Placement != PlacementReservoirGone {
		t.Fatalf("placement = %q, want reservoir_unavailable", view.Placement)
	}
	if !containsDetail(view.PlacementDetail, "node="+r.nodeA.id) {
		t.Fatalf("placement detail = %v, want the node it waits for", view.PlacementDetail)
	}
	if _, other := r.claim(t, r.nodeB, nashnet.ClaimRequest{}); other != nil {
		t.Fatalf("node-b took %s while its reservoir node was gone", other.JobID)
	}
}

// TestResumeGateReadsThePromotedCopy is the resume half of D31: for a job the
// pool ran, the gate asks the folded manifest head whether the resume files
// were promoted, so bytes that merely sit in the run dir do not open a resume.
func TestResumeGateReadsThePromotedCopy(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	// A pool run that promoted something other than the resume pair.
	r.finishPoolRun(t, "planted-job", "metrics.jsonl")

	runDir := filepath.Join(r.runsDir, "planted-job")
	if err := os.MkdirAll(filepath.Join(runDir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, resumeCheckpointPath), []byte("planted\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, resumeStatePath), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := r.disp.Resume("planted-job"); err != ErrNoResumableState {
		t.Fatalf("resume off unpromoted bytes: got %v, want ErrNoResumableState", err)
	}
}

func containsDetail(details []string, want string) bool {
	for _, d := range details {
		if d == want {
			return true
		}
	}
	return false
}
