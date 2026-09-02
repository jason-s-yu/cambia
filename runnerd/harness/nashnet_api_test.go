package harness

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// TestCredentialOnTheWrongSideIs403 is AC(6): the three credentials are not
// interchangeable, and presenting one on another's route names the reason
// rather than answering a bare 401.
func TestCredentialOnTheWrongSideIs403(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})

	// A node token on an operator route.
	resp := r.raw(t, http.MethodGet, "/harness/jobs", nil, map[string]string{
		"Authorization": "Bearer " + r.nodeToken(t, r.nodeA),
	})
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("node token on /harness/jobs: got %d, want 403", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeWrongAudience {
		t.Fatalf("node token on /harness/jobs: code = %q, want wrong_audience", code)
	}

	// An operator token on a node route.
	resp = r.raw(t, http.MethodPost, "/nashnet/claim", nashnet.ClaimRequest{}, map[string]string{
		"Authorization": "Bearer " + r.token,
	})
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("operator token on /nashnet/claim: got %d, want 403", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeWrongAudience {
		t.Fatalf("operator token on /nashnet/claim: code = %q, want wrong_audience", code)
	}

	// A lease credential on a node route.
	resp = r.raw(t, http.MethodPost, "/nashnet/nodes/register", nashnet.RegisterRequest{},
		map[string]string{nashnet.HeaderLeaseToken: "opaque-lease-token"})
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("lease token on register: got %d, want 403", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeWrongAudience {
		t.Fatalf("lease token on register: code = %q, want wrong_audience", code)
	}
}

// TestBodyNodeIDIsRefusedAndLeavesTheVictimAlone is AC(12): identity is the
// verified subject and nothing else, so naming another node in a body is
// refused and changes nothing about that node's record, epoch, or leases.
func TestBodyNodeIDIsRefusedAndLeavesTheVictimAlone(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	victimEpoch := r.register(t, r.nodeB, 2)
	r.queueJob(t, JobSpec{Name: "victim-job"})
	_, claimed := r.claim(t, r.nodeB, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("node-b should have claimed the queued job")
	}

	resp := r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/nodes/register",
		nashnet.RegisterRequest{NodeID: r.nodeB.id, Slots: 2, Capabilities: declaration(2)})
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("register naming another node: got %d, want 403", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeWrongSubject {
		t.Fatalf("register naming another node: code = %q, want wrong_subject", code)
	}

	resp = r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/claim",
		nashnet.ClaimRequest{NodeID: r.nodeB.id, SlotsFree: 1})
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("claim naming another node: got %d, want 403", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeWrongSubject {
		t.Fatalf("claim naming another node: code = %q, want wrong_subject", code)
	}

	rec, ok := r.pool.nodes.Get(r.nodeB.id)
	if !ok {
		t.Fatal("the victim's record must survive")
	}
	if rec.NodeEpoch != victimEpoch {
		t.Fatalf("victim node_epoch = %d, want %d (untouched)", rec.NodeEpoch, victimEpoch)
	}
	live := r.pool.leases.LiveForNode(r.nodeB.id)
	if len(live) != 1 || live[0].LeaseID != claimed.LeaseID {
		t.Fatalf("victim leases = %v, want the one it claimed", live)
	}
}

// TestWaiterCapAnswersImmediately is the first half of AC(5): past the cap a
// claim answers 204 waiters_full at once rather than joining the queue of held
// polls.
func TestWaiterCapAnswersImmediately(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{maxClaimWaiters: 1})
	r.register(t, r.nodeA, 2)
	r.register(t, r.nodeB, 2)

	started := make(chan struct{})
	done := make(chan struct{})
	go func() {
		close(started)
		resp := r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/claim",
			nashnet.ClaimRequest{SlotsFree: 1, WaitSeconds: 2, Capabilities: declaration(2), GateReport: admitReport(2)})
		resp.Body.Close()
		close(done)
	}()
	<-started
	// Give the first claim time to register as a waiter before the second runs.
	waitFor(t, func() bool { return r.pool.waiterCount() == 1 })

	resp := r.doNode(t, r.nodeB, http.MethodPost, "/nashnet/claim",
		nashnet.ClaimRequest{SlotsFree: 1, WaitSeconds: 2, Capabilities: declaration(2), GateReport: admitReport(2)})
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("second claim: got %d, want 204", resp.StatusCode)
	}
	if hold := holdReason(resp); hold != nashnet.HoldWaitersFull {
		t.Fatalf("second claim hold = %q, want waiters_full", hold)
	}
	<-done
}

// TestSecondEventsPollSupersedesTheFirst is the second half of AC(5): a node
// keeps exactly one events request open, and the older one is told so.
func TestSecondEventsPollSupersedesTheFirst(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)

	first := make(chan *http.Response, 1)
	go func() {
		first <- r.doNode(t, r.nodeA, http.MethodGet,
			"/nashnet/nodes/"+r.nodeA.id+"/events?wait_seconds=5", nil)
	}()
	waitFor(t, func() bool { return r.pool.sessionCount() == 1 })

	second := r.doNode(t, r.nodeA, http.MethodGet,
		"/nashnet/nodes/"+r.nodeA.id+"/events?wait_seconds=1", nil)
	if second.StatusCode != http.StatusOK {
		t.Fatalf("newer events poll: got %d, want 200", second.StatusCode)
	}
	second.Body.Close()

	older := <-first
	if older.StatusCode != http.StatusConflict {
		t.Fatalf("older events poll: got %d, want 409", older.StatusCode)
	}
	if code := errorCode(t, older); code != nashnet.CodeSessionSuperseded {
		t.Fatalf("older events poll code = %q, want session_superseded", code)
	}
}

// TestProgressProjectsOnlyEnumValues is AC(13): only procmgr statuses reach
// process.json, PGID is 0 on every projected row, the reported phase is carried
// separately in the view, and a node-reported stopping is refused.
func TestProgressProjectsOnlyEnumValues(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "proj-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}

	enum := map[string]bool{
		procmgr.StatusCreated: true, procmgr.StatusStarting: true, procmgr.StatusRunning: true,
		procmgr.StatusStopping: true, procmgr.StatusStopped: true, procmgr.StatusCrashed: true,
	}
	for _, phase := range []string{
		nashnet.PhaseClaimed, nashnet.PhaseFetching, nashnet.PhasePreparing,
		nashnet.PhaseRunning, nashnet.PhaseUploading, nashnet.PhaseCommitting,
	} {
		resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
			"/nashnet/leases/"+claimed.LeaseID+"/progress",
			nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: phase, PID: 4242})
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("progress %s: got %d, want 200", phase, resp.StatusCode)
		}
		resp.Body.Close()

		st := readProcessState(t, r.runsDir, "proj-job")
		if !enum[st.Status] {
			t.Fatalf("phase %s projected status %q, which is not a procmgr enum value", phase, st.Status)
		}
		if st.PGID != 0 {
			t.Fatalf("phase %s projected PGID %d, want 0", phase, st.PGID)
		}
		if st.Host != r.nodeA.id {
			t.Fatalf("projected Host = %q, want the node id", st.Host)
		}
		view, _ := r.disp.resolveView("proj-job")
		if view.Phase != phase {
			t.Fatalf("JobView.phase = %q, want %q", view.Phase, phase)
		}
	}

	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhaseStopping})
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("node-reported stopping: got %d, want 422", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeInvalidPhase {
		t.Fatalf("node-reported stopping code = %q, want invalid_phase", code)
	}
}

// TestEnvJSONCarriesExecutedOnWhileRunning is AC(14): the provenance record
// exists and names the executing node before the job is terminal, which is what
// v1.0 delivers from Prepare onward.
func TestEnvJSONCarriesExecutedOnWhileRunning(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "env-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	view, _ := r.disp.resolveView("env-job")
	if isTerminal(view.State) {
		t.Fatalf("job is terminal (%s) before env.json was read", view.State)
	}
	data, err := os.ReadFile(filepath.Join(r.runsDir, "env-job", "env.json"))
	if err != nil {
		t.Fatalf("env.json must exist for a running pool job: %v", err)
	}
	var rec map[string]any
	if err := json.Unmarshal(data, &rec); err != nil {
		t.Fatal(err)
	}
	if rec["executed_on"] != r.nodeA.id {
		t.Fatalf("executed_on = %v, want the plain node id %q", rec["executed_on"], r.nodeA.id)
	}
	if rec["origin_host"] != "coordinator.test" {
		t.Fatalf("origin_host = %v, want the coordinator", rec["origin_host"])
	}
}

// TestLogAppendFencesOnOffsetAndFiltersControlBytes is AC(8): a wrong offset is
// 409 carrying the true offset, control bytes never reach the file, and the WS
// tail streams what was appended.
func TestLogAppendFencesOnOffsetAndFiltersControlBytes(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{rig: rigConfig{origin: "https://client.lan"}})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "logjob"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	logsPath := "/nashnet/leases/" + claimed.LeaseID + "/logs"

	payload := []byte("iter 1 ok\n\x1b[31mred\x1b[0m\x07 tail\n")
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost, logsPath+"?offset=0", payload)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first append: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	onDisk, err := os.ReadFile(filepath.Join(r.runsDir, "logjob", "logs", "training.log"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.ContainsRune(string(onDisk), 0x1b) || strings.ContainsRune(string(onDisk), 0x07) {
		t.Fatalf("control bytes reached the log: %q", onDisk)
	}
	if !strings.Contains(string(onDisk), "red") || !strings.Contains(string(onDisk), "iter 1 ok") {
		t.Fatalf("filtered log lost its text: %q", onDisk)
	}
	view, _ := r.disp.resolveView("logjob")
	if view.LogBytesDropped == 0 {
		t.Fatal("the coordinator-owned dropped-byte counter stayed at zero")
	}

	// A wrong offset is refused with the true one.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, logsPath+"?offset=0", []byte("again\n"))
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("stale offset: got %d, want 409", resp.StatusCode)
	}
	var body nashnet.ErrorBody
	decodeInto(t, resp, &body)
	if body.Error != nashnet.CodeOffsetMismatch {
		t.Fatalf("stale offset code = %q, want offset_mismatch", body.Error)
	}
	if body.Offset != int64(len(onDisk)) {
		t.Fatalf("reported offset = %d, want the true size %d", body.Offset, len(onDisk))
	}

	// The same file is what the operator WS tail serves.
	c, _, err := r.dialWS("/ws/harness/jobs/logjob/logs", "https://client.lan", r.token)
	if err != nil {
		t.Fatalf("dial logs ws: %v", err)
	}
	defer c.CloseNow()
	env := readEnvelope(t, c)
	if env.Type != "log_backfill" {
		t.Fatalf("ws tail first message = %q, want log_backfill", env.Type)
	}
}

// TestSeedOutsideTheGrantSetIs404 is AC(7): an out-of-set read is
// indistinguishable from an unknown one, so no existence oracle exists.
func TestSeedOutsideTheGrantSetIs404(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)

	// A prior run holds the seed the evaluate job reads, plus a file the grant
	// never names.
	target := filepath.Join(r.runsDir, "prior-run")
	if err := os.MkdirAll(filepath.Join(target, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeSeedFile(t, filepath.Join(target, "snapshots", "prtcfr_checkpoint.pt"), "checkpoint bytes")
	writeSeedFile(t, filepath.Join(target, "secrets.txt"), "not a granted entry")
	if err := procmgr.WriteProcessState(target, &procmgr.ProcessState{
		Name: "prior-run", Status: procmgr.StatusStopped,
	}); err != nil {
		t.Fatal(err)
	}
	r.queueJob(t, JobSpec{Name: "eval-job", Kind: KindEvaluate, Target: "prior-run/snapshots/prtcfr_checkpoint.pt"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	if len(claimed.Seeds) != 1 || claimed.Seeds[0].SeedID != "prior-run" {
		t.Fatalf("seeds = %+v, want one grouped under prior-run", claimed.Seeds)
	}

	base := "/nashnet/leases/" + claimed.LeaseID + "/seeds/prior-run/"
	granted := r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"snapshots/prtcfr_checkpoint.pt", nil)
	if granted.StatusCode != http.StatusOK {
		t.Fatalf("granted seed: got %d, want 200", granted.StatusCode)
	}
	granted.Body.Close()

	outOfSet := r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"secrets.txt", nil)
	unknown := r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"no-such-file.pt", nil)
	if outOfSet.StatusCode != http.StatusNotFound || unknown.StatusCode != http.StatusNotFound {
		t.Fatalf("out-of-set %d and unknown %d must both be 404", outOfSet.StatusCode, unknown.StatusCode)
	}
	var a, b nashnet.ErrorBody
	decodeInto(t, outOfSet, &a)
	decodeInto(t, unknown, &b)
	if a != b {
		t.Fatalf("out-of-set body %+v differs from unknown body %+v", a, b)
	}
}

// TestNodesListingRendersTheRecord is AC(10).
func TestNodesListingRendersTheRecord(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "listed-job"})
	if _, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); claimed == nil {
		t.Fatal("expected a claim")
	}
	r.clock.advance(45 * time.Second)

	resp := r.do(http.MethodGet, "/nashnet/nodes", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("nodes listing: got %d, want 200", resp.StatusCode)
	}
	var body struct {
		Nodes []NodeView `json:"nodes"`
	}
	decodeInto(t, resp, &body)
	if len(body.Nodes) != 1 {
		t.Fatalf("listing has %d nodes, want 1", len(body.Nodes))
	}
	got := body.Nodes[0]
	if len(got.Capabilities) == 0 {
		t.Fatal("listing dropped the declaration")
	}
	if len(got.GateReport) == 0 {
		t.Fatal("listing dropped the gate report")
	}
	if got.Presence == "" {
		t.Fatal("listing dropped the session state")
	}
	if got.StaleSeconds < 45 {
		t.Fatalf("stale_seconds = %d, want at least 45", got.StaleSeconds)
	}
	if len(got.Leases) != 1 || got.Leases[0].JobID != "listed-job" {
		t.Fatalf("listing leases = %+v, want the one live lease", got.Leases)
	}
}

// TestDrainIsTwoWay is AC(16): the body carries the state to set, so lifting a
// hold costs no second route, and clear_breaker resets the counter.
func TestDrainIsTwoWay(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.pool.noteNack(r.nodeA.id, "some-job", nashnet.NackPrepareNodeFailed, time.Second)

	drainPath := "/nashnet/nodes/" + r.nodeA.id + "/drain"
	resp := r.do(http.MethodPost, drainPath, nashnet.DrainRequest{Drain: true})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("drain on: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if rec, _ := r.pool.nodes.Get(r.nodeA.id); !rec.Drained {
		t.Fatal("drain: true did not set the hold")
	}
	// A drained node claims nothing, and the hold is not its own gate.
	r.queueJob(t, JobSpec{Name: "drained-job"})
	claimResp, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatal("a drained node must not be handed work")
	}
	if hold := holdReason(claimResp); hold != nashnet.HoldNodeGated {
		t.Fatalf("drained claim hold = %q, want node_gated", hold)
	}

	resp = r.do(http.MethodPost, drainPath, nashnet.DrainRequest{Drain: false, ClearBreaker: true})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("drain off: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if rec, _ := r.pool.nodes.Get(r.nodeA.id); rec.Drained {
		t.Fatal("drain: false did not lift the hold")
	}
	r.pool.mu.Lock()
	trips := r.pool.breaker[r.nodeA.id]
	r.pool.mu.Unlock()
	if trips != 0 {
		t.Fatalf("breaker counter = %d after clear_breaker, want 0", trips)
	}
	if _, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{}); claimed == nil {
		t.Fatal("a node whose hold was lifted must claim again")
	}
}

// TestResultNeedsAFinalManifest walks the data plane end to end and pins the
// commit point of D6: upload, commit final, then post the terminal. A result
// without the final manifest is refused artifacts_incomplete.
func TestResultNeedsAFinalManifest(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "result-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	base := "/nashnet/leases/" + claimed.LeaseID

	// The job runs, so its terminal is gated on the artifact commit; a lease
	// that never launched is exempt and has its own test.
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhaseRunning, PID: 5150})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("progress: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/result",
		nashnet.ResultRequest{LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultStopped})
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("result with no manifest: got %d, want 409", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeArtifactsIncomplete {
		t.Fatalf("result with no manifest: code = %q, want artifacts_incomplete", code)
	}

	content := []byte("metrics row\n")
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, content)
	commit := quarantine.CommitRequest{
		ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1, Final: true,
		Entries: []quarantine.Entry{{
			Path: "metrics.jsonl", Digest: digest, Size: int64(len(content)),
			MTime: r.clock.now().UnixNano(),
		}},
	}
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest", commit)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("manifest commit: got %d, want 200", resp.StatusCode)
	}
	var committed quarantine.CommitResponse
	decodeInto(t, resp, &committed)
	if len(committed.Promoted) != 1 {
		t.Fatalf("promoted = %v, want the one entry", committed.Promoted)
	}

	exit := 0
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/result",
		nashnet.ResultRequest{
			LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultStopped, ExitCode: &exit,
			FinalManifestDigest: committed.Digest,
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("result: got %d, want 200", resp.StatusCode)
	}
	var recorded nashnet.ResultResponse
	decodeInto(t, resp, &recorded)
	if recorded.State != nashnet.ResultStopped {
		t.Fatalf("recorded state = %q, want stopped", recorded.State)
	}
	st := readProcessState(t, r.runsDir, "result-job")
	if st.Status != procmgr.StatusStopped {
		t.Fatalf("terminal status = %q, want stopped", st.Status)
	}

	// A replay of the same (lease, epoch) returns the recorded terminal.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/result",
		nashnet.ResultRequest{LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultStopped,
			FinalManifestDigest: committed.Digest})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("result replay: got %d, want 200", resp.StatusCode)
	}
	var replay nashnet.ResultResponse
	decodeInto(t, resp, &replay)
	if replay.RecordedAt != recorded.RecordedAt {
		t.Fatalf("replay returned a new record %+v, want the recorded one %+v", replay, recorded)
	}
}

// TestArtifactsUnionMatchesTheWalk is AC(9): the manifest-derived listing has
// the same shape and the same set as the walk-derived one, the
// coordinator-authored paths included.
func TestArtifactsUnionMatchesTheWalk(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "artifact-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	content := []byte("{\"iter\": 1}\n")
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, content)
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1,
			Entries: []quarantine.Entry{{
				Path: "metrics.jsonl", Digest: digest, Size: int64(len(content)),
				MTime: r.clock.now().UnixNano(),
			}},
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("manifest commit: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.do(http.MethodGet, "/harness/jobs/artifact-job/artifacts", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("artifacts: got %d, want 200", resp.StatusCode)
	}
	var served struct {
		JobID     string     `json:"job_id"`
		Artifacts []Artifact `json:"artifacts"`
	}
	decodeInto(t, resp, &served)

	walk := walkArtifacts(t, filepath.Join(r.runsDir, "artifact-job"))
	if len(served.Artifacts) != len(walk) {
		t.Fatalf("manifest listing has %d rows, the walk has %d:\n%+v\n%+v",
			len(served.Artifacts), len(walk), served.Artifacts, walk)
	}
	for i, got := range served.Artifacts {
		want := walk[i]
		if got.Path != want.Path || got.Size != want.Size || got.SHA256 != want.SHA256 {
			t.Fatalf("row %d = %+v, walk has %+v", i, got, want)
		}
		if got.MTime == "" {
			t.Fatalf("row %d = %+v, the shape keeps its mtime", i, got)
		}
	}
}

// walkArtifacts is the pre-manifest listing, rebuilt here so the test compares
// against the behavior it must preserve rather than against itself.
func walkArtifacts(t *testing.T, runDir string) []Artifact {
	t.Helper()
	var out []Artifact
	err := filepath.WalkDir(runDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == nashnetStateDir && path != runDir {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		rel, _ := filepath.Rel(runDir, path)
		info, _ := d.Info()
		sum, _ := sha256File(path)
		out = append(out, Artifact{Path: rel, Size: info.Size(), SHA256: sum})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// writeSeedFile writes one seed input file.
func writeSeedFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// waitFor polls cond until it holds or the test times out. It is the one place
// the suite waits on another goroutine, and it never sleeps on a fixed budget.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("condition never held within the timeout")
}
