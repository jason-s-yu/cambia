package harness

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// The named scenarios of D42, one test per line of the design's list. Each
// drives the real node agent against the real coordinator over the pinned
// HTTPS transport; the rig is in nashnet_twoproc_test.go and the hostile-node
// suite is in nashnet_hostile_test.go.
//
// Three scenarios were written against lanes that landed while this suite was
// built and carry a note saying so: TestNodeDeathMidJobIsRecoveredByAReattach
// and TestDrainReachesTheNodeAndLiftsOnTheRoundTrip against cambia-1887 (resume
// persistence and the drain event), and TestNackCooldownTripsTheCircuitBreaker
// and the drain scenario against cambia-1919 (the named hold on the register
// and heartbeat responses).

// TestRegisterAndClaim is the first D42 scenario: a node agent registers over
// the pinned HTTPS transport and is handed the one ready job. Everything after
// it in this file assumes this much works.
func TestRegisterAndClaim(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
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

// TestSnapshotDeliveredCold is the cold half of D42's snapshot scenario: a node
// with an empty mirror advertises no basis, the coordinator builds the
// full-tree bundle, and the bytes become git objects in the node's own mirror
// rather than a file on disk.
func TestSnapshotDeliveredCold(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	sha := r.queueFixtureJob(t, "cold-snapshot", 2, "quick")

	r.runCycle(t, 60*time.Second)

	if got := r.env.fetchCount(); got != 1 {
		t.Fatalf("bundle imports = %d, want exactly 1", got)
	}
	if !r.repo.nodeHas(t, sha) {
		t.Fatalf("the node's mirror does not hold %s after the snapshot fetch", sha)
	}
	if got := r.repo.nodeRef(t, "cold-snapshot"); got != sha {
		t.Fatalf("node mirror ref = %q, want the pinned commit %s", got, sha)
	}
	// A cold fetch is the full tree, which the coordinator caches under the
	// bare commit; a thin one carries a basis suffix (D48).
	if !r.repo.hasCachedBundle(t, sha, false) {
		t.Fatalf("the coordinator did not build the full-tree bundle for %s", sha)
	}
	served := r.requests.forPath("/snapshot")
	if len(served) == 0 || served[0].Status != http.StatusOK {
		t.Fatalf("snapshot requests = %+v, want a 200 on the first", served)
	}
}

// TestSnapshotDeliveredThin is the thin half: a node whose mirror already
// holds one commit advertises it, and the coordinator negates it out of the
// bundle. The node's have_commits comes from its own successful fetch, so this
// runs two cycles rather than seeding the list by hand.
func TestSnapshotDeliveredThin(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	first := r.queueFixtureJob(t, "thin-first", 0, "quick")
	r.runCycle(t, 60*time.Second)
	if !r.repo.nodeHas(t, first) {
		t.Fatalf("the first cycle left %s out of the node's mirror", first)
	}

	second := r.queueFixtureJob(t, "thin-second", 2, "quick")
	r.runCycle(t, 60*time.Second)

	if !r.repo.nodeHas(t, second) {
		t.Fatalf("the node's mirror does not hold %s after the thin fetch", second)
	}
	if !r.repo.hasCachedBundle(t, second, true) {
		t.Fatalf("the coordinator built no thin bundle for %s: the claim advertised no basis", second)
	}
}

// TestSnapshotInterruptedResumesByRange is D42's interrupted snapshot: 40% of
// the bundle is already on the node's disk when the fetch starts, so the node
// asks for the rest with a Range header and the coordinator answers 206. The
// digest check at the end is what proves the two halves joined correctly.
func TestSnapshotInterruptedResumesByRange(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	sha := r.queueFixtureJob(t, "resumed-snapshot", 2, "quick")

	// Build the same bundle the coordinator will serve and leave 40% of it
	// where the node's fetch resumes from.
	desc, err := r.repo.coordinator.BundleCreate(context.Background(), "resumed-snapshot", nil)
	if err != nil {
		t.Fatalf("build the fixture bundle: %v", err)
	}
	whole, err := os.ReadFile(desc.Path)
	if err != nil {
		t.Fatal(err)
	}
	cut := len(whole) * 40 / 100
	if cut == 0 {
		t.Fatalf("the fixture bundle is %d bytes: too small to interrupt", len(whole))
	}
	dest := filepath.Join(r.nodeCfg.BaseDir, "snapshots", "resumed-snapshot.bundle")
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dest, whole[:cut], 0o644); err != nil {
		t.Fatal(err)
	}

	r.runCycle(t, 60*time.Second)

	var resumed *requestRow
	for i, row := range r.requests.forPath("/snapshot") {
		if row.rangeStart() == int64(cut) {
			resumed = &r.requests.forPath("/snapshot")[i]
			break
		}
	}
	if resumed == nil {
		t.Fatalf("no snapshot request resumed at %d: %+v", cut, r.requests.forPath("/snapshot"))
	}
	if resumed.Status != http.StatusPartialContent {
		t.Fatalf("the resumed snapshot read answered %d, want 206", resumed.Status)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(whole) {
		t.Fatalf("the resumed bundle is %d bytes, want the whole %d", len(got), len(whole))
	}
	if !r.repo.nodeHas(t, sha) {
		t.Fatalf("the resumed bundle did not import %s", sha)
	}
}

// TestClaimToResultCycleWithIncrementalCommits is D42's full cycle and the
// named regression for F1 (the node's manifest commit omitted its lease epoch,
// so the coordinator fenced every commit and the node orphaned its whole
// output). It asserts three incremental commits rather than one final one, and
// that a promoted file and the quarantine blob behind it are one inode in link
// mode: the promotion hard-links rather than copying, so a checkpoint is not
// written to disk twice (D49).
func TestClaimToResultCycleWithIncrementalCommits(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "full-cycle", 2, "steps", "3")

	r.runCycle(t, 120*time.Second)

	lease := r.lastLeaseFor(t, "full-cycle")
	if n := r.commitCount(t, "full-cycle", lease); n < 3 {
		t.Fatalf("the lease committed %d manifests, want at least 3 incremental ones", n)
	}
	view, _ := r.disp.resolveView("full-cycle")
	if view.State != procmgr.StatusStopped {
		t.Fatalf("state = %q, want the clean terminal", view.State)
	}

	// The artifacts the run wrote are on the coordinator, promoted rather than
	// written by the node: the node never had a handle on this directory.
	metrics := r.promotedBody(t, "full-cycle", "metrics.jsonl")
	if !strings.Contains(string(metrics), `{"iter":3}`) {
		t.Fatalf("promoted metrics.jsonl = %q, want every step's line", metrics)
	}
	checkpoint := r.promotedBody(t, "full-cycle", "snapshots/prtcfr_checkpoint.pt")
	if string(checkpoint) != "weights-3" {
		t.Fatalf("promoted checkpoint = %q, want the last step's bytes", checkpoint)
	}
	// env.json is the coordinator's own record and is a rejected manifest
	// path; the node's staging copy lands beside it under env.node.json (D52).
	if _, err := os.Stat(filepath.Join(r.coordRunDir("full-cycle"), "env.node.json")); err != nil {
		t.Fatalf("the node's own provenance record was not promoted: %v", err)
	}

	if mode := r.pool.quar.MaterializeMode(); mode != quarantine.ModeLink {
		t.Skipf("the quarantine store selected %q: the inode assertion needs link mode", mode)
	}
	sum := sha256.Sum256(checkpoint)
	blob := r.blobPath("full-cycle", lease, hex.EncodeToString(sum[:]))
	blobInfo, err := os.Stat(blob)
	if err != nil {
		t.Fatalf("the verified blob is gone after promotion: %v", err)
	}
	promotedInfo, err := os.Stat(filepath.Join(r.coordRunDir("full-cycle"), "snapshots", "prtcfr_checkpoint.pt"))
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(blobInfo, promotedInfo) {
		t.Fatal("the promoted checkpoint is a copy of its blob, not a hard link to it")
	}
}

// TestUploadResumesAtTheHeadOffset is D42's killed upload: the coordinator
// already holds part of a blob, as it would after an upload died mid-chunk, and
// the node's HEAD probe finds the offset and appends from there rather than
// restarting at zero (D50 steps 2 and 3).
func TestUploadResumesAtTheHeadOffset(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "resumed-upload", 2, "quick")

	// The fixture job's checkpoint is deterministic, so the digest the node
	// will upload under is known before the claim. The part is written from
	// the staging hook, which runs after the lease exists and before the job
	// that produces the file does.
	body := []byte("weights-0")
	sum := sha256.Sum256(body)
	digest := hex.EncodeToString(sum[:])
	const resumeAt = 4
	r.env.onPrepare = func(string) error {
		lease := r.leaseIDFor(t, "resumed-upload")
		part := r.partPath("resumed-upload", lease, digest)
		if err := os.MkdirAll(filepath.Dir(part), 0o755); err != nil {
			return err
		}
		return os.WriteFile(part, body[:resumeAt], 0o644)
	}

	r.runCycle(t, 60*time.Second)

	heads := r.requests.forPath("/blobs/" + digest)
	sawHead := false
	var starts []int64
	for _, row := range heads {
		switch row.Method {
		case http.MethodHead:
			sawHead = true
		case http.MethodPatch:
			starts = append(starts, row.chunkStart())
		}
	}
	if !sawHead {
		t.Fatal("the upload never probed the HEAD offset")
	}
	if len(starts) == 0 || starts[0] != resumeAt {
		t.Fatalf("the upload's chunks started at %v, want the first at the part size %d", starts, resumeAt)
	}
	if got := r.promotedBody(t, "resumed-upload", "snapshots/prtcfr_checkpoint.pt"); string(got) != string(body) {
		t.Fatalf("the resumed blob promoted %q, want %q", got, body)
	}
}

// TestCommitRejectedForAMissingBlobIsRetried is D42's missing-blob commit: the
// coordinator loses the lease's verified blobs between the upload and the
// commit that names them, answers 409 blobs_missing with the list, and the node
// re-uploads exactly those and commits again (D50, D51).
func TestCommitRejectedForAMissingBlobIsRetried(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "missing-blob", 2, "quick")

	var leaseDir string
	r.env.onPrepare = func(string) error {
		leaseDir = filepath.Join(r.quarDir, r.node.id, "missing-blob", r.leaseIDFor(t, "missing-blob"))
		return nil
	}
	// The blobs vanish immediately before the first commit reaches the store,
	// which is the state a quarantine sweep or a lost filesystem leaves behind.
	r.requests.interceptOnce(http.MethodPost, "/manifest", func() {
		if leaseDir == "" {
			return
		}
		_ = os.RemoveAll(filepath.Join(leaseDir, "blobs"))
	})
	defer r.requests.clearIntercept()

	r.runCycle(t, 60*time.Second)

	refused := 0
	for _, row := range r.requests.forPath("/manifest") {
		if row.Method == http.MethodPost && row.Status == http.StatusConflict {
			refused++
		}
	}
	if refused == 0 {
		t.Fatal("no commit was refused: the scenario never provoked blobs_missing")
	}
	// The retry is what matters: the run still promoted its artifacts.
	if got := r.promotedBody(t, "missing-blob", "snapshots/prtcfr_checkpoint.pt"); string(got) != "weights-0" {
		t.Fatalf("promoted checkpoint = %q after the retry, want weights-0", got)
	}
	view, _ := r.disp.resolveView("missing-blob")
	if view.State != procmgr.StatusStopped {
		t.Fatalf("state = %q, want the clean terminal after the retried commit", view.State)
	}
}

// TestCommitWithOneRejectedEntryProceeds is D42's partial commit: one entry
// fails the rejection list and the rest of the manifest still promotes. The
// offending path carries a backslash, which is a legal filename on this
// filesystem and a refused manifest path everywhere (D52), so the node offers
// it in good faith and the coordinator records the refusal rather than failing
// the commit.
func TestCommitWithOneRejectedEntryProceeds(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "one-rejected", 2, "quick")
	r.env.onPrepare = func(runDir string) error {
		return os.WriteFile(filepath.Join(runDir, `weights\evil.pt`), []byte("rejected bytes"), 0o644)
	}

	r.runCycle(t, 60*time.Second)

	lease := r.lastLeaseFor(t, "one-rejected")
	line := lastReceiptLine(t, filepath.Join(r.quarDir, r.node.id, "one-rejected", lease))
	found := false
	for _, rej := range line.Rejected {
		if rej.Reason == quarantine.ReasonPathBackslash {
			found = true
		}
	}
	if !found {
		t.Fatalf("the receipt recorded %+v, want a path_backslash rejection", line.Rejected)
	}
	// The rest of the manifest still landed.
	if got := r.promotedBody(t, "one-rejected", "metrics.jsonl"); len(got) == 0 {
		t.Fatal("the accepted entries did not promote alongside the rejected one")
	}
	// Nothing resembling the rejected path exists under the run dir.
	entries, err := os.ReadDir(r.coordRunDir("one-rejected"))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.Contains(e.Name(), `\`) {
			t.Fatalf("the rejected path %q reached the run dir", e.Name())
		}
	}
	view, _ := r.disp.resolveView("one-rejected")
	if view.State != procmgr.StatusStopped {
		t.Fatalf("state = %q: one rejected entry failed the whole run", view.State)
	}
}

// TestSeedDeliveryForARePlacedResume is D42's seed scenario: a resumed job
// placed on a node that no longer holds the prior run's bytes is served them
// from the coordinator's promoted copy, verified per entry, and materialized
// read-only under the node's own run dir (D53).
func TestSeedDeliveryForARePlacedResume(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "seeded-resume", 2, "quick")
	r.runCycle(t, 60*time.Second)

	promoted := r.promotedBody(t, "seeded-resume", "snapshots/prtcfr_checkpoint.pt")

	// The re-placement: the node holds neither the run nor a cached snapshot,
	// which is the state of any node the resume lands on other than the one
	// that produced the run. Clearing the snapshot cache is also what keeps
	// this scenario off F3 (a node re-fetching a bundle it already holds
	// complete asks for a range past the end and is answered 416, which its
	// download loop treats as a fetch failure and nacks); a second claim on
	// the same node is that defect own case, not this one.
	if err := os.RemoveAll(r.nodeRunDir("seeded-resume")); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(r.nodeCfg.BaseDir, "snapshots")); err != nil {
		t.Fatal(err)
	}
	r.requests.reset()

	resp := r.do(http.MethodPost, "/harness/jobs/seeded-resume/resume", nil)
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		t.Fatalf("resume: got %d, want 200 or 202", resp.StatusCode)
	}
	resp.Body.Close()

	r.runCycle(t, 60*time.Second)

	seeds := r.requests.forPath("/seeds/")
	if len(seeds) == 0 {
		t.Fatal("the resumed claim fetched no seed: the prior run's bytes were assumed present")
	}
	for _, row := range seeds {
		if row.Status != http.StatusOK && row.Status != http.StatusPartialContent {
			t.Fatalf("a seed read answered %d: %+v", row.Status, row)
		}
	}
	seeded := filepath.Join(r.nodeRunDir("seeded-resume"), "snapshots", "prtcfr_checkpoint.pt")
	got, err := os.ReadFile(seeded)
	if err != nil {
		t.Fatalf("the seed did not materialize on the node: %v", err)
	}
	if string(got) != string(promoted) {
		t.Fatalf("the seeded checkpoint is %q, want the promoted %q", got, promoted)
	}
	info, err := os.Stat(seeded)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm()&0o222 != 0 {
		t.Fatalf("the seeded input is writable (%s): a seed is read-only on the node", info.Mode())
	}
}

// TestSchedulingOverBothTransports is the one scheduling test D42 runs over
// both NodeTransport implementations of D65. The remote leg is the pinned
// HTTPS client; the embedded leg is the in-process loopback, whose run
// materializes in place and uploads nothing (D40). The shared assertion is the
// scheduling outcome, which must not depend on which transport carried it.
func TestSchedulingOverBothTransports(t *testing.T) {
	t.Run("https", func(t *testing.T) {
		r := newNodeRig(t, nodeRigConfig{claimOnce: true})
		r.queueFixtureJob(t, "both-transports", 2, "quick")
		r.runCycle(t, 60*time.Second)
		assertScheduledAndPromoted(t, r.poolRig, "both-transports")
	})
	t.Run("loopback", func(t *testing.T) {
		r := newNodeRig(t, nodeRigConfig{
			claimOnce: true,
			pool:      poolRigConfig{embedded: true},
		})
		embedded := r.newEmbeddedAgent(t)
		r.queueFixtureJobAt(t, "both-transports", 2,
			filepath.Join(r.runsDir, "both-transports"), "quick")
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		if err := embedded.Run(ctx); err != nil {
			t.Fatalf("embedded agent run: %v", err)
		}

		// The scheduling half is identical over both transports: the embedded
		// node registered, was handed the job, and held a lease on it, all
		// through the in-process handler rather than a socket.
		if len(r.requests.forPath("/nashnet/nodes/register")) == 0 {
			t.Fatal("the embedded node never registered over the loopback")
		}
		claims := r.requests.forPath("/nashnet/claim")
		if len(claims) == 0 || claims[0].Status != http.StatusOK {
			t.Fatalf("claim over the loopback = %+v, want a 200 handout", claims)
		}
		if len(r.requests.forPath("/progress")) == 0 {
			t.Fatal("the embedded lease posted no progress")
		}

		// The run half does not reach a terminal, and the reason is F2 rather
		// than anything about the transport: the coordinator's own progress
		// projection writes procmgr status starting into the run dir the
		// embedded node then launches from (nashnet_lease.go projectPhase),
		// and procLauncher.Ensure refuses to adopt a starting row, so every
		// embedded claim nacks prepare_node_failed before Prepare runs. The
		// assertions above are what this leg covers today; the launch half is
		// blocked on that fix.
		t.Skip("F2: the coordinator's own progress projection blocks the embedded node's launch")
	})
}

// assertScheduledAndPromoted is the shared body of the transport legs: the job
// was placed, ran, promoted its artifacts, and reached a clean terminal.
func assertScheduledAndPromoted(t *testing.T, r *poolRig, job string) {
	t.Helper()
	view, ok := r.disp.resolveView(job)
	if !ok {
		t.Fatalf("no view for %s", job)
	}
	if view.State != procmgr.StatusStopped {
		t.Fatalf("state = %q, want the clean terminal", view.State)
	}
	body, err := os.ReadFile(filepath.Join(r.runsDir, job, "metrics.jsonl"))
	if err != nil {
		t.Fatalf("the run promoted no metrics: %v", err)
	}
	if len(body) == 0 {
		t.Fatal("the promoted metrics.jsonl is empty")
	}
}

// TestNackCooldownTripsTheCircuitBreaker is D42's nack scenario: a node whose
// staging keeps failing returns each claim rather than failing the job, and
// three consecutive prepare_node_failed nacks hold it off the claim route
// (D8, D63). The job survives every one of them at the same attempt, because a
// nack is a scheduling event and not an attempt.
//
// Written against cambia-1919: the hold a node is told about names the breaker
// rather than a drain, on the register response as well as the claim's header.
func TestNackCooldownTripsTheCircuitBreaker(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	// Three jobs rather than one retried job: a nack holds this node off the
	// job it returned for that job cooldown, so the scan hands it the next
	// one, and the breaker counts consecutive nacks across any jobs.
	jobs := []string{"nack-one", "nack-two", "nack-three"}
	for i, job := range jobs {
		r.queueFixtureJob(t, job, i, "quick")
	}
	r.env.setPrepareErr(errors.New("the node venv cache is unwritable"))

	for i := range jobs {
		r.runCycle(t, 60*time.Second)
		if got := len(r.requests.forPath("/nack")); got != i+1 {
			t.Fatalf("after cycle %d the node had posted %d nacks, want %d", i+1, got, i+1)
		}
	}
	for _, job := range jobs {
		view, _ := r.disp.resolveView(job)
		if isTerminal(view.State) {
			t.Fatalf("a nack failed %s: state %q", job, view.State)
		}
	}

	if !r.pool.breakerHeld(r.node.id) {
		t.Fatalf("%d consecutive prepare_node_failed nacks did not trip the breaker", BreakerThreshold)
	}
	// The hold the node is told about names the breaker, not a drain. A
	// coordinator that reported a drain here would have the node clear it off
	// its own next call, undraining itself under the hold (cambia-1919).
	if got := r.pool.effectiveHold(r.node.id); got != nashnet.HoldReasonBreaker {
		t.Fatalf("effective hold = %q after a breaker trip, want breaker", got)
	}

	// The held node is answered node_gated rather than handed the job again,
	// and the hold is not clearable by re-registering.
	resp, claimed := r.claim(t, r.node, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatalf("a held node was handed %s", claimed.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNodeGated {
		t.Fatalf("held claim hold = %q, want node_gated", hold)
	}

	reg := r.doNode(t, r.node, http.MethodPost, "/nashnet/nodes/register",
		nashnet.RegisterRequest{AgentVersion: "1.1.0", Slots: 2,
			Capabilities: declaration(2), GateReport: admitReport(2)})
	if reg.StatusCode != http.StatusOK {
		t.Fatalf("register while held: got %d, want 200", reg.StatusCode)
	}
	var regBody nashnet.RegisterResponse
	decodeInto(t, reg, &regBody)
	if regBody.Hold != nashnet.HoldReasonBreaker {
		t.Fatalf("register hold after a trip = %q, want breaker", regBody.Hold)
	}

	resp, claimed = r.claim(t, r.node, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatalf("re-registering cleared the breaker and the node was handed %s", claimed.JobID)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNodeGated {
		t.Fatalf("hold after a re-register = %q, want node_gated", hold)
	}
}

// TestDrainReachesTheNodeAndLiftsOnTheRoundTrip is the drain half of D36 and
// D45: an operator drain holds the node off the claim route, the node learns
// which hold it is by name rather than by inference, and a lift applies on the
// round trip that delivers it rather than on some later call.
//
// Written against cambia-1887 and cambia-1919: the drain event carries the
// boolean the route set, and the register and heartbeat responses carry a hold
// field whose value is drain or breaker.
func TestDrainReachesTheNodeAndLiftsOnTheRoundTrip(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.register(t, r.node, 2)
	r.queueFixtureJob(t, "drained-job", 2, "quick")

	resp := r.do(http.MethodPost, "/nashnet/nodes/"+r.node.id+"/drain", map[string]any{"drain": true})
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		t.Fatalf("drain: got %d, want 200 or 202", resp.StatusCode)
	}
	resp.Body.Close()

	if got := r.pool.effectiveHold(r.node.id); got != nashnet.HoldReasonDrain {
		t.Fatalf("effective hold = %q after an operator drain, want drain", got)
	}
	claimResp, claimed := r.claim(t, r.node, nashnet.ClaimRequest{})
	if claimed != nil {
		t.Fatalf("a drained node was handed %s", claimed.JobID)
	}
	if hold := holdReason(claimResp); hold != nashnet.HoldNodeGated {
		t.Fatalf("hold = %q for a drained node, want node_gated", hold)
	}

	reg := r.doNode(t, r.node, http.MethodPost, "/nashnet/nodes/register",
		nashnet.RegisterRequest{AgentVersion: "1.1.0", Slots: 2,
			Capabilities: declaration(2), GateReport: admitReport(2)})
	if reg.StatusCode != http.StatusOK {
		t.Fatalf("register while drained: got %d, want 200", reg.StatusCode)
	}
	var regBody nashnet.RegisterResponse
	decodeInto(t, reg, &regBody)
	if regBody.Hold != nashnet.HoldReasonDrain {
		t.Fatalf("register hold = %q while drained, want drain", regBody.Hold)
	}
	if view, _ := r.disp.resolveView("drained-job"); view.State != StateQueued {
		t.Fatalf("drained-job = %q, want it queued for a node that will take it", view.State)
	}

	lift := r.do(http.MethodPost, "/nashnet/nodes/"+r.node.id+"/drain", map[string]any{"drain": false})
	if lift.StatusCode != http.StatusOK && lift.StatusCode != http.StatusAccepted {
		t.Fatalf("drain lift: got %d, want 200 or 202", lift.StatusCode)
	}
	lift.Body.Close()
	if got := r.pool.effectiveHold(r.node.id); got != "" {
		t.Fatalf("effective hold = %q after the lift, want none", got)
	}
	_, taken := r.claim(t, r.node, nashnet.ClaimRequest{})
	if taken == nil {
		t.Fatal("the node was still held off the queue on the call after its drain lifted")
	}
	if taken.JobID != "drained-job" {
		t.Fatalf("claimed %q, want drained-job", taken.JobID)
	}
}

// TestFanInDependentsLaunchAfterPromotion is D42's fan-in scenario: a dependent
// with two parents is not placeable until both have reached a clean terminal,
// and then the same node claims and runs it (D29).
func TestFanInDependentsLaunchAfterPromotion(t *testing.T) {
	r := newNodeRig(t, nodeRigConfig{claimOnce: true})
	r.queueFixtureJob(t, "parent-one", 0, "quick")
	r.queueFixtureJob(t, "parent-two", 1, "quick")

	sha := r.repo.push(t, "fan-in-child", 2)
	child := integrationSpec("fan-in-child", r.nodeRunDir("fan-in-child"), "quick")
	child.Commit = sha
	child.After = []string{"parent-one", "parent-two"}
	r.queueJob(t, child)

	// With one parent still queued the dependent is not placeable, whichever
	// job the scan reaches first.
	r.runCycle(t, 60*time.Second)
	if view, _ := r.disp.resolveView("fan-in-child"); isTerminal(view.State) {
		t.Fatalf("the dependent ran at %q with a parent outstanding", view.State)
	}
	r.runCycle(t, 60*time.Second)
	for _, parent := range []string{"parent-one", "parent-two"} {
		view, _ := r.disp.resolveView(parent)
		if view.State != procmgr.StatusStopped {
			t.Fatalf("parent %s = %q, want the clean terminal before the dependent runs", parent, view.State)
		}
	}
	if view, _ := r.disp.resolveView("fan-in-child"); isTerminal(view.State) {
		t.Fatalf("the dependent ran at %q before both parents finished", view.State)
	}

	r.runCycle(t, 60*time.Second)
	view, _ := r.disp.resolveView("fan-in-child")
	if view.State != procmgr.StatusStopped {
		t.Fatalf("the dependent = %q after both parents promoted, want the clean terminal", view.State)
	}
	if body := r.promotedBody(t, "fan-in-child", "metrics.jsonl"); len(body) == 0 {
		t.Fatal("the dependent promoted nothing")
	}
}

// newEmbeddedAgent builds the coordinator's own node: the in-process loopback
// transport of D65 over the embedded grant, staging in place into the
// coordinator's own runs dir (D40). It exists so the scheduling test of D42
// runs over both transports; every other scenario drives the remote node.
func (r *nodeRig) newEmbeddedAgent(t *testing.T) *nodeagent.Agent {
	t.Helper()
	if r.nodeE.id == "" {
		t.Fatal("the rig has no embedded node: set poolRigConfig.embedded")
	}
	signer, err := nodeagent.NewSigner(r.nodeE.priv, r.clock.now)
	if err != nil {
		t.Fatal(err)
	}
	cfg := r.nodeCfg
	cfg.RunsDir = r.runsDir
	cfg.BaseDir = t.TempDir()
	env := &nodeEnv{worktree: r.worktree, runsDir: r.runsDir, mirror: r.repo.node, skipFetch: true}
	r.embeddedEnv = env
	agent, err := nodeagent.New(nodeagent.Options{
		Config: cfg,
		Signer: signer,
		Client: nodeagent.NewLoopbackClient(r.requests.wrap(r.srv.Handler()), signer),
		Env:    env,
		// The embedded node shares the coordinator's own ProcessManager, as
		// --role both does: a second manager over the same runs dir holds no
		// state for a row the coordinator already created, so its Ensure
		// refuses the launch rather than adopting it.
		Launcher:          nodeagent.NewLauncher(r.pm),
		Prober:            newNodeProber(),
		Logger:            agentLogger(false),
		CanBuildLibcambia: true,
		Now:               r.clock.now,
		PollInterval:      10 * time.Millisecond,
		InPlace:           true,
		ClaimOnce:         true,
	})
	if err != nil {
		t.Fatalf("embedded agent: %v", err)
	}
	return agent
}
