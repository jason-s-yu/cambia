package harness

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// loopbackClient is the embedded node's transport in a test: the production
// node client dialing this rig's own routed handler in process (D65). Nothing
// about the request differs from the HTTPS one but the round trip, so a test
// driving it exercises the real credential middleware, the real fence, and the
// real commit transaction.
func (r *poolRig) loopbackClient(t *testing.T) *nodeagent.Client {
	t.Helper()
	if r.nodeE.id == "" {
		t.Fatal("rig has no embedded node: set poolRigConfig.embedded")
	}
	signer, err := nodeagent.NewSigner(r.nodeE.priv, r.clock.now)
	if err != nil {
		t.Fatal(err)
	}
	if signer.NodeID() != r.nodeE.id {
		t.Fatalf("signer id %s, rig id %s", signer.NodeID(), r.nodeE.id)
	}
	return nodeagent.NewLoopbackClient(r.srv.Handler(), signer)
}

// registerEmbedded declares the embedded node over the loopback and returns the
// epoch, which is also the assertion that the in-process grant admits it: an
// unadmitted key is 401 here, exactly as a revoked remote node would be.
func (r *poolRig) registerEmbedded(t *testing.T, c *nodeagent.Client, slots int) {
	t.Helper()
	resp, err := c.Register(context.Background(), nashnet.RegisterRequest{
		AgentVersion: "1.1.0", Slots: slots,
		Capabilities: declaration(slots), GateReport: admitReport(slots),
	})
	if err != nil {
		t.Fatalf("register over the loopback: %v", err)
	}
	if resp.NodeEpoch == 0 {
		t.Fatal("register returned no node epoch")
	}
}

// claimEmbedded holds one non-blocking claim over the loopback.
func (r *poolRig) claimEmbedded(t *testing.T, c *nodeagent.Client) *nashnet.ClaimResponse {
	t.Helper()
	claim, hold, err := c.Claim(context.Background(), nashnet.ClaimRequest{
		NodeID: r.nodeE.id, SlotsFree: 1,
		Capabilities: declaration(1), GateReport: admitReport(1),
	})
	if err != nil {
		t.Fatalf("claim over the loopback: %v", err)
	}
	if claim == nil {
		t.Fatalf("no claim handed out (hold %+v)", hold)
	}
	return claim
}

// entryFor writes one file into the run dir and describes it as the manifest
// entry an in-place commit offers: the digest is of the bytes on disk, since
// nothing was uploaded and the coordinator proves the file where it lies.
func entryFor(t *testing.T, runDir, rel string, body []byte) quarantine.Entry {
	t.Helper()
	abs := filepath.Join(runDir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, body, 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(abs)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	return quarantine.Entry{
		Path:   rel,
		Digest: hex.EncodeToString(sum[:]),
		Size:   fi.Size(),
		MTime:  fi.ModTime().UnixNano(),
	}
}

// TestLoopbackCommitValidatesAndRecordsWithoutRemoving is the D40 acceptance:
// the embedded node uploads nothing, and its commit still runs the entry
// validator, the promotion transaction, and the receipt. The probe is a
// reserved path, which a remote commit rejects and never writes; here the
// training process already wrote it at the destination, so the validator
// records the rejection and the file stays exactly as it was.
func TestLoopbackCommitValidatesAndRecordsWithoutRemoving(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{embedded: true})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)
	r.queueJob(t, JobSpec{Name: "embedded-job"})
	claim := r.claimEmbedded(t, c)

	runDir := filepath.Join(r.runsDir, "embedded-job")
	reserved := []byte("{\"executed_on\": \"someone else\"}\n")
	accepted := entryFor(t, runDir, "metrics.jsonl", []byte("{\"iter\": 1}\n"))
	rejectedEntry := entryFor(t, runDir, "env.json", reserved)

	head, err := c.ManifestHead(context.Background(), claim.LeaseID, claim.LeaseToken)
	if err != nil {
		t.Fatalf("manifest head: %v", err)
	}
	resp, err := c.Commit(context.Background(), claim.LeaseID, claim.LeaseToken,
		quarantine.CommitRequest{
			ManifestVersion: quarantine.ManifestVersion,
			LeaseEpoch:      claim.LeaseEpoch,
			Seq:             head.Seq + 1,
			Parent:          head.Digest,
			Entries:         []quarantine.Entry{accepted, rejectedEntry},
		})
	if err != nil {
		t.Fatalf("loopback commit: %v", err)
	}

	// The accepted entry was proved without a byte crossing anything: the
	// coordinator hashed the file already at its destination (D40).
	if len(resp.Promoted) != 1 || resp.Promoted[0] != "metrics.jsonl" {
		t.Fatalf("promoted = %v, want [metrics.jsonl]", resp.Promoted)
	}
	if len(resp.Rejected) != 1 {
		t.Fatalf("rejected = %+v, want exactly the reserved path", resp.Rejected)
	}
	if resp.Rejected[0].Path != "env.json" || resp.Rejected[0].Reason != quarantine.ReasonPathReserved {
		t.Fatalf("rejected = %+v, want env.json path_reserved", resp.Rejected[0])
	}

	// Recorded, not removed: the run's own process wrote this file.
	onDisk, err := os.ReadFile(filepath.Join(runDir, "env.json"))
	if err != nil {
		t.Fatalf("the rejected file was removed: %v", err)
	}
	if string(onDisk) != string(reserved) {
		t.Fatalf("the rejected file was rewritten: %q", onDisk)
	}

	// The receipt is the audit record, and it carries the rejection the
	// response reported.
	line := lastReceiptLine(t, filepath.Join(r.quarDir, r.nodeE.id, "embedded-job", claim.LeaseID))
	if line.Offered != 2 {
		t.Fatalf("receipt offered = %d, want 2", line.Offered)
	}
	if len(line.Rejected) != 1 || line.Rejected[0].Path != "env.json" ||
		line.Rejected[0].Reason != quarantine.ReasonPathReserved {
		t.Fatalf("receipt rejected = %+v, want env.json path_reserved", line.Rejected)
	}
}

// lastReceiptLine reads the final line of a lease's receipt.jsonl.
func lastReceiptLine(t *testing.T, leaseDir string) quarantine.ReceiptLine {
	t.Helper()
	f, err := os.Open(filepath.Join(leaseDir, "receipt.jsonl"))
	if err != nil {
		t.Fatalf("open receipt: %v", err)
	}
	defer f.Close()
	var last quarantine.ReceiptLine
	seen := false
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var line quarantine.ReceiptLine
		if err := json.Unmarshal(sc.Bytes(), &line); err != nil {
			t.Fatalf("decode receipt line: %v", err)
		}
		last, seen = line, true
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("read receipt: %v", err)
	}
	if !seen {
		t.Fatal("receipt.jsonl is empty")
	}
	return last
}

// TestEmbeddedProgressLeavesTheLocalRowToItsLauncher is the coordinator half
// of cambia-2017: a phase reported by the node running inside this process is
// not mirrored into process.json. That run dir is the coordinator's own, and
// its row belongs to the ProcessManager that forks the job; a projection over
// it left a starting status the embedded node's own launcher refused to adopt,
// so every embedded claim nacked prepare_node_failed before Prepare ran and
// three of those held the coordinator's node off its own queue.
func TestEmbeddedProgressLeavesTheLocalRowToItsLauncher(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{embedded: true})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)
	r.queueJob(t, JobSpec{Name: "embedded-progress"})
	claim := r.claimEmbedded(t, c)

	for _, phase := range []string{
		nashnet.PhaseClaimed, nashnet.PhaseFetching, nashnet.PhasePreparing,
	} {
		if _, err := c.Progress(context.Background(), claim.LeaseID, claim.LeaseToken,
			nashnet.ProgressRequest{LeaseEpoch: claim.LeaseEpoch, Phase: phase}); err != nil {
			t.Fatalf("progress %s: %v", phase, err)
		}
		st := readProcessState(t, r.runsDir, "embedded-progress")
		if st.Status != procmgr.StatusCreated {
			t.Fatalf("phase %s left the local row at %q, want created", phase, st.Status)
		}
		if st.Host != "" {
			t.Fatalf("phase %s stamped Host %q on a row in this host's own pid space",
				phase, st.Host)
		}
	}

	// The rule and the launcher's adopt rule meet here: the node that is about
	// to fork this job shares the coordinator's ProcessManager, and Ensure
	// adopts only a row that is not already running (nodeagent.procLauncher).
	if err := nodeagent.NewLauncher(r.pm).Ensure("embedded-progress", KindTrain); err != nil {
		t.Fatalf("the embedded launcher refused its own run dir: %v", err)
	}

	// A remote lease on the same coordinator is still mirrored: the rule turns
	// on which node holds the lease, not on the route the progress arrived by.
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "remote-progress"})
	_, remote := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if remote == nil {
		t.Fatal("the remote node was handed no claim")
	}
	resp := r.doLease(t, remote.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+remote.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: remote.LeaseEpoch,
			Phase: nashnet.PhaseFetching, PID: 4242})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("remote progress: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	st := readProcessState(t, r.runsDir, "remote-progress")
	if st.Status != procmgr.StatusStarting || st.Host != r.nodeA.id {
		t.Fatalf("a remote lease projected status=%q host=%q, want starting on the node id",
			st.Status, st.Host)
	}
}

// TestCancelOfAnEmbeddedLeaseKeepsTheLocalPidAndPgid is the cancel half of the
// same rule (cambia-2017): the coordinator writes its stopping witness (D31,
// D35) and leaves the pid and pgid its own ProcessManager recorded at the fork.
// Blanking them, which is right for a node in another pid space, would leave a
// live local process group with no stop path to reach it once this daemon
// restarts and its supervisor map is empty.
func TestCancelOfAnEmbeddedLeaseKeepsTheLocalPidAndPgid(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{embedded: true})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)
	r.queueJob(t, JobSpec{Name: "embedded-cancel"})
	claim := r.claimEmbedded(t, c)

	// The row shape the local ProcessManager leaves after forking a job: a
	// live pid in this host's own pid space, and its group.
	runDir := filepath.Join(r.runsDir, "embedded-cancel")
	st := readProcessState(t, r.runsDir, "embedded-cancel")
	st.Status = procmgr.StatusRunning
	st.PID, st.PGID = os.Getpid(), os.Getpid()
	if err := procmgr.WriteProcessState(runDir, st); err != nil {
		t.Fatalf("seed the local row: %v", err)
	}

	r.cancelJob(t, "embedded-cancel", false)

	st = readProcessState(t, r.runsDir, "embedded-cancel")
	if st.Status != procmgr.StatusStopping {
		t.Fatalf("projection = %q, want stopping", st.Status)
	}
	if st.Host != "" {
		t.Fatalf("the embedded row carries Host %q, want none", st.Host)
	}
	if st.PID != os.Getpid() || st.PGID != os.Getpid() {
		t.Fatalf("pid=%d pgid=%d, want the local pair the fork recorded", st.PID, st.PGID)
	}
	if l, ok := r.pool.leases.Get(claim.LeaseID); !ok || l.State != nashnet.LeaseRevoking {
		t.Fatalf("lease state = %q, want revoking", l.State)
	}
}

// TestEmbeddedRunCarriesExecutedOn covers the coordinator's half of AC7: the
// run of an embedded lease gets env.json with executed_on naming the node that
// produced it, exactly as a remote lease's run does, from the moment the claim
// is granted rather than only at the result. The node's own staging record
// keeps its own name beside it, which the ingest manager's
// TestEnvJSONNameRedirectsTheStagingRecord asserts on the writer itself.
func TestEmbeddedRunCarriesExecutedOn(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{embedded: true})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)
	r.queueJob(t, JobSpec{Name: "embedded-prov"})
	claim := r.claimEmbedded(t, c)

	runDir := filepath.Join(r.runsDir, "embedded-prov")
	var coordinator struct {
		JobID      string `json:"job_id"`
		OriginHost string `json:"origin_host"`
		ExecutedOn string `json:"executed_on"`
		LeaseID    string `json:"lease_id"`
	}
	readJSONFile(t, filepath.Join(runDir, "env.json"), &coordinator)
	if coordinator.ExecutedOn != r.nodeE.id {
		t.Fatalf("env.json executed_on = %q, want the embedded node %q",
			coordinator.ExecutedOn, r.nodeE.id)
	}
	if coordinator.LeaseID != claim.LeaseID {
		t.Fatalf("env.json lease_id = %q, want %q", coordinator.LeaseID, claim.LeaseID)
	}
	if coordinator.OriginHost != "coordinator.test" {
		t.Fatalf("env.json origin_host = %q, want the coordinator's own name",
			coordinator.OriginHost)
	}

	// env.node.json is a promoted manifest path, so a node's own staging
	// record has somewhere to land; env.json is reserved and is the
	// coordinator's alone.
	if !quarantine.ReservedPath("env.json") {
		t.Error("env.json is not reserved: the coordinator's record is not protected")
	}
	if quarantine.ReservedPath(envNodeJSONPath) {
		t.Errorf("%s is reserved: the node's staging record cannot be offered", envNodeJSONPath)
	}
}

// envNodeJSONPath is the manifest path a node's own provenance record is
// promoted under (D52). The node agent spells it too; naming it here keeps the
// assertion above readable rather than reaching across the package boundary.
const envNodeJSONPath = "env.node.json"

// readJSONFile decodes a JSON file into v.
func readJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

// TestCudaJobWaitsForACudaNode covers AC6 and the first of D40's two
// deliberate behavior changes: with a pool attached, a device no node declares
// is a placement fact rather than a submit error, so the job is admitted and
// queued and runs the moment such a node enrolls. The embedded node's own cpu
// device, declared from RUNNERD_ALLOWED_DEVICES, still matches a cpu job.
func TestCudaJobWaitsForACudaNode(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{
		embedded: true,
		rig:      rigConfig{allowedDevices: map[string]bool{"cpu": true}},
	})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)

	cuda := baseSpec("needs-cuda", "fake-quick")
	cuda["device"] = "cuda"
	resp := r.do(http.MethodPost, "/harness/jobs", cuda)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit a cuda job: got %d, want 201 (it waits, it is not refused)", resp.StatusCode)
	}
	resp.Body.Close()

	// No node declares cuda, so the claim is held rather than answered.
	claim, hold, err := c.Claim(context.Background(), nashnet.ClaimRequest{
		NodeID: r.nodeE.id, SlotsFree: 1,
		Capabilities: declaration(1), GateReport: admitReport(1),
	})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if claim != nil {
		t.Fatalf("a cpu-only node was handed the cuda job: %+v", claim)
	}
	if hold == nil || hold.Hold == "" {
		t.Fatal("a held claim carried no hold reason")
	}
	if view, ok := r.disp.resolveView("needs-cuda"); !ok || view.State != StateQueued {
		t.Fatalf("cuda job view = %+v (ok=%v), want it still queued", view, ok)
	}

	// The same node takes a cpu job at once, which is the other half of AC6:
	// the allowed-devices env is now the node's declaration, not a submit gate.
	r.queueJob(t, JobSpec{Name: "plain-cpu", Kind: KindTrain, Device: "cpu"})
	got := r.claimEmbedded(t, c)
	if got.JobID != "plain-cpu" {
		t.Fatalf("claim handed out %q, want plain-cpu", got.JobID)
	}
}

// TestZeroNodeDaemonKeepsTheV1Surface covers AC2: with no pool attached the
// daemon is the v1.0 one. Its artifacts listing is compared row for row
// against the walk-derived listing it must reproduce, and its submit still
// refuses a device this runner does not enable, which is the refusal the pool
// path deliberately drops.
func TestZeroNodeDaemonKeepsTheV1Surface(t *testing.T) {
	r := newRig(t, rigConfig{allowedDevices: map[string]bool{"cpu": true}})

	cuda := baseSpec("cuda-refused", "fake-quick")
	cuda["device"] = "cuda"
	resp := r.do(http.MethodPost, "/harness/jobs", cuda)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("submit a cuda job with no pool: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.do(http.MethodPost, "/harness/jobs", baseSpec("local-job", "fake-quick"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("local-job", procmgr.StatusStopped, 5*time.Second)

	runDir := filepath.Join(r.runsDir, "local-job")
	if err := os.WriteFile(filepath.Join(runDir, "metrics.jsonl"), []byte("{\"iter\":1}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	resp = r.do(http.MethodGet, "/harness/jobs/local-job/artifacts", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("artifacts: got %d, want 200", resp.StatusCode)
	}
	var served struct {
		Artifacts []Artifact `json:"artifacts"`
	}
	decodeInto(t, resp, &served)

	walk := walkArtifacts(t, runDir)
	if len(served.Artifacts) != len(walk) {
		t.Fatalf("listing has %d rows, the walk has %d:\n%+v\n%+v",
			len(served.Artifacts), len(walk), served.Artifacts, walk)
	}
	for i, got := range served.Artifacts {
		want := walk[i]
		if got.Path != want.Path || got.Size != want.Size || got.SHA256 != want.SHA256 {
			t.Fatalf("row %d = %+v, the walk has %+v", i, got, want)
		}
	}
}

// TestReconcileLeavesALeasedRowToItsLeaseHolder pins the restart rule the
// embedded node makes load-bearing. An in-place run's process.json carries no
// Host, because the node's own liveness probe needs the local pid; that is
// exactly the shape Reconcile adopts for a v1.0 job it launched itself. With a
// live lease on the job, adopting it would put a second watcher on one process
// and race two terminal writes against the lease holder's result post.
func TestReconcileLeavesALeasedRowToItsLeaseHolder(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{embedded: true})
	c := r.loopbackClient(t)
	r.registerEmbedded(t, c, 1)
	r.queueJob(t, JobSpec{Name: "leased-row"})
	claim := r.claimEmbedded(t, c)
	if claim.JobID != "leased-row" {
		t.Fatalf("claimed %q, want leased-row", claim.JobID)
	}

	// The shape an in-place launch leaves behind: a running row this host can
	// probe, with no Host to mark it remote.
	child := startReattachedChild(t, r.runsDir, "leased-row", "fake", false)
	defer child.kill(t)

	r.disp.Reconcile()

	if active := r.disp.slots.Active(); active != 0 {
		t.Fatalf("dispatcher claimed %d slots for a leased job, want 0", active)
	}
	st := readProcessState(t, r.runsDir, "leased-row")
	if st.Status != procmgr.StatusRunning {
		t.Fatalf("status = %q, want running: the lease holder finalizes it, not Reconcile", st.Status)
	}
}

// TestPoolNarrowsTheSubmitPreflightToDisk is D19. RAM and VRAM describe
// whichever host ends up running the job, which the coordinator does not know
// at submit and may not be, so with a pool attached they are node gates,
// measured on the node and reported in its gate report; a breach between claim
// and Prepare is a nack, not a submit refusal. Disk stays, and stays measured
// on the coordinator's runs dir, because every node's artifacts land there.
func TestPoolNarrowsTheSubmitPreflightToDisk(t *testing.T) {
	starved := rigConfig{
		minRAMGB: 8,
		ramQuery: func() (float64, error) { return 0.1, nil },
	}

	// With no pool this host runs the job, so the floor describes it and the
	// v1.0 refusal stands.
	local := newRig(t, starved)
	resp := local.do(http.MethodPost, "/harness/jobs", baseSpec("ram-starved", "fake-quick"))
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("submit with no pool: got %d, want 412", resp.StatusCode)
	}
	resp.Body.Close()

	pooled := newPoolRig(t, poolRigConfig{embedded: true, rig: starved})
	resp = pooled.do(http.MethodPost, "/harness/jobs", baseSpec("ram-starved", "fake-quick"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit with a pool: got %d, want 201 (RAM is a node gate)", resp.StatusCode)
	}
	resp.Body.Close()

	// The disk floor is not narrowed away: it still refuses on the
	// coordinator's own runs dir.
	full := newPoolRig(t, poolRigConfig{
		embedded: true,
		rig:      rigConfig{minDiskGB: 1e9},
	})
	resp = full.do(http.MethodPost, "/harness/jobs", baseSpec("disk-starved", "fake-quick"))
	if resp.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("submit with no disk: got %d, want 412 (the disk floor stays)", resp.StatusCode)
	}
	resp.Body.Close()
}
