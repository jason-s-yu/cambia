package nodeagent

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// TestClaimToResultCycle drives one whole lease against the stub coordinator
// with a real forked job (a shell script), covering W2-T11 AC1 (the full
// cycle), AC7 (no reserved path in the uploaded set) and AC9 (only D5 phases,
// never stopping).
//
// It is also the one scheduling test that runs over both transports of D65.
// The two legs share this body verbatim, the stub coordinator included: the
// HTTPS leg reaches it over a pinned TLS socket and the loopback leg hands it
// the same request in process. Anything the loopback shortcut would skip shows
// up here as a leg that disagrees with the other.
func TestClaimToResultCycle(t *testing.T) {
	for _, tr := range transports() {
		t.Run(tr.name, func(t *testing.T) { claimToResultCycle(t, tr) })
	}
}

func claimToResultCycle(t *testing.T, tr transportCase) {
	stub := newStubCoordinator(t)
	worktree := t.TempDir()
	writeFakeJob(t, worktree)

	var runsDir string
	agent, _ := testAgent(t, stub, func(o *Options) {
		runsDir = o.Config.RunsDir
		o.Env = &fakeEnv{worktree: worktree, runsDir: o.Config.RunsDir, writeEnvJSON: true}
		o.Launcher = NewLauncher(procmgr.NewProcessManager(o.Config.RunsDir, "", "cambia", nil, nil))
		tr.apply(t, stub, o)
	})

	spec := Spec{
		Kind:   KindMeasure,
		Name:   "v0.4-prtcfr-r13",
		Commit: strings.Repeat("a", 40),
		Script: "cfr/scripts/fake.sh",
		Args:   []string{filepath.Join(runsDir, "v0.4-prtcfr-r13")},
	}
	stub.enqueue(nashnet.ClaimResponse{
		JobID:      spec.Name,
		LeaseID:    "01JTESTLEASE0000000000000A",
		LeaseEpoch: 7,
		Attempt:    1,
		Spec:       specJSON(t, spec),
		Snapshot: nashnet.SnapshotRef{
			URL:    "/nashnet/leases/01JTESTLEASE0000000000000A/snapshot",
			Commit: spec.Commit,
			SHA256: stub.snapshotDigest(),
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	agent.claimOnce(ctx)
	agent.wg.Wait()

	_, results, nacks, commits, phases := stub.snapshotState()
	if len(nacks) != 0 {
		t.Fatalf("expected no nack, got %+v", nacks)
	}
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	if results[0].State != nashnet.ResultStopped {
		t.Fatalf("result state = %q, want %q", results[0].State, nashnet.ResultStopped)
	}
	if results[0].ExitCode == nil || *results[0].ExitCode != 0 {
		t.Fatalf("result exit code = %v, want 0", results[0].ExitCode)
	}
	if results[0].FinalManifestDigest == "" {
		t.Fatalf("result carried no final manifest digest")
	}
	if len(commits) == 0 || !commits[len(commits)-1].Final {
		t.Fatalf("last commit is not final: %+v", commits)
	}

	uploaded := map[string]bool{}
	for _, c := range commits {
		for _, e := range c.Entries {
			uploaded[e.Path] = true
		}
	}
	for _, want := range []string{"metrics.jsonl", "snapshots/prtcfr_checkpoint.pt", envNodeJSONName} {
		if !uploaded[want] {
			t.Errorf("expected %s in the uploaded set, got %v", want, keys(uploaded))
		}
	}
	for _, reserved := range []string{
		"process.json", "env.json", "run_db.sqlite-wal", "reservoir/meta.json",
		"logs/training.log", "lease.json", "jobspec.json",
	} {
		if uploaded[reserved] {
			t.Errorf("reserved path %s reached the uploaded set", reserved)
		}
	}

	assertPhaseVocabulary(t, phases)

	stub.mu.Lock()
	logs := string(stub.logs)
	stub.mu.Unlock()
	if !strings.Contains(logs, "job finished") {
		t.Errorf("log append never carried the job output, got %q", logs)
	}
}

// TestGateBreachBeforePrepareNacks covers AC2: a gate that passed at claim and
// no longer holds when Prepare is about to run returns the claim with a
// cooldown rather than failing the job.
func TestGateBreachBeforePrepareNacks(t *testing.T) {
	stub := newStubCoordinator(t)
	prober := newFakeProber()
	floor := 32.0
	agent, _ := testAgent(t, stub, func(o *Options) {
		o.Prober = prober
		o.Config.Gates.Floors = &gates.FloorsGate{FreeRAMGB: &floor}
	})
	prober.setFreeRAM(1.0)

	job := newTestJob(t, agent, stub, Spec{Kind: KindTrain, Name: "job-gated", Commit: strings.Repeat("b", 40)})
	if job.stage(context.Background()) {
		t.Fatalf("stage should have returned false on a gate breach")
	}

	_, _, nacks, _, _ := stub.snapshotState()
	if len(nacks) != 1 {
		t.Fatalf("expected one nack, got %d", len(nacks))
	}
	if nacks[0].Reason != nashnet.NackGateBreach {
		t.Fatalf("nack reason = %q, want %q", nacks[0].Reason, nashnet.NackGateBreach)
	}
	if nacks[0].CooldownSeconds <= 0 {
		t.Fatalf("gate breach nack carried no cooldown")
	}
}

// TestPrepareFailureClassification covers AC3: a node-local staging failure
// nacks prepare_node_failed with the D63 cooldown, and a spec-fatal one fails
// the job with no retry.
func TestPrepareFailureClassification(t *testing.T) {
	t.Run("node local nacks", func(t *testing.T) {
		stub := newStubCoordinator(t)
		agent, _ := testAgent(t, stub, func(o *Options) {
			o.Env = &fakeEnv{
				worktree:   t.TempDir(),
				runsDir:    o.Config.RunsDir,
				prepareErr: errors.New("venv: uv sync failed: exit status 1"),
			}
		})
		job := newTestJob(t, agent, stub, Spec{Kind: KindTrain, Name: "job-uv", Commit: strings.Repeat("c", 40)})
		if job.stage(context.Background()) {
			t.Fatalf("stage should have returned false")
		}
		_, results, nacks, _, _ := stub.snapshotState()
		if len(nacks) != 1 || nacks[0].Reason != nashnet.NackPrepareNodeFailed {
			t.Fatalf("expected a prepare_node_failed nack, got %+v", nacks)
		}
		if nacks[0].CooldownSeconds != prepareFailedCooldownSeconds {
			t.Fatalf("cooldown = %d, want %d", nacks[0].CooldownSeconds, prepareFailedCooldownSeconds)
		}
		if len(results) != 0 {
			t.Fatalf("a node-local prepare failure must not post a result, got %+v", results)
		}
	})

	t.Run("spec fatal fails", func(t *testing.T) {
		stub := newStubCoordinator(t)
		agent, _ := testAgent(t, stub, func(o *Options) {
			o.Env = &fakeEnv{
				worktree:   t.TempDir(),
				runsDir:    o.Config.RunsDir,
				prepareErr: errors.New("render config: override targets harness-owned key device"),
			}
		})
		job := newTestJob(t, agent, stub, Spec{Kind: KindTrain, Name: "job-render", Commit: strings.Repeat("d", 40)})
		if job.stage(context.Background()) {
			t.Fatalf("stage should have returned false")
		}
		_, results, nacks, _, _ := stub.snapshotState()
		if len(nacks) != 0 {
			t.Fatalf("a spec-fatal prepare failure must not nack, got %+v", nacks)
		}
		if len(results) != 1 || results[0].State != nashnet.ResultFailed {
			t.Fatalf("expected one failed result, got %+v", results)
		}
	})
}

// TestRevokeOnEventsStopsJobGroup covers AC4's first half: a revoke arriving on
// the events poll stops the job group within one round trip and settles the
// lease as canceled.
func TestRevokeOnEventsStopsJobGroup(t *testing.T) {
	stub := newStubCoordinator(t)
	launcher := &fakeLauncher{status: ProcessStatus{Status: "running", PID: 99, Found: true}}
	agent, _ := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })

	job := newRunningJob(t, agent, stub, "job-revoked")
	done := runJobInBackground(t, agent, job)

	agent.applyEvents(nashnet.EventsResponse{
		Events:    []nashnet.Event{{Type: nashnet.EventRevoke, LeaseID: job.rec.LeaseID}},
		NodeEpoch: 3,
	})
	waitFor(t, done)

	if launcher.stopCount() == 0 {
		t.Fatalf("revoke did not stop the job group")
	}
	_, results, _, _, _ := stub.snapshotState()
	if len(results) != 1 || results[0].State != nashnet.ResultCanceled {
		t.Fatalf("expected one canceled result, got %+v", results)
	}
}

// TestFencedLeaseStopsJob covers AC4's second half (D36): either 401 or
// 409 lease_superseded from a lease route stops the job group within one
// progress interval, and the node writes nothing further for that lease.
func TestFencedLeaseStopsJob(t *testing.T) {
	cases := []struct {
		name   string
		status int
		code   string
	}{
		{"unauthorized", 401, "unauthorized"},
		{"lease superseded", 409, nashnet.CodeLeaseSuperseded},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stub := newStubCoordinator(t)
			launcher := &fakeLauncher{status: ProcessStatus{Status: "running", PID: 7, Found: true}}
			agent, _ := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })

			job := newRunningJob(t, agent, stub, "job-fenced")
			stub.setProgressFailure(tc.status, tc.code)
			done := runJobInBackground(t, agent, job)
			waitFor(t, done)

			if launcher.stopCount() == 0 {
				t.Fatalf("a fenced lease did not stop the job group")
			}
			_, results, _, _, _ := stub.snapshotState()
			if len(results) != 0 {
				t.Fatalf("an orphaned lease must post no result, got %+v", results)
			}
			if _, err := os.Stat(leasePath(agent.cfg.BaseDir, job.rec.JobID)); !os.IsNotExist(err) {
				t.Fatalf("orphaned lease record survived: %v", err)
			}
			if _, err := os.Stat(job.runDir()); err != nil {
				t.Fatalf("orphaned run dir must be kept for the debug ttl: %v", err)
			}
		})
	}
}

// TestLossOfContactStopsJob covers AC5: no coordinator contact for two lease
// TTLs obliges the node to stop, whether or not a fencing code ever arrived.
func TestLossOfContactStopsJob(t *testing.T) {
	stub := newStubCoordinator(t)
	launcher := &fakeLauncher{status: ProcessStatus{Status: "running", PID: 11, Found: true}}
	clock := &testClock{now: time.Now()}
	agent, _ := testAgent(t, stub, func(o *Options) {
		o.Launcher = launcher
		o.Now = clock.Now
	})
	// A 500 is neither 401 nor 409: only the elapsed-silence rule can stop
	// this job.
	stub.setProgressFailure(500, "boom")

	job := newRunningJob(t, agent, stub, "job-silent")
	clock.advance(10 * time.Second)
	done := runJobInBackground(t, agent, job)
	waitFor(t, done)

	if launcher.stopCount() == 0 {
		t.Fatalf("loss of contact did not stop the job group")
	}
	_, results, _, _, _ := stub.snapshotState()
	if len(results) != 0 {
		t.Fatalf("a silence-orphaned lease must post no result, got %+v", results)
	}
}

// TestRestartReattachesAndResumesUpload covers AC6: a restarted agent
// reattaches its live rows, re-registers naming them in live_leases, and
// resumes an interrupted upload at the HEAD offset.
func TestRestartReattachesAndResumesUpload(t *testing.T) {
	stub := newStubCoordinator(t)
	launcher := &fakeLauncher{status: ProcessStatus{Status: procmgr.StatusStopped, PID: 5, Found: true}}
	agent, cfg := testAgent(t, stub, func(o *Options) { o.Launcher = launcher })

	spec := Spec{Kind: KindTrain, Name: "job-restart", Commit: strings.Repeat("e", 40)}
	runDir := filepath.Join(cfg.RunsDir, spec.Name)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := []byte(strings.Repeat("checkpoint-bytes", 16))
	if err := os.WriteFile(filepath.Join(runDir, "metrics.jsonl"), payload, 0o644); err != nil {
		t.Fatal(err)
	}
	digest, err := hashFile(filepath.Join(runDir, "metrics.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	// Half the blob is already in the coordinator's part file, as it would be
	// after an agent restart mid-chunk.
	const resumeAt = 100
	stub.mu.Lock()
	stub.parts[digest] = append([]byte(nil), payload[:resumeAt]...)
	stub.token = "lease-token-restart"
	stub.mu.Unlock()

	rec := &leaseRecord{
		JobID: spec.Name, LeaseID: "01JRESTART000000000000000A", LeaseEpoch: 2,
		Token: "lease-token-restart", Attempt: 1, Commit: spec.Commit,
		Spec: specJSON(t, spec), Policy: fastPolicy(), Launched: true,
		Phase: nashnet.PhaseRunning,
	}
	if err := writeLeaseRecord(cfg.BaseDir, rec); err != nil {
		t.Fatal(err)
	}

	live, pending := agent.reattach()
	if len(live) != 1 || live[0].LeaseID != rec.LeaseID {
		t.Fatalf("reattach did not name the live lease: %+v", live)
	}
	if live[0].TokenHash != nashnet.HashLeaseToken(rec.Token) {
		t.Fatalf("live lease carried the wrong token hash")
	}
	ctx := context.Background()
	rebound, err := agent.register(ctx, live)
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	stub.mu.Lock()
	sawLive := len(stub.registers) == 1 && len(stub.registers[0].LiveLeases) == 1
	stub.mu.Unlock()
	if !sawLive {
		t.Fatalf("register did not carry live_leases")
	}
	agent.resumePending(ctx, pending, rebound)
	agent.wg.Wait()

	stub.mu.Lock()
	stored := append([]byte(nil), stub.blobs[digest]...)
	starts := append([]int64(nil), stub.patchStarts[digest]...)
	heads := stub.headProbes[digest]
	stub.mu.Unlock()
	if string(stored) != string(payload) {
		t.Fatalf("resumed upload did not reconstruct the blob (%d of %d bytes)", len(stored), len(payload))
	}
	if heads == 0 {
		t.Fatalf("the upload never probed the HEAD offset")
	}
	if len(starts) == 0 || starts[0] != resumeAt {
		t.Fatalf("upload restarted at %v, want a first chunk at the HEAD offset %d", starts, resumeAt)
	}
	_, results, _, commits, _ := stub.snapshotState()
	if len(commits) == 0 || !commits[len(commits)-1].Final {
		t.Fatalf("reattached job did not commit a final manifest")
	}
	if len(results) != 1 {
		t.Fatalf("reattached job did not post a result: %+v", results)
	}
}

// TestProgressRefusesPhaseOutsideVocabulary covers AC9 directly: stopping is
// the coordinator's word and never leaves a node (D5).
func TestProgressRefusesPhaseOutsideVocabulary(t *testing.T) {
	stub := newStubCoordinator(t)
	agent, _ := testAgent(t, stub, nil)
	job := newRunningJob(t, agent, stub, "job-phase")
	job.setPhase(nashnet.PhaseStopping)

	if _, err := job.postProgress(context.Background()); err == nil {
		t.Fatalf("postProgress accepted the phase %q", nashnet.PhaseStopping)
	}
	progress, _, _, _, _ := stub.snapshotState()
	if len(progress) != 0 {
		t.Fatalf("a refused phase still reached the coordinator: %+v", progress)
	}
}

// helpers

// newTestJob builds a jobRun ready for a direct stage() call, with a claim
// already recorded on the stub so the snapshot route answers.
func newTestJob(t *testing.T, agent *Agent, stub *stubCoordinator, spec Spec) *jobRun {
	t.Helper()
	rec := &leaseRecord{
		JobID: spec.Name, LeaseID: "01JSTAGE00000000000000000A", LeaseEpoch: 1,
		Token: "lease-token-stage", Attempt: 1, Commit: spec.Commit,
		Spec: specJSON(t, spec), Policy: fastPolicy(), Phase: nashnet.PhaseClaimed,
	}
	stub.mu.Lock()
	stub.token = rec.Token
	stub.mu.Unlock()
	if err := writeLeaseRecord(agent.cfg.BaseDir, rec); err != nil {
		t.Fatal(err)
	}
	job := &jobRun{
		agent: agent, rec: rec, spec: spec,
		snapshot: nashnet.SnapshotRef{Commit: spec.Commit, SHA256: stub.snapshotDigest()},
	}
	job.init(nashnet.PhaseClaimed)
	return job
}

// newRunningJob builds a jobRun already past launch, for the fencing and
// revoke paths.
func newRunningJob(t *testing.T, agent *Agent, stub *stubCoordinator, name string) *jobRun {
	t.Helper()
	spec := Spec{Kind: KindTrain, Name: name, Commit: strings.Repeat("f", 40)}
	rec := &leaseRecord{
		JobID: name, LeaseID: "01JRUNNING0000000000000000", LeaseEpoch: 4,
		Token: "lease-token-running", Attempt: 1, Commit: spec.Commit,
		Spec: specJSON(t, spec), Policy: fastPolicy(), Launched: true,
		Phase: nashnet.PhaseRunning,
	}
	stub.mu.Lock()
	stub.token = rec.Token
	stub.mu.Unlock()
	if err := os.MkdirAll(filepath.Join(agent.cfg.RunsDir, name), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeLeaseRecord(agent.cfg.BaseDir, rec); err != nil {
		t.Fatal(err)
	}
	job := &jobRun{agent: agent, rec: rec, spec: spec}
	job.init(nashnet.PhaseRunning)
	agent.mu.Lock()
	agent.active[name] = job
	agent.mu.Unlock()
	return job
}

// runJobInBackground supervises a job on its own goroutine and reports when it
// leaves the loop.
func runJobInBackground(t *testing.T, agent *Agent, job *jobRun) chan struct{} {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		job.supervise(context.Background())
	}()
	return done
}

func waitFor(t *testing.T, done chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatalf("job did not settle within the timeout")
	}
}

// testClock is the injected clock for the loss-of-contact rule.
type testClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *testClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// assertPhaseVocabulary checks every reported phase against D5's node
// vocabulary and against the stopping exclusion.
func assertPhaseVocabulary(t *testing.T, phases []string) {
	t.Helper()
	if len(phases) == 0 {
		t.Fatalf("the node reported no phase at all")
	}
	for _, p := range phases {
		if p == nashnet.PhaseStopping {
			t.Errorf("node reported the coordinator-only phase %q", p)
		}
		if err := nashnet.ValidateNodePhase(p); err != nil {
			t.Errorf("node reported phase %q outside the D5 vocabulary: %v", p, err)
		}
	}
}

// writeFakeJob writes the tiny shell job the full-cycle test runs.
func writeFakeJob(t *testing.T, worktree string) {
	t.Helper()
	dir := filepath.Join(worktree, "cfr", "scripts")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "fake.sh")
	body := `run="$1"
mkdir -p "$run/snapshots" "$run/reservoir"
printf '{"iter":1}\n' > "$run/metrics.jsonl"
printf 'weights' > "$run/snapshots/prtcfr_checkpoint.pt"
printf 'wal' > "$run/run_db.sqlite-wal"
printf 'sampler' > "$run/reservoir/meta.json"
echo "job finished"
`
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
