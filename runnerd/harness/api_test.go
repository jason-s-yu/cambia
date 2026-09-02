package harness

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

func TestAuthRequiredOnEveryRoute(t *testing.T) {
	r := newRig(t, rigConfig{})

	// Missing token -> 401.
	resp := r.doTok(http.MethodGet, "/harness/jobs", nil, "")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token: got %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()

	// Bad token -> 401.
	resp = r.doTok(http.MethodGet, "/harness/jobs", nil, "garbage.token.value")
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad token: got %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()

	// Token signed by a different key -> 401.
	_, otherPriv, _ := genOther(t)
	other := mintToken(t, otherPriv, "x", 0)
	resp = r.doTok(http.MethodGet, "/harness/jobs", nil, other)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("foreign token: got %d, want 401", resp.StatusCode)
	}
	resp.Body.Close()

	// Valid token -> 200.
	resp = r.do(http.MethodGet, "/harness/jobs", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("valid token: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestHealthIsTokenFree pins the one deliberate exception to bearer-on-every-
// route: GET /harness/health serves read-only counters without a token (LAN
// monitoring consumer; cambia-330/network-552). Everything else stays gated;
// see TestAuthRequiredOnEveryRoute.
func TestHealthIsTokenFree(t *testing.T) {
	r := newRig(t, rigConfig{})

	resp := r.doTok(http.MethodGet, "/harness/health", nil, "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("token-free health: got %d, want 200", resp.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode health body: %v", err)
	}
	resp.Body.Close()
	// EXACTLY these keys and nothing else: the route is token-free, so any new
	// field is exposed unauthenticated: a regression to writeJSON(w, 200, snap)
	// would leak queue/active JobViews (job IDs, commits, configs) while a
	// presence-only check stayed green. restart_preserves_jobs and build_commit
	// (cambia-655) are deliberate additions: neither identifies a job. features
	// is likewise deliberate (D30): a capability list, not host detail.
	// free_ram_gb/free_disk_gb describe the coordinator host rather than the
	// queue and are dropped from the token-free payload by D30 (ruling q3);
	// TestHealthAuthenticatedCarriesHostCounters pins their authenticated return.
	want := []string{
		"reconciled_at", "jobs_running", "queue_depth",
		"restart_preserves_jobs", "build_commit", "features",
	}
	for _, k := range want {
		if _, ok := body[k]; !ok {
			t.Fatalf("health body missing key %q: %v", k, body)
		}
	}
	if _, ok := body["free_ram_gb"]; ok {
		t.Fatalf("token-free health body must not carry free_ram_gb: %v", body)
	}
	if _, ok := body["free_disk_gb"]; ok {
		t.Fatalf("token-free health body must not carry free_disk_gb: %v", body)
	}
	if len(body) != len(want) {
		t.Fatalf("token-free health body must carry exactly %d counter keys, got %d: %v", len(want), len(body), body)
	}
}

// TestHealthRestartSemanticsFields pins the two cambia-655 health fields an
// operator reads before a redeploy: restart_preserves_jobs must be a bool that
// tracks the daemon's kill-on-stop policy (true for the job-preserving default),
// and build_commit a string identifying the serving binary ("dev" when the
// binary was not stamped with -X main.buildCommit).
func TestHealthRestartSemanticsFields(t *testing.T) {
	// Default daemon: SIGTERM detaches, so health advertises job preservation.
	r := newRig(t, rigConfig{})
	body := r.healthBody()
	if preserves, ok := body["restart_preserves_jobs"].(bool); !ok || !preserves {
		t.Fatalf("restart_preserves_jobs = %v, want bool true", body["restart_preserves_jobs"])
	}
	if commit, ok := body["build_commit"].(string); !ok || commit != "dev" {
		t.Fatalf("build_commit = %v, want \"dev\" for an unstamped build", body["build_commit"])
	}

	// A daemon started with RUNNERD_KILL_JOBS_ON_STOP reports the opposite, so a
	// monitoring consumer is never told jobs survive a stop when they do not.
	killer := newRig(t, rigConfig{killJobsOnStop: true, buildCommit: "deadbeef"})
	kbody := killer.healthBody()
	if preserves, _ := kbody["restart_preserves_jobs"].(bool); preserves {
		t.Fatal("restart_preserves_jobs = true with KillJobsOnStop set, want false")
	}
	if commit, _ := kbody["build_commit"].(string); commit != "deadbeef" {
		t.Fatalf("build_commit = %v, want the stamped sha", kbody["build_commit"])
	}
}

func TestSubmitInvalidNameAndKind(t *testing.T) {
	r := newRig(t, rigConfig{})

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("../evil", "fake"))
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad name: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.do(http.MethodPost, "/harness/jobs", baseSpec("ok-name", "not-a-kind"))
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad kind: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Absolute config path -> 400 invalid_path.
	spec := baseSpec("ok-name2", "fake")
	spec["config"] = "/etc/passwd"
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad path: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestSubmitAfterValidation covers the cambia-352 dependency admission checks:
// an unknown parent, a self-reference, and a bad on_failure are all rejected;
// a terminal parent is accepted (the gate resolves its outcome at dispatch).
func TestSubmitAfterValidation(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})

	// Unknown parent -> 400 after_not_found.
	spec := baseSpec("child-unknown", "fake")
	spec["after"] = "no-such-parent"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("after unknown parent: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Self-reference -> 400 invalid_after.
	spec = baseSpec("self-ref", "fake")
	spec["after"] = "self-ref"
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("after self-reference: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Bad on_failure -> 400 invalid_on_failure.
	spec = baseSpec("bad-onfail", "fake")
	spec["on_failure"] = "explode"
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad on_failure: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Terminal parent is allowed (accepted, gate resolves at dispatch).
	if err := procmgr.WriteProcessState(filepath.Join(r.runsDir, "term-parent"),
		&procmgr.ProcessState{Name: "term-parent", Status: procmgr.StatusStopped}); err != nil {
		t.Fatal(err)
	}
	spec = baseSpec("child-ok", "fake")
	spec["after"] = "term-parent"
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("after terminal parent: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestSubmitAfterListValidation (D29, cambia-1713): the list wire shape of
// `after` runs through the same per-parent checks as the single-string shape
// -- an unknown parent and a self-reference anywhere in the list are both
// rejected -- and a list of terminal parents is accepted.
func TestSubmitAfterListValidation(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 4})
	for _, name := range []string{"al-p1", "al-p2", "al-p3"} {
		if err := procmgr.WriteProcessState(filepath.Join(r.runsDir, name),
			&procmgr.ProcessState{Name: name, Status: procmgr.StatusStopped}); err != nil {
			t.Fatal(err)
		}
	}

	// Unknown parent inside a list -> 400 after_not_found.
	spec := baseSpec("al-child-unknown", "fake")
	spec["after"] = []string{"al-p1", "no-such-parent"}
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("after list unknown parent: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Self-reference inside a list -> 400 invalid_after.
	spec = baseSpec("al-self-ref", "fake")
	spec["after"] = []string{"al-p1", "al-self-ref"}
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("after list self-reference: got %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()

	// Every parent terminal -> accepted.
	spec = baseSpec("al-child-ok", "fake")
	spec["after"] = []string{"al-p1", "al-p2", "al-p3"}
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("after list all terminal: got %d, want 201: %s", resp.StatusCode, readBody(resp))
	}
	resp.Body.Close()
}

// TestSubmitDependencyDepthExceeded (D29) pins the exact 32-level boundary: a
// new job naming gen31 as its parent (own depth 31) reaches depth 32 and is
// accepted; naming gen32 (own depth 32) reaches depth 33 and is rejected.
func TestSubmitDependencyDepthExceeded(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})

	// Build gen00..gen32 (33 nodes, depth(genN) == N), each gen(i) after
	// gen(i-1), entirely via disk records: the depth check reads jobspec.json
	// directly, so no dispatcher submission is needed to seed the chain.
	const deepest = 32
	prev := ""
	for i := 0; i <= deepest; i++ {
		name := fmt.Sprintf("dd-gen%02d", i)
		st := &procmgr.ProcessState{Name: name, Status: procmgr.StatusStopped}
		if err := procmgr.WriteProcessState(filepath.Join(r.runsDir, name), st); err != nil {
			t.Fatal(err)
		}
		var after []string
		if prev != "" {
			after = []string{prev}
		}
		if err := writeJobSpec(filepath.Join(r.runsDir, name), &JobSpec{Name: name, Kind: "fake", After: after}); err != nil {
			t.Fatal(err)
		}
		prev = name
	}

	// gen31's own depth is 31; a dependent on it reaches exactly 32 -- accepted.
	spec := baseSpec("dd-at-cap", "fake")
	spec["after"] = "dd-gen31"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("depth-32 (at cap): got %d, want 201: %s", resp.StatusCode, readBody(resp))
	}
	resp.Body.Close()

	// gen32's own depth is 32; a dependent on it reaches 33 -- one over cap,
	// rejected.
	spec = baseSpec("dd-over-cap", "fake")
	spec["after"] = "dd-gen32"
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("depth-33 (over cap): got %d, want 400", resp.StatusCode)
	}
	var body map[string]any
	decodeBody(t, resp, &body)
	if body["error"] != "dependency_depth_exceeded" {
		t.Fatalf("error = %v, want dependency_depth_exceeded", body["error"])
	}
}

// TestSubmitTolerantOfUnknownFields (AC6): the submit decoder still ignores an
// unrecognized field rather than rejecting the request, unchanged by the
// custom JobSpec.UnmarshalJSON this ticket adds for the after list shape.
func TestSubmitTolerantOfUnknownFields(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	spec := baseSpec("unknown-field-ok", "fake")
	spec["totally_made_up_field"] = "some-future-client-sent-this"
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("unknown field: got %d, want 201: %s", resp.StatusCode, readBody(resp))
	}
	resp.Body.Close()
}

// TestStoredSingleStringAfterStillDecodes (AC1): a jobspec.json written by a
// pre-r2 daemon (or a v1.0 client) with a bare `after` string still decodes
// after the field widened to []string.
func TestStoredSingleStringAfterStillDecodes(t *testing.T) {
	dir := t.TempDir()
	raw := `{"kind":"train","name":"legacy","commit":"` + strings.Repeat("a", 40) + `","after":"legacy-parent","on_failure":"run"}`
	if err := os.WriteFile(filepath.Join(dir, jobSpecFile), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	spec := readJobSpec(dir)
	if spec == nil {
		t.Fatal("readJobSpec returned nil for a stored single-string after")
	}
	if len(spec.After) != 1 || spec.After[0] != "legacy-parent" {
		t.Fatalf("After = %v, want [\"legacy-parent\"]", spec.After)
	}
	if spec.OnFailure != "run" {
		t.Fatalf("OnFailure = %q, want run", spec.OnFailure)
	}
}

// readBody drains and returns resp's body as a string, for a failure message;
// it does not close resp.
func readBody(resp *http.Response) string {
	b, _ := io.ReadAll(resp.Body)
	return string(b)
}

func TestSubmitNameCollision(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("dup", "fake"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("first submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("dup", procmgr.StatusRunning, 3*time.Second)

	// Second submit of the same name -> 409 even with force.
	spec := baseSpec("dup", "fake")
	spec["force"] = true
	resp = r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("collision with force: got %d, want 409 (never forceable)", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestQueueAdmissionCapAnd429(t *testing.T) {
	// cap=1, small queue so 429 is reachable.
	r := newRig(t, rigConfig{maxJobs: 1, maxQueue: 2})

	// A dispatches and runs (occupies the single slot).
	var a submitResp
	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("job-a", "fake"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit a: got %d, want 201", resp.StatusCode)
	}
	decodeBody(t, resp, &a)
	r.waitForState("job-a", procmgr.StatusRunning, 3*time.Second)

	// B and C queue behind A (cap reached).
	for _, name := range []string{"job-b", "job-c"} {
		resp := r.do(http.MethodPost, "/harness/jobs", baseSpec(name, "fake"))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("submit %s: got %d, want 201", name, resp.StatusCode)
		}
		var sr submitResp
		decodeBody(t, resp, &sr)
		if sr.State != StateQueued {
			t.Fatalf("submit %s: state %q, want queued", name, sr.State)
		}
	}

	// Only A is running; queue depth is 2.
	var h map[string]any
	resp = r.do(http.MethodGet, "/harness/health", nil)
	decodeBody(t, resp, &h)
	if got := jnum(h["jobs_running"]); got != 1 {
		t.Fatalf("jobs_running = %v, want 1", h["jobs_running"])
	}
	if got := jnum(h["queue_depth"]); got != 2 {
		t.Fatalf("queue_depth = %v, want 2", h["queue_depth"])
	}

	// Queue is full -> 429.
	resp = r.do(http.MethodPost, "/harness/jobs", baseSpec("job-d", "fake"))
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("queue full: got %d, want 429", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestCancelQueuedAndRunning(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1, maxQueue: 4})

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("run-1", "fake"))
	resp.Body.Close()
	r.waitForState("run-1", procmgr.StatusRunning, 3*time.Second)

	resp = r.do(http.MethodPost, "/harness/jobs", baseSpec("queued-1", "fake"))
	resp.Body.Close()
	// queued-1 is behind run-1.
	if s, _ := r.getState("queued-1"); s != StateQueued {
		t.Fatalf("queued-1 state %q, want queued", s)
	}

	// Cancel the queued job -> dropped, marked canceled.
	resp = r.do(http.MethodDelete, "/harness/jobs/queued-1", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("cancel queued: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("queued-1", StateCanceled, 2*time.Second)

	// Cancel the running job (graceful) -> stops.
	resp = r.do(http.MethodDelete, "/harness/jobs/run-1", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("cancel running: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("run-1", procmgr.StatusStopped, 5*time.Second)
}

func TestCancelRunningForce(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("force-run", "fake"))
	resp.Body.Close()
	r.waitForState("force-run", procmgr.StatusRunning, 3*time.Second)

	// ?force sends SIGKILL immediately.
	resp = r.do(http.MethodDelete, "/harness/jobs/force-run?force=true", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("force cancel: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	// A SIGKILL-terminated child is recorded stopped (stopRequested) or crashed;
	// either way it reaches a terminal state promptly.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if s, _ := r.getState("force-run"); isTerminal(s) {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatal("force-run did not reach terminal after SIGKILL")
}

func TestPurgeRules(t *testing.T) {
	r := newRig(t, rigConfig{})

	// A terminal job on disk can be purged.
	termDir := filepath.Join(r.runsDir, "term")
	if err := procmgr.WriteProcessState(termDir, &procmgr.ProcessState{Name: "term", Status: procmgr.StatusStopped}); err != nil {
		t.Fatal(err)
	}
	resp := r.do(http.MethodDelete, "/harness/jobs/term?purge=true", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("purge terminal: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if _, err := os.Stat(termDir); !os.IsNotExist(err) {
		t.Fatalf("purge did not remove run dir: %v", err)
	}

	// A non-terminal job cannot be purged.
	liveDir := filepath.Join(r.runsDir, "live")
	if err := procmgr.WriteProcessState(liveDir, &procmgr.ProcessState{Name: "live", Status: procmgr.StatusCreated}); err != nil {
		t.Fatal(err)
	}
	resp = r.do(http.MethodDelete, "/harness/jobs/live?purge=true", nil)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("purge non-terminal: got %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()
	if _, err := os.Stat(liveDir); err != nil {
		t.Fatalf("refused purge should keep run dir: %v", err)
	}
}

func TestResumeGateRefusal(t *testing.T) {
	r := newRig(t, rigConfig{})

	// A stopped job with no resume_state.json / checkpoint refuses resume.
	dir := filepath.Join(r.runsDir, "stopped-1")
	if err := procmgr.WriteProcessState(dir, &procmgr.ProcessState{Name: "stopped-1", Status: procmgr.StatusStopped, Algorithm: "fake"}); err != nil {
		t.Fatal(err)
	}
	resp := r.do(http.MethodPost, "/harness/jobs/stopped-1/resume", nil)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("resume without state: got %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()

	// Unknown job -> 404.
	resp = r.do(http.MethodPost, "/harness/jobs/nope/resume", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("resume unknown: got %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestResumeGatePassesRelaunches(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})

	dir := filepath.Join(r.runsDir, "resumable")
	if err := os.MkdirAll(filepath.Join(dir, "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "snapshots", "prtcfr_checkpoint.pt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "resume_state.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte("device: cpu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := procmgr.WriteProcessState(dir, &procmgr.ProcessState{Name: "resumable", Status: procmgr.StatusStopped, Algorithm: "fake"}); err != nil {
		t.Fatal(err)
	}
	if err := writeJobSpec(dir, &JobSpec{Name: "resumable", Kind: "fake"}); err != nil {
		t.Fatal(err)
	}

	resp := r.do(http.MethodPost, "/harness/jobs/resumable/resume", nil)
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("resume with state: got %d, want 202", resp.StatusCode)
	}
	resp.Body.Close()
	// The relaunch dispatches and runs the fake sleeper.
	r.waitForState("resumable", procmgr.StatusRunning, 3*time.Second)
}

func TestStubEnvironmentFailsJob(t *testing.T) {
	// With the production stub Environment, a submitted job fails (ingest not
	// wired) rather than launching.
	r := newRig(t, rigConfig{env: StubEnvironment{}, maxJobs: 1})
	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("stub-job", "fake"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("stub-job", StateFailed, 3*time.Second)
}

func TestListAndGet(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("list-1", "fake"))
	resp.Body.Close()
	r.waitForState("list-1", procmgr.StatusRunning, 3*time.Second)

	var listed struct {
		Jobs []JobView `json:"jobs"`
	}
	resp = r.do(http.MethodGet, "/harness/jobs", nil)
	decodeBody(t, resp, &listed)
	found := false
	for _, j := range listed.Jobs {
		if j.JobID == "list-1" {
			found = true
			if j.Commit != strings.Repeat("a", 40) {
				t.Fatalf("commit not surfaced: %q", j.Commit)
			}
		}
	}
	if !found {
		t.Fatal("list-1 not in job list")
	}

	var jr struct {
		Job         JobView `json:"job"`
		ResolvedSHA string  `json:"resolved_sha"`
	}
	resp = r.do(http.MethodGet, "/harness/jobs/list-1", nil)
	decodeBody(t, resp, &jr)
	if jr.ResolvedSHA != strings.Repeat("a", 40) {
		t.Fatalf("resolved_sha = %q", jr.ResolvedSHA)
	}
	if jr.Job.Kind != "fake" {
		t.Fatalf("kind = %q, want fake", jr.Job.Kind)
	}
}

// TestHealthPayload uses r.do, an AUTHENTICATED request (a valid bearer
// token), so the host counters are expected here per D30.
func TestHealthPayload(t *testing.T) {
	r := newRig(t, rigConfig{ramQuery: func() (float64, error) { return 42.0, nil }})
	r.disp.Reconcile() // stamps reconciled_at

	var h map[string]any
	resp := r.do(http.MethodGet, "/harness/health", nil)
	decodeBody(t, resp, &h)
	for _, k := range []string{"reconciled_at", "jobs_running", "queue_depth", "free_ram_gb", "free_disk_gb"} {
		if _, ok := h[k]; !ok {
			t.Fatalf("health missing key %q", k)
		}
	}
	if jnum(h["free_ram_gb"]) != 42.0 {
		t.Fatalf("free_ram_gb = %v, want 42", h["free_ram_gb"])
	}
	if h["reconciled_at"] == "" {
		t.Fatal("reconciled_at empty after Reconcile")
	}
}

// TestHealthFeaturesList pins the D30 capability list exactly, so a client's
// local capability gate has something stable to check against.
func TestHealthFeaturesList(t *testing.T) {
	r := newRig(t, rigConfig{})
	body := r.healthBody()
	raw, ok := body["features"].([]any)
	if !ok {
		t.Fatalf("features = %v (%T), want a list", body["features"], body["features"])
	}
	got := make([]string, len(raw))
	for i, v := range raw {
		got[i], _ = v.(string)
	}
	want := []string{"job-sequencing", "fan-in", "nashnet-pool", "resumable-upload", "measure-kind"}
	if len(got) != len(want) {
		t.Fatalf("features = %v, want %v", got, want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("features[%d] = %q, want %q (features = %v)", i, got[i], w, got)
		}
	}
}

// TestHealthAuthenticatedCarriesHostCounters pins D30's split: free_ram_gb and
// free_disk_gb describe the coordinator host, not the queue, so the token-free
// LAN listener drops them while a request carrying a valid operator bearer
// token still gets them. An invalid token does NOT unlock the counters: health
// stays reachable on a bad token (never 401), it just stays narrow.
func TestHealthAuthenticatedCarriesHostCounters(t *testing.T) {
	r := newRig(t, rigConfig{ramQuery: func() (float64, error) { return 7.0, nil }})

	free := r.healthBody() // no token
	if _, ok := free["free_ram_gb"]; ok {
		t.Fatalf("token-free health carries free_ram_gb: %v", free)
	}

	bad := r.doTok(http.MethodGet, "/harness/health", nil, "garbage.token.value")
	if bad.StatusCode != http.StatusOK {
		t.Fatalf("bad-token health: got %d, want 200 (health never 401s)", bad.StatusCode)
	}
	var badBody map[string]any
	if err := json.NewDecoder(bad.Body).Decode(&badBody); err != nil {
		t.Fatalf("decode bad-token health body: %v", err)
	}
	bad.Body.Close()
	if _, ok := badBody["free_ram_gb"]; ok {
		t.Fatalf("bad-token health carries free_ram_gb: %v", badBody)
	}

	auth := r.do(http.MethodGet, "/harness/health", nil)
	var authBody map[string]any
	decodeBody(t, auth, &authBody)
	if jnum(authBody["free_ram_gb"]) != 7.0 {
		t.Fatalf("authenticated free_ram_gb = %v, want 7", authBody["free_ram_gb"])
	}
	if _, ok := authBody["free_disk_gb"]; !ok {
		t.Fatalf("authenticated health missing free_disk_gb: %v", authBody)
	}
}

func TestArtifactsManifest(t *testing.T) {
	r := newRig(t, rigConfig{})
	dir := filepath.Join(r.runsDir, "arts")
	if err := os.MkdirAll(filepath.Join(dir, "logs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.yaml"), []byte("device: cpu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "logs", "training.log"), []byte("hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out struct {
		JobID     string     `json:"job_id"`
		Artifacts []Artifact `json:"artifacts"`
	}
	resp := r.do(http.MethodGet, "/harness/jobs/arts/artifacts", nil)
	decodeBody(t, resp, &out)
	if len(out.Artifacts) != 2 {
		t.Fatalf("artifacts = %d, want 2", len(out.Artifacts))
	}
	for _, a := range out.Artifacts {
		if a.SHA256 == "" || a.Size == 0 || a.Path == "" || a.MTime == "" {
			t.Fatalf("incomplete artifact entry: %+v", a)
		}
	}
}

// jnum coerces a decoded JSON number (float64) to float64.
func jnum(v any) float64 {
	f, _ := v.(float64)
	return f
}
