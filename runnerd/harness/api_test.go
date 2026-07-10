package harness

import (
	"encoding/json"
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
// monitoring consumer; cambia-330/network-552). Everything else stays gated —
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
	// field is exposed unauthenticated — a regression to writeJSON(w, 200, snap)
	// would leak queue/active JobViews (job IDs, commits, configs) while a
	// presence-only check stayed green.
	want := []string{"reconciled_at", "jobs_running", "queue_depth", "free_ram_gb", "free_disk_gb"}
	for _, k := range want {
		if _, ok := body[k]; !ok {
			t.Fatalf("health body missing key %q: %v", k, body)
		}
	}
	if len(body) != len(want) {
		t.Fatalf("token-free health body must carry exactly %d counter keys, got %d: %v", len(want), len(body), body)
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
