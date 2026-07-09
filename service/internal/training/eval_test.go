package training

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// evalArgsEchoStub echoes its argument list to the log then exits 0. Used to
// assert the spawned `cambia evaluate ...` argument vector (device/games/target).
const evalArgsEchoStub = "#!/bin/sh\necho \"$@\"\nexit 0\n"

// evalFailStub exits nonzero, simulating an eval subprocess failure.
const evalFailStub = "#!/bin/sh\necho boom >&2\nexit 3\n"

// newEvalManager builds an EvalManager whose cambiaBin is a stub script with the
// given body.
func newEvalManager(t *testing.T, stubBody string) (*EvalManager, string) {
	t.Helper()
	base := t.TempDir()
	runsDir := filepath.Join(base, "runs")
	cfrDir := filepath.Join(base, "cfr")
	for _, d := range []string{runsDir, cfrDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	stub := filepath.Join(base, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(stubBody), 0o755); err != nil {
		t.Fatal(err)
	}
	return NewEvalManager(runsDir, cfrDir, stub), runsDir
}

// seedCheckpoint drops a fake *.pt under runs/<name>/<sub>/ so
// hasEvaluableCheckpoint passes.
func seedCheckpoint(t *testing.T, runsDir, name, sub, file string) {
	t.Helper()
	dir := filepath.Join(runsDir, name, sub)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// waitEvalStatus polls the run's newest job until it reaches want or the
// timeout.
func waitEvalStatus(t *testing.T, m *EvalManager, name, want string, timeout time.Duration) *EvalJob {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		jobs := m.Jobs(name)
		if len(jobs) > 0 && jobs[0].Status == want {
			return jobs[0]
		}
		time.Sleep(15 * time.Millisecond)
	}
	jobs := m.Jobs(name)
	t.Fatalf("run %q newest job did not reach %q within %s (jobs=%+v)", name, want, timeout, jobs)
	return nil
}

func TestEvalTriggerSucceeded(t *testing.T) {
	m, runsDir := newEvalManager(t, evalArgsEchoStub)
	seedCheckpoint(t, runsDir, "run1", "snapshots", "prtcfr_checkpoint.pt")

	games := 42
	job, err := m.Trigger("run1", EvalOpts{Device: "cpu", Games: games, Argmax: true})
	if err != nil {
		t.Fatalf("Trigger: %v", err)
	}
	if job.Status != EvalRunning {
		t.Errorf("status = %q, want running at return", job.Status)
	}
	if job.Target != "latest" {
		t.Errorf("target = %q, want latest", job.Target)
	}
	if job.StartedAt == "" {
		t.Error("started_at not set")
	}

	final := waitEvalStatus(t, m, "run1", EvalSucceeded, 10*time.Second)
	if final.ExitCode == nil || *final.ExitCode != 0 {
		t.Errorf("exit_code = %v, want 0", final.ExitCode)
	}
	if final.FinishedAt == "" {
		t.Error("finished_at not set on success")
	}
	// AC1: the spawned command carried the requested device/games/target.
	tail := strings.Join(final.Tail, "\n")
	for _, want := range []string{"evaluate", "--device cpu", "--games 42", "--latest", "--argmax"} {
		if !strings.Contains(tail, want) {
			t.Errorf("log tail missing %q; tail=%q", want, tail)
		}
	}
}

func TestEvalTriggerEpochTarget(t *testing.T) {
	m, runsDir := newEvalManager(t, evalArgsEchoStub)
	seedCheckpoint(t, runsDir, "run-epoch", "checkpoints", "prtcfr_checkpoint_iter_5.pt")

	epoch := 5
	job, err := m.Trigger("run-epoch", EvalOpts{Epoch: &epoch, Device: "cpu"})
	if err != nil {
		t.Fatalf("Trigger: %v", err)
	}
	if job.Target != "iter:5" {
		t.Errorf("target = %q, want iter:5", job.Target)
	}
	if job.Games != defaultEvalGames {
		t.Errorf("games = %d, want default %d (0 request falls back)", job.Games, defaultEvalGames)
	}

	final := waitEvalStatus(t, m, "run-epoch", EvalSucceeded, 10*time.Second)
	tail := strings.Join(final.Tail, "\n")
	if !strings.Contains(tail, "--epoch 5") || strings.Contains(tail, "--latest") {
		t.Errorf("epoch spawn args wrong; tail=%q", tail)
	}
}

func TestEvalTriggerFailed(t *testing.T) {
	m, runsDir := newEvalManager(t, evalFailStub)
	seedCheckpoint(t, runsDir, "run-fail", "snapshots", "prtcfr_checkpoint.pt")

	if _, err := m.Trigger("run-fail", EvalOpts{Device: "cpu"}); err != nil {
		t.Fatalf("Trigger: %v", err)
	}
	final := waitEvalStatus(t, m, "run-fail", EvalFailed, 10*time.Second)
	if final.ExitCode == nil || *final.ExitCode != 3 {
		t.Errorf("exit_code = %v, want 3", final.ExitCode)
	}
	if final.Error == "" {
		t.Error("error not set on failure")
	}
}

func TestEvalTriggerNoCheckpoint(t *testing.T) {
	m, runsDir := newEvalManager(t, evalArgsEchoStub)
	// Run dir exists but holds no *.pt.
	if err := os.MkdirAll(filepath.Join(runsDir, "empty-run", "snapshots"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Trigger("empty-run", EvalOpts{Device: "cpu"}); !errors.Is(err, ErrNoCheckpoint) {
		t.Errorf("Trigger: err = %v, want ErrNoCheckpoint", err)
	}
	// A missing run dir also 404-sources.
	if _, err := m.Trigger("ghost", EvalOpts{Device: "cpu"}); !errors.Is(err, ErrNoCheckpoint) {
		t.Errorf("Trigger(ghost): err = %v, want ErrNoCheckpoint", err)
	}
}

func TestEvalTriggerInvalidName(t *testing.T) {
	m, _ := newEvalManager(t, evalArgsEchoStub)
	for _, name := range []string{"../evil", "a/b", "..", "e..vil"} {
		if _, err := m.Trigger(name, EvalOpts{Device: "cpu"}); !errors.Is(err, procmgr.ErrInvalidName) {
			t.Errorf("Trigger(%q): err = %v, want ErrInvalidName", name, err)
		}
	}
}

// TestEvalCapReached asserts the manager-internal cap: with cap 1 and a
// still-running eval, a second trigger returns ErrEvalCapReached. The first
// stub blocks on a release file so its liveness is deterministic (no sleep race).
func TestEvalCapReached(t *testing.T) {
	base := t.TempDir()
	runsDir := filepath.Join(base, "runs")
	cfrDir := filepath.Join(base, "cfr")
	for _, d := range []string{runsDir, cfrDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	release := filepath.Join(base, "release")
	stubBody := "#!/bin/sh\nwhile [ ! -f '" + release + "' ]; do sleep 0.02; done\nexit 0\n"
	stub := filepath.Join(base, "cambia_stub.sh")
	if err := os.WriteFile(stub, []byte(stubBody), 0o755); err != nil {
		t.Fatal(err)
	}
	m := NewEvalManager(runsDir, cfrDir, stub)
	m.SetMaxConcurrent(1)
	seedCheckpoint(t, runsDir, "a", "snapshots", "prtcfr_checkpoint.pt")
	seedCheckpoint(t, runsDir, "b", "snapshots", "prtcfr_checkpoint.pt")
	t.Cleanup(func() { _ = os.WriteFile(release, []byte("x"), 0o644) })

	if _, err := m.Trigger("a", EvalOpts{Device: "cpu"}); err != nil {
		t.Fatalf("first Trigger: %v", err)
	}
	// Second trigger (different run) must hit the cap while the first blocks.
	if _, err := m.Trigger("b", EvalOpts{Device: "cpu"}); !errors.Is(err, ErrEvalCapReached) {
		t.Fatalf("second Trigger: err = %v, want ErrEvalCapReached", err)
	}

	// Release the first and confirm the slot frees (a later trigger succeeds).
	if err := os.WriteFile(release, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	waitEvalStatus(t, m, "a", EvalSucceeded, 10*time.Second)
	if _, err := m.Trigger("b", EvalOpts{Device: "cpu"}); err != nil {
		t.Fatalf("Trigger after slot freed: %v", err)
	}
	waitEvalStatus(t, m, "b", EvalSucceeded, 10*time.Second)
}

// TestEvalRetention asserts the 20-job retention cap evicts the oldest jobs and
// keeps the newest, newest-first. The cap is disabled so 25 quick evals all
// register.
func TestEvalRetention(t *testing.T) {
	m, runsDir := newEvalManager(t, evalArgsEchoStub)
	m.SetMaxConcurrent(0) // disable so all 25 spawn
	seedCheckpoint(t, runsDir, "retain", "snapshots", "prtcfr_checkpoint.pt")

	const n = 25
	ids := make([]string, 0, n)
	for i := 0; i < n; i++ {
		job, err := m.Trigger("retain", EvalOpts{Device: "cpu"})
		if err != nil {
			t.Fatalf("Trigger %d: %v", i, err)
		}
		ids = append(ids, job.ID)
	}

	jobs := m.Jobs("retain")
	if len(jobs) != maxEvalJobsPerRun {
		t.Fatalf("retained = %d, want %d", len(jobs), maxEvalJobsPerRun)
	}
	// Newest-first: the first listed is the last triggered.
	if jobs[0].ID != ids[n-1] {
		t.Errorf("jobs[0].ID = %q, want newest %q", jobs[0].ID, ids[n-1])
	}
	present := make(map[string]bool, len(jobs))
	for _, j := range jobs {
		present[j.ID] = true
	}
	// The 5 oldest are evicted; the 20 newest remain.
	for _, id := range ids[:n-maxEvalJobsPerRun] {
		if present[id] {
			t.Errorf("evicted id %q still present", id)
		}
	}
	for _, id := range ids[n-maxEvalJobsPerRun:] {
		if !present[id] {
			t.Errorf("retained id %q missing", id)
		}
	}

	// Drain so TempDir cleanup does not race a still-writing child.
	waitEvalDrained(t, m, "retain", 10*time.Second)
}

// waitEvalDrained waits until no job for name is still running.
func waitEvalDrained(t *testing.T, m *EvalManager, name string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		running := false
		for _, j := range m.Jobs(name) {
			if j.Status == EvalRunning {
				running = true
				break
			}
		}
		if !running {
			return
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("evals for %q did not drain within %s", name, timeout)
}

func TestEvalJobsEmpty(t *testing.T) {
	m, _ := newEvalManager(t, evalArgsEchoStub)
	if jobs := m.Jobs("nobody"); len(jobs) != 0 {
		t.Errorf("Jobs(unknown) = %+v, want empty", jobs)
	}
}
