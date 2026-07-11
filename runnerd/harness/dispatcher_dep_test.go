package harness

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// depSpec is a minimal cpu spec carrying an after/on_failure dependency
// (cambia-352). kind selects the fake script behavior (fake=sleep, fake-quick=
// exit 0, fake-fail=exit 3).
func depSpec(name, kind, after, onFailure string) JobSpec {
	return JobSpec{
		Kind:      kind,
		Commit:    strings.Repeat("a", 40),
		Name:      name,
		Config:    "cfr/config/prtcfr_prod.yaml",
		Device:    "cpu",
		After:     after,
		OnFailure: onFailure,
	}
}

// writeTerminalParent writes a parent job to disk as a terminal record: a
// process.json with the given status/exit code plus a jobspec, so a dependent's
// gate can read its outcome without the parent ever being a live dispatcher job.
func writeTerminalParent(t *testing.T, runsDir, name, status string, exit int) {
	t.Helper()
	code := exit
	st := &procmgr.ProcessState{Name: name, Status: status, Algorithm: "fake", ExitCode: &code, FinishedAt: procmgr.NowRFC3339()}
	if err := procmgr.WriteProcessState(filepath.Join(runsDir, name), st); err != nil {
		t.Fatal(err)
	}
	if err := writeJobSpec(filepath.Join(runsDir, name), &JobSpec{Name: name, Kind: "fake"}); err != nil {
		t.Fatal(err)
	}
}

// setupResumable stages a stopped job that satisfies the resume gate (rolling
// checkpoint + resume_state.json + config + jobspec). after/onFailure populate
// the persisted spec so a resumed job's own dependency can be exercised.
func setupResumable(t *testing.T, runsDir, name, kind, after, onFailure string) {
	t.Helper()
	dir := filepath.Join(runsDir, name)
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
	code := 0
	st := &procmgr.ProcessState{Name: name, Status: procmgr.StatusStopped, Algorithm: kind, ExitCode: &code, FinishedAt: procmgr.NowRFC3339()}
	if err := procmgr.WriteProcessState(dir, st); err != nil {
		t.Fatal(err)
	}
	if err := writeJobSpec(dir, &JobSpec{Name: name, Kind: kind, After: after, OnFailure: onFailure}); err != nil {
		t.Fatal(err)
	}
}

// TestDependencyParentSuccessLaunches: a parent that exits 0 (stopped, success)
// always runs its dependent.
func TestDependencyParentSuccessLaunches(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	if _, err := r.disp.Submit(depSpec("dep-P", "fake-quick", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("dep-P", procmgr.StatusStopped, 3*time.Second)

	if _, err := r.disp.Submit(depSpec("dep-C", "fake", "dep-P", "skip")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("dep-C", procmgr.StatusRunning, 3*time.Second)
}

// TestDependencyOnFailureSkip: a crashed parent skips a skip-policy dependent.
func TestDependencyOnFailureSkip(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	if _, err := r.disp.Submit(depSpec("skp-P", "fake-fail", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("skp-P", procmgr.StatusCrashed, 3*time.Second)

	if _, err := r.disp.Submit(depSpec("skp-C", "fake", "skp-P", "skip")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("skp-C", StateSkipped, 3*time.Second)
}

// TestDependencyOnFailureFail: a crashed parent fails a fail-policy dependent.
func TestDependencyOnFailureFail(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	if _, err := r.disp.Submit(depSpec("fl-P", "fake-fail", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("fl-P", procmgr.StatusCrashed, 3*time.Second)

	if _, err := r.disp.Submit(depSpec("fl-C", "fake", "fl-P", "fail")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("fl-C", StateFailed, 3*time.Second)
}

// TestDependencyOnFailureRun: a crashed parent still launches a run-policy
// dependent.
func TestDependencyOnFailureRun(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	if _, err := r.disp.Submit(depSpec("rn-P", "fake-fail", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("rn-P", procmgr.StatusCrashed, 3*time.Second)

	if _, err := r.disp.Submit(depSpec("rn-C", "fake", "rn-P", "run")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("rn-C", procmgr.StatusRunning, 3*time.Second)
}

// TestPurgeDependentGuardAndCascade: purging a terminal parent that still has a
// queued dependent is refused (409 has_dependents) unless cascade is set, which
// skips the dependent first.
func TestPurgeDependentGuardAndCascade(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	// Blocker occupies the single slot so the dependent cannot launch.
	if _, err := r.disp.Submit(depSpec("pg-B", "fake", "", "")); err != nil {
		t.Fatal(err)
	}
	r.waitForState("pg-B", procmgr.StatusRunning, 3*time.Second)

	// Parent is a terminal (stopped exit0) disk record; dependent waits (no slot).
	writeTerminalParent(t, r.runsDir, "pg-P", procmgr.StatusStopped, 0)
	if _, err := r.disp.Submit(depSpec("pg-C", "fake", "pg-P", "skip")); err != nil {
		t.Fatal(err)
	}
	if s, _ := r.getState("pg-C"); s != StateQueued {
		t.Fatalf("pg-C state = %q, want queued", s)
	}

	// Purge without cascade -> 409 has_dependents; the run dir survives.
	resp := r.do(http.MethodDelete, "/harness/jobs/pg-P?purge=true", nil)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("purge with dependents: got %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()
	if _, err := os.Stat(filepath.Join(r.runsDir, "pg-P")); err != nil {
		t.Fatalf("refused purge should keep parent dir: %v", err)
	}

	// Purge with cascade -> 200, dependent skipped, parent dir gone.
	resp = r.do(http.MethodDelete, "/harness/jobs/pg-P?purge=true&cascade=true", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("cascade purge: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("pg-C", StateSkipped, 2*time.Second)
	if _, err := os.Stat(filepath.Join(r.runsDir, "pg-P")); !os.IsNotExist(err) {
		t.Fatalf("cascade purge did not remove parent dir: %v", err)
	}
}

// TestGateReEvaluatesParentAcrossResume (white-box) proves the gate reads the
// parent's CURRENT state on every dispatch: a dependent launches when the parent
// is a terminal success, re-blocks when the parent is resumed (non-terminal
// again), and launches again once the parent re-completes.
func TestGateReEvaluatesParentAcrossResume(t *testing.T) {
	runsDir := t.TempDir()
	fe := &fakeEnv{runsDir: runsDir}
	pm := procmgr.NewProcessManager(runsDir, t.TempDir(), "cambia", NewRunResolver(runsDir), fakeAlgos())
	disp := NewDispatcher(pm, fe, runsDir, 2, 16, 15*time.Millisecond)

	writeTerminalParent(t, runsDir, "P", procmgr.StatusStopped, 0)
	child := &job{spec: JobSpec{Name: "C", After: "P", OnFailure: "run"}, state: StateQueued}

	disp.mu.Lock()
	d1 := disp.gateDecisionLocked(child)
	disp.mu.Unlock()
	if d1 != gateLaunch {
		t.Fatalf("parent stopped-exit0: decision = %d, want gateLaunch", d1)
	}

	// Parent resumed -> now a live pending (queued) job, i.e. non-terminal again.
	disp.mu.Lock()
	disp.pending["P"] = &job{spec: JobSpec{Name: "P"}, state: StateQueued}
	d2 := disp.gateDecisionLocked(child)
	disp.mu.Unlock()
	if d2 != gateBlocked {
		t.Fatalf("parent resumed (non-terminal): decision = %d, want gateBlocked", d2)
	}

	// Parent finishes again (out of pending, terminal on disk) -> re-arm.
	disp.mu.Lock()
	delete(disp.pending, "P")
	d3 := disp.gateDecisionLocked(child)
	disp.mu.Unlock()
	if d3 != gateLaunch {
		t.Fatalf("parent re-completed: decision = %d, want gateLaunch", d3)
	}
}

// TestReArmOnParentResume (behavioral): a resumed (running) parent blocks its
// dependent, and the dependent re-arms when the parent reaches a terminal.
func TestReArmOnParentResume(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	setupResumable(t, r.runsDir, "rr-P", "fake", "", "")

	if _, err := r.disp.Resume("rr-P"); err != nil {
		t.Fatalf("resume parent: %v", err)
	}
	r.waitForState("rr-P", procmgr.StatusRunning, 3*time.Second)

	// Dependent waits on the resumed, now-running parent.
	if _, err := r.disp.Submit(depSpec("rr-C", "fake", "rr-P", "run")); err != nil {
		t.Fatal(err)
	}
	if s, _ := r.getState("rr-C"); s != StateQueued {
		t.Fatalf("rr-C state = %q, want queued while resumed parent runs", s)
	}

	// Parent terminates -> re-arm fires -> dependent launches (on_failure=run).
	resp := r.do(http.MethodDelete, "/harness/jobs/rr-P", nil)
	resp.Body.Close()
	r.waitForState("rr-C", procmgr.StatusRunning, 5*time.Second)
}

// TestDependentResumeIgnoresAfter: resuming a dependent relaunches it regardless
// of its own after, even when that parent failed (which would otherwise skip a
// fresh submit).
func TestDependentResumeIgnoresAfter(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	writeTerminalParent(t, r.runsDir, "di-P", procmgr.StatusCrashed, 3)
	setupResumable(t, r.runsDir, "di-C", "fake", "di-P", "skip")

	if _, err := r.disp.Resume("di-C"); err != nil {
		t.Fatalf("resume dependent: %v", err)
	}
	// Launches despite di-P being crashed: a resumed dependent ignores its after.
	r.waitForState("di-C", procmgr.StatusRunning, 3*time.Second)
}
