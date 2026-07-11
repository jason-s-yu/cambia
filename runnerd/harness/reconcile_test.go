package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

func writeState(t *testing.T, runsDir, name, status string, pid int) {
	t.Helper()
	st := &procmgr.ProcessState{Name: name, Status: status, PID: pid}
	if err := procmgr.WriteProcessState(filepath.Join(runsDir, name), st); err != nil {
		t.Fatal(err)
	}
}

func readStatus(t *testing.T, runsDir, name string) *procmgr.ProcessState {
	t.Helper()
	st, err := procmgr.ReadProcessState(filepath.Join(runsDir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return st
}

// writeCreatedJob writes a created-but-never-launched job to disk: process.json
// (Status=created, PID 0) plus a jobspec.json carrying submit_seq/after/
// on_failure, exactly as Submit persists before enqueuing (cambia-352).
func writeCreatedJob(t *testing.T, runsDir, name, kind string, seq int64, after, onFailure string) {
	t.Helper()
	st := &procmgr.ProcessState{Name: name, Status: procmgr.StatusCreated, Algorithm: kind}
	if err := procmgr.WriteProcessState(filepath.Join(runsDir, name), st); err != nil {
		t.Fatal(err)
	}
	spec := &JobSpec{Name: name, Kind: kind, Config: "cfr/config/x.yaml", SubmitSeq: seq, After: after, OnFailure: onFailure}
	if err := writeJobSpec(filepath.Join(runsDir, name), spec); err != nil {
		t.Fatal(err)
	}
}

func TestReconcileOrphanAndCrashSweep(t *testing.T) {
	runsDir := t.TempDir()
	// created with no readable jobspec (admission aborted mid-write) -> failed.
	writeState(t, runsDir, "orphan", procmgr.StatusCreated, 0)
	// running with a dead pid -> crashed (procmgr sweep).
	writeState(t, runsDir, "dead", procmgr.StatusRunning, 999999)
	// cleanly stopped -> untouched.
	writeState(t, runsDir, "done", procmgr.StatusStopped, 0)

	fe := &fakeEnv{runsDir: runsDir}
	pm := procmgr.NewProcessManager(runsDir, t.TempDir(), "cambia", NewRunResolver(runsDir), fakeAlgos())
	disp := NewDispatcher(pm, fe, runsDir, 1, 16, 15*time.Millisecond)

	disp.Reconcile()

	orphan := readStatus(t, runsDir, "orphan")
	if orphan.Status != StateFailed {
		t.Fatalf("orphan status = %q, want failed", orphan.Status)
	}
	if !strings.Contains(orphan.LastError, "incomplete admission") {
		t.Fatalf("orphan last_error = %q", orphan.LastError)
	}
	if dead := readStatus(t, runsDir, "dead"); dead.Status != procmgr.StatusCrashed {
		t.Fatalf("dead status = %q, want crashed", dead.Status)
	}
	if done := readStatus(t, runsDir, "done"); done.Status != procmgr.StatusStopped {
		t.Fatalf("done status = %q, want stopped (untouched)", done.Status)
	}

	fe.mu.Lock()
	swept := len(fe.sweeps)
	fe.mu.Unlock()
	if swept == 0 {
		t.Fatal("ingest StartupSweep was not called during Reconcile")
	}

	if disp.ReconciledAt() == "" {
		t.Fatal("reconciled_at not stamped")
	}
}

func TestReconcileNeverAutoLaunches(t *testing.T) {
	runsDir := t.TempDir()
	// A created row with NO spec is an incomplete admission: reconcile fails it
	// rather than launching. (A created row WITH a spec is re-enqueued, not
	// launched by reconcile itself; see TestReconcileReenqueuesCreatedInSeqOrder.)
	writeState(t, runsDir, "ready", procmgr.StatusCreated, 0)

	fe := &fakeEnv{runsDir: runsDir}
	pm := procmgr.NewProcessManager(runsDir, t.TempDir(), "cambia", NewRunResolver(runsDir), fakeAlgos())
	disp := NewDispatcher(pm, fe, runsDir, 1, 16, 15*time.Millisecond)
	disp.Reconcile()

	if s := readStatus(t, runsDir, "ready").Status; s != StateFailed {
		t.Fatalf("reconcile launched or left created: status = %q, want failed", s)
	}
	fe.mu.Lock()
	prepared := len(fe.cleanups) // Prepare is never recorded; Cleanup only on markTerminal path (no launch)
	fe.mu.Unlock()
	_ = prepared
}

// TestReconcileReenqueuesCreatedInSeqOrder proves queue persistence (cambia-352):
// created rows with a readable spec survive a restart and are re-queued in
// submit_seq order, not disk-scan (alphabetical) order. bbb (seq 1) is named to
// sort AFTER aaa (seq 2) alphabetically, so a seq-ordered rebuild launches bbb
// first (the single slot) and leaves aaa queued.
func TestReconcileReenqueuesCreatedInSeqOrder(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 1})
	writeCreatedJob(t, r.runsDir, "aaa", "fake", 2, "", "")
	writeCreatedJob(t, r.runsDir, "bbb", "fake", 1, "", "")

	r.disp.Reconcile()

	// bbb (seq 1) dispatches into the single slot; aaa (seq 2) waits behind it.
	r.waitForState("bbb", procmgr.StatusRunning, 3*time.Second)
	if s, _ := r.getState("aaa"); s != StateQueued {
		t.Fatalf("aaa state = %q, want queued (bbb has the slot)", s)
	}
	// Seed check: a post-restart submit gets a seq past the persisted max (2).
	r.disp.mu.Lock()
	seq := r.disp.nextSeq
	r.disp.mu.Unlock()
	if seq <= 2 {
		t.Fatalf("nextSeq = %d, want > 2 (seeded past persisted max)", seq)
	}
}

// TestReconcileDependentStillWaitsAfterRestart proves a re-enqueued dependent
// re-blocks on its parent's live state across a restart: the parent (created,
// re-enqueued) launches and runs, so the dependent stays queued.
func TestReconcileDependentStillWaitsAfterRestart(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	writeCreatedJob(t, r.runsDir, "parent", "fake", 1, "", "")
	writeCreatedJob(t, r.runsDir, "child", "fake", 2, "parent", "skip")

	r.disp.Reconcile()

	r.waitForState("parent", procmgr.StatusRunning, 3*time.Second)
	// Parent is running (non-terminal), so the child gate blocks; it stays queued.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if s, _ := r.getState("child"); s != StateQueued {
			t.Fatalf("child state = %q, want queued while parent runs", s)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestReconcileCorruptSpecFailsOnlyThatJob proves per-file isolation: a created
// row with a corrupt spec is failed while a sibling created row with a valid
// spec is re-enqueued unharmed.
func TestReconcileCorruptSpecFailsOnlyThatJob(t *testing.T) {
	r := newRig(t, rigConfig{maxJobs: 2})
	writeCreatedJob(t, r.runsDir, "good", "fake", 1, "", "")
	// corrupt: created process.json but garbage jobspec.json.
	writeState(t, r.runsDir, "corrupt", procmgr.StatusCreated, 0)
	if err := os.WriteFile(filepath.Join(r.runsDir, "corrupt", "jobspec.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	// missing: created process.json, no jobspec.json at all.
	writeState(t, r.runsDir, "missing", procmgr.StatusCreated, 0)

	r.disp.Reconcile()

	if s := readStatus(t, r.runsDir, "corrupt").Status; s != StateFailed {
		t.Fatalf("corrupt status = %q, want failed", s)
	}
	if s := readStatus(t, r.runsDir, "missing").Status; s != StateFailed {
		t.Fatalf("missing status = %q, want failed", s)
	}
	// The valid sibling was re-enqueued and launches (never failed).
	r.waitForState("good", procmgr.StatusRunning, 3*time.Second)
}

// TestReconcileStartingRowCrashes proves the idempotency prerequisite's other
// half: a row left at `starting` (forked, but the running write never landed
// before the daemon died) reconciles to crashed via pid liveness, never
// re-enqueued (only `created` rows are).
func TestReconcileStartingRowCrashes(t *testing.T) {
	runsDir := t.TempDir()
	writeState(t, runsDir, "half-launched", procmgr.StatusStarting, 999999) // dead pid

	fe := &fakeEnv{runsDir: runsDir}
	pm := procmgr.NewProcessManager(runsDir, t.TempDir(), "cambia", NewRunResolver(runsDir), fakeAlgos())
	disp := NewDispatcher(pm, fe, runsDir, 1, 16, 15*time.Millisecond)
	disp.Reconcile()

	if s := readStatus(t, runsDir, "half-launched").Status; s != procmgr.StatusCrashed {
		t.Fatalf("starting row status = %q, want crashed", s)
	}
}
