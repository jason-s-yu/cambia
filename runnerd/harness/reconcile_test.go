package harness

import (
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

func TestReconcileOrphanAndCrashSweep(t *testing.T) {
	runsDir := t.TempDir()
	// created but never launched (daemon died mid-prepare) -> orphaned.
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
	if !strings.Contains(orphan.LastError, "orphaned by daemon restart") {
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
	// A created row with a config present would be launchable, but reconcile must
	// never launch: it reports (here, sweeps to failed since the queue is empty).
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
