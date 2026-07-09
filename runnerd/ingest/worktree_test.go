package ingest

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// stageWorktree pushes a job ref and adds its worktree, returning the worktree
// path. It is the common setup for worktree-lifecycle tests.
func stageWorktree(t *testing.T, m *Manager, jobID string) (worktreeDir, sha string) {
	t.Helper()
	src, s := sourceRepo(t, "lock-"+jobID)
	pushJobRef(t, m, src, s, jobID)
	wd := m.worktreePath(jobID)
	if err := m.addWorktree(context.Background(), wd, s); err != nil {
		t.Fatalf("addWorktree: %v", err)
	}
	return wd, s
}

func TestAddWorktreeChecksOutSha(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	wd, _ := stageWorktree(t, m, "job-a")

	// The checkout must contain the committed tree.
	if _, err := os.Stat(filepath.Join(wd, "cfr", "uv.lock")); err != nil {
		t.Fatalf("worktree missing cfr/uv.lock: %v", err)
	}
	// Idempotent: a second add over an existing dir is a no-op success.
	if err := m.addWorktree(context.Background(), wd, ""); err != nil {
		t.Fatalf("re-add: %v", err)
	}
}

func TestCleanupSyncedRemovesWorktreeAndRef(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	wd, _ := stageWorktree(t, m, "job-b")

	if err := m.Cleanup("job-b", false); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if _, err := os.Stat(wd); !os.IsNotExist(err) {
		t.Fatalf("worktree not removed: %v", err)
	}
	if _, err := m.git(context.Background(), "rev-parse", "--verify", "--quiet", jobRef("job-b")); err == nil {
		t.Fatal("job ref not deleted")
	}
}

func TestCleanupDebugKeepsWorktreeWithTTL(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	wd, _ := stageWorktree(t, m, "job-c")

	if err := m.Cleanup("job-c", true); err != nil {
		t.Fatalf("Cleanup keepForDebug: %v", err)
	}
	if _, err := os.Stat(wd); err != nil {
		t.Fatalf("debug worktree removed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(wd, debugTTLFile)); err != nil {
		t.Fatalf("debug TTL marker not written: %v", err)
	}
	if !m.debugTTLActive(wd) {
		t.Fatal("debug TTL should be active immediately after mark")
	}
}

func TestStartupSweepPrunesOrphanKeepsLiveAndInTTL(t *testing.T) {
	// Controllable clock so TTL expiry is deterministic.
	base := t.TempDir()
	clock := time.Now()
	m := New(Config{
		BaseDir:  base,
		RunsDir:  filepath.Join(base, "runs"),
		CoresCap: 18,
		Runner:   ExecRunner{},
		DebugTTL: 24 * time.Hour,
		Now:      func() time.Time { return clock },
	})
	if err := m.ensureMirror(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Three worktrees: live, in-TTL failure, and plain orphan.
	liveWD, _ := stageWorktree(t, m, "live")
	inTTLWD, _ := stageWorktree(t, m, "failed")
	orphanWD, _ := stageWorktree(t, m, "orphan")

	// Mark the failed one for debug retention.
	if err := m.markDebugTTL(inTTLWD); err != nil {
		t.Fatal(err)
	}

	if err := m.StartupSweep([]string{"live"}); err != nil {
		t.Fatalf("StartupSweep: %v", err)
	}

	if _, err := os.Stat(liveWD); err != nil {
		t.Fatalf("live worktree pruned: %v", err)
	}
	if _, err := os.Stat(inTTLWD); err != nil {
		t.Fatalf("in-TTL failure worktree pruned: %v", err)
	}
	if _, err := os.Stat(orphanWD); !os.IsNotExist(err) {
		t.Fatalf("orphan worktree not pruned: %v", err)
	}
	// The orphan's ref must be gone; live and failed refs remain.
	if _, err := m.git(context.Background(), "rev-parse", "--verify", "--quiet", jobRef("orphan")); err == nil {
		t.Fatal("orphan ref not deleted")
	}
	if _, err := m.git(context.Background(), "rev-parse", "--verify", "--quiet", jobRef("live")); err != nil {
		t.Fatal("live ref wrongly deleted")
	}

	// Advance past the TTL: the failed worktree now prunes.
	clock = clock.Add(25 * time.Hour)
	if err := m.StartupSweep([]string{"live"}); err != nil {
		t.Fatalf("second StartupSweep: %v", err)
	}
	if _, err := os.Stat(inTTLWD); !os.IsNotExist(err) {
		t.Fatalf("expired-TTL worktree not pruned: %v", err)
	}
}
