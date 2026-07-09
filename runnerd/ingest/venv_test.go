package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVenvCacheKeyStability(t *testing.T) {
	k1 := venvCacheKey("abc", "3.11", "linux_amd64")
	k2 := venvCacheKey("abc", "3.11", "linux_amd64")
	if k1 != k2 {
		t.Fatalf("key not stable: %q vs %q", k1, k2)
	}
	// Any input change moves the key.
	if venvCacheKey("abd", "3.11", "linux_amd64") == k1 {
		t.Fatal("lock-hash change did not move key")
	}
	if venvCacheKey("abc", "3.12", "linux_amd64") == k1 {
		t.Fatal("python-minor change did not move key")
	}
	if venvCacheKey("abc", "3.11", "linux_arm64") == k1 {
		t.Fatal("platform change did not move key")
	}
}

// fakeManager builds a Manager whose toolchain commands are faked and whose git
// runs for real. It returns the manager and the fake-call recorder.
func fakeManager(t *testing.T, fc *fakeControl) (*Manager, *fakeRunner) {
	t.Helper()
	fr := newFakeRunner()
	fr.hook = fc.hook()
	m, _ := testManager(t, fr)
	return m, fr
}

func TestEnsureVenvHitMiss(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	_, sha := sourceRepoInMirror(t, m, "job-v", "lock-content-A")
	wd := m.worktreePath("job-v")
	if err := m.addWorktree(context.Background(), wd, sha); err != nil {
		t.Fatal(err)
	}

	// Miss: builds.
	v1, err := m.ensureVenv(context.Background(), sha, wd)
	if err != nil {
		t.Fatalf("ensureVenv miss: %v", err)
	}
	venvBuilds := countUV(fr, "venv")
	if venvBuilds != 1 {
		t.Fatalf("expected 1 uv venv build, got %d", venvBuilds)
	}
	if _, err := os.Stat(v1.python); err != nil {
		t.Fatalf("venv python not present: %v", err)
	}

	// Hit: no new build.
	v2, err := m.ensureVenv(context.Background(), sha, wd)
	if err != nil {
		t.Fatalf("ensureVenv hit: %v", err)
	}
	if v2.key != v1.key {
		t.Fatalf("key changed on hit: %q vs %q", v1.key, v2.key)
	}
	if countUV(fr, "venv") != 1 {
		t.Fatalf("cache hit rebuilt the venv (%d builds)", countUV(fr, "venv"))
	}
	// lockSha256 must be the sha256 of the actual lock content.
	wantSha := sha256hex([]byte("lock-content-A"))
	if v1.lockSha256 != wantSha {
		t.Fatalf("lockSha256 = %q, want %q", v1.lockSha256, wantSha)
	}
}

func TestEnsureVenvStaleLockRefused(t *testing.T) {
	fc := newFakeControl()
	fc.lockCheckFails = true
	m, fr := fakeManager(t, fc)
	_, sha := sourceRepoInMirror(t, m, "job-stale", "lock-content")
	wd := m.worktreePath("job-stale")
	if err := m.addWorktree(context.Background(), wd, sha); err != nil {
		t.Fatal(err)
	}

	_, err := m.ensureVenv(context.Background(), sha, wd)
	if !errors.Is(err, ErrStaleLock) {
		t.Fatalf("want ErrStaleLock, got %v", err)
	}
	// The venv must NOT be built when the lock check fails.
	if countUV(fr, "venv") != 0 {
		t.Fatalf("venv built despite stale lock (%d builds)", countUV(fr, "venv"))
	}
}

func TestEvictVenvsLRUKeepsProtected(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	m.cfg.MaxVenvs = 2

	// Create 4 venv dirs with increasing mtimes; oldest two are evictable.
	names := []string{"k-old1", "k-old2", "k-new1", "k-new2"}
	base := time.Now().Add(-time.Hour)
	for i, n := range names {
		d := filepath.Join(m.venvsDir, n)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		ts := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(d, ts, ts); err != nil {
			t.Fatal(err)
		}
	}

	// Protect the oldest so it survives despite LRU order.
	m.evictVenvs(map[string]bool{"k-old1": true})

	remaining := map[string]bool{}
	ents, _ := os.ReadDir(m.venvsDir)
	for _, e := range ents {
		remaining[e.Name()] = true
	}
	if !remaining["k-old1"] {
		t.Fatal("protected venv was evicted")
	}
	if !remaining["k-new2"] {
		t.Fatal("newest venv was evicted")
	}
	if len(remaining) > 3 { // 2 cap + 1 protected can exceed cap
		t.Fatalf("too many venvs remain: %v", remaining)
	}
	if remaining["k-old2"] && remaining["k-new1"] {
		t.Fatalf("expected an eviction beyond protected+cap, got %v", remaining)
	}
}

// sourceRepoInMirror creates a source repo with the given lock content, pushes it
// as jobID's ref into the manager's mirror, and returns the src dir and sha.
func sourceRepoInMirror(t *testing.T, m *Manager, jobID, lockContent string) (string, string) {
	t.Helper()
	src, sha := sourceRepo(t, lockContent)
	pushJobRef(t, m, src, sha, jobID)
	return src, sha
}

// countUV counts recorded uv invocations whose first arg equals sub.
func countUV(fr *fakeRunner, sub string) int {
	n := 0
	for _, c := range fr.callsFor("uv") {
		if len(c.Args) > 0 && c.Args[0] == sub {
			n++
		}
	}
	return n
}
