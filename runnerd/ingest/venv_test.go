package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVenvCacheKeyStability(t *testing.T) {
	k1 := venvCacheKey("abc", "3.11", "linux_amd64", "cpu")
	k2 := venvCacheKey("abc", "3.11", "linux_amd64", "cpu")
	if k1 != k2 {
		t.Fatalf("key not stable: %q vs %q", k1, k2)
	}
	// Any input change moves the key.
	if venvCacheKey("abd", "3.11", "linux_amd64", "cpu") == k1 {
		t.Fatal("lock-hash change did not move key")
	}
	if venvCacheKey("abc", "3.12", "linux_amd64", "cpu") == k1 {
		t.Fatal("python-minor change did not move key")
	}
	if venvCacheKey("abc", "3.11", "linux_arm64", "cpu") == k1 {
		t.Fatal("platform change did not move key")
	}
	if venvCacheKey("abc", "3.11", "linux_amd64", "gpu") == k1 {
		t.Fatal("device extra change did not move key")
	}
}

// TestVenvCacheKeyCPUFormatUnchanged is a regression test (cambia-329): the
// cpu extra must produce the EXACT pre-device-support key format
// (sha-pyX.Y-platform, no suffix) so venvs already warm on a live runner stay
// valid cache hits after this change ships.
func TestVenvCacheKeyCPUFormatUnchanged(t *testing.T) {
	got := venvCacheKey("abc123", "3.11", "linux_amd64", "cpu")
	want := "abc123-py3.11-linux_amd64"
	if got != want {
		t.Fatalf("cpu key = %q, want %q (byte-identical to pre-device-support format)", got, want)
	}
}

// TestVenvCacheKeyNonCPUExtraSuffix covers the new cambia-329 behavior: a
// non-cpu extra appends "-<extra>" so it never collides with the cpu key
// sharing the same lock/interpreter/platform.
func TestVenvCacheKeyNonCPUExtraSuffix(t *testing.T) {
	cases := map[string]string{
		"gpu": "abc123-py3.11-linux_amd64-gpu",
		"xpu": "abc123-py3.11-linux_amd64-xpu",
	}
	for extra, want := range cases {
		if got := venvCacheKey("abc123", "3.11", "linux_amd64", extra); got != want {
			t.Errorf("extra=%q: key = %q, want %q", extra, got, want)
		}
	}
}

func TestDeviceExtra(t *testing.T) {
	cases := map[string]string{
		"cpu":   "cpu",
		"cuda":  "gpu",
		"xpu":   "xpu",
		"":      "cpu", // empty defaults to cpu, matching JobSpec.device()
		"bogus": "cpu", // unrecognized falls back to cpu
	}
	for device, want := range cases {
		if got := deviceExtra(device); got != want {
			t.Errorf("deviceExtra(%q) = %q, want %q", device, got, want)
		}
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
	v1, err := m.ensureVenv(context.Background(), sha, wd, "cpu")
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
	v2, err := m.ensureVenv(context.Background(), sha, wd, "cpu")
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

func TestEnsureVenvReceiptGuardsPoisonedCache(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	_, sha := sourceRepoInMirror(t, m, "job-vr", "lock-content-R")
	wd := m.worktreePath("job-vr")
	if err := m.addWorktree(context.Background(), wd, sha); err != nil {
		t.Fatal(err)
	}

	// A sync failure must not leave the created venv behind: bin/python exists
	// as soon as `uv venv` runs and would satisfy a bare existence probe.
	fc.syncFails = true
	if _, err := m.ensureVenv(context.Background(), sha, wd, "cpu"); err == nil {
		t.Fatal("expected ensureVenv to fail on uv sync")
	}
	var venvDir string
	for _, c := range fr.callsFor("uv") {
		if len(c.Args) >= 2 && c.Args[0] == "venv" {
			venvDir = c.Args[1]
		}
	}
	if venvDir == "" {
		t.Fatal("no uv venv call recorded")
	}
	if _, err := os.Stat(venvDir); !os.IsNotExist(err) {
		t.Fatalf("failed-sync venv dir still present at %s", venvDir)
	}

	// Recovery: the next ensureVenv rebuilds and writes the receipt.
	fc.syncFails = false
	v, err := m.ensureVenv(context.Background(), sha, wd, "cpu")
	if err != nil {
		t.Fatalf("rebuild after failed sync: %v", err)
	}
	if _, err := os.Stat(filepath.Join(v.dir, venvReceiptName)); err != nil {
		t.Fatalf("receipt missing after successful build: %v", err)
	}

	// A dir with bin/python but no receipt is a miss, never a hit.
	if err := os.Remove(filepath.Join(v.dir, venvReceiptName)); err != nil {
		t.Fatal(err)
	}
	before := len(fr.callsFor("uv"))
	if _, err := m.ensureVenv(context.Background(), sha, wd, "cpu"); err != nil {
		t.Fatalf("ensureVenv after receipt removal: %v", err)
	}
	if got := len(fr.callsFor("uv")); got <= before {
		t.Fatal("receipt-less venv treated as cache hit; expected rebuild")
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

	_, err := m.ensureVenv(context.Background(), sha, wd, "cpu")
	if !errors.Is(err, ErrStaleLock) {
		t.Fatalf("want ErrStaleLock, got %v", err)
	}
	// The venv must NOT be built when the lock check fails.
	if countUV(fr, "venv") != 0 {
		t.Fatalf("venv built despite stale lock (%d builds)", countUV(fr, "venv"))
	}
}

// TestEnsureVenvDeviceExtraSelection covers cambia-329: a cuda job's venv gets
// a distinct cache key from a cpu job on the same lock/interpreter/platform,
// and `uv sync` is invoked with --extra gpu, not cpu.
func TestEnsureVenvDeviceExtraSelection(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	_, sha := sourceRepoInMirror(t, m, "job-vd", "lock-content-D")
	wd := m.worktreePath("job-vd")
	if err := m.addWorktree(context.Background(), wd, sha); err != nil {
		t.Fatal(err)
	}

	cpuVenv, err := m.ensureVenv(context.Background(), sha, wd, "cpu")
	if err != nil {
		t.Fatalf("ensureVenv cpu: %v", err)
	}
	cudaVenv, err := m.ensureVenv(context.Background(), sha, wd, "cuda")
	if err != nil {
		t.Fatalf("ensureVenv cuda: %v", err)
	}
	if cpuVenv.key == cudaVenv.key {
		t.Fatalf("cpu and cuda venvs share a cache key: %q", cpuVenv.key)
	}
	if !strings.HasSuffix(cudaVenv.key, "-gpu") {
		t.Fatalf("cuda venv key %q missing -gpu suffix", cudaVenv.key)
	}
	// Two separate builds: one per device.
	if got := countUV(fr, "venv"); got != 2 {
		t.Fatalf("expected 2 uv venv builds (cpu + cuda), got %d", got)
	}

	sawCPUExtra, sawGPUExtra := false, false
	for _, c := range fr.callsFor("uv") {
		if len(c.Args) == 0 || c.Args[0] != "sync" {
			continue
		}
		for i, a := range c.Args {
			if a != "--extra" || i+1 >= len(c.Args) {
				continue
			}
			switch c.Args[i+1] {
			case "cpu":
				sawCPUExtra = true
			case "gpu":
				sawGPUExtra = true
			}
		}
	}
	if !sawCPUExtra {
		t.Fatal("no uv sync --extra cpu recorded")
	}
	if !sawGPUExtra {
		t.Fatal("no uv sync --extra gpu recorded")
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
