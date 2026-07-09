package ingest

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEnsureLibcambiaKeyedByEngineTree(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)

	// Commit 1.
	src, sha1 := sourceRepo(t, "lock-1")
	pushJobRef(t, m, src, sha1, "job-1")
	wd1 := m.worktreePath("job-1")
	if err := m.addWorktree(context.Background(), wd1, sha1); err != nil {
		t.Fatal(err)
	}
	lib1, err := m.ensureLibcambia(context.Background(), sha1, wd1)
	if err != nil {
		t.Fatalf("ensureLibcambia c1: %v", err)
	}
	if filepath.Base(lib1.path) != lib1.engineTreeSha+".so" {
		t.Fatalf("cache path %q not keyed by engine tree sha %q", lib1.path, lib1.engineTreeSha)
	}
	wantEngineSha := runGit(t, src, "rev-parse", sha1+":engine")
	if lib1.engineTreeSha != wantEngineSha {
		t.Fatalf("engineTreeSha = %q, want %q", lib1.engineTreeSha, wantEngineSha)
	}
	if lib1.sha256 == "" {
		t.Fatal("libcambia sha256 not computed")
	}

	// Commit 2 changes only cfr/, leaving the engine tree identical.
	mustWrite(t, filepath.Join(src, "cfr", "uv.lock"), "lock-2")
	runGit(t, src, "add", "-A")
	runGit(t, src, "commit", "-q", "-m", "cfr-only")
	sha2 := runGit(t, src, "rev-parse", "HEAD")
	pushJobRef(t, m, src, sha2, "job-2")
	wd2 := m.worktreePath("job-2")
	if err := m.addWorktree(context.Background(), wd2, sha2); err != nil {
		t.Fatal(err)
	}

	buildsBefore := countGoBuilds(fr)
	lib2, err := m.ensureLibcambia(context.Background(), sha2, wd2)
	if err != nil {
		t.Fatalf("ensureLibcambia c2: %v", err)
	}
	if lib2.engineTreeSha != lib1.engineTreeSha {
		t.Fatalf("engine tree sha differs across cfr-only change: %q vs %q", lib1.engineTreeSha, lib2.engineTreeSha)
	}
	if countGoBuilds(fr) != buildsBefore {
		t.Fatalf("cfr-only commit rebuilt libcambia (cache miss); builds %d -> %d", buildsBefore, countGoBuilds(fr))
	}
}

func TestEvictLibcambiaLRU(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	m.cfg.MaxLibcambia = 2
	if err := os.MkdirAll(m.libcambiaDir, 0o755); err != nil {
		t.Fatal(err)
	}

	names := []string{"eng-old.so", "eng-mid.so", "eng-new.so"}
	base := time.Now().Add(-time.Hour)
	for i, n := range names {
		p := filepath.Join(m.libcambiaDir, n)
		if err := os.WriteFile(p, []byte("so"), 0o644); err != nil {
			t.Fatal(err)
		}
		ts := base.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(p, ts, ts); err != nil {
			t.Fatal(err)
		}
	}

	m.evictLibcambia(nil)

	ents, _ := os.ReadDir(m.libcambiaDir)
	if len(ents) != 2 {
		t.Fatalf("expected 2 libcambia artifacts after eviction, got %d", len(ents))
	}
	// The oldest must be the one evicted.
	if _, err := os.Stat(filepath.Join(m.libcambiaDir, "eng-old.so")); !os.IsNotExist(err) {
		t.Fatalf("oldest libcambia not evicted: %v", err)
	}
}

func countGoBuilds(fr *fakeRunner) int {
	n := 0
	for _, c := range fr.callsFor("go") {
		if len(c.Args) > 0 && c.Args[0] == "build" {
			n++
		}
	}
	return n
}
