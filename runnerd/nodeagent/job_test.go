package nodeagent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
)

// The download and snapshot-cache half of a lease's staging (D48, D53). The
// case every test here circles is a node that already holds what it is being
// granted: the cache was keyed by job id while the coordinator keys the served
// artifact by (commit, basis), so a second claim of one job resumed a complete
// file, asked for a range past its end, and was answered 416 (cambia-2018).

// TestSnapshotCacheKeyNamesTheArtifactNotTheJob covers the key itself: the
// pinned commit for a full bundle, the commit plus a basis digest for a thin
// one, and the job id only when the claim pins no usable commit.
func TestSnapshotCacheKeyNamesTheArtifactNotTheJob(t *testing.T) {
	commit := strings.Repeat("a", 40)
	other := strings.Repeat("b", 40)

	full := jobRunForKey(t, commit, nil)
	if got := full.snapshotCacheKey(); got != commit {
		t.Fatalf("full-bundle key = %q, want the bare commit %q", got, commit)
	}

	thin := jobRunForKey(t, commit, []string{other, strings.Repeat("c", 40)})
	key := thin.snapshotCacheKey()
	if !strings.HasPrefix(key, commit+"-") || len(key) != len(commit)+1+16 {
		t.Fatalf("thin key = %q, want %q plus a 16-hex basis digest", key, commit)
	}
	if key == commit {
		t.Fatal("a thin bundle shares the full bundle's key: the basis is not in it")
	}

	// The basis is a set: the coordinator sorts it before hashing, so the
	// order a node advertised its commits in cannot fork the cache.
	reordered := jobRunForKey(t, commit, []string{strings.Repeat("c", 40), other})
	if got := reordered.snapshotCacheKey(); got != key {
		t.Fatalf("reordered basis key = %q, want %q", got, key)
	}
	narrower := jobRunForKey(t, commit, []string{other})
	if got := narrower.snapshotCacheKey(); got == key {
		t.Fatalf("two different bases share the key %q", got)
	}

	unpinned := jobRunForKey(t, "HEAD", nil)
	if got := unpinned.snapshotCacheKey(); got != unpinned.rec.JobID {
		t.Fatalf("key for an unpinned commit = %q, want the job id %q", got, unpinned.rec.JobID)
	}
}

// TestDownloadSkipsTheFetchForACompleteFile is the cheap half of the fix: a
// local file that hashes to the granted digest is the artifact, so no request
// goes out at all.
func TestDownloadSkipsTheFetchForACompleteFile(t *testing.T) {
	stub := newStubCoordinator(t)
	agent, _ := testAgent(t, stub, nil)
	job := newTestJob(t, agent, stub, Spec{Kind: KindTrain, Name: "cached-job", Commit: strings.Repeat("a", 40)})

	dest := filepath.Join(t.TempDir(), "cached.bundle")
	if err := os.WriteFile(dest, stub.snapshot, 0o644); err != nil {
		t.Fatal(err)
	}
	before := stub.requestCount()

	if err := job.download(context.Background(), job.snapshotURL(), dest, stub.snapshotDigest()); err != nil {
		t.Fatalf("download over a complete file: %v", err)
	}

	if got := stub.requestCount() - before; got != 0 {
		t.Fatalf("a complete cached file still sent %d requests", got)
	}
	body, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != string(stub.snapshot) {
		t.Fatalf("the cached file is now %q, want it untouched", body)
	}
}

// TestDownloadRestartsWhenTheResumeOffsetIsPastTheEnd is the other half: a
// local file as long as the artifact but not equal to it resumes past the end,
// is answered 416, fails the digest check, and is fetched again from zero
// rather than reported as a fetch failure.
func TestDownloadRestartsWhenTheResumeOffsetIsPastTheEnd(t *testing.T) {
	stub := newStubCoordinator(t)
	agent, _ := testAgent(t, stub, nil)
	job := newTestJob(t, agent, stub, Spec{Kind: KindTrain, Name: "stale-job", Commit: strings.Repeat("a", 40)})

	dest := filepath.Join(t.TempDir(), "stale.bundle")
	stale := []byte(strings.Repeat("x", len(stub.snapshot)+8))
	if err := os.WriteFile(dest, stale, 0o644); err != nil {
		t.Fatal(err)
	}
	before := stub.requestCount()

	if err := job.download(context.Background(), job.snapshotURL(), dest, stub.snapshotDigest()); err != nil {
		t.Fatalf("download over a stale file of the same length: %v", err)
	}

	if got := stub.requestCount() - before; got < 2 {
		t.Fatalf("requests = %d, want the 416 and the restart from zero", got)
	}
	body, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != string(stub.snapshot) {
		t.Fatalf("the restarted download left %q, want the served artifact", body)
	}
}

// TestFetchSnapshotServesASecondClaimFromTheCache is the ticket's own case:
// one job claimed twice on one node at one commit fetches the bundle once and
// imports it both times.
func TestFetchSnapshotServesASecondClaimFromTheCache(t *testing.T) {
	stub := newStubCoordinator(t)
	env := &fakeEnv{worktree: t.TempDir()}
	agent, cfg := testAgent(t, stub, func(o *Options) {
		env.runsDir = o.Config.RunsDir
		o.Env = env
	})
	spec := Spec{Kind: KindTrain, Name: "twice-claimed", Commit: strings.Repeat("a", 40)}
	ctx := context.Background()

	first := newTestJob(t, agent, stub, spec)
	if err := first.fetchSnapshot(ctx); err != nil {
		t.Fatalf("first fetchSnapshot: %v", err)
	}
	cached := filepath.Join(cfg.BaseDir, "snapshots", spec.Commit+".bundle")
	if _, err := os.Stat(cached); err != nil {
		t.Fatalf("the bundle is not cached under the pinned commit: %v", err)
	}
	after := stub.requestCount()

	second := newTestJob(t, agent, stub, spec)
	if err := second.fetchSnapshot(ctx); err != nil {
		t.Fatalf("second fetchSnapshot: %v", err)
	}

	if got := stub.requestCount() - after; got != 0 {
		t.Fatalf("the second claim sent %d requests for a bundle the node holds", got)
	}
	env.mu.Lock()
	fetched := env.fetched
	env.mu.Unlock()
	if fetched != 2 {
		t.Fatalf("bundle imports = %d, want one per claim", fetched)
	}
}

// jobRunForKey builds the minimum jobRun the cache key reads: a lease record
// with a job id and a snapshot ref.
func jobRunForKey(t *testing.T, commit string, basis []string) *jobRun {
	t.Helper()
	return &jobRun{
		rec:      &leaseRecord{JobID: "key-job", Commit: commit},
		snapshot: nashnet.SnapshotRef{Commit: commit, ThinBasis: basis},
	}
}
