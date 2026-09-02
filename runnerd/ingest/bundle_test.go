package ingest

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// commitOnTop adds a new commit to an existing repo dir (built by sourceRepo)
// and returns its sha, so basis tests have two distinct, connected commits.
func commitOnTop(t *testing.T, dir, marker string) string {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "cfr", marker+".txt"), marker)
	runGit(t, dir, "add", "-A")
	runGit(t, dir, "commit", "-q", "-m", marker)
	return runGit(t, dir, "rev-parse", "HEAD")
}

// bundleCreateCallArgv returns the argv of the recorded `git bundle create`
// invocation, or nil if none was recorded.
func bundleCreateCallArgv(fr *fakeRunner) []string {
	for _, c := range fr.callsFor("git") {
		if len(c.Args) >= 6 && c.Args[4] == "bundle" && c.Args[5] == "create" {
			return c.Args
		}
	}
	return nil
}

// TestBundleCreateFullBundleFetchPassesVerifyReceipt is AC(1): a full bundle
// (empty basis) fetched by BundleFetch into a fresh mirror passes
// verifyReceipt for the same job and commit.
func TestBundleCreateFullBundleFetchPassesVerifyReceipt(t *testing.T) {
	coord, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, coord, src, sha, "job-full")

	desc, err := coord.BundleCreate(context.Background(), "job-full", nil)
	if err != nil {
		t.Fatalf("BundleCreate: %v", err)
	}
	if desc.SHA256 == "" || desc.Size == 0 {
		t.Fatalf("descriptor incomplete: %+v", desc)
	}

	node, _ := testManager(t, ExecRunner{})
	if err := node.BundleFetch(context.Background(), "job-full", desc.Path); err != nil {
		t.Fatalf("BundleFetch: %v", err)
	}
	if err := node.verifyReceipt(context.Background(), "job-full", sha); err != nil {
		t.Fatalf("verifyReceipt after full bundle fetch: %v", err)
	}
}

// TestBundleCreateThinBundleBasisPresentAndAbsent is AC(2): a thin bundle
// against a basis the receiving mirror already has fetches cleanly; against
// one it lacks, the fetch fails with a recognizable prerequisite error.
func TestBundleCreateThinBundleBasisPresentAndAbsent(t *testing.T) {
	coord, _ := testManager(t, ExecRunner{})
	src, sha1 := sourceRepo(t, "lock-v1")
	sha2 := commitOnTop(t, src, "second")
	pushJobRef(t, coord, src, sha1, "job-a")
	pushJobRef(t, coord, src, sha2, "job-b")

	fullA, err := coord.BundleCreate(context.Background(), "job-a", nil)
	if err != nil {
		t.Fatalf("BundleCreate(job-a, full): %v", err)
	}
	thinB, err := coord.BundleCreate(context.Background(), "job-b", []string{sha1})
	if err != nil {
		t.Fatalf("BundleCreate(job-b, thin basis=sha1): %v", err)
	}

	// A mirror seeded with sha1 (via job-a's full bundle) already has the
	// basis object; the thin fetch succeeds.
	seeded, _ := testManager(t, ExecRunner{})
	if err := seeded.BundleFetch(context.Background(), "job-a", fullA.Path); err != nil {
		t.Fatalf("seed BundleFetch: %v", err)
	}
	if err := seeded.BundleFetch(context.Background(), "job-b", thinB.Path); err != nil {
		t.Fatalf("thin BundleFetch against a mirror holding the basis: %v", err)
	}

	// A fresh, empty mirror lacks sha1; the thin fetch fails recognizably.
	empty, _ := testManager(t, ExecRunner{})
	err = empty.BundleFetch(context.Background(), "job-b", thinB.Path)
	if err == nil {
		t.Fatal("expected thin BundleFetch to fail against a mirror lacking the basis")
	}
	if !errors.Is(err, ErrBundlePrereqMissing) {
		t.Fatalf("want ErrBundlePrereqMissing, got %v", err)
	}
}

// TestBundleCreateBasisValidation is AC(3): a basis that is not 40 hex, or is
// well-formed but absent from the mirror, is refused before it ever reaches a
// git argv (no `git bundle create` call is recorded).
func TestBundleCreateBasisValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		basis []string
	}{
		{"not hex", []string{"not-a-commit-sha-------------------xx"}},
		{"wrong length", []string{strings.Repeat("a", 39)}},
		{"absent from mirror", []string{strings.Repeat("0", 40)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fr := newFakeRunner()
			m, _ := testManager(t, fr)
			src, sha := sourceRepo(t, "lock-v1")
			pushJobRef(t, m, src, sha, "job-basis")

			_, err := m.BundleCreate(context.Background(), "job-basis", tc.basis)
			if !errors.Is(err, ErrInvalidBasis) {
				t.Fatalf("want ErrInvalidBasis, got %v", err)
			}
			if got := bundleCreateCallArgv(fr); got != nil {
				t.Fatalf("git bundle create ran despite an invalid basis: %v", got)
			}
		})
	}
}

// TestBundleCreateBasisCap refuses a basis list past the 16-entry cap before
// argv, same as a malformed entry.
func TestBundleCreateBasisCap(t *testing.T) {
	fr := newFakeRunner()
	m, _ := testManager(t, fr)
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-cap")

	basis := make([]string, 17)
	for i := range basis {
		basis[i] = strings.Repeat("a", 40)
	}
	_, err := m.BundleCreate(context.Background(), "job-cap", basis)
	if !errors.Is(err, ErrInvalidBasis) {
		t.Fatalf("want ErrInvalidBasis for a basis past the cap, got %v", err)
	}
	if got := bundleCreateCallArgv(fr); got != nil {
		t.Fatalf("git bundle create ran despite an over-cap basis: %v", got)
	}
}

// TestBundleCreateArgvVerbatim is AC(4): the recorded `git bundle create`
// invocation carries core.useReplaceRefs=false, the job ref, and the negated
// basis, and never carries --no-tags (git bundle create rejects it as an
// unrecognized argument, git 2.34.1).
func TestBundleCreateArgvVerbatim(t *testing.T) {
	fr := newFakeRunner()
	m, _ := testManager(t, fr)
	src, sha1 := sourceRepo(t, "lock-v1")
	sha2 := commitOnTop(t, src, "second")
	// Pushing sha2 carries sha1 (its parent) into the mirror's object store as
	// a reachable ancestor, so the basis guard's rev-parse --verify accepts it
	// without a ref of its own pointing at it directly.
	pushJobRef(t, m, src, sha2, "job-argv")

	desc, err := m.BundleCreate(context.Background(), "job-argv", []string{sha1})
	if err != nil {
		t.Fatalf("BundleCreate: %v", err)
	}

	got := bundleCreateCallArgv(fr)
	if got == nil {
		t.Fatal("no git bundle create call recorded")
	}
	if len(got) != 9 {
		t.Fatalf("argv = %v, want 9 elements", got)
	}
	want := []string{"-C", m.mirrorDir, "-c", "core.useReplaceRefs=false", "bundle", "create", got[6], "refs/harness/job-argv", "^" + sha1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("argv[%d] = %q, want %q (full argv %v)", i, got[i], want[i], got)
		}
	}
	for _, a := range got {
		if a == "--no-tags" {
			t.Fatal("--no-tags present in argv; git bundle create rejects it as an unrecognized argument")
		}
	}

	wantPath := filepath.Join(m.snapshotDir, bundleCacheKey(sha2, []string{sha1})+bundleExt)
	if desc.Path != wantPath {
		t.Fatalf("descriptor path = %q, want %q", desc.Path, wantPath)
	}
}

// TestBundleCreateDigestStableAcrossRebuilds is AC(5)'s digest-stability half:
// rebuilding the same (commit, basis) bundle from an unchanged mirror
// reproduces the same sha256 and size, so the ETag it feeds is stable.
func TestBundleCreateDigestStableAcrossRebuilds(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-stable")

	first, err := m.BundleCreate(context.Background(), "job-stable", nil)
	if err != nil {
		t.Fatalf("BundleCreate (1st): %v", err)
	}
	if err := m.removeBundleEntry(first.Path); err != nil {
		t.Fatalf("removeBundleEntry (force a rebuild): %v", err)
	}
	second, err := m.BundleCreate(context.Background(), "job-stable", nil)
	if err != nil {
		t.Fatalf("BundleCreate (2nd): %v", err)
	}
	if first.SHA256 != second.SHA256 {
		t.Fatalf("sha256 changed across rebuild: %q vs %q", first.SHA256, second.SHA256)
	}
	if first.Size != second.Size {
		t.Fatalf("size changed across rebuild: %d vs %d", first.Size, second.Size)
	}
}

// TestBundleCreateCacheHitSkipsRebuild exercises the per-commit cache's core
// claim: a second BundleCreate call for the same (commit, basis) reuses the
// cached artifact and never forks a second git bundle create.
func TestBundleCreateCacheHitSkipsRebuild(t *testing.T) {
	fr := newFakeRunner()
	m, _ := testManager(t, fr)
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-cache")

	count := func() int {
		n := 0
		for _, c := range fr.callsFor("git") {
			if len(c.Args) >= 6 && c.Args[4] == "bundle" && c.Args[5] == "create" {
				n++
			}
		}
		return n
	}

	if _, err := m.BundleCreate(context.Background(), "job-cache", nil); err != nil {
		t.Fatalf("BundleCreate (1st): %v", err)
	}
	if got := count(); got != 1 {
		t.Fatalf("expected exactly one git bundle create after the 1st call, got %d", got)
	}
	if _, err := m.BundleCreate(context.Background(), "job-cache", nil); err != nil {
		t.Fatalf("BundleCreate (2nd): %v", err)
	}
	if got := count(); got != 1 {
		t.Fatalf("cache hit forked another git bundle create: %d total calls, want 1", got)
	}
}

// TestBundleCreateConcurrentCallersCollapseToOneBuild exercises the
// singleflight half of the cache: a burst of concurrent BundleCreate calls for
// the same (commit, basis) forks exactly one git bundle create, and every
// caller gets back an identical descriptor.
func TestBundleCreateConcurrentCallersCollapseToOneBuild(t *testing.T) {
	fr := newFakeRunner()
	m, _ := testManager(t, fr)
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-race")

	const n = 8
	results := make([]BundleDescriptor, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = m.BundleCreate(context.Background(), "job-race", nil)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("caller %d: BundleCreate: %v", i, err)
		}
	}
	for i := 1; i < n; i++ {
		if results[i] != results[0] {
			t.Fatalf("caller %d descriptor %+v differs from caller 0 %+v", i, results[i], results[0])
		}
	}

	builds := 0
	for _, c := range fr.callsFor("git") {
		if len(c.Args) >= 6 && c.Args[4] == "bundle" && c.Args[5] == "create" {
			builds++
		}
	}
	if builds != 1 {
		t.Fatalf("expected exactly one git bundle create across %d concurrent callers, got %d", n, builds)
	}
}

// TestBundleCreateRejectsUnknownJob confirms BundleCreate refuses a job with
// no pushed ref rather than building an empty or misattributed bundle.
func TestBundleCreateRejectsUnknownJob(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	_, err := m.BundleCreate(context.Background(), "no-such-job", nil)
	if !errors.Is(err, ErrReceiptMismatch) {
		t.Fatalf("want ErrReceiptMismatch for an unpushed job ref, got %v", err)
	}
}
