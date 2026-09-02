package ingest

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnsureMirrorSetsGcAutoZero(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	out := runGit(t, m.mirrorDir, "config", "gc.auto")
	if strings.TrimSpace(out) != "0" {
		t.Fatalf("gc.auto = %q, want 0", out)
	}
}

func TestVerifyReceiptRightSha(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-a")

	if err := m.verifyReceipt(context.Background(), "job-a", sha); err != nil {
		t.Fatalf("verifyReceipt: %v", err)
	}
}

func TestVerifyReceiptWrongSha(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-a")

	// A different but well-formed sha (40 zeros stand in for "some other commit"
	// the submitter claimed). The ref resolves to sha, not this.
	wrong := strings.Repeat("0", 40)
	err := m.verifyReceipt(context.Background(), "job-a", wrong)
	if !errors.Is(err, ErrReceiptMismatch) {
		t.Fatalf("want ErrReceiptMismatch, got %v", err)
	}
}

func TestVerifyReceiptMissingRef(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	_, sha := sourceRepo(t, "lock-v1")
	// No push: the job ref does not exist.
	err := m.verifyReceipt(context.Background(), "nope", sha)
	if !errors.Is(err, ErrReceiptMismatch) {
		t.Fatalf("want ErrReceiptMismatch for missing ref, got %v", err)
	}
}

func TestBundleFetchFallbackCreatesRef(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	// Author the job-neutral snapshot ref inside the source repo, then bundle
	// it: that is the ref name a coordinator-built bundle carries.
	runGit(t, src, "update-ref", snapshotRef(sha), sha)
	bundle := filepath.Join(t.TempDir(), "job-b.bundle")
	runGit(t, src, "bundle", "create", bundle, snapshotRef(sha))

	if err := m.BundleFetch(context.Background(), "job-b", sha, bundle); err != nil {
		t.Fatalf("BundleFetch: %v", err)
	}
	if err := m.verifyReceipt(context.Background(), "job-b", sha); err != nil {
		t.Fatalf("verifyReceipt after bundle: %v", err)
	}
}

func TestBundleFetchRefusesForceUpdate(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src1, sha1 := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src1, sha1, "job-c")

	// A second, unrelated repo produces a divergent commit, bundled under its
	// own snapshot ref.
	src2, sha2 := sourceRepo(t, "lock-v2")
	runGit(t, src2, "update-ref", snapshotRef(sha2), sha2)
	bundle := filepath.Join(t.TempDir(), "job-c.bundle")
	runGit(t, src2, "bundle", "create", bundle, snapshotRef(sha2))

	// Non-fast-forward fetch into an existing ref must be refused (no force).
	err := m.BundleFetch(context.Background(), "job-c", sha2, bundle)
	if err == nil {
		t.Fatal("expected non-fast-forward bundle fetch to be refused")
	}
	// The ref must still point at the original sha.
	got := runGit(t, m.mirrorDir, "rev-parse", jobRef("job-c"))
	if got != sha1 {
		t.Fatalf("ref moved to %s, want unchanged %s", got, sha1)
	}
}

func TestDeleteJobRef(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-d")

	if err := m.deleteJobRef(context.Background(), "job-d"); err != nil {
		t.Fatalf("deleteJobRef: %v", err)
	}
	if _, err := m.git(context.Background(), "rev-parse", "--verify", "--quiet", jobRef("job-d")); err == nil {
		t.Fatal("ref still resolves after delete")
	}
	// Idempotent second delete.
	if err := m.deleteJobRef(context.Background(), "job-d"); err != nil {
		t.Fatalf("second deleteJobRef: %v", err)
	}
}

// TestStartupSweepReapsSnapshotRefs pins the lifetime of the job-neutral bundle
// refs (cambia-2128): a build publishes one, and the sweep that runs the
// mirror's pruning gc deletes it, so the commit a cached bundle was built from
// is not pinned past the job that needed it. The manager here stages no
// worktree, which is the shape of a coordinator that dispatches only to remote
// nodes: it builds bundles and never adds a worktree of its own.
func TestStartupSweepReapsSnapshotRefs(t *testing.T) {
	m, _ := testManager(t, ExecRunner{})
	src, sha := sourceRepo(t, "lock-v1")
	pushJobRef(t, m, src, sha, "job-sweep")

	if _, err := m.BundleCreate(context.Background(), "job-sweep", nil); err != nil {
		t.Fatalf("BundleCreate: %v", err)
	}
	refs, err := m.listSnapshotRefs(context.Background())
	if err != nil {
		t.Fatalf("listSnapshotRefs: %v", err)
	}
	if len(refs) != 1 || refs[0] != snapshotRef(sha) {
		t.Fatalf("snapshot refs after a build = %v, want exactly [%s]", refs, snapshotRef(sha))
	}

	if err := m.StartupSweep(nil); err != nil {
		t.Fatalf("StartupSweep: %v", err)
	}
	refs, err = m.listSnapshotRefs(context.Background())
	if err != nil {
		t.Fatalf("listSnapshotRefs after the sweep: %v", err)
	}
	if len(refs) != 0 {
		t.Fatalf("the sweep left snapshot refs behind: %v", refs)
	}
}

func TestValidateCommitRejectsNonHex(t *testing.T) {
	for _, bad := range []string{"", "xyz", strings.Repeat("g", 40), strings.Repeat("a", 39)} {
		if err := validateCommit(bad); !errors.Is(err, ErrInvalidCommit) {
			t.Fatalf("validateCommit(%q) = %v, want ErrInvalidCommit", bad, err)
		}
	}
	if err := validateCommit(strings.Repeat("a", 40)); err != nil {
		t.Fatalf("validateCommit(valid) = %v", err)
	}
}
