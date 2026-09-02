package quarantine

import (
	"bytes"
	"errors"
	"go/build"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestInvalidDigestRejectedBeforeAnyFilesystemCall covers the malformed-digest
// table of D41: every shape is invalid_digest on every route that lets a digest
// name a path, and nothing is created anywhere on disk, so the check runs
// before filepath.Join rather than after it (AC 12).
func TestInvalidDigestRejectedBeforeAnyFilesystemCall(t *testing.T) {
	good := strings.Repeat("a", 64)
	cases := []struct {
		name   string
		digest string
	}{
		{"empty", ""},
		{"uppercase hex", strings.ToUpper(good)},
		{"63 characters", strings.Repeat("a", 63)},
		{"65 characters", strings.Repeat("a", 65)},
		{"parent traversal", "../" + strings.Repeat("a", 61)},
		{"path separator", strings.Repeat("a", 31) + "/" + strings.Repeat("a", 32)},
		{"nul byte", strings.Repeat("a", 63) + "\x00"},
		{"non ascii", strings.Repeat("a", 63) + "é"},
		{"non hex letter", strings.Repeat("z", 64)},
		{"trailing newline", strings.Repeat("a", 63) + "\n"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, nil)
			before := treeSnapshot(t, r.quarDir, r.runsDir)

			if _, err := r.store.BlobStatus(r.lease, tc.digest); !errors.Is(err, ErrInvalidDigest) {
				t.Fatalf("BlobStatus: want ErrInvalidDigest, got %v", err)
			}
			if _, err := r.store.AppendChunk(r.lease, tc.digest, 0, 3, 4, bytes.NewReader([]byte("data"))); !errors.Is(err, ErrInvalidDigest) {
				t.Fatalf("AppendChunk: want ErrInvalidDigest, got %v", err)
			}
			if err := r.store.AbandonPart(r.lease, tc.digest); !errors.Is(err, ErrInvalidDigest) {
				t.Fatalf("AbandonPart: want ErrInvalidDigest, got %v", err)
			}
			if _, _, err := r.store.Probe(r.lease, []string{tc.digest}); !errors.Is(err, ErrInvalidDigest) {
				t.Fatalf("Probe: want ErrInvalidDigest, got %v", err)
			}

			after := treeSnapshot(t, r.quarDir, r.runsDir)
			if !sameStrings(before, after) {
				t.Fatalf("filesystem changed under a rejected digest:\nbefore %v\nafter  %v", before, after)
			}
			if _, err := os.Stat(filepath.Join(r.quarDir, "node-a")); !os.IsNotExist(err) {
				t.Fatalf("lease tree was created under a rejected digest: %v", err)
			}
		})
	}
}

// TestOffsetFencingAndHashMismatch covers AC 1: a start that is not the current
// part size is refused with the true offset, and a completed part whose content
// does not hash to its name is destroyed.
func TestOffsetFencingAndHashMismatch(t *testing.T) {
	r := newRig(t, nil)
	data := []byte("checkpoint bytes for the offset fence")
	digest := sha256hex(data)

	if _, err := r.store.AppendChunk(r.lease, digest, 0, 9, int64(len(data)), bytes.NewReader(data[:10])); err != nil {
		t.Fatalf("first chunk: %v", err)
	}

	_, err := r.store.AppendChunk(r.lease, digest, 4, 13, int64(len(data)), bytes.NewReader(data[4:14]))
	var om *OffsetMismatchError
	if !errors.As(err, &om) {
		t.Fatalf("want OffsetMismatchError, got %v", err)
	}
	if om.Offset != 10 {
		t.Fatalf("offset_mismatch reported %d, want the true part size 10", om.Offset)
	}

	// Complete the file with bytes that do not hash to the digest naming it.
	wrong := append([]byte{}, data[10:]...)
	wrong[0] ^= 0xff
	_, err = r.store.AppendChunk(r.lease, digest, 10, int64(len(data))-1, int64(len(data)), bytes.NewReader(wrong))
	if !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("want ErrHashMismatch, got %v", err)
	}
	leaseDir := filepath.Join(r.quarDir, "node-a", "job-a", "lease-1")
	if _, err := os.Stat(partPath(leaseDir, digest)); !os.IsNotExist(err) {
		t.Fatalf("part survived a hash mismatch: %v", err)
	}
	if _, err := os.Stat(blobPath(leaseDir, digest)); !os.IsNotExist(err) {
		t.Fatalf("a blob was created from mismatched bytes: %v", err)
	}
	if got := r.store.Stats(r.lease).HashMismatches; got != 1 {
		t.Fatalf("HashMismatches = %d, want 1", got)
	}
}

// TestResumeAfterCrashUsesPartSizeAlone covers AC 2: a second store over the
// same directories holds no running hash and no side-car state, so the part's
// own size is the resume offset and the rehash reconstructs the digest.
func TestResumeAfterCrashUsesPartSizeAlone(t *testing.T) {
	r := newRig(t, nil)
	data := bytes.Repeat([]byte("resume"), 300)
	digest := sha256hex(data)
	half := int64(len(data) / 2)

	if _, err := r.store.AppendChunk(r.lease, digest, 0, half-1, int64(len(data)), bytes.NewReader(data[:half])); err != nil {
		t.Fatalf("first chunk: %v", err)
	}

	restarted, err := New(Config{
		QuarantineDir: r.quarDir,
		RunsDir:       r.runsDir,
		Now:           r.clock.Now,
	})
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	st, err := restarted.BlobStatus(r.lease, digest)
	if err != nil {
		t.Fatalf("BlobStatus after restart: %v", err)
	}
	if st.Offset != half || st.Verified {
		t.Fatalf("BlobStatus = %+v, want offset %d and unverified", st, half)
	}

	res, err := restarted.AppendChunk(r.lease, digest, half, int64(len(data))-1, int64(len(data)), bytes.NewReader(data[half:]))
	if err != nil {
		t.Fatalf("resumed chunk: %v", err)
	}
	if !res.Verified || res.CommittedOffset != int64(len(data)) {
		t.Fatalf("resumed chunk = %+v, want verified at %d", res, len(data))
	}
	leaseDir := filepath.Join(r.quarDir, "node-a", "job-a", "lease-1")
	if got := mustRead(t, blobPath(leaseDir, digest)); !bytes.Equal(got, data) {
		t.Fatal("resumed blob content differs from the uploaded bytes")
	}
}

// TestRepeatedUploadOfVerifiedDigestIsFree asserts that re-offering a verified
// digest reads no body and answers with the full size (D50 step 3).
func TestRepeatedUploadOfVerifiedDigestIsFree(t *testing.T) {
	r := newRig(t, nil)
	data := []byte("already proved")
	digest := r.putBlob(t, data)

	guard := &failingReader{t: t}
	res, err := r.store.AppendChunk(r.lease, digest, 0, int64(len(data))-1, int64(len(data)), guard)
	if err != nil {
		t.Fatalf("repeat upload: %v", err)
	}
	if !res.Verified || res.CommittedOffset != int64(len(data)) {
		t.Fatalf("repeat upload = %+v, want verified at %d", res, len(data))
	}
	if guard.read {
		t.Fatal("a repeated upload of a verified digest read the request body")
	}
}

type failingReader struct {
	t    *testing.T
	read bool
}

func (f *failingReader) Read(p []byte) (int, error) {
	f.read = true
	return 0, errors.New("body should not have been read")
}

// TestProbeAnswersOverProvableSetOnly covers D50 step 1: a lease sees its own
// blobs and its grant set, and nothing else, so the probe is not an existence
// oracle for another lease's content.
func TestProbeAnswersOverProvableSetOnly(t *testing.T) {
	r := newRig(t, nil)
	own := r.putBlob(t, []byte("own blob"))

	seedPath := filepath.Join(t.TempDir(), "seed.pt")
	seedData := []byte("granted seed")
	if err := os.WriteFile(seedPath, seedData, 0o644); err != nil {
		t.Fatalf("write seed: %v", err)
	}
	granted := sha256hex(seedData)
	r.lease.Grants = map[string]Grant{granted: {Digest: granted, Size: int64(len(seedData)), SourcePath: seedPath}}

	other := Lease{NodeID: "node-b", JobID: "job-b", LeaseID: "lease-2", Epoch: 1}
	otherData := []byte("another lease's bytes")
	otherDigest := sha256hex(otherData)
	if _, err := r.store.AppendChunk(other, otherDigest, 0, int64(len(otherData))-1, int64(len(otherData)), bytes.NewReader(otherData)); err != nil {
		t.Fatalf("seed the other lease: %v", err)
	}

	unknown := sha256hex([]byte("never uploaded"))
	have, want, err := r.store.Probe(r.lease, []string{own, granted, otherDigest, unknown, own})
	if err != nil {
		t.Fatalf("Probe: %v", err)
	}
	if !sameStrings(have, []string{own, granted}) {
		t.Fatalf("have = %v, want the own blob and the granted seed", have)
	}
	if !sameStrings(want, []string{otherDigest, unknown}) {
		t.Fatalf("want = %v, want the other lease's digest and the unknown one", want)
	}
}

// TestWatermarkFiresBeforeTheConfiguredFloor covers AC 10: the store refuses an
// upload while free disk is still above the coordinator's own service floor,
// which is what keeps N nodes inside quota from driving it below that floor.
func TestWatermarkFiresBeforeTheConfiguredFloor(t *testing.T) {
	free := 25.0 // above the 20 GB floor, below the 30 GB watermark
	r := newRig(t, func(c *Config) {
		c.Limits = Limits{MinFreeDiskGB: 20, WatermarkMarginGB: 10}
		c.DiskFreeGB = func(string) float64 { return free }
	})
	data := []byte("bytes that must not land")
	digest := sha256hex(data)

	before := treeSnapshot(t, r.quarDir, r.runsDir)
	_, err := r.store.AppendChunk(r.lease, digest, 0, int64(len(data))-1, int64(len(data)), bytes.NewReader(data))
	var full *StoreFullError
	if !errors.As(err, &full) {
		t.Fatalf("want StoreFullError above the floor, got %v", err)
	}
	if full.FreeGB <= r.store.Limits().MinFreeDiskGB {
		t.Fatalf("watermark fired at %.1f GB free, at or below the %.1f GB floor: it must fire above it", full.FreeGB, r.store.Limits().MinFreeDiskGB)
	}
	if !sameStrings(before, treeSnapshot(t, r.quarDir, r.runsDir)) {
		t.Fatal("a store-full refusal wrote bytes")
	}

	free = 40
	if _, err := r.store.AppendChunk(r.lease, digest, 0, int64(len(data))-1, int64(len(data)), bytes.NewReader(data)); err != nil {
		t.Fatalf("upload above the watermark: %v", err)
	}
}

// TestPerFileAndLeaseBudgetCaps covers the two byte quotas of D56 that the
// store owns.
func TestPerFileAndLeaseBudgetCaps(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Limits = Limits{MaxFileBytes: 16, MaxLeaseBytes: 24}
	})
	big := bytes.Repeat([]byte("x"), 32)
	var tooLarge *FileTooLargeError
	if _, err := r.store.AppendChunk(r.lease, sha256hex(big), 0, 31, 32, bytes.NewReader(big)); !errors.As(err, &tooLarge) {
		t.Fatalf("want FileTooLargeError, got %v", err)
	}

	first := bytes.Repeat([]byte("a"), 16)
	r.putBlob(t, first)
	second := bytes.Repeat([]byte("b"), 16)
	var budget *LeaseBudgetError
	if _, err := r.store.AppendChunk(r.lease, sha256hex(second), 0, 15, 16, bytes.NewReader(second)); !errors.As(err, &budget) {
		t.Fatalf("want LeaseBudgetError, got %v", err)
	}
}

// TestAbandonPartRemovesInFlightUpload covers D50 step 4.
func TestAbandonPartRemovesInFlightUpload(t *testing.T) {
	r := newRig(t, nil)
	data := bytes.Repeat([]byte("abandon"), 10)
	digest := sha256hex(data)
	if _, err := r.store.AppendChunk(r.lease, digest, 0, 9, int64(len(data)), bytes.NewReader(data[:10])); err != nil {
		t.Fatalf("first chunk: %v", err)
	}
	if err := r.store.AbandonPart(r.lease, digest); err != nil {
		t.Fatalf("AbandonPart: %v", err)
	}
	st, err := r.store.BlobStatus(r.lease, digest)
	if err != nil {
		t.Fatalf("BlobStatus: %v", err)
	}
	if st.Offset != 0 || st.Verified {
		t.Fatalf("BlobStatus after abandon = %+v, want a zero offset", st)
	}
}

// TestPackageIsLeaf asserts the package imports no runnerd code beyond the two
// leaf helpers it is specified to reuse: no HTTP, no dispatcher.
func TestPackageIsLeaf(t *testing.T) {
	pkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatalf("ImportDir: %v", err)
	}
	allowed := map[string]bool{
		"github.com/jason-s-yu/cambia/runnerd/pathguard": true,
		"github.com/jason-s-yu/cambia/runnerd/procmgr":   true,
		"github.com/jason-s-yu/cambia/runnerd/sysprobe":  true,
	}
	for _, imp := range pkg.Imports {
		if strings.HasPrefix(imp, "github.com/jason-s-yu/cambia/") && !allowed[imp] {
			t.Errorf("quarantine imports %s: the package must stay a leaf", imp)
		}
		if imp == "net/http" {
			t.Errorf("quarantine imports net/http: status codes belong to the route layer")
		}
	}
}
