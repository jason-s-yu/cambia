package quarantine

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// blobStat stats the verified blob backing a digest under the rig's lease.
func (r *rig) blobStat(t *testing.T, digest string) os.FileInfo {
	t.Helper()
	fi, err := os.Stat(blobPath(filepath.Join(r.quarDir, "node-a", "job-a", "lease-1"), digest))
	if err != nil {
		t.Fatalf("stat blob: %v", err)
	}
	return fi
}

// runThreeCommits drives the three-commit sequence both materialize modes are
// asserted over: a first batch, an incremental second batch carrying the
// journal, and a third that replaces one file and deletes another.
func runThreeCommits(t *testing.T, r *rig) (map[string][]byte, map[string]string) {
	t.Helper()
	content := map[string][]byte{
		"config.yaml":    []byte("seats: 2\n"),
		"snapshots/a.pt": bytes.Repeat([]byte("weights"), 64),
		"metrics.jsonl":  []byte(`{"iter":1}` + "\n"),
		RunDBPath:        []byte("SQLite format 3\x00 journal one"),
	}
	digests := map[string]string{}
	for path, data := range content {
		digests[path] = r.putBlob(t, data)
	}

	first := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{
			entry("config.yaml", content["config.yaml"], 1_000_000_000),
			entry("snapshots/a.pt", content["snapshots/a.pt"], 1_000_000_001),
		},
	})
	if !sameStrings(first.Promoted, []string{"config.yaml", "snapshots/a.pt"}) {
		t.Fatalf("commit 1 promoted %v", first.Promoted)
	}

	second := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 2, Parent: first.Digest,
		Entries: []Entry{
			entry("metrics.jsonl", content["metrics.jsonl"], 1_000_000_002),
			entry(RunDBPath, content[RunDBPath], 1_000_000_003),
		},
	})
	if len(second.Rejected) != 0 {
		t.Fatalf("commit 2 rejected %v", second.Rejected)
	}

	replacement := bytes.Repeat([]byte("newer weights"), 32)
	content["snapshots/a.pt"] = replacement
	digests["snapshots/a.pt"] = r.putBlob(t, replacement)
	third := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 3, Parent: second.Digest,
		Entries: []Entry{entry("snapshots/a.pt", replacement, 1_000_000_004)},
		Deletes: []string{"config.yaml"},
		Final:   true,
	})
	if !sameStrings(third.Deleted, []string{"config.yaml"}) {
		t.Fatalf("commit 3 deleted %v", third.Deleted)
	}
	delete(content, "config.yaml")

	head := r.head(t)
	if head.Folded.Seq != 3 || head.Digest != third.Digest {
		t.Fatalf("head = seq %d digest %s, want seq 3 digest %s", head.Folded.Seq, head.Digest, third.Digest)
	}
	if !head.Folded.Final {
		t.Fatal("head does not carry final after a final commit")
	}
	if _, err := os.Stat(filepath.Join(r.runDir, "config.yaml")); !os.IsNotExist(err) {
		t.Fatalf("a deleted path survived promotion: %v", err)
	}
	for path, want := range content {
		if got := mustRead(t, filepath.Join(r.runDir, path)); !bytes.Equal(got, want) {
			t.Fatalf("promoted %s differs from the uploaded bytes", path)
		}
	}
	return content, digests
}

// TestThreeCommitSequenceLinkMode covers AC 3 and AC 8 in link mode: the
// promoted files share the blob's inode, run_db.sqlite is a 0644 copy rather
// than a link, and every promoted file is 0644.
func TestThreeCommitSequenceLinkMode(t *testing.T) {
	r := newRig(t, nil)
	if r.store.MaterializeMode() != ModeLink {
		t.Skipf("link probe selected %s: %s", r.store.MaterializeMode(), r.store.MaterializeReason())
	}
	content, digests := runThreeCommits(t, r)

	for path := range content {
		promoted, err := os.Stat(filepath.Join(r.runDir, path))
		if err != nil {
			t.Fatalf("stat promoted %s: %v", path, err)
		}
		if promoted.Mode().Perm() != 0o644 {
			t.Fatalf("promoted %s has mode %v, want 0644", path, promoted.Mode().Perm())
		}
		blob := r.blobStat(t, digests[path])
		same := os.SameFile(promoted, blob)
		if path == RunDBPath && same {
			t.Fatal("run_db.sqlite was linked to its blob; it must be a copy, since the rundb-checkpoint route opens it read-write")
		}
		if path != RunDBPath && !same {
			t.Fatalf("promoted %s is not the same inode as its blob: link mode must link", path)
		}
	}
}

// TestThreeCommitSequenceCopyMode covers AC 3 in copy mode, reached through an
// injected linker that reports EXDEV, so the probe of D49 selects copy.
func TestThreeCommitSequenceCopyMode(t *testing.T) {
	r := newRig(t, func(c *Config) {
		c.Link = func(string, string) error { return syscall.EXDEV }
	})
	if r.store.MaterializeMode() != ModeCopy {
		t.Fatalf("materialize mode = %s, want copy after an EXDEV probe", r.store.MaterializeMode())
	}
	content, digests := runThreeCommits(t, r)

	for path := range content {
		promoted, err := os.Stat(filepath.Join(r.runDir, path))
		if err != nil {
			t.Fatalf("stat promoted %s: %v", path, err)
		}
		if os.SameFile(promoted, r.blobStat(t, digests[path])) {
			t.Fatalf("promoted %s shares its blob's inode in copy mode", path)
		}
		if promoted.Mode().Perm() != 0o644 {
			t.Fatalf("promoted %s has mode %v, want 0644", path, promoted.Mode().Perm())
		}
	}
	if entries, err := os.ReadDir(filepath.Join(r.runDir, TmpDir)); err == nil && len(entries) > 0 {
		t.Fatalf("copy-mode staging directory still holds %d files", len(entries))
	}
}

// TestPromotionOrderingPutsTheJournalLast covers AC 9 and the D57 reader
// contract: every other path is materialized before run_db.sqlite, so a journal
// row that names a checkpoint never precedes the file it names.
func TestPromotionOrderingPutsTheJournalLast(t *testing.T) {
	var runDir string
	var order []string
	var checkpointPresentAtJournal bool
	r := newRig(t, func(c *Config) {
		c.OnMaterialize = func(p string) {
			order = append(order, p)
			if p == RunDBPath {
				_, err := os.Stat(filepath.Join(runDir, "snapshots/a.pt"))
				checkpointPresentAtJournal = err == nil
			}
		}
	})
	runDir = r.runDir

	ckpt := bytes.Repeat([]byte("checkpoint"), 100)
	journal := []byte("SQLite format 3\x00 rows naming the checkpoint")
	metrics := []byte(`{"iter":7}` + "\n")
	r.putBlob(t, ckpt)
	r.putBlob(t, journal)
	r.putBlob(t, metrics)

	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{
			entry(RunDBPath, journal, 5),
			entry("snapshots/a.pt", ckpt, 6),
			entry("metrics.jsonl", metrics, 7),
		},
	})
	if len(order) != 3 || order[len(order)-1] != RunDBPath {
		t.Fatalf("promotion order = %v, want run_db.sqlite last", order)
	}
	if !checkpointPresentAtJournal {
		t.Fatal("the journal was promoted before the checkpoint its rows name")
	}
	if resp.Promoted[len(resp.Promoted)-1] != RunDBPath {
		t.Fatalf("response promoted list = %v, want the journal last", resp.Promoted)
	}
}

// TestOutOfOrderAndReplayedCommit covers AC 4: each fence failure of D51 step 1
// carries the coordinator's head, a replay of the accepted seq with the same
// body returns the recorded response and promotes nothing twice, and the same
// seq with a different body is refused.
func TestOutOfOrderAndReplayedCommit(t *testing.T) {
	r := newRig(t, nil)
	data := []byte("first artifact")
	r.putBlob(t, data)

	req := CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{entry("metrics.jsonl", data, 11)},
	}
	first := r.mustCommit(t, req)

	// Out of order: a seq past the head.
	_, err := r.commit(t, CommitRequest{ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 5, Parent: first.Digest})
	var fence *FenceError
	if !errors.As(err, &fence) || fence.Reason != FenceSeq {
		t.Fatalf("want a seq fence failure, got %v", err)
	}
	if fence.Seq != 1 || fence.Digest != first.Digest {
		t.Fatalf("fence answer = seq %d digest %s, want the coordinator head", fence.Seq, fence.Digest)
	}

	// Wrong parent at the right seq.
	_, err = r.commit(t, CommitRequest{ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 2, Parent: "not the head"})
	if !errors.As(err, &fence) || fence.Reason != FenceParent {
		t.Fatalf("want a parent fence failure, got %v", err)
	}

	// Stale lease epoch.
	_, err = r.commit(t, CommitRequest{ManifestVersion: ManifestVersion, LeaseEpoch: 0, Seq: 2, Parent: first.Digest})
	if !errors.As(err, &fence) || fence.Reason != FenceStaleEpoch {
		t.Fatalf("want a stale-epoch fence failure, got %v", err)
	}

	// Replay of the accepted seq with the recorded body.
	replay := r.mustCommit(t, req)
	if replay.Digest != first.Digest || replay.Seq != first.Seq {
		t.Fatalf("replay = %+v, want the recorded response %+v", replay, first)
	}
	receipt, err := r.store.Receipt(r.lease)
	if err != nil {
		t.Fatalf("Receipt: %v", err)
	}
	if len(receipt) != 1 {
		t.Fatalf("receipt has %d lines, want 1: a replay must not promote twice", len(receipt))
	}

	// The same seq with a different body.
	_, err = r.commit(t, CommitRequest{ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1, Final: true})
	if !errors.As(err, &fence) || fence.Reason != FenceBodyMismatch {
		t.Fatalf("want a body-mismatch fence failure, got %v", err)
	}
}

// TestPartialAcceptancePromotesTheRest covers AC 5: a rejected entry is listed
// and the batch proceeds, and the receipt records both sides.
func TestPartialAcceptancePromotesTheRest(t *testing.T) {
	r := newRig(t, nil)
	good := []byte("promotable artifact")
	digest := r.putBlob(t, good)

	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{
			{Digest: digest, MTime: 21, Path: "metrics.jsonl", Size: int64(len(good))},
			{Digest: digest, MTime: 21, Path: "logs/training.log", Size: int64(len(good))},
			{Digest: digest, MTime: 21, Path: "snapshots/x.pt", Size: int64(len(good))},
		},
	})
	if !sameStrings(resp.Promoted, []string{"metrics.jsonl", "snapshots/x.pt"}) {
		t.Fatalf("promoted = %v, want the two acceptable paths", resp.Promoted)
	}
	if got := rejectionReason(resp, "logs/training.log"); got != ReasonPathReserved {
		t.Fatalf("rejection reason = %q, want %q", got, ReasonPathReserved)
	}
	if _, err := os.Stat(filepath.Join(r.runDir, "logs", "training.log")); !os.IsNotExist(err) {
		t.Fatalf("a reserved path was promoted: %v", err)
	}
	receipt, err := r.store.Receipt(r.lease)
	if err != nil || len(receipt) != 1 {
		t.Fatalf("Receipt = %v, %v", receipt, err)
	}
	if receipt[0].Offered != 3 || len(receipt[0].Promoted) != 2 || len(receipt[0].Rejected) != 1 {
		t.Fatalf("receipt line = %+v, want 3 offered, 2 promoted, 1 rejected", receipt[0])
	}
	if receipt[0].Rejected[0].Reason != ReasonPathReserved {
		t.Fatalf("receipt rejection reason = %q", receipt[0].Rejected[0].Reason)
	}
}

// TestInvalidJournalIsRejectedAndTheRestPromoted covers AC 11 and the D55 call
// site: a manifest naming an invalid run_db.sqlite has that entry rejected as
// rundb_invalid and every other entry promoted, and the journal never reaches
// the run dir.
func TestInvalidJournalIsRejectedAndTheRestPromoted(t *testing.T) {
	validated := []string{}
	r := newRig(t, func(c *Config) {
		c.Validator = ValidatorFunc(func(path string) (JournalVerdict, string) {
			validated = append(validated, path)
			return JournalInvalid, ReasonRunDBInvalid
		})
	})
	journal := []byte("not a journal at all")
	ckpt := bytes.Repeat([]byte("weights"), 16)
	metrics := []byte(`{"iter":2}` + "\n")
	journalDigest := r.putBlob(t, journal)
	r.putBlob(t, ckpt)
	r.putBlob(t, metrics)

	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{
			entry(RunDBPath, journal, 31),
			entry("snapshots/a.pt", ckpt, 32),
			entry("metrics.jsonl", metrics, 33),
		},
	})
	if got := rejectionReason(resp, RunDBPath); got != ReasonRunDBInvalid {
		t.Fatalf("rejection reason = %q, want %q", got, ReasonRunDBInvalid)
	}
	if !sameStrings(resp.Promoted, []string{"snapshots/a.pt", "metrics.jsonl"}) {
		t.Fatalf("promoted = %v, want the two non-journal entries", resp.Promoted)
	}
	if _, err := os.Stat(filepath.Join(r.runDir, RunDBPath)); !os.IsNotExist(err) {
		t.Fatalf("an invalid journal was promoted: %v", err)
	}
	for _, e := range r.head(t).Folded.Entries {
		if e.Path == RunDBPath {
			t.Fatal("an invalid journal was folded into the head")
		}
	}
	wantPath := blobPath(filepath.Join(r.quarDir, "node-a", "job-a", "lease-1"), journalDigest)
	if len(validated) != 1 || validated[0] != wantPath {
		t.Fatalf("validator saw %v, want exactly the verified blob %s", validated, wantPath)
	}

	// Three consecutive journal rejections degrade the lease (D55).
	for seq := int64(2); seq <= 3; seq++ {
		head := r.head(t)
		resp = r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: seq, Parent: head.Digest,
			Entries: []Entry{entry(RunDBPath, journal, 30+seq)},
		})
	}
	if !resp.Degraded || !r.store.Stats(r.lease).Degraded {
		t.Fatal("three consecutive journal rejections did not degrade the lease")
	}
}

// TestMissingBlobChangesNothing covers D51 step 3: a digest outside the
// provable set is listed and the commit is a no-op, so upload and commit stay a
// clean two-phase.
func TestMissingBlobChangesNothing(t *testing.T) {
	r := newRig(t, nil)
	present := []byte("uploaded")
	r.putBlob(t, present)
	absent := bytes.Repeat([]byte("never uploaded"), 3)

	before := treeSnapshot(t, r.runsDir)
	_, err := r.commit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{
			entry("metrics.jsonl", present, 41),
			entry("snapshots/a.pt", absent, 42),
		},
	})
	var missing *BlobsMissingError
	if !errors.As(err, &missing) {
		t.Fatalf("want BlobsMissingError, got %v", err)
	}
	if !sameStrings(missing.Missing, []string{sha256hex(absent)}) {
		t.Fatalf("missing = %v, want only the absent digest", missing.Missing)
	}
	if !sameStrings(before, treeSnapshot(t, r.runsDir)) {
		t.Fatal("a blobs_missing commit changed the runs tree")
	}
	if r.head(t).Folded.Seq != 0 {
		t.Fatal("a blobs_missing commit advanced the head")
	}
}

// TestGrantSetEntryIsCopiedAndVerified covers the D51 step 5 grant-set case: a
// seeded file the node reports unchanged is copied from the coordinator's own
// recorded source with its digest verified on the way in.
func TestGrantSetEntryIsCopiedAndVerified(t *testing.T) {
	r := newRig(t, nil)
	seedData := bytes.Repeat([]byte("seeded checkpoint"), 8)
	seedPath := filepath.Join(t.TempDir(), "prior.pt")
	if err := os.WriteFile(seedPath, seedData, 0o644); err != nil {
		t.Fatalf("write seed: %v", err)
	}
	digest := sha256hex(seedData)
	r.lease.Grants = map[string]Grant{digest: {Digest: digest, Size: int64(len(seedData)), SourcePath: seedPath}}

	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{entry("snapshots/prior.pt", seedData, 51)},
	})
	if !sameStrings(resp.Promoted, []string{"snapshots/prior.pt"}) {
		t.Fatalf("promoted = %v", resp.Promoted)
	}
	promoted := filepath.Join(r.runDir, "snapshots/prior.pt")
	if got := mustRead(t, promoted); !bytes.Equal(got, seedData) {
		t.Fatal("the promoted seed differs from its source")
	}
	src, err := os.Stat(seedPath)
	if err != nil {
		t.Fatalf("stat seed: %v", err)
	}
	dst, err := os.Stat(promoted)
	if err != nil {
		t.Fatalf("stat promoted: %v", err)
	}
	if os.SameFile(src, dst) {
		t.Fatal("a grant-set entry was linked to the seed rather than copied")
	}

	// A source that no longer hashes to its grant is reported as missing. The
	// tampered bytes keep the recorded length, so the mismatch can only be
	// caught by the digest verified during the copy.
	tampered := append([]byte(nil), seedData...)
	tampered[0] ^= 0xff
	if err := os.WriteFile(seedPath, tampered, 0o644); err != nil {
		t.Fatalf("rewrite seed: %v", err)
	}
	head := r.head(t)
	_, err = r.commit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 2, Parent: head.Digest,
		Entries: []Entry{entry("snapshots/again.pt", seedData, 52)},
	})
	var missing *BlobsMissingError
	if !errors.As(err, &missing) {
		t.Fatalf("want BlobsMissingError for a tampered grant source, got %v", err)
	}
}

// TestPromotedTreeHoldsNoNodeNamedSymlinkOrDirectory asserts the type rule of
// D52 over a real promotion: everything the node named is a regular file, and
// the staging names never survive a commit.
func TestPromotedTreeHoldsNoNodeNamedSymlinkOrDirectory(t *testing.T) {
	r := newRig(t, nil)
	runThreeCommits(t, r)

	err := filepath.Walk(r.runDir, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if fi.Mode()&os.ModeSymlink != 0 {
			t.Errorf("%s is a symlink in a promoted tree", p)
		}
		if !fi.IsDir() && !fi.Mode().IsRegular() {
			t.Errorf("%s is neither a directory nor a regular file", p)
		}
		if !fi.IsDir() && filepath.Ext(p) == TmpSuffix {
			t.Errorf("%s is a leftover staging file", p)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
}

// TestUnrecognizedEntriesAreAcceptedAndCounted covers the D52 rule that an
// entry outside the known v1.0 layout is promoted and counted rather than
// blocked, so a new artifact is visible without a code change.
func TestUnrecognizedEntriesAreAcceptedAndCounted(t *testing.T) {
	r := newRig(t, nil)
	novel := []byte("a new artifact kind")
	r.putBlob(t, novel)
	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
		Entries: []Entry{entry("evaluations/iter_5/summary.json", novel, 61)},
	})
	if !sameStrings(resp.Promoted, []string{"evaluations/iter_5/summary.json"}) {
		t.Fatalf("promoted = %v", resp.Promoted)
	}
	if !sameStrings(resp.Unrecognized, []string{"evaluations/iter_5/summary.json"}) {
		t.Fatalf("unrecognized = %v, want the novel path", resp.Unrecognized)
	}
}
