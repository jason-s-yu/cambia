package quarantine

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRejectionListTable covers the path half of AC 6: every item of the D52
// rejection list and every per-path cap has a row with its reason code, and the
// known v1.0 layout is accepted. The run dir carries a planted symlink so the
// resolved-containment row is reachable without a lexical escape.
func TestRejectionListTable(t *testing.T) {
	r := newRig(t, nil)
	if err := os.MkdirAll(r.runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	if err := os.Symlink(t.TempDir(), filepath.Join(r.runDir, "planted")); err != nil {
		t.Fatalf("plant symlink: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(r.runDir, "occupied"), 0o755); err != nil {
		t.Fatalf("plant directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(r.runDir, "held"), []byte("a file, not a directory"), 0o644); err != nil {
		t.Fatalf("plant file: %v", err)
	}
	lim := r.store.Limits()

	cases := []struct {
		name   string
		path   string
		reason string
	}{
		{"empty", "", ReasonPathEmpty},
		{"absolute", "/etc/passwd", ReasonPathAbsolute},
		{"parent traversal", "../outside.pt", ReasonPathTraversal},
		{"interior traversal", "snapshots/../../outside.pt", ReasonPathTraversal},
		{"backslash", `snapshots\a.pt`, ReasonPathBackslash},
		{"nul byte", "snapshots/a\x00.pt", ReasonPathNUL},
		{"invalid utf8", "snapshots/" + string([]byte{0xff, 0xfe}) + ".pt", ReasonPathNotUTF8},
		{"trailing slash", "snapshots/", ReasonPathDirectory},
		{"empty segment", "snapshots//a.pt", ReasonPathDirectory},
		{"reserved process.json", "process.json", ReasonPathReserved},
		{"reserved jobspec.json", "jobspec.json", ReasonPathReserved},
		{"reserved lease.json", "lease.json", ReasonPathReserved},
		{"reserved env.json", "env.json", ReasonPathReserved},
		{"reserved nashnet dir", ".nashnet/current.json", ReasonPathReserved},
		{"reserved nashnet tmp dir", ".nashnet-tmp/x.pt", ReasonPathReserved},
		{"reserved nashnet tmp suffix", "snapshots/a.pt.nashnet-tmp", ReasonPathReserved},
		{"reserved logs subtree", "logs/training.log", ReasonPathReserved},
		{"reserved reservoir subtree", "reservoir/meta.json", ReasonPathReserved},
		{"reserved tmp suffix", "metrics.jsonl.tmp", ReasonPathReserved},
		{"reserved wal sibling", "run_db.sqlite-wal", ReasonPathReserved},
		{"reserved shm sibling", "run_db.sqlite-shm", ReasonPathReserved},
		{"reserved journal sibling", "run_db.sqlite-journal", ReasonPathReserved},
		{"segment cap", strings.Repeat("s", lim.MaxSegmentBytes+1) + ".pt", ReasonSegmentTooLong},
		{"path cap", strings.TrimSuffix(strings.Repeat(strings.Repeat("p", 250)+"/", 5), "/"), ReasonPathTooLong},
		{"depth cap", strings.TrimSuffix(strings.Repeat("d/", lim.MaxDepth+1), "/"), ReasonPathTooDeep},
		{"escapes through a planted symlink", "planted/stolen.pt", ReasonPathEscapes},
		{"target is an existing directory", "occupied", ReasonPathDirectory},
		{"parent is an existing file", "held/a.pt", ReasonPathParentNotDir},

		{"accepted config", "config.yaml", ""},
		{"accepted node env", "env.node.json", ""},
		{"accepted metrics", "metrics.jsonl", ""},
		{"accepted run meta", "run_meta.json", ""},
		{"accepted eval summary", "eval_summary.jsonl", ""},
		{"accepted resume state", "resume_state.json", ""},
		{"accepted snapshot", "snapshots/x.pt", ""},
		{"accepted journal", "run_db.sqlite", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateRelPath(r.runDir, tc.path, lim); got != tc.reason {
				t.Fatalf("validateRelPath(%q) = %q, want %q", tc.path, got, tc.reason)
			}
		})
	}
}

// TestSymlinkPlantedInRunDirIsRejected covers AC 7 on the commit path: a link
// planted inside the run dir cannot launder a write outside it, in either of
// the two shapes (the link as the entry, and the link as a path prefix).
func TestSymlinkPlantedInRunDirIsRejected(t *testing.T) {
	r := newRig(t, nil)
	outside := t.TempDir()
	if err := os.MkdirAll(r.runDir, 0o755); err != nil {
		t.Fatalf("mkdir run dir: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(r.runDir, "escape")); err != nil {
		t.Fatalf("plant directory symlink: %v", err)
	}
	if err := os.Symlink(filepath.Join(outside, "target.pt"), filepath.Join(r.runDir, "direct.pt")); err != nil {
		t.Fatalf("plant file symlink: %v", err)
	}

	good := []byte("legitimate artifact")
	digest := r.putBlob(t, good)
	resp := r.mustCommit(t, CommitRequest{
		ManifestVersion: ManifestVersion,
		LeaseEpoch:      1,
		Seq:             1,
		Entries: []Entry{
			{Digest: digest, MTime: 1, Path: "config.yaml", Size: int64(len(good))},
			{Digest: digest, MTime: 1, Path: "escape/stolen.pt", Size: int64(len(good))},
			{Digest: digest, MTime: 1, Path: "direct.pt", Size: int64(len(good))},
		},
	})
	for _, p := range []string{"escape/stolen.pt", "direct.pt"} {
		if got := rejectionReason(resp, p); got != ReasonPathEscapes {
			t.Fatalf("rejection for %q = %q, want %q", p, got, ReasonPathEscapes)
		}
	}
	if !sameStrings(resp.Promoted, []string{"config.yaml"}) {
		t.Fatalf("promoted = %v, want only config.yaml", resp.Promoted)
	}
	if _, err := os.Stat(filepath.Join(outside, "stolen.pt")); !os.IsNotExist(err) {
		t.Fatalf("a write landed outside the run dir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outside, "target.pt")); !os.IsNotExist(err) {
		t.Fatalf("a write followed a planted file symlink: %v", err)
	}
}

// TestEntryRejectionReasonsTable covers the entry half of AC 6: the reasons a
// path shape cannot produce, each asserted through a real commit.
func TestEntryRejectionReasonsTable(t *testing.T) {
	t.Run("invalid_digest", func(t *testing.T) {
		r := newRig(t, nil)
		resp := r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
			Entries: []Entry{{Digest: strings.Repeat("z", 64), MTime: 1, Path: "metrics.jsonl", Size: 4}},
		})
		if got := rejectionReason(resp, "metrics.jsonl"); got != ReasonInvalidDigest {
			t.Fatalf("reason = %q, want %q", got, ReasonInvalidDigest)
		}
	})

	t.Run("size_mismatch", func(t *testing.T) {
		r := newRig(t, nil)
		data := []byte("ten bytes!")
		digest := r.putBlob(t, data)
		resp := r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
			Entries: []Entry{{Digest: digest, MTime: 1, Path: "metrics.jsonl", Size: 99}},
		})
		if got := rejectionReason(resp, "metrics.jsonl"); got != ReasonSizeMismatch {
			t.Fatalf("reason = %q, want %q", got, ReasonSizeMismatch)
		}
	})

	t.Run("file_too_large", func(t *testing.T) {
		r := newRig(t, func(c *Config) { c.Limits = Limits{MaxFileBytes: 8} })
		resp := r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
			Entries: []Entry{{Digest: strings.Repeat("a", 64), MTime: 1, Path: "snapshots/big.pt", Size: 4096}},
		})
		if got := rejectionReason(resp, "snapshots/big.pt"); got != ReasonFileTooLarge {
			t.Fatalf("reason = %q, want %q", got, ReasonFileTooLarge)
		}
	})

	t.Run("lease_bytes_exceeded", func(t *testing.T) {
		r := newRig(t, func(c *Config) { c.Limits = Limits{MaxFileBytes: 4096, MaxLeaseBytes: 16} })
		resp := r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
			Entries: []Entry{{Digest: strings.Repeat("a", 64), MTime: 1, Path: "snapshots/big.pt", Size: 4096}},
		})
		if got := rejectionReason(resp, "snapshots/big.pt"); got != ReasonLeaseBytes {
			t.Fatalf("reason = %q, want %q", got, ReasonLeaseBytes)
		}
	})

	t.Run("duplicate_path", func(t *testing.T) {
		r := newRig(t, nil)
		data := []byte("one artifact")
		digest := r.putBlob(t, data)
		e := Entry{Digest: digest, MTime: 1, Path: "metrics.jsonl", Size: int64(len(data))}
		resp := r.mustCommit(t, CommitRequest{
			ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1,
			Entries: []Entry{e, e},
		})
		if got := rejectionReason(resp, "metrics.jsonl"); got != ReasonDuplicatePath {
			t.Fatalf("reason = %q, want %q", got, ReasonDuplicatePath)
		}
		if !sameStrings(resp.Promoted, []string{"metrics.jsonl"}) {
			t.Fatalf("promoted = %v, want the first copy of the path", resp.Promoted)
		}
	})
}

// TestManifestBodyCapsAreFatal covers the two caps that reject the whole body
// rather than an entry (D51 step 2, D52).
func TestManifestBodyCapsAreFatal(t *testing.T) {
	r := newRig(t, nil)
	entries := make([]Entry, r.store.Limits().MaxManifestEntries+1)
	for i := range entries {
		entries[i] = Entry{Digest: strings.Repeat("a", 64), MTime: 1, Path: "snapshots/x.pt", Size: 1}
	}
	_, err := r.store.Commit(r.lease, CommitRequest{
		ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1, Entries: entries,
	}, []byte("{}"))
	var bad *InvalidManifestError
	if !errors.As(err, &bad) || bad.Detail != "too_many_entries" {
		t.Fatalf("want invalid_manifest too_many_entries, got %v", err)
	}

	_, err = r.store.Commit(r.lease, CommitRequest{ManifestVersion: 2, LeaseEpoch: 1, Seq: 1}, []byte("{}"))
	if !errors.As(err, &bad) || bad.Detail != "manifest_version" {
		t.Fatalf("want invalid_manifest manifest_version, got %v", err)
	}

	oversized := make([]byte, r.store.Limits().MaxManifestBytes+1)
	_, err = r.store.Commit(r.lease, CommitRequest{ManifestVersion: ManifestVersion, LeaseEpoch: 1, Seq: 1}, oversized)
	if !errors.As(err, &bad) || bad.Detail != "body_too_large" {
		t.Fatalf("want invalid_manifest body_too_large, got %v", err)
	}
}

// TestCanonicalFormIsSortedAndStable asserts the folded manifest hashes over a
// canonical document: keys in lexicographic order, entries sorted by path, no
// insignificant whitespace, and the same bytes whatever order the node offered
// its entries in.
func TestCanonicalFormIsSortedAndStable(t *testing.T) {
	folded := Folded{
		Entries: []Entry{
			{Digest: strings.Repeat("b", 64), MTime: 2, Path: "snapshots/a.pt", Size: 2},
			{Digest: strings.Repeat("a", 64), MTime: 1, Path: "config.yaml", Size: 1},
		},
		JobID: "job-a", LeaseEpoch: 1, LeaseID: "lease-1", ManifestVersion: ManifestVersion, Seq: 1,
	}
	shuffled := folded
	shuffled.Entries = []Entry{folded.Entries[1], folded.Entries[0]}

	sortEntries := func(f Folded) Folded {
		req := CommitRequest{Seq: f.Seq, Parent: f.Parent, ManifestVersion: ManifestVersion}
		return fold(Folded{Entries: []Entry{}}, Lease{JobID: f.JobID, LeaseID: f.LeaseID, Epoch: f.LeaseEpoch}, req, f.Entries, nil)
	}
	a, err := canonicalJSON(sortEntries(folded))
	if err != nil {
		t.Fatalf("canonicalJSON: %v", err)
	}
	b, err := canonicalJSON(sortEntries(shuffled))
	if err != nil {
		t.Fatalf("canonicalJSON: %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("canonical form depends on offer order:\n%s\n%s", a, b)
	}
	if strings.Contains(string(a), "\n") || strings.Contains(string(a), ": ") {
		t.Fatalf("canonical form carries insignificant whitespace: %s", a)
	}
	var keys []string
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(a, &probe); err != nil {
		t.Fatalf("unmarshal canonical form: %v", err)
	}
	for _, want := range []string{"entries", "final", "job_id", "lease_epoch", "lease_id", "manifest_version", "parent", "seq"} {
		if _, ok := probe[want]; !ok {
			t.Fatalf("canonical form is missing key %q: %s", want, a)
		}
		keys = append(keys, want)
	}
	last := -1
	for _, k := range keys {
		i := strings.Index(string(a), `"`+k+`":`)
		if i < last {
			t.Fatalf("key %q is out of lexicographic order: %s", k, a)
		}
		last = i
	}
}
