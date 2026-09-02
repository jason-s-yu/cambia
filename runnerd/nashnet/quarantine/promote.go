package quarantine

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/pathguard"
)

// errGrantDigestMismatch is raised when a grant-set source no longer hashes to
// the digest the grant recorded. D51 step 5 lists that case under
// blobs_missing, so the commit is abandoned and the node re-diffs.
var errGrantDigestMismatch = errors.New("grant source digest mismatch")

// source is where a promoted entry's bytes come from: a verified blob of this
// lease, or a grant-set file the coordinator itself holds (D53).
type source struct {
	path      string
	size      int64
	fromGrant bool
}

// Commit runs the fast-forward manifest commit and the promotion transaction of
// D51 under the job's promote mutex, single writer.
//
// body is the raw request body, hashed for the idempotent-replay rule; a nil
// body falls back to the canonical form of req, for a caller that did not
// retain the bytes.
//
// The steps are D51's: fence, validate every entry and delete against the
// rejection list, resolve digests against the provable set, content-validate
// run_db.sqlite through the journal validator, fold and hash, materialize with
// run_db.sqlite last, record the head and the receipt, answer. Rejected entries
// are listed and the batch proceeds with the rest; only a malformed body or a
// fence failure rejects the whole commit.
func (s *Store) Commit(l Lease, req CommitRequest, body []byte) (CommitResponse, error) {
	leaseDir, err := s.leaseDir(l)
	if err != nil {
		return CommitResponse{}, err
	}
	runDir, err := s.RunDir(l.JobID)
	if err != nil {
		return CommitResponse{}, err
	}
	if body == nil {
		if body, err = canonicalJSON(req); err != nil {
			return CommitResponse{}, err
		}
	}
	if int64(len(body)) > s.lim.MaxManifestBytes {
		return CommitResponse{}, &InvalidManifestError{Detail: "body_too_large"}
	}
	bodyDigest := digestOf(body)

	lock := s.jobLock(l.JobID)
	lock.Lock()
	defer lock.Unlock()

	head, err := readHead(runDir)
	if err != nil {
		return CommitResponse{}, err
	}

	// Step 1, fence. A re-post of the accepted seq with the recorded body
	// returns the recorded response unchanged, so a lost response is safe to
	// retry; the same seq with a different body is a fence failure. Artifact
	// state is therefore monotone: a replayed or reordered batch after a
	// reconnect is a no-op, never a rollback.
	if head.Folded.Seq > 0 && req.Seq == head.Folded.Seq {
		if bodyDigest == head.LastRequestSHA256 && head.LastResponse != nil {
			return *head.LastResponse, nil
		}
		return CommitResponse{}, &FenceError{Reason: FenceBodyMismatch, Seq: head.Folded.Seq, Digest: head.Digest}
	}
	if req.ManifestVersion != ManifestVersion {
		return CommitResponse{}, &InvalidManifestError{Detail: "manifest_version"}
	}
	if req.LeaseEpoch != l.Epoch {
		return CommitResponse{}, &FenceError{Reason: FenceStaleEpoch, Seq: head.Folded.Seq, Digest: head.Digest}
	}
	if req.Seq != head.Folded.Seq+1 {
		return CommitResponse{}, &FenceError{Reason: FenceSeq, Seq: head.Folded.Seq, Digest: head.Digest}
	}
	if req.Parent != head.Digest {
		return CommitResponse{}, &FenceError{Reason: FenceParent, Seq: head.Folded.Seq, Digest: head.Digest}
	}
	if len(req.Entries) > s.lim.MaxManifestEntries || len(req.Deletes) > s.lim.MaxManifestEntries {
		return CommitResponse{}, &InvalidManifestError{Detail: "too_many_entries"}
	}

	// Step 2, validate every entry and delete against D52.
	accepted, deletes, rejected, unrecognized := s.validateBatch(l, head, runDir, req)

	// Step 3, resolve digests against the provable set. Any absent digest is
	// listed and the commit changes nothing, so upload and commit are a clean
	// two-phase with no ordering requirement on the node.
	sources := make(map[string]source, len(accepted))
	var missing []string
	seenMissing := map[string]bool{}
	for _, e := range accepted {
		src, ok := s.resolveSource(l, leaseDir, e.Digest)
		if !ok {
			if !seenMissing[e.Digest] {
				seenMissing[e.Digest] = true
				missing = append(missing, e.Digest)
			}
			continue
		}
		sources[e.Path] = src
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return CommitResponse{}, &BlobsMissingError{Missing: missing}
	}

	// The declared size must match the bytes the coordinator actually holds,
	// so a manifest cannot describe a file as something other than what was
	// proved on upload.
	kept := accepted[:0]
	for _, e := range accepted {
		if src := sources[e.Path]; src.size != e.Size {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonSizeMismatch})
			continue
		}
		kept = append(kept, e)
	}
	accepted = kept

	// Step 4, content-validate run_db.sqlite (D55) on the verified blob, before
	// anything is folded or materialized. A rejection is per entry: the rest of
	// the batch proceeds.
	rundbRejected := false
	kept = accepted[:0]
	for _, e := range accepted {
		if e.Path != RunDBPath {
			kept = append(kept, e)
			continue
		}
		verdict, verr := s.validator.Validate(sources[e.Path].path, l.rundbName())
		if verr != nil {
			// The validator could not read a blob the coordinator just
			// verified: a coordinator-side fault, not a node rejection, so the
			// commit aborts without advancing the head rather than charging the
			// node a journal rejection.
			return CommitResponse{}, fmt.Errorf("validate %s: %w", RunDBPath, verr)
		}
		if verdict.Accepted {
			kept = append(kept, e)
			continue
		}
		// Every content-level verdict collapses to one per-entry reason (D51
		// step 4); the validator's own reason and detail ride along so the
		// commit response and the receipt name which check failed.
		detail := verdict.Reason
		if verdict.Detail != "" {
			detail += ": " + verdict.Detail
		}
		rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonRunDBInvalid, Detail: detail})
		rundbRejected = true
	}
	accepted = kept

	// Step 4 continued, fold and hash. The folded document is the full state at
	// this seq and its digest is the parent the next commit must carry.
	folded := fold(head.Folded, l, req, accepted, deletes)
	if len(folded.Entries) > s.lim.MaxManifestEntries {
		return CommitResponse{}, &InvalidManifestError{Detail: "too_many_files"}
	}
	canonical, err := canonicalJSON(folded)
	if err != nil {
		return CommitResponse{}, err
	}
	digest := digestOf(canonical)
	if err := ensureLeaseTree(leaseDir); err != nil {
		return CommitResponse{}, err
	}
	manifestFile := filepath.Join(leaseDir, dirManifests, fmt.Sprintf("%d-%s.json", req.Seq, digest))
	if err := os.WriteFile(manifestFile, canonical, 0o600); err != nil {
		return CommitResponse{}, err
	}

	// Step 5, materialize. Every other path first, deletes next, and
	// run_db.sqlite last (D57), so a journal row that names a checkpoint finds
	// its file already present. A materialize failure aborts before the head
	// advances, so the node re-posts the same seq and the fence passes again.
	// The run dir is created here rather than at construction: a fenced or
	// blob-missing commit must leave the filesystem exactly as it found it.
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return CommitResponse{}, err
	}
	promoted := []string{}
	var rundbEntry *Entry
	for i := range accepted {
		e := accepted[i]
		if e.Path == RunDBPath {
			rundbEntry = &accepted[i]
			continue
		}
		if err := s.materialize(runDir, e, sources[e.Path], req.Seq); err != nil {
			if errors.Is(err, errGrantDigestMismatch) {
				return CommitResponse{}, &BlobsMissingError{Missing: []string{e.Digest}}
			}
			return CommitResponse{}, err
		}
		promoted = append(promoted, e.Path)
	}
	deleted := []string{}
	for _, d := range deletes {
		target, err := pathguard.Resolve(runDir, d)
		if err != nil {
			rejected = append(rejected, Rejection{Path: d, Reason: ReasonPathEscapes})
			continue
		}
		if err := os.Remove(target); err != nil && !os.IsNotExist(err) {
			return CommitResponse{}, err
		}
		deleted = append(deleted, d)
	}
	if rundbEntry != nil {
		if err := s.materialize(runDir, *rundbEntry, sources[rundbEntry.Path], req.Seq); err != nil {
			if errors.Is(err, errGrantDigestMismatch) {
				return CommitResponse{}, &BlobsMissingError{Missing: []string{rundbEntry.Digest}}
			}
			return CommitResponse{}, err
		}
		promoted = append(promoted, rundbEntry.Path)
	}
	if err := fsyncDir(runDir); err != nil {
		return CommitResponse{}, err
	}

	// Step 6, record. The head is rewritten with the atomic pattern of
	// procmgr/state.go:74 and one receipt line records what the node offered
	// against what was promoted or rejected.
	s.mu.Lock()
	st := s.statsLocked(l)
	st.Rejections += len(rejected)
	if rundbRejected {
		st.RunDBRejections++
		st.ConsecutiveRunDBRejects++
	} else if rundbEntry != nil {
		st.ConsecutiveRunDBRejects = 0
	}
	s.markDegradedLocked(st)
	degraded := st.Degraded
	s.mu.Unlock()

	resp := CommitResponse{
		Seq:          req.Seq,
		Digest:       digest,
		Promoted:     promoted,
		Deleted:      deleted,
		Rejected:     rejected,
		Unrecognized: unrecognized,
		Degraded:     degraded,
	}
	newHead := Head{
		Digest:            digest,
		Folded:            folded,
		LastRequestSHA256: bodyDigest,
		LastResponse:      &resp,
		UpdatedAt:         s.now().UnixNano(),
	}
	if err := writeHead(runDir, newHead); err != nil {
		return CommitResponse{}, err
	}
	if err := s.appendReceipt(leaseDir, receiptLine(s.now(), req, resp)); err != nil {
		return CommitResponse{}, err
	}
	return resp, nil
}

// validateBatch applies the rejection list to every entry and delete and
// returns the accepted entries, the accepted deletes, the rejections, and the
// paths accepted but outside the known v1.0 layout.
func (s *Store) validateBatch(l Lease, head Head, runDir string, req CommitRequest) ([]Entry, []string, []Rejection, []string) {
	accepted := make([]Entry, 0, len(req.Entries))
	deletes := make([]string, 0, len(req.Deletes))
	rejected := []Rejection{}
	unrecognized := []string{}
	seen := map[string]bool{}

	budget := s.lim.MaxLeaseBytes
	if l.MaxBytes > 0 && l.MaxBytes < budget {
		budget = l.MaxBytes
	}
	foldedBytes := map[string]int64{}
	var total int64
	for _, e := range head.Folded.Entries {
		foldedBytes[e.Path] = e.Size
		total += e.Size
	}

	for _, d := range req.Deletes {
		if reason := validateRelPath(runDir, d, s.lim); reason != "" {
			rejected = append(rejected, Rejection{Path: d, Reason: reason})
			continue
		}
		if seen[d] {
			rejected = append(rejected, Rejection{Path: d, Reason: ReasonDuplicatePath})
			continue
		}
		seen[d] = true
		if prev, ok := foldedBytes[d]; ok {
			total -= prev
			delete(foldedBytes, d)
		}
		deletes = append(deletes, d)
	}

	for _, e := range req.Entries {
		if reason := validateRelPath(runDir, e.Path, s.lim); reason != "" {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: reason})
			continue
		}
		if seen[e.Path] {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonDuplicatePath})
			continue
		}
		if err := ValidateDigest(e.Digest); err != nil {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonInvalidDigest})
			continue
		}
		if e.Size < 0 || e.Size > s.lim.MaxFileBytes {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonFileTooLarge})
			continue
		}
		next := total - foldedBytes[e.Path] + e.Size
		if next > budget {
			rejected = append(rejected, Rejection{Path: e.Path, Reason: ReasonLeaseBytes})
			continue
		}
		total = next
		foldedBytes[e.Path] = e.Size
		seen[e.Path] = true
		accepted = append(accepted, e)
		if !recognized(e.Path) {
			unrecognized = append(unrecognized, e.Path)
		}
	}
	return accepted, deletes, rejected, unrecognized
}

// resolveSource finds a digest in the lease's provable set: its own verified
// blobs first, then the grant set.
func (s *Store) resolveSource(l Lease, leaseDir, digest string) (source, bool) {
	if fi, err := os.Stat(blobPath(leaseDir, digest)); err == nil && fi.Mode().IsRegular() {
		return source{path: blobPath(leaseDir, digest), size: fi.Size()}, true
	}
	g, ok := l.Grants[digest]
	if !ok || g.SourcePath == "" {
		return source{}, false
	}
	fi, err := os.Stat(g.SourcePath)
	if err != nil || !fi.Mode().IsRegular() {
		return source{}, false
	}
	return source{path: g.SourcePath, size: fi.Size(), fromGrant: true}, true
}

// materialize promotes one entry: link plus rename in link mode, copy into
// .nashnet-tmp plus rename in copy mode, so a direct write to the destination
// path never happens. run_db.sqlite is always a 0644 copy rather than a link
// (D22), because the rundb-checkpoint route opens the promoted file read-write
// and must not write through to the blob. A grant-set source is copied with its
// digest verified during the copy.
func (s *Store) materialize(runDir string, e Entry, src source, seq int64) error {
	target, err := pathguard.Resolve(runDir, e.Path)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	mtime := time.Unix(0, e.MTime)

	byCopy := s.mode == ModeCopy || src.fromGrant || e.Path == RunDBPath
	if !byCopy {
		tmp := target + TmpSuffix
		_ = os.Remove(tmp)
		if err := s.link(src.path, tmp); err != nil {
			return err
		}
		if err := os.Chtimes(tmp, mtime, mtime); err != nil {
			_ = os.Remove(tmp)
			return err
		}
		if err := os.Rename(tmp, target); err != nil {
			_ = os.Remove(tmp)
			return err
		}
		s.materialized(e.Path)
		return nil
	}

	tmpDir := filepath.Dir(target)
	if s.mode == ModeCopy {
		tmpDir = filepath.Join(runDir, TmpDir)
		if err := os.MkdirAll(tmpDir, 0o755); err != nil {
			return err
		}
	}
	tmp := filepath.Join(tmpDir, fmt.Sprintf("%d-%s%s", seq, e.Digest, TmpSuffix))
	if err := copyFile(src.path, tmp, e.Digest, src.fromGrant); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Chtimes(tmp, mtime, mtime); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, target); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	s.materialized(e.Path)
	return nil
}

func (s *Store) materialized(rel string) {
	if s.onMat != nil {
		s.onMat(rel)
	}
}

// copyFile copies src to dst as a 0644 file and fsyncs it. When verify is set
// the copy is hashed and the result must equal digest, which is how a grant-set
// file is proved on its way into a run dir.
func copyFile(src, dst, digest string, verify bool) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	var w io.Writer = out
	h := sha256.New()
	if verify {
		w = io.MultiWriter(out, h)
	}
	if _, err := io.Copy(w, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	if verify && hex.EncodeToString(h.Sum(nil)) != digest {
		return errGrantDigestMismatch
	}
	return nil
}
