package quarantine

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"hash"
	"io"
	"os"
	"path/filepath"
)

// partHash is the running sha256 of a part file and the offset it covers. It is
// deliberately in-memory only: D50 rejects a persisted running-hash side-car,
// because it would have to stay in lockstep with the part file across a crash
// between two fsyncs. After a coordinator restart the entry is simply absent
// and the part is rehashed once from its own bytes, so the part size alone
// remains the resume offset.
type partHash struct {
	h      hash.Hash
	offset int64
}

// BlobStatus answers the resume probe of D50 step 2.
type BlobStatus struct {
	// Offset is the size of parts/<digest>.part, 0 if none, or the full size
	// when the blob is already verified.
	Offset int64
	// Verified reports that blobs/<digest> exists, so a repeated upload is
	// free.
	Verified bool
}

// ChunkResult is the answer to one chunk append.
type ChunkResult struct {
	CommittedOffset int64
	// Verified reports that this chunk completed the file, its running digest
	// matched the URL digest, and the part was renamed into blobs/.
	Verified bool
}

// BlobStatus reports the resume offset for a digest under a lease. The digest
// is validated before any filesystem call, so a malformed one never names a
// path.
func (s *Store) BlobStatus(l Lease, digest string) (BlobStatus, error) {
	if err := ValidateDigest(digest); err != nil {
		return BlobStatus{}, err
	}
	dir, err := s.leaseDir(l)
	if err != nil {
		return BlobStatus{}, err
	}
	if fi, err := os.Stat(blobPath(dir, digest)); err == nil {
		return BlobStatus{Offset: fi.Size(), Verified: true}, nil
	}
	if fi, err := os.Stat(partPath(dir, digest)); err == nil {
		return BlobStatus{Offset: fi.Size()}, nil
	}
	return BlobStatus{}, nil
}

// Probe answers POST /nashnet/leases/{lease}/blobs/probe (D50 step 1). have is
// computed over the lease's provable set only: its own blobs/ plus the grant
// set listed at claim. Everything else is want, whether or not the coordinator
// holds it under another lease, so a node learns nothing it did not already
// know. Order and duplicates of the request list are preserved in the answer's
// input order, deduplicated.
func (s *Store) Probe(l Lease, digests []string) (have, want []string, err error) {
	for _, d := range digests {
		if err := ValidateDigest(d); err != nil {
			return nil, nil, err
		}
	}
	dir, err := s.leaseDir(l)
	if err != nil {
		return nil, nil, err
	}
	have, want = []string{}, []string{}
	seen := make(map[string]bool, len(digests))
	for _, d := range digests {
		if seen[d] {
			continue
		}
		seen[d] = true
		if s.provable(l, dir, d) {
			have = append(have, d)
			continue
		}
		want = append(want, d)
	}
	return have, want, nil
}

// provable reports whether a digest is in the lease's provable set: a verified
// blob of its own, or a grant-set entry whose source the coordinator still
// holds.
func (s *Store) provable(l Lease, leaseDir, digest string) bool {
	if _, err := os.Stat(blobPath(leaseDir, digest)); err == nil {
		return true
	}
	g, ok := l.Grants[digest]
	if !ok || g.SourcePath == "" {
		return false
	}
	fi, err := os.Stat(g.SourcePath)
	return err == nil && fi.Mode().IsRegular()
}

// AppendChunk appends one chunk to parts/<digest>.part (D50 step 3).
//
// start and end are the inclusive byte range of the chunk and total is the full
// file size, as carried by Content-Range. Admission runs before the first byte
// is read: the per-file cap, the lease byte budget, and the store watermark.
// The per-node token bucket, the concurrent-upload ceiling, and the per-request
// chunk cap are the route layer's (D56), because they are per-identity facts
// the store does not hold.
//
// A start that does not equal the current part size is an OffsetMismatchError
// carrying the true offset. When the chunk completes the file the running
// digest must equal the URL digest, or the part is destroyed and the call is
// ErrHashMismatch with an abuse counter increment; on a match the part is
// renamed to blobs/<digest>. A repeated upload of a verified digest reads no
// bytes and is free.
func (s *Store) AppendChunk(l Lease, digest string, start, end, total int64, body io.Reader) (ChunkResult, error) {
	if err := ValidateDigest(digest); err != nil {
		return ChunkResult{}, err
	}
	dir, err := s.leaseDir(l)
	if err != nil {
		return ChunkResult{}, err
	}
	if total == 0 {
		// A zero-byte artifact has no chunk range to send, so the route maps a
		// bodyless PATCH to this call. The digest still has to be the one the
		// empty string hashes to, which is the same proof every other blob
		// carries.
		return s.createEmptyBlob(l, dir, digest)
	}
	switch {
	case total < 0:
		return ChunkResult{}, &InvalidRangeError{Detail: "total size must not be negative"}
	case start < 0 || end < start:
		return ChunkResult{}, &InvalidRangeError{Detail: "range start must be non-negative and not past its end"}
	case end >= total:
		return ChunkResult{}, &InvalidRangeError{Detail: "range end must be inside the declared total"}
	}
	if total > s.lim.MaxFileBytes {
		return ChunkResult{}, &FileTooLargeError{Size: total, Cap: s.lim.MaxFileBytes}
	}

	lock := s.uploadLock(l.key() + "/" + digest)
	lock.Lock()
	defer lock.Unlock()

	// A verified blob costs nothing: answer with its size and read no body.
	if fi, statErr := os.Stat(blobPath(dir, digest)); statErr == nil {
		return ChunkResult{CommittedOffset: fi.Size(), Verified: true}, nil
	}

	chunkLen := end - start + 1
	if err := s.admitBytes(l, dir, chunkLen); err != nil {
		return ChunkResult{}, err
	}
	if err := ensureLeaseTree(dir); err != nil {
		return ChunkResult{}, err
	}

	part := partPath(dir, digest)
	f, err := os.OpenFile(part, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return ChunkResult{}, err
	}
	closed := false
	defer func() {
		if !closed {
			_ = f.Close()
		}
	}()
	fi, err := f.Stat()
	if err != nil {
		return ChunkResult{}, err
	}
	size := fi.Size()
	if start != size {
		s.mu.Lock()
		s.statsLocked(l).OffsetMismatches++
		s.mu.Unlock()
		return ChunkResult{}, &OffsetMismatchError{Offset: size}
	}

	ph, err := s.runningHash(l, digest, part, size)
	if err != nil {
		return ChunkResult{}, err
	}

	if _, err := io.CopyN(io.MultiWriter(f, ph.h), body, chunkLen); err != nil {
		// The write and the running hash advanced together, so rewinding the
		// file to its pre-chunk size and dropping the cached hash leaves the
		// part exactly where the last completed chunk left it. The next append
		// rehashes from the part's own bytes.
		_ = f.Truncate(size)
		s.dropHash(l, digest)
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return ChunkResult{}, &InvalidRangeError{Detail: "body is shorter than the declared range"}
		}
		return ChunkResult{}, err
	}
	if err := f.Sync(); err != nil {
		s.dropHash(l, digest)
		return ChunkResult{}, err
	}
	ph.offset = size + chunkLen
	s.addUsed(l, chunkLen)

	if ph.offset != total {
		return ChunkResult{CommittedOffset: ph.offset, Verified: false}, nil
	}

	got := hex.EncodeToString(ph.h.Sum(nil))
	s.dropHash(l, digest)
	if got != digest {
		_ = f.Close()
		closed = true
		_ = os.Remove(part)
		s.addUsed(l, -total)
		s.mu.Lock()
		st := s.statsLocked(l)
		st.HashMismatches++
		s.markDegradedLocked(st)
		s.mu.Unlock()
		return ChunkResult{}, ErrHashMismatch
	}
	if err := f.Chmod(0o644); err != nil {
		return ChunkResult{}, err
	}
	if err := f.Close(); err != nil {
		return ChunkResult{}, err
	}
	closed = true
	if err := os.Rename(part, blobPath(dir, digest)); err != nil {
		return ChunkResult{}, err
	}
	if err := fsyncDir(filepath.Join(dir, dirBlobs)); err != nil {
		return ChunkResult{}, err
	}
	return ChunkResult{CommittedOffset: total, Verified: true}, nil
}

// emptyDigest is the sha256 of the empty string, the only digest a zero-byte
// upload may carry.
const emptyDigest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

// createEmptyBlob materializes the zero-byte blob after checking that the
// digest naming it is the one the empty string hashes to.
func (s *Store) createEmptyBlob(l Lease, leaseDir, digest string) (ChunkResult, error) {
	if digest != emptyDigest {
		s.mu.Lock()
		st := s.statsLocked(l)
		st.HashMismatches++
		s.markDegradedLocked(st)
		s.mu.Unlock()
		return ChunkResult{}, ErrHashMismatch
	}
	if err := ensureLeaseTree(leaseDir); err != nil {
		return ChunkResult{}, err
	}
	if err := os.WriteFile(blobPath(leaseDir, digest), nil, 0o644); err != nil {
		return ChunkResult{}, err
	}
	if err := fsyncDir(filepath.Join(leaseDir, dirBlobs)); err != nil {
		return ChunkResult{}, err
	}
	return ChunkResult{CommittedOffset: 0, Verified: true}, nil
}

// AbandonPart removes an in-flight part (D50 step 4).
func (s *Store) AbandonPart(l Lease, digest string) error {
	if err := ValidateDigest(digest); err != nil {
		return err
	}
	dir, err := s.leaseDir(l)
	if err != nil {
		return err
	}
	lock := s.uploadLock(l.key() + "/" + digest)
	lock.Lock()
	defer lock.Unlock()

	part := partPath(dir, digest)
	fi, statErr := os.Stat(part)
	if err := os.Remove(part); err != nil && !os.IsNotExist(err) {
		return err
	}
	if statErr == nil {
		s.addUsed(l, -fi.Size())
	}
	s.dropHash(l, digest)
	return nil
}

// admitBytes runs the pre-write admission of D50: the lease byte budget, then
// the store watermark. Nothing is written when either refuses.
func (s *Store) admitBytes(l Lease, leaseDir string, n int64) error {
	budget := s.lim.MaxLeaseBytes
	if l.MaxBytes > 0 && l.MaxBytes < budget {
		budget = l.MaxBytes
	}
	used, err := s.usedBytes(l, leaseDir)
	if err != nil {
		return err
	}
	if used+n > budget {
		return &LeaseBudgetError{Used: used, Requested: n, Cap: budget}
	}
	watermark := s.lim.MinFreeDiskGB + s.lim.WatermarkMarginGB
	if free := s.diskFree(s.root); free < watermark {
		return &StoreFullError{FreeGB: free, WatermarkGB: watermark, RetryAfter: s.lim.RetryAfter}
	}
	return nil
}

// usedBytes returns the bytes a lease holds in quarantine, counting parts and
// blobs. The walk runs once per lease and the total is then carried in memory;
// a coordinator restart recomputes it from the tree, which is the same source
// of truth the part size is.
func (s *Store) usedBytes(l Lease, leaseDir string) (int64, error) {
	k := l.key()
	s.mu.Lock()
	if n, ok := s.used[k]; ok {
		s.mu.Unlock()
		return n, nil
	}
	s.mu.Unlock()

	var total int64
	for _, sub := range []string{dirParts, dirBlobs} {
		entries, err := os.ReadDir(filepath.Join(leaseDir, sub))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return 0, err
		}
		for _, e := range entries {
			fi, err := e.Info()
			if err != nil || !fi.Mode().IsRegular() {
				continue
			}
			total += fi.Size()
		}
	}
	s.mu.Lock()
	s.used[k] = total
	s.mu.Unlock()
	return total, nil
}

func (s *Store) addUsed(l Lease, delta int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if n, ok := s.used[l.key()]; ok {
		n += delta
		if n < 0 {
			n = 0
		}
		s.used[l.key()] = n
	}
}

// runningHash returns the running sha256 covering the first size bytes of the
// part. A cached entry is used only when it covers exactly that many bytes;
// otherwise the part is rehashed once from its own bytes, which is what makes a
// coordinator restart mid-upload transparent to the node.
func (s *Store) runningHash(l Lease, digest, part string, size int64) (*partHash, error) {
	k := l.key() + "/" + digest
	s.mu.Lock()
	ph := s.parts[k]
	s.mu.Unlock()
	if ph != nil && ph.offset == size {
		return ph, nil
	}
	h := sha256.New()
	if size > 0 {
		f, err := os.Open(part)
		if err != nil {
			return nil, err
		}
		n, err := io.Copy(h, io.LimitReader(f, size))
		f.Close()
		if err != nil {
			return nil, err
		}
		if n != size {
			return nil, io.ErrUnexpectedEOF
		}
	}
	ph = &partHash{h: h, offset: size}
	s.mu.Lock()
	s.parts[k] = ph
	s.mu.Unlock()
	return ph, nil
}

func (s *Store) dropHash(l Lease, digest string) {
	s.mu.Lock()
	delete(s.parts, l.key()+"/"+digest)
	s.mu.Unlock()
}
