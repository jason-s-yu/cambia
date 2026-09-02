package quarantine

import (
	"errors"
	"fmt"
	"time"
)

// The store returns typed errors and never a status code: the route layer
// (W2-T10) owns the mapping below, so this package stays free of net/http.
//
//	ErrInvalidDigest      422 invalid_digest
//	ErrHashMismatch       422 hash_mismatch
//	InvalidManifestError  422 invalid_manifest
//	OffsetMismatchError   409 offset_mismatch      (X-Nashnet-Offset: Offset)
//	FenceError            409 manifest_out_of_order, or lease_superseded on a stale epoch
//	BlobsMissingError     409 blobs_missing
//	FileTooLargeError     413 file_too_large
//	LeaseBudgetError      413 lease_bytes_exceeded
//	StoreFullError        507 store_full           (Retry-After: RetryAfter)
//	InvalidRangeError     400 invalid_range
var (
	// ErrInvalidDigest names a blob digest that is not exactly 64 lowercase
	// hex characters. It is returned before any filesystem call (D49).
	ErrInvalidDigest = errors.New("invalid_digest")

	// ErrHashMismatch names a completed part whose content hash does not equal
	// the digest that named it. The part is destroyed (D50 step 3).
	ErrHashMismatch = errors.New("hash_mismatch")
)

// OffsetMismatchError reports the coordinator's true part offset so the node
// can seek instead of guessing (D50 step 3).
type OffsetMismatchError struct {
	Offset int64
}

func (e *OffsetMismatchError) Error() string {
	return fmt.Sprintf("offset_mismatch: coordinator offset is %d", e.Offset)
}

// InvalidRangeError names a Content-Range the coordinator cannot act on: a
// non-positive total, a negative start, an end before its start, an end at or
// past the total, or a body shorter than the range it declared.
type InvalidRangeError struct {
	Detail string
}

func (e *InvalidRangeError) Error() string { return "invalid_range: " + e.Detail }

// FileTooLargeError names an upload past the per-file cap of D56.
type FileTooLargeError struct {
	Size int64
	Cap  int64
}

func (e *FileTooLargeError) Error() string {
	return fmt.Sprintf("file_too_large: %d bytes over the %d byte cap", e.Size, e.Cap)
}

// LeaseBudgetError names an upload that would take a lease past its byte
// budget (D56). It is a quota event, never a job failure.
type LeaseBudgetError struct {
	Used      int64
	Requested int64
	Cap       int64
}

func (e *LeaseBudgetError) Error() string {
	return fmt.Sprintf("lease_bytes_exceeded: %d used plus %d requested over the %d byte budget", e.Used, e.Requested, e.Cap)
}

// StoreFullError names a refusal at the store watermark: the coordinator's own
// free-disk floor plus its margin (D56). No bytes are written, so a node can
// slow the coordinator but cannot fill it.
type StoreFullError struct {
	FreeGB      float64
	WatermarkGB float64
	RetryAfter  time.Duration
}

func (e *StoreFullError) Error() string {
	return fmt.Sprintf("store_full: %.2f GB free is under the %.2f GB watermark", e.FreeGB, e.WatermarkGB)
}

// Fence reasons distinguish the four fence failures of D51 step 1. The route
// maps FenceStaleEpoch to 409 lease_superseded and the rest to
// 409 manifest_out_of_order, both carrying the coordinator's current
// {seq, digest} so the node re-diffs against the head it actually has.
const (
	FenceStaleEpoch   = "stale_epoch"
	FenceSeq          = "seq"
	FenceParent       = "parent"
	FenceBodyMismatch = "body_mismatch"
)

// FenceError reports a fast-forward fence failure and the coordinator's head.
type FenceError struct {
	Reason string
	Seq    int64
	Digest string
}

func (e *FenceError) Error() string {
	return fmt.Sprintf("manifest_out_of_order (%s): coordinator head is seq %d digest %q", e.Reason, e.Seq, e.Digest)
}

// BlobsMissingError lists digests the commit named that are outside the lease's
// provable set (D51 step 3). The commit changes nothing, so upload and commit
// stay a clean two-phase with no ordering requirement on the node.
type BlobsMissingError struct {
	Missing []string
}

func (e *BlobsMissingError) Error() string {
	return fmt.Sprintf("blobs_missing: %v", e.Missing)
}

// InvalidManifestError names a malformed commit body, the only per-body
// rejection besides a fence failure (D51 step 2).
type InvalidManifestError struct {
	Detail string
}

func (e *InvalidManifestError) Error() string { return "invalid_manifest: " + e.Detail }
