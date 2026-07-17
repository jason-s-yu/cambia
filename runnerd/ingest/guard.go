package ingest

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Sentinel errors returned by the ingest package so the daemon can map them to
// control-plane status codes via errors.Is.
var (
	// ErrInvalidCommit is returned when a spec commit is not a 40-hex sha.
	ErrInvalidCommit = errors.New("invalid commit")
	// ErrReceiptMismatch is returned when the job ref does not resolve to the
	// spec commit or the object is missing (design 3.1).
	ErrReceiptMismatch = errors.New("mirror receipt mismatch")
	// ErrSignatureVerification is returned when RequireSignedCommits is on and
	// the pinned commit fails ssh signature verification: unsigned, signed by a
	// key absent from allowed_signers, a bad signature, or an empty/missing
	// signers file (enforcement fails closed). Default-off, so it never fires
	// unless a runner sets RUNNERD_REQUIRE_SIGNED_COMMITS (cambia-550, W1).
	ErrSignatureVerification = errors.New("commit signature verification failed")
	// ErrPathEscape is returned when a spec path is absolute, contains "..", or
	// resolves outside its allowed root (design 5.4).
	ErrPathEscape = errors.New("path escapes containment")
	// ErrOwnedOverride is returned when a submitter override targets a
	// harness-owned config key (design 5.5).
	ErrOwnedOverride = errors.New("override targets a harness-owned key")
	// ErrStaleLock is returned when `uv lock --check` rejects the worktree lock
	// before a venv build (design 3.3).
	ErrStaleLock = errors.New("stale uv.lock")
)

// commitRe matches a full 40-hex git object id.
var commitRe = regexp.MustCompile(`^[0-9a-f]{40}$`)

// validateJobID reuses the daemon's run-name allowlist so job ids and run names
// share one path-traversal guard (job_id == validated run name, design 2.4).
func validateJobID(jobID string) error {
	return procmgr.ValidateName(jobID)
}

// validateCommit enforces the 40-hex commit contract.
func validateCommit(commit string) error {
	if !commitRe.MatchString(commit) {
		return fmt.Errorf("%w: %q is not a 40-hex sha", ErrInvalidCommit, commit)
	}
	return nil
}

// guardRelPath resolves rel against root after rejecting absolute paths and any
// ".." segment, then verifies the cleaned result is contained within root. It
// returns the absolute in-root path. This is the design 5.4 guard applied to the
// config path (inside the worktree) and to checkpoint paths (inside their staged
// run dir); the callable is exported-shaped for the daemon's checkpoint checks.
func guardRelPath(root, rel string) (string, error) {
	if rel == "" {
		return "", fmt.Errorf("%w: empty path", ErrPathEscape)
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("%w: %q is absolute", ErrPathEscape, rel)
	}
	// Reject any parent segment before normalization so a crafted "a/../../b"
	// cannot be laundered by Clean.
	for _, seg := range strings.Split(filepath.ToSlash(rel), "/") {
		if seg == ".." {
			return "", fmt.Errorf("%w: %q contains a .. segment", ErrPathEscape, rel)
		}
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	joined := filepath.Clean(filepath.Join(absRoot, rel))
	rootPrefix := absRoot + string(filepath.Separator)
	if joined != absRoot && !strings.HasPrefix(joined, rootPrefix) {
		return "", fmt.Errorf("%w: %q resolves outside %q", ErrPathEscape, rel, absRoot)
	}
	// Symlink containment: the lexical check above cannot see a symlink planted
	// inside the (submitter-controlled) worktree that points outside root.
	// Resolve symlinks on both sides and re-check against the resolved root so
	// such a link cannot launder an escape past the lexical guard (design 5.4).
	rRoot, err := pathguard.ResolveForContainment(absRoot)
	if err != nil {
		return "", fmt.Errorf("%w: resolve root %q: %v", ErrPathEscape, absRoot, err)
	}
	rJoined, err := pathguard.ResolveForContainment(joined)
	if err != nil {
		return "", fmt.Errorf("%w: resolve %q: %v", ErrPathEscape, joined, err)
	}
	if rJoined != rRoot && !strings.HasPrefix(rJoined, rRoot+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: %q resolves to %q, outside %q", ErrPathEscape, rel, rJoined, rRoot)
	}
	return joined, nil
}
