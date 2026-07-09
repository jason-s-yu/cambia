// Package pathguard implements the spec-field path guards of design 5.4. Job
// spec fields that name a file (config, checkpoint_a, checkpoint_b) are attacker
// controlled: without guarding, a submitter could point config at an arbitrary
// runner file whose contents then sync back to the client through the rendered
// config. The guard rejects absolute paths and any ".." segment, then resolves
// against a base directory and verifies containment.
//
// The functions are pure (no filesystem access beyond lexical cleaning) so they
// are unit-testable in isolation. Containment is enforced against a base dir
// that, in M3, is the job worktree (config) or the staged run dir (checkpoints);
// the base does not need to exist for the lexical guard to hold.
package pathguard

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// ErrAbsolute is returned when a spec path is absolute.
var ErrAbsolute = errors.New("path must be repo-relative, not absolute")

// ErrParentTraversal is returned when a spec path contains a ".." segment.
var ErrParentTraversal = errors.New("path must not contain a .. segment")

// ErrEmpty is returned when a required spec path is empty.
var ErrEmpty = errors.New("path is empty")

// ErrEscapesBase is returned when a cleaned path resolves outside its base dir.
var ErrEscapesBase = errors.New("path escapes its base directory")

// CheckRel validates the lexical shape of a repo-relative spec path without a
// base: it rejects an empty path, an absolute path, and any ".." path segment
// (checked segment-wise so a legitimate name like "..config" is not rejected,
// while "../x", "a/../b", and a bare ".." are). This is the guard applied at
// submit time in M2, before a worktree exists.
func CheckRel(rel string) error {
	if rel == "" {
		return ErrEmpty
	}
	if filepath.IsAbs(rel) {
		return fmt.Errorf("%w: %q", ErrAbsolute, rel)
	}
	// Reject a leading "/" and Windows-style volume/backslash defensively, then
	// walk segments split on the forward slash (the on-wire separator).
	if strings.HasPrefix(rel, "/") || strings.ContainsRune(rel, '\\') {
		return fmt.Errorf("%w: %q", ErrAbsolute, rel)
	}
	for _, seg := range strings.Split(rel, "/") {
		if seg == ".." {
			return fmt.Errorf("%w: %q", ErrParentTraversal, rel)
		}
	}
	return nil
}

// Resolve applies CheckRel and then joins rel under base and verifies the
// cleaned result stays inside base (a defense-in-depth containment check on top
// of the lexical guard). It returns the cleaned absolute-or-base-relative joined
// path. base is treated as the containment root; it is cleaned but not required
// to exist. This is the full guard consumed in M3 once the worktree/run dir base
// is known.
func Resolve(base, rel string) (string, error) {
	if err := CheckRel(rel); err != nil {
		return "", err
	}
	cleanBase := filepath.Clean(base)
	joined := filepath.Clean(filepath.Join(cleanBase, rel))
	// Containment: joined must be cleanBase itself or sit under cleanBase/.
	if joined != cleanBase && !strings.HasPrefix(joined, cleanBase+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: %q not under %q", ErrEscapesBase, joined, cleanBase)
	}
	return joined, nil
}
