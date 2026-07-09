// Package pathguard implements the spec-field path guards of design 5.4. Job
// spec fields that name a file (config, checkpoint_a, checkpoint_b) are attacker
// controlled: without guarding, a submitter could point config at an arbitrary
// runner file whose contents then sync back to the client through the rendered
// config. The guard rejects absolute paths and any ".." segment, then resolves
// against a base directory and verifies containment.
//
// CheckRel is pure (lexical only). Resolve additionally touches the filesystem:
// after the lexical guard it resolves symlinks on both the candidate and the
// base and re-checks containment against the resolved base, so a symlink planted
// inside a submitter-controlled worktree cannot make a guarded path escape its
// base. Containment is enforced against a base dir that, in M3, is the job
// worktree (config) or the staged run dir (checkpoints); neither the base nor
// the candidate needs to exist (a non-existent tail is resolved against its
// deepest existing ancestor).
package pathguard

import (
	"errors"
	"fmt"
	"os"
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
	// Lexical containment: joined must be cleanBase itself or sit under cleanBase/.
	if joined != cleanBase && !strings.HasPrefix(joined, cleanBase+string(filepath.Separator)) {
		return "", fmt.Errorf("%w: %q not under %q", ErrEscapesBase, joined, cleanBase)
	}
	// Symlink containment: the lexical check above cannot see a symlink planted
	// inside the (submitter-controlled) base that points outside it. Resolve
	// symlinks on both sides and re-check against the resolved base so such a
	// link cannot launder an escape past the lexical guard.
	if err := verifyResolvedContainment(cleanBase, joined); err != nil {
		return "", err
	}
	return joined, nil
}

// verifyResolvedContainment resolves symlinks on base and joined and fails if
// the resolved candidate does not sit inside the resolved base. A base that is
// itself a symlink resolves consistently on both sides, so it is accepted.
func verifyResolvedContainment(cleanBase, joined string) error {
	rBase, err := ResolveForContainment(cleanBase)
	if err != nil {
		return fmt.Errorf("%w: resolve base %q: %v", ErrEscapesBase, cleanBase, err)
	}
	rJoined, err := ResolveForContainment(joined)
	if err != nil {
		return fmt.Errorf("%w: resolve %q: %v", ErrEscapesBase, joined, err)
	}
	if rJoined != rBase && !strings.HasPrefix(rJoined, rBase+string(filepath.Separator)) {
		return fmt.Errorf("%w: %q resolves to %q, outside %q", ErrEscapesBase, joined, rJoined, rBase)
	}
	return nil
}

// ResolveForContainment returns a symlink-resolved absolute-ish path suitable
// for use as, or comparison against, a containment prefix. It resolves every
// symlink in the existing portion of path, follows a dangling final symlink to
// its (possibly non-existent) target, and leaves a genuinely non-existent tail
// component literal. This closes the gap where filepath.EvalSymlinks stops at
// the first missing component and would otherwise let a dangling symlink smuggle
// its tail past a containment check.
func ResolveForContainment(path string) (string, error) {
	path = filepath.Clean(path)
	for hops := 0; hops < 64; hops++ {
		if resolved, err := filepath.EvalSymlinks(path); err == nil {
			return resolved, nil
		}
		// path has a non-existent component. If path itself is a symlink it is
		// dangling: follow its target and retry so the link cannot hide behind
		// its missing target.
		if fi, lerr := os.Lstat(path); lerr == nil && fi.Mode()&os.ModeSymlink != 0 {
			target, rerr := os.Readlink(path)
			if rerr != nil {
				return "", rerr
			}
			if !filepath.IsAbs(target) {
				target = filepath.Join(filepath.Dir(path), target)
			}
			path = filepath.Clean(target)
			continue
		}
		// path's final component does not exist and is not a symlink: resolve the
		// parent (recursively, so an intermediate dangling symlink is handled)
		// and rejoin the literal tail.
		parent := filepath.Dir(path)
		if parent == path {
			return path, nil // reached the filesystem root
		}
		rParent, err := ResolveForContainment(parent)
		if err != nil {
			return "", err
		}
		return filepath.Join(rParent, filepath.Base(path)), nil
	}
	return "", fmt.Errorf("too many symlink hops resolving %q", path)
}
