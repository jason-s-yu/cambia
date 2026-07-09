package pathguard

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckRelAccepts(t *testing.T) {
	ok := []string{
		"cfr/config/prtcfr_prod.yaml",
		"config.yaml",
		"a/b/c.yaml",
		"..config.yaml", // ".." only as a substring of a segment, not a segment
		"snapshots/ckpt.pt",
		"dir.with.dots/f",
	}
	for _, p := range ok {
		if err := CheckRel(p); err != nil {
			t.Errorf("CheckRel(%q) = %v, want nil", p, err)
		}
	}
}

func TestCheckRelRejectsAbsolute(t *testing.T) {
	for _, p := range []string{"/etc/passwd", "/srv/cambia/keys/tls.key"} {
		if err := CheckRel(p); !errors.Is(err, ErrAbsolute) {
			t.Errorf("CheckRel(%q) = %v, want ErrAbsolute", p, err)
		}
	}
}

func TestCheckRelRejectsTraversal(t *testing.T) {
	for _, p := range []string{"..", "../x", "a/../../etc/passwd", "cfr/../../secret", "a/.."} {
		if err := CheckRel(p); !errors.Is(err, ErrParentTraversal) {
			t.Errorf("CheckRel(%q) = %v, want ErrParentTraversal", p, err)
		}
	}
}

func TestCheckRelRejectsEmpty(t *testing.T) {
	if err := CheckRel(""); !errors.Is(err, ErrEmpty) {
		t.Errorf("CheckRel(\"\") = %v, want ErrEmpty", err)
	}
}

func TestCheckRelRejectsBackslash(t *testing.T) {
	if err := CheckRel(`cfr\config`); !errors.Is(err, ErrAbsolute) {
		t.Errorf("CheckRel backslash = %v, want ErrAbsolute", err)
	}
}

func TestResolveContainment(t *testing.T) {
	base := "/srv/cambia/worktrees/job-1"
	got, err := Resolve(base, "cfr/config/x.yaml")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := filepath.Clean(base + "/cfr/config/x.yaml")
	if got != want {
		t.Fatalf("Resolve = %q, want %q", got, want)
	}
}

func TestResolveRejectsEscape(t *testing.T) {
	// Even without a lexical ".." (caught by CheckRel), a base with a trailing
	// element that a sibling could shadow must not slip through. The ".." form is
	// caught first; this asserts the containment guard is wired.
	if _, err := Resolve("/srv/cambia/worktrees/job-1", "../job-2/config.yaml"); !errors.Is(err, ErrParentTraversal) {
		t.Fatalf("Resolve traversal = %v, want ErrParentTraversal", err)
	}
}

func TestResolveBaseItself(t *testing.T) {
	// A "." rel resolves to base itself, which is contained.
	got, err := Resolve("/srv/cambia/x", ".")
	if err != nil {
		t.Fatalf("Resolve(.): %v", err)
	}
	if got != filepath.Clean("/srv/cambia/x") {
		t.Fatalf("Resolve(.) = %q", got)
	}
}

// TestResolveSymlinkInsideBaseEscapes covers the M4 case a lexical guard misses:
// a symlink planted inside the base that points outside it. The lexical join
// stays under base, but the resolved target does not, so Resolve must reject.
func TestResolveSymlinkInsideBaseEscapes(t *testing.T) {
	tmp := t.TempDir()
	base := filepath.Join(tmp, "base")
	outside := filepath.Join(tmp, "outside")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(base, "link")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "secret.yaml"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{"link", "link/secret.yaml"} {
		if _, err := Resolve(base, rel); !errors.Is(err, ErrEscapesBase) {
			t.Fatalf("Resolve(%q) = %v, want ErrEscapesBase", rel, err)
		}
	}

	// A dangling symlink to a non-existent outside target must also be rejected:
	// the link target does not exist, but a write through it would still escape.
	if err := os.Symlink(filepath.Join(tmp, "nope", "x"), filepath.Join(base, "dangling")); err != nil {
		t.Fatal(err)
	}
	if _, err := Resolve(base, "dangling"); !errors.Is(err, ErrEscapesBase) {
		t.Fatalf("Resolve(dangling) = %v, want ErrEscapesBase", err)
	}
}

// TestResolveSymlinkInsideBaseContained accepts a symlink inside base that
// points to another location within base.
func TestResolveSymlinkInsideBaseContained(t *testing.T) {
	tmp := t.TempDir()
	base := filepath.Join(tmp, "base")
	sub := filepath.Join(base, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(sub, filepath.Join(base, "link")); err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(base, "link/config.yaml")
	if err != nil {
		t.Fatalf("Resolve(link/config.yaml) = %v, want nil", err)
	}
	if got != filepath.Join(base, "link", "config.yaml") {
		t.Fatalf("Resolve returned %q, want the lexical in-base path", got)
	}
}

// TestResolveNonExistentUnderBase accepts a path that does not yet exist under a
// real base (the common case: a config or checkpoint not yet written).
func TestResolveNonExistentUnderBase(t *testing.T) {
	base := t.TempDir()
	got, err := Resolve(base, "runs/job/config.yaml")
	if err != nil {
		t.Fatalf("Resolve non-existent = %v, want nil", err)
	}
	if got != filepath.Join(base, "runs", "job", "config.yaml") {
		t.Fatalf("Resolve returned %q", got)
	}
}

// TestResolveBaseIsSymlink accepts a base that is itself a symlink: base and
// candidate resolve through the same link, so containment holds consistently.
func TestResolveBaseIsSymlink(t *testing.T) {
	tmp := t.TempDir()
	real := filepath.Join(tmp, "real")
	if err := os.MkdirAll(real, 0o755); err != nil {
		t.Fatal(err)
	}
	baseLink := filepath.Join(tmp, "baselink")
	if err := os.Symlink(real, baseLink); err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(baseLink, "config.yaml")
	if err != nil {
		t.Fatalf("Resolve through symlinked base = %v, want nil", err)
	}
	if got != filepath.Join(baseLink, "config.yaml") {
		t.Fatalf("Resolve returned %q", got)
	}
}
