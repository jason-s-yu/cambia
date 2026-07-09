package pathguard

import (
	"errors"
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
