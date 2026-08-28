// Package hygiene holds repo-hygiene checks that run as part of the normal
// runnerd test suite rather than a separate script.
package hygiene

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// emDash is U+2014 EM DASH, encoded as its 3-byte UTF-8 sequence so the scan
// works on raw file bytes and so this file is not itself a violation.
var emDash = []byte{0xE2, 0x80, 0x94}

// textExts bounds the scan to hand-written text by extension. Compiled
// binaries, checkpoints, and archives carry the em dash byte sequence by
// chance; scanning them fails the check with a meaningless byte offset
// instead of a line number (cambia-927 H1).
var textExts = map[string]bool{
	".go": true, ".md": true, ".sql": true, ".json": true, ".yaml": true,
	".yml": true, ".html": true, ".css": true, ".ts": true, ".tsx": true,
	".js": true, ".mjs": true, ".sh": true, ".txt": true, ".toml": true,
	".conf": true, ".env": true, ".mod": true, ".work": true, ".sum": true,
}

// textNames covers extensionless text files worth scanning.
var textNames = map[string]bool{
	"Dockerfile": true,
	"Makefile":   true,
}

// skipDirs are never descended into by the fallback walk: dependency trees,
// build output, and generated run/data artifacts. The git-backed enumeration
// excludes those already; this only matters when git is unavailable.
var skipDirs = map[string]bool{
	".git": true, ".vscode": true, "vendor": true, "testdata": true,
	"node_modules": true, "dist": true, "bin": true, "runs": true,
	"data": true, "tmp": true,
}

// isTextFile reports whether a file name looks like hand-written text.
func isTextFile(name string) bool {
	return textExts[filepath.Ext(name)] || textNames[name]
}

// repoFiles lists the files git considers part of the tree under root:
// tracked files plus untracked ones that are not ignored. Gitignored build
// output is excluded, which is the point of preferring git here. Returns
// ok=false when git is missing or root is not a repository, so the caller
// can fall back to a filtered walk.
func repoFiles(root string) ([]string, bool) {
	out, err := exec.Command("git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").Output()
	if err != nil {
		return nil, false
	}
	var files []string
	for _, rel := range strings.Split(string(out), "\x00") {
		if rel == "" {
			continue
		}
		files = append(files, filepath.Join(root, filepath.FromSlash(rel)))
	}
	return files, true
}

// walkFiles enumerates every file under root, skipping skipDirs. Used only
// when git cannot answer.
func walkFiles(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if path != root && skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		files = append(files, path)
		return nil
	})
	return files, err
}

// TestNoEmDash scans the runnerd module's own text files and fails on any
// U+2014 (em dash), reporting file:line. CLAUDE.md bans em dashes in all
// generated text (code comments included); this is the mechanical backstop
// (cambia-927).
func TestNoEmDash(t *testing.T) {
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatalf("resolve runnerd root: %v", err)
	}

	files, ok := repoFiles(root)
	if !ok {
		files, err = walkFiles(root)
		if err != nil {
			t.Fatalf("walk %s: %v", root, err)
		}
	}

	scanned := 0
	var violations []string
	for _, path := range files {
		if !isTextFile(filepath.Base(path)) {
			continue
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			// Deleted-but-tracked, unreadable, or a broken symlink is not
			// this check's concern.
			continue
		}
		if bytes.IndexByte(data, 0) >= 0 {
			// Binary content behind a text extension; a byte-level match
			// there is coincidence, not prose.
			continue
		}
		scanned++
		if !bytes.Contains(data, emDash) {
			continue
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			rel = path
		}
		for i, line := range bytes.Split(data, []byte("\n")) {
			if bytes.Contains(line, emDash) {
				violations = append(violations, fmt.Sprintf("%s:%d: %s", filepath.ToSlash(rel), i+1, bytes.TrimSpace(line)))
			}
		}
	}

	if scanned == 0 {
		t.Fatalf("scanned 0 files under %s: the em dash guard is checking nothing", root)
	}

	if len(violations) > 0 {
		t.Errorf("found %d em dash (U+2014) occurrence(s) across %d files:", len(violations), scanned)
		for _, v := range violations {
			t.Errorf("  %s", v)
		}
	}
}
