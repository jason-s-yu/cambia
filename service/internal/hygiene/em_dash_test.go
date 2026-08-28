// Package hygiene holds repo-hygiene checks that run as part of the normal
// service test suite rather than a separate script.
package hygiene

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// emDash is U+2014 EM DASH, encoded as its 3-byte UTF-8 sequence so the scan
// works on raw file bytes regardless of a file's declared encoding.
var emDash = []byte{0xE2, 0x80, 0x94}

// skipDirs are never descended into: dependency/build trees and fixture data
// that isn't hand-written prose.
var skipDirs = map[string]bool{
	".git":         true,
	"vendor":       true,
	"testdata":     true,
	"node_modules": true,
}

// TestNoEmDash walks the service module tree and fails on any U+2014 (em
// dash), reporting file:line. CLAUDE.md bans em dashes in all generated
// text (code comments included); this is the mechanical backstop (cambia-927).
func TestNoEmDash(t *testing.T) {
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatalf("resolve service root: %v", err)
	}

	var violations []string
	walkErr := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if skipDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			// Unreadable (permissions, broken symlink, etc.) isn't this
			// check's concern; skip rather than fail the sweep on it.
			return nil
		}
		if !bytes.Contains(data, emDash) {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			rel = path
		}
		for i, line := range bytes.Split(data, []byte("\n")) {
			if bytes.Contains(line, emDash) {
				violations = append(violations, fmt.Sprintf("%s:%d", filepath.ToSlash(rel), i+1))
			}
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walk %s: %v", root, walkErr)
	}

	if len(violations) > 0 {
		t.Errorf("found %d em dash (U+2014) occurrence(s):", len(violations))
		for _, v := range violations {
			t.Errorf("  %s", v)
		}
	}
}
