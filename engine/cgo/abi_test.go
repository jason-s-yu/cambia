package main

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jason-s-yu/cambia/engine/cgo/abiver"
)

// update regenerates testdata/abi_golden.txt from the live build instead of
// failing on a mismatch: go test ./cgo/ -run TestABIGolden -update
var update = flag.Bool("update", false, "update engine/cgo/testdata/abi_golden.txt")

const goldenPath = "testdata/abi_golden.txt"

// externLine matches a single exported C prototype line in the header
// go build -buildmode=c-shared emits alongside the .so, e.g.:
//
//	extern int32_t cambia_game_new(uint64_t seed);
//
// Only cambia_-prefixed lines are kept: the header also carries cgo's own
// runtime helpers (_GoString_ etc.), which are not part of this package's
// exported surface and change independently of it.
var externLine = regexp.MustCompile(`^extern .*\bcambia_[A-Za-z0-9_]*\(`)

// buildGolden shells out to `go build -buildmode=c-shared` and extracts the
// exported symbol prototypes from the generated header, sorted, one per
// line, prefixed by the abiver.Generation this test binary was itself built
// against. This is the actual signature surface libcambia.so hands to any
// FFI caller: comparing it against a checked-in golden is what catches an
// //export signature change that did not bump abiver.Generation - a diff
// this package's own doc comment, not just a hand-maintained list, cannot
// drift out of sync with, since it is regenerated from a real build.
func buildGolden(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	soPath := filepath.Join(dir, "libcambia_golden.so")
	cmd := exec.Command("go", "build", "-buildmode=c-shared", "-o", soPath, ".")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build c-shared for golden check: %v\n%s", err, out)
	}
	headerPath := strings.TrimSuffix(soPath, filepath.Ext(soPath)) + ".h"
	header, err := os.ReadFile(headerPath)
	if err != nil {
		t.Fatalf("read generated header %s: %v", headerPath, err)
	}

	var protos []string
	for _, line := range strings.Split(string(header), "\n") {
		line = strings.TrimSpace(line)
		if externLine.MatchString(line) {
			protos = append(protos, line)
		}
	}
	sort.Strings(protos)

	var sb strings.Builder
	sb.WriteString("# libcambia.so C ABI golden (cambia-1689). Do not hand-edit.\n")
	sb.WriteString("# Regenerate after bumping engine/cgo/abiver.Generation for any\n")
	sb.WriteString("# exported-signature change:\n")
	sb.WriteString("#   go test ./cgo/ -run TestABIGolden -update\n")
	sb.WriteString("generation=" + strconv.Itoa(abiver.Generation) + "\n")
	for _, p := range protos {
		sb.WriteString(p)
		sb.WriteString("\n")
	}
	return sb.String()
}

// TestABIGolden pins the exported C ABI (every //export symbol's full
// prototype, plus abiver.Generation) against testdata/abi_golden.txt. A
// signature change to any //export function - an added parameter, a changed
// type, a new or removed export - changes the live-built prototype text and
// fails this test unless the golden is regenerated, which forces the
// generation bump and the golden update to land in the same diff a reviewer
// sees (cambia-1689 AC2).
func TestABIGolden(t *testing.T) {
	actual := buildGolden(t)

	if *update {
		if err := os.WriteFile(goldenPath, []byte(actual), 0o644); err != nil {
			t.Fatalf("write %s: %v", goldenPath, err)
		}
		t.Logf("wrote %s", goldenPath)
		return
	}

	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read %s: %v (run `go test ./cgo/ -run TestABIGolden -update` to create it)", goldenPath, err)
	}

	if actual != string(want) {
		t.Fatalf(
			"exported C ABI drifted from %s.\n"+
				"If this is a deliberate exported-signature change, bump "+
				"abiver.Generation in engine/cgo/abiver/abiver.go, then run:\n"+
				"  go test ./cgo/ -run TestABIGolden -update\n"+
				"and review the diff.\n\n--- want (%s) ---\n%s\n--- got (live build) ---\n%s",
			goldenPath, goldenPath, string(want), actual,
		)
	}
}
