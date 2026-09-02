package nashnet

import (
	"go/parser"
	"go/token"
	"os/exec"
	"strings"
	"testing"
)

// modulePrefix is the runnerd module path. Any dependency under it makes this
// package a non-leaf.
const modulePrefix = "github.com/jason-s-yu/cambia/runnerd/"

// TestPackageImportsNoOtherRunnerdPackage is the leaf assertion of the ticket:
// nashnet must not import runnerd/harness or runnerd/procmgr, directly or
// transitively, so the dispatcher and the HTTP layer can depend on it without a
// cycle and its state machine stays testable with no server. The atomic write
// of lease.json copies the pattern of procmgr.WriteProcessState rather than
// calling it, which is the one place the rule costs anything.
func TestPackageImportsNoOtherRunnerdPackage(t *testing.T) {
	// The source scan always runs: it needs no toolchain and it names the file
	// and the import that broke the rule.
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse the package: %v", err)
	}
	for _, pkg := range pkgs {
		for name, file := range pkg.Files {
			if strings.HasSuffix(name, "_test.go") {
				continue
			}
			for _, imp := range file.Imports {
				path := strings.Trim(imp.Path.Value, `"`)
				if strings.HasPrefix(path, modulePrefix) {
					t.Errorf("%s imports %s; nashnet is a leaf package", name, path)
				}
			}
		}
	}

	// go list adds the transitive half: an import of a new leaf package that
	// itself reaches harness or procmgr fails here.
	out, err := exec.Command("go", "list", "-deps", ".").Output()
	if err != nil {
		t.Skipf("go list unavailable, source scan only: %v", err)
	}
	for _, dep := range strings.Fields(string(out)) {
		if strings.HasPrefix(dep, modulePrefix) && dep != strings.TrimSuffix(modulePrefix, "/")+"/nashnet" {
			t.Errorf("nashnet depends on %s transitively; it must be a leaf", dep)
		}
	}
}
