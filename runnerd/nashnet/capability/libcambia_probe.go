package capability

import "strings"

// RunFunc executes name with args and env, returning its combined output. It
// is the seam ProbeCanBuildLibcambia uses so tests inject fake toolchain and
// compiler responses without touching a real Go install or C compiler.
type RunFunc func(name string, args []string, env []string) ([]byte, error)

// ProbeCanBuildLibcambia reports whether this node can build libcambia.so
// (D9): the pinned Go toolchain resolves, and a C compiler the cgo build
// would invoke is present and runnable. It mirrors the build environment of
// ingest.buildLibcambia (GOTOOLCHAIN=<pin>, CGO_ENABLED=1,
// runnerd/ingest/libcambia.go:59-62) without running an actual build, so
// declaring this fact costs nothing on every heartbeat. pin is the same
// goToolchainPin value ingest.go pins builds to (runnerd/ingest/ingest.go:31,
// "go1.26.0").
//
// A node with the Go toolchain but no cc reports false: Prepare would fail
// ensureLibcambia for every needs_libcambia job placed there, and a wrong
// true declaration is the exact failure D63's per-node circuit breaker exists
// to recover from, not something this probe should risk.
func ProbeCanBuildLibcambia(pin string, run RunFunc) bool {
	if _, err := run("go", []string{"version"}, []string{"GOTOOLCHAIN=" + pin}); err != nil {
		return false
	}
	ccOut, err := run("go", []string{"env", "CC"}, []string{"CGO_ENABLED=1"})
	if err != nil {
		return false
	}
	cc := strings.TrimSpace(string(ccOut))
	if cc == "" {
		return false
	}
	if _, err := run(cc, []string{"--version"}, nil); err != nil {
		return false
	}
	return true
}
