// Package ingestapi holds the leaf data type the ingest Manager returns to the
// runnerd daemon layer (M2). It is deliberately dependency-free so both the
// ingest package and the daemon can import it without a cycle.
//
// INTEGRATION NOTE (M3 -> M2 reconciliation): this is a TEMPORARY local copy of
// the type M2 owns at github.com/jason-s-yu/cambia/runnerd/ingestapi. It exists
// here only so the ingest package compiles and tests run before M2 lands its
// leaf package. At integration the chief replaces the single import line in the
// ingest package (see the `ingestapi` import in ingest.go) with M2's real
// package; the struct shape below is identical to the agreed contract and must
// stay byte-for-byte in sync until the swap.
package ingestapi

// Prepared is the fully staged execution context the daemon needs to launch a
// job. Every path is absolute. The daemon sets the launch working directory to
// WorktreeDir/cfr, uses VenvPython as the interpreter, and layers Env over the
// base process environment.
type Prepared struct {
	// WorktreeDir is the detached git worktree checked out at the pinned sha.
	// The launch cwd is WorktreeDir/cfr.
	WorktreeDir string
	// RunDir is the run directory (outside the worktree) holding process.json,
	// config.yaml, env.json, logs/, snapshots/, and the runner-local run_db.
	RunDir string
	// VenvPython is the per-lock uv venv interpreter (…/bin/python).
	VenvPython string
	// LibcambiaPath is the cached engine shared library, consumed via
	// LIBCAMBIA_PATH.
	LibcambiaPath string
	// RenderedConfig is the materialized, rails-applied config.yaml inside
	// RunDir. Empty when the job kind does not render a config (run-dir modes).
	RenderedConfig string
	// Env is the set of harness-controlled environment entries (KEY=VALUE) the
	// daemon layers over the base process environment at launch: PYTHONPATH
	// (shim dir + worktree cfr), LIBCAMBIA_PATH, CAMBIA_EXPECTED_SRC_ROOT, and
	// hygiene pins. It is additive, not a full environment.
	Env []string
}
