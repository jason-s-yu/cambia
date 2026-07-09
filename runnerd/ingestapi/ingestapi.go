// Package ingestapi holds the shared types crossing the runnerd -> ingest
// boundary. It is a leaf package (no runnerd imports) so both the harness core
// (M2) and the ingest pipeline (M3, package runnerd/ingest) can depend on it
// without a cycle. The Environment interface that consumes these types is
// declared in the harness package, next to its consumer, per the milestone
// contract; only the data shape lives here.
package ingestapi

// Prepared is the result of an ingest Prepare step: the fully staged execution
// environment for one job. The harness launches the job's subprocess from these
// fields. In M2 the launch path still uses the manager's constructor-fixed
// cambia binary and cfr dir, so only RunDir/RenderedConfig are load-bearing;
// WorktreeDir, VenvPython, LibcambiaPath, and Env are consumed by the M3 launch
// parameterization (per-job venv interpreter, LIBCAMBIA_PATH, PYTHONPATH shim).
type Prepared struct {
	// WorktreeDir is the detached git worktree checked out at the pinned sha
	// (…/worktrees/<job-id>). Launch cwd is <WorktreeDir>/cfr in M3.
	WorktreeDir string
	// RunDir is the run directory (…/runs/<name>) holding process.json, the
	// rendered config, logs/, snapshots/, and the runner-local run_db.sqlite.
	RunDir string
	// VenvPython is the absolute path to the per-lock uv venv interpreter used
	// to execute the job (M3). Empty in M2 (manager uses its fixed binary).
	VenvPython string
	// LibcambiaPath is the absolute path to the cached libcambia.so for the
	// job's engine tree, exported to the subprocess via LIBCAMBIA_PATH (M3).
	LibcambiaPath string
	// RenderedConfig is the absolute path to the rendered, validated config.yaml
	// inside RunDir. The subprocess consumes this file verbatim.
	RenderedConfig string
	// Env is the additional environment (KEY=VALUE) the subprocess is launched
	// with (PYTHONPATH pin, LIBCAMBIA_PATH, GOTOOLCHAIN, ...). Consumed in M3.
	Env []string
}
