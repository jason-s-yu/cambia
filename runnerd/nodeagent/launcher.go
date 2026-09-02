package nodeagent

import (
	"context"
	"errors"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Environment is the node's ingest boundary (D15): the same Manager the
// coordinator drives, instantiated with node-local paths. It is an interface
// so the agent's claim-to-result cycle is testable with a fake staging step
// and no git, no uv, and no cgo toolchain.
type Environment interface {
	// BundleFetch imports a coordinator-served git bundle into the node's own
	// mirror; its non-force fetch refuses a job ref that already points
	// elsewhere (D48).
	BundleFetch(ctx context.Context, jobID, bundlePath string) error
	// Prepare stages the job at the pinned commit exactly as it does on the
	// coordinator: worktree, venv, libcambia, rendered config, env.json.
	Prepare(ctx context.Context, jobID, commit, kind, configRel, device, warmStart string, overrides map[string]string) (*ingestapi.Prepared, error)
	// Cleanup releases a terminal job's worktree, keeping it for the debug TTL
	// on a crash.
	Cleanup(jobID string, keepForDebug bool) error
}

// ProcessStatus is the node's view of one supervised process. Status is the
// row verbatim and Effective is the same row with pid liveness applied, which
// is the only way to observe the exit of a process this daemon did not fork
// (procmgr.EffectiveStatus).
type ProcessStatus struct {
	Status    string
	Effective string
	PID       int
	ExitCode  *int
	Found     bool
}

// Launcher is the procmgr boundary. The node forks jobs into their own process
// group through procmgr.StartWithOpts so the pgid, the pid-reuse guard, and
// the allowlisted child environment are the same ones the coordinator uses
// (D17).
type Launcher interface {
	// Ensure makes the run dir, its logs dir, and the process.json row a
	// launch needs, adopting an existing non-running row.
	Ensure(name, algorithm string) error
	// Start forks the job and returns its pid.
	Start(name string, l Launch) (int, error)
	// Stop signals the job's process group: SIGINT with the 30s grace, or
	// SIGKILL when force is set.
	Stop(name string, force bool) error
	// Status reports the current process.json row.
	Status(name string) ProcessStatus
}

// procLauncher is the production Launcher over procmgr.ProcessManager.
type procLauncher struct {
	pm *procmgr.ProcessManager
}

// NewLauncher wraps a ProcessManager as a Launcher.
func NewLauncher(pm *procmgr.ProcessManager) Launcher { return &procLauncher{pm: pm} }

// Ensure creates the run row, or adopts an existing one that is not running.
// A re-claim of the same job on the same node (after a nack, an expiry
// requeue, or an agent restart) finds the row already there, which is not a
// collision the node should refuse.
func (p *procLauncher) Ensure(name, algorithm string) error {
	_, err := p.pm.Create(procmgr.CreateRequest{Name: name, Algorithm: algorithm})
	if err == nil {
		return nil
	}
	if !errors.Is(err, procmgr.ErrNameCollision) {
		return err
	}
	st, ok := p.pm.GetState(name)
	if !ok {
		return err
	}
	switch st.Status {
	case procmgr.StatusRunning, procmgr.StatusStarting, procmgr.StatusStopping:
		return err
	}
	return nil
}

func (p *procLauncher) Start(name string, l Launch) (int, error) {
	opts := procmgr.LaunchOpts{Python: l.Python, Argv: l.Argv, Cwd: l.Cwd, Env: l.Env}
	var (
		st  *procmgr.ProcessState
		err error
	)
	if l.Resume {
		st, err = p.pm.ResumeWithOpts(name, procmgr.StartOpts{}, opts)
	} else {
		st, err = p.pm.StartWithOpts(name, procmgr.StartOpts{}, opts)
	}
	if err != nil {
		return 0, err
	}
	return st.PID, nil
}

func (p *procLauncher) Stop(name string, force bool) error {
	_, err := p.pm.Stop(name, force)
	return err
}

func (p *procLauncher) Status(name string) ProcessStatus {
	st, ok := p.pm.GetState(name)
	if !ok {
		return ProcessStatus{}
	}
	return ProcessStatus{
		Status:    st.Status,
		Effective: procmgr.EffectiveStatus(st),
		PID:       st.PID,
		ExitCode:  st.ExitCode,
		Found:     true,
	}
}

// NewEnvironment builds the node's ingest Manager with node-local paths (D15,
// D18): the caches live under the node's own base dir, keyed exactly as they
// are on the coordinator, so two nodes never collide and a node reuses its own
// artifacts across jobs.
func NewEnvironment(cfg Config, coresCap int) Environment {
	return ingest.New(ingestConfig(cfg, coresCap, ""))
}

// NewEmbeddedEnvironment builds the embedded node's ingest Manager. It is the
// coordinator's own base dir, mirror, worktrees, and caches, because the node
// runs in the coordinator's process and staging a second copy of them would
// double the disk for nothing (D40). The one difference is the provenance
// record: the coordinator has already authored env.json with executed_on in
// the run dir this stage writes into, so the staging record is written as
// env.node.json, the name a remote node's own copy is promoted under (D52).
func NewEmbeddedEnvironment(cfg Config, coresCap int) Environment {
	return ingest.New(ingestConfig(cfg, coresCap, envNodeJSONName))
}

// ingestConfig is the one ingest.Config both environments are built from, so a
// field added for a remote node reaches the embedded one without a second edit.
func ingestConfig(cfg Config, coresCap int, envJSONName string) ingest.Config {
	return ingest.Config{
		BaseDir:              cfg.BaseDir,
		RunsDir:              cfg.RunsDir,
		MaxVenvs:             cfg.Caches.MaxVenvs,
		MaxLibcambia:         cfg.Caches.MaxLibcambia,
		CoresCap:             coresCap,
		PythonBin:            cfg.PythonBin,
		RequireSignedCommits: cfg.RequireSignedCommits,
		AllowedSignersPath:   cfg.AllowedSignersPath,
		EnvJSONName:          envJSONName,
	}
}
