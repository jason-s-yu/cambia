package harness

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
)

// Environment is the ingest boundary the dispatcher drives when a job leaves the
// queue. It is declared here, next to its consumer, per the milestone contract;
// the real implementation lands in M3 as package runnerd/ingest (git mirror,
// worktree, uv venv, libcambia build, config render, env.json). M2 defines and
// exercises the interface against a stub (production, no ingest wired yet) and a
// fake (tests).
type Environment interface {
	// Prepare stages the execution environment for jobID at the pinned commit:
	// checkout, venv, libcambia, and the rendered+validated config. It returns
	// the staged paths, or an error the dispatcher records as a job failure. The
	// context is canceled if the job is canceled mid-prepare.
	Prepare(ctx context.Context, jobID, commit, kind, configRel string, overrides map[string]string) (*ingestapi.Prepared, error)
	// Cleanup releases a terminal job's worktree. It keeps it for a debug TTL
	// when keepForDebug is set (e.g. a crash). The job's mirror ref is NOT
	// released here: it lives as long as the run dir so a resume can
	// re-prepare from the pinned commit (see PurgeRef).
	Cleanup(jobID string, keepForDebug bool) error
	// PurgeRef deletes the job's mirror ref when its run dir is purged.
	PurgeRef(jobID string) error
	// StartupSweep reconciles staged resources at daemon start: anything staged
	// for a job not in liveJobIDs and not in a debug TTL is pruned; job refs
	// without a run dir are reaped.
	StartupSweep(liveJobIDs []string) error
}

// ErrIngestNotWired is returned by the stub Environment's Prepare. Until the M3
// ingest pipeline is wired, the daemon cannot stage a real job, so it fails the
// job honestly rather than launching a half-prepared run.
var ErrIngestNotWired = errors.New("ingest pipeline not wired (milestone M3)")

// StubEnvironment is the Environment used in production until M3 lands. Prepare
// fails closed; Cleanup and StartupSweep are no-ops. The queue, state machine,
// API, and reconcile paths are fully exercised against it (a submitted job is
// admitted, dispatched, and lands in failed with ErrIngestNotWired), so the M2
// daemon is operable and observable without ingest.
type StubEnvironment struct{}

// Prepare fails closed with ErrIngestNotWired.
func (StubEnvironment) Prepare(context.Context, string, string, string, string, map[string]string) (*ingestapi.Prepared, error) {
	return nil, ErrIngestNotWired
}

// Cleanup is a no-op in the stub.
func (StubEnvironment) Cleanup(string, bool) error { return nil }

// PurgeRef is a no-op in the stub.
func (StubEnvironment) PurgeRef(string) error { return nil }

// StartupSweep is a no-op in the stub.
func (StubEnvironment) StartupSweep([]string) error { return nil }

// jobSpecFile is the basename of the write-once accepted-spec record written
// next to process.json at admission. It lets GET report kind/commit/config/
// priority after the in-memory job is gone or after a restart, and is distinct
// from env.json (M3 provenance).
const jobSpecFile = "jobspec.json"

// writeJobSpec records the accepted spec into runDir/jobspec.json.
func writeJobSpec(runDir string, s *JobSpec) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(runDir, jobSpecFile), append(data, '\n'), 0o644)
}

// readJobSpec reads runDir/jobspec.json, returning nil if absent.
func readJobSpec(runDir string) *JobSpec {
	data, err := os.ReadFile(filepath.Join(runDir, jobSpecFile))
	if err != nil {
		return nil
	}
	var s JobSpec
	if err := json.Unmarshal(data, &s); err != nil {
		return nil
	}
	return &s
}
