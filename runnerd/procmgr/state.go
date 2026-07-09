// Package procmgr supervises detached cambia subprocesses: it owns the
// process.json current-state store, the spawn/stop/resume lifecycle, pid-reuse-
// safe liveness checks, and the launch preflight gates. The host store (the
// dashboard service's TrainingStore or runnerd's own) is injected through the
// RunResolver interface, and the algorithm allowlist through a constructor table,
// so the package depends on no concrete run store.
package procmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// processStateFile is the basename of the Go-owned current-process-state store
// written into each run directory.
const processStateFile = "process.json"

// ProcessState is the authoritative current-state record for a training run.
// It replaces train.pid as the source of truth for "is this run alive and in
// what state". The ProcessManager writes it atomically (temp file + rename) on
// every state transition; a lost write leaves a stale status that reconciliation
// repairs on the next server start via pid liveness.
type ProcessState struct {
	Name string `json:"name"`
	// Host is the origin host of a run synced from another machine (the serving
	// harness), empty for a local run. A local run's recorded pid lives in this
	// host's pid space, so pidAlive can probe it; a non-empty Host marks a
	// bounded-stale projection whose pid names a process on THAT host, so
	// EffectiveStatus and Reconcile must return/keep the synced status verbatim
	// and never run a local pid probe (a probe against a foreign pid is the
	// cross-host pid-reuse bug). omitempty keeps pre-harness rows decoding to
	// "" = local, the same compat-default style as StartTicks/BootID below.
	Host       string `json:"host,omitempty"`
	Status     string `json:"status"` // created|starting|running|stopping|stopped|crashed
	Algorithm  string `json:"algorithm"`
	PID        int    `json:"pid"`
	PGID       int    `json:"pgid"`
	ConfigPath string `json:"config_path"`
	CreatedAt  string `json:"created_at"` // RFC3339
	StartedAt  string `json:"started_at,omitempty"`
	FinishedAt string `json:"finished_at,omitempty"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	LastError  string `json:"last_error,omitempty"`
	// StartTicks is /proc/<pid>/stat field 22 (starttime, in clock ticks since
	// boot), recorded at spawn. Combined with BootID it lets pidAlive tell this
	// run's process apart from an unrelated process that later reuses the same
	// pid (after this process exits, or after a reboot resets the pid space).
	// Zero on process.json rows written before this field existed, or when
	// /proc could not be read at spawn time; pidAlive falls back to a bare pid
	// liveness probe in that case (documented compatibility gap).
	StartTicks int64 `json:"start_ticks,omitempty"`
	// BootID is /proc/stat's "btime" (boot time, seconds since the epoch),
	// recorded at spawn. It disambiguates a StartTicks collision across a
	// reboot, since the clock-tick counter resets at boot and the same tick
	// count can recur for an unrelated process in a later boot.
	BootID int64 `json:"boot_id,omitempty"`
}

// Status constants for ProcessState.Status.
const (
	StatusCreated  = "created"
	StatusStarting = "starting"
	StatusRunning  = "running"
	StatusStopping = "stopping"
	StatusStopped  = "stopped"
	StatusCrashed  = "crashed"
)

// WriteProcessState writes st to runDir/process.json atomically: it writes a
// temp file, fsyncs it, then renames over the destination (an atomic operation
// on POSIX). The run directory is created if missing.
func WriteProcessState(runDir string, st *ProcessState) error {
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	tmp := filepath.Join(runDir, processStateFile+".tmp")
	final := filepath.Join(runDir, processStateFile)

	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// ReadProcessState reads and decodes runDir/process.json.
func ReadProcessState(runDir string) (*ProcessState, error) {
	data, err := os.ReadFile(filepath.Join(runDir, processStateFile))
	if err != nil {
		return nil, err
	}
	var st ProcessState
	if err := json.Unmarshal(data, &st); err != nil {
		return nil, err
	}
	return &st, nil
}

// ScanProcessStates reads every runs/*/process.json under runsDir. Directories
// without a process.json (or with an unreadable one) are skipped. The error is
// non-nil only if runsDir itself cannot be listed.
func ScanProcessStates(runsDir string) ([]*ProcessState, error) {
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, err
	}
	var out []*ProcessState
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		st, err := ReadProcessState(filepath.Join(runsDir, e.Name()))
		if err != nil {
			continue
		}
		out = append(out, st)
	}
	return out, nil
}

// NowRFC3339 returns the current UTC time formatted as RFC3339.
func NowRFC3339() string {
	return time.Now().UTC().Format(time.RFC3339)
}
