package training

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
	Name       string `json:"name"`
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

// writeProcessState writes st to runDir/process.json atomically: it writes a
// temp file, fsyncs it, then renames over the destination (an atomic operation
// on POSIX). The run directory is created if missing.
func writeProcessState(runDir string, st *ProcessState) error {
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

// readProcessState reads and decodes runDir/process.json.
func readProcessState(runDir string) (*ProcessState, error) {
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

// scanProcessStates reads every runs/*/process.json under runsDir. Directories
// without a process.json (or with an unreadable one) are skipped. The error is
// non-nil only if runsDir itself cannot be listed.
func scanProcessStates(runsDir string) ([]*ProcessState, error) {
	entries, err := os.ReadDir(runsDir)
	if err != nil {
		return nil, err
	}
	var out []*ProcessState
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		st, err := readProcessState(filepath.Join(runsDir, e.Name()))
		if err != nil {
			continue
		}
		out = append(out, st)
	}
	return out, nil
}

// nowRFC3339 returns the current UTC time formatted as RFC3339.
func nowRFC3339() string {
	return time.Now().UTC().Format(time.RFC3339)
}
