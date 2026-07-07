package training

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"
)

// stopGracePeriod is how long Stop waits after SIGINT before escalating to
// SIGKILL for the process group.
const stopGracePeriod = 30 * time.Second

// Sentinel errors returned by the ProcessManager so callers (the HTTP layer in
// S1T7) can map them to status codes via errors.Is.
var (
	// ErrInvalidName is returned when a run name fails validation (path-traversal
	// guard). Not force-overridable.
	ErrInvalidName = errors.New("invalid run name")
	// ErrNameCollision is returned by Create when a run of that name already
	// has a process.json. Not force-overridable.
	ErrNameCollision = errors.New("run name already exists")
	// ErrConfigMissing is returned by Start/Resume when no config.yaml exists.
	ErrConfigMissing = errors.New("run config not found")
	// ErrUnsupportedAlgorithm is returned when the algorithm has no subcommand mapping.
	ErrUnsupportedAlgorithm = errors.New("unsupported algorithm")
	// ErrAlreadyRunning is returned by Start/Resume when the run is already live.
	ErrAlreadyRunning = errors.New("run already running")
)

// runNameRe is the strict run-name allowlist. Combined with the explicit ".."
// and "/" reject in validateName it forms the path-traversal guard.
var runNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// validateName enforces the run-name allowlist and rejects any name containing
// ".." or a path separator before any filesystem operation touches it.
func validateName(name string) error {
	if strings.Contains(name, "..") || strings.Contains(name, "/") {
		return fmt.Errorf("%w: %q contains a path separator or ..", ErrInvalidName, name)
	}
	if !runNameRe.MatchString(name) {
		return fmt.Errorf("%w: %q must match %s", ErrInvalidName, name, runNameRe.String())
	}
	return nil
}

// algoSubcommand maps an algorithm identifier to the cambia CLI subcommand.
func algoSubcommand(algo string) ([]string, error) {
	switch algo {
	case "prt-cfr", "prtcfr":
		return []string{"train", "prtcfr"}, nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedAlgorithm, algo)
	}
}

// CreateRequest is the input to ProcessManager.Create.
type CreateRequest struct {
	// Name is the run name; it is validated before any filesystem operation.
	Name string
	// Algorithm selects the CLI subcommand (e.g. "prt-cfr").
	Algorithm string
	// ConfigPath points to a materialized config.yaml. If set and outside the
	// run directory it is copied to runs/<name>/config.yaml; if it already is
	// that path it is adopted in place. May be empty, in which case the caller
	// is expected to have written runs/<name>/config.yaml directly.
	ConfigPath string
}

// StartOpts carries per-launch options for Start and Resume.
type StartOpts struct {
	// ExtraArgs are appended verbatim to the spawn argument list. Preflight
	// options (force, VRAM/disk thresholds) live in the HTTP layer, not here.
	ExtraArgs []string
}

// managedProc is the in-memory handle for a subprocess this server instance
// spawned. Processes inherited across a restart have no handle and are handled
// only through process.json + Reconcile.
type managedProc struct {
	name          string
	cmd           *exec.Cmd
	pgid          int
	stopRequested bool
	killTimer     *time.Timer
	done          chan struct{}
}

// ProcessManager supervises detached cambia training subprocesses. It owns the
// process.json current-state store and the lifecycle of processes it spawns.
type ProcessManager struct {
	runsDir   string
	cfrDir    string
	cambiaBin string
	// store is reserved for run_db name-collision checks performed by the S1T7
	// HTTP wiring; the supervisor itself does not read it.
	store *TrainingStore

	mu    sync.Mutex
	procs map[string]*managedProc
}

// NewProcessManager constructs a ProcessManager. runsDir is the runs root,
// cfrDir is the working directory for spawned subprocesses, cambiaBin is the
// cambia executable, and store is the read-only run database (may be nil).
func NewProcessManager(runsDir, cfrDir, cambiaBin string, store *TrainingStore) *ProcessManager {
	return &ProcessManager{
		runsDir:   runsDir,
		cfrDir:    cfrDir,
		cambiaBin: cambiaBin,
		store:     store,
		procs:     make(map[string]*managedProc),
	}
}

// runDir returns the absolute-or-relative run directory for name. Callers must
// have validated name first.
func (m *ProcessManager) runDir(name string) string {
	return filepath.Join(m.runsDir, name)
}

// Create validates the name, rejects a run that already has a process.json,
// makes runs/<name>/logs/, materializes the config into runs/<name>/config.yaml,
// and writes process.json with status "created".
func (m *ProcessManager) Create(req CreateRequest) (*ProcessState, error) {
	if err := validateName(req.Name); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	runDir := m.runDir(req.Name)
	if _, err := os.Stat(filepath.Join(runDir, processStateFile)); err == nil {
		return nil, fmt.Errorf("%w: %s", ErrNameCollision, req.Name)
	}
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return nil, err
	}

	configPath := filepath.Join(runDir, "config.yaml")
	if req.ConfigPath != "" {
		absSrc, _ := filepath.Abs(req.ConfigPath)
		absDst, _ := filepath.Abs(configPath)
		if absSrc != absDst {
			if err := copyFile(req.ConfigPath, configPath); err != nil {
				return nil, fmt.Errorf("materialize config: %w", err)
			}
		}
	}
	absConfig, err := filepath.Abs(configPath)
	if err != nil {
		absConfig = configPath
	}

	st := &ProcessState{
		Name:       req.Name,
		Status:     StatusCreated,
		Algorithm:  req.Algorithm,
		ConfigPath: absConfig,
		CreatedAt:  nowRFC3339(),
	}
	if err := writeProcessState(runDir, st); err != nil {
		return nil, err
	}
	return st, nil
}

// Start spawns the training subprocess for a created run.
func (m *ProcessManager) Start(name string, opts StartOpts) (*ProcessState, error) {
	return m.launch(name, opts, false)
}

// Resume spawns the training subprocess with --resume; the caller gates
// resumability (checkpoint presence).
func (m *ProcessManager) Resume(name string, opts StartOpts) (*ProcessState, error) {
	return m.launch(name, opts, true)
}

// launch is the shared spawn path for Start and Resume.
func (m *ProcessManager) launch(name string, opts StartOpts, resume bool) (*ProcessState, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.procs[name]; ok {
		return nil, fmt.Errorf("%w: %s", ErrAlreadyRunning, name)
	}

	runDir := m.runDir(name)
	st, err := readProcessState(runDir)
	if err != nil {
		return nil, fmt.Errorf("read process state for %q: %w", name, err)
	}

	switch st.Status {
	case StatusRunning, StatusStarting, StatusStopping:
		if pidAlive(st.PID) {
			return nil, fmt.Errorf("%w: %s is %s", ErrAlreadyRunning, name, st.Status)
		}
	}

	configPath := st.ConfigPath
	if configPath == "" {
		configPath = filepath.Join(runDir, "config.yaml")
	}
	absConfig, err := filepath.Abs(configPath)
	if err != nil {
		absConfig = configPath
	}
	if _, err := os.Stat(absConfig); err != nil {
		return nil, fmt.Errorf("%w: %s", ErrConfigMissing, absConfig)
	}

	absRunDir, err := filepath.Abs(runDir)
	if err != nil {
		absRunDir = runDir
	}

	sub, err := algoSubcommand(st.Algorithm)
	if err != nil {
		return nil, err
	}
	args := make([]string, 0, len(sub)+8+len(opts.ExtraArgs))
	args = append(args, sub...)
	args = append(args, "--config", absConfig, "--run-name", name, "--save-path", absRunDir)
	if resume {
		args = append(args, "--resume")
	}
	args = append(args, opts.ExtraArgs...)

	logDir := filepath.Join(runDir, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, err
	}
	logFile, err := os.OpenFile(filepath.Join(logDir, "training.log"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}

	cmd := exec.Command(m.cambiaBin, args...)
	cmd.Dir = m.cfrDir
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("start %q: %w", name, err)
	}
	// The child has its own dup of the log fd; drop the parent's copy.
	logFile.Close()

	pid := cmd.Process.Pid
	// With Setpgid and no explicit Pgid, the child leads a new group whose id
	// equals its pid.
	pgid := pid

	p := &managedProc{
		name: name,
		cmd:  cmd,
		pgid: pgid,
		done: make(chan struct{}),
	}
	m.procs[name] = p

	_ = m.mutateStateLocked(name, func(s *ProcessState) {
		s.Status = StatusRunning
		s.PID = pid
		s.PGID = pgid
		s.StartedAt = nowRFC3339()
		s.FinishedAt = ""
		s.ExitCode = nil
		s.LastError = ""
	})

	go m.waitFor(p)

	return readProcessState(runDir)
}

// waitFor blocks on the subprocess exit, then records the terminal state.
func (m *ProcessManager) waitFor(p *managedProc) {
	err := p.cmd.Wait()

	m.mu.Lock()
	defer m.mu.Unlock()

	if p.killTimer != nil {
		p.killTimer.Stop()
	}

	code := exitCodeFromErr(err)
	_ = m.mutateStateLocked(p.name, func(st *ProcessState) {
		st.ExitCode = &code
		st.FinishedAt = nowRFC3339()
		if p.stopRequested || code == 0 {
			st.Status = StatusStopped
		} else {
			st.Status = StatusCrashed
			st.LastError = fmt.Sprintf("exited with code %d", code)
		}
	})

	delete(m.procs, p.name)
	close(p.done)
}

// Stop requests a graceful shutdown: SIGINT to the process group, escalating to
// SIGKILL after stopGracePeriod. force sends SIGKILL immediately. The stop is
// recorded so the wait goroutine reports "stopped" rather than "crashed".
func (m *ProcessManager) Stop(name string, force bool) (*ProcessState, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	p, ok := m.procs[name]
	if !ok {
		// Not supervised by this instance: it already exited (terminal state on
		// disk) or was never started. Return the on-disk state; if a stale live
		// pid is recorded, signal its group best-effort.
		st, err := readProcessState(m.runDir(name))
		if err != nil {
			return nil, fmt.Errorf("read process state for %q: %w", name, err)
		}
		if pidAlive(st.PID) && st.PGID > 0 {
			sig := syscall.SIGINT
			if force {
				sig = syscall.SIGKILL
			}
			_ = syscall.Kill(-st.PGID, sig)
		}
		return st, nil
	}

	p.stopRequested = true
	if err := m.mutateStateLocked(name, func(st *ProcessState) {
		st.Status = StatusStopping
	}); err != nil {
		return nil, err
	}

	sig := syscall.SIGINT
	if force {
		sig = syscall.SIGKILL
	}
	_ = syscall.Kill(-p.pgid, sig)

	if !force {
		pgid := p.pgid
		p.killTimer = time.AfterFunc(stopGracePeriod, func() {
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
		})
	}

	return readProcessState(m.runDir(name))
}

// GetState returns the on-disk process state for name and whether it was found.
func (m *ProcessManager) GetState(name string) (*ProcessState, bool) {
	st, err := readProcessState(m.runDir(name))
	if err != nil {
		return nil, false
	}
	return st, true
}

// Reconcile runs at server start: any process.json still marked running,
// starting, or stopping whose recorded pid is no longer alive is flipped to
// crashed. It never spawns; it only repairs stale state.
func (m *ProcessManager) Reconcile() {
	states, err := scanProcessStates(m.runsDir)
	if err != nil {
		return
	}
	for _, st := range states {
		switch st.Status {
		case StatusRunning, StatusStarting, StatusStopping:
			if pidAlive(st.PID) {
				continue
			}
			st.Status = StatusCrashed
			st.LastError = "reconciled: process not alive at server start"
			if st.FinishedAt == "" {
				st.FinishedAt = nowRFC3339()
			}
			_ = writeProcessState(m.runDir(st.Name), st)
		}
	}
}

// mutateStateLocked reads, applies fn to, and atomically rewrites the process
// state for name. Callers must hold m.mu.
func (m *ProcessManager) mutateStateLocked(name string, fn func(*ProcessState)) error {
	runDir := m.runDir(name)
	st, err := readProcessState(runDir)
	if err != nil {
		return err
	}
	fn(st)
	return writeProcessState(runDir, st)
}

// pidAlive reports whether pid names a live process (signal 0 probe).
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

// exitCodeFromErr extracts the exit code from a cmd.Wait error: 0 for nil, the
// process exit code for an ExitError (-1 when terminated by a signal), and -1
// for any other error.
func exitCodeFromErr(err error) int {
	if err == nil {
		return 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return ee.ExitCode()
	}
	return -1
}

// copyFile copies src to dst, truncating dst if it exists.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
