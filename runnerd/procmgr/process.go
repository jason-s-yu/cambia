package procmgr

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// killGroupFunc sends signal sig to the process group led by pgid (the
// kill(2) negative-pgid convention). It is a seam: tests override it to
// observe Stop's signal-or-not decision without risking a real signal to a
// live process group, which could be the test binary's own group when a test
// fakes process.json with the test process's own pid.
var killGroupFunc = func(pgid int, sig syscall.Signal) error {
	return syscall.Kill(-pgid, sig)
}

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
	// ErrConcurrencyCapReached is the hard backstop returned by launch when the
	// in-flight count this manager itself is supervising is already at the
	// configured cap. It is checked and incremented atomically under m.mu, so
	// it closes the TOCTOU window the disk-scanning preflight ConcurrencyCapCheck
	// cannot: two concurrent launches racing the same disk snapshot.
	ErrConcurrencyCapReached = errors.New("concurrency cap reached")
)

// runNameRe is the strict run-name allowlist. Combined with the explicit ".."
// and "/" reject in ValidateName it forms the path-traversal guard.
var runNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// ValidateName enforces the run-name allowlist and rejects any name containing
// ".." or a path separator before any filesystem operation touches it.
func ValidateName(name string) error {
	if strings.Contains(name, "..") || strings.Contains(name, "/") {
		return fmt.Errorf("%w: %q contains a path separator or ..", ErrInvalidName, name)
	}
	if !runNameRe.MatchString(name) {
		return fmt.Errorf("%w: %q must match %s", ErrInvalidName, name, runNameRe.String())
	}
	return nil
}

// RunResolver supplies the run-directory layout and the pid-liveness-aware
// status view that a ProcessManager's host provides. It severs the manager's
// dependency on any concrete run store: the dashboard service injects its
// TrainingStore and runnerd injects its own thin store, each implementing these
// two methods. The manager holds the resolver for host-directed lookups; the
// supervisor's own launch/stop path uses its runsDir directly.
type RunResolver interface {
	// RunDir returns the run directory for a validated run name.
	RunDir(name string) string
	// EffectiveStatus returns st.Status with pid liveness applied.
	EffectiveStatus(st *ProcessState) string
}

// TrainAlgorithms returns the train-only algorithm to cambia-subcommand
// allowlist the dashboard service registers with a ProcessManager: PRT-CFR
// under both its canonical id and its no-dash alias. Callers own the table;
// runnerd registers its own (super)set of job kinds. A fresh map is returned on
// every call so callers cannot mutate a shared instance.
func TrainAlgorithms() map[string][]string {
	return map[string][]string{
		"prt-cfr": {"train", "prtcfr"},
		"prtcfr":  {"train", "prtcfr"},
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
	// resolver is the injected host store (dashboard TrainingStore or runnerd's
	// own). It severs the type dependency the old *TrainingStore field created;
	// the supervisor itself does not read it in this milestone, reserving it for
	// host-directed lookups. May be nil.
	resolver RunResolver
	// algos is the injected algorithm to cambia-subcommand allowlist. A launch
	// for an unregistered algorithm returns ErrUnsupportedAlgorithm; callers pick
	// which job kinds this manager will spawn (see TrainAlgorithms).
	algos map[string][]string

	mu            sync.Mutex
	procs         map[string]*managedProc
	maxConcurrent int
}

// NewProcessManager constructs a ProcessManager. runsDir is the runs root,
// cfrDir is the working directory for spawned subprocesses, cambiaBin is the
// cambia executable, resolver is the injected host store (may be nil), and algos
// is the algorithm to cambia-subcommand allowlist (see TrainAlgorithms).
// maxConcurrent defaults to 0 (disabled); call SetMaxConcurrent to enable the
// hard-backstop cap.
func NewProcessManager(runsDir, cfrDir, cambiaBin string, resolver RunResolver, algos map[string][]string) *ProcessManager {
	return &ProcessManager{
		runsDir:   runsDir,
		cfrDir:    cfrDir,
		cambiaBin: cambiaBin,
		resolver:  resolver,
		algos:     algos,
		procs:     make(map[string]*managedProc),
	}
}

// AlgoSubcommand maps an algorithm identifier to its cambia CLI subcommand via
// the constructor-injected allowlist. An unregistered algorithm is rejected with
// ErrUnsupportedAlgorithm (the reject-unknown discipline is unchanged; only the
// table is now injected rather than hardcoded).
func (m *ProcessManager) AlgoSubcommand(algo string) ([]string, error) {
	sub, ok := m.algos[algo]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedAlgorithm, algo)
	}
	return sub, nil
}

// SetMaxConcurrent sets the hard-backstop cap on processes this manager
// instance is actively supervising (len(m.procs)). max <= 0 disables it. This
// is the authority enforced atomically under m.mu inside launch; the disk-
// scanning preflight ConcurrencyCapCheck is the advisory, user-facing 409 and
// cannot by itself close the race between two concurrent launch calls.
func (m *ProcessManager) SetMaxConcurrent(max int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maxConcurrent = max
}

// KillAll force-terminates every process group this manager is currently
// supervising with SIGKILL and no grace period. It is the abrupt-shutdown path;
// callers wanting a graceful stop use Stop per run. Safe to call when no
// processes are live. The wait goroutines still record each terminal state.
func (m *ProcessManager) KillAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.procs {
		_ = killGroupFunc(p.pgid, syscall.SIGKILL)
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
	if err := ValidateName(req.Name); err != nil {
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
		CreatedAt:  NowRFC3339(),
	}
	if err := WriteProcessState(runDir, st); err != nil {
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
	if err := ValidateName(name); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.procs[name]; ok {
		return nil, fmt.Errorf("%w: %s", ErrAlreadyRunning, name)
	}

	// Hard backstop: count-then-launch happens atomically under m.mu, so two
	// concurrent launches for different run names cannot both slip past the
	// cap (the TOCTOU window the disk-scanning preflight check alone allows).
	if m.maxConcurrent > 0 && len(m.procs) >= m.maxConcurrent {
		return nil, fmt.Errorf("%w: %d/%d", ErrConcurrencyCapReached, len(m.procs), m.maxConcurrent)
	}

	runDir := m.runDir(name)
	st, err := ReadProcessState(runDir)
	if err != nil {
		return nil, fmt.Errorf("read process state for %q: %w", name, err)
	}

	switch st.Status {
	case StatusRunning, StatusStarting, StatusStopping:
		if pidAlive(st) {
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

	sub, err := m.AlgoSubcommand(st.Algorithm)
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

	// Record starttime/btime now, while the pid is freshly ours, so a later
	// liveness check (pidAlive) can tell this exact process apart from
	// whatever the kernel eventually recycles pid for. A read failure here
	// (non-Linux, or a very fast child exit) leaves both at zero, which
	// pidAlive treats as "no verification available" and falls back to the
	// bare pid probe.
	startTicks, _ := readProcStarttime(pid)
	bootID, _ := readBootTime()

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
		s.StartedAt = NowRFC3339()
		s.FinishedAt = ""
		s.ExitCode = nil
		s.LastError = ""
		s.StartTicks = startTicks
		s.BootID = bootID
	})

	go m.waitFor(p)

	return ReadProcessState(runDir)
}

// waitFor blocks on the subprocess exit, then records the terminal state.
func (m *ProcessManager) waitFor(p *managedProc) {
	err := p.cmd.Wait()

	m.mu.Lock()
	defer m.mu.Unlock()

	if p.killTimer != nil {
		p.killTimer.Stop()
	}

	code := ExitCodeFromErr(err)
	_ = m.mutateStateLocked(p.name, func(st *ProcessState) {
		st.ExitCode = &code
		st.FinishedAt = NowRFC3339()
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
	if err := ValidateName(name); err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	p, ok := m.procs[name]
	if !ok {
		// Not supervised by this instance: it already exited (terminal state on
		// disk) or was never started. Return the on-disk state; if a stale live
		// pid is recorded, signal its group best-effort.
		st, err := ReadProcessState(m.runDir(name))
		if err != nil {
			return nil, fmt.Errorf("read process state for %q: %w", name, err)
		}
		if pidAlive(st) && st.PGID > 0 {
			sig := syscall.SIGINT
			if force {
				sig = syscall.SIGKILL
			}
			_ = killGroupFunc(st.PGID, sig)
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
	_ = killGroupFunc(p.pgid, sig)

	if !force {
		pgid := p.pgid
		runDir := m.runDir(name)
		p.killTimer = time.AfterFunc(stopGracePeriod, func() {
			// Re-check liveness (starttime-validated) before escalating: by the
			// time the grace period elapses the child may already have been
			// reaped, or -- under enough pid churn -- pgid may have been
			// recycled for an unrelated process group. Only SIGKILL a pid we
			// can still positively identify as this run's.
			st, err := ReadProcessState(runDir)
			if err != nil || !pidAlive(st) {
				return
			}
			_ = killGroupFunc(pgid, syscall.SIGKILL)
		})
	}

	return ReadProcessState(m.runDir(name))
}

// GetState returns the on-disk process state for name and whether it was found.
func (m *ProcessManager) GetState(name string) (*ProcessState, bool) {
	st, err := ReadProcessState(m.runDir(name))
	if err != nil {
		return nil, false
	}
	return st, true
}

// Reconcile runs at server start: any process.json still marked running,
// starting, or stopping whose recorded pid is no longer alive is flipped to
// crashed. It never spawns; it only repairs stale state.
func (m *ProcessManager) Reconcile() {
	states, err := ScanProcessStates(m.runsDir)
	if err != nil {
		return
	}
	for _, st := range states {
		switch st.Status {
		case StatusRunning, StatusStarting, StatusStopping:
			if pidAlive(st) {
				continue
			}
			st.Status = StatusCrashed
			st.LastError = "reconciled: process not alive at server start"
			if st.FinishedAt == "" {
				st.FinishedAt = NowRFC3339()
			}
			_ = WriteProcessState(m.runDir(st.Name), st)
		}
	}
}

// mutateStateLocked reads, applies fn to, and atomically rewrites the process
// state for name. Callers must hold m.mu.
func (m *ProcessManager) mutateStateLocked(name string, fn func(*ProcessState)) error {
	runDir := m.runDir(name)
	st, err := ReadProcessState(runDir)
	if err != nil {
		return err
	}
	fn(st)
	return WriteProcessState(runDir, st)
}

// pidAlive reports whether st's pid names a live process that is plausibly
// the same process this manager recorded: the pid must answer a signal(0)
// probe, and when StartTicks was recorded at spawn, the pid's current
// /proc/<pid>/stat starttime (and, when BootID was recorded, the current
// boot's btime) must match. A pid the kernel has since recycled for an
// unrelated process -- after this run's process exited, or after a reboot --
// is treated as not alive rather than a false positive. process.json rows
// written before these fields existed have StartTicks == 0 and fall back to
// the bare pid probe (documented compatibility gap).
func pidAlive(st *ProcessState) bool {
	if st == nil || st.PID <= 0 {
		return false
	}
	proc, err := os.FindProcess(st.PID)
	if err != nil {
		return false
	}
	if proc.Signal(syscall.Signal(0)) != nil {
		return false
	}
	if st.StartTicks == 0 {
		return true
	}
	ticks, err := readProcStarttime(st.PID)
	if err != nil {
		// Can no longer positively verify identity (e.g. the process exited in
		// the gap between the signal probe and this read): treat as not alive
		// rather than risk reporting a stale/reused pid as live.
		return false
	}
	if ticks != st.StartTicks {
		return false
	}
	if st.BootID != 0 {
		if boot, err := readBootTime(); err == nil && boot != st.BootID {
			return false
		}
	}
	return true
}

// readProcStarttime reads field 22 (starttime, clock ticks since boot) from
// /proc/<pid>/stat. The comm field (2) is parenthesized and may itself
// contain spaces or parentheses, so parsing anchors on the last ')' before
// splitting the remaining whitespace-separated fields.
func readProcStarttime(pid int) (int64, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0, err
	}
	s := string(data)
	idx := strings.LastIndexByte(s, ')')
	if idx < 0 || idx+2 > len(s) {
		return 0, fmt.Errorf("unparseable /proc/%d/stat", pid)
	}
	// fields[0] here is field 3 (state); starttime is field 22, i.e. index 19.
	fields := strings.Fields(s[idx+2:])
	const starttimeIdx = 22 - 3
	if len(fields) <= starttimeIdx {
		return 0, fmt.Errorf("too few fields in /proc/%d/stat", pid)
	}
	return strconv.ParseInt(fields[starttimeIdx], 10, 64)
}

// readBootTime reads the system boot time (seconds since the epoch) from the
// "btime" line of /proc/stat. Paired with a recorded StartTicks, it
// disambiguates a starttime collision that recurs across a reboot.
func readBootTime() (int64, error) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(line, "btime "); ok {
			return strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		}
	}
	return 0, fmt.Errorf("btime not found in /proc/stat")
}

// ExitCodeFromErr extracts the exit code from a cmd.Wait error: 0 for nil, the
// process exit code for an ExitError (-1 when terminated by a signal), and -1
// for any other error.
func ExitCodeFromErr(err error) int {
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
