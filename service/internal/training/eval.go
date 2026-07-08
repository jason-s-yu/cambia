package training

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Sentinel errors returned by EvalManager.Trigger so the HTTP layer can map them
// to status codes via errors.Is.
var (
	// ErrNoCheckpoint is returned when the run directory holds no evaluable
	// checkpoint (no *.pt under checkpoints/ or snapshots/). Maps to 404.
	ErrNoCheckpoint = errors.New("no evaluable checkpoint")
	// ErrEvalCapReached is returned when the manager is already running its
	// configured maximum number of concurrent evals. Maps to 409. This cap is
	// EvalManager-internal and independent of the training ProcessManager cap.
	ErrEvalCapReached = errors.New("eval concurrency cap reached")
)

const (
	// evalTailLines is how many trailing log lines Jobs() attaches to each job.
	evalTailLines = 40
	// defaultMaxConcurrentEvals is the manager's cap before SetMaxConcurrent is
	// called. Unlike ProcessManager (which defaults its backstop off), eval
	// defaults to one at a time: a cuda eval contends for the GPU and even a cpu
	// eval competes for run_db and host cores, so serial is the safe default.
	defaultMaxConcurrentEvals = 1
	// defaultEvalGames is the game count used when a trigger omits games (games
	// <= 0). It matches the CLI's run-dir-mode default so a service-launched eval
	// behaves like `cambia evaluate <run_dir>` with no --games.
	defaultEvalGames = 5000
)

// EvalOpts carries the per-trigger request. Epoch nil selects --latest; a set
// Epoch selects --epoch N. Device empty falls back to cpu; Games <= 0 falls back
// to defaultEvalGames.
type EvalOpts struct {
	Epoch  *int
	Device string
	Games  int
	Argmax bool
}

// EvalManager supervises `cambia evaluate` subprocesses. It owns an in-memory,
// per-server-lifetime job registry (NOT process.json: an eval is a different
// entity answering a different question than a training run, so it never writes
// the training current-state store). Eval children are plain processes, not
// detached process groups: an eval is minutes and persists results only at the
// end, so a server restart that kills one loses at most wasted compute, never a
// committed result.
type EvalManager struct {
	runsDir   string
	cfrDir    string
	cambiaBin string

	mu            sync.Mutex
	reg           *jobRegistry
	running       int    // count of live eval children this manager is supervising
	maxConcurrent int    // <= 0 disables the cap
	counter       uint64 // monotonic, for per-manager unique job ids
}

// NewEvalManager constructs an EvalManager. runsDir is the runs root, cfrDir is
// the working directory for spawned evals, and cambiaBin is the cambia
// executable. The concurrency cap defaults to defaultMaxConcurrentEvals; call
// SetMaxConcurrent to change it.
func NewEvalManager(runsDir, cfrDir, cambiaBin string) *EvalManager {
	return &EvalManager{
		runsDir:       runsDir,
		cfrDir:        cfrDir,
		cambiaBin:     cambiaBin,
		reg:           newJobRegistry(),
		maxConcurrent: defaultMaxConcurrentEvals,
	}
}

// SetMaxConcurrent sets the cap on concurrent evals this manager supervises.
// n <= 0 disables the cap. Checked and enforced atomically under m.mu in
// Trigger, so two concurrent triggers cannot both slip past a cap of 1.
func (m *EvalManager) SetMaxConcurrent(n int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maxConcurrent = n
}

// Trigger validates the run name, gates on an evaluable checkpoint, enforces the
// concurrency cap, spawns `cambia evaluate <absRunDir> ...` with output
// redirected to runs/<name>/logs/eval_<id>.log, and launches a wait goroutine
// that records the terminal status. It returns a snapshot of the job as it stood
// at spawn (status running); callers observe later transitions via Jobs.
//
// The cap slot is reserved (m.running incremented) under m.mu before any
// filesystem or process I/O runs, then released on any failure path; this
// keeps the atomic cap check/reservation on the lock while letting
// MkdirAll/OpenFile/cmd.Start (which touch only this call's unique per-id log
// path) run unlocked, so concurrent triggers for different runs don't
// serialize behind that I/O.
func (m *EvalManager) Trigger(name string, opts EvalOpts) (*EvalJob, error) {
	if err := validateName(name); err != nil {
		return nil, err
	}
	runDir := filepath.Join(m.runsDir, name)
	if !hasEvaluableCheckpoint(runDir) {
		return nil, ErrNoCheckpoint
	}

	device := opts.Device
	if device == "" {
		device = "cpu"
	}
	games := opts.Games
	if games <= 0 {
		games = defaultEvalGames
	}

	m.mu.Lock()
	// Atomic cap check + reservation: mirror ProcessManager.launch so two
	// concurrent triggers cannot both pass a cap of 1.
	if m.maxConcurrent > 0 && m.running >= m.maxConcurrent {
		m.mu.Unlock()
		return nil, ErrEvalCapReached
	}
	m.running++
	m.counter++
	id := fmt.Sprintf("%s-%d", time.Now().UTC().Format("20060102T150405"), m.counter)
	m.mu.Unlock()

	releaseSlot := func() {
		m.mu.Lock()
		m.running--
		m.mu.Unlock()
	}

	target := "latest"
	if opts.Epoch != nil {
		target = fmt.Sprintf("iter:%d", *opts.Epoch)
	}

	logDir := filepath.Join(runDir, "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		releaseSlot()
		return nil, err
	}
	logPath := filepath.Join(logDir, "eval_"+id+".log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		releaseSlot()
		return nil, err
	}

	absRunDir, err := filepath.Abs(runDir)
	if err != nil {
		absRunDir = runDir
	}

	args := []string{"evaluate", absRunDir, "--device", device, "--games", strconv.Itoa(games)}
	if opts.Epoch != nil {
		args = append(args, "--epoch", strconv.Itoa(*opts.Epoch))
	} else {
		args = append(args, "--latest")
	}
	if opts.Argmax {
		args = append(args, "--argmax")
	}

	cmd := exec.Command(m.cambiaBin, args...)
	cmd.Dir = m.cfrDir
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	job := &EvalJob{
		ID:        id,
		Run:       name,
		Status:    EvalRunning,
		Target:    target,
		Device:    device,
		Games:     games,
		Argmax:    opts.Argmax,
		LogPath:   logPath,
		StartedAt: nowRFC3339(),
	}

	if err := cmd.Start(); err != nil {
		logFile.Close()
		code := -1
		job.Status = EvalFailed
		job.ExitCode = &code
		job.Error = err.Error()
		job.FinishedAt = nowRFC3339()
		m.mu.Lock()
		m.running-- // the reserved slot never actually ran
		m.reg.add(name, job)
		m.mu.Unlock()
		return nil, fmt.Errorf("start eval %q: %w", name, err)
	}
	// The child holds its own dup of the log fd; drop the parent's copy.
	logFile.Close()

	m.mu.Lock()
	m.reg.add(name, job)
	m.mu.Unlock()

	snap := *job
	go m.waitFor(job, cmd)
	return &snap, nil
}

// waitFor blocks on the eval subprocess exit, then records the terminal state
// under m.mu and frees the concurrency slot.
func (m *EvalManager) waitFor(job *EvalJob, cmd *exec.Cmd) {
	err := cmd.Wait()
	code := exitCodeFromErr(err)

	m.mu.Lock()
	defer m.mu.Unlock()

	job.ExitCode = &code
	job.FinishedAt = nowRFC3339()
	if code == 0 {
		job.Status = EvalSucceeded
	} else {
		job.Status = EvalFailed
		if err != nil {
			job.Error = err.Error()
		} else {
			job.Error = fmt.Sprintf("exited with code %d", code)
		}
	}
	if m.running > 0 {
		m.running--
	}
}

// Jobs returns name's eval jobs newest-first, each with Tail filled from the
// last evalTailLines lines of its log. Job structs are snapshotted under the
// lock so the log reads happen off-lock (tail I/O never blocks Trigger).
func (m *EvalManager) Jobs(name string) []*EvalJob {
	m.mu.Lock()
	stored := m.reg.list(name)
	snaps := make([]EvalJob, len(stored))
	for i, j := range stored {
		snaps[i] = *j
	}
	m.mu.Unlock()

	out := make([]*EvalJob, len(snaps))
	for i := range snaps {
		if snaps[i].LogPath != "" {
			if lines, _, err := readLastNLines(snaps[i].LogPath, evalTailLines); err == nil {
				snaps[i].Tail = lines
			}
		}
		out[i] = &snaps[i]
	}
	return out
}

// hasEvaluableCheckpoint reports whether runDir holds a checkpoint the CLI can
// evaluate: any *.pt file under checkpoints/ (legacy-named runs) or snapshots/
// (PRT-CFR writes its rolling checkpoint there). The eval CLI auto-detects the
// agent type from the run dir, so this is deliberately algorithm-agnostic.
func hasEvaluableCheckpoint(runDir string) bool {
	for _, sub := range []string{"checkpoints", "snapshots"} {
		entries, err := os.ReadDir(filepath.Join(runDir, sub))
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".pt") {
				return true
			}
		}
	}
	return false
}
