package harness

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Dispatcher errors surfaced to the HTTP layer.
var (
	// ErrQueueFull is returned by Submit/Resume when the in-memory queue is at
	// its depth cap (HTTP 429).
	ErrQueueFull = errors.New("job queue is full")
	// ErrNotFound is returned when no job of that name exists on disk.
	ErrNotFound = errors.New("job not found")
	// ErrNotTerminal is returned by Purge for a job that is not in a terminal
	// state (purge refuses to remove a live run dir).
	ErrNotTerminal = errors.New("job is not terminal")
	// ErrNoResumableState is returned by Resume when resume_state.json or the
	// rolling checkpoint is absent.
	ErrNoResumableState = errors.New("no resumable state")
	// ErrAlreadyQueued is returned by Resume when a job of that name is already
	// queued or preparing.
	ErrAlreadyQueued = errors.New("job already queued")
)

// job is the in-memory handle for a queued or preparing job. Once launched it is
// removed from the pending set and its state is read from process.json; once
// terminal its runnerd-level state (canceled/failed) is persisted there too.
type job struct {
	spec     JobSpec
	state    string // StateQueued | StatePreparing
	resume   bool
	canceled bool
	submitAt string
	ctx      context.Context
	cancel   context.CancelFunc
}

// Dispatcher owns the FIFO queue, the concurrency schedule, and the runnerd
// state machine over procmgr. It is the single writer of runnerd-level state
// transitions and the source of the queue/health snapshots.
type Dispatcher struct {
	pm       *procmgr.ProcessManager
	env      Environment
	runsDir  string
	maxJobs  int           // concurrency cap (<=0 = unlimited); mirrors SetMaxConcurrent
	maxQueue int           // queue depth cap (429 when exceeded)
	poll     time.Duration // process-liveness poll interval for monitor

	mu           sync.Mutex
	pending      map[string]*job // queued + preparing
	queue        []string        // FIFO of queued ids
	active       int             // preparing + running slots this daemon drives
	subs         map[chan QueueSnapshot]struct{}
	reconciledAt string
}

// NewDispatcher builds a Dispatcher. maxJobs is the concurrency cap (<=0
// unlimited); maxQueue bounds pending admissions; poll is the monitor's
// liveness poll interval (a sane default is substituted when non-positive).
func NewDispatcher(pm *procmgr.ProcessManager, env Environment, runsDir string, maxJobs, maxQueue int, poll time.Duration) *Dispatcher {
	if env == nil {
		env = StubEnvironment{}
	}
	if poll <= 0 {
		poll = 500 * time.Millisecond
	}
	if maxQueue <= 0 {
		maxQueue = 128
	}
	return &Dispatcher{
		pm:       pm,
		env:      env,
		runsDir:  runsDir,
		maxJobs:  maxJobs,
		maxQueue: maxQueue,
		poll:     poll,
		pending:  make(map[string]*job),
		subs:     make(map[chan QueueSnapshot]struct{}),
	}
}

func (d *Dispatcher) runDir(name string) string { return filepath.Join(d.runsDir, name) }

// Submit admits a validated spec: it writes process.json (Status=created) and
// jobspec.json, enqueues the job, and kicks the dispatcher. The caller has
// already run name/kind/path validation and the resource preflights; procmgr's
// Create re-checks the name collision atomically. Returns the job's view (queued
// or already preparing if a slot was free).
func (d *Dispatcher) Submit(spec JobSpec) (JobView, error) {
	d.mu.Lock()
	if len(d.queue) >= d.maxQueue {
		d.mu.Unlock()
		return JobView{}, ErrQueueFull
	}
	d.mu.Unlock()

	st, err := d.pm.Create(procmgr.CreateRequest{Name: spec.Name, Algorithm: spec.Kind})
	if err != nil {
		return JobView{}, err
	}
	if werr := writeJobSpec(d.runDir(spec.Name), &spec); werr != nil {
		return JobView{}, werr
	}

	ctx, cancel := context.WithCancel(context.Background())
	d.mu.Lock()
	j := &job{spec: spec, state: StateQueued, submitAt: st.CreatedAt, ctx: ctx, cancel: cancel}
	d.pending[spec.Name] = j
	d.queue = append(d.queue, spec.Name)
	d.dispatchLocked()
	view := d.pendingViewLocked(spec.Name)
	d.mu.Unlock()

	d.broadcast()
	return view, nil
}

// dispatchLocked launches queued jobs while a slot is free. Callers hold d.mu.
// It never blocks: the prepare+launch+monitor work runs in a per-job goroutine.
func (d *Dispatcher) dispatchLocked() {
	for (d.maxJobs <= 0 || d.active < d.maxJobs) && len(d.queue) > 0 {
		id := d.queue[0]
		d.queue = d.queue[1:]
		j := d.pending[id]
		if j == nil {
			continue
		}
		if j.canceled {
			delete(d.pending, id)
			continue
		}
		j.state = StatePreparing
		d.active++
		go d.runJob(j)
	}
}

// launchGateHook is a test seam invoked inside runJob after the launch options
// are built and immediately before the launch critical section, with d.mu NOT
// held. Production leaves it nil. A test sets it to inject a concurrent Cancel
// at the exact check-then-launch boundary, making the cancel/launch race
// deterministic. It mirrors procmgr's killGroupFunc seam.
var launchGateHook func(name string)

// runJob prepares, launches, and monitors one job. It is the only writer of the
// preparing -> running/failed/canceled transitions.
func (d *Dispatcher) runJob(j *job) {
	name := j.spec.Name
	launched := false
	defer func() {
		if launched {
			return
		}
		d.mu.Lock()
		d.active--
		delete(d.pending, name)
		d.dispatchLocked()
		d.mu.Unlock()
		d.broadcast()
	}()

	prepared, err := d.env.Prepare(j.ctx, name, j.spec.Commit, j.spec.Kind, j.spec.Config, j.spec.overridesStr())

	d.mu.Lock()
	canceled := j.canceled
	d.mu.Unlock()
	if canceled {
		d.markTerminal(name, StateCanceled, "canceled during preparing")
		return
	}
	if err != nil {
		d.markTerminal(name, StateFailed, "prepare failed: "+err.Error())
		return
	}
	// M3: launch from the ingest-staged environment (design 2.7) - the per-lock
	// venv interpreter, the pinned worktree cfr dir as cwd, the rendered config
	// consumed verbatim, and the staged env (PYTHONPATH pin, LIBCAMBIA_PATH,
	// src-containment shim). Job kind maps to its cambia subcommand via the same
	// injected algos table as the fixed-binary path. launchOpts (os.Stat for the
	// evaluate target) runs here, outside the launch lock, so the critical
	// section below stays minimal.
	lopts, cerr := d.launchOpts(j, prepared)
	if cerr != nil {
		d.markTerminal(name, StateFailed, "launch config: "+cerr.Error())
		return
	}

	if launchGateHook != nil {
		launchGateHook(name)
	}

	// L7: close the cancel/launch race. A Cancel arriving between a bare
	// canceled-check and StartWithOpts would otherwise be dropped and the job
	// would run anyway. Hold d.mu across the final canceled-check, the launch,
	// and the pending handoff so a concurrent Cancel is totally ordered against
	// the launch commit: it either lands first (seen here -> abort before
	// StartWithOpts) or after (job already removed from pending -> Cancel takes
	// the running-job Stop path). procmgr never calls back into the dispatcher,
	// so d.mu -> pm.mu is the single lock order with no inversion.
	d.mu.Lock()
	if j.canceled {
		d.mu.Unlock()
		d.markTerminal(name, StateCanceled, "canceled before launch")
		return
	}
	var lerr error
	if j.resume {
		_, lerr = d.pm.ResumeWithOpts(name, procmgr.StartOpts{}, lopts)
	} else {
		_, lerr = d.pm.StartWithOpts(name, procmgr.StartOpts{}, lopts)
	}
	if lerr != nil {
		d.mu.Unlock()
		d.markTerminal(name, StateFailed, "launch failed: "+lerr.Error())
		return
	}
	// Launched: hand off from the in-memory pending set to process.json under the
	// same lock that guarded the launch, so a Cancel can never slip into the gap.
	delete(d.pending, name)
	launched = true
	d.mu.Unlock()
	d.broadcast()

	d.monitor(name)

	d.mu.Lock()
	d.active--
	d.dispatchLocked()
	d.mu.Unlock()
	d.broadcast()
}

// launchOpts builds the parameterized procmgr launch from the ingest-staged
// Prepared (design 2.7). When the environment did not stage a venv interpreter
// (Prepared.VenvPython empty, e.g. a stub), it returns a zero LaunchOpts so
// procmgr takes the fixed-binary path unchanged. The subcommand for the argv is
// resolved from the job kind through the same algos table the fixed-binary path
// uses; an unregistered kind fails the job rather than launching.
func (d *Dispatcher) launchOpts(j *job, prepared *ingestapi.Prepared) (procmgr.LaunchOpts, error) {
	if prepared == nil || prepared.VenvPython == "" {
		return procmgr.LaunchOpts{}, nil
	}
	sub, err := d.pm.AlgoSubcommand(j.spec.Kind)
	if err != nil {
		return procmgr.LaunchOpts{}, err
	}
	argv := make([]string, 0, len(sub)+8)
	argv = append(argv, "-m", "src.cli")
	argv = append(argv, sub...)
	if j.spec.Kind == KindEvaluate {
		// `cambia evaluate` takes a positional checkpoint/run-dir target, not
		// --config (spec-review finding #1): --config is omitted entirely,
		// since run-dir mode auto-detects config.yaml from the target's own
		// run dir (cfr/src/cli.py evaluate).
		targetArgv, terr := d.evaluateTargetArgv(j.spec)
		if terr != nil {
			return procmgr.LaunchOpts{}, terr
		}
		argv = append(argv, targetArgv...)
	} else {
		// The rendered config is consumed verbatim (design 2.7). A kind that ingest
		// renders no config for (Prepare leaves RenderedConfig empty) gets no --config
		// rather than an empty-valued flag.
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
	}
	if j.resume {
		argv = append(argv, "--resume")
	}
	return procmgr.LaunchOpts{
		Python: prepared.VenvPython,
		Argv:   argv,
		Cwd:    filepath.Join(prepared.WorktreeDir, "cfr"),
		Env:    prepared.Env,
	}, nil
}

// evaluateTargetArgv builds the evaluate-only argv tail: the resolved target
// (the positional checkpoint/run-dir argument `cambia evaluate` requires),
// --latest when target is a run directory (run-dir mode requires --latest or
// --epoch; the harness always wants the newest checkpoint), --games, and
// --device. Target was already lexically guarded and containment-checked at
// submit (handlers.go step 4b); it is re-resolved here since launch happens
// in a later goroutine against the persisted spec.
func (d *Dispatcher) evaluateTargetArgv(spec JobSpec) ([]string, error) {
	targetAbs, err := pathguard.Resolve(d.runsDir, spec.Target)
	if err != nil {
		return nil, fmt.Errorf("target: %w", err)
	}
	info, err := os.Stat(targetAbs)
	if err != nil {
		return nil, fmt.Errorf("target: %w", err)
	}
	// Run-dir mode only. In file mode cli.py leaves agent_type at its deep_cfr
	// default and only recovers a run dir when the file sits under checkpoints/,
	// so a PRT-CFR snapshot target evaluates under the wrong agent wrapper and
	// reports plausible, wrong numbers instead of failing. Run-dir mode derives
	// config, algorithm, and agent type from the target's own config.yaml.
	if !info.IsDir() {
		return nil, fmt.Errorf("target %q: evaluate requires a run directory, not a checkpoint file (file mode misdetects agent type)", spec.Target)
	}
	argv := []string{targetAbs, "--latest"}
	argv = append(argv, "--games", strconv.Itoa(spec.gamesOrDefault()), "--device", spec.device())
	return argv, nil
}

// monitor polls process.json until the run reaches a procmgr terminal status,
// then releases the slot. It keys on the RAW Status (not EffectiveStatus): the
// supervisor's waitFor writes the terminal status and removes the proc from its
// own procs map together under lock, so waiting for the persisted terminal
// status also guarantees the SetMaxConcurrent backstop has freed its slot before
// the dispatcher launches the next job.
func (d *Dispatcher) monitor(name string) {
	for {
		st, ok := d.pm.GetState(name)
		if !ok {
			return
		}
		switch st.Status {
		case procmgr.StatusStopped, procmgr.StatusCrashed, StateCanceled, StateFailed:
			d.env.Cleanup(name, st.Status == procmgr.StatusCrashed)
			d.broadcast()
			return
		}
		time.Sleep(d.poll)
	}
}

// markTerminal persists a runnerd-level terminal state (canceled/failed) into
// process.json and cleans up staged resources (kept for debug on failure). It
// must be called without d.mu held.
func (d *Dispatcher) markTerminal(name, state, lastErr string) {
	runDir := d.runDir(name)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil {
		st = &procmgr.ProcessState{Name: name}
	}
	st.Status = state
	st.LastError = lastErr
	if st.FinishedAt == "" {
		st.FinishedAt = procmgr.NowRFC3339()
	}
	_ = procmgr.WriteProcessState(runDir, st)
	_ = d.env.Cleanup(name, state == StateFailed)
	d.broadcast()
}

// Cancel handles DELETE for a non-purge request. A queued job is dropped (marked
// canceled); a preparing job is flagged for cancellation (runJob aborts before
// launch); a running job is stopped (SIGINT + 30s grace, or SIGKILL on force).
func (d *Dispatcher) Cancel(name string, force bool) (*procmgr.ProcessState, error) {
	if err := procmgr.ValidateName(name); err != nil {
		return nil, err
	}
	d.mu.Lock()
	j, pending := d.pending[name]
	if pending {
		if j.state == StateQueued {
			d.removeFromQueueLocked(name)
			delete(d.pending, name)
			d.mu.Unlock()
			d.markTerminal(name, StateCanceled, "canceled while queued")
			return procmgr.ReadProcessState(d.runDir(name))
		}
		// preparing
		j.canceled = true
		j.cancel()
		d.mu.Unlock()
		st, _ := procmgr.ReadProcessState(d.runDir(name))
		return st, nil
	}
	d.mu.Unlock()

	// Not pending: running or terminal -> delegate to the supervisor. Stop is a
	// best-effort no-op on an already-terminal run.
	if _, ok := d.pm.GetState(name); !ok {
		return nil, ErrNotFound
	}
	return d.pm.Stop(name, force)
}

// Purge removes a terminal job's run dir to free its name. It refuses a
// non-terminal job.
func (d *Dispatcher) Purge(name string) error {
	if err := procmgr.ValidateName(name); err != nil {
		return err
	}
	st, err := procmgr.ReadProcessState(d.runDir(name))
	if err != nil {
		return ErrNotFound
	}
	if !isTerminal(d.effectiveState(name, st)) {
		return ErrNotTerminal
	}
	d.mu.Lock()
	d.removeFromQueueLocked(name)
	delete(d.pending, name)
	d.mu.Unlock()
	if err := os.RemoveAll(d.runDir(name)); err != nil {
		return err
	}
	d.broadcast()
	return nil
}

// Resume gates on the PRT-CFR resume contract (resume_state.json + rolling
// checkpoint) and re-enqueues the job as a resume launch. Resume creates a new
// launch, not a transition (design 2.3).
func (d *Dispatcher) Resume(name string) (JobView, error) {
	if err := procmgr.ValidateName(name); err != nil {
		return JobView{}, err
	}
	if _, err := procmgr.ReadProcessState(d.runDir(name)); err != nil {
		return JobView{}, ErrNotFound
	}
	if !hasResumableState(d.runsDir, name) {
		return JobView{}, ErrNoResumableState
	}
	spec := readJobSpec(d.runDir(name))
	if spec == nil {
		st, _ := procmgr.ReadProcessState(d.runDir(name))
		spec = &JobSpec{Name: name}
		if st != nil {
			spec.Kind = st.Algorithm
		}
	}
	spec.Resume = true

	ctx, cancel := context.WithCancel(context.Background())
	d.mu.Lock()
	if _, exists := d.pending[name]; exists {
		d.mu.Unlock()
		cancel()
		return JobView{}, ErrAlreadyQueued
	}
	if len(d.queue) >= d.maxQueue {
		d.mu.Unlock()
		cancel()
		return JobView{}, ErrQueueFull
	}
	j := &job{spec: *spec, state: StateQueued, resume: true, submitAt: procmgr.NowRFC3339(), ctx: ctx, cancel: cancel}
	d.pending[name] = j
	d.queue = append(d.queue, name)
	d.dispatchLocked()
	view := d.pendingViewLocked(name)
	d.mu.Unlock()

	d.broadcast()
	return view, nil
}

// Reconcile runs at daemon start (queue empty): the procmgr sweep flips
// running/starting/stopping with a dead pid to crashed, then the orphan sweep
// flips created rows with no live process and no queue entry to failed. Finally
// it hands the live job ids to the ingest StartupSweep. It never launches.
func (d *Dispatcher) Reconcile() {
	d.pm.Reconcile()

	states, _ := procmgr.ScanProcessStates(d.runsDir)
	var live []string
	for _, st := range states {
		if st.Status == procmgr.StatusCreated {
			d.mu.Lock()
			_, queued := d.pending[st.Name]
			d.mu.Unlock()
			if !queued {
				st.Status = StateFailed
				st.LastError = "orphaned by daemon restart"
				if st.FinishedAt == "" {
					st.FinishedAt = procmgr.NowRFC3339()
				}
				_ = procmgr.WriteProcessState(d.runDir(st.Name), st)
				continue
			}
		}
		switch procmgr.EffectiveStatus(st) {
		case procmgr.StatusRunning, procmgr.StatusStopping:
			live = append(live, st.Name)
		}
	}
	_ = d.env.StartupSweep(live)

	d.mu.Lock()
	d.reconciledAt = procmgr.NowRFC3339()
	d.mu.Unlock()
	d.broadcast()
}

// removeFromQueueLocked drops name from the FIFO queue. Callers hold d.mu.
func (d *Dispatcher) removeFromQueueLocked(name string) {
	out := d.queue[:0]
	for _, id := range d.queue {
		if id != name {
			out = append(out, id)
		}
	}
	d.queue = out
}

// queuePosLocked returns the 1-based FIFO position of a queued job, or 0.
func (d *Dispatcher) queuePosLocked(name string) int {
	for i, id := range d.queue {
		if id == name {
			return i + 1
		}
	}
	return 0
}

// effectiveState overlays the in-memory runnerd state on process.json. Callers
// pass the already-read process.json state.
func (d *Dispatcher) effectiveState(name string, st *procmgr.ProcessState) string {
	d.mu.Lock()
	j, pending := d.pending[name]
	d.mu.Unlock()
	if pending {
		return j.state
	}
	if st == nil {
		return ""
	}
	return procmgr.EffectiveStatus(st)
}

// hasResumableState mirrors the dashboard's hasCheckpoint gate: the rolling
// PRT-CFR checkpoint and the resume_state.json commit marker must both exist.
func hasResumableState(runsDir, name string) bool {
	runDir := filepath.Join(runsDir, name)
	if _, err := os.Stat(filepath.Join(runDir, "snapshots", "prtcfr_checkpoint.pt")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(runDir, "resume_state.json")); err != nil {
		return false
	}
	return true
}
