package harness

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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
	// ErrHasDependents is returned by Purge when a terminal parent still has
	// queued dependents (cambia-352); the operator must pass cascade to skip them
	// or wait for the gate to resolve.
	ErrHasDependents = errors.New("job has queued dependents")
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
	// nextSeq is the next submit sequence to assign (cambia-352). It is seeded to
	// max(persisted submit_seq)+1 at Reconcile so restarts never reuse a seq, and
	// increments under d.mu on every Submit.
	nextSeq int64
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
		nextSeq:  1,
	}
}

func (d *Dispatcher) runDir(name string) string { return filepath.Join(d.runsDir, name) }

// Submit admits a validated spec: it claims the name via procmgr.Create (which
// writes process.json Status=created), records jobspec.json (with the assigned
// submit_seq), enqueues the job, and kicks the dispatcher. The caller has
// already run name/kind/path validation and the resource preflights; procmgr's
// Create re-checks the name collision atomically. process.json is written before
// jobspec.json so the atomic name claim (and thus name-collision detection) is
// unchanged; a daemon crash in the tiny gap leaves a `created` row with no spec,
// which Reconcile fails as an incomplete admission. Returns the job's view.
func (d *Dispatcher) Submit(spec JobSpec) (JobView, error) {
	d.mu.Lock()
	if len(d.queue) >= d.maxQueue {
		d.mu.Unlock()
		return JobView{}, ErrQueueFull
	}
	spec.SubmitSeq = d.nextSeq
	d.nextSeq++
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
	d.mu.Unlock()

	d.broadcast()
	// Resolve the view after unlocking: dispatchLocked may have already launched
	// the job (a free slot) or gate-resolved it (a parent already terminal, so
	// the dependent goes straight to skipped/failed/running), and resolveView
	// reflects that from process.json + the in-memory state.
	view, _ := d.resolveView(spec.Name)
	return view, nil
}

// gateDecision is the dispatch verdict for a queued job after its `after`
// dependency (cambia-352) is evaluated against the parent's current state.
type gateDecision int

const (
	gateLaunch  gateDecision = iota // ready to run (no parent, parent succeeded, or on_failure=run)
	gateBlocked                     // parent not yet terminal; keep queued, do not head-of-line block
	gateSkip                        // parent failed and on_failure=skip -> mark skipped
	gateFail                        // parent failed and on_failure=fail -> mark failed
)

// dispatchLocked scans the queue in submit_seq order and, for each queued job,
// evaluates its dependency gate against the parent's current state. Ready jobs
// launch while a slot is free; a job whose parent has not yet reached a terminal
// stays queued but does NOT block later ready jobs (a first-ready scan, a
// documented deviation from strict FIFO); a job whose parent failed is resolved
// to skipped/failed per its on_failure policy without consuming a slot. Callers
// hold d.mu. The prepare+launch+monitor work runs in a per-job goroutine; the
// bounded parent process.json reads happen under d.mu. Skip/fail transitions are
// persisted inline (filesystem only, no broadcast, no d.mu re-entry); the
// caller's post-unlock broadcast publishes them.
func (d *Dispatcher) dispatchLocked() {
	var next []string
	for _, id := range d.queue {
		j := d.pending[id]
		if j == nil {
			continue
		}
		if j.canceled {
			delete(d.pending, id)
			continue
		}
		switch d.gateDecisionLocked(j) {
		case gateBlocked:
			next = append(next, id) // parent still pending: wait, without blocking others
		case gateSkip:
			delete(d.pending, id)
			j.cancel()
			d.writeGateTerminalLocked(id, StateSkipped, "parent "+j.spec.After+" did not succeed (on_failure=skip)")
		case gateFail:
			delete(d.pending, id)
			j.cancel()
			d.writeGateTerminalLocked(id, StateFailed, "parent "+j.spec.After+" did not succeed (on_failure=fail)")
		case gateLaunch:
			if d.maxJobs > 0 && d.active >= d.maxJobs {
				next = append(next, id) // ready but no free slot: keep in place
				continue
			}
			j.state = StatePreparing
			d.active++
			go d.runJob(j)
		}
	}
	d.queue = next
}

// gateDecisionLocked resolves whether a queued job may launch, from its `after`
// parent's current effective state (cambia-352). Callers hold d.mu. A job with
// no parent, or a resume launch (a resumed dependent ignores its own after,
// design 2.3), always launches. The parent read is a single bounded
// ReadProcessState under d.mu.
func (d *Dispatcher) gateDecisionLocked(j *job) gateDecision {
	if j.resume || j.spec.After == "" {
		return gateLaunch
	}
	parent := j.spec.After
	st, err := procmgr.ReadProcessState(d.runDir(parent))
	if err != nil {
		// Parent run dir gone (e.g. purged out from under a waiting dependent):
		// treat as a non-success terminal so the dependent is never stranded.
		return d.gateFailureLocked(j)
	}
	pstate := d.effectiveStateLocked(parent, st)
	if !isTerminal(pstate) {
		return gateBlocked // parent queued/preparing/running/starting/stopping
	}
	if pstate == procmgr.StatusStopped && exitCodeIsZero(st) {
		return gateLaunch // clean exit: parent success always runs the dependent
	}
	// crashed / failed / canceled / skipped / a graceful-stop with a nonzero exit.
	return d.gateFailureLocked(j)
}

// gateFailureLocked maps a job's on_failure policy to its parent-failure verdict.
func (d *Dispatcher) gateFailureLocked(j *job) gateDecision {
	switch j.spec.onFailureOrDefault() {
	case OnFailureRun:
		return gateLaunch
	case OnFailureFail:
		return gateFail
	default: // OnFailureSkip
		return gateSkip
	}
}

// exitCodeIsZero reports whether st recorded a clean (zero) process exit.
func exitCodeIsZero(st *procmgr.ProcessState) bool {
	return st != nil && st.ExitCode != nil && *st.ExitCode == 0
}

// writeGateTerminalLocked persists a gate-resolved terminal state (skipped or
// failed) for a queued dependent whose parent did not succeed. Callers hold
// d.mu; it does filesystem I/O only (no broadcast, no d.mu re-entry) so the
// caller's post-unlock broadcast publishes it. The job never left the queue for
// prepare, so no staged resources need cleanup.
func (d *Dispatcher) writeGateTerminalLocked(name, state, lastErr string) {
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
}

// reDispatch re-runs the dispatch scan after an external terminal transition (a
// parent canceled while queued, a parent purged) that did not itself flow
// through runJob's slot-release dispatch. It re-arms any dependent whose gate
// now resolves, and broadcasts the result.
func (d *Dispatcher) reDispatch() {
	d.mu.Lock()
	d.dispatchLocked()
	d.mu.Unlock()
	d.broadcast()
}

// pendingDependentsLocked returns the names of jobs still QUEUED (gate unresolved)
// whose parent is `parent`. Preparing/running dependents already passed the gate
// and no longer need the parent, so they do not count. Callers hold d.mu.
func (d *Dispatcher) pendingDependentsLocked(parent string) []string {
	var out []string
	for name, j := range d.pending {
		if j.state == StateQueued && j.spec.After == parent {
			out = append(out, name)
		}
	}
	return out
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

	prepared, err := d.env.Prepare(j.ctx, name, j.spec.Commit, j.spec.Kind, j.spec.Config, j.spec.device(), j.spec.WarmStart, j.spec.overridesStr())

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
	argv := make([]string, 0, len(sub)+10)
	argv = append(argv, "-m", "src.cli")
	argv = append(argv, sub...)
	// journalRun is the run whose per-run-dir run_db.sqlite this job writes:
	// the job's own run for train/head-to-head/bench, the evaluated run for
	// evaluate (eval rows must join the evaluated run's journal so they sync
	// with it, design 4.2). head-to-head and bench never write run_db rows of
	// their own (cli.py's head_to_head and benchmark commands do not call
	// run_db), so the default is inert for them beyond a harmless env var.
	journalRun := j.spec.Name
	switch j.spec.Kind {
	case KindEvaluate:
		// `cambia evaluate` takes a positional checkpoint/run-dir target, not
		// --config (spec-review finding #1): --config is omitted entirely,
		// since run-dir mode auto-detects config.yaml from the target's own
		// run dir (cfr/src/cli.py evaluate).
		targetArgv, terr := d.evaluateTargetArgv(j.spec)
		if terr != nil {
			return procmgr.LaunchOpts{}, terr
		}
		argv = append(argv, targetArgv...)
		journalRun = j.spec.Target
	case KindHeadToHead:
		// `cambia head-to-head` has no run-dir mode: it takes two bare
		// checkpoint files with no config of their own, so unlike evaluate it
		// genuinely needs --config (client-side spec.py requires it for this
		// kind, cambia-295 item 1 contract change).
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		h2hArgv, herr := d.headToHeadArgv(j.spec)
		if herr != nil {
			return procmgr.LaunchOpts{}, herr
		}
		argv = append(argv, h2hArgv...)
	case KindBench:
		// `cambia benchmark all` takes --config, --device, and --output-dir as
		// plain CLI flags -- none of them sourced from the rendered config's
		// device rail the way train's device is. --output-dir is pointed at
		// the job's own run dir (mirroring train's --save-path) so results
		// land under runs/<name>/ instead of the CLI's container-only default.
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		argv = append(argv, "--output-dir", d.runDir(j.spec.Name))
		argv = append(argv, "--device", j.spec.device())
	default:
		// train (and any test-injected kind that isn't evaluate/head-to-head/
		// bench, e.g. harness_test.go's "fake"): the rendered config is
		// consumed verbatim (design 2.7). A kind that ingest renders no config
		// for (Prepare leaves RenderedConfig empty) gets no --config rather
		// than an empty-valued flag.
		if prepared.RenderedConfig != "" {
			argv = append(argv, "--config", prepared.RenderedConfig)
		}
		if j.spec.Kind == KindTrain {
			// Without an explicit name the trainer registers its run_db row
			// under a config-derived default, decoupling the journal row from
			// the run dir the reconciler replays (found live in M5 e2e).
			// Without an explicit save path the trainer resolves runs/<name>
			// against the worktree cwd, so resume_state.json and metrics.jsonl
			// land in the worktree and die with its cleanup (also found live):
			// the fixed-binary dashboard path always passes both.
			argv = append(argv, "--run-name", j.spec.Name)
			argv = append(argv, "--save-path", d.runDir(j.spec.Name))
		}
	}
	if j.resume {
		argv = append(argv, "--resume")
	}
	// CAMBIA_RUN_DB points every run_db write of the job process (run
	// registration, checkpoints, eval persist) at the per-run journal the pull
	// loop syncs (design 4.2: runs/<name>/run_db.sqlite IS the wire format).
	env := append([]string(nil), prepared.Env...)
	env = append(env, "CAMBIA_RUN_DB="+filepath.Join(d.runsDir, journalRun, "run_db.sqlite"))
	return procmgr.LaunchOpts{
		Python: prepared.VenvPython,
		Argv:   argv,
		Cwd:    filepath.Join(prepared.WorktreeDir, "cfr"),
		Env:    env,
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

// headToHeadArgv builds the head-to-head argv tail: --checkpoint-a/-b (each
// re-resolved through the same runs-dir containment guard checkpoint_a/b
// already got at submit, handlers.go step 4b -- launch happens in a later
// goroutine against the persisted spec, so it is re-resolved here exactly as
// evaluateTargetArgv re-resolves target), --games, and --device. Unlike
// evaluate's target, checkpoints have no dir-vs-file ambiguity to guard
// against: `cambia head-to-head` declares both as typer Path(exists=True), so
// a missing or unresolvable checkpoint fails the job at launch with a clear
// CLI error rather than silently misinterpreting it.
func (d *Dispatcher) headToHeadArgv(spec JobSpec) ([]string, error) {
	a, err := pathguard.Resolve(d.runsDir, spec.CheckpointA)
	if err != nil {
		return nil, fmt.Errorf("checkpoint_a: %w", err)
	}
	b, err := pathguard.Resolve(d.runsDir, spec.CheckpointB)
	if err != nil {
		return nil, fmt.Errorf("checkpoint_b: %w", err)
	}
	return []string{
		"--checkpoint-a", a,
		"--checkpoint-b", b,
		"--games", strconv.Itoa(spec.gamesOrDefault()),
		"--device", spec.device(),
	}, nil
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
			// A canceled parent is a non-success terminal: re-arm any dependent
			// that was waiting on it (this cancel path does not flow through
			// runJob's slot-release dispatch).
			d.reDispatch()
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
// non-terminal job. If the job still has queued dependents (cambia-352) it
// returns ErrHasDependents unless cascade is set, in which case those dependents
// are marked skipped before the parent dir is removed so the purge cannot strand
// a waiting job. The gate's dir-gone failure branch is the backstop for any
// dependent that races in after the guard.
func (d *Dispatcher) Purge(name string, cascade bool) error {
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
	dependents := d.pendingDependentsLocked(name)
	if len(dependents) > 0 && !cascade {
		d.mu.Unlock()
		return ErrHasDependents
	}
	for _, dep := range dependents {
		if j := d.pending[dep]; j != nil {
			j.cancel()
		}
		d.removeFromQueueLocked(dep)
		delete(d.pending, dep)
		d.writeGateTerminalLocked(dep, StateSkipped, "parent "+name+" purged (cascade)")
	}
	d.removeFromQueueLocked(name)
	delete(d.pending, name)
	d.mu.Unlock()
	if err := os.RemoveAll(d.runDir(name)); err != nil {
		return err
	}
	// The run dir is gone, so the pinned commit no longer needs its gc anchor.
	_ = d.env.PurgeRef(name)
	// Re-arm any transitive dependent (a grandchild whose parent was just
	// cascade-skipped) now that the states have changed; reDispatch broadcasts.
	d.reDispatch()
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

// Reconcile runs at daemon start: the procmgr sweep flips running/starting/
// stopping with a dead pid to crashed, then the queue-persistence sweep rebuilds
// the FIFO (cambia-352). A `created` row provably never forked (launch writes
// `starting` before cmd.Start), so it is safe to re-enqueue: every `created` job
// with a readable jobspec.json is re-queued in submit_seq order and re-prepared
// from scratch; a `created` row whose spec is missing or corrupt is an aborted
// admission and is failed (per-file isolated). The submit_seq counter is seeded
// past every persisted spec so post-restart submits never reuse a seq. Live job
// ids are handed to the ingest StartupSweep. Reconcile itself never forks; the
// re-enqueued jobs launch through the normal dispatch scan (dependency-gated).
func (d *Dispatcher) Reconcile() {
	d.pm.Reconcile()

	states, _ := procmgr.ScanProcessStates(d.runsDir)
	var live []string
	var reenq []*job
	var maxSeq int64
	for _, st := range states {
		// Seed the submit_seq high-water mark from every persisted spec (not just
		// created rows) so a new submit after restart never collides with a seq a
		// still-created job carries.
		if spec := readJobSpec(d.runDir(st.Name)); spec != nil && spec.SubmitSeq > maxSeq {
			maxSeq = spec.SubmitSeq
		}
		if st.Status == procmgr.StatusCreated {
			spec := readJobSpec(d.runDir(st.Name))
			if spec == nil {
				// created row with no readable spec: admission aborted mid-write.
				st.Status = StateFailed
				st.LastError = "incomplete admission: jobspec.json missing or corrupt"
				if st.FinishedAt == "" {
					st.FinishedAt = procmgr.NowRFC3339()
				}
				_ = procmgr.WriteProcessState(d.runDir(st.Name), st)
				continue
			}
			ctx, cancel := context.WithCancel(context.Background())
			reenq = append(reenq, &job{spec: *spec, state: StateQueued, submitAt: st.CreatedAt, ctx: ctx, cancel: cancel})
			continue
		}
		switch procmgr.EffectiveStatus(st) {
		case procmgr.StatusRunning, procmgr.StatusStopping:
			live = append(live, st.Name)
		}
	}
	_ = d.env.StartupSweep(live)

	// Restore FIFO order from the persisted submit_seq.
	sort.Slice(reenq, func(i, j int) bool { return reenq[i].spec.SubmitSeq < reenq[j].spec.SubmitSeq })

	d.mu.Lock()
	if maxSeq+1 > d.nextSeq {
		d.nextSeq = maxSeq + 1
	}
	for _, j := range reenq {
		if _, exists := d.pending[j.spec.Name]; exists {
			j.cancel()
			continue
		}
		d.pending[j.spec.Name] = j
		d.queue = append(d.queue, j.spec.Name)
	}
	d.dispatchLocked()
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
// pass the already-read process.json state and must NOT hold d.mu.
func (d *Dispatcher) effectiveState(name string, st *procmgr.ProcessState) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.effectiveStateLocked(name, st)
}

// effectiveStateLocked is effectiveState for callers that already hold d.mu (the
// dependency gate reads a parent's effective state from inside dispatchLocked).
func (d *Dispatcher) effectiveStateLocked(name string, st *procmgr.ProcessState) string {
	if j, pending := d.pending[name]; pending {
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
