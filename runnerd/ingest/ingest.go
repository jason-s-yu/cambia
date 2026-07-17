// Package ingest stages a commit-pinned job for execution on the serving-harness
// runner. Given a job id, a 40-hex commit, a kind, and a repo-relative
// config, it verifies the pushed snapshot in the bare mirror, adds a detached
// worktree at the pinned sha, builds (or reuses) a per-lock uv venv and a
// per-engine-tree libcambia shared library, renders a rails-applied config,
// writes write-once provenance (env.json), and returns the launch context the
// daemon needs.
//
// It shells out through the CommandRunner seam only. Git runs against the real
// bare mirror; uv/python/go builds run through the same seam and are fakeable in
// tests. The package touches no ambient Python environment: every uv invocation
// addresses its target venv by explicit path plus UV_PROJECT_ENVIRONMENT, never
// --active and never ~/.pyenv.
package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
)

// goToolchainPin is the GOTOOLCHAIN value forced for every libcambia build so a
// newer host Go cannot silently change the compiler (design 3.5). engine/go.mod
// declares `go 1.26.0` as a floor with no toolchain line.
const goToolchainPin = "go1.26.0"

// defaultDebugTTL is how long a worktree is retained after a job fails without a
// confirmed artifact sync, for post-mortem inspection (design 3.2).
const defaultDebugTTL = 24 * time.Hour

// Layout subdirectory names under BaseDir (design 2.7).
const (
	dirMirror    = "mirror.git"
	dirWorktrees = "worktrees"
	dirVenvs     = "venvs"
	dirLibcambia = "libcambia"
	dirShim      = "shim"
)

// Config parameterizes a Manager. BaseDir is the /srv/cambia layout root; RunsDir
// is where run directories live (outside worktrees, never reaped here). MirrorDir
// defaults to BaseDir/mirror.git. MirrorURL, when set, seeds a fresh mirror by
// clone; the steady-state model is init-bare plus external `git push`.
type Config struct {
	// BaseDir is the on-disk layout root (design 2.7): mirror.git, worktrees/,
	// venvs/, libcambia/, shim/ live under it.
	BaseDir string
	// RunsDir is the runs root. Run dirs are RunsDir/<job-id>.
	RunsDir string
	// MirrorURL optionally seeds a fresh bare mirror by clone. Empty means the
	// mirror is created empty (init --bare) and populated by external push.
	MirrorURL string
	// MirrorDir overrides the bare mirror path. Defaults to BaseDir/mirror.git.
	MirrorDir string
	// MaxVenvs caps the retained per-lock venvs (LRU; live-referenced always
	// kept). Default 8.
	MaxVenvs int
	// MaxLibcambia caps the retained libcambia artifacts (LRU). Default 50.
	MaxLibcambia int
	// CoresCap is the job-internal worker-count ceiling injected as a rails-last
	// override (design 6: cores - 2). Non-positive disables the worker rail.
	CoresCap int
	// PythonBin is the interpreter uv builds venvs against (explicit --python).
	// Default "python3".
	PythonBin string
	// Runner is the injected command-execution seam. Default ExecRunner.
	Runner CommandRunner
	// DebugTTL overrides the failed-without-sync worktree retention. Default 24h.
	DebugTTL time.Duration
	// Now overrides the clock (TTL math). Default time.Now.
	Now func() time.Time
	// RequireSignedCommits gates ssh commit-signature verification in Prepare
	// (cambia-550, W1). Default false: verify-commit never runs and behavior is
	// byte-for-byte unchanged. When true, a pinned commit must carry a valid ssh
	// signature by a key in AllowedSignersPath or the job is rejected.
	RequireSignedCommits bool
	// AllowedSignersPath is the git gpg.ssh.allowedSignersFile consulted when
	// RequireSignedCommits is true. An empty value or a missing file fails
	// closed (the job is rejected). Ignored when enforcement is off.
	AllowedSignersPath string
}

// Manager stages jobs under one BaseDir. It is safe for sequential use by the
// daemon's single dispatcher; it holds no long-lived locks of its own.
type Manager struct {
	cfg          Config
	mirrorDir    string
	worktreesDir string
	venvsDir     string
	libcambiaDir string
	shimDir      string
	runner       CommandRunner
	debugTTL     time.Duration
	now          func() time.Time

	requireSignedCommits bool
	allowedSignersPath   string
}

// New constructs a Manager, applying defaults for unset Config fields.
func New(cfg Config) *Manager {
	if cfg.MaxVenvs <= 0 {
		cfg.MaxVenvs = 8
	}
	if cfg.MaxLibcambia <= 0 {
		cfg.MaxLibcambia = 50
	}
	if cfg.PythonBin == "" {
		cfg.PythonBin = "python3"
	}
	if cfg.Runner == nil {
		cfg.Runner = ExecRunner{}
	}
	if cfg.DebugTTL <= 0 {
		cfg.DebugTTL = defaultDebugTTL
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	mirrorDir := cfg.MirrorDir
	if mirrorDir == "" {
		mirrorDir = filepath.Join(cfg.BaseDir, dirMirror)
	}
	return &Manager{
		cfg:          cfg,
		mirrorDir:    mirrorDir,
		worktreesDir: filepath.Join(cfg.BaseDir, dirWorktrees),
		venvsDir:     filepath.Join(cfg.BaseDir, dirVenvs),
		libcambiaDir: filepath.Join(cfg.BaseDir, dirLibcambia),
		shimDir:      filepath.Join(cfg.BaseDir, dirShim),
		runner:       cfg.Runner,
		debugTTL:     cfg.DebugTTL,
		now:          cfg.Now,

		requireSignedCommits: cfg.RequireSignedCommits,
		allowedSignersPath:   cfg.AllowedSignersPath,
	}
}

// Prepare stages jobID at the pinned commit and returns its launch context. The
// pipeline: ensure mirror -> receipt-check the pushed ref -> add detached
// worktree -> preflight+build (or reuse) the per-lock, per-device venv ->
// build (or reuse) libcambia -> render the rails-applied config (train kinds)
// -> write env.json.
//
// device selects the uv sync extra (cpu/gpu/xpu) and the rendered config's
// device rail (cambia-329); an empty device defaults to cpu, matching
// JobSpec.device(). warmStart is a train job's optional runs-dir-relative
// snapshot reference (design cambia-334); it is re-resolved against RunsDir
// here (defense in depth alongside the submit-time guard in handlers.go,
// mirroring how the dispatcher re-resolves an evaluate target at launch) and
// threaded into the rendered config as the prt_cfr.warm_start_path rail.
// overrides are the submitter's dotted-key config overrides; any targeting a
// harness-owned key is rejected before render. Rails are appended AFTER the
// user overrides so they win last-write.
func (m *Manager) Prepare(ctx context.Context, jobID, commit, kind, configRel, device, warmStart string, overrides map[string]string) (*ingestapi.Prepared, error) {
	if device == "" {
		device = "cpu"
	}
	if err := validateJobID(jobID); err != nil {
		return nil, err
	}
	if err := validateCommit(commit); err != nil {
		return nil, err
	}
	// Reject submitter overrides on harness-owned keys before any work.
	if err := rejectOwnedOverrides(overrides); err != nil {
		return nil, err
	}

	if err := m.ensureMirror(ctx); err != nil {
		return nil, fmt.Errorf("ensure mirror: %w", err)
	}
	if err := m.verifyReceipt(ctx, jobID, commit); err != nil {
		return nil, fmt.Errorf("receipt check: %w", err)
	}
	// Signature gate (cambia-550, W1): a no-op when RequireSignedCommits is off.
	// At this point the sha is receipt-matched and the object is present in the
	// bare mirror, so verify-commit reads a known-local object.
	if err := m.verifyCommitSignature(ctx, commit); err != nil {
		return nil, fmt.Errorf("signature verify: %w", err)
	}

	worktreeDir := m.worktreePath(jobID)
	if err := m.addWorktree(ctx, worktreeDir, commit); err != nil {
		return nil, fmt.Errorf("worktree add: %w", err)
	}

	runDir := m.runDir(jobID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, fmt.Errorf("create run dir: %w", err)
	}

	venv, err := m.ensureVenv(ctx, commit, worktreeDir, device)
	if err != nil {
		return nil, fmt.Errorf("venv: %w", err)
	}

	lib, err := m.ensureLibcambia(ctx, commit, worktreeDir)
	if err != nil {
		return nil, fmt.Errorf("libcambia: %w", err)
	}

	warmStartPath := ""
	if warmStart != "" {
		resolved, werr := pathguard.Resolve(m.cfg.RunsDir, warmStart)
		if werr != nil {
			return nil, fmt.Errorf("warm_start: %w", werr)
		}
		warmStartPath = resolved
	}

	renderedConfig := ""
	if configRel != "" {
		renderedConfig, err = m.renderConfig(ctx, worktreeDir, runDir, venv.python, kind, configRel, device, warmStartPath, overrides)
		if err != nil {
			return nil, fmt.Errorf("config render: %w", err)
		}
	}

	env, err := m.assembleEnv(worktreeDir, lib.path)
	if err != nil {
		return nil, fmt.Errorf("env assembly: %w", err)
	}

	prov := provenance{
		JobID:         jobID,
		Commit:        commit,
		EngineTreeSha: lib.engineTreeSha,
		LibcambiaSha:  lib.sha256,
		UVLockSha:     venv.lockSha256,
		VenvCacheKey:  venv.key,
		PlatformTag:   platformTag(),
		Device:        device,
	}
	if err := m.writeEnvJSON(ctx, runDir, venv.python, prov); err != nil {
		return nil, fmt.Errorf("env.json: %w", err)
	}

	// Best-effort LRU trims, protecting the keys this job just staked.
	m.evictVenvs(map[string]bool{venv.key: true})
	m.evictLibcambia(map[string]bool{lib.engineTreeSha: true})

	return &ingestapi.Prepared{
		WorktreeDir:    worktreeDir,
		RunDir:         runDir,
		VenvPython:     venv.python,
		LibcambiaPath:  lib.path,
		RenderedConfig: renderedConfig,
		Env:            env,
	}, nil
}

// Cleanup releases a job's ingest resources. keepForDebug=false is the terminal-
// and-synced path: the worktree is removed and the job ref deleted. keepForDebug=
// true is the failed-without-sync path: the worktree is retained and stamped with
// a debug TTL so StartupSweep leaves it for post-mortem until the TTL lapses. Run
// dirs are never touched here (they live outside worktrees). Caches are not
// evicted here; their LRU keeps live-referenced keys.
func (m *Manager) Cleanup(jobID string, keepForDebug bool) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	worktreeDir := m.worktreePath(jobID)
	if keepForDebug {
		return m.markDebugTTL(worktreeDir)
	}
	ctx := context.Background()
	// The job ref is deliberately NOT deleted here: its lifetime is the run
	// dir's, not the worktree's. Resume re-runs Prepare, whose receipt check
	// needs the ref, and the mirror's pruning gc would otherwise collect the
	// pinned commit. PurgeRef (run-dir purge) and StartupSweep (ref with no
	// run dir) own ref deletion.
	return m.removeWorktree(ctx, worktreeDir)
}

// PurgeRef deletes a job's mirror ref. Called when the run dir is purged: the
// ref pins the job's commit against mirror gc for exactly as long as the run
// (and thus a possible resume) exists.
func (m *Manager) PurgeRef(jobID string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	return m.deleteJobRef(context.Background(), jobID)
}

// StartupSweep reconciles ingest state after a daemon restart. It prunes every
// worktree whose job is neither live (in liveJobIDs) nor within an unexpired
// failure TTL, deletes those job refs, prunes worktree metadata, and runs a
// pruning gc on the mirror. It then trims the venv and libcambia caches to their
// caps, protecting keys referenced by live jobs (read from their env.json). Run
// dirs are never reaped.
func (m *Manager) StartupSweep(liveJobIDs []string) error {
	ctx := context.Background()
	live := make(map[string]bool, len(liveJobIDs))
	for _, id := range liveJobIDs {
		live[id] = true
	}

	entries, err := os.ReadDir(m.worktreesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	var firstErr error
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		jobID := e.Name()
		if live[jobID] {
			continue
		}
		if m.debugTTLActive(m.worktreePath(jobID)) {
			continue
		}
		if err := m.removeWorktree(ctx, m.worktreePath(jobID)); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	// Reap job refs by run-dir lifetime, independent of worktree state: a ref
	// whose run dir still exists pins a resumable run's commit against the gc
	// below; one without a run dir (purged while the daemon was down) is dead.
	if refs, err := m.listJobRefs(ctx); err == nil {
		for _, jobID := range refs {
			if live[jobID] {
				continue
			}
			if _, statErr := os.Stat(m.runDir(jobID)); statErr == nil {
				continue
			}
			if err := m.deleteJobRef(ctx, jobID); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	} else if firstErr == nil {
		firstErr = err
	}

	// Prune dangling worktree admin entries, then pruning gc on the mirror.
	_, _ = m.git(ctx, "worktree", "prune")
	if _, err := m.git(ctx, "gc", "--prune=now"); err != nil && firstErr == nil {
		firstErr = err
	}

	// Cache trims protect keys still referenced by live jobs.
	protectVenv, protectLib := m.liveCacheKeys(liveJobIDs)
	m.evictVenvs(protectVenv)
	m.evictLibcambia(protectLib)
	return firstErr
}

// liveCacheKeys reads each live job's env.json to collect the venv cache keys and
// engine-tree shas still in use, so cache eviction never drops a running job's
// interpreter or shared library. Unreadable env.json is skipped (best-effort).
func (m *Manager) liveCacheKeys(liveJobIDs []string) (venvKeys, libKeys map[string]bool) {
	venvKeys = map[string]bool{}
	libKeys = map[string]bool{}
	for _, id := range liveJobIDs {
		p, err := readEnvJSON(filepath.Join(m.runDir(id), envJSONFile))
		if err != nil {
			continue
		}
		if p.VenvCacheKey != "" {
			venvKeys[p.VenvCacheKey] = true
		}
		if p.EngineTreeSha != "" {
			libKeys[p.EngineTreeSha] = true
		}
	}
	return venvKeys, libKeys
}

// runDir returns the run directory for jobID (outside any worktree).
func (m *Manager) runDir(jobID string) string {
	return filepath.Join(m.cfg.RunsDir, jobID)
}

// worktreePath returns the worktree directory for jobID.
func (m *Manager) worktreePath(jobID string) string {
	return filepath.Join(m.worktreesDir, jobID)
}

// platformTag is the Go-native platform identifier used in the venv cache key
// and env.json. It is contention-immune (no subprocess) and deterministic.
func platformTag() string {
	return runtime.GOOS + "_" + runtime.GOARCH
}
