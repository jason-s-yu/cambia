package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// venvResult carries the resolved per-lock venv and the provenance derived while
// building or hitting it.
type venvResult struct {
	dir        string // venvs/<key>
	python     string // <dir>/bin/python
	key        string // venv cache key
	lockSha256 string // sha256 of uv.lock blob content
	pyMinor    string // e.g. "3.11"
}

// ensureVenv resolves the per-lock, per-device uv venv for the pinned commit,
// building it on a cache miss and reusing it on a hit (design 3.3, extended by
// cambia-329 for device). The cache key is sha256(cfr/uv.lock blob content) ++
// python-minor ++ platform tag ++ device extra (cpu omitted, see
// venvCacheKey), so jobs on the same lock, interpreter, platform, and device
// share one venv and cfr-only commits that leave the lock untouched are hits.
//
// ABSOLUTE RULES enforced here: the venv is addressed only by explicit path plus
// UV_PROJECT_ENVIRONMENT and --python; uv is never invoked with --active; no
// ambient environment (~/.pyenv, user site) is touched; no editable install is
// performed (job src arrives via PYTHONPATH).
func (m *Manager) ensureVenv(ctx context.Context, commit, worktreeDir, device string) (venvResult, error) {
	lockContent, err := m.gitRaw(ctx, "cat-file", "blob", commit+":cfr/uv.lock")
	if err != nil {
		return venvResult{}, fmt.Errorf("read uv.lock blob: %w", err)
	}
	lockSha := sha256hex(lockContent)

	pyMinor, err := m.probePythonMinor(ctx)
	if err != nil {
		return venvResult{}, err
	}

	extra := deviceExtra(device)
	key := venvCacheKey(lockSha, pyMinor, platformTag(), extra)
	dir := filepath.Join(m.venvsDir, key)
	python := filepath.Join(dir, "bin", "python")

	res := venvResult{dir: dir, python: python, key: key, lockSha256: lockSha, pyMinor: pyMinor}

	if venvValid(python) {
		touch(dir, m.now())
		return res, nil
	}

	if err := m.buildVenv(ctx, worktreeDir, dir, extra); err != nil {
		return venvResult{}, err
	}
	touch(dir, m.now())
	return res, nil
}

// deviceExtra maps a job device to the uv dependency-group extra it installs
// (cfr/pyproject.toml's cpu/gpu/xpu extras, mutually exclusive per
// [tool.uv].conflicts). An empty or unrecognized device defaults to cpu,
// matching JobSpec.device()'s default.
func deviceExtra(device string) string {
	switch device {
	case "cuda":
		return "gpu"
	case "xpu":
		return "xpu"
	default:
		return "cpu"
	}
}

// venvCacheKey composes the readable, stable cache key from the lock content
// hash, python minor, platform tag, and device extra. The cpu extra keeps the
// EXACT pre-device-support key format (sha-pyX.Y-platform) so venvs already
// warm on a live runner stay valid cache hits; any other extra appends
// "-<extra>" so a cuda or xpu venv never collides with a cpu venv sharing the
// same lock, interpreter, and platform.
func venvCacheKey(lockSha, pyMinor, platform, extra string) string {
	key := fmt.Sprintf("%s-py%s-%s", lockSha, pyMinor, platform)
	if extra != "cpu" {
		key += "-" + extra
	}
	return key
}

// venvReceiptName marks a venv whose uv sync completed. A cache hit requires
// it: bin/python alone exists as soon as `uv venv` runs, so a build that died
// between venv creation and sync completion would otherwise poison the cache
// as a permanently-reused empty env.
const venvReceiptName = ".cambia-venv-ok"

// venvValid reports whether a venv is complete: interpreter present AND the
// post-sync receipt written.
func venvValid(python string) bool {
	if _, err := os.Stat(python); err != nil {
		return false
	}
	receipt := filepath.Join(filepath.Dir(filepath.Dir(python)), venvReceiptName)
	if _, err := os.Stat(receipt); err != nil {
		return false
	}
	return true
}

// buildVenv runs the `uv lock --check` staleness preflight, then creates and
// syncs the venv. The lock check runs in the worktree cfr dir BEFORE any venv
// creation so a stale lock rejects before build work (design 3.3). uv sync uses
// --frozen --extra <extra> (cpu/gpu/xpu, selected by the job's device via
// deviceExtra) with the target env pinned by both UV_PROJECT_ENVIRONMENT and
// explicit --python; --frozen installs exactly from the lock without
// re-resolving (staleness is the preflight's job; uv >= 0.5 rejects combining
// --frozen with --locked).
func (m *Manager) buildVenv(ctx context.Context, worktreeDir, venvDir, extra string) error {
	cfrDir := filepath.Join(worktreeDir, "cfr")
	uvEnv := []string{"UV_PROJECT_ENVIRONMENT=" + venvDir}

	// Preflight: reject a stale lock before building anything.
	if res, err := m.runner.Run(ctx, Command{
		Name: "uv",
		Args: []string{"lock", "--check"},
		Dir:  cfrDir,
		Env:  uvEnv,
	}); err != nil {
		return fmt.Errorf("%w: %s", ErrStaleLock, strings.TrimSpace(string(res.Stderr)))
	}

	if err := os.MkdirAll(m.venvsDir, 0o755); err != nil {
		return err
	}

	if res, err := m.runner.Run(ctx, Command{
		Name: "uv",
		// --clear replaces any stale tree at the target (uv refuses an
		// existing dir otherwise), so a rebuild over a receipt-less remnant
		// is self-healing.
		Args: []string{"venv", venvDir, "--clear", "--python", m.cfg.PythonBin},
		Dir:  cfrDir,
		Env:  uvEnv,
	}); err != nil {
		os.RemoveAll(venvDir)
		return fmt.Errorf("uv venv: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}

	if res, err := m.runner.Run(ctx, Command{
		Name: "uv",
		Args: []string{"sync", "--frozen", "--extra", extra, "--python", filepath.Join(venvDir, "bin", "python")},
		Dir:  cfrDir,
		Env:  uvEnv,
	}); err != nil {
		// Never leave a created-but-unsynced venv behind: bin/python already
		// exists, and a stale tree here would satisfy a naive existence probe.
		os.RemoveAll(venvDir)
		return fmt.Errorf("uv sync: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}

	// Receipt last: only a fully-synced venv is ever treated as a cache hit.
	if err := os.WriteFile(filepath.Join(venvDir, venvReceiptName), []byte("ok\n"), 0o644); err != nil {
		os.RemoveAll(venvDir)
		return fmt.Errorf("venv receipt: %w", err)
	}
	return nil
}

// probePythonMinor asks the configured interpreter for its "major.minor"
// version, used in the venv cache key and env.json. It addresses PythonBin
// directly and touches no project state.
func (m *Manager) probePythonMinor(ctx context.Context) (string, error) {
	res, err := m.runner.Run(ctx, Command{
		Name: m.cfg.PythonBin,
		Args: []string{"-c", "import sys;print('%d.%d'%sys.version_info[:2])"},
	})
	if err != nil {
		return "", fmt.Errorf("probe python minor: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}
	v := strings.TrimSpace(string(res.Stdout))
	if v == "" {
		return "", fmt.Errorf("probe python minor: empty output")
	}
	return v, nil
}
