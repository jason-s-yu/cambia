package ingest

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// envJSONFile is the basename of the write-once provenance record, a sibling of
// process.json in the run dir (design 3.6).
const envJSONFile = "env.json"

// originHostEnvVar overrides the host recorded in provenance (design 3.6).
const originHostEnvVar = "RUNNERD_ORIGIN_HOST"

// resolveOriginHost returns the host recorded in provenance: originHostEnvVar
// if set, else the local hostname (design 3.6).
func resolveOriginHost() string {
	if h := os.Getenv(originHostEnvVar); h != "" {
		return h
	}
	h, _ := os.Hostname()
	return h
}

// sitecustomizeName is the shim module that asserts the imported src package
// resolves under the pinned worktree (cambia-240 class).
const sitecustomizeName = "sitecustomize.py"

// srcGuardShim is a side-effect-free interpreter-startup guard. It runs before
// the job imports src and verifies that src would resolve under
// CAMBIA_EXPECTED_SRC_ROOT (the pinned worktree cfr dir), hard-failing otherwise.
// It uses importlib.util.find_spec, which locates the top-level src package
// without executing it.
const srcGuardShim = `# cambia serving-harness src-containment guard (cambia-240).
# Asserts the pinned worktree src is what the job imports, not an ambient copy
# leaked via a stray .pth or user-site editable install.
import os
import sys

_expected = os.environ.get("CAMBIA_EXPECTED_SRC_ROOT")
if _expected:
    try:
        import importlib.util

        _spec = importlib.util.find_spec("src")
    except Exception:
        _spec = None
    _origin = None
    if _spec is not None:
        if _spec.submodule_search_locations:
            _origin = list(_spec.submodule_search_locations)[0]
        elif _spec.origin:
            _origin = os.path.dirname(_spec.origin)
    _root = os.path.realpath(_expected)
    _got = os.path.realpath(_origin) if _origin else None
    if not _got or not (_got == _root or _got.startswith(_root + os.sep)):
        sys.stderr.write(
            "cambia harness: src resolves to %r, not under %r\n" % (_got, _root)
        )
        raise SystemExit(97)
`

// provenance carries the fields Prepare resolves during staging; writeEnvJSON
// fills the remaining probe-derived fields.
type provenance struct {
	JobID         string
	Commit        string
	EngineTreeSha string
	LibcambiaSha  string
	UVLockSha     string
	VenvCacheKey  string
	PlatformTag   string
	Device        string
}

// envRecord is the on-disk env.json shape (design 3.6).
type envRecord struct {
	JobID             string `json:"job_id"`
	OriginHost        string `json:"origin_host"`
	Commit            string `json:"commit"`
	EngineTreeSha     string `json:"engine_tree_sha"`
	LibcambiaSha256   string `json:"libcambia_sha256"`
	UVLockSha256      string `json:"uv_lock_sha256"`
	VenvCacheKey      string `json:"venv_cache_key"`
	PythonVersion     string `json:"python_version"`
	PipFreeze         string `json:"pip_freeze"`
	TorchVersion      string `json:"torch_version"`
	TorchWheelTag     string `json:"torch_wheel_tag"`
	GoVersion         string `json:"go_version"`
	GoToolchainPinned string `json:"go_toolchain_pinned"`
	PlatformTag       string `json:"platform_tag"`
	Kernel            string `json:"kernel"`
	CPUModel          string `json:"cpu_model"`
	Device            string `json:"device"`
	CreatedAt         string `json:"created_at"`
}

// assembleEnv builds the harness-controlled launch environment (Prepared.Env).
// It writes the src-guard shim once, then returns PYTHONPATH (shim dir first so
// sitecustomize is importable, then the pinned worktree cfr dir), LIBCAMBIA_PATH,
// the expected-src-root the shim checks, and user-site suppression.
func (m *Manager) assembleEnv(worktreeDir, libcambiaPath string) ([]string, error) {
	if err := m.writeShim(); err != nil {
		return nil, err
	}
	cfrDir := filepath.Join(worktreeDir, "cfr")
	pythonPath := m.shimDir + string(os.PathListSeparator) + cfrDir
	return []string{
		"PYTHONPATH=" + pythonPath,
		"LIBCAMBIA_PATH=" + libcambiaPath,
		"CAMBIA_EXPECTED_SRC_ROOT=" + cfrDir,
		"PYTHONNOUSERSITE=1",
	}, nil
}

// writeShim writes the sitecustomize guard into the shim dir if absent.
func (m *Manager) writeShim() error {
	if err := os.MkdirAll(m.shimDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(m.shimDir, sitecustomizeName)
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	return os.WriteFile(path, []byte(srcGuardShim), 0o644)
}

// writeEnvJSON writes the write-once provenance record. It gathers the
// probe-derived fields (python version, pip freeze, torch, go, kernel, cpu) then
// writes env.json atomically. An existing env.json is left untouched (write-once).
func (m *Manager) writeEnvJSON(ctx context.Context, runDir, venvPython string, prov provenance) error {
	path := filepath.Join(runDir, envJSONFile)
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	rec := envRecord{
		JobID:             prov.JobID,
		OriginHost:        resolveOriginHost(),
		Commit:            prov.Commit,
		EngineTreeSha:     prov.EngineTreeSha,
		LibcambiaSha256:   prov.LibcambiaSha,
		UVLockSha256:      prov.UVLockSha,
		VenvCacheKey:      prov.VenvCacheKey,
		PythonVersion:     m.probePythonVersion(ctx, venvPython),
		GoToolchainPinned: goToolchainPin,
		PlatformTag:       prov.PlatformTag,
		Kernel:            m.probeKernel(ctx),
		CPUModel:          probeCPUModel(),
		Device:            prov.Device,
		CreatedAt:         m.now().UTC().Format(time.RFC3339),
	}
	rec.GoVersion = m.probeGoVersion(ctx)
	freeze := m.probePipFreeze(ctx, venvPython)
	rec.PipFreeze = freeze
	rec.TorchVersion, rec.TorchWheelTag = torchFromFreeze(freeze)

	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// readEnvJSON decodes an env.json for the startup sweep's live-key protection.
func readEnvJSON(path string) (*envRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rec envRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, err
	}
	return &rec, nil
}

// probePythonVersion returns the venv interpreter's full version string.
func (m *Manager) probePythonVersion(ctx context.Context, venvPython string) string {
	res, err := m.runner.Run(ctx, Command{
		Name: venvPython,
		Args: []string{"-c", "import platform;print(platform.python_version())"},
	})
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(res.Stdout))
}

// probePipFreeze returns `uv pip freeze` for the venv, addressed by explicit
// interpreter path (never --active, never ambient).
func (m *Manager) probePipFreeze(ctx context.Context, venvPython string) string {
	res, err := m.runner.Run(ctx, Command{
		Name: "uv",
		Args: []string{"pip", "freeze", "--python", venvPython},
	})
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(res.Stdout))
}

// probeGoVersion returns the go toolchain version string.
func (m *Manager) probeGoVersion(ctx context.Context) string {
	res, err := m.runner.Run(ctx, Command{Name: "go", Args: []string{"version"}})
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(res.Stdout))
}

// probeKernel returns the running kernel release (uname -r).
func (m *Manager) probeKernel(ctx context.Context) string {
	res, err := m.runner.Run(ctx, Command{Name: "uname", Args: []string{"-r"}})
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(res.Stdout))
}

// probeCPUModel returns the first "model name" from /proc/cpuinfo (best-effort;
// a plain file read, not a shelled command).
func probeCPUModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "model name") {
			if i := strings.Index(line, ":"); i >= 0 {
				return strings.TrimSpace(line[i+1:])
			}
		}
	}
	return ""
}

// torchFromFreeze extracts the torch version and its wheel local tag (the "+cpu"
// segment) from pip-freeze output. Empty strings when torch is absent.
func torchFromFreeze(freeze string) (version, wheelTag string) {
	for _, line := range strings.Split(freeze, "\n") {
		l := strings.TrimSpace(line)
		if strings.HasPrefix(l, "torch==") {
			version = strings.TrimPrefix(l, "torch==")
			if i := strings.Index(version, "+"); i >= 0 {
				wheelTag = version[i+1:]
			}
			return version, wheelTag
		}
	}
	return "", ""
}
