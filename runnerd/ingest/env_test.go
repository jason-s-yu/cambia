package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteEnvJSONCompleteness(t *testing.T) {
	t.Setenv(originHostEnvVar, "testhost")
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	runDir := filepath.Join(t.TempDir(), "runs", "job-e")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	venvPython := filepath.Join(t.TempDir(), "venv", "bin", "python")

	prov := provenance{
		JobID:         "job-e",
		Commit:        strings.Repeat("a", 40),
		EngineTreeSha: "engtree123",
		LibcambiaSha:  "libsha456",
		UVLockSha:     "locksha789",
		VenvCacheKey:  "locksha789-py3.11-linux_amd64",
		PlatformTag:   "linux_amd64",
		Device:        "cpu",
	}
	if err := m.writeEnvJSON(context.Background(), runDir, venvPython, prov); err != nil {
		t.Fatalf("writeEnvJSON: %v", err)
	}

	rec, err := readEnvJSON(filepath.Join(runDir, envJSONFile))
	if err != nil {
		t.Fatalf("readEnvJSON: %v", err)
	}

	checks := map[string]string{
		"job_id":              rec.JobID,
		"origin_host":         rec.OriginHost,
		"commit":              rec.Commit,
		"engine_tree_sha":     rec.EngineTreeSha,
		"libcambia_sha256":    rec.LibcambiaSha256,
		"uv_lock_sha256":      rec.UVLockSha256,
		"venv_cache_key":      rec.VenvCacheKey,
		"python_version":      rec.PythonVersion,
		"pip_freeze":          rec.PipFreeze,
		"torch_version":       rec.TorchVersion,
		"torch_wheel_tag":     rec.TorchWheelTag,
		"go_version":          rec.GoVersion,
		"go_toolchain_pinned": rec.GoToolchainPinned,
		"platform_tag":        rec.PlatformTag,
		"kernel":              rec.Kernel,
		"device":              rec.Device,
		"created_at":          rec.CreatedAt,
	}
	for field, val := range checks {
		if val == "" {
			t.Errorf("env.json field %q is empty", field)
		}
	}

	if rec.OriginHost != "testhost" {
		t.Errorf("origin_host = %q, want testhost", rec.OriginHost)
	}
	if rec.GoToolchainPinned != goToolchainPin {
		t.Errorf("go_toolchain_pinned = %q, want %q", rec.GoToolchainPinned, goToolchainPin)
	}
	if rec.TorchVersion != "2.6.0+cpu" {
		t.Errorf("torch_version = %q, want 2.6.0+cpu", rec.TorchVersion)
	}
	if rec.TorchWheelTag != "cpu" {
		t.Errorf("torch_wheel_tag = %q, want cpu", rec.TorchWheelTag)
	}
	if !strings.Contains(rec.PipFreeze, "torch==2.6.0+cpu") {
		t.Errorf("pip_freeze missing torch line: %q", rec.PipFreeze)
	}
}

func TestWriteEnvJSONWriteOnce(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	runDir := filepath.Join(t.TempDir(), "job-w")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(runDir, envJSONFile)
	if err := os.WriteFile(path, []byte(`{"job_id":"preexisting"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := m.writeEnvJSON(context.Background(), runDir, "python", provenance{JobID: "job-w"}); err != nil {
		t.Fatalf("writeEnvJSON: %v", err)
	}
	rec, err := readEnvJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	if rec.JobID != "preexisting" {
		t.Fatalf("write-once violated: env.json was overwritten (job_id=%q)", rec.JobID)
	}
}

func TestAssembleEnvAndShim(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	worktreeDir := t.TempDir()
	libPath := "/srv/cambia/libcambia/eng.so"

	env, err := m.assembleEnv(worktreeDir, libPath)
	if err != nil {
		t.Fatalf("assembleEnv: %v", err)
	}
	kv := envMap(env)

	cfrDir := filepath.Join(worktreeDir, "cfr")
	if got := kv["CAMBIA_EXPECTED_SRC_ROOT"]; got != cfrDir {
		t.Fatalf("CAMBIA_EXPECTED_SRC_ROOT = %q, want %q", got, cfrDir)
	}
	if got := kv["LIBCAMBIA_PATH"]; got != libPath {
		t.Fatalf("LIBCAMBIA_PATH = %q, want %q", got, libPath)
	}
	// PYTHONPATH: shim dir first, then the pinned cfr dir.
	pp := kv["PYTHONPATH"]
	parts := strings.Split(pp, string(os.PathListSeparator))
	if len(parts) != 2 || parts[0] != m.shimDir || parts[1] != cfrDir {
		t.Fatalf("PYTHONPATH = %q, want shim(%q):cfr(%q)", pp, m.shimDir, cfrDir)
	}
	if kv["PYTHONNOUSERSITE"] != "1" {
		t.Fatalf("PYTHONNOUSERSITE not set")
	}

	// The shim must exist and reference the containment env var.
	shim := filepath.Join(m.shimDir, sitecustomizeName)
	data, err := os.ReadFile(shim)
	if err != nil {
		t.Fatalf("shim not written: %v", err)
	}
	if !strings.Contains(string(data), "CAMBIA_EXPECTED_SRC_ROOT") {
		t.Fatal("shim does not check CAMBIA_EXPECTED_SRC_ROOT")
	}
	if !strings.Contains(string(data), "find_spec") {
		t.Fatal("shim does not resolve src via find_spec")
	}
}

func envMap(env []string) map[string]string {
	out := map[string]string{}
	for _, e := range env {
		if i := strings.Index(e, "="); i >= 0 {
			out[e[:i]] = e[i+1:]
		}
	}
	return out
}
