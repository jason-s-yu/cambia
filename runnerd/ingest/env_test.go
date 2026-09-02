package ingest

import (
	"context"
	"errors"
	"os"
	"os/exec"
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
		JobID:             "job-e",
		Commit:            strings.Repeat("a", 40),
		EngineTreeSha:     "engtree123",
		LibcambiaCacheKey: "engtree123-abigen1",
		LibcambiaSha:      "libsha456",
		UVLockSha:         "locksha789",
		VenvCacheKey:      "locksha789-py3.11-linux_amd64",
		PlatformTag:       "linux_amd64",
		Device:            "cpu",
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
		"libcambia_cache_key": rec.LibcambiaCacheKey,
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

// TestSitecustomizeGuardFiresOnStraySrcAheadOfWorktree is AC(7): the guard
// still catches a stray `src` package that resolves ahead of the pinned
// worktree's own cfr/src (the cambia-240 class trap), spawning a real python3
// with only the harness-constructed environment (design 3.3, D17).
//
// A SystemExit raised from inside sitecustomize during interpreter startup
// does not surface as its own exit code: CPython treats an uncaught exception
// during site initialization as a fatal startup error and always exits 1
// (verified here against the system python3 3.13.2 and pyenv 3.12.7; both
// print "Fatal Python error: init_import_site"). The assertion below is on
// the guard actually firing - a non-zero exit plus its diagnostic message -
// not on a specific exit code, since 97 is never the process's own exit code
// through this path.
func TestSitecustomizeGuardFiresOnStraySrcAheadOfWorktree(t *testing.T) {
	pythonBin, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}

	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	worktreeDir := t.TempDir()
	cfrDir := filepath.Join(worktreeDir, "cfr")
	mustWrite(t, filepath.Join(cfrDir, "src", "__init__.py"), "")

	// A stray decoy src package in its own directory, placed ahead of the
	// pinned worktree's cfr dir on PYTHONPATH.
	decoyDir := t.TempDir()
	mustWrite(t, filepath.Join(decoyDir, "src", "__init__.py"), "")

	env, err := m.assembleEnv(worktreeDir, "/fake/libcambia.so")
	if err != nil {
		t.Fatalf("assembleEnv: %v", err)
	}
	kv := envMap(env)
	pythonPath := decoyDir + string(os.PathListSeparator) + kv["PYTHONPATH"]

	cmd := exec.Command(pythonBin, "-c", "import src")
	cmd.Env = []string{
		"PYTHONPATH=" + pythonPath,
		"CAMBIA_EXPECTED_SRC_ROOT=" + kv["CAMBIA_EXPECTED_SRC_ROOT"],
		"PYTHONNOUSERSITE=1",
		"HOME=" + os.Getenv("HOME"),
		"PATH=" + os.Getenv("PATH"),
	}
	out, runErr := cmd.CombinedOutput()

	var exitErr *exec.ExitError
	if !errors.As(runErr, &exitErr) {
		t.Fatalf("expected the guard to fail the process; err=%v output=%s", runErr, out)
	}
	if exitErr.ExitCode() == 0 {
		t.Fatalf("exit code = 0, want non-zero; output=%s", out)
	}
	if !strings.Contains(string(out), "src resolves to") || !strings.Contains(string(out), decoyDir) {
		t.Fatalf("guard did not report the decoy src as the resolved origin: %s", out)
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

// TestEnvJSONNameRedirectsTheStagingRecord covers the embedded node's half of
// D40: on --role both the coordinator has already authored env.json with
// executed_on in the run dir this stage writes into, so the manager is pointed
// at env.node.json. Without the redirect the write-once rule would silently
// drop the staging record, leaving the run with no venv or libcambia cache key
// on disk and the cache sweep unable to protect a live job's interpreter.
func TestEnvJSONNameRedirectsTheStagingRecord(t *testing.T) {
	fr := newFakeRunner()
	fr.hook = newFakeControl().hook()
	base := t.TempDir()
	m := New(Config{
		BaseDir:     base,
		RunsDir:     filepath.Join(base, "runs"),
		Runner:      fr,
		EnvJSONName: "env.node.json",
	})
	runDir := filepath.Join(base, "runs", "job-embedded")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	coordinator := []byte(`{"job_id":"job-embedded","executed_on":"n-abcdef012345"}`)
	if err := os.WriteFile(filepath.Join(runDir, envJSONFile), coordinator, 0o644); err != nil {
		t.Fatal(err)
	}

	prov := provenance{JobID: "job-embedded", VenvCacheKey: "lock-py311", Device: "cpu"}
	if err := m.writeEnvJSON(context.Background(), runDir, "python", prov); err != nil {
		t.Fatalf("writeEnvJSON: %v", err)
	}

	staged, err := readEnvJSON(filepath.Join(runDir, "env.node.json"))
	if err != nil {
		t.Fatalf("the staging record was not written under its own name: %v", err)
	}
	if staged.VenvCacheKey != "lock-py311" {
		t.Errorf("env.node.json venv_cache_key = %q, want lock-py311", staged.VenvCacheKey)
	}
	kept, err := os.ReadFile(filepath.Join(runDir, envJSONFile))
	if err != nil {
		t.Fatalf("read the coordinator's env.json: %v", err)
	}
	if string(kept) != string(coordinator) {
		t.Errorf("the coordinator's env.json was rewritten: %q", kept)
	}

	// The cache sweep reads the same name it writes, so a live embedded job's
	// interpreter is still protected from eviction.
	venvKeys, _ := m.liveCacheKeys([]string{"job-embedded"})
	if !venvKeys["lock-py311"] {
		t.Errorf("live venv keys = %v, want the embedded run's key protected", venvKeys)
	}
}
