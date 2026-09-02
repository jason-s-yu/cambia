package procmgr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// captureScript writes a POSIX-sh script that records its argv, physical cwd, and
// a fixed set of env vars to capturePath, then exits 0. It stands in for either
// the fixed cambia binary (empty-opts path) or the venv interpreter
// (parameterized path). It uses `pwd -P` deliberately: the child inherits the
// parent's stale PWD env, so a logical pwd would report the wrong directory.
func captureScript(t *testing.T, dir, name, capturePath string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	body := "#!/bin/sh\n{\n" +
		"  echo \"ARGV: $@\"\n" +
		"  echo \"CWD: $(pwd -P)\"\n" +
		"  echo \"MARKER: ${CAMBIA_LAUNCH_MARKER}\"\n" +
		"  echo \"LIBCAMBIA_PATH: ${LIBCAMBIA_PATH}\"\n" +
		"  echo \"PYTHONPATH: ${PYTHONPATH}\"\n" +
		"} > \"" + capturePath + "\"\nexit 0\n"
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// captureLine returns the value after "PREFIX: " for the given prefix in the
// capture file's content, or fails the test.
func captureLine(t *testing.T, content, prefix string) string {
	t.Helper()
	for _, line := range strings.Split(content, "\n") {
		if v, ok := strings.CutPrefix(line, prefix+": "); ok {
			return v
		}
	}
	t.Fatalf("capture missing %q line in:\n%s", prefix, content)
	return ""
}

// TestStartWithOptsParameterized is the M3 launch-parameterization test: a
// LaunchOpts with Python set must exec that interpreter with the given argv from
// the given cwd, with the given env appended, and drive the normal process.json
// lifecycle (running -> stopped with an exit code) exactly as the fixed-binary
// path does.
func TestStartWithOptsParameterized(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := captureScript(t, base, "fake_python.sh", capture)

	stagedCwd := filepath.Join(base, "worktree", "cfr")
	if err := os.MkdirAll(stagedCwd, 0o755); err != nil {
		t.Fatal(err)
	}

	m, _ := newTestManager(t, crashStub) // cambiaBin (crashStub) must be ignored here
	createRun(t, m, "wopts", "prt-cfr")

	rendered := filepath.Join(base, "runs", "wopts", "config.yaml")
	lopts := LaunchOpts{
		Python: interp,
		Argv:   []string{"-m", "src.cli", "train", "prtcfr", "--config", rendered},
		Cwd:    stagedCwd,
		Env: []string{
			"CAMBIA_LAUNCH_MARKER=parameterized",
			"LIBCAMBIA_PATH=/staged/libcambia.so",
			"PYTHONPATH=/staged/shim:/staged/cfr",
		},
	}

	st, err := m.StartWithOpts("wopts", StartOpts{}, lopts)
	if err != nil {
		t.Fatalf("StartWithOpts: %v", err)
	}
	if st.Status != StatusRunning {
		t.Fatalf("status = %q, want running", st.Status)
	}
	if st.PID <= 0 || st.PGID <= 0 {
		t.Fatalf("pid=%d pgid=%d, want both > 0 (process-group setup preserved)", st.PID, st.PGID)
	}

	final := waitForStatus(t, m, "wopts", StatusStopped, 10*time.Second)
	if final.ExitCode == nil || *final.ExitCode != 0 {
		t.Errorf("exit_code = %v, want 0", final.ExitCode)
	}
	if final.StartedAt == "" || final.FinishedAt == "" {
		t.Errorf("lifecycle timestamps missing: started=%q finished=%q", final.StartedAt, final.FinishedAt)
	}

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	content := string(data)

	wantArgv := "-m src.cli train prtcfr --config " + rendered
	if got := captureLine(t, content, "ARGV"); got != wantArgv {
		t.Errorf("argv = %q, want %q", got, wantArgv)
	}

	wantCwd, err := filepath.EvalSymlinks(stagedCwd)
	if err != nil {
		t.Fatal(err)
	}
	if got := captureLine(t, content, "CWD"); got != wantCwd {
		t.Errorf("cwd = %q, want %q (staged worktree cfr dir)", got, wantCwd)
	}

	if got := captureLine(t, content, "MARKER"); got != "parameterized" {
		t.Errorf("marker env = %q, want %q (LaunchOpts.Env not applied)", got, "parameterized")
	}
	if got := captureLine(t, content, "LIBCAMBIA_PATH"); got != "/staged/libcambia.so" {
		t.Errorf("LIBCAMBIA_PATH = %q, want /staged/libcambia.so", got)
	}
	if got := captureLine(t, content, "PYTHONPATH"); got != "/staged/shim:/staged/cfr" {
		t.Errorf("PYTHONPATH = %q, want /staged/shim:/staged/cfr", got)
	}
}

// TestStartWithOptsChildEnvAllowlist is AC(6): a parameterized launch's child
// environment carries only the allowlist (HOME, PATH, LANG, TZ, TMPDIR, taken
// from the agent's own ambient values) plus the harness-set variables in
// LaunchOpts.Env - never the agent's own VIRTUAL_ENV, UV_*, or ambient
// PYTHONPATH (design 3.3, D17: the editable-install trap on a dev host).
func TestStartWithOptsChildEnvAllowlist(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := captureEnvScript(t, base, capture)

	stagedCwd := filepath.Join(base, "worktree", "cfr")
	if err := os.MkdirAll(stagedCwd, 0o755); err != nil {
		t.Fatal(err)
	}

	// Ambient contamination the allowlist must drop.
	t.Setenv("VIRTUAL_ENV", "/home/dev/.venv")
	t.Setenv("UV_CACHE_DIR", "/home/dev/.cache/uv")
	t.Setenv("PYTHONPATH", "/home/dev/ambient-src")
	t.Setenv("CONDA_PREFIX", "/home/dev/conda")
	// Ambient values the allowlist must let through unchanged.
	t.Setenv("LANG", "en_US.UTF-8")

	m, _ := newTestManager(t, crashStub)
	createRun(t, m, "envopts", "prt-cfr")

	lopts := LaunchOpts{
		Python: interp,
		Argv:   []string{"-c", "pass"},
		Cwd:    stagedCwd,
		Env: []string{
			"PYTHONPATH=/staged/shim:/staged/cfr",
			"LIBCAMBIA_PATH=/staged/libcambia.so",
		},
	}
	if _, err := m.StartWithOpts("envopts", StartOpts{}, lopts); err != nil {
		t.Fatalf("StartWithOpts: %v", err)
	}
	waitForStatus(t, m, "envopts", StatusStopped, 10*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	kv := parseCaptureEnv(t, string(data))

	if got := kv["VIRTUAL_ENV"]; got != "" {
		t.Errorf("VIRTUAL_ENV leaked into child env: %q", got)
	}
	if got := kv["UV_CACHE_DIR"]; got != "" {
		t.Errorf("UV_CACHE_DIR leaked into child env: %q", got)
	}
	if got := kv["CONDA_PREFIX"]; got != "" {
		t.Errorf("CONDA_PREFIX leaked into child env: %q", got)
	}
	if got := kv["PYTHONPATH"]; got != "/staged/shim:/staged/cfr" {
		t.Errorf("PYTHONPATH = %q, want the harness-set value, not the ambient one", got)
	}
	if got := kv["LIBCAMBIA_PATH"]; got != "/staged/libcambia.so" {
		t.Errorf("LIBCAMBIA_PATH = %q, want /staged/libcambia.so", got)
	}
	if got := kv["LANG"]; got != "en_US.UTF-8" {
		t.Errorf("LANG = %q, want the allowlisted ambient value en_US.UTF-8", got)
	}
	if got := kv["HOME"]; got == "" {
		t.Error("HOME dropped; the allowlist should let it through")
	}
}

// captureEnvScript writes a POSIX-sh script that dumps its full environment
// (one KEY=VALUE per line) to capturePath, then exits 0.
func captureEnvScript(t *testing.T, dir, capturePath string) string {
	t.Helper()
	p := filepath.Join(dir, "capture_env.sh")
	body := "#!/bin/sh\nenv > \"" + capturePath + "\"\nexit 0\n"
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

// parseCaptureEnv parses `env` output (one KEY=VALUE per line) into a map.
func parseCaptureEnv(t *testing.T, content string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, line := range strings.Split(content, "\n") {
		if line == "" {
			continue
		}
		if i := strings.Index(line, "="); i >= 0 {
			out[line[:i]] = line[i+1:]
		}
	}
	return out
}

// TestStartWithOptsEmptyIsLegacy is the empty-opts regression: a zero LaunchOpts
// must reproduce the exact fixed-binary launch - cambiaBin run from cfrDir with
// the algorithm subcommand plus --config/--run-name/--save-path - proving the
// additive method did not change the dashboard's launch behavior.
func TestStartWithOptsEmptyIsLegacy(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")

	runsDir := filepath.Join(base, "runs")
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfrDir := filepath.Join(base, "cfr")
	if err := os.MkdirAll(cfrDir, 0o755); err != nil {
		t.Fatal(err)
	}
	bin := captureScript(t, base, "cambia_capture.sh", capture)

	m := NewProcessManager(runsDir, cfrDir, bin, nil, TrainAlgorithms())
	t.Cleanup(func() { m.KillAll() })
	createRun(t, m, "run-empty", "prt-cfr")

	if _, err := m.StartWithOpts("run-empty", StartOpts{}, LaunchOpts{}); err != nil {
		t.Fatalf("StartWithOpts(empty): %v", err)
	}
	waitForStatus(t, m, "run-empty", StatusStopped, 10*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	content := string(data)

	absConfig, _ := filepath.Abs(filepath.Join(runsDir, "run-empty", "config.yaml"))
	absRunDir, _ := filepath.Abs(filepath.Join(runsDir, "run-empty"))
	wantArgv := "train prtcfr --config " + absConfig + " --run-name run-empty --save-path " + absRunDir
	if got := captureLine(t, content, "ARGV"); got != wantArgv {
		t.Errorf("argv = %q, want %q (fixed-binary argv changed)", got, wantArgv)
	}

	wantCwd, err := filepath.EvalSymlinks(cfrDir)
	if err != nil {
		t.Fatal(err)
	}
	if got := captureLine(t, content, "CWD"); got != wantCwd {
		t.Errorf("cwd = %q, want %q (fixed-binary cwd = cfrDir)", got, wantCwd)
	}
}
