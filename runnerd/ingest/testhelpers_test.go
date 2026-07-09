package ingest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeRunner records every invocation and delegates git (and any other command
// its fake hook declines) to a real ExecRunner. uv/python/go/uname probes are
// intercepted by the hook so toolchain builds are fakeable while git runs for
// real against temp repositories, per the M3 testability contract.
type fakeRunner struct {
	real  CommandRunner
	mu    sync.Mutex
	calls []Command
	// hook returns handled=false to delegate to the real runner (git).
	hook func(c Command) (Result, error, bool)
}

func newFakeRunner() *fakeRunner {
	return &fakeRunner{real: ExecRunner{}}
}

func (f *fakeRunner) Run(ctx context.Context, c Command) (Result, error) {
	f.mu.Lock()
	f.calls = append(f.calls, c)
	f.mu.Unlock()
	if f.hook != nil {
		if res, err, handled := f.hook(c); handled {
			return res, err
		}
	}
	return f.real.Run(ctx, c)
}

// callsFor returns the recorded invocations whose Name has the given basename.
func (f *fakeRunner) callsFor(name string) []Command {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []Command
	for _, c := range f.calls {
		if filepath.Base(c.Name) == name {
			out = append(out, c)
		}
	}
	return out
}

// ok is a convenience for a successful faked result.
func ok(stdout string) (Result, error, bool) {
	return Result{Stdout: []byte(stdout)}, nil, true
}

// defaultFakeHook simulates a successful toolchain: uv lock --check passes, uv
// venv materializes a bin/python, uv sync passes, go build writes an .so,
// python/uv/go/uname probes return canned values, and config render writes its
// -o target. Individual tests override fields of the returned closure's captured
// state via the returned *fakeControl.
type fakeControl struct {
	lockCheckFails bool
	syncFails      bool
	pyMinor        string
	pyVersion      string
	pipFreeze      string
	goVersion      string
	kernel         string
}

func (fc *fakeControl) hook() func(Command) (Result, error, bool) {
	return func(c Command) (Result, error, bool) {
		base := filepath.Base(c.Name)
		switch base {
		case "uv":
			return fc.handleUV(c)
		case "go":
			return fc.handleGo(c)
		case "uname":
			return ok(fc.kernel + "\n")
		case "python3", "python":
			return fc.handlePython(c)
		default:
			// Not a toolchain command: could be the venv python (an absolute
			// path ending in /bin/python) driving render/validate/probes.
			if strings.HasSuffix(c.Name, "/bin/python") {
				return fc.handlePython(c)
			}
			return Result{}, nil, false // delegate (git)
		}
	}
}

func (fc *fakeControl) handleUV(c Command) (Result, error, bool) {
	if len(c.Args) >= 2 && c.Args[0] == "lock" && c.Args[1] == "--check" {
		if fc.lockCheckFails {
			return Result{Stderr: []byte("lock is out of date")}, exec.ErrNotFound, true
		}
		return ok("")
	}
	if len(c.Args) >= 1 && c.Args[0] == "venv" {
		// Args: venv <dir> --python <bin>. Materialize <dir>/bin/python.
		venvDir := c.Args[1]
		_ = os.MkdirAll(filepath.Join(venvDir, "bin"), 0o755)
		_ = os.WriteFile(filepath.Join(venvDir, "bin", "python"), []byte("#!/bin/sh\n"), 0o755)
		return ok("")
	}
	if len(c.Args) >= 1 && c.Args[0] == "sync" {
		if fc.syncFails {
			return Result{Stderr: []byte("sync failed")}, exec.ErrNotFound, true
		}
		return ok("")
	}
	if len(c.Args) >= 2 && c.Args[0] == "pip" && c.Args[1] == "freeze" {
		return ok(fc.pipFreeze)
	}
	return ok("")
}

func (fc *fakeControl) handleGo(c Command) (Result, error, bool) {
	if len(c.Args) >= 1 && c.Args[0] == "version" {
		return ok(fc.goVersion + "\n")
	}
	if len(c.Args) >= 1 && c.Args[0] == "build" {
		// Find -o <out> and write a fake .so there.
		for i := 0; i < len(c.Args)-1; i++ {
			if c.Args[i] == "-o" {
				out := c.Args[i+1]
				_ = os.MkdirAll(filepath.Dir(out), 0o755)
				_ = os.WriteFile(out, []byte("\x7fELF fake libcambia"), 0o644)
			}
		}
		return ok("")
	}
	return ok("")
}

func (fc *fakeControl) handlePython(c Command) (Result, error, bool) {
	joined := strings.Join(c.Args, " ")
	switch {
	case strings.Contains(joined, "version_info"):
		return ok(fc.pyMinor + "\n")
	case strings.Contains(joined, "python_version"):
		return ok(fc.pyVersion + "\n")
	case strings.Contains(joined, "config render"):
		// Args: -m src.cli config render <cfg> --set ... -o <out>. Write <out>.
		for i := 0; i < len(c.Args)-1; i++ {
			if c.Args[i] == "-o" {
				out := c.Args[i+1]
				_ = os.MkdirAll(filepath.Dir(out), 0o755)
				_ = os.WriteFile(out, []byte("rendered: true\n"), 0o644)
			}
		}
		return ok("")
	case strings.Contains(joined, "config validate"):
		return ok("OK")
	default:
		return ok("")
	}
}

func newFakeControl() *fakeControl {
	return &fakeControl{
		pyMinor:   "3.11",
		pyVersion: "3.11.9",
		pipFreeze: "numpy==1.26.4\ntorch==2.6.0+cpu\n",
		goVersion: "go version go1.26.0 linux/amd64",
		kernel:    "6.8.0-test",
	}
}

// runGit runs a real git command in dir and fails the test on error.
func runGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@test",
		"GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@test",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s in %s: %v\n%s", strings.Join(args, " "), dir, err, out)
	}
	return strings.TrimSpace(string(out))
}

// sourceRepo builds a real repo with cfr/uv.lock, cfr/pyproject.toml, and an
// engine/ tree, committed. It returns the repo dir and the commit sha.
func sourceRepo(t *testing.T, lockContent string) (dir, sha string) {
	t.Helper()
	dir = t.TempDir()
	runGit(t, dir, "init", "-q")
	runGit(t, dir, "config", "user.email", "test@test")
	runGit(t, dir, "config", "user.name", "test")
	mustWrite(t, filepath.Join(dir, "cfr", "uv.lock"), lockContent)
	mustWrite(t, filepath.Join(dir, "cfr", "pyproject.toml"), "[project]\nname='cambia-cfr'\n")
	mustWrite(t, filepath.Join(dir, "cfr", "src", "cli.py"), "# cli\n")
	mustWrite(t, filepath.Join(dir, "engine", "go.mod"), "module e\n\ngo 1.26.0\n")
	mustWrite(t, filepath.Join(dir, "engine", "cgo", "exports.go"), "package main\n\nfunc main(){}\n")
	runGit(t, dir, "add", "-A")
	runGit(t, dir, "commit", "-q", "-m", "init")
	sha = runGit(t, dir, "rev-parse", "HEAD")
	return dir, sha
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// testManager builds a Manager over a fresh BaseDir with the given runner and a
// controllable clock. It initializes the bare mirror so callers can push refs.
func testManager(t *testing.T, runner CommandRunner) (*Manager, string) {
	t.Helper()
	base := t.TempDir()
	runsDir := filepath.Join(base, "runs")
	m := New(Config{
		BaseDir:      base,
		RunsDir:      runsDir,
		MaxVenvs:     8,
		MaxLibcambia: 50,
		CoresCap:     18,
		PythonBin:    "python3",
		Runner:       runner,
		Now:          time.Now,
	})
	if err := m.ensureMirror(context.Background()); err != nil {
		t.Fatalf("ensureMirror: %v", err)
	}
	return m, base
}

// pushJobRef pushes sha from srcDir into the manager's mirror under
// refs/harness/<jobID>, the shape the submit-time push produces.
func pushJobRef(t *testing.T, m *Manager, srcDir, sha, jobID string) {
	t.Helper()
	runGit(t, srcDir, "push", "-q", m.mirrorDir, sha+":"+jobRef(jobID))
}
