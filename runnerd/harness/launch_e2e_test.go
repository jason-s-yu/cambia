package harness

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// stagedEnv is a fake Environment whose Prepare returns a Prepared pointing at a
// fake interpreter, a staged worktree cfr cwd, a rendered config, and a staged
// env. It exercises the M3 parameterized launch: the child must exec the venv
// interpreter from the staged cwd with the staged env, not the manager's fixed
// binary. runsDir/baseDir/interp/env are set after the rig is built (the rig owns
// runsDir), which is safe because Prepare is only called at dispatch, post-submit.
type stagedEnv struct {
	runsDir string
	baseDir string
	interp  string
	env     []string
}

func (s *stagedEnv) Prepare(ctx context.Context, jobID, commit, kind, configRel string, overrides map[string]string) (*ingestapi.Prepared, error) {
	worktreeDir := filepath.Join(s.baseDir, "worktrees", jobID)
	cfrDir := filepath.Join(worktreeDir, "cfr")
	if err := os.MkdirAll(cfrDir, 0o755); err != nil {
		return nil, err
	}
	runDir := filepath.Join(s.runsDir, jobID)
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return nil, err
	}
	rendered := filepath.Join(runDir, "config.yaml")
	if err := os.WriteFile(rendered, []byte("device: cpu\n"), 0o644); err != nil {
		return nil, err
	}
	return &ingestapi.Prepared{
		WorktreeDir:    worktreeDir,
		RunDir:         runDir,
		VenvPython:     s.interp,
		LibcambiaPath:  filepath.Join(s.baseDir, "libcambia", "x.so"),
		RenderedConfig: rendered,
		Env:            s.env,
	}, nil
}

func (s *stagedEnv) Cleanup(string, bool) error  { return nil }
func (s *stagedEnv) StartupSweep([]string) error { return nil }

// writeCaptureInterp writes a fake interpreter that records its argv, physical
// cwd, and selected env to capturePath then exits 0. `pwd -P` is deliberate: the
// child inherits the parent's stale PWD, so a logical pwd would misreport cwd.
func writeCaptureInterp(t *testing.T, dir, capturePath string) string {
	t.Helper()
	p := filepath.Join(dir, "fake_python.sh")
	body := "#!/bin/sh\n{\n" +
		"  echo \"ARGV: $@\"\n" +
		"  echo \"CWD: $(pwd -P)\"\n" +
		"  echo \"MARKER: ${CAMBIA_LAUNCH_MARKER}\"\n" +
		"  echo \"LIBCAMBIA_PATH: ${LIBCAMBIA_PATH}\"\n" +
		"} > \"" + capturePath + "\"\nexit 0\n"
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func captureField(t *testing.T, content, prefix string) string {
	t.Helper()
	for _, line := range strings.Split(content, "\n") {
		if v, ok := strings.CutPrefix(line, prefix+": "); ok {
			return v
		}
	}
	t.Fatalf("capture missing %q line in:\n%s", prefix, content)
	return ""
}

// TestDispatcherLaunchesFromStagedEnv is the dispatcher e2e for launch
// parameterization: a job dispatched against an Environment that stages a venv
// interpreter must run that interpreter from the staged worktree cfr cwd, with
// the rendered config in argv and the staged env applied - not the manager's
// fixed cambia binary.
func TestDispatcherLaunchesFromStagedEnv(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp
	se.env = []string{
		"CAMBIA_LAUNCH_MARKER=staged",
		"LIBCAMBIA_PATH=" + filepath.Join(base, "libcambia", "x.so"),
		"PYTHONPATH=" + filepath.Join(base, "shim"),
	}

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("job-staged", "fake"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()

	// kind "fake" -> subcommand "sleep"; the interpreter ignores args, captures,
	// and exits 0, so the run reaches procmgr terminal "stopped".
	r.waitForState("job-staged", procmgr.StatusStopped, 5*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture (interpreter did not run from staged env): %v", err)
	}
	content := string(data)

	rendered := filepath.Join(r.runsDir, "job-staged", "config.yaml")
	wantArgv := "-m src.cli sleep --config " + rendered
	if got := captureField(t, content, "ARGV"); got != wantArgv {
		t.Errorf("argv = %q, want %q", got, wantArgv)
	}

	wantCwd, err := filepath.EvalSymlinks(filepath.Join(base, "worktrees", "job-staged", "cfr"))
	if err != nil {
		t.Fatal(err)
	}
	if got := captureField(t, content, "CWD"); got != wantCwd {
		t.Errorf("cwd = %q, want %q (staged worktree cfr dir)", got, wantCwd)
	}
	if got := captureField(t, content, "MARKER"); got != "staged" {
		t.Errorf("marker env = %q, want staged (Prepared.Env not applied)", got)
	}
	if got := captureField(t, content, "LIBCAMBIA_PATH"); got != filepath.Join(base, "libcambia", "x.so") {
		t.Errorf("LIBCAMBIA_PATH = %q, want staged path", got)
	}
}

// h2hAlgos is a kind allowlist that includes head-to-head (mapped to the fake
// script's fast-exit subcommand) so the checkpoint-guard submit path is testable.
func h2hAlgos() map[string][]string {
	return map[string][]string{
		"fake":         {"sleep"},
		"head-to-head": {"quick"},
	}
}

func h2hSpec(name, cpA, cpB string) map[string]any {
	return map[string]any{
		"kind":         "head-to-head",
		"commit":       strings.Repeat("a", 40),
		"name":         name,
		"device":       "cpu",
		"games":        10,
		"checkpoint_a": cpA,
		"checkpoint_b": cpB,
	}
}

// TestSubmitCheckpointGuards is the design 5.4 checkpoint-guard test at submit:
// head-to-head checkpoints that are absolute or contain ".." are rejected 400
// before dispatch; repo-relative checkpoints that resolve inside the runs dir are
// accepted.
func TestSubmitCheckpointGuards(t *testing.T) {
	reject := []struct {
		name, cpA, cpB, wantField string
	}{
		{"h2h-abs-a", "/etc/passwd", "runs-x/snap.pt", "checkpoint_a"},
		{"h2h-dotdot-b", "runs-x/snap.pt", "../../etc/passwd", "checkpoint_b"},
		{"h2h-nested-dotdot", "ok/a/../../../escape.pt", "runs-y/snap.pt", "checkpoint_a"},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, rigConfig{algos: h2hAlgos()})
			resp := r.do(http.MethodPost, "/harness/jobs", h2hSpec(tc.name, tc.cpA, tc.cpB))
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("got %d, want 400 for bad %s", resp.StatusCode, tc.wantField)
			}
			var body map[string]string
			decodeBody(t, resp, &body)
			if body["error"] != "invalid_path" {
				t.Fatalf("error = %q, want invalid_path", body["error"])
			}
			if !strings.Contains(body["detail"], tc.wantField) {
				t.Fatalf("detail = %q, want mention of %q", body["detail"], tc.wantField)
			}
		})
	}

	// Accept: both checkpoints repo-relative and contained under the runs dir.
	t.Run("accept-contained", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: h2hAlgos()})
		resp := r.do(http.MethodPost, "/harness/jobs",
			h2hSpec("h2h-ok", "prior-run-a/snapshots/prtcfr_checkpoint.pt", "prior-run-b/snapshots/prtcfr_checkpoint.pt"))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201 for contained checkpoints", resp.StatusCode)
		}
		resp.Body.Close()
		// The accepted job launches the fake script's fast-exit subcommand.
		r.waitForState("h2h-ok", procmgr.StatusStopped, 5*time.Second)
	})
}
