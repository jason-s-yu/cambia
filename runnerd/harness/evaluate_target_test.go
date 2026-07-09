package harness

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// -----------------------------------------------------------------------
// JobSpec unit tests (target field: required/forbidden semantics, guards)
// -----------------------------------------------------------------------

func TestJobSpecTargetForbidden(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"train-with-target", JobSpec{Kind: KindTrain, Target: "x"}, true},
		{"train-without-target", JobSpec{Kind: KindTrain}, false},
		{"evaluate-with-target", JobSpec{Kind: KindEvaluate, Target: "x"}, false},
		{"bench-with-target", JobSpec{Kind: KindBench, Target: "x"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.targetForbidden(); got != tc.want {
				t.Errorf("targetForbidden() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecContainedTarget(t *testing.T) {
	s := JobSpec{Kind: KindEvaluate, Target: "prior-run"}
	got := s.containedTarget()
	if len(got) != 1 || got[0].label != "target" || got[0].value != "prior-run" {
		t.Fatalf("containedTarget() = %+v, want [{target prior-run}]", got)
	}

	// Non-evaluate kinds get no containment entry: target is unused for them.
	s2 := JobSpec{Kind: KindTrain, Target: ""}
	if got := s2.containedTarget(); got != nil {
		t.Fatalf("containedTarget() for train = %+v, want nil", got)
	}
	s3 := JobSpec{Kind: KindBench, Target: "x"}
	if got := s3.containedTarget(); got != nil {
		t.Fatalf("containedTarget() for bench = %+v, want nil", got)
	}
}

func TestJobSpecGamesOrDefault(t *testing.T) {
	cases := []struct {
		games int
		want  int
	}{
		{0, 5000},
		{-1, 5000},
		{250, 250},
	}
	for _, tc := range cases {
		s := JobSpec{Games: tc.games}
		if got := s.gamesOrDefault(); got != tc.want {
			t.Errorf("gamesOrDefault() with Games=%d = %d, want %d", tc.games, got, tc.want)
		}
	}
}

// -----------------------------------------------------------------------
// Submit-time guards (design 2.6/5.4): evaluate requires target, train
// forbids it, and target gets the same lexical + containment guard as
// checkpoint_a/b (TestSubmitCheckpointGuards is the sibling test).
// -----------------------------------------------------------------------

// evalAlgos maps kind "evaluate" (subcommand "evaluate") and "train"
// (subcommand "train prtcfr", mirroring HarnessAlgorithms) to the real
// subcommand text. Used only by the argv-precision tests below, which launch
// through the staged (VenvPython-set) capture interpreter that ignores argv
// and always exits fast regardless of subcommand.
func evalAlgos() map[string][]string {
	return map[string][]string{
		"evaluate": {"evaluate"},
		"train":    {"train", "prtcfr"},
	}
}

// evalGuardAlgos maps kind "evaluate"/"train" to the fake-cambia script's
// fast-exit subcommand ("quick"). Used by the submit-guard tests, which run
// through the M2 fixed-binary path (no staged interpreter) and only check
// HTTP status codes, never argv content; h2hAlgos in launch_e2e_test.go is
// the analog for head-to-head.
func evalGuardAlgos() map[string][]string {
	return map[string][]string{
		"evaluate": {"quick"},
		"train":    {"quick"},
	}
}

// evalSpec returns a minimal evaluate job spec body. games is omitted from
// the payload when 0, so the harness default (5000) is exercised.
func evalSpec(name, target string, games int) map[string]any {
	spec := map[string]any{
		"kind":   "evaluate",
		"commit": strings.Repeat("a", 40),
		"name":   name,
		"device": "cpu",
		"target": target,
	}
	if games > 0 {
		spec["games"] = games
	}
	return spec
}

func TestSubmitTargetGuards(t *testing.T) {
	reject := []struct {
		name, target string
	}{
		{"eval-abs", "/etc/passwd"},
		{"eval-dotdot", "../../etc/passwd"},
		{"eval-nested-dotdot", "ok/a/../../../escape.pt"},
		{"eval-missing", ""},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, rigConfig{algos: evalGuardAlgos()})
			resp := r.do(http.MethodPost, "/harness/jobs", evalSpec(tc.name, tc.target, 0))
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("got %d, want 400", resp.StatusCode)
			}
			var body map[string]string
			decodeBody(t, resp, &body)
			if body["error"] != "invalid_path" {
				t.Fatalf("error = %q, want invalid_path", body["error"])
			}
			if !strings.Contains(body["detail"], "target") {
				t.Fatalf("detail = %q, want mention of target", body["detail"])
			}
		})
	}

	t.Run("train-with-target-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: evalGuardAlgos()})
		spec := baseSpec("train-bad-target", "train")
		spec["target"] = "some-run"
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_target" {
			t.Fatalf("error = %q, want invalid_target", body["error"])
		}
	})

	t.Run("evaluate-without-target-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: evalGuardAlgos()})
		spec := map[string]any{
			"kind":   "evaluate",
			"commit": strings.Repeat("a", 40),
			"name":   "eval-no-target",
			"device": "cpu",
		}
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_path" {
			t.Fatalf("error = %q, want invalid_path (target required)", body["error"])
		}
	})

	t.Run("accept-contained", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: evalGuardAlgos()})
		if err := os.MkdirAll(filepath.Join(r.runsDir, "prior-run"), 0o755); err != nil {
			t.Fatal(err)
		}
		resp := r.do(http.MethodPost, "/harness/jobs", evalSpec("eval-ok", "prior-run", 0))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201 for contained target", resp.StatusCode)
		}
		resp.Body.Close()
		r.waitForState("eval-ok", procmgr.StatusStopped, 5*time.Second)
	})
}

// -----------------------------------------------------------------------
// launchOpts argv construction for kind=evaluate (spec-review finding #1):
// positional target, --latest for a dir target, direct path for a file
// target, --games propagation/default, --config omitted, --device present.
// Exercised through the staged (VenvPython-set) launch path, like
// TestDispatcherLaunchesFromStagedEnv, since the M2 fixed-binary fallback
// (VenvPython empty) never calls evaluateTargetArgv.
// -----------------------------------------------------------------------

func TestEvaluateArgvDirTargetUsesLatest(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se, algos: evalAlgos()})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp

	// target names a prior run's directory (e.g. a completed training run),
	// distinct from this evaluate job's own run dir.
	targetDir := filepath.Join(r.runsDir, "prior-run")
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		t.Fatal(err)
	}

	resp := r.do(http.MethodPost, "/harness/jobs", evalSpec("eval-dir", "prior-run", 250))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("eval-dir", procmgr.StatusStopped, 5*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture (interpreter did not run): %v", err)
	}
	argv := captureField(t, string(data), "ARGV")

	wantTarget, err := filepath.EvalSymlinks(targetDir)
	if err != nil {
		t.Fatal(err)
	}
	wantArgv := "-m src.cli evaluate " + wantTarget + " --latest --games 250 --device cpu"
	if argv != wantArgv {
		t.Errorf("argv = %q, want %q", argv, wantArgv)
	}

	// The eval job's run_db journal is the EVALUATED run's, not its own: eval
	// rows must land in the target's run_db.sqlite so they sync with that run
	// (design 4.2).
	wantDB := filepath.Join(r.runsDir, "prior-run", "run_db.sqlite")
	if got := captureField(t, string(data), "CAMBIA_RUN_DB"); got != wantDB {
		t.Errorf("CAMBIA_RUN_DB = %q, want %q", got, wantDB)
	}
}

// TestEvaluateFileTargetFailsJob asserts a checkpoint-file target never launches.
// cli.py's file mode leaves agent_type at deep_cfr and only recovers a run dir
// under checkpoints/, so a PRT-CFR snapshot would evaluate under the wrong agent
// wrapper and report plausible, wrong numbers. The job fails instead.
func TestEvaluateFileTargetFailsJob(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se, algos: evalAlgos()})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp

	targetDir := filepath.Join(r.runsDir, "prior-run", "snapshots")
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		t.Fatal(err)
	}
	targetFile := filepath.Join(targetDir, "prtcfr_checkpoint_100.pt")
	if err := os.WriteFile(targetFile, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	resp := r.do(http.MethodPost, "/harness/jobs",
		evalSpec("eval-file", "prior-run/snapshots/prtcfr_checkpoint_100.pt", 0))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("eval-file", StateFailed, 5*time.Second)

	if _, err := os.ReadFile(capture); err == nil {
		t.Error("interpreter ran for a file target; want the job failed before launch")
	}

	st, ok := r.pm.GetState("eval-file")
	if !ok {
		t.Fatal("no process state for eval-file")
	}
	if !strings.Contains(st.LastError, "run directory") {
		t.Errorf("LastError = %q, want it to explain the run-directory requirement", st.LastError)
	}
}

// TestTrainArgvUnchangedByEvaluateTarget confirms kind=train still gets the
// pre-existing --config argv (the else branch in launchOpts), unaffected by
// the evaluate-only target handling added alongside it.
func TestTrainArgvUnchangedByEvaluateTarget(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se, algos: evalAlgos()})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp

	resp := r.do(http.MethodPost, "/harness/jobs", baseSpec("train-ok", "train"))
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("train-ok", procmgr.StatusStopped, 5*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture (interpreter did not run): %v", err)
	}
	argv := captureField(t, string(data), "ARGV")

	rendered := filepath.Join(r.runsDir, "train-ok", "config.yaml")
	wantArgv := "-m src.cli train prtcfr --config " + rendered +
		" --run-name train-ok --save-path " + filepath.Join(r.runsDir, "train-ok")
	if argv != wantArgv {
		t.Errorf("argv = %q, want %q (train argv: rendered config + explicit run identity)", argv, wantArgv)
	}

	// A train job journals into its own run dir (design 4.2 wire format).
	wantDB := filepath.Join(r.runsDir, "train-ok", "run_db.sqlite")
	if got := captureField(t, string(data), "CAMBIA_RUN_DB"); got != wantDB {
		t.Errorf("CAMBIA_RUN_DB = %q, want %q", got, wantDB)
	}
}
