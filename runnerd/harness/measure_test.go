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

// -----------------------------------------------------------------------
// JobSpec unit tests (measure fields: kind-scoped, script/args/reads guards)
// -- mirrors the warm_start field tests in warm_start_test.go (design D38,
// cambia-1072).
// -----------------------------------------------------------------------

func TestJobSpecScriptForbidden(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"measure-with-script", JobSpec{Kind: KindMeasure, Script: "cfr/scripts/x.py"}, false},
		{"measure-without-script", JobSpec{Kind: KindMeasure}, false},
		{"train-with-script", JobSpec{Kind: KindTrain, Script: "cfr/scripts/x.py"}, true},
		{"evaluate-with-script", JobSpec{Kind: KindEvaluate, Script: "cfr/scripts/x.py"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.scriptForbidden(); got != tc.want {
				t.Errorf("scriptForbidden() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecScriptRequired(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"measure-no-script", JobSpec{Kind: KindMeasure}, true},
		{"measure-with-script", JobSpec{Kind: KindMeasure, Script: "cfr/scripts/x.py"}, false},
		{"train-no-script", JobSpec{Kind: KindTrain}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.scriptRequired(); got != tc.want {
				t.Errorf("scriptRequired() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecScriptRootValid(t *testing.T) {
	cases := []struct {
		script string
		want   bool
	}{
		{"cfr/scripts/measure_gate_gap.py", true},
		{"cfr/scripts/nested/x.py", true},
		{"cfr/scripts/", false},
		{"cfr/scripts", false},
		{"cfr/other/measure_gate_gap.py", false},
		{"cfr/scripts_evil/x.py", false},
		{"scripts/x.py", false},
		{"", false},
	}
	for _, tc := range cases {
		t.Run(tc.script, func(t *testing.T) {
			s := JobSpec{Script: tc.script}
			if got := s.scriptRootValid(); got != tc.want {
				t.Errorf("scriptRootValid(%q) = %v, want %v", tc.script, got, tc.want)
			}
		})
	}
}

func TestJobSpecArgsForbidden(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"measure-with-args", JobSpec{Kind: KindMeasure, Args: []string{"--x"}}, false},
		{"measure-without-args", JobSpec{Kind: KindMeasure}, false},
		{"train-with-args", JobSpec{Kind: KindTrain, Args: []string{"--x"}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.argsForbidden(); got != tc.want {
				t.Errorf("argsForbidden() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecValidateArgs(t *testing.T) {
	t.Run("ordinary-args-ok", func(t *testing.T) {
		s := JobSpec{Args: []string{"--run", "v0.4-x2r-c1", "--shards", "17"}}
		if err := s.validateArgs(); err != nil {
			t.Errorf("validateArgs() = %v, want nil", err)
		}
	})
	t.Run("nul-byte-rejected", func(t *testing.T) {
		s := JobSpec{Args: []string{"a\x00b"}}
		if err := s.validateArgs(); err == nil {
			t.Error("validateArgs() = nil for a NUL byte entry, want an error")
		}
	})
	t.Run("over-cap-rejected", func(t *testing.T) {
		s := JobSpec{Args: []string{strings.Repeat("x", maxMeasureArgLen+1)}}
		if err := s.validateArgs(); err == nil {
			t.Error("validateArgs() = nil for an over-cap entry, want an error")
		}
	})
	t.Run("at-cap-accepted", func(t *testing.T) {
		s := JobSpec{Args: []string{strings.Repeat("x", maxMeasureArgLen)}}
		if err := s.validateArgs(); err != nil {
			t.Errorf("validateArgs() at the cap = %v, want nil", err)
		}
	})
}

func TestJobSpecReadsForbidden(t *testing.T) {
	cases := []struct {
		name string
		spec JobSpec
		want bool
	}{
		{"measure-with-reads", JobSpec{Kind: KindMeasure, Reads: []string{"run-a"}}, false},
		{"measure-without-reads", JobSpec{Kind: KindMeasure}, false},
		{"train-with-reads", JobSpec{Kind: KindTrain, Reads: []string{"run-a"}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.spec.readsForbidden(); got != tc.want {
				t.Errorf("readsForbidden() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestJobSpecContainedReads(t *testing.T) {
	s := JobSpec{Kind: KindMeasure, Reads: []string{"run-a", "run-b"}}
	got := s.containedReads()
	want := []struct{ label, value string }{
		{"reads[0]", "run-a"},
		{"reads[1]", "run-b"},
	}
	if len(got) != len(want) {
		t.Fatalf("containedReads() = %+v, want %+v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("containedReads()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	// Non-measure kinds get no containment entries (already rejected earlier
	// by readsForbidden in the real submit path).
	s2 := JobSpec{Kind: KindTrain, Reads: []string{"run-a"}}
	if got := s2.containedReads(); got != nil {
		t.Fatalf("containedReads() for train = %+v, want nil", got)
	}
}

// -----------------------------------------------------------------------
// Submit-time guards (design D38, AC 1 and 4): script's root allowlist and
// existence-shape, args' NUL/length-cap shape, and reads' containment +
// existence, all enforced before the job ever reaches prepare.
// -----------------------------------------------------------------------

// measureSpec returns a minimal valid cpu measure job spec body. script/args/
// reads are included only when non-empty/non-nil, mirroring trainSpec's
// optional warm_start handling in warm_start_test.go.
func measureSpec(name, script string, args, reads []string) map[string]any {
	spec := map[string]any{
		"kind":   "measure",
		"commit": strings.Repeat("a", 40),
		"name":   name,
		"device": "cpu",
	}
	if script != "" {
		spec["script"] = script
	}
	if args != nil {
		spec["args"] = args
	}
	if reads != nil {
		spec["reads"] = reads
	}
	return spec
}

func TestSubmitMeasureScriptGuards(t *testing.T) {
	reject := []struct {
		name, script string
	}{
		{"measure-script-empty", ""},
		{"measure-script-wrong-root", "cfr/other/measure.py"},
		{"measure-script-root-only", "cfr/scripts/"},
		{"measure-script-abs", "/etc/passwd"},
		{"measure-script-dotdot", "cfr/scripts/../../etc/passwd"},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
			resp := r.do(http.MethodPost, "/harness/jobs", measureSpec(tc.name, tc.script, nil, nil))
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("got %d, want 400", resp.StatusCode)
			}
			var body map[string]string
			decodeBody(t, resp, &body)
			if body["error"] != "invalid_script" {
				t.Fatalf("error = %q, want invalid_script", body["error"])
			}
		})
	}

	t.Run("script-forbidden-on-train", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := baseSpec("train-with-script", "train")
		spec["script"] = "cfr/scripts/measure_gate_gap.py"
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_script" {
			t.Fatalf("error = %q, want invalid_script", body["error"])
		}
	})

	t.Run("accept-valid-script-shape", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		resp := r.do(http.MethodPost, "/harness/jobs",
			measureSpec("measure-shape-ok", "cfr/scripts/measure_gate_gap.py", nil, nil))
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201 for a well-formed measure script field", resp.StatusCode)
		}
		resp.Body.Close()
	})
}

func TestSubmitMeasureArgsGuards(t *testing.T) {
	t.Run("nul-byte-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := measureSpec("measure-nul-arg", "cfr/scripts/measure_gate_gap.py", []string{"a\x00b"}, nil)
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_args" {
			t.Fatalf("error = %q, want invalid_args", body["error"])
		}
	})

	t.Run("over-length-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := measureSpec("measure-long-arg", "cfr/scripts/measure_gate_gap.py",
			[]string{strings.Repeat("x", maxMeasureArgLen+1)}, nil)
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_args" {
			t.Fatalf("error = %q, want invalid_args", body["error"])
		}
	})

	t.Run("args-forbidden-on-train", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := baseSpec("train-with-args", "train")
		spec["args"] = []string{"--foo"}
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_args" {
			t.Fatalf("error = %q, want invalid_args", body["error"])
		}
	})
}

func TestSubmitMeasureReadsGuards(t *testing.T) {
	t.Run("nonexistent-read-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := measureSpec("measure-bad-read", "cfr/scripts/measure_gate_gap.py", nil, []string{"nonexistent-run"})
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "reads_not_found" {
			t.Fatalf("error = %q, want reads_not_found", body["error"])
		}
	})

	t.Run("escaping-read-rejected", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := measureSpec("measure-escape-read", "cfr/scripts/measure_gate_gap.py", nil, []string{"../../etc"})
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_path" {
			t.Fatalf("error = %q, want invalid_path", body["error"])
		}
	})

	t.Run("reads-forbidden-on-train", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		spec := baseSpec("train-with-reads", "train")
		spec["reads"] = []string{"some-run"}
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("got %d, want 400", resp.StatusCode)
		}
		var body map[string]string
		decodeBody(t, resp, &body)
		if body["error"] != "invalid_reads" {
			t.Fatalf("error = %q, want invalid_reads", body["error"])
		}
	})

	t.Run("accept-contained-and-existing-read", func(t *testing.T) {
		r := newRig(t, rigConfig{algos: HarnessAlgorithms()})
		if err := os.MkdirAll(filepath.Join(r.runsDir, "seed-run"), 0o755); err != nil {
			t.Fatal(err)
		}
		spec := measureSpec("measure-good-read", "cfr/scripts/measure_gate_gap.py", nil, []string{"seed-run"})
		resp := r.do(http.MethodPost, "/harness/jobs", spec)
		if resp.StatusCode != http.StatusCreated {
			t.Fatalf("got %d, want 201", resp.StatusCode)
		}
		resp.Body.Close()
	})
}

// -----------------------------------------------------------------------
// measureLaunchOpts: direct unit tests for the launch template (design D38,
// AC 2/3/5/7), mirroring TestHeadToHeadArgv's direct-call style in
// head_to_head_bench_argv_test.go.
// -----------------------------------------------------------------------

func TestMeasureLaunchOptsArgvAndEnv(t *testing.T) {
	runsDir := t.TempDir()
	worktreeDir := t.TempDir()
	scriptRel := "cfr/scripts/measure_gate_gap.py"
	scriptAbs := filepath.Join(worktreeDir, scriptRel)
	if err := os.MkdirAll(filepath.Dir(scriptAbs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scriptAbs, []byte("# trivial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	seedDir := filepath.Join(runsDir, "seed-run")
	if err := os.MkdirAll(seedDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// algos deliberately omits "measure": AC(7) requires AlgoSubcommand to
	// never be consulted for this kind. A successful launchOpts call against
	// an algos table with no "measure" entry proves it was never called (a
	// call would fail with ErrUnsupportedAlgorithm).
	pm := procmgr.NewProcessManager(runsDir, worktreeDir, "unused-cambia-bin", NewRunResolver(runsDir), map[string][]string{})
	d := NewDispatcher(pm, StubEnvironment{}, runsDir, 1, 16, 0)

	j := &job{spec: JobSpec{
		Kind:   KindMeasure,
		Name:   "measure-argv",
		Script: scriptRel,
		Args:   []string{"--run", "v0.4-x2r-c1", "--shards", "17"},
		Reads:  []string{"seed-run"},
	}}
	prepared := &ingestapi.Prepared{
		WorktreeDir: worktreeDir,
		Env:         []string{"PYTHONPATH=" + filepath.Join(worktreeDir, "shim")},
		VenvPython:  "/fake/venv/python3",
	}

	got, err := d.launchOpts(j, prepared)
	if err != nil {
		t.Fatalf("launchOpts() unexpected error (AlgoSubcommand may have been consulted for kind=measure): %v", err)
	}

	wantArgv := []string{scriptAbs, "--run", "v0.4-x2r-c1", "--shards", "17"}
	if len(got.Argv) != len(wantArgv) {
		t.Fatalf("argv = %v, want %v", got.Argv, wantArgv)
	}
	for i := range got.Argv {
		if got.Argv[i] != wantArgv[i] {
			t.Fatalf("argv = %v, want %v", got.Argv, wantArgv)
		}
	}
	if got.Argv[0] != scriptAbs {
		t.Errorf("argv[0] = %q, want the staged script path %q (no -m src.cli prefix)", got.Argv[0], scriptAbs)
	}

	wantDB := filepath.Join(runsDir, "measure-argv", "run_db.sqlite")
	var gotDB, gotReads string
	for _, e := range got.Env {
		if v, ok := strings.CutPrefix(e, "CAMBIA_RUN_DB="); ok {
			gotDB = v
		}
		if v, ok := strings.CutPrefix(e, "CAMBIA_MEASURE_READ_DIRS="); ok {
			gotReads = v
		}
	}
	if gotDB != wantDB {
		t.Errorf("CAMBIA_RUN_DB = %q, want %q (the job's own run dir)", gotDB, wantDB)
	}
	wantReads := filepath.Join(runsDir, "seed-run")
	if gotReads != wantReads {
		t.Errorf("CAMBIA_MEASURE_READ_DIRS = %q, want %q", gotReads, wantReads)
	}
}

func TestMeasureLaunchOptsScriptMissingAtPinnedCommit(t *testing.T) {
	runsDir := t.TempDir()
	worktreeDir := t.TempDir()
	// cfr/scripts/ exists in the staged worktree, but the named script does not
	// -- the pinned commit simply never carried it.
	if err := os.MkdirAll(filepath.Join(worktreeDir, "cfr", "scripts"), 0o755); err != nil {
		t.Fatal(err)
	}

	pm := procmgr.NewProcessManager(runsDir, worktreeDir, "unused-cambia-bin", NewRunResolver(runsDir), map[string][]string{})
	d := NewDispatcher(pm, StubEnvironment{}, runsDir, 1, 16, 0)

	j := &job{spec: JobSpec{Kind: KindMeasure, Name: "measure-missing", Script: "cfr/scripts/nonexistent.py"}}
	prepared := &ingestapi.Prepared{WorktreeDir: worktreeDir, VenvPython: "/fake/venv/python3"}

	_, err := d.launchOpts(j, prepared)
	if err == nil {
		t.Fatal("launchOpts() = nil error for a script missing at the pinned commit, want a named error")
	}
	if !strings.Contains(err.Error(), "cfr/scripts/nonexistent.py") {
		t.Errorf("err = %q, want it to name the missing script path", err.Error())
	}
}

// -----------------------------------------------------------------------
// End-to-end launch (design D38, AC 6): a trivial measure script writes a
// metrics row through the full submit -> prepare -> launch path, mirroring
// TestDispatcherLaunchesFromStagedEnv's stagedEnv pattern in
// launch_e2e_test.go, extended to stage a pinned script file.
// -----------------------------------------------------------------------

// measureStagedEnv is a fake Environment that stages a worktree containing a
// pinned measure script (scriptFiles: worktree-relative path -> content).
type measureStagedEnv struct {
	runsDir     string
	baseDir     string
	interp      string
	env         []string
	scriptFiles map[string]string
}

func (e *measureStagedEnv) Prepare(ctx context.Context, jobID, commit, kind, configRel, device, warmStart string, overrides map[string]string) (*ingestapi.Prepared, error) {
	worktreeDir := filepath.Join(e.baseDir, "worktrees", jobID)
	if err := os.MkdirAll(filepath.Join(worktreeDir, "cfr"), 0o755); err != nil {
		return nil, err
	}
	for rel, content := range e.scriptFiles {
		full := filepath.Join(worktreeDir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			return nil, err
		}
	}
	runDir := filepath.Join(e.runsDir, jobID)
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return nil, err
	}
	return &ingestapi.Prepared{
		WorktreeDir: worktreeDir,
		RunDir:      runDir,
		VenvPython:  e.interp,
		Env:         e.env,
	}, nil
}

func (e *measureStagedEnv) Cleanup(string, bool) error  { return nil }
func (e *measureStagedEnv) PurgeRef(string) error       { return nil }
func (e *measureStagedEnv) StartupSweep([]string) error { return nil }

// writeMeasureInterp writes a fake "python" interpreter for the measure e2e
// test: it treats its first argv entry as the staged script path and the
// rest as the script's own argv tail, and simulates the script's job --
// writing a metrics row -- by appending a line naming both to the file at
// $CAMBIA_RUN_DB (the run_db.sqlite the real script would open and insert
// into).
func writeMeasureInterp(t *testing.T, dir string) string {
	t.Helper()
	p := filepath.Join(dir, "fake_measure_python.sh")
	body := "#!/bin/sh\n" +
		"script=\"$1\"\nshift\n" +
		"echo \"metric_row script=$script args=$@\" >> \"$CAMBIA_RUN_DB\"\n" +
		"exit 0\n"
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestMeasureLaunchE2EWritesMetricsRow(t *testing.T) {
	base := t.TempDir()
	interp := writeMeasureInterp(t, base)

	se := &measureStagedEnv{
		interp: interp,
		scriptFiles: map[string]string{
			"cfr/scripts/measure_trivial.py": "# trivial measure script\n",
		},
	}
	r := newRig(t, rigConfig{env: se, algos: HarnessAlgorithms()})
	se.runsDir = r.runsDir
	se.baseDir = base

	spec := measureSpec("measure-e2e", "cfr/scripts/measure_trivial.py",
		[]string{"--run", "v0.4-x2r-c1"}, nil)
	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("measure-e2e", procmgr.StatusStopped, 5*time.Second)

	// CAMBIA_RUN_DB (AC 5) points at the job's own run dir; the fake
	// interpreter's write there is the "metrics row" AC 6 asks for.
	runDB := filepath.Join(r.runsDir, "measure-e2e", "run_db.sqlite")
	data, err := os.ReadFile(runDB)
	if err != nil {
		t.Fatalf("read CAMBIA_RUN_DB target (measure script did not write a metrics row): %v", err)
	}
	content := string(data)

	scriptAbs, err := filepath.EvalSymlinks(
		filepath.Join(base, "worktrees", "measure-e2e", "cfr", "scripts", "measure_trivial.py"))
	if err != nil {
		t.Fatal(err)
	}
	// AC 7: argv[0] is the staged script path (no "-m src.cli" prefix); AC 3:
	// args reach argv verbatim.
	want := "metric_row script=" + scriptAbs + " args=--run v0.4-x2r-c1\n"
	if content != want {
		t.Errorf("metrics row = %q, want %q", content, want)
	}
}
