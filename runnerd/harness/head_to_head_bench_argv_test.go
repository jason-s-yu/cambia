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
// headToHeadArgv: direct table test for the argv construction and the
// containment-refusal defense-in-depth re-resolve (design 5.4), mirroring
// evaluateTargetArgv's re-resolve at launch time against the persisted spec
// (evaluate_target_test.go's TestEvaluateArgvDirTargetUsesLatest is the
// e2e analog below; this is the unit-level table test cambia-295 item 1 asks
// for on top of it).
// -----------------------------------------------------------------------

func TestHeadToHeadArgv(t *testing.T) {
	runsDir := t.TempDir()
	d := NewDispatcher(nil, StubEnvironment{}, runsDir, 1, 16, 0)

	cases := []struct {
		name       string
		spec       JobSpec
		wantErrSub string // non-empty: want an error containing this substring
		wantArgv   []string
	}{
		{
			name: "contained-defaults-games-and-device",
			spec: JobSpec{
				Kind:        KindHeadToHead,
				CheckpointA: "run-a/snapshots/prtcfr_checkpoint.pt",
				CheckpointB: "run-b/snapshots/prtcfr_checkpoint.pt",
			},
			wantArgv: []string{
				"--checkpoint-a", filepath.Join(runsDir, "run-a", "snapshots", "prtcfr_checkpoint.pt"),
				"--checkpoint-b", filepath.Join(runsDir, "run-b", "snapshots", "prtcfr_checkpoint.pt"),
				"--games", "5000",
				"--device", "cpu",
			},
		},
		{
			name: "contained-explicit-games-and-device",
			spec: JobSpec{
				Kind:        KindHeadToHead,
				CheckpointA: "run-a/snap.pt",
				CheckpointB: "run-b/snap.pt",
				Games:       250,
				Device:      "xpu",
			},
			wantArgv: []string{
				"--checkpoint-a", filepath.Join(runsDir, "run-a", "snap.pt"),
				"--checkpoint-b", filepath.Join(runsDir, "run-b", "snap.pt"),
				"--games", "250",
				"--device", "xpu",
			},
		},
		{
			name:       "checkpoint-a-absolute-rejected",
			spec:       JobSpec{Kind: KindHeadToHead, CheckpointA: "/etc/passwd", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
		{
			name:       "checkpoint-b-parent-traversal-rejected",
			spec:       JobSpec{Kind: KindHeadToHead, CheckpointA: "run-a/snap.pt", CheckpointB: "../../etc/passwd"},
			wantErrSub: "checkpoint_b",
		},
		{
			name:       "checkpoint-a-nested-traversal-rejected",
			spec:       JobSpec{Kind: KindHeadToHead, CheckpointA: "ok/a/../../../escape.pt", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
		{
			name:       "checkpoint-a-empty-rejected",
			spec:       JobSpec{Kind: KindHeadToHead, CheckpointA: "", CheckpointB: "run-b/snap.pt"},
			wantErrSub: "checkpoint_a",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := d.headToHeadArgv(tc.spec)
			if tc.wantErrSub != "" {
				if err == nil {
					t.Fatalf("headToHeadArgv() = %v, nil error; want error containing %q", got, tc.wantErrSub)
				}
				if !strings.Contains(err.Error(), tc.wantErrSub) {
					t.Fatalf("err = %q, want it to contain %q", err.Error(), tc.wantErrSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("headToHeadArgv() unexpected error: %v", err)
			}
			if len(got) != len(tc.wantArgv) {
				t.Fatalf("argv = %v, want %v", got, tc.wantArgv)
			}
			for i := range got {
				if got[i] != tc.wantArgv[i] {
					t.Fatalf("argv = %v, want %v", got, tc.wantArgv)
				}
			}
		})
	}
}

// -----------------------------------------------------------------------
// launchOpts argv construction for kind=head-to-head and kind=bench
// (cambia-295 item 1): exercised end-to-end through the staged
// (VenvPython-set) launch path via HarnessAlgorithms(), the same production
// kind->subcommand table the daemon injects, so this also covers the
// kind=bench subcommand fix (benchmark all, not the nonexistent "bench").
// -----------------------------------------------------------------------

func TestHeadToHeadArgvPrecisionE2E(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se, algos: HarnessAlgorithms()})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp

	// Checkpoints must resolve inside the runs dir (design 5.4); write them as
	// real files so the expected paths can be safely symlink-resolved below.
	ckptA := filepath.Join(r.runsDir, "run-a", "snapshots", "prtcfr_checkpoint.pt")
	ckptB := filepath.Join(r.runsDir, "run-b", "snapshots", "prtcfr_checkpoint.pt")
	for _, p := range []string{ckptA, ckptB} {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	spec := baseSpec("h2h-argv", "head-to-head")
	spec["checkpoint_a"] = "run-a/snapshots/prtcfr_checkpoint.pt"
	spec["checkpoint_b"] = "run-b/snapshots/prtcfr_checkpoint.pt"
	spec["games"] = 128

	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("h2h-argv", procmgr.StatusStopped, 5*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture (interpreter did not run): %v", err)
	}
	argv := captureField(t, string(data), "ARGV")

	rendered := filepath.Join(r.runsDir, "h2h-argv", "config.yaml")
	wantA, err := filepath.EvalSymlinks(ckptA)
	if err != nil {
		t.Fatal(err)
	}
	wantB, err := filepath.EvalSymlinks(ckptB)
	if err != nil {
		t.Fatal(err)
	}
	wantArgv := "-m src.cli head-to-head --config " + rendered +
		" --checkpoint-a " + wantA + " --checkpoint-b " + wantB +
		" --games 128 --device cpu"
	if argv != wantArgv {
		t.Errorf("argv = %q, want %q", argv, wantArgv)
	}

	// head-to-head journals into its own run dir: no evaluated-run override
	// (unlike evaluate, which redirects CAMBIA_RUN_DB to the target's run).
	wantDB := filepath.Join(r.runsDir, "h2h-argv", "run_db.sqlite")
	if got := captureField(t, string(data), "CAMBIA_RUN_DB"); got != wantDB {
		t.Errorf("CAMBIA_RUN_DB = %q, want %q", got, wantDB)
	}
}

func TestBenchArgvPrecisionE2E(t *testing.T) {
	base := t.TempDir()
	capture := filepath.Join(base, "capture.txt")
	interp := writeCaptureInterp(t, base, capture)

	se := &stagedEnv{}
	r := newRig(t, rigConfig{env: se, algos: HarnessAlgorithms()})
	se.runsDir = r.runsDir
	se.baseDir = base
	se.interp = interp

	spec := baseSpec("bench-argv", "bench")

	resp := r.do(http.MethodPost, "/harness/jobs", spec)
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("submit: got %d, want 201", resp.StatusCode)
	}
	resp.Body.Close()
	r.waitForState("bench-argv", procmgr.StatusStopped, 5*time.Second)

	data, err := os.ReadFile(capture)
	if err != nil {
		t.Fatalf("read capture (interpreter did not run): %v", err)
	}
	argv := captureField(t, string(data), "ARGV")

	rendered := filepath.Join(r.runsDir, "bench-argv", "config.yaml")
	wantArgv := "-m src.cli benchmark all --config " + rendered +
		" --output-dir " + filepath.Join(r.runsDir, "bench-argv") + " --device cpu"
	if argv != wantArgv {
		t.Errorf("argv = %q, want %q", argv, wantArgv)
	}

	// bench journals into its own run dir, same as train/head-to-head.
	wantDB := filepath.Join(r.runsDir, "bench-argv", "run_db.sqlite")
	if got := captureField(t, string(data), "CAMBIA_RUN_DB"); got != wantDB {
		t.Errorf("CAMBIA_RUN_DB = %q, want %q", got, wantDB)
	}
}
