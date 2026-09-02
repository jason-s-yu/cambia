package harness

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// The direct table test for the head-to-head argv tail moved to
// runnerd/nodeagent with the launch template it exercises (D1); it is
// TestHeadToHeadArgv there. What stays here is the end-to-end half below,
// which proves the same argv reaches a real interpreter through the
// dispatcher's staged launch path.

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
