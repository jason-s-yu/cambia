package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPrepareEndToEnd exercises the full staging pipeline with real git and a
// faked toolchain, asserting the returned Prepared and the on-disk artifacts.
func TestPrepareEndToEnd(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	src, sha := sourceRepo(t, "prod-lock")
	pushJobRef(t, m, src, sha, "v0.4-prtcfr-r1")

	prep, err := m.Prepare(
		context.Background(),
		"v0.4-prtcfr-r1",
		sha,
		"train",
		"cfr/config/prtcfr.yaml",
		"cpu",
		map[string]string{"prt_cfr.iterations": "10"},
	)
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	// Worktree checked out with content.
	if _, err := os.Stat(filepath.Join(prep.WorktreeDir, "cfr", "uv.lock")); err != nil {
		t.Fatalf("worktree checkout incomplete: %v", err)
	}
	// Run dir outside the worktree.
	if strings.HasPrefix(prep.RunDir, prep.WorktreeDir) {
		t.Fatalf("run dir %q is inside worktree %q", prep.RunDir, prep.WorktreeDir)
	}
	// Venv python present.
	if _, err := os.Stat(prep.VenvPython); err != nil {
		t.Fatalf("venv python missing: %v", err)
	}
	// libcambia artifact present, keyed by engine tree sha.
	if _, err := os.Stat(prep.LibcambiaPath); err != nil {
		t.Fatalf("libcambia missing: %v", err)
	}
	engSha := runGit(t, src, "rev-parse", sha+":engine")
	if filepath.Base(prep.LibcambiaPath) != engSha+".so" {
		t.Fatalf("libcambia %q not keyed by engine tree %q", prep.LibcambiaPath, engSha)
	}
	// Rendered config written.
	if prep.RenderedConfig != filepath.Join(prep.RunDir, "config.yaml") {
		t.Fatalf("rendered config = %q", prep.RenderedConfig)
	}
	if _, err := os.Stat(prep.RenderedConfig); err != nil {
		t.Fatalf("rendered config not written: %v", err)
	}
	// env.json written with the pinned commit.
	rec, err := readEnvJSON(filepath.Join(prep.RunDir, envJSONFile))
	if err != nil {
		t.Fatalf("env.json: %v", err)
	}
	if rec.Commit != sha {
		t.Fatalf("env.json commit = %q, want %q", rec.Commit, sha)
	}
	if rec.Device != "cpu" {
		t.Fatalf("env.json device = %q, want cpu", rec.Device)
	}
	if rec.EngineTreeSha != engSha {
		t.Fatalf("env.json engine_tree_sha = %q, want %q", rec.EngineTreeSha, engSha)
	}

	// Launch env carries the containment pin.
	kv := envMap(prep.Env)
	if kv["LIBCAMBIA_PATH"] != prep.LibcambiaPath {
		t.Fatalf("Env LIBCAMBIA_PATH = %q, want %q", kv["LIBCAMBIA_PATH"], prep.LibcambiaPath)
	}
	if kv["CAMBIA_EXPECTED_SRC_ROOT"] != filepath.Join(prep.WorktreeDir, "cfr") {
		t.Fatalf("Env CAMBIA_EXPECTED_SRC_ROOT = %q", kv["CAMBIA_EXPECTED_SRC_ROOT"])
	}

	// libcambia built exactly once through the pinned toolchain.
	goBuilds := countGoBuilds(fr)
	if goBuilds != 1 {
		t.Fatalf("expected 1 go build, got %d", goBuilds)
	}
	// Assert GOTOOLCHAIN pin on the build invocation.
	sawPin := false
	for _, c := range fr.callsFor("go") {
		if len(c.Args) > 0 && c.Args[0] == "build" {
			for _, e := range c.Env {
				if e == "GOTOOLCHAIN="+goToolchainPin {
					sawPin = true
				}
			}
		}
	}
	if !sawPin {
		t.Fatal("go build did not pin GOTOOLCHAIN")
	}

	// No uv invocation may carry --active (absolute rule).
	for _, c := range fr.callsFor("uv") {
		for _, a := range c.Args {
			if a == "--active" {
				t.Fatalf("uv invoked with --active: %v", c.Args)
			}
		}
	}
}

// TestPrepareThreadsDeviceToProvenanceRailsAndVenv covers cambia-329: a
// non-cpu device flows through Prepare into env.json provenance, the uv sync
// extra, the venv cache key suffix, and the rendered config's device rail.
func TestPrepareThreadsDeviceToProvenanceRailsAndVenv(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	src, sha := sourceRepo(t, "lock-device")
	pushJobRef(t, m, src, sha, "job-device")

	prep, err := m.Prepare(context.Background(), "job-device", sha, "train", "cfr/config/prtcfr.yaml", "cuda", nil)
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}

	rec, err := readEnvJSON(filepath.Join(prep.RunDir, envJSONFile))
	if err != nil {
		t.Fatalf("env.json: %v", err)
	}
	if rec.Device != "cuda" {
		t.Fatalf("env.json device = %q, want cuda", rec.Device)
	}
	if !strings.HasSuffix(rec.VenvCacheKey, "-gpu") {
		t.Fatalf("venv cache key %q missing -gpu extra suffix", rec.VenvCacheKey)
	}

	sawGPUExtra := false
	for _, c := range fr.callsFor("uv") {
		if len(c.Args) > 0 && c.Args[0] == "sync" {
			for i, a := range c.Args {
				if a == "--extra" && i+1 < len(c.Args) && c.Args[i+1] == "gpu" {
					sawGPUExtra = true
				}
			}
		}
	}
	if !sawGPUExtra {
		t.Fatal("uv sync did not use --extra gpu for a cuda device job")
	}

	var renderArgs []string
	for _, c := range fr.calls {
		if strings.HasSuffix(c.Name, "/bin/python") && contains(c.Args, "render") {
			renderArgs = c.Args
			break
		}
	}
	if renderArgs == nil {
		t.Fatal("no render invocation recorded")
	}
	assertContains(t, extractSets(renderArgs), "prt_cfr.device=cuda")
}

func TestPrepareRejectsOwnedOverride(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	src, sha := sourceRepo(t, "lock")
	pushJobRef(t, m, src, sha, "job-o")

	_, err := m.Prepare(context.Background(), "job-o", sha, "train", "cfr/config/x.yaml", "cpu",
		map[string]string{"cfr_training.num_workers": "20"})
	if !errors.Is(err, ErrOwnedOverride) {
		t.Fatalf("want ErrOwnedOverride, got %v", err)
	}
}

func TestPrepareRejectsReceiptMismatch(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	src, sha := sourceRepo(t, "lock")
	pushJobRef(t, m, src, sha, "job-m")

	other := strings.Repeat("b", 40)
	_, err := m.Prepare(context.Background(), "job-m", other, "train", "cfr/config/x.yaml", "cpu", nil)
	if !errors.Is(err, ErrReceiptMismatch) {
		t.Fatalf("want ErrReceiptMismatch, got %v", err)
	}
}

func TestPrepareRejectsBadJobID(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	_, err := m.Prepare(context.Background(), "../evil", strings.Repeat("a", 40), "train", "c.yaml", "cpu", nil)
	if err == nil {
		t.Fatal("expected invalid job id rejection")
	}
}
