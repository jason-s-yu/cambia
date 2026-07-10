package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRejectOwnedOverrides(t *testing.T) {
	owned := []string{
		"prt_cfr.device",
		"deep_cfr.device",
		"device",
		"cfr_training.num_workers",
		"analysis.exploitability_num_workers",
		"prt_cfr.gen_chunk_games",
		"persistence.agent_data_save_path",
		"prt_cfr.reservoir_dir",
		"prt_cfr.snapshot_dir",
		"prt_cfr.warm_start_path",
	}
	for _, k := range owned {
		if err := rejectOwnedOverrides(map[string]string{k: "x"}); !errors.Is(err, ErrOwnedOverride) {
			t.Fatalf("override %q not rejected (err=%v)", k, err)
		}
	}
	// A non-owned key passes.
	if err := rejectOwnedOverrides(map[string]string{"prt_cfr.iterations": "500"}); err != nil {
		t.Fatalf("legitimate override rejected: %v", err)
	}
}

// TestRejectPathishOverrides covers the M5 broadening: any key whose leaf is
// path-ish (a path suffix or an exact path leaf) is rejected even when it is not
// one of the enumerated owned rails, while a normal non-path key still passes.
func TestRejectPathishOverrides(t *testing.T) {
	rejected := []string{
		"persistence.some_new_dir",   // _dir suffix, not an enumerated rail
		"logging.extra_dirs",         // _dirs suffix
		"trainer.metrics_out",        // _out suffix
		"trainer.summary_output",     // _output suffix
		"io.results_file",            // _file suffix
		"analysis.report_path",       // _path suffix
		"output",                     // bare exact leaf
		"nested.path",                // exact leaf "path"
		"nested.dir",                 // exact leaf "dir"
		"persistence.checkpoint_dir", // exact + suffix
	}
	for _, k := range rejected {
		if err := rejectOwnedOverrides(map[string]string{k: "/tmp/evil"}); !errors.Is(err, ErrOwnedOverride) {
			t.Fatalf("path-ish override %q not rejected (err=%v)", k, err)
		}
	}
	accepted := []string{
		"prt_cfr.iterations",
		"prt_cfr.k_games",
		"cfr_training.batch_size",
		"analysis.report_period", // ends in "period", not path-ish
	}
	for _, k := range accepted {
		if err := rejectOwnedOverrides(map[string]string{k: "8"}); err != nil {
			t.Fatalf("legitimate override %q rejected: %v", k, err)
		}
	}
}

func TestRenderArgOrderRailsLast(t *testing.T) {
	fc := newFakeControl()
	m, fr := fakeManager(t, fc)
	src, sha := sourceRepo(t, "lock-r")
	pushJobRef(t, m, src, sha, "job-r")
	wd := m.worktreePath("job-r")
	if err := m.addWorktree(context.Background(), wd, sha); err != nil {
		t.Fatal(err)
	}
	runDir := m.runDir("job-r")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}

	overrides := map[string]string{
		"prt_cfr.iterations": "500",
		"prt_cfr.k_games":    "8",
	}
	out, err := m.renderConfig(context.Background(), wd, runDir, filepath.Join(wd, "venv", "bin", "python"), "train", "cfr/config/prtcfr.yaml", "cpu", "", overrides)
	if err != nil {
		t.Fatalf("renderConfig: %v", err)
	}
	if out != filepath.Join(runDir, "config.yaml") {
		t.Fatalf("rendered path = %q", out)
	}

	// Inspect the render invocation's --set order.
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
	sets := extractSets(renderArgs)
	// User overrides (sorted) come first, then rails.
	wantUserPrefix := []string{"prt_cfr.iterations=500", "prt_cfr.k_games=8"}
	if len(sets) < len(wantUserPrefix) {
		t.Fatalf("too few --set entries: %v", sets)
	}
	for i, w := range wantUserPrefix {
		if sets[i] != w {
			t.Fatalf("--set[%d] = %q, want %q (full: %v)", i, sets[i], w, sets)
		}
	}
	// Rails must all appear AFTER the last user override.
	railStart := len(wantUserPrefix)
	rails := sets[railStart:]
	assertContains(t, rails, "prt_cfr.device=cpu")
	assertContains(t, rails, "cfr_training.num_workers=18")
	assertContains(t, rails, "analysis.exploitability_num_workers=18")
	assertContains(t, rails, "persistence.agent_data_save_path="+filepath.Join(runDir, "snapshots", "prtcfr_checkpoint.pt"))
	assertContains(t, rails, "prt_cfr.reservoir_dir="+filepath.Join(runDir, "reservoir"))
	assertContains(t, rails, "prt_cfr.snapshot_dir="+filepath.Join(runDir, "snapshots"))

	// No user --set may appear after a rail (rails-last invariant).
	for i, s := range sets {
		if i >= railStart {
			if strings.HasPrefix(s, "prt_cfr.iterations") || strings.HasPrefix(s, "prt_cfr.k_games") {
				t.Fatalf("user override %q appears in the rails region: %v", s, sets)
			}
		}
	}

	// The validate gate must have run on the output.
	sawValidate := false
	for _, c := range fr.calls {
		if strings.HasSuffix(c.Name, "/bin/python") && contains(c.Args, "validate") {
			sawValidate = true
		}
	}
	if !sawValidate {
		t.Fatal("config validate gate did not run")
	}
}

func TestGuardRelPath(t *testing.T) {
	root := t.TempDir()
	// Valid, contained.
	got, err := guardRelPath(root, "cfr/config/x.yaml")
	if err != nil {
		t.Fatalf("valid path rejected: %v", err)
	}
	if got != filepath.Join(root, "cfr", "config", "x.yaml") {
		t.Fatalf("resolved to %q", got)
	}
	// Rejections.
	for _, bad := range []string{"/etc/passwd", "../escape", "cfr/../../etc/passwd", "a/../../b", ""} {
		if _, err := guardRelPath(root, bad); !errors.Is(err, ErrPathEscape) {
			t.Fatalf("guardRelPath(%q) = %v, want ErrPathEscape", bad, err)
		}
	}
}

// TestGuardRelPathSymlinkEscape covers the M4 symlink cases for the ingest
// guard: a symlink inside root pointing outside root escapes (reject), one
// pointing within root is contained (accept), a non-existent tail under root is
// accepted, and a root that is itself a symlink resolves consistently (accept).
func TestGuardRelPathSymlinkEscape(t *testing.T) {
	tmp := t.TempDir()
	root := filepath.Join(tmp, "root")
	outside := filepath.Join(tmp, "outside")
	if err := os.MkdirAll(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "escape")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(root, "sub"), filepath.Join(root, "inward")); err != nil {
		t.Fatal(err)
	}

	// Escape via a symlink to an existing outside dir: reject.
	for _, rel := range []string{"escape", "escape/config.yaml"} {
		if _, err := guardRelPath(root, rel); !errors.Is(err, ErrPathEscape) {
			t.Fatalf("guardRelPath(%q) = %v, want ErrPathEscape", rel, err)
		}
	}
	// Dangling symlink to a non-existent outside target: also reject.
	if err := os.Symlink(filepath.Join(tmp, "gone", "x"), filepath.Join(root, "dangling")); err != nil {
		t.Fatal(err)
	}
	if _, err := guardRelPath(root, "dangling"); !errors.Is(err, ErrPathEscape) {
		t.Fatalf("guardRelPath(dangling) = %v, want ErrPathEscape", err)
	}
	// Symlink within root: accept.
	if _, err := guardRelPath(root, "inward/config.yaml"); err != nil {
		t.Fatalf("guardRelPath(inward/config.yaml) = %v, want nil", err)
	}
	// Non-existent tail under a real root: accept.
	if _, err := guardRelPath(root, "cfr/config/new.yaml"); err != nil {
		t.Fatalf("guardRelPath(non-existent) = %v, want nil", err)
	}

	// Root that is itself a symlink: accept, resolved consistently.
	rootLink := filepath.Join(tmp, "rootlink")
	if err := os.Symlink(root, rootLink); err != nil {
		t.Fatal(err)
	}
	if _, err := guardRelPath(rootLink, "sub/config.yaml"); err != nil {
		t.Fatalf("guardRelPath through symlinked root = %v, want nil", err)
	}
}

func TestRenderConfigPathGuardRejectsEscape(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	wd := t.TempDir()
	runDir := t.TempDir()
	_, err := m.renderConfig(context.Background(), wd, runDir, "python", "train", "../../etc/passwd", "cpu", "", nil)
	if !errors.Is(err, ErrPathEscape) {
		t.Fatalf("want ErrPathEscape, got %v", err)
	}
}

func TestRailOverridesWorkerCapDisabledWhenZero(t *testing.T) {
	m := New(Config{BaseDir: t.TempDir(), RunsDir: t.TempDir(), CoresCap: 0})
	rails := m.railOverrides("train", "/runs/x", "cpu", "")
	for _, r := range rails {
		if strings.HasPrefix(r, "cfr_training.num_workers") {
			t.Fatalf("worker rail emitted with CoresCap=0: %v", rails)
		}
	}
}

// TestRailOverridesDeviceValue covers cambia-329: the device rail carries the
// job's own device, for both train (prt_cfr.device) and non-train
// (deep_cfr.device) kinds.
func TestRailOverridesDeviceValue(t *testing.T) {
	m := New(Config{BaseDir: t.TempDir(), RunsDir: t.TempDir(), CoresCap: 0})

	trainRails := m.railOverrides("train", "/runs/x", "cuda", "")
	assertContains(t, trainRails, "prt_cfr.device=cuda")

	evalRails := m.railOverrides("evaluate", "/runs/x", "xpu", "")
	assertContains(t, evalRails, "deep_cfr.device=xpu")
}

// TestRailOverridesWarmStartPath covers cambia-334: the warm_start rail is
// emitted only for a train job carrying an already-resolved warm_start path,
// and omitted otherwise (absent warm_start, or a non-train kind -- defense in
// depth, since warm_start is kind-scoped to train before it ever reaches
// render).
func TestRailOverridesWarmStartPath(t *testing.T) {
	m := New(Config{BaseDir: t.TempDir(), RunsDir: t.TempDir(), CoresCap: 0})

	withWarmStart := m.railOverrides("train", "/runs/x", "cpu", "/runs/prior-run/snapshots/prtcfr_snapshot_iter_530.pt")
	assertContains(t, withWarmStart, "prt_cfr.warm_start_path=/runs/prior-run/snapshots/prtcfr_snapshot_iter_530.pt")

	noWarmStart := m.railOverrides("train", "/runs/x", "cpu", "")
	for _, r := range noWarmStart {
		if strings.HasPrefix(r, "prt_cfr.warm_start_path") {
			t.Fatalf("warm_start rail emitted with no warm_start: %v", noWarmStart)
		}
	}

	evalRails := m.railOverrides("evaluate", "/runs/x", "cpu", "/runs/prior-run/snapshots/x.pt")
	for _, r := range evalRails {
		if strings.HasPrefix(r, "prt_cfr.warm_start_path") {
			t.Fatalf("warm_start rail emitted for non-train kind: %v", evalRails)
		}
	}
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func assertContains(t *testing.T, ss []string, want string) {
	t.Helper()
	if !contains(ss, want) {
		t.Fatalf("missing rail %q in %v", want, ss)
	}
}

// extractSets returns the value tokens following each --set flag, in order.
func extractSets(args []string) []string {
	var out []string
	for i := 0; i < len(args)-1; i++ {
		if args[i] == "--set" {
			out = append(out, args[i+1])
		}
	}
	return out
}
