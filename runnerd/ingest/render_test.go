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
	out, err := m.renderConfig(context.Background(), wd, runDir, filepath.Join(wd, "venv", "bin", "python"), "train", "cfr/config/prtcfr.yaml", overrides)
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

func TestRenderConfigPathGuardRejectsEscape(t *testing.T) {
	fc := newFakeControl()
	m, _ := fakeManager(t, fc)
	wd := t.TempDir()
	runDir := t.TempDir()
	_, err := m.renderConfig(context.Background(), wd, runDir, "python", "train", "../../etc/passwd", nil)
	if !errors.Is(err, ErrPathEscape) {
		t.Fatalf("want ErrPathEscape, got %v", err)
	}
}

func TestRailOverridesWorkerCapDisabledWhenZero(t *testing.T) {
	m := New(Config{BaseDir: t.TempDir(), RunsDir: t.TempDir(), CoresCap: 0})
	rails := m.railOverrides("train", "/runs/x")
	for _, r := range rails {
		if strings.HasPrefix(r, "cfr_training.num_workers") {
			t.Fatalf("worker rail emitted with CoresCap=0: %v", rails)
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
