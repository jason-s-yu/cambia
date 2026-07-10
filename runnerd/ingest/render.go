package ingest

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ownedLeafKeys is the set of dotted-key leaf segments the harness owns and a
// submitter may not override (design 5.5): the device, the job-internal worker /
// process-pool counts, the save/run paths, and warm_start_path (design
// cambia-334; already caught by the pathishSuffixes "_path" check below, listed
// here too for the same explicit-documentation reason as pathishExact). A
// submitter override whose final segment is in this set is rejected before
// render; the harness sets these itself as rails-last overrides.
var ownedLeafKeys = map[string]bool{
	"device":                     true,
	"num_workers":                true,
	"exploitability_num_workers": true,
	"gen_chunk_games":            true,
	"agent_data_save_path":       true,
	"reservoir_dir":              true,
	"snapshot_dir":               true,
	"warm_start_path":            true,
}

// pathishSuffixes are leaf suffixes that mark a config key as naming a
// filesystem location. The explicit ownedLeafKeys set names only 7 leaves, but
// any key whose leaf ends in one of these suffixes also redirects where a job
// writes (or reads), so a submitter override targeting one could escape its run
// dir. Rejecting the whole pattern, not just the enumerated leaves, closes that
// gap (design 5.5, broadened).
var pathishSuffixes = []string{"_path", "_dir", "_dirs", "_file", "_out", "_output"}

// pathishExact are bare leaf keys (no path-ish suffix) that still name a
// filesystem location. save_path/checkpoint_dir/log_dir already match a suffix
// above; they are listed for documentation and defense in depth.
var pathishExact = map[string]bool{
	"path":           true,
	"dir":            true,
	"output":         true,
	"save_path":      true,
	"checkpoint_dir": true,
	"log_dir":        true,
}

// leafOf returns the last dotted segment of a config key.
func leafOf(key string) string {
	if i := strings.LastIndex(key, "."); i >= 0 {
		return key[i+1:]
	}
	return key
}

// isOwnedKey reports whether key targets an explicitly harness-owned config
// field (the enumerated rails).
func isOwnedKey(key string) bool {
	return ownedLeafKeys[leafOf(strings.TrimSpace(key))]
}

// isPathishKey reports whether key's leaf names a filesystem location by an
// exact match or a path-ish suffix. Such a key is refused even when it is not an
// enumerated rail, so a submitter cannot redirect writes outside the run dir
// through a config key the explicit list happens not to name.
func isPathishKey(key string) bool {
	leaf := leafOf(strings.TrimSpace(key))
	if pathishExact[leaf] {
		return true
	}
	for _, sfx := range pathishSuffixes {
		if strings.HasSuffix(leaf, sfx) {
			return true
		}
	}
	return false
}

// rejectOwnedOverrides fails when any submitter override targets a harness-owned
// key or any path-ish key (design 5.5): rails and output/path locations must not
// be silently clobbered, so the request is refused rather than overridden late.
// Non-rejected keys are still governed by the rails-last render ordering
// (railOverrides is appended after user overrides), so a rail always wins its
// own leaf even when a similarly named non-path key slips through.
func rejectOwnedOverrides(overrides map[string]string) error {
	var bad []string
	for k := range overrides {
		if isOwnedKey(k) || isPathishKey(k) {
			bad = append(bad, k)
		}
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("%w: %s", ErrOwnedOverride, strings.Join(bad, ", "))
	}
	return nil
}

// railOverrides returns the harness rail overrides in deterministic order for
// the given kind and device. Rails are appended AFTER user overrides in the
// render command so they win last-write (design 5.5). The device rail value is
// the job's own device (cambia-329): the rail stays harness-owned (a submitter
// still cannot set device via overrides -- rejectOwnedOverrides refuses it --
// only via the spec's device field). CoresCap <= 0 omits the worker rails.
// Train jobs get the full PRT-CFR rail set (device, worker counts, save/run
// paths); other kinds get the device and worker rails only. warmStartPath is
// the already-resolved absolute snapshot path for a train job's optional
// warm_start (design cambia-334); empty omits the rail entirely, and it is
// only ever considered for kind=train (warm_start is kind-scoped at submit).
func (m *Manager) railOverrides(kind, runDir, device, warmStartPath string) []string {
	var rails []string
	if kind == "train" {
		rails = append(rails, "prt_cfr.device="+device)
	} else {
		rails = append(rails, "deep_cfr.device="+device)
	}
	if m.cfg.CoresCap > 0 {
		workers := strconv.Itoa(m.cfg.CoresCap)
		rails = append(rails,
			"cfr_training.num_workers="+workers,
			"analysis.exploitability_num_workers="+workers,
		)
	}
	if kind == "train" {
		rails = append(rails,
			"persistence.agent_data_save_path="+filepath.Join(runDir, "snapshots", "prtcfr_checkpoint.pt"),
			"prt_cfr.reservoir_dir="+filepath.Join(runDir, "reservoir"),
			"prt_cfr.snapshot_dir="+filepath.Join(runDir, "snapshots"),
		)
		if warmStartPath != "" {
			rails = append(rails, "prt_cfr.warm_start_path="+warmStartPath)
		}
	}
	return rails
}

// renderConfig materializes runDir/config.yaml from the submitter's base config
// under the worktree, then validates it as a hard gate (design 3.4). The render
// argument order is: user overrides (sorted for determinism) first, harness rails
// last. warmStartPath is the already-resolved absolute snapshot path threaded
// into railOverrides (empty when the job has no warm_start). It returns the
// absolute rendered config path.
func (m *Manager) renderConfig(ctx context.Context, worktreeDir, runDir, venvPython, kind, configRel, device, warmStartPath string, overrides map[string]string) (string, error) {
	// Path guard: config is repo-relative at the pinned commit, inside the
	// worktree (design 5.4).
	configAbs, err := guardRelPath(worktreeDir, configRel)
	if err != nil {
		return "", err
	}
	outPath := filepath.Join(runDir, "config.yaml")
	cfrDir := filepath.Join(worktreeDir, "cfr")

	args := []string{"-m", "src.cli", "config", "render", configAbs}
	// User overrides first, deterministically ordered.
	userKeys := make([]string, 0, len(overrides))
	for k := range overrides {
		userKeys = append(userKeys, k)
	}
	sort.Strings(userKeys)
	for _, k := range userKeys {
		args = append(args, "--set", k+"="+overrides[k])
	}
	// Harness rails appended AFTER user overrides so they win last-write.
	for _, r := range m.railOverrides(kind, runDir, device, warmStartPath) {
		args = append(args, "--set", r)
	}
	args = append(args, "-o", outPath)

	if res, err := m.runner.Run(ctx, Command{
		Name: venvPython,
		Args: args,
		Dir:  cfrDir,
		Env:  m.renderEnv(worktreeDir),
	}); err != nil {
		return "", fmt.Errorf("render: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}

	// Hard validation gate.
	if res, err := m.runner.Run(ctx, Command{
		Name: venvPython,
		Args: []string{"-m", "src.cli", "config", "validate", outPath},
		Dir:  cfrDir,
		Env:  m.renderEnv(worktreeDir),
	}); err != nil {
		return "", fmt.Errorf("validate: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
	}
	return outPath, nil
}

// renderEnv is the minimal environment for the render/validate subprocesses:
// PYTHONPATH pinned to the worktree cfr dir (so `-m src.cli` resolves the pinned
// src) plus user-site suppression. It mirrors the launch env's src pinning.
func (m *Manager) renderEnv(worktreeDir string) []string {
	return []string{
		"PYTHONPATH=" + filepath.Join(worktreeDir, "cfr"),
		"PYTHONNOUSERSITE=1",
	}
}
