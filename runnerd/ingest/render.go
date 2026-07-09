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
// process-pool counts, and the save/run paths. A submitter override whose final
// segment is in this set is rejected before render; the harness sets these itself
// as rails-last overrides.
var ownedLeafKeys = map[string]bool{
	"device":                     true,
	"num_workers":                true,
	"exploitability_num_workers": true,
	"gen_chunk_games":            true,
	"agent_data_save_path":       true,
	"reservoir_dir":              true,
	"snapshot_dir":               true,
}

// leafOf returns the last dotted segment of a config key.
func leafOf(key string) string {
	if i := strings.LastIndex(key, "."); i >= 0 {
		return key[i+1:]
	}
	return key
}

// isOwnedKey reports whether key targets a harness-owned config field.
func isOwnedKey(key string) bool {
	return ownedLeafKeys[leafOf(strings.TrimSpace(key))]
}

// rejectOwnedOverrides fails when any submitter override targets a harness-owned
// key (design 5.5): rails must not be silently clobbered, so the request is
// refused rather than overridden late.
func rejectOwnedOverrides(overrides map[string]string) error {
	var bad []string
	for k := range overrides {
		if isOwnedKey(k) {
			bad = append(bad, k)
		}
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("%w: %s", ErrOwnedOverride, strings.Join(bad, ", "))
	}
	return nil
}

// railOverrides returns the harness rail overrides in deterministic order for the
// given kind. Rails are appended AFTER user overrides in the render command so
// they win last-write (design 5.5). CoresCap <= 0 omits the worker rails. Train
// jobs get the full PRT-CFR rail set (device, worker counts, save/run paths);
// other kinds get the device and worker rails only.
func (m *Manager) railOverrides(kind, runDir string) []string {
	var rails []string
	if kind == "train" {
		rails = append(rails, "prt_cfr.device=cpu")
	} else {
		rails = append(rails, "deep_cfr.device=cpu")
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
	}
	return rails
}

// renderConfig materializes runDir/config.yaml from the submitter's base config
// under the worktree, then validates it as a hard gate (design 3.4). The render
// argument order is: user overrides (sorted for determinism) first, harness rails
// last. It returns the absolute rendered config path.
func (m *Manager) renderConfig(ctx context.Context, worktreeDir, runDir, venvPython, kind, configRel string, overrides map[string]string) (string, error) {
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
	for _, r := range m.railOverrides(kind, runDir) {
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
