package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// debugTTLFile is the basename of the failed-without-sync retention marker
// written into a worktree; it holds an RFC3339 expiry timestamp.
const debugTTLFile = ".harness_debug_ttl"

// addWorktree adds a detached worktree at commit from the bare mirror (shared
// object store, checkout-only cost; design 3.2). An existing directory is kept
// only when its HEAD already matches the pinned commit (idempotent re-prepare
// after a crash mid-stage); anything else is torn down and re-added, so a job
// name reused after purge can never execute a tree older than the commit its
// jobspec and run_db record (commit-pinning invariant, design 3.1; the X2R
// resubmission ran stale code this way, cambia-518).
func (m *Manager) addWorktree(ctx context.Context, worktreeDir, commit string) error {
	if _, err := os.Stat(worktreeDir); err == nil {
		res, err := m.runner.Run(ctx, Command{
			Name: "git", Args: []string{"-C", worktreeDir, "rev-parse", "HEAD"},
		})
		if err == nil && strings.TrimSpace(string(res.Stdout)) == commit {
			return nil
		}
		if rmErr := m.removeWorktree(ctx, worktreeDir); rmErr != nil {
			return rmErr
		}
	}
	if err := os.MkdirAll(m.worktreesDir, 0o755); err != nil {
		return err
	}
	_, err := m.git(ctx, "worktree", "add", "--detach", worktreeDir, commit)
	return err
}

// removeWorktree removes a worktree via git so the mirror's admin metadata is
// updated, falling back to a filesystem removal if the git remove fails (a
// half-created worktree may not be known to git). Idempotent: a missing worktree
// is success.
func (m *Manager) removeWorktree(ctx context.Context, worktreeDir string) error {
	if _, err := os.Stat(worktreeDir); os.IsNotExist(err) {
		return nil
	}
	if _, err := m.git(ctx, "worktree", "remove", "--force", worktreeDir); err == nil {
		return nil
	}
	// git refused (unknown/corrupt worktree); remove the tree directly and let
	// the sweep's `worktree prune` reconcile the admin entry.
	return os.RemoveAll(worktreeDir)
}

// markDebugTTL stamps a worktree with a failure-retention expiry (now + DebugTTL)
// so StartupSweep keeps it for post-mortem until the TTL lapses (design 3.2).
func (m *Manager) markDebugTTL(worktreeDir string) error {
	if _, err := os.Stat(worktreeDir); err != nil {
		return err
	}
	expiry := m.now().Add(m.debugTTL).UTC().Format(time.RFC3339)
	return os.WriteFile(filepath.Join(worktreeDir, debugTTLFile), []byte(expiry+"\n"), 0o644)
}

// debugTTLActive reports whether a worktree carries an unexpired failure marker.
// A missing or malformed marker is treated as inactive (prunable).
func (m *Manager) debugTTLActive(worktreeDir string) bool {
	data, err := os.ReadFile(filepath.Join(worktreeDir, debugTTLFile))
	if err != nil {
		return false
	}
	expiry, err := time.Parse(time.RFC3339, strings.TrimSpace(string(data)))
	if err != nil {
		return false
	}
	return m.now().Before(expiry)
}
