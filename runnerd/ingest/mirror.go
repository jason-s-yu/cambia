package ingest

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// jobRef returns the job-scoped ref name for jobID (design 3.1). A job ref never
// pre-exists across jobs; it is created by the submit-time push and deleted on
// cleanup.
func jobRef(jobID string) string {
	return "refs/harness/" + jobID
}

// git runs a git subcommand against the bare mirror (git -C <mirror> ...) through
// the injected runner and returns trimmed stdout. On failure the error carries
// stderr for diagnosis.
func (m *Manager) git(ctx context.Context, args ...string) (string, error) {
	full := append([]string{"-C", m.mirrorDir}, args...)
	res, err := m.runner.Run(ctx, Command{Name: "git", Args: full})
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(res.Stderr)))
	}
	return strings.TrimSpace(string(res.Stdout)), nil
}

// gitRaw runs a git subcommand and returns raw (untrimmed) stdout bytes, used
// where content bytes matter (blob content hashing).
func (m *Manager) gitRaw(ctx context.Context, args ...string) ([]byte, error) {
	full := append([]string{"-C", m.mirrorDir}, args...)
	res, err := m.runner.Run(ctx, Command{Name: "git", Args: full})
	if err != nil {
		return nil, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(res.Stderr)))
	}
	return res.Stdout, nil
}

// ensureMirror makes m.mirrorDir a bare mirror with gc.auto disabled (design
// 2.7: runnerd controls gc explicitly). If the directory is already a git repo
// it only reasserts gc.auto=0. Otherwise it clones --bare from MirrorURL when
// set, else initializes an empty bare repo (the steady-state push target).
func (m *Manager) ensureMirror(ctx context.Context) error {
	if isGitDir(m.mirrorDir) {
		_, err := m.git(ctx, "config", "gc.auto", "0")
		return err
	}
	if err := os.MkdirAll(m.cfg.BaseDir, 0o755); err != nil {
		return err
	}
	if m.cfg.MirrorURL != "" {
		res, err := m.runner.Run(ctx, Command{Name: "git", Args: []string{"clone", "--bare", m.cfg.MirrorURL, m.mirrorDir}})
		if err != nil {
			return fmt.Errorf("clone --bare: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
		}
	} else {
		res, err := m.runner.Run(ctx, Command{Name: "git", Args: []string{"init", "--bare", m.mirrorDir}})
		if err != nil {
			return fmt.Errorf("init --bare: %w: %s", err, strings.TrimSpace(string(res.Stderr)))
		}
	}
	_, err := m.git(ctx, "config", "gc.auto", "0")
	return err
}

// verifyReceipt is the runner-side receipt check (design 3.1): the job ref must
// resolve to a commit equal to the spec commit, and that object must exist. A
// missing ref, missing object, or sha mismatch is ErrReceiptMismatch. No ref is
// created or updated here; the ref is authored solely by the submit-time push.
func (m *Manager) verifyReceipt(ctx context.Context, jobID, commit string) error {
	ref := jobRef(jobID)
	resolved, err := m.git(ctx, "rev-parse", "--verify", "--quiet", ref+"^{commit}")
	if err != nil || resolved == "" {
		return fmt.Errorf("%w: ref %s does not resolve to a commit", ErrReceiptMismatch, ref)
	}
	if resolved != commit {
		return fmt.Errorf("%w: ref %s -> %s, spec commit %s", ErrReceiptMismatch, ref, resolved, commit)
	}
	// Confirm the object is present in the mirror's object store.
	if _, err := m.git(ctx, "cat-file", "-e", commit+"^{commit}"); err != nil {
		return fmt.Errorf("%w: object %s missing from mirror", ErrReceiptMismatch, commit)
	}
	return nil
}

// BundleFetch is the file-drop fallback transport (design 3.1): it fetches from a
// git bundle into the same job-ref shape without force. Because the fetch omits
// the "+" force prefix, a job ref that already points elsewhere and is not
// fast-forwardable is refused by git rather than overwritten. After the fetch the
// caller runs verifyReceipt as usual.
func (m *Manager) BundleFetch(ctx context.Context, jobID, bundlePath string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if err := m.ensureMirror(ctx); err != nil {
		return err
	}
	refspec := "refs/harness/" + jobID + ":" + jobRef(jobID)
	if _, err := m.git(ctx, "fetch", bundlePath, refspec); err != nil {
		return fmt.Errorf("bundle fetch: %w", err)
	}
	return nil
}

// deleteJobRef removes the job-scoped ref (idempotent: a missing ref is not an
// error). Used on terminal cleanup and startup sweep.
func (m *Manager) deleteJobRef(ctx context.Context, jobID string) error {
	ref := jobRef(jobID)
	// A missing ref makes update-ref -d fail; treat "already gone" as success.
	cur, _ := m.git(ctx, "rev-parse", "--verify", "--quiet", ref)
	if cur == "" {
		return nil
	}
	_, err := m.git(ctx, "update-ref", "-d", ref)
	return err
}

// isGitDir reports whether path looks like a git repository (bare or not).
func isGitDir(path string) bool {
	if _, err := os.Stat(path); err != nil {
		return false
	}
	// A bare repo has HEAD + objects/ at its root; a worktree-backed repo has
	// a .git entry. Either presence is enough for our ensure logic.
	if _, err := os.Stat(path + "/objects"); err == nil {
		if _, err := os.Stat(path + "/HEAD"); err == nil {
			return true
		}
	}
	if _, err := os.Stat(path + "/.git"); err == nil {
		return true
	}
	return false
}
