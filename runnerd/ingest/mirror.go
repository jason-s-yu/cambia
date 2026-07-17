package ingest

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// jobRef returns the job-scoped ref name for jobID (design 3.1). A job ref never
// pre-exists across jobs; it is created by the submit-time push and lives as
// long as the run dir (deleted on purge, or by the startup sweep once the run
// dir is gone), pinning the job's commit against mirror gc for resume.
func jobRef(jobID string) string {
	return "refs/harness/" + jobID
}

// git runs a git subcommand against the bare mirror (git -C <mirror> ...) through
// the injected runner and returns trimmed stdout. On failure the error carries
// stderr for diagnosis.
func (m *Manager) git(ctx context.Context, args ...string) (string, error) {
	// core.useReplaceRefs=false: a refs/replace/<obj> ref remaps an object at
	// read time. Without this, an attacker who can push to the mirror pushes a
	// genuinely signed commit to the job ref plus a refs/replace/<good-tree> ->
	// <evil-tree>; verify-commit still passes on the untouched commit while the
	// worktree checkout materializes the evil tree (cambia-550 review finding).
	// Disabling replace substitution on every mirror op closes it on both the
	// verify and the worktree-add path; verifyCommitSignature additionally
	// rejects outright a mirror that carries any replace ref.
	full := append([]string{"-C", m.mirrorDir, "-c", "core.useReplaceRefs=false"}, args...)
	res, err := m.runner.Run(ctx, Command{Name: "git", Args: full})
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(res.Stderr)))
	}
	return strings.TrimSpace(string(res.Stdout)), nil
}

// gitRaw runs a git subcommand and returns raw (untrimmed) stdout bytes, used
// where content bytes matter (blob content hashing).
func (m *Manager) gitRaw(ctx context.Context, args ...string) ([]byte, error) {
	// core.useReplaceRefs=false: see m.git; keep content-byte reads immune to
	// replace-ref remapping too.
	full := append([]string{"-C", m.mirrorDir, "-c", "core.useReplaceRefs=false"}, args...)
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

// verifyCommitSignature enforces ssh commit-signature verification when the
// runner is configured with RequireSignedCommits (cambia-550, W1). It runs
// `git -c gpg.ssh.allowedSignersFile=<path> verify-commit <commit>` against the
// bare mirror, where the object is already receipt-matched and present. A
// non-zero exit (unsigned, wrong key, or bad signature) rejects the job with
// ErrSignatureVerification. This one hook covers fresh dispatch, resume, and
// post-restart reconcile, all of which re-run Prepare.
//
// When enforcement is off the git verify never runs, so behavior is
// byte-for-byte unchanged. Enforcement fails closed: an empty AllowedSignersPath
// or a missing signers file rejects rather than silently passing.
func (m *Manager) verifyCommitSignature(ctx context.Context, commit string) error {
	if !m.requireSignedCommits {
		return nil
	}
	if m.allowedSignersPath == "" {
		return fmt.Errorf("%w: signed-commit enforcement is on but no allowed-signers path is configured", ErrSignatureVerification)
	}
	if _, err := os.Stat(m.allowedSignersPath); err != nil {
		return fmt.Errorf("%w: allowed-signers file %q is unreadable: %v", ErrSignatureVerification, m.allowedSignersPath, err)
	}
	// Reject a mirror carrying replace refs. m.git already disables replace
	// substitution (core.useReplaceRefs=false), so a pushed refs/replace/* is
	// inert for our reads; refusing outright surfaces the tampering loud instead
	// of silently ignoring it, and defends even a future call site that forgets
	// the flag. We only ever push refs/harness/*, so a replace ref is never
	// legitimate here (cambia-550 review finding). for-each-ref lists refs by
	// name regardless of useReplaceRefs, so detection is unaffected.
	if refs, err := m.git(ctx, "for-each-ref", "--format=%(refname)", "refs/replace/"); err != nil {
		return fmt.Errorf("%w: cannot enumerate replace refs: %v", ErrSignatureVerification, err)
	} else if refs != "" {
		return fmt.Errorf("%w: mirror carries replace ref(s) [%s]; refusing to stage a possibly-remapped tree", ErrSignatureVerification, strings.ReplaceAll(refs, "\n", ","))
	}
	if _, err := m.git(ctx, "-c", "gpg.ssh.allowedSignersFile="+m.allowedSignersPath, "verify-commit", commit); err != nil {
		return fmt.Errorf("%w: commit %s: %v", ErrSignatureVerification, commit, err)
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

// listJobRefs returns the job ids of every refs/harness/* ref in the mirror.
func (m *Manager) listJobRefs(ctx context.Context) ([]string, error) {
	out, err := m.git(ctx, "for-each-ref", "--format=%(refname)", "refs/harness/")
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, line := range strings.Split(out, "\n") {
		if id := strings.TrimPrefix(line, "refs/harness/"); id != "" && id != line {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// deleteJobRef removes the job-scoped ref (idempotent: a missing ref is not an
// error). Ref lifetime follows the run dir: PurgeRef and the startup sweep's
// run-dir-absence check are the only deleters.
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
