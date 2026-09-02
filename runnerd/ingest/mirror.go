package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// jobRef returns the job-scoped ref name for jobID (design 3.1). A job ref never
// pre-exists across jobs; it is created by the submit-time push and lives as
// long as the run dir (deleted on purge, or by the startup sweep once the run
// dir is gone), pinning the job's commit against mirror gc for resume.
func jobRef(jobID string) string {
	return "refs/harness/" + jobID
}

// snapshotRefPrefix is the namespace the job-neutral bundle refs live in. It
// sits outside refs/harness/ so a snapshot ref can never collide with the job
// ref of a job whose id is "snapshot", and so the job-ref sweep keeps seeing
// exactly the job refs.
const snapshotRefPrefix = "refs/harness-snapshots/"

// snapshotRef returns the job-neutral ref a bundle for commit is built under
// (cambia-2128). A bundle records the ref named on the build argv, so building
// under the requesting job's ref put that job's identity inside an artifact the
// cache keys by commit and basis alone: the next job served that cache entry
// fetched a ref the bundle did not carry. Naming the commit instead makes the
// artifact match its key, and BundleFetch maps it onto the requesting job's
// ref.
func snapshotRef(commit string) string {
	return snapshotRefPrefix + commit
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

// resolveJobRef resolves jobID's mirror ref to its commit, without comparing it
// against any expected value. A missing ref or non-commit target is
// ErrReceiptMismatch. Shared by verifyReceipt (which additionally checks the
// resolved commit against the spec) and BundleCreate (which has no expected
// commit to check against; the ref's current target is the commit to bundle).
func (m *Manager) resolveJobRef(ctx context.Context, jobID string) (string, error) {
	ref := jobRef(jobID)
	resolved, err := m.git(ctx, "rev-parse", "--verify", "--quiet", ref+"^{commit}")
	if err != nil || resolved == "" {
		return "", fmt.Errorf("%w: ref %s does not resolve to a commit", ErrReceiptMismatch, ref)
	}
	return resolved, nil
}

// verifyReceipt is the runner-side receipt check (design 3.1): the job ref must
// resolve to a commit equal to the spec commit, and that object must exist. A
// missing ref, missing object, or sha mismatch is ErrReceiptMismatch. No ref is
// created or updated here; the ref is authored solely by the submit-time push.
func (m *Manager) verifyReceipt(ctx context.Context, jobID, commit string) error {
	resolved, err := m.resolveJobRef(ctx, jobID)
	if err != nil {
		return err
	}
	if resolved != commit {
		return fmt.Errorf("%w: ref %s -> %s, spec commit %s", ErrReceiptMismatch, jobRef(jobID), resolved, commit)
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

// BundleFetch is the file-drop fallback transport (design 3.1): it fetches the
// bundle's job-neutral snapshot ref for commit into this job's own ref, without
// force. Because the fetch omits the "+" force prefix, a job ref that already
// points elsewhere and is not fast-forwardable is refused by git rather than
// overwritten. After the fetch the caller runs verifyReceipt as usual.
//
// Naming the commit rather than the job on the bundle's side is what lets one
// cached artifact serve every job pinned to that commit (cambia-2128), and it
// makes the fetch itself assert what the bundle delivers: a bundle built for
// another commit carries no such ref and fails here rather than importing
// something the receipt check would have to catch.
//
// A thin bundle (BundleCreate with a non-empty basis) whose negated basis
// commit this mirror does not already hold fails the fetch with git's
// "Repository lacks these prerequisite commits" error; that case is reported as
// ErrBundlePrereqMissing so the node-side caller can nack bundle_prereq_miss and
// re-claim with an empty basis (design 3.3, D48) instead of treating it as a
// generic transport failure.
func (m *Manager) BundleFetch(ctx context.Context, jobID, commit, bundlePath string) error {
	if err := validateJobID(jobID); err != nil {
		return err
	}
	if err := validateCommit(commit); err != nil {
		return err
	}
	if err := m.ensureMirror(ctx); err != nil {
		return err
	}
	refspec := snapshotRef(commit) + ":" + jobRef(jobID)
	if _, err := m.git(ctx, "fetch", bundlePath, refspec); err != nil {
		if strings.Contains(err.Error(), "lacks these prerequisite commits") {
			return fmt.Errorf("%w: %v", ErrBundlePrereqMissing, err)
		}
		return fmt.Errorf("bundle fetch: %w", err)
	}
	return nil
}

// bundleBasisCap is the maximum number of have_commits entries a claim may
// advertise for thin-bundle negation (design 3.3, D48); anything past it is
// refused before it ever reaches a git argv.
const bundleBasisCap = 16

// bundleExt and bundleSidecarExt name a cached bundle and the sha256-digest
// sidecar published alongside it, so a cache hit never re-hashes a
// potentially large file to answer with its descriptor.
const (
	bundleExt        = ".bundle"
	bundleSidecarExt = ".sha256"
)

// BundleDescriptor is the built (or cache-reused) artifact BundleCreate
// returns: the on-disk bundle path, its byte size, and its content sha256 -
// the digest a claim response reports as snapshot.sha256/snapshot.size and the
// coordinator's file server reports as the ETag (design 3.3, D48).
type BundleDescriptor struct {
	Path   string
	Size   int64
	SHA256 string
	// Cached reports that the artifact was already on disk when this call
	// asked for it, which is the sharing D48 is built for: one bundle per
	// (commit, basis) rather than one per job. It is what a caller logs the
	// hit from and rides outside the digest and size a claim response carries.
	Cached bool
}

// bundleCacheKey derives the cache filename stem from the commit and a
// validated basis set: "<commit>" for the full-tree bundle (empty basis), else
// "<commit>-<basis-digest>" where basis-digest is the sha256 of the sorted
// basis list, so two claims naming the same basis set in a different order
// still share one cache entry.
func bundleCacheKey(commit string, basis []string) string {
	if len(basis) == 0 {
		return commit
	}
	sorted := append([]string(nil), basis...)
	sort.Strings(sorted)
	h := sha256.Sum256([]byte(strings.Join(sorted, "\n")))
	return commit + "-" + hex.EncodeToString(h[:])[:16]
}

// BundleCreate returns the git bundle delivering jobID's pinned commit,
// building it through the injected runner on a cache miss and reusing a
// cached artifact otherwise (design 3.3, D48). basis is the requesting node's
// have_commits list: each entry is validated as a 40-hex commit present in
// this mirror and negated in the bundle's rev-list; a malformed entry, one
// absent from the mirror, or a list past bundleBasisCap is refused with
// ErrInvalidBasis and no git subprocess runs. Concurrent calls for the same
// (commit, basis) collapse into one build (singleflight), and the cache is
// LRU-evicted at cfg.MaxSnapshots.
//
// Callers must not hold the placement/claim lock while calling this: a
// cache-miss build forks a whole-repository `git bundle create`, and no git
// subprocess may run while that lock is held (design 3.3).
func (m *Manager) BundleCreate(ctx context.Context, jobID string, basis []string) (BundleDescriptor, error) {
	if err := validateJobID(jobID); err != nil {
		return BundleDescriptor{}, err
	}
	if len(basis) > bundleBasisCap {
		return BundleDescriptor{}, fmt.Errorf("%w: %d entries exceeds cap %d", ErrInvalidBasis, len(basis), bundleBasisCap)
	}
	commit, err := m.resolveJobRef(ctx, jobID)
	if err != nil {
		return BundleDescriptor{}, err
	}

	clean := make([]string, len(basis))
	for i, b := range basis {
		if verr := validateCommit(b); verr != nil {
			return BundleDescriptor{}, fmt.Errorf("%w: %v", ErrInvalidBasis, verr)
		}
		if _, rerr := m.git(ctx, "rev-parse", "--verify", "--quiet", b+"^{commit}"); rerr != nil {
			return BundleDescriptor{}, fmt.Errorf("%w: basis %s not present in mirror", ErrInvalidBasis, b)
		}
		clean[i] = b
	}

	if err := os.MkdirAll(m.snapshotDir, 0o755); err != nil {
		return BundleDescriptor{}, err
	}
	key := bundleCacheKey(commit, clean)
	path := filepath.Join(m.snapshotDir, key+bundleExt)

	if desc, ok := m.bundleCacheHit(path); ok {
		return desc, nil
	}

	return m.bundleGroup.Do(key, func() (BundleDescriptor, error) {
		if desc, ok := m.bundleCacheHit(path); ok {
			return desc, nil
		}
		desc, berr := m.buildBundle(ctx, commit, clean, path)
		if berr != nil {
			return BundleDescriptor{}, berr
		}
		m.evictSnapshots(map[string]bool{key: true})
		return desc, nil
	})
}

// bundleCacheHit reports whether a bundle and its digest sidecar are both
// present at path, returning its descriptor and bumping its LRU mtime. A
// bundle without its sidecar (a build interrupted between the two renames in
// buildBundle) is treated as a miss so the caller rebuilds it.
func (m *Manager) bundleCacheHit(path string) (BundleDescriptor, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return BundleDescriptor{}, false
	}
	digest, err := os.ReadFile(path + bundleSidecarExt)
	if err != nil {
		return BundleDescriptor{}, false
	}
	touch(path, m.now())
	return BundleDescriptor{Path: path, Size: info.Size(), SHA256: strings.TrimSpace(string(digest)), Cached: true}, true
}

// buildBundle forks the actual `git bundle create`, hashes the result, and
// publishes it (and its sidecar digest) atomically so a reader never observes
// a partially written bundle. The build first publishes the job-neutral
// snapshot ref for the commit, because that ref name is what the bundle
// records and what every job's BundleFetch names (cambia-2128). The ref is
// idempotent across concurrent builds of one commit and is reaped by
// StartupSweep, which is where the mirror's gc runs.
func (m *Manager) buildBundle(ctx context.Context, commit string, basis []string, path string) (BundleDescriptor, error) {
	if _, err := m.git(ctx, "update-ref", snapshotRef(commit), commit); err != nil {
		return BundleDescriptor{}, fmt.Errorf("publish snapshot ref: %w", err)
	}
	tmp, err := os.CreateTemp(m.snapshotDir, filepath.Base(path)+".tmp-*")
	if err != nil {
		return BundleDescriptor{}, err
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpPath) // no-op once renamed away below

	args := []string{"bundle", "create", tmpPath, snapshotRef(commit)}
	for _, b := range basis {
		args = append(args, "^"+b)
	}
	if _, err := m.git(ctx, args...); err != nil {
		return BundleDescriptor{}, fmt.Errorf("bundle create: %w", err)
	}

	digest, size, err := sha256FileStream(tmpPath)
	if err != nil {
		return BundleDescriptor{}, err
	}

	sidecarTmp := tmpPath + bundleSidecarExt
	if err := os.WriteFile(sidecarTmp, []byte(digest), 0o644); err != nil {
		return BundleDescriptor{}, err
	}
	// Sidecar renamed first, bundle second: bundleCacheHit only accepts a
	// bundle whose sidecar already exists, so a crash between these two
	// renames leaves a stray sidecar and no bundle, never the reverse.
	if err := os.Rename(sidecarTmp, path+bundleSidecarExt); err != nil {
		return BundleDescriptor{}, err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return BundleDescriptor{}, err
	}
	return BundleDescriptor{Path: path, Size: size, SHA256: digest}, nil
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

// listSnapshotRefs returns every job-neutral bundle ref in the mirror, by full
// ref name.
func (m *Manager) listSnapshotRefs(ctx context.Context) ([]string, error) {
	out, err := m.git(ctx, "for-each-ref", "--format=%(refname)", snapshotRefPrefix)
	if err != nil {
		return nil, err
	}
	var refs []string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, snapshotRefPrefix) && line != snapshotRefPrefix {
			refs = append(refs, line)
		}
	}
	return refs, nil
}

// deleteSnapshotRefs removes every job-neutral bundle ref, and is a no-op on a
// base dir with no mirror yet. A snapshot ref is needed only while the build it
// names runs, and it outlives the build only so concurrent builds of one commit
// never delete each other's ref; the sweep is where it goes, because the sweep
// runs at daemon start with no build in flight and is the only place the
// mirror's pruning gc runs (cambia-2128).
func (m *Manager) deleteSnapshotRefs(ctx context.Context) error {
	if !isGitDir(m.mirrorDir) {
		return nil
	}
	refs, err := m.listSnapshotRefs(ctx)
	if err != nil {
		return err
	}
	var firstErr error
	for _, ref := range refs {
		if _, derr := m.git(ctx, "update-ref", "-d", ref); derr != nil && firstErr == nil {
			firstErr = derr
		}
	}
	return firstErr
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
