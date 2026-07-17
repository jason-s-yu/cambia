package ingest

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// requireGitSSHSigning skips the calling test unless the host git is >= 2.34
// (the first release with ssh commit signing/verification) and ssh-keygen is
// present. This keeps the signed-commit suite from failing on older toolchains.
func requireGitSSHSigning(t *testing.T) {
	t.Helper()
	out, err := exec.Command("git", "--version").Output()
	if err != nil {
		t.Skipf("git unavailable: %v", err)
	}
	fields := strings.Fields(string(out))
	if len(fields) < 3 {
		t.Skipf("cannot parse git version: %q", strings.TrimSpace(string(out)))
	}
	if !gitVersionAtLeast(fields[2], 2, 34) {
		t.Skipf("git %s < 2.34: ssh commit signing unsupported", fields[2])
	}
	if _, err := exec.LookPath("ssh-keygen"); err != nil {
		t.Skipf("ssh-keygen unavailable: %v", err)
	}
}

// gitVersionAtLeast reports whether a dotted git version string (e.g.
// "2.34.1.windows.1") is at least major.minor.
func gitVersionAtLeast(ver string, major, minor int) bool {
	parts := strings.Split(ver, ".")
	if len(parts) < 2 {
		return false
	}
	maj, err1 := strconv.Atoi(parts[0])
	min, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil {
		return false
	}
	if maj != major {
		return maj > major
	}
	return min >= minor
}

// genSSHSigningKey creates an ed25519 keypair with no passphrase in a temp dir
// and returns the private key path plus the trimmed public-key line.
func genSSHSigningKey(t *testing.T) (keyPath, pubLine string) {
	t.Helper()
	keyPath = filepath.Join(t.TempDir(), "id_ed25519")
	cmd := exec.Command("ssh-keygen", "-t", "ed25519", "-N", "", "-C", "signer@test", "-f", keyPath)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("ssh-keygen failed: %v\n%s", err, out)
	}
	pub, err := os.ReadFile(keyPath + ".pub")
	if err != nil {
		t.Fatalf("read pubkey: %v", err)
	}
	return keyPath, strings.TrimSpace(string(pub))
}

// writeAllowedSigners writes a git allowed_signers file authorizing pubLine for
// the git namespace under the given principal, and returns its path.
func writeAllowedSigners(t *testing.T, principal, pubLine string) string {
	t.Helper()
	f := strings.Fields(pubLine)
	if len(f) < 2 {
		t.Fatalf("malformed public key line: %q", pubLine)
	}
	path := filepath.Join(t.TempDir(), "allowed_signers")
	mustWrite(t, path, principal+` namespaces="git" `+f[0]+" "+f[1]+"\n")
	return path
}

// runGitSigned runs git in dir with ssh commit-signing config injected via -c
// flags so a fixture commit carries an ssh signature. keyPath is the private
// key; git reads keyPath.pub for the public half.
func runGitSigned(t *testing.T, dir, keyPath string, args ...string) string {
	t.Helper()
	full := append([]string{
		"-c", "gpg.format=ssh",
		"-c", "user.signingkey=" + keyPath,
	}, args...)
	return runGit(t, dir, full...)
}

// signedSourceRepo mirrors sourceRepo but signs the single commit with keyPath's
// ssh key. It returns the repo dir and the signed commit sha.
func signedSourceRepo(t *testing.T, lockContent, keyPath string) (dir, sha string) {
	t.Helper()
	dir = t.TempDir()
	runGit(t, dir, "init", "-q")
	runGit(t, dir, "config", "user.email", "test@test")
	runGit(t, dir, "config", "user.name", "test")
	mustWrite(t, filepath.Join(dir, "cfr", "uv.lock"), lockContent)
	mustWrite(t, filepath.Join(dir, "cfr", "pyproject.toml"), "[project]\nname='cambia-cfr'\n")
	mustWrite(t, filepath.Join(dir, "cfr", "src", "cli.py"), "# cli\n")
	mustWrite(t, filepath.Join(dir, "engine", "go.mod"), "module e\n\ngo 1.26.0\n")
	mustWrite(t, filepath.Join(dir, "engine", "cgo", "exports.go"), "package main\n\nfunc main(){}\n")
	runGit(t, dir, "add", "-A")
	runGitSigned(t, dir, keyPath, "commit", "-S", "-q", "-m", "init")
	sha = runGit(t, dir, "rev-parse", "HEAD")
	return dir, sha
}

// signingFakeManager builds a Manager with the ssh-signature gate configured and
// a faked toolchain runner (git runs for real), then initializes its mirror.
func signingFakeManager(t *testing.T, fc *fakeControl, require bool, signersPath string) (*Manager, *fakeRunner) {
	t.Helper()
	fr := newFakeRunner()
	fr.hook = fc.hook()
	base := t.TempDir()
	m := New(Config{
		BaseDir:              base,
		RunsDir:              filepath.Join(base, "runs"),
		MaxVenvs:             8,
		MaxLibcambia:         50,
		CoresCap:             18,
		PythonBin:            "python3",
		Runner:               fr,
		Now:                  time.Now,
		RequireSignedCommits: require,
		AllowedSignersPath:   signersPath,
	})
	if err := m.ensureMirror(context.Background()); err != nil {
		t.Fatalf("ensureMirror: %v", err)
	}
	return m, fr
}

// hasVerifyCommit reports whether any recorded git invocation ran verify-commit.
func hasVerifyCommit(fr *fakeRunner) bool {
	for _, c := range fr.callsFor("git") {
		for _, a := range c.Args {
			if a == "verify-commit" {
				return true
			}
		}
	}
	return false
}

// TestPrepareSignedCommitAccepted: a commit signed by an authorized key, with
// enforcement on, stages successfully.
func TestPrepareSignedCommitAccepted(t *testing.T) {
	requireGitSSHSigning(t)
	key, pub := genSSHSigningKey(t)
	signers := writeAllowedSigners(t, "signer@test", pub)

	fc := newFakeControl()
	m, fr := signingFakeManager(t, fc, true, signers)
	src, sha := signedSourceRepo(t, "prod-lock", key)
	pushJobRef(t, m, src, sha, "job-signed")

	if _, err := m.Prepare(context.Background(), "job-signed", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil); err != nil {
		t.Fatalf("Prepare with valid signature: %v", err)
	}
	if !hasVerifyCommit(fr) {
		t.Fatal("verify-commit was not invoked under enforcement")
	}
}

// TestPrepareUnsignedCommitRejected: an unsigned commit under enforcement is
// rejected with ErrSignatureVerification.
func TestPrepareUnsignedCommitRejected(t *testing.T) {
	requireGitSSHSigning(t)
	_, pub := genSSHSigningKey(t)
	signers := writeAllowedSigners(t, "signer@test", pub)

	fc := newFakeControl()
	m, _ := signingFakeManager(t, fc, true, signers)
	src, sha := sourceRepo(t, "prod-lock") // unsigned
	pushJobRef(t, m, src, sha, "job-unsigned")

	_, err := m.Prepare(context.Background(), "job-unsigned", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil)
	if !errors.Is(err, ErrSignatureVerification) {
		t.Fatalf("want ErrSignatureVerification, got %v", err)
	}
}

// TestPrepareWrongKeyRejected: a commit signed by a key absent from
// allowed_signers is rejected even though it carries a valid signature.
func TestPrepareWrongKeyRejected(t *testing.T) {
	requireGitSSHSigning(t)
	signKey, _ := genSSHSigningKey(t)
	_, otherPub := genSSHSigningKey(t) // a different key populates allowed_signers
	signers := writeAllowedSigners(t, "signer@test", otherPub)

	fc := newFakeControl()
	m, _ := signingFakeManager(t, fc, true, signers)
	src, sha := signedSourceRepo(t, "prod-lock", signKey)
	pushJobRef(t, m, src, sha, "job-wrongkey")

	_, err := m.Prepare(context.Background(), "job-wrongkey", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil)
	if !errors.Is(err, ErrSignatureVerification) {
		t.Fatalf("want ErrSignatureVerification, got %v", err)
	}
}

// TestPrepareEnforcementOffPassesUnsigned is the regression guard: with
// enforcement off, an unsigned commit stages and verify-commit never runs.
func TestPrepareEnforcementOffPassesUnsigned(t *testing.T) {
	fc := newFakeControl()
	m, fr := signingFakeManager(t, fc, false, "")
	src, sha := sourceRepo(t, "prod-lock") // unsigned
	pushJobRef(t, m, src, sha, "job-off")

	if _, err := m.Prepare(context.Background(), "job-off", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil); err != nil {
		t.Fatalf("Prepare with enforcement off: %v", err)
	}
	if hasVerifyCommit(fr) {
		t.Fatal("verify-commit ran while enforcement was off")
	}
}

// TestPrepareFailsClosedOnEmptySignersPath: enforcement on but no signers path
// configured rejects, even for a validly signed commit.
func TestPrepareFailsClosedOnEmptySignersPath(t *testing.T) {
	requireGitSSHSigning(t)
	key, _ := genSSHSigningKey(t)

	fc := newFakeControl()
	m, _ := signingFakeManager(t, fc, true, "")
	src, sha := signedSourceRepo(t, "prod-lock", key)
	pushJobRef(t, m, src, sha, "job-nopath")

	_, err := m.Prepare(context.Background(), "job-nopath", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil)
	if !errors.Is(err, ErrSignatureVerification) {
		t.Fatalf("want ErrSignatureVerification for empty signers path, got %v", err)
	}
}

// TestPrepareReplaceRefBypassRejected is the regression guard for the cambia-550
// review finding: an attacker with only mirror-push access pushes a genuinely
// signed commit to the job ref, then pushes a refs/replace/<good-tree> ->
// <evil-tree> that remaps the commit's tree at read time. Without the fix,
// verify-commit passes on the untouched commit while the worktree checkout
// materializes the evil tree (RCE). Prepare must reject: the mirror carries a
// replace ref, and replace substitution is disabled for our reads regardless.
func TestPrepareReplaceRefBypassRejected(t *testing.T) {
	requireGitSSHSigning(t)
	key, pub := genSSHSigningKey(t)
	signers := writeAllowedSigners(t, "signer@test", pub)

	fc := newFakeControl()
	m, _ := signingFakeManager(t, fc, true, signers)

	// The good, signed commit the attacker legitimately pushes to the job ref.
	src, sha := signedSourceRepo(t, "prod-lock", key)
	pushJobRef(t, m, src, sha, "job-replace")

	// An evil tree with different content; push it into the mirror so its objects
	// are present, then point a replace ref at the good commit's tree.
	evilSrc, evilSha := sourceRepo(t, "evil-lock") // distinct content -> distinct tree
	pushJobRef(t, m, evilSrc, evilSha, "evil-stash")

	goodTree := runGit(t, m.mirrorDir, "rev-parse", sha+"^{tree}")
	evilTree := runGit(t, m.mirrorDir, "rev-parse", evilSha+"^{tree}")
	if goodTree == evilTree {
		t.Fatalf("fixture bug: good and evil trees are identical (%s)", goodTree)
	}
	runGit(t, m.mirrorDir, "update-ref", "refs/replace/"+goodTree, evilTree)

	_, err := m.Prepare(context.Background(), "job-replace", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil)
	if !errors.Is(err, ErrSignatureVerification) {
		t.Fatalf("want ErrSignatureVerification for a mirror carrying a replace ref, got %v", err)
	}
}

// TestPrepareFailsClosedOnMissingSignersFile: enforcement on with a signers path
// that does not exist rejects rather than silently passing.
func TestPrepareFailsClosedOnMissingSignersFile(t *testing.T) {
	requireGitSSHSigning(t)
	key, _ := genSSHSigningKey(t)

	fc := newFakeControl()
	m, _ := signingFakeManager(t, fc, true, filepath.Join(t.TempDir(), "does-not-exist"))
	src, sha := signedSourceRepo(t, "prod-lock", key)
	pushJobRef(t, m, src, sha, "job-missing")

	_, err := m.Prepare(context.Background(), "job-missing", sha, "train",
		"cfr/config/prtcfr.yaml", "cpu", "", nil)
	if !errors.Is(err, ErrSignatureVerification) {
		t.Fatalf("want ErrSignatureVerification for missing signers file, got %v", err)
	}
}
