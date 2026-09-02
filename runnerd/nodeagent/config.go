// Package nodeagent is the nashnet compute-pool node role (design
// .docs/serving-harness/v1.1-compute-pool-design.md sections 1 to 7): the
// half of the protocol that runs on a node rather than on the coordinator.
// It loads the node's own `nashnet:` config, mints its own short-lived
// ed25519 tokens, talks to the coordinator over a fingerprint-pinned HTTPS
// client (D27), evaluates its own gates (D46), claims work (D2), stages it
// through the existing runnerd/ingest Manager with node-local paths (D15),
// launches it through procmgr's allowlisted child environment (D17), uploads
// its artifacts through the diff-probe-upload-commit loop (D50, D51), and
// posts the result (D6).
//
// The package holds no coordinator state and imports no coordinator code: it
// depends on runnerd/nashnet for the wire types, runnerd/nashnet/quarantine
// for the manifest shapes it has to speak, runnerd/nashnet/capability and
// .../gates for what it declares and evaluates, and runnerd/ingest,
// runnerd/procmgr, runnerd/pathguard, runnerd/sysprobe for the local work.
// Nothing here imports runnerd/harness, so the dispatcher can later depend on
// this package without a cycle (D1).
package nodeagent

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
)

// ErrConfig wraps every config load and validation failure so a caller can
// tell a bad config apart from an I/O error on the file.
var ErrConfig = errors.New("nashnet node config")

// ErrInsecureCoordinator is the D27 refusal: an agent that sets
// InsecureSkipVerify without the fingerprint compare has no authentication of
// the coordinator at all, so a coordinator URL that is not https:// is refused
// at config load rather than at the first request.
var ErrInsecureCoordinator = fmt.Errorf("%w: coordinator url must be https://", ErrConfig)

// defaultKinds is what a node offers when its config names no kinds: every
// kind runnerd supervises. A node narrows the list through gates.kinds_allowed
// (D46) or through this field, whichever the operator prefers.
var defaultKinds = []string{"train", "evaluate", "head-to-head", "bench", "measure"}

// fingerprintRe is the accepted spelling of coordinator.cert_sha256: 64 hex
// characters, case-insensitive, with an optional "sha256:" prefix so a
// fingerprint pasted from an ETag or from `openssl x509 -fingerprint` is
// accepted without hand-editing.
var fingerprintRe = regexp.MustCompile(`^(?:sha256:)?([0-9a-fA-F]{64})$`)

// File is the node's yaml document. The agent reads only the nashnet section,
// so the same file may carry unrelated top-level keys.
type File struct {
	Nashnet Config `yaml:"nashnet"`
}

// Coordinator names the one coordinator this node serves and the certificate
// it pins (D27). There is no second coordinator and no fallback: a node that
// cannot authenticate its coordinator does not run.
type Coordinator struct {
	URL string `yaml:"url"`
	// CertSHA256 is the SHA256 of the coordinator's leaf certificate in DER
	// form, the value VerifyPeerCertificate compares against.
	CertSHA256 string `yaml:"cert_sha256"`
}

// Caches carries the per-node overrides of the ingest cache ceilings (D18).
type Caches struct {
	MaxVenvs     int `yaml:"max_venvs,omitempty"`
	MaxLibcambia int `yaml:"max_libcambia,omitempty"`
}

// Config is the nashnet: section of a node's yaml (D46). Every path may be
// written with a leading ~ and is expanded at load; every path is absolute
// afterwards, so nothing downstream resolves against the process working
// directory.
type Config struct {
	// NodeID is optional and advisory: the authoritative id is derived from
	// the node's own key (D60). A configured id that disagrees with the
	// derived one is a load error rather than a silent override, since the
	// coordinator keys everything by the derived value.
	NodeID       string       `yaml:"node_id,omitempty"`
	Coordinator  Coordinator  `yaml:"coordinator"`
	KeyPath      string       `yaml:"key_path"`
	BaseDir      string       `yaml:"base_dir"`
	RunsDir      string       `yaml:"runs_dir"`
	Caches       Caches       `yaml:"caches,omitempty"`
	Gates        gates.Config `yaml:"gates,omitempty"`
	AgentVersion string       `yaml:"agent_version,omitempty"`
	PlatformTag  string       `yaml:"platform_tag,omitempty"`
	// Slots is the node's declared concurrency. It is a declaration, clamped
	// by the enrollment grant on the coordinator (D47); gates.concurrency is
	// the node's own enforcement of the same number.
	Slots int      `yaml:"slots,omitempty"`
	Kinds []string `yaml:"kinds,omitempty"`
	// Labels are declared for an operator's benefit only: the coordinator
	// matches labels_any against the grant's label set and never reads a
	// declared label (D47).
	Labels []string `yaml:"labels,omitempty"`
	// ClaimWaitSeconds is the node's requested long-poll hold, capped by the
	// coordinator at 30s (D2).
	ClaimWaitSeconds int `yaml:"claim_wait_seconds,omitempty"`
	// RequireSignedCommits mirrors RUNNERD_REQUIRE_SIGNED_COMMITS for the
	// node's own ingest Manager: the pin is checked on the node by the same
	// code that checks it on the coordinator (D48).
	RequireSignedCommits bool   `yaml:"require_signed_commits,omitempty"`
	AllowedSignersPath   string `yaml:"allowed_signers_path,omitempty"`
	// PythonBin is the interpreter uv builds the node's venvs against.
	PythonBin string `yaml:"python_bin,omitempty"`
}

// LoadConfig reads a node yaml and returns its validated nashnet section.
func LoadConfig(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("%w: read %s: %v", ErrConfig, path, err)
	}
	var f File
	if err := yaml.Unmarshal(data, &f); err != nil {
		return Config{}, fmt.Errorf("%w: parse %s: %v", ErrConfig, path, err)
	}
	cfg := f.Nashnet
	if err := cfg.normalize(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// normalize applies defaults, expands ~, and validates. It is the single gate
// every field passes before the agent starts, so no later code re-checks a
// scheme or re-expands a path.
func (c *Config) normalize() error {
	if err := c.normalizeCoordinator(); err != nil {
		return err
	}
	return c.NormalizeEmbedded()
}

// NormalizeEmbedded is normalize without the coordinator URL and certificate
// pin. The embedded node of --role both reaches its coordinator over an
// in-process handler (D65), so there is no URL to parse and no certificate to
// pin; every other default, expansion, and check is the one a remote node
// gets, because the two nodes are otherwise the same node.
func (c *Config) NormalizeEmbedded() error {
	return c.normalizeLocal()
}

// normalizeCoordinator validates the one coordinator a remote node serves.
func (c *Config) normalizeCoordinator() error {
	raw := strings.TrimSpace(c.Coordinator.URL)
	if raw == "" {
		return fmt.Errorf("%w: coordinator.url is required", ErrConfig)
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%w: coordinator.url %q: %v", ErrConfig, raw, err)
	}
	if !strings.EqualFold(u.Scheme, "https") {
		return fmt.Errorf("%w (got %q)", ErrInsecureCoordinator, raw)
	}
	if u.Host == "" {
		return fmt.Errorf("%w: coordinator.url %q names no host", ErrConfig, raw)
	}
	u.Scheme = "https"
	c.Coordinator.URL = strings.TrimSuffix(u.String(), "/")

	m := fingerprintRe.FindStringSubmatch(strings.TrimSpace(c.Coordinator.CertSHA256))
	if m == nil {
		return fmt.Errorf("%w: coordinator.cert_sha256 must be 64 hex characters", ErrConfig)
	}
	c.Coordinator.CertSHA256 = strings.ToLower(m[1])
	return nil
}

// normalizeLocal applies every default, expansion, and check that describes
// the node itself rather than its coordinator.
func (c *Config) normalizeLocal() error {
	for _, f := range []struct {
		name string
		p    *string
	}{
		{"key_path", &c.KeyPath},
		{"base_dir", &c.BaseDir},
		{"runs_dir", &c.RunsDir},
		{"allowed_signers_path", &c.AllowedSignersPath},
	} {
		expanded, err := expandPath(*f.p)
		if err != nil {
			return fmt.Errorf("%w: %s: %v", ErrConfig, f.name, err)
		}
		*f.p = expanded
	}
	if c.BaseDir == "" {
		return fmt.Errorf("%w: base_dir is required", ErrConfig)
	}
	if c.RunsDir == "" {
		c.RunsDir = filepath.Join(c.BaseDir, "runs")
	}
	if c.KeyPath == "" {
		c.KeyPath = filepath.Join(c.BaseDir, "nashnet-node.key")
	}
	if err := c.expandGatePaths(); err != nil {
		return err
	}
	if c.Slots <= 0 {
		c.Slots = 1
	}
	if len(c.Kinds) == 0 {
		c.Kinds = append([]string(nil), defaultKinds...)
	}
	if c.ClaimWaitSeconds <= 0 {
		c.ClaimWaitSeconds = 25
	}
	if c.AgentVersion == "" {
		c.AgentVersion = AgentVersion
	}
	if c.PlatformTag == "" {
		c.PlatformTag = platformTag()
	}
	if c.PythonBin == "" {
		c.PythonBin = "python3"
	}
	return nil
}

// expandGatePaths expands the one gate field that names a file. Gates are
// evaluated against a probe snapshot, so this is the only filesystem path the
// gate config carries.
func (c *Config) expandGatePaths() error {
	if c.Gates.Drain == nil || c.Gates.Drain.File == "" {
		return nil
	}
	p, err := expandPath(c.Gates.Drain.File)
	if err != nil {
		return fmt.Errorf("%w: gates.drain.file: %v", ErrConfig, err)
	}
	c.Gates.Drain.File = p
	return nil
}

// expandPath turns "" into "", expands a leading ~, and makes the result
// absolute. A bare ~ and ~/x expand against the current user's home; ~other is
// refused rather than guessed at.
func expandPath(p string) (string, error) {
	if p == "" {
		return "", nil
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		p = filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(p, "~"), "/"))
	} else if strings.HasPrefix(p, "~") {
		return "", fmt.Errorf("%q: ~user expansion is not supported", p)
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return "", err
	}
	return abs, nil
}
