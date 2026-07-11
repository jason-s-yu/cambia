// Package harnessproxy is the dashboard's client to a serving-harness runner
// control plane (design 5.1/5.2, cambia-295 v1.1). It reads the host-local
// harness config, mints short-lived EdDSA JWTs off the harness ed25519 signing
// key, and speaks to the runner over a fingerprint-pinned TLS channel. It is the
// Go mirror of cfr/src/harness/{config,transport}.py, kept separate from
// service/internal/auth (whose package-global key is the user-session key and
// whose CreateJWT stamps no audience).
//
// A nil *Client (harness config absent) is the "no proxy" state: every caller
// treats it as the pre-v1.1 behavior (remote runs are read-only, 409).
package harnessproxy

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// DashboardSubject is the JWT "sub" claim the dashboard mints its control-plane
// tokens under. runnerd does not enforce sub against any configured value (it
// discards the verified subject: runnerd/harness/server.go requireBearer), so
// the dashboard uses a subject distinct from the CLI's auth.subject purely for
// audit separation in the runner's request logs (a control action minted here
// is attributable to the dashboard, not the CLI).
const DashboardSubject = "pangu-dashboard"

// TokenAudience is the required "aud" claim; runnerd rejects any token without
// it (runnerd/authtoken.Audience). Mirrors transport.py TOKEN_AUDIENCE.
const TokenAudience = "cambia-runnerd"

// NBFBackdate absorbs small mint-host/runner clock skew so a runner whose clock
// trails the dashboard by a few seconds does not reject a fresh token. Mirrors
// transport.py NBF_BACKDATE_SECONDS; runnerd applies an independent leeway.
const NBFBackdate = 30 * time.Second

// DefaultTokenTTL is the mint lifetime when the config omits token_ttl_seconds.
const DefaultTokenTTL = 900 * time.Second

// MaxTokenTTL is the hard ceiling on a minted token's lifetime (design 5.2:
// short-lived, <= 1h). Mirrors config._MAX_TOKEN_TTL_SECONDS.
const MaxTokenTTL = 3600 * time.Second

// defaultUserConfig is the documented default config path; it holds host-local
// key paths and is never checked in. Mirrors config.DEFAULT_USER_CONFIG.
const defaultUserConfig = "~/.config/cambia/harness.yaml"

// ErrNoConfig signals that no harness config file was found on any candidate
// path. main.go maps it to a nil *Client (no proxy configured), not a fatal.
var ErrNoConfig = errors.New("harnessproxy: no harness config found")

// Config is the subset of the harness config the dashboard proxy needs: the
// runner control-plane URL, its pinned self-signed cert fingerprint, the
// ed25519 signing key path, the token TTL, and the origin host the runner
// stamps onto reconciled runs. It intentionally omits the CLI-only data-plane
// fields (ssh alias, mirror remote, rsync dir).
type Config struct {
	RunnerURL       string
	CertFingerprint string
	PrivateKeyPath  string
	// Subject is auth.subject from the file (the CLI's sub). The dashboard does
	// NOT mint under it: it mints under DashboardSubject for audit separation,
	// because runnerd does not enforce sub. Retained so the parsed Config
	// mirrors the file and a future enforced-sub world can switch to it.
	Subject    string
	TokenTTL   time.Duration
	OriginHost string
	SourcePath string
}

// yamlConfig mirrors the on-disk YAML shape (a superset; unknown keys ignored).
type yamlConfig struct {
	Runner struct {
		URL             string `yaml:"url"`
		CertFingerprint string `yaml:"cert_fingerprint"`
	} `yaml:"runner"`
	Auth struct {
		PrivateKeyPath  string `yaml:"private_key_path"`
		Subject         string `yaml:"subject"`
		TokenTTLSeconds int    `yaml:"token_ttl_seconds"`
	} `yaml:"auth"`
	DataPlane struct {
		OriginHost string `yaml:"origin_host"`
		SSHAlias   string `yaml:"ssh_alias"`
	} `yaml:"data_plane"`
}

// expandUser expands a leading ~ to the current user's home directory, matching
// Python's Path.expanduser used throughout the harness config.
func expandUser(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			if p == "~" {
				return home
			}
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// candidatePaths returns the config search order: CAMBIA_HARNESS_CONFIG (when
// set) then ~/.config/cambia/harness.yaml. First existing wins. This is the
// dashboard subset of config._candidate_paths (the repo-relative fallback is
// CLI-only and deliberately not consulted server-side).
func candidatePaths() []string {
	var paths []string
	if env := os.Getenv("CAMBIA_HARNESS_CONFIG"); env != "" {
		paths = append(paths, expandUser(env))
	}
	paths = append(paths, expandUser(defaultUserConfig))
	return paths
}

// LoadConfig reads and validates the harness config from the first candidate
// path that exists. It returns ErrNoConfig when none is found (the no-proxy
// state), or a descriptive error on a malformed/incomplete file.
func LoadConfig() (*Config, error) {
	for _, p := range candidatePaths() {
		if _, err := os.Stat(p); err == nil {
			return loadFrom(p)
		}
	}
	return nil, ErrNoConfig
}

// loadFrom parses and validates a specific config file.
func loadFrom(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("harnessproxy: read %s: %w", path, err)
	}
	var y yamlConfig
	if err := yaml.Unmarshal(data, &y); err != nil {
		return nil, fmt.Errorf("harnessproxy: parse %s: %w", path, err)
	}
	ttl := DefaultTokenTTL
	if y.Auth.TokenTTLSeconds > 0 {
		ttl = time.Duration(y.Auth.TokenTTLSeconds) * time.Second
	}
	subject := y.Auth.Subject
	if subject == "" {
		subject = "cambia-harness"
	}
	origin := y.DataPlane.OriginHost
	if origin == "" {
		origin = y.DataPlane.SSHAlias // mirror config.from_dict's origin fallback
	}
	c := &Config{
		RunnerURL:       y.Runner.URL,
		CertFingerprint: y.Runner.CertFingerprint,
		PrivateKeyPath:  expandUser(y.Auth.PrivateKeyPath),
		Subject:         subject,
		TokenTTL:        ttl,
		OriginHost:      origin,
		SourcePath:      path,
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	return c, nil
}

// validate enforces the invariants the transport depends on: an https runner
// URL, a well-formed 64-hex fingerprint, a key path, a bounded positive TTL, and
// a non-empty origin host (without it the dashboard cannot decide which remote
// runs are controllable). Mirrors config.HarnessConfig.validate.
func (c *Config) validate() error {
	if !strings.HasPrefix(c.RunnerURL, "https://") {
		return fmt.Errorf("harnessproxy: runner.url must be https:// (plaintext refused), got %q", c.RunnerURL)
	}
	if _, err := NormalizeFingerprint(c.CertFingerprint); err != nil {
		return fmt.Errorf("harnessproxy: runner.cert_fingerprint: %w", err)
	}
	if c.PrivateKeyPath == "" {
		return errors.New("harnessproxy: auth.private_key_path is required")
	}
	if c.TokenTTL <= 0 || c.TokenTTL > MaxTokenTTL {
		return fmt.Errorf("harnessproxy: auth.token_ttl_seconds must be in (0, %d]", int(MaxTokenTTL.Seconds()))
	}
	if c.OriginHost == "" {
		return errors.New("harnessproxy: data_plane.origin_host (or ssh_alias) is required")
	}
	return nil
}
