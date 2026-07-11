package harnessproxy

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

const goodFingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "harness.yaml")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadConfigViaEnv(t *testing.T) {
	body := `
runner:
  url: https://runner.example:8090
  cert_fingerprint: "` + goodFingerprint + `"
auth:
  private_key_path: /keys/ed25519
  subject: cambia-harness
  token_ttl_seconds: 600
data_plane:
  origin_host: nash
`
	p := writeConfig(t, body)
	t.Setenv("CAMBIA_HARNESS_CONFIG", p)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.RunnerURL != "https://runner.example:8090" {
		t.Errorf("RunnerURL = %q", cfg.RunnerURL)
	}
	if cfg.OriginHost != "nash" {
		t.Errorf("OriginHost = %q, want nash", cfg.OriginHost)
	}
	if cfg.TokenTTL != 600*time.Second {
		t.Errorf("TokenTTL = %v, want 600s", cfg.TokenTTL)
	}
	if cfg.PrivateKeyPath != "/keys/ed25519" {
		t.Errorf("PrivateKeyPath = %q", cfg.PrivateKeyPath)
	}
}

func TestLoadConfigNoConfigWhenAbsent(t *testing.T) {
	// Point the env override at a nonexistent path and the user default at an
	// empty HOME so neither candidate exists.
	t.Setenv("CAMBIA_HARNESS_CONFIG", filepath.Join(t.TempDir(), "missing.yaml"))
	t.Setenv("HOME", t.TempDir())
	_, err := LoadConfig()
	if !errors.Is(err, ErrNoConfig) {
		t.Fatalf("LoadConfig err = %v, want ErrNoConfig", err)
	}
}

func TestLoadConfigOriginFallsBackToSSHAlias(t *testing.T) {
	body := `
runner:
  url: https://r:8090
  cert_fingerprint: "` + goodFingerprint + `"
auth:
  private_key_path: /k
data_plane:
  ssh_alias: runner
`
	p := writeConfig(t, body)
	t.Setenv("CAMBIA_HARNESS_CONFIG", p)
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.OriginHost != "runner" {
		t.Errorf("OriginHost = %q, want runner (ssh_alias fallback)", cfg.OriginHost)
	}
	if cfg.TokenTTL != DefaultTokenTTL {
		t.Errorf("TokenTTL = %v, want default %v", cfg.TokenTTL, DefaultTokenTTL)
	}
}

func TestValidateRejects(t *testing.T) {
	base := func() *Config {
		return &Config{
			RunnerURL:       "https://r:8090",
			CertFingerprint: goodFingerprint,
			PrivateKeyPath:  "/k",
			TokenTTL:        900 * time.Second,
			OriginHost:      "nash",
		}
	}
	cases := map[string]func(*Config){
		"plaintext url":    func(c *Config) { c.RunnerURL = "http://r:8090" },
		"bad fingerprint":  func(c *Config) { c.CertFingerprint = "xyz" },
		"missing key":      func(c *Config) { c.PrivateKeyPath = "" },
		"zero ttl":         func(c *Config) { c.TokenTTL = 0 },
		"ttl over ceiling": func(c *Config) { c.TokenTTL = 2 * time.Hour },
		"missing origin":   func(c *Config) { c.OriginHost = "" },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := base()
			mutate(c)
			if err := c.validate(); err == nil {
				t.Fatalf("validate() = nil, want error for %s", name)
			}
		})
	}
	// The unmutated base must validate.
	if err := base().validate(); err != nil {
		t.Fatalf("base validate() = %v, want nil", err)
	}
}
