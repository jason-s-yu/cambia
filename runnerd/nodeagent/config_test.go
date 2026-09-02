package nodeagent

import (
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
	"github.com/jason-s-yu/cambia/runnerd/authtoken"
)

// writeNodeConfig writes a node yaml and returns its path.
func writeNodeConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "node.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

const validFingerprint = "3b1f2a7c9d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5061728394a5b6c7d8e"

// TestConfigRefusesPlaintextCoordinator covers AC8's second half: an http://
// coordinator url is refused at config load, because an agent that sets
// InsecureSkipVerify without the fingerprint compare authenticates nothing.
func TestConfigRefusesPlaintextCoordinator(t *testing.T) {
	path := writeNodeConfig(t, `
nashnet:
  coordinator:
    url: http://coordinator.example:8090
    cert_sha256: `+validFingerprint+`
  base_dir: /tmp/node
`)
	_, err := LoadConfig(path)
	if !errors.Is(err, ErrInsecureCoordinator) {
		t.Fatalf("LoadConfig error = %v, want ErrInsecureCoordinator", err)
	}
}

// TestConfigLoadsNashnetSection checks the shape the design's D46 example
// shows: paths expanded and absolute, gates parsed, defaults applied.
func TestConfigLoadsNashnetSection(t *testing.T) {
	base := t.TempDir()
	path := writeNodeConfig(t, `
nashnet:
  coordinator:
    url: https://coordinator.example:8090/
    cert_sha256: "sha256:`+strings.ToUpper(validFingerprint)+`"
  key_path: `+base+`/nashnet-node.key
  base_dir: `+base+`
  runs_dir: `+base+`/runs
  caches: {max_venvs: 4, max_libcambia: 20}
  slots: 2
  gates:
    drain: {file: `+base+`/DRAIN, on_breach: drain}
    concurrency: {max_slots: 2, max_accelerator_jobs: 1}
    kinds_allowed: [train, evaluate, measure]
    devices_allowed: [cpu, "cuda:0"]
    floors: {free_ram_gb: 8, free_disk_gb: 20, free_vram_gb: {"cuda:0": 12}, on_breach: finish}
    job_policy: {max_runtime_hours: 48}
`)
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Coordinator.URL != "https://coordinator.example:8090" {
		t.Errorf("coordinator url = %q", cfg.Coordinator.URL)
	}
	if cfg.Coordinator.CertSHA256 != validFingerprint {
		t.Errorf("fingerprint = %q, want the lowercase hex with no prefix", cfg.Coordinator.CertSHA256)
	}
	if cfg.Slots != 2 || cfg.ClaimWaitSeconds != 25 {
		t.Errorf("defaults not applied: slots=%d wait=%d", cfg.Slots, cfg.ClaimWaitSeconds)
	}
	if len(cfg.Kinds) != len(defaultKinds) {
		t.Errorf("kinds = %v, want the default set", cfg.Kinds)
	}
	if cfg.Gates.Floors == nil || cfg.Gates.Floors.FreeVRAMGB["cuda:0"] != 12 {
		t.Errorf("floors gate did not parse: %+v", cfg.Gates.Floors)
	}
	if cfg.Gates.Concurrency == nil || cfg.Gates.Concurrency.MaxSlots == nil || *cfg.Gates.Concurrency.MaxSlots != 2 {
		t.Errorf("concurrency gate did not parse: %+v", cfg.Gates.Concurrency)
	}
	if len(cfg.Gates.DevicesAllowed) != 2 {
		t.Errorf("devices_allowed = %v", cfg.Gates.DevicesAllowed)
	}
	if !filepath.IsAbs(cfg.Gates.Drain.File) {
		t.Errorf("drain file %q was not made absolute", cfg.Gates.Drain.File)
	}
}

// TestConfigRejectsBadFingerprint keeps the pin from silently degrading to no
// pin at all.
func TestConfigRejectsBadFingerprint(t *testing.T) {
	for _, fp := range []string{"", "deadbeef", strings.Repeat("z", 64)} {
		path := writeNodeConfig(t, `
nashnet:
  coordinator: {url: "https://c.example:8090", cert_sha256: "`+fp+`"}
  base_dir: /tmp/node
`)
		if _, err := LoadConfig(path); err == nil {
			t.Errorf("fingerprint %q was accepted", fp)
		}
	}
}

// TestLoadOrCreateKeyDerivesNodeID checks the key file contract: 0600, raw
// ed25519 bytes, and an id intrinsic to the key (D60).
func TestLoadOrCreateKeyDerivesNodeID(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys", "node.key")
	priv, created, err := LoadOrCreateKey(path)
	if err != nil || !created {
		t.Fatalf("LoadOrCreateKey: created=%v err=%v", created, err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Errorf("key mode = %#o, want 0600", fi.Mode().Perm())
	}
	again, created, err := LoadOrCreateKey(path)
	if err != nil || created {
		t.Fatalf("second load: created=%v err=%v", created, err)
	}
	if string(again) != string(priv) {
		t.Errorf("second load returned a different key")
	}

	signer, err := NewSigner(priv, nil)
	if err != nil {
		t.Fatal(err)
	}
	pub, _ := priv.Public().(ed25519.PublicKey)
	want, err := authtoken.DeriveNodeID(pub)
	if err != nil {
		t.Fatal(err)
	}
	if signer.NodeID() != want {
		t.Errorf("node id = %q, want the key-derived %q", signer.NodeID(), want)
	}
}

// TestSignerMintsNodeAudienceToken checks the three claims the node verifier
// adds beyond the landed operator Verify: the aud, the node:<id> sub shape,
// and both exp and iat inside the 300s cap (D25).
func TestSignerMintsNodeAudienceToken(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	signer, err := NewSigner(priv, nil)
	if err != nil {
		t.Fatal(err)
	}
	tokenStr, err := signer.Token()
	if err != nil {
		t.Fatal(err)
	}
	verifier := authtoken.NewNodeVerifier(func(string) (ed25519.PublicKey, error) {
		return priv.Public().(ed25519.PublicKey), nil
	}, nil)
	sub, claims, err := verifier.VerifyClaims(tokenStr)
	if err != nil {
		t.Fatalf("the node verifier refused a minted token: %v", err)
	}
	if sub != authtoken.SubjectForNode(signer.NodeID()) {
		t.Errorf("sub = %q, want %q", sub, authtoken.SubjectForNode(signer.NodeID()))
	}
	exp, err := claims.GetExpirationTime()
	if err != nil || exp == nil {
		t.Fatalf("token carried no exp: %v", err)
	}
	iat, err := claims.GetIssuedAt()
	if err != nil || iat == nil {
		t.Fatalf("token carried no iat: %v", err)
	}
	if life := exp.Time.Sub(iat.Time); life <= 0 || life > authtoken.MaxNodeTokenLifetime {
		t.Errorf("token lifetime %s is outside (0, %s]", life, authtoken.MaxNodeTokenLifetime)
	}
	if _, ok := jwt.GetSigningMethod("EdDSA").(*jwt.SigningMethodEd25519); !ok {
		t.Errorf("EdDSA is not the ed25519 method")
	}
}
