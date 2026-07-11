package harnessproxy

import (
	"crypto/ed25519"
	"crypto/x509"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/jason-s-yu/cambia/runnerd/authtoken"
)

func TestMintTokenClaims(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1_700_000_000, 0).UTC()
	ttl := 900 * time.Second
	tok, err := MintToken(priv, DashboardSubject, ttl, now)
	if err != nil {
		t.Fatalf("MintToken: %v", err)
	}

	claims := jwt.MapClaims{}
	parsed, err := jwt.ParseWithClaims(tok, claims, func(*jwt.Token) (interface{}, error) {
		return pub, nil
	}, jwt.WithTimeFunc(func() time.Time { return now.Add(time.Second) }))
	if err != nil || !parsed.Valid {
		t.Fatalf("parse minted token: %v valid=%v", err, parsed.Valid)
	}
	if claims["sub"] != DashboardSubject {
		t.Errorf("sub = %v, want %s", claims["sub"], DashboardSubject)
	}
	if claims["aud"] != TokenAudience {
		t.Errorf("aud = %v, want %s", claims["aud"], TokenAudience)
	}
	iat := int64(claims["iat"].(float64))
	nbf := int64(claims["nbf"].(float64))
	exp := int64(claims["exp"].(float64))
	if iat != now.Unix() {
		t.Errorf("iat = %d, want %d", iat, now.Unix())
	}
	if nbf != now.Add(-NBFBackdate).Unix() {
		t.Errorf("nbf = %d, want %d (iat - 30s backdate)", nbf, now.Add(-NBFBackdate).Unix())
	}
	if exp != now.Add(ttl).Unix() {
		t.Errorf("exp = %d, want %d (iat + ttl)", exp, now.Add(ttl).Unix())
	}
}

// TestMintedTokenVerifiesAgainstRunnerd is the end-to-end guard: a token minted
// here must be accepted by runnerd's real verify-only gate, which requires
// aud == cambia-runnerd and a present string sub. This proves the audience
// binding and signing method line up with the runner.
func TestMintedTokenVerifiesAgainstRunnerd(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := MintToken(priv, DashboardSubject, 900*time.Second, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	sub, err := authtoken.NewVerifier(pub).Verify(tok)
	if err != nil {
		t.Fatalf("runnerd verifier rejected a dashboard-minted token: %v", err)
	}
	if sub != DashboardSubject {
		t.Errorf("verified sub = %q, want %s", sub, DashboardSubject)
	}
}

func TestMintTokenRejects(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(nil)
	now := time.Now().UTC()
	if _, err := MintToken(priv, "", 900*time.Second, now); err == nil {
		t.Error("empty subject must be rejected")
	}
	if _, err := MintToken(priv, "x", 0, now); err == nil {
		t.Error("zero ttl must be rejected")
	}
	if _, err := MintToken(priv, "x", 2*time.Hour, now); err == nil {
		t.Error("ttl over ceiling must be rejected")
	}
}

func TestLoadEd25519PrivateKeyShapes(t *testing.T) {
	_, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()

	// 64-byte Go layout (seed||public).
	raw64 := filepath.Join(dir, "raw64")
	if err := os.WriteFile(raw64, priv, 0o600); err != nil {
		t.Fatal(err)
	}
	// 32-byte seed.
	raw32 := filepath.Join(dir, "raw32")
	if err := os.WriteFile(raw32, priv.Seed(), 0o600); err != nil {
		t.Fatal(err)
	}
	// PEM (PKCS8).
	der, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	pemPath := filepath.Join(dir, "key.pem")
	if err := os.WriteFile(pemPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, p := range []string{raw64, raw32, pemPath} {
		got, err := LoadEd25519PrivateKey(p)
		if err != nil {
			t.Fatalf("LoadEd25519PrivateKey(%s): %v", filepath.Base(p), err)
		}
		if !got.Equal(priv) {
			t.Errorf("%s: loaded key does not equal the original", filepath.Base(p))
		}
	}

	// A wrong-length raw file is rejected.
	bad := filepath.Join(dir, "bad")
	if err := os.WriteFile(bad, []byte("tooshort"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEd25519PrivateKey(bad); err == nil {
		t.Error("an 8-byte key file must be rejected")
	}
}
