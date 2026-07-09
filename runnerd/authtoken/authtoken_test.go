package authtoken

import (
	"crypto/ed25519"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// mintEdDSA signs a token with priv exactly as the client's CreateJWT would
// (session.go:85-98): jwt.SigningMethodEdDSA over MapClaims.
func mintEdDSA(t *testing.T, priv ed25519.PrivateKey, claims jwt.MapClaims) string {
	t.Helper()
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	s, err := tok.SignedString(priv)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	return s
}

func genKeypair(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("genkey: %v", err)
	}
	return pub, priv
}

func TestLoadRawPublicKeyRoundTrip(t *testing.T) {
	pub, priv := genKeypair(t)
	dir := t.TempDir()
	pubPath := filepath.Join(dir, "jwt_ed25519.pub")
	// Matches auth.InitAndSave: raw key bytes written verbatim.
	if err := os.WriteFile(pubPath, []byte(pub), 0o644); err != nil {
		t.Fatal(err)
	}
	v, err := Load(pubPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok := mintEdDSA(t, priv, jwt.MapClaims{"sub": "client-cli"})
	sub, err := v.Verify(tok)
	if err != nil {
		t.Fatalf("Verify valid token: %v", err)
	}
	if sub != "client-cli" {
		t.Fatalf("sub = %q, want client-cli", sub)
	}
}

func TestLoadRejectsEmptyPath(t *testing.T) {
	if _, err := Load(""); err == nil {
		t.Fatal("Load(\"\") should error")
	}
}

func TestLoadRejectsWrongSizedKey(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.pub")
	if err := os.WriteFile(p, []byte("too-short"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err == nil {
		t.Fatal("Load should reject a wrong-sized key")
	}
}

func TestVerifyRejectsWrongKey(t *testing.T) {
	pubA, _ := genKeypair(t)
	_, privB := genKeypair(t)
	v := NewVerifier(pubA)
	// Signed by B, verified against A: must fail.
	tok := mintEdDSA(t, privB, jwt.MapClaims{"sub": "x"})
	if _, err := v.Verify(tok); err == nil {
		t.Fatal("Verify should reject a token signed by a different key")
	}
}

func TestVerifyRejectsExpired(t *testing.T) {
	pub, priv := genKeypair(t)
	v := NewVerifier(pub)
	tok := mintEdDSA(t, priv, jwt.MapClaims{
		"sub": "x",
		"exp": time.Now().Add(-time.Minute).Unix(),
	})
	if _, err := v.Verify(tok); err == nil {
		t.Fatal("Verify should reject an expired token")
	}
}

func TestVerifyRejectsMissingSub(t *testing.T) {
	pub, priv := genKeypair(t)
	v := NewVerifier(pub)
	tok := mintEdDSA(t, priv, jwt.MapClaims{"role": "admin"})
	if _, err := v.Verify(tok); err == nil {
		t.Fatal("Verify should reject a token with no sub claim")
	}
}

func TestVerifyRejectsWrongAlg(t *testing.T) {
	pub, _ := genKeypair(t)
	v := NewVerifier(pub)
	// HS256 token: the signing-method pin must reject it (alg-confusion guard).
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{"sub": "x"})
	s, err := tok.SignedString([]byte("shared-secret"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := v.Verify(s); err == nil {
		t.Fatal("Verify should reject a non-EdDSA token")
	}
}

func TestVerifyRejectsGarbage(t *testing.T) {
	pub, _ := genKeypair(t)
	v := NewVerifier(pub)
	if _, err := v.Verify("not-a-jwt"); err == nil {
		t.Fatal("Verify should reject a malformed token")
	}
}
