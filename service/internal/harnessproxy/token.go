package harnessproxy

import (
	"bytes"
	"crypto/ed25519"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// LoadEd25519PrivateKey loads an ed25519 private key from a host-local path,
// accepting the same three shapes as the Python client's
// load_ed25519_private_key so the dashboard and CLI read the identical key file:
//   - PEM (PKCS8 "-----BEGIN ... KEY-----")
//   - 32 raw bytes (the ed25519 seed)
//   - 64 raw bytes (Go ed25519.PrivateKey layout: seed||public)
//
// The key bytes never leave this process.
func LoadEd25519PrivateKey(path string) (ed25519.PrivateKey, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("harnessproxy: read ed25519 key %s: %w", path, err)
	}
	if bytes.HasPrefix(data, []byte("-----BEGIN ")) {
		block, _ := pem.Decode(data)
		if block == nil {
			return nil, fmt.Errorf("harnessproxy: %s looks like PEM but no block decoded", path)
		}
		key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
		if err != nil {
			return nil, fmt.Errorf("harnessproxy: parse PKCS8 key %s: %w", path, err)
		}
		ed, ok := key.(ed25519.PrivateKey)
		if !ok {
			return nil, fmt.Errorf("harnessproxy: key %s is not ed25519 (%T)", path, key)
		}
		return ed, nil
	}
	switch len(data) {
	case ed25519.SeedSize: // 32: seed only
		return ed25519.NewKeyFromSeed(data), nil
	case ed25519.PrivateKeySize: // 64: seed||public (Go layout)
		return ed25519.PrivateKey(data), nil
	}
	return nil, fmt.Errorf("harnessproxy: unrecognized ed25519 key at %s: expected PEM, 32, or 64 raw bytes, got %d",
		path, len(data))
}

// MintToken mints a short-lived EdDSA JWT for one control-plane call. Claims mirror
// transport.mint_token exactly: sub, aud="cambia-runnerd", iat, nbf=iat-30s,
// exp=iat+ttl. The subject is passed in (the dashboard supplies DashboardSubject).
// now is injectable for tests.
func MintToken(key ed25519.PrivateKey, subject string, ttl time.Duration, now time.Time) (string, error) {
	if subject == "" {
		return "", errors.New("harnessproxy: JWT subject (sub) must be non-empty")
	}
	if ttl <= 0 || ttl > MaxTokenTTL {
		return "", fmt.Errorf("harnessproxy: token ttl must be in (0, %d], got %v", int(MaxTokenTTL.Seconds()), ttl)
	}
	claims := jwt.MapClaims{
		"sub": subject,
		"aud": TokenAudience,
		"iat": now.Unix(),
		"nbf": now.Add(-NBFBackdate).Unix(),
		"exp": now.Add(ttl).Unix(),
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	signed, err := tok.SignedString(key)
	if err != nil {
		return "", fmt.Errorf("harnessproxy: sign token: %w", err)
	}
	return signed, nil
}
