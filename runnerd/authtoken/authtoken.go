// Package authtoken is a verify-only JWT gate for runnerd's control plane. It
// loads an ed25519 PUBLIC key and verifies Bearer tokens minted on the client; it
// never signs and never holds a private key, so a compromised runner cannot
// forge its own tokens (design 5.2).
//
// Intentional verify-only duplication: the verification semantics mirror
// service/internal/auth/session.go:101-127 (AuthenticateJWT) exactly -- EdDSA
// signing-method pin, publicKey verification, token validity, and a string
// "sub" claim. runnerd is a separate Go module and MUST NOT import
// service/internal/... (cross-module internal imports are illegal), so the
// verification is re-implemented here rather than shared. The signing half
// (CreateJWT) is deliberately NOT reproduced: only the client mints tokens.
package authtoken

import (
	"crypto/ed25519"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// ErrNoPublicKey is returned by Load when the configured key path is empty.
var ErrNoPublicKey = errors.New("no JWT public key path configured")

// Audience is the required "aud" claim for a runnerd control-plane token. the client
// mints tokens carrying this audience; Verify rejects any token that omits it or
// carries a different one. This is the guard against key aliasing: if
// RUNNERD_JWT_PUBKEY is ever pointed at the same ed25519 key the service uses
// for user session JWTs, a user session token (audience-less, or a different
// audience) still cannot drive a job launch or cancel, because it does not carry
// aud == "cambia-runnerd".
const Audience = "cambia-runnerd"

// ClockSkewLeeway is applied to exp/nbf/iat validation so a mint host and
// runner whose clocks disagree by a few seconds (WSL2 drift, container hosts)
// do not spuriously reject fresh tokens. The client mint additionally backdates
// nbf; the two guards are independent halves of the same skew tolerance.
const ClockSkewLeeway = 30 * time.Second

// Verifier holds an ed25519 public key and verifies JWTs against it. It is
// immutable after Load and safe for concurrent use.
type Verifier struct {
	pub ed25519.PublicKey
}

// Load reads the raw ed25519 public key bytes from pubPath and returns a
// Verifier. The on-disk format matches auth.InitAndSave / auth.InitFromPath: the
// raw 32-byte ed25519 public key written verbatim (not PEM/DER), so the the client
// side and runnerd read the same file shape.
func Load(pubPath string) (*Verifier, error) {
	if pubPath == "" {
		return nil, ErrNoPublicKey
	}
	data, err := os.ReadFile(pubPath)
	if err != nil {
		return nil, fmt.Errorf("read JWT public key %q: %w", pubPath, err)
	}
	if len(data) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("JWT public key %q has %d bytes, want %d (raw ed25519)",
			pubPath, len(data), ed25519.PublicKeySize)
	}
	return &Verifier{pub: ed25519.PublicKey(data)}, nil
}

// NewVerifier builds a Verifier from an in-memory public key. Used by tests that
// generate an ephemeral keypair.
func NewVerifier(pub ed25519.PublicKey) *Verifier {
	return &Verifier{pub: pub}
}

// Verify parses and validates tokenString and returns its "sub" claim. It
// mirrors session.go:101-127 (signing method Ed25519, valid token, present
// string "sub") and adds one runnerd-specific requirement beyond the service
// gate: the token must carry aud == Audience ("cambia-runnerd"). The audience is
// enforced through golang-jwt's own validator (jwt.WithAudience), which fails
// when the claim is absent or mismatched. This closes the key-aliasing hole: a
// user session JWT signed with the same ed25519 key must not be accepted here,
// and it will not be, because it does not carry this audience. Any deviation
// returns an error and no subject.
func (v *Verifier) Verify(tokenString string) (string, error) {
	t, err := jwt.Parse(tokenString, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodEd25519); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return v.pub, nil
	}, jwt.WithAudience(Audience), jwt.WithLeeway(ClockSkewLeeway))
	if err != nil {
		return "", fmt.Errorf("jwt parse error: %w", err)
	}
	if !t.Valid {
		return "", errors.New("invalid token")
	}
	claims, ok := t.Claims.(jwt.MapClaims)
	if !ok {
		return "", errors.New("invalid jwt claims")
	}
	userID, ok := claims["sub"].(string)
	if !ok {
		return "", errors.New("missing sub in jwt")
	}
	return userID, nil
}
