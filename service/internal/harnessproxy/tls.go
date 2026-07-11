package harnessproxy

import (
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"fmt"
	"strings"
)

// ErrPinMismatch is returned through a TLS handshake when the peer's leaf cert
// does not match the pinned fingerprint. Callers detect it with errors.Is to
// map a pin mismatch to a distinct failure (never fall through to an unpinned
// connection). Mirrors transport.CertificatePinError.
var ErrPinMismatch = fmt.Errorf("harnessproxy: peer cert fingerprint does not match pin")

// NormalizeFingerprint lowercases hex and strips colons, validating a 64-char
// SHA256 hex string. Mirrors transport.normalize_fingerprint.
func NormalizeFingerprint(fp string) (string, error) {
	norm := strings.ToLower(strings.TrimSpace(strings.ReplaceAll(fp, ":", "")))
	if len(norm) != 64 {
		return "", fmt.Errorf("cert fingerprint must be 64 hex chars, got %d", len(norm))
	}
	for _, c := range norm {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return "", fmt.Errorf("cert fingerprint must be SHA256 hex, got %q", fp)
		}
	}
	return norm, nil
}

// pinnedTLSConfig returns a tls.Config whose sole trust anchor is a SHA256
// fingerprint pin on the peer's leaf cert (design 5.1). CA-chain and hostname
// verification are disabled (the runner serves a self-signed cert); the manual
// VerifyPeerCertificate computes sha256 over rawCerts[0] (the leaf DER) and
// constant-time-compares it to the pin. A mismatch, or an empty cert list,
// aborts the handshake with ErrPinMismatch before any application bytes (the
// Bearer token) are written. Pinning one self-signed cert is tighter than
// trusting a CA that could issue others.
func pinnedTLSConfig(fingerprint string) (*tls.Config, error) {
	pinned, err := NormalizeFingerprint(fingerprint)
	if err != nil {
		return nil, err
	}
	return &tls.Config{
		// InsecureSkipVerify disables the default CA/hostname chain so the
		// fingerprint pin below is the only trust decision. This is not an
		// unverified connection: VerifyPeerCertificate rejects any cert whose
		// fingerprint is not the pinned one.
		InsecureSkipVerify: true, //nolint:gosec // pinned by VerifyPeerCertificate
		VerifyPeerCertificate: func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
			if len(rawCerts) == 0 {
				return ErrPinMismatch
			}
			sum := sha256.Sum256(rawCerts[0])
			actual := hex.EncodeToString(sum[:])
			if subtle.ConstantTimeCompare([]byte(actual), []byte(pinned)) != 1 {
				return fmt.Errorf("%w: peer %s pinned %s", ErrPinMismatch, actual, pinned)
			}
			return nil
		},
	}, nil
}
