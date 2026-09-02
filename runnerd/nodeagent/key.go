package nodeagent

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
)

// AgentVersion is the agent_version a node declares (D9). It is the protocol
// revision the node speaks, not the runnerd build sha, which travels as
// build_commit on the coordinator's health route.
const AgentVersion = "1.1.0"

// keyFileMode is the permission a node key is created with and required to
// have (D25: 0600, never transmitted).
const keyFileMode = 0o600

// ErrKey wraps every node-key load failure.
var ErrKey = errors.New("nashnet node key")

// Signer holds the node's private half and mints the short-lived node-audience
// tokens every node route requires (D25). The coordinator holds no signing key
// of any kind, so this is the only signer in the protocol.
type Signer struct {
	nodeID string
	priv   ed25519.PrivateKey
	now    func() time.Time
	// lifetime is exp - iat on a minted token. The node verifier refuses a
	// token whose lifetime exceeds authtoken.MaxNodeTokenLifetime, so this is
	// clamped to that ceiling at construction.
	lifetime time.Duration
}

// LoadOrCreateKey reads the node's raw 64-byte ed25519 private key from path,
// creating it when absent. The on-disk form is the raw key bytes with no
// encoding wrapper, mirroring the raw-32-byte public key runnerd already reads
// for operator token verification (authtoken.Load), so the two halves of the
// same scheme are spelled the same way. created reports whether this call
// minted the key, which the caller logs alongside the derived node id so an
// operator can mint the enrollment grant (D60).
func LoadOrCreateKey(path string) (priv ed25519.PrivateKey, created bool, err error) {
	data, readErr := os.ReadFile(path)
	switch {
	case readErr == nil:
		if len(data) != ed25519.PrivateKeySize {
			return nil, false, fmt.Errorf("%w: %s holds %d bytes, want %d",
				ErrKey, path, len(data), ed25519.PrivateKeySize)
		}
		if fi, statErr := os.Stat(path); statErr == nil && fi.Mode().Perm()&0o077 != 0 {
			return nil, false, fmt.Errorf("%w: %s is mode %#o, want 0600", ErrKey, path, fi.Mode().Perm())
		}
		return ed25519.PrivateKey(data), false, nil
	case !os.IsNotExist(readErr):
		return nil, false, fmt.Errorf("%w: read %s: %v", ErrKey, path, readErr)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, false, fmt.Errorf("%w: %v", ErrKey, err)
	}
	_, generated, genErr := ed25519.GenerateKey(rand.Reader)
	if genErr != nil {
		return nil, false, fmt.Errorf("%w: generate: %v", ErrKey, genErr)
	}
	if err := os.WriteFile(path, generated, keyFileMode); err != nil {
		return nil, false, fmt.Errorf("%w: write %s: %v", ErrKey, path, err)
	}
	return generated, true, nil
}

// NewSigner derives the node id from the key and returns a token minter. The
// id is intrinsic to the key (D60), so a node cannot pick its own.
func NewSigner(priv ed25519.PrivateKey, now func() time.Time) (*Signer, error) {
	if len(priv) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("%w: private key has %d bytes, want %d",
			ErrKey, len(priv), ed25519.PrivateKeySize)
	}
	pub, ok := priv.Public().(ed25519.PublicKey)
	if !ok {
		return nil, fmt.Errorf("%w: key is not ed25519", ErrKey)
	}
	id, err := authtoken.DeriveNodeID(pub)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrKey, err)
	}
	if now == nil {
		now = time.Now
	}
	return &Signer{nodeID: id, priv: priv, now: now, lifetime: authtoken.MaxNodeTokenLifetime}, nil
}

// NodeID returns the derived node id.
func (s *Signer) NodeID() string { return s.nodeID }

// PublicKey returns the public half, which is what the coordinator's
// in-process grant for the embedded node is keyed on (D40).
func (s *Signer) PublicKey() ed25519.PublicKey {
	pub, _ := s.priv.Public().(ed25519.PublicKey)
	return pub
}

// PublicKeyBase64 renders the public half as the base64url the enrollment
// grant carries in its node_pubkey claim (D60).
func (s *Signer) PublicKeyBase64() string {
	pub, _ := s.priv.Public().(ed25519.PublicKey)
	return base64.RawURLEncoding.EncodeToString(pub)
}

// Token mints one node-audience bearer token: EdDSA, aud=nashnet-node,
// sub=node:<node_id>, with both iat and exp set and a lifetime inside the
// 300s cap the node verifier enforces (D25). A token is minted per call rather
// than cached, so a clock jump never leaves a stale credential in flight.
func (s *Signer) Token() (string, error) {
	now := s.now().UTC()
	claims := jwt.MapClaims{
		"aud": authtoken.AudienceNode,
		"sub": authtoken.SubjectForNode(s.nodeID),
		"iat": jwt.NewNumericDate(now),
		"exp": jwt.NewNumericDate(now.Add(s.lifetime)),
	}
	signed, err := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims).SignedString(s.priv)
	if err != nil {
		return "", fmt.Errorf("%w: sign: %v", ErrKey, err)
	}
	return signed, nil
}

// platformTag is the declaration's platform_tag (D9): the os-arch pair, the
// same shape the ingest pipeline stamps into a venv cache key.
func platformTag() string {
	return runtime.GOOS + "-" + runtime.GOARCH
}
