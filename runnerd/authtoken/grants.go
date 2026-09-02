package authtoken

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Grant-store failures. Each is typed so the middleware can map it without
// matching error text; HTTPStatus turns all of them into 401 unauthorized, so
// an unenrolled id and a revoked one are indistinguishable to a caller.
var (
	// ErrUnknownNode is returned when no grant file exists for a node id. A
	// removed grant file is an identity that dies at the next token, bounded by
	// the keyset cache TTL (D60).
	ErrUnknownNode = errors.New("no enrollment grant for node")
	// ErrRevoked is returned when a tombstone exists for a node id. It is
	// consulted on every verification, not at load, so a revoked node is refused
	// on its next call rather than at the next cache refresh (D60).
	ErrRevoked = errors.New("node credential revoked")
	// ErrGrantExpired is returned when a grant's own exp has passed. It is
	// re-checked at every verification, so a cached grant cannot outlive it.
	ErrGrantExpired = errors.New("enrollment grant expired")
	// ErrGrantInvalid is returned when a grant file is unsigned, signed by
	// another key, malformed, or names an id that is not the one derived from
	// its own node public key.
	ErrGrantInvalid = errors.New("invalid enrollment grant")
)

const (
	// GrantFileSuffix and TombstoneFileSuffix are the two file names a node id
	// maps to inside the nodes directory. The operator writes both; the
	// coordinator only reads them, because it holds no signing key (D60).
	GrantFileSuffix     = ".grant"
	TombstoneFileSuffix = ".revoked"

	// NodeIDPrefix and nodeIDHexLen define the derived node id:
	// "n-" + sha256(node public key) truncated to 12 hex characters, which
	// satisfies procmgr's run-name allowlist and makes id squatting impossible,
	// since the id is a function of the key the grant carries (D60).
	NodeIDPrefix = "n-"
	nodeIDHexLen = 12

	// DefaultKeysetTTL is how long a loaded grant is reused before the file is
	// read again (D60). It bounds how long a removed or replaced grant keeps
	// working; the tombstone path is immediate and is not cached.
	DefaultKeysetTTL = 30 * time.Second

	// DefaultGrantLifetime is the default grant validity, matching the default
	// of the client's mint (cfr/src/harness/nashnet.py). It is also the default
	// ceiling GrantStoreConfig.MaxGrantLifetime enforces at load, so a longer
	// grant is a deliberate operator setting on both sides rather than an
	// accident on one. Worst case for a stolen node key is one grant lifetime
	// of use unless the operator revokes sooner (D60).
	DefaultGrantLifetime = 90 * 24 * time.Hour
)

// Grant is a verified enrollment grant: the node's identity, its key, its caps,
// and its validity window. Caps is capability.Grant (D60's clamp set, D47's
// bound on every declaration field placement reads): a nil or empty field
// means the cap is unset, the enrolling operator's explicit choice to let the
// declared value pass unclamped. capability is the leaf package that also
// applies the clamp (capability.Clamp), so this package imports it rather than
// keeping a second, duplicate claim type.
type Grant struct {
	NodeID    string
	PublicKey ed25519.PublicKey
	Caps      capability.Grant
	IssuedAt  time.Time
	ExpiresAt time.Time
}

// DeriveNodeID computes the node id a public key is entitled to. The binding is
// intrinsic: a grant file whose name is not this id is refused at load, so an
// operator cannot enroll one key under another node's id and a node cannot pick
// its own id (D60).
func DeriveNodeID(pub ed25519.PublicKey) (string, error) {
	if len(pub) != ed25519.PublicKeySize {
		return "", fmt.Errorf("%w: node public key has %d bytes, want %d",
			ErrGrantInvalid, len(pub), ed25519.PublicKeySize)
	}
	sum := sha256.Sum256(pub)
	return NodeIDPrefix + hex.EncodeToString(sum[:])[:nodeIDHexLen], nil
}

// GrantStoreConfig configures a GrantStore. Dir and OperatorKey are required.
type GrantStoreConfig struct {
	// Dir is the nodes directory holding <node_id>.grant and <node_id>.revoked.
	Dir string
	// OperatorKey is the public half of the client key that mints grants, the
	// same key operator tokens are verified against. The coordinator holds no
	// private half of anything and cannot author a grant.
	OperatorKey ed25519.PublicKey
	// KeysetTTL is the grant cache lifetime. Zero means DefaultKeysetTTL.
	KeysetTTL time.Duration
	// MaxGrantLifetime bounds exp - iat of a grant at load. Zero means
	// DefaultGrantLifetime; raise it to accept longer grants.
	MaxGrantLifetime time.Duration
	// Now is the injected clock. Nil means time.Now.
	Now func() time.Time
}

type cachedGrant struct {
	grant    *Grant
	loadedAt time.Time
}

// GrantStore resolves node keys from operator-signed grant files, caching each
// verified grant for KeysetTTL. It is safe for concurrent use. Only successful
// loads are cached: a miss stays a cheap file open, so a flood of tokens naming
// unenrolled ids cannot grow the map.
type GrantStore struct {
	dir string
	// grantVerifier verifies grant files; nodeVerifier verifies the call tokens
	// the granted keys sign. Both are built once: a node verifies on every call.
	grantVerifier *Verifier
	nodeVerifier  *Verifier
	ttl           time.Duration
	now           func() time.Time

	mu    sync.Mutex
	cache map[string]cachedGrant
}

// NewGrantStore builds a store over the nodes directory.
func NewGrantStore(cfg GrantStoreConfig) (*GrantStore, error) {
	if cfg.Dir == "" {
		return nil, errors.New("nashnet nodes directory is not configured")
	}
	if len(cfg.OperatorKey) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("operator public key has %d bytes, want %d (raw ed25519)",
			len(cfg.OperatorKey), ed25519.PublicKeySize)
	}
	ttl := cfg.KeysetTTL
	if ttl <= 0 {
		ttl = DefaultKeysetTTL
	}
	life := cfg.MaxGrantLifetime
	if life <= 0 {
		life = DefaultGrantLifetime
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	s := &GrantStore{
		dir: cfg.Dir,
		grantVerifier: &Verifier{
			pub:         cfg.OperatorKey,
			audience:    AudienceEnroll,
			nodeSubject: true,
			maxLifetime: life,
			now:         now,
		},
		ttl:   ttl,
		now:   now,
		cache: map[string]cachedGrant{},
	}
	s.nodeVerifier = NewNodeVerifier(s.KeyResolver(), now)
	return s, nil
}

// Grant returns the verified grant for nodeID. It consults the tombstone and
// re-checks the grant's own exp on every call, so revocation and expiry both
// take effect at the next verification rather than at the next cache refresh.
func (s *GrantStore) Grant(nodeID string) (*Grant, error) {
	if err := procmgr.ValidateName(nodeID); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidSubject, err)
	}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.revoked(nodeID) {
		delete(s.cache, nodeID)
		return nil, fmt.Errorf("%w: %s", ErrRevoked, nodeID)
	}
	entry, ok := s.cache[nodeID]
	if !ok || now.Sub(entry.loadedAt) >= s.ttl {
		g, err := s.load(nodeID)
		if err != nil {
			delete(s.cache, nodeID)
			return nil, err
		}
		entry = cachedGrant{grant: g, loadedAt: now}
		s.cache[nodeID] = entry
	}
	// The same skew leeway the JWT validation applies, so a cached grant and a
	// freshly loaded one expire at the same moment rather than one instant apart.
	if !entry.grant.ExpiresAt.After(now.Add(-ClockSkewLeeway)) {
		delete(s.cache, nodeID)
		return nil, fmt.Errorf("%w: %s expired at %s", ErrGrantExpired, nodeID,
			entry.grant.ExpiresAt.UTC().Format(time.RFC3339))
	}
	return entry.grant, nil
}

// KeyResolver returns the resolver the node verifier uses: subject to node id
// to granted key.
func (s *GrantStore) KeyResolver() KeyResolver {
	return func(sub string) (ed25519.PublicKey, error) {
		nodeID, err := NodeIDFromSubject(sub)
		if err != nil {
			return nil, err
		}
		g, err := s.Grant(nodeID)
		if err != nil {
			return nil, err
		}
		return g.PublicKey, nil
	}
}

// NodeVerifier returns the AudienceNode verifier backed by this store.
func (s *GrantStore) NodeVerifier() *Verifier {
	return s.nodeVerifier
}

// VerifyNode verifies a node token and returns the node id its subject names
// together with the token claims, so a route handler derives the acting node
// from the verified subject and never from a request body (D25).
func (s *GrantStore) VerifyNode(tokenString string) (string, jwt.MapClaims, error) {
	sub, claims, err := s.nodeVerifier.VerifyClaims(tokenString)
	if err != nil {
		return "", nil, err
	}
	nodeID, err := NodeIDFromSubject(sub)
	if err != nil {
		return "", nil, err
	}
	return nodeID, claims, nil
}

// revoked reports whether a tombstone exists for nodeID. Callers hold s.mu.
func (s *GrantStore) revoked(nodeID string) bool {
	_, err := os.Stat(filepath.Join(s.dir, nodeID+TombstoneFileSuffix))
	return err == nil
}

// load reads and verifies <node_id>.grant. Callers hold s.mu.
func (s *GrantStore) load(nodeID string) (*Grant, error) {
	path := filepath.Join(s.dir, nodeID+GrantFileSuffix)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("%w: %s", ErrUnknownNode, nodeID)
		}
		return nil, fmt.Errorf("read enrollment grant %q: %w", path, err)
	}
	g, err := parseGrant(string(data), s.grantVerifier)
	if err != nil {
		return nil, fmt.Errorf("enrollment grant %q: %w", path, err)
	}
	if g.NodeID != nodeID {
		return nil, fmt.Errorf("%w: %q holds the grant for %s", ErrGrantInvalid, path, g.NodeID)
	}
	return g, nil
}

// parseGrant verifies a compact grant JWS against v (an AudienceEnroll
// verifier) and returns the grant it carries. An unsigned, foreign-signed,
// expired, or over-long grant is refused by the verifier; the id derived from
// the grant's own node_pubkey must equal the id its subject names.
func parseGrant(compact string, v *Verifier) (*Grant, error) {
	sub, claims, err := v.VerifyClaims(strings.TrimSpace(compact))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrGrantInvalid, err)
	}
	subjectID, err := NodeIDFromSubject(sub)
	if err != nil {
		return nil, err
	}
	raw, _ := claims["node_pubkey"].(string)
	if raw == "" {
		return nil, fmt.Errorf("%w: no node_pubkey claim", ErrGrantInvalid)
	}
	key, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(raw, "="))
	if err != nil {
		return nil, fmt.Errorf("%w: node_pubkey is not base64url: %v", ErrGrantInvalid, err)
	}
	derived, err := DeriveNodeID(key)
	if err != nil {
		return nil, err
	}
	if derived != subjectID {
		return nil, fmt.Errorf("%w: subject names %s, key derives %s",
			ErrGrantInvalid, subjectID, derived)
	}
	caps, err := parseCaps(claims["caps"])
	if err != nil {
		return nil, err
	}
	iat, _ := claims.GetIssuedAt()
	exp, _ := claims.GetExpirationTime()
	return &Grant{
		NodeID:    derived,
		PublicKey: ed25519.PublicKey(key),
		Caps:      caps,
		IssuedAt:  iat.Time,
		ExpiresAt: exp.Time,
	}, nil
}

// parseCaps decodes the caps claim strictly: an unrecognized key is refused
// rather than ignored, so a misspelled cap is a load failure instead of a
// silently unclamped field.
func parseCaps(v any) (capability.Grant, error) {
	var caps capability.Grant
	if v == nil {
		return caps, nil
	}
	buf, err := json.Marshal(v)
	if err != nil {
		return caps, fmt.Errorf("%w: caps: %v", ErrGrantInvalid, err)
	}
	dec := json.NewDecoder(bytes.NewReader(buf))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&caps); err != nil {
		return caps, fmt.Errorf("%w: caps: %v", ErrGrantInvalid, err)
	}
	return caps, nil
}
