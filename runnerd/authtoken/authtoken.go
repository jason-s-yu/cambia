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
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// ErrNoPublicKey is returned by Load when the configured key path is empty.
var ErrNoPublicKey = errors.New("no JWT public key path configured")

// Typed verification failures. The control-plane middleware maps them to wire
// status codes through HTTPStatus rather than matching on error text.
var (
	// ErrWrongAudience is returned when a token's aud claim is not the audience
	// the verifier was built for: a node or enrollment credential presented on
	// an operator route, or an operator token presented on a node route (D26).
	ErrWrongAudience = errors.New("wrong token audience")
	// ErrInvalidSubject is returned when a node-audience token's sub claim is
	// absent, empty, or not node:<node_id> with a node id that is safe as a path
	// segment and a map key (D25).
	ErrInvalidSubject = errors.New("invalid token subject")
	// ErrTokenLifetime is returned when a node-audience token omits exp or iat,
	// or declares a lifetime past the audience's ceiling. golang-jwt validates
	// exp only when the claim is present, so a token minted without one would
	// otherwise be valid for as long as its key resolves (D25).
	ErrTokenLifetime = errors.New("token lifetime out of bounds")
)

// Audience is the required "aud" claim for a runnerd control-plane token. The
// client mints tokens carrying this audience; Verify rejects any token that omits it or
// carries a different one. This is the guard against key aliasing: if
// RUNNERD_JWT_PUBKEY is ever pointed at the same ed25519 key the service uses
// for user session JWTs, a user session token (audience-less, or a different
// audience) still cannot drive a job launch or cancel, because it does not carry
// aud == "cambia-runnerd".
const Audience = "cambia-runnerd"

// AudienceNode is the required "aud" claim for a token a nashnet node mints
// with its own enrolled keypair to drive its own leases (D25). It is a separate
// audience from Audience so the two credential classes cannot cross routes: the
// node verifier resolves its key per subject from an enrollment grant, and the
// operator verifier holds one static key, so neither accepts the other's token.
const AudienceNode = "nashnet-node"

// AudienceEnroll is the required "aud" claim for an enrollment grant, the
// operator-signed JWS that binds a node id to a node public key and carries its
// capability caps (D60). A grant is verified with the same operator key as an
// operator token, so the audience is what keeps a grant from being replayed as
// a control-plane token and the other way round.
const AudienceEnroll = "nashnet-enroll"

// MaxNodeTokenLifetime is the ceiling on exp - iat for an AudienceNode token.
// A node is an untrusted minter under MIN-TRUST, so the 300s TTL of D25 is
// enforced on the verifying side rather than assumed of the minter.
const MaxNodeTokenLifetime = 300 * time.Second

// NodeSubjectPrefix is the fixed prefix of a node token's sub claim. The
// subject is the whole of a node's identity: the coordinator derives node_id
// from it and ignores any node_id a request body carries (D25).
const NodeSubjectPrefix = "node:"

// ClockSkewLeeway is applied to exp/nbf/iat validation so a mint host and
// runner whose clocks disagree by a few seconds (WSL2 drift, container hosts)
// do not spuriously reject fresh tokens. The client mint additionally backdates
// nbf; the two guards are independent halves of the same skew tolerance.
const ClockSkewLeeway = 30 * time.Second

// KeyResolver returns the ed25519 public key that must have signed a token
// carrying subject sub. It is the node-audience half of the verifier: the
// operator key is static and configured once, while a node key comes from that
// node's enrollment grant and dies when the grant is removed, expires, or is
// tombstoned (D25, D60). A resolver failure fails the verification.
type KeyResolver func(sub string) (ed25519.PublicKey, error)

// Verifier verifies JWTs for exactly one audience. It is immutable after
// construction and safe for concurrent use. One Verify implementation serves
// all three audiences (D25): the operator audience verifies against a static
// key with no lifetime rule, and the two nashnet audiences resolve the key from
// the subject and carry the subject-shape and lifetime rules that an untrusted
// minter makes necessary.
type Verifier struct {
	pub      ed25519.PublicKey
	resolve  KeyResolver
	audience string
	// nodeSubject requires sub to be NodeSubjectPrefix + a valid node id.
	nodeSubject bool
	// maxLifetime, when positive, requires both exp and iat and bounds their
	// difference. Zero leaves golang-jwt's present-claims-only validation.
	maxLifetime time.Duration
	// now is the injected clock. Nil means time.Now.
	now func() time.Time
}

// Load reads the raw ed25519 public key bytes from pubPath and returns a
// Verifier. The on-disk format matches auth.InitAndSave / auth.InitFromPath: the
// raw 32-byte ed25519 public key written verbatim (not PEM/DER), so the client
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
	return NewVerifier(ed25519.PublicKey(data)), nil
}

// NewVerifier builds an operator-audience Verifier from an in-memory public key.
// Used by tests that generate an ephemeral keypair.
func NewVerifier(pub ed25519.PublicKey) *Verifier {
	return &Verifier{pub: pub, audience: Audience}
}

// NewNodeVerifier builds the AudienceNode verifier: the key comes from resolve,
// which reads the subject's enrollment grant, and the two checks this layer
// owns beyond the operator gate apply, subject shape and token lifetime (D25).
// The third, binding a request body's node id to the verified subject, belongs
// to the route handlers, which have the body. Subject shape is checked before
// the key is resolved, because the subject is a filesystem path segment on the
// way to the grant. now is the injected clock; nil means time.Now.
func NewNodeVerifier(resolve KeyResolver, now func() time.Time) *Verifier {
	return &Verifier{
		resolve:     resolve,
		audience:    AudienceNode,
		nodeSubject: true,
		maxLifetime: MaxNodeTokenLifetime,
		now:         now,
	}
}

// Verify parses and validates tokenString and returns its "sub" claim. It
// mirrors session.go:101-127 (signing method Ed25519, valid token, present
// string "sub") and adds one runnerd-specific requirement beyond the service
// gate: the token must carry the verifier's own audience, which for an operator
// verifier is Audience ("cambia-runnerd"). The audience is enforced twice, in
// the key function and through golang-jwt's own validator (jwt.WithAudience),
// which fails when the claim is absent or mismatched. This closes the
// key-aliasing hole: a user session JWT signed with the same ed25519 key must
// not be accepted here, and it will not be, because it does not carry this
// audience. Any deviation returns an error and no subject.
func (v *Verifier) Verify(tokenString string) (string, error) {
	sub, _, err := v.VerifyClaims(tokenString)
	return sub, err
}

// VerifyClaims is Verify plus the map claims Verify discards. Grant loading
// needs them (the node public key and the caps object live there), and the
// nashnet route middleware reads them for the node epoch, so the claims are
// returned rather than parsed a second time. The subject is returned verbatim,
// so a node token yields node:<node_id>; NodeIDFromSubject splits it.
func (v *Verifier) VerifyClaims(tokenString string) (string, jwt.MapClaims, error) {
	opts := []jwt.ParserOption{jwt.WithAudience(v.audience), jwt.WithLeeway(ClockSkewLeeway)}
	if v.now != nil {
		opts = append(opts, jwt.WithTimeFunc(v.now))
	}
	t, err := jwt.Parse(tokenString, v.keyFunc, opts...)
	if err != nil {
		if errors.Is(err, jwt.ErrTokenInvalidAudience) {
			return "", nil, fmt.Errorf("%w: want %q: %w", ErrWrongAudience, v.audience, err)
		}
		return "", nil, fmt.Errorf("jwt parse error: %w", err)
	}
	if !t.Valid {
		return "", nil, errors.New("invalid token")
	}
	claims, ok := t.Claims.(jwt.MapClaims)
	if !ok {
		return "", nil, errors.New("invalid jwt claims")
	}
	userID, ok := claims["sub"].(string)
	if !ok {
		return "", nil, errors.New("missing sub in jwt")
	}
	if v.nodeSubject {
		if _, err := NodeIDFromSubject(userID); err != nil {
			return "", nil, err
		}
	}
	if err := v.checkLifetime(claims); err != nil {
		return "", nil, err
	}
	return userID, claims, nil
}

// keyFunc pins the signing method, refuses a foreign audience before any key is
// resolved, and returns the key: the static one for the operator audience, the
// subject's granted one otherwise. The audience check sits here rather than
// only in jwt.WithAudience so that a mismatch is ErrWrongAudience for every
// audience and so that an operator token never reaches a node key lookup.
func (v *Verifier) keyFunc(t *jwt.Token) (interface{}, error) {
	if _, ok := t.Method.(*jwt.SigningMethodEd25519); !ok {
		return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
	}
	aud, err := t.Claims.GetAudience()
	if err != nil || !slices.Contains(aud, v.audience) {
		return nil, fmt.Errorf("%w: token carries %v, want %q", ErrWrongAudience, aud, v.audience)
	}
	if v.resolve == nil {
		return v.pub, nil
	}
	sub, err := t.Claims.GetSubject()
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidSubject, err)
	}
	if v.nodeSubject {
		if _, err := NodeIDFromSubject(sub); err != nil {
			return nil, err
		}
	}
	return v.resolve(sub)
}

// checkLifetime enforces the exp and iat rules of the nashnet audiences. It is
// a no-op for the operator audience, whose tokens are minted by the client and
// keep the validation they have always had.
func (v *Verifier) checkLifetime(claims jwt.MapClaims) error {
	if v.maxLifetime <= 0 {
		return nil
	}
	exp, err := claims.GetExpirationTime()
	if err != nil || exp == nil {
		return fmt.Errorf("%w: token carries no exp", ErrTokenLifetime)
	}
	iat, err := claims.GetIssuedAt()
	if err != nil || iat == nil {
		return fmt.Errorf("%w: token carries no iat", ErrTokenLifetime)
	}
	life := exp.Time.Sub(iat.Time)
	if life <= 0 || life > v.maxLifetime {
		return fmt.Errorf("%w: exp - iat is %s, want (0, %s]",
			ErrTokenLifetime, life, v.maxLifetime)
	}
	return nil
}

// NodeIDFromSubject validates a node-audience subject and returns its node id.
// The subject is the key-resolution input and a path segment on the way to the
// grant file, so the id must satisfy the same allowlist a run name does
// (procmgr.ValidateName, process.go:56). An empty subject and a subject that is
// not node:<node_id> are both refused.
func NodeIDFromSubject(sub string) (string, error) {
	id, ok := strings.CutPrefix(sub, NodeSubjectPrefix)
	if !ok {
		return "", fmt.Errorf("%w: %q is not %s<node_id>", ErrInvalidSubject, sub, NodeSubjectPrefix)
	}
	if err := procmgr.ValidateName(id); err != nil {
		return "", fmt.Errorf("%w: %v", ErrInvalidSubject, err)
	}
	return id, nil
}

// SubjectForNode is the inverse of NodeIDFromSubject, so a minter and a
// verifier cannot disagree on the subject format.
func SubjectForNode(nodeID string) string {
	return NodeSubjectPrefix + nodeID
}

// HTTPStatus maps a verification failure to the status and error code the
// control-plane middleware writes. An audience mismatch is 403 wrong_audience
// (D26): the credential is real, the route is not its route. Everything else is
// 401 unauthorized, including a revoked or unknown node, so a caller cannot use
// the status code to learn which node ids are enrolled.
func HTTPStatus(err error) (int, string) {
	if err == nil {
		return http.StatusOK, ""
	}
	if errors.Is(err, ErrWrongAudience) {
		return http.StatusForbidden, "wrong_audience"
	}
	return http.StatusUnauthorized, "unauthorized"
}
