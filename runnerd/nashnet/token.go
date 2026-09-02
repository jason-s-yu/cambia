package nashnet

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"time"
)

// leaseTokenBytes is the mint size of a lease token: 32 crypto/rand bytes (D44).
const leaseTokenBytes = 32

// ErrTokenMint is returned when the entropy source fails. It is never a client
// error: a claim that cannot mint a token is refused rather than answered with
// a weak one.
var ErrTokenMint = errors.New("nashnet: lease token mint failed")

// MintLeaseToken returns a new lease token as base64url text plus its hex
// sha256 (D44). The coordinator returns the token once, in the claim response,
// and stores only the hash. Passing a nil reader uses crypto/rand.
//
// The hash is taken over the base64url text exactly as returned, not over the
// decoded bytes, so a node that stored the token string and later hashes it for
// the live_leases re-bind of D3 reproduces the coordinator's value with no
// decode step and no ambiguity about which encoding was hashed.
func MintLeaseToken(entropy io.Reader) (token, hash string, err error) {
	if entropy == nil {
		entropy = rand.Reader
	}
	buf := make([]byte, leaseTokenBytes)
	if _, err := io.ReadFull(entropy, buf); err != nil {
		return "", "", fmt.Errorf("%w: %v", ErrTokenMint, err)
	}
	token = base64.RawURLEncoding.EncodeToString(buf)
	return token, HashLeaseToken(token), nil
}

// HashLeaseToken returns the hex sha256 of a lease token as the lease record
// stores it (D44). It is defined on the empty string too, so callers never have
// to special-case a dropped token; TokenMatches rejects an empty stored hash
// before any compare runs.
func HashLeaseToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// TokenMatches reports whether the presented token hashes to storedHash, using
// crypto/subtle.ConstantTimeCompare so the compare leaks no prefix information
// to a node guessing another lease's token (D44). An empty stored hash is a
// dropped token (expired lease, bumped epoch, revoked node) and never matches,
// including against an empty presented token.
func TokenMatches(storedHash, token string) bool {
	if storedHash == "" || token == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(storedHash), []byte(HashLeaseToken(token))) == 1
}

// HashMatches reports whether two token hashes are equal under the same
// constant-time compare. It is the live_leases half of D3, where the node
// presents a hash rather than the token itself.
func HashMatches(storedHash, presentedHash string) bool {
	if storedHash == "" || presentedHash == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(storedHash), []byte(presentedHash)) == 1
}

// crockford is the ULID alphabet (Crockford base32, excluding I, L, O, and U).
const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// NewLeaseID returns a fresh lease id: a 26-character ULID over a 48-bit
// millisecond timestamp and 80 bits of randomness, as the claim response of D2
// shows. The encoding keeps ids lexically sortable by grant time, which is what
// makes a lease listing readable in the order leases were handed out, and
// satisfies the run-name allowlist shape so a lease id is safe as a path
// segment and a map key. Passing a nil reader uses crypto/rand.
func NewLeaseID(now time.Time, entropy io.Reader) (string, error) {
	if entropy == nil {
		entropy = rand.Reader
	}
	var raw [16]byte
	ms := uint64(now.UTC().UnixMilli())
	var tsBuf [8]byte
	binary.BigEndian.PutUint64(tsBuf[:], ms)
	copy(raw[0:6], tsBuf[2:8])
	if _, err := io.ReadFull(entropy, raw[6:]); err != nil {
		return "", fmt.Errorf("%w: %v", ErrTokenMint, err)
	}
	return encodeULID(raw), nil
}

// encodeULID renders the 128-bit value as 26 Crockford base32 characters, most
// significant bits first. The leading character carries only the top 3 bits,
// since 26 characters hold 130 bits.
func encodeULID(raw [16]byte) string {
	hi := binary.BigEndian.Uint64(raw[0:8])
	lo := binary.BigEndian.Uint64(raw[8:16])
	out := make([]byte, 26)
	for i := 0; i < 26; i++ {
		shift := uint(125 - 5*i)
		out[i] = crockford[shiftRight128(hi, lo, shift)&0x1f]
	}
	return string(out)
}

// shiftRight128 returns the low 64 bits of the 128-bit value (hi, lo) shifted
// right by s.
func shiftRight128(hi, lo uint64, s uint) uint64 {
	switch {
	case s == 0:
		return lo
	case s < 64:
		return (hi << (64 - s)) | (lo >> s)
	case s == 64:
		return hi
	case s < 128:
		return hi >> (s - 64)
	default:
		return 0
	}
}
