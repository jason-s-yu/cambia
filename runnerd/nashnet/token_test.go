package nashnet

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMintLeaseTokenShapeAndHash(t *testing.T) {
	token, hash, err := MintLeaseToken(testEntropy())
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		t.Fatalf("token is not base64url: %v", err)
	}
	if len(raw) != leaseTokenBytes {
		t.Fatalf("token carries %d bytes, want %d", len(raw), leaseTokenBytes)
	}
	sum := sha256.Sum256([]byte(token))
	if hash != hex.EncodeToString(sum[:]) {
		t.Fatalf("hash is not the sha256 of the returned text")
	}
	if strings.Contains(hash, token) {
		t.Fatalf("the stored hash must not carry the token itself")
	}

	second, _, err := MintLeaseToken(testEntropy())
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	third, _, err := MintLeaseToken(nil)
	if err != nil {
		t.Fatalf("mint from crypto/rand: %v", err)
	}
	if second == third {
		t.Fatalf("two mints returned the same token")
	}
}

func TestMintLeaseTokenFailsClosedOnEntropyFailure(t *testing.T) {
	_, _, err := MintLeaseToken(failingReader{})
	if !errors.Is(err, ErrTokenMint) {
		t.Fatalf("a failed entropy read must refuse the mint, got %v", err)
	}
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("no entropy") }

func TestTokenMatchesRejectsEveryNonMatch(t *testing.T) {
	token, hash, err := MintLeaseToken(testEntropy())
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	if !TokenMatches(hash, token) {
		t.Fatalf("the minted token must match its own hash")
	}
	// A token of the same length differing in one character, a token of another
	// length, and the empty pair: none may match.
	wrongSameLen := "A" + token[1:]
	if wrongSameLen == token {
		wrongSameLen = "B" + token[1:]
	}
	for name, tok := range map[string]string{
		"same length, one character off": wrongSameLen,
		"shorter":                        token[:len(token)-1],
		"longer":                         token + "A",
		"empty":                          "",
	} {
		if TokenMatches(hash, tok) {
			t.Errorf("%s must not match", name)
		}
	}
	if TokenMatches("", token) {
		t.Errorf("a dropped token (empty stored hash) must never match")
	}
	if TokenMatches("", "") {
		t.Errorf("an empty pair must never match")
	}
	if !HashMatches(hash, hash) || HashMatches(hash, "deadbeef") || HashMatches("", "") {
		t.Errorf("HashMatches is wrong on the live_leases compare")
	}
}

// TestTokenCompareIsConstantTime pins the mechanism, not only the answer: the
// compare must run through crypto/subtle so it leaks no prefix of another
// lease's token to a node guessing one (D44).
func TestTokenCompareIsConstantTime(t *testing.T) {
	src, err := os.ReadFile("token.go")
	if err != nil {
		t.Fatalf("read token.go: %v", err)
	}
	text := string(src)
	if !strings.Contains(text, `"crypto/subtle"`) {
		t.Fatalf("token.go must import crypto/subtle")
	}
	if strings.Count(text, "subtle.ConstantTimeCompare") < 2 {
		t.Fatalf("both TokenMatches and HashMatches must compare through subtle.ConstantTimeCompare")
	}
	if strings.Contains(text, "storedHash == HashLeaseToken(") {
		t.Fatalf("token.go compares hashes with ==, which is not constant time")
	}
}

func TestNewLeaseIDIsAULIDSortableByGrantTime(t *testing.T) {
	// The ULID specification's worked example: this millisecond timestamp
	// encodes to the 10-character prefix below.
	stamp := time.UnixMilli(1469918176385).UTC()
	id, err := NewLeaseID(stamp, testEntropy())
	if err != nil {
		t.Fatalf("new lease id: %v", err)
	}
	if len(id) != 26 {
		t.Fatalf("lease id %q is %d characters, want 26", id, len(id))
	}
	if got, want := id[:10], "01ARYZ6S41"; got != want {
		t.Fatalf("timestamp prefix = %q, want %q (id %q)", got, want, id)
	}
	for _, c := range id {
		if !strings.ContainsRune(crockford, c) {
			t.Fatalf("lease id %q carries %q, outside the Crockford alphabet", id, c)
		}
	}

	later, err := NewLeaseID(stamp.Add(time.Second), testEntropy())
	if err != nil {
		t.Fatalf("new lease id: %v", err)
	}
	if !(id < later) {
		t.Fatalf("lease ids must sort by grant time: %q is not before %q", id, later)
	}
}
