package authtoken

import (
	"crypto/ed25519"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// nodeClaims builds a well-formed node token claim set at t0 with the given
// lifetime, so each test varies exactly the field it is about.
func nodeClaims(sub string, t0 time.Time, life time.Duration) jwt.MapClaims {
	return jwt.MapClaims{
		"sub": sub,
		"aud": AudienceNode,
		"iat": t0.Unix(),
		"exp": t0.Add(life).Unix(),
	}
}

// staticResolver is the key half of the node verifier for tests that are about
// the verifier's own rules rather than the grant store.
func staticResolver(pub ed25519.PublicKey) KeyResolver {
	return func(string) (ed25519.PublicKey, error) { return pub, nil }
}

func TestAudienceRoutingIsExclusive(t *testing.T) {
	opPub, opPriv := genKeypair(t)
	nodePub, nodePriv := genKeypair(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }

	operator := NewVerifier(opPub)
	node := NewNodeVerifier(staticResolver(nodePub), clock)

	nodeTok := mintEdDSA(t, nodePriv, nodeClaims("node:n-20cbd59e84e3", now, time.Minute))
	// The operator verifier keeps the real clock, so its token is minted against
	// it rather than against the node verifier's injected one.
	opTok := mintEdDSA(t, opPriv, jwt.MapClaims{
		"sub": "client-cli", "aud": Audience, "iat": time.Now().Unix(),
		"exp": time.Now().Add(time.Minute).Unix(),
	})

	if _, err := node.Verify(nodeTok); err != nil {
		t.Fatalf("node verifier should accept a node token: %v", err)
	}
	if _, err := operator.Verify(opTok); err != nil {
		t.Fatalf("operator verifier should accept an operator token: %v", err)
	}

	for _, tc := range []struct {
		name string
		v    *Verifier
		tok  string
	}{
		{"node token on the operator verifier", operator, nodeTok},
		{"operator token on the node verifier", node, opTok},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.v.Verify(tc.tok)
			if !errors.Is(err, ErrWrongAudience) {
				t.Fatalf("err = %v, want ErrWrongAudience", err)
			}
			status, code := HTTPStatus(err)
			if status != http.StatusForbidden || code != "wrong_audience" {
				t.Fatalf("HTTPStatus = %d %q, want 403 wrong_audience", status, code)
			}
		})
	}
}

func TestHTTPStatusMapsEverythingElseToUnauthorized(t *testing.T) {
	if status, code := HTTPStatus(nil); status != http.StatusOK || code != "" {
		t.Fatalf("HTTPStatus(nil) = %d %q, want 200 and no code", status, code)
	}
	for _, err := range []error{ErrInvalidSubject, ErrTokenLifetime, ErrRevoked,
		ErrUnknownNode, ErrGrantExpired, ErrGrantInvalid, errors.New("parse")} {
		status, code := HTTPStatus(err)
		if status != http.StatusUnauthorized || code != "unauthorized" {
			t.Fatalf("HTTPStatus(%v) = %d %q, want 401 unauthorized", err, status, code)
		}
	}
}

func TestNodeTokenLifetimeRules(t *testing.T) {
	nodePub, nodePriv := genKeypair(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	v := NewNodeVerifier(staticResolver(nodePub), func() time.Time { return now })
	const sub = "node:n-20cbd59e84e3"

	cases := []struct {
		name   string
		claims jwt.MapClaims
		ok     bool
	}{
		{"at the ceiling", nodeClaims(sub, now, MaxNodeTokenLifetime), true},
		{"well inside the ceiling", nodeClaims(sub, now, 60*time.Second), true},
		{"one second past the ceiling",
			nodeClaims(sub, now, MaxNodeTokenLifetime+time.Second), false},
		{"a day", nodeClaims(sub, now, 24*time.Hour), false},
		{"no exp", jwt.MapClaims{"sub": sub, "aud": AudienceNode, "iat": now.Unix()}, false},
		{"no iat", jwt.MapClaims{"sub": sub, "aud": AudienceNode,
			"exp": now.Add(time.Minute).Unix()}, false},
		{"exp before iat", nodeClaims(sub, now, -time.Minute), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := v.Verify(mintEdDSA(t, nodePriv, tc.claims))
			if tc.ok {
				if err != nil {
					t.Fatalf("Verify: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Verify should have refused the token")
			}
		})
	}
}

func TestNodeSubjectShapeRules(t *testing.T) {
	nodePub, nodePriv := genKeypair(t)
	opPub, opPriv := genKeypair(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	node := NewNodeVerifier(staticResolver(nodePub), func() time.Time { return now })
	operator := NewVerifier(opPub)

	bad := []string{"", "n-20cbd59e84e3", "node:", "node:../etc", "node:a/b",
		"node:node:x", "user:someone"}
	for _, sub := range bad {
		t.Run("node refuses "+sub, func(t *testing.T) {
			tok := mintEdDSA(t, nodePriv, nodeClaims(sub, now, time.Minute))
			_, err := node.Verify(tok)
			if !errors.Is(err, ErrInvalidSubject) {
				t.Fatalf("err = %v, want ErrInvalidSubject", err)
			}
		})
		t.Run("operator is unchanged for "+sub, func(t *testing.T) {
			// The same subjects on the operator audience keep the behavior they
			// have always had: any string sub passes, because the operator token
			// subject is a log label rather than a key-resolution input.
			tok := mintEdDSA(t, opPriv, jwt.MapClaims{"sub": sub, "aud": Audience})
			got, err := operator.Verify(tok)
			if err != nil {
				t.Fatalf("operator Verify: %v", err)
			}
			if got != sub {
				t.Fatalf("sub = %q, want %q", got, sub)
			}
		})
	}
	if _, err := node.Verify(mintEdDSA(t, nodePriv,
		nodeClaims("node:n-20cbd59e84e3", now, time.Minute))); err != nil {
		t.Fatalf("a well-formed subject should verify: %v", err)
	}
}

func TestNodeIDFromSubjectRoundTrip(t *testing.T) {
	id, err := NodeIDFromSubject(SubjectForNode("n-20cbd59e84e3"))
	if err != nil {
		t.Fatalf("NodeIDFromSubject: %v", err)
	}
	if id != "n-20cbd59e84e3" {
		t.Fatalf("id = %q", id)
	}
}

func TestVerifyClaimsReturnsTheClaimsVerifyDiscards(t *testing.T) {
	pub, priv := genKeypair(t)
	v := NewVerifier(pub)
	tok := mintEdDSA(t, priv, jwt.MapClaims{
		"sub": "client-cli", "aud": Audience, "node_epoch": float64(3),
	})
	sub, claims, err := v.VerifyClaims(tok)
	if err != nil {
		t.Fatalf("VerifyClaims: %v", err)
	}
	if sub != "client-cli" || claims["node_epoch"] != float64(3) {
		t.Fatalf("sub = %q, claims = %v", sub, claims)
	}
}

func TestNodeVerifierRejectsANonEdDSAToken(t *testing.T) {
	nodePub, _ := genKeypair(t)
	v := NewNodeVerifier(staticResolver(nodePub), nil)
	tok := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"sub": "node:n-20cbd59e84e3", "aud": AudienceNode,
	})
	s, err := tok.SignedString([]byte("shared-secret"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := v.Verify(s); err == nil {
		t.Fatal("the EdDSA pin should reject an HS256 node token")
	}
}
