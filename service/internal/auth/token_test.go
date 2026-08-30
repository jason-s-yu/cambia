// internal/auth/token_test.go
package auth

import (
	"net/http/httptest"
	"testing"
)

// mustJWT mints a token for sub, failing the test if signing fails.
func mustJWT(t *testing.T, sub string) string {
	t.Helper()
	token, err := CreateJWT(sub)
	if err != nil {
		t.Fatalf("failed to create JWT for %q: %v", sub, err)
	}
	return token
}

// TestResolveAuthTokenHeaderBeatsCookie checks resolution step 1: an
// Authorization: Bearer token wins over a valid cookie, and nothing is written
// to the response (there is no cookie to heal).
func TestResolveAuthTokenHeaderBeatsCookie(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Authorization", "Bearer "+mustJWT(t, "header-user"))
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthToken(w, req)

	if !ok || !sawAny {
		t.Fatalf("expected ok=true sawAny=true, got ok=%v sawAny=%v", ok, sawAny)
	}
	if userID != "header-user" {
		t.Fatalf("expected the header token to win, got %q", userID)
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("expected no Set-Cookie, got %q", sc)
	}
}

// TestResolveAuthTokenHandshakeBeatsCookie checks resolution step 2: the
// WebSocket handshake carrier wins over a valid cookie.
func TestResolveAuthTokenHandshakeBeatsCookie(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/ws/lobby", nil)
	req.Header.Set("Sec-WebSocket-Protocol", "cambia, "+TokenSubprotocolPrefix+mustJWT(t, "ws-user"))
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthToken(w, req)

	if !ok || !sawAny {
		t.Fatalf("expected ok=true sawAny=true, got ok=%v sawAny=%v", ok, sawAny)
	}
	if userID != "ws-user" {
		t.Fatalf("expected the handshake token to win, got %q", userID)
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("expected no Set-Cookie, got %q", sc)
	}
}

// TestResolveAuthTokenHeaderBeatsHandshake pins the order between the two
// explicit carriers: the header is step 1, the handshake step 2.
func TestResolveAuthTokenHeaderBeatsHandshake(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/ws/lobby", nil)
	req.Header.Set("Authorization", "Bearer "+mustJWT(t, "header-user"))
	req.Header.Set("Sec-WebSocket-Protocol", "cambia, "+TokenSubprotocolPrefix+mustJWT(t, "ws-user"))
	w := httptest.NewRecorder()

	userID, _, ok := ResolveAuthToken(w, req)

	if !ok {
		t.Fatal("expected ok=true")
	}
	if userID != "header-user" {
		t.Fatalf("expected the header token to win over the handshake, got %q", userID)
	}
}

// TestResolveAuthTokenInvalidHeaderIgnoresCookie is the core opt-out rule: a
// caller that sends an explicit token has opted out of the cookie for that
// request. An invalid Bearer token fails the request outright; the cookie is
// neither consulted nor expired, so a valid shared session in the jar survives
// one tab's stale token.
func TestResolveAuthTokenInvalidHeaderIgnoresCookie(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Authorization", "Bearer not-a-valid-jwt")
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthToken(w, req)

	if ok {
		t.Fatalf("expected ok=false for an invalid Bearer token, got userID %q", userID)
	}
	if !sawAny {
		t.Fatal("expected sawAny=true: a token was offered")
	}
	if userID != "" {
		t.Fatalf("expected empty userID, got %q", userID)
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("an invalid Bearer token must not expire the cookie, got Set-Cookie %q", sc)
	}
}

// TestResolveAuthTokenInvalidHandshakeIgnoresCookie applies the same opt-out
// rule to the handshake carrier.
func TestResolveAuthTokenInvalidHandshakeIgnoresCookie(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/ws/lobby", nil)
	req.Header.Set("Sec-WebSocket-Protocol", "cambia, "+TokenSubprotocolPrefix+"not-a-valid-jwt")
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthToken(w, req)

	if ok {
		t.Fatalf("expected ok=false for an invalid handshake token, got userID %q", userID)
	}
	if !sawAny {
		t.Fatal("expected sawAny=true: a token was offered")
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("an invalid handshake token must not expire the cookie, got Set-Cookie %q", sc)
	}
}

// TestResolveAuthTokenBearerSchemeCaseInsensitive checks RFC 7235 scheme
// matching: the scheme is case-insensitive, the token itself is not.
func TestResolveAuthTokenBearerSchemeCaseInsensitive(t *testing.T) {
	Init()

	for _, scheme := range []string{"Bearer", "bearer", "BEARER", "BeArEr"} {
		req := httptest.NewRequest("GET", "/user/me", nil)
		req.Header.Set("Authorization", scheme+" "+mustJWT(t, "header-user"))
		w := httptest.NewRecorder()

		userID, _, ok := ResolveAuthToken(w, req)
		if !ok || userID != "header-user" {
			t.Fatalf("scheme %q: expected header-user, got %q (ok=%v)", scheme, userID, ok)
		}
	}
}

// TestResolveAuthTokenForeignAuthorizationFallsBack checks that an
// Authorization header this service does not issue (a proxy's Basic
// credentials, say) is not read as an opt-out: resolution falls through to the
// cookie.
func TestResolveAuthTokenForeignAuthorizationFallsBack(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Authorization", "Basic dXNlcjpwYXNz")
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, _, ok := ResolveAuthToken(w, req)

	if !ok || userID != "cookie-user" {
		t.Fatalf("expected the cookie identity, got %q (ok=%v)", userID, ok)
	}
}

// TestResolveAuthTokenEmptyBearerFallsBack checks that "Bearer" with no token
// text offers nothing, so it is not an opt-out either.
func TestResolveAuthTokenEmptyBearerFallsBack(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Authorization", "Bearer   ")
	req.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()

	userID, _, ok := ResolveAuthToken(w, req)

	if !ok || userID != "cookie-user" {
		t.Fatalf("expected the cookie identity, got %q (ok=%v)", userID, ok)
	}
}

// TestResolveAuthTokenHandshakeTokenPositionAndSplitHeaders checks the
// Sec-WebSocket-Protocol parse: entries can arrive comma-separated, spread over
// repeated header lines, and in any order, and the JWT's case must survive.
func TestResolveAuthTokenHandshakeTokenPositionAndSplitHeaders(t *testing.T) {
	Init()

	token := mustJWT(t, "ws-user")

	req := httptest.NewRequest("GET", "/ws/lobby", nil)
	req.Header.Add("Sec-WebSocket-Protocol", "cambia")
	req.Header.Add("Sec-WebSocket-Protocol", "other, "+TokenSubprotocolPrefix+token)
	w := httptest.NewRecorder()

	userID, _, ok := ResolveAuthToken(w, req)

	if !ok || userID != "ws-user" {
		t.Fatalf("expected ws-user from a later header line, got %q (ok=%v)", userID, ok)
	}
}

// TestResolveAuthTokenNoExplicitCarrierIsCookieWalk checks that with neither
// explicit carrier present the behavior is exactly the cookie walk, including
// the self-healing expiry of an invalid cookie.
func TestResolveAuthTokenNoExplicitCarrierIsCookieWalk(t *testing.T) {
	Init()

	valid := httptest.NewRequest("GET", "/user/me", nil)
	valid.Header.Set("Cookie", "auth_token="+mustJWT(t, "cookie-user"))
	w := httptest.NewRecorder()
	if userID, _, ok := ResolveAuthToken(w, valid); !ok || userID != "cookie-user" {
		t.Fatalf("expected cookie-user, got %q (ok=%v)", userID, ok)
	}

	invalid := httptest.NewRequest("GET", "/user/me", nil)
	invalid.Header.Set("Cookie", "auth_token=not-a-valid-jwt")
	w = httptest.NewRecorder()
	if _, sawAny, ok := ResolveAuthToken(w, invalid); ok || !sawAny {
		t.Fatalf("expected ok=false sawAny=true for an invalid cookie, got ok=%v sawAny=%v", ok, sawAny)
	}
	if sc := w.Header().Get("Set-Cookie"); sc == "" {
		t.Fatal("an invalid cookie must still be expired by the cookie walk")
	}
}

// TestResolveAuthTokenNothingPresent checks the empty case.
func TestResolveAuthTokenNothingPresent(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthToken(w, req)

	if ok || sawAny || userID != "" {
		t.Fatalf("expected an empty resolution, got userID=%q sawAny=%v ok=%v", userID, sawAny, ok)
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("expected no Set-Cookie, got %q", sc)
	}
}

// TestIsTabSession checks the header that switches a response to tab mode (no
// Set-Cookie). The value match is case-insensitive; any other value is not tab
// mode.
func TestIsTabSession(t *testing.T) {
	cases := map[string]bool{
		"tab":    true,
		"TAB":    true,
		" tab ":  true,
		"cookie": false,
		"":       false,
	}
	for value, want := range cases {
		req := httptest.NewRequest("POST", "/user/guest", nil)
		if value != "" {
			req.Header.Set(SessionModeHeader, value)
		}
		if got := IsTabSession(req); got != want {
			t.Fatalf("%s: %q -> %v, want %v", SessionModeHeader, value, got, want)
		}
	}
}
