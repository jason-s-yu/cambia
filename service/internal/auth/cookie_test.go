// internal/auth/cookie_test.go
package auth

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// TestResolveAuthTokenCookieNoCookie checks that a request without any
// auth_token cookie reports sawAny=false, ok=false, and writes no Set-Cookie
// (nothing to self-heal).
func TestResolveAuthTokenCookieNoCookie(t *testing.T) {
	req := httptest.NewRequest("GET", "/user/me", nil)
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthTokenCookie(w, req)

	if sawAny {
		t.Fatal("expected sawAny=false with no auth_token cookie")
	}
	if ok {
		t.Fatal("expected ok=false with no auth_token cookie")
	}
	if userID != "" {
		t.Fatalf("expected empty userID, got %q", userID)
	}
	if setCookie := w.Header().Get("Set-Cookie"); setCookie != "" {
		t.Fatalf("expected no Set-Cookie when no auth_token cookie was present, got %q", setCookie)
	}
}

// TestResolveAuthTokenCookieSingleValid checks that a single valid
// auth_token cookie resolves cleanly with no Set-Cookie (nothing to heal).
func TestResolveAuthTokenCookieSingleValid(t *testing.T) {
	Init()

	token, err := CreateJWT("user-single-valid")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthTokenCookie(w, req)

	if !sawAny {
		t.Fatal("expected sawAny=true with an auth_token cookie present")
	}
	if !ok {
		t.Fatal("expected ok=true for a valid auth_token cookie")
	}
	if userID != "user-single-valid" {
		t.Fatalf("expected userID %q, got %q", "user-single-valid", userID)
	}
	if setCookie := w.Header().Get("Set-Cookie"); setCookie != "" {
		t.Fatalf("expected no Set-Cookie for a single valid auth_token, got %q", setCookie)
	}
}

// TestResolveAuthTokenCookieInvalidOnly checks that one or more invalid
// auth_token cookies (and no valid one) report ok=false and trigger an
// expiring Set-Cookie so the browser drops the bad cookie.
func TestResolveAuthTokenCookieInvalidOnly(t *testing.T) {
	Init()

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Cookie", "auth_token=not-a-valid-jwt")
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthTokenCookie(w, req)

	if !sawAny {
		t.Fatal("expected sawAny=true with an auth_token cookie present")
	}
	if ok {
		t.Fatal("expected ok=false for an invalid auth_token cookie")
	}
	if userID != "" {
		t.Fatalf("expected empty userID, got %q", userID)
	}

	setCookie := w.Header().Get("Set-Cookie")
	if setCookie == "" {
		t.Fatal("expected an expiring Set-Cookie for the invalid auth_token, got none")
	}
	if !strings.Contains(setCookie, AuthCookieName+"=") || !strings.Contains(setCookie, "Max-Age=0") {
		t.Fatalf("expected expiring auth_token Set-Cookie, got %q", setCookie)
	}
}

// TestResolveAuthTokenCookieStaleFirstValidSecond checks the core
// self-heal scenario: a stale/invalid auth_token cookie sent before a valid
// one (e.g. a foreign duplicate shadowing a fresh session on a multi-app
// localhost host, or a stale cookie surviving a dev-server key rotation)
// does not shadow the valid cookie sent later in the header, and the stale
// duplicate triggers an expiring Set-Cookie.
func TestResolveAuthTokenCookieStaleFirstValidSecond(t *testing.T) {
	Init()

	token, err := CreateJWT("user-stale-first")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Cookie", "auth_token=stale-invalid-token; auth_token="+token)
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthTokenCookie(w, req)

	if !sawAny {
		t.Fatal("expected sawAny=true with auth_token cookies present")
	}
	if !ok {
		t.Fatal("expected ok=true when any auth_token cookie verifies")
	}
	if userID != "user-stale-first" {
		t.Fatalf("expected userID %q, got %q", "user-stale-first", userID)
	}

	setCookie := w.Header().Get("Set-Cookie")
	if setCookie == "" {
		t.Fatal("expected an expiring Set-Cookie to clear the stale duplicate, got none")
	}
	if !strings.Contains(setCookie, AuthCookieName+"=") || !strings.Contains(setCookie, "Max-Age=0") {
		t.Fatalf("expected expiring auth_token Set-Cookie, got %q", setCookie)
	}
}

// TestResolveAuthTokenCookieIgnoresOtherCookies checks that cookies with
// other names don't interfere with auth_token resolution.
func TestResolveAuthTokenCookieIgnoresOtherCookies(t *testing.T) {
	Init()

	token, err := CreateJWT("user-mixed-cookies")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Cookie", "session_id=unrelated; auth_token="+token+"; theme=dark")
	w := httptest.NewRecorder()

	userID, sawAny, ok := ResolveAuthTokenCookie(w, req)

	if !sawAny || !ok {
		t.Fatalf("expected sawAny=true, ok=true, got sawAny=%v ok=%v", sawAny, ok)
	}
	if userID != "user-mixed-cookies" {
		t.Fatalf("expected userID %q, got %q", "user-mixed-cookies", userID)
	}
	if setCookie := w.Header().Get("Set-Cookie"); setCookie != "" {
		t.Fatalf("expected no Set-Cookie when the only auth_token cookie is valid, got %q", setCookie)
	}
}
