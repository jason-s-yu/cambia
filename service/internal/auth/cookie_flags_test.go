// internal/auth/cookie_flags_test.go
//
// COOKIE_SECURE gates the HTTPS transport flags on every auth_token cookie.
// Unset must leave the dev behavior untouched, since browsers drop Secure
// cookies served over plain http://localhost.
package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewAuthTokenCookieUnsetKeepsDevFlags(t *testing.T) {
	t.Setenv("COOKIE_SECURE", "")

	ck := NewAuthTokenCookie("tok", 3600)

	if ck.Name != AuthCookieName || ck.Value != "tok" {
		t.Fatalf("unexpected cookie identity: %+v", ck)
	}
	if !ck.HttpOnly || ck.Path != "/" || ck.MaxAge != 3600 {
		t.Fatalf("expected HttpOnly cookie at / with MaxAge 3600, got %+v", ck)
	}
	if ck.Secure {
		t.Fatal("expected Secure=false when COOKIE_SECURE is unset")
	}
	// The zero SameSite value omits the attribute entirely, which is what the
	// dev path emitted before COOKIE_SECURE existed.
	if ck.SameSite != 0 {
		t.Fatalf("expected no SameSite attribute when COOKIE_SECURE is unset, got %v", ck.SameSite)
	}
}

func TestNewAuthTokenCookieSecureFlags(t *testing.T) {
	for _, val := range []string{"true", "1"} {
		t.Run(val, func(t *testing.T) {
			t.Setenv("COOKIE_SECURE", val)

			ck := NewAuthTokenCookie("tok", 3600)

			if !ck.Secure {
				t.Fatalf("expected Secure=true with COOKIE_SECURE=%q", val)
			}
			if ck.SameSite != http.SameSiteLaxMode {
				t.Fatalf("expected SameSite=Lax with COOKIE_SECURE=%q, got %v", val, ck.SameSite)
			}
			if !ck.HttpOnly {
				t.Fatal("expected HttpOnly to survive the secure path")
			}
		})
	}
}

func TestNewAuthTokenCookieIgnoresOtherValues(t *testing.T) {
	for _, val := range []string{"false", "0", "yes", "TRUE"} {
		t.Setenv("COOKIE_SECURE", val)
		if NewAuthTokenCookie("tok", 0).Secure {
			t.Fatalf("expected COOKIE_SECURE=%q to be treated as off", val)
		}
	}
}

// ExpireAuthTokenCookie must carry the same flags as the issuing sites, or a
// Secure-issued cookie cannot be cleared by a non-Secure deletion.
func TestExpireAuthTokenCookieCarriesSecureFlags(t *testing.T) {
	t.Setenv("COOKIE_SECURE", "true")

	w := httptest.NewRecorder()
	ExpireAuthTokenCookie(w)

	cookies := (&http.Response{Header: w.Header()}).Cookies()
	if len(cookies) != 1 {
		t.Fatalf("expected exactly one Set-Cookie, got %d", len(cookies))
	}
	ck := cookies[0]
	if ck.Name != AuthCookieName || ck.Value != "" {
		t.Fatalf("expected an empty %s cookie, got %+v", AuthCookieName, ck)
	}
	if ck.MaxAge != -1 {
		t.Fatalf("expected MaxAge=-1 to delete the cookie, got %d", ck.MaxAge)
	}
	if !ck.Secure || ck.SameSite != http.SameSiteLaxMode {
		t.Fatalf("expected Secure + SameSite=Lax on the expiring cookie, got %+v", ck)
	}
}

func TestSetAuthTokenCookieWritesHeader(t *testing.T) {
	t.Setenv("COOKIE_SECURE", "")

	w := httptest.NewRecorder()
	SetAuthTokenCookie(w, "abc", 60)

	cookies := (&http.Response{Header: w.Header()}).Cookies()
	if len(cookies) != 1 {
		t.Fatalf("expected exactly one Set-Cookie, got %d", len(cookies))
	}
	if cookies[0].Value != "abc" || cookies[0].MaxAge != 60 {
		t.Fatalf("unexpected cookie: %+v", cookies[0])
	}
}
