// internal/middleware/tab_token_test.go
package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// okHandler records whether it ran and answers 200.
func okHandler(called *bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*called = true
		w.WriteHeader(http.StatusOK)
	})
}

// TestRequireAuthBearerHeader checks that an explicit Authorization: Bearer
// token authenticates a request carrying no cookie at all (cambia-1149).
func TestRequireAuthBearerHeader(t *testing.T) {
	auth.Init()

	token, err := auth.CreateJWT("bearer-user")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	called := false
	req := httptest.NewRequest("GET", "/training/runs", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()

	RequireAuth(okHandler(&called)).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for a valid Bearer token, got %d", w.Code)
	}
	if !called {
		t.Fatal("wrapped handler should run with a valid Bearer token")
	}
}

// TestRequireAuthHandshakeToken checks the WebSocket carrier: a token offered
// as a cambia-token.<jwt> subprotocol entry authenticates the upgrade request
// the same way a Bearer header would (cambia-1149).
func TestRequireAuthHandshakeToken(t *testing.T) {
	auth.Init()

	token, err := auth.CreateJWT("ws-user")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	called := false
	req := httptest.NewRequest("GET", "/ws/training/resources", nil)
	req.Header.Set("Sec-WebSocket-Protocol", "cambia, "+auth.TokenSubprotocolPrefix+token)
	w := httptest.NewRecorder()

	RequireAuth(okHandler(&called)).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for a valid handshake token, got %d", w.Code)
	}
	if !called {
		t.Fatal("wrapped handler should run with a valid handshake token")
	}
}

// TestRequireAuthInvalidBearerWithValidCookie pins the opt-out rule at the
// middleware: a request offering an explicit token gets no cookie fallback, so
// a stale tab token is a 401 even when the browser's shared cookie is fine, and
// that cookie is left alone rather than expired out of the jar (cambia-1149).
func TestRequireAuthInvalidBearerWithValidCookie(t *testing.T) {
	auth.Init()

	cookieToken, err := auth.CreateJWT("cookie-user")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	called := false
	req := httptest.NewRequest("GET", "/training/runs", nil)
	req.Header.Set("Authorization", "Bearer not-a-valid-jwt")
	req.Header.Set("Cookie", "auth_token="+cookieToken)
	w := httptest.NewRecorder()

	RequireAuth(okHandler(&called)).ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for an invalid Bearer token, got %d", w.Code)
	}
	if called {
		t.Fatal("wrapped handler should not fall back to the cookie")
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("the valid cookie must survive an invalid Bearer token, got Set-Cookie %q", sc)
	}
}
