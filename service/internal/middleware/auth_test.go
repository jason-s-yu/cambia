// internal/middleware/auth_test.go
package middleware

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/sirupsen/logrus"
)

// TestRequireAuthNoCookie checks that a request without an auth_token cookie is
// rejected with 401 and the wrapped handler never runs.
func TestRequireAuthNoCookie(t *testing.T) {
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest("GET", "/training/runs", nil)
	w := httptest.NewRecorder()

	RequireAuth(next).ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", w.Code)
	}
	if called {
		t.Fatal("wrapped handler should not run without a cookie")
	}
}

// TestRequireAuthInvalidCookie checks that a malformed/unsigned auth_token cookie
// is rejected with 401.
func TestRequireAuthInvalidCookie(t *testing.T) {
	auth.Init()

	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest("GET", "/training/runs", nil)
	req.Header.Set("Cookie", "auth_token=not-a-valid-jwt")
	w := httptest.NewRecorder()

	RequireAuth(next).ServeHTTP(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", w.Code)
	}
	if called {
		t.Fatal("wrapped handler should not run with an invalid cookie")
	}
}

// TestRequireAuthValidCookie checks that a request with a valid CreateJWT-issued
// auth_token cookie passes through to the wrapped handler.
func TestRequireAuthValidCookie(t *testing.T) {
	auth.Init() // ephemeral ed25519 keys, no DB needed

	token, err := auth.CreateJWT("test-user-id")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}

	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest("GET", "/training/runs", nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()

	RequireAuth(next).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !called {
		t.Fatal("wrapped handler should run with a valid cookie")
	}
}

// TestRequireAuthComposesWithLogMiddleware checks that RequireAuth composes with
// LogMiddleware (outer logger, inner auth gate) without altering the auth result.
func TestRequireAuthComposesWithLogMiddleware(t *testing.T) {
	auth.Init()

	logger := logrus.New()
	logger.SetOutput(io.Discard) // silence log output; only need the wrapping to work

	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	wrapped := LogMiddleware(logger)(RequireAuth(next))

	// No cookie -> 401 through both layers.
	req := httptest.NewRequest("GET", "/training/runs", nil)
	w := httptest.NewRecorder()
	wrapped.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 through composed middleware, got %d", w.Code)
	}

	// Valid cookie -> 200 through both layers.
	token, err := auth.CreateJWT("test-user-id")
	if err != nil {
		t.Fatalf("failed to create JWT: %v", err)
	}
	req2 := httptest.NewRequest("GET", "/training/runs", nil)
	req2.Header.Set("Cookie", "auth_token="+token)
	w2 := httptest.NewRecorder()
	wrapped.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("expected 200 through composed middleware, got %d", w2.Code)
	}
}
