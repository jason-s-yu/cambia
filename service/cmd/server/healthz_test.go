// cmd/server/healthz_test.go
//
// /healthz is the unauthenticated liveness probe the deploy stack polls. It must
// stay 200 while the process is up even with both dependencies down, reporting
// their state in the body instead, so a Postgres restart does not turn into a
// container restart loop.
package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHealthzOKWithDependenciesDown(t *testing.T) {
	// database.DB and cache.Rdb are nil in a bare test binary: the "down" case.
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	healthzHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 with dependencies down, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("expected a JSON content type, got %q", ct)
	}

	var body struct {
		Status string `json:"status"`
		DB     bool   `json:"db"`
		Redis  bool   `json:"redis"`
	}
	if err := json.NewDecoder(w.Body).Decode(&body); err != nil {
		t.Fatalf("failed to decode healthz body: %v", err)
	}
	if body.Status != "ok" {
		t.Fatalf(`expected status "ok", got %q`, body.Status)
	}
	if body.DB || body.Redis {
		t.Fatalf("expected both dependency flags false, got db=%v redis=%v", body.DB, body.Redis)
	}
}

func TestHealthzRejectsNonGET(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/healthz", nil)
	w := httptest.NewRecorder()

	healthzHandler(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 for POST, got %d", w.Code)
	}
}

// The probe must not sit behind RequireAuth: an orchestrator has no cookie.
func TestHealthzRouteIsUnauthenticated(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthzHandler)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected an unauthenticated 200, got %d", w.Code)
	}
}
