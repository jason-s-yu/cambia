// internal/handlers/dev_session_test.go
package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

// devMux returns a mux with the dev routes registered under the given
// CAMBIA_DEV_ACCOUNTS value, and whether registration happened.
func devMux(t *testing.T, flag string) (*http.ServeMux, bool) {
	t.Helper()
	t.Setenv(DevAccountsEnvVar, flag)
	mux := http.NewServeMux()
	return mux, RegisterDevRoutes(mux)
}

// devAccountName returns a per-run account name matching the server's pattern,
// so repeat runs never collide on one row.
func devAccountName(t *testing.T) string {
	t.Helper()
	return "devtest-" + strings.ReplaceAll(uuid.NewString(), "-", "")[:16]
}

// postDevSession drives POST /dev/session through the mux and returns the
// recorder.
func postDevSession(t *testing.T, mux *http.ServeMux, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/dev/session", strings.NewReader(body))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

// TestDevRoutesUnregisteredWhenFlagUnset checks the gate: with
// CAMBIA_DEV_ACCOUNTS unset the route does not exist, so /dev/session 404s and
// is indistinguishable from any unknown path (cambia-1149).
func TestDevRoutesUnregisteredWhenFlagUnset(t *testing.T) {
	mux, registered := devMux(t, "")
	if registered {
		t.Fatal("dev routes must not register with CAMBIA_DEV_ACCOUNTS unset")
	}
	if DevAccountsEnabled() {
		t.Fatal("DevAccountsEnabled must be false with CAMBIA_DEV_ACCOUNTS unset")
	}

	for _, method := range []string{"GET", "POST"} {
		req := httptest.NewRequest(method, "/dev/session", strings.NewReader(`{"name":"alice"}`))
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != http.StatusNotFound {
			t.Fatalf("%s /dev/session with the flag unset: expected 404, got %d", method, w.Code)
		}
	}
}

// TestDevRoutesRegisteredWhenFlagSet checks the other side of the gate without
// needing a database: with the flag set the path is routed, so an unsupported
// method answers 405 rather than 404.
func TestDevRoutesRegisteredWhenFlagSet(t *testing.T) {
	for _, flag := range []string{"1", "true", "TRUE"} {
		mux, registered := devMux(t, flag)
		if !registered {
			t.Fatalf("dev routes must register with CAMBIA_DEV_ACCOUNTS=%s", flag)
		}

		req := httptest.NewRequest("DELETE", "/dev/session", nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != http.StatusMethodNotAllowed {
			t.Fatalf("CAMBIA_DEV_ACCOUNTS=%s: expected 405 for DELETE, got %d", flag, w.Code)
		}
	}
}

// TestDevSessionRejectsInvalidName checks name validation, which happens before
// any database work: the name becomes an email address and a username, so it is
// held to [a-z0-9_-]{1,32}.
func TestDevSessionRejectsInvalidName(t *testing.T) {
	mux, _ := devMux(t, "1")

	invalid := []string{
		`{"name":"Alice"}`,
		`{"name":"alice bob"}`,
		`{"name":"alice@example.com"}`,
		`{"name":"../etc/passwd"}`,
		`{"name":"` + strings.Repeat("a", 33) + `"}`,
	}
	for _, body := range invalid {
		w := postDevSession(t, mux, body)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("%s: expected 400, got %d: %s", body, w.Code, w.Body.String())
		}
	}

	w := postDevSession(t, mux, `{"name":`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("malformed body: expected 400, got %d", w.Code)
	}
}

// TestDevSessionUpsertIsIdempotent checks the account contract: the same name
// resolves to the same user row every time, with a fixed email and username,
// carrying a password the login form cannot reach, and no cookie is ever set
// (cambia-1149).
func TestDevSessionUpsertIsIdempotent(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	mux, _ := devMux(t, "1")
	name := devAccountName(t)
	body := `{"name":"` + name + `"}`

	decode := func(w *httptest.ResponseRecorder) devSessionResponse {
		t.Helper()
		if w.Code != http.StatusOK {
			t.Fatalf("POST /dev/session: expected 200, got %d: %s", w.Code, w.Body.String())
		}
		if sc := w.Header().Get("Set-Cookie"); sc != "" {
			t.Fatalf("POST /dev/session must not set a cookie, got %q", sc)
		}
		var resp devSessionResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode /dev/session response: %v", err)
		}
		return resp
	}

	first := decode(postDevSession(t, mux, body))
	t.Cleanup(func() { cleanupTestUserRows(t, first.User.ID) })

	if first.User.Username != name {
		t.Fatalf("expected username %q, got %q", name, first.User.Username)
	}
	if first.User.IsEphemeral {
		t.Fatal("a named dev account must be persistent, not ephemeral")
	}
	sub, err := auth.AuthenticateJWT(first.Token)
	if err != nil {
		t.Fatalf("dev session token does not verify: %v", err)
	}
	if sub != first.User.ID.String() {
		t.Fatalf("token subject %q does not match the returned user %s", sub, first.User.ID)
	}

	second := decode(postDevSession(t, mux, body))
	if second.User.ID != first.User.ID {
		t.Fatalf("upsert is not idempotent: %s then %s", first.User.ID, second.User.ID)
	}

	stored, err := database.GetUserByEmail(t.Context(), name+DevAccountEmailDomain)
	if err != nil {
		t.Fatalf("dev account not stored under %s%s: %v", name, DevAccountEmailDomain, err)
	}
	if stored.ID != first.User.ID {
		t.Fatalf("email %s%s resolves to %s, want %s", name, DevAccountEmailDomain, stored.ID, first.User.ID)
	}

	// The login form cannot reach a dev account: its password is 32 bytes of
	// crypto/rand that nothing kept.
	login := httptest.NewRequest("POST", "/user/login",
		strings.NewReader(`{"email":"`+name+DevAccountEmailDomain+`","password":"password"}`))
	loginW := httptest.NewRecorder()
	LoginHandler(loginW, login)
	if loginW.Code != http.StatusForbidden {
		t.Fatalf("expected a dev account login to be refused with 403, got %d: %s", loginW.Code, loginW.Body.String())
	}
}

// TestDevSessionGuestMint checks the no-name form: a fresh ephemeral guest,
// returned in the same shape and with no cookie.
func TestDevSessionGuestMint(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	mux, _ := devMux(t, "1")

	for _, body := range []string{`{}`, `{"name":""}`, ``} {
		w := postDevSession(t, mux, body)
		if w.Code != http.StatusOK {
			t.Fatalf("body %q: expected 200, got %d: %s", body, w.Code, w.Body.String())
		}
		if sc := w.Header().Get("Set-Cookie"); sc != "" {
			t.Fatalf("body %q: must not set a cookie, got %q", body, sc)
		}
		var resp devSessionResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("body %q: decode response: %v", body, err)
		}
		t.Cleanup(func() { cleanupTestUserRows(t, resp.User.ID) })

		if !resp.User.IsEphemeral {
			t.Fatalf("body %q: expected an ephemeral guest", body)
		}
		sub, err := auth.AuthenticateJWT(resp.Token)
		if err != nil {
			t.Fatalf("body %q: guest token does not verify: %v", body, err)
		}
		if sub != resp.User.ID.String() {
			t.Fatalf("body %q: token subject %q does not match user %s", body, sub, resp.User.ID)
		}
	}
}

// TestDevSessionList checks the listing the switcher reads: enabled, and
// carrying every account minted under the dev email domain by its short name.
func TestDevSessionList(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	mux, _ := devMux(t, "1")
	name := devAccountName(t)

	var created devSessionResponse
	w := postDevSession(t, mux, `{"name":"`+name+`"}`)
	if w.Code != http.StatusOK {
		t.Fatalf("POST /dev/session: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if err := json.Unmarshal(w.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode /dev/session response: %v", err)
	}
	t.Cleanup(func() { cleanupTestUserRows(t, created.User.ID) })

	listReq := httptest.NewRequest("GET", "/dev/session", nil)
	listW := httptest.NewRecorder()
	mux.ServeHTTP(listW, listReq)
	if listW.Code != http.StatusOK {
		t.Fatalf("GET /dev/session: expected 200, got %d: %s", listW.Code, listW.Body.String())
	}
	var listing devSessionListResponse
	if err := json.Unmarshal(listW.Body.Bytes(), &listing); err != nil {
		t.Fatalf("decode /dev/session listing: %v", err)
	}
	if !listing.Enabled {
		t.Fatal("expected enabled=true in the dev session listing")
	}
	found := false
	for _, acct := range listing.Accounts {
		if acct.Name == name {
			found = true
			if acct.ID != created.User.ID {
				t.Fatalf("listing reports %s for %q, want %s", acct.ID, name, created.User.ID)
			}
		}
		if strings.Contains(acct.Name, "@") {
			t.Fatalf("listing must carry short names, got %q", acct.Name)
		}
	}
	if !found {
		t.Fatalf("account %q missing from the dev session listing", name)
	}
}
