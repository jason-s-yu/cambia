// internal/handlers/tab_session_test.go
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// postGuest drives GuestHandler with an optional tab-session header and returns
// the recorder, so a test can assert on both the body and the cookie headers.
func postGuest(t *testing.T, tabMode bool, cookieToken string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/user/guest", nil)
	if tabMode {
		req.Header.Set(auth.SessionModeHeader, auth.SessionModeTab)
	}
	if cookieToken != "" {
		req.Header.Set("Cookie", auth.AuthCookieName+"="+cookieToken)
	}
	w := httptest.NewRecorder()
	GuestHandler(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("guest handler: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	return w
}

// TestGuestHandlerTabModeReturnsTokenWithoutCookie checks the tab-scoped guest
// contract: the token comes back in the body and no Set-Cookie is written, so
// the other tabs on the origin keep the identity they had (cambia-1149).
func TestGuestHandlerTabModeReturnsTokenWithoutCookie(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	w := postGuest(t, true, "")

	var body struct {
		ID    string `json:"id"`
		Token string `json:"token"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode guest response: %v", err)
	}
	id, err := uuid.Parse(body.ID)
	if err != nil {
		t.Fatalf("parse guest id %q: %v", body.ID, err)
	}
	t.Cleanup(func() { cleanupTestUserRows(t, id) })

	if body.Token == "" {
		t.Fatal("tab-mode guest response carried no token")
	}
	sub, err := auth.AuthenticateJWT(body.Token)
	if err != nil {
		t.Fatalf("tab-mode guest token does not verify: %v", err)
	}
	if sub != body.ID {
		t.Fatalf("token subject %q does not match the returned id %q", sub, body.ID)
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("tab-mode guest must not set a cookie, got %q", sc)
	}
}

// TestGuestHandlerTabModeMintsFreshIdentity checks that a tab guest is a new
// user even when the request carries a perfectly valid shared cookie: the
// caller is asking for an identity this tab alone holds, so handing back the
// cookie's identity would defeat the request.
func TestGuestHandlerTabModeMintsFreshIdentity(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	sharedID, sharedToken := createGuestSession(t)

	w := postGuest(t, true, sharedToken)

	var body struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode guest response: %v", err)
	}
	id, err := uuid.Parse(body.ID)
	if err != nil {
		t.Fatalf("parse guest id %q: %v", body.ID, err)
	}
	t.Cleanup(func() { cleanupTestUserRows(t, id) })

	if id == sharedID {
		t.Fatalf("tab-mode guest reused the shared cookie identity %s", sharedID)
	}
}

// TestGuestHandlerWithoutTabHeaderUnchanged is the control: with no
// X-Cambia-Session header the endpoint behaves exactly as before - a cookie is
// set and the body carries the id alone.
func TestGuestHandlerWithoutTabHeaderUnchanged(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	w := postGuest(t, false, "")

	var body map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode guest response: %v", err)
	}
	id, err := uuid.Parse(body["id"])
	if err != nil {
		t.Fatalf("parse guest id %q: %v", body["id"], err)
	}
	t.Cleanup(func() { cleanupTestUserRows(t, id) })

	if _, present := body["token"]; present {
		t.Fatal("cookie-mode guest response must not carry a token")
	}
	if w.Header().Get("Set-Cookie") == "" {
		t.Fatal("cookie-mode guest must still set the auth_token cookie")
	}
}

// TestGuestHandlerInvalidExplicitTokenRejected checks that a stale tab token on
// POST /user/guest is a 401 rather than a silently minted second identity: the
// caller asked to be somebody specific, so the client can unpin and retry
// instead of finding itself signed in as a stranger. No database is reached -
// resolution fails before any user is created.
func TestGuestHandlerInvalidExplicitTokenRejected(t *testing.T) {
	auth.Init()

	req := httptest.NewRequest("POST", "/user/guest", nil)
	req.Header.Set("Authorization", "Bearer not-a-valid-jwt")
	w := httptest.NewRecorder()
	GuestHandler(w, req)

	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for an invalid explicit token, got %d: %s", w.Code, w.Body.String())
	}
	if sc := w.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("expected no Set-Cookie on the rejection, got %q", sc)
	}
}

// TestLoginHandlerTabModeSkipsCookie checks that a tab-scoped login returns the
// same body as always but writes no Set-Cookie, so the switcher can pin a real
// account in one tab without touching the shared jar (cambia-1149).
func TestLoginHandlerTabModeSkipsCookie(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	// Per-run address, per createTestUser's fallback branch: a fixed one would
	// run the test against whatever account already held it.
	email := "tabsession-" + uuid.NewString() + "@example.com"
	password := "tab-session-password"
	createTestUser(t, email, password, "tabsession")

	payload := `{"email":"` + email + `","password":"` + password + `"}`

	tabReq := httptest.NewRequest("POST", "/user/login", strings.NewReader(payload))
	tabReq.Header.Set(auth.SessionModeHeader, auth.SessionModeTab)
	tabW := httptest.NewRecorder()
	LoginHandler(tabW, tabReq)

	if tabW.Code != http.StatusOK {
		t.Fatalf("tab-mode login: expected 200, got %d: %s", tabW.Code, tabW.Body.String())
	}
	var tabBody struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(tabW.Body.Bytes(), &tabBody); err != nil {
		t.Fatalf("decode tab-mode login response: %v", err)
	}
	if tabBody.Token == "" {
		t.Fatal("tab-mode login returned no token")
	}
	if sc := tabW.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("tab-mode login must not set a cookie, got %q", sc)
	}

	// Control: the same login without the header still sets the cookie.
	plainReq := httptest.NewRequest("POST", "/user/login", strings.NewReader(payload))
	plainW := httptest.NewRecorder()
	LoginHandler(plainW, plainReq)

	if plainW.Code != http.StatusOK {
		t.Fatalf("cookie-mode login: expected 200, got %d: %s", plainW.Code, plainW.Body.String())
	}
	if plainW.Header().Get("Set-Cookie") == "" {
		t.Fatal("cookie-mode login must still set the auth_token cookie")
	}
}

// TestMeHandlerBearerToken checks that GET /user/me authenticates off an
// explicit Bearer token with no cookie present, and that an invalid Bearer is
// refused rather than falling back to a valid cookie.
func TestMeHandlerBearerToken(t *testing.T) {
	ensureTestDB(t)
	auth.Init()

	guestID, guestToken := createGuestSession(t)

	req := httptest.NewRequest("GET", "/user/me", nil)
	req.Header.Set("Authorization", "Bearer "+guestToken)
	w := httptest.NewRecorder()
	MeHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for a Bearer token, got %d: %s", w.Code, w.Body.String())
	}
	var body struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /user/me response: %v", err)
	}
	if body.ID != guestID.String() {
		t.Fatalf("expected the Bearer identity %s, got %s", guestID, body.ID)
	}

	stale := httptest.NewRequest("GET", "/user/me", nil)
	stale.Header.Set("Authorization", "Bearer not-a-valid-jwt")
	stale.Header.Set("Cookie", auth.AuthCookieName+"="+guestToken)
	staleW := httptest.NewRecorder()
	MeHandler(staleW, stale)

	if staleW.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for an invalid Bearer token, got %d", staleW.Code)
	}
	if sc := staleW.Header().Get("Set-Cookie"); sc != "" {
		t.Fatalf("an invalid Bearer token must leave the cookie alone, got %q", sc)
	}
}

// TestHubWSHandshakeTokenAuthenticates checks the WebSocket carrier end to end:
// a client offering ["cambia", "cambia-token.<jwt>"] and no cookie connects as
// the token's user, and the negotiated subprotocol is plain "cambia" - the
// token entry is never selected (cambia-1149).
//
// Identity is proven by the private-lobby gate rather than asserted from the
// outside: only a member may connect, and the host is the sole member of a
// freshly created private lobby.
func TestHubWSHandshakeTokenAuthenticates(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, err := auth.CreateJWT(hostID.String())
	if err != nil {
		t.Fatalf("failed to create host JWT: %v", err)
	}
	lobbyID := createPrivateLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/" + lobbyID.String()
	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		Subprotocols: []string{"cambia", auth.TokenSubprotocolPrefix + hostToken},
	})
	if err != nil {
		t.Fatalf("handshake-token dial failed: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "test done")

	if got := conn.Subprotocol(); got != "cambia" {
		t.Fatalf("expected the negotiated subprotocol to be %q, got %q", "cambia", got)
	}

	// A token for somebody else is a different identity, and a non-member of a
	// private lobby: the handshake is refused. That is what pins the socket to
	// the token rather than to the first entry of the protocol list.
	strangerID := uuid.New()
	strangerToken, err := auth.CreateJWT(strangerID.String())
	if err != nil {
		t.Fatalf("failed to create stranger JWT: %v", err)
	}
	strangerConn, resp, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		Subprotocols: []string{"cambia", auth.TokenSubprotocolPrefix + strangerToken},
	})
	if err == nil {
		strangerConn.Close(websocket.StatusNormalClosure, "test done")
		t.Fatal("a non-member's handshake token must not be admitted to a private lobby")
	}
	if resp == nil || resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected 403 for a non-member handshake token, got %v (err %v)", resp, err)
	}
}

// TestHubWSInvalidHandshakeTokenRejected checks the opt-out rule at the socket:
// an explicit token that does not verify is a 401, not a silently minted guest.
func TestHubWSInvalidHandshakeTokenRejected(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, err := auth.CreateJWT(hostID.String())
	if err != nil {
		t.Fatalf("failed to create host JWT: %v", err)
	}
	lobbyID := createPrivateLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/" + lobbyID.String()
	conn, resp, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		Subprotocols: []string{"cambia", auth.TokenSubprotocolPrefix + "not-a-valid-jwt"},
	})
	if err == nil {
		conn.Close(websocket.StatusNormalClosure, "test done")
		t.Fatal("an invalid handshake token must not be admitted")
	}
	if resp == nil || resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401 for an invalid handshake token, got %v (err %v)", resp, err)
	}
}
