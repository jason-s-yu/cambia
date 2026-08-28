// internal/handlers/ws_private_lobby_test.go
package handlers

import (
	"bytes"
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

// createPrivateLobby drives POST /lobby/create with type "private" and returns the new
// lobby id.
// createPrivateLobby drives POST /lobby/create and returns the new lobby id. Registers a
// t.Cleanup that best-effort deletes any lobbies row this lobby ends up persisting once a game
// starts against it (cambia-890 F4; see cleanupLobbyDBRows).
func createPrivateLobby(t *testing.T, gs *GameServer, hostToken string) uuid.UUID {
	t.Helper()
	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBufferString(`{"type":"private","gameMode":"head_to_head"}`))
	req.Header.Set("Cookie", "auth_token="+hostToken)
	w := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create lobby: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var created struct {
		ID uuid.UUID `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode created lobby: %v", err)
	}
	t.Cleanup(func() { cleanupLobbyDBRows(t, gs, created.ID) })
	return created.ID
}

// tryDialWS attempts a WS handshake and returns the raw dial error and response (if any),
// without failing the test, so callers can assert rejection behavior.
func tryDialWS(ctx context.Context, serverURL, lobbyID, token string) (*websocket.Conn, *http.Response, error) {
	wsURL := "ws" + strings.TrimPrefix(serverURL, "http") + "/ws/" + lobbyID
	hdr := http.Header{}
	hdr.Set("Cookie", "auth_token="+token)
	return websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPHeader: hdr})
}

// TestHubWSPrivateLobbyHostCanConnect verifies that the host of a freshly created private
// lobby can open a WS connection to it without any prior invite or join call. Pre-fix, the
// host was never added to lob.Users at creation time, so the private-lobby gate in
// HubWSHandler (which checks lob.Users membership) refused the host's own connection
// (cambia-771).
func TestHubWSPrivateLobbyHostCanConnect(t *testing.T) {
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
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobUUID := createPrivateLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, resp, err := tryDialWS(ctx, ts.URL, lobUUID.String(), hostToken)
	if err != nil {
		status := "<no response>"
		if resp != nil {
			status = resp.Status
		}
		t.Fatalf("host failed to connect to own private lobby: %v (response: %s)", err, status)
	}
	defer conn.Close(websocket.StatusNormalClosure, "done")
}

// TestHubWSPrivateLobbyRejectsUninvitedUser verifies that a user who is neither the host nor
// invited is still refused a WS connection to a private lobby (the fix must not open the gate
// to everyone).
func TestHubWSPrivateLobbyRejectsUninvitedUser(t *testing.T) {
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
	hostToken, _ := auth.CreateJWT(hostID.String())
	lobUUID := createPrivateLobby(t, gs, hostToken)

	strangerID := uuid.New()
	strangerToken, _ := auth.CreateJWT(strangerID.String())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, resp, err := tryDialWS(ctx, ts.URL, lobUUID.String(), strangerToken)
	if err == nil {
		conn.Close(websocket.StatusNormalClosure, "done")
		t.Fatalf("expected uninvited user to be rejected, but connection succeeded")
	}
	if resp == nil {
		t.Fatalf("expected an HTTP response on rejection, got none (err: %v)", err)
	}
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected 403 Forbidden, got %d", resp.StatusCode)
	}
}

// TestHubWSPublicLobbyUnaffected verifies public lobby WS behavior is unchanged: any
// authenticated user (host or not) can connect without invite, and is added to lob.Users.
func TestHubWSPublicLobbyUnaffected(t *testing.T) {
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
	hostToken, _ := auth.CreateJWT(hostID.String())
	lobUUID := createPublicLobby(t, gs, hostToken)

	otherID := uuid.New()
	otherToken, _ := auth.CreateJWT(otherID.String())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	hostConn, _, err := tryDialWS(ctx, ts.URL, lobUUID.String(), hostToken)
	if err != nil {
		t.Fatalf("host failed to connect to own public lobby: %v", err)
	}
	defer hostConn.Close(websocket.StatusNormalClosure, "done")

	otherConn, _, err := tryDialWS(ctx, ts.URL, lobUUID.String(), otherToken)
	if err != nil {
		t.Fatalf("uninvited user failed to connect to public lobby: %v", err)
	}
	defer otherConn.Close(websocket.StatusNormalClosure, "done")
}
