// internal/handlers/leave_test.go
//
// Leaving a lobby, end to end (cambia-807), and what a stale lobby URL does afterwards
// (cambia-808). The distinction under test throughout is deliberate leave versus lost
// connection: only POST /lobby/{id}/leave releases membership, and a dropped WebSocket must
// leave a member exactly where they were so they can reconnect.
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// leaveLobbyAs calls POST /lobby/{id}/leave as the given token's user and returns the recorder.
func leaveLobbyAs(t *testing.T, gs *GameServer, lobbyID uuid.UUID, token string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/lobby/"+lobbyID.String()+"/leave", nil)
	if token != "" {
		req.Header.Set("Cookie", "auth_token="+token)
	}
	w := httptest.NewRecorder()
	LeaveLobbyHandler(gs).ServeHTTP(w, req)
	return w
}

// isMember reports whether the user holds any membership entry on the lobby.
func isMember(lob *lobby.Lobby, userID uuid.UUID) bool {
	lob.Mu.Lock()
	defer lob.Mu.Unlock()
	_, ok := lob.Users[userID]
	return ok
}

// TestLeaveReleasesMembership is the headline of cambia-807: before it, "leave lobby" was a
// client-local state flip that sent nothing, so the server kept the member forever and
// /lobby/active kept offering the lobby back.
func TestLeaveReleasesMembership(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	memberID := uuid.New()
	memberToken, _ := auth.CreateJWT(memberID.String())

	lob, h, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	joinLobbyAs(t, gs, lob.ID, memberToken)

	if resp := getActiveSession(t, gs, memberToken); resp.Active == nil {
		t.Fatalf("expected the joined member to have an active session before leaving")
	}

	if w := leaveLobbyAs(t, gs, lob.ID, memberToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}

	if isMember(lob, memberID) {
		t.Fatalf("member %s still holds lobby membership after leaving", memberID)
	}
	if !isMember(lob, hostID) {
		t.Fatalf("one member leaving must not disturb the others")
	}
	if resp := getActiveSession(t, gs, memberToken); resp.Active != nil {
		t.Fatalf("expected no active session after leaving, got %+v", *resp.Active)
	}
	if resp := getActiveSession(t, gs, hostToken); resp.Active == nil {
		t.Fatalf("expected the remaining member to keep their active session")
	}
	if _, exists := gs.LobbyStore.GetLobby(lob.ID); !exists {
		t.Fatalf("a lobby that still has members must not be deleted")
	}
	if !h.Alive() {
		t.Fatalf("a hub whose lobby still has members must keep running")
	}
}

// TestLastLeaveTearsDownLobbyAndHub covers the other end of the lifecycle: with RemoveUser
// unreachable, OnEmpty never fired and lobbies (and their hubs) lived for the whole process.
func TestLastLeaveTearsDownLobbyAndHub(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lob, h, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)

	if w := leaveLobbyAs(t, gs, lob.ID, hostToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}

	if _, exists := gs.LobbyStore.GetLobby(lob.ID); exists {
		t.Fatalf("the last member leaving must delete the lobby from the store")
	}
	if _, exists := gs.HubStore.GetHub(lob.ID); exists {
		t.Fatalf("the last member leaving must deregister the lobby's hub")
	}
	waitHubAlive(t, h, false)

	if resp := getActiveSession(t, gs, hostToken); resp.Active != nil {
		t.Fatalf("expected no active session for a torn-down lobby, got %+v", *resp.Active)
	}

	// Leaving again is a no-op the client never has to handle.
	if w := leaveLobbyAs(t, gs, lob.ID, hostToken); w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 leaving a lobby that is gone, got %d: %s", w.Code, w.Body.String())
	}
}

// TestLeaveRefusedDuringGame gates the leave to non-in-game phases: a seat in a running game is
// not something a lobby-level leave can release, and the membership must survive the refusal.
func TestLeaveRefusedDuringGame(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()
	playerToken, _ := auth.CreateJWT(playerID.String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	joinLobbyAs(t, gs, lob.ID, playerToken)
	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	if w := leaveLobbyAs(t, gs, lob.ID, playerToken); w.Code != http.StatusConflict {
		t.Fatalf("expected 409 leaving mid-game, got %d: %s", w.Code, w.Body.String())
	}
	if !isMember(lob, playerID) {
		t.Fatalf("a refused leave must not release membership")
	}
	if _, exists := gs.LobbyStore.GetLobby(lob.ID); !exists {
		t.Fatalf("a refused leave must not tear the lobby down")
	}
}

// TestLeaveByNonMemberIsANoOp keeps the endpoint idempotent for the client: a double-click, or
// a leave that lost a race with someone else emptying the lobby, is not an error to surface.
func TestLeaveByNonMemberIsANoOp(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	strangerToken, _ := auth.CreateJWT(uuid.New().String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)

	w := leaveLobbyAs(t, gs, lob.ID, strangerToken)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for a non-member leave, got %d: %s", w.Code, w.Body.String())
	}
	var body struct {
		Status  string `json:"status"`
		Removed bool   `json:"removed"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("failed to decode leave response: %v", err)
	}
	if body.Status != "left" || body.Removed {
		t.Fatalf("expected a no-op leave response, got %+v", body)
	}
	if _, exists := gs.LobbyStore.GetLobby(lob.ID); !exists {
		t.Fatalf("a non-member leave must not tear the lobby down")
	}
}

// TestLeaveRejectsUnauthenticatedAndWrongMethod covers the endpoint's gates.
func TestLeaveRejectsUnauthenticatedAndWrongMethod(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	lobbyID := uuid.New()
	if w := leaveLobbyAs(t, gs, lobbyID, ""); w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without an auth cookie, got %d: %s", w.Code, w.Body.String())
	}

	token, _ := auth.CreateJWT(uuid.New().String())
	req := httptest.NewRequest("GET", "/lobby/"+lobbyID.String()+"/leave", nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	LeaveLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 for GET, got %d: %s", w.Code, w.Body.String())
	}
}

// TestDisconnectKeepsMembershipAndReconnectWorks drives the transient case over the real
// WebSocket handler. A client that drops keeps its membership and its resume entry, and its
// reconnect is answered: before cambia-808, dropping the only connection dissolved the hub, so
// the reconnect was accepted by a hub that no longer existed and simply never replied.
func TestDisconnectKeepsMembershipAndReconnectWorks(t *testing.T) {
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

	lobbyUUID := createPublicLobby(t, gs, hostToken)
	lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", lobbyUUID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	first := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	if first.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("first connection never received lobby_state")
	}

	// Drop the only connection, the way a closed tab does.
	first.close()
	time.Sleep(200 * time.Millisecond)

	if !isMember(lob, hostID) {
		t.Fatalf("a dropped connection must not release lobby membership")
	}
	if resp := getActiveSession(t, gs, hostToken); resp.Active == nil {
		t.Fatalf("expected the dropped member's session to still be resumable")
	}

	second := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	defer second.close()
	if second.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("reconnect after a full disconnect was accepted but never served")
	}
}

// TestLeaveUpdatesRosterForRemainingPlayers checks what the other players see: the departure
// has to reach them, and it reaches them as the lobby snapshot the hub broadcasts.
func TestLeaveUpdatesRosterForRemainingPlayers(t *testing.T) {
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
	memberID := uuid.New()
	memberToken, _ := auth.CreateJWT(memberID.String())

	lobbyUUID := createPublicLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	defer host.close()
	member := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), memberToken)
	host.settle()
	member.settle()

	member.close()
	if w := leaveLobbyAs(t, gs, lobbyUUID, memberToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if !rosterContains(t, host, memberID) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("the remaining player never saw a roster without the departed member %s", memberID)
}

// rosterContains reports whether the most recent lobby_state the client received still lists
// the given user.
func rosterContains(t *testing.T, c *wsTestClient, userID uuid.UUID) bool {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := len(c.frames) - 1; i >= 0; i-- {
		if c.frames[i].Type != "lobby_state" {
			continue
		}
		var snapshot struct {
			LobbyStatus struct {
				Users []struct {
					ID string `json:"id"`
				} `json:"users"`
			} `json:"lobby_status"`
		}
		if err := json.Unmarshal(c.frames[i].Payload, &snapshot); err != nil {
			t.Fatalf("failed to decode lobby_state: %v", err)
		}
		for _, u := range snapshot.LobbyStatus.Users {
			if u.ID == userID.String() {
				return true
			}
		}
		return false
	}
	return true // no snapshot yet: treat as "still listed" so the caller keeps waiting
}

// TestConnectToTornDownLobbyIsRefused is the stale-URL half of cambia-808: once the lobby is
// gone the WebSocket must be refused outright. The failure being guarded against is not a
// rejection but an acceptance nothing ever answers.
func TestConnectToTornDownLobbyIsRefused(t *testing.T) {
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

	lobbyUUID := createPublicLobby(t, gs, hostToken)
	joinLobbyAs(t, gs, lobbyUUID, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}
	client.close()
	time.Sleep(100 * time.Millisecond)

	if w := leaveLobbyAs(t, gs, lobbyUUID, hostToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}

	// The stale URL must now be refused before the upgrade, with an HTTP status rather than an
	// accepted socket. Dialling is done directly so the handshake result is visible.
	wsURL := "ws" + ts.URL[len("http"):] + "/ws/" + lobbyUUID.String()
	hdr := http.Header{}
	hdr.Set("Cookie", "auth_token="+hostToken)
	dialCtx, dialCancel := context.WithTimeout(ctx, 3*time.Second)
	defer dialCancel()
	conn, resp, err := websocket.Dial(dialCtx, wsURL, &websocket.DialOptions{HTTPHeader: hdr})
	if err == nil {
		conn.Close(websocket.StatusNormalClosure, "unexpected success")
		t.Fatalf("expected the stale lobby URL to be refused, but the socket was accepted")
	}
	if resp == nil || resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected a 404 refusal for a torn-down lobby, got %v (resp %v)", err, resp)
	}
}
