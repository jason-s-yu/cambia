// internal/handlers/host_migration_test.go
//
// Host migration end to end (cambia-835). A host who left took the lobby's settings with them:
// HostUserID still named them, and the remaining connections carried the IsHost flag they were
// accepted with, so no one in the lobby could change a rule and no client had any reason to
// re-render the settings panel.
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

// sendWith transmits a typed message carrying extra top-level fields, echoing the freshest seq
// the client has seen. The hub's ReadPump treats a frame with no explicit "body" as the body
// itself, which is how the real client sends update_rules.
func (c *wsTestClient) sendWith(msgType string, extra map[string]interface{}) {
	c.mu.Lock()
	seq := c.maxSeq
	c.mu.Unlock()
	frame := map[string]interface{}{"type": msgType, "last_seq": seq}
	for k, v := range extra {
		frame[k] = v
	}
	data, err := json.Marshal(frame)
	if err != nil {
		c.t.Fatalf("marshal %q frame: %v", msgType, err)
	}
	if err := c.conn.Write(c.ctx, websocket.MessageText, data); err != nil {
		c.t.Fatalf("ws write %q: %v", msgType, err)
	}
}

// sendReliableWith is sendReliable for a message with a payload: it retries past the hub's
// staleness gate, which private frames to other users can trip.
func (c *wsTestClient) sendReliableWith(msgType string, extra map[string]interface{}) {
	for attempt := 0; attempt < 8; attempt++ {
		before := c.countType("sync_state")
		c.sendWith(msgType, extra)
		deadline := time.Now().Add(400 * time.Millisecond)
		bounced := false
		for time.Now().Before(deadline) {
			if c.countType("sync_state") > before {
				bounced = true
				break
			}
			time.Sleep(15 * time.Millisecond)
		}
		if !bounced {
			return
		}
	}
	c.t.Fatalf("message %q never accepted after retries", msgType)
}

// latestHostView returns the host id and your_is_host flag from the most recent lobby_state the
// client received, or ("", false) if it has seen none.
func latestHostView(t *testing.T, c *wsTestClient) (string, bool) {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()

	for i := len(c.frames) - 1; i >= 0; i-- {
		if c.frames[i].Type != "lobby_state" {
			continue
		}
		var snapshot struct {
			HostID     string `json:"host_id"`
			YourIsHost bool   `json:"your_is_host"`
		}
		if err := json.Unmarshal(c.frames[i].Payload, &snapshot); err != nil {
			t.Fatalf("failed to decode lobby_state: %v", err)
		}
		return snapshot.HostID, snapshot.YourIsHost
	}
	return "", false
}

// waitForHostView polls until the client's latest snapshot names wantHost and reports the caller
// as host, or the timeout expires.
func waitForHostView(t *testing.T, c *wsTestClient, wantHost uuid.UUID, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if hostID, isHost := latestHostView(t, c); hostID == wantHost.String() && isHost {
			return true
		}
		time.Sleep(25 * time.Millisecond)
	}
	return false
}

// drawFromDiscardAllowed reads the house rule the tests toggle, under the lobby lock.
func drawFromDiscardAllowed(lob *lobby.Lobby) bool {
	lob.Mu.Lock()
	defer lob.Mu.Unlock()
	return lob.HouseRules.AllowDrawFromDiscardPile
}

// TestHostLeaveMigratesAndMemberCanEditRules is the whole ticket over the real handlers: the host
// leaves, the remaining member is told they are the host, and the rule change they then send is
// applied rather than refused.
func TestHostLeaveMigratesAndMemberCanEditRules(t *testing.T) {
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
	lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", lobbyUUID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	host.settle()
	member := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), memberToken)
	defer member.close()
	member.settle()

	// Before the departure the member is not the host and their edits are refused.
	member.sendReliableWith("update_rules", map[string]interface{}{
		"rules": map[string]interface{}{"houseRules": map[string]interface{}{"allowDrawFromDiscardPile": true}},
	})
	if member.waitForType("error", 2*time.Second) == nil {
		t.Fatalf("a plain member's update_rules should have been refused")
	}
	if drawFromDiscardAllowed(lob) {
		t.Fatalf("a refused update must not change the rules")
	}

	// The host leaves the way the web client does: close the socket, then release membership.
	host.close()
	time.Sleep(100 * time.Millisecond)
	if w := leaveLobbyAs(t, gs, lobbyUUID, hostToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}

	if !waitForHostView(t, member, memberID, 3*time.Second) {
		gotHost, gotIsHost := latestHostView(t, member)
		t.Fatalf("the remaining member was never told they are the host (host_id=%q your_is_host=%v)", gotHost, gotIsHost)
	}

	member.settle()
	member.sendReliableWith("update_rules", map[string]interface{}{
		"rules": map[string]interface{}{"houseRules": map[string]interface{}{"allowDrawFromDiscardPile": true}},
	})

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if drawFromDiscardAllowed(lob) {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("the migrated host's rule change was never applied")
}

// TestDepartedHostIsRefusedByPrivateLobbyGate covers the allowance cambia-807 added and cambia-835
// makes dead: with the role migrating, a host who leaves a private lobby is an outsider like any
// other, and letting them past the invite gate would only re-admit somebody who left.
func TestDepartedHostIsRefusedByPrivateLobbyGate(t *testing.T) {
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

	lobbyUUID := createPrivateLobby(t, gs, hostToken)
	lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", lobbyUUID)
	}
	lob.Mu.Lock()
	lob.InviteUser(memberID)
	lob.Mu.Unlock()
	joinLobbyAs(t, gs, lobbyUUID, hostToken)
	joinLobbyAs(t, gs, lobbyUUID, memberToken)

	if w := leaveLobbyAs(t, gs, lobbyUUID, hostToken); w.Code != http.StatusOK {
		t.Fatalf("expected 200 from leave, got %d: %s", w.Code, w.Body.String())
	}
	lob.Mu.Lock()
	newHost := lob.HostUserID
	lob.Mu.Unlock()
	if newHost != memberID {
		t.Fatalf("expected the host role to migrate to %s, got %s", memberID, newHost)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, resp, err := tryDialWS(ctx, ts.URL, lobbyUUID.String(), hostToken)
	if err == nil {
		conn.Close(websocket.StatusNormalClosure, "unexpected success")
		t.Fatalf("a host who left a private lobby must not be let back in without an invite")
	}
	if resp == nil || resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected 403 for the departed host, got %v (resp %v)", err, resp)
	}
}
