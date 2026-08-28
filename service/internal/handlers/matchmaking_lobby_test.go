// internal/handlers/matchmaking_lobby_test.go
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// postCreateLobby runs CreateLobbyHandler for the given raw JSON body as token's user.
func postCreateLobby(t *testing.T, gs *GameServer, token, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBufferString(body))
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(w, req)
	return w
}

// createdLobby decodes a 200 create response into a Lobby, failing the test on any other status.
func createdLobby(t *testing.T, w *httptest.ResponseRecorder) *lobby.Lobby {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}
	lob := &lobby.Lobby{}
	if err := json.Unmarshal(w.Body.Bytes(), lob); err != nil {
		t.Fatalf("failed to decode lobby: %v", err)
	}
	return lob
}

// postSearch runs SearchLobbyHandler for POST /lobby/{id}/search as token's user.
func postSearch(t *testing.T, gs *GameServer, token string, lobbyID uuid.UUID) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", fmt.Sprintf("/lobby/%s/search", lobbyID), nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	SearchLobbyHandler(gs).ServeHTTP(w, req)
	return w
}

// TestCreateMatchmakingLobbyDerivesFromQueueConfig is the cambia-933 regression for the create
// half of the contract: a matchmaking lobby is defined by its queue id, and every mode-shaped
// field on it is derived from that queue's config rather than supplied by the client.
func TestCreateMatchmakingLobbyDerivesFromQueueConfig(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	cases := []struct {
		queueID  string
		gameMode string
		mode     string
	}{
		{"h2h_quickplay", "head_to_head", "ranked"},
		{"ffa4_standard", "group_of_4", "ranked"},
	}
	for _, tc := range cases {
		t.Run(tc.queueID, func(t *testing.T) {
			body := fmt.Sprintf(`{"type":"matchmaking","queueID":%q}`, tc.queueID)
			lob := createdLobby(t, postCreateLobby(t, gs, token, body))
			if lob.GameMode != tc.gameMode {
				t.Fatalf("expected gameMode %q, got %q", tc.gameMode, lob.GameMode)
			}
			if lob.Mode != tc.mode {
				t.Fatalf("expected mode %q, got %q", tc.mode, lob.Mode)
			}
			if lob.QueueID != tc.queueID {
				t.Fatalf("expected queueID %q, got %q", tc.queueID, lob.QueueID)
			}
			if lob.Type != "matchmaking" {
				t.Fatalf("expected type matchmaking, got %q", lob.Type)
			}
		})
	}
}

// TestCreateMatchmakingLobbyRejectsBadQueue covers the two 400s the contract names: an id no
// queue answers to, and no id at all (which must not silently default to some queue).
func TestCreateMatchmakingLobbyRejectsBadQueue(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	w := postCreateLobby(t, gs, token, `{"type":"matchmaking","queueID":"h2h_turbo"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for an unknown queue, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Unknown matchmaking queue: h2h_turbo") {
		t.Fatalf("400 body should name the queue, got %q", w.Body.String())
	}

	w = postCreateLobby(t, gs, token, `{"type":"matchmaking"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for a missing queueID, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Matchmaking lobby requires queueID") {
		t.Fatalf("unexpected 400 body for a missing queueID: %q", w.Body.String())
	}

	// A game mode is not a queue id: the mode-shaped field must not stand in for one.
	w = postCreateLobby(t, gs, token, `{"type":"matchmaking","gameMode":"head_to_head"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for gameMode-without-queueID, got %d: %s", w.Code, w.Body.String())
	}
}

// TestCreateMatchmakingLobbyAcceptsLegacyGameModeQueueID covers the transitional shape: a web
// bundle cached from before cambia-933 sends the queue id as gameMode and no queueID at all.
// It is accepted (and logged as deprecated) so a stale tab keeps working across the deploy.
func TestCreateMatchmakingLobbyAcceptsLegacyGameModeQueueID(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"matchmaking","gameMode":"h2h_quickplay"}`))
	if lob.QueueID != "h2h_quickplay" {
		t.Fatalf("expected the legacy gameMode to become queueID h2h_quickplay, got %q", lob.QueueID)
	}
	if lob.GameMode != "head_to_head" {
		t.Fatalf("expected derived gameMode head_to_head, got %q", lob.GameMode)
	}
	if lob.Mode != "ranked" {
		t.Fatalf("expected derived mode ranked, got %q", lob.Mode)
	}
}

// TestCreatePublicLobbyUnchanged pins the non-matchmaking half of the handler: game-mode
// validation still applies exactly as before, and a bogus mode is still a 400.
func TestCreatePublicLobbyUnchanged(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"public","gameMode":"head_to_head"}`))
	if lob.GameMode != "head_to_head" || lob.Type != "public" {
		t.Fatalf("public lobby changed shape: type=%q gameMode=%q", lob.Type, lob.GameMode)
	}
	if lob.QueueID != "" {
		t.Fatalf("public lobby should carry no queue id, got %q", lob.QueueID)
	}

	w := postCreateLobby(t, gs, token, `{"type":"public","gameMode":"h2h_quickplay"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for a bogus public game mode, got %d: %s", w.Code, w.Body.String())
	}
}

// TestMatchmakingCreateThenSearch proves the two halves of the flow agree: the lobby the create
// handler produces is one SearchLobbyHandler accepts, with no WebSocket connection in between
// (the web client queues straight from the dashboard).
func TestMatchmakingCreateThenSearch(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	w := postSearch(t, gs, token, lob.ID)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 from /lobby/{id}/search, got %d: %s", w.Code, w.Body.String())
	}
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode search response: %v", err)
	}
	if resp["status"] != "searching" {
		t.Fatalf("expected status searching, got %v", resp["status"])
	}
	if resp["queue_id"] != "h2h_quickplay" {
		t.Fatalf("expected queue_id h2h_quickplay, got %v", resp["queue_id"])
	}
}

// TestMatchmakingMatchMovesBothPlayersToOneLobby is the end-to-end half: two solo parties queue
// into h2h_quickplay from separate lobbies, and when the matchmaker pairs them both clients are
// told, in the same match_found frame, which lobby the match is played in. Without the lobby id
// the second party has no way to reach the match at all (cambia-933).
func TestMatchmakingMatchMovesBothPlayersToOneLobby(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	mux := http.NewServeMux()
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	gs.WireMatchmaker()
	go gs.Matchmaker.Run(ctx)

	tokenA, _ := auth.CreateJWT(uuid.New().String())
	tokenB, _ := auth.CreateJWT(uuid.New().String())

	lobA := createdLobby(t, postCreateLobby(t, gs, tokenA, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	lobB := createdLobby(t, postCreateLobby(t, gs, tokenB, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	clientA := dialWSClient(t, ctx, ts.URL, lobA.ID.String(), tokenA)
	defer clientA.close()
	clientB := dialWSClient(t, ctx, ts.URL, lobB.ID.String(), tokenB)
	defer clientB.close()
	if clientA.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("client A never received lobby_state")
	}
	if clientB.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("client B never received lobby_state")
	}

	if w := postSearch(t, gs, tokenA, lobA.ID); w.Code != http.StatusOK {
		t.Fatalf("search for lobby A failed: %d %s", w.Code, w.Body.String())
	}
	if w := postSearch(t, gs, tokenB, lobB.ID); w.Code != http.StatusOK {
		t.Fatalf("search for lobby B failed: %d %s", w.Code, w.Body.String())
	}

	// The matchmaker ticks every 5 seconds; allow two ticks plus slack.
	envA := clientA.waitForType("match_found", 15*time.Second)
	if envA == nil {
		t.Fatalf("client A never received match_found")
	}
	envB := clientB.waitForType("match_found", 15*time.Second)
	if envB == nil {
		t.Fatalf("client B never received match_found")
	}

	var payloadA, payloadB struct {
		LobbyID string `json:"lobby_id"`
		QueueID string `json:"queue_id"`
	}
	if err := json.Unmarshal(envA.Payload, &payloadA); err != nil {
		t.Fatalf("failed to decode client A match_found: %v", err)
	}
	if err := json.Unmarshal(envB.Payload, &payloadB); err != nil {
		t.Fatalf("failed to decode client B match_found: %v", err)
	}
	if payloadA.LobbyID == "" || payloadA.LobbyID != payloadB.LobbyID {
		t.Fatalf("both clients must be pointed at one lobby, got A=%q B=%q", payloadA.LobbyID, payloadB.LobbyID)
	}
	if payloadA.QueueID != "h2h_quickplay" || payloadB.QueueID != "h2h_quickplay" {
		t.Fatalf("match_found carried the wrong queue: A=%q B=%q", payloadA.QueueID, payloadB.QueueID)
	}

	matchLobbyID, err := uuid.Parse(payloadA.LobbyID)
	if err != nil {
		t.Fatalf("match_found lobby_id is not a uuid: %v", err)
	}
	matchLob, ok := gs.LobbyStore.GetLobby(matchLobbyID)
	if !ok {
		t.Fatalf("match lobby %s is not in the store", matchLobbyID)
	}
	// Both players must be members of the match lobby, or the second one's WebSocket upgrade
	// and the game's seating have nothing to go on.
	matchLob.Mu.Lock()
	members := len(matchLob.Users)
	matchLob.Mu.Unlock()
	if members != 2 {
		t.Fatalf("expected both players in the match lobby, got %d members", members)
	}
}
