// internal/handlers/active_session_test.go
//
// Tests for GET /lobby/active (cambia-783): the endpoint a client calls on the home screen to
// find the lobby or in-progress game it should offer to resume after a refresh or a lost tab.
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// newRunningLobby creates a lobby through the real create handler (which registers the lobby,
// creates its hub and starts the hub's Run loop) and waits until that hub reports alive. The
// returned stop function dissolves the hub; it is idempotent and also runs at test cleanup.
// Also registers a t.Cleanup that best-effort deletes any lobbies row this lobby ends up
// persisting once a game starts against it via startTestGame below (cambia-890 F4; see
// cleanupLobbyDBRows).
func newRunningLobby(t *testing.T, gs *GameServer, hostToken, body string) (*lobby.Lobby, *hub.Hub, func()) {
	t.Helper()

	req := httptest.NewRequest("POST", "/lobby/create", bytes.NewBufferString(body))
	req.Header.Set("Cookie", "auth_token="+hostToken)
	w := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("lobby create failed: %d %s", w.Code, w.Body.String())
	}

	var created lobby.Lobby
	if err := json.Unmarshal(w.Body.Bytes(), &created); err != nil {
		t.Fatalf("failed to decode created lobby: %v", err)
	}
	t.Cleanup(func() { cleanupLobbyDBRows(t, gs, created.ID) })
	lob, exists := gs.LobbyStore.GetLobby(created.ID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", created.ID)
	}
	h, hasHub := gs.HubStore.GetHub(created.ID)
	if !hasHub {
		t.Fatalf("hub %s missing from store after create", created.ID)
	}
	waitHubAlive(t, h, true)

	var once sync.Once
	stop := func() {
		once.Do(func() {
			h.Shutdown()
			waitHubAlive(t, h, false)
		})
	}
	t.Cleanup(stop)
	return lob, h, stop
}

// waitHubAlive blocks until the hub's liveness matches want, or fails the test. The Run loop
// starts and stops on its own goroutine, so both transitions are observed rather than assumed.
func waitHubAlive(t *testing.T, h *hub.Hub, want bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.Alive() == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("hub %s: expected Alive()==%v within timeout", h.ID, want)
}

// getActiveSession calls the handler as the given token's user and returns the decoded body.
func getActiveSession(t *testing.T, gs *GameServer, token string) ActiveSessionResponse {
	t.Helper()
	req := httptest.NewRequest("GET", "/lobby/active", nil)
	if token != "" {
		req.Header.Set("Cookie", "auth_token="+token)
	}
	w := httptest.NewRecorder()
	ActiveSessionHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}
	var resp ActiveSessionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode active session response: %v", err)
	}
	return resp
}

// joinLobbyAs runs the real join handler so lobby membership is established the same way the
// server establishes it.
func joinLobbyAs(t *testing.T, gs *GameServer, lobbyID uuid.UUID, token string) {
	t.Helper()
	req := httptest.NewRequest("POST", "/lobby/"+lobbyID.String()+"/join", nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	JoinLobbyHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("join lobby failed: %d %s", w.Code, w.Body.String())
	}
}

// TestActiveSessionMemberOfLobby covers the plain lobby case: a joined member of a live lobby
// gets that lobby back, with no game attached.
func TestActiveSessionMemberOfLobby(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	memberID := uuid.New()
	memberToken, _ := auth.CreateJWT(memberID.String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head","name":"Resume Me"}`)
	joinLobbyAs(t, gs, lob.ID, memberToken)

	resp := getActiveSession(t, gs, memberToken)
	if resp.Active == nil {
		t.Fatalf("expected an active session for a joined lobby member")
	}
	if resp.Active.LobbyID != lob.ID.String() {
		t.Fatalf("expected lobby %s, got %s", lob.ID, resp.Active.LobbyID)
	}
	if resp.Active.Phase != "open" {
		t.Fatalf("expected phase %q, got %q", "open", resp.Active.Phase)
	}
	if resp.Active.GameID != "" {
		t.Fatalf("expected no game id for a lobby that is not in game, got %q", resp.Active.GameID)
	}
	if resp.Active.Name != "Resume Me" || resp.Active.GameMode != "head_to_head" || resp.Active.LobbyType != "public" {
		t.Fatalf("unexpected lobby context: %+v", *resp.Active)
	}
	if resp.Active.PlayerCount != 1 {
		t.Fatalf("expected playerCount 1 (the joined member), got %d", resp.Active.PlayerCount)
	}
	if resp.Active.Seated {
		t.Fatalf("expected seated=false with no game running")
	}
}

// TestActiveSessionInGame covers the in-progress game case: a seated player gets the lobby plus
// the live game id, and the phase reports in_game.
func TestActiveSessionInGame(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()
	playerToken, _ := auth.CreateJWT(playerID.String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	joinLobbyAs(t, gs, lob.ID, playerToken)

	g := startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	resp := getActiveSession(t, gs, playerToken)
	if resp.Active == nil {
		t.Fatalf("expected an active session for a seated player")
	}
	if resp.Active.LobbyID != lob.ID.String() {
		t.Fatalf("expected lobby %s, got %s", lob.ID, resp.Active.LobbyID)
	}
	if resp.Active.Phase != "in_game" {
		t.Fatalf("expected phase %q, got %q", "in_game", resp.Active.Phase)
	}
	if resp.Active.GameID != g.ID.String() {
		t.Fatalf("expected game %s, got %q", g.ID, resp.Active.GameID)
	}
	if !resp.Active.Seated {
		t.Fatalf("expected seated=true for a player in the game")
	}
	if resp.Active.PlayerCount != 2 {
		t.Fatalf("expected playerCount 2, got %d", resp.Active.PlayerCount)
	}
}

// TestActiveSessionInGameLobbyMemberWithoutSeat checks that a lobby member who holds no seat in
// the running game still gets routed back to the lobby, but is not reported as seated.
func TestActiveSessionInGameLobbyMemberWithoutSeat(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()
	lateID := uuid.New()
	lateToken, _ := auth.CreateJWT(lateID.String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"group_of_4"}`)
	joinLobbyAs(t, gs, lob.ID, lateToken)

	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	resp := getActiveSession(t, gs, lateToken)
	if resp.Active == nil {
		t.Fatalf("expected an active session for a lobby member during a game")
	}
	if resp.Active.Phase != "in_game" {
		t.Fatalf("expected phase %q, got %q", "in_game", resp.Active.Phase)
	}
	if resp.Active.Seated {
		t.Fatalf("expected seated=false for a member with no seat in the running game")
	}
}

// TestActiveSessionNoneWhenIdle covers the empty cases the client must handle silently: no
// lobbies at all, and a live lobby the caller never joined.
func TestActiveSessionNoneWhenIdle(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	strangerID := uuid.New()
	strangerToken, _ := auth.CreateJWT(strangerID.String())

	if resp := getActiveSession(t, gs, strangerToken); resp.Active != nil {
		t.Fatalf("expected no active session with no lobbies, got %+v", *resp.Active)
	}

	newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)

	if resp := getActiveSession(t, gs, strangerToken); resp.Active != nil {
		t.Fatalf("expected no active session for a non-member, got %+v", *resp.Active)
	}
}

// TestActiveSessionSkipsFinishedGame checks that a lobby still pointing at a finished game is
// reported as a plain lobby rather than a resumable game: OnGameEnd clears the lobby
// asynchronously, so the game's own GameOver flag is the authority.
func TestActiveSessionSkipsFinishedGame(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()
	playerToken, _ := auth.CreateJWT(playerID.String())

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, playerToken)

	g := startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})
	g.GameOver = true

	resp := getActiveSession(t, gs, playerToken)
	if resp.Active == nil {
		t.Fatalf("expected the lobby to still be resumable after its game finished")
	}
	if resp.Active.Phase != "open" || resp.Active.GameID != "" {
		t.Fatalf("expected a plain lobby session, got %+v", *resp.Active)
	}
}

// TestActiveSessionSkipsDissolvedHub checks that a lobby whose hub has stopped is not offered.
// Such a hub stays in the hub store and would accept a reconnect it can never answer.
func TestActiveSessionSkipsDissolvedHub(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	memberID := uuid.New()
	memberToken, _ := auth.CreateJWT(memberID.String())

	lob, _, stop := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, memberToken)

	if resp := getActiveSession(t, gs, memberToken); resp.Active == nil {
		t.Fatalf("expected an active session while the hub is alive")
	}

	stop()

	if resp := getActiveSession(t, gs, memberToken); resp.Active != nil {
		t.Fatalf("expected no active session after the hub dissolved, got %+v", *resp.Active)
	}
}

// TestActiveSessionPrefersGameOverIdleLobby checks the precedence rule for a caller who belongs
// to more than one live lobby: the one running a game wins.
func TestActiveSessionPrefersGameOverIdleLobby(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()
	playerToken, _ := auth.CreateJWT(playerID.String())

	idleLobby, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, idleLobby.ID, playerToken)

	gameLobby, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, gameLobby.ID, playerToken)
	g := startTestGame(t, gs, gameLobby, []uuid.UUID{hostID, playerID})

	resp := getActiveSession(t, gs, playerToken)
	if resp.Active == nil {
		t.Fatalf("expected an active session")
	}
	if resp.Active.LobbyID != gameLobby.ID.String() || resp.Active.GameID != g.ID.String() {
		t.Fatalf("expected the in-game lobby %s, got %+v", gameLobby.ID, *resp.Active)
	}
}

// TestActiveSessionRejectsUnauthenticated checks the auth gate and the method gate.
func TestActiveSessionRejectsUnauthenticated(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	req := httptest.NewRequest("GET", "/lobby/active", nil)
	w := httptest.NewRecorder()
	ActiveSessionHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without an auth cookie, got %d: %s", w.Code, w.Body.String())
	}

	token, _ := auth.CreateJWT(uuid.New().String())
	postReq := httptest.NewRequest("POST", "/lobby/active", nil)
	postReq.Header.Set("Cookie", "auth_token="+token)
	postW := httptest.NewRecorder()
	ActiveSessionHandler(gs).ServeHTTP(postW, postReq)
	if postW.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 for POST, got %d: %s", postW.Code, postW.Body.String())
	}
}

// startTestGame registers a game for the lobby with the given seats and marks the lobby in-game,
// mirroring what the hub does when a countdown elapses. The game is left un-dealt: the endpoint
// reads only membership and lifecycle flags, and dealing would pull in the historian and DB.
func startTestGame(t *testing.T, gs *GameServer, lob *lobby.Lobby, playerIDs []uuid.UUID) *game.CambiaGame {
	t.Helper()

	lob.Mu.Lock()
	lobbyID, hostID, gameMode, lobbyType := lob.ID, lob.HostUserID, lob.GameMode, lob.Type
	houseRules, circuit := lob.HouseRules, lob.Circuit
	lob.Mu.Unlock()

	g := gs.CreateGameInstance(context.Background(), lobbyID, hostID, gameMode, lobbyType, false, houseRules, circuit, playerIDs, nil, nil)
	if g == nil {
		t.Fatalf("failed to create game instance for lobby %s", lobbyID)
	}

	lob.Mu.Lock()
	lob.InGame = true
	lob.GameID = g.ID
	lob.GameInstanceCreated = true
	lob.Mu.Unlock()

	return g
}

// rawActiveSession drives ActiveSessionHandler and returns the status code and the exact
// response body, unparsed. getActiveSession above decodes into ActiveSessionResponse, which
// silently tolerates renamed, added or dropped JSON keys; the doc-sample test below needs the
// wire bytes themselves.
func rawActiveSession(t *testing.T, gs *GameServer, token string) (int, []byte) {
	t.Helper()
	req := httptest.NewRequest("GET", "/lobby/active", nil)
	if token != "" {
		req.Header.Set("Cookie", "auth_token="+token)
	}
	w := httptest.NewRecorder()
	ActiveSessionHandler(gs).ServeHTTP(w, req)
	return w.Code, w.Body.Bytes()
}

// TestActiveSessionDocSample is the test service/doc/rest_api.md's GET /lobby/active samples
// cite (cambia-942 F7). The doc claimed its bodies were captured from TestActiveSessionInGame,
// which decodes into a typed struct and never sees a raw body, so nothing tied the documented
// wire shape to the handler and the samples could drift silently. This asserts all three
// documented bodies: the in-game session's exact key set, the empty case, and the 401 text.
func TestActiveSessionDocSample(t *testing.T) {
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

	code, body := rawActiveSession(t, gs, playerToken)
	if code != http.StatusOK {
		t.Fatalf("expected 200 for a seated player, got %d: %s", code, body)
	}
	var envelope struct {
		Active map[string]json.RawMessage `json:"active"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatalf("decode active session body %s: %v", body, err)
	}
	// The lobby is unnamed, so `name` is legitimately absent (omitempty) - the doc sample omits
	// it too. Every other key the sample shows must be present, and no key it does not show may
	// appear.
	want := map[string]bool{
		"lobbyId": true, "lobbyType": true, "gameMode": true,
		"phase": true, "gameId": true, "seated": true, "playerCount": true,
	}
	for key := range want {
		if _, present := envelope.Active[key]; !present {
			t.Fatalf("documented field %q is missing from the in-game response: %s", key, body)
		}
	}
	for key := range envelope.Active {
		if !want[key] {
			t.Fatalf("field %q appears in the in-game response but not in the rest_api.md sample: %s", key, body)
		}
	}

	// The empty case: a token with no lobby membership documents an explicit null, not an
	// absent key or an empty object.
	idleToken, _ := auth.CreateJWT(uuid.New().String())
	code, body = rawActiveSession(t, gs, idleToken)
	if code != http.StatusOK {
		t.Fatalf("expected 200 for a caller with no membership, got %d: %s", code, body)
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, body); err != nil {
		t.Fatalf("compact empty-case body %s: %v", body, err)
	}
	if got := compact.String(); got != `{"active":null}` {
		t.Fatalf("empty-case body = %s, want %s", got, `{"active":null}`)
	}

	// The documented 401 body.
	code, body = rawActiveSession(t, gs, "")
	if code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without an auth cookie, got %d: %s", code, body)
	}
	if got := strings.TrimSpace(string(body)); got != "Missing authentication token" {
		t.Fatalf("401 body = %q, want %q", got, "Missing authentication token")
	}
}
