// internal/handlers/lobby_list_test.go
//
// GET /lobby/list presence filtering (cambia-884). playerCount counts lobby membership, and
// membership survives a dropped socket by design (cambia-807), so a table that closed its tabs
// after a game kept listing itself as a full, unjoinable lobby for the whole idle window. The
// list now answers "what can somebody join right now", which is a question about live
// connections; the members' own way back is GET /lobby/active, which still offers the lobby.
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// listLobbies drives ListLobbiesHandler and returns the decoded response map.
func listLobbies(t *testing.T, gs *GameServer, token string) map[string]ListLobbiesResponse {
	t.Helper()
	req := httptest.NewRequest("GET", "/lobby/list", nil)
	if token != "" {
		req.Header.Set("Cookie", "auth_token="+token)
	}
	w := httptest.NewRecorder()
	ListLobbiesHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /lobby/list, got %d: %s", w.Code, w.Body.String())
	}
	var resp map[string]ListLobbiesResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode lobby list: %v", err)
	}
	return resp
}

// backdateLobby pushes a stored lobby's CreatedAt into the past, simulating one old enough that
// the creation grace (cambia-887 F3) no longer applies to it.
func backdateLobby(t *testing.T, gs *GameServer, id uuid.UUID, age time.Duration) {
	t.Helper()
	lob, exists := gs.LobbyStore.GetLobby(id)
	if !exists {
		t.Fatalf("lobby %s missing from store", id)
	}
	lob.Mu.Lock()
	lob.CreatedAt = time.Now().Add(-age)
	lob.Mu.Unlock()
}

// waitListed polls the list until the lobby's presence matches want, or the timeout expires.
func waitListed(t *testing.T, gs *GameServer, token string, id uuid.UUID, want bool, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		_, listed := listLobbies(t, gs, token)[id.String()]
		if listed == want {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestListLobbiesExcludesLobbyWithNoLiveConnections is the ticket's case: everyone closed their
// tab, so the lobby holds two members and no sockets. It must leave the public list, while the
// lobby itself stays alive for its members to reconnect to until the reaper takes it.
func TestListLobbiesExcludesLobbyWithNoLiveConnections(t *testing.T) {
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

	lobbyID := createPublicLobby(t, gs, hostToken)

	// Backdate past the creation grace (cambia-887 F3) so this test exercises pure
	// presence-based filtering, same as before that grace existed. A lobby still inside its
	// grace window is covered separately by TestListLobbiesListsFreshLobbyDuringCreationGrace.
	backdateLobby(t, gs, lobbyID, 31*time.Second)

	// A lobby nobody has connected to yet is not joinable either: the create response is what
	// takes its host to it, not the list.
	if _, listed := listLobbies(t, gs, hostToken)[lobbyID.String()]; listed {
		t.Fatalf("lobby %s was listed before anyone connected to it", lobbyID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client := dialWSClient(t, ctx, ts.URL, lobbyID.String(), hostToken)
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}

	if !waitListed(t, gs, hostToken, lobbyID, true, 2*time.Second) {
		t.Fatalf("lobby %s with a live connection is missing from the list", lobbyID)
	}
	if got := listLobbies(t, gs, hostToken)[lobbyID.String()].PlayerCount; got != 1 {
		t.Fatalf("expected playerCount 1 for the connected host, got %d", got)
	}

	// Closing the tab: the socket goes, the membership stays.
	client.close()

	if !waitListed(t, gs, hostToken, lobbyID, false, 3*time.Second) {
		t.Fatalf("lobby %s stayed in the public list with nobody connected", lobbyID)
	}

	// Excluding from the list is not tearing down: the lobby is still there and its member is
	// still offered the way back.
	if _, exists := gs.LobbyStore.GetLobby(lobbyID); !exists {
		t.Fatalf("the list filter must not release lobby %s", lobbyID)
	}
	resp := getActiveSession(t, gs, hostToken)
	if resp.Active == nil || resp.Active.LobbyID != lobbyID.String() {
		t.Fatalf("expected the abandoned lobby to stay resumable for its member, got %+v", resp.Active)
	}
}

// TestListLobbiesKeepsAnInGameLobby guards the other side of the filter: a running game holds
// its listing even when every player has dropped, the same exemption the idle reaper makes. The
// lobby is also backdated past its creation grace (cambia-887 F3), so this pins the inGame
// exemption as independent of the grace window rather than merely riding on it.
func TestListLobbiesKeepsAnInGameLobby(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})
	backdateLobby(t, gs, lob.ID, 31*time.Second)

	entry, listed := listLobbies(t, gs, hostToken)[lob.ID.String()]
	if !listed {
		t.Fatalf("an in-game lobby must stay listed, lobby %s is missing", lob.ID)
	}
	if !entry.Lobby.InGame {
		t.Fatalf("expected the listed lobby to report inGame")
	}
}

// TestListLobbiesListsFreshLobbyDuringCreationGrace is the F3 case: a lobby whose host has not
// opened their WebSocket yet has no live connection, but it must not vanish from the list in the
// gap between POST /lobby/create and that first upgrade, and it must survive a brief socket blip
// right after without flickering out of everyone else's list.
func TestListLobbiesListsFreshLobbyDuringCreationGrace(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobbyID := createPublicLobby(t, gs, hostToken)

	entry, listed := listLobbies(t, gs, hostToken)[lobbyID.String()]
	if !listed {
		t.Fatalf("a freshly created lobby with no connections yet must stay listed during its creation grace, lobby %s is missing", lobbyID)
	}
	if entry.Lobby.CreatedAt.IsZero() {
		t.Fatalf("expected CreatedAt to be stamped on the listed lobby")
	}
}

// TestListLobbiesHidesLobbyPastCreationGraceWithNoConnections is the other edge: once the
// creation grace has elapsed, a lobby with no live connection and no game in progress goes back
// to being excluded, the same as any other abandoned lobby.
func TestListLobbiesHidesLobbyPastCreationGraceWithNoConnections(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobbyID := createPublicLobby(t, gs, hostToken)
	backdateLobby(t, gs, lobbyID, 31*time.Second)

	if _, listed := listLobbies(t, gs, hostToken)[lobbyID.String()]; listed {
		t.Fatalf("lobby %s past its creation grace with no connections must not be listed", lobbyID)
	}
}

// TestListLobbiesExcludesPrivateLobby is the cambia-900 L1 case: ListLobbiesHandler applied no
// visibility filter, so a private lobby's id, host id, host-typed name and house rules leaked to
// any unauthenticated caller polling the public list. The lobby is kept inside its creation
// grace and given a live connection, the exact state that keeps a public lobby listed, to prove
// the exclusion is driven by type rather than by presence or the grace window. Its member still
// reaches it through the dedicated resume endpoint (createPrivateLobby is defined in
// ws_private_lobby_test.go).
func TestListLobbiesExcludesPrivateLobby(t *testing.T) {
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

	lobbyID := createPrivateLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	client := dialWSClient(t, ctx, ts.URL, lobbyID.String(), hostToken)
	defer client.close()
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}

	if _, listed := listLobbies(t, gs, hostToken)[lobbyID.String()]; listed {
		t.Fatalf("private lobby %s leaked into the public /lobby/list", lobbyID)
	}

	resp := getActiveSession(t, gs, hostToken)
	if resp.Active == nil || resp.Active.LobbyID != lobbyID.String() {
		t.Fatalf("expected the private lobby to stay resumable for its member via /lobby/active, got %+v", resp.Active)
	}
}

// TestListLobbiesUnauthenticatedReturnsWellFormedBody is the F4 case: GET /lobby/list carries no
// identity requirement (the handler never reads one), so an unauthenticated caller must get the
// same single well-formed JSON body as an authenticated one, not the auth helper's 401 text
// followed by the handler's own JSON encoding.
func TestListLobbiesUnauthenticatedReturnsWellFormedBody(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	req := httptest.NewRequest("GET", "/lobby/list", nil)
	w := httptest.NewRecorder()
	ListLobbiesHandler(gs).ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK for an unauthenticated list request, got %d: %s", w.Code, w.Body.String())
	}
	var resp map[string]ListLobbiesResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unauthenticated /lobby/list body is not well-formed JSON: %v, body: %s", err, w.Body.String())
	}
}

// TestListLobbiesOmitsInternalLobbyFields pins the payload shape: the lobby struct carries its
// own mutex, which was being serialised as an empty "Mu" object, and the fields the web client
// reads must survive the fix untouched.
func TestListLobbiesOmitsInternalLobbyFields(t *testing.T) {
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

	lobbyID := createPublicLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	client := dialWSClient(t, ctx, ts.URL, lobbyID.String(), hostToken)
	defer client.close()
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}
	if !waitListed(t, gs, hostToken, lobbyID, true, 2*time.Second) {
		t.Fatalf("lobby %s with a live connection is missing from the list", lobbyID)
	}

	req := httptest.NewRequest("GET", "/lobby/list", nil)
	req.Header.Set("Cookie", "auth_token="+hostToken)
	w := httptest.NewRecorder()
	ListLobbiesHandler(gs).ServeHTTP(w, req)

	var raw map[string]struct {
		Lobby map[string]json.RawMessage `json:"lobby"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
		t.Fatalf("failed to decode lobby list: %v", err)
	}
	entry, ok := raw[lobbyID.String()]
	if !ok {
		t.Fatalf("lobby %s missing from the list", lobbyID)
	}
	for _, internal := range []string{"Mu", "Users", "ReadyStates", "CountdownTimer", "OnEmpty", "GameInstanceCreated"} {
		if _, leaked := entry.Lobby[internal]; leaked {
			t.Fatalf("internal field %q is serialised in the lobby list payload", internal)
		}
	}
	// A separate lobby.Visibility field was retired (cambia-907 F1): Type alone carries
	// public/private, so a reintroduced visibility field would be redundant state that could
	// drift out of sync with it. Guard against that regression directly rather than only via
	// the internal-field loop above, since "visibility" was never an internal-only field name.
	if _, leaked := entry.Lobby["visibility"]; leaked {
		t.Fatalf("field %q is serialised in the lobby list payload; lobby.Type alone carries public/private (cambia-907 F1)", "visibility")
	}
	// Everything the web client reads off an entry (web/src/types LobbyState) stays.
	for _, want := range []string{"id", "hostUserID", "type", "gameMode", "inGame", "houseRules", "circuit", "lobbySettings", "mode", "name"} {
		if _, present := entry.Lobby[want]; !present {
			t.Fatalf("client-visible field %q disappeared from the lobby list payload", want)
		}
	}

	// The create response is the same struct and must be equally clean.
	createReq := httptest.NewRequest("POST", "/lobby/create", nil)
	createReq.Header.Set("Cookie", "auth_token="+hostToken)
	createW := httptest.NewRecorder()
	CreateLobbyHandler(gs).ServeHTTP(createW, createReq)
	var created map[string]json.RawMessage
	if err := json.Unmarshal(createW.Body.Bytes(), &created); err != nil {
		t.Fatalf("failed to decode create response: %v", err)
	}
	if _, leaked := created["Mu"]; leaked {
		t.Fatalf("the create response still serialises the lobby mutex")
	}
}
