// internal/handlers/username_population_test.go
//
// Regression coverage for cambia-877: ObfPlayerState.username arrived empty in every in-game
// event and the game_results lobby_status roster came back empty, because CreateGameInstance
// built each models.Player.User with only an ID (Username left at its zero value) and
// attachOnGameEnd's game_results roster never enriched usernames at all. Both symptoms are
// driven here end to end, over real WS connections, against real DB-backed accounts, following
// the forfeit_flow_test.go pattern.
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

// TestE2EGameCarriesRealUsernames drives a real two-player game to Started and then to
// game_results, and asserts the username each player authenticated with reaches both the
// live in-game player roster (private_sync_state's ObfPlayerState.username) and the post-game
// roster (game_results' lobby_status.users), instead of arriving empty and forcing the web
// client's own User_xxxx placeholder (web/src/stores/lobbyStore.ts).
func TestE2EGameCarriesRealUsernames(t *testing.T) {
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template)")
	}
	database.ConnectDB()

	runID := uuid.New().String()
	hostUser := createTestUser(t, fmt.Sprintf("cambia877-host-%s@test.local", runID), "pw12345678", "cambia877_host")
	p2User := createTestUser(t, fmt.Sprintf("cambia877-p2-%s@test.local", runID), "pw12345678", "cambia877_p2")

	gs, ts := newForfeitTestServer(t)

	auth.Init()
	hostToken, err := auth.CreateJWT(hostUser.ID.String())
	if err != nil {
		t.Fatalf("create host JWT: %v", err)
	}
	p2Token, err := auth.CreateJWT(p2User.ID.String())
	if err != nil {
		t.Fatalf("create p2 JWT: %v", err)
	}

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	host.settle()
	p2.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("host never received game_started")
	}

	wantUsernames := map[string]string{
		hostUser.ID.String(): hostUser.Username,
		p2User.ID.String():   p2User.Username,
	}

	// The pre-game reveal's private_sync_state carries the full player roster (opponents'
	// usernames are not secret; only hand contents are), so it is the public-in-game-event
	// vehicle for ObfPlayerState.username.
	syncEnv := host.waitForType("private_sync_state", 5*time.Second)
	if syncEnv == nil {
		t.Fatalf("host never received private_sync_state")
	}
	// The GameEvent envelope carries the ObfGameState under "state" (game.GameEvent.State),
	// not at the top level of the payload - see game.CambiaGame.sendSyncState.
	var syncPayload struct {
		State struct {
			Players []struct {
				PlayerID string `json:"playerId"`
				Username string `json:"username"`
			} `json:"players"`
		} `json:"state"`
	}
	if err := json.Unmarshal(syncEnv.Payload, &syncPayload); err != nil {
		t.Fatalf("decode private_sync_state payload: %v", err)
	}
	if len(syncPayload.State.Players) != 2 {
		t.Fatalf("expected 2 players in private_sync_state, got %d: %+v", len(syncPayload.State.Players), syncPayload.State.Players)
	}
	for _, p := range syncPayload.State.Players {
		want, ok := wantUsernames[p.PlayerID]
		if !ok {
			t.Fatalf("private_sync_state carries unknown player %s", p.PlayerID)
		}
		if p.Username == "" {
			t.Fatalf("player %s username arrived empty in private_sync_state", p.PlayerID)
		}
		if p.Username != want {
			t.Fatalf("player %s username = %q, want %q", p.PlayerID, p.Username, want)
		}
	}

	// End the game (mirrors the mid-game-drop forfeit path in forfeit_flow_test.go) and assert
	// game_results' lobby_status roster carries the same real usernames rather than an empty
	// roster entry the client would fall back away from.
	p2.close()
	results := host.waitForType("game_results", 5*time.Second)
	if results == nil {
		t.Fatalf("host never received game_results")
	}
	var resultsPayload struct {
		LobbyStatus struct {
			Users []struct {
				ID       string `json:"id"`
				Username string `json:"username"`
			} `json:"users"`
		} `json:"lobby_status"`
	}
	if err := json.Unmarshal(results.Payload, &resultsPayload); err != nil {
		t.Fatalf("decode game_results payload: %v", err)
	}
	if len(resultsPayload.LobbyStatus.Users) != 2 {
		t.Fatalf("expected 2 users in game_results lobby_status roster, got %d: %+v", len(resultsPayload.LobbyStatus.Users), resultsPayload.LobbyStatus.Users)
	}
	for _, u := range resultsPayload.LobbyStatus.Users {
		want, ok := wantUsernames[u.ID]
		if !ok {
			t.Fatalf("game_results lobby_status carries unknown user %s", u.ID)
		}
		if u.Username == "" {
			t.Fatalf("user %s username arrived empty in game_results lobby_status", u.ID)
		}
		if u.Username != want {
			t.Fatalf("user %s username = %q, want %q", u.ID, u.Username, want)
		}
	}
}

// createGuestSession drives GuestHandler directly (no route registration needed) and returns
// the new ephemeral user's id and auth token, mirroring how a real unauthenticated client's
// first GET /user/guest precedes opening its WS connection.
func createGuestSession(t *testing.T) (uuid.UUID, string) {
	t.Helper()
	r := httptest.NewRequest("GET", "/user/guest", nil)
	w := httptest.NewRecorder()
	GuestHandler(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("guest handler: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var body struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode guest handler response: %v", err)
	}
	id, err := uuid.Parse(body.ID)
	if err != nil {
		t.Fatalf("parse guest id %q: %v", body.ID, err)
	}
	var token string
	for _, c := range w.Result().Cookies() {
		if c.Name == auth.AuthCookieName {
			token = c.Value
		}
	}
	if token == "" {
		t.Fatalf("guest handler did not set an %s cookie", auth.AuthCookieName)
	}
	return id, token
}

// assertDistinctLobbyUsernames decodes a lobby_state/sync_state envelope's lobby_status.users
// roster and fails if any entry carries an empty username or if two entries share one, the
// literal symptom of cambia-890 (every ephemeral user stored the fixed string "Guest").
func assertDistinctLobbyUsernames(t *testing.T, env *wsEnvelope, source string) {
	t.Helper()
	var payload struct {
		LobbyStatus struct {
			Users []struct {
				ID       string `json:"id"`
				Username string `json:"username"`
			} `json:"users"`
		} `json:"lobby_status"`
	}
	if err := json.Unmarshal(env.Payload, &payload); err != nil {
		t.Fatalf("decode %s payload: %v", source, err)
	}
	if len(payload.LobbyStatus.Users) != 2 {
		t.Fatalf("%s: expected 2 users, got %d: %+v", source, len(payload.LobbyStatus.Users), payload.LobbyStatus.Users)
	}
	seen := make(map[string]string, 2)
	for _, u := range payload.LobbyStatus.Users {
		if u.Username == "" {
			t.Fatalf("%s: user %s carries an empty username", source, u.ID)
		}
		if other, ok := seen[u.Username]; ok {
			t.Fatalf("%s: users %s and %s share the identical username %q", source, other, u.ID, u.Username)
		}
		seen[u.Username] = u.ID
	}
}

// TestE2ETwoFreshGuestsGetDistinctUsernames is a regression test for cambia-890.
// EnsureEphemeralUser (internal/handlers/user.go) stored the literal string "Guest" as the
// Username for every ephemeral user it created, and database.CreateUser derived no per-user
// name of its own. Since cambia-877 (this file's TestE2EGameCarriesRealUsernames, merged at
// master 57638eb) made the live lobby roster, sync_state and game_results all read that stored
// DB username via hubFetchUsername/buildLobbySnapshot, a table of guests rendered the identical
// "Guest" label in every seat, in the lobby list, in chat, and on the results standings - a
// regression from the pre-877 behavior where the client's own User_xxxx fallback (built from
// the id, since the roster username arrived empty) at least told guests apart. Two brand-new
// guests joining one lobby must carry distinct, non-empty usernames in both a lobby_state
// snapshot (the live lobby view) and a sync_state repair frame (same buildLobbySnapshot payload
// shape, see hub.sendSyncState).
//
// One-off SQL to rename existing 'Guest' rows in the dev DB to the new pattern, NOT run here
// (existing rows are left as-is per the ticket):
//
//	UPDATE users
//	SET username = 'Guest-' || upper(substr(replace(id::text, '-', ''), 1, 6))
//	WHERE is_ephemeral AND username = 'Guest';
func TestE2ETwoFreshGuestsGetDistinctUsernames(t *testing.T) {
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template)")
	}
	database.ConnectDB()
	auth.Init()

	gs, ts := newForfeitTestServer(t)

	_, hostToken := createGuestSession(t)
	_, p2Token := createGuestSession(t)

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	defer p2.close()
	host.settle()
	p2.settle()

	// The host's own join sends it a private one-user lobby_state (sendLobbyState) before p2
	// has joined; only the broadcast after p2's join carries the full two-user roster, so wait
	// for that one specifically rather than the first lobby_state frame recorded.
	lobbyEnv := waitForLobbyRoster(t, host, "lobby_state", 2, 5*time.Second)
	if lobbyEnv == nil {
		t.Fatalf("host never received a lobby_state with 2 users")
	}
	assertDistinctLobbyUsernames(t, lobbyEnv, "lobby_state")

	// h.seq is already > 0 from the two joins' broadcastLobbyUpdate calls, so a deliberately
	// stale last_seq reliably trips the staleness gate and gets a private sync_state repair
	// back (sendSyncState reuses buildLobbySnapshot, so the roster shape matches lobby_state).
	host.sendStale("ready")
	syncEnv := waitForLobbyRoster(t, host, "sync_state", 2, 5*time.Second)
	if syncEnv == nil {
		t.Fatalf("host never received a sync_state with 2 users")
	}
	assertDistinctLobbyUsernames(t, syncEnv, "sync_state")
}

// TestClaimEphemeralPersistsChosenUsername is a regression test for cambia-890 finding F1.
// ClaimEphemeralHandler sets u.Username = req.Username in memory (internal/handlers/user.go)
// before calling database.UpdateUserCredentials, but that function's UPDATE statement carried
// no username column, so a claiming player's chosen name was silently dropped and the row kept
// its id-derived guest label ("Guest-A1B2C3") permanently, with is_ephemeral flipped to false.
// Drives POST /user/claim directly, then re-reads the row from the DB (not the in-memory
// *models.User ClaimEphemeralHandler already mutated) to confirm the write actually landed.
func TestClaimEphemeralPersistsChosenUsername(t *testing.T) {
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template)")
	}
	database.ConnectDB()
	auth.Init()

	guestID, guestToken := createGuestSession(t)

	runID := uuid.New().String()
	chosenUsername := "ClaimedName_" + runID[:8]
	body, _ := json.Marshal(map[string]string{
		"email":    fmt.Sprintf("cambia890-claim-%s@test.local", runID),
		"password": "pw12345678",
		"username": chosenUsername,
	})
	r := httptest.NewRequest("POST", "/user/claim", bytes.NewReader(body))
	r.AddCookie(&http.Cookie{Name: auth.AuthCookieName, Value: guestToken})
	w := httptest.NewRecorder()
	ClaimEphemeralHandler(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("claim handler: expected 200, got %d: %s", w.Code, w.Body.String())
	}

	persisted, err := database.GetUserByID(context.Background(), guestID)
	if err != nil {
		t.Fatalf("re-fetch claimed user: %v", err)
	}
	if persisted.IsEphemeral {
		t.Fatalf("claimed user %s still carries is_ephemeral=true", guestID)
	}
	if persisted.Username != chosenUsername {
		t.Fatalf("claimed user %s username = %q, want %q (chosen username was not persisted)", guestID, persisted.Username, chosenUsername)
	}
}

// waitForLobbyRoster polls c's recorded frames for one of msgType whose lobby_status.users
// roster has reached wantUsers entries, up to timeout. A plain waitForType would return the
// first frame of msgType, which for a joiner's own lobby_state is the private one-user snapshot
// sent before any other player has joined - too early for a roster-shape assertion.
func waitForLobbyRoster(t *testing.T, c *wsTestClient, msgType string, wantUsers int, timeout time.Duration) *wsEnvelope {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		for i := range c.frames {
			if c.frames[i].Type != msgType {
				continue
			}
			var payload struct {
				LobbyStatus struct {
					Users []json.RawMessage `json:"users"`
				} `json:"lobby_status"`
			}
			if err := json.Unmarshal(c.frames[i].Payload, &payload); err == nil && len(payload.LobbyStatus.Users) == wantUsers {
				env := c.frames[i]
				c.mu.Unlock()
				return &env
			}
		}
		c.mu.Unlock()
		time.Sleep(25 * time.Millisecond)
	}
	return nil
}
