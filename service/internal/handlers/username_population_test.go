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
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
)

// TestE2EGameCarriesRealUsernames drives a real two-player game to Started and then to
// game_results, and asserts the username each player authenticated with reaches both the
// live in-game player roster (private_sync_state's ObfPlayerState.username) and the post-game
// roster (game_results' lobby_status.users), instead of arriving empty and forcing the web
// client's own User_xxxx placeholder (web/src/stores/lobbyStore.ts).
func TestE2EGameCarriesRealUsernames(t *testing.T) {
	ensureTestDB(t) // dbAvailable gate + single package-wide database.ConnectDB() (cambia-908).

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
	awaitGameEndPersistence(t, gs)
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
