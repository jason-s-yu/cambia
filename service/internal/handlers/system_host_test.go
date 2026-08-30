// internal/handlers/system_host_test.go
//
// Quick play matches belong to their queue, not to whoever clicked Play first (cambia-1087), and
// the rules they run on are the queue's on every path that can set them (cambia-1089).
package handlers

import (
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
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// lastFrameOfType returns the most recent frame of a type, unlike waitForType which returns the
// first. A lobby roster is broadcast on connect and again when the match forms, and it is the
// second one that carries the handover to the system host.
func lastFrameOfType(c *wsTestClient, msgType string, timeout time.Duration) *wsEnvelope {
	deadline := time.Now().Add(timeout)
	for {
		c.mu.Lock()
		var found *wsEnvelope
		for i := range c.frames {
			if c.frames[i].Type == msgType {
				env := c.frames[i]
				found = &env
			}
		}
		c.mu.Unlock()
		if found != nil || time.Now().After(deadline) {
			return found
		}
		time.Sleep(15 * time.Millisecond)
	}
}

// TestCreateQueueBackedLobbyRefusesClientRules is the cambia-1089 regression. The ranked rules
// lock lived only on the WebSocket, and CreateLobbyHandler applied the request body's rules
// before it had worked out that a queue was involved, so a hand-written create seated a rated
// match of a public queue on rules of the caller's choosing.
func TestCreateQueueBackedLobbyRefusesClientRules(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	// Both ways a create resolves to a queue: typed matchmaking, and a standing lobby carrying
	// a queue id (which SearchLobbyHandler queues just the same).
	bodies := map[string]string{
		"houseRules on a matchmaking lobby": `{"type":"matchmaking","queueID":"h2h_quickplay","houseRules":{"cardsPerPlayer":2}}`,
		"circuit on a matchmaking lobby":    `{"type":"matchmaking","queueID":"h2h_quickplay","circuit":{"enabled":true}}`,
		"settings on a matchmaking lobby":   `{"type":"matchmaking","queueID":"h2h_quickplay","settings":{"autoStart":false}}`,
		"lobbySettings spelling":            `{"type":"matchmaking","queueID":"h2h_quickplay","lobbySettings":{"autoStart":false}}`,
		"houseRules on a queued public":     `{"type":"public","queueID":"ffa4_standard","houseRules":{"cardsPerPlayer":2}}`,
		"legacy gameMode queue id":          `{"type":"matchmaking","gameMode":"h2h_quickplay","houseRules":{"cardsPerPlayer":2}}`,
	}
	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			w := postCreateLobby(t, gs, token, body)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected 400 for client rules on a queue-backed lobby, got %d: %s", w.Code, w.Body.String())
			}
			if !strings.Contains(w.Body.String(), "queue sets its own rules") {
				t.Fatalf("the 400 must say the queue owns the rules, got %q", w.Body.String())
			}
		})
	}
}

// TestQueueRulesWinOnCreate is the positive half: a queue-backed lobby is built on the ruleset
// the queue's matches are played with, and nothing a client sent moved them.
//
// That ruleset was game.DefaultHouseRules until cambia-1123, not because the queue played the
// defaults - it never did, the game was built from the queue preset (cambia-1088) - but because
// the lobby object was left on whatever NewLobbyWithDefaults gave it. Asserting the defaults here
// was pinning the gap between what the lobby said and what its match played.
func TestQueueRulesWinOnCreate(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	preset, _ := lobby.GetPreset("h2h_quickplay")
	if lob.HouseRules != preset.HouseRules {
		t.Fatalf("a queue-backed lobby must carry its queue's rule set, got %+v", lob.HouseRules)
	}
	if lob.PresetID != "h2h_quickplay" {
		t.Fatalf("expected the lobby to name the queue it plays, got %q", lob.PresetID)
	}
	if !lob.LobbySettings.AutoStart {
		t.Fatal("auto-start must stay on: it is what starts a match with no player host")
	}
	if lob.Circuit.Enabled {
		t.Fatal("a queue-backed lobby must not come up with circuit scoring on")
	}
}

// TestCreatePlainLobbyStillTakesRules is the control: taking rules away from queue-backed
// creates must not take them away from the lobbies people open to play their own way.
func TestCreatePlainLobbyStillTakesRules(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	body := `{"type":"public","gameMode":"head_to_head","houseRules":{"cardsPerPlayer":6},"settings":{"autoStart":false}}`
	lob := createdLobby(t, postCreateLobby(t, gs, token, body))
	if lob.HouseRules.CardsPerPlayer != 6 {
		t.Fatalf("a lobby with no queue still takes its creator's rules, got cardsPerPlayer=%d", lob.HouseRules.CardsPerPlayer)
	}
	if lob.LobbySettings.AutoStart {
		t.Fatal("a lobby with no queue still takes its creator's auto-start setting")
	}
}

// TestMatchFormationHandsHostToSystem is the cambia-1087 core: the party leader whose lobby
// happens to hold the match had host powers over everybody else's ranked game, and the seat
// list said so. After the match forms nobody does, and the creator is kept only where the
// database needs a real user.
func TestMatchFormationHandsHostToSystem(t *testing.T) {
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

	userA, userB := uuid.New(), uuid.New()
	tokenA, _ := auth.CreateJWT(userA.String())
	tokenB, _ := auth.CreateJWT(userB.String())

	lobA := createdLobby(t, postCreateLobby(t, gs, tokenA, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	lobB := createdLobby(t, postCreateLobby(t, gs, tokenB, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	// The party leader holds the role while the lobby is still only a party: it is what
	// DELETE /lobby/{id}/search is gated on.
	storedA, ok := gs.LobbyStore.GetLobby(lobA.ID)
	if !ok {
		t.Fatal("lobby A is not in the store")
	}
	if storedA.HostUserID != userA {
		t.Fatalf("a searching party keeps its leader, got host %s want %s", storedA.HostUserID, userA)
	}

	clientA := dialWSClient(t, ctx, ts.URL, lobA.ID.String(), tokenA)
	defer clientA.close()
	clientB := dialWSClient(t, ctx, ts.URL, lobB.ID.String(), tokenB)
	defer clientB.close()
	if clientA.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatal("client A never received lobby_state")
	}
	if clientB.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatal("client B never received lobby_state")
	}

	// Drive the callback directly rather than waiting on the matchmaker's 5-second tick: this
	// test is about what match formation does to the lobby, not about pairing.
	gs.HandleMatchFormed(matchmaking.MatchResult{
		HostLobbyID: lobA.ID,
		QueueID:     "h2h_quickplay",
		TargetCount: 2,
		IsRanked:    true,
		Parties: []matchmaking.QueuedLobby{
			{LobbyID: lobA.ID, PlayerCount: 1, QueueID: "h2h_quickplay", TargetCount: 2, IsRanked: true},
			{LobbyID: lobB.ID, PlayerCount: 1, QueueID: "h2h_quickplay", TargetCount: 2, IsRanked: true},
		},
	})

	storedA.Mu.Lock()
	host, creator := storedA.HostUserID, storedA.CreatorUserID
	storedA.Mu.Unlock()
	if host != lobby.SystemHostUserID {
		t.Fatalf("the match lobby must be system-hosted, got host %s", host)
	}
	if creator != userA {
		t.Fatalf("the creator must survive the handover for the lobbies FK, got %s want %s", creator, userA)
	}

	// match_found is what the client seats the match from, so no seat in it may claim the role.
	env := clientA.waitForType("match_found", 5*time.Second)
	if env == nil {
		t.Fatal("client A never received match_found")
	}
	var payload struct {
		Players []struct {
			UserID string `json:"UserID"`
			IsHost bool   `json:"IsHost"`
		} `json:"players"`
	}
	if err := json.Unmarshal(env.Payload, &payload); err != nil {
		t.Fatalf("failed to decode match_found: %v", err)
	}
	if len(payload.Players) != 2 {
		t.Fatalf("expected 2 matched players, got %d", len(payload.Players))
	}
	for _, p := range payload.Players {
		if p.IsHost {
			t.Fatalf("no seat of a matchmade match holds the host role, %s does", p.UserID)
		}
	}

	// The roster the leader is already looking at has to be corrected, or their Host badge and
	// Start game button stay on screen over a service that now refuses both.
	state := lastFrameOfType(clientA, "lobby_state", 5*time.Second)
	if state == nil {
		t.Fatal("client A never received a refreshed lobby_state")
	}
	var snap struct {
		YourIsHost  bool   `json:"your_is_host"`
		SystemHost  bool   `json:"system_host"`
		HostID      string `json:"host_id"`
		LobbyStatus struct {
			Users []struct {
				IsHost bool `json:"is_host"`
			} `json:"users"`
		} `json:"lobby_status"`
	}
	if err := json.Unmarshal(state.Payload, &snap); err != nil {
		t.Fatalf("failed to decode lobby_state: %v", err)
	}
	if snap.YourIsHost {
		t.Fatal("the party leader must stop being told they host the match")
	}
	if !snap.SystemHost {
		t.Fatal("the snapshot must say the system holds the host role")
	}
	if snap.HostID != uuid.Nil.String() {
		t.Fatalf("host_id must be the documented sentinel, got %q", snap.HostID)
	}
	for _, u := range snap.LobbyStatus.Users {
		if u.IsHost {
			t.Fatal("no seat in the roster may render a Host badge")
		}
	}
}

// TestSystemHostedLobbyRefusesSearchEndpoints closes the REST half: a lobby that has held a
// match cannot be put back in a queue or pulled out of one by a player, because there is no
// player host left to authorise it.
func TestSystemHostedLobbyRefusesSearchEndpoints(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	userID := uuid.New()
	token, _ := auth.CreateJWT(userID.String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	stored, ok := gs.LobbyStore.GetLobby(lob.ID)
	if !ok {
		t.Fatal("lobby is not in the store")
	}

	// Searching first, while the party still has its leader: that is the power a party keeps.
	if w := postSearch(t, gs, token, lob.ID); w.Code != http.StatusOK {
		t.Fatalf("a party leader may queue their own party, got %d: %s", w.Code, w.Body.String())
	}

	stored.Mu.Lock()
	stored.Searching = false
	stored.AdoptSystemHostUnsafe()
	stored.Mu.Unlock()

	if w := postSearch(t, gs, token, lob.ID); w.Code != http.StatusForbidden {
		t.Fatalf("a system-hosted lobby must not be requeued by a player, got %d: %s", w.Code, w.Body.String())
	}

	req := httptest.NewRequest("DELETE", fmt.Sprintf("/lobby/%s/search", lob.ID), nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	CancelSearchHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Fatalf("a system-hosted lobby has no player to cancel its search, got %d: %s", w.Code, w.Body.String())
	}
}
