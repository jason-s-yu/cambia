// internal/handlers/lobby_preset_test.go
//
// Ruleset presets over HTTP (cambia-1088): GET /lobby/presets, and the presetId POST
// /lobby/create accepts.
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// getPresets drives PresetsHandler and returns the decoded list.
func getPresets(t *testing.T) []lobby.Preset {
	t.Helper()
	req := httptest.NewRequest("GET", "/lobby/presets", nil)
	w := httptest.NewRecorder()
	http.HandlerFunc(PresetsHandler).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from /lobby/presets, got %d: %s", w.Code, w.Body.String())
	}
	var presets []lobby.Preset
	if err := json.Unmarshal(w.Body.Bytes(), &presets); err != nil {
		t.Fatalf("failed to decode presets: %v", err)
	}
	return presets
}

// TestPresetsEndpointServesTheCanonicalList: the endpoint is a window onto lobby.Presets, so a
// client and the create handler cannot end up applying different values for the same id.
func TestPresetsEndpointServesTheCanonicalList(t *testing.T) {
	got := getPresets(t)
	want := lobby.Presets()
	if len(got) != len(want) {
		t.Fatalf("expected %d presets, got %d", len(want), len(got))
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("preset %d: expected id %q, got %q", i, want[i].ID, got[i].ID)
		}
		if got[i].HouseRules != want[i].HouseRules {
			t.Fatalf("preset %q: house rules do not match the canonical definition", want[i].ID)
		}
		if got[i].GameMode != want[i].GameMode || got[i].Name != want[i].Name {
			t.Fatalf("preset %q: shape does not match the canonical definition", want[i].ID)
		}
	}
	if got[0].ID != lobby.DefaultPresetID {
		t.Fatalf("expected the default preset first, got %q", got[0].ID)
	}
}

// TestPresetsEndpointRejectsNonGET pins the method gate.
func TestPresetsEndpointRejectsNonGET(t *testing.T) {
	req := httptest.NewRequest("POST", "/lobby/presets", nil)
	w := httptest.NewRecorder()
	http.HandlerFunc(PresetsHandler).ServeHTTP(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405, got %d", w.Code)
	}
}

// TestCreateLobbyAppliesPreset: a create carrying a preset id comes back playing that preset's
// rules and its game mode, without the client sending either.
func TestCreateLobbyAppliesPreset(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	for _, presetID := range []string{lobby.DefaultPresetID, "h2h_rapid", "ffa4_standard"} {
		t.Run(presetID, func(t *testing.T) {
			body := fmt.Sprintf(`{"type":"public","presetId":%q}`, presetID)
			lob := createdLobby(t, postCreateLobby(t, gs, token, body))
			want, _ := lobby.GetPreset(presetID)
			if lob.HouseRules != want.HouseRules {
				t.Fatalf("expected the preset's house rules, got %+v", lob.HouseRules)
			}
			if want.GameMode != "" && lob.GameMode != want.GameMode {
				t.Fatalf("expected gameMode %q, got %q", want.GameMode, lob.GameMode)
			}
			if lob.Mode != "casual" || lob.QueueID != "" {
				t.Fatalf("a preset must not make a lobby ranked or queued: mode %q queueID %q", lob.Mode, lob.QueueID)
			}
		})
	}
}

// TestCreateLobbyPresetOverridesClientGameMode: the preset fixes the player count, so a
// mismatched gameMode in the same request loses rather than silently seating the wrong table.
// The client disables its game-mode control for exactly this reason.
func TestCreateLobbyPresetOverridesClientGameMode(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	body := `{"type":"public","gameMode":"head_to_head","presetId":"ffa4_standard"}`
	lob := createdLobby(t, postCreateLobby(t, gs, token, body))
	if lob.GameMode != "group_of_4" {
		t.Fatalf("expected the preset's group_of_4 to win, got %q", lob.GameMode)
	}
}

// TestCreateLobbyPresetWithExplicitRules: the host may depart from a preset in the same request,
// and the fields they name land on top of it.
func TestCreateLobbyPresetWithExplicitRules(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	body := `{"type":"private","presetId":"h2h_rapid","houseRules":{"turnTimerSec":30}}`
	lob := createdLobby(t, postCreateLobby(t, gs, token, body))
	if lob.HouseRules.TurnTimerSec != 30 {
		t.Fatalf("expected the explicit turnTimerSec 30, got %d", lob.HouseRules.TurnTimerSec)
	}
	if !lob.HouseRules.SnapRace || lob.HouseRules.LockCallerHand {
		t.Fatalf("expected the rest of the preset to survive, got %+v", lob.HouseRules)
	}
}

// TestCreateLobbyUnknownPreset: a bad id fails the create rather than quietly producing a
// default-ruleset lobby the host did not ask for.
func TestCreateLobbyUnknownPreset(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	w := postCreateLobby(t, gs, token, `{"type":"public","presetId":"no_such_preset"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

// TestCreateLobbyPresetRefusedForQueuedLobby: a queue owns its lobby's rules, so a preset is
// refused on both shapes that carry a queue id.
func TestCreateLobbyPresetRefusedForQueuedLobby(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	bodies := []string{
		`{"type":"matchmaking","queueID":"h2h_rapid","presetId":"h2h_blitz"}`,
		`{"type":"public","queueID":"h2h_rapid","presetId":"h2h_blitz"}`,
	}
	for _, body := range bodies {
		w := postCreateLobby(t, gs, token, body)
		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d: %s", body, w.Code, w.Body.String())
		}
	}
}

// TestQueuedGameIsBuiltFromTheQueuePreset is what makes a queue preset a claim the service
// keeps: the game a queued lobby produces plays the ruleset the client is shown for that queue,
// not the defaults the lobby object was constructed with. Before cambia-1088 only the reconnect
// grace crossed over, so every ranked queue played game.DefaultHouseRules and the T1C fix in
// MATCHMAKING.md 5 was document-only.
func TestQueuedGameIsBuiltFromTheQueuePreset(t *testing.T) {
	gs := NewGameServer()

	hostID := uuid.New()
	p2ID := uuid.New()

	lob := lobby.NewLobbyWithDefaults(hostID)
	lob.JoinUser(hostID)
	lob.JoinUser(p2ID)
	lob.Type = "matchmaking"
	lob.Mode = "ranked"
	lob.GameMode = "head_to_head"
	lob.QueueID = "h2h_rapid"
	gs.LobbyStore.AddLobby(lob)

	h := hub.NewHub(lob)
	gs.HubStore.CreateHub(h)

	g := gs.NewCambiaGameFromLobby(context.Background(), lob, []uuid.UUID{hostID, p2ID}, nil, h)
	if g == nil {
		t.Fatal("failed to create the queued game instance")
	}
	want, _ := lobby.GetPreset("h2h_rapid")
	if g.HouseRules != want.HouseRules {
		t.Fatalf("expected the queue preset's house rules, got %+v", g.HouseRules)
	}
}

// TestUnqueuedGameKeepsTheLobbyRules is the other side of the same rule: a custom lobby's host
// sets its rules, and nothing about queue presets may reach across and rewrite them.
func TestUnqueuedGameKeepsTheLobbyRules(t *testing.T) {
	gs := NewGameServer()

	hostID := uuid.New()
	p2ID := uuid.New()

	lob := lobby.NewLobbyWithDefaults(hostID)
	lob.JoinUser(hostID)
	lob.JoinUser(p2ID)
	lob.HouseRules.TurnTimerSec = 42
	gs.LobbyStore.AddLobby(lob)

	h := hub.NewHub(lob)
	gs.HubStore.CreateHub(h)

	g := gs.NewCambiaGameFromLobby(context.Background(), lob, []uuid.UUID{hostID, p2ID}, nil, h)
	if g == nil {
		t.Fatal("failed to create the game instance")
	}
	if g.HouseRules != lob.HouseRules {
		t.Fatalf("expected the lobby's own rules, got %+v", g.HouseRules)
	}
}

// TestQueueListNamesComeFromTheQueueConfig: the display names the queue cards render and the
// names the presets carry are one definition (matchmaking.QueueConfig.DisplayName), so a
// renamed queue cannot show up under two labels.
func TestQueueListNamesComeFromTheQueueConfig(t *testing.T) {
	gs := NewGameServer()
	req := httptest.NewRequest("GET", "/matchmaking/queues", nil)
	w := httptest.NewRecorder()
	ListQueuesHandler(gs).ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}
	var entries []struct {
		QueueID string `json:"queueId"`
		Name    string `json:"name"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &entries); err != nil {
		t.Fatalf("failed to decode queue list: %v", err)
	}
	if len(entries) != len(matchmaking.QueueConfigs) {
		t.Fatalf("expected %d queues, got %d", len(matchmaking.QueueConfigs), len(entries))
	}
	for _, e := range entries {
		cfg := matchmaking.QueueConfigs[e.QueueID]
		if e.Name == "" || e.Name != cfg.DisplayName {
			t.Fatalf("queue %q: expected name %q, got %q", e.QueueID, cfg.DisplayName, e.Name)
		}
		p, known := lobby.GetPreset(e.QueueID)
		if !known || p.Name != e.Name {
			t.Fatalf("queue %q: preset name %q does not match the card name %q", e.QueueID, p.Name, e.Name)
		}
	}
}
