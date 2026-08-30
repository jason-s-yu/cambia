// internal/handlers/preset_identity_test.go
//
// The two defects cambia-1123 fixes, at the HTTP and match-formation boundaries.
//
// A: a lobby records which preset it carries, because the six queue presets are byte-identical
// (MATCHMAKING.md 5.2 fixes one ruleset for every ranked queue) and a client recognising a preset
// by value names whichever one its list returns first.
//
// B: a matchmade lobby's rule sheet showed the defaults NewLobbyWithDefaults built it with while
// its game ran the queue preset. The queue's ruleset now lands on the lobby itself, at creation
// and again at match formation, and the game is built from the lobby.
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// wireRuleSheet is the part of a lobby_state payload the client's rule sheet renders.
type wireRuleSheet struct {
	HouseRules game.HouseRules `json:"house_rules"`
	PresetID   string          `json:"preset_id"`
	SystemHost bool            `json:"system_host"`
}

// ruleSheetOf decodes a lobby_state envelope into the fields the sheet renders.
func ruleSheetOf(t *testing.T, env *wsEnvelope) wireRuleSheet {
	t.Helper()
	if env == nil {
		t.Fatalf("expected a lobby_state envelope")
	}
	var sheet wireRuleSheet
	if err := json.Unmarshal(env.Payload, &sheet); err != nil {
		t.Fatalf("decode lobby_state: %v", err)
	}
	return sheet
}

// TestQueueBackedLobbyIsCreatedOnTheQueueRuleset is defect B at the create boundary: the lobby
// object itself carries the queue's rules from the moment it exists, on both shapes that resolve
// to a queue. Before this it kept the defaults, which are the opposite of the queue ruleset on
// all four rules MATCHMAKING.md 5.2 departs from.
func TestQueueBackedLobbyIsCreatedOnTheQueueRuleset(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	for _, tc := range []struct {
		name    string
		body    string
		queueID string
	}{
		{"matchmaking type", `{"type":"matchmaking","queueID":"h2h_rapid"}`, "h2h_rapid"},
		{"public lobby carrying a queue id", `{"type":"public","queueID":"ffa4_standard"}`, "ffa4_standard"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, known := lobby.GetPreset(tc.queueID)
			if !known {
				t.Fatalf("%s must be a preset", tc.queueID)
			}
			if want.HouseRules == game.DefaultHouseRules() {
				t.Fatalf("a queue preset equal to the defaults would make this test vacuous")
			}

			lob := createdLobby(t, postCreateLobby(t, gs, token, tc.body))
			if lob.HouseRules != want.HouseRules {
				t.Fatalf("expected the queue's ruleset on the lobby, got %+v", lob.HouseRules)
			}
			if lob.PresetID != tc.queueID {
				t.Fatalf("expected the lobby to name queue %q, got %q", tc.queueID, lob.PresetID)
			}
			// The queue's reconnect grace is part of the ruleset it plays (MATCHMAKING.md 8),
			// and taking the whole preset is what keeps that override rather than the default.
			cfg, _ := matchmaking.GetQueueConfig(tc.queueID)
			if lob.HouseRules.DisconnectGraceSec != cfg.DisconnectGraceSec {
				t.Fatalf("expected the queue's reconnect grace %d, got %d", cfg.DisconnectGraceSec, lob.HouseRules.DisconnectGraceSec)
			}
		})
	}
}

// TestCreateLobbyRecordsWhichPresetItCarries is defect A at the create boundary: two presets with
// identical rules are still two presets, and the lobby comes back naming the one that was asked
// for. Value-matching cannot tell them apart, which is how a lobby created from H2H Rapid was
// shown as H2H Quick.
func TestCreateLobbyRecordsWhichPresetItCarries(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	first := lobby.Presets()[1]
	for _, presetID := range []string{lobby.DefaultPresetID, "h2h_rapid", "ffa4_classical"} {
		t.Run(presetID, func(t *testing.T) {
			body := fmt.Sprintf(`{"type":"public","presetId":%q}`, presetID)
			lob := createdLobby(t, postCreateLobby(t, gs, token, body))
			if lob.PresetID != presetID {
				t.Fatalf("expected the lobby to name preset %q, got %q", presetID, lob.PresetID)
			}
			if presetID != first.ID && presetID != lobby.DefaultPresetID && lob.HouseRules != first.HouseRules {
				t.Fatalf("the queue presets are supposed to be rule-identical; this test no longer proves anything")
			}
		})
	}
}

// TestCreateLobbyDropsThePresetItDepartedFrom: a create that names a preset and overrides a rule
// in the same call produces a lobby on nobody's preset, so nothing downstream claims it plays a
// named ruleset it does not.
func TestCreateLobbyDropsThePresetItDepartedFrom(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token,
		`{"type":"private","presetId":"h2h_rapid","houseRules":{"turnTimerSec":30}}`))
	if lob.PresetID != "" {
		t.Fatalf("expected no recorded preset after an explicit override, got %q", lob.PresetID)
	}
	if lob.HouseRules.TurnTimerSec != 30 || !lob.HouseRules.SnapRace {
		t.Fatalf("the override must still have landed on top of the preset: %+v", lob.HouseRules)
	}
}

// TestMatchmadeSheetIsTheRulesetTheGamePlays is defect B end to end, and the only place both
// halves are visible at once: the lobby_state a seated player reads after match_found, and the
// game NewCambiaGameFromLobby builds from that same lobby. They were opposites - the sheet said
// draw-from-discard off, replaced abilities off, snap race off, caller's hand locked, and the
// game played all four the other way.
func TestMatchmadeSheetIsTheRulesetTheGamePlays(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	const queueID = "h2h_quickplay"
	preset, known := lobby.GetPreset(queueID)
	if !known {
		t.Fatalf("%s must be a preset", queueID)
	}

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	mux := http.NewServeMux()
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	hostID, guestID := uuid.New(), uuid.New()
	tokenHost := tokenFor(t, hostID)
	tokenGuest := tokenFor(t, guestID)

	body := fmt.Sprintf(`{"type":"matchmaking","queueID":%q}`, queueID)
	lobHost := createdLobby(t, postCreateLobby(t, gs, tokenHost, body))
	lobGuest := createdLobby(t, postCreateLobby(t, gs, tokenGuest, body))

	hostClient := dialWSClient(t, ctx, ts.URL, lobHost.ID.String(), tokenHost)
	defer hostClient.close()
	guestClient := dialWSClient(t, ctx, ts.URL, lobGuest.ID.String(), tokenGuest)
	defer guestClient.close()
	if hostClient.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("the host client never received lobby_state")
	}
	if guestClient.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("the guest client never received lobby_state")
	}

	if w := postSearch(t, gs, tokenHost, lobHost.ID); w.Code != http.StatusOK {
		t.Fatalf("search for the host lobby failed: %d %s", w.Code, w.Body.String())
	}
	if w := postSearch(t, gs, tokenGuest, lobGuest.ID); w.Code != http.StatusOK {
		t.Fatalf("search for the guest lobby failed: %d %s", w.Code, w.Body.String())
	}
	// Stand in for commitMatch, which takes both entries out of the queue before calling back.
	gs.Matchmaker.Dequeue(lobHost.ID)
	gs.Matchmaker.Dequeue(lobGuest.ID)

	party := func(id uuid.UUID) matchmaking.QueuedLobby {
		return matchmaking.QueuedLobby{
			LobbyID:     id,
			PlayerCount: 1,
			QueueID:     queueID,
			TargetCount: 2,
			IsRanked:    true,
			QueuedAt:    time.Now(),
		}
	}
	gs.HandleMatchFormed(matchmaking.MatchResult{
		HostLobbyID: lobHost.ID,
		Parties:     []matchmaking.QueuedLobby{party(lobHost.ID), party(lobGuest.ID)},
		QueueID:     queueID,
		TargetCount: 2,
		IsRanked:    true,
	})

	if hostClient.waitForType("match_found", 5*time.Second) == nil {
		t.Fatalf("the host client was never told the match formed")
	}
	// settle first: the post-match broadcast is a separate frame from match_found, and
	// lastFrameOfType returns as soon as it sees any lobby_state, including the connect-time one.
	hostClient.settle()

	sheet := ruleSheetOf(t, lastFrameOfType(hostClient, "lobby_state", 2*time.Second))
	if !sheet.SystemHost {
		t.Fatalf("the match lobby must be system-hosted by now")
	}
	if sheet.HouseRules != preset.HouseRules {
		t.Fatalf("the rule sheet is not the queue's ruleset:\n got %+v\nwant %+v", sheet.HouseRules, preset.HouseRules)
	}
	if sheet.PresetID != queueID {
		t.Fatalf("expected the sheet to name queue %q, got %q", queueID, sheet.PresetID)
	}

	// The other half: the game the match produces is built from that same lobby.
	stored, ok := gs.LobbyStore.GetLobby(lobHost.ID)
	if !ok {
		t.Fatalf("the match lobby is gone")
	}
	h, ok := gs.HubStore.GetHub(lobHost.ID)
	if !ok {
		t.Fatalf("the match lobby has no hub")
	}
	g := gs.NewCambiaGameFromLobby(ctx, stored, []uuid.UUID{hostID, guestID}, nil, h)
	if g == nil {
		t.Fatalf("failed to create the match's game instance")
	}
	if g.HouseRules != sheet.HouseRules {
		t.Fatalf("the game plays rules the sheet never showed:\ngame %+v\nsheet %+v", g.HouseRules, sheet.HouseRules)
	}
}
