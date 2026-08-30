// internal/lobby/presets_test.go
//
// Ruleset presets (cambia-1088): the list's shape and the "presetId" expansion in UpdateUnsafe.
package lobby

import (
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// TestPresetsListShape pins what the dialog and the rule sheet are handed: the default first,
// then one entry per configured queue in the queue list's order, each naming a game mode the
// create handler accepts.
func TestPresetsListShape(t *testing.T) {
	presets := Presets()
	if len(presets) != len(matchmaking.QueueConfigs)+1 {
		t.Fatalf("expected %d presets (default + queues), got %d", len(matchmaking.QueueConfigs)+1, len(presets))
	}
	if presets[0].ID != DefaultPresetID {
		t.Fatalf("expected the default preset first, got %q", presets[0].ID)
	}
	if presets[0].GameMode != "" {
		t.Fatalf("the default preset must fix no game mode, got %q", presets[0].GameMode)
	}
	if presets[0].HouseRules != game.DefaultHouseRules() {
		t.Fatalf("the default preset must be the house rules a fresh lobby is built with")
	}

	want := matchmaking.OrderedQueueIDs()
	for i, id := range want {
		p := presets[i+1]
		if p.ID != id {
			t.Fatalf("preset %d: expected queue id %q, got %q", i+1, id, p.ID)
		}
		cfg := matchmaking.QueueConfigs[id]
		if p.Name != cfg.DisplayName || p.Name == "" {
			t.Fatalf("preset %q: expected display name %q, got %q", id, cfg.DisplayName, p.Name)
		}
		if p.Rounds != cfg.Rounds || p.Players != cfg.Players || p.Ranked != cfg.Ranked {
			t.Fatalf("preset %q: queue shape not carried through: %+v", id, p)
		}
		switch cfg.Players {
		case 2:
			if p.GameMode != "head_to_head" {
				t.Fatalf("preset %q: expected head_to_head, got %q", id, p.GameMode)
			}
		case 4:
			if p.GameMode != "group_of_4" {
				t.Fatalf("preset %q: expected group_of_4, got %q", id, p.GameMode)
			}
		}
	}
}

// TestQueuePresetsCarryTheRankedRuleset pins the four departures from the defaults that
// MATCHMAKING.md 5.2 specifies for every ranked queue, plus the queue's own reconnect grace.
// These are what a queue preset exists to carry: without them every preset would be the
// default ruleset under six different names.
func TestQueuePresetsCarryTheRankedRuleset(t *testing.T) {
	for _, id := range matchmaking.OrderedQueueIDs() {
		cfg := matchmaking.QueueConfigs[id]
		p, known := GetPreset(id)
		if !known {
			t.Fatalf("queue %q has no preset", id)
		}
		hr := p.HouseRules
		if !hr.AllowDrawFromDiscardPile {
			t.Errorf("preset %q: allowDrawFromDiscardPile must be on (MATCHMAKING.md 5.2)", id)
		}
		if !hr.AllowReplaceAbilities {
			t.Errorf("preset %q: allowReplaceAbilities must be on (MATCHMAKING.md 5.2)", id)
		}
		if !hr.SnapRace {
			t.Errorf("preset %q: snapRace must be on (MATCHMAKING.md 5.2)", id)
		}
		if hr.LockCallerHand {
			t.Errorf("preset %q: lockCallerHand must be off, it is the T1C fix (MATCHMAKING.md 5.1)", id)
		}
		if hr.NumJokers != 2 {
			t.Errorf("preset %q: expected a full 54-card deck, got numJokers %d", id, hr.NumJokers)
		}
		if hr.DisconnectGraceSec != cfg.DisconnectGraceSec {
			t.Errorf("preset %q: expected the queue's grace %d, got %d", id, cfg.DisconnectGraceSec, hr.DisconnectGraceSec)
		}
	}
}

// TestGetPresetUnknownID refuses anything that is neither the default nor a configured queue.
func TestGetPresetUnknownID(t *testing.T) {
	for _, id := range []string{"", "nope", "h2h_", "DEFAULT"} {
		if _, known := GetPreset(id); known {
			t.Fatalf("expected %q to be an unknown preset", id)
		}
	}
}

// TestUpdateUnsafeExpandsPreset is the update-path half: naming a preset fills the whole sheet,
// not just the fields the caller happened to send.
func TestUpdateUnsafeExpandsPreset(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_rapid"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want, _ := GetPreset("h2h_rapid")
	if lob.HouseRules != want.HouseRules {
		t.Fatalf("expected the preset's house rules, got %+v", lob.HouseRules)
	}
	if lob.LobbySettings != want.Settings {
		t.Fatalf("expected the preset's lobby settings, got %+v", lob.LobbySettings)
	}
}

// TestUpdateUnsafePresetThenExplicitFields pins the layering: a sheet sent alongside a preset is
// the host's departure from it and lands on top, not under.
func TestUpdateUnsafePresetThenExplicitFields(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	err := lob.UpdateUnsafe(map[string]interface{}{
		"presetId":   "h2h_rapid",
		"houseRules": map[string]interface{}{"lockCallerHand": true, "turnTimerSec": float64(30)},
		"settings":   map[string]interface{}{"autoStart": false},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !lob.HouseRules.LockCallerHand {
		t.Fatalf("expected the explicit lockCallerHand to win over the preset's")
	}
	if lob.HouseRules.TurnTimerSec != 30 {
		t.Fatalf("expected turnTimerSec 30, got %d", lob.HouseRules.TurnTimerSec)
	}
	if !lob.HouseRules.SnapRace {
		t.Fatalf("expected the preset's snapRace to survive fields the caller did not send")
	}
	if lob.LobbySettings.AutoStart {
		t.Fatalf("expected the explicit autoStart to win over the preset's")
	}
}

// TestUpdateUnsafeRejectsBadPreset covers the two refusals: an id naming nothing, and a value
// that is not a string at all.
func TestUpdateUnsafeRejectsBadPreset(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	before := lob.HouseRules

	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "nope"}); err == nil {
		t.Fatalf("expected an unknown preset id to be refused")
	}
	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": float64(3)}); err == nil {
		t.Fatalf("expected a non-string preset id to be refused")
	}
	if lob.HouseRules != before {
		t.Fatalf("a refused preset must leave the rules exactly as they were")
	}
}

// TestUpdateUnsafeRefusesPresetForRankedLobby: a queue owns a ranked or matchmade lobby's rules,
// so a preset cannot reach one even through a caller that does not gate first (hub.go does).
func TestUpdateUnsafeRefusesPresetForRankedLobby(t *testing.T) {
	for _, tc := range []struct {
		name      string
		lobbyType string
		mode      string
	}{
		{"matchmaking type", "matchmaking", "casual"},
		{"ranked mode", "public", "ranked"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lob := NewLobbyWithDefaults(uuid.New())
			lob.Type = tc.lobbyType
			lob.Mode = tc.mode
			before := lob.HouseRules
			if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_rapid"}); err == nil {
				t.Fatalf("expected the preset to be refused")
			}
			if lob.HouseRules != before {
				t.Fatalf("a refused preset must leave the rules exactly as they were")
			}
		})
	}
}

// TestUpdateUnsafeWithoutPresetIsUnchanged guards the restructured field-by-field path: an
// update carrying no presetId still applies exactly what it names and nothing else.
func TestUpdateUnsafeWithoutPresetIsUnchanged(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	before := lob.HouseRules
	err := lob.UpdateUnsafe(map[string]interface{}{
		"houseRules": map[string]interface{}{"turnTimerSec": float64(45)},
		"settings":   map[string]interface{}{"autoStart": false},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lob.HouseRules.TurnTimerSec != 45 {
		t.Fatalf("expected turnTimerSec 45, got %d", lob.HouseRules.TurnTimerSec)
	}
	before.TurnTimerSec = 45
	if lob.HouseRules != before {
		t.Fatalf("no other rule may move: got %+v", lob.HouseRules)
	}
	if lob.LobbySettings.AutoStart {
		t.Fatalf("expected autoStart off")
	}
}
