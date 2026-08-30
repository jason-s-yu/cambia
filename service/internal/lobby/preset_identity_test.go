// internal/lobby/preset_identity_test.go
//
// Which preset a lobby carries (cambia-1123). The id is recorded because the rules cannot carry
// it: MATCHMAKING.md 5.2 fixes one ruleset for every ranked queue, so all six queue presets hold
// byte-identical house rules and recognising a preset by value names whichever one is listed
// first. These pin when UpdateUnsafe writes the id and when it lets it go.
package lobby

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
)

// TestQueuePresetsAreRuleIdentical is the premise the recorded id exists for. If this ever fails,
// the queues have grown distinguishable rulesets and value-matching would be a real answer; the
// recorded id is still the right one, but the reason written all over this change would be stale.
func TestQueuePresetsAreRuleIdentical(t *testing.T) {
	presets := Presets()
	if len(presets) < 3 {
		t.Fatalf("expected the default plus at least two queue presets, got %d", len(presets))
	}
	first := presets[1]
	for _, p := range presets[2:] {
		if p.HouseRules != first.HouseRules {
			t.Fatalf("preset %q no longer matches %q; queue rulesets have diverged", p.ID, first.ID)
		}
	}
	if first.HouseRules == presets[0].HouseRules {
		t.Fatalf("the queue ruleset must depart from the default one (MATCHMAKING.md 5.2)")
	}
}

// TestUpdateUnsafeRecordsThePreset: naming a preset stamps the lobby with that id, and the id is
// the one that was named rather than the first one holding those rules.
func TestUpdateUnsafeRecordsThePreset(t *testing.T) {
	for _, id := range []string{DefaultPresetID, "h2h_rapid", "ffa4_classical"} {
		t.Run(id, func(t *testing.T) {
			lob := NewLobbyWithDefaults(uuid.New())
			if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": id}); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if lob.PresetID != id {
				t.Fatalf("expected the lobby to carry preset %q, got %q", id, lob.PresetID)
			}
		})
	}
}

// TestUpdateUnsafeDropsThePresetItDepartedFrom: explicit fields alongside a preset land on top of
// it, so what the lobby ends up playing is not that preset and must not claim to be. Sending the
// same values the preset already holds is not a departure and keeps the id.
func TestUpdateUnsafeDropsThePresetItDepartedFrom(t *testing.T) {
	preset, _ := GetPreset("h2h_rapid")

	departed := NewLobbyWithDefaults(uuid.New())
	err := departed.UpdateUnsafe(map[string]interface{}{
		"presetId":   "h2h_rapid",
		"houseRules": map[string]interface{}{"turnTimerSec": float64(30)},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if departed.PresetID != "" {
		t.Fatalf("a sheet that departed from the preset in the same call must carry no id, got %q", departed.PresetID)
	}

	restated := NewLobbyWithDefaults(uuid.New())
	err = restated.UpdateUnsafe(map[string]interface{}{
		"presetId":   "h2h_rapid",
		"houseRules": map[string]interface{}{"snapRace": preset.HouseRules.SnapRace},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if restated.PresetID != "h2h_rapid" {
		t.Fatalf("restating a value the preset already holds is no departure, got %q", restated.PresetID)
	}
}

// TestUpdateUnsafeClearsThePresetOnALaterEdit: the first edit that moves a rule the preset covers,
// without naming a preset, is the host leaving it. Both halves of the sheet a preset expresses
// count - house rules and the auto-start setting.
func TestUpdateUnsafeClearsThePresetOnALaterEdit(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit map[string]interface{}
	}{
		{"a house rule", map[string]interface{}{"houseRules": map[string]interface{}{"turnTimerSec": float64(45)}}},
		{"the auto-start setting", map[string]interface{}{"settings": map[string]interface{}{"autoStart": false}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lob := NewLobbyWithDefaults(uuid.New())
			if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_rapid"}); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err := lob.UpdateUnsafe(tc.edit); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if lob.PresetID != "" {
				t.Fatalf("expected the preset to be released, got %q", lob.PresetID)
			}
		})
	}
}

// TestUpdateUnsafeKeepsThePresetThroughANoOpEdit: an update that names a preset's own values, or
// only circuit settings, has not left the preset. Circuit settings are not part of a preset at
// all (presets.go), so a lobby that turns circuit scoring on still plays the ruleset it named -
// and clearing the id there would hand the client back to value-matching, which is the misnaming
// this whole change removes.
func TestUpdateUnsafeKeepsThePresetThroughANoOpEdit(t *testing.T) {
	preset, _ := GetPreset("h2h_rapid")

	lob := NewLobbyWithDefaults(uuid.New())
	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_rapid"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	edits := []map[string]interface{}{
		{"houseRules": map[string]interface{}{"turnTimerSec": float64(preset.HouseRules.TurnTimerSec)}},
		{"circuit": map[string]interface{}{"enabled": true}},
		{"settings": map[string]interface{}{"autoStart": preset.Settings.AutoStart}},
	}
	for _, edit := range edits {
		if err := lob.UpdateUnsafe(edit); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if lob.PresetID != "h2h_rapid" {
			t.Fatalf("edit %+v released the preset; it changed no rule the preset expresses", edit)
		}
	}
	if !lob.Circuit.Enabled {
		t.Fatalf("the circuit edit must still have applied")
	}
}

// TestUpdateUnsafeSwitchesBetweenIdenticalPresets is defect A at the lobby: two queue presets
// holding identical rules are still two different presets, and the id follows the one that was
// named. Nothing about the rules changes here, so an implementation that only wrote the id when
// the sheet moved would leave the lobby claiming the preset it no longer carries.
func TestUpdateUnsafeSwitchesBetweenIdenticalPresets(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_rapid"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	before := lob.HouseRules

	if err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_quickplay"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if lob.HouseRules != before {
		t.Fatalf("the two queue presets hold the same rules; nothing should have moved")
	}
	if lob.PresetID != "h2h_quickplay" {
		t.Fatalf("expected the lobby to carry h2h_quickplay, got %q", lob.PresetID)
	}
}

// TestPresetIDSerialization pins the shape POST /lobby/create returns (service/doc/rest_api.md):
// presetId is a plain string with omitempty, so a lobby on nobody's preset leaves the key out
// rather than naming one, and a lobby carrying a preset reports its id.
func TestPresetIDSerialization(t *testing.T) {
	decode := func(lob *Lobby) map[string]json.RawMessage {
		t.Helper()
		raw, err := json.Marshal(lob)
		if err != nil {
			t.Fatalf("marshal lobby: %v", err)
		}
		var decoded map[string]json.RawMessage
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("decode marshalled lobby: %v", err)
		}
		return decoded
	}

	if _, present := decode(&Lobby{ID: uuid.New(), Type: "public"})["presetId"]; present {
		t.Fatalf("presetId must be omitted while the lobby is on no preset")
	}

	named := decode(&Lobby{ID: uuid.New(), Type: "public", PresetID: "h2h_rapid"})
	raw, present := named["presetId"]
	if !present {
		t.Fatalf("presetId is missing from a lobby that carries one")
	}
	var got string
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("presetId is not a JSON string: %s", raw)
	}
	if got != "h2h_rapid" {
		t.Fatalf("presetId = %q, want %q", got, "h2h_rapid")
	}
}
