// internal/lobby/preset_game_mode_test.go
//
// A preset carries a lobby shape as well as a ruleset (cambia-1099 Q1). CreateLobbyHandler has
// always derived GameMode from the preset it expands; update_rules recorded the id and left the
// mode alone, so a head_to_head lobby could come back naming a 4-player ruleset while still
// calling itself head_to_head, and the game built from it was persisted with the wrong mode.
package lobby

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// seatedLobby builds a custom lobby in the given mode with n joined members.
func seatedLobby(mode string, n int) *Lobby {
	lob := NewLobbyWithDefaults(uuid.New())
	lob.GameMode = mode
	for i := 0; i < n; i++ {
		lob.JoinUser(uuid.New())
	}
	return lob
}

// TestPresetAdoptsItsGameMode is the fix: a preset that fixes a player count fixes the mode with
// it, in both directions, and the recorded id names the ruleset that landed.
func TestPresetAdoptsItsGameMode(t *testing.T) {
	for _, tc := range []struct {
		name     string
		from     string
		seated   int
		presetID string
		want     string
	}{
		{"h2h lobby takes a 4-player ruleset", "head_to_head", 2, "ffa4_standard", "group_of_4"},
		{"4-player lobby takes a 2-player ruleset", "group_of_4", 2, "h2h_rapid", "head_to_head"},
		{"a full table takes the ruleset that seats it", "head_to_head", 4, "ffa4_classical", "group_of_4"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lob := seatedLobby(tc.from, tc.seated)

			require.NoError(t, lob.UpdateUnsafe(map[string]interface{}{"presetId": tc.presetID}))

			assert.Equal(t, tc.want, lob.GameMode, "the preset's player count fixes the lobby's mode")
			assert.Equal(t, tc.presetID, lob.PresetID, "the recorded id must name the ruleset that landed")
		})
	}
}

// TestDefaultPresetLeavesTheGameModeAlone: the default preset fixes no player count, so a lobby
// that adopts it keeps the mode its host chose. Same rule as POST /lobby/create.
func TestDefaultPresetLeavesTheGameModeAlone(t *testing.T) {
	lob := seatedLobby("group_of_4", 3)

	require.NoError(t, lob.UpdateUnsafe(map[string]interface{}{"presetId": DefaultPresetID}))

	assert.Equal(t, "group_of_4", lob.GameMode, "a preset that fixes no count must not move the mode")
	assert.Equal(t, DefaultPresetID, lob.PresetID)
}

// TestPresetRefusedWhenTheLobbyIsFullerThanItSeats is the other half: the seats are taken, so the
// only ways to fit a 2-player ruleset are to refuse it or to remove somebody, and nothing here
// removes anybody. The refusal must leave the lobby untouched, rules included.
func TestPresetRefusedWhenTheLobbyIsFullerThanItSeats(t *testing.T) {
	lob := seatedLobby("group_of_4", 3)
	rulesBefore := lob.HouseRules
	settingsBefore := lob.LobbySettings

	err := lob.UpdateUnsafe(map[string]interface{}{
		"presetId":   "h2h_rapid",
		"houseRules": map[string]interface{}{"turnTimerSec": float64(30)},
		"settings":   map[string]interface{}{"autoStart": false},
	})

	require.Error(t, err, "a 2-player ruleset cannot seat three players")
	assert.Contains(t, err.Error(), "h2h_rapid", "the refusal must name the ruleset it refused")
	assert.Equal(t, "group_of_4", lob.GameMode, "a refused update must not move the mode")
	assert.Empty(t, lob.PresetID, "a refused preset must not be recorded")
	assert.Equal(t, rulesBefore, lob.HouseRules, "a refused update writes nothing, not even the fields alongside the preset")
	assert.Equal(t, settingsBefore, lob.LobbySettings, "a refused update writes nothing")
}

// TestPresetCountsSeatsNotInvitations: an invitation is not a player at the table, so a lobby with
// one joined member and three outstanding invites still fits a 2-player ruleset.
func TestPresetCountsSeatsNotInvitations(t *testing.T) {
	lob := seatedLobby("group_of_4", 1)
	lob.Mu.Lock()
	for i := 0; i < 3; i++ {
		lob.InviteUser(uuid.New())
	}
	lob.Mu.Unlock()

	require.NoError(t, lob.UpdateUnsafe(map[string]interface{}{"presetId": "h2h_quickplay"}))
	assert.Equal(t, "head_to_head", lob.GameMode)
	assert.Equal(t, "h2h_quickplay", lob.PresetID)
}

// TestRankedLobbyStillRefusesAnyPreset is the control on the other refusal in the same path: a
// matchmade or ranked lobby's rules are its queue's, so the preset never gets far enough to move
// its mode (cambia-966).
func TestRankedLobbyStillRefusesAnyPreset(t *testing.T) {
	for _, tc := range []struct {
		name  string
		apply func(*Lobby)
	}{
		{"matchmaking type", func(l *Lobby) { l.Type = "matchmaking" }},
		{"ranked mode", func(l *Lobby) { l.Mode = "ranked" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lob := seatedLobby("head_to_head", 2)
			tc.apply(lob)

			err := lob.UpdateUnsafe(map[string]interface{}{"presetId": "ffa4_standard"})

			require.Error(t, err)
			assert.Equal(t, "head_to_head", lob.GameMode, "a refused preset must not move the mode")
			assert.Empty(t, lob.PresetID)
		})
	}
}
