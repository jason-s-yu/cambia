// internal/hub/preset_state_test.go
//
// What the rule sheet reads off the wire (cambia-1123). Two claims, both about lobby_state: it
// names the ruleset the lobby carries rather than leaving the client to recognise it by value,
// and the sheet a matchmade lobby shows after match_found is the ruleset its game is built from
// rather than the defaults the lobby object was constructed with.
package hub

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// ruleSheet is the part of a lobby_state payload the rule sheet renders.
type ruleSheet struct {
	HouseRules game.HouseRules `json:"house_rules"`
	PresetID   string          `json:"preset_id"`
	SystemHost bool            `json:"system_host"`
}

// sheetOf decodes the rule sheet out of a lobby_state envelope.
func sheetOf(t *testing.T, env *Envelope) ruleSheet {
	t.Helper()
	require.NotNil(t, env, "expected a lobby_state envelope")
	var sheet ruleSheet
	require.NoError(t, json.Unmarshal(env.Payload, &sheet))
	return sheet
}

// TestLobbyStateNamesTheRecordedPreset: the id travels, because the values cannot carry it.
// Every ranked queue plays the one ruleset MATCHMAKING.md 5.2 fixes, so a client that recognised
// a preset by comparing house rules would name whichever of the six identical queue presets its
// list returned first.
func TestLobbyStateNamesTheRecordedPreset(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	preset, known := lobby.GetPreset("h2h_rapid")
	require.True(t, known, "h2h_rapid must be a preset")
	lob.HouseRules = preset.HouseRules
	lob.PresetID = preset.ID

	h := NewHub(lob)
	conn := newFakeConn(host, "host")
	h.conns[host] = conn

	h.sendLobbyState(conn)
	sheet := sheetOf(t, findByType(drainEnvelopes(t, conn), "lobby_state"))

	assert.Equal(t, "h2h_rapid", sheet.PresetID, "lobby_state must name the preset the lobby carries")
	assert.Equal(t, preset.HouseRules, sheet.HouseRules)
}

// TestLobbyStateLeavesThePresetEmptyForACustomSheet: a lobby on nobody's preset says so. An
// absent key and a recorded empty id read the same to a client, and only one of them means "the
// service has no answer", so the key is always sent.
func TestLobbyStateLeavesThePresetEmptyForACustomSheet(t *testing.T) {
	host := uuid.New()
	lob := lobby.NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.HouseRules.TurnTimerSec = 42

	h := NewHub(lob)
	conn := newFakeConn(host, "host")
	h.conns[host] = conn

	h.sendLobbyState(conn)
	env := findByType(drainEnvelopes(t, conn), "lobby_state")
	require.NotNil(t, env)

	var payload map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(env.Payload, &payload))
	raw, present := payload["preset_id"]
	require.True(t, present, "preset_id must always be sent, empty included")
	assert.JSONEq(t, `""`, string(raw))
}

// TestMatchFoundSheetIsTheQueueRuleset is defect B: the read-only sheet a matchmade lobby shows
// used to be the defaults NewLobbyWithDefaults built the lobby with - draw-from-discard off,
// replaced abilities off, snap race off, the caller's hand locked - while the game the match
// produced ran the queue preset, which is the exact opposite on all four. The lobby now carries
// the queue's ruleset (handlers.CreateLobbyHandler and handlers.HandleMatchFormed), so the
// broadcast that follows match_found shows what the game is built from.
//
// The game factory lives in handlers, which imports this package; the ruleset it builds from is
// lobby.GetPreset(queueID).HouseRules, which is what this compares against.
// handlers.TestMatchmadeSheetIsTheRulesetTheGamePlays makes the same comparison against a real
// game instance.
func TestMatchFoundSheetIsTheQueueRuleset(t *testing.T) {
	const queueID = "h2h_quickplay"
	preset, known := lobby.GetPreset(queueID)
	require.True(t, known, "%s must be a preset", queueID)
	require.NotEqual(t, game.DefaultHouseRules(), preset.HouseRules,
		"a queue preset that equalled the defaults would make this test vacuous")

	leader := uuid.New()
	other := uuid.New()
	lob := lobby.NewLobbyWithDefaults(leader)
	lob.Type = "matchmaking"
	lob.Mode = "ranked"
	lob.QueueID = queueID
	lob.JoinUser(leader)
	lob.JoinUser(other)
	// The state handlers.HandleMatchFormed leaves behind: no player host, and the queue's
	// ruleset on the lobby.
	lob.Mu.Lock()
	lob.AdoptSystemHostUnsafe()
	lob.HouseRules = preset.HouseRules
	lob.LobbySettings = preset.Settings
	lob.PresetID = preset.ID
	lob.Mu.Unlock()

	h := NewHub(lob)
	h.QueueID = queueID
	h.IsRanked = true
	h.Phase = PhaseSearching
	leaderConn := newFakeConn(leader, "leader")
	otherConn := newFakeConn(other, "other")
	h.conns[leader] = leaderConn
	h.conns[other] = otherConn

	h.handleMatchFound(MatchNotice{LobbyID: lob.ID, Players: []MatchedPlayer{
		{UserID: leader, Username: "leader"},
		{UserID: other, Username: "other"},
	}})

	for name, conn := range map[string]*Connection{"leader": leaderConn, "other": otherConn} {
		envs := drainEnvelopes(t, conn)
		require.True(t, containsType(envs, "match_found"), "%s must be told the match formed", name)
		sheet := sheetOf(t, findByType(envs, "lobby_state"))
		assert.Equal(t, preset.HouseRules, sheet.HouseRules,
			"%s reads a rule sheet that is not the ruleset their game plays", name)
		assert.Equal(t, queueID, sheet.PresetID, "%s must be told which queue's ruleset it is", name)
		assert.True(t, sheet.SystemHost, "%s: a matchmade lobby has no player host", name)
	}
}
