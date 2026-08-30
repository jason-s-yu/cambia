// internal/hub/rules_broadcast_test.go
//
// An accepted update_rules reaches every seat, not just the host who sent it (cambia-1099 Q10).
// The handler applied the edit and emitted nothing, so a second player's rule sheet went on showing
// the rules the lobby had before - and the preset name and game mode with them - until a reload,
// while a chat message on the same socket arrived immediately.
package hub

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// presetRulesMsg builds an update_rules frame naming a whole ruleset.
func presetRulesMsg(userID uuid.UUID, seq uint64, presetID string) ClientMsg {
	body, _ := json.Marshal(map[string]interface{}{
		"rules": map[string]interface{}{"presetId": presetID},
	})
	return ClientMsg{UserID: userID, LastSeq: seq, Type: "update_rules", Body: body}
}

// TestUpdateRulesReachesTheOtherSeats is the user-visible failure: the seat that did not send the
// edit is the one that has to be told about it.
func TestUpdateRulesReachesTheOtherSeats(t *testing.T) {
	h, host, _, _, memberConn := newTwoMemberHub(t)

	h.dispatch(presetRulesMsg(host, h.seq, "h2h_rapid"))

	env := findByType(drainEnvelopes(t, memberConn), "lobby_state")
	require.NotNil(t, env, "a non-host seat must be sent the rules it is about to play by")
	payload := payloadOf(t, *env)

	assert.Equal(t, "h2h_rapid", payload["preset_id"], "the snapshot must name the ruleset that landed")
	houseRules, ok := payload["house_rules"].(map[string]interface{})
	require.True(t, ok, "the snapshot must carry the rule sheet")
	assert.Equal(t, true, houseRules["allowDrawFromDiscardPile"],
		"the broadcast rules must be the new ones, not the sheet the lobby opened on")
	settings, ok := payload["settings"].(map[string]interface{})
	require.True(t, ok, "the snapshot must carry the lobby settings the same message can move")
	assert.Equal(t, true, settings["autoStart"], "the preset's auto-start travels with its rules")
}

// TestUpdateRulesAlsoAnswersTheHost: the host edits from the same rule sheet everyone else reads,
// so the broadcast includes the sender rather than leaving them to assume it worked.
func TestUpdateRulesAlsoAnswersTheHost(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)

	h.dispatch(presetRulesMsg(host, h.seq, "h2h_rapid"))

	frames := drainEnvelopes(t, hostConn)
	require.Nil(t, findByType(frames, "error"), "the host's own edit must be accepted")
	require.NotNil(t, findByType(frames, "lobby_state"), "the host is served the same snapshot")
}

// TestRefusedUpdateRulesBroadcastsNothing: a refused edit changed nothing, so there is nothing to
// tell the table. Sending a snapshot anyway would have every seat re-render on a non-event.
func TestRefusedUpdateRulesBroadcastsNothing(t *testing.T) {
	h, _, member, _, memberConn := newTwoMemberHub(t)

	h.dispatch(presetRulesMsg(member, h.seq, "h2h_rapid"))

	frames := drainEnvelopes(t, memberConn)
	require.NotNil(t, findByType(frames, "error"), "a non-host must still be refused")
	assert.Nil(t, findByType(frames, "lobby_state"), "a refused edit must not broadcast a snapshot")
}

// TestUpdateRulesBroadcastsTheAdoptedGameMode ties the two halves of this change together: a
// preset that fixes a player count moves the lobby's mode (cambia-1099 Q1), and the mode is only
// real to the table once it is broadcast.
func TestUpdateRulesBroadcastsTheAdoptedGameMode(t *testing.T) {
	h, host, _, _, memberConn := newTwoMemberHub(t)
	require.Equal(t, "head_to_head", h.Lobby.GameMode, "precondition: a default lobby is head to head")

	h.dispatch(presetRulesMsg(host, h.seq, "ffa4_standard"))

	env := findByType(drainEnvelopes(t, memberConn), "lobby_state")
	require.NotNil(t, env)
	payload := payloadOf(t, *env)
	assert.Equal(t, "group_of_4", payload["game_mode"], "the preset's player count fixes the mode every seat reads")
	assert.Equal(t, "ffa4_standard", payload["preset_id"])
}
