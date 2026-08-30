// internal/hub/ranked_rules_lock_test.go
package hub

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// TestMatchmakingLobbyCannotUpdateRules is the cambia-966 regression for the review finding that
// a ranked matchmaking lobby's host could still edit house rules over the WS: the queue config,
// not the host, is what the matchmaker paired the players on and what NewCambiaGameFromLobby
// builds the game from, so a late edit would leave the two sides disagreeing about what they
// agreed to play.
func TestMatchmakingLobbyCannotUpdateRules(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)
	h.Lobby.Mu.Lock()
	h.Lobby.Type = "matchmaking"
	h.Lobby.Mode = "ranked"
	h.Lobby.Mu.Unlock()

	h.dispatch(updateRulesMsg(host, h.seq))

	errEnv := findByType(drainEnvelopes(t, hostConn), "error")
	require.NotNil(t, errEnv, "the host of a matchmaking lobby must be refused")
	assert.Contains(t, string(errEnv.Payload), "locked", "the refusal must name the ranked lock")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.False(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "a refused update must change nothing")
}

// TestRankedPartyLobbyCannotUpdateRules covers the non-matchmaking side of the same rule: a
// standing public/private lobby that queued its party into a ranked queue (lob.Mode "ranked",
// cambia-966 item 2) is just as locked as one typed "matchmaking".
func TestRankedPartyLobbyCannotUpdateRules(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)
	h.Lobby.Mu.Lock()
	require.Equal(t, "private", h.Lobby.Type, "precondition: not a matchmaking-typed lobby")
	h.Lobby.Mode = "ranked"
	h.Lobby.Mu.Unlock()

	h.dispatch(updateRulesMsg(host, h.seq))

	errEnv := findByType(drainEnvelopes(t, hostConn), "error")
	require.NotNil(t, errEnv, "the host of a ranked party lobby must be refused")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.False(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "a refused update must change nothing")
}

// TestCasualLobbyStillCanUpdateRules pins the negative: a plain casual lobby (the default
// newTwoMemberHub shape) is unaffected by the lock and keeps working exactly as
// TestMigratedHostCanUpdateRules already checks.
func TestCasualLobbyStillCanUpdateRules(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)
	h.Lobby.Mu.Lock()
	require.Equal(t, "casual", h.Lobby.Mode, "precondition: casual lobby")
	h.Lobby.Mu.Unlock()

	h.dispatch(updateRulesMsg(host, h.seq))

	assert.Nil(t, findByType(drainEnvelopes(t, hostConn), "error"), "a casual lobby's host must not be refused")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.True(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "the update must apply")
}

// wireRulesLocked reads the rules_locked flag out of a lobby_state payload, the way the client's
// rule sheet does. A pointer, so an absent key fails loudly rather than reading as false.
func wireRulesLocked(t *testing.T, env *Envelope) bool {
	t.Helper()
	require.NotNil(t, env, "expected a lobby_state envelope")
	var payload struct {
		RulesLocked *bool `json:"rules_locked"`
	}
	require.NoError(t, json.Unmarshal(env.Payload, &payload))
	require.NotNil(t, payload.RulesLocked, "rules_locked must always be sent")
	return *payload.RulesLocked
}

// rankedQueueID names a configured ranked queue, read off the queue table rather than written
// here so the fixture cannot outlive the queue it names.
func rankedQueueID(t *testing.T) string {
	t.Helper()
	for _, id := range matchmaking.OrderedQueueIDs() {
		if cfg, known := matchmaking.GetQueueConfig(id); known && cfg.Ranked {
			return id
		}
	}
	t.Fatal("no ranked queue is configured")
	return ""
}

// TestLobbyStateReportsTheRankedLock is the cambia-1099 K2 half of the same rule, on the wire: a
// standing private lobby that queued its party into a ranked queue is locked by its mode, and
// mode is the one field lobby_state never carried. A client re-deriving the lock from lobby_type
// alone saw "private", offered its host the editable rule sheet, and every Save came back with
// the refusal below.
func TestLobbyStateReportsTheRankedLock(t *testing.T) {
	queueID := rankedQueueID(t)
	h, host, _, hostConn, _ := newTwoMemberHub(t)
	h.Lobby.Mu.Lock()
	require.Equal(t, "private", h.Lobby.Type, "precondition: not a matchmaking-typed lobby")
	h.Lobby.QueueID = queueID
	// What CreateLobbyHandler derives from the queue config for a standing lobby carrying a
	// queue id (cambia-966): the id alone is not the lock, the mode it implies is.
	h.Lobby.Mode = "ranked"
	h.Lobby.Mu.Unlock()

	h.sendLobbyState(hostConn)
	assert.True(t, wireRulesLocked(t, findByType(drainEnvelopes(t, hostConn), "lobby_state")),
		"a lobby queued into %s must report its rules locked", queueID)

	// The flag has to be the refusal, not a second opinion about it.
	h.dispatch(updateRulesMsg(host, h.seq))
	assert.NotNil(t, findByType(drainEnvelopes(t, hostConn), "error"),
		"the same lobby's update_rules must be refused")
}

// TestLobbyStateReportsACustomLobbyUnlocked pins the negative: a plain custom lobby's host keeps
// an editable sheet, so the flag names the queue lock rather than switching the sheet off.
func TestLobbyStateReportsACustomLobbyUnlocked(t *testing.T) {
	h, _, _, hostConn, _ := newTwoMemberHub(t)
	h.Lobby.Mu.Lock()
	require.Equal(t, "casual", h.Lobby.Mode, "precondition: casual lobby")
	require.Empty(t, h.Lobby.QueueID, "precondition: on no queue")
	h.Lobby.Mu.Unlock()

	h.sendLobbyState(hostConn)
	assert.False(t, wireRulesLocked(t, findByType(drainEnvelopes(t, hostConn), "lobby_state")),
		"a custom lobby must report its rules editable")
}
