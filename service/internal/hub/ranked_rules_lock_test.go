// internal/hub/ranked_rules_lock_test.go
package hub

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
