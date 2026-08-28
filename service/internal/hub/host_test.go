// internal/hub/host_test.go
//
// Live host derivation (cambia-835). Connection.IsHost was stamped at WebSocket accept and never
// revisited, so once the host role moved the permission checks were reading a snapshot of the
// past: the promoted host was refused and the departed one was still authorised.
package hub

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// updateRulesMsg builds an update_rules frame that flips allowDrawFromDiscardPile on.
func updateRulesMsg(userID uuid.UUID, seq uint64) ClientMsg {
	body, _ := json.Marshal(map[string]interface{}{
		"rules": map[string]interface{}{
			"houseRules": map[string]interface{}{"allowDrawFromDiscardPile": true},
		},
	})
	return ClientMsg{UserID: userID, LastSeq: seq, Type: "update_rules", Body: body}
}

// newTwoMemberHub builds a hub whose lobby has host and member joined in that order, both
// connected, with auto-start off so a stray ready cannot drag the hub into a countdown.
func newTwoMemberHub(t *testing.T) (*Hub, uuid.UUID, uuid.UUID, *Connection, *Connection) {
	t.Helper()

	host := uuid.New()
	member := uuid.New()

	lob := lobby.NewLobbyWithDefaults(host)
	lob.LobbySettings.AutoStart = false
	lob.JoinUser(host)
	lob.JoinUser(member)

	h := NewHub(lob)
	hostConn := newFakeConn(host, "host")
	memberConn := newFakeConn(member, "member")
	h.conns[host] = hostConn
	h.conns[member] = memberConn

	return h, host, member, hostConn, memberConn
}

// TestMigratedHostCanUpdateRules is the user-visible failure: after the host left, nobody in the
// lobby could change a setting, because the remaining connection still carried IsHost=false and
// HostUserID still named someone who was gone.
func TestMigratedHostCanUpdateRules(t *testing.T) {
	h, host, member, _, memberConn := newTwoMemberHub(t)

	require.True(t, h.Lobby.RemoveUser(host), "the host leaves deliberately")
	require.Equal(t, member, h.Lobby.HostUserID, "precondition: the role migrated")

	h.dispatch(updateRulesMsg(member, h.seq))

	frames := drainEnvelopes(t, memberConn)
	assert.Nil(t, findByType(frames, "error"), "the migrated host must not be refused")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.True(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "the migrated host's rule change must apply")
}

// TestDepartedHostCannotUpdateRules is the same check from the other side: leaving gives the role
// up, and a cached flag would have kept the leaver in charge of a lobby they are not in.
func TestDepartedHostCannotUpdateRules(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)

	require.True(t, h.Lobby.RemoveUser(host))

	h.dispatch(updateRulesMsg(host, h.seq))

	errEnv := findByType(drainEnvelopes(t, hostConn), "error")
	require.NotNil(t, errEnv, "a departed host must be refused")
	assert.Contains(t, string(errEnv.Payload), "only the host", "the refusal must be the host gate")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.False(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "a refused update must change nothing")
}

// TestNonHostCannotUpdateRules keeps the gate closed for an ordinary member, so the fix is not
// just "everyone is the host now".
func TestNonHostCannotUpdateRules(t *testing.T) {
	h, _, member, _, memberConn := newTwoMemberHub(t)

	h.dispatch(updateRulesMsg(member, h.seq))

	errEnv := findByType(drainEnvelopes(t, memberConn), "error")
	require.NotNil(t, errEnv, "a non-host must be refused")
	assert.Contains(t, string(errEnv.Payload), "only the host")
	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.False(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile)
}

// TestStartGameGateFollowsTheLiveHost covers the second host-gated lobby action through the same
// derivation, so the two cannot drift apart.
func TestStartGameGateFollowsTheLiveHost(t *testing.T) {
	h, host, member, _, memberConn := newTwoMemberHub(t)
	require.True(t, h.Lobby.RemoveUser(host))

	h.dispatch(ClientMsg{UserID: member, LastSeq: h.seq, Type: "start_game"})

	errEnv := findByType(drainEnvelopes(t, memberConn), "error")
	require.NotNil(t, errEnv, "the sole remaining member cannot be ready, so a start is still refused")
	assert.NotContains(t, string(errEnv.Payload), "only the host",
		"the migrated host must clear the host gate and be refused on readiness instead")
}

// TestCancelSearchGateFollowsTheLiveHost covers the searching-phase gate, the third place the
// cached flag was read.
func TestCancelSearchGateFollowsTheLiveHost(t *testing.T) {
	h, host, member, _, memberConn := newTwoMemberHub(t)
	require.True(t, h.Lobby.RemoveUser(host))
	h.Phase = PhaseSearching

	h.dispatch(ClientMsg{UserID: member, LastSeq: h.seq, Type: "cancel_search"})

	frames := drainEnvelopes(t, memberConn)
	assert.Nil(t, findByType(frames, "error"), "the migrated host must be able to cancel the search")
	assert.Equal(t, PhaseOpen, h.Phase, "cancelling must return the hub to the open phase")
}

// TestLobbySnapshotReportsTheMigratedHost is what reaches the client: settings editability is
// rendered from your_is_host, so the migration is only real once it is broadcast.
func TestLobbySnapshotReportsTheMigratedHost(t *testing.T) {
	h, host, member, _, memberConn := newTwoMemberHub(t)
	require.True(t, h.Lobby.RemoveUser(host))
	delete(h.conns, host)

	h.broadcastLobbyUpdate()

	snapshot := findByType(drainEnvelopes(t, memberConn), "lobby_state")
	require.NotNil(t, snapshot, "the remaining member must receive a refreshed snapshot")
	payload := payloadOf(t, *snapshot)
	assert.Equal(t, member.String(), payload["host_id"], "the snapshot must name the new host")
	assert.Equal(t, true, payload["your_is_host"], "the new host's own snapshot must say so")
}
