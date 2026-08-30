// internal/hub/system_host_test.go
//
// A matchmade lobby is run by its queue, not by a player (cambia-1087). Every quick play queue
// is ranked, so the settings the match runs on are the queue's and the start is the ready
// check's; the host role that used to survive match formation is handed to the system, and these
// tests pin what that means at the hub's message boundary.
package hub

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/lobby"
)

// newSystemHostedHub builds the state HandleMatchFormed leaves behind: a matchmaking lobby with
// two seated, ready players and no player host. Auto-start is left on, which is what actually
// starts a matchmade game, so the hub is driven with dispatch() rather than a live Run loop.
func newSystemHostedHub(t *testing.T) (*Hub, uuid.UUID, uuid.UUID, *Connection, *Connection) {
	t.Helper()

	leader := uuid.New()
	other := uuid.New()

	lob := lobby.NewLobbyWithDefaults(leader)
	lob.Type = "matchmaking"
	lob.Mode = "ranked"
	lob.QueueID = "h2h_quickplay"
	lob.JoinUser(leader)
	lob.JoinUser(other)

	lob.Mu.Lock()
	lob.AdoptSystemHostUnsafe()
	lob.Mu.Unlock()

	h := NewHub(lob)
	h.Phase = PhaseReadyCheck
	leaderConn := newFakeConn(leader, "leader")
	otherConn := newFakeConn(other, "other")
	h.conns[leader] = leaderConn
	h.conns[other] = otherConn

	return h, leader, other, leaderConn, otherConn
}

// TestSystemHostedLobbyRefusesStartGame covers the power the party leader used to keep after the
// match formed: forcing everybody else's ranked game to start. The refusal names what does start
// it, since no player can take the host role back.
func TestSystemHostedLobbyRefusesStartGame(t *testing.T) {
	h, leader, other, leaderConn, otherConn := newSystemHostedHub(t)

	for _, tc := range []struct {
		name string
		user uuid.UUID
		conn *Connection
	}{
		{"party leader", leader, leaderConn},
		{"other party", other, otherConn},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h.dispatch(ClientMsg{UserID: tc.user, LastSeq: h.seq, Type: "start_game"})

			errEnv := findByType(drainEnvelopes(t, tc.conn), "error")
			require.NotNil(t, errEnv, "nobody may force a matchmade match to start")
			assert.Contains(t, string(errEnv.Payload), "starts on its own",
				"the refusal must point at the ready check, not at a host nobody is")
			assert.Equal(t, PhaseReadyCheck, h.Phase, "the refused frame must not have started anything")
		})
	}
}

// TestSystemHostedLobbyStartsOnReadyCheck is the other half of the same rule: taking start_game
// away from the players must not leave the match unable to begin. Auto-start is what drives it,
// and it needs no host.
func TestSystemHostedLobbyStartsOnReadyCheck(t *testing.T) {
	h, leader, other, _, _ := newSystemHostedHub(t)

	h.dispatch(ClientMsg{UserID: leader, LastSeq: h.seq, Type: "ready"})
	require.Equal(t, PhaseReadyCheck, h.Phase, "one ready seat is not a full table")

	h.dispatch(ClientMsg{UserID: other, LastSeq: h.seq, Type: "ready"})
	assert.Equal(t, PhaseCountdown, h.Phase, "the last ready seat must begin the countdown with no host involved")
}

// TestSystemHostedLobbyRefusesRuleEdits pins the reason a matchmade lobby refuses update_rules.
// The queue's config is what paired the players and what the game inherits, so the answer has to
// be the lock rather than the host gate: cambia-966 checked the roles first, which told every
// player in a hostless lobby that somebody else was in charge of rules nobody could change.
func TestSystemHostedLobbyRefusesRuleEdits(t *testing.T) {
	h, leader, _, leaderConn, _ := newSystemHostedHub(t)

	h.dispatch(updateRulesMsg(leader, h.seq))

	errEnv := findByType(drainEnvelopes(t, leaderConn), "error")
	require.NotNil(t, errEnv, "a matchmade lobby's rules are fixed")
	assert.Contains(t, string(errEnv.Payload), "locked by its queue",
		"the refusal must name the queue, not a host role")

	h.Lobby.Mu.Lock()
	defer h.Lobby.Mu.Unlock()
	assert.False(t, h.Lobby.HouseRules.AllowDrawFromDiscardPile, "the edit must not have landed")
}

// TestSystemHostedSnapshotHasNoPlayerHost is what the client renders from: no seat is flagged
// is_host, nobody's own snapshot says your_is_host, and system_host says the nil host_id is
// deliberate rather than a lobby whose host has not loaded yet.
func TestSystemHostedSnapshotHasNoPlayerHost(t *testing.T) {
	h, leader, other, _, _ := newSystemHostedHub(t)

	for _, uid := range []uuid.UUID{leader, other} {
		snap := h.buildLobbySnapshot(uid)
		assert.Equal(t, false, snap["your_is_host"], "no seat holds the host role")
		assert.Equal(t, true, snap["system_host"], "the snapshot must say the system holds it")
		assert.Equal(t, uuid.Nil.String(), snap["host_id"], "host_id is the documented sentinel")

		status, ok := snap["lobby_status"].(map[string]interface{})
		require.True(t, ok, "lobby_status must be present")
		users, ok := status["users"].([]map[string]interface{})
		require.True(t, ok, "the roster must be present")
		require.Len(t, users, 2)
		for _, u := range users {
			assert.Equal(t, false, u["is_host"], "no seat renders a Host badge")
		}
	}
}

// TestPlayerHostedLobbyKeepsItsHost is the control: the system host is scoped to a matchmade
// lobby, and an ordinary lobby's host keeps every power this ticket took away.
func TestPlayerHostedLobbyKeepsItsHost(t *testing.T) {
	h, host, _, hostConn, _ := newTwoMemberHub(t)

	snap := h.buildLobbySnapshot(host)
	assert.Equal(t, true, snap["your_is_host"])
	assert.Equal(t, false, snap["system_host"])

	h.dispatch(updateRulesMsg(host, h.seq))
	assert.Nil(t, findByType(drainEnvelopes(t, hostConn), "error"), "a real host still sets the rules")
}
