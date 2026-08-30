// internal/lobby/system_host_test.go
//
// The system host (cambia-1087): a matchmade lobby's host role belongs to the queue that seated
// the match in it, and the role never comes back to a player.
package lobby

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdoptSystemHostTakesTheRoleFromThePlayer(t *testing.T) {
	leader := uuid.New()
	lob := NewLobbyWithDefaults(leader)
	require.False(t, lob.SystemHosted(), "a fresh lobby is hosted by whoever created it")

	lob.Mu.Lock()
	lob.AdoptSystemHostUnsafe()
	lob.Mu.Unlock()

	assert.True(t, lob.SystemHosted())
	assert.Equal(t, SystemHostUserID, lob.HostUserID)
	assert.Equal(t, leader, lob.CreatorUserID, "the creator is not a role and does not move")
	assert.NotEqual(t, leader, lob.HostUserID, "the leader must not still hold host powers")
}

// TestSystemHostSurvivesDeparture is the case that would quietly undo the whole ticket: host
// migration hands the role to the longest-standing member when a host leaves, and a matchmade
// lobby must not acquire a player host that way when the party leader closes their tab.
func TestSystemHostSurvivesDeparture(t *testing.T) {
	leader := uuid.New()
	other := uuid.New()
	lob := NewLobbyWithDefaults(leader)
	lob.JoinUser(leader)
	lob.JoinUser(other)

	lob.Mu.Lock()
	lob.AdoptSystemHostUnsafe()
	lob.Mu.Unlock()

	require.True(t, lob.RemoveUser(leader), "the leader leaves deliberately")

	assert.True(t, lob.SystemHosted(), "the remaining player must not inherit the host role")
	assert.NotEqual(t, other, lob.HostUserID)
}

// TestPlayerHostStillMigrates is the control for the guard added to RemoveUser: an ordinary
// lobby still hands the role on, which is what keeps its settings reachable (cambia-835).
func TestPlayerHostStillMigrates(t *testing.T) {
	host := uuid.New()
	member := uuid.New()
	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(member)

	require.True(t, lob.RemoveUser(host))

	assert.Equal(t, member, lob.HostUserID, "an ordinary lobby still migrates its host")
	assert.False(t, lob.SystemHosted())
}

// TestAdoptSystemHostIsIdempotent: match formation is the only caller, but a requeued party can
// be seated more than once over a lobby's life and the second handover must be a no-op.
func TestAdoptSystemHostIsIdempotent(t *testing.T) {
	lob := NewLobbyWithDefaults(uuid.New())
	lob.Mu.Lock()
	lob.AdoptSystemHostUnsafe()
	lob.AdoptSystemHostUnsafe()
	lob.Mu.Unlock()

	assert.True(t, lob.SystemHosted())
}
