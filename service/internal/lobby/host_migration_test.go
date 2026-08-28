// internal/lobby/host_migration_test.go
//
// Host migration (cambia-835). Releasing membership on a deliberate leave (cambia-807) gave a
// host a way out of their own lobby, but HostUserID stayed pointing at them: the lobby was left
// with a host who was not a member, and every host-gated action was refused for everyone still
// in it. The role now moves to the earliest remaining joiner.
package lobby

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// hostOf reads the current host under the lock.
func hostOf(l *Lobby) uuid.UUID {
	l.Mu.Lock()
	defer l.Mu.Unlock()
	return l.HostUserID
}

// TestHostLeavingMigratesToEarliestJoiner is the headline: the successor is picked by join
// order, not by whatever the membership map happens to yield first.
func TestHostLeavingMigratesToEarliestJoiner(t *testing.T) {
	host := uuid.New()
	first := uuid.New()
	second := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(first)
	lob.JoinUser(second)

	require.True(t, lob.RemoveUser(host))

	assert.Equal(t, first, hostOf(lob), "the host role must land on the earliest remaining joiner")
	assert.NotEqual(t, host, hostOf(lob), "a departed host must not stay host")
}

// TestHostMigrationIsDeterministicAcrossLobbies pins the choice rather than the mechanism: map
// iteration is randomised per range, so a picker that read the map order would disagree with
// itself across identical lobbies.
func TestHostMigrationIsDeterministicAcrossLobbies(t *testing.T) {
	host := uuid.New()
	first := uuid.New()
	others := []uuid.UUID{uuid.New(), uuid.New(), uuid.New(), uuid.New()}

	for i := 0; i < 25; i++ {
		lob := NewLobbyWithDefaults(host)
		lob.JoinUser(host)
		lob.JoinUser(first)
		for _, o := range others {
			lob.JoinUser(o)
		}
		require.True(t, lob.RemoveUser(host))
		require.Equal(t, first, hostOf(lob), "every run must migrate to the same member")
	}
}

// TestHostMigrationSkipsUnacceptedInvites keeps the role on somebody who is actually present: an
// invitation nobody accepted is not a member and cannot run the lobby.
func TestHostMigrationSkipsUnacceptedInvites(t *testing.T) {
	host := uuid.New()
	member := uuid.New()
	invitee := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(member)
	lob.Mu.Lock()
	lob.InviteUser(invitee)
	lob.Mu.Unlock()

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, member, hostOf(lob), "an unaccepted invitation must not inherit the lobby")
}

// TestNonHostLeavingKeepsHost guards the other direction: an ordinary member leaving must not
// move the role.
func TestNonHostLeavingKeepsHost(t *testing.T) {
	host := uuid.New()
	member := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(member)

	require.True(t, lob.RemoveUser(member))
	assert.Equal(t, host, hostOf(lob), "a non-host departure must leave the host alone")
}

// TestLastMemberLeavingHasNoSuccessor covers the teardown case: with nobody left there is
// nothing to migrate to, and OnEmpty still fires exactly once.
func TestLastMemberLeavingHasNoSuccessor(t *testing.T) {
	host := uuid.New()
	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	emptied := 0
	lob.OnEmpty = func(uuid.UUID) { emptied++ }

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, 1, emptied, "the last member leaving must still tear the lobby down")
	assert.Equal(t, host, hostOf(lob), "a lobby with nobody left keeps its record of who hosted it")
}

// TestJoinOrderSurvivesAReconnect ties migration to the 807 rule that a lost socket keeps
// membership: rejoining a lobby you never left must not push you to the back of the queue, or a
// refresh would quietly change who inherits the lobby.
func TestJoinOrderSurvivesAReconnect(t *testing.T) {
	host := uuid.New()
	first := uuid.New()
	second := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(first)
	lob.JoinUser(second)

	// first refreshes their tab: the WebSocket upgrade re-marks them joined without ever
	// releasing membership.
	lob.JoinUser(first)

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, first, hostOf(lob), "a reconnect must not restamp join order")
}

// TestRejoiningAfterLeavingTakesANewPlace is the complement: a deliberate leave does drop the
// stamp, so coming back puts you behind everyone who stayed.
func TestRejoiningAfterLeavingTakesANewPlace(t *testing.T) {
	host := uuid.New()
	first := uuid.New()
	second := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(first)
	lob.JoinUser(second)

	require.True(t, lob.RemoveUser(first))
	lob.JoinUser(first)

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, second, hostOf(lob), "a member who left and came back joined after the ones who stayed")
}
