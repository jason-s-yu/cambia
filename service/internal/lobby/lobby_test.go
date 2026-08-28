// internal/lobby/lobby_test.go
//
// Membership release (cambia-807). RemoveUser had no callers at all, so nothing ever left a
// lobby and OnEmpty - the callback that deletes the lobby from the store - was unreachable.
package lobby

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRemoveUserReleasesMembership checks the core release: the user is gone from both the
// membership and ready maps, and a lobby that still has members is left alone.
func TestRemoveUserReleasesMembership(t *testing.T) {
	host := uuid.New()
	other := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(other)
	lob.Mu.Lock()
	lob.ReadyStates[other] = true
	lob.Mu.Unlock()

	emptied := 0
	lob.OnEmpty = func(uuid.UUID) { emptied++ }

	require.True(t, lob.RemoveUser(other), "removing a joined member must report the removal")

	lob.Mu.Lock()
	defer lob.Mu.Unlock()
	_, stillMember := lob.Users[other]
	assert.False(t, stillMember, "the departed user must not remain in the membership map")
	_, stillReady := lob.ReadyStates[other]
	assert.False(t, stillReady, "the departed user's ready state must go with them")
	assert.Equal(t, 1, lob.JoinedCount(), "the remaining member must still be joined")
	assert.Equal(t, 0, emptied, "a lobby that still has members must not fire OnEmpty")
}

// TestRemoveLastMemberFiresOnEmpty is the lifecycle half: the last member leaving is what makes
// a lobby collectable, and OnEmpty must fire exactly once, with the lock released so the
// callback can take it.
func TestRemoveLastMemberFiresOnEmpty(t *testing.T) {
	host := uuid.New()
	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	var emptiedWith []uuid.UUID
	lob.OnEmpty = func(id uuid.UUID) {
		// Taking the lock inside the callback proves RemoveUser is not holding it: the real
		// callback deletes from the lobby store and stops the hub.
		lob.Mu.Lock()
		lob.Mu.Unlock()
		emptiedWith = append(emptiedWith, id)
	}

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, []uuid.UUID{lob.ID}, emptiedWith, "the last member leaving must fire OnEmpty once, for this lobby")
}

// TestRemoveUserIsIdempotent guards teardown from running twice. OnEmpty stops the hub by
// closing a channel, so a second fire on a repeated or unknown leave would be fatal.
func TestRemoveUserIsIdempotent(t *testing.T) {
	host := uuid.New()
	stranger := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)

	emptied := 0
	lob.OnEmpty = func(uuid.UUID) { emptied++ }

	require.True(t, lob.RemoveUser(host))
	assert.False(t, lob.RemoveUser(host), "a repeated leave must report that nothing was removed")
	assert.False(t, lob.RemoveUser(stranger), "leaving a lobby you never joined must remove nothing")
	assert.Equal(t, 1, emptied, "teardown must run exactly once")
}

// TestRemoveUserCountsJoinedMembersForEmptiness covers the auto-invite case (cambia-771): the
// host is invited at creation, so map entries outlive the people who accepted them. A lobby
// holding nothing but unaccepted invitations has nobody left and must be collected.
func TestRemoveUserCountsJoinedMembersForEmptiness(t *testing.T) {
	host := uuid.New()
	invitee := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.Mu.Lock()
	lob.InviteUser(invitee)
	lob.Mu.Unlock()

	emptied := 0
	lob.OnEmpty = func(uuid.UUID) { emptied++ }

	require.True(t, lob.RemoveUser(host))
	assert.Equal(t, 1, emptied, "an unaccepted invitation must not keep a lobby alive")
}

// TestRemoveUserCancelsCountdown checks that a pending start does not survive the departure
// that invalidated it.
func TestRemoveUserCancelsCountdown(t *testing.T) {
	host := uuid.New()
	other := uuid.New()

	lob := NewLobbyWithDefaults(host)
	lob.JoinUser(host)
	lob.JoinUser(other)

	lob.Mu.Lock()
	started := lob.StartCountdownUnsafe(60, func(*Lobby) {})
	lob.Mu.Unlock()
	require.True(t, started, "two joined members should be able to start a countdown")

	require.True(t, lob.RemoveUser(other))

	lob.Mu.Lock()
	defer lob.Mu.Unlock()
	assert.Nil(t, lob.CountdownTimer, "leaving must cancel a countdown that can no longer complete")
}
