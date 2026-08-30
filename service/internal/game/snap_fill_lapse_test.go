// internal/game/snap_fill_lapse_test.go
// A snap fill that lapses has to reach the client that is holding the prompt for it. The prompt
// (player_snap_move_required) blocks every other action the snapper can send, and the only thing
// that clears it is a snapshot: the obligation is server state, carried per player in
// ObfGameState.SnapMoves, and the client re-reads that whole list from every sync. Every lapse path
// used to return silently, so the snapper sat on a prompt for a debt the server had already written
// off, unable to play (cambia-1118).
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// requireSnapFillCleared asserts the snapper was sent a snapshot that no longer owes a fill, which
// is what clears the client's pendingSnapMove.
func requireSnapFillCleared(t *testing.T, mb *mockBroadcaster, snapperID uuid.UUID) {
	t.Helper()
	mb.mu.Lock()
	defer mb.mu.Unlock()
	for i := len(mb.playerEvents[snapperID]) - 1; i >= 0; i-- {
		ev := mb.playerEvents[snapperID][i]
		if ev.Type != EventPrivateSyncState {
			continue
		}
		require.NotNil(t, ev.State, "a sync event carries the snapshot")
		for _, sm := range ev.State.SnapMoves {
			if sm.SnapperID == snapperID {
				t.Fatal("the snapshot still owes the snapper's fill, so the client keeps its prompt")
			}
		}
		return
	}
	t.Fatal("the snapper was sent no state update, so its prompt never clears")
}

// snapperAndVictim seats the snapper as the acting player, so they can still be handed the turn
// actions a test needs after the snap.
func snapperAndVictim(t *testing.T, g *CambiaGame, players []*models.Player) (snapper, victim *models.Player) {
	t.Helper()
	snapper = currentTurnPlayer(g)
	victim = otherPlayer(t, players, snapper.ID)
	return snapper, victim
}

// TestSnapFillLapseNotifiesWhenVictimHandLocks: the victim calls Cambia while the fill is still
// outstanding, so it cannot be paid into their frozen hand (RULES.md 3C).
func TestSnapFillLapseNotifiesWhenVictimHandLocks(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim := currentTurnPlayer(g)
	snapper := otherPlayer(t, players, victim.ID)
	snapperIdx := g.PlayerToEngine[snapper.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	g.HandlePlayerAction(victim.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.handLocked(g.PlayerToEngine[victim.ID]))

	givenUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[0]
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": givenUUID.String(), "idx": float64(0)},
	})

	require.False(t, g.owesSnapFill(snapper.ID), "the obligation lapses against a locked hand")
	assert.Nil(t, mb.findEventByType(EventPlayerSnapMove), "no card moved, so no move event fires")
	requireSnapFillCleared(t, mb, snapper.ID)
}

// TestSnapFillLapseNotifiesWhenVictimHandFull: the victim drew back up to a full hand while the
// fill was outstanding, so there is no slot left to fill.
func TestSnapFillLapseNotifiesWhenVictimHandFull(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper, victim := snapperAndVictim(t, g, players)
	snapperIdx := g.PlayerToEngine[snapper.ID]
	victimIdx := g.PlayerToEngine[victim.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	for g.Engine.Players[victimIdx].HandLen < engine.MaxHandSize {
		slot := g.Engine.Players[victimIdx].HandLen
		g.Engine.Players[victimIdx].Hand[slot] = engine.NewCard(engine.SuitClubs, engine.RankTwo)
		g.Engine.Players[victimIdx].HandLen++
		g.CardTracker.Players[victimIdx].HandUUIDs[slot] = uuid.New()
	}
	g.syncPlayerHandsFromEngine()

	givenUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[0]
	snapperBefore := int(g.Engine.Players[snapperIdx].HandLen)
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": givenUUID.String(), "idx": float64(0)},
	})

	require.False(t, g.owesSnapFill(snapper.ID), "the obligation lapses against a full hand")
	assert.Equal(t, snapperBefore, int(g.Engine.Players[snapperIdx].HandLen), "the snapper keeps the card they would have given up")
	requireSnapFillCleared(t, mb, snapper.ID)
}

// TestSnapFillLapseNotifiesWhenSnapperEmpties: the deadline finds a snapper with nothing left to
// give, so it settles nothing and the debt is written off.
func TestSnapFillLapseNotifiesWhenSnapperEmpties(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper, victim := snapperAndVictim(t, g, players)
	snapperIdx := g.PlayerToEngine[snapper.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	emptyHand(g, snapperIdx)
	mb.clear()

	g.mu.Lock()
	fill := g.snapFills[snapper.ID]
	require.NotNil(t, fill)
	g.autoSnapFill(fill)
	g.mu.Unlock()

	require.False(t, g.owesSnapFill(snapper.ID), "a fill with nothing to pay it lapses")
	requireSnapFillCleared(t, mb, snapper.ID)
}

// TestUnpayableSnapFillLapseNotifies: the same debt written off by the action gate instead of a
// deadline, on a table with no turn timer where no deadline is ever armed.
func TestUnpayableSnapFillLapseNotifies(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper, victim := snapperAndVictim(t, g, players)
	snapperIdx := g.PlayerToEngine[snapper.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	emptyHand(g, snapperIdx)
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	require.False(t, g.owesSnapFill(snapper.ID), "a debt with nothing left to pay it lapses")
	requireSnapFillCleared(t, mb, snapper.ID)
}

// emptyHand takes every card off a seat, mirror included, without any of the rules that would
// normally do it: the point is a snapper who is left with nothing to pay a fill with.
func emptyHand(g *CambiaGame, seat uint8) {
	for i := uint8(0); i < g.Engine.Players[seat].HandLen; i++ {
		g.Engine.Players[seat].Hand[i] = engine.EmptyCard
		g.CardTracker.Players[seat].HandUUIDs[i] = uuid.Nil
	}
	g.Engine.Players[seat].HandLen = 0
	g.syncPlayerHandsFromEngine()
}
