// internal/game/snap_fill_test.go
// RULES.md 5: snapping an opponent's card obliges the snapper to move one of their own cards into
// the slot it left. The service resolved snaps outside the engine's snap phase and never asked for
// that card, so an opponent snap was a free card off the victim (cambia-936).
package game

import (
	"testing"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// snapOpponentCard plants a matching pair, has snapper take victim's card, and returns the planted
// card's UUID and the slot it left.
func snapOpponentCard(t *testing.T, g *CambiaGame, mb *mockBroadcaster, snapper, victim *models.Player) (uuid.UUID, int) {
	t.Helper()
	cardUUID, slot := plantSnapPair(t, g, victim.ID)
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})
	return cardUUID, slot
}

// TestOpponentSnapOwesFill pins the obligation itself: the victim loses the card, the snapper keeps
// their hand for now, and the table is told who owes what.
func TestOpponentSnapOwesFill(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]
	victimIdx, snapperIdx := g.PlayerToEngine[victim.ID], g.PlayerToEngine[snapper.ID]

	victimBefore := int(g.Engine.Players[victimIdx].HandLen)
	snapperBefore := int(g.Engine.Players[snapperIdx].HandLen)

	_, slot := snapOpponentCard(t, g, mb, snapper, victim)

	assert.Equal(t, victimBefore, int(g.Engine.Players[victimIdx].HandLen), "the victim lost the snapped card and has not been paid back yet")
	assert.Equal(t, snapperBefore, int(g.Engine.Players[snapperIdx].HandLen), "the snapper's hand only moves once they choose the card to give")
	assert.True(t, g.owesSnapFill(snapper.ID), "the snapper should owe a fill")

	ev := mb.findEventByType(EventPlayerSnapMoveRequired)
	require.NotNil(t, ev, "the snap should prompt the snapper for the card they owe")
	require.NotNil(t, ev.User)
	assert.Equal(t, snapper.ID, ev.User.ID, "the prompt names the snapper who owes the card")
	require.NotNil(t, ev.Card)
	require.NotNil(t, ev.Card.User)
	assert.Equal(t, victim.ID, ev.Card.User.ID, "the prompt names the hand the card goes into")
	require.NotNil(t, ev.Card.Idx)
	assert.Equal(t, slot, *ev.Card.Idx, "the prompt names the slot the snapped card left")
}

// TestSnapFillMovesChosenCard walks the whole rule: the snapper names one of their cards and it
// lands in the slot they emptied, in both the engine hand and the UUID mirror.
func TestSnapFillMovesChosenCard(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]
	victimIdx, snapperIdx := g.PlayerToEngine[victim.ID], g.PlayerToEngine[snapper.ID]

	_, slot := snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	victimAfterSnap := int(g.Engine.Players[victimIdx].HandLen)
	snapperAfterSnap := int(g.Engine.Players[snapperIdx].HandLen)
	givenUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[0]
	givenCard := g.Engine.Players[snapperIdx].Hand[0]
	keptUUIDs := []uuid.UUID{
		g.CardTracker.Players[snapperIdx].HandUUIDs[1],
		g.CardTracker.Players[snapperIdx].HandUUIDs[2],
		g.CardTracker.Players[snapperIdx].HandUUIDs[3],
	}

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": givenUUID.String(), "idx": float64(0)},
	})

	assert.False(t, g.owesSnapFill(snapper.ID), "paying the card settles the obligation")
	assert.Equal(t, snapperAfterSnap-1, int(g.Engine.Players[snapperIdx].HandLen), "the snapper gave a card away")
	assert.Equal(t, victimAfterSnap+1, int(g.Engine.Players[victimIdx].HandLen), "the victim's empty slot was filled")
	assert.Equal(t, givenCard, g.Engine.Players[victimIdx].Hand[slot], "the given card sits in the slot the snapped card left")
	assert.Equal(t, givenUUID, g.CardTracker.Players[victimIdx].HandUUIDs[slot], "the UUID mirror follows the card into the victim's hand")

	// The snapper's remaining ids shift left with the engine hand, and the vacated tail is cleared.
	for i, want := range keptUUIDs {
		assert.Equal(t, want, g.CardTracker.Players[snapperIdx].HandUUIDs[i], "kept card %d shifts left", i)
	}
	assert.Equal(t, uuid.Nil, g.CardTracker.Players[snapperIdx].HandUUIDs[len(keptUUIDs)], "the snapper's freed slot holds no id")

	ev := mb.findEventByType(EventPlayerSnapMove)
	require.NotNil(t, ev, "the move is public")
	require.NotNil(t, ev.User)
	assert.Equal(t, snapper.ID, ev.User.ID)
	require.NotNil(t, ev.Card)
	assert.Equal(t, givenUUID, ev.Card.ID)
	assert.Empty(t, ev.Card.Rank, "the card changes hands unseen, so no face crosses the wire")
	require.NotNil(t, ev.Card.Idx)
	assert.Equal(t, slot, *ev.Card.Idx)
	require.NotNil(t, ev.Card.User)
	assert.Equal(t, victim.ID, ev.Card.User.ID)
	assert.Equal(t, 0, ev.Payload["fromIdx"])
	assert.Equal(t, false, ev.Payload["auto"])
}

// TestSnapFillRejectsCardNotHeld keeps the fill honest: an id the snapper does not hold settles
// nothing and leaves the obligation open.
func TestSnapFillRejectsCardNotHeld(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]
	victimIdx := g.PlayerToEngine[victim.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	victimAfterSnap := int(g.Engine.Players[victimIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": uuid.New().String()},
	})

	assert.True(t, g.owesSnapFill(snapper.ID), "an unknown card does not settle the obligation")
	assert.Equal(t, victimAfterSnap, int(g.Engine.Players[victimIdx].HandLen), "no card moved")
	require.NotNil(t, mb.getLastPlayerEvent(snapper.ID), "the snapper is told why")
}

// TestSnapFillBlocksOtherActions holds the snapper to the rule: nothing else they send counts
// until they have paid the card, and the refusal reaches them.
func TestSnapFillBlocksOtherActions(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper := currentTurnPlayer(g)
	var victim *models.Player
	for _, p := range players {
		if p.ID != snapper.ID {
			victim = p
		}
	}
	require.NotNil(t, victim)

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	stockBefore := g.Engine.StockLen
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	assert.Equal(t, stockBefore, g.Engine.StockLen, "the draw is refused while the fill is owed")
	require.NotNil(t, mb.getLastPlayerEvent(snapper.ID), "the snapper is told what they owe")

	// A second snap is refused with the rest: it would open a second obligation on the same hand.
	otherUUID, _ := plantSnapPair(t, g, victim.ID)
	victimIdx := g.PlayerToEngine[victim.ID]
	victimBefore := int(g.Engine.Players[victimIdx].HandLen)
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": otherUUID.String()},
	})
	assert.Equal(t, victimBefore, int(g.Engine.Players[victimIdx].HandLen), "the second snap never resolved")
}

// TestSnapFillDeadlineSettlesIt covers the snapper who never answers: the victim cannot be left a
// card short, so the deadline gives up the snapper's last slot for them.
func TestSnapFillDeadlineSettlesIt(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]
	victimIdx, snapperIdx := g.PlayerToEngine[victim.ID], g.PlayerToEngine[snapper.ID]

	// The turn timer is off in this fixture, so arming a fill deadline here arms nothing else.
	g.TurnDuration = 40 * time.Millisecond

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	g.mu.Lock()
	snapperHand := int(g.Engine.Players[snapperIdx].HandLen)
	victimHand := int(g.Engine.Players[victimIdx].HandLen)
	lastUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[snapperHand-1]
	g.mu.Unlock()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		g.mu.Lock()
		settled := !g.owesSnapFill(snapper.ID)
		g.mu.Unlock()
		if settled {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	require.False(t, g.owesSnapFill(snapper.ID), "the deadline should have settled the fill")
	assert.Equal(t, snapperHand-1, int(g.Engine.Players[snapperIdx].HandLen), "the snapper still paid a card")
	assert.Equal(t, victimHand+1, int(g.Engine.Players[victimIdx].HandLen), "the victim was paid back")
	assert.Equal(t, lastUUID, g.CardTracker.Players[victimIdx].HandUUIDs[victimHand], "the deadline gives up the snapper's last slot")

	ev := mb.findEventByType(EventPlayerSnapMove)
	require.NotNil(t, ev)
	assert.Equal(t, true, ev.Payload["auto"], "the event marks a fill the deadline chose")
}

// TestOpponentSnapWithEmptyHandFails mirrors the engine's own answer (engine/snap.go snapOpponent):
// a snapper with nothing to give cannot take, so the attempt is a failed snap and a penalty.
func TestOpponentSnapWithEmptyHandFails(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]
	victimIdx, snapperIdx := g.PlayerToEngine[victim.ID], g.PlayerToEngine[snapper.ID]

	// Empty the snapper's hand, mirror included.
	for i := uint8(0); i < g.Engine.Players[snapperIdx].HandLen; i++ {
		g.Engine.Players[snapperIdx].Hand[i] = engine.EmptyCard
		g.CardTracker.Players[snapperIdx].HandUUIDs[i] = uuid.Nil
	}
	g.Engine.Players[snapperIdx].HandLen = 0
	g.syncPlayerHandsFromEngine()

	cardUUID, _ := plantSnapPair(t, g, victim.ID)
	victimBefore := int(g.Engine.Players[victimIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, victimBefore, int(g.Engine.Players[victimIdx].HandLen), "the victim keeps the card")
	assert.False(t, g.owesSnapFill(snapper.ID), "a failed snap owes nothing")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail), "the snap failed")
	assert.Equal(t, 2, int(g.Engine.Players[snapperIdx].HandLen), "the failed snap drew the penalty")
}

// TestUnpayableSnapFillLapses covers the debt that can never be settled: another player can snap
// the snapper's last card away, and a table with no turn timer arms no deadline to clear it, so
// the gate would refuse that player's every action for the rest of the game.
func TestUnpayableSnapFillLapses(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper := currentTurnPlayer(g)
	var victim *models.Player
	for _, p := range players {
		if p.ID != snapper.ID {
			victim = p
		}
	}
	require.NotNil(t, victim)
	snapperIdx := g.PlayerToEngine[snapper.ID]

	snapOpponentCard(t, g, mb, snapper, victim)
	require.True(t, g.owesSnapFill(snapper.ID))

	// The snapper's hand empties without them acting at all.
	for i := uint8(0); i < g.Engine.Players[snapperIdx].HandLen; i++ {
		g.Engine.Players[snapperIdx].Hand[i] = engine.EmptyCard
		g.CardTracker.Players[snapperIdx].HandUUIDs[i] = uuid.Nil
	}
	g.Engine.Players[snapperIdx].HandLen = 0
	g.syncPlayerHandsFromEngine()

	stockBefore := g.Engine.StockLen
	g.HandlePlayerAction(snapper.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	assert.False(t, g.owesSnapFill(snapper.ID), "a debt with nothing left to pay it lapses")
	assert.Equal(t, stockBefore-1, g.Engine.StockLen, "and the refused action goes through instead")
}

// TestSyncStateCarriesSnapFill: a client that resyncs mid-obligation has to see it, or it sits on a
// table refusing its every action with nothing on screen to explain why.
func TestSyncStateCarriesSnapFill(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim, snapper := players[0], players[1]

	_, slot := snapOpponentCard(t, g, mb, snapper, victim)

	obf := g.GetCurrentObfuscatedGameState(snapper.ID)
	require.Len(t, obf.SnapMoves, 1, "the outstanding fill should be in the snapshot")
	assert.Equal(t, snapper.ID, obf.SnapMoves[0].SnapperID)
	assert.Equal(t, victim.ID, obf.SnapMoves[0].VictimID)
	assert.Equal(t, slot, obf.SnapMoves[0].Slot)

	g.mu.Lock()
	fill := g.snapFills[snapper.ID]
	g.applySnapFill(fill, 0, false)
	g.mu.Unlock()

	obf = g.GetCurrentObfuscatedGameState(snapper.ID)
	assert.Empty(t, obf.SnapMoves, "a settled fill leaves the snapshot")
}
