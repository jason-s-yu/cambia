// internal/game/snap_owner_test.go
// player_snap_success has to name the hand the card left, not just the player who snapped
// (cambia-913). The two are the same only for an own-hand snap; on an opponent snap the event's
// top-level user is the snapper while the card leaves the victim's hand at the event's idx, so
// without an owner every client shrinks the wrong seat.
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// plantSnapPair puts a seven on the discard pile and a matching seven in the given player's hand,
// returning the planted card's UUID and the hand slot it went into.
func plantSnapPair(t *testing.T, g *CambiaGame, ownerID uuid.UUID) (uuid.UUID, int) {
	t.Helper()

	top := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	g.Engine.DiscardPile[g.Engine.DiscardLen] = top
	g.Engine.DiscardLen++
	topUUID := uuid.New()
	g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen-1] = topUUID
	g.CardTracker.Registry[topUUID] = engineCardToDetails(top, topUUID)
	g.CardTracker.DiscardLen = g.Engine.DiscardLen

	ownerIdx, ok := g.PlayerToEngine[ownerID]
	require.True(t, ok, "owner should have an engine index")

	match := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	slot := int(g.Engine.Players[ownerIdx].HandLen)
	g.Engine.Players[ownerIdx].Hand[slot] = match
	g.Engine.Players[ownerIdx].HandLen++
	cardUUID := uuid.New()
	g.CardTracker.Players[ownerIdx].HandUUIDs[slot] = cardUUID
	g.CardTracker.Registry[cardUUID] = engineCardToDetails(match, cardUUID)
	g.syncPlayerHandsFromEngine()

	return cardUUID, slot
}

// TestSnapSuccessNamesOpponentOwner pins the owner on an opponent snap: the snapper acts, the
// victim's hand shrinks, and the event has to say so.
func TestSnapSuccessNamesOpponentOwner(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	victim := players[0]
	snapper := players[1]

	cardUUID, slot := plantSnapPair(t, g, victim.ID)
	victimIdx := g.PlayerToEngine[victim.ID]
	snapperIdx := g.PlayerToEngine[snapper.ID]
	victimHandBefore := int(g.Engine.Players[victimIdx].HandLen)
	snapperHandBefore := int(g.Engine.Players[snapperIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, victimHandBefore-1, int(g.Engine.Players[victimIdx].HandLen), "the victim's hand should lose the snapped card")
	assert.Equal(t, snapperHandBefore, int(g.Engine.Players[snapperIdx].HandLen), "the snapper's hand should be untouched")

	ev := mb.findEventByType(EventPlayerSnapSuccess)
	require.NotNil(t, ev, "expected a public snap success event")
	require.NotNil(t, ev.User, "snap success should name the snapper")
	assert.Equal(t, snapper.ID, ev.User.ID, "top-level user is the snapper")
	require.NotNil(t, ev.Card, "snap success should carry the card")
	assert.Equal(t, cardUUID, ev.Card.ID)
	require.NotNil(t, ev.Card.User, "snap success should carry the card owner")
	assert.Equal(t, victim.ID, ev.Card.User.ID, "card owner is the hand the card left")
	require.NotNil(t, ev.Card.Idx, "snap success should carry the hand slot")
	assert.Equal(t, slot, *ev.Card.Idx, "idx indexes the owner's hand")
}

// TestSnapSuccessNamesSelfOwner keeps the own-hand branch on the same contract: owner equals
// snapper, so a client that reads the owner needs no special case.
func TestSnapSuccessNamesSelfOwner(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	snapper := players[1]

	cardUUID, slot := plantSnapPair(t, g, snapper.ID)
	snapperIdx := g.PlayerToEngine[snapper.ID]
	handBefore := int(g.Engine.Players[snapperIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, handBefore-1, int(g.Engine.Players[snapperIdx].HandLen), "the snapper's own hand should shrink")

	ev := mb.findEventByType(EventPlayerSnapSuccess)
	require.NotNil(t, ev, "expected a public snap success event")
	require.NotNil(t, ev.Card, "snap success should carry the card")
	require.NotNil(t, ev.Card.User, "snap success should carry the card owner")
	assert.Equal(t, snapper.ID, ev.Card.User.ID, "own-hand snap owner is the snapper")
	require.NotNil(t, ev.Card.Idx)
	assert.Equal(t, slot, *ev.Card.Idx)
}
