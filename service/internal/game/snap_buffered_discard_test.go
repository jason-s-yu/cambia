// internal/game/snap_buffered_discard_test.go
// A discard whose card carries an ability is announced to every client immediately and only
// applied to the engine once the discarder resolves (or skips) the ability. Snapping is legal
// "any time a card is discarded" (RULES.md section 5), so snaps arrive inside that window and have
// to be judged against the card the clients were shown, not the card it covered (cambia-956).
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// bufferedAbilityDiscard walks the acting player through drawing the given ability card off the
// stockpile and discarding it, leaving the service holding the discard behind the ability choice.
// It returns the acting player, the non-acting player, and the discarded card's UUID.
func bufferedAbilityDiscard(t *testing.T, g *CambiaGame, players []*models.Player, card engine.Card) (actor, other *models.Player, cardUUID uuid.UUID) {
	t.Helper()

	actor = currentTurnPlayer(g)
	for _, p := range players {
		if p.ID != actor.ID {
			other = p
			break
		}
	}
	require.NotNil(t, other, "the table needs a second seat")

	actorIdx, ok := g.PlayerToEngine[actor.ID]
	require.True(t, ok, "the acting player should have an engine index")

	// Plant the ability card on top of the stockpile so the draw is deterministic.
	require.Greater(t, int(g.Engine.StockLen), 0, "the deal should leave a stocked pile")
	topStock := g.Engine.StockLen - 1
	g.Engine.Stockpile[topStock] = card
	cardUUID = uuid.New()
	g.CardTracker.StockUUIDs[topStock] = cardUUID
	g.CardTracker.Registry[cardUUID] = engineCardToDetails(card, cardUUID)

	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, cardUUID, g.CardTracker.Players[actorIdx].DrawnCardUUID, "the planted card should be the drawn one")

	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})
	require.True(t, g.pendingDiscardAbilityChoice, "an ability card drawn from the stockpile buffers its discard behind the ability choice")

	return actor, other, cardUUID
}

// plantDiscardTop overwrites the card already sitting on top of the discard pile, so the card an
// ability discard covers is fixed rather than whatever the deal turned up.
func plantDiscardTop(t *testing.T, g *CambiaGame, card engine.Card) uuid.UUID {
	t.Helper()

	require.Greater(t, int(g.Engine.DiscardLen), 0, "the deal should turn one card face up")
	topIdx := g.Engine.DiscardLen - 1
	g.Engine.DiscardPile[topIdx] = card
	cardUUID := uuid.New()
	g.CardTracker.DiscardUUIDs[topIdx] = cardUUID
	g.CardTracker.Registry[cardUUID] = engineCardToDetails(card, cardUUID)

	return cardUUID
}

// plantHandCard overwrites a seat's hand slot with the given card and mints a tracked UUID for it.
func plantHandCard(t *testing.T, g *CambiaGame, ownerID uuid.UUID, slot uint8, card engine.Card) uuid.UUID {
	t.Helper()

	ownerIdx, ok := g.PlayerToEngine[ownerID]
	require.True(t, ok, "owner should have an engine index")
	require.Less(t, slot, g.Engine.Players[ownerIdx].HandLen, "slot should be inside the dealt hand")

	g.Engine.Players[ownerIdx].Hand[slot] = card
	cardUUID := uuid.New()
	g.CardTracker.Players[ownerIdx].HandUUIDs[slot] = cardUUID
	g.CardTracker.Registry[cardUUID] = engineCardToDetails(card, cardUUID)
	g.syncPlayerHandsFromEngine()

	return cardUUID
}

// TestSnapMatchesBufferedAbilityDiscard reproduces the false rejection: the discarder puts a seven
// on the pile, every client is told so, and an opponent snaps their own seven before the ability
// resolves. The engine's discard pile still holds the card the seven covered, so the snap used to
// be judged against that card and paid the invalid-snap penalty; retried after the ability
// resolved, the same card snapped cleanly against the same visible top.
func TestSnapMatchesBufferedAbilityDiscard(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))

	seven := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	actor, snapper, discardedUUID := bufferedAbilityDiscard(t, g, players, seven)

	discardEv := mb.findEventByType(EventPlayerDiscard)
	require.NotNil(t, discardEv, "clients are told the seven is on the pile before the ability resolves")
	require.NotNil(t, discardEv.Card)
	require.Equal(t, discardedUUID, discardEv.Card.ID)
	require.Equal(t, actor.ID, discardEv.User.ID)

	matchUUID := plantHandCard(t, g, snapper.ID, 0, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	snapperIdx := g.PlayerToEngine[snapper.ID]
	handBefore := int(g.Engine.Players[snapperIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": matchUUID.String()},
	})

	assert.Nil(t, mb.findEventByType(EventPlayerSnapFail), "a rank match against the announced discard top must not be rejected")
	assert.Nil(t, mb.findEventByType(EventPlayerSnapPenalty), "a valid snap must draw no penalty")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapSuccess), "the snap should succeed")
	assert.Equal(t, handBefore-1, int(g.Engine.Players[snapperIdx].HandLen), "the snapper's hand should lose the snapped card")
}

// TestSnapAgainstCoveredCardRejectedDuringAbilityWindow pins the other half of the same read: once
// the ability card is announced, the card it covered is no longer the top, so a snap matching the
// covered card is invalid and must take the penalty. Judging snaps against the engine's unapplied
// pile accepted it.
func TestSnapAgainstCoveredCardRejectedDuringAbilityWindow(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	buried := engine.NewCard(engine.SuitClubs, engine.RankFive)
	plantDiscardTop(t, g, buried)

	seven := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	_, snapper, _ := bufferedAbilityDiscard(t, g, players, seven)

	// The snapper holds a match for the card the seven covered, not for the seven.
	staleUUID := plantHandCard(t, g, snapper.ID, 0, engine.NewCard(engine.SuitSpades, buried.Rank()))
	snapperIdx := g.PlayerToEngine[snapper.ID]
	handBefore := int(g.Engine.Players[snapperIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": staleUUID.String()},
	})

	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail), "matching the covered card is not a match for the top")
	assert.Nil(t, mb.findEventByType(EventPlayerSnapSuccess), "the snap must not succeed")
	assert.Equal(t, handBefore+2, int(g.Engine.Players[snapperIdx].HandLen), "the invalid snap should draw the two-card penalty")
}

// TestBufferedDiscardTimeoutSkipsAbility pins the reachable path a turn timeout takes over a
// buffered ability discard. handleTimeoutEngine used to carry a second check for exactly this state
// (g.pendingDiscardAbilityChoice && g.SpecialAction.Active && ...) below one that already handles
// every SpecialAction.Active case and always returns, so the second check could never run: the
// buffered-choice SpecialAction this test builds (handleDiscardViaEngine) sets Active true for the
// same player it sets pendingDiscardAbilityChoice for, so the first check always catches it first
// (cambia-1054). The actually-reachable route is SpecialAction.MustResolve() reading false for this
// non-Mandatory prompt, falling into processSkipSpecialAction, which resolves the same buffered
// discard as no-ability (special_actions.go).
func TestBufferedDiscardTimeoutSkipsAbility(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	seven := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	actor, _, discardedUUID := bufferedAbilityDiscard(t, g, players, seven)
	require.True(t, g.SpecialAction.Active, "the buffered choice activates the special-action prompt")
	require.False(t, g.SpecialAction.Mandatory, "a buffered ability choice is declinable, unlike an engine-armed one")

	g.mu.Lock()
	g.handleTimeoutEngine(actor.ID)
	g.mu.Unlock()

	assert.False(t, g.pendingDiscardAbilityChoice, "the timeout resolves the buffer instead of leaving it pending")
	assert.False(t, g.SpecialAction.Active, "and the prompt does not outlive it")
	assert.Equal(t, discardedUUID, g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen-1], "the buffered card lands on the pile without a second announcement")
	assert.Equal(t, 1, countDiscardsOfCard(mb, discardedUUID), "the card was already announced when it was played; the timeout must not announce it again")
}
