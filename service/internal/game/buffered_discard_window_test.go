// internal/game/buffered_discard_window_test.go
// The pile, the announcement and the snapshot across the window an ability discard opens: the card
// is announced to every client when it is played and only applied to the engine once the ability is
// resolved, skipped or timed out (cambia-1033). Snaps land inside that window (cambia-956), so the
// order the engine ends up with, the events it fires and the state it hands a resyncing client all
// have to agree with the pile the table was already shown.
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

// countDiscardsOfCard returns how many player_discard events announced the given card.
func countDiscardsOfCard(mb *mockBroadcaster, cardID uuid.UUID) int {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	n := 0
	for _, ev := range mb.allEvents {
		if ev.Type == EventPlayerDiscard && ev.Card != nil && ev.Card.ID == cardID {
			n++
		}
	}
	return n
}

// pileUUIDs returns the discard pile's UUIDs bottom-up, as the engine holds them.
func pileUUIDs(g *CambiaGame) []uuid.UUID {
	out := make([]uuid.UUID, 0, g.Engine.DiscardLen)
	for i := uint8(0); i < g.Engine.DiscardLen; i++ {
		out = append(out, g.CardTracker.DiscardUUIDs[i])
	}
	return out
}

// snapDuringWindow has snapper snap the card at ownerID's slot, which must match the announced
// discard top, and returns that card's UUID.
func snapDuringWindow(t *testing.T, g *CambiaGame, snapper *models.Player, ownerID uuid.UUID, slot uint8, card engine.Card) uuid.UUID {
	t.Helper()

	cardUUID := plantHandCard(t, g, ownerID, slot, card)
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})
	require.Equal(t, cardUUID, g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen-1], "the snapped card should be on the engine pile")
	return cardUUID
}

// TestWindowSnapOrderSurvivesAbilityResolve pins A1: a snap taken while the ability discard is
// buffered is applied to the engine pile first, so the buffered card has to be rotated back under
// it when the ability resolves. Left on top it inverted the order every client rendered, which made
// action_draw_discardpile hand out the ability card instead of the snapped one.
func TestWindowSnapOrderSurvivesAbilityResolve(t *testing.T) {
	rules := testHouseRules(0, 2)
	rules.AllowDrawFromDiscardPile = true
	g, players, mb := setupTestGame(t, 2, rules)

	buriedUUID := plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	actor, snapper, sevenUUID := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	snapUUID := snapDuringWindow(t, g, snapper, snapper.ID, 0, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	require.NotNil(t, mb.findEventByType(EventPlayerSnapSuccess), "the window snap should succeed")

	// Resolve the ability. The buffered discard applies with it.
	actorIdx := g.PlayerToEngine[actor.ID]
	ownSlot0 := g.CardTracker.Players[actorIdx].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_self", cardTarget(ownSlot0, actor.ID, 0), nil)
	require.False(t, g.pendingDiscardAbilityChoice, "resolving the ability applies the buffered discard")

	assert.Equal(t, []uuid.UUID{buriedUUID, sevenUUID, snapUUID}, pileUUIDs(g),
		"the engine pile must end up in the order the clients rendered: the announced card, then what was snapped onto it")
	top := g.Engine.DiscardPile[g.Engine.DiscardLen-1]
	assert.Equal(t, engine.SuitSpades, top.Suit(), "the snapped card is the top card, not the ability card of the same rank")

	// The next player draws the pile's top back off it: with the order inverted this handed out the
	// ability card's identity instead of the snapped card's.
	require.Equal(t, snapper.ID, currentTurnPlayer(g).ID, "the ability resolution should advance the turn")
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{ActionType: "action_draw_discardpile"})

	snapperIdx := g.PlayerToEngine[snapper.ID]
	assert.Equal(t, snapUUID, g.CardTracker.Players[snapperIdx].DrawnCardUUID, "the drawn card is the snapped one")
	drawEv := mb.findEventByType(EventPlayerDrawStockpile)
	require.NotNil(t, drawEv, "drawing from the discard pile announces the card")
	require.NotNil(t, drawEv.Card)
	assert.Equal(t, snapUUID, drawEv.Card.ID)
	assert.Equal(t, "S", drawEv.Card.Suit, "the event names the snapped card's suit, not the ability card's")
}

// TestAbilitySkipAnnouncesTheDiscardOnce pins A2 on the skip path: the card was announced when it
// was played, so applying the buffered discard as a no-ability discard must not announce it again.
// The duplicate counted every client's pile up by one and pulled its rendered top back onto the
// ability card. The rotation is checked here too, since skip settles the buffer through the full
// apply path rather than the raw one an ability resolution uses.
func TestAbilitySkipAnnouncesTheDiscardOnce(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	buriedUUID := plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	actor, snapper, sevenUUID := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	require.Equal(t, 1, countDiscardsOfCard(mb, sevenUUID), "playing the card announces it once")

	snapUUID := snapDuringWindow(t, g, snapper, snapper.ID, 0, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	sizeDuringWindow := g.discardSize()

	g.ProcessSpecialAction(actor.ID, "skip", nil, nil)
	require.False(t, g.pendingDiscardAbilityChoice, "skipping the ability applies the buffered discard")

	assert.Equal(t, 1, countDiscardsOfCard(mb, sevenUUID), "the card must not be announced a second time when the buffer settles")
	assert.Equal(t, sizeDuringWindow, g.discardSize(), "the pile the table sees is the same size before and after the buffer settles")
	assert.Equal(t, []uuid.UUID{buriedUUID, sevenUUID, snapUUID}, pileUUIDs(g),
		"the skipped ability card still sinks under the card snapped onto it")
}

// TestAbilityTimeoutAnnouncesTheDiscardOnce pins A2 on the timeout path: the turn timer settles an
// unanswered ability the same way a skip does, and must not re-announce the card either. Asserted
// per card, since the timer keeps running and the next player's turn times out in the same way.
func TestAbilityTimeoutAnnouncesTheDiscardOnce(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(1, 2))

	plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	actor, _, sevenUUID := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))
	require.Equal(t, 1, countDiscardsOfCard(mb, sevenUUID), "playing the card announces it once")

	// The timer fires on its own goroutine, so every read below goes through the locked snapshot.
	require.Eventually(t, func() bool {
		return g.GetCurrentObfuscatedGameState(actor.ID).SpecialAction == nil
	}, 2*time.Second, 10*time.Millisecond, "the turn timer should settle the unanswered ability")

	assert.Equal(t, 1, countDiscardsOfCard(mb, sevenUUID), "a timed-out ability must not announce its card again")

	g.mu.Lock()
	defer g.mu.Unlock()
	onPile := 0
	for i := uint8(0); i < g.Engine.DiscardLen; i++ {
		if g.CardTracker.DiscardUUIDs[i] == sevenUUID {
			onPile++
		}
	}
	assert.Equal(t, 1, onPile, "the timed-out ability card sits on the pile exactly once")
	assert.False(t, g.pendingDiscardAbilityChoice, "the window is closed once the timer settles it")
}

// TestSyncDuringAbilityWindowShowsAnnouncedTop pins A3: a snapshot taken inside the window has to
// describe the pile the clients are already looking at. It used to name the covered card as the top
// and count the pile a card short, and it handed the discarder back the card it had just told
// everyone they discarded as their still-drawn card.
func TestSyncDuringAbilityWindowShowsAnnouncedTop(t *testing.T) {
	g, players, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	sizeBefore := g.discardSize()
	actor, snapper, sevenUUID := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	obf := g.GetCurrentObfuscatedGameState(snapper.ID)
	require.NotNil(t, obf.DiscardTop, "the pile has a top card during the window")
	assert.Equal(t, sevenUUID, obf.DiscardTop.ID, "the top is the announced card, not the one it covered")
	assert.Equal(t, "7", obf.DiscardTop.Rank)
	assert.Equal(t, sizeBefore+1, obf.DiscardSize, "the announced card counts towards the pile the table sees")

	// The discarder's own view: the card is on the pile, so it is no longer their drawn card, and the
	// ability they owe is what the snapshot puts them on.
	self := g.GetCurrentObfuscatedGameState(actor.ID)
	var actorState *ObfPlayerState
	for i := range self.Players {
		if self.Players[i].PlayerID == actor.ID {
			actorState = &self.Players[i]
		}
	}
	require.NotNil(t, actorState)
	assert.Nil(t, actorState.DrawnCard, "an announced discard is not still in hand as a drawn card")
	require.NotNil(t, self.SpecialAction, "the ability the discarder owes rides the snapshot")
	assert.Equal(t, "7", self.SpecialAction.CardRank)

	// A snap covers the announced card in turn, and the snapshot follows it.
	snapUUID := snapDuringWindow(t, g, snapper, snapper.ID, 0, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	obf = g.GetCurrentObfuscatedGameState(snapper.ID)
	require.NotNil(t, obf.DiscardTop)
	assert.Equal(t, snapUUID, obf.DiscardTop.ID, "the snapped card covers the announced one")
	assert.Equal(t, sizeBefore+2, obf.DiscardSize)

	// And once the buffer settles, the snapshot reports the same pile it did inside the window.
	actorIdx := g.PlayerToEngine[actor.ID]
	ownSlot0 := g.CardTracker.Players[actorIdx].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_self", cardTarget(ownSlot0, actor.ID, 0), nil)

	obf = g.GetCurrentObfuscatedGameState(snapper.ID)
	require.NotNil(t, obf.DiscardTop)
	assert.Equal(t, snapUUID, obf.DiscardTop.ID, "settling the buffer must not change the top the table saw")
	assert.Equal(t, sizeBefore+2, obf.DiscardSize, "nor the size")
}

// TestFailedWindowSnapReportsVisiblePileSize covers the other count a client takes from the server
// inside the window: a snap that fails there draws a penalty, and its event carries the pile sizes
// the client displays outright. Reporting the engine's count knocked the rendered pile down by the
// announced card.
func TestFailedWindowSnapReportsVisiblePileSize(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	sizeBefore := g.discardSize()
	_, snapper, _ := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	// A card that matches neither the announced seven nor the five it covers.
	missUUID := plantHandCard(t, g, snapper.ID, 0, engine.NewCard(engine.SuitSpades, engine.RankThree))
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": missUUID.String()},
	})

	require.NotNil(t, mb.findEventByType(EventPlayerSnapFail), "the snap should fail")
	penaltyEv := mb.findEventByType(EventPlayerSnapPenalty)
	require.NotNil(t, penaltyEv, "a failed snap draws a penalty")
	assert.Equal(t, sizeBefore+1, penaltyEv.Payload["discardSize"], "the penalty event counts the announced card the client already rendered")
}

// TestWindowSnapOfOpponentCardKeepsOrderWithFill composes the window with the fill an opponent snap
// owes (RULES.md 5, cambia-936): the snap takes a card out of the discarder's hand mid-window, so
// the pile has to end up in the announced order and the fill still has to settle on top of it.
func TestWindowSnapOfOpponentCardKeepsOrderWithFill(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	buriedUUID := plantDiscardTop(t, g, engine.NewCard(engine.SuitClubs, engine.RankFive))
	actor, snapper, sevenUUID := bufferedAbilityDiscard(t, g, players, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	// The snapper takes the discarder's own seven out of their hand, which owes a fill back.
	snapUUID := snapDuringWindow(t, g, snapper, actor.ID, 1, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	require.NotNil(t, mb.findEventByType(EventPlayerSnapSuccess), "the opponent snap should succeed")
	require.NotNil(t, mb.findEventByType(EventPlayerSnapMoveRequired), "an opponent snap owes a fill")
	require.True(t, g.owesSnapFill(snapper.ID), "the snapper owes the fill")

	// The discarder resolves the ability they still owe.
	actorIdx := g.PlayerToEngine[actor.ID]
	ownSlot0 := g.CardTracker.Players[actorIdx].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_self", cardTarget(ownSlot0, actor.ID, 0), nil)
	require.False(t, g.pendingDiscardAbilityChoice, "resolving the ability applies the buffered discard")

	assert.Equal(t, []uuid.UUID{buriedUUID, sevenUUID, snapUUID}, pileUUIDs(g),
		"an opponent snap orders the pile the same way an own snap does")

	// Paying the fill moves a card between hands and leaves the pile alone.
	snapperIdx := g.PlayerToEngine[snapper.ID]
	fillUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[0]
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": fillUUID.String()},
	})

	moveEv := mb.findEventByType(EventPlayerSnapMove)
	require.NotNil(t, moveEv, "the fill should settle")
	require.NotNil(t, moveEv.Card)
	assert.Equal(t, fillUUID, moveEv.Card.ID)
	assert.False(t, g.owesSnapFill(snapper.ID), "the obligation is paid")
	assert.Equal(t, []uuid.UUID{buriedUUID, sevenUUID, snapUUID}, pileUUIDs(g), "the fill leaves the pile untouched")
}
