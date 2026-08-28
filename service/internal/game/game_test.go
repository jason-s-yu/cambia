// internal/game/game_test.go
package game

import (
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockBroadcaster captures game events for testing assertions.
// It implements the Emitter interface.
type mockBroadcaster struct {
	mu           sync.Mutex
	allEvents    []GameEvent
	playerEvents map[uuid.UUID][]GameEvent
}

// newMockBroadcaster creates an instance of the mock broadcaster.
func newMockBroadcaster() *mockBroadcaster {
	return &mockBroadcaster{
		playerEvents: make(map[uuid.UUID][]GameEvent),
	}
}

// Emit implements Emitter - captures broadcast events.
func (mb *mockBroadcaster) Emit(eventType string, payload any) {
	ev, ok := payload.(GameEvent)
	if !ok {
		return
	}
	mb.mu.Lock()
	defer mb.mu.Unlock()
	mb.allEvents = append(mb.allEvents, ev)
}

// EmitTo implements Emitter - captures per-player events.
func (mb *mockBroadcaster) EmitTo(playerID uuid.UUID, eventType string, payload any) {
	ev, ok := payload.(GameEvent)
	if !ok {
		return
	}
	mb.mu.Lock()
	defer mb.mu.Unlock()
	mb.playerEvents[playerID] = append(mb.playerEvents[playerID], ev)
}

func (mb *mockBroadcaster) clear() {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	mb.allEvents = []GameEvent{}
	mb.playerEvents = make(map[uuid.UUID][]GameEvent)
}

func (mb *mockBroadcaster) getLastEvent() *GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	if len(mb.allEvents) == 0 {
		return nil
	}
	return &mb.allEvents[len(mb.allEvents)-1]
}

func (mb *mockBroadcaster) getLastPlayerEvent(playerID uuid.UUID) *GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	events, ok := mb.playerEvents[playerID]
	if !ok || len(events) == 0 {
		return nil
	}
	return &events[len(events)-1]
}

func (mb *mockBroadcaster) findEventByType(eventType GameEventType) *GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	for i := len(mb.allEvents) - 1; i >= 0; i-- {
		if mb.allEvents[i].Type == eventType {
			return &mb.allEvents[i]
		}
	}
	return nil
}

// testHouseRules returns the service defaults with the turn timer and invalid-snap penalty
// overridden, the two knobs these fixtures vary. It builds on DefaultHouseRules rather than a
// bare literal so the deal knobs (cards per player, jokers, decks, pregame peek count) hold the
// values a real lobby starts with: a bare HouseRules literal zeroes them, which since cambia-782
// means a peek-less, joker-less deal rather than the old hardcoded engine config.
func testHouseRules(turnTimerSec, penaltyDrawCount int) *HouseRules {
	hr := DefaultHouseRules()
	hr.TurnTimerSec = turnTimerSec
	hr.PenaltyDrawCount = penaltyDrawCount
	return &hr
}

// setupTestGame initializes a CambiaGame instance with mock players and broadcasters for testing.
func setupTestGame(t *testing.T, numPlayers int, rules *HouseRules) (*CambiaGame, []*models.Player, *mockBroadcaster) {
	if numPlayers < 2 {
		numPlayers = 2
	}

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb

	if rules != nil {
		g.HouseRules = *rules
		// Ensure penalties are set for tests that need them.
	}
	// Use a very short turn duration for timeout tests, but allow disabling.
	if g.HouseRules.TurnTimerSec > 0 {
		g.TurnDuration = 100 * time.Millisecond
	} else {
		g.TurnDuration = 0
	}

	players := make([]*models.Player, numPlayers)
	for i := 0; i < numPlayers; i++ {
		player := &models.Player{
			ID:        uuid.New(),
			Connected: true,
			Conn:      nil,
			User:      &models.User{ID: uuid.New(), Username: "Player" + string(rune('A'+i))},
		}
		players[i] = player
		g.AddPlayer(player)
	}

	// Start the game flow.
	g.BeginPreGame()
	require.True(t, g.PreGameActive, "PreGame should be active after BeginPreGame")
	g.StartGame()
	require.True(t, g.Started, "Game should be marked as started")
	require.False(t, g.PreGameActive, "PreGame should be inactive after StartGame")

	mb.clear() // Clear events generated during setup.

	return g, players, mb
}

// currentTurnPlayer returns the player whose turn it currently is.
func currentTurnPlayer(g *CambiaGame) *models.Player {
	actingIdx := g.Engine.ActingPlayer()
	playerID := g.EngineToPlayer[actingIdx]
	return g.getPlayerByID(playerID)
}

// getPlayerIndex finds the index of a player within the game's Players slice.
func getPlayerIndex(g *CambiaGame, playerID uuid.UUID) int {
	for i, p := range g.Players {
		if p.ID == playerID {
			return i
		}
	}
	return -1
}

// TestBasicDrawDiscard verifies the standard draw from stockpile -> discard flow.
func TestBasicDrawDiscard(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	playerA := players[0]
	playerB := players[1]

	// Determine who goes first based on engine.
	firstPlayer := currentTurnPlayer(g)
	var currentPlayer, otherPlayer *models.Player
	if firstPlayer.ID == playerA.ID {
		currentPlayer = playerA
		otherPlayer = playerB
	} else {
		currentPlayer = playerB
		otherPlayer = playerA
	}

	// Action: current player draws from stockpile.
	g.HandlePlayerAction(currentPlayer.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	// Verify engine has pending discard.
	engineIdx := g.PlayerToEngine[currentPlayer.ID]
	require.Equal(t, engine.PendingDiscard, g.Engine.Pending.Type, "Engine should have pending discard")
	require.Equal(t, engineIdx, g.Engine.Pending.PlayerID, "Pending should belong to current player")

	// Verify drawn card UUID is set.
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "Drawn card UUID should be set")

	// Assert Events: Public draw, Private draw.
	lastPublicEvent := mb.getLastEvent()
	require.NotNil(t, lastPublicEvent, "Expected public draw event")
	assert.Equal(t, EventPlayerDrawStockpile, lastPublicEvent.Type)
	assert.Equal(t, currentPlayer.ID, lastPublicEvent.User.ID)
	require.NotNil(t, lastPublicEvent.Card, "Public draw event card missing")
	assert.NotEqual(t, uuid.Nil, lastPublicEvent.Card.ID, "Public draw event card ID missing")

	lastPrivateEvent := mb.getLastPlayerEvent(currentPlayer.ID)
	require.NotNil(t, lastPrivateEvent, "Expected private draw event")
	assert.Equal(t, EventPrivateDrawStockpile, lastPrivateEvent.Type)
	require.NotNil(t, lastPrivateEvent.Card, "Private draw event card missing")
	assert.NotEmpty(t, lastPrivateEvent.Card.Rank, "Private draw event should reveal rank")
	assert.Equal(t, drawnUUID, lastPrivateEvent.Card.ID)

	// Action: discard the drawn card.
	discardAction := models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	}
	preDiscardLen := int(g.Engine.DiscardLen)
	g.HandlePlayerAction(currentPlayer.ID, discardAction)

	// After discard (no ability card path or special triggered):
	// Check that either special is active or turn advanced.
	specialActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == currentPlayer.ID
	nextTurnPlayer := currentTurnPlayer(g)

	if specialActive {
		// Special was triggered - skip it.
		lastPublicEvent = mb.getLastEvent()
		require.NotNil(t, lastPublicEvent)
		assert.Equal(t, EventPlayerSpecialChoice, lastPublicEvent.Type)

		g.ProcessSpecialAction(currentPlayer.ID, "skip", nil, nil)
		nextTurnPlayer = currentTurnPlayer(g)
		assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should advance to other player after skip")
	} else {
		// No special - check turn advanced and discard pile grew.
		newDiscardLen := int(g.Engine.DiscardLen)
		assert.Greater(t, newDiscardLen, preDiscardLen, "Discard pile should have grown")
		assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should have advanced to other player")
	}
}

// TestStockpileDrawReshuffleEmitsEvent verifies that a stockpile draw off an empty stockpile
// (which forces the engine to reshuffle the discard pile back into the stockpile first, see
// engine.attemptReshuffle) broadcasts EventGameReshuffleStockpile with the post-reshuffle counts
// (cambia-763 F3). Before this, the constant had zero emission call sites and client-side discard
// counts only corrected on the next full sync_state.
func TestStockpileDrawReshuffleEmitsEvent(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)

	// Force an empty stockpile with a 4-card discard pile so the next stockpile draw must
	// reshuffle: attemptReshuffle keeps the top card and moves the other 3 into the stockpile,
	// then the draw itself pops one of those 3, leaving StockLen=2, DiscardLen=1.
	g.Engine.StockLen = 0
	g.CardTracker.StockLen = 0
	discardCards := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankFive),
		engine.NewCard(engine.SuitClubs, engine.RankSix),
		engine.NewCard(engine.SuitDiamonds, engine.RankSeven),
		engine.NewCard(engine.SuitSpades, engine.RankEight), // top
	}
	for i, c := range discardCards {
		g.Engine.DiscardPile[i] = c
		id := uuid.New()
		g.CardTracker.DiscardUUIDs[i] = id
		g.CardTracker.Registry[id] = engineCardToDetails(c, id)
	}
	g.Engine.DiscardLen = uint8(len(discardCards))
	g.CardTracker.DiscardLen = g.Engine.DiscardLen

	mb.clear()
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	reshuffleEvent := mb.findEventByType(EventGameReshuffleStockpile)
	require.NotNil(t, reshuffleEvent, "a stockpile draw off an empty stockpile must emit a reshuffle event")
	require.NotNil(t, reshuffleEvent.Payload, "reshuffle event must carry corrected counts")
	assert.EqualValues(t, 2, reshuffleEvent.Payload["stockpileSize"], "post-reshuffle stockpile size should reflect the 3 moved cards minus the 1 just drawn")
	assert.EqualValues(t, 1, reshuffleEvent.Payload["discardSize"], "post-reshuffle discard size should be just the preserved top card")
	assert.EqualValues(t, g.Engine.StockLen, reshuffleEvent.Payload["stockpileSize"])
	assert.EqualValues(t, g.Engine.DiscardLen, reshuffleEvent.Payload["discardSize"])

	// A subsequent draw (stockpile no longer empty) must not fire a spurious reshuffle event.
	discardUUID := g.CardTracker.Players[g.PlayerToEngine[actor.ID]].DrawnCardUUID
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": discardUUID.String()},
	})
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == actor.ID {
		g.ProcessSpecialAction(actor.ID, "skip", nil, nil)
	}
	nextActor := currentTurnPlayer(g)
	mb.clear()
	g.HandlePlayerAction(nextActor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	assert.Nil(t, mb.findEventByType(EventGameReshuffleStockpile), "a draw with cards already in the stockpile must not emit a reshuffle event")
}

// TestStockpileDrawReshuffleUpdatesTracker verifies that a stockpile draw which forces a reshuffle
// rebuilds the CardUUIDTracker mirror instead of leaving it stale (cambia-819). Before this fix,
// updateCardTracker's `if preStockLen > 0` guard skipped the tracker update outright whenever a
// draw reshuffled, so DrawnCardUUID kept its previous value and StockUUIDs/StockLen were never
// rebuilt: the drawn card would reach the client under a nil or duplicate UUID.
func TestStockpileDrawReshuffleUpdatesTracker(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]

	// Collect every UUID already tracked elsewhere (both hands) before the draw, so the drawn
	// card's UUID can be checked against them for uniqueness after the reshuffle.
	preexisting := map[uuid.UUID]bool{}
	for _, p := range g.CardTracker.Players {
		for i := uint8(0); i < engine.MaxHandSize; i++ {
			if id := p.HandUUIDs[i]; id != uuid.Nil {
				preexisting[id] = true
			}
		}
	}

	// Pull 2 still-undealt cards straight off the current stockpile (guaranteed distinct from
	// every hand and already registered by initCardTracker) and move them into the discard pile,
	// then empty the stockpile. Reshuffle keeps the top card and moves the other 1 back into the
	// stockpile, leaving a 1-card stockpile the draw immediately empties (StockLen 0 -> 0 through
	// a reshuffle, exercising the smallest non-trivial mirror rebuild).
	nonTopCard, nonTopUUID := g.Engine.Stockpile[0], g.CardTracker.StockUUIDs[0]
	topCard, topUUID := g.Engine.Stockpile[1], g.CardTracker.StockUUIDs[1]
	require.NotEqual(t, uuid.Nil, nonTopUUID)
	require.NotEqual(t, uuid.Nil, topUUID)

	g.Engine.StockLen = 0
	g.CardTracker.StockLen = 0
	g.Engine.DiscardPile[0], g.CardTracker.DiscardUUIDs[0] = nonTopCard, nonTopUUID
	g.Engine.DiscardPile[1], g.CardTracker.DiscardUUIDs[1] = topCard, topUUID // top
	g.Engine.DiscardLen = 2
	g.CardTracker.DiscardLen = g.Engine.DiscardLen
	preexisting[topUUID] = true

	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})

	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	assert.NotEqual(t, uuid.Nil, drawnUUID, "drawn card's UUID must not be nil after a reshuffling draw")
	assert.False(t, preexisting[drawnUUID], "drawn card's UUID must not be shared with any other previously tracked card")
	assert.Equal(t, g.Engine.StockLen, g.CardTracker.StockLen, "CardTracker.StockLen must match the post-reshuffle-and-draw stockpile length")
	assert.EqualValues(t, 0, g.Engine.StockLen, "the single reshuffled stockpile card should have been drawn, leaving the stockpile empty")

	// The drawn card's identity should be the same UUID the non-top discard card carried before
	// the reshuffle (identity travels with the UUID across a reshuffle), not a stale slot or a
	// freshly minted placeholder.
	assert.Equal(t, nonTopUUID, drawnUUID, "the reshuffled non-top discard card should be the one drawn, under its pre-reshuffle UUID")
	drawnDetails := g.CardTracker.Registry[drawnUUID]
	require.NotNil(t, drawnDetails, "drawn card must have a registry entry")
}

// TestBasicDrawReplace verifies the draw -> replace card flow.
func TestBasicDrawReplace(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	playerA := players[0]
	playerB := players[1]

	// Determine first player.
	firstPlayer := currentTurnPlayer(g)
	var currentPlayer, otherPlayer *models.Player
	if firstPlayer.ID == playerA.ID {
		currentPlayer = playerA
		otherPlayer = playerB
	} else {
		currentPlayer = playerB
		otherPlayer = playerA
	}

	engineIdx := g.PlayerToEngine[currentPlayer.ID]
	require.Greater(t, int(g.Engine.Players[engineIdx].HandLen), 0, "Player must have cards")

	originalHandLen := int(g.Engine.Players[engineIdx].HandLen)
	originalCardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[0]

	// Action: draw.
	g.HandlePlayerAction(currentPlayer.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "Drawn card UUID should be set")

	mb.clear()

	// Action: replace card at index 0.
	replaceAction := models.GameAction{
		ActionType: "action_replace",
		Payload: map[string]interface{}{
			"id":  originalCardUUID.String(),
			"idx": float64(0),
		},
	}
	g.HandlePlayerAction(currentPlayer.ID, replaceAction)

	// Verify state changes.
	require.Equal(t, originalHandLen, int(g.Engine.Players[engineIdx].HandLen), "Hand size should remain the same")
	newCardUUID := g.CardTracker.Players[engineIdx].HandUUIDs[0]
	assert.Equal(t, drawnUUID, newCardUUID, "Drawn card should now be at index 0 in hand")

	// Verify discard pile grew.
	discardLen := int(g.Engine.DiscardLen)
	require.Greater(t, discardLen, 0, "Discard pile should not be empty")
	discardTopUUID := g.CardTracker.DiscardUUIDs[discardLen-1]
	assert.Equal(t, originalCardUUID, discardTopUUID, "Original card should be on discard pile")

	// Assert events: find the discard event among all public events.
	// (A turn-start event may fire after the discard, so check by type.)
	discardEvent := mb.findEventByType(EventPlayerDiscard)
	require.NotNil(t, discardEvent, "Expected public discard event for replaced card")
	assert.Equal(t, EventPlayerDiscard, discardEvent.Type)
	assert.Equal(t, currentPlayer.ID, discardEvent.User.ID)
	require.NotNil(t, discardEvent.Card, "Discard event card missing")
	assert.Equal(t, originalCardUUID, discardEvent.Card.ID)
	require.NotNil(t, discardEvent.Card.Idx, "Discard event for replaced card should include index")
	assert.Equal(t, 0, *discardEvent.Card.Idx)

	// If no special action was triggered, turn should advance.
	specialActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == currentPlayer.ID
	nextTurnPlayer := currentTurnPlayer(g)

	// Default house rules: AllowReplaceAbilities=false, so no special on replace.
	require.False(t, specialActive, "Special action should NOT be active if AllowReplaceAbilities is false")
	assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should advance if AllowReplaceAbilities is false")
}

// TestSnapSuccess verifies a correct snap action.
func TestSnapSuccess(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	playerA := players[0]
	playerB := players[1]

	// Setup: Force a specific rank onto the discard pile.
	// We need to manipulate engine state directly to set up the test condition.
	// Inject a '7' (rank 6 in engine) at the top of discard pile.
	sevenCard := engine.NewCard(engine.SuitHearts, engine.RankSeven) // H7
	g.Engine.DiscardPile[g.Engine.DiscardLen] = sevenCard
	g.Engine.DiscardLen++

	// Generate a UUID for the discard top.
	discardTopUUID, _ := uuid.NewRandom()
	g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen-1] = discardTopUUID
	g.CardTracker.Registry[discardTopUUID] = engineCardToDetails(sevenCard, discardTopUUID)

	// Now inject a matching '7' into playerB's hand.
	// Find whichever player is NOT the current acting player for snap testing.
	// Snap can be done by either player (out of turn).
	snappingPlayer := playerB
	snappingEngineIdx := g.PlayerToEngine[snappingPlayer.ID]

	// Add a Seven to snapping player's hand in the engine.
	snapCard := engine.NewCard(engine.SuitSpades, engine.RankSeven) // S7
	handLen := g.Engine.Players[snappingEngineIdx].HandLen
	g.Engine.Players[snappingEngineIdx].Hand[handLen] = snapCard
	g.Engine.Players[snappingEngineIdx].HandLen++

	// Register UUID for snap card.
	snapCardUUID, _ := uuid.NewRandom()
	g.CardTracker.Players[snappingEngineIdx].HandUUIDs[handLen] = snapCardUUID
	g.CardTracker.Registry[snapCardUUID] = engineCardToDetails(snapCard, snapCardUUID)
	g.syncPlayerHandsFromEngine()

	initialHandSize := int(g.Engine.Players[snappingEngineIdx].HandLen)
	initialDiscardSize := int(g.Engine.DiscardLen)

	mb.clear()

	// Action: snap the '7'.
	snapAction := models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": snapCardUUID.String()},
	}
	g.HandlePlayerAction(snappingPlayer.ID, snapAction)

	// Assert State Changes.
	newHandSize := int(g.Engine.Players[snappingEngineIdx].HandLen)
	newDiscardSize := int(g.Engine.DiscardLen)
	assert.Equal(t, initialHandSize-1, newHandSize, "Player hand size should decrease by 1")
	assert.Equal(t, initialDiscardSize+1, newDiscardSize, "Discard pile size should increase by 1")

	// Check discard top is the snapped card.
	discardTopUUIDAfter := g.CardTracker.DiscardUUIDs[g.Engine.DiscardLen-1]
	assert.Equal(t, snapCardUUID, discardTopUUIDAfter, "Snapped card should now be top of discard")

	// Verify card is no longer in hand.
	_, foundIdx := g.findCardByID(snappingPlayer.ID, snapCardUUID)
	assert.Equal(t, -1, foundIdx, "Snapped card should no longer be in hand")

	// Assert Events: snap success.
	lastPublicEvent := mb.getLastEvent()
	require.NotNil(t, lastPublicEvent, "Expected public snap success event")
	assert.Equal(t, EventPlayerSnapSuccess, lastPublicEvent.Type)
	assert.Equal(t, snappingPlayer.ID, lastPublicEvent.User.ID)
	require.NotNil(t, lastPublicEvent.Card, "Snap success event card missing")
	assert.Equal(t, snapCardUUID, lastPublicEvent.Card.ID)
	_ = playerA
}

// TestSnapFailPenalty verifies penalties for incorrect snaps (wrong rank).
func TestSnapFailPenalty(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	playerA := players[0]
	playerB := players[1]
	penaltyCount := g.HouseRules.PenaltyDrawCount

	// Force a '7' on top of discard pile.
	sevenCard := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	discardTopIdx := g.Engine.DiscardLen
	g.Engine.DiscardPile[discardTopIdx] = sevenCard
	g.Engine.DiscardLen++

	discardUUID, _ := uuid.NewRandom()
	g.CardTracker.DiscardUUIDs[discardTopIdx] = discardUUID
	g.CardTracker.Registry[discardUUID] = engineCardToDetails(sevenCard, discardUUID)

	// Give playerB an '8' (wrong rank) to snap with.
	snappingPlayer := playerB
	snappingEngineIdx := g.PlayerToEngine[snappingPlayer.ID]

	// Reduce hand to 3 cards to ensure there's room for penaltyCount=2 penalty cards (max hand=6).
	// Cards are dealt face-down so we can just truncate.
	for g.Engine.Players[snappingEngineIdx].HandLen > 3 {
		lastIdx := g.Engine.Players[snappingEngineIdx].HandLen - 1
		removedUUID := g.CardTracker.Players[snappingEngineIdx].HandUUIDs[lastIdx]
		g.Engine.Players[snappingEngineIdx].Hand[lastIdx] = engine.EmptyCard
		g.Engine.Players[snappingEngineIdx].HandLen--
		g.CardTracker.Players[snappingEngineIdx].HandUUIDs[lastIdx] = uuid.Nil
		delete(g.CardTracker.Registry, removedUUID)
	}

	eightCard := engine.NewCard(engine.SuitSpades, engine.RankEight)
	handLen := g.Engine.Players[snappingEngineIdx].HandLen
	g.Engine.Players[snappingEngineIdx].Hand[handLen] = eightCard
	g.Engine.Players[snappingEngineIdx].HandLen++

	eightUUID, _ := uuid.NewRandom()
	g.CardTracker.Players[snappingEngineIdx].HandUUIDs[handLen] = eightUUID
	g.CardTracker.Registry[eightUUID] = engineCardToDetails(eightCard, eightUUID)
	g.syncPlayerHandsFromEngine()

	initialHandSize := int(g.Engine.Players[snappingEngineIdx].HandLen)
	initialStockSize := int(g.Engine.StockLen)

	mb.clear()

	// Action: snap incorrectly with '8'.
	snapAction := models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": eightUUID.String()},
	}
	g.HandlePlayerAction(snappingPlayer.ID, snapAction)

	// Assert State Changes: hand should grow by penalty count.
	newHandSize := int(g.Engine.Players[snappingEngineIdx].HandLen)
	newStockSize := int(g.Engine.StockLen)
	assert.Equal(t, initialHandSize+penaltyCount, newHandSize, "Player hand size should increase by penalty count")
	assert.Equal(t, initialStockSize-penaltyCount, newStockSize, "Stockpile size should decrease by penalty count")

	// Original card should still be in hand.
	foundCard, _ := g.findCardByID(snappingPlayer.ID, eightUUID)
	require.NotNil(t, foundCard, "Original card should still be in player's hand after failed snap")

	// Check events.
	publicEvents := mb.allEvents
	privateEventsB := mb.playerEvents[snappingPlayer.ID]

	require.GreaterOrEqual(t, len(publicEvents), 1+penaltyCount, "Expected snap fail + penalty public events")
	assert.Equal(t, EventPlayerSnapFail, publicEvents[0].Type)
	assert.Equal(t, snappingPlayer.ID, publicEvents[0].User.ID)

	// Check penalty events.
	publicPenaltyEventCount := 0
	for i := 1; i < len(publicEvents); i++ {
		if publicEvents[i].Type == EventPlayerSnapPenalty {
			publicPenaltyEventCount++
		}
	}
	assert.Equal(t, penaltyCount, publicPenaltyEventCount, "Expected correct number of public penalty events")

	require.Len(t, privateEventsB, penaltyCount, "Expected correct number of private penalty events")
	for i := 0; i < penaltyCount; i++ {
		assert.Equal(t, EventPrivateSnapPenalty, privateEventsB[i].Type)
		require.NotNil(t, privateEventsB[i].Card, "Private penalty event card missing")
		require.NotNil(t, privateEventsB[i].Card.Idx, "Private penalty event should name the hand slot")
		// Penalty cards are drawn unseen (cambia-820); see TestSnapPenaltyRevealsNoCardFace.
		assert.Empty(t, privateEventsB[i].Card.Rank, "Private penalty event must not reveal the card face")
	}
	_ = playerA
}

// TestCambiaCallAndEndgame verifies calling Cambia and the subsequent final round logic.
// Engine only supports 2 players, so this tests the 2-player Cambia flow.
func TestCambiaCallAndEndgame(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	playerA := players[0]
	playerB := players[1]

	// Identify first player.
	firstPlayer := currentTurnPlayer(g)
	var first, second *models.Player
	if firstPlayer.ID == playerA.ID {
		first = playerA
		second = playerB
	} else {
		first = playerB
		second = playerA
	}

	// Turn 1: First player draws and discards (skip any special).
	doSimpleTurn := func(player *models.Player) {
		g.HandlePlayerAction(player.ID, models.GameAction{ActionType: "action_draw_stockpile"})
		engineIdx := g.PlayerToEngine[player.ID]
		drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
		if drawnUUID != uuid.Nil {
			g.HandlePlayerAction(player.ID, models.GameAction{
				ActionType: "action_discard",
				Payload:    map[string]interface{}{"id": drawnUUID.String()},
			})
			if g.SpecialAction.Active && g.SpecialAction.PlayerID == player.ID {
				g.ProcessSpecialAction(player.ID, "skip", nil, nil)
			}
		}
	}

	doSimpleTurn(first)

	// Verify turn advanced to second.
	require.Equal(t, second.ID, currentTurnPlayer(g).ID, "Should be second player's turn")

	// Turn 2: Second player calls Cambia.
	g.HandlePlayerAction(second.ID, models.GameAction{ActionType: "action_cambia"})

	// Assert Cambia state.
	assert.True(t, g.Engine.IsCambiaCalled(), "Engine: CambiaCalled flag should be true")
	callerID := g.cambiaCallerID()
	assert.Equal(t, second.ID, callerID, "cambiaCallerID should be second player")

	// Assert events: find the cambia event (a turn event may follow it).
	cambiaEvent := mb.findEventByType(EventPlayerCambia)
	require.NotNil(t, cambiaEvent, "Expected player_cambia event")
	assert.Equal(t, EventPlayerCambia, cambiaEvent.Type)
	assert.Equal(t, second.ID, cambiaEvent.User.ID)

	// Turn advanced to first (only 2 players, so Cambia caller's turn just ended).
	// In 2-player game with Cambia called: first player gets their final turn,
	// then game ends.
	require.Equal(t, first.ID, currentTurnPlayer(g).ID, "Should be first player's turn (final)")
	require.False(t, g.GameOver, "Game should not be over yet")

	// Turn 3 (final): First player draws and discards.
	doSimpleTurn(first)

	// Game should end after first player's turn (they were the player before the caller in 2p game).
	assert.True(t, g.GameOver, "Game should be over after the final turn")

	// Assert game end event.
	gameEndEvent := mb.findEventByType(EventGameEnd)
	require.NotNil(t, gameEndEvent, "Expected game end event")
	assert.Equal(t, EventGameEnd, gameEndEvent.Type)
	require.NotNil(t, gameEndEvent.Payload, "Game end payload missing")
	assert.Contains(t, gameEndEvent.Payload, "scores", "Game end payload missing scores")
	assert.Contains(t, gameEndEvent.Payload, "winner", "Game end payload missing winner")
}

// TestCambiaLock verifies that swapping with a player who has called Cambia fails.
func TestCambiaLock(t *testing.T) {
	hr := testHouseRules(0, 2)
	hr.AllowDrawFromDiscardPile = true
	g, players, mb := setupTestGame(t, 2, hr)
	playerA := players[0]
	playerB := players[1]

	// Identify first player.
	firstPlayer := currentTurnPlayer(g)
	var first, second *models.Player
	if firstPlayer.ID == playerA.ID {
		first = playerA
		second = playerB
	} else {
		first = playerB
		second = playerA
	}

	firstEngineIdx := g.PlayerToEngine[first.ID]
	secondEngineIdx := g.PlayerToEngine[second.ID]

	// Turn 1: First player does a simple turn.
	g.HandlePlayerAction(first.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[firstEngineIdx].DrawnCardUUID
	if drawnUUID != uuid.Nil {
		g.HandlePlayerAction(first.ID, models.GameAction{
			ActionType: "action_discard",
			Payload:    map[string]interface{}{"id": drawnUUID.String()},
		})
		if g.SpecialAction.Active && g.SpecialAction.PlayerID == first.ID {
			g.ProcessSpecialAction(first.ID, "skip", nil, nil)
		}
	}

	// Turn 2: Second player calls Cambia.
	require.Equal(t, second.ID, currentTurnPlayer(g).ID)
	g.HandlePlayerAction(second.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	// Mark second player's HasCalledCambia in the Player model for Cambia lock check.
	secondPlayerModel := g.getPlayerByID(second.ID)
	require.NotNil(t, secondPlayerModel)
	secondPlayerModel.HasCalledCambia = true

	// Turn 3 (final): First player draws and discards a Jack to trigger blind swap.
	require.Equal(t, first.ID, currentTurnPlayer(g).ID, "Should be first player's final turn")

	// Inject a Jack into first player's hand (as drawn card) via engine Pending.
	// We'll do this by drawing from stockpile and then manually setting up a Jack scenario.
	// For simplicity: draw, then manually trigger a swap_blind attempt via ProcessSpecialAction.
	g.HandlePlayerAction(first.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID2 := g.CardTracker.Players[firstEngineIdx].DrawnCardUUID

	// Inject a Jack as the drawn card in engine state.
	jackCard := engine.NewCard(engine.SuitClubs, engine.RankJack)
	jackUUID, _ := uuid.NewRandom()
	g.Engine.Pending.Data[0] = uint8(jackCard)
	g.Engine.Pending.Data[1] = engine.DrawnFromStockpile
	g.CardTracker.Players[firstEngineIdx].DrawnCardUUID = jackUUID
	g.CardTracker.Registry[jackUUID] = engineCardToDetails(jackCard, jackUUID)
	_ = drawnUUID2

	// Discard the Jack (has ability: swap_blind).
	g.HandlePlayerAction(first.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": jackUUID.String()},
	})

	// Verify special action is active.
	require.True(t, g.SpecialAction.Active && g.SpecialAction.PlayerID == first.ID && g.SpecialAction.CardRank == "J")

	mb.clear()

	// First player's hand and second player's hand.
	require.Greater(t, int(g.Engine.Players[firstEngineIdx].HandLen), 0, "First player needs cards")
	require.Greater(t, int(g.Engine.Players[secondEngineIdx].HandLen), 0, "Second player needs cards")

	cardA_UUID := g.CardTracker.Players[firstEngineIdx].HandUUIDs[0]
	cardB_UUID := g.CardTracker.Players[secondEngineIdx].HandUUIDs[0]

	// Attempt blind swap involving second player (who called Cambia).
	swapCard1Data := map[string]interface{}{
		"id":  cardA_UUID.String(),
		"idx": float64(0),
		"user": map[string]interface{}{"id": first.ID.String()},
	}
	swapCard2Data := map[string]interface{}{
		"id":  cardB_UUID.String(),
		"idx": float64(0),
		"user": map[string]interface{}{"id": second.ID.String()},
	}
	g.ProcessSpecialAction(first.ID, "swap_blind", swapCard1Data, swapCard2Data)

	// Assert: special action should still be active (Cambia lock prevented swap).
	specialStillActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == first.ID

	assert.True(t, specialStillActive, "Special action should still be active after failed swap attempt due to Cambia lock")

	// Assert: private fail event sent.
	lastPrivateEvent := mb.getLastPlayerEvent(first.ID)
	require.NotNil(t, lastPrivateEvent, "Expected a private event for first player")
	assert.Equal(t, EventPrivateSpecialFail, lastPrivateEvent.Type)
	assert.Equal(t, "swap_blind", lastPrivateEvent.Special)
	require.NotNil(t, lastPrivateEvent.Payload)
	assert.Contains(t, lastPrivateEvent.Payload["message"], "called Cambia")

	// Assert: cards were NOT swapped.
	assert.Equal(t, cardA_UUID, g.CardTracker.Players[firstEngineIdx].HandUUIDs[0], "Card A should not have been swapped")
	assert.Equal(t, cardB_UUID, g.CardTracker.Players[secondEngineIdx].HandUUIDs[0], "Card B should not have been swapped")
}
