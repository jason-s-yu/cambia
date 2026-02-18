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

func (mb *mockBroadcaster) broadcastFn(ev GameEvent) {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	mb.allEvents = append(mb.allEvents, ev)
}

func (mb *mockBroadcaster) broadcastToPlayerFn(playerID uuid.UUID, ev GameEvent) {
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

// setupTestGame initializes a CambiaGame instance with mock players and broadcasters for testing.
// The engine requires exactly 2 players.
func setupTestGame(t *testing.T, numPlayers int, rules *HouseRules) (*CambiaGame, []*models.Player, *mockBroadcaster) {
	// Engine requires exactly 2 players; clamp to 2 for engine-backed tests.
	if numPlayers != engine.MaxPlayers {
		t.Logf("Note: engine requires %d players, adjusting from %d", engine.MaxPlayers, numPlayers)
		numPlayers = engine.MaxPlayers
	}

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.BroadcastFn = mb.broadcastFn
	g.BroadcastToPlayerFn = mb.broadcastToPlayerFn

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
	g, players, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
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
	g.Mu.Lock()
	specialActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == currentPlayer.ID
	nextTurnPlayer := currentTurnPlayer(g)
	g.Mu.Unlock()

	if specialActive {
		// Special was triggered — skip it.
		lastPublicEvent = mb.getLastEvent()
		require.NotNil(t, lastPublicEvent)
		assert.Equal(t, EventPlayerSpecialChoice, lastPublicEvent.Type)

		g.ProcessSpecialAction(currentPlayer.ID, "skip", nil, nil)
		g.Mu.Lock()
		nextTurnPlayer = currentTurnPlayer(g)
		g.Mu.Unlock()
		assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should advance to other player after skip")
	} else {
		// No special — check turn advanced and discard pile grew.
		newDiscardLen := int(g.Engine.DiscardLen)
		assert.Greater(t, newDiscardLen, preDiscardLen, "Discard pile should have grown")
		assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should have advanced to other player")
	}
}

// TestBasicDrawReplace verifies the draw -> replace card flow.
func TestBasicDrawReplace(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
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
	g.Mu.Lock()
	specialActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == currentPlayer.ID
	nextTurnPlayer := currentTurnPlayer(g)
	g.Mu.Unlock()

	// Default house rules: AllowReplaceAbilities=false, so no special on replace.
	require.False(t, specialActive, "Special action should NOT be active if AllowReplaceAbilities is false")
	assert.Equal(t, otherPlayer.ID, nextTurnPlayer.ID, "Turn should advance if AllowReplaceAbilities is false")
}

// TestSnapSuccess verifies a correct snap action.
func TestSnapSuccess(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
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
	g, players, mb := setupTestGame(t, 2, &HouseRules{PenaltyDrawCount: 2, TurnTimerSec: 0})
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
		assert.NotEmpty(t, privateEventsB[i].Card.Rank, "Private penalty event should reveal card details")
	}
	_ = playerA
}

// TestCambiaCallAndEndgame verifies calling Cambia and the subsequent final round logic.
// Engine only supports 2 players, so this tests the 2-player Cambia flow.
func TestCambiaCallAndEndgame(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
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
	g, players, mb := setupTestGame(t, 2, &HouseRules{AllowDrawFromDiscardPile: true, TurnTimerSec: 0, PenaltyDrawCount: 2})
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
	g.Mu.Lock()
	specialStillActive := g.SpecialAction.Active && g.SpecialAction.PlayerID == first.ID
	g.Mu.Unlock()

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
