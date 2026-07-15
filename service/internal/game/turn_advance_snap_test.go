// internal/game/turn_advance_snap_test.go
//
// Regression tests for cambia-506: turn advancement wedges after a special-ability resolution
// (or a skipped ability) when the discarded card triggers a snap phase. The engine advances the
// turn while resolving the auto-passed snap phase, but the service adapter only notified its turn
// lifecycle (TurnID++, timer re-arm, game_player_turn broadcast) on the non-snap path. On the snap
// path the previous player's turn timer stayed armed and no game_player_turn was ever emitted, so
// clients were stranded and the stale timer later auto-played the wrong (previous) player.
package game

import (
	"testing"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

// stopGameTimer neutralizes any armed turn timer so it cannot fire after a test returns.
func stopGameTimer(g *CambiaGame) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.GameOver = true
	g.Started = false
	if g.turnTimer != nil {
		g.turnTimer.Stop()
		g.turnTimer = nil
	}
}

// giveMatchingCard overwrites playerEngineIdx's hand slot with card and keeps the UUID tracker
// registry consistent, so the card is a live snap candidate for the given rank.
func giveMatchingCard(g *CambiaGame, playerEngineIdx uint8, slot uint8, card engine.Card) {
	g.Engine.Players[playerEngineIdx].Hand[slot] = card
	handUUID := g.CardTracker.Players[playerEngineIdx].HandUUIDs[slot]
	g.CardTracker.Registry[handUUID] = engineCardToDetails(card, handUUID)
	g.syncPlayerHandsFromEngine()
}

// opponentOf returns the id in ids that is not cur.
func opponentOf(ids []uuid.UUID, cur uuid.UUID) uuid.UUID {
	for _, id := range ids {
		if id != cur {
			return id
		}
	}
	return uuid.Nil
}

// driveDrawDiscardSeven forces the acting player to draw a 7 (an ability card) and discard it,
// leaving a pending peek_self choice. Returns the acting player's id and engine index.
func driveDrawDiscardSeven(t *testing.T, g *CambiaGame) (uuid.UUID, uint8) {
	t.Helper()
	cur := currentTurnPlayer(g)
	curIdx := g.PlayerToEngine[cur.ID]

	_ = forceStockTop(g, engine.NewCard(engine.SuitClubs, engine.RankSeven))

	g.HandlePlayerAction(cur.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[curIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "draw should set a drawn card UUID")

	g.HandlePlayerAction(cur.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "discarding a 7 should open an ability choice")
	require.Equal(t, "7", g.SpecialAction.CardRank)
	require.Equal(t, curIdx, g.Engine.ActingPlayer(),
		"turn must still belong to the actor while the ability is pending")
	return cur.ID, curIdx
}

// TestPeekSelfAdvancesTurnThroughSnapPhase drives draw -> discard(7) -> resolve peek_self where the
// opponent holds a matching 7 (so the discard opens a snap phase). The turn must advance to the
// opponent: TurnID increments, a game_player_turn for the opponent is emitted, and the timer
// re-arms. Before the fix, onTurnAdvanced was skipped on the snap path and none of that happened.
func TestPeekSelfAdvancesTurnThroughSnapPhase(t *testing.T) {
	g, ids, mb := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	curID, curIdx := driveDrawDiscardSeven(t, g)
	oppID := opponentOf(ids, curID)
	oppIdx := g.PlayerToEngine[oppID]

	// Opponent holds a matching 7 -> the discarded 7 opens a snap phase on ability resolution.
	giveMatchingCard(g, oppIdx, 0, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	preTurnID := g.TurnID
	preStock := g.Engine.StockLen

	mb.clear()
	ownSlot0 := g.CardTracker.Players[curIdx].HandUUIDs[0]
	g.ProcessSpecialAction(curID, "peek_self", cardTarget(ownSlot0, curID, 0), nil)

	assert.Equal(t, oppIdx, g.Engine.ActingPlayer(),
		"turn must advance to the opponent after the ability + auto-passed snap phase")
	assert.Equal(t, preTurnID+1, g.TurnID, "service TurnID must increment exactly once for the advance")

	ev := mb.findEventByType(EventGamePlayerTurn)
	require.NotNil(t, ev, "a game_player_turn must be emitted for the next player after the ability resolves")
	require.NotNil(t, ev.User)
	assert.Equal(t, oppID, ev.User.ID, "the emitted turn event must name the opponent")

	require.NotNil(t, g.turnTimer, "a fresh turn timer must be armed for the new player")
	assert.False(t, g.TurnDeadline.IsZero(), "the turn deadline must re-arm on the advance")
	assert.True(t, g.TurnDeadline.After(time.Now()), "the re-armed deadline must be in the future")

	assert.Equal(t, preStock, g.Engine.StockLen, "no stray timeout auto-draw should have run after the draw")

	// Wedge check: the previous player is rejected, the opponent can act.
	mb.clear()
	g.HandlePlayerAction(curID, models.GameAction{ActionType: "action_draw_stockpile"})
	rej := mb.getLastPlayerEvent(curID)
	require.NotNil(t, rej, "previous player's extra draw should get a private failure")
	assert.Equal(t, EventPrivateSpecialFail, rej.Type)
	assert.Equal(t, "It's not your turn.", rej.Payload["message"])

	g.HandlePlayerAction(oppID, models.GameAction{ActionType: "action_draw_stockpile"})
	assert.Equal(t, engine.PendingDiscard, g.Engine.Pending.Type, "opponent's draw should be accepted")
	assert.Equal(t, oppIdx, g.Engine.Pending.PlayerID, "opponent's draw should register as their pending")
}

// TestSkipAbilityAdvancesTurnThroughSnapPhase covers the skip path: draw -> discard(7) -> skip.
// The skip discards the 7 with no ability, which still opens a snap phase (opponent holds a 7), so
// the same turn-advance lifecycle must fire.
func TestSkipAbilityAdvancesTurnThroughSnapPhase(t *testing.T) {
	g, ids, mb := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	curID, _ := driveDrawDiscardSeven(t, g)
	oppID := opponentOf(ids, curID)
	oppIdx := g.PlayerToEngine[oppID]

	giveMatchingCard(g, oppIdx, 0, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	preTurnID := g.TurnID

	mb.clear()
	g.ProcessSpecialAction(curID, "skip", nil, nil)

	assert.Equal(t, oppIdx, g.Engine.ActingPlayer(),
		"turn must advance to the opponent after skipping the ability through the snap phase")
	assert.Equal(t, preTurnID+1, g.TurnID, "service TurnID must increment exactly once for the advance")

	ev := mb.findEventByType(EventGamePlayerTurn)
	require.NotNil(t, ev, "a game_player_turn must be emitted for the next player after skip")
	require.NotNil(t, ev.User)
	assert.Equal(t, oppID, ev.User.ID, "the emitted turn event must name the opponent")

	require.NotNil(t, g.turnTimer, "a fresh turn timer must be armed for the new player")
	assert.False(t, g.TurnDeadline.IsZero(), "the turn deadline must re-arm on the advance")
	assert.True(t, g.TurnDeadline.After(time.Now()), "the re-armed deadline must be in the future")
}

// TestNoStaleTimeoutAfterAbilitySnapAdvance is the end-to-end reproduction of the evidence-log
// wedge: the previous player's turn timer must not fire and auto-play them after the ability
// resolves through a snap phase. The opponent is disconnected before resolution so no fresh timer
// arms for them; the only timer that could fire is the previous player's. Before the fix it fired
// ("timed out without drawing") and drew for the previous player; after the fix it is neutralized.
func TestNoStaleTimeoutAfterAbilitySnapAdvance(t *testing.T) {
	g, ids, mb := buildTimedTestGame(t, 80*time.Millisecond)
	defer stopGameTimer(g)

	curID, curIdx := driveDrawDiscardSeven(t, g)
	oppID := opponentOf(ids, curID)
	oppIdx := g.PlayerToEngine[oppID]

	giveMatchingCard(g, oppIdx, 0, engine.NewCard(engine.SuitHearts, engine.RankSeven))

	// Disconnect the opponent so the advance arms no new timer; only the previous player's stale
	// timer remains as a candidate to fire.
	g.mu.Lock()
	if p := g.getPlayerByID(oppID); p != nil {
		p.Connected = false
	}
	g.mu.Unlock()

	ownSlot0 := g.CardTracker.Players[curIdx].HandUUIDs[0]
	g.ProcessSpecialAction(curID, "peek_self", cardTarget(ownSlot0, curID, 0), nil)

	require.Equal(t, oppIdx, g.Engine.ActingPlayer(), "engine turn should have advanced to the opponent")

	// Observe past the stale timer's original deadline.
	mb.clear()
	time.Sleep(250 * time.Millisecond)

	g.mu.Lock()
	defer g.mu.Unlock()
	for _, ev := range mb.allEvents {
		if ev.User != nil && ev.User.ID == curID &&
			(ev.Type == EventPlayerDrawStockpile || ev.Type == EventPlayerDiscard) {
			t.Fatalf("stale turn timer auto-played the previous player: got %s for %s", ev.Type, curID)
		}
	}
}
