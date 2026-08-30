// internal/game/replace_ability_test.go
//
// Tests for the ability a replace triggers when AllowReplaceAbilities is on (cambia-1125). Ranked
// queues play that rule (MATCHMAKING.md 5.2), so a replace that puts a 7/8/9/T/J/Q/K on the discard
// pile is the first live path that reaches engine replace()'s ability arm.
//
// Two invariants live here:
//
//   - The prompt comes from the engine's pending state, never from a rank the service re-derives.
//     replace() arms an ability only when the drawn card came off the STOCKPILE and the ability has
//     a legal target (engine/actions.go, canUseAbility), so a service that prompts on the rank alone
//     prompts for abilities the engine never armed.
//   - An ability the engine armed is mandatory. The engine folds the decline into the discard action
//     (DiscardNoAbility vs DiscardWithAbility) and legalAbilitySelect offers an armed ability only
//     targets, so nothing in the action space declines one. Skipping it service-side leaves the
//     engine holding the pending action forever, which refuses every later action at the table and
//     kills the turn timer on its own refused timeout draw.
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

// replaceRules returns the ranked-preset rules this file exercises: replace abilities on, and the
// discard pile drawable so the "drawn from the discard pile" branch is reachable.
func replaceRules(turnTimerSec int) *HouseRules {
	hr := testHouseRules(turnTimerSec, 2)
	hr.AllowReplaceAbilities = true
	hr.AllowDrawFromDiscardPile = true
	return hr
}

// forceHandCard overwrites one hand slot, wiring a fresh UUID into the tracker so the card the
// replace moves onto the discard pile resolves through the registry like a dealt one.
func forceHandCard(g *CambiaGame, seat uint8, slot uint8, c engine.Card) uuid.UUID {
	g.Engine.Players[seat].Hand[slot] = c
	id := uuid.New()
	g.CardTracker.Players[seat].HandUUIDs[slot] = id
	g.CardTracker.Registry[id] = engineCardToDetails(c, id)
	return id
}

// drawThenReplace forces card into the actor's slot 2, draws a plain 3 off the stockpile and
// replaces slot 2 with it, which puts card on the discard pile.
func drawThenReplace(t *testing.T, g *CambiaGame, actor *models.Player, card engine.Card) uuid.UUID {
	t.Helper()
	seat := g.PlayerToEngine[actor.ID]
	require.GreaterOrEqual(t, int(g.Engine.Players[seat].HandLen), 3, "need a slot 2 to replace")
	handUUID := forceHandCard(g, seat, 2, card)
	forceStockTop(g, engine.NewCard(engine.SuitHearts, engine.RankThree))

	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_replace",
		Payload:    map[string]interface{}{"id": handUUID.String(), "idx": float64(2)},
	})
	return handUUID
}

// lastEventOfType returns the most recent broadcast event of the given type.
func lastEventOfType(mb *mockBroadcaster, evType GameEventType) *GameEvent {
	for i := len(mb.allEvents) - 1; i >= 0; i-- {
		if mb.allEvents[i].Type == evType {
			return &mb.allEvents[i]
		}
	}
	return nil
}

// TestReplaceAbilityPromptsFromEnginePending verifies a stockpile draw replaced onto an ability
// card arms the engine's pending ability, prompts its owner, marks the prompt mandatory and leaves
// the turn timer running so the player has a window to answer in.
func TestReplaceAbilityPromptsFromEnginePending(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, replaceRules(15))
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	tenUUID := drawThenReplace(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankTen))

	require.Equal(t, engine.PendingPeekOther, g.Engine.Pending.Type, "a replaced 10 arms peek-other")
	require.Equal(t, seat, g.Engine.Pending.PlayerID, "the ability belongs to the replacing seat")
	assert.True(t, g.SpecialAction.Active, "the service should prompt for the ability the engine armed")
	assert.Equal(t, actor.ID, g.SpecialAction.PlayerID)
	assert.Equal(t, "T", g.SpecialAction.CardRank)
	assert.True(t, g.SpecialAction.Mandatory, "an engine-armed ability cannot be declined")
	assert.False(t, g.TurnDeadline.IsZero(), "the ability window needs an armed turn deadline")

	choice := lastEventOfType(mb, EventPlayerSpecialChoice)
	require.NotNil(t, choice, "the table should be told the ability is pending")
	assert.Equal(t, "peek_other", choice.Special)
	require.NotNil(t, choice.Card)
	assert.Equal(t, tenUUID, choice.Card.ID, "the prompt names the card that went to the pile")
	assert.Equal(t, true, choice.Payload["mandatory"], "the client needs to know it cannot offer a skip")

	obf := g.GetCurrentObfuscatedGameState(actor.ID)
	require.NotNil(t, obf.SpecialAction, "a resync mid-ability restores the prompt")
	assert.True(t, obf.SpecialAction.Mandatory, "and restores that it cannot be skipped")
}

// TestReplaceAbilitySkipIsRefused verifies skipping an engine-armed ability is refused rather than
// silently clearing the prompt: the wedge was that the service cleared its own state and announced
// a new turn while the engine kept the pending ability, refusing everything sent afterwards.
func TestReplaceAbilitySkipIsRefused(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, replaceRules(15))
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankTen))
	require.True(t, g.SpecialAction.Active)
	mb.clear()

	g.ProcessSpecialAction(actor.ID, "skip", nil, nil)

	assert.True(t, g.SpecialAction.Active, "the prompt stands: the ability still has to be answered")
	assert.Equal(t, engine.PendingPeekOther, g.Engine.Pending.Type, "the engine still holds the ability")
	assert.Equal(t, seat, g.Engine.ActingPlayer(), "the turn has not moved")
	assert.Nil(t, lastEventOfType(mb, EventGamePlayerTurn), "no turn may be announced over a pending ability")
	assert.Equal(t, 1, countPlayerEvents(mb, actor.ID, EventPrivateSpecialFail), "the player is told why")

	// The table is still playable: the ability resolves and the turn ends.
	oppSeat := 1 - seat
	oppID := g.EngineToPlayer[oppSeat]
	oppCard := g.CardTracker.Players[oppSeat].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(oppCard, oppID, 0), nil)

	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "resolving the ability clears the engine")
	assert.False(t, g.SpecialAction.Active, "and the prompt")
	assert.Equal(t, oppSeat, g.Engine.ActingPlayer(), "and ends the turn")
}

// TestReplaceAbilityTimeoutResolvesTheTurn verifies the turn timer settles an engine-armed ability
// instead of skipping it. A skipped one left the engine pending, so the next timeout's fallback
// draw was refused and the timer was never rescheduled: the table sat on a dead 0:00 clock.
func TestReplaceAbilityTimeoutResolvesTheTurn(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, replaceRules(15))
	g.TurnDuration = 60 * time.Millisecond
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankTen))
	require.True(t, g.SpecialAction.Active)

	require.Eventually(t, func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.Engine.Pending.Type == engine.PendingNone
	}, 2*time.Second, 10*time.Millisecond, "the timeout must resolve the ability the engine armed")

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.False(t, g.SpecialAction.Active, "the prompt is settled")
	assert.Equal(t, 1-seat, g.Engine.ActingPlayer(), "the turn moved on")
	assert.False(t, g.TurnDeadline.IsZero(), "the next player's turn is armed, not left on a dead clock")
}

// TestReplaceAbilityKingTimeoutKeepsCardsInPlace verifies the King's auto-resolution takes the
// defensive line the rest of the timeout path takes: it looks, then declines the swap, so no card
// moves that the player did not ask to move.
func TestReplaceAbilityKingTimeoutKeepsCardsInPlace(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, replaceRules(15))
	g.TurnDuration = 60 * time.Millisecond
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]
	oppSeat := 1 - seat

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing))
	require.Equal(t, engine.PendingKingLook, g.Engine.Pending.Type)

	ownSlot0 := g.CardTracker.Players[seat].HandUUIDs[0]
	oppSlot0 := g.CardTracker.Players[oppSeat].HandUUIDs[0]

	require.Eventually(t, func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.Engine.Pending.Type == engine.PendingNone
	}, 2*time.Second, 10*time.Millisecond, "the King has to settle for the turn to end")

	g.mu.Lock()
	defer g.mu.Unlock()
	assert.Equal(t, ownSlot0, g.CardTracker.Players[seat].HandUUIDs[0], "no card of the actor's moved")
	assert.Equal(t, oppSlot0, g.CardTracker.Players[oppSeat].HandUUIDs[0], "and none of the opponent's")
	assert.Equal(t, oppSeat, g.Engine.ActingPlayer(), "the turn moved on")
}

// TestReplaceFromDiscardPileDoesNotPrompt verifies a replace fed by a discard-pile draw never
// prompts. RULES.md 3B is explicit that abilities do not fire on a discard-pile draw, and
// engine replace() honours it, so a service that prompted on the rank alone left the previous
// player holding a prompt the engine had never armed while the turn had already moved on.
func TestReplaceFromDiscardPileDoesNotPrompt(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, replaceRules(15))
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	tenUUID := forceHandCard(g, seat, 2, engine.NewCard(engine.SuitClubs, engine.RankTen))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_discardpile"})
	require.Equal(t, engine.PendingDiscard, g.Engine.Pending.Type, "the discard draw is pending")
	mb.clear()

	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_replace",
		Payload:    map[string]interface{}{"id": tenUUID.String(), "idx": float64(2)},
	})

	assert.False(t, g.SpecialAction.Active, "no ability fires off a discard-pile draw")
	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type)
	assert.Equal(t, 1-seat, g.Engine.ActingPlayer(), "the turn ends on the replace")
	assert.Nil(t, lastEventOfType(mb, EventPlayerSpecialChoice), "and nothing prompts for one")
}

// TestReplaceAbilityAtFFASeat verifies a replace-triggered ability from a seat past 1 at a 4-seat
// table prompts and then resolves against the seat the client names. The ranked FFA-4 queue plays
// AllowReplaceAbilities, and engine replace()'s ability gate used to resolve its opponent as
// 1-acting, which underflows from seat 2 and panicked the game on the WS handler goroutine
// (cambia-1125).
func TestReplaceAbilityAtFFASeat(t *testing.T) {
	g, _, mb := setupTestGame(t, 4, replaceRules(15))
	g.Engine.CurrentPlayer = 2
	actor := currentTurnPlayer(g)
	require.Equal(t, uint8(2), g.PlayerToEngine[actor.ID], "the fixture should be acting from seat 2")

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankTen))

	require.Equal(t, engine.PendingPeekOther, g.Engine.Pending.Type, "seat 2's replaced 10 arms peek-other")
	require.True(t, g.SpecialAction.Active)
	require.NotNil(t, lastEventOfType(mb, EventPlayerSpecialChoice))

	// Resolve it against seat 3, which is neither seat 1 nor the seat a 2-player derivation names.
	targetID := g.EngineToPlayer[3]
	targetCard := g.CardTracker.Players[3].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(targetCard, targetID, 0), nil)

	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "the ability resolves")
	assert.False(t, g.SpecialAction.Active)
	assert.Equal(t, uint8(3), g.Engine.ActingPlayer(), "and the turn moves to the next seat")
}

// TestReplaceAbilityFizzleDoesNotPrompt verifies an ability the engine refuses to arm because it
// has no legal target never reaches the client either. LockCallerHand plus a called Cambia is the
// reachable case: canUseAbility returns false for a swap that could only target the caller.
func TestReplaceAbilityFizzleDoesNotPrompt(t *testing.T) {
	hr := replaceRules(15)
	hr.LockCallerHand = true
	g, _, mb := setupTestGame(t, 2, hr)

	// The opponent calls Cambia, then the turn comes back round to the actor.
	caller := currentTurnPlayer(g)
	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled(), "Cambia should be called")

	actor := currentTurnPlayer(g)
	require.NotEqual(t, caller.ID, actor.ID, "the turn should have moved to the other seat")
	mb.clear()

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankJack))

	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "a swap that can only target the locked caller fizzles")
	assert.False(t, g.SpecialAction.Active, "so nothing is pending service-side either")
	assert.Nil(t, lastEventOfType(mb, EventPlayerSpecialChoice), "and nothing prompts for one")
}
