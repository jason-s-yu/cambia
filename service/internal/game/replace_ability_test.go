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

// handSnapshot copies a seat's hand as both the tracker and the engine hold it, so a test can
// assert that no card of that seat's moved.
func handSnapshot(g *CambiaGame, seat uint8) ([]uuid.UUID, []engine.Card) {
	n := g.Engine.Players[seat].HandLen
	return append([]uuid.UUID(nil), g.CardTracker.Players[seat].HandUUIDs[:n]...),
		append([]engine.Card(nil), g.Engine.Players[seat].Hand[:n]...)
}

// lastPlayerEventOfType returns the most recent private event of the given type sent to a player.
func lastPlayerEventOfType(mb *mockBroadcaster, playerID uuid.UUID, evType GameEventType) *GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	events := mb.playerEvents[playerID]
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Type == evType {
			return &events[i]
		}
	}
	return nil
}

// lockedCallerTable seats four players with LockCallerHand on, has seat 0 call Cambia and hands
// back the game with seat 1 to act. Seat 1 is the seat that exposes a target chosen by seat order:
// its lowest-numbered opponent still holding a card is the locked caller.
func lockedCallerTable(t *testing.T) (*CambiaGame, *mockBroadcaster, *models.Player, *models.Player) {
	t.Helper()
	hr := replaceRules(15)
	hr.LockCallerHand = true
	g, _, mb := setupTestGame(t, 4, hr)
	g.TurnDuration = 60 * time.Millisecond

	g.Engine.CurrentPlayer = 0
	caller := currentTurnPlayer(g)
	require.Equal(t, uint8(0), g.PlayerToEngine[caller.ID], "seat 0 calls Cambia")
	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled(), "Cambia should be called")

	actor := currentTurnPlayer(g)
	require.Equal(t, uint8(1), g.PlayerToEngine[actor.ID], "the turn should have moved to seat 1")
	return g, mb, caller, actor
}

// TestReplaceAbilityTimeoutSkipsLockedCaller verifies the timeout auto-resolution picks a swap
// target the engine's own legal mask accepts. LockCallerHand freezes the Cambia caller's hand
// against swaps (RULES.md 3C), which canUseAbility and both legal-mask builders honour for a blind
// swap and a King look (engine/legal.go reachableOpponent, nplayerLegalAbilitySelect). The
// auto-resolution named the lowest-numbered seat still holding a card instead, and the engine's
// N-player apply path carries no guard of its own, so the swap landed on the locked hand.
func TestReplaceAbilityTimeoutSkipsLockedCaller(t *testing.T) {
	g, _, caller, actor := lockedCallerTable(t)
	callerSeat := g.PlayerToEngine[caller.ID]
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankJack))
	require.Equal(t, engine.PendingBlindSwap, g.Engine.Pending.Type, "seats 2 and 3 keep the swap armed")

	lockedUUIDs, lockedCards := handSnapshot(g, callerSeat)

	require.Eventually(t, func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.Engine.Pending.Type == engine.PendingNone
	}, 2*time.Second, 10*time.Millisecond, "the timeout has to settle the ability for the turn to end")

	g.mu.Lock()
	defer g.mu.Unlock()
	gotUUIDs, gotCards := handSnapshot(g, callerSeat)
	assert.Equal(t, lockedUUIDs, gotUUIDs, "the locked caller's hand may not be swapped into")
	assert.Equal(t, lockedCards, gotCards, "and the engine must hold the cards it did")
	assert.NotEqual(t, seat, g.Engine.ActingPlayer(), "the turn moved on")
	assert.False(t, g.TurnDeadline.IsZero(), "and the next turn is armed, not left on a dead clock")
}

// TestReplaceAbilityKingTimeoutSkipsLockedCaller is the King half of the same rule. The look moves
// no card, so the seat it named is read off the reveal it fires: naming the locked caller there
// puts the King's swap decision on a hand no action may move, and shows the actor a card the lock
// keeps out of reach.
func TestReplaceAbilityKingTimeoutSkipsLockedCaller(t *testing.T) {
	g, mb, caller, actor := lockedCallerTable(t)
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing))
	require.Equal(t, engine.PendingKingLook, g.Engine.Pending.Type)

	require.Eventually(t, func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.Engine.Pending.Type == engine.PendingNone
	}, 2*time.Second, 10*time.Millisecond, "the King has to settle for the turn to end")

	g.mu.Lock()
	defer g.mu.Unlock()
	reveal := lastPlayerEventOfType(mb, actor.ID, EventPrivateSpecialSuccess)
	require.NotNil(t, reveal, "the auto-resolved look reveals the pair it bound")
	require.Equal(t, "swap_peek_reveal", reveal.Special)
	require.NotNil(t, reveal.Card2)
	require.NotNil(t, reveal.Card2.User)
	assert.NotEqual(t, caller.ID, reveal.Card2.User.ID, "the King may not bind the locked caller")
	assert.NotEqual(t, seat, g.Engine.ActingPlayer(), "the turn moved on")
	assert.False(t, g.TurnDeadline.IsZero(), "and the next turn is armed")
}

// TestReplaceAbilityKingTimeoutLeavesATimerArmed pins the invariant the auto-resolution owes the
// table on the King path: it takes two engine actions, and a refused second one leaves the engine
// holding PendingKingDecision. Either the turn advances or a timer is armed; a pending state on a
// dead clock is the wedge this whole path exists to avoid.
func TestReplaceAbilityKingTimeoutLeavesATimerArmed(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, replaceRules(15))
	g.TurnDuration = 60 * time.Millisecond
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing))
	require.Equal(t, engine.PendingKingLook, g.Engine.Pending.Type)

	require.Eventually(t, func() bool {
		g.mu.Lock()
		defer g.mu.Unlock()
		return g.Engine.Pending.Type != engine.PendingKingLook
	}, 2*time.Second, 10*time.Millisecond, "the timeout has to move the King off its look")

	g.mu.Lock()
	defer g.mu.Unlock()
	turnAdvanced := g.Engine.ActingPlayer() != seat
	assert.True(t, turnAdvanced || !g.TurnDeadline.IsZero(),
		"after auto-resolution the turn has advanced or a timer is armed")
	if !turnAdvanced {
		assert.True(t, g.SpecialAction.Active, "an unadvanced turn still owes its owner a prompt")
	}
}

// TestKingDecisionPromptRestoresAfterRefusedDecline covers the recovery the King path falls back on
// when the engine refuses the auto-declined swap: the prompt comes back marked as the King's second
// step, so a live client and the next timeout both have an action that settles it, and the clock is
// re-armed rather than left dead under a pending engine.
func TestKingDecisionPromptRestoresAfterRefusedDecline(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, replaceRules(15))
	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]
	oppSeat := 1 - seat

	g.mu.Lock()
	g.SpecialAction = SpecialActionState{}
	g.TurnDeadline = time.Time{}
	g.restoreKingDecisionPrompt(actor.ID, seat, oppSeat, 0, 0)
	g.mu.Unlock()

	assert.True(t, g.SpecialAction.Active, "the prompt is back")
	assert.Equal(t, actor.ID, g.SpecialAction.PlayerID)
	assert.Equal(t, "K", g.SpecialAction.CardRank)
	assert.True(t, g.SpecialAction.Mandatory, "the ability is still one the engine armed")
	assert.True(t, g.SpecialAction.FirstStepDone, "and its look is done, so the decline is on offer")
	require.NotNil(t, g.SpecialAction.Card1)
	require.NotNil(t, g.SpecialAction.Card2)
	assert.Equal(t, g.CardTracker.Players[seat].HandUUIDs[0], g.SpecialAction.Card1.ID)
	assert.Equal(t, g.CardTracker.Players[oppSeat].HandUUIDs[0], g.SpecialAction.Card2.ID)
	assert.False(t, g.TurnDeadline.IsZero(), "and the clock is re-armed")

	// The restored prompt is answerable: the skip is the decline the engine models.
	g.ProcessSpecialAction(actor.ID, "skip", nil, nil)
	assert.False(t, g.SpecialAction.Active, "the restored prompt settles")
}

// TestInteractiveSwapRefusesLockedCaller verifies the server refuses a blind swap that names the
// Cambia caller's card while LockCallerHand is on. Nothing server-side used to stop one: the
// resolver never consulted the lock, the models.Player.HasCalledCambia flag the blind-swap branch
// read is never written outside tests, and the engine's N-player apply path has no guard, so a
// client that sent the swap anyway landed it on a hand RULES.md 3C freezes.
func TestInteractiveSwapRefusesLockedCaller(t *testing.T) {
	g, mb, caller, actor := lockedCallerTable(t)
	callerSeat := g.PlayerToEngine[caller.ID]
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankJack))
	require.Equal(t, engine.PendingBlindSwap, g.Engine.Pending.Type)
	lockedUUIDs, lockedCards := handSnapshot(g, callerSeat)
	mb.clear()

	ownCard := g.CardTracker.Players[seat].HandUUIDs[0]
	lockedCard := g.CardTracker.Players[callerSeat].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownCard, actor.ID, 0), cardTarget(lockedCard, caller.ID, 0))

	assert.Equal(t, 1, countPlayerEvents(mb, actor.ID, EventPrivateSpecialFail), "the swap is refused")
	gotUUIDs, gotCards := handSnapshot(g, callerSeat)
	assert.Equal(t, lockedUUIDs, gotUUIDs, "and the locked hand is untouched")
	assert.Equal(t, lockedCards, gotCards)
	assert.Equal(t, engine.PendingBlindSwap, g.Engine.Pending.Type, "the ability is still owed")
	assert.True(t, g.SpecialAction.Active, "so the prompt stands for a legal retry")

	// A legal target still resolves, so the refusal rejects and waits rather than wedging the turn.
	targetID := g.EngineToPlayer[2]
	targetCard := g.CardTracker.Players[2].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownCard, actor.ID, 0), cardTarget(targetCard, targetID, 0))
	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "a reachable seat resolves the ability")
	assert.False(t, g.SpecialAction.Active)
}

// TestInteractiveKingLookRefusesLockedCaller is the King half: the look binds the pair its decision
// may then swap, so the lock has to refuse it at the look rather than at the swap.
func TestInteractiveKingLookRefusesLockedCaller(t *testing.T) {
	g, mb, caller, actor := lockedCallerTable(t)
	callerSeat := g.PlayerToEngine[caller.ID]
	seat := g.PlayerToEngine[actor.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing))
	require.Equal(t, engine.PendingKingLook, g.Engine.Pending.Type)
	mb.clear()

	ownCard := g.CardTracker.Players[seat].HandUUIDs[0]
	lockedCard := g.CardTracker.Players[callerSeat].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownCard, actor.ID, 0), cardTarget(lockedCard, caller.ID, 0))

	assert.Equal(t, 1, countPlayerEvents(mb, actor.ID, EventPrivateSpecialFail), "the look is refused")
	assert.Nil(t, lastPlayerEventOfType(mb, actor.ID, EventPrivateSpecialSuccess), "and reveals nothing")
	assert.Equal(t, engine.PendingKingLook, g.Engine.Pending.Type, "the ability is still owed")
	assert.True(t, g.SpecialAction.Active, "so the prompt stands for a legal retry")
	assert.False(t, g.SpecialAction.FirstStepDone, "and the King's step has not moved")
}

// TestInteractivePeekReachesLockedCaller pins the other side of the rule: LockCallerHand freezes
// the caller's hand against moves, not against being looked at, and neither engine legal mask
// withholds peek-other from that seat. The server-side lock guard must not over-reach into peeks.
func TestInteractivePeekReachesLockedCaller(t *testing.T) {
	g, mb, caller, actor := lockedCallerTable(t)
	callerSeat := g.PlayerToEngine[caller.ID]

	drawThenReplace(t, g, actor, engine.NewCard(engine.SuitClubs, engine.RankTen))
	require.Equal(t, engine.PendingPeekOther, g.Engine.Pending.Type)
	mb.clear()

	lockedCard := g.CardTracker.Players[callerSeat].HandUUIDs[0]
	g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(lockedCard, caller.ID, 0), nil)

	assert.Equal(t, 0, countPlayerEvents(mb, actor.ID, EventPrivateSpecialFail), "a peek of a locked hand is legal")
	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "and resolves the ability")
	assert.False(t, g.SpecialAction.Active)
}
