package game

import (
	"testing"

	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestArmedAbilityWithNoLegalTargetResolvesInsteadOfRearming covers the escape from an armed
// ability no action can resolve. autoResolveArmedAbility's !resolvable branch used to log and call
// scheduleNextTurnTimer for the same player, and nothing could change the condition: the engine
// refuses every action while it holds a pending ability, including the fallback draw the rest of
// the timeout path would take, so the same timeout fired against the same state forever. The
// cambia-1173 ruling bars a decline action, so the escape is the engine discharging the ability
// (cambia-1171).
//
// The engine no longer arms this state - discardWithAbility and discardWithAbilityNPlayer both
// fizzle an ability with an empty legal set - so the state is built directly here, which is also
// the shape a game carried over from before that fix holds. The loop was never reproduced; this
// pins the guard.
func TestArmedAbilityWithNoLegalTargetResolvesInsteadOfRearming(t *testing.T) {
	hr := replaceRules(0) // no turn timer, so nothing races the manual call below
	hr.LockCallerHand = true
	g, _, mb := setupTestGame(t, 4, hr)

	g.Engine.CurrentPlayer = 0
	caller := currentTurnPlayer(g)
	require.Equal(t, uint8(0), g.PlayerToEngine[caller.ID], "seat 0 calls Cambia")
	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]
	require.Equal(t, uint8(1), seat, "the turn moved to seat 1")

	g.mu.Lock()
	defer g.mu.Unlock()

	// Empty the only seats a blind swap could still reach, leaving the locked caller as the sole
	// opponent holding cards. LockCallerHand freezes that hand against swaps (RULES.md 3C), so no
	// action in either space can resolve the ability.
	for _, empty := range []uint8{2, 3} {
		g.Engine.Players[empty].HandLen = 0
	}
	g.Engine.Pending.Type = engine.PendingBlindSwap
	g.Engine.Pending.PlayerID = seat
	g.SpecialAction = SpecialActionState{
		Active:    true,
		PlayerID:  actor.ID,
		CardRank:  "J",
		Mandatory: true,
	}
	require.Empty(t, g.Engine.NPlayerLegalActionsList(),
		"premise: no action in this table's space can resolve the armed swap")

	g.autoResolveArmedAbility(actor.ID)

	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "the ability is discharged, not left armed")
	assert.False(t, g.SpecialAction.Active, "and the prompt does not outlive it")
	assert.False(t, g.Engine.Snap.Active, "the snap phase the resolution opened is played out")
	assert.True(t, g.Engine.ActingPlayer() != seat || g.Engine.IsTerminal(),
		"the turn moves off the seat that could not act")

	// The frame that tells the player why carries the special enum the protocol defines
	// (game_actions.md), not the card rank that armed the ability: this emitter passed the rank
	// straight through onto the wire and into the game_actions row (cambia-1239).
	var fail *GameEvent
	for i, ev := range mb.playerEvents[actor.ID] {
		if ev.Type == EventPrivateSpecialFail {
			fail = &mb.playerEvents[actor.ID][i]
		}
	}
	require.NotNil(t, fail, "the discharged player is told why")
	assert.Equal(t, "swap_blind", fail.Special, "the Jack's ability is swap_blind on the wire")
}

// TestArmedAbilityWithALegalTargetIsPlayedNotDischarged is the other half: an ability the mask can
// still resolve is not the stranded case, so the timeout plays it against a legal target instead of
// discharging it out from under a player who could have played it. Same table as above with the
// swap's targets left in place, driven through the same entry point, so the two tests differ only
// in whether a target survives.
//
// It went by TestArmedAbilityWithALegalTargetStillRearms and never called autoResolveArmedAbility,
// asserting the engine predicate the test above already covers; the adapter branch that name points
// at is a guard against the adapter and the engine disagreeing about one action, which no state
// reachable from play produces (cambia-1239).
func TestArmedAbilityWithALegalTargetIsPlayedNotDischarged(t *testing.T) {
	hr := replaceRules(0)
	hr.LockCallerHand = true
	g, _, _ := setupTestGame(t, 4, hr)

	g.Engine.CurrentPlayer = 0
	caller := currentTurnPlayer(g)
	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]
	require.Equal(t, uint8(1), seat, "the turn moved to seat 1")

	g.mu.Lock()
	defer g.mu.Unlock()

	// Seats 2 and 3 still hold cards, so the swap has a target the mask accepts, and the lock
	// keeps seat 0 out of reach: the auto-resolution must take the first unlocked one.
	g.Engine.Pending.Type = engine.PendingBlindSwap
	g.Engine.Pending.PlayerID = seat
	g.SpecialAction = SpecialActionState{
		Active:    true,
		PlayerID:  actor.ID,
		CardRank:  "J",
		Mandatory: true,
	}
	require.NotEmpty(t, g.Engine.NPlayerLegalActionsList(), "premise: the ability is resolvable")

	ownBefore := g.Engine.Players[seat].Hand[0]
	targetBefore := g.Engine.Players[2].Hand[0]
	callerBefore := g.Engine.Players[0].Hand

	g.autoResolveArmedAbility(actor.ID)

	assert.Equal(t, targetBefore, g.Engine.Players[seat].Hand[0], "the swap was played against seat 2")
	assert.Equal(t, ownBefore, g.Engine.Players[2].Hand[0], "and seat 2 got the actor's card")
	assert.Equal(t, callerBefore, g.Engine.Players[0].Hand, "the locked caller's hand is not a target")
	assert.Equal(t, engine.PendingNone, g.Engine.Pending.Type, "the ability was resolved, not left armed")
	assert.False(t, g.SpecialAction.Active, "and the prompt does not outlive it")
}
