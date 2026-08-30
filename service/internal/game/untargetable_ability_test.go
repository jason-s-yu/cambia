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
	g, _, _ := setupTestGame(t, 4, hr)

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
}

// TestArmedAbilityWithALegalTargetStillRearms is the other half: a targetable ability whose chosen
// action the engine refused is not the stranded case, so the prompt and the clock stand rather than
// the ability being discharged out from under a player who can still play it.
func TestArmedAbilityWithALegalTargetStillRearms(t *testing.T) {
	hr := replaceRules(0)
	hr.LockCallerHand = true
	g, _, _ := setupTestGame(t, 4, hr)

	g.Engine.CurrentPlayer = 0
	caller := currentTurnPlayer(g)
	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	actor := currentTurnPlayer(g)
	seat := g.PlayerToEngine[actor.ID]

	g.mu.Lock()
	defer g.mu.Unlock()

	// Seats 2 and 3 still hold cards, so the swap has a target the mask accepts.
	g.Engine.Pending.Type = engine.PendingBlindSwap
	g.Engine.Pending.PlayerID = seat
	require.NotEmpty(t, g.Engine.NPlayerLegalActionsList(), "premise: the ability is resolvable")

	if g.Engine.ResolveUntargetableArmedAbility(g.isNPlayerTable()) {
		t.Fatal("a resolvable ability was discharged")
	}
	assert.Equal(t, engine.PendingBlindSwap, g.Engine.Pending.Type, "the ability stands")
}
