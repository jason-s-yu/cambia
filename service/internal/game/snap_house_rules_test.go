// internal/game/snap_house_rules_test.go
// handleSnapViaEngine resolves snaps outside the engine's own snap phase (see the comments on
// handleSnapViaEngine and beginSnapFill), so it never went through engine.ApplyAction's legal-action
// gating that AllowOpponentSnapping and LockCallerHand rely on elsewhere. Only the client checked
// AllowOpponentSnapping before sending action_snap, and nothing checked LockCallerHand at all, so a
// crafted frame could snap an opponent's card with the rule off, snap the Cambia caller's locked
// hand, or have the caller themselves snap after locking (cambia-1043).
package game

import (
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// otherPlayer returns the first player in players whose ID is not exclude.
func otherPlayer(t *testing.T, players []*models.Player, exclude uuid.UUID) *models.Player {
	t.Helper()
	for _, p := range players {
		if p.ID != exclude {
			return p
		}
	}
	t.Fatalf("no other player found besides %s", exclude)
	return nil
}

// TestOpponentSnapRefusedWhenRuleOff covers hole 1: AllowOpponentSnapping off refuses an
// opponent-target snap outright. This mirrors engine/snap.go snapOpponent and
// nplayer_actions.go nplayerSnapOpponent, both of which return a plain error - before touching
// any hand or drawing a penalty - when the rule is off, because the action was never legal to
// attempt rather than a legal attempt that turned out wrong (RULES.md 5's penalty covers the
// latter, a mismatched card or nothing left to pay with, not the former).
func TestOpponentSnapRefusedWhenRuleOff(t *testing.T) {
	hr := testHouseRules(0, 2)
	hr.AllowOpponentSnapping = false
	g, players, mb := setupTestGame(t, 2, hr)
	victim, snapper := players[0], players[1]
	victimIdx := g.PlayerToEngine[victim.ID]
	snapperIdx := g.PlayerToEngine[snapper.ID]

	cardUUID, _ := plantSnapPair(t, g, victim.ID)
	victimBefore := int(g.Engine.Players[victimIdx].HandLen)
	snapperBefore := int(g.Engine.Players[snapperIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, victimBefore, int(g.Engine.Players[victimIdx].HandLen), "the victim keeps the card: opponent snapping is off")
	assert.Equal(t, snapperBefore, int(g.Engine.Players[snapperIdx].HandLen), "the attempt is refused outright, no penalty drawn")
	assert.False(t, g.owesSnapFill(snapper.ID), "a refused attempt owes no fill")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail), "the attempt is reported as a failed snap")
}

// TestLockedCallerHandNotSnappable covers hole 2's first direction: once LockCallerHand has
// frozen the caller's hand (RULES.md 3C, "cannot be altered by any player, including yourself"),
// nobody else can snap a card out of it either, and the attempt draws no penalty (the target was
// never a legal one to name, matching engine/snap.go initiateSnapPhase, which never offers the
// caller's cards as a snap target once they have called).
func TestLockedCallerHandNotSnappable(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2)) // LockCallerHand defaults true.
	caller := currentTurnPlayer(g)
	other := otherPlayer(t, players, caller.ID)

	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	cardUUID, _ := plantSnapPair(t, g, caller.ID)
	callerIdx := g.PlayerToEngine[caller.ID]
	otherIdx := g.PlayerToEngine[other.ID]
	callerBefore := int(g.Engine.Players[callerIdx].HandLen)
	otherBefore := int(g.Engine.Players[otherIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(other.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, callerBefore, int(g.Engine.Players[callerIdx].HandLen), "the locked caller's hand must not be snappable")
	assert.Equal(t, otherBefore, int(g.Engine.Players[otherIdx].HandLen), "the refused attempt draws no penalty")
	assert.False(t, g.owesSnapFill(other.ID), "a refused attempt owes no fill")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail))
}

// TestLockedCallerCannotSnapOwnCard covers hole 2's second direction: the caller cannot snap
// their own card once locked, matching the same RULES.md 3C text ("including yourself").
func TestLockedCallerCannotSnapOwnCard(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	caller := currentTurnPlayer(g)

	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	cardUUID, _ := plantSnapPair(t, g, caller.ID)
	callerIdx := g.PlayerToEngine[caller.ID]
	before := int(g.Engine.Players[callerIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(caller.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, before, int(g.Engine.Players[callerIdx].HandLen), "the locked caller cannot snap their own card")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail))
}

// TestLockedCallerCannotSnapOpponent completes hole 2's second direction: the caller cannot snap
// an opponent's card either, once locked.
func TestLockedCallerCannotSnapOpponent(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	caller := currentTurnPlayer(g)
	other := otherPlayer(t, players, caller.ID)

	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())

	cardUUID, _ := plantSnapPair(t, g, other.ID)
	otherIdx := g.PlayerToEngine[other.ID]
	callerIdx := g.PlayerToEngine[caller.ID]
	otherBefore := int(g.Engine.Players[otherIdx].HandLen)
	callerBefore := int(g.Engine.Players[callerIdx].HandLen)

	mb.clear()
	g.HandlePlayerAction(caller.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})

	assert.Equal(t, otherBefore, int(g.Engine.Players[otherIdx].HandLen), "the locked caller cannot snap an opponent's card either")
	assert.Equal(t, callerBefore, int(g.Engine.Players[callerIdx].HandLen), "the refused attempt draws no penalty")
	assert.False(t, g.owesSnapFill(caller.ID), "a refused attempt owes no fill")
	assert.NotNil(t, mb.findEventByType(EventPlayerSnapFail))
}

// TestSnapFillDropsWhenVictimHandLocksBeforeSettling covers the race RULES.md does not describe:
// a fill obligation opens against a hand that is not yet locked (the victim has not called
// Cambia at the moment they are snapped), but the victim calls Cambia on their own turn before
// the snapper settles what they owe. HandlePlayerAction only blocks the snapper's actions while a
// fill is outstanding (game.go, owesSnapFill), not the victim's, so the victim's Cambia call goes
// through and locks a hand a fill still targets. RULES.md is silent on this interaction; this
// takes the conservative reading applySnapFill already uses for a hand that is simply full: the
// fill lapses rather than writing into a hand LockCallerHand has frozen, and the snapper keeps the
// card they would have given up.
func TestSnapFillDropsWhenVictimHandLocksBeforeSettling(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))
	// The victim must still be the acting player after the snap (snaps do not advance the turn) so
	// they can legitimately call Cambia next.
	victim := currentTurnPlayer(g)
	snapper := otherPlayer(t, players, victim.ID)
	victimIdx := g.PlayerToEngine[victim.ID]
	snapperIdx := g.PlayerToEngine[snapper.ID]

	cardUUID, _ := plantSnapPair(t, g, victim.ID)
	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": cardUUID.String()},
	})
	require.True(t, g.owesSnapFill(snapper.ID), "the opponent snap should have opened a fill obligation")
	require.NotNil(t, mb.findEventByType(EventPlayerSnapSuccess))

	// The victim calls Cambia on their own turn while the fill is still outstanding. Nothing in
	// HandlePlayerAction blocks this: only the snapper owes a fill.
	g.HandlePlayerAction(victim.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.Engine.IsCambiaCalled())
	require.Equal(t, uint8(g.Engine.CambiaCaller), victimIdx, "the victim is now the Cambia caller")
	require.True(t, g.handLocked(victimIdx), "the victim's hand should now be locked")
	require.True(t, g.owesSnapFill(snapper.ID), "calling Cambia does not itself settle the snapper's debt")

	victimBefore := int(g.Engine.Players[victimIdx].HandLen)
	snapperBefore := int(g.Engine.Players[snapperIdx].HandLen)
	givenUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[0]

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap_move",
		Payload:    map[string]interface{}{"id": givenUUID.String(), "idx": float64(0)},
	})

	assert.False(t, g.owesSnapFill(snapper.ID), "the obligation clears even though it could not be paid into a locked hand")
	assert.Equal(t, victimBefore, int(g.Engine.Players[victimIdx].HandLen), "the locked hand must not receive the fill card")
	assert.Equal(t, snapperBefore, int(g.Engine.Players[snapperIdx].HandLen), "the snapper keeps the card they would have given up")
	assert.Nil(t, mb.findEventByType(EventPlayerSnapMove), "no move happened, so no move event fires")
}
