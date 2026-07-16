// internal/game/cambia_call_validation_test.go
//
// Regression tests for cambia-507: a player who already drew this turn and then calls Cambia
// was rejected with a misleading/generic message instead of an accurate one explaining that
// Cambia must be called before drawing, at turn start (RULES.md). These tests pin the corrected
// message for the already-drew case and confirm the genuinely out-of-turn case still reports
// "It's not your turn."
package game

import (
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCambiaCallAfterDrawRejectedWithAccurateMessage covers the already-drew-this-turn case: the
// acting player draws, then calls Cambia before discarding. It IS their turn, so the rejection
// must explain the actual rule violation rather than reporting a turn-ownership problem.
func TestCambiaCallAfterDrawRejectedWithAccurateMessage(t *testing.T) {
	g, _, mb := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	cur := currentTurnPlayer(g)
	curIdx := g.PlayerToEngine[cur.ID]

	g.HandlePlayerAction(cur.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, curIdx, g.Engine.Pending.PlayerID, "draw should register a pending discard for the current player")

	mb.clear()
	g.HandlePlayerAction(cur.ID, models.GameAction{ActionType: "action_cambia"})

	rej := mb.getLastPlayerEvent(cur.ID)
	require.NotNil(t, rej, "the post-draw Cambia call should get a private failure event")
	assert.Equal(t, EventPrivateSpecialFail, rej.Type)
	assert.Equal(t, "Cambia must be called before drawing.", rej.Payload["message"])

	// The engine must not have registered a Cambia call, and the pending draw must remain intact.
	assert.False(t, g.Engine.IsCambiaCalled(), "the rejected Cambia call must not take effect")
	assert.Equal(t, curIdx, g.Engine.Pending.PlayerID, "the pending discard must remain intact")
}

// TestCambiaCallOutOfTurnStillRejectedAsNotYourTurn covers the genuinely out-of-turn case: a
// player who is not the acting player calls Cambia. This must still be rejected as a turn-
// ownership problem, unchanged by the cambia-507 fix.
func TestCambiaCallOutOfTurnStillRejectedAsNotYourTurn(t *testing.T) {
	g, ids, mb := buildTimedTestGame(t, 30*time.Second)
	defer stopGameTimer(g)

	cur := currentTurnPlayer(g)
	oppID := opponentOf(ids, cur.ID)

	mb.clear()
	g.HandlePlayerAction(oppID, models.GameAction{ActionType: "action_cambia"})

	rej := mb.getLastPlayerEvent(oppID)
	require.NotNil(t, rej, "the out-of-turn Cambia call should get a private failure event")
	assert.Equal(t, EventPrivateSpecialFail, rej.Type)
	assert.Equal(t, "It's not your turn.", rej.Payload["message"])

	assert.False(t, g.Engine.IsCambiaCalled(), "the rejected Cambia call must not take effect")
}
