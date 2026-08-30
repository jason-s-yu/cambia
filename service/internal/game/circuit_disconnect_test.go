// internal/game/circuit_disconnect_test.go
//
// The circuit disconnect path (cambia-1117 D4). A circuit round used to return out of
// HandleDisconnect on a 60s literal of its own, ahead of the grace and forfeit block: the rule
// sheet's DisconnectGraceSec never armed, nothing forfeited, and because a circuit is created with
// ForfeitOnDisconnect off (handlers.CreateGameInstance) the turn scheduler then declined to clock a
// disconnected acting player. A round whose actor dropped had neither a clock nor a forfeit and
// stopped there.
package game

import (
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

// buildDropTestGame builds a started game with a turn timer armed and a reconnect window
// configured. turnDuration and grace override what BeginPreGame derives from the rule sheet (the
// same override order buildTimedTestGame uses: BeginPreGame reads the rule sheet, the test writes
// over it, StartGame arms the first timer), so a test does not have to wait whole seconds for
// either. graceSec stays the rule-sheet figure, which is what the player_reconnecting payload
// quotes and what this ticket's 60s literal used to ignore.
func buildDropTestGame(t *testing.T, numPlayers int, circuit, forfeit bool, graceSec int, turnDuration, grace time.Duration) (*CambiaGame, []*models.Player, *mockBroadcaster) {
	t.Helper()

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	if circuit {
		g.Circuit = Circuit{Enabled: true, Mode: "circuit_" + strconv.Itoa(numPlayers) + "p"}
	}
	g.HouseRules = *testHouseRules(15, 2)
	g.HouseRules.ForfeitOnDisconnect = forfeit
	g.HouseRules.DisconnectGraceSec = graceSec

	players := make([]*models.Player, numPlayers)
	for i := range players {
		id := uuid.New()
		players[i] = &models.Player{
			ID:        id,
			Connected: true,
			Hand:      []*models.Card{},
			User:      &models.User{ID: id, Username: "P" + strconv.Itoa(i)},
		}
		g.AddPlayer(players[i])
	}

	g.BeginPreGame()
	require.True(t, g.PreGameActive, "the pregame reveal should be running")
	g.TurnDuration = turnDuration
	g.DisconnectGrace = grace
	g.StartGame()
	require.True(t, g.Started, "the game should be started")

	t.Cleanup(func() { g.EndGame() }) // stops whatever timer is still armed

	mb.clear()
	return g, players, mb
}

// playSimpleTurn takes one plain turn for player: draw from the stockpile, discard the drawn card,
// and decline any ability that discard triggers. Used to hand the turn to the next seat without
// waiting on the turn timer.
func playSimpleTurn(t *testing.T, g *CambiaGame, player *models.Player) {
	t.Helper()

	g.HandlePlayerAction(player.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	engineIdx := g.PlayerToEngine[player.ID]
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "the draw should have produced a card")

	g.HandlePlayerAction(player.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	})
	if g.SpecialAction.Active && g.SpecialAction.PlayerID == player.ID {
		g.ProcessSpecialAction(player.ID, "skip", nil, nil)
	}
}

// TestCircuitDropHoldsTheSeatForTheRuleSheetWindow pins the grace a circuit drop opens to
// HouseRules.DisconnectGraceSec. The circuit branch used to arm a 60s AfterFunc of its own and
// return, so the window every other game reads off the rule sheet was neither armed nor advertised:
// no player_reconnecting event reached the table and nothing published a reconnect deadline.
func TestCircuitDropHoldsTheSeatForTheRuleSheetWindow(t *testing.T) {
	g, players, mb := buildDropTestGame(t, 2, true, false, 5, 30*time.Second, 5*time.Second)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	observer := otherPlayer(t, players, acting.ID)

	before := time.Now()
	g.HandleDisconnect(acting.ID)

	ev := mb.findEventByType(EventPlayerReconnecting)
	require.NotNil(t, ev, "a circuit drop must advertise the reconnect window it opened")
	require.NotNil(t, ev.User)
	assert.Equal(t, acting.ID, ev.User.ID)
	assert.Equal(t, 5, ev.Payload["graceSeconds"],
		"the window must be the rule sheet's DisconnectGraceSec, not the 60s literal the circuit branch carried")

	deadlineMs, ok := ev.Payload["deadline"].(int64)
	require.True(t, ok, "the reconnect window must carry an absolute deadline")
	deadline := time.UnixMilli(deadlineMs)
	assert.WithinDuration(t, before.Add(5*time.Second), deadline, time.Second,
		"the advertised deadline must close 5s out, the configured window")

	state := g.GetCurrentObfuscatedGameState(observer.ID)
	var dropped *ObfPlayerState
	for i := range state.Players {
		if state.Players[i].PlayerID == acting.ID {
			dropped = &state.Players[i]
		}
	}
	require.NotNil(t, dropped)
	assert.False(t, dropped.Connected, "the dropped player must be marked disconnected")
	assert.False(t, dropped.Forfeited, "a circuit drop must not forfeit the seat")
	require.NotNil(t, dropped.ReconnectDeadline, "a client resyncing mid-window must see the countdown")
}

// TestCircuitDropOfTheActingPlayerLeavesTheTurnClocked is the acceptance case: the player on turn
// drops, and inside their reconnect window the round still has a clock running on that turn. The
// clock is deliberately not restarted by the drop (memory cambia-991: a disconnect neither pauses
// nor extends the turn timer), so the deadline the table was already counting down to stands.
func TestCircuitDropOfTheActingPlayerLeavesTheTurnClocked(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, true, false, 5, 30*time.Second, 5*time.Second)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	observer := otherPlayer(t, players, acting.ID)

	armed := g.GetCurrentObfuscatedGameState(observer.ID)
	require.NotNil(t, armed.TurnDeadline, "the turn should be clocked before the drop")

	g.HandleDisconnect(acting.ID)

	state := g.GetCurrentObfuscatedGameState(observer.ID)
	assert.False(t, state.GameOver, "one drop of two must not end a circuit round")
	assert.Equal(t, acting.ID, state.CurrentPlayerID, "the drop must not skip the dropped player's turn")
	require.NotNil(t, state.TurnDeadline, "the turn must still be clocked inside the grace window")
	assert.Equal(t, *armed.TurnDeadline, *state.TurnDeadline, "the drop must neither pause nor restart the turn clock")
	assert.Greater(t, *state.TurnDeadline, time.Now().UnixMilli(), "the armed deadline must still be ahead of us")
}

// TestCircuitTurnPassingToADroppedPlayerIsStillClocked covers the stall itself. The drop is not
// what left the round stuck: the turn already had a clock when the actor dropped. It stalled at the
// next turn boundary, where the scheduler declined to arm anything for a disconnected acting player
// whose game had ForfeitOnDisconnect off, which every circuit round does.
func TestCircuitTurnPassingToADroppedPlayerIsStillClocked(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, true, false, 5, 30*time.Second, 5*time.Second)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	victim := otherPlayer(t, players, acting.ID)

	g.HandleDisconnect(victim.ID)
	playSimpleTurn(t, g, acting)

	state := g.GetCurrentObfuscatedGameState(acting.ID)
	require.False(t, state.GameOver, "the round must still be live")
	require.Equal(t, victim.ID, state.CurrentPlayerID, "the turn must have passed to the dropped player")
	require.NotNil(t, state.TurnDeadline,
		"a turn belonging to a disconnected player must still be clocked, or nothing ever ends it")
	assert.Greater(t, *state.TurnDeadline, time.Now().UnixMilli(), "the new turn's deadline must be ahead of us")
}

// TestDropWithoutForfeitOrCircuitIsStillClocked is the same stall outside circuit mode: a private
// lobby that turned the forfeit rule off got the same unclocked turn, and with no forfeit to end it
// either the table waited on a player who may never come back.
func TestDropWithoutForfeitOrCircuitIsStillClocked(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, false, false, 0, 30*time.Second, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	victim := otherPlayer(t, players, acting.ID)

	g.HandleDisconnect(victim.ID)
	playSimpleTurn(t, g, acting)

	state := g.GetCurrentObfuscatedGameState(acting.ID)
	require.False(t, state.GameOver, "the game must still be live with the forfeit rule off")
	require.Equal(t, victim.ID, state.CurrentPlayerID, "the turn must have passed to the dropped player")
	require.NotNil(t, state.TurnDeadline, "the turn must be clocked whatever the forfeit rule says")
}

// TestDropOfTheActingPlayerDoesNotRestartTheClock pins the other half of the same rule outside
// circuit mode. With nothing holding the seat, HandleDisconnect asks for the abandoned turn to be
// clocked - but only where it is not clocked already, since a drop that re-armed the timer would
// buy the player a fresh turn's worth of thinking time (memory cambia-991).
func TestDropOfTheActingPlayerDoesNotRestartTheClock(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, false, false, 0, 30*time.Second, 0)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	observer := otherPlayer(t, players, acting.ID)

	armed := g.GetCurrentObfuscatedGameState(observer.ID)
	require.NotNil(t, armed.TurnDeadline)

	g.HandleDisconnect(acting.ID)

	state := g.GetCurrentObfuscatedGameState(observer.ID)
	assert.Equal(t, acting.ID, state.CurrentPlayerID, "the drop must not skip the dropped player's turn")
	require.NotNil(t, state.TurnDeadline, "the abandoned turn must still be clocked")
	assert.Equal(t, *armed.TurnDeadline, *state.TurnDeadline, "the drop must not restart the turn clock")
}

// TestCircuitGraceExpiryTakesTheSeatWithoutForfeiting is what the closed window costs a circuit
// player: the seat is marked unattended and keeps playing on the turn clock, and the player stays in
// the round to be scored (RULES.md T5). A circuit is created with ForfeitOnDisconnect off, so there
// is no forfeit to fall back on and an unclocked seat would have been the end of the round.
func TestCircuitGraceExpiryTakesTheSeatWithoutForfeiting(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, true, false, 1, 30*time.Second, 40*time.Millisecond)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	victim := otherPlayer(t, players, acting.ID)

	g.HandleDisconnect(victim.ID)
	require.False(t, g.IsCircuitAIControlled(victim.ID), "the seat is only taken over once the window closes")

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && !g.IsCircuitAIControlled(victim.ID) {
		time.Sleep(5 * time.Millisecond)
	}
	require.True(t, g.IsCircuitAIControlled(victim.ID), "the closed window must hand the seat over")

	state := g.GetCurrentObfuscatedGameState(acting.ID)
	assert.False(t, state.GameOver, "an unattended circuit seat must not end the round")
	for _, ps := range state.Players {
		if ps.PlayerID == victim.ID {
			assert.False(t, ps.Forfeited, "a circuit seat is played on, not forfeited")
			assert.Nil(t, ps.ReconnectDeadline, "the closed window must stop being advertised")
		}
	}
	require.NotNil(t, state.TurnDeadline, "the round must still be clocked once the seat is unattended")
}

// TestCircuitReconnectInsideTheWindowHandsTheSeatBack is the return leg: a player who comes back
// before the window closes keeps their seat, and one who comes back after it closed takes it back
// from the takeover marker.
func TestCircuitReconnectInsideTheWindowHandsTheSeatBack(t *testing.T) {
	g, players, mb := buildDropTestGame(t, 2, true, false, 5, 30*time.Second, 5*time.Second)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	victim := otherPlayer(t, players, acting.ID)

	g.HandleDisconnect(victim.ID)
	g.HandleReconnect(victim.ID, nil)

	assert.False(t, g.IsCircuitAIControlled(victim.ID), "a player back inside the window keeps their seat")
	require.NotNil(t, mb.findEventByType(EventPlayerReconnected), "the table must be told the seat is back")

	state := g.GetCurrentObfuscatedGameState(acting.ID)
	for _, ps := range state.Players {
		if ps.PlayerID == victim.ID {
			assert.True(t, ps.Connected, "the returning player must be connected again")
			assert.Nil(t, ps.ReconnectDeadline, "the reconnect must close the window")
		}
	}

	// The window that already closed: the seat comes back off the takeover marker too.
	g.HandleDisconnect(victim.ID)
	g.mu.Lock()
	g.circuitAIControlled[victim.ID] = true
	g.mu.Unlock()
	g.HandleReconnect(victim.ID, nil)
	assert.False(t, g.IsCircuitAIControlled(victim.ID), "a late return must take the seat back from the takeover")
}

// TestCircuitDropUnderTheForfeitRuleStillForfeits keeps the two rules from cancelling each other.
// Nothing in production turns the forfeit rule on for a circuit today (handlers.CreateGameInstance
// forces it off), but the rule sheet can carry it, and the circuit branch used to swallow the drop
// before the forfeit block ever saw it.
func TestCircuitDropUnderTheForfeitRuleStillForfeits(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 3, true, true, 1, 30*time.Second, 40*time.Millisecond)

	acting := currentTurnPlayer(g)
	require.NotNil(t, acting)
	victim := otherPlayer(t, players, acting.ID)

	g.HandleDisconnect(victim.ID)

	forfeited := func() bool {
		for _, ps := range g.GetCurrentObfuscatedGameState(acting.ID).Players {
			if ps.PlayerID == victim.ID {
				return ps.Forfeited
			}
		}
		return false
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && !forfeited() {
		time.Sleep(5 * time.Millisecond)
	}
	assert.True(t, forfeited(), "the forfeit rule must still land on a circuit drop that never returns")
	assert.False(t, g.IsCircuitAIControlled(victim.ID), "a forfeited seat is out of the round, not taken over")
}
