// internal/game/special_targeting_test.go
//
// Tests for opponent-targeting special abilities and the fail-path wedge (cambia-509).
//
//   - Opponent-facing abilities (9/T peek_other, J/Q swap_blind, K swap_peek) must succeed
//     end-to-end when the client targets a real opponent-card UUID sourced from the obfuscated
//     sync state.
//   - An invalid/garbage card target must reject-and-wait: fire the private fail event, retain the
//     pending ability + buffered discard, and never advance the turn, so the same player can retry
//     and the game never wedges (the King failure previously left the engine's pending discard
//     divergent while advanceTurn rebroadcast a turn to the same player).
//   - The obfuscated state must expose opponent hand slots as id references with Known:false and no
//     face fields for every viewer.
package game

import (
	"fmt"
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// oppOf returns the opponent player, engine index, and UUID for a 2-player game.
func oppOf(g *CambiaGame, actorID uuid.UUID) (uuid.UUID, uint8) {
	actorIdx := g.PlayerToEngine[actorID]
	oppIdx := uint8(1 - int(actorIdx))
	return g.EngineToPlayer[oppIdx], oppIdx
}

// drawDiscardAbility force-draws card for actor and discards it, activating the buffered ability.
func drawDiscardAbility(t *testing.T, g *CambiaGame, actor *models.Player, card engine.Card, wantRank string) {
	t.Helper()
	forceStockTop(g, card)
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	engineIdx := g.PlayerToEngine[actor.ID]
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "draw should set a drawn card UUID")
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": drawnUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "ability discard should activate a special action")
	require.Equal(t, wantRank, g.SpecialAction.CardRank)
	require.True(t, g.pendingDiscardAbilityChoice, "buffered discard should be pending after ability discard")
}

// actorFailEvents returns every private fail event captured for playerID.
func actorFailEvents(mb *mockBroadcaster, playerID uuid.UUID) []GameEvent {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	var out []GameEvent
	for _, ev := range mb.playerEvents[playerID] {
		if ev.Type == EventPrivateSpecialFail {
			out = append(out, ev)
		}
	}
	return out
}

// placeholderTarget reproduces the pre-fix client fabrication (`${playerId}-card-${idx}`) that the
// server can never uuid.Parse, so it exercises the reject-and-wait path.
func placeholderTarget(ownerID uuid.UUID, idx int) map[string]interface{} {
	return map[string]interface{}{
		"id":   fmt.Sprintf("%s-card-%d", ownerID.String(), idx),
		"idx":  float64(idx),
		"user": map[string]interface{}{"id": ownerID.String()},
	}
}

// TestPeekOtherSucceedsWithRealOpponentUUID verifies 9/T peek_other resolves end-to-end when the
// target is a real opponent-card UUID.
func TestPeekOtherSucceedsWithRealOpponentUUID(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	oppID, oppIdx := oppOf(g, actor.ID)
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankNine), "9")
	startActing := g.Engine.ActingPlayer()

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(oppSlot0, oppID, 0), nil)

	assert.Empty(t, actorFailEvents(mb, actor.ID), "a valid peek_other target must not fail")
	assert.False(t, g.SpecialAction.Active, "special action should be cleared after peek_other")
	assert.False(t, g.pendingDiscardAbilityChoice, "buffered discard should be resolved after peek_other")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "peek_other should advance the turn")
}

// TestSwapBlindSucceedsWithRealOpponentUUID verifies J/Q swap_blind resolves end-to-end with a real
// opponent-card UUID.
func TestSwapBlindSucceedsWithRealOpponentUUID(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppID, oppIdx := oppOf(g, actor.ID)
	ownSlot0 := g.CardTracker.Players[engineIdx].HandUUIDs[0]
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankJack), "J")
	startActing := g.Engine.ActingPlayer()

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownSlot0, actor.ID, 0),
		cardTarget(oppSlot0, oppID, 0),
	)

	assert.Empty(t, actorFailEvents(mb, actor.ID), "a valid swap_blind must not fail")
	assert.False(t, g.SpecialAction.Active, "special action should be cleared after swap_blind")
	assert.False(t, g.pendingDiscardAbilityChoice, "buffered discard should be resolved after swap_blind")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "swap_blind should advance the turn")
	// The opponent's slot-0 card should now sit in the actor's slot 0.
	assert.Equal(t, oppSlot0, g.CardTracker.Players[engineIdx].HandUUIDs[0], "swapped-in opponent card should occupy actor slot 0")
}

// TestKingSwapPeekSucceedsWithRealOpponentUUID verifies the K look-then-swap flow resolves
// end-to-end with a real opponent-card UUID across both steps.
func TestKingSwapPeekSucceedsWithRealOpponentUUID(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppID, oppIdx := oppOf(g, actor.ID)
	ownSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing), "K")
	startActing := g.Engine.ActingPlayer()

	// Step 1: look. Turn must not advance; the second step is still pending.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownSlot2, actor.ID, 2),
		cardTarget(oppSlot0, oppID, 0),
	)
	assert.Empty(t, actorFailEvents(mb, actor.ID), "a valid King look must not fail")
	require.True(t, g.SpecialAction.Active, "King look should keep the special action active")
	require.True(t, g.SpecialAction.FirstStepDone, "King look should mark the first step done")
	assert.False(t, g.pendingDiscardAbilityChoice, "buffered discard should be resolved after the King look")
	assert.Equal(t, startActing, g.Engine.ActingPlayer(), "King look must not advance the turn")

	// Step 2: confirm swap. Now the turn advances and the opponent card moves in.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_peek_swap", nil, nil)
	assert.Empty(t, actorFailEvents(mb, actor.ID), "the King swap confirmation must not fail")
	assert.False(t, g.SpecialAction.Active, "special action should be cleared after the King swap")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "the King swap should advance the turn")
	assert.Equal(t, oppSlot0, g.CardTracker.Players[engineIdx].HandUUIDs[2], "swapped-in opponent card should occupy actor slot 2")
}

// assertRejectAndWait checks the reject-and-wait invariant after an invalid special payload: the
// same actor still owns an active, non-advanced turn with the buffered discard intact.
func assertRejectAndWait(t *testing.T, g *CambiaGame, mb *mockBroadcaster, actor *models.Player, startActing uint8, wantFirstStepDone bool) {
	t.Helper()
	assert.NotEmpty(t, actorFailEvents(mb, actor.ID), "an invalid target must fire a private fail event")
	require.True(t, g.SpecialAction.Active, "the pending ability must survive an invalid target")
	assert.Equal(t, actor.ID, g.SpecialAction.PlayerID, "the ability must still belong to the same player")
	assert.Equal(t, wantFirstStepDone, g.SpecialAction.FirstStepDone, "an invalid target must not change the King step")
	assert.True(t, g.pendingDiscardAbilityChoice, "the buffered discard must stay pending after a reject")
	assert.Equal(t, startActing, g.Engine.ActingPlayer(), "an invalid target must not advance the turn")
}

// TestPeekOtherInvalidTargetRejectsAndRetries verifies a garbage peek_other target rejects without
// wedging and the same player can retry successfully.
func TestPeekOtherInvalidTargetRejectsAndRetries(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	oppID, oppIdx := oppOf(g, actor.ID)
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankNine), "9")
	startActing := g.Engine.ActingPlayer()

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "peek_other", placeholderTarget(oppID, 0), nil)
	assertRejectAndWait(t, g, mb, actor, startActing, false)

	// Retry with the real UUID: it must now succeed and advance the turn.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "peek_other", cardTarget(oppSlot0, oppID, 0), nil)
	assert.Empty(t, actorFailEvents(mb, actor.ID), "the retry with a real UUID must succeed")
	assert.False(t, g.SpecialAction.Active, "the retry should resolve the ability")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "the successful retry should advance the turn")
}

// TestSwapBlindInvalidTargetRejectsAndRetries verifies a garbage swap_blind target rejects without
// wedging and the same player can retry successfully.
func TestSwapBlindInvalidTargetRejectsAndRetries(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppID, oppIdx := oppOf(g, actor.ID)
	ownSlot0 := g.CardTracker.Players[engineIdx].HandUUIDs[0]
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankJack), "J")
	startActing := g.Engine.ActingPlayer()

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownSlot0, actor.ID, 0),
		placeholderTarget(oppID, 0),
	)
	assertRejectAndWait(t, g, mb, actor, startActing, false)

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownSlot0, actor.ID, 0),
		cardTarget(oppSlot0, oppID, 0),
	)
	assert.Empty(t, actorFailEvents(mb, actor.ID), "the retry with a real UUID must succeed")
	assert.False(t, g.SpecialAction.Active, "the retry should resolve the ability")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "the successful retry should advance the turn")
}

// TestKingInvalidTargetRejectsAndRetries verifies a garbage King-look target rejects without
// wedging (the original permanent-wedge case) and the same player can retry through both steps.
func TestKingInvalidTargetRejectsAndRetries(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppID, oppIdx := oppOf(g, actor.ID)
	ownSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	oppSlot0 := g.CardTracker.Players[oppIdx].HandUUIDs[0]

	drawDiscardAbility(t, g, actor, engine.NewCard(engine.SuitSpades, engine.RankKing), "K")
	startActing := g.Engine.ActingPlayer()

	// Garbage opponent target on the look step: the classic wedge trigger.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownSlot2, actor.ID, 2),
		placeholderTarget(oppID, 0),
	)
	assertRejectAndWait(t, g, mb, actor, startActing, false)

	// Retry the look with real UUIDs: first step should complete without advancing.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownSlot2, actor.ID, 2),
		cardTarget(oppSlot0, oppID, 0),
	)
	assert.Empty(t, actorFailEvents(mb, actor.ID), "the King look retry must succeed")
	require.True(t, g.SpecialAction.FirstStepDone, "the retried look should mark the first step done")
	assert.Equal(t, startActing, g.Engine.ActingPlayer(), "the King look must not advance the turn")

	// Confirm the swap: the game remains fully playable and the turn advances.
	mb.clear()
	g.ProcessSpecialAction(actor.ID, "swap_peek_swap", nil, nil)
	assert.False(t, g.SpecialAction.Active, "the King swap should clear the ability")
	assert.NotEqual(t, startActing, g.Engine.ActingPlayer(), "the King swap should advance the turn")
}

// TestObfStateExposesOpponentIDsHiddenForAllViewers verifies every viewer sees opponent hand slots
// as id references with Known:false and no leaked face fields.
func TestObfStateExposesOpponentIDsHiddenForAllViewers(t *testing.T) {
	g, players, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	for _, viewer := range players {
		obf := g.GetCurrentObfuscatedGameState(viewer.ID)
		sawOpponent := false
		for _, ps := range obf.Players {
			if ps.PlayerID == viewer.ID {
				continue // Self-view gating is covered by sync_seen_test.go.
			}
			sawOpponent = true
			require.NotEmpty(t, ps.RevealedHand, "opponent hand should be exposed as id references")
			require.Equal(t, ps.HandSize, len(ps.RevealedHand), "opponent revealed slots should match hand size")
			for j, c := range ps.RevealedHand {
				assert.Falsef(t, c.Known, "opponent slot %d must be hidden", j)
				assert.NotEqualf(t, uuid.Nil, c.ID, "opponent slot %d must carry a real UUID", j)
				assert.Emptyf(t, c.Rank, "opponent slot %d must not leak a rank", j)
				assert.Emptyf(t, c.Suit, "opponent slot %d must not leak a suit", j)
				assert.Zerof(t, c.Value, "opponent slot %d must not leak a value", j)
				require.NotNilf(t, c.Idx, "opponent slot %d must carry its index", j)
				assert.Equalf(t, j, *c.Idx, "opponent slot %d index should match position", j)
			}
		}
		require.True(t, sawOpponent, "viewer %s should see at least one opponent", viewer.ID)
	}
}
