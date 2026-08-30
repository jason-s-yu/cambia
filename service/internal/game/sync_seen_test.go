// internal/game/sync_seen_test.go
//
// Tests for the self-view rule (cambia-1094): no own card is ever persistently face-up. The
// obfuscated sync state hides rank/suit/value for EVERY own hand slot in every phase - pregame,
// live play, and a reconnect - because the physical game gives you the pregame peek and then turns
// every card down, leaving you to play the round on memory. This replaces the cambia-505 memory
// aid, which kept a card face-up for the rest of the round once its owner had legitimately seen it.
//
// The reveals a player is entitled to travel in their own events and nowhere else:
// private_initial_cards for the pregame peek, private_draw_stockpile (and the snapshot's DrawnCard)
// for a card in hand off the pile, private_special_action_success for a 7/8 peek or a King look.
// Each is a window the client shows and then closes.
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// selfRevealedHand returns the RevealedHand the snapshot exposes for playerID's own seat.
func selfRevealedHand(obf ObfGameState, playerID uuid.UUID) []ObfCard {
	for _, p := range obf.Players {
		if p.PlayerID == playerID {
			return p.RevealedHand
		}
	}
	return nil
}

// forceStockTop overwrites the stockpile top with card, wiring a fresh UUID into the tracker so a
// subsequent draw yields a deterministic card. Returns the UUID assigned to it.
func forceStockTop(g *CambiaGame, card engine.Card) uuid.UUID {
	top := g.Engine.StockLen - 1
	g.Engine.Stockpile[top] = card
	id := uuid.New()
	g.CardTracker.StockUUIDs[top] = id
	g.CardTracker.Registry[id] = engineCardToDetails(card, id)
	return id
}

// cardTarget builds a special-action card payload (id, idx, owner) matching the client wire shape.
func cardTarget(id, owner uuid.UUID, idx int) map[string]interface{} {
	return map[string]interface{}{
		"id":   id.String(),
		"idx":  float64(idx),
		"user": map[string]interface{}{"id": owner.String()},
	}
}

// countPlayerEvents returns how many events of the given type playerID has been sent privately.
func countPlayerEvents(mb *mockBroadcaster, playerID uuid.UUID, evType GameEventType) int {
	n := 0
	for _, ev := range mb.playerEvents[playerID] {
		if ev.Type == evType {
			n++
		}
	}
	return n
}

// assertHandFullyHidden checks the self-view invariant on one hand: every slot carries its card id
// and its index (ability targeting names an own card by id, cambia-509) and nothing else.
func assertHandFullyHidden(t *testing.T, hand []ObfCard, where string) {
	t.Helper()
	require.NotEmptyf(t, hand, "%s: the self view should still carry a slot per card", where)
	for i, c := range hand {
		assert.Falsef(t, c.Known, "%s: slot %d must not be Known", where, i)
		assert.Emptyf(t, c.Rank, "%s: slot %d leaked a rank", where, i)
		assert.Emptyf(t, c.Suit, "%s: slot %d leaked a suit", where, i)
		assert.Zerof(t, c.Value, "%s: slot %d leaked a value", where, i)
		assert.NotEqualf(t, uuid.Nil, c.ID, "%s: slot %d should still carry its card id", where, i)
		if assert.NotNilf(t, c.Idx, "%s: slot %d should still carry its index", where, i) {
			assert.Equalf(t, i, *c.Idx, "%s: slot %d index should match its position", where, i)
		}
	}
}

// TestPregameSelfViewHidesEveryOwnCard verifies the pregame snapshot hides the peeked cards too.
// The peek travels in private_initial_cards alone: a snapshot that repeated it would keep those
// faces up for the whole round, which is the bug this replaces (cambia-1094).
func TestPregameSelfViewHidesEveryOwnCard(t *testing.T) {
	rules := DefaultHouseRules()
	rules.TurnTimerSec = 0
	g, ids, mb := beginPreGameWithRules(t, rules)

	for _, id := range ids {
		hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(id), id)
		assertHandFullyHidden(t, hand, "pregame self view")

		// The reveal itself must still carry the faces, or the peek would be lost outright.
		ev := lastPlayerInitialCards(mb, id)
		require.NotNilf(t, ev, "player %s should receive a private_initial_cards event", id)
		require.NotEmpty(t, ev.Cards, "the pregame reveal should carry the peeked cards")
		for _, c := range ev.Cards {
			assert.NotEmpty(t, c.Rank, "the pregame reveal carries the face")
		}
	}
}

// TestPostStartSelfViewHidesEveryOwnCard verifies the transition to live play does not resurrect
// the pregame faces: the sync StartGame broadcasts is what turns the peeked cards down on screen.
func TestPostStartSelfViewHidesEveryOwnCard(t *testing.T) {
	g, players, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	for _, p := range players {
		hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(p.ID), p.ID)
		assertHandFullyHidden(t, hand, "post-start self view")
	}
}

// TestReplaceFromDrawLeavesSlotHidden verifies a card put into the hand from a draw is face-down in
// the snapshot. The drawn card is revealed while it is still in hand (DrawnCard below, and
// private_draw_stockpile), and the client shows the swap for its window; the slot it lands in is
// not a permanent face.
func TestReplaceFromDrawLeavesSlotHidden(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	require.GreaterOrEqual(t, int(g.Engine.Players[engineIdx].HandLen), 3, "need a slot 2 to replace")

	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "draw should set a drawn card UUID")

	// The pending drawn card IS revealed: it is in the player's hand, not their fan, and the
	// client needs its face to choose between discarding and replacing (cambia-1094 keeps this).
	pending := g.GetCurrentObfuscatedGameState(actor.ID)
	self := pending.Players[0]
	for i := range pending.Players {
		if pending.Players[i].PlayerID == actor.ID {
			self = pending.Players[i]
		}
	}
	require.NotNil(t, self.DrawnCard, "the drawn card should still be revealed to its drawer")
	assert.True(t, self.DrawnCard.Known, "the drawn card is the one own card the snapshot names")
	assert.NotEmpty(t, self.DrawnCard.Rank, "the drawn card should carry its face")

	oldSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_replace",
		Payload:    map[string]interface{}{"id": oldSlot2.String(), "idx": float64(2)},
	})

	hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	require.GreaterOrEqual(t, len(hand), 3)
	assert.Equal(t, drawnUUID, hand[2].ID, "drawn card should now occupy slot 2")
	assertHandFullyHidden(t, hand, "self view after a replace")
}

// TestPeekSelfLeavesSlotHidden verifies a 7/8 peek-own does not turn the card up in the snapshot.
// The face travels in private_special_action_success, which the client holds on screen for its
// window and then drops.
func TestPeekSelfLeavesSlotHidden(t *testing.T) {
	g, _, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	ownSlot0 := g.CardTracker.Players[engineIdx].HandUUIDs[0]

	sevenUUID := forceStockTop(g, engine.NewCard(engine.SuitSpades, engine.RankSeven))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, sevenUUID, g.CardTracker.Players[engineIdx].DrawnCardUUID, "drawn card should be the forced 7")
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": sevenUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "a 7 discard should activate a special action")

	mb.clear()
	g.ProcessSpecialAction(actor.ID, "peek_self", cardTarget(ownSlot0, actor.ID, 0), nil)

	// The private reveal is the carrier of the face.
	var reveal *GameEvent
	for i := range mb.playerEvents[actor.ID] {
		if mb.playerEvents[actor.ID][i].Type == EventPrivateSpecialSuccess {
			reveal = &mb.playerEvents[actor.ID][i]
		}
	}
	require.NotNil(t, reveal, "a peek_self should privately reveal the card")
	require.NotNil(t, reveal.Card1)
	assert.NotEmpty(t, reveal.Card1.Rank, "the private reveal should carry the peeked face")

	hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, ownSlot0, hand[0].ID, "the peeked card should still be in slot 0")
	assertHandFullyHidden(t, hand, "self view after a peek_self")
}

// TestKingLookAndSwapLeaveSlotsHidden verifies neither half of a King turns an own card up in the
// snapshot: not the card the actor looked at, and not the opponent card the swap moved in.
func TestKingLookAndSwapLeaveSlotsHidden(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppEngineIdx := uint8(1 - int(engineIdx))
	oppID := g.EngineToPlayer[oppEngineIdx]

	ownSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	oppSlot0 := g.CardTracker.Players[oppEngineIdx].HandUUIDs[0]

	kUUID := forceStockTop(g, engine.NewCard(engine.SuitSpades, engine.RankKing))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, kUUID, g.CardTracker.Players[engineIdx].DrawnCardUUID, "drawn card should be the forced King")

	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": kUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "King discard should activate a special action")
	require.Equal(t, "K", g.SpecialAction.CardRank)

	// King look: view own slot 2 and opponent slot 0.
	g.ProcessSpecialAction(actor.ID, "swap_peek",
		cardTarget(ownSlot2, actor.ID, 2),
		cardTarget(oppSlot0, oppID, 0),
	)
	afterLook := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, ownSlot2, afterLook[2].ID, "the looked-at card should still be in slot 2")
	assertHandFullyHidden(t, afterLook, "self view after a King look")

	// King swap: the peeked opponent card moves into slot 2 and stays hidden as well.
	g.ProcessSpecialAction(actor.ID, "swap_peek_swap", nil, nil)
	afterSwap := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, oppSlot0, afterSwap[2].ID, "slot 2 should now hold the opponent's card")
	assertHandFullyHidden(t, afterSwap, "self view after a King swap")
}

// TestBlindSwapMovesIdAndStaysHidden verifies the UUID tracker moves the swapped card into the
// actor's slot and that the slot is a hidden id reference on both sides of the swap.
func TestBlindSwapMovesIdAndStaysHidden(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, testHouseRules(0, 2))

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppEngineIdx := uint8(1 - int(engineIdx))
	oppID := g.EngineToPlayer[oppEngineIdx]

	ownSlot0 := g.CardTracker.Players[engineIdx].HandUUIDs[0]
	oppSlot0 := g.CardTracker.Players[oppEngineIdx].HandUUIDs[0]

	jUUID := forceStockTop(g, engine.NewCard(engine.SuitSpades, engine.RankJack))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, jUUID, g.CardTracker.Players[engineIdx].DrawnCardUUID, "drawn card should be the forced Jack")

	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": jUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "Jack discard should activate a special action")
	require.Equal(t, "J", g.SpecialAction.CardRank)

	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownSlot0, actor.ID, 0),
		cardTarget(oppSlot0, oppID, 0),
	)

	hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, oppSlot0, hand[0].ID, "slot 0 should now hold the opponent's card")
	assertHandFullyHidden(t, hand, "self view after a blind swap")
}

// TestReconnectSelfViewHidesEveryOwnCard verifies the repair snapshot a returning player is sent
// carries no own faces either. A reconnect used to be where a stale client re-learned its hand;
// under cambia-1094 it re-learns ids and slots only.
func TestReconnectSelfViewHidesEveryOwnCard(t *testing.T) {
	rules := testHouseRules(0, 2)
	rules.ForfeitOnDisconnect = false
	g, players, mb := setupTestGame(t, 2, rules)

	// Give the returning player a card they have legitimately seen, so a hidden hand is not hidden
	// merely because nothing was ever revealed to them.
	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	oldSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_replace",
		Payload:    map[string]interface{}{"id": oldSlot2.String(), "idx": float64(2)},
	})
	require.True(t, g.hasSeenCard(engineIdx, drawnUUID), "the drawn card should be in the seen set")

	mb.clear()
	g.HandleDisconnect(actor.ID)
	g.HandleReconnect(actor.ID, nil)

	var sync *GameEvent
	for i := range mb.playerEvents[actor.ID] {
		if mb.playerEvents[actor.ID][i].Type == EventPrivateSyncState {
			sync = &mb.playerEvents[actor.ID][i]
		}
	}
	require.NotNil(t, sync, "a reconnecting player should be sent a private_sync_state")
	require.NotNil(t, sync.State)
	assertHandFullyHidden(t, selfRevealedHand(*sync.State, actor.ID), "reconnect self view")

	// Nothing re-fires the pregame reveal once the game is live: that window is over.
	assert.Zero(t, countPlayerEvents(mb, actor.ID, EventPrivateInitialCards),
		"a mid-game reconnect must not replay the pregame peek")
	_ = players
}

// TestPregameReconnectRefiresInitialCards verifies a player who reloads during the initial reveal
// gets it again. private_initial_cards is the only frame that ever carries those faces, so a
// reconnect answered with a sync alone would cost the returning player the whole peek for the rest
// of the round (cambia-1094).
func TestPregameReconnectRefiresInitialCards(t *testing.T) {
	rules := DefaultHouseRules()
	rules.TurnTimerSec = 0
	rules.ForfeitOnDisconnect = false
	g, ids, mb := beginPreGameWithRules(t, rules)

	first := lastPlayerInitialCards(mb, ids[0])
	require.NotNil(t, first, "the pregame reveal should have been sent once already")
	want := make(map[int]string, len(first.Cards))
	for _, c := range first.Cards {
		require.NotNil(t, c.Idx)
		want[*c.Idx] = c.Rank + c.Suit
	}
	require.NotEmpty(t, want, "the default house rules peek at least one card")

	mb.clear()
	g.HandleDisconnect(ids[0])
	g.HandleReconnect(ids[0], nil)

	require.True(t, g.PreGameActive, "the reveal window should still be open")
	assert.Equal(t, 1, countPlayerEvents(mb, ids[0], EventPrivateInitialCards),
		"a reconnect inside the pregame window should re-fire the reveal exactly once")

	again := lastPlayerInitialCards(mb, ids[0])
	require.NotNil(t, again)
	got := make(map[int]string, len(again.Cards))
	for _, c := range again.Cards {
		require.NotNil(t, c.Idx)
		got[*c.Idx] = c.Rank + c.Suit
		assert.NotEmpty(t, c.Rank, "the re-fired reveal should carry the face")
	}
	assert.Equal(t, want, got, "the re-fired reveal should name the same slots and faces")

	// The snapshot that accompanies it still hides everything.
	assertHandFullyHidden(t, selfRevealedHand(g.GetCurrentObfuscatedGameState(ids[0]), ids[0]), "pregame reconnect self view")

	// The other seat is not re-sent anything: only the returning player lost the frame.
	assert.Zero(t, countPlayerEvents(mb, ids[1], EventPrivateInitialCards),
		"a reconnect must not replay another player's pregame reveal")
}

// TestSyncStateSerializesPendingSpecialAction verifies a reconnecting client's sync_state carries
// the pending special action (cambia-763 F1): before it, ObfGameState never surfaced
// SpecialActionState, so a client resyncing mid-King could not restore its pendingAction UI.
func TestSyncStateSerializesPendingSpecialAction(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppEngineIdx := uint8(1 - int(engineIdx))
	oppID := g.EngineToPlayer[oppEngineIdx]

	// Precondition: no special action pending yet, so sync_state must omit it entirely.
	preState := g.GetCurrentObfuscatedGameState(actor.ID)
	require.Nil(t, preState.SpecialAction, "sync_state must not report a special action before one is pending")

	// Force a King (look-and-swap ability) as the drawn card, then discard it.
	kUUID := forceStockTop(g, engine.NewCard(engine.SuitSpades, engine.RankKing))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, kUUID, g.CardTracker.Players[engineIdx].DrawnCardUUID, "drawn card should be the forced King")
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": kUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "King discard should activate a special action")

	// Both the acting player's and the opponent's sync_state must now report the pending action
	// (it is public-safe: who owes an action and which rank triggered it, no peeked card values).
	actorState := g.GetCurrentObfuscatedGameState(actor.ID)
	require.NotNil(t, actorState.SpecialAction, "acting player's sync_state must report the pending special action")
	assert.True(t, actorState.SpecialAction.Active)
	assert.Equal(t, actor.ID, actorState.SpecialAction.PlayerID)
	assert.Equal(t, "K", actorState.SpecialAction.CardRank)

	oppState := g.GetCurrentObfuscatedGameState(oppID)
	require.NotNil(t, oppState.SpecialAction, "opponent's sync_state must also report the pending special action")
	assert.Equal(t, actor.ID, oppState.SpecialAction.PlayerID)
	assert.Equal(t, "K", oppState.SpecialAction.CardRank)

	// Resolving the action (skip) must clear it from the next sync_state.
	g.ProcessSpecialAction(actor.ID, "skip", nil, nil)
	require.False(t, g.SpecialAction.Active, "skip should resolve the pending special action")
	afterState := g.GetCurrentObfuscatedGameState(actor.ID)
	assert.Nil(t, afterState.SpecialAction, "sync_state must omit the special action once it resolves")
}

// TestSnapRemovalDropsCardFromSelfView verifies a snapped own card leaves the owner's self view
// entirely, rather than lingering as a slot the client would still target by id.
func TestSnapRemovalDropsCardFromSelfView(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	// Force a 7 onto the discard top so a matching 7 snaps successfully.
	sevenTop := engine.NewCard(engine.SuitHearts, engine.RankSeven)
	topIdx := g.Engine.DiscardLen
	g.Engine.DiscardPile[topIdx] = sevenTop
	g.Engine.DiscardLen++
	topUUID := uuid.New()
	g.CardTracker.DiscardUUIDs[topIdx] = topUUID
	g.CardTracker.Registry[topUUID] = engineCardToDetails(sevenTop, topUUID)

	// Give playerB a matching 7 in a fresh hand slot and mark it seen (as if peeked/drawn).
	snapper := players[1]
	snapperIdx := g.PlayerToEngine[snapper.ID]
	slot := g.Engine.Players[snapperIdx].HandLen
	snapCard := engine.NewCard(engine.SuitSpades, engine.RankSeven)
	g.Engine.Players[snapperIdx].Hand[slot] = snapCard
	g.Engine.Players[snapperIdx].HandLen++
	snapUUID := uuid.New()
	g.CardTracker.Players[snapperIdx].HandUUIDs[slot] = snapUUID
	g.CardTracker.Registry[snapUUID] = engineCardToDetails(snapCard, snapUUID)
	g.markCardSeen(snapperIdx, snapUUID)
	g.syncPlayerHandsFromEngine()

	// Precondition: the card is in the self view as a hidden slot reference.
	before := selfRevealedHand(g.GetCurrentObfuscatedGameState(snapper.ID), snapper.ID)
	require.True(t, containsCard(before, snapUUID), "the snap card should be a slot in the self view before snapping")
	assertHandFullyHidden(t, before, "self view before a snap")

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": snapUUID.String()},
	})

	after := selfRevealedHand(g.GetCurrentObfuscatedGameState(snapper.ID), snapper.ID)
	assert.False(t, containsCard(after, snapUUID), "snapped card must no longer appear in the self view")
	assertHandFullyHidden(t, after, "self view after a snap")
}

// containsCard reports whether hand holds a slot for cardID.
func containsCard(hand []ObfCard, cardID uuid.UUID) bool {
	for _, c := range hand {
		if c.ID == cardID {
			return true
		}
	}
	return false
}
