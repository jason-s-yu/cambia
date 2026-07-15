// internal/game/sync_seen_test.go
//
// Tests for the self-view knowledge gate (cambia-505): a player's obfuscated sync state must
// reveal rank/suit only for own cards they have legitimately seen (pregame peek, own draw,
// peek-own ability, King look of the own card). Unseen own cards render as face-down backs so
// the memory mechanic is preserved.
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

// TestPregameSyncRevealsOnlyPeekedCards verifies a fresh game exposes exactly the two pregame
// peeks as Known and every other own card as a hidden back.
func TestPregameSyncRevealsOnlyPeekedCards(t *testing.T) {
	g, players, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	for _, p := range players {
		engineIdx := g.PlayerToEngine[p.ID]
		peek := g.Engine.Players[engineIdx].InitialPeek
		peekCount := g.Engine.Players[engineIdx].InitialPeekCount
		peeked := make(map[int]bool)
		for i := uint8(0); i < peekCount; i++ {
			peeked[int(peek[i])] = true
		}

		hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(p.ID), p.ID)
		require.NotEmpty(t, hand, "self view should carry a revealed hand")

		knownCount := 0
		for j, c := range hand {
			if peeked[j] {
				assert.Truef(t, c.Known, "peeked slot %d should be known", j)
				assert.NotEmptyf(t, c.Rank, "peeked slot %d should carry a rank", j)
			} else {
				assert.Falsef(t, c.Known, "unpeeked slot %d must stay hidden", j)
				assert.Emptyf(t, c.Rank, "unpeeked slot %d must not leak a rank", j)
				assert.Emptyf(t, c.Suit, "unpeeked slot %d must not leak a suit", j)
			}
			if c.Known {
				knownCount++
			}
		}
		assert.Equalf(t, int(peekCount), knownCount, "player %s should know exactly the peeked cards", p.ID)
	}
}

// TestReplaceFromDrawMarksNewCardSeen verifies the card placed into a hand from a draw is seen.
func TestReplaceFromDrawMarksNewCardSeen(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	require.GreaterOrEqual(t, int(g.Engine.Players[engineIdx].HandLen), 3, "need a slot 2 to replace")

	// Slot 2 is not a pregame peek, so it must start hidden.
	pre := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	require.False(t, pre[2].Known, "slot 2 should start hidden")

	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
	require.NotEqual(t, uuid.Nil, drawnUUID, "draw should set a drawn card UUID")

	oldSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_replace",
		Payload:    map[string]interface{}{"id": oldSlot2.String(), "idx": float64(2)},
	})

	hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	require.GreaterOrEqual(t, len(hand), 3)
	assert.Equal(t, drawnUUID, hand[2].ID, "drawn card should now occupy slot 2")
	assert.True(t, hand[2].Known, "the drawn card placed into the hand must be seen")
	assert.NotEmpty(t, hand[2].Rank, "seen card must carry a rank")
}

// TestBlindSwapInYieldsUnknown verifies a card moved into a hand by a J/Q blind swap is hidden,
// even though the outgoing pregame-peeked card was seen.
func TestBlindSwapInYieldsUnknown(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppEngineIdx := uint8(1 - int(engineIdx))
	oppID := g.EngineToPlayer[oppEngineIdx]

	// Slot 0 (a pregame peek) starts seen; capture the opponent's slot-0 card to swap in.
	pre := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	require.True(t, pre[0].Known, "slot 0 should start seen (pregame peek)")
	ownSlot0 := g.CardTracker.Players[engineIdx].HandUUIDs[0]
	oppSlot0 := g.CardTracker.Players[oppEngineIdx].HandUUIDs[0]

	// Force a Jack (blind-swap ability) as the drawn card.
	jUUID := forceStockTop(g, engine.NewCard(engine.SuitSpades, engine.RankJack))
	g.HandlePlayerAction(actor.ID, models.GameAction{ActionType: "action_draw_stockpile"})
	require.Equal(t, jUUID, g.CardTracker.Players[engineIdx].DrawnCardUUID, "drawn card should be the forced Jack")

	// Discard the Jack -> buffers the swap_blind ability choice.
	g.HandlePlayerAction(actor.ID, models.GameAction{
		ActionType: "action_discard",
		Payload:    map[string]interface{}{"id": jUUID.String()},
	})
	require.True(t, g.SpecialAction.Active, "Jack discard should activate a special action")
	require.Equal(t, "J", g.SpecialAction.CardRank)

	// Swap own slot 0 with opponent slot 0.
	g.ProcessSpecialAction(actor.ID, "swap_blind",
		cardTarget(ownSlot0, actor.ID, 0),
		cardTarget(oppSlot0, oppID, 0),
	)

	hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, oppSlot0, hand[0].ID, "slot 0 should now hold the opponent's card")
	assert.False(t, hand[0].Known, "a blind-swapped-in card must be hidden")
	assert.Empty(t, hand[0].Rank, "hidden swapped-in card must not leak a rank")
	// The untouched pregame peek at slot 1 must remain visible.
	assert.True(t, hand[1].Known, "an untouched pregame-peeked card should stay seen")
}

// TestKingLookMarksOwnSeenThenSwapInHidden verifies a King look reveals the actor's own looked
// card, but a subsequent King swap moves an unseen opponent card into the hand.
func TestKingLookMarksOwnSeenThenSwapInHidden(t *testing.T) {
	g, _, _ := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

	actor := currentTurnPlayer(g)
	engineIdx := g.PlayerToEngine[actor.ID]
	oppEngineIdx := uint8(1 - int(engineIdx))
	oppID := g.EngineToPlayer[oppEngineIdx]

	// Slot 2 is not a pregame peek, so it starts hidden.
	pre := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	require.False(t, pre[2].Known, "slot 2 should start hidden")
	ownSlot2 := g.CardTracker.Players[engineIdx].HandUUIDs[2]
	oppSlot0 := g.CardTracker.Players[oppEngineIdx].HandUUIDs[0]

	// Force a King (look-and-swap ability) as the drawn card.
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
	assert.True(t, afterLook[2].Known, "King look must reveal the actor's own looked card")
	assert.NotEmpty(t, afterLook[2].Rank, "revealed own card must carry a rank")

	// King swap: the peeked opponent card moves into slot 2 and must be hidden (not auto-revealed).
	g.ProcessSpecialAction(actor.ID, "swap_peek_swap", nil, nil)

	afterSwap := selfRevealedHand(g.GetCurrentObfuscatedGameState(actor.ID), actor.ID)
	assert.Equal(t, oppSlot0, afterSwap[2].ID, "slot 2 should now hold the opponent's card")
	assert.False(t, afterSwap[2].Known, "a King-swapped-in card must be hidden")
	assert.Empty(t, afterSwap[2].Rank, "hidden swapped-in card must not leak a rank")
}

// TestSnapRemovalDropsSeenCardFromView verifies a seen own card that is snapped away no longer
// appears face-up in the owner's self view.
func TestSnapRemovalDropsSeenCardFromView(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, &HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2})

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

	// Precondition: the seen snap card is visible in the self view.
	before := selfRevealedHand(g.GetCurrentObfuscatedGameState(snapper.ID), snapper.ID)
	require.True(t, containsKnownCard(before, snapUUID), "seen snap card should be visible before snapping")

	mb.clear()
	g.HandlePlayerAction(snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": snapUUID.String()},
	})

	after := selfRevealedHand(g.GetCurrentObfuscatedGameState(snapper.ID), snapper.ID)
	for _, c := range after {
		assert.NotEqualf(t, snapUUID, c.ID, "snapped card must no longer appear in the self view")
	}
}

// containsKnownCard reports whether hand holds cardID as a Known (face-up) card.
func containsKnownCard(hand []ObfCard, cardID uuid.UUID) bool {
	for _, c := range hand {
		if c.ID == cardID && c.Known {
			return true
		}
	}
	return false
}
