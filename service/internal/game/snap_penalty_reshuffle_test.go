// internal/game/snap_penalty_reshuffle_test.go: snap-penalty draws against an empty or near-empty
// stockpile (cambia-799).
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// snapPenaltyFixture is a two-player game whose snapper holds a card that cannot match the discard
// top, with both piles forced to known contents.
type snapPenaltyFixture struct {
	game        *CambiaGame
	broadcaster *mockBroadcaster
	snapper     *models.Player
	snapperIdx  uint8
	snapCardID  uuid.UUID // The mismatched card the snapper plays.
	discardIDs  []uuid.UUID
	stockIDs    []uuid.UUID
}

// setupSnapPenaltyGame forces the stockpile and discard pile to the given cards (index 0 is the
// bottom of each pile, so the last discard card is the top one) and hands the snapper a card whose
// rank cannot match the discard top, so that snapping it fails and draws the penalty.
func setupSnapPenaltyGame(t *testing.T, stockCards, discardCards []engine.Card, mismatch engine.Card) *snapPenaltyFixture {
	t.Helper()
	require.NotEmpty(t, discardCards, "the discard pile needs a top card for a snap to resolve against")
	require.NotEqual(t, discardCards[len(discardCards)-1].Rank(), mismatch.Rank(),
		"the snapper's card must not match the discard top, otherwise the snap succeeds")

	g, players, mb := setupTestGame(t, 2, &HouseRules{PenaltyDrawCount: 2, TurnTimerSec: 0})
	snapper := players[1]
	snapperIdx := g.PlayerToEngine[snapper.ID]

	// Trim the snapper's hand so the penalty cards have room (MaxHandSize is 6).
	for g.Engine.Players[snapperIdx].HandLen > 2 {
		lastIdx := g.Engine.Players[snapperIdx].HandLen - 1
		removedUUID := g.CardTracker.Players[snapperIdx].HandUUIDs[lastIdx]
		g.Engine.Players[snapperIdx].Hand[lastIdx] = engine.EmptyCard
		g.Engine.Players[snapperIdx].HandLen--
		g.CardTracker.Players[snapperIdx].HandUUIDs[lastIdx] = uuid.Nil
		delete(g.CardTracker.Registry, removedUUID)
	}

	// Force the stockpile. Slots beyond StockLen keep the UUIDs the deal assigned them, which is
	// what a live game looks like once cards have been drawn out: the mirror above StockLen is
	// stale, so a reshuffle that is not mirrored hands out an ID belonging to another card.
	stockIDs := make([]uuid.UUID, len(stockCards))
	for i, c := range stockCards {
		id := uuid.New()
		g.Engine.Stockpile[i] = c
		g.CardTracker.StockUUIDs[i] = id
		g.CardTracker.Registry[id] = engineCardToDetails(c, id)
		stockIDs[i] = id
	}
	g.Engine.StockLen = uint8(len(stockCards))
	g.CardTracker.StockLen = g.Engine.StockLen

	// Force the discard pile.
	discardIDs := make([]uuid.UUID, len(discardCards))
	for i, c := range discardCards {
		id := uuid.New()
		g.Engine.DiscardPile[i] = c
		g.CardTracker.DiscardUUIDs[i] = id
		g.CardTracker.Registry[id] = engineCardToDetails(c, id)
		discardIDs[i] = id
	}
	g.Engine.DiscardLen = uint8(len(discardCards))
	g.CardTracker.DiscardLen = g.Engine.DiscardLen

	// Give the snapper the mismatched card.
	handLen := g.Engine.Players[snapperIdx].HandLen
	snapCardID := uuid.New()
	g.Engine.Players[snapperIdx].Hand[handLen] = mismatch
	g.Engine.Players[snapperIdx].HandLen++
	g.CardTracker.Players[snapperIdx].HandUUIDs[handLen] = snapCardID
	g.CardTracker.Registry[snapCardID] = engineCardToDetails(mismatch, snapCardID)

	g.syncPlayerHandsFromEngine()
	mb.clear()

	return &snapPenaltyFixture{
		game:        g,
		broadcaster: mb,
		snapper:     snapper,
		snapperIdx:  snapperIdx,
		snapCardID:  snapCardID,
		discardIDs:  discardIDs,
		stockIDs:    stockIDs,
	}
}

// snap plays the fixture's mismatched card, which fails and triggers the penalty draw.
func (f *snapPenaltyFixture) snap() {
	f.game.HandlePlayerAction(f.snapper.ID, models.GameAction{
		ActionType: "action_snap",
		Payload:    map[string]interface{}{"id": f.snapCardID.String()},
	})
}

// countEvents returns how many broadcast events of the given type were captured.
func (mb *mockBroadcaster) countEvents(eventType GameEventType) int {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	n := 0
	for _, ev := range mb.allEvents {
		if ev.Type == eventType {
			n++
		}
	}
	return n
}

// assertNoDuplicateCardIDs checks that no card ID is mirrored in two places at once, which is what
// happens when the stockpile mirror is not rebuilt after a reshuffle: a reshuffled card is handed
// out under the ID of a card already sitting in someone's hand.
func assertNoDuplicateCardIDs(t *testing.T, g *CambiaGame) {
	t.Helper()
	seen := make(map[uuid.UUID]string)
	record := func(id uuid.UUID, where string) {
		if id == uuid.Nil {
			return
		}
		if prev, ok := seen[id]; ok {
			t.Errorf("card ID %s appears in both %s and %s", id, prev, where)
			return
		}
		seen[id] = where
	}
	for p := uint8(0); p < engine.MaxPlayers; p++ {
		for i := uint8(0); i < g.Engine.Players[p].HandLen; i++ {
			record(g.CardTracker.Players[p].HandUUIDs[i], "a player hand")
		}
	}
	for i := uint8(0); i < g.Engine.StockLen; i++ {
		record(g.CardTracker.StockUUIDs[i], "the stockpile")
	}
	for i := uint8(0); i < g.Engine.DiscardLen; i++ {
		record(g.CardTracker.DiscardUUIDs[i], "the discard pile")
	}
}

// assertPenaltyCardsMirrored checks that every card the penalty added to the snapper's hand is
// mirrored by an ID whose registry entry describes that exact card.
func assertPenaltyCardsMirrored(t *testing.T, g *CambiaGame, snapperIdx uint8, firstPenaltySlot uint8) {
	t.Helper()
	for i := firstPenaltySlot; i < g.Engine.Players[snapperIdx].HandLen; i++ {
		id := g.CardTracker.Players[snapperIdx].HandUUIDs[i]
		require.NotEqual(t, uuid.Nil, id, "penalty card at hand slot %d has no ID", i)
		details := g.CardTracker.Registry[id]
		require.NotNil(t, details, "penalty card at hand slot %d is not in the registry", i)
		assert.True(t, registryCardMatches(details, g.Engine.Players[snapperIdx].Hand[i]),
			"penalty card ID at hand slot %d names a different card (%s%s) than the engine holds", i, details.Rank, details.Suit)
	}
}

// TestSnapPenaltyEmptyStockpileReshuffles covers a failed snap whose penalty is owed against a
// completely empty stockpile. The engine reshuffles the discard pile back in and pays the penalty
// in full; before cambia-799 the service's hand-rolled draw broke out of the loop instead, paying
// nothing.
func TestSnapPenaltyEmptyStockpileReshuffles(t *testing.T) {
	discard := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankFive),
		engine.NewCard(engine.SuitClubs, engine.RankSix),
		engine.NewCard(engine.SuitDiamonds, engine.RankSeven),
		engine.NewCard(engine.SuitSpades, engine.RankEight), // top
	}
	f := setupSnapPenaltyGame(t, nil, discard, engine.NewCard(engine.SuitHearts, engine.RankNine))
	g := f.game
	handBefore := g.Engine.Players[f.snapperIdx].HandLen

	// The engine's own penalty draw, run on a copy taken before the snap, is the parity reference:
	// GameState carries its RNG, so the copy reshuffles identically.
	expected := g.Engine
	for i := 0; i < g.HouseRules.PenaltyDrawCount; i++ {
		expected.DrawPenaltyCard(f.snapperIdx)
	}

	f.snap()

	assert.EqualValues(t, handBefore+2, g.Engine.Players[f.snapperIdx].HandLen, "the full 2-card penalty should be paid off the reshuffled stockpile")
	assert.EqualValues(t, 1, g.Engine.StockLen, "3 cards reshuffle in and 2 are drawn as penalty")
	assert.EqualValues(t, 1, g.Engine.DiscardLen, "the reshuffle leaves only the top discard card")

	// Deck state must match the engine's own penalty draw card for card.
	assert.Equal(t, expected.StockLen, g.Engine.StockLen, "stockpile size diverged from the engine")
	assert.Equal(t, expected.DiscardLen, g.Engine.DiscardLen, "discard size diverged from the engine")
	assert.Equal(t, expected.Stockpile[:expected.StockLen], g.Engine.Stockpile[:g.Engine.StockLen], "stockpile contents diverged from the engine")
	assert.Equal(t, expected.DiscardPile[:expected.DiscardLen], g.Engine.DiscardPile[:g.Engine.DiscardLen], "discard contents diverged from the engine")
	assert.Equal(t, expected.Players[f.snapperIdx].Hand, g.Engine.Players[f.snapperIdx].Hand, "penalised hand diverged from the engine")
	assert.Equal(t, expected.RNG, g.Engine.RNG, "the reshuffle consumed a different amount of randomness than the engine's own")

	// Card identity is mirrored across the reshuffle.
	assertPenaltyCardsMirrored(t, g, f.snapperIdx, handBefore)
	assertNoDuplicateCardIDs(t, g)
	assert.Equal(t, f.discardIDs[len(f.discardIDs)-1], g.CardTracker.DiscardUUIDs[0], "the preserved top discard card keeps its ID")
	movedIDs := map[uuid.UUID]bool{}
	for _, id := range f.discardIDs[:len(f.discardIDs)-1] {
		movedIDs[id] = true
	}
	for i := handBefore; i < g.Engine.Players[f.snapperIdx].HandLen; i++ {
		assert.True(t, movedIDs[g.CardTracker.Players[f.snapperIdx].HandUUIDs[i]],
			"penalty card at slot %d should carry the ID of a card that was in the discard pile", i)
	}
	assert.EqualValues(t, g.Engine.StockLen, g.CardTracker.StockLen, "tracker stockpile length should follow the engine")
	assert.EqualValues(t, g.Engine.DiscardLen, g.CardTracker.DiscardLen, "tracker discard length should follow the engine")

	// Events: one reshuffle notice carrying the settled counts, plus the usual penalty events.
	require.Equal(t, 1, f.broadcaster.countEvents(EventGameReshuffleStockpile), "exactly one reshuffle event")
	ev := f.broadcaster.findEventByType(EventGameReshuffleStockpile)
	require.NotNil(t, ev.Payload, "the reshuffle event must carry the corrected counts")
	assert.EqualValues(t, g.Engine.StockLen, ev.Payload["stockpileSize"])
	assert.EqualValues(t, g.Engine.DiscardLen, ev.Payload["discardSize"])
	assert.Equal(t, 2, f.broadcaster.countEvents(EventPlayerSnapPenalty), "one public event per penalty card")
	assert.Len(t, f.broadcaster.playerEvents[f.snapper.ID], 2, "one private event per penalty card")
}

// TestSnapPenaltyNearEmptyStockpileReshufflesMidPenalty covers the penalty that empties the
// stockpile partway through: the first card comes off the stockpile, the second forces a reshuffle.
func TestSnapPenaltyNearEmptyStockpileReshufflesMidPenalty(t *testing.T) {
	stock := []engine.Card{engine.NewCard(engine.SuitClubs, engine.RankTwo)}
	discard := []engine.Card{
		engine.NewCard(engine.SuitDiamonds, engine.RankThree),
		engine.NewCard(engine.SuitHearts, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankFive), // top
	}
	f := setupSnapPenaltyGame(t, stock, discard, engine.NewCard(engine.SuitHearts, engine.RankNine))
	g := f.game
	handBefore := g.Engine.Players[f.snapperIdx].HandLen

	expected := g.Engine
	for i := 0; i < g.HouseRules.PenaltyDrawCount; i++ {
		expected.DrawPenaltyCard(f.snapperIdx)
	}

	f.snap()

	assert.EqualValues(t, handBefore+2, g.Engine.Players[f.snapperIdx].HandLen, "both penalty cards should be paid")
	assert.EqualValues(t, 1, g.Engine.StockLen, "2 cards reshuffle in, 1 of them is drawn")
	assert.EqualValues(t, 1, g.Engine.DiscardLen, "the reshuffle leaves only the top discard card")
	assert.Equal(t, expected.Stockpile[:expected.StockLen], g.Engine.Stockpile[:g.Engine.StockLen], "stockpile contents diverged from the engine")
	assert.Equal(t, expected.Players[f.snapperIdx].Hand, g.Engine.Players[f.snapperIdx].Hand, "penalised hand diverged from the engine")

	// The first penalty card is the one that was already in the stockpile; the second comes from the
	// reshuffle and must carry a discard-pile ID, not a stale stockpile one.
	assert.Equal(t, f.stockIDs[0], g.CardTracker.Players[f.snapperIdx].HandUUIDs[handBefore], "the first penalty card keeps its stockpile ID")
	assert.Contains(t, f.discardIDs[:len(f.discardIDs)-1], g.CardTracker.Players[f.snapperIdx].HandUUIDs[handBefore+1],
		"the second penalty card should carry the ID of a reshuffled discard card")
	assertPenaltyCardsMirrored(t, g, f.snapperIdx, handBefore)
	assertNoDuplicateCardIDs(t, g)

	require.Equal(t, 1, f.broadcaster.countEvents(EventGameReshuffleStockpile), "exactly one reshuffle event")
	assert.Equal(t, 2, f.broadcaster.countEvents(EventPlayerSnapPenalty), "one public event per penalty card")
}

// TestSnapPenaltyExhaustedDeckPaysShort covers the deck being genuinely exhausted: an empty
// stockpile and a discard pile too thin to reshuffle. The penalty is paid short, as the engine pays
// it, without panicking or fabricating cards.
func TestSnapPenaltyExhaustedDeckPaysShort(t *testing.T) {
	discard := []engine.Card{engine.NewCard(engine.SuitSpades, engine.RankEight)} // top only
	f := setupSnapPenaltyGame(t, nil, discard, engine.NewCard(engine.SuitHearts, engine.RankNine))
	g := f.game
	handBefore := g.Engine.Players[f.snapperIdx].HandLen

	f.snap()

	assert.Equal(t, handBefore, g.Engine.Players[f.snapperIdx].HandLen, "no cards exist to pay the penalty with")
	assert.EqualValues(t, 0, g.Engine.StockLen, "the stockpile stays empty")
	assert.EqualValues(t, 1, g.Engine.DiscardLen, "a single discard card cannot be reshuffled")
	assert.Equal(t, 0, f.broadcaster.countEvents(EventGameReshuffleStockpile), "nothing was reshuffled")
	assert.Equal(t, 0, f.broadcaster.countEvents(EventPlayerSnapPenalty), "no penalty cards were drawn")
	require.NotNil(t, f.broadcaster.findEventByType(EventPlayerSnapFail), "the snap still fails")
	assertNoDuplicateCardIDs(t, g)
}

// TestSnapPenaltyHealthyStockpileDoesNotReshuffle verifies the ordinary penalty is untouched: two
// cards off the stockpile, no reshuffle, no reshuffle event.
func TestSnapPenaltyHealthyStockpileDoesNotReshuffle(t *testing.T) {
	stock := []engine.Card{
		engine.NewCard(engine.SuitClubs, engine.RankTwo),
		engine.NewCard(engine.SuitDiamonds, engine.RankThree),
		engine.NewCard(engine.SuitHearts, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankSix),
	}
	discard := []engine.Card{
		engine.NewCard(engine.SuitClubs, engine.RankQueen),
		engine.NewCard(engine.SuitSpades, engine.RankEight), // top
	}
	f := setupSnapPenaltyGame(t, stock, discard, engine.NewCard(engine.SuitHearts, engine.RankNine))
	g := f.game
	handBefore := g.Engine.Players[f.snapperIdx].HandLen

	f.snap()

	assert.EqualValues(t, handBefore+2, g.Engine.Players[f.snapperIdx].HandLen, "the full penalty is paid")
	assert.EqualValues(t, len(stock)-2, g.Engine.StockLen, "penalty cards come off the stockpile")
	assert.EqualValues(t, len(discard), g.Engine.DiscardLen, "the discard pile is untouched")
	assert.Equal(t, 0, f.broadcaster.countEvents(EventGameReshuffleStockpile), "a stocked pile must not emit a reshuffle event")

	// The penalty cards come off the top of the stockpile, keeping their mirrored IDs.
	assert.Equal(t, f.stockIDs[len(stock)-1], g.CardTracker.Players[f.snapperIdx].HandUUIDs[handBefore])
	assert.Equal(t, f.stockIDs[len(stock)-2], g.CardTracker.Players[f.snapperIdx].HandUUIDs[handBefore+1])
	assertPenaltyCardsMirrored(t, g, f.snapperIdx, handBefore)
	assertNoDuplicateCardIDs(t, g)
}
