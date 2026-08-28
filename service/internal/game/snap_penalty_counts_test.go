// internal/game/snap_penalty_counts_test.go: pile sizes carried by the snap-penalty events
// (cambia-821).
package game

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// clientPiles replays the broadcast events the way the web client does (gameStore.ts), starting
// from the counts a sync_state would have handed it before the snap, and returns the stockpile and
// discard sizes the client ends up displaying. Every event that carries pile sizes is applied
// directly, which is the whole point of carrying them: the client never has to reconstruct a count
// the server already knows.
func clientPiles(f *snapPenaltyFixture, beforeStock, beforeDiscard int) (int, int) {
	stock, discard := beforeStock, beforeDiscard
	for _, ev := range f.broadcaster.allEvents {
		switch ev.Type {
		case EventPlayerSnapPenalty, EventGameReshuffleStockpile:
			if v, ok := ev.Payload["stockpileSize"].(int); ok {
				stock = v
			}
			if v, ok := ev.Payload["discardSize"].(int); ok {
				discard = v
			}
		}
	}
	return stock, discard
}

// TestSnapPenaltyEventsCarryPileSizes verifies every public penalty event reports the stockpile and
// discard sizes as they stand after that card was drawn, and that a client applying them lands on
// the engine's own counts. Pile sizes are public knowledge (sync_state broadcasts both), so they
// ride the public event; the penalized player already receives it alongside the private one.
//
// The healthy-stockpile case is the ticket's own acceptance case. The reshuffle and exhausted-deck
// cases are here because they are where a client reconstructing the count by subtraction has the
// least to go on: the stockpile can grow mid-penalty, and a penalty can be paid short with no event
// at all.
func TestSnapPenaltyEventsCarryPileSizes(t *testing.T) {
	mismatch := engine.NewCard(engine.SuitSpades, engine.RankEight)
	stocked := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankFive),
	}
	deepDiscard := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankFive),
		engine.NewCard(engine.SuitClubs, engine.RankSix),
		engine.NewCard(engine.SuitDiamonds, engine.RankNine),
		engine.NewCard(engine.SuitSpades, engine.RankSeven),
	}
	thinDiscard := []engine.Card{engine.NewCard(engine.SuitHearts, engine.RankSeven)}

	cases := []struct {
		name          string
		stock         []engine.Card
		discard       []engine.Card
		wantPenalties int
	}{
		{name: "healthy stockpile", stock: stocked, discard: thinDiscard, wantPenalties: 2},
		{name: "empty stockpile reshuffles", stock: nil, discard: deepDiscard, wantPenalties: 2},
		{name: "reshuffle mid penalty", stock: stocked[:1], discard: deepDiscard, wantPenalties: 2},
		{name: "exhausted deck pays short", stock: nil, discard: thinDiscard, wantPenalties: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := setupSnapPenaltyGame(t, tc.stock, tc.discard, mismatch)
			beforeStock := int(f.game.Engine.StockLen)
			beforeDiscard := int(f.game.Engine.DiscardLen)

			f.snap()

			penalties := 0
			for _, ev := range f.broadcaster.allEvents {
				if ev.Type != EventPlayerSnapPenalty {
					continue
				}
				penalties++
				require.NotNil(t, ev.Payload, "penalty event should carry a payload")
				stock, ok := ev.Payload["stockpileSize"].(int)
				require.Truef(t, ok, "penalty event %d should report the stockpile size", penalties)
				discard, ok := ev.Payload["discardSize"].(int)
				require.Truef(t, ok, "penalty event %d should report the discard size", penalties)
				assert.GreaterOrEqual(t, stock, 0, "reported stockpile size should never be negative")
				assert.Greater(t, discard, 0, "the discard pile always keeps its top card")
			}
			require.Equal(t, tc.wantPenalties, penalties)

			gotStock, gotDiscard := clientPiles(f, beforeStock, beforeDiscard)
			assert.EqualValues(t, f.game.Engine.StockLen, gotStock, "client stockpile size should match the engine after the penalty")
			assert.EqualValues(t, f.game.Engine.DiscardLen, gotDiscard, "client discard size should match the engine after the penalty")
		})
	}
}

// TestSnapPenaltyEventsReportSizesPerCard verifies the sizes are the ones in force when each card
// was drawn rather than a single end-of-penalty figure repeated, so a client applying them in
// order never renders a count the server was not holding at that moment.
func TestSnapPenaltyEventsReportSizesPerCard(t *testing.T) {
	stock := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankFive),
	}
	discard := []engine.Card{engine.NewCard(engine.SuitHearts, engine.RankSeven)}
	f := setupSnapPenaltyGame(t, stock, discard, engine.NewCard(engine.SuitSpades, engine.RankEight))
	beforeStock := int(f.game.Engine.StockLen)

	f.snap()

	var reported []int
	for _, ev := range f.broadcaster.allEvents {
		if ev.Type != EventPlayerSnapPenalty {
			continue
		}
		reported = append(reported, ev.Payload["stockpileSize"].(int))
	}
	require.Len(t, reported, 2)
	assert.Equal(t, []int{beforeStock - 1, beforeStock - 2}, reported, "each penalty event should report the stockpile as it stood after that card was drawn")
}
