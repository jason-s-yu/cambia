// internal/game/initial_peek_test.go
package game

import (
	"fmt"
	"testing"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// beginPreGameWithRules builds a two-player game on the given house rules and runs the pre-game
// reveal only, leaving every event emitted along the way in the broadcaster. setupTestGame cannot
// be used here: it runs StartGame and then clears the broadcaster, discarding the very events the
// pregame peek tests inspect.
func beginPreGameWithRules(t *testing.T, rules HouseRules) (*CambiaGame, []uuid.UUID, *mockBroadcaster) {
	t.Helper()

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	g.HouseRules = rules
	g.TurnDuration = 0

	ids := make([]uuid.UUID, 2)
	for i := range ids {
		ids[i] = uuid.New()
		g.AddPlayer(&models.Player{
			ID:        ids[i],
			Connected: true,
			User:      &models.User{ID: ids[i], Username: fmt.Sprintf("P%d", i)},
		})
	}

	g.BeginPreGame()
	require.True(t, g.PreGameActive, "pre-game should be active after BeginPreGame")
	return g, ids, mb
}

// lastPlayerInitialCards returns the most recent private_initial_cards event captured for playerID.
func lastPlayerInitialCards(mb *mockBroadcaster, playerID uuid.UUID) *GameEvent {
	events := mb.playerEvents[playerID]
	var last *GameEvent
	for i := range events {
		if events[i].Type == EventPrivateInitialCards {
			last = &events[i]
		}
	}
	return last
}

// TestPrivateInitialCardsCarriesEveryPeekedSlot verifies that the pregame reveal ships one entry
// per peeked slot regardless of how many the initialViewCount house rule asks for. The event used
// to carry a fixed card1/card2 pair, so a three- or four-card peek silently lost everything past
// the second slot even though the engine peeked (and the owner was entitled to see) all of them
// (cambia-817). 0 and 2 are covered alongside 3 and 4 to pin the unchanged cases.
func TestPrivateInitialCardsCarriesEveryPeekedSlot(t *testing.T) {
	for _, viewCount := range []int{0, 1, 2, 3, 4} {
		t.Run(fmt.Sprintf("initialViewCount=%d", viewCount), func(t *testing.T) {
			rules := DefaultHouseRules()
			rules.TurnTimerSec = 0
			rules.InitialViewCount = viewCount
			g, ids, mb := beginPreGameWithRules(t, rules)

			for _, id := range ids {
				ev := lastPlayerInitialCards(mb, id)
				require.NotNilf(t, ev, "player %s should receive a private_initial_cards event", id)
				require.Lenf(t, ev.Cards, viewCount, "player %s: reveal should carry one card per peeked slot", id)

				engineIdx := g.PlayerToEngine[id]
				for i, card := range ev.Cards {
					require.NotNilf(t, card, "player %s: card %d should be populated", id, i)
					require.NotNilf(t, card.Idx, "player %s: card %d should carry its hand index", id, i)

					slot := uint8(*card.Idx)
					require.Lessf(t, slot, g.Engine.Players[engineIdx].HandLen, "peeked slot out of hand range")

					want := g.Engine.Players[engineIdx].Hand[slot]
					assert.Equal(t, g.CardTracker.Players[engineIdx].HandUUIDs[slot], card.ID, "revealed card id should match the tracked hand slot")
					assert.Equal(t, engineRankToString(want.Rank()), card.Rank, "revealed rank should match the engine hand")
					assert.Equal(t, engineSuitToString(want.Suit()), card.Suit, "revealed suit should match the engine hand")
					assert.Equal(t, int(want.Value()), card.Value, "revealed value should match the engine hand")
				}
			}
		})
	}
}

// TestPrivateInitialCardsPeekMarksExactlyThoseSlotsSeen verifies the reveal and the server's
// seen-set agree past the old two-card wire limit: with a peek wider than two, every peeked slot
// (and only those) must come back Known in the owner's own sync_state, so the client renders the
// same cards the event revealed rather than a hand the two of them disagree about.
func TestPrivateInitialCardsPeekMarksExactlyThoseSlotsSeen(t *testing.T) {
	rules := DefaultHouseRules()
	rules.TurnTimerSec = 0
	rules.CardsPerPlayer = 4
	rules.InitialViewCount = 3
	g, ids, mb := beginPreGameWithRules(t, rules)

	for _, id := range ids {
		ev := lastPlayerInitialCards(mb, id)
		require.NotNilf(t, ev, "player %s should receive a private_initial_cards event", id)
		require.Len(t, ev.Cards, 3)

		revealed := make(map[int]bool, len(ev.Cards))
		for _, card := range ev.Cards {
			require.NotNil(t, card.Idx)
			revealed[*card.Idx] = true
		}

		obf := g.GetCurrentObfuscatedGameState(id)
		var self *ObfPlayerState
		for i := range obf.Players {
			if obf.Players[i].PlayerID == id {
				self = &obf.Players[i]
			}
		}
		require.NotNilf(t, self, "player %s should appear in its own sync state", id)
		require.Len(t, self.RevealedHand, 4)

		for slot, card := range self.RevealedHand {
			assert.Equalf(t, revealed[slot], card.Known, "slot %d: sync_state visibility should match the pregame reveal", slot)
		}
	}
	_ = mb
}

// TestInitialViewCountBound verifies the lobby-edge bound on initialViewCount. The ceiling used to
// be a flat 2 because the wire shape could not express more; it is now the hand size, enforced
// against cardsPerPlayer in the same update so both keys can move together (cambia-817).
func TestInitialViewCountBound(t *testing.T) {
	tests := []struct {
		name    string
		update  map[string]interface{}
		wantErr bool
		want    int
	}{
		{name: "zero", update: map[string]interface{}{"initialViewCount": float64(0)}, want: 0},
		{name: "two unchanged", update: map[string]interface{}{"initialViewCount": float64(2)}, want: 2},
		{name: "three now accepted", update: map[string]interface{}{"initialViewCount": float64(3)}, want: 3},
		{name: "whole hand accepted", update: map[string]interface{}{"initialViewCount": float64(4)}, want: 4},
		{name: "above cards per player rejected", update: map[string]interface{}{"initialViewCount": float64(5)}, wantErr: true},
		{name: "above max hand size rejected", update: map[string]interface{}{"initialViewCount": float64(7)}, wantErr: true},
		{name: "negative rejected", update: map[string]interface{}{"initialViewCount": float64(-1)}, wantErr: true},
		{
			name:   "raised together with cards per player",
			update: map[string]interface{}{"cardsPerPlayer": float64(6), "initialViewCount": float64(6)},
			want:   6,
		},
		{
			name:    "lowering cards per player below the peek rejected",
			update:  map[string]interface{}{"cardsPerPlayer": float64(2), "initialViewCount": float64(4)},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rules := DefaultHouseRules() // cardsPerPlayer 4, initialViewCount 2
			err := rules.Update(tc.update)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, rules.InitialViewCount)
		})
	}
}

// TestInitialViewCountRejectedWhenHandShrinks verifies the cross-field check also fires when only
// cardsPerPlayer moves: a lobby peeking 4 cards that drops to a 2-card deal is rejected rather than
// silently played as a 2-card peek by the engine's own clamp.
func TestInitialViewCountRejectedWhenHandShrinks(t *testing.T) {
	rules := DefaultHouseRules()
	require.NoError(t, rules.Update(map[string]interface{}{"initialViewCount": float64(4)}))

	err := rules.Update(map[string]interface{}{"cardsPerPlayer": float64(2)})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "initialViewCount")
}
