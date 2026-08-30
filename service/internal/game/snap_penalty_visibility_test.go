// internal/game/snap_penalty_visibility_test.go: what a snap penalty is allowed to reveal
// (cambia-820).
package game

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSnapPenaltyRevealsNoCardFace pins the penalty draw as an unseen draw. doc/game_actions.md
// specifies private_snap_penalty as an id plus a hand slot: "Note that no card details are to be
// revealed, just the new cards." The client renders the slot face-down and the engine's agent state
// buckets penalty cards as unknown, so a face on the wire was information no consumer used but
// anyone reading the socket could. Both the public and the private event are checked, and the
// serialized JSON is checked alongside the struct so an omitempty tag cannot hide a regression.
func TestSnapPenaltyRevealsNoCardFace(t *testing.T) {
	stock := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankFive),
	}
	discard := []engine.Card{engine.NewCard(engine.SuitHearts, engine.RankSeven)}
	f := setupSnapPenaltyGame(t, stock, discard, engine.NewCard(engine.SuitSpades, engine.RankEight))

	f.snap()

	assertNoCardFace := func(t *testing.T, ev GameEvent, label string) {
		t.Helper()
		require.NotNilf(t, ev.Card, "%s should still name the drawn card", label)
		assert.NotEqualf(t, uuid.Nil, ev.Card.ID, "%s should carry the card id", label)
		assert.Emptyf(t, ev.Card.Rank, "%s must not reveal the card rank", label)
		assert.Emptyf(t, ev.Card.Suit, "%s must not reveal the card suit", label)
		assert.Zerof(t, ev.Card.Value, "%s must not reveal the card value", label)

		raw, err := json.Marshal(ev)
		require.NoError(t, err)
		for _, key := range []string{`"rank"`, `"suit"`, `"value"`} {
			assert.NotContainsf(t, string(raw), key, "%s serialized %s onto the wire: %s", label, key, string(raw))
		}
	}

	privateCount := 0
	for _, ev := range f.broadcaster.playerEvents[f.snapper.ID] {
		if ev.Type != EventPrivateSnapPenalty {
			continue
		}
		privateCount++
		require.NotNil(t, ev.Card.Idx, "the private event should still name the hand slot the card landed in")
		assertNoCardFace(t, ev, "private_snap_penalty")
	}
	assert.Equal(t, f.game.HouseRules.PenaltyDrawCount, privateCount, "one private event per penalty card")

	publicCount := 0
	for _, ev := range f.broadcaster.allEvents {
		if ev.Type != EventPlayerSnapPenalty {
			continue
		}
		publicCount++
		assertNoCardFace(t, ev, "player_snap_penalty")
	}
	assert.Equal(t, f.game.HouseRules.PenaltyDrawCount, publicCount, "one public event per penalty card")
}

// TestSnapPenaltyCardStaysUnseen verifies the seen-set agrees with the wire: a penalty card is
// never marked seen, so the penalized player's own sync_state renders it face-down like any other
// unknown card in their hand. This is what keeps the drawn-unseen rule intact against a client that
// reads the socket directly rather than the rendered table.
func TestSnapPenaltyCardStaysUnseen(t *testing.T) {
	stock := []engine.Card{
		engine.NewCard(engine.SuitHearts, engine.RankTwo),
		engine.NewCard(engine.SuitClubs, engine.RankThree),
		engine.NewCard(engine.SuitDiamonds, engine.RankFour),
		engine.NewCard(engine.SuitSpades, engine.RankFive),
	}
	discard := []engine.Card{engine.NewCard(engine.SuitHearts, engine.RankSeven)}
	f := setupSnapPenaltyGame(t, stock, discard, engine.NewCard(engine.SuitSpades, engine.RankEight))
	handBefore := int(f.game.Engine.Players[f.snapperIdx].HandLen)

	// Control: the card the snapper plays is one they have seen, so a hand rendered entirely
	// face-down cannot pass the face-down assertions below vacuously.
	seenSlot := handBefore - 1
	f.game.markCardSeen(f.snapperIdx, f.snapCardID)

	f.snap()

	obf := f.game.GetCurrentObfuscatedGameState(f.snapper.ID)
	var self *ObfPlayerState
	for i := range obf.Players {
		if obf.Players[i].PlayerID == f.snapper.ID {
			self = &obf.Players[i]
		}
	}
	require.NotNil(t, self, "the snapper should appear in its own sync state")
	require.Len(t, self.RevealedHand, handBefore+f.game.HouseRules.PenaltyDrawCount)

	// The control is the seen-set, not the wire: since cambia-1094 sync_state renders every own
	// card face-down whether or not its owner has seen it, so "face-down on the wire" is no longer
	// evidence about a single card. The played card is one the snapper has seen and it is face-down
	// too; what pins the penalty cards as unseen is their absence from SeenByPlayer.
	assert.True(t, f.game.hasSeenCard(f.snapperIdx, f.snapCardID), "control: the card the snapper played is one they had seen")
	assert.False(t, self.RevealedHand[seenSlot].Known, "no own card is ever face-up in sync_state (cambia-1094)")

	for slot := handBefore; slot < len(self.RevealedHand); slot++ {
		card := self.RevealedHand[slot]
		assert.Falsef(t, card.Known, "penalty card in slot %d should stay face-down for its owner", slot)
		assert.Emptyf(t, card.Rank, "penalty card in slot %d leaked a rank into sync_state", slot)
		assert.Falsef(t, f.game.hasSeenCard(f.snapperIdx, card.ID), "penalty card in slot %d was marked seen", slot)
	}
}
