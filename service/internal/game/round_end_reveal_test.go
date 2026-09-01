// internal/game/round_end_reveal_test.go
//
// The round-end reveal (RULES.md 3C, cambia-1542). Before this, no frame carried a hand face when
// a round ended: game_end carried scores and a winner, the snapshot marked every slot
// Known:false in every phase (cambia-1094), and the only serialization of the final hands went
// into games.final_game_state, which nothing reads. A finished table therefore sat entirely
// face-down under the results.
package game

import (
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// engineHandOf reads a seat's hand straight off the engine as the round left it, which is the
// truth the reveal has to reproduce.
func engineHandOf(g *CambiaGame, playerID uuid.UUID) []FinalHandCard {
	idx := g.PlayerToEngine[playerID]
	handLen := g.Engine.Players[idx].HandLen
	out := make([]FinalHandCard, handLen)
	for j := uint8(0); j < handLen; j++ {
		card := g.Engine.Players[idx].Hand[j]
		out[j] = FinalHandCard{
			ID:    g.CardTracker.Players[idx].HandUUIDs[j],
			Idx:   int(j),
			Rank:  engineRankToString(card.Rank()),
			Suit:  engineSuitToString(card.Suit()),
			Value: int(card.Value()),
		}
	}
	return out
}

// driveToCambiaEnd plays a 2-seat table to a Cambia end: one plain turn, the call, then the one
// final turn the caller's opponent is owed (RULES.md 3C).
func driveToCambiaEnd(t *testing.T, g *CambiaGame, players []*models.Player) {
	t.Helper()

	first := currentTurnPlayer(g)
	firstP, secondP := players[0], players[1]
	if first.ID != players[0].ID {
		firstP, secondP = players[1], players[0]
	}

	playSimpleTurn(t, g, firstP)
	g.HandlePlayerAction(secondP.ID, models.GameAction{ActionType: "action_cambia"})
	playSimpleTurn(t, g, firstP)

	require.True(t, g.GameOver, "the table should be over once the caller's opponent has taken their final turn")
	require.True(t, g.Engine.IsCambiaCalled(), "the round should have ended on a Cambia call")
}

// TestGameEndRevealsEveryScoredHand is the headline case: a real 2-seat game played to a Cambia
// end must emit every scored card's rank on the game_end frame, matching the engine hands the
// same endGame scored.
func TestGameEndRevealsEveryScoredHand(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	// The pre-terminal snapshot is the cambia-1094 rule, unchanged: not one face, own or opponent.
	for _, viewer := range players {
		obf := g.GetCurrentObfuscatedGameState(viewer.ID)
		for _, ps := range obf.Players {
			require.NotEmpty(t, ps.RevealedHand, "every seat should carry its hand slots mid-round")
			for _, card := range ps.RevealedHand {
				assert.False(t, card.Known, "no hand card is face-up before the round ends (cambia-1094)")
				assert.Empty(t, card.Rank, "a face-down slot carries no rank")
			}
		}
	}

	driveToCambiaEnd(t, g, players)

	ev := mb.findEventByType(EventGameEnd)
	require.NotNil(t, ev, "game_end must be broadcast when the round ends")
	payload := ev.Payload
	hands, ok := payload["finalHands"].([]FinalHand)
	require.True(t, ok, "game_end must carry finalHands (RULES.md 3C)")
	require.Len(t, hands, len(players), "every seat that was scored must appear in the reveal")

	scores := payload["scores"].(map[string]int)
	for _, hand := range hands {
		want := engineHandOf(g, hand.PlayerID)
		require.NotEmpty(t, want, "a scored seat holds cards at the end of a round")
		// Every scored card is named by rank, suit, value, id and slot.
		assert.Equal(t, want, hand.Cards, "the reveal must name the hand the engine scored, card for card")

		// The revealed values must add up to the score the same frame reports, before the Cambia
		// penalty (the caller's adjusted score can carry +1 on a failed call).
		total := 0
		for _, c := range hand.Cards {
			assert.NotEmpty(t, c.Rank, "every revealed card names a rank")
			assert.NotEqual(t, uuid.Nil, c.ID, "every revealed card carries the id its events named it by")
			total += c.Value
		}
		reported := scores[hand.PlayerID.String()]
		assert.True(t, reported == total || reported == total+1,
			"seat %s: revealed values sum to %d, frame reports %d", hand.PlayerID, total, reported)
	}
}

// TestSyncStateRevealsHandsOnceGameOver covers the snapshot half: the same faces reach a client
// that resyncs or reconnects into a finished round, for its own hand as well as its opponent's.
func TestSyncStateRevealsHandsOnceGameOver(t *testing.T) {
	g, players, _ := setupTestGame(t, 2, testHouseRules(0, 2))
	driveToCambiaEnd(t, g, players)

	for _, viewer := range players {
		obf := g.GetCurrentObfuscatedGameState(viewer.ID)
		require.True(t, obf.GameOver, "the snapshot should report the round as over")
		for _, ps := range obf.Players {
			want := engineHandOf(g, ps.PlayerID)
			require.Len(t, ps.RevealedHand, len(want), "seat %s hand size", ps.PlayerID)
			for j, card := range ps.RevealedHand {
				assert.True(t, card.Known, "seat %s slot %d must be face-up once the round is over", ps.PlayerID, j)
				assert.Equal(t, want[j].Rank, card.Rank, "seat %s slot %d rank", ps.PlayerID, j)
				assert.Equal(t, want[j].Suit, card.Suit, "seat %s slot %d suit", ps.PlayerID, j)
				assert.Equal(t, want[j].Value, card.Value, "seat %s slot %d value", ps.PlayerID, j)
				assert.Equal(t, want[j].ID, card.ID, "seat %s slot %d id", ps.PlayerID, j)
				require.NotNil(t, card.Idx, "a hand slot keeps its index")
				assert.Equal(t, j, *card.Idx, "seat %s slot %d index", ps.PlayerID, j)
			}
		}
	}
}

// TestTurnCapEndAlsoReveals pins the decision recorded against RULES.md 3C: the reveal is not
// specific to the Cambia path. A round the engine ends on its own terminal condition reveals the
// same way, because endGame is the one funnel every terminal path reaches.
func TestTurnCapEndAlsoReveals(t *testing.T) {
	g, players, mb := setupTestGame(t, 2, testHouseRules(0, 2))

	// End without a Cambia call: EndGame is the same entry the turn-cap and stock-exhaustion
	// paths take once the engine reports terminal (engine_adapter.go settleEngineResolution).
	g.EndGame()
	require.True(t, g.GameOver)
	require.False(t, g.Engine.IsCambiaCalled(), "this round ends with no call, so the reveal cannot be keyed off one")

	ev := mb.findEventByType(EventGameEnd)
	require.NotNil(t, ev, "game_end must be broadcast")
	hands := ev.Payload["finalHands"].([]FinalHand)
	require.Len(t, hands, len(players), "an uncalled end reveals every scored seat too")
	for _, hand := range hands {
		assert.Equal(t, engineHandOf(g, hand.PlayerID), hand.Cards)
	}
}

// TestForfeitedSeatIsNotRevealed holds the one carve-out: a forfeited seat is not scored
// (computeScoresFromEngine), so it takes no part in the result and stays face-down in both the
// frame and the snapshot. The two must agree, or the table would show a hand the results do not.
func TestForfeitedSeatIsNotRevealed(t *testing.T) {
	g, players, mb := setupTestGame(t, 3, testHouseRules(0, 2))

	gone := players[2]
	g.mu.Lock()
	g.forfeited[gone.ID] = true
	g.mu.Unlock()

	g.EndGame()
	require.True(t, g.GameOver)

	ev := mb.findEventByType(EventGameEnd)
	require.NotNil(t, ev)
	payload := ev.Payload
	hands := payload["finalHands"].([]FinalHand)
	require.Len(t, hands, 2, "the forfeited seat must be left out of the reveal")
	for _, hand := range hands {
		assert.NotEqual(t, gone.ID, hand.PlayerID, "the forfeited seat must not be revealed")
	}
	_, scored := payload["scores"].(map[string]int)[gone.ID.String()]
	assert.False(t, scored, "the forfeited seat is not scored either, which is why it is not revealed")

	obf := g.GetCurrentObfuscatedGameState(players[0].ID)
	for _, ps := range obf.Players {
		faceUp := ps.RevealedHand[0].Known
		if ps.PlayerID == gone.ID {
			assert.False(t, faceUp, "the forfeited seat stays face-down in the snapshot too")
		} else {
			assert.True(t, faceUp, "a scored seat is face-up in the snapshot")
		}
	}
}

// TestBuildFinalRevealCoversEveryEngineSeat guards the seat walk itself: it runs over engine
// indices, so the list order is seating order and no mapped seat is skipped.
func TestBuildFinalRevealCoversEveryEngineSeat(t *testing.T) {
	g, players, _ := setupTestGame(t, 4, testHouseRules(0, 2))

	g.mu.Lock()
	hands := g.buildFinalReveal()
	g.mu.Unlock()

	require.Len(t, hands, len(players))
	for i, hand := range hands {
		assert.Equal(t, g.EngineToPlayer[uint8(i)], hand.PlayerID, "reveal entry %d should be engine seat %d", i, i)
		assert.Len(t, hand.Cards, int(g.Engine.Players[uint8(i)].HandLen))
	}
	assert.LessOrEqual(t, len(hands), engine.MaxPlayers)
}
