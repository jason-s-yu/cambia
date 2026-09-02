// internal/hub/round_end_reveal_test.go
//
// The hub's half of the round-end reveal (RULES.md 3C, cambia-1542): a client that reconnects
// into the results is answered with the hub's held game_results and nothing else (the game is
// dropped from the store the moment that frame is emitted, see handlers.attachOnGameEnd), so the
// reveal has to ride that frame and the finished game's snapshot. A ranked round carries the same
// list on round_end, or on match_end where the round that just ended was the match's last.
package hub

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/service/internal/game"
)

// revealFrame is the slice of a results frame this file cares about.
type revealFrame struct {
	FinalHands []game.FinalHand `json:"finalHands"`
}

// syncFrame is the slice of a private_sync_state this file cares about: whether the hands it
// carries are face-up. The snapshot rides the event's `state` field, not `payload`.
type syncFrame struct {
	State struct {
		GameOver bool `json:"gameOver"`
		Players  []struct {
			PlayerID     uuid.UUID `json:"playerId"`
			RevealedHand []struct {
				Known bool   `json:"known"`
				Rank  string `json:"rank"`
			} `json:"revealedHand"`
		} `json:"players"`
	} `json:"state"`
}

func findEnvelope(envs []Envelope, typ string) *Envelope {
	for i := range envs {
		if envs[i].Type == typ {
			return &envs[i]
		}
	}
	return nil
}

// TestReconnectIntoPostGameGetsTheRoundEndReveal is the cambia-1542 reconnect case: a player who
// reloads into the results screen must be handed the revealed hands, not a face-down table. The
// hub's stored terminal frame is the only carrier, since game_end is long gone by then.
func TestReconnectIntoPostGameGetsTheRoundEndReveal(t *testing.T) {
	h, ids, g, _ := newInGameHubStopped(t, 2, false, 0, 0, 0)

	var reveal []game.FinalHand
	g.OnGameEnd = func(_ uuid.UUID, _ uuid.UUID, _ map[uuid.UUID]int, _ map[uuid.UUID]string, _ map[uuid.UUID]int, _ uuid.UUID, finalHands []game.FinalHand, _ game.EndReason) {
		reveal = finalHands
	}

	conn := newFakeConn(ids[0], "A")
	h.conns[ids[0]] = conn

	g.EndGame()
	require.Len(t, reveal, 2, "endGame must hand the callback every scored seat's hand")

	// What handlers.attachOnGameEnd broadcasts once the game ends. Emit records it as the hub's
	// terminal frame (rememberTerminal), which is what a later reconnect is answered with.
	h.Phase = PhasePostGame
	h.Emit("game_results", map[string]interface{}{
		"type":       "game_results",
		"winner":     ids[0].String(),
		"scores":     map[string]int{},
		"finalHands": reveal,
	})
	drainEnvelopes(t, conn)

	// The reload: a fresh connection for the same seat.
	rejoin := newFakeConn(ids[0], "A")
	h.conns[ids[0]] = rejoin
	h.notePlayerReconnected(ids[0])

	envs := drainEnvelopes(t, rejoin)

	results := findEnvelope(envs, "game_results")
	require.NotNil(t, results, "a reconnect into PhasePostGame must be re-sent the results frame")
	var got revealFrame
	require.NoError(t, json.Unmarshal(results.Payload, &got))
	assert.Equal(t, reveal, got.FinalHands, "the resent results must carry the same reveal the table saw")
	for _, hand := range got.FinalHands {
		require.NotEmpty(t, hand.Cards)
		for _, card := range hand.Cards {
			assert.NotEmpty(t, card.Rank, "every resent card names its rank")
		}
	}

	// The snapshot sent ahead of it carries the same faces, so the table under the results is
	// face-up too rather than a wall of card backs.
	sync := findEnvelope(envs, "private_sync_state")
	require.NotNil(t, sync, "a reconnect is sent the finished table's snapshot")
	var snapshot syncFrame
	require.NoError(t, json.Unmarshal(sync.Payload, &snapshot))
	require.True(t, snapshot.State.GameOver)
	require.Len(t, snapshot.State.Players, 2)
	for _, ps := range snapshot.State.Players {
		require.NotEmpty(t, ps.RevealedHand)
		for _, card := range ps.RevealedHand {
			assert.True(t, card.Known, "seat %s should be face-up in a finished round's snapshot", ps.PlayerID)
			assert.NotEmpty(t, card.Rank)
		}
	}
}

// TestRoundEndFrameCarriesTheReveal covers the ranked path: the round's reveal rides round_end
// while rounds remain, and match_end for the last one (which gets no round_end of its own, and is
// the frame a reconnect into a finished match is re-sent).
func TestRoundEndFrameCarriesTheReveal(t *testing.T) {
	idA := uuid.New()
	idB := uuid.New()
	reveal := []game.FinalHand{
		{PlayerID: idA, Cards: []game.FinalHandCard{{ID: uuid.New(), Idx: 0, Rank: "K", Suit: "H", Value: -1}}},
		{PlayerID: idB, Cards: []game.FinalHandCard{{ID: uuid.New(), Idx: 0, Rank: "9", Suit: "S", Value: 9}}},
	}
	scores := map[uuid.UUID]int{idA: -1, idB: 9}

	for _, tc := range []struct {
		name        string
		totalRounds int
		wantType    string
	}{
		{"a round with more to come", 3, "round_end"},
		{"the match's last round", 1, "match_end"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, _, _, _ := newInGameHubStopped(t, 2, false, 0, 0, 0)
			h.TotalRounds = tc.totalRounds
			conn := newFakeConn(idA, "A")
			h.conns[idA] = conn

			h.HandleRoundEnd(scores, idA, reveal)

			envs := drainEnvelopes(t, conn)
			frame := findEnvelope(envs, tc.wantType)
			require.NotNil(t, frame, "%s must be emitted", tc.wantType)
			var got revealFrame
			require.NoError(t, json.Unmarshal(frame.Payload, &got))
			assert.Equal(t, reveal, got.FinalHands, "%s must carry the round's reveal", tc.wantType)
		})
	}
}
