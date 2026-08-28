// internal/game/circuit_rating_test.go
package game

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/require"
)

// TestRatePerGameExcludesCircuitRounds pins the cadence rule itself (RULES.md T6,
// MATCHMAKING.md 6.2/6.3): a rated game rates on its own result unless it is one round of a
// circuit, in which case the circuit's conclusion carries the single update instead. Unrated
// games rate either way, which is to say not at all.
func TestRatePerGameExcludesCircuitRounds(t *testing.T) {
	cases := []struct {
		name    string
		rated   bool
		circuit bool
		want    bool
	}{
		{"rated standalone game rates on its own result", true, false, true},
		{"rated circuit round defers to the circuit's single update", true, true, false},
		{"unrated standalone game never rates", false, false, false},
		{"unrated circuit round never rates", false, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewCambiaGame()
			g.Rated = tc.rated
			g.Circuit.Enabled = tc.circuit
			require.Equal(t, tc.want, g.ratePerGame())
		})
	}
}

// TestCircuitRoundPersistsResultsWithoutRating drives a real rated 2-player circuit round
// through EndGame() and asserts the split the cadence requires: the round is still recorded
// (games row completed, game_results rows written, results broadcast unchanged), but no rating
// moves and no ratings row is written. The circuit's own conclusion is what rates, once, from
// cumulative totals (handlers.finalizeCircuitRatings -> database.RecordCircuitRatings).
//
// The counterpart for a non-circuit rated game is TestEndGameRecordsResultsAndRating, which must
// keep passing: this change narrows the per-game rating path, it does not remove it.
func TestCircuitRoundPersistsResultsWithoutRating(t *testing.T) {
	setupGameDBTest(t)

	userA := createGameDBTestUser(t, "circuit-round-a-"+uuid.NewString())
	userB := createGameDBTestUser(t, "circuit-round-b-"+uuid.NewString())

	g := NewCambiaGame()
	g.Emitter = newMockBroadcaster()
	g.LobbyID = uuid.New()
	g.HostUserID = userA.ID
	g.LobbyType = "private"
	g.Rated = true
	g.Circuit = Circuit{Enabled: true, Mode: "quick", Rules: CircuitRules{WinBonus: -1, FalseCambiaPenalty: 2}}
	g.HouseRules = *testHouseRules(0, 2)
	g.TurnDuration = 0
	g.PersistWG = &sync.WaitGroup{}

	playerA := &models.Player{ID: userA.ID, Connected: true, User: &models.User{ID: userA.ID}}
	playerB := &models.Player{ID: userB.ID, Connected: true, User: &models.User{ID: userB.ID}}
	g.AddPlayer(playerA)
	g.AddPlayer(playerB)

	g.BeginPreGame()
	g.StartGame()
	waitForGameStatus(t, g.ID, "in_progress", 2*time.Second)

	first := currentTurnPlayer(g)
	var firstP, secondP *models.Player
	if first.ID == playerA.ID {
		firstP, secondP = playerA, playerB
	} else {
		firstP, secondP = playerB, playerA
	}

	doSimpleTurn := func(player *models.Player) {
		g.HandlePlayerAction(player.ID, models.GameAction{ActionType: "action_draw_stockpile"})
		engineIdx := g.PlayerToEngine[player.ID]
		drawnUUID := g.CardTracker.Players[engineIdx].DrawnCardUUID
		if drawnUUID != uuid.Nil {
			g.HandlePlayerAction(player.ID, models.GameAction{
				ActionType: "action_discard",
				Payload:    map[string]interface{}{"id": drawnUUID.String()},
			})
			if g.SpecialAction.Active && g.SpecialAction.PlayerID == player.ID {
				g.ProcessSpecialAction(player.ID, "skip", nil, nil)
			}
		}
	}

	doSimpleTurn(firstP)
	g.HandlePlayerAction(secondP.ID, models.GameAction{ActionType: "action_cambia"})
	doSimpleTurn(firstP)

	require.True(t, g.GameOver, "game should be over after the final turn")
	awaitPersistence(t, g.PersistWG, 5*time.Second)

	ctx := context.Background()

	var resultRows int
	require.NoError(t, database.DB.QueryRow(ctx,
		`SELECT count(*) FROM game_results WHERE game_id = $1`, g.ID).Scan(&resultRows))
	require.Equal(t, 2, resultRows, "a circuit round still records its results: only the rating update is deferred")

	var ratingRows int
	require.NoError(t, database.DB.QueryRow(ctx,
		`SELECT count(*) FROM ratings WHERE game_id = $1`, g.ID).Scan(&ratingRows))
	require.Zero(t, ratingRows, "a circuit round must not write a ratings row: the circuit rates once, at its conclusion")

	afterA, err := database.GetUserByID(ctx, userA.ID)
	require.NoError(t, err)
	afterB, err := database.GetUserByID(ctx, userB.ID)
	require.NoError(t, err)
	require.Equal(t, 1500, afterA.Elo1v1, "no rating moves on a circuit round")
	require.Equal(t, 1500, afterB.Elo1v1, "no rating moves on a circuit round")
	require.Equal(t, 350.0, afterA.Phi1v1, "no rating deviation moves on a circuit round either")
	require.Equal(t, 350.0, afterB.Phi1v1, "no rating deviation moves on a circuit round either")
}
