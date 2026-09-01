// internal/game/forfeit_scoring_test.go
//
// What a forfeited seat is worth at game end (cambia-1541). The seat used to be left out of the
// score map entirely, and every consumer that indexed that map by player id read the miss as a 0:
// game_results stored a 0 and the rating sort, where lower is better, read the 0 as the best score
// at the table. Quitting a rated 2-seat game was therefore a rating gain, and the player who
// stayed took the loss. The seat now scores engine.ForfeitRoundScore, the 41 points
// MATCHMAKING.md 8 and RULES.md T5 put a forfeited round at.
package game

import (
	"strconv"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// endResult captures the arguments OnGameEnd is handed, which is where the two score maps a
// finished game produces become separately observable: rawScores is the full map every record
// path takes (game_results, the rating roster, a circuit's cumulative totals) and scores is the
// display map the results frame is built from.
type endResult struct {
	winner    uuid.UUID
	scores    map[uuid.UUID]int
	rawScores map[uuid.UUID]int
}

// captureGameEnd wires OnGameEnd to copy what it is given into the returned channel. The callback
// runs under the game mutex, so it copies rather than calling back into the game.
func captureGameEnd(g *CambiaGame) chan endResult {
	ended := make(chan endResult, 1)
	g.OnGameEnd = func(_ uuid.UUID, winner uuid.UUID, scores map[uuid.UUID]int, _ map[uuid.UUID]string, rawScores map[uuid.UUID]int, _ uuid.UUID, _ []FinalHand) {
		res := endResult{
			winner:    winner,
			scores:    make(map[uuid.UUID]int, len(scores)),
			rawScores: make(map[uuid.UUID]int, len(rawScores)),
		}
		for id, s := range scores {
			res.scores[id] = s
		}
		for id, s := range rawScores {
			res.rawScores[id] = s
		}
		select {
		case ended <- res:
		default:
		}
	}
	return ended
}

// setSeatHand fills a seat's whole engine hand with one rank so the seat's score is fixed rather
// than dealt, and returns that score. Scoring reads the engine hand, so this is what
// computeScoresFromEngine will add up.
func setSeatHand(t *testing.T, g *CambiaGame, playerID uuid.UUID, suit, rank uint8) int {
	t.Helper()

	seat, ok := g.PlayerToEngine[playerID]
	require.True(t, ok, "player %s holds no engine seat", playerID)

	card := engine.NewCard(suit, rank)
	total := 0
	for j := uint8(0); j < g.Engine.Players[seat].HandLen; j++ {
		g.Engine.Players[seat].Hand[j] = card
		total += int(card.Value())
	}
	require.Greater(t, total, 0, "the fixed hand must be worth something to compare against")
	return total
}

// awaitEnd reads the captured game-end result, failing the test if the game never ended.
func awaitEnd(t *testing.T, ended chan endResult) endResult {
	t.Helper()
	select {
	case res := <-ended:
		return res
	case <-time.After(2 * time.Second):
		t.Fatal("the game never ended")
		return endResult{}
	}
}

// TestForfeitInARatedTwoSeatGameScoresTheQuitterAtTheForfeitScore is the reported bug on the
// default rated path: h2h_quickplay is rated with the circuit off, so the game rates itself as it
// ends, and ForfeitOnDisconnect defaults on. The seat that walked away has to be scored, at the
// forfeit score, and must not come out ahead of the seat that stayed.
func TestForfeitInARatedTwoSeatGameScoresTheQuitterAtTheForfeitScore(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, false, true, 0, 30*time.Second, 0)
	g.Rated = true
	ended := captureGameEnd(g)

	stayer, quitter := players[0], players[1]
	stayerScore := setSeatHand(t, g, stayer.ID, engine.SuitClubs, engine.RankAce)
	// The quitter's hand is deliberately the cheaper one: the forfeit score has to replace what
	// they were holding, not be read off it.
	setSeatHand(t, g, quitter.ID, engine.SuitClubs, engine.RankAce)

	g.HandleDisconnect(quitter.ID)
	res := awaitEnd(t, ended)

	assert.Equal(t, engine.ForfeitRoundScore, res.rawScores[quitter.ID],
		"the forfeited seat must be scored at the forfeit score, not left out of the map")
	assert.Equal(t, stayerScore, res.rawScores[stayer.ID], "the seat that stayed is scored on its hand")
	assert.Greater(t, res.rawScores[quitter.ID], res.rawScores[stayer.ID],
		"lower is better: the forfeit must not outrank the hand that was played")
	assert.Equal(t, stayer.ID, res.winner, "the seat that stayed wins the forfeit")

	// The results frame stays a scoreboard of the seats that played (cambia-837/955): the
	// forfeited seat is reported by player_forfeited and rendered from the forfeited flag.
	assert.NotContains(t, res.scores, quitter.ID, "the results payload keeps omitting the forfeited seat")
	assert.Contains(t, res.scores, stayer.ID)
}

// TestVoluntaryForfeitScoresLikeAClosedReconnectWindow holds the two forfeits to one rule: giving
// the seat up on purpose (ForfeitSeat, cambia-1520) costs exactly what letting the reconnect
// window close costs, so leaving the table is never the cheaper exit.
func TestVoluntaryForfeitScoresLikeAClosedReconnectWindow(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, false, true, 60, 30*time.Second, 60*time.Second)
	g.Rated = true
	ended := captureGameEnd(g)

	stayer, quitter := players[0], players[1]
	stayerScore := setSeatHand(t, g, stayer.ID, engine.SuitClubs, engine.RankAce)
	setSeatHand(t, g, quitter.ID, engine.SuitClubs, engine.RankAce)

	require.True(t, g.ForfeitSeat(quitter.ID), "the seat must be given up on the spot")
	res := awaitEnd(t, ended)

	assert.Equal(t, engine.ForfeitRoundScore, res.rawScores[quitter.ID],
		"a voluntary forfeit costs the same as a closed reconnect window")
	assert.Equal(t, stayerScore, res.rawScores[stayer.ID])
	assert.Equal(t, stayer.ID, res.winner)
}

// TestFourSeatForfeitNeverOutranksASeatThatFinished is the forfeit that does not end the game:
// three seats are still there, so the table plays on and the forfeited seat is scored alongside
// them when it does end. The map asserted here is the one that becomes both the game_results rows
// and the rating roster's scores, so ordering in it is ordering in both.
func TestFourSeatForfeitNeverOutranksASeatThatFinished(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 4, false, true, 0, 30*time.Second, 0)
	g.Rated = true
	ended := captureGameEnd(g)

	quitter := players[3]
	finisherScores := map[uuid.UUID]int{
		players[0].ID: setSeatHand(t, g, players[0].ID, engine.SuitClubs, engine.RankAce),
		players[1].ID: setSeatHand(t, g, players[1].ID, engine.SuitClubs, engine.RankTwo),
		players[2].ID: setSeatHand(t, g, players[2].ID, engine.SuitClubs, engine.RankThree),
	}
	setSeatHand(t, g, quitter.ID, engine.SuitClubs, engine.RankAce)

	g.HandleDisconnect(quitter.ID)
	require.True(t, g.IsForfeited(quitter.ID), "the drop must forfeit the seat")
	require.False(t, g.GetCurrentObfuscatedGameState(players[0].ID).GameOver,
		"one forfeit of four leaves three seats playing, so the game continues")

	g.EndGame()
	res := awaitEnd(t, ended)

	require.Equal(t, engine.ForfeitRoundScore, res.rawScores[quitter.ID])
	for id, want := range finisherScores {
		assert.Equal(t, want, res.rawScores[id], "a seat that finished is scored on its hand")
		assert.Greater(t, res.rawScores[quitter.ID], res.rawScores[id],
			"the forfeited seat must never rank ahead of a seat that finished the round")
	}
	assert.NotEqual(t, quitter.ID, res.winner, "a forfeited seat cannot win the round")
	assert.NotContains(t, res.scores, quitter.ID, "the results payload keeps omitting the forfeited seat")
	assert.Len(t, res.scores, 3, "the results payload carries exactly the seats that played")
}

// TestForfeitScoreIsNotAdjustedByTheCambiaPenalty keeps the forfeit score flat. It is what a round
// nobody played is worth, not a hand, so the false-Cambia penalty has nothing to adjust; stacking
// it would make one seat's forfeit cost more than another's.
func TestForfeitScoreIsNotAdjustedByTheCambiaPenalty(t *testing.T) {
	g, players, _ := buildDropTestGame(t, 2, false, true, 60, 30*time.Second, 60*time.Second)
	ended := captureGameEnd(g)

	caller := currentTurnPlayer(g)
	require.NotNil(t, caller)
	other := otherPlayer(t, players, caller.ID)

	// The caller is left holding more than the other seat, so the call loses and the penalty
	// would apply, and then gives the seat up before the round is scored.
	setSeatHand(t, g, caller.ID, engine.SuitClubs, engine.RankAce)
	setSeatHand(t, g, other.ID, engine.SuitClubs, engine.RankAce)

	g.HandlePlayerAction(caller.ID, models.GameAction{ActionType: "action_cambia"})
	require.True(t, g.ForfeitSeat(caller.ID))
	res := awaitEnd(t, ended)

	assert.Equal(t, engine.ForfeitRoundScore, res.rawScores[caller.ID],
		"the forfeit score stands on its own, penalty or no penalty")
}

// TestEveryScoredSeatCarriesAScore is the contract the database layer now refuses to write
// without: a finished game hands persistence and rating one score per seated player, so nothing
// downstream has to decide what a missing key means.
func TestEveryScoredSeatCarriesAScore(t *testing.T) {
	for _, seats := range []int{2, 4} {
		t.Run(strconv.Itoa(seats)+"seat", func(t *testing.T) {
			g, players, _ := buildDropTestGame(t, seats, false, true, 60, 30*time.Second, 60*time.Second)
			ended := captureGameEnd(g)

			require.True(t, g.ForfeitSeat(players[seats-1].ID))
			if !g.GetCurrentObfuscatedGameState(players[0].ID).GameOver {
				g.EndGame()
			}
			res := awaitEnd(t, ended)

			require.Len(t, res.rawScores, seats, "every seated player must carry a score")
			for _, p := range players {
				_, ok := res.rawScores[p.ID]
				assert.True(t, ok, "seat %s has no score", p.ID)
			}
		})
	}
}
