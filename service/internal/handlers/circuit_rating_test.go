// internal/handlers/circuit_rating_test.go
package handlers

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/game"
	"github.com/stretchr/testify/require"
)

// TestCircuitCumulativeScoresMapsStandingsOntoPlayers covers the translation the circuit's single
// rating update runs on: engine-side standings (engine player ids, cumulative totals with
// subsidies already applied) become the service-side roster and score map the rating path reads.
// Roster order follows the standings, so the update is deterministic rather than map-ordered.
func TestCircuitCumulativeScoresMapsStandingsOntoPlayers(t *testing.T) {
	alice, bob, carol := uuid.New(), uuid.New(), uuid.New()
	playerMap := map[uuid.UUID]int{alice: 0, bob: 1, carol: 2}
	standings := []engine.CircuitPlayerState{
		{PlayerID: 2, CumulativeScore: 38},
		{PlayerID: 0, CumulativeScore: 44},
		{PlayerID: 1, CumulativeScore: 61},
	}

	roster, scores := circuitCumulativeScores(standings, playerMap)

	require.Equal(t, []uuid.UUID{carol, alice, bob}, roster, "roster follows the standings order, best cumulative first")
	require.Equal(t, map[uuid.UUID]int{carol: 38, alice: 44, bob: 61}, scores)
}

// TestCircuitCumulativeScoresDropsUnmappedStandings guards the mismatch case: a standings entry
// with no service-side player (a roster that changed under the circuit) is dropped rather than
// contributing a nil-UUID entry to the rating update.
func TestCircuitCumulativeScoresDropsUnmappedStandings(t *testing.T) {
	alice := uuid.New()
	roster, scores := circuitCumulativeScores(
		[]engine.CircuitPlayerState{{PlayerID: 0, CumulativeScore: 40}, {PlayerID: 7, CumulativeScore: 50}},
		map[uuid.UUID]int{alice: 0},
	)

	require.Equal(t, []uuid.UUID{alice}, roster)
	require.Equal(t, map[uuid.UUID]int{alice: 40}, scores)
}

// seedCircuitGameRow inserts the lobbies and games rows a ratings row's game_id FK needs, for a
// test that writes ratings without driving a game through the normal start path. Mirrors
// internal/database/game_test.go's seedGameRow.
func seedCircuitGameRow(t *testing.T, gameID, hostUserID uuid.UUID) {
	t.Helper()
	ctx := context.Background()
	var lobbyID uuid.UUID
	require.NoError(t, database.DB.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type) VALUES ($1, 'private') RETURNING id`,
		hostUserID,
	).Scan(&lobbyID))
	_, err := database.DB.Exec(ctx,
		`INSERT INTO games (id, lobby_id, status) VALUES ($1, $2, 'completed')`,
		gameID, lobbyID,
	)
	require.NoError(t, err)
}

// circuitRatingRows returns the (user_id -> new_rating) ratings rows written for a game.
func circuitRatingRows(t *testing.T, gameID uuid.UUID) map[uuid.UUID]int {
	t.Helper()
	rows, err := database.DB.Query(context.Background(),
		`SELECT user_id, new_rating FROM ratings WHERE game_id = $1`, gameID)
	require.NoError(t, err)
	defer rows.Close()
	out := map[uuid.UUID]int{}
	for rows.Next() {
		var uid uuid.UUID
		var newRating int
		require.NoError(t, rows.Scan(&uid, &newRating))
		out[uid] = newRating
	}
	return out
}

// newCircuitRatingGame builds a minimal circuit game for the finalization helper: it reads the
// game's id (what the ratings rows are attributed to) and its rated flag, nothing else.
func newCircuitRatingGame(rated bool) *game.CambiaGame {
	g := game.NewCambiaGame()
	g.Rated = rated
	g.Circuit = game.Circuit{Enabled: true, Mode: "quick"}
	return g
}

// TestFinalizeCircuitRatingsWritesOneUpdateFromCumulativeTotals is the wiring half of the circuit
// cadence (RULES.md T6, MATCHMAKING.md 6.2/6.3): at a circuit's conclusion exactly one rating
// update is written, computed from the final cumulative standings rather than any single round.
// The rounds themselves rate nothing (game.TestCircuitRoundPersistsResultsWithoutRating).
func TestFinalizeCircuitRatingsWritesOneUpdateFromCumulativeTotals(t *testing.T) {
	ensureTestDB(t)

	winner := createTestUser(t, "circuit-fin-w-"+uuid.NewString()+"@example.com", "pw", "circuit-fin-w-"+uuid.NewString())
	loser := createTestUser(t, "circuit-fin-l-"+uuid.NewString()+"@example.com", "pw", "circuit-fin-l-"+uuid.NewString())

	gs := NewGameServer()
	gs.PersistWG = &sync.WaitGroup{}

	g := newCircuitRatingGame(true)
	seedCircuitGameRow(t, g.ID, winner.ID)

	playerMap := map[uuid.UUID]int{winner.ID: 0, loser.ID: 1}
	standings := []engine.CircuitPlayerState{
		{PlayerID: 0, CumulativeScore: 120},
		{PlayerID: 1, CumulativeScore: 155},
	}

	gs.finalizeCircuitRatings(g, standings, playerMap)
	awaitCircuitPersistence(t, gs)

	written := circuitRatingRows(t, g.ID)
	require.Len(t, written, 2, "one ratings row per player: a circuit rates exactly once, at its conclusion")
	require.Greater(t, written[winner.ID], 1500, "the lower cumulative total wins the circuit")
	require.Less(t, written[loser.ID], 1500, "the higher cumulative total loses it")
}

// TestFinalizeCircuitRatingsTiesFinishersWithinThreeCumulativePoints holds the 3-point band at
// the wiring level: it is the cumulative totals it applies to, not any round's scores.
func TestFinalizeCircuitRatingsTiesFinishersWithinThreeCumulativePoints(t *testing.T) {
	ensureTestDB(t)

	first := createTestUser(t, "circuit-tie-1-"+uuid.NewString()+"@example.com", "pw", "circuit-tie-1-"+uuid.NewString())
	second := createTestUser(t, "circuit-tie-2-"+uuid.NewString()+"@example.com", "pw", "circuit-tie-2-"+uuid.NewString())

	gs := NewGameServer()
	gs.PersistWG = &sync.WaitGroup{}

	g := newCircuitRatingGame(true)
	seedCircuitGameRow(t, g.ID, first.ID)

	playerMap := map[uuid.UUID]int{first.ID: 0, second.ID: 1}
	standings := []engine.CircuitPlayerState{
		{PlayerID: 0, CumulativeScore: 118},
		{PlayerID: 1, CumulativeScore: 120},
	}

	gs.finalizeCircuitRatings(g, standings, playerMap)
	awaitCircuitPersistence(t, gs)

	written := circuitRatingRows(t, g.ID)
	require.Len(t, written, 2)
	require.Equal(t, 1500, written[first.ID], "a 2-point cumulative margin is a tie, so no rating changes hands")
	require.Equal(t, 1500, written[second.ID], "a 2-point cumulative margin is a tie, so no rating changes hands")
}

// TestFinalizeCircuitRatingsSkipsUnratedCircuits keeps the unrated case unrated: a casual circuit
// concludes without touching the rating tables at all.
func TestFinalizeCircuitRatingsSkipsUnratedCircuits(t *testing.T) {
	ensureTestDB(t)

	playerA := createTestUser(t, "circuit-unrated-a-"+uuid.NewString()+"@example.com", "pw", "circuit-unrated-a-"+uuid.NewString())
	playerB := createTestUser(t, "circuit-unrated-b-"+uuid.NewString()+"@example.com", "pw", "circuit-unrated-b-"+uuid.NewString())

	gs := NewGameServer()
	gs.PersistWG = &sync.WaitGroup{}

	g := newCircuitRatingGame(false)
	seedCircuitGameRow(t, g.ID, playerA.ID)

	gs.finalizeCircuitRatings(g,
		[]engine.CircuitPlayerState{{PlayerID: 0, CumulativeScore: 100}, {PlayerID: 1, CumulativeScore: 160}},
		map[uuid.UUID]int{playerA.ID: 0, playerB.ID: 1},
	)
	awaitCircuitPersistence(t, gs)

	require.Empty(t, circuitRatingRows(t, g.ID), "an unrated circuit writes no ratings rows")
	after, err := database.GetUserByID(context.Background(), playerA.ID)
	require.NoError(t, err)
	require.Equal(t, 1500, after.Elo1v1)
	require.Equal(t, 350.0, after.Phi1v1)
}

// awaitCircuitPersistence drains the background write finalizeCircuitRatings registers on the
// server's PersistWG. The Add runs synchronously inside the helper (finalizeCircuitRatings is
// called from OnGameEnd, under the game mutex, and must not block there), so a Wait ordered after
// the call is safe under sync.WaitGroup's Add-before-Wait rule; see awaitGameEndPersistence for
// the same argument applied to the game-end writes.
func awaitCircuitPersistence(t *testing.T, gs *GameServer) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		gs.PersistWG.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("circuit rating write did not finish within 5s")
	}
}
