// internal/game/forfeit_persist_db_test.go
//
// The persisted half of cambia-1541, driven through the production game-end path against a real
// database: what game_results stores for a forfeited seat and which way the rating moves. Skips
// cleanly where no dev Postgres is reachable (setupGameDBTest), the same gate the rest of this
// package's DB-backed tests use.
package game

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// TestForfeitPersistsTheForfeitScoreAndRatesTheQuitterDown is the reported bug end to end on the
// default rated path: a rated two-seat game with the forfeit rule on, one seat dropping and never
// coming back. game_results has to store the forfeit score for that seat rather than the 0 a
// missing map key used to write, and the rating has to move against the seat that left. Measured
// on two fresh 1500s before the fix: quitter 1662, stayer 1338.
func TestForfeitPersistsTheForfeitScoreAndRatesTheQuitterDown(t *testing.T) {
	setupGameDBTest(t)

	stayerUser := createGameDBTestUser(t, "forfeit-stayer-"+uuid.NewString())
	quitterUser := createGameDBTestUser(t, "forfeit-quitter-"+uuid.NewString())

	g := NewCambiaGame()
	g.Emitter = newMockBroadcaster()
	g.LobbyID = uuid.New()
	g.HostUserID = stayerUser.ID
	g.LobbyType = "private"
	g.Rated = true
	g.HouseRules = *testHouseRules(0, 2)
	// The forfeit rule with no window to wait out: the drop itself closes the seat, which is what
	// a reconnect window expiring does and what h2h_quickplay ships with once the 60s elapses.
	g.HouseRules.ForfeitOnDisconnect = true
	g.HouseRules.DisconnectGraceSec = 0
	g.TurnDuration = 0
	g.PersistWG = &sync.WaitGroup{}

	stayer := &models.Player{ID: stayerUser.ID, Connected: true, User: &models.User{ID: stayerUser.ID}}
	quitter := &models.Player{ID: quitterUser.ID, Connected: true, User: &models.User{ID: quitterUser.ID}}
	g.AddPlayer(stayer)
	g.AddPlayer(quitter)

	g.BeginPreGame()
	g.StartGame()
	require.True(t, g.Started, "the game should be started")
	g.DisconnectGrace = 0

	// The games row is written asynchronously; RecordGameAndResults' completion UPDATE needs it.
	waitForGameStatus(t, g.ID, "in_progress", 2*time.Second)

	stayerScore := setSeatHand(t, g, stayer.ID, engine.SuitClubs, engine.RankAce)
	setSeatHand(t, g, quitter.ID, engine.SuitClubs, engine.RankAce)

	g.HandleDisconnect(quitter.ID)
	require.True(t, g.IsForfeited(quitter.ID), "the drop must forfeit the seat")
	require.True(t, g.IsGameOver(), "one seat left connected ends a two-seat game")

	awaitPersistence(t, g.PersistWG, 5*time.Second)

	scores := map[uuid.UUID]int{}
	wins := map[uuid.UUID]bool{}
	rows, err := database.DB.Query(context.Background(),
		`SELECT player_id, score, did_win FROM game_results WHERE game_id = $1`, g.ID)
	require.NoError(t, err)
	for rows.Next() {
		var pid uuid.UUID
		var score int
		var didWin bool
		require.NoError(t, rows.Scan(&pid, &score, &didWin))
		scores[pid] = score
		wins[pid] = didWin
	}
	rows.Close()

	require.Contains(t, scores, quitterUser.ID, "the forfeited seat must still have a game_results row")
	assert.Equal(t, engine.ForfeitRoundScore, scores[quitterUser.ID],
		"the forfeited seat's persisted score must be the forfeit score, not the 0 a missing key wrote")
	assert.Equal(t, stayerScore, scores[stayerUser.ID], "the seat that stayed is recorded on its hand")
	assert.False(t, wins[quitterUser.ID], "a forfeited seat does not win the game")
	assert.True(t, wins[stayerUser.ID], "the seat that stayed wins the forfeit")

	afterStayer, err := database.GetUserByID(context.Background(), stayerUser.ID)
	require.NoError(t, err)
	afterQuitter, err := database.GetUserByID(context.Background(), quitterUser.ID)
	require.NoError(t, err)
	t.Logf("rating after the forfeit: stayer %d (from %d), quitter %d (from %d)",
		afterStayer.Elo1v1, stayerUser.Elo1v1, afterQuitter.Elo1v1, quitterUser.Elo1v1)

	assert.LessOrEqual(t, afterQuitter.Elo1v1, quitterUser.Elo1v1, "the seat that forfeited must not gain rating")
	assert.GreaterOrEqual(t, afterStayer.Elo1v1, stayerUser.Elo1v1, "the seat that stayed must not lose rating")
}
