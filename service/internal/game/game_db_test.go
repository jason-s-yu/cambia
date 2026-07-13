// internal/game/game_db_test.go
package game

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/require"
)

// dbAvailable reports whether a Postgres instance matching this package's standard connection
// env vars (PG_HOST, PG_PORT, POSTGRES_USER, POSTGRES_PASSWORD, PG_DATABASE - the same ones
// database.ConnectDB reads, see service/.env.template) is reachable. DB-dependent tests check
// this flag and skip cleanly on machines without a running dev database. Mirrors
// internal/handlers/main_test.go and internal/database/game_test.go.
var dbAvailable bool

func TestMain(m *testing.M) {
	dbAvailable = pingTestDB()
	os.Exit(m.Run())
}

// pingTestDB attempts a short-timeout connection to the database configured via the package's
// standard env vars. An unreachable DB is an expected condition on dev machines; callers use
// the returned bool to skip DB-dependent tests.
func pingTestDB() bool {
	connStr := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s",
		os.Getenv("POSTGRES_USER"),
		os.Getenv("POSTGRES_PASSWORD"),
		os.Getenv("PG_HOST"),
		os.Getenv("PG_PORT"),
		os.Getenv("PG_DATABASE"),
	)

	config, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return false
	}
	defer pool.Close()

	return pool.Ping(ctx) == nil
}

// setupGameDBTest skips the test when no dev Postgres is reachable, otherwise connects.
func setupGameDBTest(t *testing.T) {
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template); set these to point at a running dev database to run this test")
	}
	database.ConnectDB()
}

// createGameDBTestUser inserts a bare user (unique random username, no email) for use as a
// rating-update participant.
func createGameDBTestUser(t *testing.T, uname string) models.User {
	u := models.User{Username: uname}
	err := database.CreateUser(context.Background(), &u)
	require.NoError(t, err, "CreateUser failed")
	return u
}

// waitForGameStatus polls games.status until it matches want or the timeout elapses.
// EndGame/BeginPreGame dispatch their database writes via fire-and-forget goroutines (matching
// the existing persistInitialGameState/persistFinalGameState pattern), so tests observing their
// effect must poll rather than assert immediately after the triggering call returns.
func waitForGameStatus(t *testing.T, gameID uuid.UUID, want string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var status string
	for time.Now().Before(deadline) {
		err := database.DB.QueryRow(context.Background(), `SELECT status FROM games WHERE id = $1`, gameID).Scan(&status)
		if err == nil && status == want {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for games.status = %q for game %v (last seen: %q)", want, gameID, status)
}

// TestEndGameRecordsResultsAndRating drives a real 2-player game through EndGame() (not a
// direct RecordGameAndResults call, unlike internal/database/game_test.go) and asserts the
// production wiring persists game_results and updates ratings. Before cambia-450,
// RecordGameAndResults had zero production callers, so real games never updated ratings despite
// the pool-aware persistence logic added in cambia-381; this test exercises the actual
// game-completion path (BeginPreGame -> play -> EndGame) to catch a regression in that wiring,
// not just in RecordGameAndResults itself.
func TestEndGameRecordsResultsAndRating(t *testing.T) {
	setupGameDBTest(t)

	userA := createGameDBTestUser(t, "endgame-a-"+uuid.NewString())
	userB := createGameDBTestUser(t, "endgame-b-"+uuid.NewString())

	g := NewCambiaGame()
	mb := newMockBroadcaster()
	g.Emitter = mb
	g.LobbyID = uuid.New()
	g.HostUserID = userA.ID
	g.LobbyType = "private"
	g.Rated = true
	g.HouseRules = HouseRules{TurnTimerSec: 0, PenaltyDrawCount: 2}
	g.TurnDuration = 0

	playerA := &models.Player{ID: userA.ID, Connected: true, User: &models.User{ID: userA.ID}}
	playerB := &models.Player{ID: userB.ID, Connected: true, User: &models.User{ID: userB.ID}}
	g.AddPlayer(playerA)
	g.AddPlayer(playerB)

	g.BeginPreGame()
	require.True(t, g.PreGameActive, "PreGame should be active after BeginPreGame")
	g.StartGame()
	require.True(t, g.Started, "game should be marked as started")

	// The games row (with lobby_id satisfying the FK to a stub lobbies row) is created
	// asynchronously from persistInitialGameState; wait for it before proceeding so
	// RecordGameAndResults' completion UPDATE at game-end has a row to match.
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

	waitForGameStatus(t, g.ID, "completed", 2*time.Second)

	rows, err := database.DB.Query(context.Background(), `SELECT player_id, did_win FROM game_results WHERE game_id = $1`, g.ID)
	require.NoError(t, err)
	seen := map[uuid.UUID]bool{}
	for rows.Next() {
		var pid uuid.UUID
		var didWin bool
		require.NoError(t, rows.Scan(&pid, &didWin))
		seen[pid] = true
	}
	rows.Close()
	require.True(t, seen[userA.ID], "player A should have a game_results row")
	require.True(t, seen[userB.ID], "player B should have a game_results row")

	// Rating: a rated 2p game should move elo_1v1 off the 1500 default for at least one
	// player (cannot assert direction here without recomputing Glicko-2; the point is that
	// EndGame actually reached the rating step, not RecordGameAndResults' math).
	afterA, err := database.GetUserByID(context.Background(), userA.ID)
	require.NoError(t, err)
	afterB, err := database.GetUserByID(context.Background(), userB.ID)
	require.NoError(t, err)
	require.False(t, afterA.Elo1v1 == 1500 && afterB.Elo1v1 == 1500,
		"expected elo_1v1 to move for at least one player after a rated 2p game reached via EndGame()")
}
