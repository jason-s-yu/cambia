// internal/database/game_test.go
package database

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jason-s-yu/cambia/service/internal/models"
	_ "github.com/joho/godotenv/autoload" // Load .env for database connection.
	"github.com/stretchr/testify/require"
)

// dbAvailable reports whether a Postgres instance matching this package's standard
// connection env vars (PG_HOST, PG_PORT, POSTGRES_USER, POSTGRES_PASSWORD, PG_DATABASE
// - the same ones database.ConnectDB reads, see service/.env.template) is reachable.
// DB-dependent tests check this flag and skip cleanly on machines without a running dev
// database instead of failing the whole package. Mirrors internal/handlers/main_test.go.
var dbAvailable bool

func TestMain(m *testing.M) {
	dbAvailable = pingTestDB()
	os.Exit(m.Run())
}

// pingTestDB attempts a short-timeout connection to the database configured via the
// package's standard env vars. An unreachable DB is an expected condition on dev
// machines; callers use the returned bool to skip DB-dependent tests.
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

// setupGameTest skips the test when no dev Postgres is reachable, otherwise connects.
func setupGameTest(t *testing.T) {
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template); set these to point at a running dev database to run this test")
	}
	ConnectDB()
}

// createGameTestUser inserts a bare user (unique random username, no email) directly via
// CreateUser for use as a rating-update participant.
func createGameTestUser(t *testing.T, uname string) models.User {
	u := models.User{Username: uname}
	err := CreateUser(context.Background(), &u)
	require.NoError(t, err, "CreateUser failed")
	return u
}

// seedGameRow inserts a lobby and a games row for gameID directly, bypassing the normal
// lobby/game-start flow. RecordGameAndResults only upserts games.status ('completed') and
// has no notion of lobby_id, but games.lobby_id is NOT NULL with no default, so a fresh
// gameID needs a backing row seeded here or the upsert's INSERT branch fails Postgres's
// constraint. In production this row already exists by the time a game finishes.
func seedGameRow(t *testing.T, gameID, hostUserID uuid.UUID) {
	ctx := context.Background()
	var lobbyID uuid.UUID
	require.NoError(t, DB.QueryRow(ctx,
		`INSERT INTO lobbies (host_user_id, type) VALUES ($1, 'private') RETURNING id`,
		hostUserID,
	).Scan(&lobbyID))
	_, err := DB.Exec(ctx,
		`INSERT INTO games (id, lobby_id, status) VALUES ($1, $2, 'in_progress')`,
		gameID, lobbyID,
	)
	require.NoError(t, err, "seedGameRow: failed to insert games row")
}

// TestRecordGameAndResultsShrinksRD1v1 plays two sequential 1v1 games between the same
// two players and asserts phi_1v1 (Glicko-2 rating deviation) shrinks after each game
// instead of resetting to the DB default (350) every time (cambia-381): before this fix
// RecordGameAndResults only ever persisted elo_1v1, so every game read back the DB
// defaults for phi/sigma and rating deviation never converged with play.
func TestRecordGameAndResultsShrinksRD1v1(t *testing.T) {
	setupGameTest(t)

	winner := createGameTestUser(t, "rd1v1-winner-"+uuid.NewString())
	loser := createGameTestUser(t, "rd1v1-loser-"+uuid.NewString())

	players := []*models.Player{{ID: winner.ID}, {ID: loser.ID}}
	scores := map[uuid.UUID]int{winner.ID: 0, loser.ID: 10}
	winners := []uuid.UUID{winner.ID}

	ctx := context.Background()

	game1 := uuid.New()
	seedGameRow(t, game1, winner.ID)
	require.NoError(t, RecordGameAndResults(ctx, game1, players, scores, winners))

	afterGame1, err := GetUserByID(ctx, winner.ID)
	require.NoError(t, err)
	require.Less(t, afterGame1.Phi1v1, 350.0, "phi_1v1 should have shrunk below the default RD after one game")
	require.NotZero(t, afterGame1.Sigma1v1, "sigma_1v1 should be persisted, not left at zero")

	game2 := uuid.New()
	seedGameRow(t, game2, winner.ID)
	require.NoError(t, RecordGameAndResults(ctx, game2, players, scores, winners))

	afterGame2, err := GetUserByID(ctx, winner.ID)
	require.NoError(t, err)
	require.Less(t, afterGame2.Phi1v1, afterGame1.Phi1v1, "phi_1v1 should keep shrinking across successive games, not reset to the default each time")
}

// TestRecordGameAndResultsShrinksRD4p mirrors TestRecordGameAndResultsShrinksRD1v1 for the
// 4p pool: before this fix, elo_4p/elo_7p8p had no phi/sigma columns at all (cambia-381
// item 3), so multiplayer rating deviation could not be tracked no matter what
// RecordGameAndResults wrote. Also asserts a 4p game does not touch the 1v1 columns.
func TestRecordGameAndResultsShrinksRD4p(t *testing.T) {
	setupGameTest(t)

	var users []models.User
	for i := 0; i < 4; i++ {
		users = append(users, createGameTestUser(t, fmt.Sprintf("rd4p-%d-%s", i, uuid.NewString())))
	}

	players := make([]*models.Player, len(users))
	scores := make(map[uuid.UUID]int, len(users))
	for i, u := range users {
		players[i] = &models.Player{ID: u.ID}
		scores[u.ID] = i * 5 // distinct ranks: lower score is a better placement
	}
	winners := []uuid.UUID{users[0].ID}

	ctx := context.Background()

	game1 := uuid.New()
	seedGameRow(t, game1, users[0].ID)
	require.NoError(t, RecordGameAndResults(ctx, game1, players, scores, winners))

	afterGame1, err := GetUserByID(ctx, users[0].ID)
	require.NoError(t, err)
	require.Less(t, afterGame1.Phi4p, 350.0, "phi_4p should have shrunk below the default RD after one game")
	require.NotZero(t, afterGame1.Sigma4p, "sigma_4p should be persisted, not left at zero")
	require.Equal(t, 1500, afterGame1.Elo1v1, "a 4p game must not touch the 1v1 pool's elo_1v1 column")
	require.Equal(t, 350.0, afterGame1.Phi1v1, "a 4p game must not touch the 1v1 pool's phi_1v1 column")

	game2 := uuid.New()
	seedGameRow(t, game2, users[0].ID)
	require.NoError(t, RecordGameAndResults(ctx, game2, players, scores, winners))

	afterGame2, err := GetUserByID(ctx, users[0].ID)
	require.NoError(t, err)
	require.Less(t, afterGame2.Phi4p, afterGame1.Phi4p, "phi_4p should keep shrinking across successive games, not reset to the default each time")
}
