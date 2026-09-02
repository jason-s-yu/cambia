// internal/game/game_db_test.go
package game

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/jason-s-yu/cambia/service/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// dbAvailable reports whether a Postgres instance matching this package's standard connection
// env vars (PG_HOST, PG_PORT, POSTGRES_USER, POSTGRES_PASSWORD, PG_DATABASE - the same ones
// database.ConnectDB reads, see service/.env.template) is reachable. DB-dependent tests check
// this flag and skip cleanly on machines without a running dev database. Mirrors
// internal/handlers/main_test.go and internal/database/game_test.go.
var dbAvailable bool

func TestMain(m *testing.M) {
	testutil.LoadServiceEnv()
	dbAvailable = testutil.PingPostgres()
	os.Exit(m.Run())
}

// gameDBOnce guards the single pool this package's test binary opens, mirroring
// internal/handlers/main_test.go's dbOnce/ensureTestDB (cambia-908): every DB-backed test used to
// call database.ConnectDB() itself, which reassigns the package-level database.DB pool on every
// call, so a later test's call could race under -race against a background goroutine from an
// earlier test's game-end persistence still reading the pool it was about to replace. Connecting
// exactly once for the whole binary removes that race regardless of how many DB-backed tests this
// package grows.
var gameDBOnce sync.Once

// gameDBPoolErr records a failure from that single pool open, so every test that needs the pool
// reports it rather than only the one that happened to run first.
var gameDBPoolErr error

// setupGameDBTest skips the test when no dev Postgres is reachable, otherwise installs this
// package's single database pool (gameDBOnce).
//
// The pool comes from testutil.NewBoundedPool rather than database.ConnectDB because the dev
// Postgres is shared by every checkout on the machine: ConnectDB takes pgxpool's default MaxConns
// of max(4, NumCPU), and concurrent test runs across worktrees exhaust the server's 100
// connections that way (testutil.TestPoolMaxConns carries the measurement, cambia-1830). Building
// the pool here also keeps a test binary from running migrations against that shared database,
// which ConnectDB does whenever RUN_MIGRATIONS is set in the environment.
func setupGameDBTest(t *testing.T) {
	if !dbAvailable {
		t.Skip(testutil.SkipMessage)
	}
	gameDBOnce.Do(func() {
		pool, err := testutil.NewBoundedPool(context.Background())
		if err != nil {
			gameDBPoolErr = err
			return
		}
		database.DB = pool
	})
	require.NoError(t, gameDBPoolErr, "open this package's bounded test pool")
}

// awaitPersistence blocks, bounded by timeout, until every background DB-write goroutine
// tracked by wg (persistInitialGameState, persistFinalGameState) completes. RecordGameAndResults
// runs the game_results insert and the games.status='completed' update in one transaction and
// the rating (users elo/phi/sigma + ratings row) update in a second, separate transaction, both
// inside the same goroutine persistFinalGameState launches; polling games.status alone observes
// only the first transaction; waiting on wg (Add'd before the goroutine launches, Done'd when it
// returns - see CambiaGame.PersistWG) is what actually proves the rating write has landed before
// a caller reads it (cambia-908 L4).
func awaitPersistence(t *testing.T, wg *sync.WaitGroup, timeout time.Duration) {
	t.Helper()
	if wg == nil {
		return
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		t.Fatalf("cambia-908: game-end persistence goroutines did not finish within %s", timeout)
	}
}

// createGameDBTestUser inserts a bare user (unique random username, no email) for use as a
// rating-update participant, and returns it as the database holds it. Registers a t.Cleanup
// deleting the created row - and anything a test drove it to accumulate as a lobby host or game
// participant - in FK-safe order, mirroring internal/handlers/main_test.go's cleanupTestUserRows
// (cambia-890 F4 spirit: this package's own DB-backed test leaked a user, a lobby, a game, two
// game_results rows and two ratings rows per run before this fix, confirmed by comparing
// `select count(*)` across those tables before and after a run).
//
// The read-back is what makes the returned value a usable "before" rating. CreateUser's INSERT
// names only id/email/password/username/is_ephemeral/is_admin, so the elo/phi/sigma columns take
// their schema defaults in the database while the struct it was handed keeps their Go zero
// values; a caller comparing a post-game rating against that struct compares against 0 rather
// than the 1500 the row actually holds. TestForfeitPersistsTheForfeitScoreAndRatesTheQuitterDown
// failed on every run that way, asserting 1338 <= 0 (cambia-1830).
func createGameDBTestUser(t *testing.T, uname string) models.User {
	u := models.User{Username: uname}
	err := database.CreateUser(context.Background(), &u)
	require.NoError(t, err, "CreateUser failed")
	t.Cleanup(func() { cleanupGameDBTestUserRows(t, u.ID) })

	created, err := database.GetUserByID(context.Background(), u.ID)
	require.NoError(t, err, "read back the created user")
	return *created
}

// cleanupGameDBTestUserRows deletes every row this package's DB-backed test could have left
// behind for a single test-created user, in FK-safe order, then the user row itself: game_results
// (by player_id, which carries no cascade from users) first, then lobbies (by host_user_id, which
// cascades lobby_participants and games - which in turn cascades game_actions, game_results, and
// ratings for that game_id), then the user row itself (which cascades friends and any remaining
// ratings row via user_id).
func cleanupGameDBTestUserRows(t *testing.T, userID uuid.UUID) {
	t.Helper()
	if database.DB == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := database.DB.Exec(ctx, `DELETE FROM game_actions WHERE actor_user_id = $1`, userID); err != nil {
		t.Logf("cleanupGameDBTestUserRows: delete game_actions for %s: %v", userID, err)
	}
	if _, err := database.DB.Exec(ctx, `DELETE FROM game_results WHERE player_id = $1`, userID); err != nil {
		t.Logf("cleanupGameDBTestUserRows: delete game_results for %s: %v", userID, err)
	}
	if _, err := database.DB.Exec(ctx, `DELETE FROM lobbies WHERE host_user_id = $1`, userID); err != nil {
		t.Logf("cleanupGameDBTestUserRows: delete lobbies hosted by %s: %v", userID, err)
	}
	if _, err := database.DB.Exec(ctx, `DELETE FROM users WHERE id = $1`, userID); err != nil {
		t.Logf("cleanupGameDBTestUserRows: delete user %s: %v", userID, err)
	}
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
	g.HouseRules = *testHouseRules(0, 2)
	g.TurnDuration = 0
	// Tracks persistInitialGameState's and persistFinalGameState's background DB-write
	// goroutines so the test can wait for them deterministically instead of polling a single
	// column that only one of the goroutine's two transactions actually touches (cambia-908 L4).
	g.PersistWG = &sync.WaitGroup{}

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

	// Waits for persistFinalGameState's goroutine to finish both of RecordGameAndResults'
	// transactions (game_results + games.status, then the separate rating transaction), not just
	// the first: waitForGameStatus("completed") alone would only prove the first transaction
	// landed, racing the rating assertions below against the second (cambia-908 L4).
	awaitPersistence(t, g.PersistWG, 5*time.Second)

	rows, err := database.DB.Query(context.Background(), `SELECT player_id, score FROM game_results WHERE game_id = $1`, g.ID)
	require.NoError(t, err)
	scores := map[uuid.UUID]int{}
	for rows.Next() {
		var pid uuid.UUID
		var score int
		require.NoError(t, rows.Scan(&pid, &score))
		scores[pid] = score
	}
	rows.Close()
	require.Contains(t, scores, userA.ID, "player A should have a game_results row")
	require.Contains(t, scores, userB.ID, "player B should have a game_results row")

	// Rating: the outcome-independent proof that EndGame reached the rating step is the ratings
	// rows applyRatingUpdate writes, one per player, attributed to this game and stamped with the
	// pool the 2-player roster selects. Asserting on those rather than on elo alone is what makes
	// this test deterministic: the deal is unseeded, so roughly one run in six deals a tie, and a
	// Glicko-2 draw between two fresh 1500/350 ratings leaves both elo values at exactly 1500,
	// which the old "elo moved for at least one player" assertion read as the rating step never
	// having run (cambia-1244).
	var ratingRows int
	require.NoError(t, database.DB.QueryRow(context.Background(),
		`SELECT count(*) FROM ratings WHERE game_id = $1 AND rating_mode = '1v1' AND old_rating = 1500`,
		g.ID).Scan(&ratingRows))
	require.Equal(t, 2, ratingRows,
		"a rated 2p game reached via EndGame() should write one 1v1 ratings row per player")

	afterA, err := database.GetUserByID(context.Background(), userA.ID)
	require.NoError(t, err)
	afterB, err := database.GetUserByID(context.Background(), userB.ID)
	require.NoError(t, err)

	// Both deals the unseeded shuffle can produce are asserted, so the run's outcome selects the
	// expectation instead of deciding whether the test passes. Lower score wins (RULES.md 6), and
	// FinalizeRatings maps the 2-player ranking to a decisive 1/0 or a shared 0.5, so the sign of
	// each move is fixed once the scores are known.
	switch {
	case scores[userA.ID] == scores[userB.ID]:
		// A draw between two identical fresh ratings has expected score 0.5 against an actual
		// 0.5, so mu does not move and both elo values stay at 1500. The deviation still shrinks
		// off the 350.0 fresh-user default, which is what proves the update was applied.
		assert.Equal(t, 1500, afterA.Elo1v1, "a tied 2p game must leave player A's elo_1v1 at 1500")
		assert.Equal(t, 1500, afterB.Elo1v1, "a tied 2p game must leave player B's elo_1v1 at 1500")
		assert.Less(t, afterA.Phi1v1, 350.0, "a tied rated game must still shrink player A's phi_1v1 off the default")
		assert.Less(t, afterB.Phi1v1, 350.0, "a tied rated game must still shrink player B's phi_1v1 off the default")
	case scores[userA.ID] < scores[userB.ID]:
		assert.Greater(t, afterA.Elo1v1, 1500, "the lower-scoring player A should gain elo_1v1")
		assert.Less(t, afterB.Elo1v1, 1500, "the higher-scoring player B should lose elo_1v1")
	default:
		assert.Greater(t, afterB.Elo1v1, 1500, "the lower-scoring player B should gain elo_1v1")
		assert.Less(t, afterA.Elo1v1, 1500, "the higher-scoring player A should lose elo_1v1")
	}
}
