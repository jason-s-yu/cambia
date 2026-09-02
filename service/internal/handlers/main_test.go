// internal/handlers/main_test.go
package handlers

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/testutil"
	"github.com/stretchr/testify/require"
)

// dbAvailable reports whether a Postgres instance matching this package's
// standard connection env vars (PG_HOST, PG_PORT, POSTGRES_USER,
// POSTGRES_PASSWORD, PG_DATABASE - the same ones database.ConnectDB reads,
// see service/.env.template) is reachable. DB-dependent tests check this
// flag and skip cleanly on machines without a running dev database instead
// of failing the whole package.
var dbAvailable bool

func TestMain(m *testing.M) {
	testutil.LoadServiceEnv()
	dbAvailable = testutil.PingPostgres()
	os.Exit(m.Run())
}

// dbOnce guards the single database.ConnectDB() call this package makes for its whole test
// binary. Every DB-backed test used to call database.ConnectDB() itself, and since it
// reassigns the package-level database.DB pool on every call, a later test's call raced under
// -race against a background goroutine from an earlier test's game-end persistence
// (persistFinalGameState) still reading the pool it was about to replace (cambia-908).
// ensureTestDB replaces every direct database.ConnectDB() call site in this package's tests so
// the connect happens exactly once; database.ConnectDB's production call path is untouched.
var dbOnce sync.Once

// dbPoolErr records a failure from that single pool open so every DB-backed test reports it.
var dbPoolErr error

// ensureTestDB skips the calling test up front when no database is reachable (dbAvailable, set
// once in TestMain), then opens testutil.NewBoundedPool exactly once for the whole package: the
// dev Postgres is shared by every checkout on the machine and database.ConnectDB's default
// MaxConns exhausted its connections under concurrent worktrees (cambia-1830).
// Every caller after the first still synchronizes with that one connect (sync.Once guarantees
// this), so a game-end persistence goroutine spawned by any test always observes the same,
// never-reassigned database.DB.
func ensureTestDB(t *testing.T) {
	t.Helper()
	if !dbAvailable {
		t.Skip(testutil.SkipMessage)
	}
	dbOnce.Do(func() {
		pool, err := testutil.NewBoundedPool(context.Background())
		if err != nil {
			dbPoolErr = err
			return
		}
		database.DB = pool
	})
	require.NoError(t, dbPoolErr, "open this package's bounded test pool")
}

// TestEnsureTestDBConnectsOnce is the cambia-908 regression for the first half of the fix:
// whichever test calls ensureTestDB first is the only one that actually reconnects; every later
// call (from this test or any other) must observe the same database.DB pool rather than a fresh
// one. Deterministic, unlike the original failure: it checks pointer identity directly rather
// than depending on a background goroutine losing a race, which is why the ticket's acceptance
// command (go test -race, -count=3) is the primary evidence and this is a supporting unit check.
func TestEnsureTestDBConnectsOnce(t *testing.T) {
	ensureTestDB(t)
	pool := database.DB
	if pool == nil {
		t.Fatalf("ensureTestDB left database.DB nil after connecting")
	}
	ensureTestDB(t)
	if database.DB != pool {
		t.Fatalf("a second ensureTestDB call reassigned database.DB: got %p, want the original %p", database.DB, pool)
	}
}

// cleanupTestUserRows deletes every row this package's DB-backed tests could have left behind
// for a single test-created user, in FK-safe order, then the user row itself (cambia-890 F4).
// Registered via t.Cleanup by every helper that inserts a real users row (createTestUser,
// createGuestSession), so repeated test runs against the shared dev DB do not grow users,
// lobbies, games, game_results, or ratings without bound.
//
// Self-contained regardless of t.Cleanup's LIFO ordering relative to any other test-created
// user's cleanup or a lobby's own cleanup (cleanupLobbyDBRows below): game_results.player_id and
// game_actions.actor_user_id carry no ON DELETE CASCADE from users (only game_id cascades from
// games, and lobbies.host_user_id/game_results.player_id/game_actions.actor_user_id are all plain
// NO ACTION references to users), so deleting a user row while an unrelated cleanup still owns a
// row that references it as a *participant* (not host) would fail the delete. Explicitly clearing
// game_actions/game_results by this user's id first - not just lobbies by host_user_id - makes
// that failure impossible no matter which cleanup happens to run first. ratings.user_id and
// friends.user1_id/user2_id do carry ON DELETE CASCADE from users, so those need no explicit
// delete here.
func cleanupTestUserRows(t *testing.T, userID uuid.UUID) {
	t.Helper()
	if database.DB == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := database.DB.Exec(ctx, `DELETE FROM game_actions WHERE actor_user_id = $1`, userID); err != nil {
		t.Logf("cleanupTestUserRows: delete game_actions for %s: %v", userID, err)
	}
	if _, err := database.DB.Exec(ctx, `DELETE FROM game_results WHERE player_id = $1`, userID); err != nil {
		t.Logf("cleanupTestUserRows: delete game_results for %s: %v", userID, err)
	}
	// Cascades lobby_participants, and games (which in turn cascades game_actions,
	// game_results, and ratings tied to that game_id) for any lobby this user hosted.
	if _, err := database.DB.Exec(ctx, `DELETE FROM lobbies WHERE host_user_id = $1`, userID); err != nil {
		t.Logf("cleanupTestUserRows: delete lobbies hosted by %s: %v", userID, err)
	}
	// Cascades friends (user1_id and user2_id) and any remaining ratings row (user_id).
	if _, err := database.DB.Exec(ctx, `DELETE FROM users WHERE id = $1`, userID); err != nil {
		t.Logf("cleanupTestUserRows: delete user %s: %v", userID, err)
	}
}

// cleanupLobbyDBRows best-effort deletes the lobbies row (if any) a test-driven POST
// /lobby/create ended up persisting, by lobby id (cambia-890 F4). CreateLobbyHandler itself
// writes nothing to Postgres - the row exists only once a game actually starts against the
// lobby (game.persistInitialGameState -> database.UpsertInitialGameState upserts it to satisfy
// games.lobby_id's FK) - so this is a 0-row no-op for the many callers (most of this package's
// hub/lobby protocol tests) that never drive a lobby that far, and reaches the DB at all only
// once some earlier test in this binary has connected it (database.DB != nil): plain in-memory
// lobby/hub tests never acquire a DB dependency they did not already have.
//
// Deleting by lobby id cascades lobby_participants and games (which cascades game_actions,
// game_results, ratings for that game_id) without needing to know which users were involved.
// A test always creates its users before calling createPublicLobby/createPrivateLobby (the host
// token has to exist first), so this cleanup is always registered after theirs - and t.Cleanup
// runs cleanups in LIFO order, so this one fires first, clearing any game_results/game_actions
// row a participant's cleanupTestUserRows might otherwise race against before it ever runs.
//
// The DELETE is preceded by drainLobbyPersistence (cambia-942 F4): the writes this deletes are
// issued by fire-and-forget goroutines, so deleting first leaves a still-running write to fail
// on the vanished FK target or, worse, to re-insert the lobbies/games rows straight after the
// DELETE and strand them. That was the pre-fix ordering: registering this inside the lobby
// helpers put the DELETE ahead of startTwoPlayerGame's drain under t.Cleanup LIFO, so a late
// UpsertInitialGameState or RecordGameAndResults could land behind it and leave a whole
// lobbies/games/game_results triple; the host's cleanupTestUserRows then fails its users DELETE
// on lobbies.host_user_id (that FK has no cascade), only t.Logf's, and leaks the fixture user
// too. Measure that shape on a database this suite has to itself: cambia-dev also carries rows
// from any dev server running against it, so a row-count delta there proves nothing.
func cleanupLobbyDBRows(t *testing.T, gs *GameServer, lobbyID uuid.UUID) {
	t.Helper()
	drainLobbyPersistence(t, gs, lobbyID)
	if database.DB == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := database.DB.Exec(ctx, `DELETE FROM lobbies WHERE id = $1`, lobbyID); err != nil {
		t.Logf("cleanupLobbyDBRows: delete lobby %s: %v", lobbyID, err)
	}
}

// drainLobbyPersistence waits out the background DB writes a game on this lobby may still have
// in flight, so a caller can delete the lobby's rows without racing them (cambia-942 F4).
//
// A no-op unless the server tracks its persistence goroutines (gs.PersistWG; only test servers
// set it - see newForfeitTestServer). Otherwise it works off whether the lobby's game is still
// registered, which is exactly the signal needed for sync.WaitGroup's Add-before-Wait rule:
//
//   - Still registered: the game has not ended, so persistFinalGameState's Add has not run and
//     a bare Wait would see a zero counter and return ahead of it. Poll GameOver first (endGame
//     sets it under the game's lock, in the same call that Adds), then Wait.
//   - Not registered: endGame already ran. It calls persistFinalGameState, and only afterwards
//     OnGameEnd, whose last act is GameStore.DeleteGame - so reading the game's absence through
//     the store's own mutex is itself proof that both game-end Adds already happened, as is the
//     initial-state Add from BeginPreGame, earlier still. Wait alone is correct here, and this
//     is the common case: the store drops a game as soon as it ends.
//
// Both phases are deadline-bounded, since a test may legitimately end while its game is still
// running; in that case only the initial-state upsert is outstanding and Wait covers it.
func drainLobbyPersistence(t *testing.T, gs *GameServer, lobbyID uuid.UUID) {
	t.Helper()
	if gs == nil || gs.PersistWG == nil {
		return
	}
	if g := gs.GameStore.GetGameByLobbyID(lobbyID); g != nil {
		deadline := time.Now().Add(5 * time.Second)
		// forUser is irrelevant here: ObfGameState.GameOver is derived from engine/game state,
		// not from the requesting player's view.
		for !g.GetCurrentObfuscatedGameState(uuid.Nil).GameOver {
			if time.Now().After(deadline) {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	done := make(chan struct{})
	go func() {
		gs.PersistWG.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		// Errorf, not Fatalf: this runs from t.Cleanup, and the DELETE that follows is still
		// worth attempting. A test that reaches this has an undrained write and its cleanup is
		// no longer race-free, so it must not pass silently.
		t.Errorf("cambia-942 F4: persistence goroutines for lobby %s did not finish within 5s; deleting its rows anyway", lobbyID)
	}
}
