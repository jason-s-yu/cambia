// internal/handlers/main_test.go
package handlers

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/jason-s-yu/cambia/service/internal/database"
)

// dbAvailable reports whether a Postgres instance matching this package's
// standard connection env vars (PG_HOST, PG_PORT, POSTGRES_USER,
// POSTGRES_PASSWORD, PG_DATABASE - the same ones database.ConnectDB reads,
// see service/.env.template) is reachable. DB-dependent tests check this
// flag and skip cleanly on machines without a running dev database instead
// of failing the whole package.
var dbAvailable bool

func TestMain(m *testing.M) {
	dbAvailable = pingTestDB()
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

// ensureTestDB skips the calling test up front when no database is reachable (dbAvailable, set
// once in TestMain), then connects via database.ConnectDB exactly once for the whole package.
// Every caller after the first still synchronizes with that one connect (sync.Once guarantees
// this), so a game-end persistence goroutine spawned by any test always observes the same,
// never-reassigned database.DB.
func ensureTestDB(t *testing.T) {
	t.Helper()
	if !dbAvailable {
		t.Skip("skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template); set these to point at a running dev database to run this test")
	}
	dbOnce.Do(database.ConnectDB)
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

// pingTestDB attempts a short-timeout connection to the database configured
// via the package's standard env vars. Unlike database.ConnectDB, it never
// calls log.Fatalf: an unreachable DB is an expected condition on dev
// machines and callers use the returned bool to skip DB-dependent tests.
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
