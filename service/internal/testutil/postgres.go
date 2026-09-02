// Package testutil holds helpers the service's DB-backed tests share: locating the module's .env
// and opening a connection pool that is a good citizen on the single dev Postgres every checkout
// on the machine points at.
//
// It deliberately does not import internal/database. That package's own DB-backed tests live in
// package database rather than database_test, so an import back into it would be a cycle; each
// caller assigns the pool this package returns to its own database.DB itself.
package testutil

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"
)

// TestPoolMaxConns bounds one test binary's share of the dev Postgres connection budget, and is
// the isolation mechanism this package exists for (cambia-1830).
//
// The dev database is one shared server (service/docker-compose.yml, published on port 5434) at
// Postgres' default max_connections of 100, and every DB-backed test binary opens a pool of its
// own. pgxpool's default MaxConns is max(4, runtime.NumCPU()), which is 32 on a 32-core dev box:
// three DB-backed packages (internal/game, internal/database, internal/handlers) build separate
// binaries that `go test ./...` runs in parallel, so a single run can ask for 96 connections and
// two checkouts running service tests at the same time saturate the server. The failure that
// produces is `FATAL: sorry, too many clients already (SQLSTATE 53300)` on whichever pool lost
// the race, surfacing as an unrelated-looking assertion failure in the test that happened to be
// holding it: measured at 6 concurrent runs of internal/game, peak connections pinned at 100 and
// TestEndGameRecordsResultsAndRating failed on the exhausted pool.
//
// Four covers the most connection-hungry test in the tree (internal/game's
// TestPersistInitialGameStateIsWaitable holds one connection in an open transaction while a
// background write blocks on a second) and leaves room for around eight concurrent checkouts.
// Rows need no isolation of their own: every DB-backed test here keys its fixtures on a fresh
// uuid, so concurrent runs never touch each other's users, lobbies, games or ratings.
const TestPoolMaxConns = 4

// pingTimeout bounds the reachability probe. An unreachable database is an expected condition on
// a machine with no dev stack up, so the probe fails fast rather than stalling the package.
const pingTimeout = 2 * time.Second

// LoadServiceEnv loads the module's .env (service/.env) into the process environment without
// overriding a variable the caller already exported, and is a no-op when there is no such file.
//
// The godotenv autoload some DB-backed test packages carry reads a .env in the process's working
// directory, and `go test` runs each package's binary in that package's own source directory, so
// service/.env is never the file it finds and the DB-backed tests silently skip for anyone who
// keeps their settings there instead of exporting them. Walking up to the directory holding
// go.mod finds it from any package in the module, and stops there rather than wandering further
// up the filesystem.
func LoadServiceEnv() {
	dir, err := os.Getwd()
	if err != nil {
		return
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			envPath := filepath.Join(dir, ".env")
			if _, err := os.Stat(envPath); err == nil {
				_ = godotenv.Load(envPath)
			}
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return
		}
		dir = parent
	}
}

// PostgresDSN builds the connection string from the same env vars database.ConnectDB reads:
// PG_HOST, PG_PORT, POSTGRES_USER, POSTGRES_PASSWORD, PG_DATABASE (see service/.env.template).
func PostgresDSN() string {
	return fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s",
		os.Getenv("POSTGRES_USER"),
		os.Getenv("POSTGRES_PASSWORD"),
		os.Getenv("PG_HOST"),
		os.Getenv("PG_PORT"),
		os.Getenv("PG_DATABASE"),
	)
}

// NewBoundedPool opens a pool against the configured dev Postgres capped at TestPoolMaxConns.
// ctx bounds pool construction only; pgxpool does not retain it for later acquires.
func NewBoundedPool(ctx context.Context) (*pgxpool.Pool, error) {
	config, err := pgxpool.ParseConfig(PostgresDSN())
	if err != nil {
		return nil, fmt.Errorf("parse test pool config: %w", err)
	}
	config.MaxConns = TestPoolMaxConns
	return pgxpool.NewWithConfig(ctx, config)
}

// PingPostgres reports whether the configured dev Postgres answers within pingTimeout. Callers
// hold the result for the life of the test binary and skip DB-backed tests when it is false.
func PingPostgres() bool {
	ctx, cancel := context.WithTimeout(context.Background(), pingTimeout)
	defer cancel()

	pool, err := NewBoundedPool(ctx)
	if err != nil {
		return false
	}
	defer pool.Close()

	return pool.Ping(ctx) == nil
}

// SkipMessage is the reason DB-backed tests report when no dev Postgres is reachable.
const SkipMessage = "skipping: no Postgres reachable via PG_HOST/PG_PORT/POSTGRES_USER/POSTGRES_PASSWORD/PG_DATABASE (see service/.env.template); set these to point at a running dev database, or keep them in service/.env, to run this test"
