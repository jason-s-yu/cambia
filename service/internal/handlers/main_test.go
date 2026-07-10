// internal/handlers/main_test.go
package handlers

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
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
