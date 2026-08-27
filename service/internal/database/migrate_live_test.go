// internal/database/migrate_live_test.go
//
// Live migrator coverage against a real Postgres. Skipped unless
// CAMBIA_MIGRATE_TEST_DSN is set, and it is deliberately a dedicated variable
// rather than the package's usual PG_* vars: every test here starts by dropping
// and recreating the public schema. Point it only at a throwaway database.
//
//	docker run -d --name cambia-mig-check -e POSTGRES_PASSWORD=migcheck \
//	  -e POSTGRES_USER=migcheck -e POSTGRES_DB=migcheck \
//	  -p 127.0.0.1:55432:5432 postgres:17-alpine
//	CAMBIA_MIGRATE_TEST_DSN=postgres://migcheck:migcheck@127.0.0.1:55432/migcheck \
//	  go test ./internal/database/ -run TestLive
package database

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

func liveDSN(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv("CAMBIA_MIGRATE_TEST_DSN")
	if dsn == "" {
		t.Skip("CAMBIA_MIGRATE_TEST_DSN not set")
	}
	return dsn
}

func freshPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	pool, err := pgxpool.New(context.Background(), liveDSN(t))
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	t.Cleanup(pool.Close)
	if _, err := pool.Exec(context.Background(), "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"); err != nil {
		t.Fatalf("reset schema: %v", err)
	}
	return pool
}

func count(t *testing.T, pool *pgxpool.Pool, q string, args ...any) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), q, args...).Scan(&n); err != nil {
		t.Fatalf("query %q: %v", q, err)
	}
	return n
}

func exists(t *testing.T, pool *pgxpool.Pool, table string) bool {
	t.Helper()
	var ok bool
	if err := pool.QueryRow(context.Background(),
		"SELECT to_regclass('public.'||$1) IS NOT NULL", table).Scan(&ok); err != nil {
		t.Fatalf("regclass %s: %v", table, err)
	}
	return ok
}

func TestLiveMigrateFreshDatabase(t *testing.T) {
	pool := freshPool(t)
	ctx := context.Background()

	if err := Migrate(ctx, pool); err != nil {
		t.Fatalf("first Migrate: %v", err)
	}

	files, _ := migrationFiles()
	if n := count(t, pool, "SELECT count(*) FROM schema_migrations"); n != len(files) {
		t.Fatalf("expected %d recorded migrations, got %d", len(files), n)
	}
	for _, table := range []string{"users", "games", "game_actions", "ratings", "lobbies", "lobby_participants"} {
		if !exists(t, pool, table) {
			t.Fatalf("expected table %s after a fresh migrate", table)
		}
	}

	// Re-running is a no-op.
	if err := Migrate(ctx, pool); err != nil {
		t.Fatalf("second Migrate: %v", err)
	}
	if n := count(t, pool, "SELECT count(*) FROM schema_migrations"); n != len(files) {
		t.Fatalf("re-run changed the recorded set: %d", n)
	}
}

func TestLiveMigratePartialCatchUp(t *testing.T) {
	pool := freshPool(t)
	ctx := context.Background()

	if err := Migrate(ctx, pool); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	// Forget the last file, as if the image shipped a new migration.
	files, _ := migrationFiles()
	last := files[len(files)-1]
	if _, err := pool.Exec(ctx, "DELETE FROM schema_migrations WHERE version=$1", last); err != nil {
		t.Fatalf("delete: %v", err)
	}
	// The enum has to go too: 5_add_lobby_persistence.sql creates lobby_type
	// unguarded, so a leftover type makes the re-apply fail on its own setup.
	if _, err := pool.Exec(ctx, "DROP TABLE IF EXISTS lobby_participants, lobbies CASCADE; DROP TYPE IF EXISTS lobby_type"); err != nil {
		t.Fatalf("drop: %v", err)
	}

	if err := Migrate(ctx, pool); err != nil {
		t.Fatalf("catch-up Migrate: %v", err)
	}
	if !exists(t, pool, "lobbies") {
		t.Fatal("expected the pending migration to be re-applied")
	}
	if n := count(t, pool, "SELECT count(*) FROM schema_migrations WHERE version=$1", last); n != 1 {
		t.Fatalf("expected %s recorded once, got %d", last, n)
	}
}

func TestLiveMigrateBaselinesInitdbDatabase(t *testing.T) {
	pool := freshPool(t)
	ctx := context.Background()

	// Stand in for docker-entrypoint-initdb.d: 0_init.sql ran, nothing recorded.
	body, err := readMigration("0_init.sql")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if _, err := pool.Exec(ctx, body); err != nil {
		t.Fatalf("seed init: %v", err)
	}
	if exists(t, pool, "schema_migrations") {
		t.Fatal("precondition: schema_migrations should not exist yet")
	}

	if err := Migrate(ctx, pool); err != nil {
		t.Fatalf("baseline Migrate: %v", err)
	}

	files, _ := migrationFiles()
	if n := count(t, pool, "SELECT count(*) FROM schema_migrations"); n != len(files) {
		t.Fatalf("expected all %d files baselined, got %d", len(files), n)
	}
	// Baseline records without executing, so a later file's table stays absent.
	if exists(t, pool, "lobbies") {
		t.Fatal("baseline executed a migration it should only have recorded")
	}
}
