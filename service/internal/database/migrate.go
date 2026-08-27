package database

import (
	"context"
	"fmt"
	"io/fs"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jason-s-yu/cambia/service/migrations"
)

// migrationLockID is the pg_advisory_lock key serializing migration runs. Two
// containers booting against the same database (server plus historian, or a
// rolling replace) must not apply the same file concurrently. The value is
// arbitrary but must stay stable across releases.
const migrationLockID int64 = 8734129045

// migrationTimeout bounds the whole migration pass, lock acquisition included,
// so a boot cannot hang forever behind a stuck lock holder.
const migrationTimeout = 2 * time.Minute

// migrateOnce keeps the migrator to a single pass per process. ConnectDB is
// re-entered by ConnectDBAsync's reconnect loop, and a reconnect is not a boot.
var migrateOnce sync.Once

// MigrateIfEnabled applies any un-applied embedded migration to the connected
// database when RUN_MIGRATIONS is truthy, and is a no-op otherwise. It runs at
// most once per process.
//
// Any failure is fatal: a half-migrated process must not go on to serve
// traffic against a schema it does not match.
func MigrateIfEnabled() {
	switch os.Getenv("RUN_MIGRATIONS") {
	case "true", "1":
	default:
		return
	}

	migrateOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), migrationTimeout)
		defer cancel()
		if err := Migrate(ctx, DB); err != nil {
			log.Fatalf("migrations failed: %v", err)
		}
	})
}

// Migrate applies every embedded migration that schema_migrations does not
// already record, in version order, each in its own transaction.
//
// It holds a session-level advisory lock for the duration, so a second process
// booting at the same time waits rather than racing.
func Migrate(ctx context.Context, pool *pgxpool.Pool) error {
	if pool == nil {
		return fmt.Errorf("no database pool")
	}

	files, err := migrationFiles()
	if err != nil {
		return err
	}

	// The advisory lock is session-scoped, so it has to be taken and released on
	// one pinned connection rather than anywhere in the pool.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("acquire migration connection: %w", err)
	}
	defer conn.Release()

	if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock($1)", migrationLockID); err != nil {
		return fmt.Errorf("acquire migration lock: %w", err)
	}
	defer func() {
		// Best effort: a failed unlock still releases when the connection closes.
		unlockCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if _, err := conn.Exec(unlockCtx, "SELECT pg_advisory_unlock($1)", migrationLockID); err != nil {
			log.Printf("migrations: failed to release advisory lock: %v", err)
		}
	}()

	baseline, err := needsBaseline(ctx, conn)
	if err != nil {
		return err
	}

	if _, err := conn.Exec(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version    TEXT PRIMARY KEY,
			applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`); err != nil {
		return fmt.Errorf("create schema_migrations: %w", err)
	}

	if baseline {
		// The database was initialized out of band (docker-entrypoint-initdb.d, or
		// a hand-built dev database) and already carries the full schema. Re-running
		// 0_init.sql would be mostly idempotent but the later ALTERs are not, so
		// record the current files as applied instead of executing them.
		log.Printf("MIGRATIONS BASELINE: schema_migrations was absent but the users table exists. "+
			"Recording all %d migration files as applied WITHOUT executing them. "+
			"Verify the schema matches these files before trusting future migrations.", len(files))
		for _, name := range files {
			if err := recordApplied(ctx, conn, name); err != nil {
				return err
			}
		}
		return nil
	}

	applied, err := appliedVersions(ctx, conn)
	if err != nil {
		return err
	}

	pending := 0
	for _, name := range files {
		if applied[name] {
			continue
		}
		body, err := readMigration(name)
		if err != nil {
			return err
		}
		if err := applyMigration(ctx, conn, name, body); err != nil {
			return err
		}
		log.Printf("migrations: applied %s", name)
		pending++
	}

	if pending == 0 {
		log.Printf("migrations: schema up to date (%d applied)", len(applied))
	} else {
		log.Printf("migrations: applied %d file(s)", pending)
	}
	return nil
}

// applyMigration runs one migration file and records it in the same
// transaction, so a version is never marked applied unless its SQL committed.
func applyMigration(ctx context.Context, conn *pgxpool.Conn, name, body string) error {
	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin %s: %w", name, err)
	}
	defer tx.Rollback(context.Background()) //nolint:errcheck // no-op after commit

	if _, err := tx.Exec(ctx, body); err != nil {
		return fmt.Errorf("apply %s: %w", name, err)
	}
	if _, err := tx.Exec(ctx,
		"INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING", name,
	); err != nil {
		return fmt.Errorf("record %s: %w", name, err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit %s: %w", name, err)
	}
	return nil
}

// recordApplied marks a version applied without running it (baseline path).
func recordApplied(ctx context.Context, conn *pgxpool.Conn, name string) error {
	if _, err := conn.Exec(ctx,
		"INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING", name,
	); err != nil {
		return fmt.Errorf("baseline %s: %w", name, err)
	}
	return nil
}

// needsBaseline reports whether the database predates schema_migrations: no
// bookkeeping table, but the schema those files build is already present.
func needsBaseline(ctx context.Context, conn *pgxpool.Conn) (bool, error) {
	var hasMigrations, hasUsers bool
	err := conn.QueryRow(ctx,
		"SELECT to_regclass('public.schema_migrations') IS NOT NULL, to_regclass('public.users') IS NOT NULL",
	).Scan(&hasMigrations, &hasUsers)
	if err != nil {
		return false, fmt.Errorf("probe existing schema: %w", err)
	}
	return !hasMigrations && hasUsers, nil
}

// appliedVersions reads the set of versions already recorded.
func appliedVersions(ctx context.Context, conn *pgxpool.Conn) (map[string]bool, error) {
	rows, err := conn.Query(ctx, "SELECT version FROM schema_migrations")
	if err != nil {
		return nil, fmt.Errorf("read schema_migrations: %w", err)
	}
	defer rows.Close()

	applied := make(map[string]bool)
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, fmt.Errorf("scan schema_migrations: %w", err)
		}
		applied[v] = true
	}
	return applied, rows.Err()
}

// migrationFiles lists the embedded .sql files in version order: by the leading
// integer where one is present, by name otherwise. Plain lexicographic order
// would run 10_ before 2_ once the set grows past nine files.
func migrationFiles() ([]string, error) {
	entries, err := fs.ReadDir(migrations.FS, ".")
	if err != nil {
		return nil, fmt.Errorf("read embedded migrations: %w", err)
	}

	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".sql") {
			names = append(names, e.Name())
		}
	}
	sort.Slice(names, func(i, j int) bool {
		ni, oki := versionPrefix(names[i])
		nj, okj := versionPrefix(names[j])
		if oki && okj && ni != nj {
			return ni < nj
		}
		if oki != okj {
			return oki // numbered files sort ahead of unnumbered ones
		}
		return names[i] < names[j]
	})
	return names, nil
}

// readMigration returns the body of one embedded migration file.
func readMigration(name string) (string, error) {
	body, err := migrations.FS.ReadFile(name)
	if err != nil {
		return "", fmt.Errorf("read migration %s: %w", name, err)
	}
	return string(body), nil
}

// versionPrefix parses the leading "<n>_" of a migration filename.
func versionPrefix(name string) (int, bool) {
	i := strings.IndexByte(name, '_')
	if i <= 0 {
		return 0, false
	}
	n, err := strconv.Atoi(name[:i])
	if err != nil {
		return 0, false
	}
	return n, true
}
