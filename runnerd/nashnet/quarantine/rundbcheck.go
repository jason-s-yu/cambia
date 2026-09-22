// Package quarantine holds the nashnet per-lease quarantine store: parts,
// blobs, manifests, and promotion (design v1.1-compute-pool-design.md
// section 4). rundbcheck.go is one leaf of that package: the D55 journal
// validator, called by the manifest commit transaction (D51 step 4) before a
// node-authored run_db.sqlite is folded into runs/<job>/.
package quarantine

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	_ "modernc.org/sqlite" // pure-Go driver, no cgo (rundb_status.go precedent)
)

// Reason codes returned on a rejected Verdict. Stable strings so a caller (or
// a test) can switch on them; the manifest commit response collapses every
// one of them to the single per-entry reason "rundb_invalid" (D51 step 4),
// these are for the coordinator's own logs and for telling fixtures apart in
// tests.
const (
	ReasonSizeCap        = "size_cap"
	ReasonIntegrityCheck = "integrity_check"
	ReasonSchema         = "schema"
	ReasonIdentity       = "identity"
	ReasonEnum           = "enum"
	ReasonRowCount       = "row_count"
)

// DefaultMaxRunDBBytes is the D55 size cap checked before the file is ever
// opened. The design's default is 256 MiB, overridable in production via
// RUNNERD_NASHNET_MAX_RUNDB_BYTES; this package reads no environment itself,
// so the daemon's config-loading layer threads the resolved value into
// RunDBConfig.MaxBytes.
const DefaultMaxRunDBBytes int64 = 256 * 1024 * 1024

// DefaultQueryTimeout mirrors the 5s budget of runnerd/harness/rundb_status.go
// (runDBQueryTimeout), applied here as one shared timeout for the whole
// validation session rather than per query.
const DefaultQueryTimeout = 5 * time.Second

// DefaultMaxRowsPerTable bounds every table but runs, whose row count is
// pinned to exactly one by the identity check rather than a cap. A generous
// default: real per-run journals carry at most a few thousand checkpoint and
// eval rows; this exists to bound a hostile node stuffing many tiny rows
// within the byte cap, not to constrain a normal run.
const DefaultMaxRowsPerTable = 200000

// runDBSchema is the set of tables and columns a journal's schema must fit
// inside (D55): every column cfr/src/run_db.py can create, through its _DDL
// or the _COLUMN_MIGRATIONS get_db applies after it. Kept in sync by hand;
// TestRunDBSchemaParity parses run_db.py and fails on any column missing
// here, and the shared fixture corpus under runnerd/harness/testdata/rundb/
// (generated from the real schema, consumed by both suites) is held current
// by TestCorpusMatchesRunDBSchema (cambia-2358).
var runDBSchema = map[string]map[string]bool{
	"runs": set(
		"id", "name", "algorithm", "status", "config_hash", "house_rules_hash",
		"config_schema_version", "engine_commit_hash", "origin_host",
		"executed_on", "best_metric_name", "best_metric_value",
		"best_metric_iter", "tags", "notes", "parent_run_id", "created_at",
		"updated_at",
	),
	"config_snapshots": set(
		"id", "run_id", "config_yaml", "config_hash", "created_at",
	),
	"checkpoints": set(
		"id", "run_id", "iteration", "file_path", "file_size_bytes",
		"created_at", "is_best", "is_retained", "compressed",
	),
	"eval_results": set(
		"id", "run_id", "checkpoint_id", "iteration", "baseline", "win_rate",
		"ci_low", "ci_high", "games_played", "p0_wins", "p1_wins", "ties",
		"avg_game_turns", "t1_cambia_rate", "avg_score_margin", "adv_loss",
		"strat_loss", "seat_balanced", "selection_mode", "crn_seed",
		"run_seed", "seat_scheme", "policy_errors", "engine_errors",
		"belief_protocol", "served_policy", "timestamp",
	),
	"head_to_head": set(
		"id", "run_id", "iter_a", "iter_b", "label", "a_wins", "b_wins",
		"ties", "a_win_rate", "avg_game_turns", "timestamp",
	),
	"harness_sync": set(
		"run_name", "origin_host", "last_sync_at", "last_status", "unpullable",
		"last_error",
	),
	"harness_reflection": set(
		"origin_host", "run_name", "last_reflected_state", "item_handle",
		"last_reflected_at",
	),
}

// runDBChildTables lists every runDBSchema table but runs, in row-count-cap
// check order.
var runDBChildTables = []string{
	"config_snapshots", "checkpoints", "eval_results", "head_to_head",
	"harness_sync", "harness_reflection",
}

// allowedStatus mirrors cfr/src/harness/reconciler.py's _ALLOWED_STATUS at
// the commit this file was written against. The two lists are independent by
// design (D61): the shared fixture corpus is what catches drift between
// them, not a shared source of truth.
var allowedStatus = set(
	"created", "queued", "preparing", "starting", "running", "stopping",
	"stopped", "crashed", "canceled", "cancelled", "failed", "completed",
	"finished", "done", "interrupted",
)

func set(items ...string) map[string]bool {
	m := make(map[string]bool, len(items))
	for _, item := range items {
		m[item] = true
	}
	return m
}

// Verdict is the result of validating a node-authored run_db.sqlite journal.
type Verdict struct {
	// Accepted is true only when every D55 check passed.
	Accepted bool
	// Reason is empty when Accepted, else one of the Reason* constants above.
	Reason string
	// Detail is a human-readable explanation for the coordinator's own logs.
	// It is never a trusted echo of file content beyond short identifiers
	// (row counts, table/column names already confirmed against the fixed
	// schema, the rejected status string capped in length below).
	Detail string
}

// RunDBConfig bounds a single Validate call. The zero value selects every
// Default* constant above.
type RunDBConfig struct {
	MaxBytes        int64
	QueryTimeout    time.Duration
	MaxRowsPerTable int
}

func (c RunDBConfig) withDefaults() RunDBConfig {
	if c.MaxBytes <= 0 {
		c.MaxBytes = DefaultMaxRunDBBytes
	}
	if c.QueryTimeout <= 0 {
		c.QueryTimeout = DefaultQueryTimeout
	}
	if c.MaxRowsPerTable <= 0 {
		c.MaxRowsPerTable = DefaultMaxRowsPerTable
	}
	return c
}

func reject(reason, detail string) Verdict {
	return Verdict{Accepted: false, Reason: reason, Detail: detail}
}

// Validate opens path as a node-authored run_db.sqlite journal, strictly
// read-only, and checks it against the D55 contract: a size cap enforced
// before the file is ever opened; PRAGMA integrity_check; a schema that is a
// subset of runDBSchema; exactly one `runs` row whose name equals
// expectedName (the job id for a train/measure job, or the resolved
// spec.target for an evaluate job, D64); a `runs.status` in allowedStatus;
// and row-count caps on every other table. It applies DefaultMaxRunDBBytes,
// DefaultQueryTimeout, and DefaultMaxRowsPerTable; use ValidateWithConfig for
// a non-default cap.
//
// Validate never writes to path: it opens with the mode=ro&immutable=1 URI
// parameters and issues only SELECT/PRAGMA statements. It never executes,
// imports, or unpickles anything the file contains; the file is parsed only
// as SQLite content through database/sql and modernc.org/sqlite.
//
// A non-nil error return means Validate could not evaluate path at all (for
// example the file does not exist), which signals a caller-side problem: the
// caller is expected to point Validate at a blob it already holds and has
// already digest-verified. Every content-level problem with the journal
// itself, including a file that is not a valid SQLite database, is reported
// as a rejected Verdict with a nil error, never as an error return.
func Validate(path, expectedName string) (Verdict, error) {
	return ValidateWithConfig(path, expectedName, RunDBConfig{})
}

// ValidateWithConfig is Validate with an explicit RunDBConfig. Production
// callers pass the env-derived RUNNERD_NASHNET_MAX_RUNDB_BYTES cap here; this
// package's own tests use it to keep the oversized fixture small and fast.
func ValidateWithConfig(path, expectedName string, cfg RunDBConfig) (Verdict, error) {
	cfg = cfg.withDefaults()

	info, err := os.Stat(path)
	if err != nil {
		return Verdict{}, fmt.Errorf("stat %s: %w", path, err)
	}
	if info.Size() > cfg.MaxBytes {
		return reject(ReasonSizeCap, fmt.Sprintf(
			"%d bytes exceeds the %d byte cap", info.Size(), cfg.MaxBytes,
		)), nil
	}

	// mode=ro plus immutable=1 are SQLite's own URI parameters (recognized
	// because the DSN is prefixed "file:" and the driver opens with
	// SQLITE_OPEN_URI), not something this package parses; the path is never
	// interpolated into a query, only into this DSN, and quarantine paths are
	// built from names already restricted to a safe charset by the caller
	// (D49), so no URI-escaping is needed here (rundb_status.go precedent).
	dsn := "file:" + path + "?mode=ro&immutable=1"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return reject(ReasonIntegrityCheck, "opening the journal: "+err.Error()), nil
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), cfg.QueryTimeout)
	defer cancel()

	if v := checkIntegrity(ctx, db); !v.Accepted {
		return v, nil
	}
	present, v := checkSchema(ctx, db)
	if !v.Accepted {
		return v, nil
	}
	if v := checkIdentity(ctx, db, expectedName); !v.Accepted {
		return v, nil
	}
	if v := checkRowCounts(ctx, db, present, cfg.MaxRowsPerTable); !v.Accepted {
		return v, nil
	}
	return Verdict{Accepted: true}, nil
}

func checkIntegrity(ctx context.Context, db *sql.DB) Verdict {
	rows, err := db.QueryContext(ctx, "PRAGMA integrity_check")
	if err != nil {
		return reject(ReasonIntegrityCheck, "running integrity_check: "+err.Error())
	}
	defer rows.Close()

	var lines []string
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			return reject(ReasonIntegrityCheck, "reading integrity_check output: "+err.Error())
		}
		lines = append(lines, line)
	}
	if err := rows.Err(); err != nil {
		return reject(ReasonIntegrityCheck, "iterating integrity_check output: "+err.Error())
	}
	if len(lines) != 1 || lines[0] != "ok" {
		return reject(ReasonIntegrityCheck, fmt.Sprintf(
			"integrity_check reported %d issue(s)", len(lines),
		))
	}
	return Verdict{Accepted: true}
}

// checkSchema validates every table present is a known one with only known
// columns, and returns the set of known tables actually present so the
// row-count pass does not re-query sqlite_master.
func checkSchema(ctx context.Context, db *sql.DB) (map[string]bool, Verdict) {
	rows, err := db.QueryContext(ctx, "SELECT name FROM sqlite_master WHERE type='table'")
	if err != nil {
		return nil, reject(ReasonSchema, "reading sqlite_master: "+err.Error())
	}
	var tableNames []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			rows.Close()
			return nil, reject(ReasonSchema, "reading table name: "+err.Error())
		}
		tableNames = append(tableNames, name)
	}
	rowsErr := rows.Err()
	rows.Close()
	if rowsErr != nil {
		return nil, reject(ReasonSchema, "iterating sqlite_master: "+rowsErr.Error())
	}

	present := make(map[string]bool)
	for _, name := range tableNames {
		// sqlite_sequence is created implicitly by SQLite itself for any
		// AUTOINCREMENT column (every table in runDBSchema has one); it is
		// not a node-authored table and is not part of _DDL.
		if name == "sqlite_sequence" || strings.HasPrefix(name, "sqlite_") {
			continue
		}
		cols, known := runDBSchema[name]
		if !known {
			return nil, reject(ReasonSchema, fmt.Sprintf("unexpected table %q", name))
		}
		present[name] = true

		colRows, err := db.QueryContext(ctx, fmt.Sprintf("PRAGMA table_info(%s)", quoteIdent(name)))
		if err != nil {
			return nil, reject(ReasonSchema, fmt.Sprintf("reading columns of %q: %v", name, err))
		}
		for colRows.Next() {
			var cid int
			var colName, colType string
			var notNull, pk int
			var dflt interface{}
			if err := colRows.Scan(&cid, &colName, &colType, &notNull, &dflt, &pk); err != nil {
				colRows.Close()
				return nil, reject(ReasonSchema, fmt.Sprintf("scanning columns of %q: %v", name, err))
			}
			if !cols[colName] {
				colRows.Close()
				return nil, reject(ReasonSchema, fmt.Sprintf("table %q has unexpected column %q", name, colName))
			}
		}
		colErr := colRows.Err()
		colRows.Close()
		if colErr != nil {
			return nil, reject(ReasonSchema, fmt.Sprintf("iterating columns of %q: %v", name, colErr))
		}
	}

	if !present["runs"] {
		return nil, reject(ReasonSchema, "the runs table is missing")
	}
	return present, Verdict{Accepted: true}
}

func checkIdentity(ctx context.Context, db *sql.DB, expectedName string) Verdict {
	var count int
	if err := db.QueryRowContext(ctx, "SELECT COUNT(*) FROM runs").Scan(&count); err != nil {
		return reject(ReasonIdentity, "counting runs rows: "+err.Error())
	}
	if count != 1 {
		return reject(ReasonIdentity, fmt.Sprintf("runs has %d row(s), want exactly 1", count))
	}

	var name, status string
	if err := db.QueryRowContext(ctx, "SELECT name, status FROM runs").Scan(&name, &status); err != nil {
		return reject(ReasonIdentity, "reading the runs row: "+err.Error())
	}
	if name != expectedName {
		return reject(ReasonIdentity, fmt.Sprintf("runs.name %q does not match the expected identity %q", name, expectedName))
	}
	if !allowedStatus[status] {
		return reject(ReasonEnum, fmt.Sprintf("runs.status %q is not a known value", truncateForLog(status)))
	}
	return Verdict{Accepted: true}
}

func checkRowCounts(ctx context.Context, db *sql.DB, present map[string]bool, maxRows int) Verdict {
	for _, table := range runDBChildTables {
		if !present[table] {
			continue
		}
		var n int
		if err := db.QueryRowContext(ctx, fmt.Sprintf("SELECT COUNT(*) FROM %s", quoteIdent(table))).Scan(&n); err != nil {
			return reject(ReasonRowCount, fmt.Sprintf("counting %s: %v", table, err))
		}
		if n > maxRows {
			return reject(ReasonRowCount, fmt.Sprintf("%s has %d rows, exceeds the %d row cap", table, n, maxRows))
		}
	}
	return Verdict{Accepted: true}
}

// quoteIdent double-quotes a SQLite identifier for use inside a PRAGMA or
// SELECT COUNT(*) statement, where database/sql offers no bound-parameter
// form for identifiers. In practice every call site passes a name already
// confirmed to be one of runDBSchema's fixed literal keys (checkSchema
// rejects anything else before this is ever reached), so this is
// defense-in-depth rather than a reachable escape.
func quoteIdent(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}

// truncateForLog caps a value pulled from the journal before it is embedded
// in a Detail string, so a pathological status value cannot inflate a log
// line without bound.
func truncateForLog(s string) string {
	const maxLen = 128
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "...(truncated)"
}
