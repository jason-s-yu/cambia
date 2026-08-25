// Run-db lifecycle read (cambia-655): the exit-status oracle for a reattached
// job.
//
// A job adopted at Reconcile was forked by a previous daemon incarnation, so
// this daemon can never waitpid it and its true exit code is unrecoverable. The
// run's own journal (runs/<name>/run_db.sqlite, design 4.2) is the next best
// witness: the trainer flips its `runs` row to `completed` on a clean end, and
// the stale sweep flips an abandoned one to `interrupted`. Reading it lets a
// reattached job that finished normally still satisfy a dependent's success
// gate instead of being recorded as a crash.
package harness

import (
	"context"
	"database/sql"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite" // pure-Go driver: keeps the runnerd static build static (no cgo)
)

// runDBStatusCompleted is the `runs.status` value the trainer writes on a clean
// end (cfr/src/cfr/deep_trainer.py update_run_status(..., "completed")). It is
// the only value that certifies a zero exit; every other value (`running` left
// behind by an abrupt death, `interrupted` from the stale sweep, `created`) is
// not a clean finish.
const runDBStatusCompleted = "completed"

// runDBQueryTimeout bounds the journal read so a locked or pathological
// run_db.sqlite cannot wedge a reattach watcher's finalize.
const runDBQueryTimeout = 5 * time.Second

// runDBRunStatus returns the `runs.status` recorded in runDir/run_db.sqlite for
// name, or "" if the journal is absent, unreadable, locked, or has no row. It is
// read-only (mode=ro) and never creates the file: a job that never got far
// enough to register a run must not gain a journal as a side effect of being
// finalized.
//
// The lookup mirrors the client-side reader (cfr/src/harness/pull.py
// read_run_status): by name first, then the newest row in the file. The
// fallback matters because the per-run journal holds exactly one run and the
// trainer may have registered it under a name that differs from the run dir
// (an evaluate job writes into the evaluated run's journal, design 4.2).
func runDBRunStatus(runDir, name string) string {
	dbPath := filepath.Join(runDir, "run_db.sqlite")
	dsn := "file:" + dbPath + "?mode=ro&_pragma=busy_timeout(2000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return ""
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), runDBQueryTimeout)
	defer cancel()

	var status string
	err = db.QueryRowContext(ctx, "SELECT status FROM runs WHERE name = ?", name).Scan(&status)
	if err == nil {
		return status
	}
	if err = db.QueryRowContext(ctx,
		"SELECT status FROM runs ORDER BY updated_at DESC LIMIT 1").Scan(&status); err != nil {
		return ""
	}
	return status
}
