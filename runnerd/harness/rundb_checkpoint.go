// Run-db WAL-checkpoint endpoint (cambia-295 item 5).
//
// The per-run runs/<id>/run_db.sqlite is the artifact wire format the client
// pull loop rsyncs down (design 4.2); the pull loop already copies the -wal
// and -shm siblings as a fallback. In WAL mode, though, recently committed
// rows can sit only in the -wal file while the main db file lags behind. This
// endpoint folds the WAL into the main file before the client pulls, so the
// synced main file is current on its own.
package harness

import (
	"database/sql"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite" // pure-Go driver: keeps the runnerd static build static (no cgo)

	"github.com/jason-s-yu/cambia/runnerd/pathguard"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// handleRunDBCheckpoint is POST /harness/jobs/{id}/rundb-checkpoint. It runs
// PRAGMA wal_checkpoint(TRUNCATE) against runs/<id>/run_db.sqlite if that file
// exists. id is untrusted (design 5.7): validated by the same allowlist as
// every other job-id route, then resolved through pathguard so a symlink
// planted inside the run dir cannot smuggle the checkpoint target outside the
// runs directory -- the same containment treatment handleCreateJob gives
// checkpoint/target/warm_start spec paths.
func (s *Server) handleRunDBCheckpoint(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	dbPath, err := pathguard.Resolve(s.runsDir, filepath.Join(id, "run_db.sqlite"))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_path", err.Error())
		return
	}
	if fi, statErr := os.Stat(dbPath); statErr != nil || fi.IsDir() {
		writeJSONError(w, http.StatusNotFound, "not_found", "run_db.sqlite not found")
		return
	}

	result, err := checkpointRunDB(dbPath)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "checkpoint_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"job_id":       id,
		"busy":         result.busy,
		"log_frames":   result.log,
		"checkpointed": result.checkpointed,
	})
}

// walCheckpointResult mirrors the three integer columns "PRAGMA
// wal_checkpoint(MODE)" returns: busy (1 if the checkpoint could not run to
// completion because of a conflicting lock, e.g. a long-lived reader), log
// (WAL frames present), checkpointed (frames actually moved into the main db
// file). A TRUNCATE checkpoint that completes with busy=0 truncates the -wal
// file to zero length.
type walCheckpointResult struct {
	busy         int
	log          int
	checkpointed int
}

// checkpointRunDB opens dbPath with a bounded busy timeout and runs "PRAGMA
// wal_checkpoint(TRUNCATE)". The busy_timeout gives a concurrent writer (the
// training process still appending to run_db.sqlite) a short window to
// release its lock rather than failing the checkpoint outright.
func checkpointRunDB(dbPath string) (walCheckpointResult, error) {
	dsn := "file:" + dbPath + "?_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return walCheckpointResult{}, fmt.Errorf("open %s: %w", dbPath, err)
	}
	defer db.Close()

	var res walCheckpointResult
	row := db.QueryRow("PRAGMA wal_checkpoint(TRUNCATE);")
	if scanErr := row.Scan(&res.busy, &res.log, &res.checkpointed); scanErr != nil {
		return walCheckpointResult{}, fmt.Errorf("wal_checkpoint: %w", scanErr)
	}
	return res, nil
}
