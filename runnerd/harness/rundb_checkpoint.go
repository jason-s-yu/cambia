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
	"net/http"
	"os"
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
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

	result, err := nodeagent.FoldRunDB(dbPath)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "checkpoint_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"job_id":       id,
		"busy":         result.Busy,
		"log_frames":   result.Log,
		"checkpointed": result.Checkpointed,
	})
}
