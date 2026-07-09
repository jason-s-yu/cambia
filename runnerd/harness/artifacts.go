package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Artifact is one entry of the run-dir manifest: bytes move by rsync out of
// band, this is the index (design 2.4).
type Artifact struct {
	Path   string `json:"path"`   // relative to the run dir
	Size   int64  `json:"size"`   // bytes
	SHA256 string `json:"sha256"` // hex digest
	MTime  string `json:"mtime"`  // RFC3339
}

// handleArtifacts is GET /harness/jobs/{id}/artifacts. It walks runs/<id>/ and
// returns a manifest of every regular file (path relative to the run dir, size,
// sha256, mtime), sorted by path.
func (s *Server) handleArtifacts(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	runDir := filepath.Join(s.runsDir, id)
	if fi, err := os.Stat(runDir); err != nil || !fi.IsDir() {
		writeJSONError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}

	var manifest []Artifact
	err := filepath.WalkDir(runDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries rather than aborting the walk
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		rel, rerr := filepath.Rel(runDir, path)
		if rerr != nil {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		sum, serr := sha256File(path)
		if serr != nil {
			return nil
		}
		manifest = append(manifest, Artifact{
			Path:   rel,
			Size:   info.Size(),
			SHA256: sum,
			MTime:  info.ModTime().UTC().Format("2006-01-02T15:04:05Z07:00"),
		})
		return nil
	})
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "walk_failed", err.Error())
		return
	}
	sort.Slice(manifest, func(i, j int) bool { return manifest[i].Path < manifest[j].Path })
	writeJSON(w, http.StatusOK, map[string]any{"job_id": id, "artifacts": manifest})
}

// sha256File returns the hex sha256 of a file's contents.
func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
