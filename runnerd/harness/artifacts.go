package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
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
	if manifest, ok := s.manifestArtifacts(id, runDir); ok {
		writeJSON(w, http.StatusOK, map[string]any{"job_id": id, "artifacts": manifest})
		return
	}

	var manifest []Artifact
	err := filepath.WalkDir(runDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries rather than aborting the walk
		}
		if d.IsDir() {
			// .nashnet/ is the coordinator's own manifest state, not an
			// artifact: it is a rejected manifest path (D52), the client never
			// pulls it, and skipping it keeps the walk-derived listing equal to
			// the manifest-derived one (D58). No pre-pool run dir has it.
			if d.Name() == nashnetStateDir && path != runDir {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
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

// nashnetStateDir is the run-dir subtree holding the coordinator's folded
// manifest head (D49), named by the quarantine store that owns it. It is
// coordinator bookkeeping rather than a run artifact, so neither listing
// returns it.
const nashnetStateDir = quarantine.NashnetDir

// coordinatorAuthoredPaths are the files a manifest can never contain, because
// D52 makes each one a rejected manifest path, and which the walk-derived
// listing has always returned. The manifest-derived listing unions them in so
// the response set is unchanged, on an embedded run as well as a remote one
// (D58).
var coordinatorAuthoredPaths = []string{
	"process.json", "jobspec.json", "env.json", "env.node.json", "lease.json",
	"logs/training.log",
}

// manifestArtifacts serves the artifacts listing from the folded manifest
// instead of rehashing every file in the run dir on every call (D58): the
// promoted digests are the ones the node proved on upload. It returns ok=false
// for a run dir with no manifest, which keeps the walk for every v1.0 run.
func (s *Server) manifestArtifacts(id, runDir string) ([]Artifact, bool) {
	if s.pool == nil {
		return nil, false
	}
	head, err := s.pool.quar.ReadHead(id)
	if err != nil || head.Digest == "" {
		return nil, false
	}
	seen := map[string]bool{}
	manifest := make([]Artifact, 0, len(head.Folded.Entries)+len(coordinatorAuthoredPaths))
	for _, e := range head.Folded.Entries {
		if seen[e.Path] {
			continue
		}
		seen[e.Path] = true
		manifest = append(manifest, Artifact{
			Path:   e.Path,
			Size:   e.Size,
			SHA256: e.Digest,
			MTime:  time.Unix(0, e.MTime).UTC().Format(time.RFC3339),
		})
	}
	for _, rel := range coordinatorAuthoredPaths {
		if seen[rel] {
			continue
		}
		path := filepath.Join(runDir, filepath.FromSlash(rel))
		info, serr := os.Stat(path)
		if serr != nil || !info.Mode().IsRegular() {
			continue
		}
		sum, herr := sha256File(path)
		if herr != nil {
			continue
		}
		seen[rel] = true
		manifest = append(manifest, Artifact{
			Path: rel, Size: info.Size(), SHA256: sum,
			MTime: info.ModTime().UTC().Format(time.RFC3339),
		})
	}
	sort.Slice(manifest, func(i, j int) bool { return manifest[i].Path < manifest[j].Path })
	return manifest, true
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
