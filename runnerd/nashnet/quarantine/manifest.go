package quarantine

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/jason-s-yu/cambia/runnerd/pathguard"
)

// ManifestVersion is the only manifest_version this coordinator accepts (D51).
const ManifestVersion = 1

// Entry is one file a manifest names. Fields are declared in lexicographic key
// order because the canonical form of a folded manifest is what gets hashed
// into the fast-forward chain, and Go marshals struct fields in declaration
// order. MTime is Unix nanoseconds, an integer like every other numeric field:
// the canonical form carries no floats.
type Entry struct {
	Digest string `json:"digest"`
	MTime  int64  `json:"mtime"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
}

// CommitRequest is the body of POST /nashnet/leases/{lease}/manifest (D51).
type CommitRequest struct {
	Deletes         []string `json:"deletes"`
	Entries         []Entry  `json:"entries"`
	Final           bool     `json:"final"`
	LeaseEpoch      int64    `json:"lease_epoch"`
	ManifestVersion int      `json:"manifest_version"`
	Parent          string   `json:"parent"`
	Seq             int64    `json:"seq"`
}

// Folded is the full manifest state at a seq: the parent folded with this
// commit's accepted entries and deletes. It is the document that is
// canonicalized and hashed, and its digest is the parent the node's next commit
// must carry. Field order is lexicographic for the same reason as Entry.
type Folded struct {
	Entries         []Entry `json:"entries"`
	Final           bool    `json:"final"`
	JobID           string  `json:"job_id"`
	LeaseEpoch      int64   `json:"lease_epoch"`
	LeaseID         string  `json:"lease_id"`
	ManifestVersion int     `json:"manifest_version"`
	Parent          string  `json:"parent"`
	Seq             int64   `json:"seq"`
}

// Head is runs/<job>/.nashnet/current.json: the folded manifest plus the
// bookkeeping the fast-forward fence needs. It is coordinator-authored, is a
// rejected manifest path (D52), and survives a restart, so last_seq,
// last_manifest_digest, and the idempotent-replay record all come off disk
// rather than out of memory.
type Head struct {
	Digest string `json:"digest"`
	Folded Folded `json:"folded"`
	// LastRequestSHA256 and LastResponse record the accepted commit at
	// Folded.Seq so a re-post of the same seq with the same body returns the
	// recorded response unchanged (D51 step 1).
	LastRequestSHA256 string          `json:"last_request_sha256"`
	LastResponse      *CommitResponse `json:"last_response,omitempty"`
	UpdatedAt         int64           `json:"updated_at"`
}

// Rejection is one per-entry refusal. A rejection is typed and per entry, never
// a session kill (D52). Detail is optional and carries the journal validator's
// own reason and explanation behind the fixed rundb_invalid code, so the commit
// response and the receipt name which D55 check failed instead of swallowing
// it; every other rejection leaves it empty.
type Rejection struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
	Detail string `json:"detail,omitempty"`
}

// CommitResponse is the body of an accepted commit (D51 step 7).
type CommitResponse struct {
	Seq          int64       `json:"seq"`
	Digest       string      `json:"digest"`
	Promoted     []string    `json:"promoted"`
	Deleted      []string    `json:"deleted"`
	Rejected     []Rejection `json:"rejected"`
	Unrecognized []string    `json:"unrecognized,omitempty"`
	Degraded     bool        `json:"degraded,omitempty"`
}

// Rejection reason codes. Every item of the D52 rejection list and every
// per-entry cap has one, so a receipt and a commit response name why a path was
// refused rather than reporting a bare count.
const (
	ReasonPathEmpty        = "path_empty"
	ReasonPathAbsolute     = "path_absolute"
	ReasonPathTraversal    = "path_traversal"
	ReasonPathBackslash    = "path_backslash"
	ReasonPathNUL          = "path_nul"
	ReasonPathNotUTF8      = "path_not_utf8"
	ReasonPathDirectory    = "path_directory"
	ReasonPathParentNotDir = "parent_not_a_directory"
	ReasonPathReserved     = "path_reserved"
	ReasonSegmentTooLong   = "segment_too_long"
	ReasonPathTooLong      = "path_too_long"
	ReasonPathTooDeep      = "path_too_deep"
	ReasonPathEscapes      = "path_escapes_run_dir"
	ReasonInvalidDigest    = "invalid_digest"
	ReasonSizeMismatch     = "size_mismatch"
	ReasonFileTooLarge     = "file_too_large"
	ReasonLeaseBytes       = "lease_bytes_exceeded"
	ReasonDuplicatePath    = "duplicate_path"
	ReasonRunDBInvalid     = "rundb_invalid"
)

// reservedExact are coordinator-authored files and directories a node may never
// name (D52). The directory names are listed too, so an entry naming the
// directory itself as a file cannot reach a rename over it.
var reservedExact = map[string]bool{
	"process.json": true,
	"jobspec.json": true,
	"lease.json":   true,
	"env.json":     true,
	NashnetDir:     true,
	TmpDir:         true,
	"logs":         true,
	"reservoir":    true,
}

// reservedPrefixes are subtrees the coordinator owns. logs/** is the one
// allowlist finding kept from the r1 review: without it a manifest entry could
// rename over the file harness/logstream.go:192 serves.
var reservedPrefixes = []string{
	NashnetDir + "/",
	TmpDir + "/",
	"logs/",
	"reservoir/",
}

// reservedSuffixes cover the staging names and the journal siblings. Exactly
// one journal file ever crosses the wire (D22), so -wal, -shm, and -journal are
// rejected outright.
var reservedSuffixes = []string{
	TmpSuffix,
	".tmp",
	"-wal",
	"-shm",
	"-journal",
}

// knownLayout is the v1.0 run-dir layout. An entry outside it is accepted and
// counted as unrecognized in the receipt, so a new artifact is visible without
// being blocked; a promotable-basename allowlist was rejected because it caps
// what any run may ever return (D52).
var knownLayout = map[string]bool{
	"config.yaml":        true,
	"env.node.json":      true,
	"metrics.jsonl":      true,
	"run_meta.json":      true,
	"eval_summary.jsonl": true,
	"resume_state.json":  true,
	RunDBPath:            true,
}

// recognized reports whether a path is part of the known v1.0 layout.
func recognized(rel string) bool {
	if knownLayout[rel] {
		return true
	}
	dir, base := path2Dir(rel)
	return dir == "snapshots" && strings.HasSuffix(base, ".pt")
}

func path2Dir(rel string) (dir, base string) {
	i := strings.LastIndex(rel, "/")
	if i < 0 {
		return "", rel
	}
	return rel[:i], rel[i+1:]
}

// validateRelPath applies the D52 rejection list to one manifest path and
// returns the reason code, or "" when the path is acceptable. Cheap pure checks
// run before pathguard.Resolve, which is the only step that touches the
// filesystem: the resolved containment check of pathguard.go:114 is what closes
// a symlink planted inside a run dir.
func validateRelPath(runDir, rel string, lim Limits) string {
	if rel == "" {
		return ReasonPathEmpty
	}
	if strings.ContainsRune(rel, 0) {
		return ReasonPathNUL
	}
	if !utf8.ValidString(rel) {
		return ReasonPathNotUTF8
	}
	if strings.ContainsRune(rel, '\\') {
		return ReasonPathBackslash
	}
	if err := pathguard.CheckRel(rel); err != nil {
		switch {
		case errors.Is(err, pathguard.ErrEmpty):
			return ReasonPathEmpty
		case errors.Is(err, pathguard.ErrParentTraversal):
			return ReasonPathTraversal
		default:
			return ReasonPathAbsolute
		}
	}
	if strings.HasSuffix(rel, "/") {
		return ReasonPathDirectory
	}
	if len(rel) > lim.MaxPathBytes {
		return ReasonPathTooLong
	}
	segs := strings.Split(rel, "/")
	if len(segs) > lim.MaxDepth {
		return ReasonPathTooDeep
	}
	for _, seg := range segs {
		if seg == "" || seg == "." {
			// An empty or dot segment is a directory shape, not a file path.
			return ReasonPathDirectory
		}
		if len(seg) > lim.MaxSegmentBytes {
			return ReasonSegmentTooLong
		}
	}
	if reservedPath(rel) {
		return ReasonPathReserved
	}
	target, err := pathguard.Resolve(runDir, rel)
	if err != nil {
		return ReasonPathEscapes
	}
	return checkPlacement(runDir, rel, target)
}

// checkPlacement rejects an entry whose promotion could not be a per-file
// rename: one whose target is an existing directory, and one whose parent chain
// runs through an existing regular file. Both are per-entry rejections rather
// than batch failures, so a node that named a directory as a file cannot wedge
// every later commit for its own job behind a materialize error.
func checkPlacement(runDir, rel, target string) string {
	if fi, err := os.Lstat(target); err == nil && fi.IsDir() {
		return ReasonPathDirectory
	}
	segs := strings.Split(rel, "/")
	for i := 1; i < len(segs); i++ {
		ancestor := filepath.Join(runDir, filepath.Join(segs[:i]...))
		fi, err := os.Lstat(ancestor)
		if err != nil {
			// A missing ancestor is created by the coordinator at promotion.
			continue
		}
		if !fi.IsDir() {
			return ReasonPathParentNotDir
		}
	}
	return ""
}

// ReservedPath reports whether rel names something the coordinator authors and
// a node may therefore never offer in a manifest (D52). It is exported so the
// node agent filters its own run-dir scan against this list rather than
// keeping a second copy that could drift from the enforcement here.
func ReservedPath(rel string) bool { return reservedPath(rel) }

// reservedPath reports whether rel names something the coordinator authors.
func reservedPath(rel string) bool {
	if reservedExact[rel] {
		return true
	}
	for _, p := range reservedPrefixes {
		if strings.HasPrefix(rel, p) {
			return true
		}
	}
	for _, sfx := range reservedSuffixes {
		if strings.HasSuffix(rel, sfx) {
			return true
		}
	}
	return false
}

// fold applies a commit's accepted deletes and entries to the parent state and
// returns the new full manifest with entries sorted by path, so the canonical
// form of a given state is independent of the order the node sent it in.
func fold(parent Folded, l Lease, req CommitRequest, entries []Entry, deletes []string) Folded {
	byPath := make(map[string]Entry, len(parent.Entries)+len(entries))
	for _, e := range parent.Entries {
		byPath[e.Path] = e
	}
	for _, d := range deletes {
		delete(byPath, d)
	}
	for _, e := range entries {
		byPath[e.Path] = e
	}
	out := make([]Entry, 0, len(byPath))
	for _, e := range byPath {
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return Folded{
		Entries:         out,
		Final:           req.Final,
		JobID:           l.JobID,
		LeaseEpoch:      l.Epoch,
		LeaseID:         l.LeaseID,
		ManifestVersion: ManifestVersion,
		Parent:          req.Parent,
		Seq:             req.Seq,
	}
}

// canonicalJSON encodes v with sorted keys (by field declaration order), no
// insignificant whitespace, UTF-8 rather than escaped HTML, and integers only.
// It is the form the manifest digest is taken over.
func canonicalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// digestOf returns the hex sha256 of b.
func digestOf(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// readHead loads runs/<job>/.nashnet/current.json. A missing file is the
// genesis head: seq 0 and an empty parent digest, which is what the node's
// first commit must carry.
func readHead(runDir string) (Head, error) {
	data, err := os.ReadFile(filepath.Join(runDir, NashnetDir, HeadFile))
	if err != nil {
		if os.IsNotExist(err) {
			return Head{Folded: Folded{Entries: []Entry{}, ManifestVersion: ManifestVersion}}, nil
		}
		return Head{}, err
	}
	var h Head
	if err := json.Unmarshal(data, &h); err != nil {
		return Head{}, err
	}
	if h.Folded.Entries == nil {
		h.Folded.Entries = []Entry{}
	}
	return h, nil
}

// ReadHead returns the folded manifest head for a job, for the manifest GET
// route and for serving artifacts from the manifest (D58).
func (s *Store) ReadHead(jobID string) (Head, error) {
	runDir, err := s.RunDir(jobID)
	if err != nil {
		return Head{}, err
	}
	return readHead(runDir)
}

// writeHead rewrites current.json with the temp-fsync-rename pattern of
// procmgr/state.go:74 and fsyncs the directory holding it.
func writeHead(runDir string, h Head) error {
	dir := filepath.Join(runDir, NashnetDir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(h, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp := filepath.Join(dir, HeadFile+".tmp")
	final := filepath.Join(dir, HeadFile)
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		os.Remove(tmp)
		return err
	}
	return fsyncDir(dir)
}
