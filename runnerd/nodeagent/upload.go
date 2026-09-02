package nodeagent

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"

	_ "modernc.org/sqlite"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// runDBName is the per-run journal, the one node-authored format the
// coordinator parses (D55) and the entry every commit materializes last (D57).
const runDBName = quarantine.RunDBPath

// envJSONName is the provenance record the node's own Prepare writes. The
// coordinator authors the run's env.json itself (D23), so the node offers its
// own copy under the promoted name env.node.json; env.json is a rejected
// manifest path.
const (
	envJSONName     = "env.json"
	envNodeJSONName = "env.node.json"
)

// localFile is one candidate manifest entry found in the node's run dir.
type localFile struct {
	// rel is the path as the manifest names it, which is the run-dir-relative
	// path except for the env.json to env.node.json rename.
	rel string
	abs string
	fi  os.FileInfo
}

// uploader drives the diff-probe-upload-commit loop for one lease (D50, D51).
// It holds no lock of its own: the agent runs exactly one uploader per lease
// and calls Sync from the job's own goroutine.
type uploader struct {
	client  NodeTransport
	leaseID string
	token   string
	runDir  string
	index   *Index
	policy  nashnet.Policy
	log     *log.Logger
	// rejectedRunDB counts consecutive rundb_invalid rejections. The journal
	// is not retried on a rejection, so the count is reported rather than
	// acted on: three consecutive ones mark the lease degraded coordinator
	// side (D55).
	rejectedRunDB int
}

// Sync runs one commit cycle: fold the journal's WAL, scan the run dir, diff
// against the coordinator's head, upload what it lacks, and commit. final
// marks the commit that closes the lease's artifact stream, which the result
// route requires (D6).
func (u *uploader) Sync(ctx context.Context, final bool) (quarantine.CommitResponse, error) {
	if err := foldRunDB(filepath.Join(u.runDir, runDBName)); err != nil {
		// A journal the node cannot fold is still worth offering: the
		// coordinator validates the bytes it receives and rejects them per
		// entry, so a fold failure degrades to a possibly-rejected entry
		// rather than to a stalled upload.
		u.logf("wal fold: %v", err)
	}
	files, err := scanRunDir(u.runDir)
	if err != nil {
		return quarantine.CommitResponse{}, err
	}
	entries, err := u.entries(files)
	if err != nil {
		return quarantine.CommitResponse{}, err
	}
	_ = u.index.Save()

	// Two attempts: the second one re-reads the head after a fast-forward
	// fence failure or re-uploads what the coordinator says it lacks. A third
	// failure is reported to the caller, which retries at the next tick.
	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		head, herr := u.client.ManifestHead(ctx, u.leaseID, u.token)
		if herr != nil {
			return quarantine.CommitResponse{}, herr
		}
		changed, deletes := diff(head, entries)
		if len(changed) == 0 && len(deletes) == 0 && !final {
			return quarantine.CommitResponse{Seq: head.Seq, Digest: head.Digest}, nil
		}
		if err := u.push(ctx, changed); err != nil {
			return quarantine.CommitResponse{}, err
		}
		req := quarantine.CommitRequest{
			ManifestVersion: quarantine.ManifestVersion,
			Seq:             head.Seq + 1,
			Parent:          head.Digest,
			Entries:         changed,
			Deletes:         deletes,
			Final:           final,
		}
		resp, cerr := u.client.Commit(ctx, u.leaseID, u.token, req)
		if cerr == nil {
			u.note(resp)
			return resp, nil
		}
		if IsFenced(cerr) {
			return quarantine.CommitResponse{}, cerr
		}
		lastErr = cerr
		if api, ok := AsAPIError(cerr); ok {
			switch api.Code {
			case nashnet.CodeBlobsMissing:
				if err := u.pushDigests(ctx, changed, api.Missing); err != nil {
					return quarantine.CommitResponse{}, err
				}
				continue
			case nashnet.CodeManifestOutOfOrder:
				continue
			}
		}
		return quarantine.CommitResponse{}, cerr
	}
	return quarantine.CommitResponse{}, lastErr
}

// entries hashes every scanned file into a manifest entry.
func (u *uploader) entries(files []localFile) ([]quarantine.Entry, error) {
	out := make([]quarantine.Entry, 0, len(files))
	for _, f := range files {
		digest, err := u.index.Digest(f.rel, f.abs, f.fi)
		if err != nil {
			// A file that vanished between the scan and the hash is not an
			// error: it simply is not part of this commit.
			if errors.Is(err, os.ErrNotExist) {
				u.index.Forget(f.rel)
				continue
			}
			return nil, fmt.Errorf("hash %s: %w", f.rel, err)
		}
		out = append(out, quarantine.Entry{
			Path:   f.rel,
			Digest: digest,
			Size:   f.fi.Size(),
			MTime:  f.fi.ModTime().UnixNano(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// push probes for what the coordinator lacks and uploads exactly that.
func (u *uploader) push(ctx context.Context, entries []quarantine.Entry) error {
	if len(entries) == 0 {
		return nil
	}
	digests := make([]string, 0, len(entries))
	seen := map[string]bool{}
	for _, e := range entries {
		if !seen[e.Digest] {
			seen[e.Digest] = true
			digests = append(digests, e.Digest)
		}
	}
	probe, err := u.client.Probe(ctx, u.leaseID, u.token, digests)
	if err != nil {
		return err
	}
	return u.pushDigests(ctx, entries, probe.Want)
}

// pushDigests uploads the named digests, resolving each back to a local file.
func (u *uploader) pushDigests(ctx context.Context, entries []quarantine.Entry, want []string) error {
	if len(want) == 0 {
		return nil
	}
	byDigest := make(map[string]quarantine.Entry, len(entries))
	for _, e := range entries {
		byDigest[e.Digest] = e
	}
	for _, digest := range want {
		e, ok := byDigest[digest]
		if !ok {
			continue
		}
		if err := u.uploadOne(ctx, e); err != nil {
			return err
		}
	}
	return nil
}

// uploadOne uploads one blob, resuming at the offset the coordinator reports.
// A zero-byte artifact has no range to send and goes as a bodyless PATCH with
// total 0.
func (u *uploader) uploadOne(ctx context.Context, e quarantine.Entry) error {
	abs := u.absFor(e.Path)
	if e.Size == 0 {
		_, err := u.client.AppendChunk(ctx, u.leaseID, u.token, e.Digest, 0, 0, 0, nil)
		return err
	}
	offset, err := u.client.BlobOffset(ctx, u.leaseID, u.token, e.Digest)
	if err != nil {
		return err
	}
	if offset >= e.Size {
		return nil
	}
	f, err := os.Open(abs)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return err
	}
	chunk := u.policy.ChunkBytes
	if chunk <= 0 {
		chunk = nashnet.DefaultChunkBytes
	}
	buf := make([]byte, chunk)
	for offset < e.Size {
		n, rerr := io.ReadFull(f, buf[:min64(chunk, e.Size-offset)])
		if n == 0 {
			if rerr == nil || errors.Is(rerr, io.EOF) {
				break
			}
			return rerr
		}
		committed, aerr := u.client.AppendChunk(ctx, u.leaseID, u.token, e.Digest,
			offset, offset+int64(n)-1, e.Size, buf[:n])
		if aerr != nil {
			api, ok := AsAPIError(aerr)
			if ok && api.Code == nashnet.CodeOffsetMismatch && api.Offset != offset {
				// The coordinator holds more or less than the node assumed;
				// seek to its truth and continue rather than restarting.
				offset = api.Offset
				if _, serr := f.Seek(offset, io.SeekStart); serr != nil {
					return serr
				}
				continue
			}
			return aerr
		}
		if committed > offset {
			offset = committed
		} else {
			offset += int64(n)
		}
		if _, serr := f.Seek(offset, io.SeekStart); serr != nil {
			return serr
		}
	}
	return nil
}

// absFor maps a manifest path back to the local file it came from, undoing the
// env.node.json rename.
func (u *uploader) absFor(rel string) string {
	if rel == envNodeJSONName {
		return filepath.Join(u.runDir, envJSONName)
	}
	return filepath.Join(u.runDir, filepath.FromSlash(rel))
}

// note records what the coordinator rejected. A rundb_invalid rejection is
// logged and not retried: the node cannot fix a journal the validator
// refused, and re-offering the same bytes only earns the lease a degraded
// mark.
func (u *uploader) note(resp quarantine.CommitResponse) {
	sawRunDB := false
	for _, r := range resp.Rejected {
		if r.Reason == quarantine.ReasonRunDBInvalid {
			sawRunDB = true
			u.rejectedRunDB++
			u.logf("journal rejected (%s): %s", r.Reason, r.Detail)
			continue
		}
		u.logf("entry rejected %s: %s", r.Path, r.Reason)
	}
	if !sawRunDB {
		u.rejectedRunDB = 0
	}
}

func (u *uploader) logf(format string, args ...any) {
	if u.log == nil {
		return
	}
	u.log.Printf("lease %s: "+format, append([]any{u.leaseID}, args...)...)
}

// diff computes the entries whose digest differs from the coordinator's head
// and the head paths the run dir no longer holds.
func diff(head ManifestHead, local []quarantine.Entry) (changed []quarantine.Entry, deletes []string) {
	remote := make(map[string]string, len(head.Entries))
	for _, e := range head.Entries {
		remote[e.Path] = e.Digest
	}
	present := make(map[string]bool, len(local))
	for _, e := range local {
		present[e.Path] = true
		if remote[e.Path] != e.Digest {
			changed = append(changed, e)
		}
	}
	for _, e := range head.Entries {
		if !present[e.Path] {
			deletes = append(deletes, e.Path)
		}
	}
	sort.Strings(deletes)
	return changed, deletes
}

// scanRunDir walks a run dir and returns every regular file a node may offer.
// Reserved paths are filtered here rather than left to the coordinator, so the
// uploaded set never contains one: the filter is quarantine.ReservedPath, the
// same predicate the coordinator enforces with, so the two cannot drift.
func scanRunDir(runDir string) ([]localFile, error) {
	var out []localFile
	err := filepath.WalkDir(runDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		rel, rerr := filepath.Rel(runDir, path)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if rel == "." {
			return nil
		}
		if d.IsDir() {
			if quarantine.ReservedPath(rel) || quarantine.ReservedPath(rel+"/") {
				return filepath.SkipDir
			}
			return nil
		}
		// Symlinks, sockets, FIFOs, and device nodes are unrepresentable in
		// the protocol; only regular files are offered.
		if !d.Type().IsRegular() {
			return nil
		}
		name := rel
		if rel == envJSONName {
			name = envNodeJSONName
		}
		if quarantine.ReservedPath(name) {
			return nil
		}
		fi, ferr := d.Info()
		if ferr != nil {
			if os.IsNotExist(ferr) {
				return nil
			}
			return ferr
		}
		out = append(out, localFile{rel: name, abs: path, fi: fi})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].rel < out[j].rel })
	return out, nil
}

// foldRunDB runs PRAGMA wal_checkpoint(TRUNCATE) against the node's local
// journal before it is hashed (D22). A blob is immutable once verified, so the
// fold has to happen node-side: exactly one self-contained journal file ever
// crosses the wire, and its -wal, -shm, and -journal siblings are rejected
// manifest paths.
func foldRunDB(dbPath string) error {
	if _, err := os.Stat(dbPath); err != nil {
		return nil
	}
	dsn := "file:" + dbPath + "?_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return fmt.Errorf("open %s: %w", dbPath, err)
	}
	defer db.Close()
	var busy, logFrames, checkpointed int
	row := db.QueryRow("PRAGMA wal_checkpoint(TRUNCATE);")
	if err := row.Scan(&busy, &logFrames, &checkpointed); err != nil {
		return fmt.Errorf("wal_checkpoint: %w", err)
	}
	if busy != 0 {
		return fmt.Errorf("wal_checkpoint reported busy (%d frames pending)", logFrames)
	}
	return nil
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// logTail reads new bytes from a job's log file starting at offset.
func logTail(path string, offset int64, limit int64) ([]byte, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, offset, nil
		}
		return nil, offset, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, offset, err
	}
	if fi.Size() <= offset {
		return nil, offset, nil
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, offset, err
	}
	n := fi.Size() - offset
	if limit > 0 && n > limit {
		n = limit
	}
	buf := make([]byte, n)
	read, err := io.ReadFull(f, buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return nil, offset, err
	}
	return buf[:read], offset + int64(read), nil
}

// logPath is the file procmgr writes a job's stdout and stderr to, and the
// exact path the coordinator's log route appends into.
func logPath(runDir string) string {
	return filepath.Join(runDir, "logs", "training.log")
}
