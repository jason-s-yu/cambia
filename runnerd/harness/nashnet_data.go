package harness

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/pathguard"
)

// HeaderOffset carries the coordinator's true part offset on a resume probe and
// on an offset mismatch (D50). It is nashnet's own spelling, so the node's
// HeaderBlobOffset and this cannot drift.
const HeaderOffset = nashnet.HeaderBlobOffset

// quarantineLease projects a lease record onto the quarantine store's view of
// it: the node, job, and lease ids come from the authenticated record and never
// from a request, so a node cannot address another lease's tree (D49).
func (p *Pool) quarantineLease(l nashnet.Lease) quarantine.Lease {
	p.mu.Lock()
	grants := p.quarGrants[l.LeaseID]
	p.mu.Unlock()
	return quarantine.Lease{
		NodeID:    l.NodeID,
		JobID:     l.JobID,
		LeaseID:   l.LeaseID,
		Epoch:     l.LeaseEpoch,
		Grants:    grants,
		RunDBName: p.runDBName(l.JobID),
	}
}

// runDBName is the runs.name this job's journal must carry (D55): the job id,
// or the target's run name for an evaluate job, whose rows belong to the run it
// evaluated (D64).
func (p *Pool) runDBName(jobID string) string {
	spec := readJobSpec(filepath.Join(p.runsDir, jobID))
	if spec == nil || spec.Kind != KindEvaluate || spec.Target == "" {
		return jobID
	}
	name, _, err := splitSeedRef(spec.Target)
	if err != nil {
		return jobID
	}
	return name
}

// handleBlobProbe is POST /nashnet/leases/{lease}/blobs/probe (D50 step 1). The
// answer is computed over the lease's provable set alone, so a node learns
// nothing it did not already know: a digest the coordinator holds under another
// lease is want, not have.
func (s *Server) handleBlobProbe(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	var req struct {
		Digests []string `json:"digests"`
	}
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	have, want, err := p.quar.Probe(p.quarantineLease(lease), req.Digests)
	if err != nil {
		writeQuarantineError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"have": have, "want": want})
}

// handleBlobHead is HEAD /nashnet/leases/{lease}/blobs/{digest} (D50 step 2):
// the resume offset is the part file's own size, so no side-car state exists
// that could skew from it across a crash.
func (s *Server) handleBlobHead(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	st, err := p.quar.BlobStatus(p.quarantineLease(lease), r.PathValue("digest"))
	if err != nil {
		writeQuarantineError(w, err)
		return
	}
	w.Header().Set(HeaderOffset, strconv.FormatInt(st.Offset, 10))
	if st.Verified {
		w.Header().Set("X-Nashnet-Verified", "1")
	}
	w.WriteHeader(http.StatusOK)
}

// handleBlobPatch is PATCH /nashnet/leases/{lease}/blobs/{digest} (D50 step 3).
// The route layer owns the per-request chunk cap, the concurrent-upload
// ceiling, and the per-tick lease budget; the per-file cap, the lease byte
// budget, and the store watermark are the store's, which holds the state they
// read.
func (s *Server) handleBlobPatch(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	if !p.uploads.acquire(lease.NodeID) {
		writeNashnetError(w, http.StatusTooManyRequests, nashnet.ErrorBody{
			Error: nashnet.CodeRateLimited, Detail: "concurrent upload ceiling reached", RetryAfterSeconds: 1,
		})
		return
	}
	defer p.uploads.release(lease.NodeID)

	start, end, total, err := parseContentRange(r.Header.Get("Content-Range"))
	if err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_range", err.Error())
		return
	}
	chunk := end - start + 1
	if total == 0 {
		chunk = 0
	}
	if chunk > p.ceilings.ChunkBytes {
		writeNashnetError(w, http.StatusRequestEntityTooLarge,
			nashnet.ErrorBody{Error: nashnet.CodeOverCap, Detail: "chunk over the per-request cap"})
		return
	}
	if !p.spendTick(lease.LeaseID, chunk) {
		writeNashnetError(w, http.StatusRequestEntityTooLarge, nashnet.ErrorBody{
			Error: nashnet.CodeOverCap, Detail: "lease byte budget for this tick exhausted",
		})
		return
	}
	body := http.MaxBytesReader(w, r.Body, p.ceilings.ChunkBytes)
	defer body.Close()
	res, err := p.quar.AppendChunk(p.quarantineLease(lease), r.PathValue("digest"), start, end, total, body)
	if err != nil {
		writeQuarantineError(w, err)
		return
	}
	w.Header().Set(HeaderOffset, strconv.FormatInt(res.CommittedOffset, 10))
	writeJSON(w, http.StatusOK, map[string]any{
		"committed_offset": res.CommittedOffset, "verified": res.Verified,
	})
}

// spendTick charges a lease's per-tick upload budget, which resets on every
// progress post (D56).
func (p *Pool) spendTick(leaseID string, n int64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.tickBytes[leaseID]+n > p.ceilings.LeaseBytesPerTick {
		return false
	}
	p.tickBytes[leaseID] += n
	return true
}

// handleBlobDelete is DELETE /nashnet/leases/{lease}/blobs/{digest} (D50 step
// 4): the node abandons a part.
func (s *Server) handleBlobDelete(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	if err := p.quar.AbandonPart(p.quarantineLease(lease), r.PathValue("digest")); err != nil {
		writeQuarantineError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleManifestGet is GET /nashnet/leases/{lease}/manifest: the coordinator's
// current head, which the node diffs its run dir against before it uploads.
func (s *Server) handleManifestGet(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	head, err := p.quar.ReadHead(lease.JobID)
	if err != nil {
		writeQuarantineError(w, err)
		return
	}
	entries := head.Folded.Entries
	if entries == nil {
		entries = []quarantine.Entry{}
	}
	// final rides along with the head the node diffs against, so a restarted
	// agent knows whether it already committed the final manifest before it
	// re-posts a result.
	writeJSON(w, http.StatusOK, map[string]any{
		"seq": head.Folded.Seq, "digest": head.Digest, "entries": entries,
		"final": head.Folded.Final,
	})
}

// handleManifestPost is POST /nashnet/leases/{lease}/manifest (D51): the
// fast-forward commit and the promotion transaction. The fence is the lease
// store's; the rest is the quarantine store's, which validates every entry,
// resolves digests against the provable set, content-validates the journal, and
// materializes with run_db.sqlite last.
func (s *Server) handleManifestPost(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, p.quar.Limits().MaxManifestBytes))
	if err != nil {
		writeNashnetError(w, http.StatusRequestEntityTooLarge,
			nashnet.ErrorBody{Error: nashnet.CodeOverCap, Detail: "manifest body over the cap"})
		return
	}
	var req quarantine.CommitRequest
	if err := json.Unmarshal(body, &req); err != nil {
		nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeInvalidManifest, err.Error())
		return
	}
	route := nashnet.RouteManifest
	if req.Final {
		route = nashnet.RouteManifestFinal
	}
	fenced, err := p.leases.Fence(lease.LeaseID, req.LeaseEpoch, p.nodeEpochFence(lease.NodeID),
		r.Header.Get(nashnet.HeaderLeaseToken), route)
	if err != nil {
		writeLeaseFenceError(w, err)
		return
	}
	resp, err := p.quar.Commit(p.quarantineLease(fenced), req, body)
	if err != nil {
		writeQuarantineError(w, err)
		return
	}
	for _, rej := range resp.Rejected {
		if rej.Detail != "" {
			poolLog("nashnet manifest: %s rejected %s: %s (%s)", lease.JobID, rej.Path, rej.Reason, rej.Detail)
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleLogs is POST /nashnet/leases/{lease}/logs?offset=N (D54). The
// coordinator appends only at the file's current size, so the log has one
// writer and one ordering; a wrong offset is 409 with the true offset so the
// node seeks. What it appends is filtered: C0 control bytes other than tab and
// newline are dropped and ESC-introduced sequences are neutralized, which takes
// terminal escape sequences out of a node's reach on the WS tail an operator
// reads.
func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	offset, err := strconv.ParseInt(r.URL.Query().Get("offset"), 10, 64)
	if err != nil || offset < 0 {
		nashnetError(w, http.StatusBadRequest, "invalid_offset", "offset must be a non-negative integer")
		return
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, p.ceilings.LogBytesPerCall))
	if err != nil {
		writeNashnetError(w, http.StatusRequestEntityTooLarge,
			nashnet.ErrorBody{Error: nashnet.CodeOverCap, Detail: "log body over the per-call cap"})
		return
	}
	logPath := filepath.Join(p.runsDir, lease.JobID, "logs", "training.log")
	if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
		nashnetError(w, http.StatusInternalServerError, "log_failed", err.Error())
		return
	}
	size := int64(0)
	if fi, serr := os.Stat(logPath); serr == nil {
		size = fi.Size()
	}
	if offset != size {
		writeNashnetError(w, http.StatusConflict, nashnet.ErrorBody{
			Error: nashnet.CodeOffsetMismatch, Detail: "append only at the current size", Offset: size,
		})
		return
	}
	clean, dropped := filterLogBytes(raw)
	if size+int64(len(clean)) > p.ceilings.LogBytesPerJob {
		p.noteLogDropped(lease.LeaseID, int64(len(clean))+dropped)
		writeJSON(w, http.StatusOK, map[string]any{"offset": size, "dropped": int64(len(clean)) + dropped})
		return
	}
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		nashnetError(w, http.StatusInternalServerError, "log_failed", err.Error())
		return
	}
	defer f.Close()
	if dropped > 0 {
		// The in-band marker is a convenience for someone tailing the file; the
		// authoritative count is the coordinator-owned counter below.
		fmt.Fprintf(f, "\n[nashnet: log truncated, %d bytes dropped]\n", dropped)
	}
	if _, err := f.Write(clean); err != nil {
		nashnetError(w, http.StatusInternalServerError, "log_failed", err.Error())
		return
	}
	p.noteLogDropped(lease.LeaseID, dropped)
	fi, _ := f.Stat()
	writeJSON(w, http.StatusOK, map[string]any{"offset": fi.Size(), "dropped": dropped})
}

// noteLogDropped advances the coordinator-owned dropped-byte counter (D54).
func (p *Pool) noteLogDropped(leaseID string, n int64) {
	if n <= 0 {
		return
	}
	p.mu.Lock()
	p.logDropped[leaseID] += n
	p.mu.Unlock()
}

// filterLogBytes drops the C0 control bytes other than tab and newline and
// neutralizes ESC-introduced sequences, returning the filtered bytes and how
// many were removed (D54).
func filterLogBytes(in []byte) ([]byte, int64) {
	out := make([]byte, 0, len(in))
	var dropped int64
	for i := 0; i < len(in); i++ {
		b := in[i]
		if b == 0x1b {
			// ESC: drop the introducer plus its sequence. A CSI runs to a final
			// byte in 0x40-0x7e; every other form is a single following byte.
			dropped++
			if i+1 < len(in) && in[i+1] == '[' {
				i++
				dropped++
				for i+1 < len(in) {
					i++
					dropped++
					if in[i] >= 0x40 && in[i] <= 0x7e {
						break
					}
				}
			} else if i+1 < len(in) {
				i++
				dropped++
			}
			continue
		}
		if b == '\t' || b == '\n' {
			out = append(out, b)
			continue
		}
		if b < 0x20 || b == 0x7f {
			dropped++
			continue
		}
		out = append(out, b)
	}
	return out, dropped
}

// handleSnapshot is GET /nashnet/leases/{lease}/snapshot (D48): the git bundle
// at the receipt commit, served with an ETag and Range support so an
// interrupted fetch resumes at its offset. A node fetches by the resolved
// commit's cached artifact, never by a job ref: the cache is shared across jobs
// at one commit.
func (s *Server) handleSnapshot(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	if !p.egress.acquire(lease.NodeID) {
		writeNashnetError(w, http.StatusTooManyRequests, nashnet.ErrorBody{
			Error: nashnet.CodeRateLimited, Detail: "concurrent download ceiling reached", RetryAfterSeconds: 1,
		})
		return
	}
	defer p.egress.release(lease.NodeID)

	p.mu.Lock()
	desc, ok := p.snapshots[lease.LeaseID]
	p.mu.Unlock()
	if !ok {
		if p.bundles == nil {
			nashnetError(w, http.StatusServiceUnavailable, "no_bundle_builder", "no snapshot source configured")
			return
		}
		// A coordinator restart loses the in-memory descriptor; the cache key is
		// the commit, so rebuilding hits the same artifact (D34).
		built, err := p.resolveSnapshot(r.Context(), lease.NodeID, lease.JobID, nil)
		if err != nil {
			nashnetError(w, http.StatusInternalServerError, "snapshot_failed", err.Error())
			return
		}
		if lease.GrantSet.Snapshot != "" && built.SHA256 != lease.GrantSet.Snapshot {
			poolLog("nashnet snapshot: rebuilt bundle for %s differs from the granted digest", lease.JobID)
		}
		desc = built
		p.mu.Lock()
		p.snapshots[lease.LeaseID] = desc
		p.mu.Unlock()
	}
	setDeadlines(w, r, deadlineEgress)
	if err := ingest.ServeRangedFile(w, r, desc.Path, ingest.FormatETag(desc.SHA256)); err != nil {
		poolLog("nashnet snapshot: serving %s: %v", lease.JobID, err)
	}
}

// handleSeed is GET /nashnet/leases/{lease}/seeds/{seed_id}/{path...} (D53):
// one granted seed file, range-resumable. A seed id or path outside the grant
// set returns the same 404 body as an unknown one, so an out-of-set read is
// indistinguishable from a miss and no existence oracle exists.
func (s *Server) handleSeed(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	if !p.egress.acquire(lease.NodeID) {
		writeNashnetError(w, http.StatusTooManyRequests, nashnet.ErrorBody{
			Error: nashnet.CodeRateLimited, Detail: "concurrent download ceiling reached", RetryAfterSeconds: 1,
		})
		return
	}
	defer p.egress.release(lease.NodeID)

	seedID := r.PathValue("seed_id")
	rel := r.PathValue("path")
	digest, allowed := lease.GrantSet.Allows(seedID, rel)
	if !allowed {
		nashnetError(w, http.StatusNotFound, "not_found", "no such seed entry")
		return
	}
	abs, err := pathguard.Resolve(p.runsDir, seedID+"/"+rel)
	if err != nil {
		nashnetError(w, http.StatusNotFound, "not_found", "no such seed entry")
		return
	}
	if fi, serr := os.Stat(abs); serr != nil || !fi.Mode().IsRegular() {
		// The grant named it, so a live lease asking for an entry that has since
		// vanished is the one place 409 seed_missing survives (D53).
		writeNashnetError(w, http.StatusConflict,
			nashnet.ErrorBody{Error: nashnet.CodeSeedMissing, Detail: "granted seed entry is gone"})
		return
	}
	setDeadlines(w, r, deadlineEgress)
	if err := ingest.ServeRangedFile(w, r, abs, ingest.FormatETag(digest)); err != nil {
		poolLog("nashnet seed: serving %s/%s: %v", seedID, rel, err)
	}
}

// parseContentRange reads a Content-Range: bytes start-end/total header into
// its three numbers (D50 step 3).
func parseContentRange(h string) (start, end, total int64, err error) {
	const prefix = "bytes "
	if !strings.HasPrefix(h, prefix) {
		return 0, 0, 0, errors.New("Content-Range must be bytes <start>-<end>/<size>")
	}
	rest := strings.TrimSpace(h[len(prefix):])
	if rest == "*/0" {
		// A zero-byte artifact has no range to send, so the node sends a
		// bodyless PATCH and the store creates the empty blob against the
		// digest the empty string hashes to (D50).
		return 0, -1, 0, nil
	}
	slash := strings.LastIndex(rest, "/")
	dash := strings.Index(rest, "-")
	if slash < 0 || dash < 0 || dash > slash {
		return 0, 0, 0, errors.New("Content-Range must be bytes <start>-<end>/<size>")
	}
	if start, err = strconv.ParseInt(rest[:dash], 10, 64); err != nil {
		return 0, 0, 0, err
	}
	if end, err = strconv.ParseInt(rest[dash+1:slash], 10, 64); err != nil {
		return 0, 0, 0, err
	}
	if total, err = strconv.ParseInt(rest[slash+1:], 10, 64); err != nil {
		return 0, 0, 0, err
	}
	return start, end, total, nil
}

// writeQuarantineError maps the quarantine store's typed errors to the statuses
// of D50, D51, and D56. The store returns errors and never a status code, so
// this table is the whole of the mapping.
func writeQuarantineError(w http.ResponseWriter, err error) {
	var offset *quarantine.OffsetMismatchError
	var fence *quarantine.FenceError
	var missing *quarantine.BlobsMissingError
	var invalid *quarantine.InvalidManifestError
	var rng *quarantine.InvalidRangeError
	var big *quarantine.FileTooLargeError
	var budget *quarantine.LeaseBudgetError
	var full *quarantine.StoreFullError
	switch {
	case errors.Is(err, quarantine.ErrInvalidDigest):
		nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeInvalidDigest, err.Error())
	case errors.Is(err, quarantine.ErrHashMismatch):
		nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeHashMismatch, err.Error())
	case errors.As(err, &offset):
		w.Header().Set(HeaderOffset, strconv.FormatInt(offset.Offset, 10))
		writeNashnetError(w, http.StatusConflict, nashnet.ErrorBody{
			Error: nashnet.CodeOffsetMismatch, Detail: err.Error(), Offset: offset.Offset,
		})
	case errors.As(err, &fence):
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": nashnet.CodeManifestOutOfOrder, "detail": fence.Reason,
			"seq": fence.Seq, "digest": fence.Digest,
		})
	case errors.As(err, &missing):
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": nashnet.CodeBlobsMissing, "missing": missing.Missing,
		})
	case errors.As(err, &invalid):
		nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeInvalidManifest, invalid.Detail)
	case errors.As(err, &rng):
		nashnetError(w, http.StatusBadRequest, "invalid_range", rng.Detail)
	case errors.As(err, &big):
		nashnetError(w, http.StatusRequestEntityTooLarge, nashnet.CodeOverCap, err.Error())
	case errors.As(err, &budget):
		nashnetError(w, http.StatusRequestEntityTooLarge, nashnet.CodeOverCap, err.Error())
	case errors.As(err, &full):
		writeNashnetError(w, http.StatusInsufficientStorage, nashnet.ErrorBody{
			Error: nashnet.CodeStoreFull, Detail: err.Error(),
			RetryAfterSeconds: int(full.RetryAfter.Seconds()),
		})
	case errors.Is(err, os.ErrNotExist):
		nashnetError(w, http.StatusNotFound, "not_found", "no such entry")
	default:
		nashnetError(w, http.StatusInternalServerError, "quarantine_failed", err.Error())
	}
}
