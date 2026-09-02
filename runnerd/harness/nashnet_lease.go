package harness

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// envJSONFile is the coordinator-authored provenance record (D23). It is a
// rejected manifest path (D52), so the coordinator is its only writer on a pool
// run.
const envJSONFile = "env.json"

// projectPhase maps a node-reported phase onto the procmgr status enum (D5).
// The two alphabets are not the same: preparing is a runnerd job state rather
// than a procmgr status, and stopping is outside the node's vocabulary
// entirely, so the projection maps rather than copies and only enum values ever
// reach process.json.
func projectPhase(phase string) (string, bool) {
	switch phase {
	case nashnet.PhaseClaimed, nashnet.PhaseFetching, nashnet.PhasePreparing:
		return procmgr.StatusStarting, true
	case nashnet.PhaseRunning, nashnet.PhaseUploading, nashnet.PhaseCommitting:
		return procmgr.StatusRunning, true
	default:
		return "", false
	}
}

// handleProgress is POST /nashnet/leases/{lease}/progress (D5): one call is the
// lease heartbeat and the liveness mirror the coordinator projects into
// process.json. A revoking lease is admitted, renews nothing, and is answered
// with revoke, which is the second delivery path beside the events poll.
func (s *Server) handleProgress(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	var req nashnet.ProgressRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if req.Phase != "" {
		if err := nashnet.ValidateNodePhase(req.Phase); err != nil {
			nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeInvalidPhase,
				"phase is not one a node may report: "+req.Phase)
			return
		}
	}
	updated, err := p.leases.Renew(nashnet.ProgressUpdate{
		LeaseID:        lease.LeaseID,
		LeaseEpoch:     req.LeaseEpoch,
		NodeEpoch:      p.nodeEpochFence(lease.NodeID),
		Token:          r.Header.Get(nashnet.HeaderLeaseToken),
		Phase:          req.Phase,
		PIDProjected:   req.PID > 0,
		ManifestSeq:    req.ManifestSeq,
		ManifestDigest: req.ManifestDigest,
	})
	if err != nil {
		writeLeaseFenceError(w, err)
		return
	}
	if status, ok := projectPhase(req.Phase); ok {
		p.project(updated, status, req)
	}
	p.mu.Lock()
	delete(p.tickBytes, lease.LeaseID)
	p.mu.Unlock()

	rec, _ := p.nodes.Get(updated.NodeID)
	resp := nashnet.ProgressResponse{
		Drain:         rec.Drained,
		LeaseDeadline: rfc3339(updated.Deadline),
	}
	if updated.State == nashnet.LeaseRevoking {
		// The second delivery path of D31: a node that missed the revoke event
		// reads the same {revoke, force} pair off its next progress tick.
		resp.Revoke = true
		resp.Force = p.stopForceFor(updated.LeaseID)
		resp.RetryAfterSeconds = p.policy.ProgressIntervalSeconds
	}
	writeJSON(w, http.StatusOK, resp)
}

// project mirrors a node's liveness into process.json (D5). Host is the node
// id, the field's documented purpose, and PGID is always 0: a node-supplied
// pgid is never accepted, because ProcessManager.Stop's unsupervised branch
// would otherwise turn an ordinary cancel into an arbitrary process-group
// signal on the coordinator host.
func (p *Pool) project(lease nashnet.Lease, status string, req nashnet.ProgressRequest) {
	runDir := filepath.Join(p.runsDir, lease.JobID)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil {
		st = &procmgr.ProcessState{Name: lease.JobID, CreatedAt: procmgr.NowRFC3339()}
	}
	st.Host = lease.NodeID
	st.Status = status
	st.PID = req.PID
	st.PGID = 0
	if req.StartedAt != "" && st.StartedAt == "" {
		st.StartedAt = req.StartedAt
	}
	if err := procmgr.WriteProcessState(runDir, st); err != nil {
		poolLog("nashnet progress: projecting %s: %v", lease.JobID, err)
	}
}

// projectReady un-projects a job returning to the ready set. The row a node's
// progress ticks left behind names a host and a phase that describe nothing any
// more, and the restart path of D34 re-enqueues created rows: a requeued job
// left at starting or running is one a restart neither re-enqueues nor
// reattaches, because its non-empty Host marks a process on another machine.
// Resetting it to created is what puts it back in submit_seq order across a
// restart, where the live queue already holds it.
//
// A terminal row is left alone. The two ways a return ends in a terminal
// instead, a spent attempt budget and a promoted checkpoint, are written by the
// caller before this runs.
//
// So is a job holding a promoted checkpoint, which is a resume the operator
// asked for: the resume intent lives in the queue handle rather than in
// jobspec.json, so a created row would come back from a restart as a fresh
// launch over the run dir the checkpoint sits in. Leaving that row alone costs
// an operator act after a restart, which is what D33 asks for anyway, rather
// than restarting a job that ran.
func (p *Pool) projectReady(jobID string) {
	runDir := filepath.Join(p.runsDir, jobID)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil || isTerminal(procmgr.EffectiveStatus(st)) || p.promotedCheckpoint(jobID) {
		return
	}
	st.Status = procmgr.StatusCreated
	st.Host = ""
	st.PID = 0
	st.PGID = 0
	st.StartedAt = ""
	st.FinishedAt = ""
	st.ExitCode = nil
	if err := procmgr.WriteProcessState(runDir, st); err != nil {
		poolLog("nashnet: returning %s to ready: %v", jobID, err)
	}
}

// nodeEpochFence is the registry's current epoch for a lease's node, supplied
// by the coordinator rather than by the node: a lease route carries the lease
// token and nothing else (D26). A node that has not registered since a
// coordinator restart leaves no epoch to compare, which is the one case the
// fence skips (D34).
func (p *Pool) nodeEpochFence(nodeID string) int64 {
	if rec, ok := p.nodes.Get(nodeID); ok {
		return rec.NodeEpoch
	}
	return nashnet.SkipEpoch
}

// handleNack is POST /nashnet/leases/{lease}/nack (D8): the node returns a
// claim it cannot honor. The job goes back to the ready set at its original
// position with no attempt increment, and the node is excluded from re-matching
// it for the cooldown. A wrong placement is a scheduling event, not a job
// failure: failing it would burn a name.
func (s *Server) handleNack(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	var req nashnet.NackRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if _, err := p.leases.Fence(lease.LeaseID, req.LeaseEpoch, p.nodeEpochFence(lease.NodeID),
		r.Header.Get(nashnet.HeaderLeaseToken), nashnet.RouteNack); err != nil {
		writeLeaseFenceError(w, err)
		return
	}
	out, err := p.leases.Return(lease.LeaseID, req.Reason)
	if err != nil {
		nashnetError(w, http.StatusInternalServerError, "nack_failed", err.Error())
		return
	}
	if p.noteNack(lease.NodeID, lease.JobID, req.Reason,
		time.Duration(req.CooldownSeconds)*time.Second) {
		// Three consecutive prepare_node_failed nacks: the node's claims are
		// refused node_gated until its own cooldown runs out or an operator lifts
		// the hold (D63). No event is posted, because the only one that would say
		// this is drain, which names an operator act the node clears off its next
		// heartbeat; the 204 and its Retry-After are the whole signal, and the
		// queue is untouched, so the job this nack returned goes to the next
		// capable node.
		poolLog("nashnet: circuit breaker tripped for node %s", lease.NodeID)
	}
	// The lease settles through the same outcome path as every other ended one,
	// so a nack posted after the process started finalizes rather than returning
	// a job that ran (D32, D33).
	p.applyOutcome(out)
	writeJSON(w, http.StatusOK, map[string]any{
		"job_id":   lease.JobID,
		"returned": out.Verdict == nashnet.VerdictRequeue,
		"attempt":  out.NextAttempt,
	})
}

// handleResult is POST /nashnet/leases/{lease}/result (D6), the commit point.
// It is accepted only once the node has committed a final manifest whose digest
// the result carries and whose journal was promoted; otherwise the lease holds
// for one more grace period. A replay of a recorded (job, lease, epoch) returns
// the recorded terminal rather than a conflict.
func (s *Server) handleResult(w http.ResponseWriter, r *http.Request, lease nashnet.Lease) {
	p := s.pool
	var req nashnet.ResultRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	p.mu.Lock()
	recorded, replay := p.results[lease.LeaseID]
	p.mu.Unlock()
	if replay && recorded.LeaseEpoch == req.LeaseEpoch {
		writeJSON(w, http.StatusOK, recorded)
		return
	}
	if _, err := p.leases.Fence(lease.LeaseID, req.LeaseEpoch, p.nodeEpochFence(lease.NodeID),
		r.Header.Get(nashnet.HeaderLeaseToken), nashnet.RouteResult); err != nil {
		writeLeaseFenceError(w, err)
		return
	}
	if !validResultState(req.State) {
		nashnetError(w, http.StatusUnprocessableEntity, "invalid_state", "unknown terminal state: "+req.State)
		return
	}
	if !lease.PermitsResult(req.State) {
		nashnetError(w, http.StatusConflict, nashnet.CodeLeaseSuperseded,
			"a stopping lease accepts only canceled or preempted")
		return
	}
	// A job that never launched has no artifacts to commit and no journal to
	// promote, so the gate of D6 does not apply to it: holding its lease for a
	// grace period would only delay a terminal the node already knows. Every
	// lease that reached a launched phase still needs its final manifest.
	if nashnet.PhaseLaunched(lease.Phase) || lease.PIDProjected {
		head, err := p.quar.ReadHead(lease.JobID)
		if err != nil || !head.Folded.Final || head.Digest == "" || head.Digest != req.FinalManifestDigest {
			nashnetError(w, http.StatusConflict, nashnet.CodeArtifactsIncomplete,
				"result needs a committed final manifest whose digest it carries")
			return
		}
	}

	// A gate-driven stop is not an operator cancel (D62). Preempted with no
	// promoted checkpoint is a gate release: the job returns to ready at its
	// original submit_seq with no attempt increment and no terminal is written,
	// so a capable node picks it up when its gate reopens. Preempted with a
	// checkpoint is terminal and waits for an explicit operator resume, which is
	// D33 unchanged.
	gateRelease := req.State == nashnet.ResultPreempted && !p.promotedCheckpoint(lease.JobID)
	if !gateRelease {
		p.writeTerminal(lease, req)
	}
	if err := p.writeEnvRecord(lease.JobID, lease, "", req.State, &req); err != nil {
		poolLog("nashnet result: env.json for %s: %v", lease.JobID, err)
	}
	if _, err := p.leases.Release(lease.LeaseID); err != nil {
		poolLog("nashnet result: releasing %s: %v", lease.LeaseID, err)
	}
	p.releaseLeaseState(lease.LeaseID)
	if err := p.quar.Retire(p.quarantineLease(lease)); err != nil {
		poolLog("nashnet result: retiring quarantine for %s: %v", lease.LeaseID, err)
	}
	resp := nashnet.ResultResponse{
		JobID:      lease.JobID,
		LeaseID:    lease.LeaseID,
		LeaseEpoch: lease.LeaseEpoch,
		State:      req.State,
		ExitCode:   req.ExitCode,
		RecordedAt: rfc3339(p.now()),
	}
	p.recordResult(lease.LeaseID, resp,
		nashnet.HashLeaseToken(r.Header.Get(nashnet.HeaderLeaseToken)))
	if gateRelease {
		// The queue still holds the job at its original position, so returning it
		// costs one re-dispatch: nothing is dequeued and no attempt is charged.
		p.projectReady(lease.JobID)
		p.disp.reDispatch()
	} else {
		// Dependents gate on a settled parent, so the dispatch scan re-runs only
		// after the terminal is on disk (D6).
		p.disp.clearPlaced(lease.JobID)
	}
	p.signalPlacement()
	writeJSON(w, http.StatusOK, resp)
}

// validResultState reports whether a node-posted terminal is in the D6 set.
func validResultState(state string) bool {
	switch state {
	case nashnet.ResultStopped, nashnet.ResultCrashed, nashnet.ResultCanceled,
		nashnet.ResultFailed, nashnet.ResultPreempted:
		return true
	}
	return false
}

// writeTerminal flips process.json to the posted terminal. The runnerd-level
// terminals (canceled, failed, preempted) are persisted into Status exactly as
// the local gate terminals already are; the phase projection above is the path
// that only ever writes procmgr enum values.
func (p *Pool) writeTerminal(lease nashnet.Lease, req nashnet.ResultRequest) {
	runDir := filepath.Join(p.runsDir, lease.JobID)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil {
		st = &procmgr.ProcessState{Name: lease.JobID, CreatedAt: procmgr.NowRFC3339()}
	}
	st.Host = lease.NodeID
	st.Status = req.State
	st.PGID = 0
	st.ExitCode = req.ExitCode
	st.LastError = req.LastError
	if req.FinishedAt != "" {
		st.FinishedAt = req.FinishedAt
	} else {
		st.FinishedAt = procmgr.NowRFC3339()
	}
	if err := procmgr.WriteProcessState(runDir, st); err != nil {
		poolLog("nashnet result: writing terminal for %s: %v", lease.JobID, err)
	}
}

// envRecord is the coordinator-authored provenance of D23. origin_host answers
// "who owns and serves this run" and executed_on answers "which node produced
// these numbers"; the second is a plain node-id string, read from the lease and
// never from node-written data.
type envRecord struct {
	JobID      string          `json:"job_id"`
	OriginHost string          `json:"origin_host"`
	ExecutedOn string          `json:"executed_on"`
	LeaseID    string          `json:"lease_id"`
	NodeEpoch  int64           `json:"node_epoch,omitempty"`
	Commit     string          `json:"commit,omitempty"`
	State      string          `json:"state,omitempty"`
	FinishedAt string          `json:"finished_at,omitempty"`
	CreatedAt  string          `json:"created_at"`
	NodeReport json.RawMessage `json:"node_reported,omitempty"`
}

// writeEnvRecord authors runs/<job>/env.json. It is an explicit overwrite, not
// the write-once ingest path: routing the coordinator's record through a
// write-once writer would make it a silent no-op wherever a run dir already
// holds one, leaving executed_on empty for the run's whole life (D23).
func (p *Pool) writeEnvRecord(jobID string, lease nashnet.Lease, commit, state string, res *nashnet.ResultRequest) error {
	runDir := filepath.Join(p.runsDir, jobID)
	rec := envRecord{
		JobID:      jobID,
		OriginHost: p.originHost,
		ExecutedOn: lease.NodeID,
		LeaseID:    lease.LeaseID,
		NodeEpoch:  lease.NodeEpoch,
		Commit:     commit,
		State:      state,
		CreatedAt:  rfc3339(p.now()),
	}
	if rec.Commit == "" {
		if spec := readJobSpec(runDir); spec != nil {
			rec.Commit = spec.Commit
		}
	}
	if res != nil {
		rec.FinishedAt = res.FinishedAt
	}
	if node, ok := p.nodes.Get(lease.NodeID); ok && len(node.Capabilities) > 0 {
		rec.NodeReport = node.Capabilities
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	path := filepath.Join(runDir, envJSONFile)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// writeLeaseFenceError maps a fence or renew refusal to its status (D4).
func writeLeaseFenceError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, nashnet.ErrInvalidPhase):
		nashnetError(w, http.StatusUnprocessableEntity, nashnet.CodeInvalidPhase, err.Error())
	case errors.Is(err, nashnet.ErrLeaseSuperseded):
		nashnetError(w, http.StatusConflict, nashnet.CodeLeaseSuperseded, "lease superseded")
	case errors.Is(err, nashnet.ErrUnknownLease), errors.Is(err, nashnet.ErrLeaseTokenDropped):
		nashnetError(w, http.StatusUnauthorized, "unauthorized", "lease credential refused")
	default:
		nashnetError(w, http.StatusConflict, nashnet.CodeLeaseSuperseded, err.Error())
	}
}
