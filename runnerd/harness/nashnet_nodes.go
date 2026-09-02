package harness

import (
	"errors"
	"net/http"
	"strconv"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// handleNodeRegister is POST /nashnet/nodes/register (D3). Enrollment is the
// operator-signed grant; this is a declaration, keyed by the node id derived
// from the verified token subject and never by a node_id in the body. A body id
// that disagrees is 403 wrong_subject, which is what stops a node holding a
// valid grant for one id from overwriting another node's record, bumping its
// epoch, and revoking the leases it never named (D25).
func (s *Server) handleNodeRegister(w http.ResponseWriter, r *http.Request, nodeID string) {
	p := s.pool
	var req nashnet.RegisterRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	if req.NodeID != "" && req.NodeID != nodeID {
		nashnetError(w, http.StatusForbidden, nashnet.CodeWrongSubject, "node_id does not match the verified subject")
		return
	}
	if _, err := p.declarationFor(nodeID, req.Capabilities); err != nil {
		nashnetError(w, http.StatusUnprocessableEntity, "invalid_declaration", err.Error())
		return
	}
	rec, err := p.nodes.Register(nodeID, req)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	rebind, err := p.leases.ReBind(nodeID, rec.NodeEpoch, req.LiveLeases)
	if err != nil {
		nashnetError(w, http.StatusInternalServerError, "rebind_failed", err.Error())
		return
	}
	revoked := make([]string, 0, len(rebind.Revoked))
	for _, o := range rebind.Revoked {
		revoked = append(revoked, o.LeaseID)
		p.applyOutcome(o)
	}
	writeJSON(w, http.StatusOK, nashnet.RegisterResponse{
		NodeID:        nodeID,
		NodeEpoch:     rec.NodeEpoch,
		Policy:        p.policy,
		ReboundLeases: rebind.Rebound,
		RevokedLeases: revoked,
		ServerTime:    rfc3339(p.now()),
	})
}

// handleNodeHeartbeat is POST /nashnet/nodes/{node}/heartbeat (D3): an idle
// node refreshes its volatile facts and its gate report every 30s. Liveness
// while busy is the events long poll, not this.
func (s *Server) handleNodeHeartbeat(w http.ResponseWriter, r *http.Request, nodeID string) {
	p := s.pool
	var req nashnet.HeartbeatRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	rec, err := p.nodes.Heartbeat(nodeID, req)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, nashnet.HeartbeatResponse{
		NodeEpoch:  rec.NodeEpoch,
		Drain:      rec.Drained,
		ServerTime: rfc3339(p.now()),
	})
}

// handleNodeEvents is GET /nashnet/nodes/{node}/events (D45): the held
// back-channel and the session liveness fact. It is answered early when an
// event is pending, at the wait otherwise, and a second concurrent request for
// one node supersedes the first, which returns 409 session_superseded.
func (s *Server) handleNodeEvents(w http.ResponseWriter, r *http.Request, nodeID string) {
	p := s.pool
	wait := clampWait(queryInt(r, "wait_seconds", maxClaimWaitSeconds))
	deadline := p.now().Add(wait)
	if _, err := p.nodes.HoldSession(nodeID, deadline); err != nil {
		writeRegistryError(w, err)
		return
	}
	sess, older := p.openSession(nodeID, deadline)
	defer p.closeSession(nodeID, sess)
	if older != nil {
		close(older.superseded)
	}
	// The held request may outlive the per-route write deadline the middleware
	// set, so it carries its own.
	setDeadlines(w, r, wait+deadlineDefault)

	timer := time.NewTimer(wait)
	defer timer.Stop()
	for {
		if evs := p.takeEvents(nodeID); len(evs) > 0 {
			p.writeEvents(w, nodeID, evs)
			return
		}
		select {
		case <-sess.ready:
		case <-sess.superseded:
			nashnetError(w, http.StatusConflict, nashnet.CodeSessionSuperseded,
				"a newer events request holds this node's session")
			return
		case <-timer.C:
			p.writeEvents(w, nodeID, nil)
			return
		case <-r.Context().Done():
			return
		}
	}
}

// writeEvents answers an events poll with the node's current epoch, which is
// how a node learns its leases were superseded even when no event was queued.
func (p *Pool) writeEvents(w http.ResponseWriter, nodeID string, evs []nashnet.Event) {
	epoch := int64(0)
	if rec, ok := p.nodes.Get(nodeID); ok {
		epoch = rec.NodeEpoch
	}
	if evs == nil {
		evs = []nashnet.Event{}
	}
	writeJSON(w, http.StatusOK, nashnet.EventsResponse{
		Events:     evs,
		NodeEpoch:  epoch,
		ServerTime: rfc3339(p.now()),
	})
}

// openSession registers this request as the node's one held events request and
// returns the older session it supersedes, if any (D45).
func (p *Pool) openSession(nodeID string, until time.Time) (*nodeSession, *nodeSession) {
	s := &nodeSession{
		ready:      make(chan struct{}, 1),
		superseded: make(chan struct{}),
		until:      until,
	}
	p.mu.Lock()
	older := p.sessions[nodeID]
	p.sessions[nodeID] = s
	p.mu.Unlock()
	return s, older
}

// closeSession drops a held events request, leaving a newer one in place.
func (p *Pool) closeSession(nodeID string, s *nodeSession) {
	p.mu.Lock()
	if p.sessions[nodeID] == s {
		delete(p.sessions, nodeID)
	}
	p.mu.Unlock()
}

// handleNodesList is GET /nashnet/nodes (operator): the declaration, the gate
// report, the session state and staleness, and the node's live leases.
func (s *Server) handleNodesList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"nodes": s.pool.nodeViews()})
}

// handleNodeGet is GET /nashnet/nodes/{node} (operator).
func (s *Server) handleNodeGet(w http.ResponseWriter, r *http.Request) {
	p := s.pool
	id := r.PathValue("node")
	if err := procmgr.ValidateName(id); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	rec, ok := p.nodes.Get(id)
	if !ok {
		nashnetError(w, http.StatusNotFound, "not_found", "unknown node")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"node": p.nodeView(rec, p.now())})
}

// handleNodeDrain is POST /nashnet/nodes/{node}/drain (operator). It is two-way:
// the body carries the state to set, so lifting a hold costs no second route
// and does not require restarting an agent that holds live leases.
// clear_breaker also resets the D63 breaker counter and the per-job degraded
// marks of D8, which is the operator clearing referenced there; there is no
// node-initiated path to any of it.
func (s *Server) handleNodeDrain(w http.ResponseWriter, r *http.Request) {
	p := s.pool
	id := r.PathValue("node")
	if err := procmgr.ValidateName(id); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	var req nashnet.DrainRequest
	if err := decodeJSON(r, &req); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_body", err.Error())
		return
	}
	rec, err := p.nodes.SetDrained(id, req.Drain)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	if req.ClearBreaker {
		p.clearBreaker(id)
	}
	p.postEvent(id, nashnet.Event{Type: nashnet.EventDrain})
	if !req.Drain {
		p.signalPlacement()
	}
	writeJSON(w, http.StatusOK, map[string]any{"node": p.nodeView(rec, p.now())})
}

// handleNodeRevoke is POST /nashnet/nodes/{node}/revoke (operator, D60). It
// writes the tombstone every token and lease-token verification consults, bumps
// the node epoch so every lease it held is superseded, and settles each of
// those jobs against whatever was already promoted. Revocation is the one stop
// with no grace period: the operator has declared the node untrusted, so it
// gets no window in which to write.
func (s *Server) handleNodeRevoke(w http.ResponseWriter, r *http.Request) {
	p := s.pool
	id := r.PathValue("node")
	if err := procmgr.ValidateName(id); err != nil {
		nashnetError(w, http.StatusBadRequest, "invalid_name", err.Error())
		return
	}
	if err := p.writeTombstone(id); err != nil {
		nashnetError(w, http.StatusInternalServerError, "revoke_failed", err.Error())
		return
	}
	rec, err := p.nodes.Revoke(id)
	if err != nil && !errors.Is(err, nashnet.ErrUnknownNode) {
		writeRegistryError(w, err)
		return
	}
	outcomes, lerr := p.leases.RevokeNode(id)
	if lerr != nil {
		nashnetError(w, http.StatusInternalServerError, "revoke_failed", lerr.Error())
		return
	}
	revoked := make([]string, 0, len(outcomes))
	for _, o := range outcomes {
		revoked = append(revoked, o.LeaseID)
		p.applyOutcome(o)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"node_id":        id,
		"revoked":        true,
		"node_epoch":     rec.NodeEpoch,
		"revoked_leases": revoked,
	})
}

// writeRegistryError maps a node-registry refusal to its status. An unknown
// node on a route that needs a record is 409 not_registered rather than 404: the
// credential is good, the node simply has not declared itself since the
// coordinator last started, and the fix is a register call.
func writeRegistryError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, nashnet.ErrNodeRevoked):
		nashnetError(w, http.StatusUnauthorized, "unauthorized", "node credential refused")
	case errors.Is(err, nashnet.ErrUnknownNode):
		nashnetError(w, http.StatusConflict, "not_registered", "register before calling this route")
	default:
		nashnetError(w, http.StatusInternalServerError, "registry_failed", err.Error())
	}
}

// queryInt reads a bounded integer query parameter, falling back on anything
// malformed rather than refusing the call.
func queryInt(r *http.Request, key string, fallback int) int {
	raw := r.URL.Query().Get(key)
	if raw == "" {
		return fallback
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return n
}

// clampWait applies the coordinator's own cap to a node-supplied wait (D2).
func clampWait(seconds int) time.Duration {
	if seconds <= 0 {
		return 0
	}
	if seconds > maxClaimWaitSeconds {
		seconds = maxClaimWaitSeconds
	}
	return time.Duration(seconds) * time.Second
}
