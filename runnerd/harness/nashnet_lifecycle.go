package harness

import (
	"path/filepath"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// The two files the PRT-CFR resume contract needs, run-dir relative. They are
// named once here because the resume gate reads them twice: off the filesystem,
// and out of the promoted manifest head for a job the pool ran (D31).
const (
	resumeCheckpointPath = "snapshots/prtcfr_checkpoint.pt"
	resumeStatePath      = "resume_state.json"
)

// cancelLeased is the revoke path of D31 and reports whether it handled the
// cancel. A job a node holds a live lease for is stopped through that lease and
// through nothing else: the local stop path signals the pid on process.json,
// and a row this daemon never forked is exactly the shape it would still try to
// signal (D5). The ordering is the point rather than an optimization, so this
// runs before Cancel consults the pending set or the supervisor at all.
//
// A lease that was released between the lookup and the revoke is reported
// unhandled: the job is no longer on a node, so the ordinary path is the
// correct one for it.
func (d *Dispatcher) cancelLeased(name string, force bool) (*procmgr.ProcessState, bool) {
	d.mu.Lock()
	pool, leases := d.pool, d.leases
	d.mu.Unlock()
	if pool == nil || leases == nil {
		return nil, false
	}
	l, ok := leases.ByJob(name)
	if !ok || !l.Live() {
		return nil, false
	}
	if err := pool.stopLease(l, force); err != nil {
		return nil, false
	}
	st, _ := procmgr.ReadProcessState(d.runDir(name))
	d.broadcast()
	return st, true
}

// stopLease is the coordinator half of a cancel that landed on a leased job
// (D31). The lease moves to revoking with stop_requested_at recorded, keeping
// its token and its epoch for one grace period so the node can commit a final
// manifest and post canceled; the projection is written stopping, which is the
// coordinator's own witness and the first of the two the D7 verdict rests on;
// and the revoke reaches the node on its held events request rather than at its
// next progress tick. force carries SIGKILL to the node, which is the only
// place any signal is sent.
func (p *Pool) stopLease(l nashnet.Lease, force bool) error {
	updated, err := p.leases.BeginRevoking(l.LeaseID, true)
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.stopForce[updated.LeaseID] = force
	p.mu.Unlock()
	p.projectStopping(updated)
	p.postEvent(updated.NodeID, nashnet.Event{
		Type: nashnet.EventRevoke, LeaseID: updated.LeaseID, Force: force,
	})
	return nil
}

// stopForceFor reports whether the stop recorded for a lease was a forced
// one, so a progress response carries the same {revoke, force} pair the events
// poll does and a node that missed the event still sends SIGKILL (D31, D45).
func (p *Pool) stopForceFor(leaseID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stopForce[leaseID]
}

// projectStopping writes the operator-stop marker into process.json. stopping
// is outside the node's phase vocabulary (D5), so this is the one path that
// writes it and the witness a node cannot forge. PGID stays 0 and Host stays
// the node id, so no local stop path can reach a process group on this host.
//
// A row that is already terminal is left alone: a result the node committed in
// the gap between the lease lookup and the revoke is the authoritative outcome,
// and overwriting it with stopping would strand the job non-terminal with a
// released lease no sweep will revisit.
//
// On an in-place lease the witness is written and nothing else is: the pid and
// pgid on that row were recorded by this daemon's own ProcessManager at the
// fork, so blanking them would orphan a live local process group from every
// stop path this daemon has left after a restart, which is the opposite of what
// the Host and PGID rules above protect against (D40, cambia-2017).
func (p *Pool) projectStopping(l nashnet.Lease) {
	runDir := filepath.Join(p.runsDir, l.JobID)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil {
		st = &procmgr.ProcessState{Name: l.JobID, CreatedAt: procmgr.NowRFC3339()}
	}
	if isTerminal(procmgr.EffectiveStatus(st)) {
		return
	}
	if !p.inPlaceLease(l) {
		st.Host = l.NodeID
		st.PGID = 0
	}
	st.Status = procmgr.StatusStopping
	if err := procmgr.WriteProcessState(runDir, st); err != nil {
		poolLog("nashnet cancel: projecting stopping for %s: %v", l.JobID, err)
	}
}

// purgeQuarantine deletes every lease's quarantine tree for a job (D31, D49).
func (p *Pool) purgeQuarantine(jobID string) error {
	_, err := p.quar.PurgeJob(jobID)
	return err
}

// nodePresent reports whether a node could still claim: enrolled, not
// tombstoned, not held by an operator drain, and seen inside the node TTL. It
// is what the reservoir pin reads to decide between waiting for its node and
// rendering the hold that names it (D12).
func (p *Pool) nodePresent(nodeID string) bool {
	if nodeID == "" {
		return false
	}
	rec, ok := p.nodes.Get(nodeID)
	if !ok || rec.Revoked || rec.Drained {
		return false
	}
	return rec.Presence(p.now(), p.nodeTTL, p.sessGrace) != nashnet.PresenceStale
}

// manifestHead returns a job's promoted manifest head. A job the pool never ran
// has no head, which is how the resume gate tells a pool run from a local one.
func (p *Pool) manifestHead(jobID string) (quarantine.Head, bool) {
	head, err := p.quar.ReadHead(jobID)
	if err != nil {
		return quarantine.Head{}, false
	}
	return head, true
}

// purgeLeaseTrees removes the quarantine trees a purged job left behind (D31).
// It runs after the run dir is gone, because the trees exist to be promoted
// into it and a tree under a node that no longer holds the lease would
// otherwise sit until the debug TTL with nothing left to describe.
func (d *Dispatcher) purgeLeaseTrees(name string) {
	pool := d.placementSourceRef()
	if pool == nil {
		return
	}
	if err := pool.purgeQuarantine(name); err != nil {
		poolLog("purge: quarantine trees for %s: %v", name, err)
	}
}

// hasPromotedResumableState is the resume gate read against the promoted copy
// (D31). For a job the pool ran, the run dir is written by the promotion
// transaction alone, so the folded manifest head is asked whether the two
// resume files were actually promoted rather than trusting bytes that happen to
// sit in the directory. A job with no head never ran in the pool and keeps the
// filesystem gate unchanged, which is the zero-node behavior of D40.
func (d *Dispatcher) hasPromotedResumableState(name string) bool {
	if !hasResumableState(d.runsDir, name) {
		return false
	}
	pool := d.placementSourceRef()
	if pool == nil {
		return true
	}
	head, ok := pool.manifestHead(name)
	if !ok || head.Folded.Seq == 0 {
		return true
	}
	promoted := make(map[string]bool, len(head.Folded.Entries))
	for _, e := range head.Folded.Entries {
		promoted[e.Path] = true
	}
	return promoted[resumeCheckpointPath] && promoted[resumeStatePath]
}

// pinResume applies the reservoir pin of D12 to a resume and returns the node
// it pinned to, or "" when the prior run was local. The pin is authoritative
// rather than an optimization: a PRT-CFR resume loads a per-player disk
// reservoir that is excluded from both the manifest and the seed set, so it
// never travels, and a resume placed on any other node dies loading it.
func (d *Dispatcher) pinResume(spec *JobSpec) string {
	node := d.priorExecutor(spec.Name)
	if node == "" {
		return ""
	}
	req := capability.Requires{}
	if spec.Requires != nil {
		req = *spec.Requires
	}
	req.Node = node
	spec.Requires = &req
	return node
}

// priorExecutor names the node whose runs dir holds the job's reservoir: the
// node its retained lease record names, or the executed_on the coordinator
// wrote into env.json, which survives a lease the store no longer holds. Both
// are coordinator-authored, so neither is a node assertion.
func (d *Dispatcher) priorExecutor(name string) string {
	d.mu.Lock()
	leases := d.leases
	d.mu.Unlock()
	if leases != nil {
		if l, ok := leases.ByJob(name); ok && l.NodeID != "" {
			return l.NodeID
		}
	}
	if env := readEnvJSON(d.runsDir, name); env != nil {
		if node, ok := env["executed_on"].(string); ok {
			return node
		}
	}
	return ""
}

// refreshPinHolds re-renders the reservoir pin over the queued resumes: a pin
// whose node is gone holds as reservoir_unavailable naming the node it waits
// for, and a pin whose node came back drops the hold (D12, D14). It never
// places the job elsewhere, which is the whole point of the pin: an operator
// who wants it to run somewhere else starts a fresh run.
//
// The presence read is the registry's, so it runs outside d.mu.
func (d *Dispatcher) refreshPinHolds() {
	pool := d.placementSourceRef()
	if pool == nil {
		return
	}
	type pinned struct{ job, node string }
	var list []pinned
	d.mu.Lock()
	for _, id := range d.queue {
		j := d.pending[id]
		if j == nil || j.canceled || !j.spec.Resume || j.state != StateQueued {
			continue
		}
		if node := requiresFor(&j.spec).Node; node != "" {
			list = append(list, pinned{job: id, node: node})
		}
	}
	d.mu.Unlock()

	for _, p := range list {
		if pool.nodePresent(p.node) {
			d.clearHoldReason(p.job, PlacementReservoirGone)
			continue
		}
		d.markUnplaceable(p.job, PlacementReservoirGone, "node="+p.node)
	}
}

// clearHoldReason drops a job's placement hold when it is the named one, so a
// pin whose node returned stops rendering unplaceable without erasing a hold
// some other scan recorded for another reason.
func (d *Dispatcher) clearHoldReason(jobID, reason string) {
	d.mu.Lock()
	if h := d.holds[jobID]; h != nil && h.reason == reason {
		delete(d.holds, jobID)
	}
	d.mu.Unlock()
}
