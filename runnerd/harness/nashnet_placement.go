package harness

import (
	"sort"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Placement hold reasons rendered in a JobView after the unplaceable grace
// (D14). They name a fact rather than a failure, and none of them fails a job.
const (
	PlacementWaitingForNode     = "waiting_for_node"
	PlacementWaitingForNodeGate = "waiting_for_node_gate"
	PlacementSeedMissing        = "seed_missing"
	PlacementReservoirGone      = "reservoir_unavailable"
	// PlacementPlaced is the rendering of a job a node holds a live lease for.
	PlacementPlaced = "placed"
)

// placementSource is the coordinator pool the dispatcher reads for the
// placement half of a view and wakes when the queue moves. It is an interface
// so the dispatcher keeps compiling with no pool attached, which is the
// zero-node daemon of D40.
type placementSource interface {
	leaseForJob(jobID string) (nashnet.Lease, bool)
	nodeViews() []NodeView
	logDroppedFor(jobID string) int64
	signalPlacement()
	// stopLease moves a live lease to revoking and tells its node to stop,
	// which is the whole of a cancel on a leased job (D31).
	stopLease(lease nashnet.Lease, force bool) error
	// purgeQuarantine deletes every lease tree a purged job left behind (D31).
	purgeQuarantine(jobID string) error
	// nodePresent reports whether a pinned node could still claim (D12).
	nodePresent(nodeID string) bool
	// manifestHead is the job's promoted manifest head, which is what the
	// resume gate reads rather than the run dir's raw contents (D31).
	manifestHead(jobID string) (quarantine.Head, bool)
}

// placementHold is one job's accumulated match rejection, rendered after the
// grace (D14).
type placementHold struct {
	since   time.Time
	reason  string
	details map[string]bool
	// nextEligible is the earliest gate reopen across the nodes that refused,
	// so an operator sees a time window rather than a capability gap.
	nextEligible time.Time
}

// candidate is one claiming node as the placement scan reads it: the grant-
// clamped declaration (D47), the node's own gate report (D46), and the claim's
// volatile facts. Every field is either coordinator-held or clamped, so a node
// inflating its declaration attracts nothing beyond its enrollment.
type candidate struct {
	nodeID      string
	declaration capability.Declaration
	grantLabels []string
	report      gates.Report
	kinds       map[string]bool
	slotsFree   int
	leaseRoom   int
	busy        map[string]bool
}

// pick is one placement decision the scan reserved for a node.
type pick struct {
	jobID   string
	spec    JobSpec
	attempt int
	resume  bool
}

// attachPool wires the coordinator pool into the dispatcher: the lease store
// the view rendering reads, the placement source, and the unplaceable grace.
func (d *Dispatcher) attachPool(p placementSource, leases *nashnet.LeaseStore, grace time.Duration, now func() time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.pool = p
	d.leases = leases
	d.unplaceableGrace = grace
	d.now = now
	d.holds = map[string]*placementHold{}
	d.placing = map[string]string{}
}

// scanForNode is the placement scan of D11, run under d.mu with no I/O beyond
// the bounded parent process.json reads the dependency gate already does. It
// walks the queue in submit_seq order and, for each ready job, applies the node
// gates' admit verdict, Match, exclusivity, the pin, and the lease ceiling, in
// that order; the first match wins, as dispatchLocked already does for
// gate-blocked jobs. It reserves the picked job so a concurrent claim skips it,
// and it forks nothing: seed resolution and the bundle build run after it
// returns, outside the lock (D48).
func (d *Dispatcher) scanForNode(c candidate, skip map[string]bool, held func(jobID string) bool) (*pick, string) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if c.leaseRoom <= 0 || c.slotsFree <= 0 {
		return nil, nashnet.HoldNoMatch
	}
	// The node's exclusive lease is alone on its node: while it stands, the
	// coordinator hands that node nothing else (D13).
	for _, l := range d.leasesForNodeLocked(c.nodeID) {
		if d.exclusiveLocked(l.JobID) {
			return nil, nashnet.HoldExclusivePending
		}
	}

	hold := nashnet.HoldNoMatch
	exclusiveBarrier := false
	for _, id := range d.queue {
		j := d.pending[id]
		if j == nil || j.canceled || j.state != StateQueued {
			continue
		}
		if skip[id] || d.placing[id] != "" {
			continue
		}
		if decision, _ := d.gateDecisionLocked(j); decision != gateLaunch {
			continue
		}
		if d.leases != nil {
			if l, ok := d.leases.ByJob(id); ok && l.Live() {
				continue
			}
		}
		if held(id) || c.busy[id] {
			continue
		}
		if exclusiveBarrier {
			// A later ready job this node matches waits behind an older
			// exclusive ready job it also matches (D13).
			if ok, _ := d.matchesLocked(j, c); ok {
				hold = nashnet.HoldExclusivePending
			}
			continue
		}
		ok, reasons := d.matchesLocked(j, c)
		if !ok {
			d.recordHoldLocked(id, reasons, c)
			continue
		}
		if j.spec.Exclusive && len(d.leasesForNodeLocked(c.nodeID)) > 0 {
			exclusiveBarrier = true
			hold = nashnet.HoldExclusivePending
			continue
		}
		d.placing[id] = c.nodeID
		delete(d.holds, id)
		return &pick{jobID: id, spec: j.spec, attempt: 1, resume: j.resume}, ""
	}
	return nil, hold
}

// noteGateDenied records the gate hold of D14 for every ready job this node
// would otherwise match. A denying report is node-wide, so no scan hands work
// out; without this pass a job held only because every capable node reports
// admit: false would render as an unexplained wait rather than as a time window
// with the failing check names.
func (d *Dispatcher) noteGateDenied(c candidate) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, id := range d.queue {
		j := d.pending[id]
		if j == nil || j.canceled || j.state != StateQueued {
			continue
		}
		if decision, _ := d.gateDecisionLocked(j); decision != gateLaunch {
			continue
		}
		if ok, _ := d.matchesLocked(j, c); !ok {
			continue
		}
		d.recordHoldLocked(id, nil, c)
	}
}

// matchesLocked applies the coordinator-authoritative half of admission to one
// job (D11): the node's kinds_allowed and devices_allowed gates, the pin, the
// node's own per-job runtime policy, and the pure Match over the clamped
// declaration. Callers hold d.mu.
func (d *Dispatcher) matchesLocked(j *job, c candidate) (bool, []capability.Reason) {
	req := requiresFor(&j.spec)
	if req.Node != "" && req.Node != c.nodeID {
		return false, []capability.Reason{capability.ReasonNodeMismatch}
	}
	if len(c.kinds) > 0 && !c.kinds[j.spec.Kind] {
		return false, []capability.Reason{capability.Reason("kinds_allowed")}
	}
	if h := nodeMaxRuntimeHours(c.report); h > 0 && j.spec.MaxRuntimeHours > h {
		return false, []capability.Reason{capability.Reason("job_policy")}
	}
	decl := c.declaration
	decl.Devices = allowedDevices(decl.Devices, c.report.DevicesAllowed)
	return capability.Match(j.spec.Kind, req, capability.Candidate{
		NodeID:      c.nodeID,
		Declaration: decl,
		GrantLabels: c.grantLabels,
	})
}

// requiresFor resolves a spec's placement constraints with the defaults of D10:
// device from the job's resolved device, min_cores 1, needs_libcambia true.
func requiresFor(spec *JobSpec) capability.Requires {
	var req capability.Requires
	if spec.Requires != nil {
		req = *spec.Requires
	}
	return req.Normalize(spec.device())
}

// allowedDevices filters a declaration's devices to the gate's resolved id set
// (D46). An empty set means the node configured no devices_allowed gate, which
// never blocks.
func allowedDevices(devices []capability.Device, allowed []string) []capability.Device {
	if len(allowed) == 0 {
		return devices
	}
	ids := map[string]bool{}
	for _, id := range allowed {
		ids[id] = true
	}
	out := make([]capability.Device, 0, len(devices))
	for _, dev := range devices {
		if ids[dev.ID] {
			out = append(out, dev)
		}
	}
	return out
}

// nodeMaxRuntimeHours reads the node's own job_policy.max_runtime_hours from
// its gate report, where a node that runs the check publishes it as a check
// with its required value. It is a placement input only: the coordinator's own
// runtime bound is the lease cap of D4, which does not depend on a node number.
func nodeMaxRuntimeHours(report gates.Report) float64 {
	for _, c := range report.Checks {
		if c.Gate == "job_policy.max_runtime_hours" && c.Required != nil {
			return *c.Required
		}
	}
	return 0
}

// exclusiveLocked reports whether a leased job runs alone on its node (D13). A
// leased job keeps its in-memory handle, so the common path reads no file;
// after a coordinator restart the persisted spec answers instead. Callers hold
// d.mu.
func (d *Dispatcher) exclusiveLocked(jobID string) bool {
	if j := d.pending[jobID]; j != nil {
		return j.spec.Exclusive
	}
	spec := readJobSpec(d.runDir(jobID))
	return spec != nil && spec.Exclusive
}

// leasesForNodeLocked lists a node's live leases through the attached store.
// Callers hold d.mu.
func (d *Dispatcher) leasesForNodeLocked(nodeID string) []nashnet.Lease {
	if d.leases == nil {
		return nil
	}
	return d.leases.LiveForNode(nodeID)
}

// recordHoldLocked accumulates the union of match rejection reasons for a job
// no node matched, with the earliest gate reopen across the nodes that refused
// (D14). Callers hold d.mu.
func (d *Dispatcher) recordHoldLocked(jobID string, reasons []capability.Reason, c candidate) {
	h := d.holds[jobID]
	if h == nil {
		h = &placementHold{since: d.nowFn(), details: map[string]bool{}, reason: PlacementWaitingForNode}
		d.holds[jobID] = h
	}
	for _, r := range reasons {
		h.details[string(r)] = true
	}
	if !c.report.Admit {
		h.reason = PlacementWaitingForNodeGate
		for _, chk := range c.report.Checks {
			if !chk.OK {
				h.details["gate:"+chk.Gate] = true
			}
		}
		if c.report.NextEligibleAt != nil {
			if h.nextEligible.IsZero() || c.report.NextEligibleAt.Before(h.nextEligible) {
				h.nextEligible = *c.report.NextEligibleAt
			}
		}
	}
}

// markUnplaceable records a job-level placement defect the scan found (D14,
// D53): a seed it could not resolve where the referenced run dir still exists,
// or a resume pinned to a node that is gone. The job stays queued.
func (d *Dispatcher) markUnplaceable(jobID, reason, detail string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	h := d.holds[jobID]
	if h == nil {
		h = &placementHold{since: d.nowFn(), details: map[string]bool{}}
		d.holds[jobID] = h
	}
	h.reason = reason
	if detail != "" {
		h.details[detail] = true
	}
}

// releasePlacing drops the reservation the scan took on a job, so the next
// claim may consider it again.
func (d *Dispatcher) releasePlacing(jobID string) {
	d.mu.Lock()
	delete(d.placing, jobID)
	d.mu.Unlock()
}

// placementViewLocked renders a job's placement fields (D14, D23). Callers hold
// d.mu.
func (d *Dispatcher) placementViewLocked(jobID string) (string, []string) {
	h := d.holds[jobID]
	if h == nil {
		return "", nil
	}
	if h.reason != PlacementSeedMissing && h.reason != PlacementReservoirGone &&
		d.nowFn().Sub(h.since) < d.unplaceableGrace {
		return "", nil
	}
	details := make([]string, 0, len(h.details))
	for k := range h.details {
		details = append(details, k)
	}
	sort.Strings(details)
	if !h.nextEligible.IsZero() {
		details = append(details, "next_eligible_at="+rfc3339(h.nextEligible))
	}
	return h.reason, details
}

// nowFn is the dispatcher's clock, injected by the pool so the unplaceable
// grace is testable without sleeping.
func (d *Dispatcher) nowFn() time.Time {
	if d.now != nil {
		return d.now()
	}
	return time.Now()
}

// recordPoolTerminal writes a coordinator-decided terminal for a pool job and
// takes it out of the queue. It is the canceled row of the D7 verdict table and
// the seed-missing spec-fatal of D53.
func (d *Dispatcher) recordPoolTerminal(jobID, state, lastErr string) {
	runDir := d.runDir(jobID)
	st, err := procmgr.ReadProcessState(runDir)
	if err != nil {
		st = &procmgr.ProcessState{Name: jobID}
	}
	st.Status = state
	st.LastError = lastErr
	if st.FinishedAt == "" {
		st.FinishedAt = procmgr.NowRFC3339()
	}
	_ = procmgr.WriteProcessState(runDir, st)

	d.mu.Lock()
	if j := d.pending[jobID]; j != nil {
		j.cancel()
		delete(d.pending, jobID)
	}
	d.removeFromQueueLocked(jobID)
	delete(d.placing, jobID)
	delete(d.holds, jobID)
	d.dispatchLocked()
	d.mu.Unlock()
	d.broadcast()
}

// finalizePoolJob settles a job whose lease ended after its process had run, by
// the two-witness rule against promoted state (D7, D35).
func (d *Dispatcher) finalizePoolJob(jobID string) {
	d.mu.Lock()
	if j := d.pending[jobID]; j != nil {
		j.cancel()
		delete(d.pending, jobID)
	}
	d.removeFromQueueLocked(jobID)
	delete(d.placing, jobID)
	delete(d.holds, jobID)
	d.mu.Unlock()

	d.finalizeReattached(jobID)
	d.reDispatch()
}

// clearPlaced removes a settled job from the queue and the reservation table
// after a node posted its result (D6).
func (d *Dispatcher) clearPlaced(jobID string) {
	d.mu.Lock()
	if j := d.pending[jobID]; j != nil {
		j.cancel()
		delete(d.pending, jobID)
	}
	d.removeFromQueueLocked(jobID)
	delete(d.placing, jobID)
	delete(d.holds, jobID)
	d.dispatchLocked()
	d.mu.Unlock()
	d.broadcast()
}
