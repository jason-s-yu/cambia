package harness

import "github.com/jason-s-yu/cambia/runnerd/procmgr"

// JobView is the API projection of a job: the runnerd-level effective state plus
// the process.json record and the accepted spec's provenance fields.
type JobView struct {
	JobID    string `json:"job_id"`
	State    string `json:"state"`
	Kind     string `json:"kind,omitempty"`
	Priority string `json:"priority,omitempty"`
	QueuePos int    `json:"queue_pos,omitempty"`
	Commit   string `json:"commit,omitempty"`
	Config   string `json:"config,omitempty"`
	Resume   bool   `json:"resume,omitempty"`
	// After keeps its pre-r2 string shape and carries the first parent, so a
	// v1.0 client parsing `after` as it always did keeps working unannounced
	// (D29). AfterAll carries every parent of a fan-in job; a client that
	// advertised understanding of the `fan-in` feature reads it instead.
	// Widening After itself to an array here would be the unannounced payload
	// shape break D30's features list exists to prevent.
	After    string   `json:"after,omitempty"`
	AfterAll []string `json:"after_all,omitempty"`
	// Exclusive echoes the accepted spec's run-alone flag (cambia-655) so an
	// operator listing jobs sees which one holds (or will hold) the daemon. Omitted
	// when false (a normal, concurrency-shared job).
	Exclusive bool `json:"exclusive,omitempty"`
	// HubItem echoes the accepted spec's Codebridge hub link (cambia-353) so the
	// client-side reflector reads it from a single list_jobs poll, no run-dir read
	// required. Telemetry-only; empty for an unlinked job.
	HubItem    string `json:"hub_item,omitempty"`
	PID        int    `json:"pid,omitempty"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	LastError  string `json:"last_error,omitempty"`
	CreatedAt  string `json:"created_at,omitempty"`
	StartedAt  string `json:"started_at,omitempty"`
	FinishedAt string `json:"finished_at,omitempty"`
	// Node names the node holding this job's lease, and Placement carries either
	// "placed" or the hold reason a job that matched no node accumulated past
	// the unplaceable grace, with the union of match rejection reasons in
	// PlacementDetail (D14, D23). All three are empty on a job no pool touched.
	Node            string   `json:"node,omitempty"`
	Placement       string   `json:"placement,omitempty"`
	PlacementDetail []string `json:"placement_detail,omitempty"`
	// Phase is the node-reported lease phase (D5). It is deliberately not the
	// process.json status: only procmgr enum values reach that file, and the
	// fine-grained phase is carried here, where an operator wants the detail.
	Phase string `json:"phase,omitempty"`
	// LogBytesDropped is the coordinator-owned count of log bytes refused at the
	// append route (D54). The in-band truncation marker is advisory text inside
	// a stream the node authors; this counter is the evidence.
	LogBytesDropped int64 `json:"log_bytes_dropped,omitempty"`
}

// viewAfterFields derives a JobView's after/after_all pair from a spec's After
// list (D29): after keeps its pre-r2 string shape, carrying the first parent,
// so a v1.0 client's existing parse of "after" keeps working unannounced;
// after_all carries every parent for a client that reads it. Both are the zero
// value for a job with no parents, matching omitempty on both fields. The
// queue snapshot (Queue/Active, both []JobView) renders the same way, since
// every entry is a JobView built through this helper.
func viewAfterFields(after []string) (string, []string) {
	if len(after) == 0 {
		return "", nil
	}
	return after[0], after
}

// QueueSnapshot is the payload of GET /harness/health-adjacent listings and the
// /ws/harness/queue stream: the reconcile heartbeat, depth/running counts, and
// the queued/active job views.
type QueueSnapshot struct {
	ReconciledAt string    `json:"reconciled_at"`
	QueueDepth   int       `json:"queue_depth"`
	JobsRunning  int       `json:"jobs_running"`
	Queue        []JobView `json:"queue"`
	Active       []JobView `json:"active"`
	// Nodes carries the pool's node records so harness status, harness nodes,
	// and the dashboard show placement without reading run dirs (D23). It is
	// absent on a daemon with no pool attached.
	Nodes []NodeView `json:"nodes,omitempty"`
}

// pendingViewLocked builds the view of a queued/preparing job from its in-memory
// handle. Callers hold d.mu.
func (d *Dispatcher) pendingViewLocked(name string) JobView {
	j := d.pending[name]
	if j == nil {
		return JobView{JobID: name}
	}
	after, afterAll := viewAfterFields(j.spec.After)
	v := JobView{
		JobID:     name,
		State:     j.state,
		Kind:      j.spec.Kind,
		Priority:  j.spec.Priority,
		Commit:    j.spec.Commit,
		Config:    j.spec.Config,
		Resume:    j.spec.Resume,
		After:     after,
		AfterAll:  afterAll,
		Exclusive: j.spec.Exclusive,
		HubItem:   j.spec.HubItem,
		CreatedAt: j.submitAt,
	}
	if j.state == StateQueued {
		v.QueuePos = d.queuePosLocked(name)
	}
	return v
}

// resolveView builds a job view from process.json + jobspec.json, overlaid with
// the in-memory runnerd state (queued/preparing) when present. Returns false
// when the job is unknown on disk and not pending.
func (d *Dispatcher) resolveView(name string) (JobView, bool) {
	st, derr := procmgr.ReadProcessState(d.runDir(name))

	d.mu.Lock()
	j, pending := d.pending[name]
	var pendingState string
	var pos int
	var isResume bool
	var submitAt string
	if pending {
		pendingState = j.state
		isResume = j.spec.Resume
		submitAt = j.submitAt
		if j.state == StateQueued {
			pos = d.queuePosLocked(name)
		}
	}
	d.mu.Unlock()

	if derr != nil && !pending {
		return JobView{}, false
	}

	v := JobView{JobID: name}
	if st != nil {
		v.State = procmgr.EffectiveStatus(st)
		v.Kind = st.Algorithm
		v.PID = st.PID
		v.ExitCode = st.ExitCode
		v.LastError = st.LastError
		v.CreatedAt = st.CreatedAt
		v.StartedAt = st.StartedAt
		v.FinishedAt = st.FinishedAt
	}
	if spec := readJobSpec(d.runDir(name)); spec != nil {
		if v.Kind == "" {
			v.Kind = spec.Kind
		}
		v.Commit = spec.Commit
		v.Config = spec.Config
		v.Priority = spec.Priority
		v.Resume = spec.Resume
		v.After, v.AfterAll = viewAfterFields(spec.After)
		v.Exclusive = spec.Exclusive
		v.HubItem = spec.HubItem
	}
	if pending {
		v.State = pendingState
		v.QueuePos = pos
		v.Resume = isResume
		if v.CreatedAt == "" {
			v.CreatedAt = submitAt
		}
	}
	d.decoratePlacement(&v)
	return v, true
}

// decoratePlacement fills a view's pool fields (D14, D23). A job a node holds a
// live lease for never renders as queued: the node is preparing or running it,
// so the projection's own status wins, and the queue position is dropped
// because the job is no longer waiting for a slot.
func (d *Dispatcher) decoratePlacement(v *JobView) {
	pool := d.placementSourceRef()
	if pool == nil {
		return
	}
	if l, ok := pool.leaseForJob(v.JobID); ok && l.Live() {
		v.Node = l.NodeID
		v.Placement = PlacementPlaced
		v.Phase = l.Phase
		v.QueuePos = 0
		if v.State == StateQueued || v.State == "" || v.State == procmgr.StatusCreated {
			v.State = StatePreparing
		}
		v.LogBytesDropped = pool.logDroppedFor(v.JobID)
		return
	}
	d.mu.Lock()
	reason, detail := d.placementViewLocked(v.JobID)
	d.mu.Unlock()
	v.Placement = reason
	v.PlacementDetail = detail
}

// List returns every known job (queued, preparing, running, terminal), FIFO
// queue order first then the rest by disk scan.
func (d *Dispatcher) List() []JobView {
	seen := map[string]bool{}
	var out []JobView

	d.mu.Lock()
	queueIDs := append([]string(nil), d.queue...)
	d.mu.Unlock()
	for _, id := range queueIDs {
		if v, ok := d.resolveView(id); ok {
			out = append(out, v)
			seen[id] = true
		}
	}

	states, _ := procmgr.ScanProcessStates(d.runsDir)
	for _, st := range states {
		if seen[st.Name] {
			continue
		}
		if v, ok := d.resolveView(st.Name); ok {
			out = append(out, v)
			seen[st.Name] = true
		}
	}
	return out
}

// Snapshot builds the queue/health snapshot: queued views (FIFO), active views
// (preparing + running), and the running count.
func (d *Dispatcher) Snapshot() QueueSnapshot {
	d.mu.Lock()
	queueIDs := append([]string(nil), d.queue...)
	reconciledAt := d.reconciledAt
	var preparingIDs []string
	for name, j := range d.pending {
		if j.state == StatePreparing {
			preparingIDs = append(preparingIDs, name)
		}
	}
	d.mu.Unlock()

	snap := QueueSnapshot{ReconciledAt: reconciledAt, QueueDepth: len(queueIDs)}
	for _, id := range queueIDs {
		if v, ok := d.resolveView(id); ok {
			snap.Queue = append(snap.Queue, v)
		}
	}
	states, _ := procmgr.ScanProcessStates(d.runsDir)
	for _, st := range states {
		switch procmgr.EffectiveStatus(st) {
		case procmgr.StatusRunning, procmgr.StatusStopping:
			snap.JobsRunning++
			if v, ok := d.resolveView(st.Name); ok {
				snap.Active = append(snap.Active, v)
			}
		}
	}
	for _, id := range preparingIDs {
		if v, ok := d.resolveView(id); ok {
			snap.Active = append(snap.Active, v)
		}
	}
	if pool := d.placementSourceRef(); pool != nil {
		snap.Nodes = pool.nodeViews()
	}
	return snap
}

// placementSourceRef reads the attached pool under the lock.
func (d *Dispatcher) placementSourceRef() placementSource {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.pool
}

// ReconciledAt returns the last reconcile timestamp (RFC3339).
func (d *Dispatcher) ReconciledAt() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.reconciledAt
}

// Subscribe registers a snapshot channel for the /ws/harness/queue stream and
// returns an unsubscribe func. The channel is buffered; a slow subscriber drops
// intermediate snapshots (latest-wins).
func (d *Dispatcher) Subscribe() (<-chan QueueSnapshot, func()) {
	ch := make(chan QueueSnapshot, 4)
	d.mu.Lock()
	d.subs[ch] = struct{}{}
	d.mu.Unlock()
	return ch, func() {
		d.mu.Lock()
		delete(d.subs, ch)
		d.mu.Unlock()
	}
}

// broadcast pushes the current snapshot to every subscriber. It builds the
// snapshot without holding d.mu (Snapshot locks briefly itself), then copies the
// subscriber set under the lock and sends non-blocking.
func (d *Dispatcher) broadcast() {
	snap := d.Snapshot()
	d.mu.Lock()
	subs := make([]chan QueueSnapshot, 0, len(d.subs))
	for ch := range d.subs {
		subs = append(subs, ch)
	}
	pool := d.pool
	d.mu.Unlock()
	if pool != nil {
		// Every queue transition is a placement opportunity: waking the held
		// claims here is what lets a chain of ready jobs run without poll
		// latency (D2).
		pool.signalPlacement()
	}
	for _, ch := range subs {
		select {
		case ch <- snap:
		default:
		}
	}
}
