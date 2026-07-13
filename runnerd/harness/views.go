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
	After    string `json:"after,omitempty"`
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
}

// pendingViewLocked builds the view of a queued/preparing job from its in-memory
// handle. Callers hold d.mu.
func (d *Dispatcher) pendingViewLocked(name string) JobView {
	j := d.pending[name]
	if j == nil {
		return JobView{JobID: name}
	}
	v := JobView{
		JobID:     name,
		State:     j.state,
		Kind:      j.spec.Kind,
		Priority:  j.spec.Priority,
		Commit:    j.spec.Commit,
		Config:    j.spec.Config,
		Resume:    j.resume,
		After:     j.spec.After,
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
		isResume = j.resume
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
		v.After = spec.After
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
	return v, true
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
	return snap
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
	d.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- snap:
		default:
		}
	}
}
