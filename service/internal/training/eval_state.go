package training

// EvalJob is the in-memory record for one `cambia evaluate` subprocess. It is
// the authoritative answer to "is my eval job queued / running / done / failed"
// for one server lifetime; the eval RESULTS are durable in run_db, this
// registry is not (a server restart drops it, the user re-triggers). It marshals
// to the pinned wire shape consumed by the frontend EvalControls panel.
type EvalJob struct {
	ID         string   `json:"id"`     // compact-timestamp + counter, unique per manager
	Run        string   `json:"run"`    // run name (validated before use)
	Status     string   `json:"status"` // queued|running|succeeded|failed
	Target     string   `json:"target"` // "latest" | "iter:<N>"
	Device     string   `json:"device"` // cpu|cuda
	Games      int      `json:"games"`
	Argmax     bool     `json:"argmax"`
	LogPath    string   `json:"log_path"`
	StartedAt  string   `json:"started_at,omitempty"`
	FinishedAt string   `json:"finished_at,omitempty"`
	ExitCode   *int     `json:"exit_code,omitempty"`
	Error      string   `json:"error,omitempty"`
	Tail       []string `json:"tail,omitempty"` // last ~40 log lines, populated on GET only
}

// Eval job status constants.
const (
	EvalQueued    = "queued"
	EvalRunning   = "running"
	EvalSucceeded = "succeeded"
	EvalFailed    = "failed"
)

// maxEvalJobsPerRun bounds the per-run job history (contract R6). The oldest
// job is evicted when a new one pushes the count past the cap.
const maxEvalJobsPerRun = 20

// jobRegistry holds per-run eval jobs newest-first with a bounded retention
// window. It carries no lock of its own: every method assumes the caller (the
// EvalManager) holds m.mu, so the cap check and the registry mutation stay one
// atomic critical section (mirroring ProcessManager.mutateStateLocked).
type jobRegistry struct {
	jobs map[string][]*EvalJob
}

// newJobRegistry constructs an empty registry.
func newJobRegistry() *jobRegistry {
	return &jobRegistry{jobs: make(map[string][]*EvalJob)}
}

// add prepends j to name's history (newest-first) and evicts entries beyond the
// retention cap. Caller holds the manager lock.
func (r *jobRegistry) add(name string, j *EvalJob) {
	lst := append([]*EvalJob{j}, r.jobs[name]...)
	if len(lst) > maxEvalJobsPerRun {
		lst = lst[:maxEvalJobsPerRun]
	}
	r.jobs[name] = lst
}

// list returns a shallow copy of name's job pointers, newest-first. Caller holds
// the manager lock. The pointers are the live records; callers that release the
// lock before touching them must snapshot the struct values first.
func (r *jobRegistry) list(name string) []*EvalJob {
	src := r.jobs[name]
	out := make([]*EvalJob, len(src))
	copy(out, src)
	return out
}
