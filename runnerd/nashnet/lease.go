package nashnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// LeaseFileName is the basename of the per-job lease record, written beside
// process.json and jobspec.json in the run directory (D4). Reconcile reads it
// at startup, which is why a coordinator restart does not lose placement (D34).
const LeaseFileName = "lease.json"

// Lease states (D4). revoking is the wind-down state a coordinator-initiated
// stop of a node in good standing enters: the lease token stays valid and
// lease_epoch does not advance for one grace period, so the node can commit a
// final manifest and post its terminal. released is terminal for the lease, by
// a posted result or by a revocation that dropped the token.
const (
	LeaseActive   = "active"
	LeaseRevoking = "revoking"
	LeaseReleased = "released"
)

// Verdict is what an ended lease means for its job (D7). It is a function of
// the furthest phase reached and whether a pid was ever projected, never of the
// projection's status string.
type Verdict string

const (
	// VerdictRequeue is the never-launched row: the job returns to the ready set
	// at its original submit_seq with attempt++ (D32).
	VerdictRequeue Verdict = "requeue"
	// VerdictFinalize is the executed-outcome-unknown row: the job is settled by
	// the two-witness rule against the promoted run_db.sqlite (D35). Never a
	// retry: a job that ran is never re-run (D33).
	VerdictFinalize Verdict = "finalize"
	// VerdictCanceled is the operator-stop row: canceled with no exit code, keyed
	// on the lease record's stop_requested_at, which only the coordinator writes.
	VerdictCanceled Verdict = "canceled"
)

// Reasons carried on an Outcome, naming why the lease ended.
const (
	ReasonExpired        = "lease_expired"
	ReasonRuntimeCap     = "runtime_cap"
	ReasonGraceElapsed   = "revoke_grace_elapsed"
	ReasonNodeReRegister = "node_reregistered"
	ReasonNodeRevoked    = "node_revoked"
)

// ErrInvalidPhase is returned for a phase outside the node's vocabulary (D5),
// which the server maps to 422 invalid_phase.
var ErrInvalidPhase = errors.New("nashnet: invalid phase")

// phaseRank orders the phases so a lease can record the furthest one reached
// rather than the last one reported. stopping ranks with the post-launch
// phases: it is coordinator-written and only ever follows a launch.
var phaseRank = map[string]int{
	PhaseClaimed:    1,
	PhaseFetching:   2,
	PhasePreparing:  3,
	PhaseRunning:    4,
	PhaseStopping:   5,
	PhaseUploading:  6,
	PhaseCommitting: 7,
}

// nodePhases is the phase vocabulary a node may report (D5). stopping is
// absent: the coordinator writes it, and only when it initiated the stop, which
// keeps the first witness of the two-witness rule a coordinator fact.
var nodePhases = map[string]bool{
	PhaseClaimed:    true,
	PhaseFetching:   true,
	PhasePreparing:  true,
	PhaseRunning:    true,
	PhaseUploading:  true,
	PhaseCommitting: true,
}

// ValidateNodePhase reports whether phase is one a node may report (D5).
func ValidateNodePhase(phase string) error {
	if !nodePhases[phase] {
		return fmt.Errorf("%w: %q", ErrInvalidPhase, phase)
	}
	return nil
}

// PhaseLaunched reports whether a phase is one the job could only have reached
// by launching a process: the second row of the D7 table.
func PhaseLaunched(phase string) bool {
	switch phase {
	case PhaseRunning, PhaseStopping, PhaseUploading, PhaseCommitting:
		return true
	}
	return false
}

// GrantEntry is one file of the lease grant set: the path as the seed named it
// and the digest the node must verify (D4, D53).
type GrantEntry struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// GrantSet is what a lease authorizes its holder to read: the snapshot bundle
// digest and the seed files resolved at placement (D4). Every read on a lease
// route is fenced on membership here, so a lease cannot name a coordinator path
// or another lease's input.
type GrantSet struct {
	Snapshot string                  `json:"snapshot,omitempty"`
	Seeds    map[string][]GrantEntry `json:"seeds,omitempty"`
}

// Allows reports whether seedID and path are in the grant set and returns the
// digest recorded for them. A miss is answered with the same 404 body as an
// unknown seed, so the route is no existence oracle (D53).
func (g GrantSet) Allows(seedID, path string) (string, bool) {
	for _, e := range g.Seeds[seedID] {
		if e.Path == path {
			return e.SHA256, true
		}
	}
	return "", false
}

// clone deep-copies the grant set so a caller holding a returned lease cannot
// mutate the store's copy through the seed map.
func (g GrantSet) clone() GrantSet {
	out := GrantSet{Snapshot: g.Snapshot}
	if g.Seeds == nil {
		return out
	}
	out.Seeds = make(map[string][]GrantEntry, len(g.Seeds))
	for k, v := range g.Seeds {
		entries := make([]GrantEntry, len(v))
		copy(entries, v)
		out.Seeds[k] = entries
	}
	return out
}

// Lease is the in-memory lease record. Its persisted form is lease.json (D4);
// timestamps are RFC3339 on disk and time.Time here so the sweeper and the
// fences do their arithmetic without reparsing.
//
// Phase, PIDProjected, and MaxRuntime are not in the D4 sample record but are
// persisted with it, because the D7 verdict is a function of the furthest phase
// reached and whether a pid was ever projected, and both facts must survive the
// coordinator restart D34 restores a lease across. MaxRuntime is the
// spec-supplied max_runtime_hours resolved at grant, so the runtime cap of D4
// stays a lease fact rather than a second lookup into the spec at sweep time.
//
// NextAttempt is the attempt the job runs at once this lease has ended: equal
// to Attempt for every verdict but requeue, one higher for that one (D32). It
// is written with the released record rather than held in memory, so a
// coordinator restart resumes the count where the last verdict left it instead
// of handing every requeued job a fresh attempt 1 and never exhausting
// max_attempts.
type Lease struct {
	JobID           string
	NodeID          string
	NodeEpoch       int64
	LeaseID         string
	LeaseEpoch      int64
	State           string
	TokenHash       string
	GrantedAt       time.Time
	Deadline        time.Time
	StopRequestedAt time.Time
	Attempt         int
	NextAttempt     int
	GrantSet        GrantSet
	ManifestSeq     int64
	ManifestDigest  string
	Phase           string
	PIDProjected    bool
	MaxRuntime      time.Duration
}

// Clone returns a deep copy. The store hands out clones so no caller holds a
// pointer into state the sweeper mutates under the lock.
func (l *Lease) Clone() Lease {
	c := *l
	c.GrantSet = l.GrantSet.clone()
	return c
}

// Live reports whether the lease still holds its job: active or winding down,
// but not released.
func (l *Lease) Live() bool {
	return l.State == LeaseActive || l.State == LeaseRevoking
}

// Verdict applies the D7 table to an ended lease.
//
// Row three keys on stop_requested_at, which only the coordinator writes and
// only when it initiated the stop (D31, D60), so a node reporting a stop it was
// never asked for cannot forge the operator-stop witness. It is evaluated first
// rather than last: the phase column of that row reads "any", and an operator
// cancel of a lease still fetching would otherwise fall into the requeue row
// and relaunch the job the operator just stopped.
//
// Rows one and two split on whether the job could have run. A pid ever
// projected settles it; so does a post-launch phase with no pid yet recorded,
// which is the conservative reading and the one D35 wants, since finalizing by
// two witnesses against promoted state costs a wrong crashed verdict while
// requeueing a job that ran costs a second execution.
func (l *Lease) Verdict() Verdict {
	if !l.StopRequestedAt.IsZero() {
		return VerdictCanceled
	}
	if l.PIDProjected || PhaseLaunched(l.Phase) {
		return VerdictFinalize
	}
	return VerdictRequeue
}

// Expired reports whether the lease is past its renewal deadline.
func (l *Lease) Expired(now time.Time) bool {
	return !l.Deadline.IsZero() && now.After(l.Deadline)
}

// RuntimeCapReached reports whether elapsed time from granted_at has passed the
// smaller of the pool cap and the spec's own max_runtime_hours (D4). It is
// checked regardless of renewals: renewals alone are not a liveness argument,
// since a node posting progress forever would otherwise hold its lease forever,
// block the job's dependents, and keep an exclusive barrier standing.
func (l *Lease) RuntimeCapReached(now time.Time, poolCap time.Duration) bool {
	limit := poolCap
	if l.MaxRuntime > 0 && (limit <= 0 || l.MaxRuntime < limit) {
		limit = l.MaxRuntime
	}
	if limit <= 0 || l.GrantedAt.IsZero() {
		return false
	}
	return !now.Before(l.GrantedAt.Add(limit))
}

// advancePhase records phase when it is further along than the furthest one
// reached, so a node that reports an earlier phase after a later one does not
// walk the D7 verdict backwards.
func (l *Lease) advancePhase(phase string) {
	if phaseRank[phase] > phaseRank[l.Phase] {
		l.Phase = phase
	}
}

// Route names a lease route for the revoking-state gate below. The manifest
// has three of them rather than one because the grace admits them differently:
// RouteManifestHead is the GET of the folded head every commit fast-forwards
// from, RouteManifest is a rolling non-final commit, and RouteManifestFinal is
// the commit that closes the artifact stream.
type Route string

const (
	RouteProgress      Route = "progress"
	RouteLogs          Route = "logs"
	RouteBlobs         Route = "blobs"
	RouteManifestHead  Route = "manifest_head"
	RouteManifest      Route = "manifest"
	RouteManifestFinal Route = "manifest_final"
	RouteSnapshot      Route = "snapshot"
	RouteSeeds         Route = "seeds"
	RouteNack          Route = "nack"
	RouteResult        Route = "result"
)

// PermitsRoute reports whether the lease's current state admits a call on
// route. While revoking the coordinator accepts log appends, blob chunks, a
// final manifest, and a terminal; a progress post is admitted but renews
// nothing and answers revoke (D4). It also admits the manifest head, because
// the grace exists so the final commit can land and that commit fast-forwards
// from the head it reads first: refusing the read refuses the commit, which
// orphans the output D62 says the node still posts. Refused is the writing
// half, the non-final commit, along with the input reads a stopping job no
// longer needs. Everything else is lease_superseded, which is what keeps a
// superseded writer and a fresh lease off one run dir.
func (l *Lease) PermitsRoute(route Route) bool {
	switch l.State {
	case LeaseActive:
		return true
	case LeaseRevoking:
		switch route {
		case RouteLogs, RouteBlobs, RouteManifestHead, RouteManifestFinal, RouteResult, RouteProgress:
			return true
		}
		return false
	default:
		return false
	}
}

// PermitsResult reports whether a terminal state may be posted in the lease's
// current state. A revoking lease accepts only canceled or preempted: a node
// asked to stop cannot report the run as a clean stop or a crash (D4, D62).
func (l *Lease) PermitsResult(state string) bool {
	switch l.State {
	case LeaseActive:
		return true
	case LeaseRevoking:
		return state == ResultCanceled || state == ResultPreempted
	default:
		return false
	}
}

// leaseFile is the on-disk form of a lease record, exactly the shape of D4 with
// the three restart-visibility fields the verdict needs.
type leaseFile struct {
	JobID           string   `json:"job_id"`
	NodeID          string   `json:"node_id"`
	NodeEpoch       int64    `json:"node_epoch"`
	LeaseID         string   `json:"lease_id"`
	LeaseEpoch      int64    `json:"lease_epoch"`
	State           string   `json:"state"`
	TokenHash       string   `json:"token_hash"`
	GrantedAt       string   `json:"granted_at"`
	Deadline        string   `json:"deadline"`
	StopRequestedAt string   `json:"stop_requested_at"`
	Attempt         int      `json:"attempt"`
	NextAttempt     int      `json:"next_attempt,omitempty"`
	GrantSet        GrantSet `json:"grant_set"`
	ManifestSeq     int64    `json:"manifest_seq"`
	ManifestDigest  string   `json:"manifest_digest"`
	Phase           string   `json:"phase,omitempty"`
	PIDProjected    bool     `json:"pid_projected,omitempty"`
	MaxRuntimeSecs  int64    `json:"max_runtime_seconds,omitempty"`
}

// rfc3339 renders a timestamp for the record, with the zero time rendering as
// the empty string the D4 sample shows for an unset stop_requested_at.
func rfc3339(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339Nano)
}

// parseTime is rfc3339's inverse, tolerating the empty string as the zero time.
func parseTime(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	return time.Parse(time.RFC3339Nano, s)
}

func (l *Lease) toFile() leaseFile {
	return leaseFile{
		JobID:           l.JobID,
		NodeID:          l.NodeID,
		NodeEpoch:       l.NodeEpoch,
		LeaseID:         l.LeaseID,
		LeaseEpoch:      l.LeaseEpoch,
		State:           l.State,
		TokenHash:       l.TokenHash,
		GrantedAt:       rfc3339(l.GrantedAt),
		Deadline:        rfc3339(l.Deadline),
		StopRequestedAt: rfc3339(l.StopRequestedAt),
		Attempt:         l.Attempt,
		NextAttempt:     l.NextAttempt,
		GrantSet:        l.GrantSet,
		ManifestSeq:     l.ManifestSeq,
		ManifestDigest:  l.ManifestDigest,
		Phase:           l.Phase,
		PIDProjected:    l.PIDProjected,
		MaxRuntimeSecs:  int64(l.MaxRuntime / time.Second),
	}
}

func (f *leaseFile) toLease() (*Lease, error) {
	granted, err := parseTime(f.GrantedAt)
	if err != nil {
		return nil, fmt.Errorf("granted_at: %w", err)
	}
	deadline, err := parseTime(f.Deadline)
	if err != nil {
		return nil, fmt.Errorf("deadline: %w", err)
	}
	stop, err := parseTime(f.StopRequestedAt)
	if err != nil {
		return nil, fmt.Errorf("stop_requested_at: %w", err)
	}
	return &Lease{
		JobID:           f.JobID,
		NodeID:          f.NodeID,
		NodeEpoch:       f.NodeEpoch,
		LeaseID:         f.LeaseID,
		LeaseEpoch:      f.LeaseEpoch,
		State:           f.State,
		TokenHash:       f.TokenHash,
		GrantedAt:       granted,
		Deadline:        deadline,
		StopRequestedAt: stop,
		Attempt:         f.Attempt,
		NextAttempt:     f.NextAttempt,
		GrantSet:        f.GrantSet,
		ManifestSeq:     f.ManifestSeq,
		ManifestDigest:  f.ManifestDigest,
		Phase:           f.Phase,
		PIDProjected:    f.PIDProjected,
		MaxRuntime:      time.Duration(f.MaxRuntimeSecs) * time.Second,
	}, nil
}

// WriteLease writes the lease record into runDir atomically: temp file, fsync,
// rename over the destination. It is the pattern of procmgr.WriteProcessState
// (procmgr/state.go:74) rather than a call to it, because nashnet is a leaf
// package and imports no other runnerd package.
func WriteLease(runDir string, l *Lease) error {
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(l.toFile(), "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')

	tmp := filepath.Join(runDir, LeaseFileName+".tmp")
	final := filepath.Join(runDir, LeaseFileName)

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
	return nil
}

// ReadLease reads and decodes runDir/lease.json.
func ReadLease(runDir string) (*Lease, error) {
	data, err := os.ReadFile(filepath.Join(runDir, LeaseFileName))
	if err != nil {
		return nil, err
	}
	var f leaseFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	return f.toLease()
}
