package nashnet

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Lease-store errors. The server ticket maps them to status codes: an unknown
// lease and a dropped token are 401, which is the code D36 obliges a node to
// treat exactly as it treats 409, and everything superseded is
// 409 lease_superseded.
var (
	// ErrUnknownLease is an id no lease record carries.
	ErrUnknownLease = errors.New("nashnet: unknown lease")
	// ErrLeaseTokenDropped is a lease whose token died with it: the grace
	// expired, the epoch was bumped, or the node was revoked (D44).
	ErrLeaseTokenDropped = errors.New("nashnet: lease token dropped")
	// ErrLeaseSuperseded is the fence: a stale lease_epoch or node_epoch, a
	// wrong token, or a route the lease's state does not admit (D4).
	ErrLeaseSuperseded = errors.New("nashnet: lease superseded")
	// ErrJobLeased is a grant for a job that already holds a live lease. It is a
	// coordinator bug rather than a node error: the placement scan must not hand
	// one job to two nodes, and a lease in revoking blocks a re-claim until it
	// clears (D4).
	ErrJobLeased = errors.New("nashnet: job already leased")
	// ErrInvalidJobID rejects a job id that is unsafe as a path segment. The
	// authority is procmgr.ValidateName at submit; this is the leaf package's own
	// restatement, since it joins the id onto the runs dir.
	ErrInvalidJobID = errors.New("nashnet: invalid job id")
)

// SkipEpoch tells the fence that the caller has no value for that half of it.
// It is spelled out rather than left to a zero value so a forgotten field fails
// closed: a lease route that carries an epoch fences on it strictly, and only a
// caller that means "I hold no epoch to compare" says so. The two callers that
// do are the lease routes carrying no epoch in their body (logs, blob chunks,
// snapshot and seed reads) and the window after a coordinator restart, where a
// restored lease is fenced on a node epoch the fresh registry does not yet hold
// (D34).
const SkipEpoch int64 = -1

// StoreConfig configures a LeaseStore. Every field has a documented default, so
// the zero value plus a runs dir is a working store.
type StoreConfig struct {
	// RunsDir is the coordinator's runs directory; a lease persists to
	// <RunsDir>/<job>/lease.json (D4).
	RunsDir string
	// Policy carries the lease TTL and the pool runtime cap (D2, D56).
	Policy Policy
	// RevokeGrace is how long a revoking lease keeps its token and its epoch
	// before the coordinator advances the epoch and applies the D7 verdict (D4).
	// The brief fixes no number for it; the default is one lease TTL, which is
	// the window a cooperating node needs to notice the revoke on its events
	// poll or its next progress tick, stop the process group with the 30s grace,
	// and commit a final manifest.
	RevokeGrace time.Duration
	// Now is the injected clock. Nil means time.Now.
	Now func() time.Time
	// Entropy is the source for lease ids and tokens. Nil means crypto/rand.
	Entropy io.Reader
}

func (c StoreConfig) withDefaults() StoreConfig {
	if c.Policy.LeaseTTLSeconds <= 0 {
		c.Policy.LeaseTTLSeconds = DefaultLeaseTTLSeconds
	}
	if c.Policy.MaxLeaseSeconds <= 0 {
		c.Policy.MaxLeaseSeconds = DefaultMaxLeaseSeconds
	}
	if c.RevokeGrace <= 0 {
		c.RevokeGrace = time.Duration(c.Policy.LeaseTTLSeconds) * time.Second
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// LeaseStore owns every lease record: it grants, fences, renews, revokes, and
// expires them, and it is the only writer of runs/<job>/lease.json (D4). It
// holds no HTTP, no dispatcher, and no clock of its own.
type LeaseStore struct {
	cfg StoreConfig

	mu     sync.Mutex
	leases map[string]*Lease // by lease id, retained past release for D6 replay and the D7 verdict
	byJob  map[string]*Lease // by job id, the job's current (highest-epoch) lease
}

// NewLeaseStore returns a store rooted at cfg.RunsDir.
func NewLeaseStore(cfg StoreConfig) (*LeaseStore, error) {
	if strings.TrimSpace(cfg.RunsDir) == "" {
		return nil, errors.New("nashnet: runs dir is required")
	}
	return &LeaseStore{
		cfg:    cfg.withDefaults(),
		leases: make(map[string]*Lease),
		byJob:  make(map[string]*Lease),
	}, nil
}

// Policy returns the pool policy this store hands to nodes.
func (s *LeaseStore) Policy() Policy { return s.cfg.Policy }

// TTL is the lease renewal window.
func (s *LeaseStore) TTL() time.Duration {
	return time.Duration(s.cfg.Policy.LeaseTTLSeconds) * time.Second
}

// poolRuntimeCap is the pool's own ceiling on lease lifetime from granted_at.
func (s *LeaseStore) poolRuntimeCap() time.Duration {
	return time.Duration(s.cfg.Policy.MaxLeaseSeconds) * time.Second
}

// Now reads the injected clock.
func (s *LeaseStore) Now() time.Time { return s.cfg.Now() }

// runDir joins a job id onto the runs dir after the path guard.
func (s *LeaseStore) runDir(jobID string) (string, error) {
	if err := validJobID(jobID); err != nil {
		return "", err
	}
	return filepath.Join(s.cfg.RunsDir, jobID), nil
}

// validJobID rejects an empty id and anything carrying a path separator or a
// parent reference before it is joined onto the runs dir.
func validJobID(jobID string) error {
	if jobID == "" {
		return fmt.Errorf("%w: empty", ErrInvalidJobID)
	}
	if strings.Contains(jobID, "/") || strings.Contains(jobID, `\`) ||
		strings.Contains(jobID, "..") || strings.ContainsRune(jobID, 0) {
		return fmt.Errorf("%w: %q", ErrInvalidJobID, jobID)
	}
	return nil
}

// GrantRequest is one placement decision: the coordinator has matched a job to
// a node and is handing out the right to run it (D4).
type GrantRequest struct {
	JobID     string
	NodeID    string
	NodeEpoch int64
	// Attempt is the job's infrastructure attempt counter (D32). Zero means the
	// first attempt.
	Attempt int
	// GrantSet is the snapshot digest and the seeds the placement scan resolved
	// (D53). Every later read on the lease is fenced on membership in it.
	GrantSet GrantSet
	// MaxRuntime is spec.max_runtime_hours resolved to a duration, or zero when
	// the submitter set none. The effective cap is the smaller of it and the
	// pool's (D4).
	MaxRuntime time.Duration
}

// Grant mints a lease for one job on one node: it advances the job's lease
// epoch, mints a token, writes runs/<job>/lease.json, and returns the record
// plus the token, which is the only time the token exists in the clear (D4,
// D44). A job already holding a live lease is ErrJobLeased, including one in
// revoking, which cannot be re-claimed until it clears.
func (s *LeaseStore) Grant(req GrantRequest) (Lease, string, error) {
	if req.NodeID == "" {
		return Lease{}, "", errors.New("nashnet: grant needs a node id")
	}
	dir, err := s.runDir(req.JobID)
	if err != nil {
		return Lease{}, "", err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	prev := s.byJob[req.JobID]
	if prev != nil && prev.Live() {
		return Lease{}, "", fmt.Errorf("%w: %s holds %s", ErrJobLeased, req.JobID, prev.LeaseID)
	}
	epoch := int64(1)
	if prev != nil {
		epoch = prev.LeaseEpoch + 1
	}

	now := s.cfg.Now()
	leaseID, err := NewLeaseID(now, s.cfg.Entropy)
	if err != nil {
		return Lease{}, "", err
	}
	token, hash, err := MintLeaseToken(s.cfg.Entropy)
	if err != nil {
		return Lease{}, "", err
	}
	// The attempt is resolved here rather than by the caller, so every path
	// that grants a lease counts the same way (D32). The previous lease's
	// verdict already decided the number: NextAttempt is one higher than its
	// own attempt for a requeue and equal to it for a nack, and it is on disk,
	// so a coordinator restart between the verdict and the re-claim does not
	// restart the count.
	attempt := req.Attempt
	if prev != nil {
		if prev.NextAttempt > attempt {
			attempt = prev.NextAttempt
		}
		if prev.Attempt > attempt {
			attempt = prev.Attempt
		}
	}
	if attempt <= 0 {
		attempt = 1
	}

	l := &Lease{
		JobID:      req.JobID,
		NodeID:     req.NodeID,
		NodeEpoch:  req.NodeEpoch,
		LeaseID:    leaseID,
		LeaseEpoch: epoch,
		State:      LeaseActive,
		TokenHash:  hash,
		GrantedAt:  now,
		Deadline:   now.Add(s.TTL()),
		Attempt:    attempt,
		GrantSet:   req.GrantSet.clone(),
		Phase:      PhaseClaimed,
		MaxRuntime: req.MaxRuntime,
	}
	if err := WriteLease(dir, l); err != nil {
		return Lease{}, "", err
	}
	s.leases[leaseID] = l
	s.byJob[req.JobID] = l
	return l.Clone(), token, nil
}

// ProgressUpdate is one progress post (D5), already decoded and carrying the
// facts the lease record keeps.
type ProgressUpdate struct {
	LeaseID    string
	LeaseEpoch int64
	// NodeEpoch is the registry's current epoch for the lease's node, supplied by
	// the coordinator rather than by the node: a lease route carries the lease
	// token and nothing else (D26). SkipEpoch means the caller holds no registry
	// epoch to compare, which is the state a coordinator restart leaves until the
	// node re-registers (D34).
	NodeEpoch int64
	Token     string
	// Phase must be one a node may report; stopping is refused with
	// ErrInvalidPhase (D5). Empty leaves the recorded phase untouched.
	Phase string
	// PIDProjected records that the coordinator projected a pid for this lease.
	// It only ever latches true: it is the second witness of the D7 table and a
	// job that once ran is never requeued.
	PIDProjected   bool
	ManifestSeq    int64
	ManifestDigest string
}

// Renew fences a progress post and renews the lease (D5). A revoking lease
// admits the post and records its facts but is not renewed and its deadline is
// not extended, so the grace period is a hard window; the caller reads the
// returned State and answers revoke (D4).
func (s *LeaseStore) Renew(u ProgressUpdate) (Lease, error) {
	if u.Phase != "" {
		if err := ValidateNodePhase(u.Phase); err != nil {
			return Lease{}, err
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	l, err := s.fenceLocked(u.LeaseID, u.LeaseEpoch, u.NodeEpoch, u.Token)
	if err != nil {
		return Lease{}, err
	}
	if !l.PermitsRoute(RouteProgress) {
		return Lease{}, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}

	l.advancePhase(u.Phase)
	if u.PIDProjected {
		l.PIDProjected = true
	}
	if u.ManifestSeq > l.ManifestSeq {
		l.ManifestSeq = u.ManifestSeq
		l.ManifestDigest = u.ManifestDigest
	}
	if l.State == LeaseActive {
		l.Deadline = s.cfg.Now().Add(s.TTL())
	}
	if err := s.persistLocked(l); err != nil {
		return Lease{}, err
	}
	return l.Clone(), nil
}

// Fence authorizes a lease route that carries an epoch, without renewing (the
// manifest, result, and nack routes). Route is checked against the lease state,
// so a revoking lease refuses everything but the routes D4 lists.
func (s *LeaseStore) Fence(leaseID string, leaseEpoch, nodeEpoch int64, token string, route Route) (Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, err := s.fenceLocked(leaseID, leaseEpoch, nodeEpoch, token)
	if err != nil {
		return Lease{}, err
	}
	if !l.PermitsRoute(route) {
		return Lease{}, fmt.Errorf("%w: state %s refuses %s", ErrLeaseSuperseded, l.State, route)
	}
	return l.Clone(), nil
}

// Authorize is Fence for the lease routes that carry no epoch in their body
// (logs, blob chunks, snapshot and seed reads): the token alone authorizes, and
// the lease state still gates the route.
func (s *LeaseStore) Authorize(leaseID, token string, route Route) (Lease, error) {
	return s.Fence(leaseID, SkipEpoch, SkipEpoch, token, route)
}

// fenceLocked resolves a lease and applies the (lease_id, lease_epoch,
// node_epoch) fence plus the constant-time token compare (D4, D44). SkipEpoch
// on either half omits that comparison; every other value is compared exactly,
// so a body carrying no epoch at all fences out rather than through.
// Callers hold s.mu.
func (s *LeaseStore) fenceLocked(leaseID string, leaseEpoch, nodeEpoch int64, token string) (*Lease, error) {
	l, ok := s.leases[leaseID]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownLease, leaseID)
	}
	if l.TokenHash == "" {
		return nil, fmt.Errorf("%w: %s", ErrLeaseTokenDropped, leaseID)
	}
	if !TokenMatches(l.TokenHash, token) {
		return nil, fmt.Errorf("%w: token mismatch on %s", ErrLeaseSuperseded, leaseID)
	}
	if !l.Live() {
		return nil, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}
	if leaseEpoch != SkipEpoch && leaseEpoch != l.LeaseEpoch {
		return nil, fmt.Errorf("%w: lease epoch %d, have %d", ErrLeaseSuperseded, leaseEpoch, l.LeaseEpoch)
	}
	if nodeEpoch != SkipEpoch && nodeEpoch != l.NodeEpoch {
		return nil, fmt.Errorf("%w: node epoch %d, have %d", ErrLeaseSuperseded, nodeEpoch, l.NodeEpoch)
	}
	return l, nil
}

// BeginRevoking moves a live lease into the wind-down state of D4: the token
// stays valid and the epoch does not advance for one grace period, during which
// the node may append logs and blobs, commit a final manifest, and post
// canceled or preempted. operatorStop records stop_requested_at, the only
// witness of an operator stop and the key of the third D7 row; a gate-driven
// stop and the runtime cap pass false, so neither is recorded as a cancel.
//
// The grace window is carried on Deadline rather than in a field of its own, so
// a revoking lease that goes quiet expires through the same sweep as any other
// and a restored one keeps exactly its remaining grace (D34).
func (s *LeaseStore) BeginRevoking(leaseID string, operatorStop bool) (Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, ok := s.leases[leaseID]
	if !ok {
		return Lease{}, fmt.Errorf("%w: %s", ErrUnknownLease, leaseID)
	}
	if !l.Live() {
		return Lease{}, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}
	now := s.cfg.Now()
	if l.State != LeaseRevoking {
		l.State = LeaseRevoking
		l.Deadline = now.Add(s.cfg.RevokeGrace)
	}
	if operatorStop && l.StopRequestedAt.IsZero() {
		l.StopRequestedAt = now
		l.advancePhase(PhaseStopping)
	}
	if err := s.persistLocked(l); err != nil {
		return Lease{}, err
	}
	return l.Clone(), nil
}

// Outcome is what the coordinator must do about a lease the store just moved.
// A revoking outcome asks for a revoke event on the node's events poll; a
// released one carries the D7 verdict for the phase the lease reached.
type Outcome struct {
	JobID      string
	LeaseID    string
	NodeID     string
	LeaseEpoch int64
	// State is the lease state after the action: revoking or released.
	State string
	// Verdict is empty while revoking and carries the D7 verdict once released.
	Verdict Verdict
	// NextAttempt is the attempt the job requeues at, and equals the lease's own
	// attempt for every verdict but requeue (D32).
	NextAttempt int
	Reason      string
}

// Revoke ends a lease now: it advances the lease epoch, drops the token, marks
// the record released, and returns the D7 verdict. The record is retained,
// because a lease whose pid was ever projected is settled by the two-witness
// rule against promoted state and the verdict table must still apply to it
// (D3, D35). Revocation takes no grace period; a stop of a node in good
// standing goes through BeginRevoking first (D60).
func (s *LeaseStore) Revoke(leaseID, reason string) (Outcome, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.revokeLocked(leaseID, reason)
}

func (s *LeaseStore) revokeLocked(leaseID, reason string) (Outcome, error) {
	l, ok := s.leases[leaseID]
	if !ok {
		return Outcome{}, fmt.Errorf("%w: %s", ErrUnknownLease, leaseID)
	}
	if !l.Live() {
		return Outcome{}, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}
	l.LeaseEpoch++
	l.TokenHash = ""
	l.State = LeaseReleased
	o := s.outcomeLocked(l, reason)
	l.NextAttempt = o.NextAttempt
	if err := s.persistLocked(l); err != nil {
		return Outcome{}, err
	}
	return o, nil
}

// Return ends a lease its node handed back (D8): the job returns to the ready
// set at its original position with no attempt increment, since a wrong
// placement is a scheduling event rather than a job failure. The lease record
// is retained like any other released one.
//
// A lease that already reached a launched phase is revoked instead, and settles
// by the D7 verdict for that phase. A nack is a pre-launch act by construction,
// so this only bites a node that returns a claim after starting the process,
// and re-running a job that ran is the one thing the retry rules never do
// (D33).
func (s *LeaseStore) Return(leaseID, reason string) (Outcome, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, ok := s.leases[leaseID]
	if !ok {
		return Outcome{}, fmt.Errorf("%w: %s", ErrUnknownLease, leaseID)
	}
	if !l.Live() {
		return Outcome{}, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}
	if l.PIDProjected || PhaseLaunched(l.Phase) || !l.StopRequestedAt.IsZero() {
		return s.revokeLocked(leaseID, reason)
	}
	l.LeaseEpoch++
	l.TokenHash = ""
	l.State = LeaseReleased
	l.NextAttempt = l.Attempt
	if err := s.persistLocked(l); err != nil {
		return Outcome{}, err
	}
	return Outcome{
		JobID:       l.JobID,
		LeaseID:     l.LeaseID,
		NodeID:      l.NodeID,
		LeaseEpoch:  l.LeaseEpoch,
		State:       l.State,
		Verdict:     VerdictRequeue,
		NextAttempt: l.Attempt,
		Reason:      reason,
	}, nil
}

// outcomeLocked renders a released lease as an Outcome. Callers hold s.mu.
func (s *LeaseStore) outcomeLocked(l *Lease, reason string) Outcome {
	v := l.Verdict()
	next := l.Attempt
	if v == VerdictRequeue {
		next = l.Attempt + 1
	}
	return Outcome{
		JobID:       l.JobID,
		LeaseID:     l.LeaseID,
		NodeID:      l.NodeID,
		LeaseEpoch:  l.LeaseEpoch,
		State:       l.State,
		Verdict:     v,
		NextAttempt: next,
		Reason:      reason,
	}
}

// Release is the result commit point (D6): the node posted a terminal the
// coordinator accepted, so the lease is released and its token dropped. The
// epoch does not advance, so a replayed result on the same (lease, epoch) is
// still recognizable as the recorded one. The caller fences first.
func (s *LeaseStore) Release(leaseID string) (Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	l, ok := s.leases[leaseID]
	if !ok {
		return Lease{}, fmt.Errorf("%w: %s", ErrUnknownLease, leaseID)
	}
	if !l.Live() {
		return Lease{}, fmt.Errorf("%w: state %s", ErrLeaseSuperseded, l.State)
	}
	l.State = LeaseReleased
	l.TokenHash = ""
	if err := s.persistLocked(l); err != nil {
		return Lease{}, err
	}
	return l.Clone(), nil
}

// ReBindResult is the outcome of a register call's live_leases list (D3).
type ReBindResult struct {
	// Rebound names the leases the node kept: its own, with a matching token
	// hash, now stamped with the new node epoch.
	Rebound []string
	// Refused names entries the node listed that re-bound nothing: an unknown
	// lease, a lease whose record belongs to another node, or a hash mismatch.
	// A refused entry never touches the record it named, so listing another
	// node's lease id is inert rather than an attack.
	Refused []string
	// Revoked carries every other live lease the node held, with its D7 verdict:
	// a restarted agent cannot be supervising a job it never reattached to.
	Revoked []Outcome
}

// ReBind applies the live_leases half of a registration (D3). It re-binds only
// the leases whose record names this node and whose token hash compares equal
// under crypto/subtle.ConstantTimeCompare, so a node-supplied hash alone
// authorizes nothing, and revokes every other live lease the node held. A
// revoked lease that never projected a pid requeues with attempt++; one whose
// pid was ever projected goes to the finalizer instead, which is the case a
// node whose host rebooted inside the lease TTL produces (D32).
func (s *LeaseStore) ReBind(nodeID string, nodeEpoch int64, live []LiveLease) (ReBindResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var res ReBindResult
	rebound := make(map[string]bool, len(live))
	for _, entry := range live {
		l, ok := s.leases[entry.LeaseID]
		if !ok || l.NodeID != nodeID || !l.Live() || !HashMatches(l.TokenHash, entry.TokenHash) {
			res.Refused = append(res.Refused, entry.LeaseID)
			continue
		}
		l.NodeEpoch = nodeEpoch
		if err := s.persistLocked(l); err != nil {
			return res, err
		}
		rebound[l.LeaseID] = true
		res.Rebound = append(res.Rebound, l.LeaseID)
	}

	for _, id := range s.liveLeaseIDsLocked(nodeID) {
		if rebound[id] {
			continue
		}
		out, err := s.revokeLocked(id, ReasonNodeReRegister)
		if err != nil {
			return res, err
		}
		res.Revoked = append(res.Revoked, out)
	}
	return res, nil
}

// RevokeNode ends every live lease a node holds, with no grace period: the
// operator has declared the node untrusted, so it gets no window in which to
// write, and each of its jobs is settled by the returned verdict against
// whatever was already promoted (D60).
func (s *LeaseStore) RevokeNode(nodeID string) ([]Outcome, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var out []Outcome
	for _, id := range s.liveLeaseIDsLocked(nodeID) {
		o, err := s.revokeLocked(id, ReasonNodeRevoked)
		if err != nil {
			return out, err
		}
		out = append(out, o)
	}
	return out, nil
}

// Sweep is one expiry pass (D7), run by the Sweeper every TTL/4 under the
// injected clock. An active lease past the runtime cap moves to revoking so the
// node gets its wind-down window; one past its renewal deadline, and a revoking
// one past its grace, are revoked and carry the verdict for the phase reached.
func (s *LeaseStore) Sweep() ([]Outcome, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.cfg.Now()
	var out []Outcome
	for _, id := range s.liveLeaseIDsLocked("") {
		l := s.leases[id]
		switch {
		case l.Expired(now):
			// Expiry outranks the runtime cap: a lease nobody is renewing gets no
			// wind-down window, since the grace exists for a node that is still
			// answering.
			reason := ReasonExpired
			if l.State == LeaseRevoking {
				reason = ReasonGraceElapsed
			}
			o, err := s.revokeLocked(id, reason)
			if err != nil {
				return out, err
			}
			out = append(out, o)
		case l.State == LeaseActive && l.RuntimeCapReached(now, s.poolRuntimeCap()):
			l.State = LeaseRevoking
			l.Deadline = now.Add(s.cfg.RevokeGrace)
			if err := s.persistLocked(l); err != nil {
				return out, err
			}
			out = append(out, Outcome{
				JobID: l.JobID, LeaseID: l.LeaseID, NodeID: l.NodeID,
				LeaseEpoch: l.LeaseEpoch, State: l.State, NextAttempt: l.Attempt,
				Reason: ReasonRuntimeCap,
			})
		}
	}
	return out, nil
}

// liveLeaseIDsLocked returns the sorted ids of the live leases, filtered to one
// node when nodeID is non-empty. Sorting keeps a sweep pass deterministic.
// Callers hold s.mu.
func (s *LeaseStore) liveLeaseIDsLocked(nodeID string) []string {
	var ids []string
	for id, l := range s.leases {
		if !l.Live() {
			continue
		}
		if nodeID != "" && l.NodeID != nodeID {
			continue
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Get returns a lease by id.
func (s *LeaseStore) Get(leaseID string) (Lease, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.leases[leaseID]
	if !ok {
		return Lease{}, false
	}
	return l.Clone(), true
}

// ByJob returns a job's current lease, live or not.
func (s *LeaseStore) ByJob(jobID string) (Lease, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.byJob[jobID]
	if !ok {
		return Lease{}, false
	}
	return l.Clone(), true
}

// LiveForNode returns the node's live leases, ordered by lease id.
func (s *LeaseStore) LiveForNode(nodeID string) []Lease {
	s.mu.Lock()
	defer s.mu.Unlock()
	ids := s.liveLeaseIDsLocked(nodeID)
	out := make([]Lease, 0, len(ids))
	for _, id := range ids {
		out = append(out, s.leases[id].Clone())
	}
	return out
}

// LiveCountForNode is the input to the per-node lease ceiling of D47, which the
// dispatcher applies as a placement fact.
func (s *LeaseStore) LiveCountForNode(nodeID string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.liveLeaseIDsLocked(nodeID))
}

// NodeEpochFloor returns the highest node epoch any retained lease records for
// a node, or zero. A restarted coordinator seeds the registry with it so the
// next registration cannot hand back an epoch a restored lease already carries
// (D34).
func (s *LeaseStore) NodeEpochFloor(nodeID string) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	var floor int64
	for _, l := range s.leases {
		if l.NodeID == nodeID && l.NodeEpoch > floor {
			floor = l.NodeEpoch
		}
	}
	return floor
}

// Restore rebuilds the store from runs/*/lease.json after a coordinator restart
// (D34): a lease inside its deadline comes back with its token hash, state,
// stop_requested_at, granted_at, sequence, digest, and grant set, and the node
// keeps posting progress under the same lease epoch. A lapsed lease is restored
// too and goes to the sweeper on its next pass rather than being dropped, since
// its verdict is the only thing that settles its job. Unreadable records are
// skipped and counted; the error is non-nil only when the runs dir cannot be
// listed.
func (s *LeaseStore) Restore() (restored, skipped int, err error) {
	entries, err := os.ReadDir(s.cfg.RunsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, 0, nil
		}
		return 0, 0, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		l, err := ReadLease(filepath.Join(s.cfg.RunsDir, e.Name()))
		if err != nil {
			if !os.IsNotExist(err) {
				skipped++
			}
			continue
		}
		if l.LeaseID == "" || l.JobID != e.Name() {
			skipped++
			continue
		}
		s.leases[l.LeaseID] = l
		if prev, ok := s.byJob[l.JobID]; !ok || l.LeaseEpoch >= prev.LeaseEpoch {
			s.byJob[l.JobID] = l
		}
		restored++
	}
	return restored, skipped, nil
}

// persistLocked rewrites the lease record. Callers hold s.mu.
func (s *LeaseStore) persistLocked(l *Lease) error {
	dir, err := s.runDir(l.JobID)
	if err != nil {
		return err
	}
	return WriteLease(dir, l)
}
