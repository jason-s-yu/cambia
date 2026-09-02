package harness

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// BundleBuilder is the coordinator's code-delivery seam (D48): it returns the
// git bundle descriptor for a job's pinned commit, negating the basis commits
// the claiming node already holds. ingest.Manager satisfies it. It is an
// interface rather than a concrete manager so the route suite drives the
// snapshot path with a fake and asserts that no git subprocess runs while the
// placement lock is held (D48).
type BundleBuilder interface {
	BundleCreate(ctx context.Context, jobID string, basis []string) (ingest.BundleDescriptor, error)
}

// Ceilings are the coordinator-side quotas and connection bounds of D56 that
// live in the route layer rather than in the quarantine store, because each is
// a per-identity fact the store does not hold. Every zero value takes the
// documented default.
type Ceilings struct {
	// RequestsPerMinute is the per-node token bucket over all node and lease
	// routes.
	RequestsPerMinute int
	// ConnectionsPerNode caps concurrent in-flight requests per identity.
	ConnectionsPerNode int
	// ConcurrentUploads caps concurrent blob chunk appends per node.
	ConcurrentUploads int
	// ConcurrentDownloads caps concurrent snapshot and seed reads per node.
	ConcurrentDownloads int
	// BundleBuildsPerMinute caps forced bundle builds per node; over budget the
	// claim is served the cached full-tree bundle instead of forking git.
	BundleBuildsPerMinute int
	// MaxClaimWaiters caps held claim long polls across the pool
	// (RUNNERD_NASHNET_MAX_CLAIM_WAITERS); past it a claim answers
	// 204 waiters_full at once.
	MaxClaimWaiters int
	// ChunkBytes is the per-request blob body cap (MaxBytesReader).
	ChunkBytes int64
	// LeaseBytesPerTick caps the bytes one lease may upload between two
	// progress posts.
	LeaseBytesPerTick int64
	// LogBytesPerCall and LogBytesPerJob bound the log append route (D54).
	LogBytesPerCall int64
	LogBytesPerJob  int64
	// NodeMBPS is RUNNERD_NASHNET_NODE_MBPS: the per-node byte-rate token
	// bucket over the ingress routes (blob chunks, log appends) and the egress
	// ones (snapshot and seed reads), which share one bucket because they share
	// one link. Zero is unlimited, the documented default on a LAN.
	NodeMBPS int
	// GlobalInFlight and GlobalConnections are the pre-authentication caps: a
	// stranger that reaches the listener holds no more than this many requests
	// and connections across the daemon (D56, ruling q3).
	GlobalInFlight    int
	GlobalConnections int
	// SourceRequestsPerMinute is the pre-authentication per-source-address token
	// bucket, and ConcurrentVerifications ceilings concurrent token
	// verifications so an unauthenticated peer cannot force unbounded EdDSA
	// work.
	SourceRequestsPerMinute int
	ConcurrentVerifications int
}

// Ceiling defaults, the bounds table of D56.
const (
	DefaultRequestsPerMinute     = 600
	DefaultConnectionsPerNode    = 8
	DefaultConcurrentUploads     = 4
	DefaultConcurrentDownloads   = 2
	DefaultBundleBuildsPerMinute = 4
	DefaultMaxClaimWaiters       = 64
	DefaultChunkBytes            = 64 << 20
	DefaultLeaseBytesPerTick     = 2 << 30
	DefaultLogBytesPerCall       = 2 << 20
	DefaultLogBytesPerJob        = 1 << 30
	// DefaultNodeMBPS is 0: unlimited, the LAN default of D56. An operator
	// meters a node over a slow link by setting RUNNERD_NASHNET_NODE_MBPS.
	DefaultNodeMBPS                = 0
	DefaultGlobalInFlight          = 256
	DefaultGlobalConnections       = 512
	DefaultSourceRequestsPerMinute = 1200
	DefaultConcurrentVerifications = 32
	// DefaultUnplaceableGrace is RUNNERD_NASHNET_UNPLACEABLE_GRACE (D14): how
	// long a ready job may match no node before its hold is rendered.
	DefaultUnplaceableGrace = 600 * time.Second
	// DefaultNackCooldown is the D8 default cooldown excluding a node from
	// re-matching the job it returned.
	DefaultNackCooldown = 300 * time.Second
	// DefaultDegradedJobHold is RUNNERD_NASHNET_DEGRADED_JOB_HOLD (D8): how
	// long a per-job degraded mark stands before its own timer clears it.
	DefaultDegradedJobHold = time.Hour
	// DefaultMaxAttempts is the infrastructure attempt budget of D32, used for
	// a job whose spec sets no max_attempts of its own.
	DefaultMaxAttempts = 3
	// resultRetention is how long a posted terminal stays replayable (D6). It
	// covers the whole window in which a node retries a lost result response,
	// which D36 bounds at 2 x lease TTL of lost contact, with room to spare.
	resultRetention = time.Hour
	// maxResultRecords is the hard ceiling on retained terminals, so a
	// coordinator that settles more leases per hour than this drops the oldest
	// replay record rather than growing without bound.
	maxResultRecords = 4096
	// maxClaimWaitSeconds is the coordinator's cap on a node-supplied
	// wait_seconds (D2), shared by the claim and the events long poll.
	maxClaimWaitSeconds = 30
)

func (c Ceilings) withDefaults() Ceilings {
	if c.RequestsPerMinute <= 0 {
		c.RequestsPerMinute = DefaultRequestsPerMinute
	}
	if c.ConnectionsPerNode <= 0 {
		c.ConnectionsPerNode = DefaultConnectionsPerNode
	}
	if c.ConcurrentUploads <= 0 {
		c.ConcurrentUploads = DefaultConcurrentUploads
	}
	if c.ConcurrentDownloads <= 0 {
		c.ConcurrentDownloads = DefaultConcurrentDownloads
	}
	if c.BundleBuildsPerMinute <= 0 {
		c.BundleBuildsPerMinute = DefaultBundleBuildsPerMinute
	}
	if c.MaxClaimWaiters <= 0 {
		c.MaxClaimWaiters = DefaultMaxClaimWaiters
	}
	if c.ChunkBytes <= 0 {
		c.ChunkBytes = DefaultChunkBytes
	}
	if c.LeaseBytesPerTick <= 0 {
		c.LeaseBytesPerTick = DefaultLeaseBytesPerTick
	}
	if c.LogBytesPerCall <= 0 {
		c.LogBytesPerCall = DefaultLogBytesPerCall
	}
	if c.LogBytesPerJob <= 0 {
		c.LogBytesPerJob = DefaultLogBytesPerJob
	}
	if c.GlobalInFlight <= 0 {
		c.GlobalInFlight = DefaultGlobalInFlight
	}
	if c.GlobalConnections <= 0 {
		c.GlobalConnections = DefaultGlobalConnections
	}
	if c.SourceRequestsPerMinute <= 0 {
		c.SourceRequestsPerMinute = DefaultSourceRequestsPerMinute
	}
	if c.ConcurrentVerifications <= 0 {
		c.ConcurrentVerifications = DefaultConcurrentVerifications
	}
	return c
}

// PoolConfig configures the coordinator half of the nashnet compute pool. The
// four stores are required; every other field has a documented default, so a
// daemon wires the pool with a nodes directory and a runs directory.
type PoolConfig struct {
	Dispatcher *Dispatcher
	Grants     *authtoken.GrantStore
	Leases     *nashnet.LeaseStore
	Registry   *nashnet.NodeRegistry
	Quarantine *quarantine.Store
	// Bundles serves the claim's snapshot (D48). Nil leaves the snapshot route
	// answering 503, which is the state of a daemon with no ingest wired.
	Bundles BundleBuilder
	RunsDir string
	// NodesDir is RUNNERD_NASHNET_NODES_DIR: the operator-owned directory
	// holding <node_id>.grant and the <node_id>.revoked tombstones the revoke
	// route writes (D60).
	NodesDir string
	// OriginHost is the coordinator's own name, written into every pool run's
	// env.json as origin_host (D20, D23). Empty resolves the hostname.
	OriginHost string
	Policy     nashnet.Policy
	Ceilings   Ceilings
	// MaxLeasesPerNode is RUNNERD_NASHNET_MAX_LEASES_PER_NODE, the coordinator's
	// own ceiling on concurrent leases per node (D47).
	MaxLeasesPerNode int
	// MaxAttempts is RUNNERD_NASHNET_MAX_ATTEMPTS, the pool-wide infrastructure
	// attempt budget a job with no max_attempts of its own runs under (D32).
	// Zero means DefaultMaxAttempts.
	MaxAttempts int
	// UnplaceableGrace is how long a ready job matches no node before its hold
	// is rendered in the view (D14).
	UnplaceableGrace time.Duration
	// NodeTTL and SessionGrace drive the presence rendering of D3 and D45.
	NodeTTL      time.Duration
	SessionGrace time.Duration
	// Now is the injected clock. Nil means time.Now.
	Now func() time.Time
}

// Pool is the coordinator's nashnet state: the node sessions of D45, the claim
// waiters of D2, the per-identity ceilings of D56, and the wiring between the
// lease store, the node registry, the quarantine store, and the dispatcher's
// placement scan. One Pool serves every /nashnet/ route.
type Pool struct {
	disp    *Dispatcher
	grants  *authtoken.GrantStore
	leases  *nashnet.LeaseStore
	nodes   *nashnet.NodeRegistry
	quar    *quarantine.Store
	bundles BundleBuilder

	runsDir    string
	nodesDir   string
	originHost string
	policy     nashnet.Policy
	ceilings   Ceilings
	maxLeases  int
	maxAttempt int
	nodeTTL    time.Duration
	sessGrace  time.Duration
	now        func() time.Time

	mu sync.Mutex
	// sessions holds the one events request a node may have open (D45); a
	// second one supersedes the first.
	sessions map[string]*nodeSession
	// pending queues the events a node has not yet collected.
	pending map[string][]nashnet.Event
	// waiters are the held claim long polls, capped by Ceilings.MaxClaimWaiters.
	waiters  map[*claimWaiter]struct{}
	requests *rateTable
	conns    *countTable
	uploads  *countTable
	egress   *countTable
	builds   *rateTable
	// bytes is the per-node byte-rate bucket shared by the upload and the
	// egress routes (D56).
	bytes *byteTable
	// snapshots caches the bundle descriptor a claim resolved, keyed by lease
	// id, so the snapshot route serves the exact artifact the claim reported.
	snapshots map[string]snapshotDescriptor
	// quarGrants is the lease's grant set in the quarantine store's shape,
	// keyed by digest: a seeded file the node reports unchanged is copied from
	// the recorded source path rather than uploaded (D50, D51).
	quarGrants map[string]map[string]quarantine.Grant
	// tickBytes is the per-lease upload budget since its last progress post.
	tickBytes map[string]int64
	// stopForce records whether the stop a lease is winding down under was a
	// forced one, so the progress response of D5 carries the same force flag
	// the revoke event does (D31).
	stopForce map[string]bool
	// logDropped is the coordinator-owned dropped-byte counter of D54; the
	// in-band marker is advisory text and never the evidence.
	logDropped map[string]int64
	// results records a posted terminal so a replay after a lost response
	// returns the recorded body rather than a conflict (D6). resultAuth keeps
	// the hash of the token that posted it, because the lease store drops the
	// token at release and the replay must still prove it holds it.
	// resultAt is when each was recorded, which is what bounds the three maps:
	// they are per lease and a coordinator serves leases for its whole life, so
	// without a retention window they are an unbounded leak.
	results    map[string]nashnet.ResultResponse
	resultAuth map[string]string
	resultAt   map[string]time.Time
	// cooldowns exclude a node from re-matching a job it nacked (D8).
	cooldowns map[string]map[string]time.Time
	// nacks counts a node's nacks per job; three mark the job degraded for it.
	nacks map[string]map[string]int
	// breaker is the per-node circuit breaker of D63: the consecutive
	// prepare_node_failed counter, the trip count the cooldown ladder doubles
	// on, and the standing hold.
	breaker map[string]*breakerState
	// degraded marks a (node, job) pair held after three nacks (D8).
	degraded map[string]map[string]time.Time

	preauth *preAuthGate
}

// nodeSession is one node's held events request (D45).
type nodeSession struct {
	ready      chan struct{}
	superseded chan struct{}
	until      time.Time
}

// claimWaiter is one held claim long poll (D2).
type claimWaiter struct {
	wake chan struct{}
}

// NewPool builds the coordinator pool. It returns an error when a required
// store is missing: a daemon must not serve node routes without the grant store
// that authenticates them.
func NewPool(cfg PoolConfig) (*Pool, error) {
	if cfg.Dispatcher == nil {
		return nil, errors.New("nashnet pool: nil dispatcher")
	}
	if cfg.Grants == nil {
		return nil, errors.New("nashnet pool: nil grant store (node routes cannot authenticate)")
	}
	if cfg.Leases == nil {
		return nil, errors.New("nashnet pool: nil lease store")
	}
	if cfg.Registry == nil {
		return nil, errors.New("nashnet pool: nil node registry")
	}
	if cfg.Quarantine == nil {
		return nil, errors.New("nashnet pool: nil quarantine store")
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	policy := cfg.Policy
	if policy.LeaseTTLSeconds <= 0 {
		policy = nashnet.DefaultPolicy()
	}
	origin := cfg.OriginHost
	if origin == "" {
		if h, err := os.Hostname(); err == nil {
			origin = h
		}
	}
	maxLeases := cfg.MaxLeasesPerNode
	if maxLeases <= 0 {
		maxLeases = nashnet.DefaultMaxLeasesPerNode
	}
	maxAttempt := cfg.MaxAttempts
	if maxAttempt <= 0 {
		maxAttempt = DefaultMaxAttempts
	}
	nodeTTL := cfg.NodeTTL
	if nodeTTL <= 0 {
		nodeTTL = nashnet.DefaultNodeTTLSeconds * time.Second
	}
	sessGrace := cfg.SessionGrace
	if sessGrace <= 0 {
		sessGrace = nashnet.DefaultSessionGraceSeconds * time.Second
	}
	ceil := cfg.Ceilings.withDefaults()

	p := &Pool{
		disp:       cfg.Dispatcher,
		grants:     cfg.Grants,
		leases:     cfg.Leases,
		nodes:      cfg.Registry,
		quar:       cfg.Quarantine,
		bundles:    cfg.Bundles,
		runsDir:    cfg.RunsDir,
		nodesDir:   cfg.NodesDir,
		originHost: origin,
		policy:     policy,
		ceilings:   ceil,
		maxLeases:  maxLeases,
		maxAttempt: maxAttempt,
		nodeTTL:    nodeTTL,
		sessGrace:  sessGrace,
		now:        now,
		sessions:   map[string]*nodeSession{},
		pending:    map[string][]nashnet.Event{},
		waiters:    map[*claimWaiter]struct{}{},
		requests:   newRateTable(ceil.RequestsPerMinute, time.Minute, now),
		conns:      newCountTable(ceil.ConnectionsPerNode),
		uploads:    newCountTable(ceil.ConcurrentUploads),
		egress:     newCountTable(ceil.ConcurrentDownloads),
		builds:     newRateTable(ceil.BundleBuildsPerMinute, time.Minute, now),
		bytes:      newByteTable(ceil.NodeMBPS, now),
		snapshots:  map[string]snapshotDescriptor{},
		quarGrants: map[string]map[string]quarantine.Grant{},
		tickBytes:  map[string]int64{},
		stopForce:  map[string]bool{},
		logDropped: map[string]int64{},
		results:    map[string]nashnet.ResultResponse{},
		resultAuth: map[string]string{},
		resultAt:   map[string]time.Time{},
		cooldowns:  map[string]map[string]time.Time{},
		nacks:      map[string]map[string]int{},
		breaker:    map[string]*breakerState{},
		degraded:   map[string]map[string]time.Time{},
		preauth:    newPreAuthGate(ceil, now),
	}
	grace := cfg.UnplaceableGrace
	if grace <= 0 {
		grace = DefaultUnplaceableGrace
	}
	cfg.Dispatcher.attachPool(p, cfg.Leases, grace, now)
	return p, nil
}

// Sweeper returns the lease expiry sweeper wired to this pool's outcome
// handling (D7): a revoking outcome raises a revoke event on the node's events
// poll, a released one applies the verdict for the phase the lease reached.
func (p *Pool) Sweeper() *nashnet.Sweeper {
	return nashnet.NewSweeper(p.leases, p.applyOutcome, func(err error) {
		poolLog("nashnet sweep: %v", err)
	})
}

// applyOutcome is what the coordinator does about a lease the store moved (D7,
// D32). A revoking lease is told to stop; a released one is requeued,
// finalized, or recorded canceled. It never fails a job over a quota or a
// scheduling event.
func (p *Pool) applyOutcome(o nashnet.Outcome) {
	switch o.State {
	case nashnet.LeaseRevoking:
		p.postEvent(o.NodeID, nashnet.Event{Type: nashnet.EventRevoke, LeaseID: o.LeaseID})
		return
	case nashnet.LeaseReleased:
	default:
		return
	}
	p.releaseLeaseState(o.LeaseID)
	switch o.Verdict {
	case nashnet.VerdictRequeue:
		p.requeue(o)
	case nashnet.VerdictCanceled:
		p.disp.recordPoolTerminal(o.JobID, StateCanceled, "lease stopped by the coordinator")
	case nashnet.VerdictFinalize:
		p.disp.finalizePoolJob(o.JobID)
	}
	p.signalPlacement()
}

// requeue returns a job to the ready set, or fails it when its attempt budget
// is spent (the last row of D32). The job never left the queue: a leased job
// stays at its original submit_seq and the placement scan skips it while a
// lease stands, so returning it is one re-dispatch and no queue surgery.
//
// A requeue is refused outright for a job that promoted a checkpoint under the
// lease that just ended (D33): the retry rules restart only jobs that never
// launched or that a gate stopped before any checkpoint existed, and anything
// else waits for an explicit operator resume. The requeue verdict is
// pre-launch by construction, so this is the assertion of that invariant rather
// than a path the sweeper reaches on a healthy pool.
//
// The test is per lease rather than per job, because an operator resume runs
// precisely because a checkpoint from an earlier run exists: reading the job's
// whole promoted state here would make a nack of a resumed job unrecoverable
// without a second operator act.
func (p *Pool) requeue(o nashnet.Outcome) {
	if p.promotedUnderLease(o.LeaseID, o.JobID) {
		p.disp.recordPoolTerminal(o.JobID, StatePreempted,
			"a promoted checkpoint is never auto-resumed: "+o.Reason)
		return
	}
	if max := p.maxAttempts(o.JobID); o.NextAttempt > max {
		p.disp.recordPoolTerminal(o.JobID, StateFailed,
			"attempts exhausted after "+itoa(max)+": "+o.Reason)
		return
	}
	p.projectReady(o.JobID)
	p.disp.reDispatch()
}

// maxAttempts is the job's own infrastructure attempt budget, falling back on
// the pool's (D32). It reads the persisted spec rather than the queue handle,
// so the budget survives the coordinator restart of D34.
func (p *Pool) maxAttempts(jobID string) int {
	if spec := readJobSpec(filepath.Join(p.runsDir, jobID)); spec != nil && spec.MaxAttempts > 0 {
		return spec.MaxAttempts
	}
	return p.maxAttempt
}

// promotedCheckpoint reports whether the job's folded manifest holds a
// checkpoint the coordinator has already materialized (D33, D62). It reads
// promoted state only, which the coordinator validated before materializing
// (D55), and the head is on disk, so the answer survives a restart.
//
// This is the whole-job question, which is the one D62 asks of a gate stop: a
// run resumed from an earlier checkpoint has partial state whether or not this
// lease added to it, so its gate stop is terminal and waits for an operator.
func (p *Pool) promotedCheckpoint(jobID string) bool {
	head, ok := p.manifestHead(jobID)
	return ok && headHasCheckpoint(head)
}

// promotedUnderLease narrows that question to one lease: whether the lease
// being settled is the one whose commit put the checkpoint in the head. It is
// what the requeue path asks, so that returning a job to ready turns on what
// this placement produced rather than on what the job already carried.
func (p *Pool) promotedUnderLease(leaseID, jobID string) bool {
	head, ok := p.manifestHead(jobID)
	return ok && head.Folded.LeaseID == leaseID && headHasCheckpoint(head)
}

// headHasCheckpoint reports whether a folded manifest names resumable state.
func headHasCheckpoint(head quarantine.Head) bool {
	for _, e := range head.Folded.Entries {
		if isCheckpointPath(e.Path) {
			return true
		}
	}
	return false
}

// isCheckpointPath reports whether a promoted manifest path is state a run
// could resume from: the snapshots directory it writes its rolling checkpoint
// into, or the resume marker beside it.
//
// It is deliberately broader than hasPromotedResumableState, which asks for
// both named files before an operator may resume. This one guards the opposite
// decision, whether the daemon may re-place the job on its own, so any promoted
// snapshot is enough to refuse: a wrongly held job waits for an operator, and a
// wrongly re-placed one runs twice over its own partial state.
func isCheckpointPath(rel string) bool {
	return rel == resumeStatePath || strings.HasPrefix(rel, path.Dir(resumeCheckpointPath)+"/")
}

// releaseLeaseState drops the per-lease route state a released lease no longer
// needs. The lease record itself is retained by the store for the D35 verdict.
func (p *Pool) releaseLeaseState(leaseID string) {
	p.mu.Lock()
	delete(p.snapshots, leaseID)
	delete(p.quarGrants, leaseID)
	delete(p.tickBytes, leaseID)
	delete(p.stopForce, leaseID)
	p.mu.Unlock()
}

// postEvent queues an event for a node and wakes its held events request, so a
// revoke, a drain, or a policy change reaches the node within one round trip
// instead of at its next progress tick (D45).
func (p *Pool) postEvent(nodeID string, ev nashnet.Event) {
	if nodeID == "" {
		return
	}
	p.mu.Lock()
	p.pending[nodeID] = append(p.pending[nodeID], ev)
	s := p.sessions[nodeID]
	p.mu.Unlock()
	if s != nil {
		select {
		case s.ready <- struct{}{}:
		default:
		}
	}
}

// takeEvents drains a node's pending events.
func (p *Pool) takeEvents(nodeID string) []nashnet.Event {
	p.mu.Lock()
	defer p.mu.Unlock()
	evs := p.pending[nodeID]
	delete(p.pending, nodeID)
	return evs
}

// signalPlacement wakes every held claim so a newly ready job is handed out
// without poll latency (D2). The dispatcher calls it through the
// placementSource seam on every queue transition.
func (p *Pool) signalPlacement() {
	p.mu.Lock()
	waiters := make([]*claimWaiter, 0, len(p.waiters))
	for w := range p.waiters {
		waiters = append(waiters, w)
	}
	p.mu.Unlock()
	for _, w := range waiters {
		select {
		case w.wake <- struct{}{}:
		default:
		}
	}
}

// addWaiter registers a held claim and reports whether the waiter cap admitted
// it (D2, D56); past the cap the claim answers 204 waiters_full at once.
func (p *Pool) addWaiter() (*claimWaiter, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.waiters) >= p.ceilings.MaxClaimWaiters {
		return nil, false
	}
	w := &claimWaiter{wake: make(chan struct{}, 1)}
	p.waiters[w] = struct{}{}
	return w, true
}

func (p *Pool) dropWaiter(w *claimWaiter) {
	p.mu.Lock()
	delete(p.waiters, w)
	p.mu.Unlock()
}

// replayLease admits a repeat result post on a released lease whose token hash
// matches the one that posted the recorded terminal (D6). It admits nothing
// else: another route, another token, or a lease with no recorded result falls
// through to the ordinary refusal.
func (p *Pool) replayLease(leaseID, token string, route nashnet.Route) (nashnet.Lease, bool) {
	if route != nashnet.RouteResult {
		return nashnet.Lease{}, false
	}
	p.mu.Lock()
	hash := p.resultAuth[leaseID]
	_, recorded := p.results[leaseID]
	p.mu.Unlock()
	if !recorded || !nashnet.TokenMatches(hash, token) {
		return nashnet.Lease{}, false
	}
	return p.leases.Get(leaseID)
}

// leaseForJob answers the dispatcher's view rendering: the job's current lease,
// live or not.
func (p *Pool) leaseForJob(jobID string) (nashnet.Lease, bool) {
	return p.leases.ByJob(jobID)
}

// logDroppedFor reports the coordinator-owned dropped-byte count for a job's
// current lease (D54).
func (p *Pool) logDroppedFor(jobID string) int64 {
	l, ok := p.leases.ByJob(jobID)
	if !ok {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.logDropped[l.LeaseID]
}

// NodeView is one row of GET /nashnet/nodes: the stored declaration and gate
// report, the session and staleness rendering of D3 and D45, the coordinator's
// own holds, and the node's live leases.
type NodeView struct {
	nashnet.NodeRecord
	Presence     string      `json:"presence"`
	StaleSeconds int64       `json:"stale_seconds"`
	Leases       []LeaseView `json:"leases,omitempty"`
	BreakerTrips int         `json:"breaker_trips,omitempty"`
	// BreakerHeldSeconds is how long the D63 hold has left, so an operator
	// reads a time window rather than a bare trip count.
	BreakerHeldSeconds int64            `json:"breaker_held_seconds,omitempty"`
	Degraded           map[string]int64 `json:"degraded_jobs,omitempty"`
}

// LeaseView renders one lease inside a node listing.
type LeaseView struct {
	LeaseID    string `json:"lease_id"`
	JobID      string `json:"job_id"`
	LeaseEpoch int64  `json:"lease_epoch"`
	State      string `json:"state"`
	Phase      string `json:"phase,omitempty"`
	GrantedAt  string `json:"granted_at"`
	Deadline   string `json:"deadline"`
	Attempt    int    `json:"attempt,omitempty"`
}

// nodeViews renders every node record for the operator listing and the queue
// snapshot's nodes array (D23).
func (p *Pool) nodeViews() []NodeView {
	now := p.now()
	records := p.nodes.List()
	out := make([]NodeView, 0, len(records))
	for _, rec := range records {
		out = append(out, p.nodeView(rec, now))
	}
	return out
}

func (p *Pool) nodeView(rec nashnet.NodeRecord, now time.Time) NodeView {
	v := NodeView{
		NodeRecord:   rec,
		Presence:     rec.Presence(now, p.nodeTTL, p.sessGrace),
		StaleSeconds: int64(now.Sub(rec.LastSeen) / time.Second),
	}
	for _, l := range p.leases.LiveForNode(rec.NodeID) {
		v.Leases = append(v.Leases, LeaseView{
			LeaseID:    l.LeaseID,
			JobID:      l.JobID,
			LeaseEpoch: l.LeaseEpoch,
			State:      l.State,
			Phase:      l.Phase,
			GrantedAt:  rfc3339(l.GrantedAt),
			Deadline:   rfc3339(l.Deadline),
			Attempt:    l.Attempt,
		})
	}
	v.BreakerTrips, v.BreakerHeldSeconds = p.breakerReport(rec.NodeID, now)
	p.mu.Lock()
	if marks := p.degraded[rec.NodeID]; len(marks) > 0 {
		v.Degraded = map[string]int64{}
		for job, until := range marks {
			if until.After(now) {
				v.Degraded[job] = int64(until.Sub(now) / time.Second)
			}
		}
		if len(v.Degraded) == 0 {
			v.Degraded = nil
		}
	}
	p.mu.Unlock()
	return v
}

// noteNack records a returned claim: the node is excluded from re-matching that
// job for the cooldown, three nacks of one job mark it degraded for that node
// (D8), and three consecutive prepare_node_failed nacks across any jobs trip
// its circuit breaker (D63). It reports whether this nack tripped the breaker,
// which the route turns into the node-wide hold. No mark is node-clearable;
// each clears on its own timer or on an operator act through the drain route.
func (p *Pool) noteNack(nodeID, jobID, reason string, cooldown time.Duration) (tripped bool) {
	if cooldown <= 0 {
		cooldown = DefaultNackCooldown
	}
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cooldowns[nodeID] == nil {
		p.cooldowns[nodeID] = map[string]time.Time{}
	}
	p.cooldowns[nodeID][jobID] = now.Add(cooldown)
	if p.nacks[nodeID] == nil {
		p.nacks[nodeID] = map[string]int{}
	}
	p.nacks[nodeID][jobID]++
	if p.nacks[nodeID][jobID] >= 3 {
		if p.degraded[nodeID] == nil {
			p.degraded[nodeID] = map[string]time.Time{}
		}
		p.degraded[nodeID][jobID] = now.Add(DefaultDegradedJobHold)
	}
	if p.noteBreakerLocked(nodeID, reason, now) {
		tripped = true
	}
	p.pruneHoldsLocked(now)
	return tripped
}

// heldFor reports whether the coordinator is holding this (node, job) pair: a
// nack cooldown or a degraded mark, both time-based and coordinator-owned.
func (p *Pool) heldFor(nodeID, jobID string) bool {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()
	if until, ok := p.cooldowns[nodeID][jobID]; ok && until.After(now) {
		return true
	}
	if until, ok := p.degraded[nodeID][jobID]; ok && until.After(now) {
		return true
	}
	return false
}

// clearBreaker lifts the D63 hold, resets its counter and its trip ladder, and
// drops the per-job degraded marks and cooldowns of D8 for one node. It is
// reachable only from the operator drain route: registration is a node route,
// so a node-clearable hold would be no hold at all.
func (p *Pool) clearBreaker(nodeID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.breaker, nodeID)
	delete(p.degraded, nodeID)
	delete(p.nacks, nodeID)
	delete(p.cooldowns, nodeID)
}

// pruneHoldsLocked drops the (node, job) marks of D8 whose timer has run out.
// They are keyed per pair and a coordinator holds nodes for its whole life, so
// without this pass the tables grow one entry per job every node ever returns.
// Callers hold p.mu.
func (p *Pool) pruneHoldsLocked(now time.Time) {
	for node, jobs := range p.cooldowns {
		for job, until := range jobs {
			if !until.After(now) {
				delete(jobs, job)
				// The per-job nack count is only read against the degraded
				// threshold while a hold stands, so it ages out with the hold
				// rather than counting a node's whole history.
				delete(p.nacks[node], job)
			}
		}
		if len(jobs) == 0 {
			delete(p.cooldowns, node)
			delete(p.nacks, node)
		}
	}
	for node, jobs := range p.degraded {
		for job, until := range jobs {
			if !until.After(now) {
				delete(jobs, job)
			}
		}
		if len(jobs) == 0 {
			delete(p.degraded, node)
		}
	}
}

// recordResult keeps a posted terminal replayable for the retention window
// (D6). The three per-lease maps are bounded here rather than dropped at
// release, because the replay a lost response needs happens after the lease is
// gone: a record ages out of the window, or the oldest goes when the count
// reaches the ceiling.
func (p *Pool) recordResult(leaseID string, resp nashnet.ResultResponse, tokenHash string) {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.results[leaseID] = resp
	p.resultAuth[leaseID] = tokenHash
	p.resultAt[leaseID] = now

	cutoff := now.Add(-resultRetention)
	for id, at := range p.resultAt {
		if at.Before(cutoff) {
			p.forgetResultLocked(id)
		}
	}
	for len(p.resultAt) > maxResultRecords {
		oldest, oldestAt := "", time.Time{}
		for id, at := range p.resultAt {
			if oldest == "" || at.Before(oldestAt) {
				oldest, oldestAt = id, at
			}
		}
		p.forgetResultLocked(oldest)
	}
}

// forgetResultLocked drops one lease's retained route records. Callers hold
// p.mu.
func (p *Pool) forgetResultLocked(leaseID string) {
	delete(p.results, leaseID)
	delete(p.resultAuth, leaseID)
	delete(p.resultAt, leaseID)
	delete(p.logDropped, leaseID)
}

// StartupSweep is the coordinator-restart half of D34 and D59: quarantine trees
// whose lease no longer lives, and the parts under them, are dropped once past
// their debug TTL, so a restart does not inherit the upload state of leases it
// has no record of. A live lease's tree is never touched, which is what makes
// the restart invisible to a node with an upload in flight.
func (p *Pool) StartupSweep() (int, error) {
	return p.quar.Sweep(func(nodeID, jobID, leaseID string) bool {
		l, ok := p.leases.Get(leaseID)
		return ok && l.Live() && l.JobID == jobID && l.NodeID == nodeID
	})
}

// writeTombstone records an operator revocation where authtoken consults it: a
// <node_id>.revoked file beside the grant, read on every verification so a
// revoked node is refused on its next call (D60).
func (p *Pool) writeTombstone(nodeID string) error {
	if p.nodesDir == "" {
		return errors.New("no nodes directory configured")
	}
	path := filepath.Join(p.nodesDir, nodeID+authtoken.TombstoneFileSuffix)
	return os.WriteFile(path, []byte(rfc3339(p.now())+"\n"), 0o600)
}

// rfc3339 renders a timestamp the way every other runnerd record does, and the
// zero time as an empty string.
func rfc3339(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339)
}

// writeNashnetError writes the {error, detail} shape every nashnet route uses,
// with the optional retry and offset fields of D50 and D54.
func writeNashnetError(w http.ResponseWriter, status int, body nashnet.ErrorBody) {
	if body.RetryAfterSeconds > 0 {
		w.Header().Set("Retry-After", itoa(body.RetryAfterSeconds))
	}
	writeJSON(w, status, body)
}

// nashnetError is the common one-line refusal.
func nashnetError(w http.ResponseWriter, status int, code, detail string) {
	writeNashnetError(w, status, nashnet.ErrorBody{Error: code, Detail: detail})
}
