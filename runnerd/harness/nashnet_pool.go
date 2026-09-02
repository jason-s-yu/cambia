package harness

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
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
	DefaultRequestsPerMinute       = 600
	DefaultConnectionsPerNode      = 8
	DefaultConcurrentUploads       = 4
	DefaultConcurrentDownloads     = 2
	DefaultBundleBuildsPerMinute   = 4
	DefaultMaxClaimWaiters         = 64
	DefaultChunkBytes              = 64 << 20
	DefaultLeaseBytesPerTick       = 2 << 30
	DefaultLogBytesPerCall         = 2 << 20
	DefaultLogBytesPerJob          = 1 << 30
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
	// UnplaceableGrace is how long a ready job matches no node before its hold
	// is rendered in the view (D14).
	UnplaceableGrace time.Duration
	// NodeTTL and SessionGrace drive the presence rendering of D3 and D45.
	NodeTTL      time.Duration
	SessionGrace time.Duration
	// EmbeddedNodeID names the node running inside this process over the
	// loopback transport (D40, D65). Its leases materialize in place, because
	// its jobs write straight into the run dir the manifest names. It is
	// configured here rather than reported by the node, so no remote node can
	// claim the right to have its manifest believed without uploading a byte.
	EmbeddedNodeID string
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
	embedded   string
	policy     nashnet.Policy
	ceilings   Ceilings
	maxLeases  int
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
	// snapshots caches the bundle descriptor a claim resolved, keyed by lease
	// id, so the snapshot route serves the exact artifact the claim reported.
	snapshots map[string]snapshotDescriptor
	// quarGrants is the lease's grant set in the quarantine store's shape,
	// keyed by digest: a seeded file the node reports unchanged is copied from
	// the recorded source path rather than uploaded (D50, D51).
	quarGrants map[string]map[string]quarantine.Grant
	// tickBytes is the per-lease upload budget since its last progress post.
	tickBytes map[string]int64
	// logDropped is the coordinator-owned dropped-byte counter of D54; the
	// in-band marker is advisory text and never the evidence.
	logDropped map[string]int64
	// results records a posted terminal so a replay after a lost response
	// returns the recorded body rather than a conflict (D6). resultAuth keeps
	// the hash of the token that posted it, because the lease store drops the
	// token at release and the replay must still prove it holds it.
	results    map[string]nashnet.ResultResponse
	resultAuth map[string]string
	// cooldowns exclude a node from re-matching a job it nacked (D8).
	cooldowns map[string]map[string]time.Time
	// nacks counts a node's nacks per job; three mark the job degraded for it.
	nacks map[string]map[string]int
	// breaker is the per-node consecutive prepare_node_failed counter of D63.
	// W3-T14 owns its cooldown ladder; the counter and the operator reset live
	// here because the drain route is this ticket's.
	breaker map[string]int
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
		embedded:   cfg.EmbeddedNodeID,
		policy:     policy,
		ceilings:   ceil,
		maxLeases:  maxLeases,
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
		snapshots:  map[string]snapshotDescriptor{},
		quarGrants: map[string]map[string]quarantine.Grant{},
		tickBytes:  map[string]int64{},
		logDropped: map[string]int64{},
		results:    map[string]nashnet.ResultResponse{},
		resultAuth: map[string]string{},
		cooldowns:  map[string]map[string]time.Time{},
		nacks:      map[string]map[string]int{},
		breaker:    map[string]int{},
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
		// The job never left the queue: a leased job stays at its original
		// submit_seq and the placement scan simply skips it while a lease
		// stands (D32, "requeue at the original submit_seq").
		p.disp.reDispatch()
	case nashnet.VerdictCanceled:
		p.disp.recordPoolTerminal(o.JobID, StateCanceled, "lease stopped by the coordinator")
	case nashnet.VerdictFinalize:
		p.disp.finalizePoolJob(o.JobID)
	}
	p.signalPlacement()
}

// releaseLeaseState drops the per-lease route state a released lease no longer
// needs. The lease record itself is retained by the store for the D35 verdict.
func (p *Pool) releaseLeaseState(leaseID string) {
	p.mu.Lock()
	delete(p.snapshots, leaseID)
	delete(p.quarGrants, leaseID)
	delete(p.tickBytes, leaseID)
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
	Presence     string           `json:"presence"`
	StaleSeconds int64            `json:"stale_seconds"`
	Leases       []LeaseView      `json:"leases,omitempty"`
	BreakerTrips int              `json:"breaker_trips,omitempty"`
	Degraded     map[string]int64 `json:"degraded_jobs,omitempty"`
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
	p.mu.Lock()
	v.BreakerTrips = p.breaker[rec.NodeID]
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
// job for the cooldown, and three nacks of one job mark it degraded for that
// node (D8). Neither mark is node-clearable; both clear on their own timer or
// on an operator act through the drain route (D63).
func (p *Pool) noteNack(nodeID, jobID, reason string, cooldown time.Duration) {
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
	if reason == nashnet.NackPrepareNodeFailed {
		p.breaker[nodeID]++
	} else {
		p.breaker[nodeID] = 0
	}
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

// clearBreaker resets the D63 counter and the per-job degraded marks of D8 for
// one node. It is reachable only from the operator drain route: registration is
// a node route, so a node-clearable hold would be no hold at all.
func (p *Pool) clearBreaker(nodeID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.breaker, nodeID)
	delete(p.degraded, nodeID)
	delete(p.nacks, nodeID)
	delete(p.cooldowns, nodeID)
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
