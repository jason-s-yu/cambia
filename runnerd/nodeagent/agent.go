package nodeagent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
)

// haveCommitsCap mirrors the coordinator's thin-basis ceiling: nothing past 16
// entries reaches a bundle argv, so advertising more is wasted bytes (D48).
const haveCommitsCap = 16

// idleHeartbeatInterval is the D3 cadence for an idle node's heartbeat.
const idleHeartbeatInterval = 30 * time.Second

// Options configures an Agent. Every collaborator is injectable so the whole
// claim-to-result cycle runs offline against a stub coordinator, a fake ingest
// environment, and a fake launcher.
type Options struct {
	Config Config
	Signer *Signer
	// Client is the coordinator transport: the pinned HTTPS client a remote
	// node runs, or the coordinator's own in-process loopback (D65).
	Client   NodeTransport
	Env      Environment
	Launcher Launcher
	Prober   Prober
	Logger   *log.Logger
	Now      func() time.Time
	// PollInterval is how often the agent re-reads a launched job's
	// process.json while waiting for it to reach a terminal status.
	PollInterval time.Duration
	// CanBuildLibcambia is the declaration's can_build_libcambia. The caller
	// probes it once at startup rather than on every heartbeat.
	CanBuildLibcambia bool
	// ClaimOnce stops the claim loop after one claim attempt. It exists for
	// tests, which drive one cycle deterministically instead of racing a
	// background loop.
	ClaimOnce bool
}

// Agent is one node's participation in the pool: it registers, holds the
// events poll, claims work, stages and supervises it, uploads its artifacts,
// and posts its results.
type Agent struct {
	cfg      Config
	signer   *Signer
	client   NodeTransport
	env      Environment
	launcher Launcher
	prober   Prober
	log      *log.Logger
	now      func() time.Time
	poll     time.Duration
	claimOne bool

	canBuildLibcambia bool

	mu          sync.Mutex
	nodeEpoch   int64
	policy      nashnet.Policy
	drained     bool
	active      map[string]*jobRun
	haveCommits []string

	wg sync.WaitGroup
}

// New builds an Agent, defaulting the injectable collaborators to their
// production implementations.
func New(opts Options) (*Agent, error) {
	if opts.Signer == nil {
		return nil, errors.New("nodeagent: signer is required")
	}
	if opts.Client == nil {
		return nil, errors.New("nodeagent: client is required")
	}
	if opts.Env == nil {
		return nil, errors.New("nodeagent: environment is required")
	}
	if opts.Launcher == nil {
		return nil, errors.New("nodeagent: launcher is required")
	}
	if opts.Config.NodeID != "" && opts.Config.NodeID != opts.Signer.NodeID() {
		return nil, fmt.Errorf("%w: node_id %q does not match the id derived from %s (%s)",
			ErrConfig, opts.Config.NodeID, opts.Config.KeyPath, opts.Signer.NodeID())
	}
	if opts.Prober == nil {
		opts.Prober = NewHostProber()
	}
	if opts.Logger == nil {
		opts.Logger = log.New(os.Stderr, "nashnet-node: ", log.LstdFlags)
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.PollInterval <= 0 {
		opts.PollInterval = time.Second
	}
	return &Agent{
		cfg:               opts.Config,
		signer:            opts.Signer,
		client:            opts.Client,
		env:               opts.Env,
		launcher:          opts.Launcher,
		prober:            opts.Prober,
		log:               opts.Logger,
		now:               opts.Now,
		poll:              opts.PollInterval,
		claimOne:          opts.ClaimOnce,
		canBuildLibcambia: opts.CanBuildLibcambia,
		policy:            nashnet.DefaultPolicy(),
		active:            map[string]*jobRun{},
	}, nil
}

// NodeID returns the derived id this agent acts under.
func (a *Agent) NodeID() string { return a.signer.NodeID() }

// Run reattaches to whatever this node was already supervising, registers,
// holds the events poll, and claims until ctx is done.
func (a *Agent) Run(ctx context.Context) error {
	live, pending := a.reattach()
	rebound, err := a.register(ctx, live)
	if err != nil {
		return err
	}
	a.resumePending(ctx, pending, rebound)
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		a.eventsLoop(ctx)
	}()
	a.claimLoop(ctx)
	a.wg.Wait()
	return nil
}

// register declares this node and adopts the returned policy and epoch (D3).
// Every lease the node did not name in live_leases is revoked by the
// coordinator, so the response also tells a restarted agent which of its jobs
// it must orphan.
func (a *Agent) register(ctx context.Context, live []nashnet.LiveLease) (map[string]bool, error) {
	report, obs := a.gateReport()
	decl := a.declaration(obs)
	req := nashnet.RegisterRequest{
		NodeID:       a.signer.NodeID(),
		AgentVersion: a.cfg.AgentVersion,
		PlatformTag:  a.cfg.PlatformTag,
		Slots:        a.cfg.Slots,
		Kinds:        a.cfg.Kinds,
		Capabilities: mustJSON(decl),
		GateReport:   mustJSON(report),
		HaveCommits:  a.commits(),
		LiveLeases:   live,
	}
	resp, err := a.client.Register(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("register: %w", err)
	}
	a.mu.Lock()
	a.nodeEpoch = resp.NodeEpoch
	if resp.Policy.LeaseTTLSeconds > 0 {
		a.policy = resp.Policy
	}
	a.mu.Unlock()

	rebound := map[string]bool{}
	for _, id := range resp.ReboundLeases {
		rebound[id] = true
	}
	a.log.Printf("registered as %s (epoch %d, %d of %d leases rebound)",
		a.signer.NodeID(), resp.NodeEpoch, len(rebound), len(live))
	return rebound, nil
}

// claimLoop is the node's main loop: evaluate gates, claim when admitted and
// a slot is free, heartbeat when not.
func (a *Agent) claimLoop(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		wait := a.claimOnce(ctx)
		if a.claimOne {
			return
		}
		if wait <= 0 {
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}

// claimOnce runs one pass and returns how long to wait before the next. A
// denying gate report means the node heartbeats with it and does not claim, so
// an idle node always explains itself (D46).
func (a *Agent) claimOnce(ctx context.Context) time.Duration {
	report, obs := a.gateReport()
	free := a.slotsFree(report)

	if !report.Admit || free <= 0 || a.isDrained() {
		a.heartbeat(ctx, report, obs, free)
		return jitter(idleHeartbeatInterval)
	}

	req := nashnet.ClaimRequest{
		NodeID:       a.signer.NodeID(),
		NodeEpoch:    a.epoch(),
		AgentVersion: a.cfg.AgentVersion,
		SlotsFree:    free,
		BusyJobIDs:   a.busyJobIDs(),
		Capabilities: mustJSON(a.declaration(obs)),
		GateReport:   mustJSON(report),
		Kinds:        a.cfg.Kinds,
		HaveCommits:  a.commits(),
		WaitSeconds:  a.cfg.ClaimWaitSeconds,
	}
	claim, hold, err := a.client.Claim(ctx, req)
	if err != nil {
		if ctx.Err() != nil {
			return 0
		}
		a.log.Printf("claim: %v", err)
		return jitter(15 * time.Second)
	}
	if hold != nil {
		return jitter(time.Duration(hold.RetryAfterSeconds) * time.Second)
	}
	a.startJob(ctx, claim)
	// A node re-claims immediately after a handout so a chain of ready jobs
	// runs without poll latency (D2).
	return 0
}

// startJob records the lease and runs it on its own goroutine.
func (a *Agent) startJob(ctx context.Context, claim *nashnet.ClaimResponse) {
	rec := &leaseRecord{
		JobID:       claim.JobID,
		LeaseID:     claim.LeaseID,
		LeaseEpoch:  claim.LeaseEpoch,
		Token:       claim.LeaseToken,
		Deadline:    claim.LeaseDeadline,
		GrantedAt:   a.now().UTC().Format(time.RFC3339Nano),
		Attempt:     claim.Attempt,
		Resume:      claim.Resume,
		Commit:      claim.Snapshot.Commit,
		Spec:        claim.Spec,
		Policy:      claim.Policy,
		Seeds:       claim.Seeds,
		SnapshotURL: claim.Snapshot.URL,
		SnapshotSHA: claim.Snapshot.SHA256,
		Phase:       nashnet.PhaseClaimed,
	}
	if rec.Policy.LeaseTTLSeconds == 0 {
		rec.Policy = a.currentPolicy()
	}
	if err := writeLeaseRecord(a.cfg.BaseDir, rec); err != nil {
		a.log.Printf("persist lease %s: %v", rec.LeaseID, err)
	}
	job := &jobRun{agent: a, rec: rec, snapshot: claim.Snapshot}
	a.mu.Lock()
	a.active[rec.JobID] = job
	a.mu.Unlock()

	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		defer a.finishJob(rec.JobID)
		job.run(ctx)
	}()
}

// finishJob drops a job from the active set.
func (a *Agent) finishJob(jobID string) {
	a.mu.Lock()
	delete(a.active, jobID)
	a.mu.Unlock()
}

// eventsLoop keeps exactly one events request open at all times, re-issuing it
// on every response (D45). Cancel, revocation, and drain therefore reach the
// node within one round trip.
func (a *Agent) eventsLoop(ctx context.Context) {
	backoff := time.Second
	for {
		if ctx.Err() != nil {
			return
		}
		resp, err := a.client.Events(ctx, a.signer.NodeID(), a.cfg.ClaimWaitSeconds)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			a.log.Printf("events: %v", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(jitter(backoff)):
			}
			if backoff < 30*time.Second {
				backoff *= 2
			}
			continue
		}
		backoff = time.Second
		a.applyEvents(resp)
		if a.claimOne {
			return
		}
	}
}

// applyEvents dispatches one events response.
func (a *Agent) applyEvents(resp nashnet.EventsResponse) {
	if resp.NodeEpoch != 0 {
		a.mu.Lock()
		a.nodeEpoch = resp.NodeEpoch
		a.mu.Unlock()
	}
	for _, ev := range resp.Events {
		switch ev.Type {
		case nashnet.EventRevoke:
			a.revokeLease(ev.LeaseID, ev.Force)
		case nashnet.EventDrain:
			a.setDrained(true)
			a.log.Printf("coordinator drained this node")
		case nashnet.EventPolicy:
			// Policy changes arrive with the next claim or heartbeat response;
			// nothing to do beyond noting the event.
		}
	}
}

// revokeLease stops the job group of the named lease within this round trip
// (D36, D62). The lease token stays valid through the coordinator's grace
// period, so the job still commits a final manifest and posts its result.
func (a *Agent) revokeLease(leaseID string, force bool) {
	a.mu.Lock()
	var target *jobRun
	for _, j := range a.active {
		if j.rec.LeaseID == leaseID || leaseID == "" {
			target = j
			break
		}
	}
	a.mu.Unlock()
	if target == nil {
		return
	}
	target.requestStop(stopReason{kind: stopRevoked, force: force})
}

// heartbeat refreshes the volatile facts and the gate report while idle (D3).
func (a *Agent) heartbeat(ctx context.Context, report gates.Report, obs Observation, free int) {
	resp, err := a.client.Heartbeat(ctx, a.signer.NodeID(), nashnet.HeartbeatRequest{
		AgentVersion: a.cfg.AgentVersion,
		SlotsFree:    free,
		Capabilities: mustJSON(a.declaration(obs)),
		GateReport:   mustJSON(report),
		HaveCommits:  a.commits(),
	})
	if err != nil {
		if ctx.Err() == nil {
			a.log.Printf("heartbeat: %v", err)
		}
		return
	}
	a.mu.Lock()
	if resp.NodeEpoch != 0 {
		a.nodeEpoch = resp.NodeEpoch
	}
	a.drained = resp.Drain
	a.mu.Unlock()
}

// gateReport evaluates the node's gates against a fresh observation.
func (a *Agent) gateReport() (gates.Report, Observation) {
	return evaluateGates(a.cfg, a.prober, a.running(), a.now())
}

// declaration assembles the capability declaration from one observation.
func (a *Agent) declaration(obs Observation) capability.Declaration {
	return buildDeclaration(a.cfg, obs, a.canBuildLibcambia, a.commits())
}

// running is the node's own live-job accounting.
func (a *Agent) running() running {
	a.mu.Lock()
	defer a.mu.Unlock()
	r := running{slots: len(a.active)}
	for _, j := range a.active {
		if j.usesAccelerator() {
			r.acceleratorJobs++
		}
	}
	return r
}

// slotsFree is the smaller of the node's own free slots and what its gates
// offer, so a coordinator bug cannot oversubscribe the node (D13).
func (a *Agent) slotsFree(report gates.Report) int {
	a.mu.Lock()
	free := a.cfg.Slots - len(a.active)
	a.mu.Unlock()
	if report.SlotsOffered < free {
		free = report.SlotsOffered
	}
	if free < 0 {
		return 0
	}
	return free
}

func (a *Agent) busyJobIDs() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, 0, len(a.active))
	for id := range a.active {
		out = append(out, id)
	}
	return out
}

func (a *Agent) epoch() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.nodeEpoch
}

func (a *Agent) currentPolicy() nashnet.Policy {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.policy
}

func (a *Agent) isDrained() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.drained
}

func (a *Agent) setDrained(v bool) {
	a.mu.Lock()
	a.drained = v
	a.mu.Unlock()
}

// commits returns the mirror commits this node can negate a thin bundle
// against (D48), capped at what the coordinator will accept.
func (a *Agent) commits() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.haveCommits) <= haveCommitsCap {
		return append([]string(nil), a.haveCommits...)
	}
	return append([]string(nil), a.haveCommits[len(a.haveCommits)-haveCommitsCap:]...)
}

// noteCommit records a commit this node's mirror now holds.
func (a *Agent) noteCommit(commit string) {
	if len(commit) != 40 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, c := range a.haveCommits {
		if c == commit {
			return
		}
	}
	a.haveCommits = append(a.haveCommits, commit)
}

// mustJSON marshals a value that cannot fail to marshal; an impossible failure
// yields a null body rather than a panic in the claim path.
func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage("null")
	}
	return b
}

// jitter spreads retries so a pool of nodes recovering from an outage does not
// arrive in lockstep.
func jitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	spread := d / 4
	if spread <= 0 {
		return d
	}
	return d - spread/2 + time.Duration(rand.Int63n(int64(spread)))
}
