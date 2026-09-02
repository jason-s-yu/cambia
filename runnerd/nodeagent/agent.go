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
	// AlgoSubcommand resolves a job kind to its cambia subcommand. Nil selects
	// this package's own Subcommand table.
	AlgoSubcommand AlgoResolver
	// InPlace runs the node against the coordinator's own runs dir and mirror,
	// which is what --role both does: no snapshot fetch, no seed fetch, no log
	// push, and no blob upload, because every byte those steps would move is
	// already at its destination (D40). The manifest commit still runs, so a
	// reserved path an embedded run writes is validated and recorded exactly
	// as a remote one would be.
	InPlace bool
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
	algo     AlgoResolver
	inPlace  bool
	// slots is this node's launch accounting, moved here from the dispatcher
	// with the launch path it guards (D1).
	slots *Slots

	canBuildLibcambia bool

	mu        sync.Mutex
	nodeEpoch int64
	policy    nashnet.Policy
	// hold is the coordinator's hold on this node, its reason verbatim off the
	// wire and empty when there is none. Every call that can carry it sets it,
	// so a hold learned from one of them is never undone by the next.
	hold string
	// gateDrain is this node's own hold, set when a gate breaches mid job with
	// on_breach: drain (D46). It is separate from the coordinator's: the two
	// have different owners and different lifetimes, and sharing one field let
	// the next heartbeat answer for a gate it knows nothing about. It closes
	// the window between a mid-job breach and the claim loop's next gate
	// evaluation, and lifts on that evaluation once the gates admit again.
	gateDrain   bool
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
	if opts.AlgoSubcommand == nil {
		opts.AlgoSubcommand = Subcommand
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
		algo:              opts.AlgoSubcommand,
		inPlace:           opts.InPlace,
		slots:             NewSlots(opts.Config.Slots),
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
	// A registration is the first call after a node restart, so the hold it
	// answers with is what keeps a held node from claiming before its first
	// heartbeat (D3, D63).
	a.applyHold(resp.Hold)

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

	if report.Admit {
		// The gates this node evaluates are the authority on its own hold, so an
		// admitting report lifts the one a mid-job breach set (D46).
		a.setGateDrain(false)
	}
	if !report.Admit || free <= 0 || a.isHeld() || a.isGateDrained() {
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
	if spec, derr := decodeSpec(claim.Spec); derr == nil {
		// The exclusive flag is read before the goroutine starts because the
		// slot it holds is claimed here, in the caller, so a second claim
		// racing this one sees the occupancy (D13).
		job.spec = spec
	}
	a.mu.Lock()
	a.active[rec.JobID] = job
	a.mu.Unlock()
	a.slots.Claim(job.spec.Exclusive)

	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		defer a.finishJob(job)
		job.run(ctx)
	}()
}

// finishJob drops a job from the active set and frees the slot it held.
func (a *Agent) finishJob(j *jobRun) {
	a.mu.Lock()
	delete(a.active, j.rec.JobID)
	a.mu.Unlock()
	a.slots.Release(j.spec.Exclusive)
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
			// The event's own hold decides, so both a new hold and a lift take
			// effect on the round trip that delivers them rather than waiting for
			// the next heartbeat.
			a.applyHold(ev.Hold)
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
	a.noteHeartbeat(resp)
}

// noteHeartbeat applies one heartbeat answer: the epoch it fences on and the
// hold it reports, which is the coordinator's alone and says nothing about the
// node's own gates.
func (a *Agent) noteHeartbeat(resp nashnet.HeartbeatResponse) {
	a.mu.Lock()
	if resp.NodeEpoch != 0 {
		a.nodeEpoch = resp.NodeEpoch
	}
	a.mu.Unlock()
	a.applyHold(resp.Hold)
}

// applyHold records a hold off the wire and logs the transitions. A hold and a
// lift are both worth one line; a standing hold repeated on every heartbeat is
// not.
func (a *Agent) applyHold(hold string) {
	if !a.setHold(hold) {
		return
	}
	if hold == "" {
		a.log.Printf("coordinator lifted this node's hold")
		return
	}
	a.log.Printf("coordinator is holding this node: %s", hold)
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
// offer, so a coordinator bug cannot oversubscribe the node (D13). The slot
// accounting reports zero while an exclusive job holds the node, so a claim
// never advertises capacity the launch would refuse.
func (a *Agent) slotsFree(report gates.Report) int {
	free := a.slots.Free()
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

// isHeld reports whether the coordinator is refusing this node's claims, for
// either of its reasons (D63).
func (a *Agent) isHeld() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.hold != ""
}

// setGateDrain records this node's own gate-driven hold (D46).
func (a *Agent) setGateDrain(v bool) {
	a.mu.Lock()
	a.gateDrain = v
	a.mu.Unlock()
}

// isGateDrained reports whether a mid-job gate breach is holding this node off
// new work.
func (a *Agent) isGateDrained() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.gateDrain
}

// setHold records the hold a response or an event carried, and reports whether
// it changed, which is all the caller needs to log a transition rather than
// every repeat of a standing hold.
func (a *Agent) setHold(hold string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.hold == hold {
		return false
	}
	a.hold = hold
	return true
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
