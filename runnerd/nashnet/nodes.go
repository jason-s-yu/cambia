package nashnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Node registry errors.
var (
	// ErrUnknownNode is a node id with no record: it never registered, or the
	// coordinator restarted and it has not re-registered yet.
	ErrUnknownNode = errors.New("nashnet: unknown node")
	// ErrNodeRevoked is a tombstoned node. The authority is the tombstone beside
	// the enrollment grant, consulted on every token verification (D60); the
	// record's flag is the second refusal, so a revoked node cannot re-register
	// its way back into the pool through a stale credential.
	ErrNodeRevoked = errors.New("nashnet: node revoked")
)

// Presence values rendered for an operator (D3, D45).
const (
	// PresenceOnline is a node seen inside the node TTL and holding an events
	// request.
	PresenceOnline = "online"
	// PresenceDisconnected is a node with no held events request for the session
	// grace. The lease TTL stays the only expiry authority: the session grace
	// changes what an operator sees, never what the sweeper does (D45).
	PresenceDisconnected = "disconnected"
	// PresenceStale is a node that has sent nothing at all inside the node TTL
	// (D3). It outranks disconnected, which it implies.
	PresenceStale = "stale"
	// PresenceRevoked is a tombstoned node, kept in the listing for audit (D60).
	PresenceRevoked = "revoked"
)

// NodeRecord is the coordinator's record of one node (D3, D45). It is state,
// not trust: enrollment is the operator-signed grant, and every field below
// except the epochs and the timestamps is a node assertion, stored verbatim and
// clamped by the grant before placement reads it (D9, D47).
type NodeRecord struct {
	NodeID    string `json:"node_id"`
	NodeEpoch int64  `json:"node_epoch"`
	// AgentVersion, PlatformTag, Slots, and Kinds are the declaration summary an
	// operator listing renders without parsing the declaration itself.
	AgentVersion string   `json:"agent_version,omitempty"`
	PlatformTag  string   `json:"platform_tag,omitempty"`
	Slots        int      `json:"slots,omitempty"`
	Kinds        []string `json:"kinds,omitempty"`
	// Capabilities is the D9 declaration and GateReport the D46 report, both held
	// verbatim; runnerd/nashnet/capability parses them (W1-T3).
	Capabilities json.RawMessage `json:"capabilities,omitempty"`
	GateReport   json.RawMessage `json:"gate_report,omitempty"`
	HaveCommits  []string        `json:"have_commits,omitempty"`
	SlotsFree    int             `json:"slots_free,omitempty"`
	RegisteredAt time.Time       `json:"registered_at"`
	// LastSeen is the last call of any kind from the node, which is what the node
	// TTL measures (D3).
	LastSeen time.Time `json:"last_seen"`
	// ConnectedUntil is the deadline of the node's currently held events request,
	// which is the session liveness fact of D45.
	ConnectedUntil time.Time `json:"connected_until,omitzero"`
	// Drained is the operator drain alone, set and lifted only by the drain
	// route. The D63 circuit breaker holds a node without touching it, so the
	// question "is this node held" is the pool's effectiveHold rather than this
	// flag. It is also not the node's own drain gate, which is node-evaluated
	// and arrives inside the gate report (D46).
	Drained   bool      `json:"drained,omitempty"`
	Revoked   bool      `json:"revoked,omitempty"`
	RevokedAt time.Time `json:"revoked_at,omitzero"`
}

// Presence renders the node's liveness at now (D3, D45). Stale outranks
// disconnected: a node that has sent nothing at all is necessarily holding no
// events request, and the broader fact is the one an operator needs.
func (n *NodeRecord) Presence(now time.Time, nodeTTL, sessionGrace time.Duration) string {
	switch {
	case n.Revoked:
		return PresenceRevoked
	case now.Sub(n.LastSeen) > nodeTTL:
		return PresenceStale
	case now.After(n.ConnectedUntil.Add(sessionGrace)):
		return PresenceDisconnected
	default:
		return PresenceOnline
	}
}

// Clone returns a copy safe to hand out: the raw declaration and the string
// slices are copied, so a caller cannot mutate registry state through them.
func (n *NodeRecord) Clone() NodeRecord {
	c := *n
	c.Capabilities = append(json.RawMessage(nil), n.Capabilities...)
	c.GateReport = append(json.RawMessage(nil), n.GateReport...)
	c.Kinds = append([]string(nil), n.Kinds...)
	c.HaveCommits = append([]string(nil), n.HaveCommits...)
	return c
}

// RegistryConfig configures a NodeRegistry. The zero value is the documented
// default set.
type RegistryConfig struct {
	// NodeTTL is RUNNERD_NASHNET_NODE_TTL (D3). Zero means the default.
	NodeTTL time.Duration
	// SessionGrace is RUNNERD_NASHNET_SESSION_GRACE (D45). Zero means the
	// default.
	SessionGrace time.Duration
	// Now is the injected clock. Nil means time.Now.
	Now func() time.Time
}

func (c RegistryConfig) withDefaults() RegistryConfig {
	if c.NodeTTL <= 0 {
		c.NodeTTL = DefaultNodeTTLSeconds * time.Second
	}
	if c.SessionGrace <= 0 {
		c.SessionGrace = DefaultSessionGraceSeconds * time.Second
	}
	if c.Now == nil {
		c.Now = time.Now
	}
	return c
}

// NodeRegistry holds the node records. Registration is a declaration, not an
// admission: a record here grants nothing on its own, since every route is
// authorized by the node's enrollment grant (D3, D60).
type NodeRegistry struct {
	cfg RegistryConfig

	mu    sync.Mutex
	nodes map[string]*NodeRecord
	// floors carries a node epoch a restored lease already used, so a restarted
	// coordinator cannot hand back an epoch below one a live lease is fenced on
	// (D34).
	floors map[string]int64
}

// NewNodeRegistry returns an empty registry.
func NewNodeRegistry(cfg RegistryConfig) *NodeRegistry {
	return &NodeRegistry{
		cfg:    cfg.withDefaults(),
		nodes:  make(map[string]*NodeRecord),
		floors: make(map[string]int64),
	}
}

// Now reads the injected clock.
func (r *NodeRegistry) Now() time.Time { return r.cfg.Now() }

// SeedEpoch raises the floor the next registration of nodeID must exceed. The
// coordinator calls it once per restored lease at startup: the lease is fenced
// on the node epoch it carries, and a fresh registry would otherwise restart
// the counter at 1 and fence out a node that never noticed the restart (D34).
func (r *NodeRegistry) SeedEpoch(nodeID string, epoch int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if epoch > r.floors[nodeID] {
		r.floors[nodeID] = epoch
	}
}

// Register upserts the record keyed by nodeID, which the caller derives from
// the verified token subject and never from the request body (D3, D25), and
// returns it with a bumped node epoch. The live_leases half of a registration
// is LeaseStore.ReBind, which the caller runs with the epoch returned here.
func (r *NodeRegistry) Register(nodeID string, req RegisterRequest) (NodeRecord, error) {
	if nodeID == "" {
		return NodeRecord{}, fmt.Errorf("%w: empty node id", ErrUnknownNode)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	now := r.cfg.Now()
	n, ok := r.nodes[nodeID]
	if !ok {
		n = &NodeRecord{NodeID: nodeID, RegisteredAt: now}
		r.nodes[nodeID] = n
	}
	if n.Revoked {
		return NodeRecord{}, fmt.Errorf("%w: %s", ErrNodeRevoked, nodeID)
	}
	n.NodeEpoch = r.nextEpochLocked(nodeID, n.NodeEpoch)
	n.AgentVersion = req.AgentVersion
	n.PlatformTag = req.PlatformTag
	n.Slots = req.Slots
	n.Kinds = append([]string(nil), req.Kinds...)
	n.HaveCommits = append([]string(nil), req.HaveCommits...)
	if req.Capabilities != nil {
		n.Capabilities = append(json.RawMessage(nil), req.Capabilities...)
	}
	if req.GateReport != nil {
		n.GateReport = append(json.RawMessage(nil), req.GateReport...)
	}
	n.LastSeen = now
	return n.Clone(), nil
}

// nextEpochLocked advances a node epoch past both its current value and any
// floor a restored lease set. Callers hold r.mu.
func (r *NodeRegistry) nextEpochLocked(nodeID string, current int64) int64 {
	next := current
	if floor := r.floors[nodeID]; floor > next {
		next = floor
	}
	return next + 1
}

// Heartbeat refreshes the volatile facts and the gate report of an idle node
// (D3). It is not a registration: it never bumps the epoch, and an unknown node
// is refused rather than created, so a node that missed a coordinator restart
// learns to register again.
func (r *NodeRegistry) Heartbeat(nodeID string, req HeartbeatRequest) (NodeRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n, err := r.liveLocked(nodeID)
	if err != nil {
		return NodeRecord{}, err
	}
	if req.AgentVersion != "" {
		n.AgentVersion = req.AgentVersion
	}
	n.SlotsFree = req.SlotsFree
	if req.Capabilities != nil {
		n.Capabilities = append(json.RawMessage(nil), req.Capabilities...)
	}
	if req.GateReport != nil {
		n.GateReport = append(json.RawMessage(nil), req.GateReport...)
	}
	if req.HaveCommits != nil {
		n.HaveCommits = append([]string(nil), req.HaveCommits...)
	}
	n.LastSeen = r.cfg.Now()
	return n.Clone(), nil
}

// ObserveClaim refreshes the same facts from a claim, which carries them on
// every call (D9). Like Heartbeat it never bumps the epoch.
func (r *NodeRegistry) ObserveClaim(nodeID string, req ClaimRequest) (NodeRecord, error) {
	return r.Heartbeat(nodeID, HeartbeatRequest{
		AgentVersion: req.AgentVersion,
		SlotsFree:    req.SlotsFree,
		Capabilities: req.Capabilities,
		GateReport:   req.GateReport,
		HaveCommits:  req.HaveCommits,
	})
}

// HoldSession records the deadline of the node's currently held events request,
// which is the connected_until liveness fact of D45. It refreshes LastSeen too:
// a busy node's only regular call is this one.
func (r *NodeRegistry) HoldSession(nodeID string, until time.Time) (NodeRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n, err := r.liveLocked(nodeID)
	if err != nil {
		return NodeRecord{}, err
	}
	n.ConnectedUntil = until
	n.LastSeen = r.cfg.Now()
	return n.Clone(), nil
}

// SetDrained sets or lifts the coordinator-side hold (the operator drain route,
// which is two-way, and the D63 breaker). It is not a substitute for the node's
// own drain gate, which no route can override (D46).
func (r *NodeRegistry) SetDrained(nodeID string, drained bool) (NodeRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n, err := r.liveLocked(nodeID)
	if err != nil {
		return NodeRecord{}, err
	}
	n.Drained = drained
	return n.Clone(), nil
}

// Revoke tombstones a node and bumps its epoch, which supersedes every lease it
// held (D60). The record is kept for audit. Revoking every one of its leases is
// LeaseStore.RevokeNode, which the caller runs alongside this.
func (r *NodeRegistry) Revoke(nodeID string) (NodeRecord, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	n, ok := r.nodes[nodeID]
	if !ok {
		return NodeRecord{}, fmt.Errorf("%w: %s", ErrUnknownNode, nodeID)
	}
	n.Revoked = true
	n.RevokedAt = r.cfg.Now()
	n.NodeEpoch = r.nextEpochLocked(nodeID, n.NodeEpoch)
	return n.Clone(), nil
}

// Get returns one record.
func (r *NodeRegistry) Get(nodeID string) (NodeRecord, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n, ok := r.nodes[nodeID]
	if !ok {
		return NodeRecord{}, false
	}
	return n.Clone(), true
}

// List returns every record, ordered by node id.
func (r *NodeRegistry) List() []NodeRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.nodes))
	for id := range r.nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]NodeRecord, 0, len(ids))
	for _, id := range ids {
		out = append(out, r.nodes[id].Clone())
	}
	return out
}

// Presence renders one node's liveness under the registry's clock and windows.
func (r *NodeRegistry) Presence(nodeID string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n, ok := r.nodes[nodeID]
	if !ok {
		return "", fmt.Errorf("%w: %s", ErrUnknownNode, nodeID)
	}
	return n.Presence(r.cfg.Now(), r.cfg.NodeTTL, r.cfg.SessionGrace), nil
}

// liveLocked resolves a known, un-revoked node. Callers hold r.mu.
func (r *NodeRegistry) liveLocked(nodeID string) (*NodeRecord, error) {
	n, ok := r.nodes[nodeID]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownNode, nodeID)
	}
	if n.Revoked {
		return nil, fmt.Errorf("%w: %s", ErrNodeRevoked, nodeID)
	}
	return n, nil
}
