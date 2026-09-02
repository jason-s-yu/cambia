// Package nashnet holds the coordinator-side core of the compute pool: the
// node protocol wire types of design section 1, the fenced lease store and its
// tokens (D4, D44), the node registry record (D3, D45), and the expiry sweeper
// (D7). It is a leaf package by construction: it imports nothing from
// runnerd/harness or runnerd/procmgr, so the dispatcher and the HTTP layer can
// depend on it without a cycle, and its state machine is testable with no
// server, no network, and no clock of its own (see leaf_test.go).
//
// Ownership seams with the sibling wave-1 tickets, so the types below do not
// collide with theirs: the capability declaration (D9) and the gate report
// (D46) are carried as json.RawMessage and parsed by runnerd/nashnet/capability
// and .../gates (W1-T3); the persisted JobSpec is carried as json.RawMessage
// because harness owns it (W1-T4); the blob, manifest, and journal bodies of
// D50, D51, and D55 belong to the quarantine store (W1-T7) and are referenced
// here only by their scalar digest and sequence fields.
package nashnet

import "encoding/json"

// HeaderLeaseToken carries the opaque lease token on every lease route (D44).
// The token never travels in a URL, a query parameter, or a body, so it stays
// out of access logs and out of anything fronting the listener.
const HeaderLeaseToken = "X-Nashnet-Lease"

// Node-reported lease phases (D5). The coordinator writes PhaseStopping itself
// and only when it initiated the stop; it is outside the node's vocabulary, so
// a node cannot launder a crash into an operator cancel by reporting it.
const (
	PhaseClaimed    = "claimed"
	PhaseFetching   = "fetching"
	PhasePreparing  = "preparing"
	PhaseRunning    = "running"
	PhaseUploading  = "uploading"
	PhaseCommitting = "committing"
	PhaseStopping   = "stopping"
)

// Terminal states a node may post with a result (D6, D62).
const (
	ResultStopped   = "stopped"
	ResultCrashed   = "crashed"
	ResultCanceled  = "canceled"
	ResultFailed    = "failed"
	ResultPreempted = "preempted"
)

// Hold reasons on a 204 claim (D2).
const (
	HoldNodeGated        = "node_gated"
	HoldExclusivePending = "exclusive_pending"
	HoldNoMatch          = "no_match"
	HoldWaitersFull      = "waiters_full"
)

// Nack reasons (D8, D63). A nack is a scheduling event: the job returns to the
// ready set at its original position with no attempt increment.
const (
	NackGateBreach        = "gate_breach"
	NackSnapshotFailed    = "snapshot_failed"
	NackBundlePrereqMiss  = "bundle_prereq_miss"
	NackPrepareNodeFailed = "prepare_node_failed"
	NackSeedUnavailable   = "seed_unavailable"
	NackCapacity          = "capacity"
)

// Session event types on the held events poll (D45).
const (
	EventRevoke = "revoke"
	EventDrain  = "drain"
	EventPolicy = "policy"
)

// Error codes returned in an ErrorBody (the status-code summary closing section
// 1). The HTTP status each maps to is the server ticket's table; the codes are
// wire facts and live with the messages that carry them.
const (
	CodeWrongAudience       = "wrong_audience"
	CodeWrongSubject        = "wrong_subject"
	CodeForbiddenPath       = "forbidden_path"
	CodeLeaseSuperseded     = "lease_superseded"
	CodeSessionSuperseded   = "session_superseded"
	CodeManifestOutOfOrder  = "manifest_out_of_order"
	CodeBlobsMissing        = "blobs_missing"
	CodeOffsetMismatch      = "offset_mismatch"
	CodeArtifactsIncomplete = "artifacts_incomplete"
	CodeSeedMissing         = "seed_missing"
	CodeOverCap             = "over_cap"
	CodeHashMismatch        = "hash_mismatch"
	CodeRunDBInvalid        = "rundb_invalid"
	CodeInvalidManifest     = "invalid_manifest"
	CodeInvalidDigest       = "invalid_digest"
	CodeInvalidPhase        = "invalid_phase"
	CodeRateLimited         = "rate_limited"
	CodeStoreFull           = "store_full"
)

// Policy is the pool policy the coordinator returns at registration and with
// every claim (D2, D56). The byte bounds are echoed to the node so it can size
// its own uploads; the coordinator enforces them independently.
type Policy struct {
	LeaseTTLSeconds         int   `json:"lease_ttl_seconds"`
	MaxLeaseSeconds         int   `json:"max_lease_seconds"`
	ProgressIntervalSeconds int   `json:"progress_interval_seconds"`
	ChunkBytes              int64 `json:"chunk_bytes"`
	MaxFileBytes            int64 `json:"max_file_bytes"`
	MaxLeaseBytes           int64 `json:"max_lease_bytes"`
	LogBytesPerCall         int64 `json:"log_bytes_per_call"`
}

// Policy defaults (the policy block of D2, the bounds table of D56). Env
// overrides (RUNNERD_NASHNET_LEASE_TTL and friends) are read by the server
// ticket; these are the values it defaults to.
const (
	DefaultLeaseTTLSeconds = 120
	// DefaultMaxLeaseSeconds is 72h, checked from granted_at regardless of
	// renewals so a node posting progress forever cannot hold a lease forever
	// (D4).
	DefaultMaxLeaseSeconds         = 259200
	DefaultProgressIntervalSeconds = 30
	DefaultChunkBytes              = 8 << 20
	DefaultMaxFileBytes            = 8 << 30
	DefaultMaxLeaseBytes           = 64 << 30
	DefaultLogBytesPerCall         = 2 << 20
	// DefaultNodeTTLSeconds is RUNNERD_NASHNET_NODE_TTL: a node unseen for this
	// long renders stale (D3).
	DefaultNodeTTLSeconds = 180
	// DefaultSessionGraceSeconds is RUNNERD_NASHNET_SESSION_GRACE: a node with no
	// held events request for this long renders disconnected (D45). It changes
	// what an operator sees, never what the sweeper does.
	DefaultSessionGraceSeconds = 90
	// DefaultMaxLeasesPerNode is RUNNERD_NASHNET_MAX_LEASES_PER_NODE, the
	// coordinator's own ceiling on concurrent leases per node (D47).
	DefaultMaxLeasesPerNode = 8
)

// DefaultPolicy returns the pool policy with every field at its documented
// default.
func DefaultPolicy() Policy {
	return Policy{
		LeaseTTLSeconds:         DefaultLeaseTTLSeconds,
		MaxLeaseSeconds:         DefaultMaxLeaseSeconds,
		ProgressIntervalSeconds: DefaultProgressIntervalSeconds,
		ChunkBytes:              DefaultChunkBytes,
		MaxFileBytes:            DefaultMaxFileBytes,
		MaxLeaseBytes:           DefaultMaxLeaseBytes,
		LogBytesPerCall:         DefaultLogBytesPerCall,
	}
}

// LiveLease is one entry of a register request's live_leases list (D3): a lease
// the restarting agent reattached to, named with the hash of the lease token it
// still holds. It re-binds only when the lease record's own node_id equals the
// authenticated subject and the hash compares equal under
// crypto/subtle.ConstantTimeCompare, so a node-supplied hash alone authorizes
// nothing.
type LiveLease struct {
	LeaseID   string `json:"lease_id"`
	JobID     string `json:"job_id,omitempty"`
	TokenHash string `json:"token_hash"`
}

// RegisterRequest is POST /nashnet/nodes/register (D3). NodeID is echoed for
// logging only: the coordinator keys the record by the node id it derives from
// the verified token sub and refuses a body id that disagrees with
// 403 wrong_subject (D25).
type RegisterRequest struct {
	NodeID       string          `json:"node_id,omitempty"`
	AgentVersion string          `json:"agent_version,omitempty"`
	PlatformTag  string          `json:"platform_tag,omitempty"`
	Slots        int             `json:"slots,omitempty"`
	Kinds        []string        `json:"kinds,omitempty"`
	Capabilities json.RawMessage `json:"capabilities,omitempty"`
	GateReport   json.RawMessage `json:"gate_report,omitempty"`
	HaveCommits  []string        `json:"have_commits,omitempty"`
	LiveLeases   []LiveLease     `json:"live_leases,omitempty"`
}

// RegisterResponse carries the pool policy and the bumped node epoch (D3), plus
// the outcome of the live_leases re-bind so a restarted agent learns
// immediately which of its leases it may keep supervising and which it must
// abandon (D36).
type RegisterResponse struct {
	NodeID        string   `json:"node_id"`
	NodeEpoch     int64    `json:"node_epoch"`
	Policy        Policy   `json:"policy"`
	ReboundLeases []string `json:"rebound_leases,omitempty"`
	RevokedLeases []string `json:"revoked_leases,omitempty"`
	ServerTime    string   `json:"server_time"`
}

// HeartbeatRequest is POST /nashnet/nodes/{node}/heartbeat every 30s while idle
// (D3): it refreshes the volatile facts and the gate report. Liveness while
// busy is the events long poll (D45), not this.
type HeartbeatRequest struct {
	AgentVersion string          `json:"agent_version,omitempty"`
	SlotsFree    int             `json:"slots_free,omitempty"`
	Capabilities json.RawMessage `json:"capabilities,omitempty"`
	GateReport   json.RawMessage `json:"gate_report,omitempty"`
	HaveCommits  []string        `json:"have_commits,omitempty"`
}

// HeartbeatResponse tells an idle node its current epoch and whether the
// coordinator holds it (an operator drain or the circuit breaker, D63).
type HeartbeatResponse struct {
	NodeEpoch  int64  `json:"node_epoch"`
	Drain      bool   `json:"drain,omitempty"`
	ServerTime string `json:"server_time"`
}

// Event is one entry of an events response (D45).
//
// Drain carries the state a drain event sets, so the event says which way the
// operator moved the hold. It is the whole payload of that event type: absent
// is false, which is the lift, and a node applies the value rather than reading
// the event's arrival as a drain.
type Event struct {
	Type    string `json:"type"`
	LeaseID string `json:"lease_id,omitempty"`
	Force   bool   `json:"force,omitempty"`
	Drain   bool   `json:"drain,omitempty"`
}

// EventsResponse answers GET /nashnet/nodes/{node}/events, held up to 30s and
// answered early when an event is pending (D45).
type EventsResponse struct {
	Events     []Event `json:"events"`
	NodeEpoch  int64   `json:"node_epoch"`
	ServerTime string  `json:"server_time"`
}

// ClaimRequest is POST /nashnet/claim (D2). Capabilities and GateReport are
// stored verbatim and parsed by the capability package (D9, D46, D47).
type ClaimRequest struct {
	NodeID       string          `json:"node_id,omitempty"`
	NodeEpoch    int64           `json:"node_epoch,omitempty"`
	AgentVersion string          `json:"agent_version,omitempty"`
	SlotsFree    int             `json:"slots_free"`
	BusyJobIDs   []string        `json:"busy_job_ids,omitempty"`
	Capabilities json.RawMessage `json:"capabilities,omitempty"`
	GateReport   json.RawMessage `json:"gate_report,omitempty"`
	Kinds        []string        `json:"kinds,omitempty"`
	HaveCommits  []string        `json:"have_commits,omitempty"`
	WaitSeconds  int             `json:"wait_seconds,omitempty"`
}

// SnapshotRef names the coordinator-served git bundle at the receipt commit
// (D48). ThinBasis is empty for a full bundle.
type SnapshotRef struct {
	URL       string   `json:"url"`
	Commit    string   `json:"commit"`
	SHA256    string   `json:"sha256"`
	Size      int64    `json:"size"`
	ThinBasis []string `json:"thin_basis,omitempty"`
}

// SeedEntry is one file of a served seed (D53).
type SeedEntry struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// Seed is one seed group of the lease grant set (D53): the resolved,
// existence-verified inputs a job reads. The snapshot digest plus the seed
// entries are the whole of what the lease authorizes the node to fetch, so a
// path outside the set answers the same 404 as an unknown one.
type Seed struct {
	SeedID  string      `json:"seed_id"`
	Kind    string      `json:"kind"`
	Entries []SeedEntry `json:"entries"`
}

// ClaimResponse is the 200 answer to a claim: one job, its lease, and the
// one-time lease token (D2, D44). LeaseToken is returned once and never stored
// in the clear; the coordinator keeps only its hash.
type ClaimResponse struct {
	JobID         string          `json:"job_id"`
	LeaseID       string          `json:"lease_id"`
	LeaseEpoch    int64           `json:"lease_epoch"`
	LeaseDeadline string          `json:"lease_deadline"`
	LeaseToken    string          `json:"lease_token"`
	Spec          json.RawMessage `json:"spec"`
	Resume        bool            `json:"resume,omitempty"`
	Attempt       int             `json:"attempt"`
	Snapshot      SnapshotRef     `json:"snapshot"`
	Seeds         []Seed          `json:"seeds,omitempty"`
	Policy        Policy          `json:"policy"`
}

// ClaimHold is the 204 answer to a claim: nothing placeable, with the reason
// and how long to wait (D2).
type ClaimHold struct {
	RetryAfterSeconds int    `json:"retry_after_seconds"`
	Hold              string `json:"hold"`
}

// RunDBRows is the node's count of journal rows staged so far, carried on
// progress for operator visibility only; the coordinator counts its own after
// validation (D55).
type RunDBRows struct {
	Checkpoints int `json:"checkpoints"`
	Evals       int `json:"evals"`
}

// ProgressRequest is POST /nashnet/leases/{lease}/progress every 30s (D5): one
// call is the lease heartbeat and the liveness mirror the coordinator projects
// into process.json. Log lines never ride this body (D54).
//
// The node epoch is not a field here: a lease route carries the lease token and
// nothing else (D26), so the coordinator resolves the node from the lease
// record and fences (lease_id, lease_epoch, node_epoch) against the record and
// the registry rather than against a node-supplied number.
type ProgressRequest struct {
	LeaseEpoch     int64           `json:"lease_epoch"`
	Phase          string          `json:"phase"`
	PID            int             `json:"pid,omitempty"`
	StartedAt      string          `json:"started_at,omitempty"`
	ManifestSeq    int64           `json:"manifest_seq,omitempty"`
	ManifestDigest string          `json:"manifest_digest,omitempty"`
	LogOffset      int64           `json:"log_offset,omitempty"`
	BytesUploaded  int64           `json:"bytes_uploaded,omitempty"`
	RunDBRows      RunDBRows       `json:"rundb_rows"`
	GateReport     json.RawMessage `json:"gate_report,omitempty"`
}

// ProgressResponse answers a progress post (D5). RetryAfterSeconds spells out
// the brief's retry_after in the same units the claim hold uses, so the two
// backoff fields of section 1 read alike.
type ProgressResponse struct {
	Revoke            bool   `json:"revoke,omitempty"`
	Force             bool   `json:"force,omitempty"`
	Drain             bool   `json:"drain,omitempty"`
	LeaseDeadline     string `json:"lease_deadline,omitempty"`
	RetryAfterSeconds int    `json:"retry_after_seconds,omitempty"`
}

// NackRequest is POST /nashnet/leases/{lease}/nack (D8): the node returns a
// claim it cannot honor. The job goes back to the ready set at its original
// position with no attempt increment.
type NackRequest struct {
	LeaseEpoch      int64  `json:"lease_epoch"`
	Reason          string `json:"reason"`
	CooldownSeconds int    `json:"cooldown_seconds,omitempty"`
	Detail          string `json:"detail,omitempty"`
}

// ResultRequest is POST /nashnet/leases/{lease}/result, the commit point (D6).
// It is accepted only once the node has committed a final manifest whose digest
// FinalManifestDigest carries and whose journal was promoted.
type ResultRequest struct {
	LeaseEpoch          int64  `json:"lease_epoch"`
	State               string `json:"state"`
	ExitCode            *int   `json:"exit_code,omitempty"`
	LastError           string `json:"last_error,omitempty"`
	FinishedAt          string `json:"finished_at,omitempty"`
	FinalManifestDigest string `json:"final_manifest_digest,omitempty"`
	Attempt             int    `json:"attempt,omitempty"`
}

// ResultResponse echoes the recorded terminal. A result is idempotent by
// (job_id, lease_id, lease_epoch), so a replay after a lost response returns
// the same body rather than a conflict (D6).
type ResultResponse struct {
	JobID      string `json:"job_id"`
	LeaseID    string `json:"lease_id"`
	LeaseEpoch int64  `json:"lease_epoch"`
	State      string `json:"state"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	RecordedAt string `json:"recorded_at"`
}

// DrainRequest is POST /nashnet/nodes/{node}/drain (operator, the endpoint
// table of section 1). It is two-way: the body carries the state to set, so
// lifting a hold costs no second route and does not require restarting an agent
// holding live leases. ClearBreaker also resets the D63 breaker counter and the
// per-job degraded marks of D8.
type DrainRequest struct {
	Drain        bool `json:"drain"`
	ClearBreaker bool `json:"clear_breaker,omitempty"`
}

// ErrorBody is the error shape of every nashnet route, matching the harness
// convention of {"error": code, "detail": text}. Offset carries the true offset
// on offset_mismatch (D54) and RetryAfterSeconds accompanies rate_limited and
// store_full.
type ErrorBody struct {
	Error             string `json:"error"`
	Detail            string `json:"detail,omitempty"`
	RetryAfterSeconds int    `json:"retry_after_seconds,omitempty"`
	Offset            int64  `json:"offset,omitempty"`
}
