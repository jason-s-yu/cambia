// Package quarantine implements the coordinator half of the nashnet data plane
// (design 4). Everything a node uploads lands in a per-node, per-job, per-lease
// quarantine tree, is verified against the digest that names it, is validated
// against the rejection list, and is promoted into runs/<job>/ by a commit
// transaction the coordinator performs. Nothing a node sends is executed,
// imported, or unpickled here; run_db.sqlite is the only node-authored format
// parsed at all, and it goes through the JournalValidator seam (D55) before any
// of its bytes reach a run dir.
//
// Leaf package: no HTTP and no dispatcher import. Every condition a request can
// provoke is a typed error the route layer maps to a status code, so the store
// decides policy and the route decides presentation.
package quarantine

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"
	"syscall"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/runnerd/sysprobe"
)

// Subdirectory and file names of a lease's quarantine tree (D49):
//
//	quarantine/<node_id>/<job_id>/<lease_id>/
//	  parts/<digest>.part        in-flight uploads; the file's own size is the resume offset
//	  blobs/<digest>             verified content, 0644, named by its sha256
//	  manifests/<seq>-<digest>.json
//	  receipt.jsonl              one line per commit
//	  retired.json               coordinator-written result stamp, drives GC
const (
	dirParts     = "parts"
	dirBlobs     = "blobs"
	dirManifests = "manifests"
	partSuffix   = ".part"
	receiptFile  = "receipt.jsonl"
	retiredFile  = "retired.json"
)

// Names the coordinator owns inside a promoted run dir. NashnetDir holds the
// folded manifest head, TmpDir is the copy-mode staging directory, and
// TmpSuffix names the link-mode staging file. All three are rejected manifest
// paths (D52), so a node can never name one.
const (
	NashnetDir = ".nashnet"
	HeadFile   = "current.json"
	TmpDir     = ".nashnet-tmp"
	TmpSuffix  = ".nashnet-tmp"

	// RunDBPath is the one node-authored format the coordinator parses (D55)
	// and the one entry materialized last in every commit (D57).
	RunDBPath = "run_db.sqlite"
)

// MaterializeMode is how a verified blob becomes a promoted file: a hard link
// when the quarantine and the runs dir share a filesystem, a byte copy when
// they do not (D49).
type MaterializeMode string

// The two materialize modes.
const (
	ModeLink MaterializeMode = "link"
	ModeCopy MaterializeMode = "copy"
)

// Limits are the quota and cap defaults of D56 plus the GC horizons of D59.
// The route layer owns the token buckets, the connection ceilings, and the
// per-request body caps; everything here is enforced inside the store because
// it needs the store's own state (lease usage, disk free, tree age).
type Limits struct {
	// MaxFileBytes caps one uploaded file (RUNNERD_NASHNET_MAX_FILE_BYTES).
	MaxFileBytes int64
	// MaxLeaseBytes caps every byte a lease may hold in quarantine
	// (RUNNERD_NASHNET_MAX_LEASE_BYTES), lowered per lease by Lease.MaxBytes.
	MaxLeaseBytes int64
	// MaxManifestEntries caps entries per manifest and files per lease.
	MaxManifestEntries int
	// MaxManifestBytes caps the manifest request body.
	MaxManifestBytes int64
	// MaxPathBytes, MaxSegmentBytes, and MaxDepth cap a manifest path's shape.
	MaxPathBytes    int
	MaxSegmentBytes int
	MaxDepth        int
	// MaxRejections is the rejection count past which a lease is degraded.
	MaxRejections int
	// MaxRunDBRejections is the consecutive rundb_invalid count past which a
	// lease is degraded (D55).
	MaxRunDBRejections int
	// MinFreeDiskGB is the coordinator's own service floor
	// (RUNNERD_MIN_FREE_DISK_GB); WatermarkMarginGB sits above it so N nodes
	// each inside quota cannot drive the coordinator below its own floor.
	MinFreeDiskGB     float64
	WatermarkMarginGB float64
	// RetryAfter is reported with a store-full refusal.
	RetryAfter time.Duration
	// PartTTL reaps an idle part; DebugTTL retains a retired lease tree and its
	// receipt for post-mortem reading (D59).
	PartTTL  time.Duration
	DebugTTL time.Duration
}

// DefaultLimits returns the D56 and D59 defaults.
func DefaultLimits() Limits {
	return Limits{
		MaxFileBytes:       8 << 30,
		MaxLeaseBytes:      64 << 30,
		MaxManifestEntries: 20000,
		MaxManifestBytes:   4 << 20,
		MaxPathBytes:       1024,
		MaxSegmentBytes:    255,
		MaxDepth:           16,
		MaxRejections:      100,
		MaxRunDBRejections: 3,
		MinFreeDiskGB:      20,
		WatermarkMarginGB:  10,
		RetryAfter:         60 * time.Second,
		PartTTL:            24 * time.Hour,
		DebugTTL:           24 * time.Hour,
	}
}

func (l Limits) withDefaults() Limits {
	d := DefaultLimits()
	if l.MaxFileBytes <= 0 {
		l.MaxFileBytes = d.MaxFileBytes
	}
	if l.MaxLeaseBytes <= 0 {
		l.MaxLeaseBytes = d.MaxLeaseBytes
	}
	if l.MaxManifestEntries <= 0 {
		l.MaxManifestEntries = d.MaxManifestEntries
	}
	if l.MaxManifestBytes <= 0 {
		l.MaxManifestBytes = d.MaxManifestBytes
	}
	if l.MaxPathBytes <= 0 {
		l.MaxPathBytes = d.MaxPathBytes
	}
	if l.MaxSegmentBytes <= 0 {
		l.MaxSegmentBytes = d.MaxSegmentBytes
	}
	if l.MaxDepth <= 0 {
		l.MaxDepth = d.MaxDepth
	}
	if l.MaxRejections <= 0 {
		l.MaxRejections = d.MaxRejections
	}
	if l.MaxRunDBRejections <= 0 {
		l.MaxRunDBRejections = d.MaxRunDBRejections
	}
	if l.WatermarkMarginGB <= 0 {
		l.WatermarkMarginGB = d.WatermarkMarginGB
	}
	if l.RetryAfter <= 0 {
		l.RetryAfter = d.RetryAfter
	}
	if l.PartTTL <= 0 {
		l.PartTTL = d.PartTTL
	}
	if l.DebugTTL <= 0 {
		l.DebugTTL = d.DebugTTL
	}
	return l
}

// Grant is one entry of a lease's claim-time grant set (D53): a file the
// coordinator already holds and the node may name in a manifest without
// uploading a byte. SourcePath is coordinator-resolved and never appears in any
// request, so a node cannot name a coordinator path through this field.
type Grant struct {
	Digest     string
	Size       int64
	SourcePath string
}

// Lease names one lease's quarantine tree and carries the facts the store needs
// to answer for it. NodeID, JobID, and LeaseID come from the authenticated
// lease record and never from a request (D49), so a node cannot address,
// enumerate, or observe another node's or another lease's tree.
type Lease struct {
	NodeID  string
	JobID   string
	LeaseID string
	// Epoch fences a manifest commit against a superseded lease (D51 step 1).
	Epoch int64
	// Grants is the grant set keyed by digest. Together with the lease's own
	// blobs/ it is the provable set of D50: a digest outside it is want,
	// whether or not the coordinator holds it elsewhere.
	Grants map[string]Grant
	// MaxBytes lowers Limits.MaxLeaseBytes for this lease (the grant's
	// caps.max_lease_bytes, D56). Zero means the store default.
	MaxBytes int64
	// RunDBName is the runs.name this lease's journal must carry (D55). Empty
	// means the job id, which is right for every kind but evaluate, where the
	// coordinator resolves spec.target instead (D64).
	RunDBName string
}

// rundbName is the identity the journal validator checks the runs row against.
func (l Lease) rundbName() string {
	if l.RunDBName != "" {
		return l.RunDBName
	}
	return l.JobID
}

func (l Lease) validate() error {
	for _, name := range []string{l.NodeID, l.JobID, l.LeaseID} {
		if err := procmgr.ValidateName(name); err != nil {
			return fmt.Errorf("lease identity: %w", err)
		}
	}
	return nil
}

func (l Lease) key() string {
	return l.NodeID + "/" + l.JobID + "/" + l.LeaseID
}

// digestRe is the one request-derived path component in the protocol: exactly
// 64 lowercase hex characters. Go anchors ^ and $ to the text rather than a
// line, so a trailing newline does not slip through.
var digestRe = regexp.MustCompile(`^[0-9a-f]{64}$`)

// ValidateDigest enforces the blob-digest shape of D49. It is a pure string
// check with no filesystem access, and every entry point that lets a digest
// reach filepath.Join calls it first, so an uppercase, short, long,
// traversal-bearing, separator-bearing, NUL-bearing, or non-ASCII digest never
// names a path. The route layer maps ErrInvalidDigest to 422 invalid_digest.
func ValidateDigest(digest string) error {
	if !digestRe.MatchString(digest) {
		return fmt.Errorf("%w: %q", ErrInvalidDigest, digest)
	}
	return nil
}

// Config configures a Store. Every seam the offline suite of D41 needs to drive
// (the clock, the linker, the disk probe, the journal validator) is injected,
// so no test depends on a real filesystem condition or a real journal.
type Config struct {
	// QuarantineDir is RUNNERD_NASHNET_QUARANTINE_DIR, created 0700.
	QuarantineDir string
	// RunsDir is RUNNERD_RUNS_DIR, the promotion target root.
	RunsDir string
	Limits  Limits
	// Now defaults to time.Now.
	Now func() time.Time
	// Link defaults to os.Link. A test returning syscall.EXDEV drives the copy
	// path without a second filesystem.
	Link func(oldname, newname string) error
	// DiskFreeGB defaults to sysprobe.DiskFreeGB, the moved statfs probe that
	// keeps the semantics of harness.diskFreeGB (server.go:238) and
	// procmgr.DiskSpaceCheck.
	DiskFreeGB func(path string) float64
	// Validator content-validates run_db.sqlite before promotion (D55). Nil
	// selects this package's own Validate under RunDB, so the commit path can
	// never promote an unvalidated journal by omission; a test injects its own
	// to force a verdict.
	Validator JournalValidator
	// RunDB bounds the journal validator: the daemon threads the env-derived
	// RUNNERD_NASHNET_MAX_RUNDB_BYTES cap here. The zero value takes the D55
	// defaults.
	RunDB RunDBConfig
	// MaterializeMode forces link or copy. Empty runs the link() probe of D49
	// at construction: success selects link, a cross-device or unsupported
	// link selects copy.
	MaterializeMode MaterializeMode
	// OnMaterialize, when set, is called with each run-dir-relative path as it
	// is promoted, in promotion order. The D57 ordering test reads it.
	OnMaterialize func(path string)
}

// LeaseStats are the abuse counters a lease accumulates. The lease record (the
// nashnet package) reads them to mark a lease degraded; the store never fails a
// job over one, because a quota or validation event is not a job outcome (D56).
type LeaseStats struct {
	HashMismatches          int
	OffsetMismatches        int
	Rejections              int
	RunDBRejections         int
	ConsecutiveRunDBRejects int
	Degraded                bool
}

// Store owns one coordinator's quarantine tree and the promotion transaction
// into its runs dir.
type Store struct {
	root       string
	runsDir    string
	lim        Limits
	now        func() time.Time
	link       func(oldname, newname string) error
	diskFree   func(string) float64
	validator  JournalValidator
	mode       MaterializeMode
	modeReason string
	onMat      func(string)

	mu       sync.Mutex
	jobLocks map[string]*sync.Mutex
	upLocks  map[string]*sync.Mutex
	parts    map[string]*partHash
	used     map[string]int64
	stats    map[string]*LeaseStats
}

// New creates the quarantine root, selects the materialize mode, and returns a
// store ready to accept uploads.
func New(cfg Config) (*Store, error) {
	if cfg.QuarantineDir == "" {
		return nil, errors.New("quarantine: QuarantineDir is required")
	}
	if cfg.RunsDir == "" {
		return nil, errors.New("quarantine: RunsDir is required")
	}
	if cfg.Validator == nil {
		cfg.Validator = runDBValidator{cfg: cfg.RunDB}
	}
	s := &Store{
		root:      filepath.Clean(cfg.QuarantineDir),
		runsDir:   filepath.Clean(cfg.RunsDir),
		lim:       cfg.Limits.withDefaults(),
		now:       cfg.Now,
		link:      cfg.Link,
		diskFree:  cfg.DiskFreeGB,
		validator: cfg.Validator,
		mode:      cfg.MaterializeMode,
		onMat:     cfg.OnMaterialize,
		jobLocks:  map[string]*sync.Mutex{},
		upLocks:   map[string]*sync.Mutex{},
		parts:     map[string]*partHash{},
		used:      map[string]int64{},
		stats:     map[string]*LeaseStats{},
	}
	if s.now == nil {
		s.now = time.Now
	}
	if s.link == nil {
		s.link = os.Link
	}
	if s.diskFree == nil {
		s.diskFree = sysprobe.DiskFreeGB
	}
	if err := os.MkdirAll(s.root, 0o700); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(s.runsDir, 0o755); err != nil {
		return nil, err
	}
	if s.mode == "" {
		s.mode, s.modeReason = probeMaterializeMode(s.root, s.runsDir, s.link)
	} else {
		s.modeReason = "configured"
	}
	return s, nil
}

// MaterializeMode reports the mode selected at construction, for the health
// route and the startup log (D49).
func (s *Store) MaterializeMode() MaterializeMode { return s.mode }

// MaterializeReason reports why that mode was selected.
func (s *Store) MaterializeReason() string { return s.modeReason }

// Limits reports the effective caps, so the route layer can size its own
// MaxBytesReader from the same numbers.
func (s *Store) Limits() Limits { return s.lim }

// probeMaterializeMode tries one hard link from the quarantine into the runs
// dir. Success selects link; a cross-device link, or any other refusal by the
// filesystem, selects copy, which is correct in every case and costs one byte
// copy per promoted file.
func probeMaterializeMode(quarantineDir, runsDir string, link func(string, string) error) (MaterializeMode, string) {
	src := filepath.Join(quarantineDir, ".nashnet-link-probe")
	dst := filepath.Join(runsDir, ".nashnet-link-probe")
	_ = os.Remove(src)
	_ = os.Remove(dst)
	if err := os.WriteFile(src, []byte("probe"), 0o600); err != nil {
		return ModeCopy, "probe write failed: " + err.Error()
	}
	defer func() { _ = os.Remove(src) }()
	err := link(src, dst)
	_ = os.Remove(dst)
	switch {
	case err == nil:
		return ModeLink, "link probe succeeded"
	case errors.Is(err, syscall.EXDEV):
		return ModeCopy, "quarantine and runs dir are on different filesystems"
	default:
		return ModeCopy, "link probe failed: " + err.Error()
	}
}

// leaseDir validates the lease identity and returns its quarantine tree.
func (s *Store) leaseDir(l Lease) (string, error) {
	if err := l.validate(); err != nil {
		return "", err
	}
	return filepath.Join(s.root, l.NodeID, l.JobID, l.LeaseID), nil
}

// RunDir returns the promotion target for a job.
func (s *Store) RunDir(jobID string) (string, error) {
	if err := procmgr.ValidateName(jobID); err != nil {
		return "", err
	}
	return filepath.Join(s.runsDir, jobID), nil
}

func partPath(leaseDir, digest string) string {
	return filepath.Join(leaseDir, dirParts, digest+partSuffix)
}

func blobPath(leaseDir, digest string) string {
	return filepath.Join(leaseDir, dirBlobs, digest)
}

// ensureLeaseTree creates the lease's quarantine tree 0700. It runs only after
// every pure check has passed.
func ensureLeaseTree(leaseDir string) error {
	for _, d := range []string{dirParts, dirBlobs, dirManifests} {
		if err := os.MkdirAll(filepath.Join(leaseDir, d), 0o700); err != nil {
			return err
		}
	}
	return nil
}

// Stats returns a copy of a lease's abuse counters.
func (s *Store) Stats(l Lease) LeaseStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return *s.statsLocked(l)
}

func (s *Store) statsLocked(l Lease) *LeaseStats {
	k := l.key()
	st := s.stats[k]
	if st == nil {
		st = &LeaseStats{}
		s.stats[k] = st
	}
	return st
}

// markDegradedLocked applies the two degrade rules: more than MaxRejections
// rejections on the lease (D52), or MaxRunDBRejections consecutive journal
// rejections (D55).
func (s *Store) markDegradedLocked(st *LeaseStats) {
	if st.Rejections > s.lim.MaxRejections || st.ConsecutiveRunDBRejects >= s.lim.MaxRunDBRejections {
		st.Degraded = true
	}
}

func (s *Store) jobLock(jobID string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.jobLocks[jobID]
	if m == nil {
		m = &sync.Mutex{}
		s.jobLocks[jobID] = m
	}
	return m
}

func (s *Store) uploadLock(key string) *sync.Mutex {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.upLocks[key]
	if m == nil {
		m = &sync.Mutex{}
		s.upLocks[key] = m
	}
	return m
}

// fsyncDir flushes a directory entry so a rename survives a crash.
func fsyncDir(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
