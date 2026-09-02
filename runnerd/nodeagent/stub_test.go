package nodeagent

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// stubCoordinator is an in-memory coordinator speaking the wire types of
// design section 1. It exists so the node agent's whole claim-to-result cycle
// runs offline: no network beyond loopback TLS, no GPU, no git, no ssh.
type stubCoordinator struct {
	t   *testing.T
	srv *httptest.Server

	mu sync.Mutex

	// requests counts every handled request, which is what the certificate-pin
	// test asserts stays at zero.
	requests int

	registers  []nashnet.RegisterRequest
	claims     []nashnet.ClaimRequest
	progress   []nashnet.ProgressRequest
	nacks      []nashnet.NackRequest
	results    []nashnet.ResultRequest
	commits    []quarantine.CommitRequest
	logs       []byte
	phasesSeen []string

	// queue holds the claims to hand out, in order.
	queue []nashnet.ClaimResponse
	// events is the pending events queue for the held poll.
	events []nashnet.Event
	// snapshot is the bundle body served on the snapshot route.
	snapshot []byte
	// seeds is keyed "<seed_id>/<path>".
	seeds map[string][]byte

	token string
	// leaseEpoch is the epoch the stub granted alongside token. Every manifest
	// commit must carry it, because the real coordinator fences a commit on
	// (lease_id, lease_epoch, node_epoch) and refuses any other value with
	// 409 lease_superseded. A stub that ignored the field let this package
	// pass commits the coordinator rejects, which is exactly what shipped
	// (cambia-1726).
	leaseEpoch int64

	blobs map[string][]byte
	parts map[string][]byte
	// patchStarts records the Content-Range start of every chunk append per
	// digest, so a test can assert an upload resumed at an offset instead of
	// restarting from zero.
	patchStarts map[string][]int64
	// headProbes counts HEAD requests per digest, the D50 resume probe.
	headProbes map[string]int

	head      ManifestHead
	nodeEpoch int64
	policy    nashnet.Policy

	// failProgress, when non-zero, is the status the progress route answers
	// with instead of renewing, plus the error code it carries.
	failProgress     int
	failProgressCode string
	// revokeOnProgress makes the next progress response carry revoke:true.
	revokeOnProgress bool
	// holdOnProgress makes progress responses carry that hold reason.
	holdOnProgress string
}

func newStubCoordinator(t *testing.T) *stubCoordinator {
	t.Helper()
	s := &stubCoordinator{
		t:           t,
		seeds:       map[string][]byte{},
		blobs:       map[string][]byte{},
		parts:       map[string][]byte{},
		patchStarts: map[string][]int64{},
		headProbes:  map[string]int{},
		nodeEpoch:   3,
		policy:      fastPolicy(),
		snapshot:    []byte("stub bundle bytes"),
	}
	s.srv = httptest.NewTLSServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.srv.Close)
	return s
}

// fastPolicy is the pool policy the stub hands out: seconds rather than the
// production minutes, so a test drives several ticks in well under a second.
func fastPolicy() nashnet.Policy {
	p := nashnet.DefaultPolicy()
	p.LeaseTTLSeconds = 1
	p.ProgressIntervalSeconds = 1
	p.ChunkBytes = 64
	return p
}

// fingerprint is the sha256 of the stub's leaf certificate in DER form, the
// value a node pins.
func (s *stubCoordinator) fingerprint() string {
	sum := sha256.Sum256(s.srv.Certificate().Raw)
	return hex.EncodeToString(sum[:])
}

func (s *stubCoordinator) url() string { return s.srv.URL }

// enqueue adds one claim handout and records its lease token.
func (s *stubCoordinator) enqueue(resp nashnet.ClaimResponse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if resp.LeaseToken == "" {
		resp.LeaseToken = "lease-token-" + resp.LeaseID
	}
	if resp.Policy.LeaseTTLSeconds == 0 {
		resp.Policy = s.policy
	}
	if resp.LeaseEpoch == 0 {
		resp.LeaseEpoch = 1
	}
	s.token = resp.LeaseToken
	s.leaseEpoch = resp.LeaseEpoch
	s.queue = append(s.queue, resp)
}

func (s *stubCoordinator) pushEvent(ev nashnet.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
}

func (s *stubCoordinator) snapshotDigest() string {
	sum := sha256.Sum256(s.snapshot)
	return hex.EncodeToString(sum[:])
}

func (s *stubCoordinator) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	s.requests++
	s.mu.Unlock()

	path := r.URL.Path
	switch {
	case path == "/nashnet/nodes/register":
		s.handleRegister(w, r)
	case strings.HasSuffix(path, "/heartbeat"):
		writeJSON(w, http.StatusOK, nashnet.HeartbeatResponse{NodeEpoch: s.epoch(), ServerTime: nowText()})
	case strings.HasSuffix(path, "/events"):
		s.handleEvents(w)
	case path == "/nashnet/claim":
		s.handleClaim(w, r)
	case strings.HasPrefix(path, "/nashnet/leases/"):
		s.handleLease(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (s *stubCoordinator) handleRegister(w http.ResponseWriter, r *http.Request) {
	var req nashnet.RegisterRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	s.mu.Lock()
	s.registers = append(s.registers, req)
	epoch := s.nodeEpoch
	policy := s.policy
	s.mu.Unlock()

	rebound := make([]string, 0, len(req.LiveLeases))
	for _, l := range req.LiveLeases {
		rebound = append(rebound, l.LeaseID)
	}
	writeJSON(w, http.StatusOK, nashnet.RegisterResponse{
		NodeID:        req.NodeID,
		NodeEpoch:     epoch,
		Policy:        policy,
		ReboundLeases: rebound,
		ServerTime:    nowText(),
	})
}

func (s *stubCoordinator) handleEvents(w http.ResponseWriter) {
	s.mu.Lock()
	events := s.events
	s.events = nil
	epoch := s.nodeEpoch
	s.mu.Unlock()
	if events == nil {
		events = []nashnet.Event{}
	}
	writeJSON(w, http.StatusOK, nashnet.EventsResponse{Events: events, NodeEpoch: epoch, ServerTime: nowText()})
}

// claimCount is how many claims the node has sent, which is what a hold is
// supposed to stop.
func (s *stubCoordinator) claimCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.claims)
}

func (s *stubCoordinator) handleClaim(w http.ResponseWriter, r *http.Request) {
	var req nashnet.ClaimRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	s.mu.Lock()
	s.claims = append(s.claims, req)
	if len(s.queue) == 0 {
		s.mu.Unlock()
		writeJSON(w, http.StatusNoContent, nashnet.ClaimHold{RetryAfterSeconds: 1, Hold: nashnet.HoldNoMatch})
		return
	}
	next := s.queue[0]
	s.queue = s.queue[1:]
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, next)
}

// handleLease routes every lease-credentialed request after checking the lease
// token, which is the only credential a lease route accepts (D44).
func (s *stubCoordinator) handleLease(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/nashnet/leases/")
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}
	tail := parts[1]

	s.mu.Lock()
	want := s.token
	s.mu.Unlock()
	if got := r.Header.Get(nashnet.HeaderLeaseToken); want != "" && got != want {
		writeJSON(w, http.StatusUnauthorized, nashnet.ErrorBody{Error: "unauthorized"})
		return
	}

	switch {
	case tail == "snapshot":
		s.serveBytes(w, r, s.snapshot)
	case strings.HasPrefix(tail, "seeds/"):
		s.mu.Lock()
		body, ok := s.seeds[strings.TrimPrefix(tail, "seeds/")]
		s.mu.Unlock()
		if !ok {
			writeJSON(w, http.StatusNotFound, nashnet.ErrorBody{Error: "not_found"})
			return
		}
		s.serveBytes(w, r, body)
	case tail == "progress":
		s.handleProgress(w, r)
	case tail == "logs":
		s.handleLogs(w, r)
	case tail == "blobs/probe":
		s.handleProbe(w, r)
	case strings.HasPrefix(tail, "blobs/"):
		s.handleBlob(w, r, strings.TrimPrefix(tail, "blobs/"))
	case tail == "manifest":
		s.handleManifest(w, r)
	case tail == "nack":
		var req nashnet.NackRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		s.mu.Lock()
		s.nacks = append(s.nacks, req)
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"status": "returned"})
	case tail == "result":
		var req nashnet.ResultRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		s.mu.Lock()
		s.results = append(s.results, req)
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, nashnet.ResultResponse{State: req.State, RecordedAt: nowText()})
	default:
		http.NotFound(w, r)
	}
}

func (s *stubCoordinator) handleProgress(w http.ResponseWriter, r *http.Request) {
	var req nashnet.ProgressRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	s.mu.Lock()
	s.progress = append(s.progress, req)
	s.phasesSeen = append(s.phasesSeen, req.Phase)
	status, code := s.failProgress, s.failProgressCode
	revoke, hold := s.revokeOnProgress, s.holdOnProgress
	s.revokeOnProgress = false
	s.mu.Unlock()

	if status != 0 {
		writeJSON(w, status, nashnet.ErrorBody{Error: code})
		return
	}
	writeJSON(w, http.StatusOK, nashnet.ProgressResponse{
		Revoke:        revoke,
		Hold:          hold,
		LeaseDeadline: time.Now().Add(time.Minute).UTC().Format(time.RFC3339Nano),
	})
}

func (s *stubCoordinator) handleLogs(w http.ResponseWriter, r *http.Request) {
	offset, _ := strconv.ParseInt(r.URL.Query().Get("offset"), 10, 64)
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	if offset != int64(len(s.logs)) {
		writeJSON(w, http.StatusConflict, nashnet.ErrorBody{Error: nashnet.CodeOffsetMismatch, Offset: int64(len(s.logs))})
		return
	}
	s.logs = append(s.logs, body...)
	writeJSON(w, http.StatusOK, map[string]int64{"offset": int64(len(s.logs))})
}

func (s *stubCoordinator) handleProbe(w http.ResponseWriter, r *http.Request) {
	var req ProbeRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	resp := ProbeResponse{Have: []string{}, Want: []string{}}
	s.mu.Lock()
	for _, d := range req.Digests {
		if _, ok := s.blobs[d]; ok {
			resp.Have = append(resp.Have, d)
		} else {
			resp.Want = append(resp.Want, d)
		}
	}
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, resp)
}

func (s *stubCoordinator) handleBlob(w http.ResponseWriter, r *http.Request, digest string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch r.Method {
	case http.MethodHead:
		s.headProbes[digest]++
		if b, ok := s.blobs[digest]; ok {
			w.Header().Set(HeaderBlobOffset, strconv.Itoa(len(b)))
		} else {
			w.Header().Set(HeaderBlobOffset, strconv.Itoa(len(s.parts[digest])))
		}
		w.WriteHeader(http.StatusOK)
	case http.MethodPatch:
		start, _, total, ok := parseContentRange(r.Header.Get("Content-Range"))
		if !ok {
			writeJSON(w, http.StatusBadRequest, nashnet.ErrorBody{Error: "invalid_range"})
			return
		}
		if total == 0 {
			s.blobs[digest] = []byte{}
			writeJSON(w, http.StatusOK, ChunkResponse{CommittedOffset: 0})
			return
		}
		have := int64(len(s.parts[digest]))
		s.patchStarts[digest] = append(s.patchStarts[digest], start)
		if start != have {
			writeJSON(w, http.StatusConflict, nashnet.ErrorBody{Error: nashnet.CodeOffsetMismatch, Offset: have})
			return
		}
		chunk, _ := io.ReadAll(r.Body)
		s.parts[digest] = append(s.parts[digest], chunk...)
		if int64(len(s.parts[digest])) == total {
			sum := sha256.Sum256(s.parts[digest])
			if hex.EncodeToString(sum[:]) != digest {
				delete(s.parts, digest)
				writeJSON(w, http.StatusUnprocessableEntity, nashnet.ErrorBody{Error: nashnet.CodeHashMismatch})
				return
			}
			s.blobs[digest] = s.parts[digest]
			delete(s.parts, digest)
		}
		writeJSON(w, http.StatusOK, ChunkResponse{CommittedOffset: int64(len(s.parts[digest])) + int64(len(s.blobs[digest]))})
	case http.MethodDelete:
		delete(s.parts, digest)
		w.WriteHeader(http.StatusNoContent)
	default:
		http.Error(w, "method", http.StatusMethodNotAllowed)
	}
}

func (s *stubCoordinator) handleManifest(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		s.mu.Lock()
		head := s.head
		s.mu.Unlock()
		if head.Entries == nil {
			head.Entries = []quarantine.Entry{}
		}
		writeJSON(w, http.StatusOK, head)
		return
	}
	var req quarantine.CommitRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.leaseEpoch != 0 && req.LeaseEpoch != s.leaseEpoch {
		writeJSON(w, http.StatusConflict, nashnet.ErrorBody{
			Error:  nashnet.CodeLeaseSuperseded,
			Detail: fmt.Sprintf("commit carried lease epoch %d, lease is at %d", req.LeaseEpoch, s.leaseEpoch),
		})
		return
	}
	if req.Seq != s.head.Seq+1 || req.Parent != s.head.Digest {
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": nashnet.CodeManifestOutOfOrder, "seq": s.head.Seq, "digest": s.head.Digest,
		})
		return
	}
	var missing []string
	for _, e := range req.Entries {
		if _, ok := s.blobs[e.Digest]; !ok {
			missing = append(missing, e.Digest)
		}
	}
	if len(missing) > 0 {
		writeJSON(w, http.StatusConflict, map[string]any{"error": nashnet.CodeBlobsMissing, "missing": missing})
		return
	}
	s.commits = append(s.commits, req)
	byPath := map[string]quarantine.Entry{}
	for _, e := range s.head.Entries {
		byPath[e.Path] = e
	}
	for _, d := range req.Deletes {
		delete(byPath, d)
	}
	promoted := make([]string, 0, len(req.Entries))
	for _, e := range req.Entries {
		byPath[e.Path] = e
		promoted = append(promoted, e.Path)
	}
	entries := make([]quarantine.Entry, 0, len(byPath))
	for _, e := range byPath {
		entries = append(entries, e)
	}
	s.head = ManifestHead{
		Seq:     req.Seq,
		Digest:  fmt.Sprintf("digest-%d", req.Seq),
		Entries: entries,
		Final:   req.Final,
	}
	writeJSON(w, http.StatusOK, quarantine.CommitResponse{
		Seq: s.head.Seq, Digest: s.head.Digest, Promoted: promoted,
		Deleted: req.Deletes, Rejected: []quarantine.Rejection{},
	})
}

// serveBytes answers a ranged GET the way the snapshot and seed routes do.
func (s *stubCoordinator) serveBytes(w http.ResponseWriter, r *http.Request, body []byte) {
	w.Header().Set("Accept-Ranges", "bytes")
	rng := r.Header.Get("Range")
	if rng == "" {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
		return
	}
	var start int64
	// A first byte at or past the end is unsatisfiable, which is what
	// net/http.ServeContent answers on the real routes. The stub admitted
	// start == len and replied 206 with an empty body, so this package could
	// not see the defect the two-process suite caught (cambia-2018).
	if _, err := fmt.Sscanf(rng, "bytes=%d-", &start); err != nil || start >= int64(len(body)) {
		w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", len(body)))
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		return
	}
	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = w.Write(body[start:])
}

// snapshotAt returns a copy of the recorded state a test asserts against.
func (s *stubCoordinator) snapshotState() (progress []nashnet.ProgressRequest, results []nashnet.ResultRequest, nacks []nashnet.NackRequest, commits []quarantine.CommitRequest, phases []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]nashnet.ProgressRequest(nil), s.progress...),
		append([]nashnet.ResultRequest(nil), s.results...),
		append([]nashnet.NackRequest(nil), s.nacks...),
		append([]quarantine.CommitRequest(nil), s.commits...),
		append([]string(nil), s.phasesSeen...)
}

func (s *stubCoordinator) epoch() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.nodeEpoch
}

func (s *stubCoordinator) requestCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests
}

// setLease records the credential pair a hand-built lease record holds, so a
// test driving a jobRun directly fences the same way a claimed lease does.
func (s *stubCoordinator) setLease(token string, epoch int64) {
	s.mu.Lock()
	s.token = token
	s.leaseEpoch = epoch
	s.mu.Unlock()
}

func (s *stubCoordinator) setProgressFailure(status int, code string) {
	s.mu.Lock()
	s.failProgress = status
	s.failProgressCode = code
	s.mu.Unlock()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func nowText() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// parseContentRange reads "bytes start-end/total".
func parseContentRange(v string) (start, end, total int64, ok bool) {
	if strings.HasPrefix(v, "bytes */") {
		t, err := strconv.ParseInt(strings.TrimPrefix(v, "bytes */"), 10, 64)
		return 0, 0, t, err == nil
	}
	if _, err := fmt.Sscanf(v, "bytes %d-%d/%d", &start, &end, &total); err != nil {
		return 0, 0, 0, false
	}
	return start, end, total, true
}

// fakeProber returns a fixed observation and a switchable gate outcome.
type fakeProber struct {
	mu  sync.Mutex
	obs Observation
}

func newFakeProber() *fakeProber {
	ram := 64.0
	disk := 500.0
	load := 0.5
	return &fakeProber{obs: Observation{
		Cores:       8,
		RAMFreeGB:   &ram,
		RAMTotalGB:  &ram,
		DiskFreeGB:  &disk,
		DiskTotalGB: &disk,
		Load1:       &load,
		Devices:     []gates.DeviceSnapshot{{ID: "cpu", Kind: "cpu"}},
	}}
}

func (p *fakeProber) Observe(string) Observation {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.obs
}

func (p *fakeProber) setFreeRAM(v float64) {
	p.mu.Lock()
	p.obs.RAMFreeGB = &v
	p.mu.Unlock()
}

// fakeEnv is the ingest boundary stand-in: it stages nothing and reports a
// prepared environment pointing at a temp worktree.
type fakeEnv struct {
	worktree   string
	runsDir    string
	prepareErr error
	fetchErr   error
	python     string
	// writeEnvJSON mirrors what the real Prepare does: it writes the node's
	// own env.json into the run dir, which the uploader must offer as
	// env.node.json.
	writeEnvJSON bool

	mu       sync.Mutex
	prepared int
	fetched  int
	cleaned  int
}

func (e *fakeEnv) BundleFetch(context.Context, string, string) error {
	e.mu.Lock()
	e.fetched++
	e.mu.Unlock()
	return e.fetchErr
}

func (e *fakeEnv) Prepare(_ context.Context, jobID, _, _, _, _, _ string, _ map[string]string) (*ingestapi.Prepared, error) {
	e.mu.Lock()
	e.prepared++
	e.mu.Unlock()
	if e.prepareErr != nil {
		return nil, e.prepareErr
	}
	runDir := filepath.Join(e.runsDir, jobID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return nil, err
	}
	if e.writeEnvJSON {
		if err := os.WriteFile(filepath.Join(runDir, "env.json"), []byte(`{"origin_host":"node-a"}`), 0o644); err != nil {
			return nil, err
		}
	}
	python := e.python
	if python == "" {
		python = "/bin/sh"
	}
	return &ingestapi.Prepared{
		WorktreeDir: e.worktree,
		RunDir:      runDir,
		VenvPython:  python,
	}, nil
}

func (e *fakeEnv) Cleanup(string, bool) error {
	e.mu.Lock()
	e.cleaned++
	e.mu.Unlock()
	return nil
}

// fakeLauncher records launches and stops without forking anything.
type fakeLauncher struct {
	mu       sync.Mutex
	ensured  []string
	started  []Launch
	stops    []bool
	status   ProcessStatus
	startErr error
}

func (l *fakeLauncher) Ensure(name, _ string) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ensured = append(l.ensured, name)
	return nil
}

func (l *fakeLauncher) Start(_ string, launch Launch) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.startErr != nil {
		return 0, l.startErr
	}
	l.started = append(l.started, launch)
	if l.status.Status == "" {
		l.status = ProcessStatus{Status: "running", PID: 4242, Found: true}
	}
	return l.status.PID, nil
}

func (l *fakeLauncher) Stop(_ string, force bool) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.stops = append(l.stops, force)
	l.status = ProcessStatus{Status: "stopped", PID: l.status.PID, Found: true}
	return nil
}

func (l *fakeLauncher) Status(string) ProcessStatus {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.status
}

func (l *fakeLauncher) setStatus(st ProcessStatus) {
	l.mu.Lock()
	l.status = st
	l.mu.Unlock()
}

func (l *fakeLauncher) stopCount() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.stops)
}

// testAgent builds an agent wired to the stub with everything else faked.
func testAgent(t *testing.T, stub *stubCoordinator, opts func(*Options)) (*Agent, Config) {
	t.Helper()
	base := t.TempDir()
	cfg := Config{
		Coordinator: Coordinator{URL: stub.url(), CertSHA256: stub.fingerprint()},
		BaseDir:     base,
		RunsDir:     filepath.Join(base, "runs"),
		KeyPath:     filepath.Join(base, "node.key"),
	}
	if err := cfg.normalize(); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if err := os.MkdirAll(cfg.RunsDir, 0o755); err != nil {
		t.Fatalf("runs dir: %v", err)
	}
	signer := testSigner(t)
	client, err := NewClient(cfg, signer)
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	o := Options{
		Config:       cfg,
		Signer:       signer,
		Client:       client,
		Env:          &fakeEnv{worktree: t.TempDir(), runsDir: cfg.RunsDir},
		Launcher:     &fakeLauncher{},
		Prober:       newFakeProber(),
		Logger:       log.New(io.Discard, "", 0),
		PollInterval: 10 * time.Millisecond,
	}
	if opts != nil {
		opts(&o)
	}
	agent, err := New(o)
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	return agent, o.Config
}

func testSigner(t *testing.T) *Signer {
	t.Helper()
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	signer, err := NewSigner(priv, nil)
	if err != nil {
		t.Fatalf("signer: %v", err)
	}
	return signer
}

// specJSON renders a node spec as the claim's raw spec payload.
func specJSON(t *testing.T, s Spec) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("marshal spec: %v", err)
	}
	return b
}

// transportCase is one of the two NodeTransport implementations of D65, in the
// shape a test drives them through: the fingerprint-pinned HTTPS client a
// remote node runs, and the in-process loopback the coordinator's own embedded
// node runs. Both reach the same stub coordinator, so a test parameterized
// over this proves one path rather than two similar ones.
type transportCase struct {
	name  string
	apply func(t *testing.T, stub *stubCoordinator, o *Options)
}

// transports returns both legs. The HTTPS leg leaves the client testAgent
// already built; the loopback leg replaces it with one dialing the stub's own
// handler, which is the same function the TLS server serves.
func transports() []transportCase {
	return []transportCase{
		{name: "https", apply: func(*testing.T, *stubCoordinator, *Options) {}},
		{name: "loopback", apply: func(_ *testing.T, stub *stubCoordinator, o *Options) {
			o.Client = NewLoopbackClient(http.HandlerFunc(stub.serve), o.Signer)
		}},
	}
}
