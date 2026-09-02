package harness

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/capability"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// fakeBundles stands in for ingest.Manager on the snapshot path. Its Create
// hook is what asserts D48's invariant: it tries the dispatcher lock and fails
// the test when it is held, so a git fork under the placement lock is a test
// failure rather than a latent stall.
type fakeBundles struct {
	mu     sync.Mutex
	dir    string
	calls  int
	basis  [][]string
	onCall func(basis []string) error
}

func newFakeBundles(t *testing.T) *fakeBundles {
	t.Helper()
	dir := t.TempDir()
	body := []byte("PACK bundle bytes for the pinned commit\n")
	if err := os.WriteFile(filepath.Join(dir, "job.bundle"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	return &fakeBundles{dir: dir}
}

func (f *fakeBundles) BundleCreate(ctx context.Context, jobID string, basis []string) (ingest.BundleDescriptor, error) {
	f.mu.Lock()
	f.calls++
	f.basis = append(f.basis, basis)
	hook := f.onCall
	f.mu.Unlock()
	if hook != nil {
		if err := hook(basis); err != nil {
			return ingest.BundleDescriptor{}, err
		}
	}
	path := filepath.Join(f.dir, "job.bundle")
	data, err := os.ReadFile(path)
	if err != nil {
		return ingest.BundleDescriptor{}, err
	}
	sum := sha256.Sum256(data)
	return ingest.BundleDescriptor{
		Path: path, Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:]),
	}, nil
}

func (f *fakeBundles) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// poolRig is a rig plus an attached coordinator pool: a grant directory holding
// the two fixture nodes' enrollment grants, a lease store, a node registry, a
// quarantine store over temp dirs, and a fake bundle builder.
type poolRig struct {
	*testRig
	pool     *Pool
	bundles  *fakeBundles
	grantDir string
	quarDir  string
	cfg      poolRigConfig
	nodeA    fixtureNode
	nodeB    fixtureNode
	opPriv   ed25519.PrivateKey
	clock    *testClock
}

// fixtureNode is one enrolled node: node-a and node-b are the only fixture
// names, and each is described by its grant and its declaration alone.
type fixtureNode struct {
	name string
	id   string
	priv ed25519.PrivateKey
}

// testClock is the injected pool clock; no test here reads the wall clock for
// anything the grace periods depend on.
type testClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *testClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

type poolRigConfig struct {
	rig             rigConfig
	maxClaimWaiters int
	maxLeases       int
	grace           time.Duration
}

func newPoolRig(t *testing.T, cfg poolRigConfig) *poolRig {
	t.Helper()
	opPub, opPriv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	grantDir := t.TempDir()
	quarDir := t.TempDir()
	clock := &testClock{t: time.Date(2026, time.September, 1, 12, 0, 0, 0, time.UTC)}
	bundles := newFakeBundles(t)

	var pool *Pool
	cfg.rig.attach = func(srv *Server, disp *Dispatcher) {
		grants, gerr := authtoken.NewGrantStore(authtoken.GrantStoreConfig{
			Dir: grantDir, OperatorKey: opPub, Now: clock.now,
		})
		if gerr != nil {
			t.Fatal(gerr)
		}
		leases, lerr := nashnet.NewLeaseStore(nashnet.StoreConfig{
			RunsDir: srv.runsDir, Policy: nashnet.DefaultPolicy(), Now: clock.now,
		})
		if lerr != nil {
			t.Fatal(lerr)
		}
		limits := quarantine.DefaultLimits()
		limits.MinFreeDiskGB = 0.0001
		limits.WatermarkMarginGB = 0.0001
		quar, qerr := quarantine.New(quarantine.Config{
			QuarantineDir: quarDir, RunsDir: srv.runsDir, Limits: limits, Now: clock.now,
		})
		if qerr != nil {
			t.Fatal(qerr)
		}
		p, perr := NewPool(PoolConfig{
			Dispatcher:       disp,
			Grants:           grants,
			Leases:           leases,
			Registry:         nashnet.NewNodeRegistry(nashnet.RegistryConfig{Now: clock.now}),
			Quarantine:       quar,
			Bundles:          bundles,
			RunsDir:          srv.runsDir,
			NodesDir:         grantDir,
			OriginHost:       "coordinator.test",
			Policy:           nashnet.DefaultPolicy(),
			Ceilings:         Ceilings{MaxClaimWaiters: cfg.maxClaimWaiters},
			MaxLeasesPerNode: cfg.maxLeases,
			UnplaceableGrace: cfg.grace,
			Now:              clock.now,
		})
		if perr != nil {
			t.Fatal(perr)
		}
		pool = p
		srv.AttachPool(p)
	}
	base := newRig(t, cfg.rig)

	pr := &poolRig{
		testRig: base, pool: pool, bundles: bundles, grantDir: grantDir,
		quarDir: quarDir, cfg: cfg, opPriv: opPriv, clock: clock,
	}
	pr.nodeA = pr.enroll(t, "node-a", capability.Grant{})
	pr.nodeB = pr.enroll(t, "node-b", capability.Grant{})
	return pr
}

// enroll writes an operator-signed enrollment grant for a fresh node keypair,
// which is the only way a node becomes known: the coordinator holds no signing
// key and cannot author trust.
func (r *poolRig) enroll(t *testing.T, name string, caps capability.Grant) fixtureNode {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	id, err := authtoken.DeriveNodeID(pub)
	if err != nil {
		t.Fatal(err)
	}
	capsJSON, err := json.Marshal(caps)
	if err != nil {
		t.Fatal(err)
	}
	var capsMap map[string]any
	if err := json.Unmarshal(capsJSON, &capsMap); err != nil {
		t.Fatal(err)
	}
	now := r.clock.now()
	claims := jwt.MapClaims{
		"aud":         authtoken.AudienceEnroll,
		"sub":         authtoken.SubjectForNode(id),
		"node_pubkey": base64.RawURLEncoding.EncodeToString(pub),
		"caps":        capsMap,
		"iat":         now.Unix(),
		"exp":         now.Add(30 * 24 * time.Hour).Unix(),
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	signed, err := tok.SignedString(r.opPriv)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(r.grantDir, id+authtoken.GrantFileSuffix), []byte(signed), 0o600); err != nil {
		t.Fatal(err)
	}
	return fixtureNode{name: name, id: id, priv: priv}
}

// nodeToken mints a short node call token, the shape a node agent mints per
// call: aud=nashnet-node, sub=node:<id>, exp within the 300s ceiling.
func (r *poolRig) nodeToken(t *testing.T, n fixtureNode) string {
	t.Helper()
	now := r.clock.now()
	claims := jwt.MapClaims{
		"aud": authtoken.AudienceNode,
		"sub": authtoken.SubjectForNode(n.id),
		"iat": now.Unix(),
		"exp": now.Add(60 * time.Second).Unix(),
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	signed, err := tok.SignedString(n.priv)
	if err != nil {
		t.Fatal(err)
	}
	return signed
}

// doNode issues a request with a node credential.
func (r *poolRig) doNode(t *testing.T, n fixtureNode, method, path string, body any) *http.Response {
	t.Helper()
	return r.raw(t, method, path, body, map[string]string{
		"Authorization": "Bearer " + r.nodeToken(t, n),
	})
}

// doLease issues a request with a lease credential and nothing else.
func (r *poolRig) doLease(t *testing.T, token, method, path string, body any) *http.Response {
	t.Helper()
	return r.raw(t, method, path, body, map[string]string{nashnet.HeaderLeaseToken: token})
}

// raw issues a request with explicit headers, so a test can present the wrong
// credential on purpose.
func (r *poolRig) raw(t *testing.T, method, path string, body any, headers map[string]string) *http.Response {
	t.Helper()
	var rdr io.Reader
	switch b := body.(type) {
	case nil:
	case []byte:
		rdr = bytes.NewReader(b)
	default:
		data, err := json.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		rdr = bytes.NewReader(data)
	}
	req, err := http.NewRequest(method, r.baseURL+path, rdr)
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// decodeInto decodes a JSON response body and closes it.
func decodeInto(t *testing.T, resp *http.Response, v any) {
	t.Helper()
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatalf("decode %s: %v", resp.Request.URL.Path, err)
	}
}

// errorCode reads the {error} code from a refusal body.
func errorCode(t *testing.T, resp *http.Response) string {
	t.Helper()
	var body nashnet.ErrorBody
	decodeInto(t, resp, &body)
	return body.Error
}

// declaration is the fixture declaration both nodes send: one cpu device with
// enough cores and RAM for the default requires block, plus a working
// libcambia build.
func declaration(slots int, devices ...capability.Device) json.RawMessage {
	if len(devices) == 0 {
		devices = []capability.Device{cpuDevice(32, 62)}
	}
	disk := 900.0
	decl := capability.Declaration{
		Schema: 1, AgentVersion: "1.1.0", PlatformTag: "linux-x86_64", Slots: slots,
		Kinds:       []string{KindTrain, KindEvaluate, KindMeasure, KindHeadToHead, KindBench},
		Devices:     devices,
		DiskTotalGB: &disk,
		Toolchain: capability.Toolchain{
			Go: "1.26.0", UV: "0.9.4", Python: "3.13", Git: "2.47",
		},
		CanBuildLibcambia: true,
	}
	data, _ := json.Marshal(decl)
	return data
}

func cpuDevice(cores int, ramGB float64) capability.Device {
	return capability.Device{ID: "cpu", Kind: "cpu", Cores: &cores, RAMTotalGB: &ramGB}
}

func cudaDevice(id string, vramGB float64) capability.Device {
	return capability.Device{ID: id, Kind: "cuda", VRAMTotalGB: &vramGB}
}

// admitReport is a gate report that admits, the shape a node with every gate
// passing sends.
func admitReport(slots int, devicesAllowed ...string) json.RawMessage {
	rep := gates.Report{
		Admit: true, SlotsOffered: slots, EvaluatedAt: time.Now().UTC(),
		Checks: []gates.Check{{Gate: "drain", OK: true}}, DevicesAllowed: devicesAllowed,
	}
	data, _ := json.Marshal(rep)
	return data
}

// denyReport is a gate report that refuses admission with a reopen time, the
// window an operator reads as waiting_for_node_gate.
func denyReport(next time.Time) json.RawMessage {
	rep := gates.Report{
		Admit: false, EvaluatedAt: time.Now().UTC(), NextEligibleAt: &next,
		Checks: []gates.Check{{Gate: "windows", OK: false, NextEligibleAt: &next,
			Detail: "outside every configured window"}},
	}
	data, _ := json.Marshal(rep)
	return data
}

// register declares a node and returns its epoch.
func (r *poolRig) register(t *testing.T, n fixtureNode, slots int) int64 {
	t.Helper()
	resp := r.doNode(t, n, http.MethodPost, "/nashnet/nodes/register", nashnet.RegisterRequest{
		AgentVersion: "1.1.0", Slots: slots, Capabilities: declaration(slots),
		GateReport: admitReport(slots),
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("register %s: got %d, want 200", n.name, resp.StatusCode)
	}
	var out nashnet.RegisterResponse
	decodeInto(t, resp, &out)
	return out.NodeEpoch
}

// claim issues one non-blocking claim and returns the response plus the parsed
// body when a job was handed out.
func (r *poolRig) claim(t *testing.T, n fixtureNode, req nashnet.ClaimRequest) (*http.Response, *nashnet.ClaimResponse) {
	t.Helper()
	if req.Capabilities == nil {
		req.Capabilities = declaration(2)
	}
	if req.GateReport == nil {
		req.GateReport = admitReport(2)
	}
	if req.SlotsFree == 0 {
		req.SlotsFree = 2
	}
	resp := r.doNode(t, n, http.MethodPost, "/nashnet/claim", req)
	if resp.StatusCode != http.StatusOK {
		return resp, nil
	}
	var out nashnet.ClaimResponse
	decodeInto(t, resp, &out)
	return resp, &out
}

// queueJob admits a job through the dispatcher without letting the local
// executor take it, which is what the pool's placement scan then sees. It
// writes the spec and process.json exactly as Submit does.
func (r *poolRig) queueJob(t *testing.T, spec JobSpec) {
	t.Helper()
	if spec.Kind == "" {
		spec.Kind = KindTrain
	}
	if spec.Commit == "" {
		spec.Commit = "0123456789abcdef0123456789abcdef01234567"
	}
	if _, err := r.disp.Submit(spec); err != nil {
		t.Fatalf("submit %s: %v", spec.Name, err)
	}
}

// holdReason reads a 204 claim's hold, which rides a header because a 204 body
// is not deliverable.
func holdReason(resp *http.Response) string {
	defer resp.Body.Close()
	return resp.Header.Get(HeaderClaimHold)
}

// readProcessState is the projection assertion helper.
func readProcessState(t *testing.T, runsDir, job string) *procmgr.ProcessState {
	t.Helper()
	st, err := procmgr.ReadProcessState(filepath.Join(runsDir, job))
	if err != nil {
		t.Fatalf("read process.json for %s: %v", job, err)
	}
	return st
}

// uploadBlob PATCHes one whole file as a single chunk and returns its digest.
func (r *poolRig) uploadBlob(t *testing.T, token, leaseID string, content []byte) string {
	t.Helper()
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])
	path := fmt.Sprintf("/nashnet/leases/%s/blobs/%s", leaseID, digest)
	req, err := http.NewRequest(http.MethodPatch, r.baseURL+path, bytes.NewReader(content))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set(nashnet.HeaderLeaseToken, token)
	req.Header.Set("Content-Range", fmt.Sprintf("bytes 0-%d/%d", len(content)-1, len(content)))
	resp, err := r.client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("upload blob: got %d, want 200 (%s)", resp.StatusCode, body)
	}
	return digest
}

// waiterCount and sessionCount are the two pieces of pool bookkeeping the
// long-poll tests must observe to avoid racing a held request.
func (p *Pool) waiterCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.waiters)
}

func (p *Pool) sessionCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.sessions)
}
