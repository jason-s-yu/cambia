package harness

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// fakeEnv is the test Environment. Prepare writes a config.yaml into the run dir
// (so the procmgr launch's os.Stat gate passes) and returns a Prepared, unless
// prepareErr is set or prepareBlock is closed-gated. It records Cleanup and
// StartupSweep calls.
type fakeEnv struct {
	runsDir    string
	prepareErr error
	block      chan struct{} // if non-nil, Prepare waits on it (or ctx) before returning

	mu       sync.Mutex
	cleanups []string
	sweeps   [][]string
}

func (f *fakeEnv) Prepare(ctx context.Context, jobID, commit, kind, configRel string, overrides map[string]string) (*ingestapi.Prepared, error) {
	if f.block != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-f.block:
		}
	}
	if f.prepareErr != nil {
		return nil, f.prepareErr
	}
	runDir := filepath.Join(f.runsDir, jobID)
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return nil, err
	}
	cfg := filepath.Join(runDir, "config.yaml")
	if err := os.WriteFile(cfg, []byte("device: cpu\n"), 0o644); err != nil {
		return nil, err
	}
	return &ingestapi.Prepared{RunDir: runDir, RenderedConfig: cfg}, nil
}

func (f *fakeEnv) Cleanup(jobID string, keepForDebug bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cleanups = append(f.cleanups, jobID)
	return nil
}

func (f *fakeEnv) StartupSweep(ids []string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sweeps = append(f.sweeps, ids)
	return nil
}

// fakeAlgos maps the test job kinds to the fake cambia script's mode arg.
func fakeAlgos() map[string][]string {
	return map[string][]string{
		"fake":       {"sleep"},
		"fake-quick": {"quick"},
		"fake-fail":  {"fail"},
	}
}

// writeFakeCambia writes a shell script standing in for the cambia binary. It
// dispatches on its first arg (the injected subcommand): sleep long, exit 0, or
// exit nonzero. It ignores the trailing --config/--run-name/--save-path flags.
func writeFakeCambia(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "fake-cambia.sh")
	script := "#!/bin/sh\ncase \"$1\" in\n  quick) exit 0 ;;\n  fail) exit 3 ;;\n  *) exec sleep 30 ;;\nesac\n"
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

type rigConfig struct {
	maxJobs   int
	maxQueue  int
	poll      time.Duration
	env       Environment
	algos     map[string][]string
	origin    string
	minRAMGB  float64
	minDiskGB float64
	ramQuery  RAMQueryFunc
	gpuQuery  procmgr.GPUQueryFunc
}

type testRig struct {
	t       *testing.T
	ts      *httptest.Server
	client  *http.Client
	baseURL string
	token   string
	priv    ed25519.PrivateKey
	runsDir string
	disp    *Dispatcher
	pm      *procmgr.ProcessManager
	env     *fakeEnv
	origin  string
}

func newRig(t *testing.T, cfg rigConfig) *testRig {
	t.Helper()
	runsDir := t.TempDir()
	cfrDir := t.TempDir()
	script := writeFakeCambia(t)

	if cfg.maxJobs == 0 {
		cfg.maxJobs = 1
	}
	if cfg.maxQueue == 0 {
		cfg.maxQueue = 16
	}
	if cfg.poll == 0 {
		cfg.poll = 15 * time.Millisecond
	}
	if cfg.algos == nil {
		cfg.algos = fakeAlgos()
	}
	if cfg.origin == "" {
		cfg.origin = "https://client.lan"
	}
	if cfg.minRAMGB == 0 {
		cfg.minRAMGB = 1
	}
	if cfg.minDiskGB == 0 {
		cfg.minDiskGB = 0.001
	}
	if cfg.ramQuery == nil {
		cfg.ramQuery = func() (float64, error) { return 999, nil }
	}
	if cfg.gpuQuery == nil {
		cfg.gpuQuery = procmgr.DefaultGPUQuery
	}

	fe, _ := cfg.env.(*fakeEnv)
	if cfg.env == nil {
		fe = &fakeEnv{runsDir: runsDir}
		cfg.env = fe
	}

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	verifier := authtoken.NewVerifier(pub)

	pm := procmgr.NewProcessManager(runsDir, cfrDir, script, NewRunResolver(runsDir), cfg.algos)
	pm.SetMaxConcurrent(cfg.maxJobs)
	disp := NewDispatcher(pm, cfg.env, runsDir, cfg.maxJobs, cfg.maxQueue, cfg.poll)

	srv, err := NewServer(ServerConfig{
		Dispatcher:    disp,
		Verifier:      verifier,
		RunsDir:       runsDir,
		AllowedOrigin: cfg.origin,
		MinFreeRAMGB:  cfg.minRAMGB,
		MinFreeDiskGB: cfg.minDiskGB,
		Algos:         cfg.algos,
		GPUQuery:      cfg.gpuQuery,
		RAMQuery:      cfg.ramQuery,
	})
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewTLSServer(srv.Handler())
	t.Cleanup(func() {
		pm.KillAll()
		// Wait for supervised processes and their monitor goroutines to settle
		// (final process.json writes done) before TempDir removal, so a late
		// write does not race RemoveAll into a "directory not empty" error.
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			disp.mu.Lock()
			a := disp.active
			disp.mu.Unlock()
			if a == 0 {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		ts.Close()
	})

	tok := mintToken(t, priv, "client-cli", 0)
	return &testRig{
		t:       t,
		ts:      ts,
		client:  ts.Client(),
		baseURL: ts.URL,
		token:   tok,
		priv:    priv,
		runsDir: runsDir,
		disp:    disp,
		pm:      pm,
		env:     fe,
		origin:  cfg.origin,
	}
}

// genOther returns a fresh ed25519 keypair for negative auth tests.
func genOther(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	t.Helper()
	return ed25519.GenerateKey(nil)
}

// mintToken signs an EdDSA JWT with priv, as the client's CLI would: it carries the
// runnerd audience so the Verifier (which now requires aud == cambia-runnerd)
// accepts it.
func mintToken(t *testing.T, priv ed25519.PrivateKey, sub string, exp time.Duration) string {
	t.Helper()
	claims := jwt.MapClaims{"sub": sub, "aud": authtoken.Audience}
	if exp != 0 {
		claims["exp"] = time.Now().Add(exp).Unix()
	}
	tok := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	s, err := tok.SignedString(priv)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// do issues an authenticated request with the rig's default token.
func (r *testRig) do(method, path string, body any) *http.Response {
	return r.doTok(method, path, body, r.token)
}

func (r *testRig) doTok(method, path string, body any, token string) *http.Response {
	r.t.Helper()
	var rdr io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, r.baseURL+path, rdr)
	if err != nil {
		r.t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		r.t.Fatalf("%s %s: %v", method, path, err)
	}
	return resp
}

// decode reads a JSON body into v and closes it.
func decodeBody(t *testing.T, resp *http.Response, v any) {
	t.Helper()
	defer resp.Body.Close()
	if v == nil {
		io.Copy(io.Discard, resp.Body)
		return
	}
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatalf("decode body: %v", err)
	}
}

// jobResp is the {job: JobView} envelope of GET/DELETE.
type jobResp struct {
	Job JobView `json:"job"`
}

// submitResp is the POST /harness/jobs 201 body.
type submitResp struct {
	JobID    string `json:"job_id"`
	State    string `json:"state"`
	QueuePos int    `json:"queue_pos"`
}

// getState fetches a job's current state via GET /harness/jobs/{id}.
func (r *testRig) getState(id string) (string, bool) {
	resp := r.do(http.MethodGet, "/harness/jobs/"+id, nil)
	if resp.StatusCode == http.StatusNotFound {
		resp.Body.Close()
		return "", false
	}
	var jr jobResp
	decodeBody(r.t, resp, &jr)
	return jr.Job.State, true
}

// waitForState polls until the job reaches want or the timeout elapses.
func (r *testRig) waitForState(id, want string, timeout time.Duration) {
	r.t.Helper()
	deadline := time.Now().Add(timeout)
	var last string
	for time.Now().Before(deadline) {
		s, ok := r.getState(id)
		if ok {
			last = s
			if s == want {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	r.t.Fatalf("job %s: state %q, want %q after %s", id, last, want, timeout)
}

// baseSpec returns a minimal valid cpu train spec for the given name/kind.
func baseSpec(name, kind string) map[string]any {
	return map[string]any{
		"kind":   kind,
		"commit": strings.Repeat("a", 40),
		"name":   name,
		"config": "cfr/config/prtcfr_prod.yaml",
		"device": "cpu",
	}
}
