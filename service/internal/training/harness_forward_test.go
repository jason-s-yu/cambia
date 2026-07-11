package training

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/service/internal/harnessproxy"
)

// tlsRunnerStub is a TLS httptest server standing in for runnerd's control plane
// in training-side handler tests.
type tlsRunnerStub struct {
	ts        *httptest.Server
	lastPath  string
	lastQuery string
}

func newTLSRunnerStub(t *testing.T) *tlsRunnerStub {
	t.Helper()
	s := &tlsRunnerStub{}
	mux := http.NewServeMux()
	mux.HandleFunc("DELETE /harness/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		s.lastPath, s.lastQuery = r.URL.Path, r.URL.RawQuery
		if r.PathValue("id") == "gone" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"not_found","detail":"job not found"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"job":{"job_id":"` + r.PathValue("id") + `"}}`))
	})
	mux.HandleFunc("POST /harness/jobs/{id}/resume", func(w http.ResponseWriter, r *http.Request) {
		s.lastPath = r.URL.Path
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"job_id":"` + r.PathValue("id") + `","state":"queued"}`))
	})
	s.ts = httptest.NewTLSServer(mux)
	t.Cleanup(s.ts.Close)
	return s
}

func (s *tlsRunnerStub) fingerprint() string {
	sum := sha256.Sum256(s.ts.Certificate().Raw)
	return hex.EncodeToString(sum[:])
}

// proxyClientFor builds a harnessproxy.Client aimed at url with the given pin and
// origin host, using a throwaway ed25519 key.
func proxyClientFor(t *testing.T, url, fingerprint, origin string) *harnessproxy.Client {
	t.Helper()
	_, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	keyPath := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(keyPath, priv, 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := harnessproxy.New(&harnessproxy.Config{
		RunnerURL:       url,
		CertFingerprint: fingerprint,
		PrivateKeyPath:  keyPath,
		Subject:         "cambia-harness",
		TokenTTL:        900 * time.Second,
		OriginHost:      origin,
	})
	if err != nil {
		t.Fatalf("harnessproxy.New: %v", err)
	}
	return c
}

// markRemoteHost writes a process.json carrying host so RemoteHost resolves the
// run as remote from that origin.
func markRemoteHost(t *testing.T, f *handlerFixture, name, host string) {
	t.Helper()
	writeState(t, f.runsDir, name, &procmgr.ProcessState{
		Name: name, Host: host, Status: procmgr.StatusRunning, Algorithm: "prt-cfr",
		PID: 9999999, PGID: 9999999, CreatedAt: procmgr.NowRFC3339(),
	})
}

func TestHandleStopRemoteForwarded(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	markRemoteHost(t, f, "remote-stop", "runner1")

	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/remote-stop/stop", `{"force":true}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["status"] != "requested" || body["action"] != "stop" || body["remote"] != true {
		t.Errorf("body = %v, want requested/stop/remote", body)
	}
	if runner.lastPath != "/harness/jobs/remote-stop" || runner.lastQuery != "force=true" {
		t.Errorf("forwarded %s?%s, want /harness/jobs/remote-stop?force=true", runner.lastPath, runner.lastQuery)
	}
}

func TestHandleResumeRemoteForwarded(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	markRemoteHost(t, f, "remote-resume", "runner1")

	w := doReq(f.ph.HandleResume, http.MethodPost, "/training/runs/remote-resume/resume", `{}`)
	if w.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want 202: %s", w.Code, w.Body.String())
	}
	if runner.lastPath != "/harness/jobs/remote-resume/resume" {
		t.Errorf("forwarded path = %s", runner.lastPath)
	}
}

func TestHandleStopRemoteMismatchOriginRefused(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	// Proxy origin is runner1 but the run originates on runner2: no pinned path,
	// keep the read-only 409.
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	markRemoteHost(t, f, "other-origin", "runner2")

	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/other-origin/stop", `{}`)
	assertRemoteRefused(t, w)
}

// TestHandleStartRemoteForwardableStays409 guards that a remote run cannot be
// started from the dashboard even when its origin is forwardable: start has no
// runner control-plane equivalent (fresh start / purge stay dashboard-forbidden).
func TestHandleStartRemoteForwardableStays409(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	markRemoteHost(t, f, "remote-start", "runner1")

	w := doReq(f.ph.HandleStart, http.MethodPost, "/training/runs/remote-start/start", `{}`)
	assertRemoteRefused(t, w)
	if runner.lastPath != "" {
		t.Errorf("start must not forward to the runner, got path %s", runner.lastPath)
	}
}

func TestHandleStopRemoteRunnerErrorForwarded(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	markRemoteHost(t, f, "gone", "runner1")

	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/gone/stop", `{}`)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (runner error forwarded): %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "not_found" {
		t.Errorf("error = %v, want not_found", body["error"])
	}
}

func TestHandleStopRemoteRunnerUnreachable(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	fp := runner.fingerprint()
	url := runner.ts.URL
	runner.ts.Close() // now unreachable (connection refused)
	f.ph.SetHarnessProxy(proxyClientFor(t, url, fp, "runner1"))
	markRemoteHost(t, f, "unreachable", "runner1")

	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/unreachable/stop", `{}`)
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "runner_unreachable" {
		t.Errorf("error = %v, want runner_unreachable", body["error"])
	}
}

func TestHandleStopRemotePinMismatch(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	wrongPin := strings.Repeat("00", 32)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, wrongPin, "runner1"))
	markRemoteHost(t, f, "pinbad", "runner1")

	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/pinbad/stop", `{}`)
	if w.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	if body["error"] != "runner_pin_mismatch" {
		t.Errorf("error = %v, want runner_pin_mismatch", body["error"])
	}
}

// TestHandleStopLocalUnaffectedByProxy confirms a local run still takes the local
// path (404 not found here, since it is not registered in the manager) even when
// a proxy is configured: the proxy only intercepts remote-origin runs.
func TestHandleStopLocalStillLocal(t *testing.T) {
	f := newHandlerFixture(t)
	runner := newTLSRunnerStub(t)
	f.ph.SetHarnessProxy(proxyClientFor(t, runner.ts.URL, runner.fingerprint(), "runner1"))
	// No process.json -> RemoteHost == "" -> local path -> 404 not_found.
	w := doReq(f.ph.HandleStop, http.MethodPost, "/training/runs/local-run/stop", `{}`)
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (local path): %s", w.Code, w.Body.String())
	}
	if runner.lastPath != "" {
		t.Errorf("a local run must not forward to the runner, got %s", runner.lastPath)
	}
}

// TestRemoteLogsFallbackOnDialFailure exercises the WS log proxy fallback: when
// the runner WS is unreachable, the browser gets a log_notice frame followed by
// the locally synced file's backfill.
func TestRemoteLogsFallbackOnDialFailure(t *testing.T) {
	store, tmpDir := setupTestDB(t)
	runner := newTLSRunnerStub(t)
	fp := runner.fingerprint()
	url := runner.ts.URL
	runner.ts.Close() // WS dial will fail
	store.SetHarnessProxy(proxyClientFor(t, url, fp, "runner1"))

	// Mark the run remote (origin runner1) and give it a locally synced log.
	writeState(t, tmpDir, "remote-log", &procmgr.ProcessState{
		Name: "remote-log", Host: "runner1", Status: procmgr.StatusRunning,
		Algorithm: "prt-cfr", PID: 9999999, PGID: 9999999, CreatedAt: procmgr.NowRFC3339(),
	})
	logDir := filepath.Join(tmpDir, "remote-log", "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(logDir, "training.log"), []byte("synced line 1\nsynced line 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/", store.HandleLogStream)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/remote-log/logs"
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer c.CloseNow()

	// First frame: the runner-unreachable notice.
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatalf("read notice: %v", err)
	}
	var notice map[string]any
	_ = json.Unmarshal(data, &notice)
	if notice["type"] != "log_notice" {
		t.Fatalf("first frame type = %v, want log_notice", notice["type"])
	}

	// Next frame: the local synced backfill.
	_, data, err = c.Read(ctx)
	if err != nil {
		t.Fatalf("read backfill: %v", err)
	}
	var backfill map[string]any
	_ = json.Unmarshal(data, &backfill)
	if backfill["type"] != "log_backfill" {
		t.Fatalf("second frame type = %v, want log_backfill", backfill["type"])
	}
	c.Close(websocket.StatusNormalClosure, "done")
}
