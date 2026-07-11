package harnessproxy

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/jason-s-yu/cambia/runnerd/authtoken"
)

// stubRunner is a TLS httptest server standing in for runnerd's control plane.
// It records the last request line and Bearer token so a test can assert the
// forwarded method/path/query and verify the minted token.
type stubRunner struct {
	ts *httptest.Server

	mu         sync.Mutex
	lastMethod string
	lastPath   string
	lastQuery  string
	lastToken  string
}

func newStubRunner(t *testing.T) *stubRunner {
	t.Helper()
	s := &stubRunner{}
	mux := http.NewServeMux()
	record := func(r *http.Request) {
		s.mu.Lock()
		s.lastMethod = r.Method
		s.lastPath = r.URL.Path
		s.lastQuery = r.URL.RawQuery
		s.lastToken = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		s.mu.Unlock()
	}
	mux.HandleFunc("DELETE /harness/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		record(r)
		if r.PathValue("id") == "missing" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"not_found","detail":"job not found"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"job":{"job_id":"` + r.PathValue("id") + `"}}`))
	})
	mux.HandleFunc("POST /harness/jobs/{id}/resume", func(w http.ResponseWriter, r *http.Request) {
		record(r)
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"job_id":"` + r.PathValue("id") + `","state":"queued"}`))
	})
	mux.HandleFunc("GET /ws/harness/jobs/{id}/logs", func(w http.ResponseWriter, r *http.Request) {
		record(r)
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
		if err != nil {
			return
		}
		defer c.CloseNow()
		_ = c.Write(r.Context(), websocket.MessageText,
			[]byte(`{"type":"log_backfill","data":{"lines":["hello from runner"]}}`))
		c.Close(websocket.StatusNormalClosure, "done")
	})
	s.ts = httptest.NewTLSServer(mux)
	t.Cleanup(s.ts.Close)
	return s
}

func (s *stubRunner) fingerprint() string {
	sum := sha256.Sum256(s.ts.Certificate().Raw)
	return hex.EncodeToString(sum[:])
}

func (s *stubRunner) token() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastToken
}

// newClientFor builds a Client aimed at the stub with the given pin, plus the
// public half of its throwaway signing key for token verification.
func newClientFor(t *testing.T, url, fingerprint string) (*Client, ed25519.PublicKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	keyPath := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(keyPath, priv, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		RunnerURL:       url,
		CertFingerprint: fingerprint,
		PrivateKeyPath:  keyPath,
		Subject:         "cambia-harness",
		TokenTTL:        900 * time.Second,
		OriginHost:      "nash",
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, pub
}

func TestStopForwardsWithPinAndToken(t *testing.T) {
	runner := newStubRunner(t)
	c, pub := newClientFor(t, runner.ts.URL, runner.fingerprint())

	resp, err := c.Stop(context.Background(), "run-a", true)
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if resp.Status != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.Status)
	}
	if runner.lastMethod != http.MethodDelete || runner.lastPath != "/harness/jobs/run-a" {
		t.Errorf("forwarded %s %s, want DELETE /harness/jobs/run-a", runner.lastMethod, runner.lastPath)
	}
	if runner.lastQuery != "force=true" {
		t.Errorf("query = %q, want force=true", runner.lastQuery)
	}
	// The minted Bearer token must carry aud and verify against runnerd's gate,
	// under the dashboard subject (audit separation).
	sub, verr := authtoken.NewVerifier(pub).Verify(runner.token())
	if verr != nil {
		t.Fatalf("forwarded token failed runnerd verify: %v", verr)
	}
	if sub != DashboardSubject {
		t.Errorf("token sub = %q, want %s", sub, DashboardSubject)
	}
}

func TestStopWithoutForceOmitsQuery(t *testing.T) {
	runner := newStubRunner(t)
	c, _ := newClientFor(t, runner.ts.URL, runner.fingerprint())
	if _, err := c.Stop(context.Background(), "run-a", false); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if runner.lastQuery != "" {
		t.Errorf("query = %q, want empty (no force)", runner.lastQuery)
	}
}

func TestResumeForwards(t *testing.T) {
	runner := newStubRunner(t)
	c, _ := newClientFor(t, runner.ts.URL, runner.fingerprint())
	resp, err := c.Resume(context.Background(), "run-b")
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if resp.Status != http.StatusAccepted {
		t.Errorf("status = %d, want 202", resp.Status)
	}
	if runner.lastMethod != http.MethodPost || runner.lastPath != "/harness/jobs/run-b/resume" {
		t.Errorf("forwarded %s %s, want POST /harness/jobs/run-b/resume", runner.lastMethod, runner.lastPath)
	}
}

func TestRunnerErrorForwarded(t *testing.T) {
	runner := newStubRunner(t)
	c, _ := newClientFor(t, runner.ts.URL, runner.fingerprint())
	resp, err := c.Stop(context.Background(), "missing", false)
	if err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if resp.Status != http.StatusNotFound {
		t.Errorf("status = %d, want 404 (runner error forwarded)", resp.Status)
	}
	var body map[string]any
	if err := json.Unmarshal(resp.Body, &body); err != nil {
		t.Fatalf("body not json: %v", err)
	}
	if body["error"] != "not_found" {
		t.Errorf("error = %v, want not_found", body["error"])
	}
}

func TestPinMismatchRejected(t *testing.T) {
	runner := newStubRunner(t)
	wrongPin := strings.Repeat("00", 32) // valid 64-hex shape, wrong value
	c, _ := newClientFor(t, runner.ts.URL, wrongPin)

	_, err := c.Stop(context.Background(), "run-a", false)
	if err == nil {
		t.Fatal("Stop against a mismatched pin must error")
	}
	if !errors.Is(err, ErrPinMismatch) {
		t.Fatalf("err = %v, want ErrPinMismatch", err)
	}
}

func TestDialLogsCopiesFrames(t *testing.T) {
	runner := newStubRunner(t)
	c, _ := newClientFor(t, runner.ts.URL, runner.fingerprint())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	up, _, err := c.DialLogs(ctx, "run-c")
	if err != nil {
		t.Fatalf("DialLogs: %v", err)
	}
	defer up.CloseNow()

	_, data, err := up.Read(ctx)
	if err != nil {
		t.Fatalf("read frame: %v", err)
	}
	var msg map[string]any
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatalf("frame not json: %v", err)
	}
	if msg["type"] != "log_backfill" {
		t.Errorf("frame type = %v, want log_backfill", msg["type"])
	}
}

func TestDialLogsPinMismatch(t *testing.T) {
	runner := newStubRunner(t)
	c, _ := newClientFor(t, runner.ts.URL, strings.Repeat("00", 32))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, _, err := c.DialLogs(ctx, "run-c")
	if err == nil {
		t.Fatal("DialLogs against a mismatched pin must error")
	}
	if !errors.Is(err, ErrPinMismatch) {
		t.Fatalf("err = %v, want ErrPinMismatch", err)
	}
}
