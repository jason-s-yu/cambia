package harness

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func (r *testRig) dialWS(path, origin, token string) (*websocket.Conn, *http.Response, error) {
	wsURL := strings.Replace(r.baseURL, "https://", "wss://", 1) + path
	hdr := http.Header{}
	if token != "" {
		hdr.Set("Authorization", "Bearer "+token)
	}
	if origin != "" {
		hdr.Set("Origin", origin)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	return websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPClient: r.client,
		HTTPHeader: hdr,
	})
}

func readEnvelope(t *testing.T, c *websocket.Conn) wsEnvelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatalf("ws read: %v", err)
	}
	var env wsEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		t.Fatalf("ws unmarshal: %v", err)
	}
	return env
}

func TestQueueWSCorrectOrigin(t *testing.T) {
	r := newRig(t, rigConfig{origin: "https://client.lan"})
	c, _, err := r.dialWS("/ws/harness/queue", "https://client.lan", r.token)
	if err != nil {
		t.Fatalf("dial with correct origin: %v", err)
	}
	defer c.CloseNow()
	env := readEnvelope(t, c)
	if env.Type != "queue_snapshot" {
		t.Fatalf("first message type = %q, want queue_snapshot", env.Type)
	}
}

func TestQueueWSWrongOriginRejected(t *testing.T) {
	r := newRig(t, rigConfig{origin: "https://client.lan"})
	// Valid token, disallowed origin -> 403 (never wildcard).
	c, resp, err := r.dialWS("/ws/harness/queue", "https://evil.example", r.token)
	if err == nil {
		c.CloseNow()
		t.Fatal("dial with a disallowed origin should be rejected")
	}
	if resp != nil && resp.StatusCode != http.StatusForbidden {
		t.Fatalf("wrong origin status = %d, want 403", resp.StatusCode)
	}
}

func TestQueueWSRequiresAuth(t *testing.T) {
	r := newRig(t, rigConfig{origin: "https://client.lan"})
	// Correct origin, no token -> 401 (auth precedes origin).
	c, resp, err := r.dialWS("/ws/harness/queue", "https://client.lan", "")
	if err == nil {
		c.CloseNow()
		t.Fatal("dial without a token should be rejected")
	}
	if resp != nil && resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no-token status = %d, want 401", resp.StatusCode)
	}
}

func TestLogsWSBackfill(t *testing.T) {
	r := newRig(t, rigConfig{origin: "https://client.lan"})
	dir := filepath.Join(r.runsDir, "logjob", "logs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "training.log"), []byte("line-1\nline-2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	c, _, err := r.dialWS("/ws/harness/jobs/logjob/logs", "https://client.lan", r.token)
	if err != nil {
		t.Fatalf("dial logs ws: %v", err)
	}
	defer c.CloseNow()
	env := readEnvelope(t, c)
	if env.Type != "log_backfill" {
		t.Fatalf("first logs message = %q, want log_backfill", env.Type)
	}
}
