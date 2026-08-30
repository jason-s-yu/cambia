// internal/training/ws_subprotocol_test.go
package training

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// dialWithProtocols opens a WS to wsURL offering the given subprotocols and
// returns the negotiated one.
func dialWithProtocols(t *testing.T, wsURL string, protocols []string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	c, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{Subprotocols: protocols})
	if err != nil {
		t.Fatalf("dial %s offering %v: %v", wsURL, protocols, err)
	}
	defer c.CloseNow()
	return c.Subprotocol()
}

// TestResourceWSSelectsCambiaSubprotocol checks that the resource socket
// negotiates "cambia". A client that offers subprotocols and is given none back
// must fail the handshake per RFC 6455, so this socket has to select "cambia"
// now that a tab-pinned client always offers it alongside its token entry
// (cambia-1149). A client that offers nothing still gets an empty selection.
func TestResourceWSSelectsCambiaSubprotocol(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/resources", m.HandleWS)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/resources"

	if got := dialWithProtocols(t, wsURL, []string{"cambia", "cambia-token.some.jwt.value"}); got != "cambia" {
		t.Fatalf("expected the resource socket to select %q, got %q", "cambia", got)
	}
	if got := dialWithProtocols(t, wsURL, []string{"cambia"}); got != "cambia" {
		t.Fatalf("expected the resource socket to select %q, got %q", "cambia", got)
	}
	if got := dialWithProtocols(t, wsURL, nil); got != "" {
		t.Fatalf("expected no subprotocol when the client offers none, got %q", got)
	}
}

// TestLogStreamWSSelectsCambiaSubprotocol checks the same for the per-run log
// socket, the other WebSocket a browser tab opens against the training API.
func TestLogStreamWSSelectsCambiaSubprotocol(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	runDir := filepath.Join(tmpDir, "subprotocol-run", "logs")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "training.log"), []byte("line 0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/", store.HandleLogStream)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := fmt.Sprintf("ws%s/ws/training/subprotocol-run/logs", strings.TrimPrefix(ts.URL, "http"))

	if got := dialWithProtocols(t, wsURL, []string{"cambia", "cambia-token.some.jwt.value"}); got != "cambia" {
		t.Fatalf("expected the log socket to select %q, got %q", "cambia", got)
	}
	if got := dialWithProtocols(t, wsURL, nil); got != "" {
		t.Fatalf("expected no subprotocol when the client offers none, got %q", got)
	}
}
