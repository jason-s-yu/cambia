package training

import (
	"context"
	"encoding/json"
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

func TestBackfillReadsLast200Lines(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	// Create log file with 250 lines.
	runDir := filepath.Join(tmpDir, "test-run-1", "logs")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "training.log")
	var lines []string
	for i := 0; i < 250; i++ {
		lines = append(lines, fmt.Sprintf("line %d", i))
	}
	if err := os.WriteFile(logPath, []byte(strings.Join(lines, "\n")+"\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Set up server with WS handler.
	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/", store.HandleLogStream)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/test-run-1/logs"
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	c, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()

	// Read backfill message.
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var msg WSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Type != "log_backfill" {
		t.Fatalf("expected log_backfill, got %s", msg.Type)
	}

	dataMap, ok := msg.Data.(map[string]interface{})
	if !ok {
		t.Fatal("expected data to be map")
	}
	backfillLines, ok := dataMap["lines"].([]interface{})
	if !ok {
		t.Fatal("expected lines array")
	}

	if len(backfillLines) != 200 {
		t.Fatalf("expected 200 backfill lines, got %d", len(backfillLines))
	}
	// First backfilled line should be "line 50" (250 - 200).
	firstLine := backfillLines[0].(string)
	if firstLine != "line 50" {
		t.Errorf("expected 'line 50', got %q", firstLine)
	}

	c.Close(websocket.StatusNormalClosure, "done")
}

func TestLiveTailing(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	runDir := filepath.Join(tmpDir, "test-run-1", "logs")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "training.log")
	if err := os.WriteFile(logPath, []byte("initial line\n"), 0644); err != nil {
		t.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/", store.HandleLogStream)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/test-run-1/logs"
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	c, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()

	// Read and discard backfill.
	_, _, err = c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Give fsnotify time to set up its watch before writing.
	time.Sleep(200 * time.Millisecond)

	// Append a new line to the log file.
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("new tailed line\n"); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Sync()
	f.Close()

	// Read the live-tailed line.
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var msg WSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Type != "log_line" {
		t.Fatalf("expected log_line, got %s", msg.Type)
	}
	dataMap := msg.Data.(map[string]interface{})
	line := dataMap["line"].(string)
	if line != "new tailed line" {
		t.Errorf("expected 'new tailed line', got %q", line)
	}

	c.Close(websocket.StatusNormalClosure, "done")
}

func TestFindLogFile(t *testing.T) {
	store, tmpDir := setupTestDB(t)

	// No log file exists yet.
	if got := store.findLogFile("test-run-1"); got != "" {
		t.Errorf("expected empty, got %s", got)
	}

	// Create training.log in logs/.
	runDir := filepath.Join(tmpDir, "test-run-1", "logs")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(runDir, "training.log")
	if err := os.WriteFile(logPath, []byte("test"), 0644); err != nil {
		t.Fatal(err)
	}

	if got := store.findLogFile("test-run-1"); got != logPath {
		t.Errorf("expected %s, got %s", logPath, got)
	}

	// Create train.log in run root — training.log should still take priority.
	trainLog := filepath.Join(tmpDir, "test-run-1", "train.log")
	if err := os.WriteFile(trainLog, []byte("test2"), 0644); err != nil {
		t.Fatal(err)
	}
	if got := store.findLogFile("test-run-1"); got != logPath {
		t.Errorf("training.log should take priority, got %s", got)
	}

	// Remove training.log — should fall back to train.log.
	os.Remove(logPath)
	if got := store.findLogFile("test-run-1"); got != trainLog {
		t.Errorf("expected train.log fallback, got %s", got)
	}
}

func TestLogStreamNotFound(t *testing.T) {
	store, _ := setupTestDB(t)

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/", store.HandleLogStream)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Try to access log for a run without any log file.
	resp, err := http.Get(ts.URL + "/ws/training/test-run-1/logs")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for missing log, got %d", resp.StatusCode)
	}
}
