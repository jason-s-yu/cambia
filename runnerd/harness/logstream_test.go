package harness

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fsnotify/fsnotify"
)

// TestReadNewLinesPartialLine covers L9: a partial trailing line (no newline
// yet) must not be consumed or emitted, and once its newline arrives the whole
// line is returned exactly once. The old +len(line)+1 accounting over-counted
// the unterminated line, which split the next mid-line write.
func TestReadNewLinesPartialLine(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "log.txt")

	// First write: a complete line plus a partial (unterminated) tail.
	if err := os.WriteFile(p, []byte("alpha\npar"), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, off, err := readNewLines(p, 0)
	if err != nil {
		t.Fatalf("readNewLines: %v", err)
	}
	if len(lines) != 1 || lines[0] != "alpha" {
		t.Fatalf("first read lines = %v, want [alpha]", lines)
	}
	// Offset must sit just past "alpha\n" (6 bytes), leaving "par" unconsumed.
	if off != 6 {
		t.Fatalf("offset after first read = %d, want 6", off)
	}

	// Complete the partial line and append another.
	if err := os.WriteFile(p, []byte("alpha\npartial-then-rest\nomega\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, off, err = readNewLines(p, off)
	if err != nil {
		t.Fatalf("readNewLines 2: %v", err)
	}
	// The formerly partial line must appear once, whole, not split or duplicated.
	if len(lines) != 2 || lines[0] != "partial-then-rest" || lines[1] != "omega" {
		t.Fatalf("second read lines = %v, want [partial-then-rest omega]", lines)
	}
	if off != int64(len("alpha\npartial-then-rest\nomega\n")) {
		t.Fatalf("final offset = %d, want %d", off, len("alpha\npartial-then-rest\nomega\n"))
	}

	// A read at EOF yields nothing and does not move the offset.
	lines, off2, err := readNewLines(p, off)
	if err != nil {
		t.Fatalf("readNewLines 3: %v", err)
	}
	if len(lines) != 0 || off2 != off {
		t.Fatalf("EOF read = (%v, %d), want ([], %d)", lines, off2, off)
	}
}

// TestReadLastNLinesBoundary checks that backfill returns only complete lines
// and reports a line-boundary offset, so a trailing partial line is streamed
// later rather than double-counted at the backfill/stream seam.
func TestReadLastNLinesBoundary(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "log.txt")
	if err := os.WriteFile(p, []byte("one\ntwo\nthree-partial"), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, off, err := readLastNLines(p, 200)
	if err != nil {
		t.Fatalf("readLastNLines: %v", err)
	}
	if len(lines) != 2 || lines[0] != "one" || lines[1] != "two" {
		t.Fatalf("backfill lines = %v, want [one two]", lines)
	}
	if off != int64(len("one\ntwo\n")) {
		t.Fatalf("boundary offset = %d, want %d", off, len("one\ntwo\n"))
	}
	// Streaming from the reported offset picks up the partial line once completed.
	if err := os.WriteFile(p, []byte("one\ntwo\nthree-partial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	streamed, _, err := readNewLines(p, off)
	if err != nil {
		t.Fatalf("readNewLines: %v", err)
	}
	if len(streamed) != 1 || streamed[0] != "three-partial" {
		t.Fatalf("streamed = %v, want [three-partial]", streamed)
	}
}

// TestLogsWSClientDisconnectReleasesWatcher covers M6 for the log handler: an
// abrupt client disconnect must return the handler goroutine and close its
// fsnotify watcher (freeing the inotify fd), not leak them. The seams are set
// before newRig so the happens-before chain through server startup keeps the
// hook writes race-free under -race.
func TestLogsWSClientDisconnectReleasesWatcher(t *testing.T) {
	done := make(chan string, 1)
	watcherCh := make(chan *fsnotify.Watcher, 1)
	wsHandlerDone = func(h string) {
		select {
		case done <- h:
		default:
		}
	}
	logWatcherHook = func(w *fsnotify.Watcher) {
		select {
		case watcherCh <- w:
		default:
		}
	}
	t.Cleanup(func() { wsHandlerDone = nil; logWatcherHook = nil })

	r := newRig(t, rigConfig{origin: "https://client.lan"})
	logDir := filepath.Join(r.runsDir, "leakjob", "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(logDir, "training.log"), []byte("x\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	c, _, err := r.dialWS("/ws/harness/jobs/leakjob/logs", "https://client.lan", r.token)
	if err != nil {
		t.Fatalf("dial logs ws: %v", err)
	}
	if env := readEnvelope(t, c); env.Type != "log_backfill" {
		t.Fatalf("first message = %q, want log_backfill", env.Type)
	}
	var w *fsnotify.Watcher
	select {
	case w = <-watcherCh:
	case <-time.After(2 * time.Second):
		t.Fatal("log handler never created a watcher")
	}

	// Abrupt disconnect: no clean close handshake.
	_ = c.CloseNow()

	select {
	case h := <-done:
		if h != "logs" {
			t.Fatalf("done handler = %q, want logs", h)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("log handler goroutine leaked (never returned) after client disconnect")
	}

	// The watcher must have been closed: fsnotify closes Events on Close.
	deadline := time.After(2 * time.Second)
	for {
		select {
		case _, open := <-w.Events:
			if !open {
				return // closed -> pass
			}
		case <-deadline:
			t.Fatal("fsnotify watcher not closed after handler return")
		}
	}
}

// TestQueueWSClientDisconnectReturns covers M6 for the queue handler: an abrupt
// client disconnect must return the handler goroutine (and drop its
// subscription), not block on the never-cancelled request context of a hijacked
// connection.
func TestQueueWSClientDisconnectReturns(t *testing.T) {
	done := make(chan string, 1)
	wsHandlerDone = func(h string) {
		select {
		case done <- h:
		default:
		}
	}
	t.Cleanup(func() { wsHandlerDone = nil })

	r := newRig(t, rigConfig{origin: "https://client.lan"})
	c, _, err := r.dialWS("/ws/harness/queue", "https://client.lan", r.token)
	if err != nil {
		t.Fatalf("dial queue ws: %v", err)
	}
	if env := readEnvelope(t, c); env.Type != "queue_snapshot" {
		t.Fatalf("first message = %q, want queue_snapshot", env.Type)
	}

	_ = c.CloseNow()

	select {
	case h := <-done:
		if h != "queue" {
			t.Fatalf("done handler = %q, want queue", h)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("queue handler goroutine leaked (never returned) after client disconnect")
	}
}
