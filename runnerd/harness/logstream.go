package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/coder/websocket"
	"github.com/fsnotify/fsnotify"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// logBackfillLines is the number of trailing log lines sent before streaming.
const logBackfillLines = 200

// wsHandlerDone and logWatcherHook are test seams (nil in production, mirroring
// procmgr's killGroupFunc). wsHandlerDone fires from a defer registered so it
// runs last, after the watcher and connection are closed, letting a test assert
// via a done-channel that a WS handler goroutine actually returns on client
// disconnect. logWatcherHook captures the log handler's fsnotify watcher so a
// test can assert the watcher (and its inotify fd) is closed on disconnect.
var (
	wsHandlerDone  func(handler string)
	logWatcherHook func(w *fsnotify.Watcher)
)

// wsEnvelope is the WebSocket message envelope.
type wsEnvelope struct {
	Type string `json:"type"`
	Data any    `json:"data"`
}

// acceptWS enforces the single allowed origin (design 5.3), then upgrades. The
// exact-match origin check runs before Accept; OriginPatterns is set to the
// configured origin's host as defense-in-depth, never wildcard.
func (s *Server) acceptWS(w http.ResponseWriter, r *http.Request) (*websocket.Conn, bool) {
	if !s.checkOrigin(r) {
		http.Error(w, "forbidden origin", http.StatusForbidden)
		return nil, false
	}
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		OriginPatterns: []string{s.originHostPattern()},
	})
	if err != nil {
		return nil, false
	}
	return c, true
}

// handleQueueWS is GET /ws/harness/queue: emits a snapshot on connect and on
// every state transition.
func (s *Server) handleQueueWS(w http.ResponseWriter, r *http.Request) {
	c, ok := s.acceptWS(w, r)
	if !ok {
		return
	}
	defer func() {
		if wsHandlerDone != nil {
			wsHandlerDone("queue")
		}
	}()
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	// Drain control frames so a client close/disconnect is noticed: CloseRead
	// actively reads (responding to ping/pong/close) and cancels the returned
	// context when the peer goes away. Without it a dead client would leak this
	// goroutine and its subscription until process exit.
	ctx = c.CloseRead(ctx)

	ch, unsub := s.disp.Subscribe()
	defer unsub()

	if err := writeWS(ctx, c, wsEnvelope{Type: "queue_snapshot", Data: s.disp.Snapshot()}); err != nil {
		return
	}
	for {
		select {
		case <-ctx.Done():
			c.Close(websocket.StatusGoingAway, "context done")
			return
		case snap, open := <-ch:
			if !open {
				return
			}
			if err := writeWS(ctx, c, wsEnvelope{Type: "queue_snapshot", Data: snap}); err != nil {
				return
			}
		}
	}
}

// handleLogsWS is GET /ws/harness/jobs/{id}/logs: backfills the last 200 lines
// then tails logs/training.log via fsnotify.
func (s *Server) handleLogsWS(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := procmgr.ValidateName(id); err != nil {
		http.Error(w, "invalid run name", http.StatusBadRequest)
		return
	}
	logPath := s.findLogFile(id)
	if logPath == "" {
		http.Error(w, "log file not found", http.StatusNotFound)
		return
	}
	c, ok := s.acceptWS(w, r)
	if !ok {
		return
	}
	defer func() {
		if wsHandlerDone != nil {
			wsHandlerDone("logs")
		}
	}()
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	// Drain control frames so a client close/disconnect cancels ctx; otherwise a
	// dead client would leak this goroutine plus the fsnotify watcher and its fd
	// until inotify exhaustion. watcher.Close() and c.CloseNow() are deferred
	// below, so the cancelled loop releases both.
	ctx = c.CloseRead(ctx)

	lines, offset, err := readLastNLines(logPath, logBackfillLines)
	if err != nil {
		c.Close(websocket.StatusInternalError, "failed to read log")
		return
	}
	if err := writeWS(ctx, c, wsEnvelope{Type: "log_backfill", Data: map[string]any{"lines": lines}}); err != nil {
		return
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		c.Close(websocket.StatusInternalError, "failed to create watcher")
		return
	}
	defer watcher.Close()
	if err := watcher.Add(logPath); err != nil {
		c.Close(websocket.StatusInternalError, "failed to watch log")
		return
	}
	if logWatcherHook != nil {
		logWatcherHook(watcher)
	}

	for {
		select {
		case <-ctx.Done():
			c.Close(websocket.StatusGoingAway, "context done")
			return
		case ev, open := <-watcher.Events:
			if !open {
				return
			}
			if ev.Has(fsnotify.Write) {
				newLines, newOffset, rerr := readNewLines(logPath, offset)
				if rerr != nil {
					continue
				}
				offset = newOffset
				ts := time.Now().UTC().Format("2006-01-02T15:04:05Z07:00")
				for _, line := range newLines {
					if err := writeWS(ctx, c, wsEnvelope{Type: "log_line", Data: map[string]any{"line": line, "ts": ts}}); err != nil {
						return
					}
				}
			}
			if ev.Has(fsnotify.Remove) || ev.Has(fsnotify.Create) {
				_ = watcher.Remove(logPath)
				time.Sleep(100 * time.Millisecond)
				if err := watcher.Add(logPath); err == nil {
					offset = 0
				}
			}
		case _, open := <-watcher.Errors:
			if !open {
				return
			}
		}
	}
}

// findLogFile locates the run's training log.
func (s *Server) findLogFile(name string) string {
	p := filepath.Join(s.runsDir, name, "logs", "training.log")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// writeWS marshals and sends msg as a text frame.
func writeWS(ctx context.Context, c *websocket.Conn, msg wsEnvelope) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.Write(ctx, websocket.MessageText, data)
}

// readLastNLines returns the last n complete lines of a file and the byte offset
// of the first byte after the last newline. A trailing partial line (no final
// newline) is deliberately not returned and not counted in the offset: it is
// streamed later by readNewLines once its newline arrives, which keeps the
// backfill/stream handoff on a line boundary so a mid-line write is never split.
func readLastNLines(path string, n int) ([]string, int64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return []string{}, 0, err
	}
	lastNL := bytes.LastIndexByte(data, '\n')
	if lastNL < 0 {
		// No complete line yet: nothing to backfill, offset stays at 0 so the
		// partial line streams in full once its newline lands.
		return []string{}, 0, nil
	}
	lines := splitLines(data[:lastNL+1])
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines, int64(lastNL + 1), nil
}

// readNewLines reads complete lines from offset to EOF and returns them with the
// advanced offset. Only bytes belonging to a newline-terminated line are
// consumed: a partial trailing line (no newline yet) is left unconsumed so the
// next call re-reads it and emits the whole line exactly once when the newline
// arrives. The offset advances by the true consumed byte count (line bytes plus
// the one newline), never over-counting a final unterminated line.
func readNewLines(path string, offset int64) ([]string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, offset, err
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, offset, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, offset, err
	}
	lastNL := bytes.LastIndexByte(data, '\n')
	if lastNL < 0 {
		return nil, offset, nil // only a partial line so far; consume nothing
	}
	lines := splitLines(data[:lastNL+1])
	return lines, offset + int64(lastNL) + 1, nil
}

// splitLines splits a newline-terminated byte slice into lines, stripping a
// trailing carriage return per line to match bufio.ScanLines' CRLF handling.
// The input must end in '\n'; the final empty field that '\n'-splitting would
// otherwise produce is dropped.
func splitLines(data []byte) []string {
	body := data[:len(data)-1] // drop the trailing '\n'
	out := make([]string, 0, bytes.Count(body, []byte{'\n'})+1)
	for _, ln := range bytes.Split(body, []byte{'\n'}) {
		out = append(out, string(bytes.TrimSuffix(ln, []byte{'\r'})))
	}
	return out
}
