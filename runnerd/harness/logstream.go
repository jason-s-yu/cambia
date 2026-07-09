package harness

import (
	"bufio"
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
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

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
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

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

// readLastNLines returns the last n lines of a file and the byte offset at EOF.
func readLastNLines(path string, n int) ([]string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return []string{}, 0, err
	}
	defer f.Close()

	var all []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		all = append(all, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, 0, err
	}
	offset, _ := f.Seek(0, io.SeekEnd)
	if len(all) > n {
		all = all[len(all)-n:]
	}
	return all, offset, nil
}

// readNewLines reads lines from offset to EOF, tracking the new offset manually
// (bufio.Scanner buffers ahead of what it returns).
func readNewLines(path string, offset int64) ([]string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, offset, err
	}
	defer f.Close()
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, offset, err
	}
	var lines []string
	newOffset := offset
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		lines = append(lines, sc.Text())
		newOffset += int64(len(sc.Bytes())) + 1
	}
	return lines, newOffset, sc.Err()
}
