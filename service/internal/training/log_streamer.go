package training

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/coder/websocket"
	"github.com/fsnotify/fsnotify"
)

const logBackfillLines = 200

// WSMessage is the envelope for WebSocket messages.
type WSMessage struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

// HandleLogStream upgrades to WebSocket and streams training log lines in real time.
func (s *TrainingStore) HandleLogStream(w http.ResponseWriter, r *http.Request) {
	name := extractRunName(r)
	if name == "" {
		http.Error(w, "missing run name", http.StatusBadRequest)
		return
	}

	logPath := s.findLogFile(name)
	if logPath == "" {
		http.Error(w, "log file not found", http.StatusNotFound)
		return
	}

	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		OriginPatterns: []string{"*"},
	})
	if err != nil {
		return
	}
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Send backfill.
	lines, offset, err := readLastNLines(logPath, logBackfillLines)
	if err != nil {
		c.Close(websocket.StatusInternalError, "failed to read log")
		return
	}
	if err := writeWSMessage(ctx, c, WSMessage{
		Type: "log_backfill",
		Data: map[string]interface{}{"lines": lines},
	}); err != nil {
		return
	}

	// Set up file watcher.
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

	// Tail loop.
	for {
		select {
		case <-ctx.Done():
			c.Close(websocket.StatusGoingAway, "context done")
			return
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			if event.Has(fsnotify.Write) {
				newLines, newOffset, err := readNewLines(logPath, offset)
				if err != nil {
					continue
				}
				offset = newOffset
				ts := time.Now().UTC().Format(time.RFC3339)
				for _, line := range newLines {
					if err := writeWSMessage(ctx, c, WSMessage{
						Type: "log_line",
						Data: map[string]interface{}{"line": line, "ts": ts},
					}); err != nil {
						return
					}
				}
			}
			if event.Has(fsnotify.Remove) || event.Has(fsnotify.Create) {
				// Log rotation — re-watch if possible.
				_ = watcher.Remove(logPath)
				time.Sleep(100 * time.Millisecond)
				if err := watcher.Add(logPath); err == nil {
					offset = 0
				}
			}
		case _, ok := <-watcher.Errors:
			if !ok {
				return
			}
		}
	}
}

// findLogFile locates the log file for a run, checking common names.
func (s *TrainingStore) findLogFile(runName string) string {
	runDir := filepath.Join(s.runsDir, runName)

	// Check common locations.
	candidates := []string{
		filepath.Join(runDir, "logs", "training.log"),
		filepath.Join(runDir, "train.log"),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// Fall back to latest file in logs/.
	logsDir := filepath.Join(runDir, "logs")
	entries, err := os.ReadDir(logsDir)
	if err != nil || len(entries) == 0 {
		return ""
	}
	sort.Slice(entries, func(i, j int) bool {
		fi, _ := entries[i].Info()
		fj, _ := entries[j].Info()
		if fi == nil || fj == nil {
			return false
		}
		return fi.ModTime().After(fj.ModTime())
	})
	return filepath.Join(logsDir, entries[0].Name())
}

// readLastNLines reads the last n lines from a file and returns them plus the final offset.
func readLastNLines(path string, n int) ([]string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return []string{}, 0, err
	}
	defer f.Close()

	var allLines []string
	scanner := bufio.NewScanner(f)
	// Increase buffer for long log lines.
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		allLines = append(allLines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, err
	}

	offset, _ := f.Seek(0, io.SeekEnd)

	if len(allLines) > n {
		allLines = allLines[len(allLines)-n:]
	}
	return allLines, offset, nil
}

// readNewLines reads lines from a file starting at the given byte offset.
// Tracks offset manually since bufio.Scanner buffers ahead of what it returns.
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
	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
		newOffset += int64(len(scanner.Bytes())) + 1 // +1 for newline
	}

	return lines, newOffset, scanner.Err()
}

func writeWSMessage(ctx context.Context, c *websocket.Conn, msg WSMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return c.Write(ctx, websocket.MessageText, data)
}
