package training

import (
	"context"
	"net/http"
	"time"

	"github.com/coder/websocket"
)

// remoteRetryBackoff is the pause before the single reconnect attempt after a
// mid-stream runner drop. One retry, then fall back to the synced-file tail: no
// reconnect storm against a runner that is down.
const remoteRetryBackoff = 1 * time.Second

// serveRemoteLogs bridges a browser WS to the runner's pinned WS log stream for
// a remote run. It accepts the browser connection, proxies runner log frames
// verbatim, and on any dial or mid-stream failure (after one retry) sends a
// log_notice frame and falls back to tailing the locally synced file. The token
// validity is checked only at the runner's upgrade; a tail outliving the token
// TTL is fine.
func (s *TrainingStore) serveRemoteLogs(w http.ResponseWriter, r *http.Request, name string) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		OriginPatterns: []string{"*"},
	})
	if err != nil {
		return
	}
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	// Read-pump the browser side so a client disconnect cancels ctx and tears
	// down the upstream stream (and any fallback tail) instead of leaking it.
	ctx = c.CloseRead(ctx)

	if s.proxyRemoteLogs(ctx, c, name) {
		return // stream ended cleanly (browser closed or upstream closed normally)
	}

	// Proxy failed or dropped: a log_notice was already sent. Fall back to the
	// locally synced file if one exists; otherwise hold the connection open
	// until the browser goes away (nothing to tail yet).
	if logPath := s.findLogFile(name); logPath != "" {
		s.streamLocalFile(ctx, c, logPath)
		return
	}
	<-ctx.Done()
	c.Close(websocket.StatusGoingAway, "context done")
}

// proxyRemoteLogs dials the runner's pinned WS log stream and copies its text
// frames verbatim to the browser. It returns true when the stream ended because
// the browser disconnected or the upstream closed normally (nothing more to do),
// and false after sending a log_notice frame when the upstream was unreachable
// or dropped mid-stream. A mid-stream drop is retried once (with backoff) before
// falling back. A pin mismatch is treated like any dial failure here: the WS
// falls back to the last synced log rather than showing an unpinned stream.
func (s *TrainingStore) proxyRemoteLogs(ctx context.Context, browser *websocket.Conn, name string) bool {
	const maxAttempts = 2 // initial dial + one reconnect on a runner drop
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return true // browser went away during backoff
			case <-time.After(remoteRetryBackoff):
			}
		}
		up, _, err := s.proxy.DialLogs(ctx, name)
		if err != nil {
			continue // unreachable / pin mismatch: retry once, then fall back
		}
		cerr := copyFrames(ctx, up, browser)
		up.CloseNow()
		if ctx.Err() != nil {
			return true // browser closed; done
		}
		switch websocket.CloseStatus(cerr) {
		case websocket.StatusNormalClosure, websocket.StatusGoingAway:
			return true // runner ended the stream cleanly
		}
		// Abnormal drop or read error: loop retries once, then falls back.
	}
	sendLogNotice(ctx, browser)
	return false
}

// copyFrames forwards frames from the upstream runner connection to the browser
// until either side errors. It returns the first error (an upstream read error,
// including a normal close, or a browser write error).
func copyFrames(ctx context.Context, up, browser *websocket.Conn) error {
	for {
		typ, data, err := up.Read(ctx)
		if err != nil {
			return err
		}
		if err := browser.Write(ctx, typ, data); err != nil {
			return err
		}
	}
}

// sendLogNotice tells the browser the runner is unreachable and the tail below
// is the last synced log. The frame is a new log_notice type; a client that does
// not render it simply ignores it (the log_backfill/log_line stream is
// unaffected).
func sendLogNotice(ctx context.Context, c *websocket.Conn) {
	_ = writeWSMessage(ctx, c, WSMessage{
		Type: "log_notice",
		Data: map[string]interface{}{"message": "runner unreachable, showing last synced log"},
	})
}
