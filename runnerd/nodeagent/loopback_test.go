package nodeagent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"
)

// TestLoopbackStreamsBeforeTheHandlerReturns pins the one property a buffering
// in-process transport would lose: the response headers reach the caller when
// the handler writes them, not when it returns. The held claim and events polls
// of D2 and D45 depend on it, and so does a ranged snapshot read, which must
// stream rather than materialize in memory.
func TestLoopbackStreamsBeforeTheHandlerReturns(t *testing.T) {
	release := make(chan struct{})
	rt := LoopbackTransport{Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Probe", "set-before-write")
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte("first"))
		<-release
		_, _ = w.Write([]byte("-second"))
	})}

	req, err := http.NewRequest(http.MethodGet, "http://loopback/x", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("round trip: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("status = %d, want 202", resp.StatusCode)
	}
	if got := resp.Header.Get("X-Probe"); got != "set-before-write" {
		t.Fatalf("header = %q, want the value set before the first write", got)
	}
	head := make([]byte, 5)
	if _, err := io.ReadFull(resp.Body, head); err != nil {
		t.Fatalf("read the first chunk: %v", err)
	}
	if string(head) != "first" {
		t.Fatalf("first chunk = %q, want %q", head, "first")
	}

	close(release)
	rest, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read the rest: %v", err)
	}
	if string(rest) != "-second" {
		t.Fatalf("rest = %q, want %q", rest, "-second")
	}
}

// TestLoopbackCancelsAHeldRequest is the other half: a node whose context is
// cancelled while a long poll is held gets its error back instead of blocking
// on a handler that answers only when work exists.
func TestLoopbackCancelsAHeldRequest(t *testing.T) {
	held := make(chan struct{})
	rt := LoopbackTransport{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(held)
		<-r.Context().Done()
	})}

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://loopback/nashnet/claim", nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, rerr := rt.RoundTrip(req)
		done <- rerr
	}()

	select {
	case <-held:
	case <-time.After(5 * time.Second):
		t.Fatal("the handler never ran")
	}
	cancel()

	select {
	case rerr := <-done:
		if !errors.Is(rerr, context.Canceled) {
			t.Fatalf("round trip error = %v, want context.Canceled", rerr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("round trip did not return after the context was cancelled")
	}
}
