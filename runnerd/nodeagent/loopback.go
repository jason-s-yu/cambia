package nodeagent

import (
	"errors"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"
)

// loopbackBase is the authority the embedded node's requests carry. Nothing
// resolves it: the round tripper below never opens a socket, and the host is
// present only because an http.Client requires an absolute URL.
const loopbackBase = "http://embedded-node.nashnet.invalid"

// LoopbackTransport is the in-process half of D65: an http.RoundTripper that
// hands a request straight to the coordinator's own routed handler instead of
// writing it to a socket. It exists so the embedded node of --role both runs
// the transport it runs remotely, byte-identical request and response shapes
// included, rather than a second code path with its own failure semantics.
//
// It shortcuts nothing. The request reaches the same mux, the same node and
// lease credential middlewares, the same fence, the same validators, and the
// same commit transaction a request off the wire reaches; what it skips is the
// TCP connection, the TLS handshake, and the certificate pin, none of which
// authenticate anything between two halves of one process.
type LoopbackTransport struct {
	// Handler is the coordinator's routed control plane, i.e. Server.Handler().
	Handler http.Handler
}

// RoundTrip serves req against the handler and streams the answer back. The
// body is an io.Pipe rather than a buffer, so a ranged snapshot read does not
// materialize in memory and a held long poll returns its response headers the
// moment the handler writes them rather than when the handler returns.
func (t LoopbackTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		defer req.Body.Close()
	}
	pr, pw := io.Pipe()
	w := &loopbackWriter{header: make(http.Header), body: pw, ready: make(chan struct{})}

	go func() {
		defer func() {
			// A handler that panicked has written no terminator, so the reader
			// would block forever; closing the pipe with the panic makes it a
			// transport error on the node's side instead.
			if rec := recover(); rec != nil {
				w.start(http.StatusInternalServerError)
				_ = pw.CloseWithError(errPanic)
				return
			}
			w.start(http.StatusOK)
			_ = pw.Close()
		}()
		t.Handler.ServeHTTP(w, req)
	}()

	select {
	case <-w.ready:
	case <-req.Context().Done():
		_ = pr.CloseWithError(req.Context().Err())
		return nil, req.Context().Err()
	}
	return &http.Response{
		Status:        http.StatusText(w.status),
		StatusCode:    w.status,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        w.sent,
		Body:          pr,
		ContentLength: -1,
		Request:       req,
	}, nil
}

// errPanic is what a panicking handler surfaces as on the node's side.
var errPanic = errors.New("nashnet: loopback handler panicked")

// loopbackWriter is the http.ResponseWriter the in-process handler writes
// through. It publishes the status and headers once, on the first Write or
// WriteHeader, and pipes every body byte to the reader the round trip returned.
type loopbackWriter struct {
	header http.Header
	body   *io.PipeWriter
	once   sync.Once
	ready  chan struct{}
	status int
	// sent is the header map as it stood when the status line was published.
	// net/http snapshots at WriteHeader too, and taking a copy is also what
	// keeps a handler that keeps mutating its header map from racing the
	// reader holding the response.
	sent http.Header
}

func (w *loopbackWriter) Header() http.Header { return w.header }

// start publishes the status line exactly once. Later calls are dropped the
// way net/http drops a second WriteHeader.
func (w *loopbackWriter) start(status int) {
	w.once.Do(func() {
		w.status = status
		w.sent = w.header.Clone()
		close(w.ready)
	})
}

func (w *loopbackWriter) WriteHeader(status int) { w.start(status) }

func (w *loopbackWriter) Write(b []byte) (int, error) {
	w.start(http.StatusOK)
	return w.body.Write(b)
}

// Flush is a no-op: the pipe hands every write straight to the reader, so
// there is nothing buffered to push. It exists because a handler may assert
// http.Flusher before streaming.
func (w *loopbackWriter) Flush() {}

// NewLoopbackClient builds the embedded node's transport: the same Client the
// remote node runs, dialing an in-process handler (D65). Every request the
// node makes is constructed, authorized, and decoded by the production code
// path; only the round tripper differs, which is the whole of what "two
// transports, one interface" is meant to buy.
func NewLoopbackClient(h http.Handler, signer *Signer) *Client {
	base, _ := url.Parse(loopbackBase)
	rt := LoopbackTransport{Handler: h}
	return &Client{
		base:   base,
		signer: signer,
		hc:     &http.Client{Transport: rt, Timeout: 120 * time.Second},
		// The held claim and events polls answer only when work or an event
		// exists, so they run without a client deadline exactly as they do
		// over HTTPS; their bound is the request context.
		longPollHTTP: &http.Client{Transport: rt},
	}
}
