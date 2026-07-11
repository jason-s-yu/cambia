package harnessproxy

import (
	"context"
	"crypto/ed25519"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/coder/websocket"
)

// restTimeout bounds a whole control-plane REST call (mint + connect + request +
// response). Long-lived WS streams use a client with no overall timeout.
const restTimeout = 15 * time.Second

// handshakeTimeout bounds the handshake phase of every call at the transport
// level: TCP connect, TLS handshake, and (for WS) the wait for the 101 upgrade
// response. It is set on the transport rather than via a dial-context deadline
// because coder/websocket hijacks the connection; a cancelled dial context would
// tear down a live stream, so the WS read ctx (not the dial ctx) governs the
// stream and only the transport timeouts bound the handshake.
const handshakeTimeout = 15 * time.Second

// Client speaks to one runner control plane over a fingerprint-pinned TLS
// channel, minting a fresh short-lived token per call. Safe for concurrent use.
// A nil *Client is the no-proxy state; callers nil-check before use.
type Client struct {
	cfg      *Config
	key      ed25519.PrivateKey
	baseURL  *url.URL
	rest     *http.Client
	wsClient *http.Client
	now      func() time.Time // test seam for token iat/exp
}

// ProxyResponse is a runner REST reply the dashboard forwards upward: the
// runner's HTTP status and raw JSON body.
type ProxyResponse struct {
	Status int
	Body   []byte
}

// New builds a Client from a validated Config: it loads the ed25519 signing key
// and constructs the pinned TLS transport shared by the REST and WS clients.
func New(cfg *Config) (*Client, error) {
	key, err := LoadEd25519PrivateKey(cfg.PrivateKeyPath)
	if err != nil {
		return nil, err
	}
	base, err := url.Parse(cfg.RunnerURL)
	if err != nil {
		return nil, fmt.Errorf("harnessproxy: parse runner url: %w", err)
	}
	tlsCfg, err := pinnedTLSConfig(cfg.CertFingerprint)
	if err != nil {
		return nil, err
	}
	tr := &http.Transport{
		TLSClientConfig:       tlsCfg,
		DialContext:           (&net.Dialer{Timeout: handshakeTimeout}).DialContext,
		TLSHandshakeTimeout:   handshakeTimeout,
		ResponseHeaderTimeout: handshakeTimeout,
	}
	return &Client{
		cfg:      cfg,
		key:      key,
		baseURL:  base,
		rest:     &http.Client{Timeout: restTimeout, Transport: tr},
		wsClient: &http.Client{Transport: tr}, // no overall timeout: streams outlive it
		now:      time.Now,
	}, nil
}

// OriginHost is the host the runner stamps onto reconciled runs. A remote run is
// controllable from this dashboard only when its Host equals this value.
func (c *Client) OriginHost() string {
	if c == nil {
		return ""
	}
	return c.cfg.OriginHost
}

// mintToken mints a fresh token for one call under DashboardSubject (audit
// separation; runnerd does not enforce sub). Tokens are never persisted.
func (c *Client) mintToken() (string, error) {
	return MintToken(c.key, DashboardSubject, c.cfg.TokenTTL, c.now().UTC())
}

// Stop cancels a runner job: DELETE /harness/jobs/{name}, with ?force=true for a
// SIGKILL, else SIGINT + grace (design 4.5). Returns the runner's status/body,
// or a transport error (errors.Is ErrPinMismatch on a pin failure).
func (c *Client) Stop(ctx context.Context, name string, force bool) (*ProxyResponse, error) {
	q := ""
	if force {
		q = "?force=true"
	}
	return c.do(ctx, http.MethodDelete, "/harness/jobs/"+name+q)
}

// Resume re-enqueues a stopped runner job: POST /harness/jobs/{name}/resume.
func (c *Client) Resume(ctx context.Context, name string) (*ProxyResponse, error) {
	return c.do(ctx, http.MethodPost, "/harness/jobs/"+name+"/resume")
}

// do issues one pinned, token-authenticated request and reads the full body.
func (c *Client) do(ctx context.Context, method, path string) (*ProxyResponse, error) {
	tok, err := c.mintToken()
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL.String()+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Accept", "application/json")
	resp, err := c.rest.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	return &ProxyResponse{Status: resp.StatusCode, Body: body}, nil
}

// DialLogs opens the pinned wss log stream for a job:
// wss://<runner>/ws/harness/jobs/{name}/logs, with a Bearer token. The runner URL
// is https:// (config-validated), so the stream is always wss. The handshake is
// bounded by the transport timeouts; the returned connection is not (a multi-hour
// tail is expected). A pin mismatch surfaces as an error satisfying errors.Is
// ErrPinMismatch with no application bytes sent. The caller reads text frames and
// closes the connection. ctx governs the stream's lifetime (not the handshake).
func (c *Client) DialLogs(ctx context.Context, name string) (*websocket.Conn, *http.Response, error) {
	tok, err := c.mintToken()
	if err != nil {
		return nil, nil, err
	}
	wsURL := "wss://" + c.baseURL.Host + "/ws/harness/jobs/" + name + "/logs"
	return websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPClient: c.wsClient,
		HTTPHeader: http.Header{"Authorization": {"Bearer " + tok}},
	})
}
