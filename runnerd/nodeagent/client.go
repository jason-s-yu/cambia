package nodeagent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// HeaderBlobOffset is the resume-probe header a HEAD on a blob answers with
// (D50 step 2).
const HeaderBlobOffset = "X-Nashnet-Offset"

// ErrCertificatePin is the handshake failure a mismatched coordinator
// certificate produces (D27). It fires inside VerifyPeerCertificate, so it
// aborts the TLS handshake and no request line or body ever reaches the wire.
var ErrCertificatePin = errors.New("nashnet: coordinator certificate fingerprint mismatch")

// ProbeRequest is the body of POST /nashnet/leases/{lease}/blobs/probe. The
// design fixes the answer shape ({have[], want[]}) and describes the request
// only as "a digest list"; this is the node's spelling of it.
type ProbeRequest struct {
	Digests []string `json:"digests"`
}

// ProbeResponse is the answer: have is the lease's provable set, want is
// everything else (D50 step 1).
type ProbeResponse struct {
	Have []string `json:"have"`
	Want []string `json:"want"`
}

// ChunkResponse is the answer to a blob PATCH (D50 step 3).
type ChunkResponse struct {
	CommittedOffset int64 `json:"committed_offset"`
}

// ManifestHead is the answer to GET /nashnet/leases/{lease}/manifest: the
// coordinator's current head, which the node diffs its run dir against (D50).
// The design names {seq, digest, entries}; Final is carried too because the
// node's own restart path wants to know whether it already committed the
// final manifest before it re-posts a result.
type ManifestHead struct {
	Seq     int64              `json:"seq"`
	Digest  string             `json:"digest"`
	Entries []quarantine.Entry `json:"entries"`
	Final   bool               `json:"final,omitempty"`
}

// APIError is any non-2xx answer, decoded into the error body every nashnet
// route shares. Status and Code together are what the node classifies on: D36
// keys on 401 and on 409 lease_superseded, the uploader keys on 409
// blobs_missing and 409 manifest_out_of_order, and the log tail keys on 409
// offset_mismatch plus the true offset the body carries.
type APIError struct {
	Status            int
	Code              string
	Detail            string
	Offset            int64
	RetryAfterSeconds int
	Missing           []string
	// Head carries the coordinator's current {seq, digest} on a
	// manifest_out_of_order refusal so the node re-diffs without a second
	// round trip.
	Head *ManifestHead
}

func (e *APIError) Error() string {
	if e.Code == "" {
		return fmt.Sprintf("nashnet: http %d", e.Status)
	}
	if e.Detail == "" {
		return fmt.Sprintf("nashnet: http %d %s", e.Status, e.Code)
	}
	return fmt.Sprintf("nashnet: http %d %s: %s", e.Status, e.Code, e.Detail)
}

// errorBody is the decode target: the shared nashnet.ErrorBody plus the two
// route-specific extras (blobs_missing's list, manifest_out_of_order's head).
type errorBody struct {
	nashnet.ErrorBody
	Missing []string      `json:"missing,omitempty"`
	Head    *ManifestHead `json:"head,omitempty"`
	Seq     *int64        `json:"seq,omitempty"`
	Digest  string        `json:"digest,omitempty"`
}

// IsFenced reports whether err is the D36 stop signal: a lease route answering
// 401 (the token was dropped: the grace expired, node_epoch bumped, or the
// node was revoked) or 409 lease_superseded (the lease was reassigned). Both
// oblige the node to kill its job and stop uploading; keying on the 409 alone
// would miss the very revocation the rule exists for.
func IsFenced(err error) bool {
	var api *APIError
	if !errors.As(err, &api) {
		return false
	}
	return api.Status == http.StatusUnauthorized ||
		(api.Status == http.StatusConflict && api.Code == nashnet.CodeLeaseSuperseded)
}

// StatusIs reports whether err is an APIError with the given status.
func StatusIs(err error, status int) bool {
	var api *APIError
	return errors.As(err, &api) && api.Status == status
}

// CodeIs reports whether err is an APIError carrying the given error code.
func CodeIs(err error, code string) bool {
	var api *APIError
	return errors.As(err, &api) && api.Code == code
}

// AsAPIError returns err as an *APIError when it is one.
func AsAPIError(err error) (*APIError, bool) {
	var api *APIError
	ok := errors.As(err, &api)
	return api, ok
}

// Client is the node's half of the transport: one fingerprint-pinned HTTPS
// connection pool to one coordinator, with the node token minted per request
// and the lease token carried in its own header (D44).
type Client struct {
	base   *url.URL
	hc     *http.Client
	signer *Signer
	// longPollHTTP is a second client with no response-header timeout, used by
	// the held claim and events requests (D2, D45), which are answered only
	// when work or an event exists.
	longPollHTTP *http.Client
}

// NewClient builds the pinned client of D27: InsecureSkipVerify with a
// VerifyPeerCertificate that compares sha256(rawCerts[0]) against
// coordinator.cert_sha256 and fails the handshake on mismatch. The two halves
// are not separable, which is why they are built together here and the
// fingerprint is validated at config load.
func NewClient(cfg Config, signer *Signer) (*Client, error) {
	base, err := url.Parse(cfg.Coordinator.URL)
	if err != nil {
		return nil, fmt.Errorf("%w: coordinator.url: %v", ErrConfig, err)
	}
	if base.Scheme != "https" {
		return nil, ErrInsecureCoordinator
	}
	pinned, err := hex.DecodeString(cfg.Coordinator.CertSHA256)
	if err != nil || len(pinned) != sha256.Size {
		return nil, fmt.Errorf("%w: coordinator.cert_sha256 is not a sha256", ErrConfig)
	}
	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   15 * time.Second,
		MaxIdleConnsPerHost:   8,
		IdleConnTimeout:       90 * time.Second,
		ExpectContinueTimeout: time.Second,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
			// The coordinator serves a self-signed certificate, so the chain
			// is not verifiable against a CA set; the fingerprint compare
			// below is the whole of the authentication and is mandatory.
			InsecureSkipVerify:    true,
			VerifyPeerCertificate: pinVerifier(pinned),
		},
	}
	return &Client{
		base:         base,
		signer:       signer,
		hc:           &http.Client{Transport: transport, Timeout: 120 * time.Second},
		longPollHTTP: &http.Client{Transport: transport},
	}, nil
}

// pinVerifier returns the VerifyPeerCertificate callback: it hashes the leaf's
// raw DER and compares it against the pinned fingerprint in constant time. It
// reads only rawCerts[0], so it is independent of chain building and of the
// system trust store, and it runs during the handshake, so a mismatch aborts
// the connection before any request line or body is written.
func pinVerifier(pinned []byte) func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
	return func(rawCerts [][]byte, _ [][]*x509.Certificate) error {
		if len(rawCerts) == 0 {
			return fmt.Errorf("%w: peer presented no certificate", ErrCertificatePin)
		}
		sum := sha256.Sum256(rawCerts[0])
		if !constantTimeEqual(sum[:], pinned) {
			return fmt.Errorf("%w: got %s", ErrCertificatePin, hex.EncodeToString(sum[:]))
		}
		return nil
	}
}

// Register declares this node and returns the pool policy and node epoch (D3).
func (c *Client) Register(ctx context.Context, req nashnet.RegisterRequest) (nashnet.RegisterResponse, error) {
	var out nashnet.RegisterResponse
	err := c.nodeJSON(ctx, http.MethodPost, "/nashnet/nodes/register", req, &out, c.hc)
	return out, err
}

// Heartbeat refreshes the volatile facts and the gate report while idle (D3).
func (c *Client) Heartbeat(ctx context.Context, nodeID string, req nashnet.HeartbeatRequest) (nashnet.HeartbeatResponse, error) {
	var out nashnet.HeartbeatResponse
	err := c.nodeJSON(ctx, http.MethodPost, "/nashnet/nodes/"+nodeID+"/heartbeat", req, &out, c.hc)
	return out, err
}

// Events holds the back-channel long poll (D45). The node keeps exactly one
// of these open at all times and re-issues it on every response.
func (c *Client) Events(ctx context.Context, nodeID string, waitSeconds int) (nashnet.EventsResponse, error) {
	var out nashnet.EventsResponse
	path := fmt.Sprintf("/nashnet/nodes/%s/events?wait_seconds=%d", nodeID, waitSeconds)
	err := c.nodeJSON(ctx, http.MethodGet, path, nil, &out, c.longPollHTTP)
	return out, err
}

// Claim holds the claim long poll (D2). A 200 returns the claim response and a
// nil hold; a 204 returns a nil claim and the hold telling the node how long to
// wait and why.
func (c *Client) Claim(ctx context.Context, req nashnet.ClaimRequest) (*nashnet.ClaimResponse, *nashnet.ClaimHold, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, nil, err
	}
	httpReq, err := c.newRequest(ctx, http.MethodPost, "/nashnet/claim", bytes.NewReader(body))
	if err != nil {
		return nil, nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if err := c.authorize(httpReq); err != nil {
		return nil, nil, err
	}
	resp, err := c.longPollHTTP.Do(httpReq)
	if err != nil {
		return nil, nil, err
	}
	defer drainClose(resp)

	switch resp.StatusCode {
	case http.StatusOK:
		var out nashnet.ClaimResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return nil, nil, fmt.Errorf("decode claim: %w", err)
		}
		return &out, nil, nil
	case http.StatusNoContent, http.StatusAccepted:
		// A 204 carries no body by the HTTP rules the design's own example
		// bends; the hold and its retry_after_seconds are read from the body
		// when one is present and defaulted otherwise, so a strict 204 and the
		// design's illustrated body both work.
		hold := nashnet.ClaimHold{Hold: nashnet.HoldNoMatch}
		_ = json.NewDecoder(resp.Body).Decode(&hold)
		if hold.RetryAfterSeconds <= 0 {
			hold.RetryAfterSeconds = retryAfterHeader(resp, defaultClaimBackoffSeconds)
		}
		return nil, &hold, nil
	default:
		return nil, nil, apiError(resp)
	}
}

// defaultClaimBackoffSeconds is what a hold with no stated retry_after falls
// back to.
const defaultClaimBackoffSeconds = 15

// Progress renews the lease and drives the coordinator's projection (D5).
func (c *Client) Progress(ctx context.Context, leaseID, leaseToken string, req nashnet.ProgressRequest) (nashnet.ProgressResponse, error) {
	var out nashnet.ProgressResponse
	err := c.leaseJSON(ctx, http.MethodPost, "/nashnet/leases/"+leaseID+"/progress", leaseToken, req, &out)
	return out, err
}

// Nack returns a claim the node cannot honor (D8).
func (c *Client) Nack(ctx context.Context, leaseID, leaseToken string, req nashnet.NackRequest) error {
	return c.leaseJSON(ctx, http.MethodPost, "/nashnet/leases/"+leaseID+"/nack", leaseToken, req, nil)
}

// Result posts the terminal commit (D6).
func (c *Client) Result(ctx context.Context, leaseID, leaseToken string, req nashnet.ResultRequest) (nashnet.ResultResponse, error) {
	var out nashnet.ResultResponse
	err := c.leaseJSON(ctx, http.MethodPost, "/nashnet/leases/"+leaseID+"/result", leaseToken, req, &out)
	return out, err
}

// Probe asks which of the digests the lease can already prove (D50 step 1).
func (c *Client) Probe(ctx context.Context, leaseID, leaseToken string, digests []string) (ProbeResponse, error) {
	var out ProbeResponse
	err := c.leaseJSON(ctx, http.MethodPost, "/nashnet/leases/"+leaseID+"/blobs/probe", leaseToken,
		ProbeRequest{Digests: digests}, &out)
	return out, err
}

// BlobOffset is the resume probe: the size of the in-flight part, 0 when none,
// and the full size when the blob is already verified (D50 step 2).
func (c *Client) BlobOffset(ctx context.Context, leaseID, leaseToken, digest string) (int64, error) {
	req, err := c.newRequest(ctx, http.MethodHead, "/nashnet/leases/"+leaseID+"/blobs/"+digest, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(nashnet.HeaderLeaseToken, leaseToken)
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer drainClose(resp)
	if resp.StatusCode/100 != 2 {
		return 0, apiError(resp)
	}
	n, err := strconv.ParseInt(resp.Header.Get(HeaderBlobOffset), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("blob offset: %s header %q: %w", HeaderBlobOffset, resp.Header.Get(HeaderBlobOffset), err)
	}
	return n, nil
}

// AppendChunk uploads one chunk at an exact offset (D50 step 3). A zero-byte
// artifact has no range to send and is a bodyless PATCH with total 0.
func (c *Client) AppendChunk(ctx context.Context, leaseID, leaseToken, digest string, start, end, total int64, chunk []byte) (int64, error) {
	var body io.Reader
	if total > 0 {
		body = bytes.NewReader(chunk)
	}
	req, err := c.newRequest(ctx, http.MethodPatch, "/nashnet/leases/"+leaseID+"/blobs/"+digest, body)
	if err != nil {
		return 0, err
	}
	req.Header.Set(nashnet.HeaderLeaseToken, leaseToken)
	if total > 0 {
		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, total))
		req.ContentLength = int64(len(chunk))
	} else {
		req.Header.Set("Content-Range", "bytes */0")
		req.ContentLength = 0
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer drainClose(resp)
	if resp.StatusCode/100 != 2 {
		return 0, apiError(resp)
	}
	var out ChunkResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil && !errors.Is(err, io.EOF) {
		return 0, fmt.Errorf("decode chunk response: %w", err)
	}
	return out.CommittedOffset, nil
}

// ManifestHead reads the coordinator's current folded head (D51).
func (c *Client) ManifestHead(ctx context.Context, leaseID, leaseToken string) (ManifestHead, error) {
	var out ManifestHead
	err := c.leaseJSON(ctx, http.MethodGet, "/nashnet/leases/"+leaseID+"/manifest", leaseToken, nil, &out)
	return out, err
}

// Commit posts a fast-forward manifest commit (D51).
func (c *Client) Commit(ctx context.Context, leaseID, leaseToken string, req quarantine.CommitRequest) (quarantine.CommitResponse, error) {
	var out quarantine.CommitResponse
	err := c.leaseJSON(ctx, http.MethodPost, "/nashnet/leases/"+leaseID+"/manifest", leaseToken, req, &out)
	return out, err
}

// AppendLog appends raw log bytes at an exact offset (D54). A 409
// offset_mismatch carries the true offset so the node seeks.
func (c *Client) AppendLog(ctx context.Context, leaseID, leaseToken string, offset int64, body []byte) error {
	path := fmt.Sprintf("/nashnet/leases/%s/logs?offset=%d", leaseID, offset)
	req, err := c.newRequest(ctx, http.MethodPost, path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set(nashnet.HeaderLeaseToken, leaseToken)
	req.Header.Set("Content-Type", "application/octet-stream")
	req.ContentLength = int64(len(body))
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer drainClose(resp)
	if resp.StatusCode/100 != 2 {
		return apiError(resp)
	}
	return nil
}

// Download streams a lease-scoped GET (the snapshot bundle of D48 or one seed
// entry of D53) into w, resuming at offset with a Range header. rawPath is a
// coordinator-relative path; an absolute URL is refused so a claim response
// can never steer a node at a third host.
func (c *Client) Download(ctx context.Context, leaseToken, rawPath string, offset int64, w io.Writer) (int64, error) {
	if !strings.HasPrefix(rawPath, "/") {
		return 0, fmt.Errorf("nashnet: download path %q is not coordinator-relative", rawPath)
	}
	req, err := c.newRequest(ctx, http.MethodGet, rawPath, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(nashnet.HeaderLeaseToken, leaseToken)
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer drainClose(resp)
	switch resp.StatusCode {
	case http.StatusOK:
		if offset > 0 {
			// The coordinator ignored the Range and is replaying from zero;
			// the caller's writer is positioned at offset, so restarting is
			// the only correct answer.
			return 0, errRangeIgnored
		}
	case http.StatusPartialContent:
	default:
		return 0, apiError(resp)
	}
	return io.Copy(w, resp.Body)
}

// errRangeIgnored tells the fetch loop to truncate and start over.
var errRangeIgnored = errors.New("nashnet: coordinator ignored the range request")

// newRequest builds a request against the coordinator base URL.
func (c *Client) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	return http.NewRequestWithContext(ctx, method, c.base.String()+path, body)
}

// authorize mints and attaches a fresh node-audience bearer token.
func (c *Client) authorize(req *http.Request) error {
	tok, err := c.signer.Token()
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	return nil
}

// nodeJSON performs a node-credentialed JSON round trip.
func (c *Client) nodeJSON(ctx context.Context, method, path string, in, out any, hc *http.Client) error {
	req, err := c.jsonRequest(ctx, method, path, in)
	if err != nil {
		return err
	}
	if err := c.authorize(req); err != nil {
		return err
	}
	return c.do(req, out, hc)
}

// leaseJSON performs a lease-credentialed JSON round trip. A lease route
// requires the lease token alone and carries no node token (D26, D44).
func (c *Client) leaseJSON(ctx context.Context, method, path, leaseToken string, in, out any) error {
	req, err := c.jsonRequest(ctx, method, path, in)
	if err != nil {
		return err
	}
	req.Header.Set(nashnet.HeaderLeaseToken, leaseToken)
	return c.do(req, out, c.hc)
}

func (c *Client) jsonRequest(ctx context.Context, method, path string, in any) (*http.Request, error) {
	var body io.Reader
	if in != nil {
		encoded, err := json.Marshal(in)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(encoded)
	}
	req, err := c.newRequest(ctx, method, path, body)
	if err != nil {
		return nil, err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")
	return req, nil
}

func (c *Client) do(req *http.Request, out any, hc *http.Client) error {
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	defer drainClose(resp)
	if resp.StatusCode/100 != 2 {
		return apiError(resp)
	}
	if out == nil {
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("decode %s: %w", req.URL.Path, err)
	}
	return nil
}

// apiError decodes a non-2xx answer. A body that is not the shared error shape
// still yields a typed error carrying the status, so status-only classification
// (D36's 401) never depends on the coordinator having sent a body.
func apiError(resp *http.Response) error {
	out := &APIError{Status: resp.StatusCode}
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var body errorBody
	if err := json.Unmarshal(raw, &body); err == nil {
		out.Code = body.Error
		out.Detail = body.Detail
		out.Offset = body.Offset
		out.RetryAfterSeconds = body.RetryAfterSeconds
		out.Missing = body.Missing
		switch {
		case body.Head != nil:
			out.Head = body.Head
		case body.Seq != nil:
			out.Head = &ManifestHead{Seq: *body.Seq, Digest: body.Digest}
		}
	}
	if out.RetryAfterSeconds == 0 {
		out.RetryAfterSeconds = retryAfterHeader(resp, 0)
	}
	return out
}

// retryAfterHeader reads a Retry-After seconds value, falling back to def.
func retryAfterHeader(resp *http.Response, def int) int {
	if v := resp.Header.Get("Retry-After"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// drainClose consumes and closes a response body so the connection returns to
// the pool instead of being torn down after every call.
func drainClose(resp *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	_ = resp.Body.Close()
}

// constantTimeEqual is the fingerprint compare, kept in one place so the pin
// never degrades to bytes.Equal.
func constantTimeEqual(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}
