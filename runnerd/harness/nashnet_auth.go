package harness

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// Per-route body deadlines (D56). A chunk append gets the long one because an
// 8 MiB body over a slow uplink is legitimate; everything else is a small JSON
// body. The egress deadline bounds a ranged snapshot or seed read, and the
// long-poll routes extend the write deadline to their own wait instead.
const (
	deadlineChunk   = 120 * time.Second
	deadlineDefault = 30 * time.Second
	deadlineEgress  = 15 * time.Minute
)

// nodeHandler is a route authorized by a node credential: the acting node id is
// derived from the verified token subject and never from a request body or a
// path segment (D25).
type nodeHandler func(w http.ResponseWriter, r *http.Request, nodeID string)

// leaseHandler is a route authorized by the opaque lease token alone (D44): no
// node token accompanies it, and the lease record names the node, the job, and
// the lease tree the handler may touch.
type leaseHandler func(w http.ResponseWriter, r *http.Request, lease nashnet.Lease)

// setDeadlines applies the per-route body and response deadlines of D56. A
// failure to set them is not fatal: the connection simply keeps the server's
// own timeouts.
func setDeadlines(w http.ResponseWriter, r *http.Request, d time.Duration) {
	rc := http.NewResponseController(w)
	now := time.Now()
	_ = rc.SetReadDeadline(now.Add(d))
	_ = rc.SetWriteDeadline(now.Add(d))
}

// requireNodeBearer verifies an aud=nashnet-node token and hands the route the
// node id its subject names (D25). It is a separate middleware from
// requireBearer: an operator token here and a node token there are both
// 403 wrong_audience, and a lease credential is refused before any JWT parsing
// because a lease route carries no node token and the reverse is equally true
// (D26, D44).
func (s *Server) requireNodeBearer(d time.Duration, next nodeHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := s.pool
		if p == nil {
			nashnetError(w, http.StatusServiceUnavailable, "pool_disabled", "nashnet is not configured on this daemon")
			return
		}
		done, ok := p.preauth.enter(r)
		if !ok {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "coordinator at its pre-authentication cap", RetryAfterSeconds: 1})
			return
		}
		defer done()

		if r.Header.Get(nashnet.HeaderLeaseToken) != "" {
			nashnetError(w, http.StatusForbidden, nashnet.CodeWrongAudience, "a lease credential does not authorize a node route")
			return
		}
		tok, ok := bearerToken(r)
		if !ok {
			nashnetError(w, http.StatusUnauthorized, "unauthorized", "missing bearer token")
			return
		}
		slot, ok := p.preauth.verifySlot()
		if !ok {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "verification queue full", RetryAfterSeconds: 1})
			return
		}
		nodeID, _, err := p.grants.VerifyNode(tok)
		slot()
		if err != nil {
			status, code := authtoken.HTTPStatus(err)
			nashnetError(w, status, code, "node credential refused")
			return
		}
		// A path-carried {node} names the acting node only when it equals the
		// verified subject; a mismatch is refused rather than served, so a node
		// holding a valid grant cannot act on another node's record (D25).
		if named := r.PathValue("node"); named != "" && named != nodeID {
			nashnetError(w, http.StatusForbidden, nashnet.CodeWrongSubject, "credential does not name this node")
			return
		}
		if !p.requests.allow(nodeID) {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "node request budget exhausted", RetryAfterSeconds: 1})
			return
		}
		if !p.conns.acquire(nodeID) {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "node connection ceiling reached", RetryAfterSeconds: 1})
			return
		}
		defer p.conns.release(nodeID)
		setDeadlines(w, r, d)
		next(w, r, nodeID)
	})
}

// requireLeaseToken authorizes a lease route by the opaque token in
// X-Nashnet-Lease and nothing else: it parses no JWT, looks the lease record
// up, compares the stored hash under crypto/subtle.ConstantTimeCompare, and
// consults the node tombstone so a revoked node is refused on its next call
// (D44, D60). An unknown lease and a dropped token are both 401, which is the
// code D36 obliges a node to treat exactly as it treats 409.
func (s *Server) requireLeaseToken(route nashnet.Route, d time.Duration, next leaseHandler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := s.pool
		if p == nil {
			nashnetError(w, http.StatusServiceUnavailable, "pool_disabled", "nashnet is not configured on this daemon")
			return
		}
		done, ok := p.preauth.enter(r)
		if !ok {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "coordinator at its pre-authentication cap", RetryAfterSeconds: 1})
			return
		}
		defer done()

		if _, hasBearer := bearerToken(r); hasBearer {
			nashnetError(w, http.StatusForbidden, nashnet.CodeWrongAudience, "a lease route takes the lease token alone")
			return
		}
		token := r.Header.Get(nashnet.HeaderLeaseToken)
		if token == "" {
			nashnetError(w, http.StatusUnauthorized, "unauthorized", "missing lease token")
			return
		}
		leaseID := r.PathValue("lease")
		if err := procmgr.ValidateName(leaseID); err != nil {
			nashnetError(w, http.StatusUnauthorized, "unauthorized", "unknown lease")
			return
		}
		lease, err := p.leases.Authorize(leaseID, token, route)
		if err != nil {
			// A result is idempotent by (job, lease, epoch), and a released
			// lease has already dropped its token, so a replay after a lost
			// response would otherwise be refused by the credential rather than
			// answered with the recorded terminal (D6). The replay is still
			// authenticated: it is admitted only against the hash of the token
			// that posted the result.
			replayed, ok := p.replayLease(leaseID, token, route)
			if !ok {
				writeLeaseAuthError(w, err)
				return
			}
			lease = replayed
		}
		if _, gerr := p.grants.Grant(lease.NodeID); gerr != nil {
			nashnetError(w, http.StatusUnauthorized, "unauthorized", "node credential refused")
			return
		}
		if !p.requests.allow(lease.NodeID) {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "node request budget exhausted", RetryAfterSeconds: 1})
			return
		}
		if !p.conns.acquire(lease.NodeID) {
			writeNashnetError(w, http.StatusTooManyRequests,
				nashnet.ErrorBody{Error: nashnet.CodeRateLimited, Detail: "node connection ceiling reached", RetryAfterSeconds: 1})
			return
		}
		defer p.conns.release(lease.NodeID)
		setDeadlines(w, r, d)
		next(w, r, lease)
	})
}

// requireOperatorNashnet gates the four operator routes under /nashnet/. It is
// requireBearer plus the lease-credential refusal, so an operator route answers
// 403 wrong_audience for a node or lease credential rather than a bare 401.
func (s *Server) requireOperatorNashnet(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.pool == nil {
			nashnetError(w, http.StatusServiceUnavailable, "pool_disabled", "nashnet is not configured on this daemon")
			return
		}
		s.requireBearer(next).ServeHTTP(w, r)
	})
}

// writeLeaseAuthError maps a lease-store refusal to its status (D4, D44).
func writeLeaseAuthError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, nashnet.ErrLeaseSuperseded):
		nashnetError(w, http.StatusConflict, nashnet.CodeLeaseSuperseded, "lease superseded")
	case errors.Is(err, nashnet.ErrUnknownLease), errors.Is(err, nashnet.ErrLeaseTokenDropped):
		nashnetError(w, http.StatusUnauthorized, "unauthorized", "lease credential refused")
	default:
		nashnetError(w, http.StatusUnauthorized, "unauthorized", "lease credential refused")
	}
}

// bearerToken extracts an Authorization: Bearer credential.
func bearerToken(r *http.Request) (string, bool) {
	const prefix = "Bearer "
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, prefix) {
		return "", false
	}
	return strings.TrimSpace(h[len(prefix):]), true
}
