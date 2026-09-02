// Package harness is the runnerd control-plane core: the in-memory FIFO job
// queue, the dispatcher and job state machine layered over procmgr, and the
// HTTPS+Bearer-JWT API (design 2.2-2.4, 5). It launches jobs through the
// injected Environment (the M3 ingest boundary) and never serves plaintext or
// accepts cookies.
package harness

import (
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
	"github.com/jason-s-yu/cambia/runnerd/sysprobe"
)

// ServerConfig configures NewServer. Dispatcher and Verifier are required.
type ServerConfig struct {
	Dispatcher    *Dispatcher
	Verifier      *authtoken.Verifier
	RunsDir       string
	AllowedOrigin string // single allowed WS Origin (design 5.3); never wildcard
	MinFreeRAMGB  float64
	MinFreeDiskGB float64
	Algos         map[string][]string // kind allowlist (design 2.6)
	// AllowedDevices is the per-runner device capability gate (cambia-329):
	// a job whose device is not a key here is rejected at submit as
	// device_unsupported, not forceable. Defaults to cpu-only.
	AllowedDevices map[string]bool
	// BuildCommit is the daemon's source commit, stamped at link time and
	// echoed by GET /harness/health so an operator can tell which binary is
	// serving. Empty renders as "dev".
	BuildCommit string
	// KillJobsOnStop mirrors the daemon's RUNNERD_KILL_JOBS_ON_STOP override
	// (cambia-655): when set, SIGTERM kills the job process groups instead of
	// detaching, so health reports restart_preserves_jobs=false. The zero value
	// is the default job-preserving restart, which is what health reports for
	// any caller that does not set it.
	KillJobsOnStop bool
	// GPUQuery/RAMQuery/RenderNodeGlob/XPUQuery are seams so tests inject
	// preflight inputs without touching real hardware.
	GPUQuery       procmgr.GPUQueryFunc
	RAMQuery       RAMQueryFunc
	RenderNodeGlob procmgr.RenderNodeGlobFunc
	XPUQuery       procmgr.XPUQueryFunc
}

// ParseAllowedDevices parses a comma-separated RUNNERD_ALLOWED_DEVICES value
// into the capability-gate set (cambia-329). Blank segments are dropped; a
// wholly blank or empty input defaults to cpu-only, matching the v1 cpu-only
// runner behavior when the env var is unset.
func ParseAllowedDevices(raw string) map[string]bool {
	out := map[string]bool{}
	for _, d := range strings.Split(raw, ",") {
		d = strings.TrimSpace(d)
		if d != "" {
			out[d] = true
		}
	}
	if len(out) == 0 {
		out["cpu"] = true
	}
	return out
}

// Server serves the control-plane API for a Dispatcher.
type Server struct {
	disp           *Dispatcher
	verifier       *authtoken.Verifier
	runsDir        string
	allowedOrigin  string
	minRAMGB       float64
	minDiskGB      float64
	minVRAMGB      float64
	algos          map[string][]string
	allowedDevices map[string]bool
	buildCommit    string
	killJobsOnStop bool
	gpuQuery       procmgr.GPUQueryFunc
	ramQuery       RAMQueryFunc
	renderNodeGlob procmgr.RenderNodeGlobFunc
	xpuQuery       procmgr.XPUQueryFunc
	// pool is the nashnet coordinator. Nil is the v1.0 surface: no /nashnet/
	// route is registered and nothing else changes (D39).
	pool *Pool
}

// NewServer builds the control-plane server. It fills preflight floors and the
// GPU/RAM query seams with defaults. It returns an error if the dispatcher or
// verifier is missing (the daemon must not serve without JWT verification).
func NewServer(cfg ServerConfig) (*Server, error) {
	if cfg.Dispatcher == nil {
		return nil, errors.New("nil dispatcher")
	}
	if cfg.Verifier == nil {
		return nil, errors.New("nil verifier: JWT verification is mandatory")
	}
	minRAM := cfg.MinFreeRAMGB
	if minRAM <= 0 {
		minRAM = DefaultMinFreeRAMGB
	}
	minDisk := cfg.MinFreeDiskGB
	if minDisk <= 0 {
		minDisk = DefaultMinFreeDiskGB
	}
	algos := cfg.Algos
	if algos == nil {
		algos = HarnessAlgorithms()
	}
	gq := cfg.GPUQuery
	if gq == nil {
		gq = procmgr.DefaultGPUQuery
	}
	rq := cfg.RAMQuery
	if rq == nil {
		rq = DefaultRAMQuery
	}
	devices := cfg.AllowedDevices
	if len(devices) == 0 {
		devices = map[string]bool{"cpu": true}
	}
	rng := cfg.RenderNodeGlob
	if rng == nil {
		rng = procmgr.DefaultRenderNodeGlob
	}
	xq := cfg.XPUQuery
	if xq == nil {
		xq = procmgr.DefaultXPUQuery
	}
	commit := cfg.BuildCommit
	if commit == "" {
		commit = "dev"
	}
	return &Server{
		disp:           cfg.Dispatcher,
		verifier:       cfg.Verifier,
		runsDir:        cfg.RunsDir,
		allowedOrigin:  cfg.AllowedOrigin,
		minRAMGB:       minRAM,
		minDiskGB:      minDisk,
		minVRAMGB:      procmgr.DefaultMinVRAMGB,
		algos:          algos,
		allowedDevices: devices,
		buildCommit:    commit,
		killJobsOnStop: cfg.KillJobsOnStop,
		gpuQuery:       gq,
		ramQuery:       rq,
		renderNodeGlob: rng,
		xpuQuery:       xq,
	}, nil
}

// Handler returns the routed control-plane handler with Bearer auth on every
// route except GET /harness/health, which serves read-only capacity counters
// token-free for LAN monitoring (engelbart tile; cambia-330/network-552:
// reachability is already LAN-scoped by the host firewall). Routes use
// go1.22+ method+path patterns; {id} is the validated run name.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	a := s.requireBearer

	mux.Handle("POST /harness/jobs", a(http.HandlerFunc(s.handleCreateJob)))
	mux.Handle("GET /harness/jobs", a(http.HandlerFunc(s.handleListJobs)))
	mux.Handle("GET /harness/jobs/{id}", a(http.HandlerFunc(s.handleGetJob)))
	mux.Handle("DELETE /harness/jobs/{id}", a(http.HandlerFunc(s.handleDeleteJob)))
	mux.Handle("POST /harness/jobs/{id}/resume", a(http.HandlerFunc(s.handleResumeJob)))
	mux.Handle("GET /harness/jobs/{id}/artifacts", a(http.HandlerFunc(s.handleArtifacts)))
	mux.Handle("POST /harness/jobs/{id}/rundb-checkpoint", a(http.HandlerFunc(s.handleRunDBCheckpoint)))
	mux.Handle("GET /harness/health", http.HandlerFunc(s.handleHealth))

	mux.Handle("GET /ws/harness/queue", a(http.HandlerFunc(s.handleQueueWS)))
	mux.Handle("GET /ws/harness/jobs/{id}/logs", a(http.HandlerFunc(s.handleLogsWS)))

	s.registerNashnet(mux)
	return mux
}

// registerNashnet adds the twenty pool routes beside the v1.0 table (D39). They
// are additive: every existing route keeps its path, method, semantics, and
// status codes, and a daemon with no pool attached registers none of these, so
// its surface is byte-for-byte the v1.0 one.
func (s *Server) registerNashnet(mux *http.ServeMux) {
	if s.pool == nil {
		return
	}
	node := func(h nodeHandler) http.Handler { return s.requireNodeBearer(deadlineDefault, h) }
	lease := func(route nashnet.Route, h leaseHandler) http.Handler {
		return s.requireLeaseToken(route, deadlineDefault, h)
	}
	op := func(h http.HandlerFunc) http.Handler { return s.requireOperatorNashnet(h) }

	mux.Handle("POST /nashnet/nodes/register", node(s.handleNodeRegister))
	mux.Handle("POST /nashnet/nodes/{node}/heartbeat", node(s.handleNodeHeartbeat))
	mux.Handle("GET /nashnet/nodes/{node}/events", node(s.handleNodeEvents))
	mux.Handle("POST /nashnet/claim", node(s.handleClaim))

	mux.Handle("GET /nashnet/leases/{lease}/snapshot", lease(nashnet.RouteSnapshot, s.handleSnapshot))
	mux.Handle("GET /nashnet/leases/{lease}/seeds/{seed_id}/{path...}", lease(nashnet.RouteSeeds, s.handleSeed))
	mux.Handle("POST /nashnet/leases/{lease}/progress", lease(nashnet.RouteProgress, s.handleProgress))
	mux.Handle("POST /nashnet/leases/{lease}/logs", lease(nashnet.RouteLogs, s.handleLogs))
	mux.Handle("POST /nashnet/leases/{lease}/blobs/probe", lease(nashnet.RouteBlobs, s.handleBlobProbe))
	mux.Handle("HEAD /nashnet/leases/{lease}/blobs/{digest}", lease(nashnet.RouteBlobs, s.handleBlobHead))
	mux.Handle("PATCH /nashnet/leases/{lease}/blobs/{digest}",
		s.requireLeaseToken(nashnet.RouteBlobs, deadlineChunk, s.handleBlobPatch))
	mux.Handle("DELETE /nashnet/leases/{lease}/blobs/{digest}", lease(nashnet.RouteBlobs, s.handleBlobDelete))
	// The manifest GET carries its own route because the grace admits it while
	// refusing the non-final commit on the same path: the final commit the
	// grace exists for reads this head to fast-forward from (D4, D62).
	mux.Handle("GET /nashnet/leases/{lease}/manifest", lease(nashnet.RouteManifestHead, s.handleManifestGet))
	// The manifest POST admits the state a final commit is granted, because
	// whether this one is final is in a body the middleware has not read. The
	// per-body gate is the handler's own fence: a non-final commit on a
	// revoking lease is refused there, and the final commit the grace exists
	// for gets through (D4, D31).
	mux.Handle("POST /nashnet/leases/{lease}/manifest", lease(nashnet.RouteManifestFinal, s.handleManifestPost))
	mux.Handle("POST /nashnet/leases/{lease}/nack", lease(nashnet.RouteNack, s.handleNack))
	mux.Handle("POST /nashnet/leases/{lease}/result", lease(nashnet.RouteResult, s.handleResult))

	mux.Handle("GET /nashnet/nodes", op(s.handleNodesList))
	mux.Handle("GET /nashnet/nodes/{node}", op(s.handleNodeGet))
	mux.Handle("POST /nashnet/nodes/{node}/drain", op(s.handleNodeDrain))
	mux.Handle("POST /nashnet/nodes/{node}/revoke", op(s.handleNodeRevoke))
}

// AttachPool wires a coordinator pool into an already-built server, so the
// daemon constructs the server, the pool, and the dispatcher in one direction
// and the routes appear only once every store exists.
func (s *Server) AttachPool(p *Pool) { s.pool = p }

// ListenAndServeTLS serves the control plane over HTTPS only (design 5.1). There
// is no plaintext listener: callers pass the cert and key, and a missing pair is
// a hard startup failure surfaced by http.Server.ListenAndServeTLS.
func (s *Server) ListenAndServeTLS(addr, certFile, keyFile string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// IdleTimeout bounds a kept-alive connection a peer holds without
		// sending anything, which is the pre-authentication exposure ruling q3
		// opened by making a node behind an arbitrary NAT an intended peer
		// (D56). ReadTimeout and WriteTimeout stay unset here deliberately: the
		// two WebSocket streams and the two long polls are open-ended by
		// design, so a whole-server deadline would cut a running log tail. The
		// per-route deadlines of setDeadlines carry the bound instead, and the
		// connection cap below bounds what an unauthenticated peer can hold.
		IdleTimeout: 120 * time.Second,
	}
	if s.pool == nil {
		return srv.ListenAndServeTLS(certFile, keyFile)
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return srv.ServeTLS(newLimitListener(ln, s.pool.ceilings.GlobalConnections), certFile, keyFile)
}

// requireBearer verifies an Authorization: Bearer <jwt> header on every request.
// No cookies are consulted anywhere (design 2.4/5.3: kills the CSWSH class).
func (s *Server) requireBearer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		const prefix = "Bearer "
		h := r.Header.Get("Authorization")
		if !strings.HasPrefix(h, prefix) {
			writeJSONError(w, http.StatusUnauthorized, "unauthorized", "missing bearer token")
			return
		}
		tok := strings.TrimSpace(h[len(prefix):])
		if _, err := s.verifier.Verify(tok); err != nil {
			// A node or enrollment credential is real but not this route's, so
			// it answers 403 wrong_audience rather than a bare 401; everything
			// else stays 401, including a revoked or unknown node, so a caller
			// cannot use the status to enumerate ids (D26).
			status, code := authtoken.HTTPStatus(err)
			writeJSONError(w, status, code, "invalid token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// checkOrigin enforces the single configured allowed WS origin (design 5.3),
// exact match, never wildcard. The Origin header is a browser artifact: a
// request without one comes from a non-browser client (the CLI) and is
// admitted on Bearer auth alone, which is the CSWSH threat model; a request
// that carries one must match exactly, and an unset allowed origin fails
// closed for any browser-originated request.
func (s *Server) checkOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true
	}
	if s.allowedOrigin == "" {
		return false
	}
	return origin == s.allowedOrigin
}

// originHostPattern extracts the host of the configured origin for the
// coder/websocket OriginPatterns defense-in-depth check.
func (s *Server) originHostPattern() string {
	if u, err := url.Parse(s.allowedOrigin); err == nil && u.Host != "" {
		return u.Host
	}
	return s.allowedOrigin
}

// diskFreeGB returns the unprivileged-available space in GiB on the filesystem
// backing path (Bavail, matching procmgr.DiskSpaceCheck semantics). Moved to
// runnerd/sysprobe (design D46); this forwarder keeps the name so callers in
// this package need no change.
func diskFreeGB(path string) float64 {
	return sysprobe.DiskFreeGB(path)
}

// writeJSON writes v as JSON with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeJSONError writes {error, detail} with the given status.
func writeJSONError(w http.ResponseWriter, status int, code, detail string) {
	writeJSON(w, status, map[string]string{"error": code, "detail": detail})
}

// decodeJSON decodes an optional JSON body, tolerating an empty body and
// preserving numeric override values as json.Number.
func decodeJSON(r *http.Request, v any) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	return nil
}
