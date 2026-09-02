package harness

import (
	"log"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// poolLog is the pool's single log seam, so a route never writes to stdout
// directly and a test can read the format in one place.
func poolLog(format string, args ...any) { log.Printf(format, args...) }

// itoa is the header-friendly integer render used by Retry-After and
// X-Nashnet-Offset.
func itoa(n int) string { return strconv.Itoa(n) }

// rateTable is a per-identity token bucket over a fixed window (D56). It holds
// no goroutine and no timer: a bucket refills from the injected clock on the
// call that reads it, so a test moves time by hand.
type rateTable struct {
	mu     sync.Mutex
	limit  int
	window time.Duration
	now    func() time.Time
	seen   map[string]*bucket
}

type bucket struct {
	tokens  float64
	updated time.Time
}

func newRateTable(limit int, window time.Duration, now func() time.Time) *rateTable {
	return &rateTable{limit: limit, window: window, now: now, seen: map[string]*bucket{}}
}

// allow spends one token for key and reports whether the bucket had it.
func (t *rateTable) allow(key string) bool {
	if t == nil || t.limit <= 0 {
		return true
	}
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	b := t.seen[key]
	if b == nil {
		b = &bucket{tokens: float64(t.limit), updated: now}
		t.seen[key] = b
	}
	rate := float64(t.limit) / t.window.Seconds()
	b.tokens += now.Sub(b.updated).Seconds() * rate
	if b.tokens > float64(t.limit) {
		b.tokens = float64(t.limit)
	}
	b.updated = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// byteTable is the per-node byte-rate token bucket of D56
// (RUNNERD_NASHNET_NODE_MBPS). It is rateTable's shape with bytes for tokens:
// the bucket holds one second of transfer, refills from the injected clock on
// the call that reads it, and admits a whole request or none of it, so a node
// under its rate never sees a partial transfer. Ingress (blob chunks, log
// appends) and egress (snapshot and seed reads) share one bucket per node
// because they share one link.
type byteTable struct {
	mu    sync.Mutex
	rate  float64 // bytes per second; zero disables the bucket
	burst float64
	now   func() time.Time
	seen  map[string]*bucket
}

// newByteTable returns a bucket table metering each node at mbps megabits per
// second. A non-positive rate returns a table that admits everything, which is
// the unlimited LAN default.
func newByteTable(mbps int, now func() time.Time) *byteTable {
	t := &byteTable{now: now, seen: map[string]*bucket{}}
	if mbps > 0 {
		t.rate = float64(mbps) * 125000 // megabits/s -> bytes/s
		t.burst = t.rate
	}
	return t
}

// spend charges n bytes to key and reports whether the bucket had them. A
// request larger than the whole burst is admitted once the bucket is full
// rather than refused forever, since a chunk cap above the per-second rate is a
// configuration the node cannot fix by retrying smaller.
func (t *byteTable) spend(key string, n int64) bool {
	if t == nil || t.rate <= 0 || n <= 0 {
		return true
	}
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	b := t.seen[key]
	if b == nil {
		b = &bucket{tokens: t.burst, updated: now}
		t.seen[key] = b
	}
	b.tokens += now.Sub(b.updated).Seconds() * t.rate
	if b.tokens > t.burst {
		b.tokens = t.burst
	}
	b.updated = now
	want := float64(n)
	if want > t.burst {
		want = t.burst
	}
	if b.tokens < want {
		return false
	}
	b.tokens -= want
	return true
}

// countTable bounds concurrent holders per identity: connections, uploads, and
// downloads all use it (D56).
type countTable struct {
	mu    sync.Mutex
	limit int
	held  map[string]int
}

func newCountTable(limit int) *countTable {
	return &countTable{limit: limit, held: map[string]int{}}
}

// acquire takes one slot for key, returning false when the ceiling is reached.
func (t *countTable) acquire(key string) bool {
	if t == nil || t.limit <= 0 {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.held[key] >= t.limit {
		return false
	}
	t.held[key]++
	return true
}

func (t *countTable) release(key string) {
	if t == nil || t.limit <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.held[key] > 0 {
		t.held[key]--
		if t.held[key] == 0 {
			delete(t.held, key)
		}
	}
}

// preAuthGate holds the caps that must bind before a credential is verified
// (D56, ruling q3): a node behind an arbitrary NAT is an intended peer, so an
// unauthenticated stranger is now a reachable one and every per-identity bound
// below sits behind these.
type preAuthGate struct {
	inFlight chan struct{}
	verify   chan struct{}
	source   *rateTable
}

func newPreAuthGate(c Ceilings, now func() time.Time) *preAuthGate {
	return &preAuthGate{
		inFlight: make(chan struct{}, c.GlobalInFlight),
		verify:   make(chan struct{}, c.ConcurrentVerifications),
		source:   newRateTable(c.SourceRequestsPerMinute, time.Minute, now),
	}
}

// enter admits a request before any verification runs. It returns a release
// func and false when the daemon is at its global in-flight cap or the source
// address is over its bucket.
func (g *preAuthGate) enter(r *http.Request) (func(), bool) {
	if !g.source.allow(sourceAddr(r)) {
		return func() {}, false
	}
	select {
	case g.inFlight <- struct{}{}:
		return func() { <-g.inFlight }, true
	default:
		return func() {}, false
	}
}

// verifySlot ceilings concurrent token verifications so an unauthenticated peer
// cannot force unbounded EdDSA work.
func (g *preAuthGate) verifySlot() (func(), bool) {
	select {
	case g.verify <- struct{}{}:
		return func() { <-g.verify }, true
	default:
		return func() {}, false
	}
}

// sourceAddr is the peer address the pre-auth bucket keys on, without its port
// so a client's connection churn cannot buy fresh buckets.
func sourceAddr(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// limitListener caps the accepted connections a daemon holds at once, the
// connection half of the pre-authentication caps (D56). It is the standard
// bounded-accept wrapper rather than a dependency, since runnerd links no
// golang.org/x/net.
type limitListener struct {
	net.Listener
	sem chan struct{}
}

// newLimitListener wraps l so no more than n connections are open at once. A
// non-positive n returns l unchanged.
func newLimitListener(l net.Listener, n int) net.Listener {
	if n <= 0 {
		return l
	}
	return &limitListener{Listener: l, sem: make(chan struct{}, n)}
}

func (l *limitListener) Accept() (net.Conn, error) {
	l.sem <- struct{}{}
	c, err := l.Listener.Accept()
	if err != nil {
		<-l.sem
		return nil, err
	}
	return &limitConn{Conn: c, release: func() { <-l.sem }}, nil
}

type limitConn struct {
	net.Conn
	once    sync.Once
	release func()
}

func (c *limitConn) Close() error {
	err := c.Conn.Close()
	c.once.Do(c.release)
	return err
}
