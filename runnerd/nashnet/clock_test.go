package nashnet

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

// fakeClock is the injected clock every lease test runs on: no test sleeps, and
// an expiry is a clock move rather than a wait.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newClock(t *testing.T, stamp string) *fakeClock {
	t.Helper()
	return &fakeClock{t: mustTime(t, stamp)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func mustTime(t *testing.T, stamp string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339Nano, stamp)
	if err != nil {
		t.Fatalf("parse %q: %v", stamp, err)
	}
	return parsed.UTC()
}

// testEntropy is a deterministic reader, so lease ids and tokens are stable
// across runs and a failure names the same id twice.
func testEntropy() *rand.Rand { return rand.New(rand.NewSource(1)) }
