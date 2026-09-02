package nashnet

import (
	"context"
	"time"
)

// Sweeper drives the lease expiry pass of D7. It holds no clock of its own: the
// verdicts come from the store's injected clock, and the ticker only decides
// how often the pass runs, so a test calls Tick directly and moves the clock by
// hand.
type Sweeper struct {
	store    *LeaseStore
	interval time.Duration
	handle   func(Outcome)
	onError  func(error)
}

// NewSweeper returns a sweeper over store. handle is called once per outcome,
// in lease-id order, outside no lock the caller holds; it is where the
// coordinator requeues, finalizes, or emits a revoke event. onError may be nil.
// The interval defaults to LEASE_TTL/4 (D7).
func NewSweeper(store *LeaseStore, handle func(Outcome), onError func(error)) *Sweeper {
	return &Sweeper{
		store:    store,
		interval: store.TTL() / 4,
		handle:   handle,
		onError:  onError,
	}
}

// Interval is how often Run passes over the leases.
func (s *Sweeper) Interval() time.Duration { return s.interval }

// SetInterval overrides the pass interval, for a deployment that tunes the TTL
// and for tests that want a short one.
func (s *Sweeper) SetInterval(d time.Duration) {
	if d > 0 {
		s.interval = d
	}
}

// Tick runs one expiry pass and dispatches its outcomes. It returns them too,
// so a test asserts on the pass without installing a handler.
func (s *Sweeper) Tick() []Outcome {
	out, err := s.store.Sweep()
	if err != nil && s.onError != nil {
		s.onError(err)
	}
	if s.handle != nil {
		for _, o := range out {
			s.handle(o)
		}
	}
	return out
}

// Run sweeps every interval until ctx is done. It is the daemon's entry point;
// the pass itself is Tick.
func (s *Sweeper) Run(ctx context.Context) {
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.Tick()
		}
	}
}
