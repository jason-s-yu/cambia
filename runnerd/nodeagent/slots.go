package nodeagent

import "sync"

// Slots is one node's launch accounting: the concurrency ceiling, the jobs
// occupying it, and the exclusive holds of D13. It moved here from the
// dispatcher with the launch path it guards (D1), because how many jobs may
// run at once is a fact about the machine that forks them, not about the
// queue that orders them. The coordinator keeps the placement half: the
// per-node lease ceiling of D47 and the per-node exclusive barrier of D13.
//
// Exclusivity is per node. On a node running one exclusive job nothing else
// launches, and an exclusive job launches only into an idle node; a second
// node in the pool is unaffected, which is the whole of the change D13 makes
// to the daemon-wide rule the single-host runner had.
type Slots struct {
	mu sync.Mutex
	// max is the concurrency cap; zero or negative means unlimited.
	max int
	// active counts the slots this node drives: preparing and running jobs it
	// launched, plus every live job it reattached at start (cambia-723 -- a
	// reattached job occupies the runner exactly like a launched one, so
	// admission must see it).
	active int
	// exclusiveHolds counts the exclusive jobs currently holding the node.
	// While it is non-zero nothing else launches. It is a count rather than a
	// flag so the defensive multi-reattach case holds until the LAST exclusive
	// job exits.
	exclusiveHolds int
}

// NewSlots returns the accounting for a node whose ceiling is max concurrent
// jobs. A max of zero or less is unlimited concurrency.
func NewSlots(max int) *Slots { return &Slots{max: max} }

// Admit reports whether a gate-passed job may claim a slot now. While an
// exclusive job holds the node nothing launches; an exclusive job launches
// only into an idle node; a normal job launches while a slot is free.
func (s *Slots) Admit(exclusive bool) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.admitLocked(exclusive)
}

func (s *Slots) admitLocked(exclusive bool) bool {
	if s.exclusiveHolds > 0 {
		return false
	}
	if exclusive {
		return s.active == 0
	}
	return s.max <= 0 || s.active < s.max
}

// Claim reserves a slot for a launching or reattached job and, for an
// exclusive job, raises the exclusive hold.
func (s *Slots) Claim(exclusive bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.active++
	if exclusive {
		s.exclusiveHolds++
	}
}

// Release frees the slot a preparing, running, or reattached job held and, for
// an exclusive job, drops its exclusive hold.
func (s *Slots) Release(exclusive bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.active--
	if exclusive && s.exclusiveHolds > 0 {
		s.exclusiveHolds--
	}
}

// Active reports how many slots are occupied.
func (s *Slots) Active() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.active
}

// ExclusiveHolds reports how many exclusive jobs hold the node.
func (s *Slots) ExclusiveHolds() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.exclusiveHolds
}

// Free reports how many further jobs the node would admit. It is zero while an
// exclusive job holds the node, so a claim never offers a slot the launch
// would refuse and a coordinator bug cannot oversubscribe the node (D13).
func (s *Slots) Free() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.exclusiveHolds > 0 {
		return 0
	}
	if s.max <= 0 {
		return 1
	}
	free := s.max - s.active
	if free < 0 {
		return 0
	}
	return free
}
