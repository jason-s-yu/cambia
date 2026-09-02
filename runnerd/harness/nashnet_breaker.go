package harness

import (
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
)

// Circuit breaker bounds (D63). Three consecutive prepare_node_failed nacks
// across any jobs hold the node; the hold starts at a minute and doubles per
// trip to an hour, so a node that is briefly broken costs the pool one minute
// and one that is persistently broken drains itself instead of the ready queue.
const (
	// BreakerThreshold is how many consecutive prepare_node_failed nacks trip
	// the breaker.
	BreakerThreshold = 3
	// BreakerCooldown is the first hold, doubling per trip.
	BreakerCooldown = 60 * time.Second
	// BreakerCooldownCap is the ceiling the doubling stops at.
	BreakerCooldownCap = time.Hour
)

// breakerState is one node's circuit breaker (D63). Every field is
// coordinator-held: registration is a node route, so a node-clearable hold
// would be no hold at all, and the node that earned the mark would erase it and
// resume claiming at request rate.
type breakerState struct {
	// consecutive counts prepare_node_failed nacks since the last nack of
	// another reason and since the last trip.
	consecutive int
	// trips counts how many times this node has tripped, which is what the
	// cooldown doubles on. It survives the hold expiring, so a node that trips,
	// is let back in, and trips again is held twice as long.
	trips int
	// heldUntil is when the current hold lifts. Zero means no hold.
	heldUntil time.Time
}

// cooldownFor is the hold length for the nth trip: the first at
// BreakerCooldown, doubling per trip, capped at BreakerCooldownCap.
func cooldownFor(trips int) time.Duration {
	d := BreakerCooldown
	for i := 1; i < trips; i++ {
		d *= 2
		if d >= BreakerCooldownCap {
			return BreakerCooldownCap
		}
	}
	return d
}

// noteBreakerLocked folds one nack into the node's breaker and reports whether
// it tripped (D63). A nack of any other reason clears the consecutive run but
// leaves the trip count and any standing hold alone: the ladder measures how
// often this node has been broken, not how recently it last succeeded.
// Callers hold p.mu.
func (p *Pool) noteBreakerLocked(nodeID, reason string, now time.Time) bool {
	b := p.breaker[nodeID]
	if b == nil {
		b = &breakerState{}
		p.breaker[nodeID] = b
	}
	if reason != nashnet.NackPrepareNodeFailed {
		b.consecutive = 0
		return false
	}
	b.consecutive++
	if b.consecutive < BreakerThreshold {
		return false
	}
	b.consecutive = 0
	b.trips++
	b.heldUntil = now.Add(cooldownFor(b.trips))
	return true
}

// breakerHeld reports whether the coordinator is holding this node off the
// claim route. The hold clears on its own timer or on an operator act and never
// on a re-register (D8, D63).
func (p *Pool) breakerHeld(nodeID string) bool {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()
	b := p.breaker[nodeID]
	return b != nil && b.heldUntil.After(now)
}

// breakerReport renders a node's breaker for the operator listing: how many
// times it has tripped and how long the current hold has left.
func (p *Pool) breakerReport(nodeID string, now time.Time) (trips int, heldSeconds int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	b := p.breaker[nodeID]
	if b == nil {
		return 0, 0
	}
	if b.heldUntil.After(now) {
		heldSeconds = int64(b.heldUntil.Sub(now) / time.Second)
	}
	return b.trips, heldSeconds
}
