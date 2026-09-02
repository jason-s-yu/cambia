// Package baselines implements, engine-side, the heuristic evaluation
// baselines the mean_imp battery scores an agent against.
//
// The reference implementation is cfr/src/agents/baseline_agents.py and it
// stays the oracle: mean_imp is a historical metric, so every policy here
// reproduces its Python counterpart decision for decision on the same state,
// quirks included. Two of those quirks drive the shape of this package:
//
//   - The Python agents hold their memory in plain dicts keyed by hand slot and
//     read them by ITERATION order, not by slot number. Insertion order is
//     therefore part of the policy: "the first unknown slot" is the first
//     unknown in insertion order, and the snap-move handler deletes a key,
//     which leaves a hole the reference never renumbers. slotMemory below
//     reproduces that ordering exactly.
//   - The Python agents dispatch on the FIRST entry of the legal-action list,
//     which is the lowest legal action index. lowestLegal reproduces that.
//
// Decisions are computed from the engine state and the legal-action bitmask,
// so a caller never has to materialize a decoded action list.
package baselines

import (
	engine "github.com/jason-s-yu/cambia/engine"
)

// unknownCardExpectedValue mirrors baseline_agents.UNKNOWN_CARD_EXPECTED_VALUE:
// the value an unseen slot is scored at.
const unknownCardExpectedValue = 6.5

// slotEntry is one remembered hand slot. The Python reference keeps a value
// dict and a rank dict in lockstep (every write and delete touches both), so
// one entry carries both and `known` stands for the reference's `None`.
type slotEntry struct {
	slot  uint8
	value int
	rank  uint8
	known bool
}

// slotMemory is an insertion-ordered slot -> (value, rank) map, the Go stand-in
// for the reference's pair of dicts. Hands hold at most engine.MaxHandSize
// cards, so a linear scan beats any index.
type slotMemory struct {
	entries []slotEntry
}

func (m *slotMemory) reset() {
	m.entries = m.entries[:0]
}

// set records a known value for slot, appending the slot at the end when it is
// not already present. This is Python's `d[slot] = v`.
func (m *slotMemory) set(slot uint8, value int, rank uint8) {
	for i := range m.entries {
		if m.entries[i].slot == slot {
			m.entries[i].value = value
			m.entries[i].rank = rank
			m.entries[i].known = true
			return
		}
	}
	m.entries = append(m.entries, slotEntry{slot: slot, value: value, rank: rank, known: true})
}

// setUnknown records slot as seen-but-unknown, appending it when absent. This
// is Python's `d[slot] = None`.
func (m *slotMemory) setUnknown(slot uint8) {
	for i := range m.entries {
		if m.entries[i].slot == slot {
			m.entries[i].known = false
			return
		}
	}
	m.entries = append(m.entries, slotEntry{slot: slot})
}

// setFrom copies another slot's record, known or not, onto slot. This is the
// snap-move handler's `opponent_memory[target] = own_memory.get(source)`, where
// a missing source key reads as None.
func (m *slotMemory) setFrom(slot uint8, src slotEntry, present bool) {
	if !present || !src.known {
		m.setUnknown(slot)
		return
	}
	m.set(slot, src.value, src.rank)
}

// get returns the record for slot and whether the slot is present at all.
func (m *slotMemory) get(slot uint8) (slotEntry, bool) {
	for i := range m.entries {
		if m.entries[i].slot == slot {
			return m.entries[i], true
		}
	}
	return slotEntry{}, false
}

// remove deletes slot, closing the gap so the remaining slots keep their
// relative order. This is Python's `del d[slot]`.
func (m *slotMemory) remove(slot uint8) {
	for i := range m.entries {
		if m.entries[i].slot == slot {
			m.entries = append(m.entries[:i], m.entries[i+1:]...)
			return
		}
	}
}

func (m *slotMemory) size() int { return len(m.entries) }

// knownCount is `sum(1 for v in d.values() if v is not None)`.
func (m *slotMemory) knownCount() int {
	n := 0
	for i := range m.entries {
		if m.entries[i].known {
			n++
		}
	}
	return n
}

// estimatedValue is the reference's _estimate_own_hand_value: known values
// summed, unknown slots charged unknownCardExpectedValue.
func (m *slotMemory) estimatedValue() float64 {
	total := 0.0
	for i := range m.entries {
		if m.entries[i].known {
			total += float64(m.entries[i].value)
		} else {
			total += unknownCardExpectedValue
		}
	}
	return total
}

// firstUnknownSlot is _find_unknown_own_slot: the first unknown slot in
// insertion order.
func (m *slotMemory) firstUnknownSlot() (uint8, bool) {
	for i := range m.entries {
		if !m.entries[i].known {
			return m.entries[i].slot, true
		}
	}
	return 0, false
}

// highestKnownSlot is _find_highest_known_own_slot. The reference compares with
// a strict `>` while iterating in insertion order, so the FIRST slot holding
// the maximum wins a tie.
func (m *slotMemory) highestKnownSlot() (uint8, bool) {
	best := uint8(0)
	bestVal := 0
	found := false
	for i := range m.entries {
		e := m.entries[i]
		if !e.known {
			continue
		}
		if !found || e.value > bestVal {
			found = true
			bestVal = e.value
			best = e.slot
		}
	}
	return best, found
}

// highestEffectiveSlot is the snap-move handler's slot choice: the highest
// value with unknown slots charged unknownCardExpectedValue, again first-wins
// on a tie.
func (m *slotMemory) highestEffectiveSlot() (uint8, bool) {
	best := uint8(0)
	bestVal := 0.0
	found := false
	for i := range m.entries {
		e := m.entries[i]
		v := unknownCardExpectedValue
		if e.known {
			v = float64(e.value)
		}
		if !found || v > bestVal {
			found = true
			bestVal = v
			best = e.slot
		}
	}
	return best, found
}

// rankMatches reports whether the slot's remembered rank equals want. An absent
// or unknown slot never matches, which is the reference's
// `_own_card_matches_discard` returning False on a None rank.
func (m *slotMemory) rankMatches(slot uint8, want uint8, haveWant bool) bool {
	if !haveWant {
		return false
	}
	e, ok := m.get(slot)
	if !ok || !e.known {
		return false
	}
	return e.rank == want
}

// cardValue and cardRank read a dealt card the way the Python Card object does.
func cardValue(c engine.Card) int  { return int(c.Value()) }
func cardRank(c engine.Card) uint8 { return c.Rank() }
