package baselines

import "testing"

// The reference policies read their memory by dict ITERATION order, so a slot
// deleted by the snap-move handler leaves a hole that is never renumbered and a
// slot written for the first time lands at the end. Both facts change which
// slot "the first unknown" and "the highest known" name, which is why they are
// pinned here rather than left to the end-to-end parity replay to catch.

func newMemory(slots ...uint8) *slotMemory {
	m := &slotMemory{}
	for _, s := range slots {
		m.setUnknown(s)
	}
	return m
}

func TestRemoveLeavesTheRemainingOrder(t *testing.T) {
	m := newMemory(0, 1, 2, 3)
	m.set(2, 4, 3)
	m.remove(0)

	if got := m.size(); got != 3 {
		t.Fatalf("size after remove = %d, want 3", got)
	}
	if slot, ok := m.firstUnknownSlot(); !ok || slot != 1 {
		t.Fatalf("firstUnknownSlot = (%d, %v), want (1, true)", slot, ok)
	}
	if _, present := m.get(0); present {
		t.Fatal("removed slot 0 is still present")
	}
}

func TestSetAppendsAnAbsentSlotAtTheEnd(t *testing.T) {
	m := newMemory(0, 1)
	m.remove(0)
	// A replace onto the removed slot re-adds it, and it is now LAST in
	// iteration order, so it is no longer the first unknown that a later
	// setUnknown would find.
	m.set(0, 9, 8)
	m.setUnknown(0)

	if slot, ok := m.firstUnknownSlot(); !ok || slot != 1 {
		t.Fatalf("firstUnknownSlot = (%d, %v), want (1, true)", slot, ok)
	}
}

func TestHighestKnownSlotKeepsTheFirstOnATie(t *testing.T) {
	m := newMemory(0, 1, 2)
	m.set(2, 7, 6)
	m.set(1, 7, 6)

	// Insertion order is 0, 1, 2, and writing a value in place does not move a
	// slot, so slot 1 is reached first and the strict > keeps it.
	slot, ok := m.highestKnownSlot()
	if !ok || slot != 1 {
		t.Fatalf("highestKnownSlot = (%d, %v), want (1, true)", slot, ok)
	}
}

func TestHighestEffectiveSlotChargesUnknownsTheExpectedValue(t *testing.T) {
	m := newMemory(0, 1)
	m.set(0, 6, 5) // below unknownCardExpectedValue, so slot 1 wins
	if slot, ok := m.highestEffectiveSlot(); !ok || slot != 1 {
		t.Fatalf("highestEffectiveSlot = (%d, %v), want (1, true)", slot, ok)
	}

	m.set(0, 7, 6) // above it, so slot 0 wins
	if slot, ok := m.highestEffectiveSlot(); !ok || slot != 0 {
		t.Fatalf("highestEffectiveSlot = (%d, %v), want (0, true)", slot, ok)
	}
}

func TestSetFromCopiesUnknownAsUnknown(t *testing.T) {
	src := newMemory(0)
	dst := newMemory(0, 1)

	entry, present := src.get(0)
	dst.setFrom(1, entry, present)
	if e, _ := dst.get(1); e.known {
		t.Fatal("copying an unknown source slot left the target known")
	}

	src.set(0, 11, 10)
	entry, present = src.get(0)
	dst.setFrom(1, entry, present)
	e, _ := dst.get(1)
	if !e.known || e.value != 11 || e.rank != 10 {
		t.Fatalf("copied entry = %+v, want value 11 rank 10 known", e)
	}
}

func TestEstimatedValueAndKnownCount(t *testing.T) {
	m := newMemory(0, 1, 2)
	m.set(0, 3, 2)
	m.set(1, 5, 4)

	if got := m.knownCount(); got != 2 {
		t.Fatalf("knownCount = %d, want 2", got)
	}
	want := 3.0 + 5.0 + unknownCardExpectedValue
	if got := m.estimatedValue(); got != want {
		t.Fatalf("estimatedValue = %v, want %v", got, want)
	}
}

func TestRankMatchesNeedsAKnownSlot(t *testing.T) {
	m := newMemory(0)
	if m.rankMatches(0, 5, true) {
		t.Fatal("an unknown slot matched a rank")
	}
	m.set(0, 6, 5)
	if !m.rankMatches(0, 5, true) {
		t.Fatal("a known slot failed to match its own rank")
	}
	if m.rankMatches(0, 5, false) {
		t.Fatal("a slot matched an empty discard pile")
	}
	if m.rankMatches(1, 5, true) {
		t.Fatal("an absent slot matched a rank")
	}
}
