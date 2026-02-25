package agent

import (
	"math/rand/v2"
	"testing"
)

// TestMemoryArchetype_Default verifies that a freshly constructed AgentState
// has MemoryPerfect as the default archetype and zero values for lambda/capacity.
func TestMemoryArchetype_Default(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	if a.MemoryArchetype != MemoryPerfect {
		t.Errorf("expected MemoryPerfect, got %d", a.MemoryArchetype)
	}
	if a.MemoryDecayLambda != 0 {
		t.Errorf("expected MemoryDecayLambda=0, got %f", a.MemoryDecayLambda)
	}
	if a.MemoryCapacity != 0 {
		t.Errorf("expected MemoryCapacity=0, got %d", a.MemoryCapacity)
	}
}

// TestMemoryPerfect_NoDecay verifies that ApplyMemoryDecay with MemoryPerfect
// never removes slots from the active mask.
func TestMemoryPerfect_NoDecay(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.MemoryArchetype = MemoryPerfect

	// Manually add 3 PrivOwn slots.
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketHighKing
	a.SlotTags[1] = TagPrivOwn
	a.SlotBuckets[1] = BucketZero
	a.SlotTags[2] = TagPrivOwn
	a.SlotBuckets[2] = BucketNegKing
	a.OwnActiveMask[0] = 0
	a.OwnActiveMask[1] = 1
	a.OwnActiveMask[2] = 2
	a.OwnActiveMaskLen = 3

	rng := rand.New(rand.NewPCG(42, 0))
	for i := 0; i < 100; i++ {
		a.ApplyMemoryDecay(rng)
	}

	if a.OwnActiveMaskLen != 3 {
		t.Errorf("MemoryPerfect: expected mask len 3 after 100 decay calls, got %d", a.OwnActiveMaskLen)
	}
	for i := uint8(0); i < 3; i++ {
		if a.SlotTags[i] != TagPrivOwn {
			t.Errorf("MemoryPerfect: slot %d should still be TagPrivOwn", i)
		}
	}
}

// TestMemoryDecaying_EventualDecay verifies that with high lambda, all slots
// eventually decay to Unk.
func TestMemoryDecaying_EventualDecay(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.MemoryArchetype = MemoryDecaying
	a.MemoryDecayLambda = 10.0 // p ≈ 1.0, every slot decays immediately

	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketHighKing
	a.SlotTags[1] = TagPrivOwn
	a.SlotBuckets[1] = BucketAce
	a.OwnActiveMask[0] = 0
	a.OwnActiveMask[1] = 1
	a.OwnActiveMaskLen = 2

	rng := rand.New(rand.NewPCG(42, 0))
	a.ApplyMemoryDecay(rng)

	if a.OwnActiveMaskLen != 0 {
		t.Errorf("expected all slots decayed (mask len 0), got %d", a.OwnActiveMaskLen)
	}
	if a.SlotTags[0] != TagUnk || a.SlotTags[1] != TagUnk {
		t.Error("expected both slots to be TagUnk after high-lambda decay")
	}
}

// TestMemoryDecaying_DeterministicSeed verifies that the same seed produces
// the same decay pattern.
func TestMemoryDecaying_DeterministicSeed(t *testing.T) {
	setup := func() AgentState {
		a := NewAgentState(0, 1, 0, 0)
		a.MemoryArchetype = MemoryDecaying
		a.MemoryDecayLambda = 0.5
		for i := uint8(0); i < 3; i++ {
			a.SlotTags[i] = TagPrivOwn
			a.SlotBuckets[i] = BucketMidNum
			a.OwnActiveMask[i] = i
		}
		a.OwnActiveMaskLen = 3
		return a
	}

	a1 := setup()
	rng1 := rand.New(rand.NewPCG(99, 0))
	for i := 0; i < 10; i++ {
		a1.ApplyMemoryDecay(rng1)
	}

	a2 := setup()
	rng2 := rand.New(rand.NewPCG(99, 0))
	for i := 0; i < 10; i++ {
		a2.ApplyMemoryDecay(rng2)
	}

	if a1.OwnActiveMaskLen != a2.OwnActiveMaskLen {
		t.Errorf("deterministic seed: mask len %d != %d", a1.OwnActiveMaskLen, a2.OwnActiveMaskLen)
	}
	for i := uint8(0); i < 3; i++ {
		if a1.SlotTags[i] != a2.SlotTags[i] {
			t.Errorf("deterministic seed: slot %d tag differs", i)
		}
	}
}

// TestMemoryHumanLike_CapacityLimit verifies that after observing 6 cards
// (exceeding capacity 3), only 3 remain in the active mask.
func TestMemoryHumanLike_CapacityLimit(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.MemoryArchetype = MemoryHumanLike
	a.MemoryCapacity = 3

	// Use a larger active mask for this test — directly set 6 slots.
	// We need to override MaxActiveMask (3) for testing purposes by directly
	// manipulating the internal fields. Since the mask is [MaxActiveMask]uint8 = [3]uint8,
	// we can only set 3 entries via OwnActiveMask. Instead, we test via ApplyMemoryDecay
	// starting from a state where 3 slots exist and capacity is 2.
	a.MemoryCapacity = 2
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketHighKing // saliency = |13-4.5| = 8.5
	a.SlotTags[1] = TagPrivOwn
	a.SlotBuckets[1] = BucketAce      // saliency = |1-4.5| = 3.5
	a.SlotTags[2] = TagPrivOwn
	a.SlotBuckets[2] = BucketZero     // saliency = |0-4.5| = 4.5
	a.OwnActiveMask[0] = 0
	a.OwnActiveMask[1] = 1
	a.OwnActiveMask[2] = 2
	a.OwnActiveMaskLen = 3

	a.ApplyMemoryDecay(nil) // rng unused for HumanLike

	if a.OwnActiveMaskLen != 2 {
		t.Errorf("expected mask len 2, got %d", a.OwnActiveMaskLen)
	}
}

// TestMemoryHumanLike_SaliencyEviction verifies that the lowest-saliency slot
// is evicted first.
func TestMemoryHumanLike_SaliencyEviction(t *testing.T) {
	a := NewAgentState(0, 1, 0, 0)
	a.MemoryArchetype = MemoryHumanLike
	a.MemoryCapacity = 2

	// Slot 0: BucketAce (saliency 3.5) — LOWEST, evict first
	// Slot 1: BucketHighKing (saliency 8.5)
	// Slot 2: BucketZero (saliency 4.5)
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketAce
	a.SlotTags[1] = TagPrivOwn
	a.SlotBuckets[1] = BucketHighKing
	a.SlotTags[2] = TagPrivOwn
	a.SlotBuckets[2] = BucketZero
	a.OwnActiveMask[0] = 0
	a.OwnActiveMask[1] = 1
	a.OwnActiveMask[2] = 2
	a.OwnActiveMaskLen = 3

	a.ApplyMemoryDecay(nil)

	// Slot 0 (BucketAce, saliency 3.5) should have been evicted.
	if a.SlotTags[0] != TagUnk {
		t.Errorf("expected slot 0 (BucketAce, lowest saliency) to be evicted (TagUnk), got tag=%d", a.SlotTags[0])
	}
	if a.OwnActiveMaskLen != 2 {
		t.Errorf("expected mask len 2, got %d", a.OwnActiveMaskLen)
	}
	// Slots 1 and 2 should remain.
	if a.SlotTags[1] != TagPrivOwn {
		t.Errorf("expected slot 1 (BucketHighKing) to remain TagPrivOwn")
	}
	if a.SlotTags[2] != TagPrivOwn {
		t.Errorf("expected slot 2 (BucketZero) to remain TagPrivOwn")
	}
}
