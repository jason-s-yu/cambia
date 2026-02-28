package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// TestAgentInitializeDefaultPeeks verifies Initialize() with default InitialViewCount=2.
func TestAgentInitializeDefaultPeeks(t *testing.T) {
	rules := engine.DefaultHouseRules() // InitialViewCount=2
	g := engine.NewGame(42, rules)
	g.Deal()

	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	// Exactly 2 own hand slots should be known.
	knownCount := 0
	for i := uint8(0); i < a.OwnHandLen; i++ {
		if a.OwnHand[i].Bucket != BucketUnknown {
			knownCount++
		}
	}
	if knownCount != 2 {
		t.Errorf("agent knows %d cards, want 2 (default InitialViewCount)", knownCount)
	}
}

// TestAgentInitializeZeroPeeks verifies Initialize() with InitialViewCount=0.
func TestAgentInitializeZeroPeeks(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.InitialViewCount = 0
	g := engine.NewGame(42, rules)
	g.Deal()

	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	for i := uint8(0); i < a.OwnHandLen; i++ {
		if a.OwnHand[i].Bucket != BucketUnknown {
			t.Errorf("slot %d: expected unknown with InitialViewCount=0, got bucket %v", i, a.OwnHand[i].Bucket)
		}
	}
}

// TestAgentInitializeFullPeeks verifies Initialize() with InitialViewCount=4 (all cards).
func TestAgentInitializeFullPeeks(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.InitialViewCount = 4
	rules.CardsPerPlayer = 4
	g := engine.NewGame(42, rules)
	g.Deal()

	a := NewAgentState(0, 1, 0, 0)
	a.Initialize(&g)

	for i := uint8(0); i < a.OwnHandLen; i++ {
		if a.OwnHand[i].Bucket == BucketUnknown {
			t.Errorf("slot %d: expected known with InitialViewCount=4, got BucketUnknown", i)
		}
	}
}

// TestNPlayerInitializeDefaultPeeks verifies NPlayerInitialize() with default InitialViewCount=2.
func TestNPlayerInitializeDefaultPeeks(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.NumPlayers = 3
	rules.InitialViewCount = 2
	g := engine.NewGame(42, rules)
	g.Deal()

	a := NewAgentState(0, 1, 0, 0)
	a.InitializeNPlayer(&g)

	knownCount := 0
	for i := 0; i < int(engine.MaxHandSize); i++ {
		slot := 0*int(engine.MaxHandSize) + i
		if a.NPlayerSlotKnown[slot] {
			knownCount++
		}
	}
	if knownCount != 2 {
		t.Errorf("N-player agent knows %d own slots, want 2", knownCount)
	}
}

// TestNPlayerInitializeZeroPeeks verifies NPlayerInitialize() with InitialViewCount=0.
func TestNPlayerInitializeZeroPeeks(t *testing.T) {
	rules := engine.DefaultHouseRules()
	rules.NumPlayers = 3
	rules.InitialViewCount = 0
	g := engine.NewGame(42, rules)
	g.Deal()

	a := NewAgentState(0, 1, 0, 0)
	a.InitializeNPlayer(&g)

	for i := 0; i < int(engine.MaxHandSize); i++ {
		slot := 0*int(engine.MaxHandSize) + i
		if a.NPlayerSlotKnown[slot] {
			t.Errorf("slot %d: expected unknown with InitialViewCount=0", slot)
		}
	}
}
