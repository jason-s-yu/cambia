package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// newInitializedAgent creates an agent initialized with a fresh dealt game.
func newInitializedAgent(playerID, opponentID uint8) (AgentState, engine.GameState) {
	g := newDealtGame()
	a := NewAgentState(playerID, opponentID, 1, 3)
	a.Initialize(&g)
	return a, g
}

// TestEPPBSEncodingDim verifies that EncodeEPPBS produces exactly EPPBSInputDim floats.
func TestEPPBSEncodingDim(t *testing.T) {
	a, g := newInitializedAgent(0, 1)
	_ = g
	var out [EPPBSInputDim]float32
	a.EncodeEPPBS(engine.CtxStartTurn, -1, &out)

	// Verify output is populated (at least one non-zero).
	anyNonZero := false
	for _, v := range out {
		if v != 0 {
			anyNonZero = true
			break
		}
	}
	if !anyNonZero {
		t.Error("EncodeEPPBS output is all zeros")
	}
	// EPPBSInputDim must equal 200.
	if EPPBSInputDim != 200 {
		t.Errorf("EPPBSInputDim = %d, want 200", EPPBSInputDim)
	}
}

// TestEPPBSSaliencyEviction verifies that when OwnActiveMask is full and we append
// a new slot, the slot with minimum BucketSaliency is evicted.
func TestEPPBSSaliencyEviction(t *testing.T) {
	var a AgentState
	// Set up 3 slots manually with distinct buckets:
	// slot 0: BucketMidNum  → saliency |5.5-4.5|=1.0  (LOWEST → should be evicted)
	// slot 1: BucketLowNum  → saliency |3.0-4.5|=1.5
	// slot 2: BucketPeekSelf → saliency |7.5-4.5|=3.0
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketMidNum
	a.SlotTags[1] = TagPrivOwn
	a.SlotBuckets[1] = BucketLowNum
	a.SlotTags[2] = TagPrivOwn
	a.SlotBuckets[2] = BucketPeekSelf
	a.OwnActiveMask[0] = 0
	a.OwnActiveMask[1] = 1
	a.OwnActiveMask[2] = 2
	a.OwnActiveMaskLen = 3

	// Set slot 3 as new known slot to append.
	a.SlotTags[3] = TagUnk
	a.SlotBuckets[3] = 0
	a.OwnHandLen = 4

	// appendOwnActive(3) should evict slot 0 (BucketMidNum, saliency 1.0).
	a.appendOwnActive(3)

	if a.OwnActiveMaskLen != 3 {
		t.Fatalf("OwnActiveMaskLen = %d, want 3", a.OwnActiveMaskLen)
	}
	// Slot 0 should have been evicted → tag reset to TagUnk.
	if a.SlotTags[0] != TagUnk {
		t.Errorf("evicted slot 0 tag = %v, want TagUnk", a.SlotTags[0])
	}
	// Remaining mask should contain slots 1, 2, 3.
	mask := map[uint8]bool{}
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		mask[a.OwnActiveMask[i]] = true
	}
	for _, expected := range []uint8{1, 2, 3} {
		if !mask[expected] {
			t.Errorf("OwnActiveMask missing slot %d; mask = %v", expected, a.OwnActiveMask[:a.OwnActiveMaskLen])
		}
	}
	if mask[0] {
		t.Errorf("OwnActiveMask still contains evicted slot 0")
	}
}

// TestEPPBSFIFOEviction verifies FIFO eviction in OppActiveMask.
func TestEPPBSFIFOEviction(t *testing.T) {
	var a AgentState
	// Fill OppActiveMask with slots 6, 7, 8 (oldest = 6 at index 0).
	a.SlotTags[6] = TagPrivOpp
	a.SlotTags[7] = TagPrivOpp
	a.SlotTags[8] = TagPrivOpp
	a.OppActiveMask[0] = 6
	a.OppActiveMask[1] = 7
	a.OppActiveMask[2] = 8
	a.OppActiveMaskLen = 3

	// Append slot 9 — should evict slot 6 (oldest).
	a.SlotTags[9] = TagUnk
	a.appendOppActive(9)

	if a.OppActiveMaskLen != 3 {
		t.Fatalf("OppActiveMaskLen = %d, want 3", a.OppActiveMaskLen)
	}
	// Slot 6 should be evicted → TagUnk.
	if a.SlotTags[6] != TagUnk {
		t.Errorf("evicted slot 6 tag = %v, want TagUnk", a.SlotTags[6])
	}
	// Mask should contain 7, 8, 9.
	mask := map[uint8]bool{}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		mask[a.OppActiveMask[i]] = true
	}
	for _, expected := range []uint8{7, 8, 9} {
		if !mask[expected] {
			t.Errorf("OppActiveMask missing slot %d; mask = %v", expected, a.OppActiveMask[:a.OppActiveMaskLen])
		}
	}
	if mask[6] {
		t.Errorf("OppActiveMask still contains evicted slot 6")
	}
}

// TestEPPBSTagTransitions tests all tag transition rules via setOwnSlotKnown and setOppSlotPrivOpp.
func TestEPPBSTagTransitions(t *testing.T) {
	t.Run("Unk→PrivOwn when we peek", func(t *testing.T) {
		var a AgentState
		a.SlotTags[2] = TagUnk
		a.setOwnSlotKnown(2, BucketPeekSelf)
		if a.SlotTags[2] != TagPrivOwn {
			t.Errorf("tag = %v, want TagPrivOwn", a.SlotTags[2])
		}
		if a.SlotBuckets[2] != BucketPeekSelf {
			t.Errorf("bucket = %v, want BucketPeekSelf", a.SlotBuckets[2])
		}
		if a.OwnActiveMaskLen != 1 || a.OwnActiveMask[0] != 2 {
			t.Errorf("OwnActiveMask = %v (len=%d), want [2]", a.OwnActiveMask, a.OwnActiveMaskLen)
		}
	})

	t.Run("Unk→PrivOpp when opp peeks", func(t *testing.T) {
		var a AgentState
		a.SlotTags[3] = TagUnk
		a.setOppSlotPrivOpp(3)
		if a.SlotTags[3] != TagPrivOpp {
			t.Errorf("tag = %v, want TagPrivOpp", a.SlotTags[3])
		}
		if a.OppActiveMaskLen != 1 || a.OppActiveMask[0] != 3 {
			t.Errorf("OppActiveMask = %v (len=%d), want [3]", a.OppActiveMask, a.OppActiveMaskLen)
		}
	})

	t.Run("PrivOpp→Pub when we peek what opp knew", func(t *testing.T) {
		var a AgentState
		a.SlotTags[7] = TagPrivOpp
		a.OppActiveMask[0] = 7
		a.OppActiveMaskLen = 1
		a.setOwnSlotKnown(7, BucketLowNum)
		if a.SlotTags[7] != TagPub {
			t.Errorf("tag = %v, want TagPub", a.SlotTags[7])
		}
		if a.OppActiveMaskLen != 0 {
			t.Errorf("OppActiveMaskLen = %d, want 0", a.OppActiveMaskLen)
		}
		// Not added to OwnActiveMask (it's public).
		if a.OwnActiveMaskLen != 0 {
			t.Errorf("OwnActiveMaskLen = %d, want 0 (slot is public)", a.OwnActiveMaskLen)
		}
	})

	t.Run("PrivOwn→Pub when opp peeks what we knew", func(t *testing.T) {
		var a AgentState
		a.SlotTags[1] = TagPrivOwn
		a.SlotBuckets[1] = BucketHighKing
		a.OwnActiveMask[0] = 1
		a.OwnActiveMaskLen = 1
		a.setOppSlotPrivOpp(1)
		if a.SlotTags[1] != TagPub {
			t.Errorf("tag = %v, want TagPub", a.SlotTags[1])
		}
		if a.OwnActiveMaskLen != 0 {
			t.Errorf("OwnActiveMaskLen = %d, want 0", a.OwnActiveMaskLen)
		}
	})

	t.Run("Re-peek PrivOwn stays PrivOwn, updates bucket", func(t *testing.T) {
		var a AgentState
		a.SlotTags[0] = TagPrivOwn
		a.SlotBuckets[0] = BucketAce
		a.OwnActiveMask[0] = 0
		a.OwnActiveMaskLen = 1
		a.setOwnSlotKnown(0, BucketMidNum)
		if a.SlotTags[0] != TagPrivOwn {
			t.Errorf("tag = %v, want TagPrivOwn", a.SlotTags[0])
		}
		if a.SlotBuckets[0] != BucketMidNum {
			t.Errorf("bucket = %v, want BucketMidNum", a.SlotBuckets[0])
		}
		if a.OwnActiveMaskLen != 1 {
			t.Errorf("OwnActiveMaskLen = %d, want 1 (no duplicate)", a.OwnActiveMaskLen)
		}
	})

	t.Run("Pub stays Pub on peek (bucket updated)", func(t *testing.T) {
		var a AgentState
		a.SlotTags[5] = TagPub
		a.SlotBuckets[5] = BucketZero
		a.setOwnSlotKnown(5, BucketHighKing)
		if a.SlotTags[5] != TagPub {
			t.Errorf("tag = %v, want TagPub", a.SlotTags[5])
		}
		if a.SlotBuckets[5] != BucketHighKing {
			t.Errorf("bucket = %v, want BucketHighKing", a.SlotBuckets[5])
		}
	})
}

// TestEPPBSBlindSwap verifies that both slots become TagUnk after a blind swap action.
func TestEPPBSBlindSwap(t *testing.T) {
	g := newDealtGame()
	// Player 0 draws from stockpile, then uses blind swap ability.
	a0 := NewAgentState(0, 1, 1, 3)
	a0.Initialize(&g)
	a1 := NewAgentState(1, 0, 1, 3)
	a1.Initialize(&g)

	// Check initial EP-PBS: a0 should know slots 0 and 1 (initial peek).
	if a0.SlotTags[0] != TagPrivOwn || a0.SlotTags[1] != TagPrivOwn {
		t.Logf("a0 initial tags: slot0=%v slot1=%v", a0.SlotTags[0], a0.SlotTags[1])
	}

	// Advance to a blind swap action (J/Q discard with ability).
	// For a controlled test, directly invoke processBlindSwap.
	a0.SlotTags[0] = TagPrivOwn
	a0.SlotBuckets[0] = BucketAce
	a0.OwnActiveMask[0] = 0
	a0.OwnActiveMaskLen = 1
	a0.SlotTags[6] = TagPrivOwn // We peeked opp slot 0
	a0.SlotBuckets[6] = BucketLowNum
	a0.OwnActiveMask[1] = 6
	a0.OwnActiveMaskLen = 2

	// Simulate blind swap: we swap own slot 0 with opp slot 0 (6).
	// Directly call eppbsSetSlotUnk to match processBlindSwap behavior.
	a0.eppbsSetSlotUnk(0)
	a0.eppbsSetSlotUnk(OppSlotsStart + 0) // = 6

	if a0.SlotTags[0] != TagUnk {
		t.Errorf("after blind swap: own slot 0 tag = %v, want TagUnk", a0.SlotTags[0])
	}
	if a0.SlotTags[6] != TagUnk {
		t.Errorf("after blind swap: opp slot 0 (=6) tag = %v, want TagUnk", a0.SlotTags[6])
	}
	if a0.OwnActiveMaskLen != 0 {
		t.Errorf("OwnActiveMaskLen = %d, want 0 after both slots set to Unk", a0.OwnActiveMaskLen)
	}
}

// TestEPPBSKingSwap verifies that eppbsSwapSlots correctly swaps epistemic state.
func TestEPPBSKingSwap(t *testing.T) {
	var a AgentState
	// Set up: we know own slot 0 (BucketAce) and opp slot 6 (BucketHighKing).
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketAce
	a.OwnActiveMask[0] = 0
	a.OwnActiveMaskLen = 1

	a.SlotTags[6] = TagPrivOwn
	a.SlotBuckets[6] = BucketHighKing
	a.OwnActiveMask[1] = 6
	a.OwnActiveMaskLen = 2

	// King swap: swap own slot 0 with opp slot 0 (global = 6).
	a.eppbsSwapSlots(0, 6)

	// After swap: slot 0 has what was at 6 (HighKing, TagPrivOwn).
	if a.SlotTags[0] != TagPrivOwn {
		t.Errorf("slot 0 tag = %v, want TagPrivOwn", a.SlotTags[0])
	}
	if a.SlotBuckets[0] != BucketHighKing {
		t.Errorf("slot 0 bucket = %v, want BucketHighKing", a.SlotBuckets[0])
	}
	// Slot 6 has what was at 0 (Ace, TagPrivOwn).
	if a.SlotTags[6] != TagPrivOwn {
		t.Errorf("slot 6 tag = %v, want TagPrivOwn", a.SlotTags[6])
	}
	if a.SlotBuckets[6] != BucketAce {
		t.Errorf("slot 6 bucket = %v, want BucketAce", a.SlotBuckets[6])
	}
	// Active mask references should be swapped.
	mask := map[uint8]bool{}
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		mask[a.OwnActiveMask[i]] = true
	}
	if !mask[0] || !mask[6] {
		t.Errorf("OwnActiveMask = %v, want to contain both 0 and 6", a.OwnActiveMask[:a.OwnActiveMaskLen])
	}
}

// TestEPPBSSlotIdentityZeroing verifies that PrivOpp and Unk slots have zero bucket dims.
func TestEPPBSSlotIdentityZeroing(t *testing.T) {
	var a AgentState
	// Set up 4 slots across own and opp with various tags.
	a.OwnHandLen = 3
	a.OppHandLen = 2

	// Slot 0: TagPrivOwn, BucketAce — identity should be encoded at index 2 (BucketAce=2).
	a.SlotTags[0] = TagPrivOwn
	a.SlotBuckets[0] = BucketAce

	// Slot 1: TagUnk — identity should be all zeros.
	a.SlotTags[1] = TagUnk
	a.SlotBuckets[1] = 0

	// Slot 2: TagPub, BucketHighKing — identity encoded at index 8.
	a.SlotTags[2] = TagPub
	a.SlotBuckets[2] = BucketHighKing

	// Slot 6 (opp 0): TagPrivOpp — identity should be all zeros.
	a.SlotTags[6] = TagPrivOpp
	a.SlotBuckets[6] = 0

	// Slot 7 (opp 1): TagPub, BucketZero — identity encoded at index 0.
	a.SlotTags[7] = TagPub
	a.SlotBuckets[7] = BucketZero

	var out [EPPBSInputDim]float32
	a.EncodeEPPBS(engine.CtxStartTurn, -1, &out)

	// Identity section starts at offset 88, 9 dims per slot.
	identStart := 88

	// Slot 0 (offset 88): BucketAce=2 → out[88+2]=1.
	if out[identStart+0+int(BucketAce)] != 1.0 {
		t.Errorf("slot 0 identity[BucketAce] = %v, want 1.0", out[identStart+0+int(BucketAce)])
	}
	for b := 0; b < 9; b++ {
		if b != int(BucketAce) && out[identStart+0+b] != 0 {
			t.Errorf("slot 0 identity[%d] = %v, want 0", b, out[identStart+0+b])
		}
	}

	// Slot 1 (offset 88+9=97): all zeros.
	for b := 0; b < 9; b++ {
		if out[identStart+9+b] != 0 {
			t.Errorf("slot 1 (TagUnk) identity[%d] = %v, want 0", b, out[identStart+9+b])
		}
	}

	// Slot 2 (offset 88+18=106): BucketHighKing=8 → out[106+8]=1.
	if out[identStart+18+int(BucketHighKing)] != 1.0 {
		t.Errorf("slot 2 identity[BucketHighKing] = %v, want 1.0", out[identStart+18+int(BucketHighKing)])
	}

	// Slot 6 (offset 88+54=142): TagPrivOpp → all zeros.
	for b := 0; b < 9; b++ {
		if out[identStart+54+b] != 0 {
			t.Errorf("slot 6 (TagPrivOpp) identity[%d] = %v, want 0", b, out[identStart+54+b])
		}
	}

	// Slot 7 (offset 88+63=151): TagPub, BucketZero=0 → out[151]=1.
	if out[identStart+63+int(BucketZero)] != 1.0 {
		t.Errorf("slot 7 identity[BucketZero] = %v, want 1.0", out[identStart+63+int(BucketZero)])
	}
}

// TestEPPBSInitialize verifies initial EP-PBS state matches initial peek configuration.
func TestEPPBSInitialize(t *testing.T) {
	a, g := newInitializedAgent(0, 1)
	_ = g

	// Player 0 peeks their initial 2 cards (indices 0 and 1 by default).
	// Those slots should be TagPrivOwn; others TagUnk.
	if a.SlotTags[0] != TagPrivOwn {
		t.Errorf("slot 0 tag = %v, want TagPrivOwn (initial peek)", a.SlotTags[0])
	}
	if a.SlotTags[1] != TagPrivOwn {
		t.Errorf("slot 1 tag = %v, want TagPrivOwn (initial peek)", a.SlotTags[1])
	}
	for i := 2; i < int(a.OwnHandLen); i++ {
		if a.SlotTags[i] != TagUnk {
			t.Errorf("slot %d tag = %v, want TagUnk", i, a.SlotTags[i])
		}
	}
	// All opp slots should be TagUnk.
	for i := OppSlotsStart; i < MaxSlots; i++ {
		if a.SlotTags[i] != TagUnk {
			t.Errorf("opp slot %d tag = %v, want TagUnk", i, a.SlotTags[i])
		}
	}
	// OwnActiveMask should have the peeked slots.
	if a.OwnActiveMaskLen != 2 {
		t.Errorf("OwnActiveMaskLen = %d, want 2", a.OwnActiveMaskLen)
	}
	if a.OppActiveMaskLen != 0 {
		t.Errorf("OppActiveMaskLen = %d, want 0", a.OppActiveMaskLen)
	}
}

// TestEPPBSEncodeTagSection verifies slot tag one-hot encoding in EncodeEPPBS output.
func TestEPPBSEncodeTagSection(t *testing.T) {
	var a AgentState
	a.SlotTags[0] = TagPrivOwn
	a.SlotTags[1] = TagUnk
	a.SlotTags[2] = TagPrivOpp
	a.SlotTags[3] = TagPub

	var out [EPPBSInputDim]float32
	a.EncodeEPPBS(engine.CtxStartTurn, -1, &out)

	// Tag section starts at offset 40.
	tagStart := 40
	tests := []struct {
		slot    int
		wantTag EpistemicTag
	}{
		{0, TagPrivOwn},
		{1, TagUnk},
		{2, TagPrivOpp},
		{3, TagPub},
	}
	for _, tt := range tests {
		base := tagStart + tt.slot*4
		for v := 0; v < 4; v++ {
			want := float32(0)
			if EpistemicTag(v) == tt.wantTag {
				want = 1.0
			}
			if out[base+v] != want {
				t.Errorf("slot %d tag dim %d = %v, want %v (tag=%v)", tt.slot, v, out[base+v], want, tt.wantTag)
			}
		}
	}
}
