package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// helper: create a minimal AgentState with 0 cards
func newTestAgent() AgentState {
	return NewAgentState(0, 1, 0, 0)
}

// TestEncodeKnownHand verifies one-hot positions for known buckets in own hand.
func TestEncodeKnownHand(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 3
	a.OwnHand[0] = KnownCardInfo{Bucket: BucketZero}
	a.OwnHand[1] = KnownCardInfo{Bucket: BucketAce}
	a.OwnHand[2] = KnownCardInfo{Bucket: BucketMidNum}

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)

	// Slot 0: BucketZero(0) → index 0 → feature[0] = 1.0
	if out[0] != 1.0 {
		t.Errorf("slot0 BucketZero: expected out[0]=1.0, got %f", out[0])
	}
	// Slot 1: BucketAce(2) → index 2 → feature[15+2] = 1.0
	if out[17] != 1.0 {
		t.Errorf("slot1 BucketAce: expected out[17]=1.0, got %f", out[17])
	}
	// Slot 2: BucketMidNum(4) → index 4 → feature[30+4] = 1.0
	if out[34] != 1.0 {
		t.Errorf("slot2 BucketMidNum: expected out[34]=1.0, got %f", out[34])
	}
}

// TestEncodeUnknownSlot verifies BucketUnknown maps to one-hot index 13 (not 9).
func TestEncodeUnknownSlot(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 1
	a.OwnHand[0] = KnownCardInfo{Bucket: BucketUnknown}

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)

	// Slot 0: BucketUnknown → index 13 → feature[13] = 1.0
	if out[13] != 1.0 {
		t.Errorf("BucketUnknown: expected out[13]=1.0, got %f", out[13])
	}
	// Index 9 should NOT be set for BucketUnknown
	if out[9] != 0.0 {
		t.Errorf("BucketUnknown: expected out[9]=0.0, got %f", out[9])
	}
}

// TestEncodeEmptySlot verifies that slots beyond OwnHandLen use one-hot index 14.
func TestEncodeEmptySlot(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 2
	a.OwnHand[0] = KnownCardInfo{Bucket: BucketZero}
	a.OwnHand[1] = KnownCardInfo{Bucket: BucketZero}

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)

	// Slot 2 (index 2, offset 30): EMPTY → index 14 → feature[30+14] = 1.0
	if out[44] != 1.0 {
		t.Errorf("empty slot2: expected out[44]=1.0, got %f", out[44])
	}
	// Slot 5 (offset 75): EMPTY → index 14 → feature[75+14] = 1.0
	if out[89] != 1.0 {
		t.Errorf("empty slot5: expected out[89]=1.0, got %f", out[89])
	}
}

// TestEncodeDecayCategories verifies DecayCategory values map to indices 10-13.
func TestEncodeDecayCategories(t *testing.T) {
	a := newTestAgent()
	a.OppHandLen = 4
	a.OppBelief[0] = DecayBelief(DecayLikelyLow)  // → index 10
	a.OppBelief[1] = DecayBelief(DecayLikelyMid)  // → index 11
	a.OppBelief[2] = DecayBelief(DecayLikelyHigh) // → index 12
	a.OppBelief[3] = DecayBelief(DecayUnknown)    // → index 13

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)

	// Opponent beliefs start at offset 90.
	// Slot 0 of opp: offset 90, index 10 → feature[100]
	if out[100] != 1.0 {
		t.Errorf("opp slot0 DecayLikelyLow: expected out[100]=1.0, got %f", out[100])
	}
	// Slot 1: offset 105, index 11 → feature[116]
	if out[116] != 1.0 {
		t.Errorf("opp slot1 DecayLikelyMid: expected out[116]=1.0, got %f", out[116])
	}
	// Slot 2: offset 120, index 12 → feature[132]
	if out[132] != 1.0 {
		t.Errorf("opp slot2 DecayLikelyHigh: expected out[132]=1.0, got %f", out[132])
	}
	// Slot 3: offset 135, index 13 → feature[148]
	if out[148] != 1.0 {
		t.Errorf("opp slot3 DecayUnknown: expected out[148]=1.0, got %f", out[148])
	}
}

// TestEncodeCardCounts verifies normalized card counts.
func TestEncodeCardCounts(t *testing.T) {
	a := newTestAgent()
	a.OwnHandLen = 4
	a.OppHandLen = 3
	for i := uint8(0); i < a.OwnHandLen; i++ {
		a.OwnHand[i] = KnownCardInfo{Bucket: BucketZero}
	}
	for i := uint8(0); i < a.OppHandLen; i++ {
		a.OppBelief[i] = BucketBelief(BucketUnknown)
	}

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)

	// feature[180] = 4/6
	expected180 := float32(4) / float32(6)
	if out[180] != expected180 {
		t.Errorf("own card count: expected out[180]=%f, got %f", expected180, out[180])
	}
	// feature[181] = 3/6
	expected181 := float32(3) / float32(6)
	if out[181] != expected181 {
		t.Errorf("opp card count: expected out[181]=%f, got %f", expected181, out[181])
	}
}

// TestEncodeDrawnCardBucket verifies drawn card one-hot encoding.
func TestEncodeDrawnCardBucket(t *testing.T) {
	// drawnCardBucket=2 (BucketAce) → index 2 → feature[182+2] = feature[184]
	a := newTestAgent()
	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, 2, &out)
	if out[184] != 1.0 {
		t.Errorf("drawn=2 (BucketAce): expected out[184]=1.0, got %f", out[184])
	}

	// drawnCardBucket=-1 (NONE) → index 10 → feature[182+10] = feature[192]
	var out2 [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out2)
	if out2[192] != 1.0 {
		t.Errorf("drawn=-1 (NONE): expected out2[192]=1.0, got %f", out2[192])
	}
}

// TestEncodeDiscardTop verifies discard top bucket one-hot encoding.
func TestEncodeDiscardTop(t *testing.T) {
	a := newTestAgent()
	a.DiscardTopBucket = BucketMidNum // 4 → index 4 → feature[193+4] = feature[197]

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)
	if out[197] != 1.0 {
		t.Errorf("discard BucketMidNum: expected out[197]=1.0, got %f", out[197])
	}
}

// TestEncodeStockEstimate verifies stockpile estimate one-hot encoding.
func TestEncodeStockEstimate(t *testing.T) {
	a := newTestAgent()
	a.StockEstimate = StockMedium // 1 → index 1 → feature[203+1] = feature[204]

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)
	if out[204] != 1.0 {
		t.Errorf("StockMedium: expected out[204]=1.0, got %f", out[204])
	}
}

// TestEncodeGamePhase verifies game phase one-hot encoding.
func TestEncodeGamePhase(t *testing.T) {
	a := newTestAgent()
	a.Phase = PhaseMid // 2 → index 2 → feature[207+2] = feature[209]

	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)
	if out[209] != 1.0 {
		t.Errorf("PhaseMid: expected out[209]=1.0, got %f", out[209])
	}
}

// TestEncodeDecisionContext verifies decision context one-hot encoding.
func TestEncodeDecisionContext(t *testing.T) {
	a := newTestAgent()
	// CtxPostDraw(1) → index 1 → feature[213+1] = feature[214]
	var out [InputDim]float32
	a.Encode(engine.CtxPostDraw, -1, &out)
	if out[214] != 1.0 {
		t.Errorf("CtxPostDraw: expected out[214]=1.0, got %f", out[214])
	}
}

// TestEncodeCambiaState verifies cambia state one-hot encoding.
func TestEncodeCambiaState(t *testing.T) {
	// CambiaSelf → index 0 → feature[219]
	a := newTestAgent()
	a.CambiaState = CambiaSelf
	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)
	if out[219] != 1.0 {
		t.Errorf("CambiaSelf: expected out[219]=1.0, got %f", out[219])
	}

	// CambiaOpponent → index 1 → feature[220]
	a.CambiaState = CambiaOpponent
	var out2 [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out2)
	if out2[220] != 1.0 {
		t.Errorf("CambiaOpponent: expected out2[220]=1.0, got %f", out2[220])
	}

	// CambiaNone → index 2 → feature[221]
	a.CambiaState = CambiaNone
	var out3 [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out3)
	if out3[221] != 1.0 {
		t.Errorf("CambiaNone: expected out3[221]=1.0, got %f", out3[221])
	}
}

// TestEncodeFullVector verifies properties of a fully populated encoding.
func TestEncodeFullVector(t *testing.T) {
	a := newTestAgent()
	// Mix: 3 own cards (known/unknown/known), 2 opp (bucket/decay), rest empty
	a.OwnHandLen = 3
	a.OwnHand[0] = KnownCardInfo{Bucket: BucketPeekSelf}
	a.OwnHand[1] = KnownCardInfo{Bucket: BucketUnknown}
	a.OwnHand[2] = KnownCardInfo{Bucket: BucketHighKing}

	a.OppHandLen = 2
	a.OppBelief[0] = BucketBelief(BucketLowNum)
	a.OppBelief[1] = DecayBelief(DecayLikelyMid)

	a.DiscardTopBucket = BucketPeekOther
	a.StockEstimate = StockLow
	a.Phase = PhaseLate
	a.CambiaState = CambiaNone

	var out [InputDim]float32
	a.Encode(engine.CtxAbilitySelect, 5, &out)

	// Count one-hot entries (values == 1.0)
	oneHotCount := 0
	for _, v := range out {
		if v == 1.0 {
			oneHotCount++
		}
	}
	// 6 own slots + 6 opp slots + drawn + discard + stock + phase + ctx + cambia = 18
	if oneHotCount != 18 {
		t.Errorf("expected 18 one-hot entries, got %d", oneHotCount)
	}

	// No negative values
	for i, v := range out {
		if v < 0 {
			t.Errorf("negative value at index %d: %f", i, v)
		}
	}

	// No values > 1.0
	for i, v := range out {
		if v > 1.0 {
			t.Errorf("value > 1.0 at index %d: %f", i, v)
		}
	}

	// Sum check: 18 one-hots + ownCount/6 + oppCount/6
	var sum float32
	for _, v := range out {
		sum += v
	}
	ownNorm := float32(3) / float32(6)
	oppNorm := float32(2) / float32(6)
	expectedSum := float32(18) + ownNorm + oppNorm
	// Allow tiny float tolerance
	diff := sum - expectedSum
	if diff < -0.001 || diff > 0.001 {
		t.Errorf("vector sum: expected ~%f, got %f", expectedSum, sum)
	}
}

// TestEncodeVectorDimension verifies the output is exactly 222 elements.
func TestEncodeVectorDimension(t *testing.T) {
	a := newTestAgent()
	var out [InputDim]float32
	a.Encode(engine.CtxStartTurn, -1, &out)
	if len(out) != 222 {
		t.Errorf("expected 222 dimensions, got %d", len(out))
	}
}

// TestActionMask verifies that ActionMask correctly maps bitmask bits to boolean array.
func TestActionMask(t *testing.T) {
	var legal [3]uint64
	// Set bits 0, 5, 64, 97, 145
	legal[0] = (1 << 0) | (1 << 5)
	legal[1] = (1 << 0) // bit 64
	legal[1] |= (1 << 33) // bit 97
	legal[2] = (1 << 17) // bit 145

	var mask [NumActions]bool
	ActionMask(legal, &mask)

	expected := map[int]bool{0: true, 5: true, 64: true, 97: true, 145: true}
	for i := 0; i < NumActions; i++ {
		want := expected[i]
		if mask[i] != want {
			t.Errorf("mask[%d]: expected %v, got %v", i, want, mask[i])
		}
	}
}

// TestActionMaskEmpty verifies all-zero bitmask produces all-false output.
func TestActionMaskEmpty(t *testing.T) {
	var legal [3]uint64
	var mask [NumActions]bool
	ActionMask(legal, &mask)

	for i, v := range mask {
		if v {
			t.Errorf("empty mask: expected mask[%d]=false, got true", i)
		}
	}
}

// TestActionMaskFull verifies bits 0-145 all set produce all-true output.
func TestActionMaskFull(t *testing.T) {
	var legal [3]uint64
	// Bits 0-63 in word 0
	legal[0] = ^uint64(0)
	// Bits 64-127 in word 1
	legal[1] = ^uint64(0)
	// Bits 128-145 in word 2: need bits 0-17 set (145-128=17)
	legal[2] = (1 << 18) - 1 // bits 0-17

	var mask [NumActions]bool
	ActionMask(legal, &mask)

	for i, v := range mask {
		if !v {
			t.Errorf("full mask: expected mask[%d]=true, got false", i)
		}
	}
}

// BenchmarkEncode measures encoding throughput and allocation.
func BenchmarkEncode(b *testing.B) {
	a := NewAgentState(0, 1, 0, 0)
	a.OwnHandLen = 4
	for i := uint8(0); i < a.OwnHandLen; i++ {
		a.OwnHand[i] = KnownCardInfo{Bucket: BucketMidNum}
	}
	a.OppHandLen = 4
	for i := uint8(0); i < a.OppHandLen; i++ {
		a.OppBelief[i] = BucketBelief(BucketUnknown)
	}
	a.DiscardTopBucket = BucketPeekOther
	a.StockEstimate = StockMedium
	a.Phase = PhaseMid
	a.CambiaState = CambiaNone

	var out [InputDim]float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Encode(engine.CtxStartTurn, -1, &out)
	}
}
