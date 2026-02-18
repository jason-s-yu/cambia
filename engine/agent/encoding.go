package agent

import engine "github.com/jason-s-yu/cambia/engine"

const (
	InputDim    = 222
	NumActions  = 146
	MaxHand     = engine.MaxHandSize // 6
	SlotDim     = 15                 // per-slot one-hot dimension
	EmptySlotIdx = 14
)

// bucketSlotIndex returns the one-hot index (0-14) for a CardBucket.
// BucketZero(0)..BucketHighKing(8) map to indices 0-8.
// BucketUnknown(9) maps to index 13.
func bucketSlotIndex(b CardBucket) uint8 {
	if b == BucketUnknown {
		return 13
	}
	return uint8(b) // 0-8 map directly
}

// beliefSlotIndex returns the one-hot index (0-14) for a BeliefValue.
func beliefSlotIndex(bv BeliefValue) uint8 {
	if bv.IsBucket() {
		return bucketSlotIndex(bv.Bucket())
	}
	// Decay category: DecayLikelyLow=0→10, DecayLikelyMid=1→11, DecayLikelyHigh=2→12, DecayUnknown=3→13
	d := bv.Decay()
	if d == DecayUnknown {
		return 13
	}
	return 10 + uint8(d)
}

// cambiaOneHotIndex returns the one-hot index for CambiaState.
// Python order: SELF=0, OPPONENT=1, NONE=2.
// Go values:    CambiaNone=0, CambiaSelf=1, CambiaOpponent=2.
func cambiaOneHotIndex(cs CambiaState) uint8 {
	switch cs {
	case CambiaSelf:
		return 0
	case CambiaOpponent:
		return 1
	default: // CambiaNone
		return 2
	}
}

// drawnCardOneHotIndex returns the one-hot index for the drawn card encoding.
// bucket=-1 means no drawn card → index 10 (NONE).
// bucket 0-9 → index 0-9 directly.
func drawnCardOneHotIndex(bucket int8) uint8 {
	if bucket < 0 {
		return 10 // NONE
	}
	return uint8(bucket) // 0-9 map directly
}

// Encode writes the 222-dim feature vector into out.
// drawnCardBucket is -1 for no drawn card, or 0-9 for a CardBucket value.
// out is zeroed internally before writing.
func (a *AgentState) Encode(ctx engine.DecisionContext, drawnCardBucket int8, out *[InputDim]float32) {
	// Zero out
	*out = [InputDim]float32{}

	offset := 0

	// Own hand: 6 slots × 15-dim one-hot = 90
	for i := uint8(0); i < MaxHand; i++ {
		var idx uint8
		if i < a.OwnHandLen {
			idx = bucketSlotIndex(a.OwnHand[i].Bucket)
		} else {
			idx = EmptySlotIdx
		}
		out[offset+int(idx)] = 1.0
		offset += SlotDim
	}
	// offset = 90

	// Opponent beliefs: 6 slots × 15-dim one-hot = 90
	for i := uint8(0); i < MaxHand; i++ {
		var idx uint8
		if i < a.OppHandLen {
			idx = beliefSlotIndex(a.OppBelief[i])
		} else {
			idx = EmptySlotIdx
		}
		out[offset+int(idx)] = 1.0
		offset += SlotDim
	}
	// offset = 180

	// Own card count (normalized)
	ownCount := a.OwnHandLen
	if ownCount > MaxHand {
		ownCount = MaxHand
	}
	out[offset] = float32(ownCount) / float32(MaxHand)
	offset++
	// offset = 181

	// Opponent card count (normalized)
	oppCount := a.OppHandLen
	if oppCount > MaxHand {
		oppCount = MaxHand
	}
	out[offset] = float32(oppCount) / float32(MaxHand)
	offset++
	// offset = 182

	// Drawn card bucket: 11-dim one-hot
	out[offset+int(drawnCardOneHotIndex(drawnCardBucket))] = 1.0
	offset += 11
	// offset = 193

	// Discard top bucket: 10-dim one-hot
	out[offset+int(a.DiscardTopBucket)] = 1.0
	offset += 10
	// offset = 203

	// Stockpile estimate: 4-dim one-hot
	out[offset+int(a.StockEstimate)] = 1.0
	offset += 4
	// offset = 207

	// Game phase: 6-dim one-hot
	out[offset+int(a.Phase)] = 1.0
	offset += 6
	// offset = 213

	// Decision context: 6-dim one-hot
	out[offset+int(ctx)] = 1.0
	offset += 6
	// offset = 219

	// Cambia state: 3-dim one-hot
	out[offset+int(cambiaOneHotIndex(a.CambiaState))] = 1.0
	// offset = 222
}

// ActionMask writes the legal action mask into out.
// legalActions is the bitmask from GameState.LegalActions().
func ActionMask(legalActions [3]uint64, out *[NumActions]bool) {
	*out = [NumActions]bool{}
	for i := uint16(0); i < NumActions; i++ {
		word := i / 64
		bit := i % 64
		if legalActions[word]&(1<<bit) != 0 {
			out[i] = true
		}
	}
}
