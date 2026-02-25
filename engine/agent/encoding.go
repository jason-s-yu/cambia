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

// EncodeEPPBS writes the 200-dim EP-PBS feature vector into out.
// Layout (200 total):
//
//	[0-9]   discard top bucket (10-dim one-hot: BucketZero..BucketHighKing=0..8, BucketUnknown=9)
//	[10-13] stockpile estimate (4-dim one-hot)
//	[14-19] game phase (6-dim one-hot)
//	[20-25] decision context (6-dim one-hot)
//	[26-28] cambia state (3-dim one-hot, same mapping as Encode)
//	[29-39] drawn card bucket (11-dim one-hot: 0-9=bucket, 10=no card)
//	[40-87] slot tags (12 slots × 4-dim one-hot EpistemicTag)
//	[88-195] slot identities (12 slots × 9-dim one-hot bucket, 0=BucketZero..8=BucketHighKing)
//	          Non-zero only when tag is TagPrivOwn or TagPub.
//	[196-199] padding (always 0)
func (a *AgentState) EncodeEPPBS(ctx engine.DecisionContext, drawnCardBucket int8, out *[EPPBSInputDim]float32) {
	*out = [EPPBSInputDim]float32{}
	offset := 0

	// [0-9] Discard top bucket (10-dim).
	out[offset+int(a.DiscardTopBucket)] = 1.0
	offset += 10

	// [10-13] Stockpile estimate (4-dim).
	out[offset+int(a.StockEstimate)] = 1.0
	offset += 4

	// [14-19] Game phase (6-dim).
	out[offset+int(a.Phase)] = 1.0
	offset += 6

	// [20-25] Decision context (6-dim).
	out[offset+int(ctx)] = 1.0
	offset += 6

	// [26-28] Cambia state (3-dim, same mapping as Encode).
	out[offset+int(cambiaOneHotIndex(a.CambiaState))] = 1.0
	offset += 3

	// [29-39] Drawn card bucket (11-dim).
	out[offset+int(drawnCardOneHotIndex(drawnCardBucket))] = 1.0
	offset += 11
	// offset = 40

	// [40-87] Slot tags (12 slots × 4-dim one-hot).
	for i := 0; i < MaxSlots; i++ {
		out[offset+int(a.SlotTags[i])] = 1.0
		offset += 4
	}
	// offset = 88

	// [88-195] Slot identities (12 slots × 9-dim one-hot).
	// Only encode when we know the card (TagPrivOwn or TagPub).
	for i := 0; i < MaxSlots; i++ {
		tag := a.SlotTags[i]
		if tag == TagPrivOwn || tag == TagPub {
			b := a.SlotBuckets[i]
			if b < BucketUnknown { // known bucket (0-8)
				out[offset+int(b)] = 1.0
			}
		}
		offset += 9
	}
	// offset = 196 — remaining 4 bytes are padding (already zero).
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

// EncodeNPlayer writes the 580-dim N-player feature vector into out.
// Layout:
//
//	[0-215]   Powerset masks: 36 slots × 6 bits (which players know each card)
//	[216-539] Slot identities: 36 slots × 9 buckets (one-hot, zeroed if agent doesn't know)
//	[540-579] Public features: discard(10) + stock(4) + phase(6) + ctx(6) + cambia(3) + drawn(11) = 40
func (a *AgentState) EncodeNPlayer(ctx engine.DecisionContext, drawnCardBucket int8, out *[NPlayerInputDim]float32) {
	*out = [NPlayerInputDim]float32{}
	offset := 0

	// Powerset masks: 36 slots × 6 bits = 216
	for slot := 0; slot < MaxTotalSlots; slot++ {
		for p := 0; p < MaxKnowledgePlayers; p++ {
			if a.KnowledgeMask[slot][p] {
				out[offset] = 1.0
			}
			offset++
		}
	}
	// offset = 216

	// Slot identities: 36 slots × 9 buckets = 324
	for slot := 0; slot < MaxTotalSlots; slot++ {
		if a.NPlayerSlotKnown[slot] {
			b := a.NPlayerSlotBuckets[slot]
			if b < BucketUnknown { // 0-8
				out[offset+int(b)] = 1.0
			}
		}
		offset += 9
	}
	// offset = 540

	// Public features: 40 dims total
	// Discard top bucket (10)
	out[offset+int(a.DiscardTopBucket)] = 1.0
	offset += 10
	// Stock estimate (4)
	out[offset+int(a.StockEstimate)] = 1.0
	offset += 4
	// Phase (6)
	out[offset+int(a.Phase)] = 1.0
	offset += 6
	// Decision context (6)
	out[offset+int(ctx)] = 1.0
	offset += 6
	// Cambia state (3)
	out[offset+int(cambiaOneHotIndex(a.CambiaState))] = 1.0
	offset += 3
	// Drawn card (11)
	out[offset+int(drawnCardOneHotIndex(drawnCardBucket))] = 1.0
	// offset = 580
}

// NPlayerActionMask writes the N-player legal action mask into out.
// legalActions is the 8×uint64 bitmask from GameState.NPlayerLegalActions().
func NPlayerActionMask(legalActions [8]uint64, out *[NPlayerNumActions]bool) {
	*out = [NPlayerNumActions]bool{}
	for i := uint16(0); i < engine.NPlayerNumActions; i++ {
		word := i / 64
		bit := i % 64
		if legalActions[word]&(1<<bit) != 0 {
			out[i] = true
		}
	}
}
