package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// nplayer_encoding_seat_test.go covers what cambia-1551 added to EncodeNPlayer: the
// agent's own seat, the table's seat count, and every seat's hand length, with the slot
// blocks ordered relative to the acting seat. The vector used to carry none of the three,
// so a four-seat table and an eight-seat one encoded alike and every slot past a hand's
// end read as a card.

// encodeNPlayer is a small wrapper that returns the vector rather than filling a buffer.
func encodeNPlayer(a *AgentState, ctx engine.DecisionContext, drawn int8) [NPlayerInputDim]float32 {
	var out [NPlayerInputDim]float32
	a.EncodeNPlayer(ctx, drawn, &out)
	return out
}

// TestNPlayerEncodingSeparatesTheSeatCount takes one belief state and changes nothing but
// the number of seats at the table.
func TestNPlayerEncodingSeparatesTheSeatCount(t *testing.T) {
	g := makeNPlayerGame(t, 4, 4551)
	four := NewNPlayerAgentState(0, 4, 1, 5)
	four.InitializeNPlayer(&g)
	five := four
	five.NumPlayers = 5

	if encodeNPlayer(&four, engine.CtxStartTurn, -1) == encodeNPlayer(&five, engine.CtxStartTurn, -1) {
		t.Error("a four-seat and a five-seat table encode to the same vector")
	}
}

// TestNPlayerEncodingSeparatesAnOpponentHandLength takes one belief state and changes
// nothing but how many cards an opponent holds.
func TestNPlayerEncodingSeparatesAnOpponentHandLength(t *testing.T) {
	g := makeNPlayerGame(t, 4, 4552)
	full := NewNPlayerAgentState(0, 4, 1, 5)
	full.InitializeNPlayer(&g)
	short := full
	short.NPlayerHandLen[1]--

	if encodeNPlayer(&full, engine.CtxStartTurn, -1) == encodeNPlayer(&short, engine.CtxStartTurn, -1) {
		t.Error("an opponent holding one card fewer encodes to the same vector")
	}
}

// TestNPlayerEncodingOrdersSlotBlocksRelativeToTheActingSeat reads the encoding of an agent
// sitting at seat 2: its own hand must occupy the first slot block, and its opponents the
// blocks after it in the order the action space indexes them (0, 1, 3).
func TestNPlayerEncodingOrdersSlotBlocksRelativeToTheActingSeat(t *testing.T) {
	g := makeNPlayerGame(t, 4, 4553)
	a := NewNPlayerAgentState(2, 4, 1, 5)
	a.InitializeNPlayer(&g)
	// Clear the initial peeks so the only knowledge in the vector is what this test plants.
	for slot := 0; slot < MaxTotalSlots; slot++ {
		a.nplayerClearKnowledge(slot)
	}
	// One card per seat, each a distinct bucket, planted at that seat's slot 0.
	planted := map[uint8]CardBucket{
		2: BucketZero,     // own seat -> relative block 0
		0: BucketAce,      // first opponent -> relative block 1
		1: BucketLowNum,   // second opponent -> relative block 2
		3: BucketHighKing, // third opponent -> relative block 3
	}
	for seat, bucket := range planted {
		a.nplayerSetKnown(nplayerSlot(seat, 0), bucket)
	}

	out := encodeNPlayer(&a, engine.CtxStartTurn, -1)
	wantRel := []struct {
		rel    int
		bucket CardBucket
	}{{0, BucketZero}, {1, BucketAce}, {2, BucketLowNum}, {3, BucketHighKing}}
	for _, w := range wantRel {
		base := NPlayerPowersetDim + w.rel*engine.MaxHandSize*9
		if out[base+int(w.bucket)] != 1.0 {
			t.Errorf("relative block %d slot 0: bucket %d not set", w.rel, w.bucket)
		}
	}
	// The knower axis is relative too: this agent is knower 0 of every slot it planted.
	for _, w := range wantRel {
		base := w.rel * engine.MaxHandSize * MaxKnowledgePlayers
		if out[base] != 1.0 {
			t.Errorf("relative block %d slot 0: the acting seat is not knower 0", w.rel)
		}
	}
}

// TestNPlayerEncodingMasksSlotsPastTheHandLength plants a record on a slot no card sits in
// and reads it back out of the vector, where it must not appear.
func TestNPlayerEncodingMasksSlotsPastTheHandLength(t *testing.T) {
	g := makeNPlayerGame(t, 4, 4554)
	a := NewNPlayerAgentState(0, 4, 1, 5)
	a.InitializeNPlayer(&g)
	handLen := a.NPlayerHandLen[0]
	if handLen >= engine.MaxHandSize {
		t.Fatalf("hand length %d leaves no slot past the hand's end to test", handLen)
	}
	a.nplayerSetKnown(nplayerSlot(0, handLen), BucketHighKing)

	out := encodeNPlayer(&a, engine.CtxStartTurn, -1)
	identity := NPlayerPowersetDim + int(handLen)*9
	for b := 0; b < 9; b++ {
		if out[identity+b] != 0.0 {
			t.Errorf("identity dim %d of the slot past the hand's end is set", b)
		}
	}
	powerset := int(handLen) * MaxKnowledgePlayers
	for k := 0; k < MaxKnowledgePlayers; k++ {
		if out[powerset+k] != 0.0 {
			t.Errorf("knower bit %d of the slot past the hand's end is set", k)
		}
	}
}

// TestNPlayerAgentStateConstructsAtEightSeats covers the OpponentIDs resize: the array held
// five entries while the engine deals up to eight seats, so seven and eight seat agents
// wrote past its end.
func TestNPlayerAgentStateConstructsAtEightSeats(t *testing.T) {
	for _, seats := range []uint8{7, 8} {
		a := NewNPlayerAgentState(0, seats, 1, 5)
		if a.NumOpponents != seats-1 {
			t.Errorf("%d seats: NumOpponents = %d, want %d", seats, a.NumOpponents, seats-1)
		}
		for i := uint8(0); i < a.NumOpponents; i++ {
			if a.OpponentIDs[i] != i+1 {
				t.Errorf("%d seats: OpponentIDs[%d] = %d, want %d", seats, i, a.OpponentIDs[i], i+1)
			}
		}
	}
	// A table larger than the engine can deal clamps rather than overruns.
	a := NewNPlayerAgentState(0, 12, 1, 5)
	if a.NumPlayers != engine.MaxPlayers {
		t.Errorf("NumPlayers = %d, want %d (clamped)", a.NumPlayers, engine.MaxPlayers)
	}
}
