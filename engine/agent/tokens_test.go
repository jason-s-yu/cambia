package agent

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// TestTokenVocabLayoutContiguous asserts the vocabulary blocks tile
// [0, vocabSize) with no gaps or overlaps, matching sequence_encoding.py.
func TestTokenVocabLayoutContiguous(t *testing.T) {
	type span struct{ start, end int32 }
	spans := []span{
		{0, numSpecial},
		{frameBase, frameBase + numFrameIDs},
		{actorBase, actorBase + tokMaxActors},
		{actionBase, actionBase + numActionIDs},
		{cardBase, cardBase + numCardIDs},
		{slotBase, slotBase + tokNumSlotIDs},
		{outcomeBase, outcomeBase + numSnapOutcomeIDs},
	}
	cursor := int32(0)
	for _, s := range spans {
		if s.start != cursor {
			t.Fatalf("vocab gap/overlap: expected %d, got %d", cursor, s.start)
		}
		cursor = s.end
	}
	if cursor != vocabSize {
		t.Fatalf("vocab blocks end at %d, vocabSize=%d", cursor, vocabSize)
	}
	// Frozen expected values (must match the Python tokenizer commit: 325 ids).
	if vocabSize != 325 {
		t.Fatalf("vocabSize=%d, want 325", vocabSize)
	}
	if numActionIDs != 240 {
		t.Fatalf("numActionIDs=%d, want 240", numActionIDs)
	}
}

// TestActionTokensDistinctAndInRange asserts every 2-player action index maps to
// a valid, distinct ACTION-block token (the property that makes the public path
// determine the legal-action count).
func TestActionTokensDistinctAndInRange(t *testing.T) {
	seen := map[int32]uint16{}
	for idx := uint16(0); idx < engine.NumActions; idx++ {
		tok := actionToken(idx)
		if tok < actionBase || tok >= actionBase+numActionIDs {
			t.Fatalf("action %d -> token %d out of ACTION block [%d,%d)", idx, tok, actionBase, actionBase+numActionIDs)
		}
		if prev, ok := seen[tok]; ok {
			t.Fatalf("action token collision: idx %d and %d both -> %d", prev, idx, tok)
		}
		seen[tok] = idx
	}
	if len(seen) != int(engine.NumActions) {
		t.Fatalf("expected %d distinct action tokens, got %d", engine.NumActions, len(seen))
	}
}

// TestCardTokensCoverIdentities asserts every canonical card index maps into the
// CARD block, jokers collapse to one identity, and EmptyCard maps to none.
func TestCardTokensCoverIdentities(t *testing.T) {
	for goIdx := uint8(0); goIdx < 54; goIdx++ {
		tok := EncodeCardToken(goIdx)
		if tok < cardBase || tok >= cardBase+numCardIDs {
			t.Fatalf("card idx %d -> token %d out of CARD block", goIdx, tok)
		}
	}
	// Both physical jokers collapse to the single joker identity.
	if EncodeCardToken(52) != EncodeCardToken(53) {
		t.Fatalf("jokers 52/53 must share a token: %d vs %d", EncodeCardToken(52), EncodeCardToken(53))
	}
	if EncodeCardToken(52) != cardBase+jokerLocalID {
		t.Fatalf("joker token = %d, want %d", EncodeCardToken(52), cardBase+jokerLocalID)
	}
	if cardToken(engine.EmptyCard) != cardBase+cardNoneLocal {
		t.Fatalf("EmptyCard token = %d, want %d", cardToken(engine.EmptyCard), cardBase+cardNoneLocal)
	}
	// Ace of Spades is CARD_IDENTITIES[0] (rank A pos 0, suit S pos 0) -> local 0.
	as := engine.NewCard(engine.SuitSpades, engine.RankAce)
	if cardLocalID(as) != 0 {
		t.Fatalf("Ace of Spades local id = %d, want 0", cardLocalID(as))
	}
	// Ace of Hearts -> suit H pos 1 -> local 1.
	ah := engine.NewCard(engine.SuitHearts, engine.RankAce)
	if cardLocalID(ah) != 1 {
		t.Fatalf("Ace of Hearts local id = %d, want 1", cardLocalID(ah))
	}
}

// TestInitPeekFramesTinyGame checks the init-peek frames on a freshly dealt game.
func TestInitPeekFramesTinyGame(t *testing.T) {
	g := engine.NewGame(12345, engine.DefaultHouseRules())
	g.Deal()
	var ts TokenStream
	if err := ts.Init(&g, 0); err != nil {
		t.Fatalf("Init overflow: %v", err)
	}
	// Default deal peeks slots 0 and 1: two init_peek frames, 3 tokens each.
	if ts.Len() != 6 {
		t.Fatalf("init-peek body length = %d, want 6 (2 frames x 3)", ts.Len())
	}
	if ts.Tokens[0] != frameToken(frameInitPeek) || ts.Tokens[3] != frameToken(frameInitPeek) {
		t.Fatalf("init-peek frames must start with the init_peek marker")
	}
	// Slot tokens ascending (0 then 1).
	if ts.Tokens[1] != slotToken(0) || ts.Tokens[4] != slotToken(1) {
		t.Fatalf("init-peek slot order wrong: %d, %d", ts.Tokens[1], ts.Tokens[4])
	}
}

// TestTokenOverflowIsHardError forces the stream past the cap and asserts a hard
// error rather than silent truncation.
func TestTokenOverflowIsHardError(t *testing.T) {
	var ts TokenStream
	ts.Length = MaxTokenStream - 2 // room for < one public frame (4 tokens)
	g := engine.NewGame(1, engine.DefaultHouseRules())
	g.Deal()
	// Apply one action so LastAction is populated.
	mask := g.LegalActions()
	var idx uint16
	for i := uint16(0); i < engine.NumActions; i++ {
		if mask[i/64]&(1<<(i%64)) != 0 {
			idx = i
			break
		}
	}
	if err := g.ApplyAction(idx); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if err := ts.Observe(&g, 0); err != ErrTokenOverflow {
		t.Fatalf("expected ErrTokenOverflow, got %v (len=%d)", err, ts.Length)
	}
}
