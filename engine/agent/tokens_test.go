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
		{peekFrameBase, peekFrameBase + numPeekFrameIDs},
		{raceFrameBase, raceFrameBase + numRaceFrameIDs},
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
	// Frozen expected values (must match the Python tokenizer): 325 ids before the
	// cambia-529 peek-result block (+1 marker = 326), then the cambia-564 race
	// block (+1 marker appended at the end = 327).
	if vocabSize != 327 {
		t.Fatalf("vocabSize=%d, want 327", vocabSize)
	}
	if peekFrameBase != 325 {
		t.Fatalf("peekFrameBase=%d, want 325 (appended after the outcome block)", peekFrameBase)
	}
	if raceFrameBase != 326 {
		t.Fatalf("raceFrameBase=%d, want 326 (appended after the peek block)", raceFrameBase)
	}
	if numActionIDs != 240 {
		t.Fatalf("numActionIDs=%d, want 240", numActionIDs)
	}
}

// TestTokenizerVersionFrozen pins the tokenizer stream version. Bump it (and the
// mirror in cfr/src/sequence_encoding.py::TOKENIZER_VERSION) on every change to
// the produced token stream; the FFI cross-check test asserts Go == Python.
func TestTokenizerVersionFrozen(t *testing.T) {
	if TokenizerVersion != 3 {
		t.Fatalf("TokenizerVersion=%d, want 3 (v3 = cambia-564 race frame; bump "+
			"with the Python mirror on any token-stream change)", TokenizerVersion)
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

// fourSeatRules returns house rules for a 4-player game, matching the pattern engine's own
// nplayerRules test helper uses (engine/nplayer_test.go), which package agent cannot import
// directly since it lives in a _test.go file.
func fourSeatRules() engine.HouseRules {
	r := engine.DefaultHouseRules()
	r.NumPlayers = 4
	r.MaxGameTurns = 200
	return r
}

// TestKingLookPeekPairNamesRecordedTargetSeatAtFourSeats pins the king-look peek-pair frame's
// opponent seat to the target Pending.Data[3] records, not OpponentOf(actor) (1-acting), which
// underflows past seat 1 (cambia-1171 fixed the apply paths against exactly this bug; this frame
// was the one 3+-seat token consumer still deriving the seat that way). Actor 1 targets seat 3
// (Opponents(1) = [0, 2, 3], relative index 2): neither OpponentOf(1) (underflows to 255, then
// clamps) nor seatOpponent's next-seat default ((1+1)%4 = 2) would name seat 3, so a regression to
// either reads as a wrong-but-plausible seat rather than an obvious break.
func TestKingLookPeekPairNamesRecordedTargetSeatAtFourSeats(t *testing.T) {
	g := engine.NewGame(777, fourSeatRules())
	g.Deal()

	const actor = uint8(1)
	const target = uint8(3)
	g.CurrentPlayer = actor
	g.Pending.Type = engine.PendingKingLook
	g.Pending.PlayerID = actor

	ownCard := g.Players[actor].Hand[0]
	targetCard := g.Players[target].Hand[0]

	// Opponents(1) = [0, 2, 3]; relative index 2 resolves to absolute seat 3.
	if err := g.ApplyNPlayerAction(engine.NPlayerEncodeKingLook(0, 0, 2)); err != nil {
		t.Fatalf("KingLook(ownSlot=0, oppSlot=0, relIdx=2) from seat %d: %v", actor, err)
	}
	if g.Pending.Type != engine.PendingKingDecision {
		t.Fatalf("Pending.Type=%d after the look, want PendingKingDecision", g.Pending.Type)
	}

	var ts TokenStream
	if err := ts.Observe(&g, actor); err != nil {
		t.Fatalf("Observe: %v", err)
	}

	// Collect the peek frames (frameToken, actorToken, slotToken, cardToken) in emission order:
	// own card first, then the target's.
	var frames [][4]int32
	for i := int32(0); i+3 < ts.Length; i++ {
		if ts.Tokens[i] == peekFrameToken() {
			var f [4]int32
			copy(f[:], ts.Tokens[i:i+4])
			frames = append(frames, f)
			i += 3
		}
	}
	if len(frames) != 2 {
		t.Fatalf("expected 2 peek frames, found %d (tokens=%v)", len(frames), ts.Tokens[:ts.Length])
	}
	ownFrame, oppFrame := frames[0], frames[1]

	if ownFrame[1] != actorToken(int(actor)) || ownFrame[3] != cardToken(ownCard) {
		t.Errorf("own peek frame = %v, want actor token %d and card token %d",
			ownFrame, actorToken(int(actor)), cardToken(ownCard))
	}
	if oppFrame[1] != actorToken(int(target)) {
		t.Errorf("opponent peek frame names actor token %d, want seat %d's actor token %d (the recorded "+
			"target, not OpponentOf(1) or seat+1)", oppFrame[1], target, actorToken(int(target)))
	}
	if oppFrame[3] != cardToken(targetCard) {
		t.Errorf("opponent peek frame card token = %d, want seat %d's slot-0 card token %d",
			oppFrame[3], target, cardToken(targetCard))
	}
}
