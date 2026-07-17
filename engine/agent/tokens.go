// tokens.go implements the PRT-CFR perfect-recall sequence tokenizer, mirroring
// cfr/src/sequence_encoding.py byte-for-byte.
//
// A player's observation-action event stream is tokenized into a flat int32
// token buffer as a game advances. The buffer stores the FULL frame body
// (init-peek frames followed by one group of frames per observed action) with
// NO BOS/EOS and NO truncation. BOS/EOS wrapping and the frame-aligned
// keep-most-recent SEQ_CAP window are the consumer's job at encode time (see
// the bridge helper frame_aligned_window); the Go side never truncates.
//
// Single source of truth: cfr/src/sequence_encoding.py. The vocabulary layout
// here is computed the same way Python computes it (a frozen per-tag stride
// table), and a constants cross-check test asserts the exported Go layout
// against the Python module so drift cannot pass silently.
package agent

import (
	engine "github.com/jason-s-yu/cambia/engine"
)

// ---------------------------------------------------------------------------
// Design constants (mirror sequence_encoding.py)
// ---------------------------------------------------------------------------

const (
	// MaxTokenStream is the hard per-agent token buffer cap. A stream that would
	// exceed this is an explicit overflow ERROR, never silent truncation.
	//
	// PAIRED CONSTANT: cfr/src/cfr/prtcfr_worker.py::PRODUCTION_SEQ_CAP (12288).
	// This value MUST stay >= PRODUCTION_SEQ_CAP -- the durable invariant is
	// asserted live by cfr/tests/test_prtcfr_go_bridge_integration.py::
	// test_go_token_stream_cap_at_least_production_cap (reads this value via
	// the cambia_token_stream_cap FFI export, never hardcodes it). Raised
	// 4096 -> 12288 in S1W12 after S1W3's real P100 instrumentation
	// (scripts/prtcfr_p100_instrument.py, ~8800 games, production 300-turn
	// rule profile, avoid_cambia cohort) measured worst-case token length 7284
	// and rising with sample size at n=10000 -- the original 4096 cap (built
	// from the pre-P100 "~726 mean, ~1200 worst" estimate in
	// sequence_encoding.py, measured under the SHORTER 46-turn tiny/test rule
	// profile, not the 300-turn production one) was already below production
	// reality and would hard-error long self-play trajectories. If a future
	// larger P100 run ever approaches this cap, raise both constants together
	// and re-run the P100 script per PRODUCTION_SEQ_CAP's own docstring.
	MaxTokenStream = 12288

	// tokMaxSlots is the maximum hand-slot index the tokenizer represents
	// (MAX_SLOTS in Python). SLOT ids = tokMaxSlots concrete + one "no slot".
	tokMaxSlots      = 8
	tokNumSlotIDs    = tokMaxSlots + 1 // 9
	tokSlotNoneLocal = tokMaxSlots     // 8

	// tokMaxActors is the maximum actor id (MAX_ACTORS in Python).
	tokMaxActors = 8

	// TokSeqCap is the design sequence cap (SEQ_CAP in Python). It is the
	// reference cap for the consumer's frame-aligned window; the Go buffer
	// itself never truncates. Exposed for the cross-check test.
	TokSeqCap = 256

	// Special ids.
	tokPAD     = 0
	tokBOS     = 1
	tokEOS     = 2
	tokSEP     = 3
	numSpecial = 4

	// Card block: 53 distinct identities + a none/unknown id.
	numCardIDs    = 54
	cardNoneLocal = 53
	jokerLocalID  = 52

	// Frame kinds (fixed order): init_peek, public, drawn, snap, cambia.
	frameInitPeek = 0
	framePublic   = 1
	frameDrawn    = 2
	frameSnap     = 3
	frameCambia   = 4
	numFrameIDs   = 5

	// Snap outcomes (fixed order).
	outcomePenalty      = 0
	outcomeSuccessOwn   = 1
	outcomeSuccessOpp   = 2
	outcomeSuccessOther = 3
	outcomeFail         = 4
	numSnapOutcomeIDs   = 5

	// Peek-result block (cambia-529): APPENDED after the snap-outcome block so
	// every id defined above keeps its value (existing token ids never shift).
	// One marker heads each actor-private peek-result frame; the payload reuses
	// the ACTOR (card owner), SLOT, and CARD blocks: [PEEK, OWNER, SLOT, CARD].
	numPeekFrameIDs = 1
	peekFrameWidth  = 4
)

// Block bases. FRAME_BASE..OUTCOME_BASE are computed to match Python's fixed
// assignment order. numActionIDs is derived from the action spec below.
var (
	frameBase     int32 = numSpecial
	actorBase     int32
	actionBase    int32
	cardBase      int32
	slotBase      int32
	outcomeBase   int32
	peekFrameBase int32
	vocabSize     int32

	// numActionIDs is the total size of the ACTION block (sum of per-tag strides).
	numActionIDs int32

	// actionTagOffset maps a tag name to its cumulative offset within the ACTION
	// block. actionTagStride maps a tag name to its per-tag id stride.
	actionTagOffset map[string]int32
)

// Action arity kinds and per-kind strides (MAX_SLOTS = tokMaxSlots).
const (
	tagNone  = 0
	tagFlag  = 1
	tagSlot1 = 2
	tagSlot2 = 3
)

type actionSpecEntry struct {
	name string
	kind int
}

// actionSpec is FROZEN: ids are assigned by cumulative stride in this order.
// It mirrors _ACTION_SPEC in sequence_encoding.py exactly.
var actionSpec = []actionSpecEntry{
	{"draw_stockpile", tagNone},
	{"draw_discard", tagNone},
	{"call_cambia", tagNone},
	{"discard_ability", tagNone},
	{"discard_no_ability", tagNone},
	{"replace", tagSlot1},
	{"peek_own", tagSlot1},
	{"peek_other", tagSlot1},
	{"blind_swap", tagSlot2},
	{"king_look", tagSlot2},
	{"king_swap", tagFlag},
	{"pass_snap", tagNone},
	{"snap_own", tagSlot1},
	{"snap_opp", tagSlot1},
	{"snap_opp_move", tagSlot2},
}

func tagStride(kind int) int32 {
	switch kind {
	case tagNone:
		return 1
	case tagFlag:
		return 2
	case tagSlot1:
		return tokMaxSlots
	case tagSlot2:
		return tokMaxSlots * tokMaxSlots
	}
	return 0
}

func init() {
	actionTagOffset = make(map[string]int32, len(actionSpec))
	var acc int32
	for _, s := range actionSpec {
		actionTagOffset[s.name] = acc
		acc += tagStride(s.kind)
	}
	numActionIDs = acc

	actorBase = frameBase + numFrameIDs
	actionBase = actorBase + tokMaxActors
	cardBase = actionBase + numActionIDs
	slotBase = cardBase + numCardIDs
	outcomeBase = slotBase + tokNumSlotIDs
	peekFrameBase = outcomeBase + numSnapOutcomeIDs
	vocabSize = peekFrameBase + numPeekFrameIDs
}

// ---------------------------------------------------------------------------
// Card <-> id mapping
// ---------------------------------------------------------------------------

// cardLocalID returns the local card id within the CARD block, matching
// Python's CARD_IDENTITIES ordering: identities are [(rank, suit) for rank in
// _SUITED_RANKS for suit in ALL_SUITS] + [(joker, None)], where _SUITED_RANKS
// is A..K (positions 0..12, identical to Go rank encoding) and ALL_SUITS is
// [S, H, D, C]. Both physical jokers collapse to a single identity.
func cardLocalID(card engine.Card) int32 {
	if card == engine.EmptyCard {
		return cardNoneLocal
	}
	r := card.Rank()
	if r == engine.RankJoker {
		return jokerLocalID
	}
	var suitPos int32
	switch card.Suit() {
	case engine.SuitSpades:
		suitPos = 0
	case engine.SuitHearts:
		suitPos = 1
	case engine.SuitDiamonds:
		suitPos = 2
	case engine.SuitClubs:
		suitPos = 3
	default:
		return cardNoneLocal
	}
	if int32(r) > 12 {
		return cardNoneLocal
	}
	return int32(r)*4 + suitPos
}

func cardToken(card engine.Card) int32 { return cardBase + cardLocalID(card) }

// ---------------------------------------------------------------------------
// Actor / slot / action / frame / outcome token encoders
// ---------------------------------------------------------------------------

func clampSlot(idx int) int32 {
	if idx < 0 {
		return 0
	}
	if idx >= tokMaxSlots {
		return tokMaxSlots - 1
	}
	return int32(idx)
}

func actorToken(actor int) int32 {
	a := actor
	if a < 0 || a >= tokMaxActors {
		a = tokMaxActors - 1
	}
	return actorBase + int32(a)
}

func frameToken(kind int) int32 { return frameBase + int32(kind) }

func slotToken(slot int) int32 { return slotBase + clampSlot(slot) }

func slotTokenSigned(slot int) int32 {
	if slot < 0 {
		return slotBase + tokSlotNoneLocal
	}
	return slotBase + clampSlot(slot)
}

func outcomeToken(local int) int32 { return outcomeBase + int32(local) }

// peekFrameToken returns the marker id heading an actor-private peek-result
// frame (cambia-529).
func peekFrameToken() int32 { return peekFrameBase }

// actionLocalTag decodes a 2-player action index into its Python tag name and
// the relative offset within that tag's stride. It mirrors _action_local_id.
// Returns ok=false for indices that do not encode a known action.
func actionLocalTag(idx uint16) (name string, rel int32, ok bool) {
	switch idx {
	case engine.ActionDrawStockpile:
		return "draw_stockpile", 0, true
	case engine.ActionDrawDiscard:
		return "draw_discard", 0, true
	case engine.ActionCallCambia:
		return "call_cambia", 0, true
	case engine.ActionDiscardWithAbility:
		return "discard_ability", 0, true
	case engine.ActionDiscardNoAbility:
		return "discard_no_ability", 0, true
	case engine.ActionPassSnap:
		return "pass_snap", 0, true
	case engine.ActionKingSwapNo:
		return "king_swap", 0, true
	case engine.ActionKingSwapYes:
		return "king_swap", 1, true
	}
	if t, k := engine.ActionIsReplace(idx); k {
		return "replace", clampSlot(int(t)), true
	}
	if t, k := engine.ActionIsPeekOwn(idx); k {
		return "peek_own", clampSlot(int(t)), true
	}
	if t, k := engine.ActionIsPeekOther(idx); k {
		return "peek_other", clampSlot(int(t)), true
	}
	if own, opp, k := engine.ActionIsBlindSwap(idx); k {
		return "blind_swap", clampSlot(int(own))*tokMaxSlots + clampSlot(int(opp)), true
	}
	if own, opp, k := engine.ActionIsKingLook(idx); k {
		return "king_look", clampSlot(int(own))*tokMaxSlots + clampSlot(int(opp)), true
	}
	if t, k := engine.ActionIsSnapOwn(idx); k {
		return "snap_own", clampSlot(int(t)), true
	}
	if t, k := engine.ActionIsSnapOpponent(idx); k {
		return "snap_opp", clampSlot(int(t)), true
	}
	if own, slot, k := engine.ActionIsSnapOpponentMove(idx); k {
		return "snap_opp_move", clampSlot(int(own))*tokMaxSlots + clampSlot(int(slot)), true
	}
	return "", 0, false
}

// actionToken returns the ACTION-block token for a 2-player action index, or -1
// if the index does not encode a known action.
func actionToken(idx uint16) int32 {
	name, rel, ok := actionLocalTag(idx)
	if !ok {
		return -1
	}
	return actionBase + actionTagOffset[name] + rel
}

// isLoggedSnapAction reports whether an action index is one whose snap outcome
// is recorded in the per-phase snap results log (PassSnap, SnapOwn, SnapOpponent).
// SnapOpponentMove is NOT logged (it only completes a successful opponent snap).
func isLoggedSnapAction(idx uint16) bool {
	if idx == engine.ActionPassSnap {
		return true
	}
	if _, ok := engine.ActionIsSnapOwn(idx); ok {
		return true
	}
	if _, ok := engine.ActionIsSnapOpponent(idx); ok {
		return true
	}
	return false
}

// classifySnap reconstructs the (outcome, slot) of the last snap action from
// LastAction, mirroring _classify_snap over the Python snap_results log entry.
func classifySnap(g *engine.GameState) (outcome int, slot int) {
	idx := g.LastAction.ActionIdx
	if idx == engine.ActionPassSnap {
		return outcomeFail, -1
	}
	if t, ok := engine.ActionIsSnapOwn(idx); ok {
		if g.LastAction.SnapSuccess {
			return outcomeSuccessOwn, int(t)
		}
		return outcomePenalty, -1
	}
	if t, ok := engine.ActionIsSnapOpponent(idx); ok {
		if g.LastAction.SnapSuccess {
			return outcomeSuccessOpp, int(t)
		}
		return outcomePenalty, -1
	}
	// Should not be reached for logged snap actions.
	return outcomeFail, -1
}

// ---------------------------------------------------------------------------
// TokenStream
// ---------------------------------------------------------------------------

const maxSnapFrameTokens = 64 // up to 16 snap frames (4 tokens each) per phase.

// TokenStream holds one player's append-only observation token body plus the
// running snap-phase frame accumulator. It is a flat value type (fixed arrays)
// so it copies with = for agent clone and state snapshot/restore.
type TokenStream struct {
	Tokens [MaxTokenStream]int32
	Length int32

	// snapFrames accumulates the emitted snap frame tokens for the CURRENT snap
	// phase (public, shared across observers). It is reset whenever the snap
	// phase is not active, mirroring snap_results_log's clear-on-phase-boundary.
	snapFrames [maxSnapFrameTokens]int32
	snapLen    int32
}

// tokenOverflowError is returned when appending would exceed MaxTokenStream.
type tokenOverflowError struct{}

func (tokenOverflowError) Error() string { return "token stream overflow (> MaxTokenStream)" }

// ErrTokenOverflow is the sentinel returned by Observe/Init on hard overflow.
var ErrTokenOverflow error = tokenOverflowError{}

func (ts *TokenStream) appendTokens(vals []int32) bool {
	n := int32(len(vals))
	if ts.Length+n > MaxTokenStream {
		return false
	}
	copy(ts.Tokens[ts.Length:ts.Length+n], vals)
	ts.Length += n
	return true
}

// Reset clears the stream to empty.
func (ts *TokenStream) Reset() { *ts = TokenStream{} }

// Init resets the stream and writes the observer's private peeked initial-hand
// frames (one init_peek frame per peeked slot, ascending). Call once at agent
// creation, before any observation is appended. Mirrors initial_peek_frames.
func (ts *TokenStream) Init(g *engine.GameState, observerID uint8) error {
	ts.Reset()
	ps := &g.Players[observerID]
	// InitialPeek indices are ascending as populated by the engine; sort
	// defensively so the frame order matches Python's sorted(peek_indices).
	var peeks [engine.MaxHandSize]uint8
	pn := ps.InitialPeekCount
	for i := uint8(0); i < pn; i++ {
		peeks[i] = ps.InitialPeek[i]
	}
	// insertion sort (tiny)
	for i := uint8(1); i < pn; i++ {
		for j := i; j > 0 && peeks[j-1] > peeks[j]; j-- {
			peeks[j-1], peeks[j] = peeks[j], peeks[j-1]
		}
	}
	for i := uint8(0); i < pn; i++ {
		slot := peeks[i]
		if slot >= ps.HandLen {
			continue
		}
		frame := [3]int32{frameToken(frameInitPeek), slotToken(int(slot)), cardToken(ps.Hand[slot])}
		if !ts.appendTokens(frame[:]) {
			return ErrTokenOverflow
		}
	}
	return nil
}

// Observe appends the token frame group for the most recently applied action,
// as seen by observerID. It must be called exactly once per applied action, per
// observer, using the post-apply game state (g.LastAction identifies the action).
//
// Frame order matches observation_frame_groups: (drawn?) public (cambia?) snaps.
// Snap semantics: obs.snap_results is empty whenever the snap phase is inactive
// in the post-apply state (the phase-end clear), otherwise it is the running
// accumulation of snap outcomes since the phase started; this reproduces the
// clear-on-phase-start / clear-on-phase-end / append-per-snap-action lifecycle
// of Python's snap_results_log.
func (ts *TokenStream) Observe(g *engine.GameState, observerID uint8) error {
	var scratch [16 + maxSnapFrameTokens]int32
	n := 0
	put := func(v int32) { scratch[n] = v; n++ }
	putPeek := func(owner uint8, slot uint8, card engine.Card) {
		put(peekFrameToken())
		put(actorToken(int(owner)))
		put(slotToken(int(slot)))
		put(cardToken(card))
	}

	actor := g.LastAction.ActingPlayer
	idx := g.LastAction.ActionIdx

	// 1. Private own-draw frame: emitted for the actor-observer at the post-draw
	// decision node (right after a draw action). The freshly drawn card is held
	// in the pending-discard state (Pending.Data[0]) that drawStockpile/
	// drawDiscard set before the actor chooses discard vs replace. Surfacing it
	// here -- one event BEFORE the discard/replace decision, not on the later
	// Discard/Replace action -- puts the drawn card in the token prefix that
	// conditions that decision, so the legal-action mask at the post-draw node
	// is determined by the infoset (cambia-528; re-armed Phase-1 bug #21). The
	// draw source (stockpile/discard) does not change the frame: both surface
	// the actor's own drawn card privately. Mirrors sequence_encoding.py's
	// draw-gated drawn frame byte-for-byte.
	if actor == observerID &&
		(idx == engine.ActionDrawStockpile || idx == engine.ActionDrawDiscard) &&
		g.Pending.Type == engine.PendingDiscard {
		put(frameToken(frameDrawn))
		put(cardToken(engine.Card(g.Pending.Data[0])))
	}

	// 2. Public turn frame: ACTOR, ACTION, CARD(discard top after action).
	atok := actionToken(idx)
	if atok < 0 {
		atok = tokSEP
	}
	put(frameToken(framePublic))
	put(actorToken(int(actor)))
	put(atok)
	put(cardToken(g.DiscardTop()))

	// 2b. Private peek-result frames (cambia-529): the card(s) the actor saw via
	// an own-peek (7/8), other-peek (9/T), or King-look (K) ability. Emitted only
	// for the actor-observer (the opponent never sees the peeked identities), one
	// frame per revealed card carrying the card OWNER, its SLOT, and the CARD.
	// peek_own/peek_other reveal one card, recorded in LastAction (RevealedOwner/
	// RevealedIdx/RevealedCard). King-look reveals two cards and leaves the state
	// in PendingKingDecision with both looked cards + slots in Pending.Data (own:
	// Data[0]/Data[2], opponent: Data[1]/Data[3]); the own card is emitted first,
	// matching sequence_encoding.py's own-owner-first ordering.
	if actor == observerID {
		if _, ok := engine.ActionIsPeekOwn(idx); ok {
			putPeek(g.LastAction.RevealedOwner, g.LastAction.RevealedIdx, g.LastAction.RevealedCard)
		} else if _, ok := engine.ActionIsPeekOther(idx); ok {
			putPeek(g.LastAction.RevealedOwner, g.LastAction.RevealedIdx, g.LastAction.RevealedCard)
		} else if _, _, ok := engine.ActionIsKingLook(idx); ok && g.Pending.Type == engine.PendingKingDecision {
			opp := g.OpponentOf(actor)
			putPeek(actor, g.Pending.Data[0], engine.Card(g.Pending.Data[2]))
			putPeek(opp, g.Pending.Data[1], engine.Card(g.Pending.Data[3]))
		}
	}

	// 3. Public cambia frame: only on the CallCambia action.
	if idx == engine.ActionCallCambia && g.IsCambiaCalled() {
		caller := int(g.CambiaCaller)
		if caller < 0 {
			caller = int(actor)
		}
		put(frameToken(frameCambia))
		put(actorToken(caller))
	}

	// 4. Public snap frames. See the snap lifecycle note above.
	if g.Snap.Active {
		if isLoggedSnapAction(idx) {
			outcome, slot := classifySnap(g)
			frame := [4]int32{
				frameToken(frameSnap),
				actorToken(int(actor)),
				outcomeToken(outcome),
				slotTokenSigned(slot),
			}
			if ts.snapLen+4 <= maxSnapFrameTokens {
				copy(ts.snapFrames[ts.snapLen:ts.snapLen+4], frame[:])
				ts.snapLen += 4
			}
		}
		for i := int32(0); i < ts.snapLen; i++ {
			put(ts.snapFrames[i])
		}
	} else {
		ts.snapLen = 0
	}

	if !ts.appendTokens(scratch[:n]) {
		return ErrTokenOverflow
	}
	return nil
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

// Len returns the number of tokens currently in the stream body.
func (ts *TokenStream) Len() int32 { return ts.Length }

// CopyTo copies up to max tokens of the full stream body into out and returns
// the number copied.
func (ts *TokenStream) CopyTo(out []int32) int32 {
	n := ts.Length
	if int32(len(out)) < n {
		n = int32(len(out))
	}
	copy(out[:n], ts.Tokens[:n])
	return n
}

// CopySince copies the tail tokens[since:] (up to max) into out and returns the
// number copied. A negative or over-large `since` clamps to a valid range.
func (ts *TokenStream) CopySince(since int32, out []int32) int32 {
	if since < 0 {
		since = 0
	}
	if since > ts.Length {
		since = ts.Length
	}
	n := ts.Length - since
	if int32(len(out)) < n {
		n = int32(len(out))
	}
	copy(out[:n], ts.Tokens[since:since+n])
	return n
}

// ---------------------------------------------------------------------------
// Vocabulary layout export (for the constants cross-check test)
// ---------------------------------------------------------------------------

// TokenVocabFields is the number of layout integers TokenVocab writes, in a
// stable positional order asserted against sequence_encoding.py by the
// constants cross-check test.
const TokenVocabFields = 23

// TokenVocab writes the vocabulary layout constants into out (length >=
// TokenVocabFields) and returns the number written, or -1 if out is too small.
func TokenVocab(out []int32) int {
	if len(out) < TokenVocabFields {
		return -1
	}
	vals := [TokenVocabFields]int32{
		vocabSize,         // 0  VOCAB_SIZE
		tokPAD,            // 1  PAD_ID
		tokBOS,            // 2  BOS_ID
		tokEOS,            // 3  EOS_ID
		tokSEP,            // 4  SEP_ID
		numSpecial,        // 5  NUM_SPECIAL
		frameBase,         // 6  FRAME_BASE
		numFrameIDs,       // 7  NUM_FRAME_IDS
		actorBase,         // 8  ACTOR_BASE
		tokMaxActors,      // 9  MAX_ACTORS
		actionBase,        // 10 ACTION_BASE
		numActionIDs,      // 11 NUM_ACTION_IDS
		cardBase,          // 12 CARD_BASE
		numCardIDs,        // 13 NUM_CARD_IDS
		slotBase,          // 14 SLOT_BASE
		tokNumSlotIDs,     // 15 NUM_SLOT_IDS
		outcomeBase,       // 16 OUTCOME_BASE
		numSnapOutcomeIDs, // 17 NUM_SNAP_OUTCOME_IDS
		tokMaxSlots,       // 18 MAX_SLOTS
		TokSeqCap,         // 19 SEQ_CAP
		MaxTokenStream,    // 20 Go per-agent hard cap
		peekFrameBase,     // 21 PEEK_FRAME_BASE
		numPeekFrameIDs,   // 22 NUM_PEEK_FRAME_IDS
	}
	copy(out, vals[:])
	return TokenVocabFields
}

// EncodeCardToken returns the CARD-block token for a canonical Go card index
// (suit*13+rank; jokers 52/53). For the constants cross-check test.
func EncodeCardToken(goCardIndex uint8) int32 {
	return cardToken(goIndexToCard(goCardIndex))
}

// EncodeActionToken returns the ACTION-block token for a 2-player action index,
// or -1 if the index does not encode a known action. For the cross-check test.
func EncodeActionToken(actionIdx uint16) int32 { return actionToken(actionIdx) }

// goIndexToCard mirrors the canonical Python card-index encoding used across the
// FFI: suit*13+rank with C=0,D=1,H=2,S=3 and A=0..K=12; 52=RedJoker, 53=BlackJoker.
func goIndexToCard(idx uint8) engine.Card {
	if idx == 52 {
		return engine.NewCard(engine.SuitRedJoker, engine.RankJoker)
	}
	if idx == 53 {
		return engine.NewCard(engine.SuitBlackJoker, engine.RankJoker)
	}
	suit := idx / 13
	rank := idx % 13
	var goSuit uint8
	switch suit {
	case 0:
		goSuit = engine.SuitClubs
	case 1:
		goSuit = engine.SuitDiamonds
	case 2:
		goSuit = engine.SuitHearts
	default:
		goSuit = engine.SuitSpades
	}
	return engine.NewCard(goSuit, rank)
}
