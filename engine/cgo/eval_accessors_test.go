package main

import (
	"testing"

	engine "github.com/jason-s-yu/cambia/engine"
)

// Tests for the cambia-1425 evaluation-surface accessors. Every case runs on a
// scripted deck so the assertions pin exact card identities rather than
// whatever a seeded shuffle happened to deal.
//
// Canonical card indices used below (suit*13 + rank; C=0, D=1, H=2, S=3;
// ranks A=0..K=12):
//
//	0  = C-A   1  = C-2   2  = C-3   3  = C-4
//	13 = D-A   14 = D-2   15 = D-3   16 = D-4
//	25 = D-K (red King)   26 = H-A   27 = H-2

const (
	cardCA = 0
	cardC2 = 1
	cardC3 = 2
	cardC4 = 3
	cardDA = 13
	cardD2 = 14
	cardD3 = 15
	cardD4 = 16
	cardDK = 25
	cardHA = 26
	cardH2 = 27
)

// scriptedDeck builds a full 54-index deck whose first ten entries fix the
// deal: P0 gets slots 0..3, P1 gets slots 0..3 (round-robin, so the two hands
// interleave), then the discard flip, then the next stockpile draw. The rest is
// filler in ascending index order, skipping anything already placed.
func scriptedDeck(nextDraw uint8) []uint8 {
	head := []uint8{
		cardCA, cardDA, // slot 0
		cardC2, cardD2, // slot 1
		cardC3, cardD3, // slot 2
		cardC4, cardD4, // slot 3
		cardHA,   // discard flip
		nextDraw, // first card drawn from the stockpile
	}
	used := make(map[uint8]bool, len(head))
	for _, c := range head {
		used[c] = true
	}
	deck := append([]uint8(nil), head...)
	for i := uint8(0); i < 54; i++ {
		if !used[i] {
			deck = append(deck, i)
		}
	}
	return deck
}

// newScriptedGame deals the scripted deck to two seats, four cards each, with
// nextDraw waiting on top of the stockpile.
func newScriptedGame(t *testing.T, nextDraw uint8) int32 {
	t.Helper()
	h := testGameNewWithDeck(scriptedDeck(nextDraw), 2, 4, 0, 2, 2)
	if h < 0 {
		t.Fatalf("testGameNewWithDeck = %d, want a valid handle", h)
	}
	t.Cleanup(func() { testGameFree(h) })
	return h
}

func TestCardIndexRoundTrip(t *testing.T) {
	for idx := uint8(0); idx < 54; idx++ {
		if got := testCardIndexRoundTrip(idx); got != idx {
			t.Errorf("cardToIndex(indexToCard(%d)) = %d, want %d", idx, got, idx)
		}
	}
}

func TestCardToIndexEmptyCard(t *testing.T) {
	if got := cardToIndex(engine.EmptyCard); got != cardIndexNone {
		t.Errorf("cardToIndex(EmptyCard) = %d, want %d", got, cardIndexNone)
	}
}

func TestGameGetHand(t *testing.T) {
	h := newScriptedGame(t, cardH2)

	handLen, slots := testGameGetHand(h, 0)
	if handLen != 4 {
		t.Fatalf("seat 0 hand length = %d, want 4", handLen)
	}
	want0 := [engine.MaxHandSize]uint8{cardCA, cardC2, cardC3, cardC4, cardIndexNone, cardIndexNone}
	if slots != want0 {
		t.Errorf("seat 0 hand = %v, want %v", slots, want0)
	}

	handLen, slots = testGameGetHand(h, 1)
	if handLen != 4 {
		t.Fatalf("seat 1 hand length = %d, want 4", handLen)
	}
	want1 := [engine.MaxHandSize]uint8{cardDA, cardD2, cardD3, cardD4, cardIndexNone, cardIndexNone}
	if slots != want1 {
		t.Errorf("seat 1 hand = %v, want %v", slots, want1)
	}
}

func TestGameGetHandRejectsBadInput(t *testing.T) {
	h := newScriptedGame(t, cardH2)

	if got, _ := testGameGetHand(h, 2); got != -1 {
		t.Errorf("seat 2 of a 2-seat game = %d, want -1", got)
	}
	if got := testGameGetHandShortBuf(h, 0); got != -1 {
		t.Errorf("short buffer = %d, want -1", got)
	}
	if got, _ := testGameGetHand(-1, 0); got != -1 {
		t.Errorf("invalid handle = %d, want -1", got)
	}
}

func TestGameGetDiscardPile(t *testing.T) {
	h := newScriptedGame(t, cardH2)

	if got := testGameDiscardLen(h); got != 1 {
		t.Fatalf("discard length after the deal = %d, want 1", got)
	}
	n, pile := testGameGetDiscardPile(h)
	if n != 1 || len(pile) != 1 || pile[0] != cardHA {
		t.Fatalf("discard pile = %v (n=%d), want [%d]", pile, n, cardHA)
	}
	if got := testGameDiscardTopCard(h); got != cardHA {
		t.Errorf("discard top card = %d, want %d", got, cardHA)
	}

	// Draw and discard: the pile grows bottom-to-top, so the new card lands
	// last and becomes the top.
	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	if rc := testGameApplyAction(h, engine.ActionDiscardNoAbility); rc < 0 {
		t.Fatalf("discard failed: %d", rc)
	}
	n, pile = testGameGetDiscardPile(h)
	if n != 2 || len(pile) != 2 {
		t.Fatalf("discard pile after one discard = %v (n=%d), want 2 entries", pile, n)
	}
	if pile[0] != cardHA || pile[1] != cardH2 {
		t.Errorf("discard pile = %v, want [%d %d] (bottom first)", pile, cardHA, cardH2)
	}
	if got := testGameDiscardTopCard(h); got != cardH2 {
		t.Errorf("discard top card = %d, want %d", got, cardH2)
	}
}

func TestGameGetDiscardPileRejectsShortBuffer(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	if got := testGameGetDiscardPileShortBuf(h, 0); got != -1 {
		t.Errorf("pile of 1 into a 0-byte buffer = %d, want -1", got)
	}
	if got := testGameGetDiscardPileShortBuf(h, 1); got != 1 {
		t.Errorf("pile of 1 into a 1-byte buffer = %d, want 1", got)
	}
	if got := testGameDiscardLen(-1); got != -1 {
		t.Errorf("discard length on an invalid handle = %d, want -1", got)
	}
	if got := testGameDiscardTopCard(-1); got != -1 {
		t.Errorf("discard top on an invalid handle = %d, want -1", got)
	}
}

func TestGameGetPendingNone(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields {
		t.Fatalf("get_pending = %d, want %d", rc, evalPendingFields)
	}
	if rec[0] != uint8(engine.PendingNone) {
		t.Errorf("pending type = %d, want %d (PendingNone)", rec[0], engine.PendingNone)
	}
	for i := 1; i < evalPendingFields-1; i++ {
		if rec[i] != cardIndexNone {
			t.Errorf("field [%d] = %d, want the %d sentinel with no pending action", i, rec[i], cardIndexNone)
		}
	}
	if rec[9] != 0 {
		t.Errorf("reserved field = %d, want 0", rec[9])
	}
}

func TestGameGetPendingDiscardCarriesDrawnCard(t *testing.T) {
	h := newScriptedGame(t, cardDK)
	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields {
		t.Fatalf("get_pending = %d, want %d", rc, evalPendingFields)
	}
	if rec[0] != uint8(engine.PendingDiscard) {
		t.Fatalf("pending type = %d, want %d (PendingDiscard)", rec[0], engine.PendingDiscard)
	}
	if rec[1] != 0 {
		t.Errorf("acting seat = %d, want 0", rec[1])
	}
	if rec[2] != cardDK {
		t.Errorf("drawn card = %d, want %d (red King)", rec[2], cardDK)
	}
	if rec[3] != engine.DrawnFromStockpile {
		t.Errorf("drawn from = %d, want %d (stockpile)", rec[3], engine.DrawnFromStockpile)
	}
	for _, i := range []int{4, 5, 6, 7, 8} {
		if rec[i] != cardIndexNone {
			t.Errorf("field [%d] = %d, want the %d sentinel on a discard pending", i, rec[i], cardIndexNone)
		}
	}
}

func TestGameGetPendingDrawnFromDiscard(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	if rc := testGameApplyAction(h, engine.ActionDrawDiscard); rc < 0 {
		t.Fatalf("draw from discard failed: %d", rc)
	}
	_, rec := testGameGetPending(h)
	if rec[2] != cardHA {
		t.Errorf("drawn card = %d, want %d (the flipped H-A)", rec[2], cardHA)
	}
	if rec[3] != engine.DrawnFromDiscard {
		t.Errorf("drawn from = %d, want %d (discard)", rec[3], engine.DrawnFromDiscard)
	}
}

func TestGameGetPendingKingDecisionCarriesBothLookedCards(t *testing.T) {
	h := newScriptedGame(t, cardDK)
	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	if rc := testGameApplyAction(h, engine.ActionDiscardWithAbility); rc < 0 {
		t.Fatalf("discard King with ability failed: %d", rc)
	}
	// Look at own slot 0 (C-A) and the opponent's slot 1 (D-2).
	if rc := testGameApplyAction(h, engine.EncodeKingLook(0, 1)); rc < 0 {
		t.Fatalf("king look failed: %d", rc)
	}

	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields {
		t.Fatalf("get_pending = %d, want %d", rc, evalPendingFields)
	}
	if rec[0] != uint8(engine.PendingKingDecision) {
		t.Fatalf("pending type = %d, want %d (PendingKingDecision)", rec[0], engine.PendingKingDecision)
	}
	if rec[1] != 0 {
		t.Errorf("acting seat = %d, want 0", rec[1])
	}
	if rec[4] != 0 {
		t.Errorf("own slot = %d, want 0", rec[4])
	}
	if rec[5] != 1 {
		t.Errorf("target slot = %d, want 1", rec[5])
	}
	if rec[6] != 1 {
		t.Errorf("target seat = %d, want 1", rec[6])
	}
	if rec[7] != cardCA {
		t.Errorf("own looked card = %d, want %d (C-A)", rec[7], cardCA)
	}
	if rec[8] != cardD2 {
		t.Errorf("target looked card = %d, want %d (D-2)", rec[8], cardD2)
	}
	if rec[2] != cardIndexNone || rec[3] != cardIndexNone {
		t.Errorf("drawn-card fields = (%d, %d), want the %d sentinel on a King decision", rec[2], rec[3], cardIndexNone)
	}
}

func TestGameGetSnapStateInactive(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	rc, rec := testGameGetSnapState(h)
	if rc != evalSnapFields {
		t.Fatalf("get_snap_state = %d, want %d", rc, evalSnapFields)
	}
	want := [evalSnapFields]uint8{0, cardIndexNone, cardIndexNone, 0, 0, cardIndexNone}
	if rec != want {
		t.Errorf("snap record with no open window = %v, want %v", rec, want)
	}
}

func TestGameGetSnapStateActive(t *testing.T) {
	// H-2 discarded opens a rank-2 window: both seats hold a 2 (C-2, D-2).
	h := newScriptedGame(t, cardH2)
	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	if rc := testGameApplyAction(h, engine.ActionDiscardNoAbility); rc < 0 {
		t.Fatalf("discard failed: %d", rc)
	}

	rc, rec := testGameGetSnapState(h)
	if rc != evalSnapFields {
		t.Fatalf("get_snap_state = %d, want %d", rc, evalSnapFields)
	}
	if rec[0] != 1 {
		t.Fatalf("snap active = %d, want 1 after discarding a matched rank", rec[0])
	}
	if rec[1] != engine.RankTwo {
		t.Errorf("snapped rank = %d, want %d (RankTwo)", rec[1], engine.RankTwo)
	}
	if rec[2] != cardH2 {
		t.Errorf("snapped card = %d, want %d (H-2, the card that opened the window)", rec[2], cardH2)
	}
	if rec[3] != 2 {
		t.Errorf("snapper count = %d, want 2 (both seats hold a 2)", rec[3])
	}
	if rec[4] != 0 {
		t.Errorf("snapper cursor = %d, want 0", rec[4])
	}
	if rec[5] != 0 {
		t.Errorf("snapper seat at the cursor = %d, want 0 (the discarder snaps first)", rec[5])
	}
}

func TestGameGetPendingSnapMove(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	if rc := testGameApplyAction(h, engine.ActionDrawStockpile); rc < 0 {
		t.Fatalf("draw stockpile failed: %d", rc)
	}
	if rc := testGameApplyAction(h, engine.ActionDiscardNoAbility); rc < 0 {
		t.Fatalf("discard failed: %d", rc)
	}
	// Seat 0 snaps the opponent's slot 1 (D-2), which leaves seat 0 owing a
	// card to the vacated slot.
	if rc := testGameApplyAction(h, engine.EncodeSnapOpponent(1)); rc < 0 {
		t.Fatalf("snap opponent failed: %d", rc)
	}

	rc, rec := testGameGetPending(h)
	if rc != evalPendingFields {
		t.Fatalf("get_pending = %d, want %d", rc, evalPendingFields)
	}
	if rec[0] != uint8(engine.PendingSnapMove) {
		t.Fatalf("pending type = %d, want %d (PendingSnapMove)", rec[0], engine.PendingSnapMove)
	}
	if rec[1] != 0 {
		t.Errorf("acting seat = %d, want 0 (the snapper owes the move)", rec[1])
	}
	if rec[6] != 1 {
		t.Errorf("target seat = %d, want 1 (the snapped seat)", rec[6])
	}
	if rec[5] != 1 {
		t.Errorf("target slot = %d, want 1 (the vacated slot)", rec[5])
	}
	if rec[2] != cardIndexNone || rec[3] != cardIndexNone || rec[4] != cardIndexNone {
		t.Errorf("drawn/own fields = (%d, %d, %d), want the %d sentinel on a snap move",
			rec[2], rec[3], rec[4], cardIndexNone)
	}
}

func TestGameGetHouseRules(t *testing.T) {
	h := newScriptedGame(t, cardH2)
	rc, rec := testGameGetHouseRules(h)
	if rc != evalHouseRuleFields {
		t.Fatalf("get_house_rules = %d, want %d", rc, evalHouseRuleFields)
	}
	want := [evalHouseRuleFields]uint8{
		0, 0, // max game turns = 0 (unlimited)
		4, // cards per player
		0, // cambia allowed round
		2, // penalty draw count
		1, // allow draw from discard
		0, // allow replace abilities
		1, // allow opponent snapping
		0, // snap race
		2, // jokers
		1, // lock caller hand
		2, // players
		2, // initial view count
		1, // decks
		// deck rank mask: the 0 sentinel reads back as all 13 suited ranks
		0xFF, 0x1F,
	}
	if rec != want {
		t.Errorf("house rules record = %v, want %v", rec, want)
	}
}

func TestGameGetHouseRulesNormalizesPlayerCount(t *testing.T) {
	// cambia_game_new defaults to DefaultHouseRules, which sets NumPlayers=2
	// explicitly; the sentinel path is exercised through the effective count
	// staying in agreement with cambia_game_num_players.
	h := testGameNew(7)
	if h < 0 {
		t.Fatalf("testGameNew = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	_, rec := testGameGetHouseRules(h)
	if got := testGameNumPlayers(h); rec[11] != got {
		t.Errorf("house-rules player count = %d, want %d (cambia_game_num_players)", rec[11], got)
	}
}

func TestEvalAccessorsRejectBadHandles(t *testing.T) {
	if rc, _ := testGameGetPending(-1); rc != -1 {
		t.Errorf("get_pending on an invalid handle = %d, want -1", rc)
	}
	if rc, _ := testGameGetSnapState(-1); rc != -1 {
		t.Errorf("get_snap_state on an invalid handle = %d, want -1", rc)
	}
	if rc, _ := testGameGetHouseRules(-1); rc != -1 {
		t.Errorf("get_house_rules on an invalid handle = %d, want -1", rc)
	}
}
