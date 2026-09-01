package main

import "testing"

// The FFI boundary for the rule bounds Validate() gained in cambia-1555 and
// the deck-rank mask it gained in cambia-1478. Same shape and same reason as
// nplayer_bounds_test.go: there is no recover() anywhere in this package, so
// an out-of-range rule that reaches the deal takes the process down instead of
// failing as something Python can catch. These tests assert the constructors
// return the -1 rejection sentinel; they must never reach the panic path.

func rejects(t *testing.T, what string, h int32) {
	t.Helper()
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_rules(%s) = %d, want -1 (rejected)", what, h)
	}
}

func accepts(t *testing.T, what string, h int32) {
	t.Helper()
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(%s) = %d, want a valid handle", what, h)
	}
	testGameFree(h)
}

func TestCambiaGameNewWithRulesRejectsOversizedHand(t *testing.T) {
	// CardsPerPlayer=7 indexed past the fixed [MaxHandSize]Card hand in Deal.
	r := defaultTestRules()
	r.cardsPerPlayer = 7
	rejects(t, "cardsPerPlayer=7", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesAcceptsMaxHandSize(t *testing.T) {
	r := defaultTestRules()
	r.cardsPerPlayer = 6
	accepts(t, "cardsPerPlayer=6", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesRejectsEmptyHand(t *testing.T) {
	r := defaultTestRules()
	r.cardsPerPlayer = 0
	r.initialViewCount = 0
	rejects(t, "cardsPerPlayer=0", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesRejectsPeekAboveHandSize(t *testing.T) {
	r := defaultTestRules() // cardsPerPlayer = 4
	r.initialViewCount = 5
	rejects(t, "initialViewCount=5 over a 4-card hand", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesRejectsThreeJokers(t *testing.T) {
	// NumJokers=3 sized StockLen as 55 while the build loop wrote 54, leaving
	// an unwritten Card(0) - a phantom second Ace of Hearts - in the deck.
	r := defaultTestRules()
	r.numJokers = 3
	rejects(t, "numJokers=3", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesAcceptsEveryValidJokerCount(t *testing.T) {
	for _, jokers := range []uint8{0, 1, 2} {
		r := defaultTestRules()
		r.numJokers = jokers
		accepts(t, "numJokers in range", testGameNewWithRulesFull(1, r))
	}
}

func TestCambiaGameNewWithRulesRejectsFiveDecks(t *testing.T) {
	// NumDecks=5 wrote 270 cards into a [MaxDeckSize]Card array and panicked
	// inside NewGame, before any deal.
	r := defaultTestRules()
	r.numDecks = 5
	rejects(t, "numDecks=5", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesAcceptsEveryValidDeckCount(t *testing.T) {
	for _, decks := range []uint8{0, 1, 2, 3, 4} {
		r := defaultTestRules()
		r.numDecks = decks
		accepts(t, "numDecks in range", testGameNewWithRulesFull(1, r))
	}
}

func TestCambiaGameNewWithRulesRejectsJokersWithFourDecks(t *testing.T) {
	// The combination the ticket names: three jokers across four decks
	// overran the stockpile and panicked in Deal's shuffle.
	r := defaultTestRules()
	r.numJokers = 3
	r.numDecks = 4
	rejects(t, "numJokers=3 numDecks=4", testGameNewWithRulesFull(1, r))
}

// --- Deck rank mask (cambia-1478) ---

const (
	maskAce    = 1 << 0  // engine.RankAce
	maskSix    = 1 << 5  // engine.RankSix
	maskJoker  = 1 << 13 // engine.RankJoker: not a suited rank
	maskAceSix = maskAce | maskSix
)

// tinyRules is the {A,6} 2-card game tiny_2card_plateau.yaml configures, the
// deal that used to come out as a full 13-rank game on this engine.
func tinyRules() testRules {
	r := defaultTestRules()
	r.deckRanks = maskAceSix
	r.numJokers = 0
	r.cardsPerPlayer = 2
	r.initialViewCount = 1
	return r
}

func TestCambiaGameNewWithRulesDealsOnlyTheMaskedRanks(t *testing.T) {
	h := testGameNewWithRulesFull(1, tinyRules())
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(deckRanks={A,6}) = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	for seat := uint8(0); seat < 2; seat++ {
		handLen, slots := testGameGetHand(h, seat)
		if handLen != 2 {
			t.Fatalf("seat %d holds %d cards, want 2", seat, handLen)
		}
		for i := int32(0); i < handLen; i++ {
			// Card index is suit*13 + rank (see the eval surface docs).
			if rank := slots[i] % 13; rank != 0 && rank != 5 {
				t.Errorf("seat %d slot %d holds rank %d, want Ace (0) or Six (5)", seat, i, rank)
			}
		}
	}
	// 2 ranks x 4 suits = 8 cards, less 4 dealt and 1 flipped to the discard.
	if got := testGameStockLen(h); got != 3 {
		t.Errorf("stock length = %d, want 3 (an 8-card deck, 4 dealt, 1 flipped)", got)
	}
}

func TestCambiaGameNewWithRulesReadsBackTheMask(t *testing.T) {
	h := testGameNewWithRulesFull(1, tinyRules())
	if h < 0 {
		t.Fatalf("cambia_game_new_with_rules(deckRanks={A,6}) = %d, want a valid handle", h)
	}
	defer testGameFree(h)

	rc, rec := testGameGetHouseRules(h)
	if rc != evalHouseRuleFields {
		t.Fatalf("get_house_rules = %d, want %d", rc, evalHouseRuleFields)
	}
	if got := uint16(rec[14]) | uint16(rec[15])<<8; got != maskAceSix {
		t.Errorf("deck rank mask reads back as %#04x, want %#04x", got, maskAceSix)
	}
}

func TestCambiaGameNewWithRulesRejectsAMaskWithNoSuitedRank(t *testing.T) {
	r := tinyRules()
	r.deckRanks = maskJoker
	rejects(t, "deckRanks=joker bit only", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesRejectsAMaskBitOutsideTheSuitedRanks(t *testing.T) {
	r := tinyRules()
	r.deckRanks = maskJoker | maskAce
	rejects(t, "deckRanks=joker bit plus Ace", testGameNewWithRulesFull(1, r))
}

func TestCambiaGameNewWithRulesRejectsADeckTooSmallForTheDeal(t *testing.T) {
	// One rank, no jokers: 4 cards, against a 2x2 deal plus the opening flip.
	r := tinyRules()
	r.deckRanks = maskAce
	rejects(t, "deckRanks={A} dealing 2x2+1", testGameNewWithRulesFull(1, r))
}

// --- cambia_game_new_with_deck ---

// fullDeck is a 54-card deal order in canonical card indices.
func fullDeck() []uint8 {
	deck := make([]uint8, 54)
	for i := range deck {
		deck[i] = uint8(i)
	}
	return deck
}

func TestCambiaGameNewWithDeckRejectsStartingPlayerAtTheSeatCount(t *testing.T) {
	// startingPlayer is written straight into CurrentPlayer, and the first
	// post-draw legal-action mask then indexes Players by it.
	h := testGameNewWithDeck(fullDeck(), 2, 4, 2, 2, 2)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(startingPlayer=2, numPlayers=2) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithDeckRejectsStartingPlayerAboveTheSeatCount(t *testing.T) {
	h := testGameNewWithDeck(fullDeck(), 2, 4, 200, 2, 2)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(startingPlayer=200) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithDeckAcceptsTheLastSeat(t *testing.T) {
	h := testGameNewWithDeck(fullDeck(), 2, 4, 1, 2, 2)
	if h < 0 {
		t.Fatalf("cambia_game_new_with_deck(startingPlayer=1, numPlayers=2) = %d, want a valid handle", h)
	}
	testGameFree(h)
}

func TestCambiaGameNewWithDeckRejectsADeckTooShortForTheDeal(t *testing.T) {
	// 2 seats x 4 cards plus the opening flip needs 9; 8 underflowed StockLen
	// from 0 to 255 and dealt out of the rest of the array.
	h := testGameNewWithDeck(fullDeck()[:8], 2, 4, 0, 2, 2)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(deckLen=8, dealing 2x4+1) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithDeckAcceptsTheExactDealLength(t *testing.T) {
	h := testGameNewWithDeck(fullDeck()[:9], 2, 4, 0, 2, 2)
	if h < 0 {
		t.Fatalf("cambia_game_new_with_deck(deckLen=9, dealing 2x4+1) = %d, want a valid handle", h)
	}
	testGameFree(h)
}

func TestCambiaGameNewWithDeckRejectsADeckPastTheStockpile(t *testing.T) {
	// Longer than [MaxDeckSize]Card: silently truncating it dealt a different
	// deck than the caller handed over.
	deck := make([]uint8, 217)
	for i := range deck {
		deck[i] = uint8(i % 54)
	}
	h := testGameNewWithDeck(deck, 2, 4, 0, 2, 2)
	if h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(deckLen=217) = %d, want -1 (rejected)", h)
	}
}

func TestCambiaGameNewWithDeckRejectsOutOfRangeRules(t *testing.T) {
	// The same Validate gate as cambia_game_new_with_rules, reached before
	// NewGame builds anything.
	if h := testGameNewWithDeck(fullDeck(), 9, 4, 0, 2, 2); h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(numPlayers=9) = %d, want -1 (rejected)", h)
	}
	if h := testGameNewWithDeck(fullDeck(), 2, 7, 0, 2, 2); h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(cardsPerPlayer=7) = %d, want -1 (rejected)", h)
	}
	if h := testGameNewWithDeck(fullDeck(), 2, 4, 0, 3, 2); h >= 0 {
		testGameFree(h)
		t.Fatalf("cambia_game_new_with_deck(numJokers=3) = %d, want -1 (rejected)", h)
	}
}
