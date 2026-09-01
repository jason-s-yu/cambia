package engine

import "testing"

// Deck composition: what NewGame actually writes into the stockpile, for the
// DeckRanks mask (cambia-1478) and for the joker/deck-count grid whose StockLen
// used to be taken from the requested counts rather than the written cards
// (cambia-1555).

// stockCounts returns how many of each (suit, rank) the built stockpile holds,
// keyed by the packed Card, counting only the StockLen cards NewGame reports.
func stockCounts(g *GameState) map[Card]int {
	counts := make(map[Card]int)
	for i := uint8(0); i < g.StockLen; i++ {
		counts[g.Stockpile[i]]++
	}
	return counts
}

// TestNewGameDeckRanksDealsOnlyTheSelectedRanks is the cambia-1478 case: the
// tiny {A,6} game the Python engine has always dealt, now expressible in the
// rules the Go engine is built from.
func TestNewGameDeckRanksDealsOnlyTheSelectedRanks(t *testing.T) {
	r := DefaultHouseRules()
	r.DeckRanks = (1 << RankAce) | (1 << RankSix)
	r.NumJokers = 0
	r.CardsPerPlayer = 2
	r.InitialViewCount = 1
	if err := r.Validate(); err != nil {
		t.Fatalf("the {A,6} tiny rules did not validate: %v", err)
	}

	g := NewGame(42, r)
	if g.StockLen != 8 {
		t.Errorf("StockLen = %d, want 8 (2 ranks x 4 suits, no jokers)", g.StockLen)
	}
	for card, n := range stockCounts(&g) {
		if rank := card.Rank(); rank != RankAce && rank != RankSix {
			t.Errorf("stockpile holds rank %d, want only Ace (%d) and Six (%d)", rank, RankAce, RankSix)
		}
		if n != 1 {
			t.Errorf("card %#02x appears %d times, want 1", uint8(card), n)
		}
	}
	// One of each suit per included rank.
	for _, rank := range []uint8{RankAce, RankSix} {
		for suit := uint8(0); suit < 4; suit++ {
			if stockCounts(&g)[NewCard(suit, rank)] != 1 {
				t.Errorf("suit %d of rank %d missing from the stockpile", suit, rank)
			}
		}
	}
}

// TestNewGameDeckRanksCountsJokersSeparately: the mask governs suited ranks
// only, jokers stay governed by NumJokers.
func TestNewGameDeckRanksCountsJokersSeparately(t *testing.T) {
	r := DefaultHouseRules()
	r.DeckRanks = (1 << RankAce) | (1 << RankSix)
	r.CardsPerPlayer = 2
	r.InitialViewCount = 1
	for jokers := uint8(0); jokers <= MaxJokersPerDeck; jokers++ {
		r.NumJokers = jokers
		g := NewGame(42, r)
		want := uint8(8 + jokers)
		if g.StockLen != want {
			t.Errorf("NumJokers=%d: StockLen = %d, want %d", jokers, g.StockLen, want)
		}
		if got := r.DeckSize(); got != int(want) {
			t.Errorf("NumJokers=%d: DeckSize() = %d, want %d", jokers, got, want)
		}
	}
}

// TestNewGameDefaultMaskIsTheStandardDeck pins the compatibility guarantee: the
// DeckRanks=0 sentinel builds the same deck, in the same order, as the
// mask-free build did, so every seeded game deals exactly what it dealt before.
func TestNewGameDefaultMaskIsTheStandardDeck(t *testing.T) {
	sentinel := NewGame(42, DefaultHouseRules())

	explicit := DefaultHouseRules()
	explicit.DeckRanks = AllDeckRanks
	spelled := NewGame(42, explicit)

	if sentinel.StockLen != spelled.StockLen {
		t.Fatalf("StockLen = %d with the sentinel, %d spelled out", sentinel.StockLen, spelled.StockLen)
	}
	if sentinel.StockLen != StandardDeckSize {
		t.Errorf("StockLen = %d, want %d", sentinel.StockLen, StandardDeckSize)
	}
	for i := uint8(0); i < sentinel.StockLen; i++ {
		if sentinel.Stockpile[i] != spelled.Stockpile[i] {
			t.Fatalf("stockpile[%d] = %#02x with the sentinel, %#02x spelled out",
				i, uint8(sentinel.Stockpile[i]), uint8(spelled.Stockpile[i]))
		}
	}
	// Every suited card exactly once, plus both jokers.
	counts := stockCounts(&sentinel)
	if len(counts) != StandardDeckSize {
		t.Errorf("stockpile holds %d distinct cards, want %d", len(counts), StandardDeckSize)
	}
	for suit := uint8(0); suit < 4; suit++ {
		for rank := uint8(0); rank <= RankKing; rank++ {
			if counts[NewCard(suit, rank)] != 1 {
				t.Errorf("suit %d rank %d appears %d times, want 1", suit, rank, counts[NewCard(suit, rank)])
			}
		}
	}
}

// TestNewGameStockLenMatchesCardsWritten walks the full 0-2 jokers by 1-4 decks
// grid and asserts each deck copy contributes exactly 52 + NumJokers cards and
// that StockLen equals what was written (cambia-1555 AC4).
func TestNewGameStockLenMatchesCardsWritten(t *testing.T) {
	for jokers := uint8(0); jokers <= MaxJokersPerDeck; jokers++ {
		for decks := uint8(1); decks <= MaxDecks; decks++ {
			r := DefaultHouseRules()
			r.NumJokers = jokers
			r.NumDecks = decks
			if err := r.Validate(); err != nil {
				t.Fatalf("jokers=%d decks=%d: unexpectedly invalid: %v", jokers, decks, err)
			}
			g := NewGame(42, r)

			perDeck := 52 + int(jokers)
			want := int(decks) * perDeck
			if int(g.StockLen) != want {
				t.Errorf("jokers=%d decks=%d: StockLen = %d, want %d", jokers, decks, g.StockLen, want)
			}
			if got := r.DeckSize(); got != want {
				t.Errorf("jokers=%d decks=%d: DeckSize() = %d, want %d", jokers, decks, got, want)
			}

			// Per-copy contribution: every suited card, and every joker the
			// copy includes, appears once per deck copy and nothing else does.
			counts := stockCounts(&g)
			written := 0
			for _, n := range counts {
				written += n
			}
			if written != int(g.StockLen) {
				t.Errorf("jokers=%d decks=%d: counted %d cards, StockLen says %d",
					jokers, decks, written, g.StockLen)
			}
			for suit := uint8(0); suit < 4; suit++ {
				for rank := uint8(0); rank <= RankKing; rank++ {
					if got := counts[NewCard(suit, rank)]; got != int(decks) {
						t.Errorf("jokers=%d decks=%d: suit %d rank %d appears %d times, want %d",
							jokers, decks, suit, rank, got, decks)
					}
				}
			}
			for j := uint8(0); j < MaxJokersPerDeck; j++ {
				suit := SuitRedJoker
				if j == 1 {
					suit = SuitBlackJoker
				}
				want := 0
				if j < jokers {
					want = int(decks)
				}
				if got := counts[NewCard(suit, RankJoker)]; got != want {
					t.Errorf("jokers=%d decks=%d: joker suit %d appears %d times, want %d",
						jokers, decks, suit, got, want)
				}
			}
		}
	}
}

// TestNewGameExcessJokersLeaveNoPhantomCard is the cambia-1555 regression:
// StockLen used to be sized as NumDecks*(52+NumJokers) while the build loop
// wrote at most 2 jokers per copy, so NumJokers=3 left an unwritten Card(0) -
// the Ace of Hearts - inside StockLen's range, and the stockpile reported 55
// cards holding two Aces of Hearts.
//
// Validate rejects NumJokers=3 at the FFI edge now; this covers the direct
// in-process caller that skips it.
func TestNewGameExcessJokersLeaveNoPhantomCard(t *testing.T) {
	r := DefaultHouseRules()
	r.NumJokers = MaxJokersPerDeck + 1

	g := NewGame(42, r)
	if g.StockLen != StandardDeckSize {
		t.Errorf("StockLen = %d, want %d (the joker loop writes at most %d)",
			g.StockLen, StandardDeckSize, MaxJokersPerDeck)
	}
	counts := stockCounts(&g)
	if got := counts[NewCard(SuitHearts, RankAce)]; got != 1 {
		t.Errorf("Ace of Hearts appears %d times, want 1 (a second one is the phantom card)", got)
	}
	if len(counts) != StandardDeckSize {
		t.Errorf("stockpile holds %d distinct cards, want %d", len(counts), StandardDeckSize)
	}
}

// TestNewGameClampsAnUnvalidatedDeckCount is the numDecks() defense in depth:
// NumDecks=5 asked for 270 cards from a [MaxDeckSize]Card array and panicked
// inside NewGame. Validate rejects it at the FFI edge; a direct caller that
// skips Validate gets the clamp instead of a crash.
func TestNewGameClampsAnUnvalidatedDeckCount(t *testing.T) {
	r := DefaultHouseRules()
	r.NumDecks = MaxDecks + 1

	g := NewGame(42, r)
	if int(g.StockLen) != MaxDecks*StandardDeckSize {
		t.Errorf("StockLen = %d, want %d (clamped to MaxDecks)", g.StockLen, MaxDecks*StandardDeckSize)
	}
}

// TestDealClampsAnUnvalidatedHandSize is the cardsPerPlayer() defense in depth:
// CardsPerPlayer=7 indexed past the fixed [MaxHandSize]Card hand and panicked
// inside Deal. Validate rejects it at the FFI edge; a direct caller gets the
// clamp.
func TestDealClampsAnUnvalidatedHandSize(t *testing.T) {
	r := DefaultHouseRules()
	r.CardsPerPlayer = MaxHandSize + 1

	g := NewGame(42, r)
	g.Deal()
	for p := uint8(0); p < g.NumActivePlayers(); p++ {
		if g.Players[p].HandLen != MaxHandSize {
			t.Errorf("seat %d holds %d cards, want %d (clamped to MaxHandSize)",
				p, g.Players[p].HandLen, MaxHandSize)
		}
	}
}
